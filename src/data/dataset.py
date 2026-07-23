import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, SequentialSampler
from typing import List, Dict, Any, Optional

import glob
import numpy as np
import cv2
import importlib
import logging
import random
import pandas as pd

from src.utils.huggingface import load_editing_dataset, is_huggingface_repo
from src.utils.opencv_runtime import (
    configure_opencv_dataloader_worker,
    configure_opencv_runtime,
)
from src.data.config import DatasetInitArgs
import re
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


_pat_end = re.compile(r'control_(\d+)\.(?:png|jpe?g|webp)$', re.IGNORECASE)


class EpochShuffleSampler(Sampler[int]):
    """Deterministic sampler whose order depends only on (seed, epoch)."""

    def __init__(self, data_source, seed: int = 0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.data_source), generator=generator).tolist()

    def __len__(self):
        return len(self.data_source)


def is_control_image(path: str):
    """返回 (ok, d)。ok=True 表示以 control_{d}.png/jpg/jpeg/webp 结尾；d 为整数。"""
    name = Path(path).name
    m = _pat_end.search(name)
    return m is not None


def _first_existing(base_dir: str, stem: str, exts=IMG_EXTS) -> Optional[str]:
    for ext in exts:
        p = os.path.join(base_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def get_number_of_controls(control_dir: str, stem: str) -> int:
    for ext in IMG_EXTS:
        control_paths = glob.glob(os.path.join(control_dir, f"{stem}_control_[0-99]*{ext}"))
        if len(control_paths) > 0:
            return len(control_paths)
    return 0


def _collect_extra_controls(control_dir: str, stem: str, num_controls: int) -> List[str]:
    """匹配 stem_control_1.*, stem_control_2.*, …（使用 _control_N 格式），排除 *_mask."""
    out = []
    for i in range(1, num_controls + 1):
        for ext in IMG_EXTS:
            control_path = os.path.join(control_dir, f"{stem}_control_{i}{ext}")
            if os.path.exists(control_path):
                out.append(control_path)
                break
    return out


def _find_mask(images_dir: str, control_dir: str, stem: str) -> Optional[str]:
    cands = [
        os.path.join(images_dir, f"{stem}_mask.png"),
        os.path.join(control_dir, f"{stem}_mask.png"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


class ImageDataset(Dataset):
    """
    图像-控制图像-文本 三元组数据集。

    使用示例:
    ```python
    from src.data.dataset import ImageDataset, collate_fn
    from torch.utils.data import DataLoader

        data_config = {
            'dataset_path': '/path/to/dataset',
            'image_size': (512, 512),
        }
    dataset = ImageDataset(data_config)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    ```
    """
    def __init__(self, data_config: DatasetInitArgs):
        """
        初始化数据集。
        Args:
            data_config: 包含数据集路径和处理配置的字典，常见键：
                - dataset_path: 数据集根目录路径（str 或 [str]）
                - image_size: 目标尺寸 [img_w, img_h] 或 int，默认 None（不缩放）
                - selected_control_indexes: 选择的控制图像索引，默认 None

        期望的数据集目录结构（每个根路径下）:
            dataset_root/
              training_images/
                xxx.png
                xxx.txt          # 与图像同名的 caption 文本
              control_images/
                xxx.png
                xxx_1.png
                xxx_2.png
                xxx_mask.png     # 可选，若存在则会返回 'mask'

        示例:
        ```python
        cfg = {
            'dataset_path': ['/data/ds1', '/data/ds2'],
        }
        ds = ImageDataset(cfg)
        sample = ds[0]
        ```
        """
        self.data_config = data_config
        dataset_path = data_config.dataset_path

        # 支持多个数据集路径
        if isinstance(dataset_path, (list, tuple)):
            self.dataset_paths = list(dataset_path)
        else:
            self.dataset_paths = [dataset_path]

        self.hf_datasets = {}

        self.selected_control_indexes = data_config.selected_control_indexes
        # loading datsets
        self._load_all_datasets()
        if len(self.all_samples) == 0:
            raise ValueError(f"No samples loaded from dataset_path={dataset_path}")
        self.load_processor()

    def load_processor(self):
        """load processor"""
        from src.utils.tools import instantiate_class
        class_path = self.data_config.processor.class_path
        init_args = self.data_config.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def _load_all_datasets(self):
        """Load datasets from local directories or Hugging Face repositories."""
        self.all_samples = []  # Unified list of all samples
        # [file_info]
        # file_info:
        # dict['dataset_type', 'local_index', 'repo_id','global_index' ]

        for dataset_path in self.dataset_paths:
            split = None
            if isinstance(dataset_path, dict):
                repo_id = dataset_path['repo_id']
                split = dataset_path['split']
                dataset_path = repo_id
            if is_huggingface_repo(dataset_path):
                samples = self._load_huggingface_dataset(dataset_path, split=split)
            elif isinstance(dataset_path, str) and dataset_path.endswith('.csv'):
                samples = self._load_csv_dataset(dataset_path)
            else:
                samples = self._load_local_dataset(dataset_path)
                # [image_file, control_files, prompt_file, mask_file, dataset_type, local_index, global_index]
            if samples is None:
                raise ValueError(f"No samples loaded from {dataset_path}")
            self.all_samples += samples

    def __len__(self):
        """Return total number of samples across all datasets."""
        return len(self.all_samples)

    def _load_huggingface_dataset(self, repo_id: str, split: Optional[str] = None):
        """
        Load dataset from Hugging Face using lazy loading approach.

        No enumeration happens here - we just store the dataset object
        and metadata for later access.
        """
        # Load the dataset (this is fast, just creates the dataset object)
        dataset = load_editing_dataset(repo_id, split=split)
        # Store HF dataset reference
        dataset_info = {
            'type': 'huggingface',
            'repo_id': repo_id,
            'dataset': dataset,
            'length': len(dataset),
            'start_idx': len(self.all_samples)  # Track where this dataset starts
        }
        logging.info(f"Loaded Hugging Face dataset: {repo_id}, {dataset_info}")

        self.hf_datasets[repo_id] = dataset_info

        # Add entries for each sample without iterating
        # We'll create lightweight placeholders
        samples = []
        for idx in range(dataset_info['length']):
            sample_ref = {
                'dataset_type': 'huggingface',
                'repo_id': repo_id,
                'local_index': idx,
                'global_index': dataset_info['start_idx'] + idx
            }
            samples.append(sample_ref)
        return samples

    def _load_local_dataset(self, dataset_path: str) -> List[dict]:
        """Load dataset from local directory."""
        # under the path, find
        # 1. find the images_dir and control_dir
        image_dir, control_dir = self._find_directories(dataset_path)
        # 2. find the image_files and control_files, prompt file or mask file [optional]
        samples = self._scan_image_files(image_dir, control_dir)
        return samples

    def _load_csv_dataset(self, dataset_path: str) -> List[dict]:
        """Load dataset from CSV file.
        samples.append(
                {
                    "image": image_path,
                    "control": controls,
                    "caption": prompt_file,
                    "mask_file": mask_file,
                    "dataset_type": "local",
                    "local_index": n,
                    "global_index": start_idx + n,
                }
            )
        """

        df = pd.read_csv(dataset_path)
        start_idx = len(self.all_samples)
        # calculate contrl numbers
        columns = df.columns
        columns = [x for x in columns if 'path_control' in x]
        control_keys = sorted(columns)
        samples = []
        for idx, row in df.iterrows():
            controls = [row[x] for x in control_keys]
            prompt = row["prompt"]
            data = {
                "image": row["path_target"],
                "control": controls,
                "caption": prompt,
                "dataset_type": "local_csv",
                "local_index": idx,
                "global_index": start_idx + idx,

            }
            if 'path_mask' in row:
                mask_file = row["path_mask"]
                data['mask_file'] = mask_file
            samples.append(data)
        return samples

    def _find_directories(self, dataset_path: str) -> List[str]:
        """
        查找所有数据集根路径下的图像目录与控制图像目录。
        优先匹配:
            images: ['training_images', 'images', 'target_images']
            control: ['control_images', 'control', 'condition_images']
        若都未找到则回退到默认 'training_images' 与 'control_images'。
        Returns:
            images_dirs, control_dirs
        """

        image_possible_names = ['training_images', 'images', 'target_images', 'target', 'targets']
        control_possible_names = ['control_images', 'control', 'condition_images', 'controls']

        # 查找图像目录
        images_dir = None
        for name in image_possible_names:
            path = os.path.join(dataset_path, name)
            if os.path.exists(path):
                logging.info(f"Found images directory: {path}")
                images_dir = path
                break

        # 查找控制图像目录
        control_dir = None
        for name in control_possible_names:
            path = os.path.join(dataset_path, name)
            if os.path.exists(path):
                logging.info(f"Found control directory: {path}")
                control_dir = path
                break
        if images_dir is None:
            raise FileNotFoundError(f"No image directory found under {dataset_path}")
        if control_dir is None:
            raise FileNotFoundError(f"No control directory found under {dataset_path}")
        return images_dir, control_dir

    def _scan_image_files(self, images_dir, control_dir):
        """
        扫描所有数据集中的成对样本。
        假定 target 图片一定存在
        样本需满足：
            - images_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz.(jpg|jpeg|png|bmp)
            - control_dir/xyz_control_1.(jpg|jpeg|png|bmp) [additional control images]
            - control_dir/xyz_control_2.(jpg|jpeg|png|bmp) [additional control images]
        prompt file could be in either images_dir or control_dir
            - images_dir/xyz.txt 作为 prompt
            - control_dir/xyz.txt 作为 prompt
        (mask could be either in control_dir or images_dir)
            - control_dir/xxx_mask.png  [additional mask images]
            - images_dir/xyz_mask.png [additional mask images]

        Returns:
            List[dict]: 'image' 'control: list[str]' 'caption' 'dataset_index' 'mask_file'

        """
        # first looking for prompt text

        # first search target images
        target_images = glob.glob(os.path.join(images_dir, '*.*'))
        target_images = [img for img in target_images if img.endswith(IMG_EXTS)]
        # exclude mask images
        target_images = [img for img in target_images if not img.endswith('_mask.png')]
        target_images = [img for img in target_images if not is_control_image(img)]
        # exclude control images

        samples = []

        start_idx = len(self.all_samples)

        # 用 stem 归并（同 stem 出现两边时优先 images_dir 文本）
        stems = [os.path.splitext(os.path.basename(p))[0] for p in target_images]

        # filter these stems that dont have correspoding images
        stems = [s for s in stems if _first_existing(images_dir, s) is not None]
        if len(stems) == 0:
            raise ValueError(f"No target image stems found under {images_dir}")

        logging.info('found %d prompts', len(stems))

        # 2) 对每个 stem 按需拼装样本
        n = 0
        from tqdm import tqdm

        num_controls = get_number_of_controls(control_dir, stems[0])

        logging.info('found %d controls', num_controls)
        logging.info(f'found with stem {control_dir}/{stems[0]}')
        for stem in tqdm(stems, desc='matching prompts'):
            # source image（必须在 images_dir）
            image_path = _first_existing(images_dir, stem)
            if image_path is None:
                raise FileNotFoundError(f"Missing target image for stem {stem} under {images_dir}")

            # control image（必须在 control_dir）
            main_control = _first_existing(control_dir, stem)

            # 额外控制图
            if main_control is not None:
                extras = _collect_extra_controls(control_dir, stem, num_controls)
                controls = [main_control] + extras
            else:
                raise FileNotFoundError(f"Missing main control image for stem {stem} under {control_dir}")

            img_txt = os.path.join(images_dir, f"{stem}.txt")
            ctl_txt = os.path.join(control_dir, f"{stem}.txt")
            if os.path.exists(img_txt):
                prompt_file = img_txt
            elif os.path.exists(ctl_txt):
                prompt_file = ctl_txt
            else:
                raise FileNotFoundError(f"Missing prompt file for stem {stem} under {images_dir} or {control_dir}")

            # mask（可选）
            mask_file = _find_mask(images_dir, control_dir, stem)

            samples.append(
                {
                    "image": image_path,
                    "control": controls,
                    "caption": prompt_file,
                    "mask_file": mask_file,
                    "dataset_type": "local",
                    "local_index": n,
                    "global_index": start_idx + n,
                }
            )
            n += 1
        logging.info(f" samples[0]: {samples[0]}")
        return samples

    def __repr__(self) -> str:
        msg = f"""ImageDataset(
            dataset_paths={self.dataset_paths},
        """
        return msg

    def data_key_exist(self, data, key):
        if key in data and data[key] is not None:
            return True
        return False

    def load_data(self, idx):
        if idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.__len__()}")
        sample = self.all_samples[idx]
        data = {}
        if sample['dataset_type'] == 'huggingface':
            local_index = sample['local_index']
            repo_id = sample['repo_id']
            data_item = self.hf_datasets[repo_id]['dataset'][local_index]
            if data_item['target_image'] is not None:
                image = data_item['target_image'].convert('RGB')
                data['image'] = image
            control = data_item['control_images']
            if control is not None:
                data['control'] = control[0].convert('RGB')
                if len(control) > 1:
                    data['controls'] = control[1:]
                    data['controls'] = [img.convert('RGB') for img in data['controls']]
                    if self.selected_control_indexes is not None:
                        data['controls'] = [data['controls'][i-1] for i in self.selected_control_indexes]

            prompt = data_item['prompt']
            data['prompt'] = prompt
            if data_item['control_mask'] is not None:
                data['mask'] = np.array(data_item['control_mask'].convert('L'))

        else:  # loaded locally
            data_item = self.all_samples[idx]
            #  {
            #         "image": image_path,
            #         "control": controls,
            #         "caption": prompt_file,
            #         "mask_file": mask_file,
            #         "dataset_type": "local",
            #         "local_index": n,
            #         "global_index": start_idx + n,
            #     }
            # ) If, not exist return None
            # 读取提示文本
            if self.data_key_exist(data_item, 'image'):
                data['image'] = data_item['image']
            if self.data_key_exist(data_item, 'control'):
                data['control'] = data_item['control'][0]
                if len(data_item['control']) > 1:
                    data['controls'] = data_item['control'][1:]
                    if self.selected_control_indexes is not None:
                        data['controls'] = [data['controls'][i-1] for i in self.selected_control_indexes]
            if self.data_key_exist(data_item, 'mask_file'):
                data['mask'] = cv2.imread(data_item['mask_file'], 0)
            if self.data_key_exist(data_item, 'caption') and data_item['dataset_type'] == 'local':
                with open(data_item['caption'], 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                data['prompt'] = prompt
            else:
                prompt = data_item['caption']
                data['prompt'] = prompt
        return data

    def __getitem__(self, idx):
        """
        获取索引样本。
        Args:
            idx: 样本索引
        Returns:
            dict: 根据缓存状态返回原始图像+文本或缓存嵌入的字典

        示例（非缓存模式）:
        ```python
        item = dataset[0]
        image = item['image']     # (C,H,W) numpy
        control = item['control'] # (C,H,W) numpy
        prompt = item['prompt']   # str
        ```
        """
        data = self.load_data(idx)
        data = self.preprocessor.preprocess(data)
        if 'controls' in data:
            n_controls = len(data['controls'])
            for i in range(n_controls):
                data[f'control_{i+1}'] = data['controls'][i]
            del data['controls']
            data['n_controls'] = n_controls
        else:
            data['n_controls'] = 0
        return data


def pad_to_max_shape(tensors, padding_value=0):
    """
    将一组不同形状但维度数一致的张量沿各维右侧填充到最大形状后堆叠。
    Args:
        tensors: List[Tensor]，形状可能不同
        padding_value: 填充值，默认 0
    Returns:
        Tensor: 形状为 (N, ...) 的批次张量

    示例:
    ```python
    x = [torch.ones(2,3), torch.zeros(4,1)]
    y = pad_to_max_shape(x, padding_value=0)  # -> (2,4,3)
    ```
    """
    # Find maximum shape
    max_shape = [max(sizes) for sizes in zip(*[t.shape for t in tensors])]
    padded = []
    for t in tensors:
        pad_sizes = []
        for i in range(len(max_shape) - 1, -1, -1):
            diff = max_shape[i] - t.shape[i]
            pad_sizes.extend([0, diff])  # (left, right) per dimension
        padded_tensor = F.pad(t, pad_sizes, value=padding_value)
        padded.append(padded_tensor)

    return torch.stack(padded, dim=0)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate 函数，自动对张量右端填充并对嵌套字典递归聚合。
    Args:
        batch: 由 __getitem__ 返回的样本字典组成的列表
    Returns:
        Dict[str, Any]: 已聚合并对齐后的批次字典

    示例:
    ```python
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    ```
    """
    # [{a:1,b:2, c:{d:1,g:2}},{a:3,b:4, c:{e:3,g:4}}] -> {a: [1,3], b: [2,4], c:{d: [1,3], e: [2,4], g: [2,4]}}
    keys = list(batch[0].keys())
    # flattten
    batch_dict = {key: [item[key] for item in batch] for key in keys}
    # if torch tensor, padding to maximal length
    for key in batch_dict:
        if isinstance(batch_dict[key][0], np.ndarray):
            batch_dict[key] = [torch.from_numpy(item) for item in batch_dict[key]]
        if isinstance(batch_dict[key][0], torch.Tensor):
            batch_dict[key] = pad_to_max_shape(batch_dict[key])
        elif isinstance(batch_dict[key][0], dict):
            batch_list = batch_dict[key]
            batch_list = collate_fn(batch_list)
            batch_dict[key] = batch_list
            # [ {d:1,g:2}, {e:3,g:4}] -> {d: [1,3], g: [2,4]}
            # [{a:1,b:2, c:{d:1,g:2}},{a:3,b:4, c:{d:3,g:4}}] -> {a: [1,3], b: [2,4], c:{d: [1,3], g: [2,4]}}
    return batch_dict


def loader(
        class_path: str,
        init_args: dict,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: Optional[int] = 4,
        shuffle: bool = True,
        drop_last=True,
        seed: int = 42,
        teacher_routes: Optional[Dict[str, List[str]]] = None,
        fallback_teacher: Optional[str] = None) -> DataLoader:
    """
    动态加载数据集类并创建 DataLoader。
    Args:
        class_path: 类的完整路径，如 'src.data.dataset.ImageDataset'
        init_args: 用于初始化该类的参数字典（传给其 __init__）
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
    Returns:
        torch.utils.data.DataLoader

    示例:
    ```python
    from src.data.dataset import loader
    dl = loader(
        class_path='src.data.dataset.ImageDataset',
        init_args={'dataset_path': '/data/ds'},
        batch_size=2,
        num_workers=2,
        shuffle=True,
    )
    for batch in dl:
        pass
    ```
    """
    # Also configure direct callers that bypass ``src.main``.  This is
    # idempotent in the parent process; worker_init_fn below reapplies it once
    # per DataLoader child without touching PyTorch/NumPy/Python RNG state.
    configure_opencv_runtime()

    # 解析类路径
    module_path, class_name = class_path.rsplit('.', 1)

    # 动态导入模块
    module = importlib.import_module(module_path)

    # 获取类对象
    dataset_class = getattr(module, class_name)

    # 使用init_args实例化类
    dataset = dataset_class(init_args)
    if teacher_routes is not None:
        from src.distillation.routing import CategoryTeacherRouter, RoundRobinTeacherSampler

        router = CategoryTeacherRouter(teacher_routes, fallback=fallback_teacher)
        sampler = RoundRobinTeacherSampler(dataset, router, seed=seed)
    else:
        sampler = EpochShuffleSampler(dataset, seed=seed) if shuffle else SequentialSampler(dataset)
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = persistent_workers
        worker_kwargs["worker_init_fn"] = configure_opencv_dataloader_worker
        if prefetch_factor is not None:
            worker_kwargs["prefetch_factor"] = prefetch_factor

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **worker_kwargs,
    )
    return dataloader

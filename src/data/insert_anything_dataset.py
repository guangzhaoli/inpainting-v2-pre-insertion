"""
InsertAnythingDataset for the mask_prompt inpainting task.

Directory structure:
    data/
    ├── mask_prompt/  # Optional wrapper directory
    │   ├── accessory/
    │   │   ├── ref_image/      # Reference image containing the element
    │   │   ├── ref_mask/       # Mask for the inserted element
    │   │   ├── tar_image/      # Ground-truth target image
    │   │   └── tar_mask/       # Target editing region mask
    │   ├── object/
    │   └── person/
"""

import csv
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.config import DatasetInitArgs
from src.utils.adaptive_crop import build_adaptive_crop_plan, crop_numpy_pair
from src.utils.images import extract_foreground_from_image_and_mask

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class InsertAnythingDataset(Dataset):
    """
    Dataset for Insert Anything mask_prompt inpainting.

    Returns:
        - ref_image: Reference image containing element to insert [C,H,W]
        - ref_mask: Mask of element in reference image [H,W]
        - tar_image: Target ground-truth image [C,H,W]
        - tar_mask: Mask of editing region in target [H,W]
        - prompt: Text description
        - category: Image category
    """

    def __init__(self, data_config: DatasetInitArgs):
        self.data_config = data_config
        self.dataset_path = data_config.dataset_path
        self.manifest_path = data_config.manifest_path
        self.categories = data_config.categories
        self.require_prompt = data_config.require_prompt
        self.samples = []
        self._load_dataset()
        self.load_processor()
        logging.info(f"Loaded {len(self.samples)} samples from {self.dataset_path}")

    def load_processor(self):
        from src.utils.tools import instantiate_class

        processor_config = self.data_config.processor
        self.preprocessor = instantiate_class(
            processor_config.class_path,
            processor_config.init_args,
        )

    @staticmethod
    def _resolve_mask_prompt_root(root: Path) -> Path:
        mask_prompt_root = root / "mask_prompt"
        return mask_prompt_root if mask_prompt_root.is_dir() else root

    def _load_dataset(self):
        manifest_path = self._resolve_manifest_path()
        if manifest_path is not None and manifest_path.is_file():
            logging.info(f"Loading InsertAnything manifest from {manifest_path}")
            self._load_from_manifest(manifest_path)
        elif manifest_path is not None:
            logging.warning(f"manifest_path not found, falling back to dataset scan: {manifest_path}")
            self._load_dataset_without_manifest()
        else:
            self._load_dataset_without_manifest()

        if len(self.samples) == 0:
            raise ValueError(f"No InsertAnything samples loaded from {self.dataset_path}")

    def _load_dataset_without_manifest(self):
        if self._is_huggingface_repo(self.dataset_path):
            self._load_from_huggingface()
        elif self._is_local_folder_dataset(self.dataset_path):
            logging.info(f"Detected local mask_prompt folder dataset at {self.dataset_path}")
            self._load_from_local()
        elif self._is_local_arrow_dataset(self.dataset_path):
            logging.info(f"Detected local Arrow dataset at {self.dataset_path}")
            self._load_from_local_arrow()
        elif self._is_local_parquet_dataset(self.dataset_path):
            logging.info(f"Detected local Parquet dataset at {self.dataset_path}")
            self._load_from_local_parquet()
        else:
            raise FileNotFoundError(f"Unsupported or missing dataset_path: {self.dataset_path}")

    def _is_local_folder_dataset(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False

        root = Path(path)
        candidate_roots = []
        mask_prompt_root = root / "mask_prompt"
        if mask_prompt_root.is_dir():
            candidate_roots.append(mask_prompt_root)
        candidate_roots.append(root)

        for candidate_root in candidate_roots:
            if not candidate_root.is_dir():
                continue
            for category_dir in candidate_root.iterdir():
                if not category_dir.is_dir():
                    continue
                if (
                    (category_dir / "ref_image").is_dir()
                    and (category_dir / "ref_mask").is_dir()
                    and (category_dir / "tar_image").is_dir()
                    and (category_dir / "tar_mask").is_dir()
                ):
                    return True
        return False

    def _is_local_arrow_dataset(self, path: str) -> bool:
        if not os.path.isdir(path):
            return False
        root = Path(path)
        if (root / "dataset_info.json").exists() and (root / "state.json").exists():
            return True
        return any(root.rglob("*.arrow"))

    def _is_local_parquet_dataset(self, path: str) -> bool:
        return os.path.isdir(path) and any(Path(path).rglob("*.parquet"))

    def _is_huggingface_repo(self, path: str) -> bool:
        from src.utils.huggingface import is_huggingface_repo

        return is_huggingface_repo(path)

    def _load_from_local(self):
        root = self._resolve_mask_prompt_root(Path(self.dataset_path))
        logging.info(f"Loading mask_prompt dataset from {root}")

        category_dirs = [d for d in root.iterdir() if d.is_dir()]
        if self.categories:
            category_dirs = [d for d in category_dirs if d.name in self.categories]

        if len(category_dirs) == 0:
            raise ValueError(f"No category directories found under {root}")

        logging.info(f"Found categories: {[d.name for d in category_dirs]}")
        for category_dir in category_dirs:
            self._scan_category(category_dir, category_dir.name)

    def _resolve_manifest_path(self) -> Optional[Path]:
        if not self.manifest_path:
            return None

        path = Path(self.manifest_path)
        if path.is_absolute():
            return path
        if isinstance(self.dataset_path, str):
            return Path(self.dataset_path) / path
        return path

    def _category_filter(self) -> Optional[set[str]]:
        if not self.categories:
            return None
        if isinstance(self.categories, str):
            return {self.categories}
        return set(self.categories)

    def _load_from_manifest(self, manifest_path: Path):
        category_filter = self._category_filter()
        skipped = 0

        for idx, row in enumerate(self._iter_manifest_rows(manifest_path)):
            sample = self._sample_from_manifest_row(row, idx)
            if category_filter is not None and sample.get("category") not in category_filter:
                skipped += 1
                continue
            self.samples.append(sample)

        logging.info(
            f"Loaded {len(self.samples)} samples from manifest {manifest_path}"
            + (f", skipped {skipped} by categories" if skipped else "")
        )

    def _iter_manifest_rows(self, manifest_path: Path):
        suffix = manifest_path.suffix.lower()
        if suffix == ".jsonl":
            with manifest_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL manifest row {line_no}: {manifest_path}") from exc
            return

        if suffix == ".json":
            with manifest_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rows = data.get("samples") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ValueError("JSON manifest must be a list or an object with a 'samples' list")
            yield from rows
            return

        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with manifest_path.open("r", encoding="utf-8", newline="") as f:
                yield from csv.DictReader(f, delimiter=delimiter)
            return

        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    def _sample_from_manifest_row(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        required = ["ref_image", "ref_mask", "tar_image", "tar_mask"]
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"Manifest sample {idx} is missing required fields: {missing}")

        return {
            "dataset_type": "local_manifest",
            "ref_image": row["ref_image"],
            "ref_mask": row["ref_mask"],
            "tar_image": row["tar_image"],
            "tar_mask": row["tar_mask"],
            "category": row.get("category", "unknown"),
            "stem": row.get("stem", ""),
            "prompt": row.get("prompt", ""),
        }

    def _scan_category(self, category_dir: Path, category: str):
        ref_image_dir = category_dir / "ref_image"
        ref_mask_dir = category_dir / "ref_mask"
        tar_image_dir = category_dir / "tar_image"
        tar_mask_dir = category_dir / "tar_mask"

        for required_dir in (ref_image_dir, ref_mask_dir, tar_image_dir, tar_mask_dir):
            if not required_dir.exists():
                raise FileNotFoundError(f"Missing required directory: {required_dir}")

        ref_images = self._find_images(ref_image_dir)
        logging.info(f"Found {len(ref_images)} samples in category '{category}'")

        for ref_img_path in ref_images:
            stem = ref_img_path.stem
            ref_mask_path = self._find_file_with_stem(ref_mask_dir, stem)
            tar_image_path = self._find_file_with_stem(tar_image_dir, stem)
            tar_mask_path = self._find_file_with_stem(tar_mask_dir, stem)
            if ref_mask_path is None:
                raise FileNotFoundError(f"Missing ref_mask for {stem} in {ref_mask_dir}")
            if tar_image_path is None:
                raise FileNotFoundError(f"Missing tar_image for {stem} in {tar_image_dir}")
            if tar_mask_path is None:
                raise FileNotFoundError(f"Missing tar_mask for {stem} in {tar_mask_dir}")

            prompt_path = tar_image_dir / f"{stem}.txt"
            if prompt_path.exists():
                prompt = prompt_path.read_text(encoding="utf-8").strip()
            elif self.require_prompt:
                raise FileNotFoundError(f"Missing prompt file for {stem} in {tar_image_dir}")
            else:
                prompt = ""

            self.samples.append(
                {
                    "dataset_type": "local",
                    "ref_image": str(ref_img_path),
                    "ref_mask": str(ref_mask_path),
                    "tar_image": str(tar_image_path),
                    "tar_mask": str(tar_mask_path),
                    "category": category,
                    "stem": stem,
                    "prompt": prompt,
                }
            )

    def _find_images(self, directory: Path) -> List[Path]:
        images = []
        for ext in IMG_EXTS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(images)

    def _find_file_with_stem(self, directory: Path, stem: str) -> Optional[Path]:
        for ext in IMG_EXTS:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path
            upper_path = directory / f"{stem}{ext.upper()}"
            if upper_path.exists():
                return upper_path
        return None

    def _load_from_huggingface(self):
        from datasets import DatasetDict, concatenate_datasets
        from src.utils.huggingface import load_editing_dataset

        dataset = load_editing_dataset(self.dataset_path)
        if isinstance(dataset, DatasetDict):
            split_names = list(dataset.keys())
            hf_dataset = concatenate_datasets([dataset[name] for name in split_names])
            logging.info(
                f"Loaded HuggingFace dataset: {self.dataset_path}, merged_splits={split_names}, samples={len(hf_dataset)}"
            )
        else:
            hf_dataset = dataset
            logging.info(f"Loaded HuggingFace dataset: {self.dataset_path}, samples={len(hf_dataset)}")

        for idx in range(len(hf_dataset)):
            item = hf_dataset[idx]
            sample = {
                "dataset_type": "huggingface",
                "hf_index": idx,
                "ref_image": item.get("ref_image"),
                "ref_mask": item.get("ref_mask"),
                "tar_image": item.get("tar_image"),
                "tar_mask": item.get("tar_mask"),
                "category": item.get("category", "unknown"),
                "prompt": item.get("prompt", ""),
            }
            self._validate_sample_info(sample, idx)
            self.samples.append(sample)

        self.hf_dataset = hf_dataset

    def _load_from_local_arrow(self):
        from datasets import DatasetDict, concatenate_datasets, load_from_disk

        dataset = load_from_disk(self.dataset_path)
        if isinstance(dataset, DatasetDict):
            split_names = list(dataset.keys())
            hf_dataset = concatenate_datasets([dataset[name] for name in split_names])
            logging.info(
                f"Loaded local Arrow dataset: {self.dataset_path}, merged_splits={split_names}, samples={len(hf_dataset)}"
            )
        else:
            hf_dataset = dataset
            logging.info(f"Loaded local Arrow dataset: {self.dataset_path}, samples={len(hf_dataset)}")

        for idx in range(len(hf_dataset)):
            self.samples.append({"dataset_type": "local_arrow", "hf_index": idx})

        self.hf_dataset = hf_dataset

    def _load_from_local_parquet(self):
        from datasets import load_dataset

        root = self._resolve_mask_prompt_root(Path(self.dataset_path))
        parquet_files = sorted(root.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {root}")

        hf_dataset = load_dataset(
            "parquet",
            data_files={"data": [str(path) for path in parquet_files]},
        )["data"]
        logging.info(f"Loaded local parquet dataset: {self.dataset_path}, samples={len(hf_dataset)}")

        for idx in range(len(hf_dataset)):
            self.samples.append({"dataset_type": "local_parquet", "hf_index": idx})

        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.samples)

    def _read_image(self, image_path_or_pil):
        import PIL.Image

        if isinstance(image_path_or_pil, str):
            path = image_path_or_pil
            if not os.path.isabs(path):
                path = os.path.join(self.dataset_path, path)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if isinstance(image_path_or_pil, dict):
            image_bytes = image_path_or_pil.get("bytes")
            image_path = image_path_or_pil.get("path")
            if image_bytes is not None:
                img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    with PIL.Image.open(BytesIO(image_bytes)) as pil_img:
                        return np.array(pil_img.convert("RGB"))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if image_path:
                path = image_path if os.path.isabs(image_path) else os.path.join(self.dataset_path, image_path)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to read image: {path}")
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            raise ValueError("Image dict must contain either 'bytes' or 'path'.")
        if isinstance(image_path_or_pil, PIL.Image.Image):
            return np.array(image_path_or_pil.convert("RGB"))
        raise ValueError(f"Unsupported image type: {type(image_path_or_pil)}")

    def _read_mask(self, mask_path_or_pil):
        import PIL.Image

        if isinstance(mask_path_or_pil, str):
            path = mask_path_or_pil
            if not os.path.isabs(path):
                path = os.path.join(self.dataset_path, path)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask: {path}")
            return mask
        if isinstance(mask_path_or_pil, dict):
            mask_bytes = mask_path_or_pil.get("bytes")
            mask_path = mask_path_or_pil.get("path")
            if mask_bytes is not None:
                mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    with PIL.Image.open(BytesIO(mask_bytes)) as pil_mask:
                        return np.array(pil_mask.convert("L"))
                return mask
            if mask_path:
                path = mask_path if os.path.isabs(mask_path) else os.path.join(self.dataset_path, mask_path)
                mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError(f"Failed to read mask: {path}")
                return mask
            raise ValueError("Mask dict must contain either 'bytes' or 'path'.")
        if isinstance(mask_path_or_pil, PIL.Image.Image):
            return np.array(mask_path_or_pil.convert("L"))
        raise ValueError(f"Unsupported mask type: {type(mask_path_or_pil)}")

    def _build_sample_info_from_hf_item(self, idx: int) -> Dict[str, Any]:
        sample_info = dict(self.samples[idx])
        item = self.hf_dataset[sample_info["hf_index"]]
        sample_info["ref_image"] = item.get("ref_image")
        sample_info["ref_mask"] = item.get("ref_mask")
        sample_info["tar_image"] = item.get("tar_image")
        sample_info["tar_mask"] = item.get("tar_mask")
        sample_info["category"] = item.get("category", "unknown")
        sample_info["stem"] = item.get("stem", "")
        sample_info["prompt"] = item.get("prompt", "")
        return sample_info

    def _get_sample_info(self, idx: int) -> Dict[str, Any]:
        sample_info = dict(self.samples[idx])
        dataset_type = sample_info.get("dataset_type", "local")
        if dataset_type in {"huggingface", "local_parquet", "local_arrow"}:
            sample_info = self._build_sample_info_from_hf_item(idx)
        self._validate_sample_info(sample_info, idx)
        return sample_info

    @staticmethod
    def _validate_sample_info(sample_info: Dict[str, Any], idx: int) -> None:
        required = ["ref_image", "ref_mask", "tar_image", "tar_mask"]
        missing = [key for key in required if sample_info.get(key) is None]
        if missing:
            raise ValueError(f"Sample {idx} is missing required fields: {missing}")

    def _describe_sample_source(self, value: Any) -> str:
        """Return a stable path string for diagnostics without retaining image payloads."""

        path = None
        if isinstance(value, str):
            path = value
        elif isinstance(value, dict):
            path = value.get("path")
        if not path:
            return ""
        if (
            not os.path.isabs(path)
            and isinstance(self.dataset_path, str)
            and os.path.isdir(self.dataset_path)
        ):
            path = os.path.join(self.dataset_path, path)
        return os.path.abspath(path) if os.path.isabs(path) or os.path.exists(path) else str(path)

    def _sample_metadata(self, sample_info: Dict[str, Any], idx: int) -> Dict[str, Any]:
        category = str(sample_info.get("category") or "unknown")
        stem = str(sample_info.get("stem") or "")
        if not stem:
            for key in ("tar_image", "ref_image"):
                source_path = self._describe_sample_source(sample_info.get(key))
                if source_path:
                    stem = Path(source_path).stem
                    break
        if not stem:
            stem = str(idx)
        return {
            "dataset_index": int(idx),
            "sample_id": f"{category}/{stem}",
            "category": category,
            "stem": stem,
            "ref_image_path": self._describe_sample_source(sample_info.get("ref_image")),
            "ref_mask_path": self._describe_sample_source(sample_info.get("ref_mask")),
            "tar_image_path": self._describe_sample_source(sample_info.get("tar_image")),
            "tar_mask_path": self._describe_sample_source(sample_info.get("tar_mask")),
        }

    def __getitem__(self, idx):
        sample_metadata = {"dataset_index": int(idx)}
        try:
            sample_info = self._get_sample_info(idx)
            sample_metadata = self._sample_metadata(sample_info, idx)
            return self._load_and_process_sample(sample_info, sample_metadata)
        except Exception as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    "InsertAnythingDataset sample context: "
                    + ", ".join(f"{key}={value}" for key, value in sample_metadata.items() if value != "")
                )
            raise

    def _load_and_process_sample(self, sample_info: Dict[str, Any], sample_metadata: Dict[str, Any]):
        data = {
            "ref_image": self._read_image(sample_info["ref_image"]),
            "ref_mask": self._read_mask(sample_info["ref_mask"]),
            "tar_image": self._read_image(sample_info["tar_image"]),
            "tar_mask": self._read_mask(sample_info["tar_mask"]),
            "prompt": sample_info.get("prompt", ""),
            "category": sample_info.get("category", ""),
            "sample_metadata": sample_metadata,
        }

        adaptive_crop_config = self.data_config.adaptive_crop
        profile = getattr(self.data_config, "category_conditioning", {}).get(data["category"])
        if profile is not None and profile.adaptive_crop is not None:
            adaptive_crop_config = profile.adaptive_crop
        if adaptive_crop_config.train_enabled:
            tar_height, tar_width = data["tar_image"].shape[:2]
            crop_plan = build_adaptive_crop_plan(
                image_size=(tar_width, tar_height),
                mask=data["tar_mask"],
                config=adaptive_crop_config,
            )
            data["tar_image"], data["tar_mask"] = crop_numpy_pair(
                data["tar_image"],
                data["tar_mask"],
                crop_plan,
            )

        data["ref_image"], data["ref_mask"] = extract_foreground_from_image_and_mask(
            data["ref_image"],
            data["ref_mask"],
            padding_ratio=0.15,
            min_size=64,
            align_to=16,
        )

        processed = self.preprocessor.preprocess(
            {
                "image": data["tar_image"],
                "mask": data["tar_mask"],
                "controls": [data["ref_image"]],
                "control_masks": [data["ref_mask"]],
            }
        )

        data["tar_image"] = processed["image"]
        data["tar_mask"] = processed["mask"]
        data["ref_image"] = processed["controls"][0]
        data["ref_mask"] = processed["control_masks"][0]
        if self.data_config.face_identity_supervision:
            from src.data.face_regions import face_region_mask

            data["target_face_mask"] = face_region_mask(data["tar_image"], data["tar_mask"])
            data["reference_face_mask"] = face_region_mask(data["ref_image"], data["ref_mask"])
        return self._convert_to_tensors(data)

    def _convert_to_tensors(self, data):
        for key in ["ref_image", "tar_image"]:
            if key in data and isinstance(data[key], np.ndarray):
                img = data[key].astype(np.float32) / 255.0
                if img.ndim == 2:
                    img = img[np.newaxis, ...]
                elif img.ndim == 3:
                    img = img.transpose(2, 0, 1)
                data[key] = torch.from_numpy(img)

        for key in ["ref_mask", "tar_mask"]:
            if key in data and isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key].astype(np.float32) / 255.0)

        return data

    def __repr__(self):
        return f"InsertAnythingDataset(path={self.dataset_path}, samples={len(self)})"


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch.nn.functional as F

    def pad_to_max_shape(tensors, padding_value=0):
        max_shape = [max(sizes) for sizes in zip(*[t.shape for t in tensors])]
        padded = []
        for tensor in tensors:
            pad_sizes = []
            for i in range(len(max_shape) - 1, -1, -1):
                diff = max_shape[i] - tensor.shape[i]
                pad_sizes.extend([0, diff])
            padded.append(F.pad(tensor, pad_sizes, value=padding_value))
        return torch.stack(padded, dim=0)

    keys = batch[0].keys()
    batch_dict = {}
    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            batch_dict[key] = pad_to_max_shape(values)
        else:
            batch_dict[key] = values
    return batch_dict

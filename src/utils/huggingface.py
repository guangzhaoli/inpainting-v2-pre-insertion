# pip install -U datasets huggingface_hub hf-transfer
# export HUGGINGFACE_HUB_TOKEN=hf_xxx   # or HF_TOKEN
from pathlib import Path
import hashlib
import logging
from typing import Dict, List, Optional
import os
import re
from datasets import Dataset, DatasetDict, Features, Image, Value, Sequence, concatenate_datasets
from huggingface_hub import create_repo, hf_hub_download, HfApi
from src.trainer.constants import LORA_FILE_BASE_NAME

logger = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

FEATURES = Features(
    {
        "id": Value("string"),
        "control_images": Sequence(Image()),  # list of control images (>=1)
        "control_mask": Image(),  # None if absent
        "target_image": Image(),  # None in test if absent
        "prompt": Value("string"),
    }
)

_digit_suffix_re = re.compile(r"_(\d+)$")  # matches ..._<n>

# Insert Anything dataset schema
INSERT_ANYTHING_FEATURES = Features(
    {
        "id": Value("string"),           # Sample ID (stem)
        "ref_image": Image(),            # Reference image with element to insert
        "ref_mask": Image(),             # Mask of element in reference image
        "tar_image": Image(),            # Target ground truth image
        "tar_mask": Image(),             # Editing region mask
        "prompt": Value("string"),       # Text description
        "category": Value("string"),     # accessory, object, person, garment, etc.
    }
)


def _pick_first_existing(base: Path) -> Optional[Path]:
    """Return first existing file among known image extensions for a base path."""
    stem = base.with_suffix("")  # drop any ext
    for ext in _IMG_EXTS:
        p = Path(stem.as_posix() + ext)
        if p.exists():
            return p
        pU = Path(stem.as_posix() + ext.upper())
        if pU.exists():
            return pU
    return None


def _find_control_images(ctrl_dir: Path, base: str) -> List[Path]:
    """
    Collect control images for a basename:
      <base>.* (primary), <base>_control_1.*, <base>_control_2.*, ...
      Excludes <base>_mask.*
      Returns sorted: primary first, then _1, _2, ...
    """
    results: List[Path] = []
    # primary
    primary = _pick_first_existing(ctrl_dir / base)
    if primary:
        results.append(primary)

    # numbered variants by glob then filter
    for ext in _IMG_EXTS:
        for p in sorted(ctrl_dir.glob(f"{base}_*{ext}")):
            stem = p.stem
            if stem.endswith("_mask"):  # exclude mask
                continue
            m = _digit_suffix_re.search(stem[len(base):])
            if m:  # only accept pure numeric suffixes
                results.append(p)

    # natural sort: primary first, then by numeric suffix
    def _order_key(p: Path):
        s = p.stem
        if s == base:
            return (0, 0)
        m = _digit_suffix_re.search(s[len(base):])
        n = int(m.group(1)) if m else 1_000_000
        return (1, n)

    results = sorted(list(dict.fromkeys(results)), key=_order_key)
    return results


def _find_mask(ctrl_dir: Path, base: str) -> Optional[Path]:
    """Return <base>_mask.* if exists, else None."""
    for ext in _IMG_EXTS:
        p = ctrl_dir / f"{base}_mask{ext}"
        if p.exists():
            return p
    return None


def _collect_split(root: Path, split: str) -> Dataset:
    """
    Build a HF Dataset (schema = FEATURES) from:
      root/split/control_images/
      root/split/training_images/
    Requires a prompt file <base>.txt in training_images.
    Target image <base>.* optional (None if missing).
    """
    split_dir = root / split
    ctrl_dir = split_dir / "control_images"
    tgt_dir = split_dir / "training_images"

    if not ctrl_dir.is_dir() or not tgt_dir.is_dir():
        raise FileNotFoundError(
            f"[{split}] missing subfolders: {ctrl_dir} and/or {tgt_dir}"
        )

    txt_files = sorted(tgt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"[{split}] no *.txt prompts in {tgt_dir}")

    rows: List[dict] = []
    for txt in txt_files:
        base = txt.stem
        control_list = _find_control_images(ctrl_dir, base)
        if not control_list:
            raise FileNotFoundError(f"[{split}] no control image found for prompt {txt}")

        mask = _find_mask(ctrl_dir, base)
        tgt = _pick_first_existing(tgt_dir / base)  # optional

        rows.append(
            {
                "id": base,
                "control_images": [str(p) for p in control_list],  # list of paths
                "control_mask": str(mask) if mask is not None else None,
                "target_image": str(tgt) if tgt is not None else None,
                "prompt": txt.read_text(encoding="utf-8").strip(),
            }
        )

    if not rows:
        raise RuntimeError(
            f"[{split}] collected 0 valid items (missing control images?)."
        )
    return Dataset.from_list(rows, features=FEATURES)


def _find_image_with_any_format(root: Path, csv_path: str) -> Optional[Path]:
    """
    根据 CSV 中的路径查找实际存在的图像文件，自动尝试所有支持的格式。

    Args:
        root: 数据集根目录
        csv_path: CSV 中记录的相对路径（可能格式不正确）

    Returns:
        实际存在的图像文件路径，如果未找到则返回 None
    """
    csv_path = Path(csv_path)

    # 首先尝试 CSV 中记录的确切路径
    exact_path = root / csv_path
    if exact_path.exists():
        return exact_path

    # 如果确切路径不存在，尝试所有支持的格式
    stem_path = exact_path.with_suffix("")  # 去掉扩展名

    for ext in _IMG_EXTS:
        # 尝试小写扩展名
        candidate = Path(str(stem_path) + ext)
        if candidate.exists():
            return candidate

        # 尝试大写扩展名
        candidate = Path(str(stem_path) + ext.upper())
        if candidate.exists():
            return candidate

    return None


def _collect_split_from_csv(root: Path, split: str) -> Dataset:
    """
    Build a HF Dataset from CSV metadata file.

    Expected structure:
      root/split.csv - CSV with columns: image,control,control_1,prompt (train)
                       or control,control_1,prompt (test)
      root/ - all image files referenced in CSV

    CSV columns:
    - image: target image path (optional, e.g., train/target/xxx.png)
    - control: main control image path (e.g., train/control/xxx.png)
    - control_1, control_2, ...: additional control images
    - prompt: text prompt
    """
    import pandas as pd

    csv_file = root / f"{split}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # 检测CSV列结构
    required_cols = ["control", "prompt"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file}")

    rows: List[dict] = []

    for idx, row in df.iterrows():
        # 生成ID
        if "image" in df.columns and pd.notna(row["image"]):
            # 从target图像路径生成ID
            base_id = Path(row["image"]).stem
        else:
            # 从control图像路径生成ID
            base_id = Path(row["control"]).stem

        # 收集控制图像 - 自动查找正确格式
        control_images = []

        # 主控制图像
        main_control_path = _find_image_with_any_format(root, row["control"])
        if main_control_path:
            control_images.append(str(main_control_path))

        # 额外控制图像 (control_1, control_2, ...)
        control_idx = 1
        while f"control_{control_idx}" in df.columns:
            if pd.notna(row[f"control_{control_idx}"]):
                extra_control_path = _find_image_with_any_format(root, row[f"control_{control_idx}"])
                if extra_control_path:
                    control_images.append(str(extra_control_path))
            control_idx += 1

        if not control_images:
            raise ValueError(f"CSV row {idx} has no control images")

        # 目标图像 (可选) - 自动查找正确格式
        target_image = None
        if "image" in df.columns and pd.notna(row["image"]):
            target_path = _find_image_with_any_format(root, row["image"])
            if target_path:
                target_image = str(target_path)

        # 查找mask (可选) - 可能在 control 或 target 目录中
        control_mask = None
        control_parent = Path(row['control']).parent
        control_stem = Path(row['control']).stem

        # 构建可能的 mask 路径 - 支持多种图像格式
        potential_masks = []

        # 在 control 目录中查找所有支持的格式
        for ext in _IMG_EXTS:
            potential_masks.extend([
                root / control_parent / f"{control_stem}_mask{ext}",
                root / control_parent / f"{control_stem}_mask{ext.upper()}",
            ])

        # 如果有 target 图像，也在 target 目录中查找
        if "image" in df.columns and pd.notna(row["image"]):
            target_parent = Path(row["image"]).parent
            target_stem = Path(row["image"]).stem
            for ext in _IMG_EXTS:
                potential_masks.extend([
                    root / target_parent / f"{target_stem}_mask{ext}",
                    root / target_parent / f"{target_stem}_mask{ext.upper()}",
                    # 也尝试用 control 的 stem 在 target 目录中查找
                    root / target_parent / f"{control_stem}_mask{ext}",
                    root / target_parent / f"{control_stem}_mask{ext.upper()}",
                ])

        for potential_mask in potential_masks:
            if potential_mask.exists():
                control_mask = str(potential_mask)
                break

        rows.append({
            "id": base_id,
            "control_images": control_images,
            "control_mask": control_mask,
            "target_image": target_image,
            "prompt": str(row["prompt"]).strip(),
        })

    if not rows:
        raise RuntimeError(f"[{split}] No valid samples found in CSV")

    logger.info("[%s] Loaded %s samples from CSV", split, len(rows))
    return Dataset.from_list(rows, features=FEATURES)


def upload_editing_dataset_from_csv(root_dir: str, repo_id: str,
                                    private: bool = True) -> None:
    """
    Build dataset from CSV metadata files and push to HF Hub.

    Expected structure:
      root_dir/
        train.csv - CSV with image,control,control_1,prompt columns
        test.csv  - CSV with control,control_1,prompt columns
        train/target/xxx.png - target images
        train/control/xxx.png - control images
        test/control/xxx.png - control images

    Args:
        root_dir: Directory containing CSV files and images
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        private: Whether to create a private repository
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        msg = "Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment."
        raise EnvironmentError(msg)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    splits: Dict[str, Dataset] = {}

    # 检查并加载CSV文件
    for split in ("train", "test"):
        csv_file = root / f"{split}.csv"
        if csv_file.exists():
            try:
                splits[split] = _collect_split_from_csv(root, split)
            except Exception as e:
                raise RuntimeError(f"Failed to load {split} split: {e}") from e

    if not splits:
        msg = "No valid splits found. Expected train.csv and/or test.csv"
        raise RuntimeError(msg)

    dsd = DatasetDict(splits)

    create_repo(
        repo_id, repo_type="dataset", private=private, exist_ok=True, token=token
    )
    dsd.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload dataset from CSV (multi-control, unified schema)",
    )
    size_info = ', '.join([k+':'+str(len(v)) for k, v in dsd.items()])
    logger.info("Pushed %s | splits=%s | sizes={%s}", repo_id, list(dsd.keys()), size_info)


def upload_editing_dataset(root_dir: str, repo_id: str, private: bool = True) -> None:
    """
    Build {train,test} with unified schema and push to HF Hub (dataset repo).
    Token read from env: HUGGINGFACE_HUB_TOKEN or HF_TOKEN.
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    splits: Dict[str, Dataset] = {}
    for split in ("train", "test"):
        if (root / split).exists():
            splits[split] = _collect_split(root, split)

    if not splits:
        raise RuntimeError(
            "No splits found. Expect ./train and/or ./test under root_dir."
        )

    dsd = DatasetDict(splits)

    create_repo(
        repo_id, repo_type="dataset", private=private, exist_ok=True, token=token
    )
    dsd.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload dataset (multi-control, optional mask, unified schema)",
    )
    logger.info(
        "Pushed %s | splits=%s | sizes={ %s }",
        repo_id,
        list(dsd.keys()),
        ', '.join([k+':'+str(len(v)) for k, v in dsd.items()]),
    )


def load_editing_dataset(repo_id: str, split: Optional[str] = None):
    """
    Load back from Hub. Works for private repos via env token.
    Returns DatasetDict if split=None, else a single Dataset.
    """
    from datasets import load_dataset

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    kwargs = {"token": token} if token else {}
    if split is None:
        return load_dataset(repo_id, **kwargs)
    return load_dataset(repo_id, split=split, **kwargs)


def is_huggingface_repo(path: str) -> bool:
    """
    Detect if the path is a Hugging Face repository ID.

    HF repo patterns:
    - username/dataset-name
    - organization/dataset-name
    - dataset-name (for datasets without namespace)

    Local path patterns:
    - /absolute/path/to/dataset
    - ./relative/path
    - ../relative/path
    - path/without/leading/slash (treated as relative)
    """
    # Check if it's an absolute path
    if os.path.isabs(path):
        return False

    # Check if it's a relative path with ./ or ../
    if path.startswith('./') or path.startswith('../'):
        return False

    # Check if the path exists locally
    if os.path.exists(path):
        return False

    # Check HF repo pattern: should contain at most one '/'
    # and not contain invalid characters for repo names
    parts = path.split('/')
    if len(parts) <= 2 and all(part.replace('-', '').replace('_', '').isalnum() for part in parts):

        from datasets import get_dataset_config_names
        try:
            a = get_dataset_config_names(path)
            if a:
                return True
        except Exception:
            return False
    return False


def download_lora(
        repo_id='TsienDragon/qwen-image-edit-lora-face-segmentation',
        filename='pytorch_lora_weights.safetensors'):
    return hf_hub_download(repo_id=repo_id, filename=filename)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_sha_index(api: HfApi, repo_id: str) -> Dict[str, str]:
    """返回 {path_in_repo: sha256_hex}（自动处理 LFS或非LFS）"""
    idx = {}
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    for s in getattr(info, "siblings", []) or []:
        if getattr(s, "lfs", None) and "oid" in s.lfs and str(s.lfs["oid"]).startswith("sha256:"):
            idx[s.rfilename] = s.lfs["oid"].split(":", 1)[1]
        elif getattr(s, "sha", None):  # 非LFS少见
            idx[s.rfilename] = s.sha
    return idx



def upload_lora_safetensors(
    src_path: str,                 # 本地 .safetensors 文件路径，或包含它的目录
    repo_id: str,                  # "user_or_org/repo-name"
    *,
    token: Optional[str] = None,   # 不传则读 HF_TOKEN 或 HUGGINGFACE_HUB_TOKEN
    private: bool = True,
    remote_name: Optional[str] = None,  # 远端文件名；默认仍叫 pytorch_lora_weights.safetensors
    commit_message: str = "Upload LoRA weights",
    extra_files: Optional[Dict[str, str]] = None,  # {本地路径: 远端路径}
) -> str:
    """
    只上传 LoRA 权重文件到 Hugging Face Hub（仓库类型为 model）。
    若远端已有同名且内容一致则跳过。返回仓库的 Web URL。
    """
    token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise ValueError("Missing token. Set HF_TOKEN or pass token=...")

    # 1) 解析本地文件路径：支持传目录或文件
    if os.path.isdir(src_path):
        local_file = os.path.join(src_path, LORA_FILE_BASE_NAME)
    else:
        local_file = src_path

    if not os.path.isfile(local_file):
        raise FileNotFoundError(f"LoRA file not found: {local_file}")
    if not local_file.endswith(".safetensors"):
        raise ValueError("Expect a .safetensors file")

    # 2) 远端文件名
    remote_file = remote_name or LORA_FILE_BASE_NAME

    # 3) 创建仓库（存在即复用）
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    api = HfApi(token=token)

    # 4) 构建远端文件→sha 索引，用于跳过相同文件
    remote_idx = _remote_sha_index(api, repo_id)

    # 5) 主 LoRA 文件：仅在不存在或内容不同时上传
    local_sha = _sha256(local_file)
    if remote_idx.get(remote_file) != local_sha:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=remote_file,         # 可用 "lora/xxx.safetensors" 放子目录
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

    # 6) 可选：额外文件（README / train_config.yaml / adapter_config.json 等）
    if extra_files:
        for lp, rp in extra_files.items():
            if not os.path.isfile(lp):
                raise FileNotFoundError(f"Extra file not found: {lp}")
            sha = _sha256(lp)
            # 若远端已有且内容一致，则跳过
            if remote_idx.get(rp) == sha:
                continue
            api.upload_file(
                path_or_fileobj=lp,
                path_in_repo=rp,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Add/Update {os.path.basename(rp)}",
            )

    # 7) 返回网页地址（不要再取 .html_url）
    return f"https://huggingface.co/{repo_id}"

def _resolve_insert_anything_root(root: Path) -> Path:
    mask_prompt_root = root / "mask_prompt"
    return mask_prompt_root if mask_prompt_root.exists() else root


def _collect_insert_anything_dataset(root: Path, require_prompt: bool = True) -> Dataset:
    """
    Build a HF Dataset for Insert Anything from:
      root/category/ref_image/
      root/category/ref_mask/
      root/category/tar_image/
      root/category/tar_mask/

    Returns:
        Dataset with INSERT_ANYTHING_FEATURES schema
    """
    dataset_root = _resolve_insert_anything_root(root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    # Get all category directories
    category_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]
    if not category_dirs:
        raise FileNotFoundError(f"No category directories found in {dataset_root}")

    rows: List[dict] = []
    for category_dir in category_dirs:
        category = category_dir.name

        ref_image_dir = category_dir / 'ref_image'
        ref_mask_dir = category_dir / 'ref_mask'
        tar_image_dir = category_dir / 'tar_image'
        tar_mask_dir = category_dir / 'tar_mask'

        if (
            not ref_image_dir.exists()
            or not ref_mask_dir.exists()
            or not tar_image_dir.exists()
            or not tar_mask_dir.exists()
        ):
            raise FileNotFoundError(f"Missing required directories in {category_dir}")

        # Find all reference images
        ref_images = []
        for ext in _IMG_EXTS:
            ref_images.extend(ref_image_dir.glob(f"*{ext}"))
            ref_images.extend(ref_image_dir.glob(f"*{ext.upper()}"))

        logger.info("[%s] Found %s samples", category, len(ref_images))

        for ref_img_path in ref_images:
            stem = ref_img_path.stem

            # Find required files
            ref_mask = _pick_first_existing(ref_mask_dir / stem)
            tar_image = _pick_first_existing(tar_image_dir / stem)
            tar_mask = _pick_first_existing(tar_mask_dir / stem)

            if not ref_mask or not tar_image or not tar_mask:
                raise FileNotFoundError(f"Missing ref_mask, tar_image, or tar_mask for {stem} in {category_dir}")

            # Load prompt
            prompt_path = tar_image_dir / f"{stem}.txt"
            if prompt_path.exists():
                prompt = prompt_path.read_text(encoding='utf-8').strip()
            elif require_prompt:
                raise FileNotFoundError(f"Missing prompt file for {stem} in {tar_image_dir}")
            else:
                prompt = ""

            rows.append(
                {
                    "id": stem,
                    "ref_image": str(ref_img_path),
                    "ref_mask": str(ref_mask),
                    "tar_image": str(tar_image),
                    "tar_mask": str(tar_mask),
                    "prompt": prompt,
                    "category": category,
                }
            )

    if not rows:
        raise RuntimeError("Collected 0 valid Insert Anything items")

    return Dataset.from_list(rows, features=INSERT_ANYTHING_FEATURES)


def upload_insert_anything_dataset(
    root_dir: str,
    repo_id: str,
    private: bool = True,
    require_prompt: bool = True,
) -> None:
    """
    Build Insert Anything dataset and push to HF Hub.

    Expected structure:
      root_dir/
        mask_prompt/  # optional wrapper
          category/
            ref_image/
            ref_mask/
            tar_image/
            tar_mask/

    Args:
        root_dir: Root directory, optionally containing a mask_prompt wrapper directory
        repo_id: HuggingFace repository ID (e.g., "username/insert-anything-dataset")
        private: Whether to create a private repository
        require_prompt: Whether each tar_image stem must have a matching .txt prompt

    Example:
        >>> upload_insert_anything_dataset(
        ...     '/path/to/data',
        ...     'username/insert-anything-dataset',
        ...     private=True
        ... )
    """
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HUGGINGFACE_HUB_TOKEN or HF_TOKEN in environment.")

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    root = Path(root_dir)
    dataset = _collect_insert_anything_dataset(root, require_prompt=require_prompt)

    # Create repo and push
    create_repo(
        repo_id, repo_type="dataset", private=private, exist_ok=True, token=token
    )
    dataset.push_to_hub(
        repo_id,
        private=private,
        token=token,
        commit_message="upload Insert Anything dataset",
    )

    logger.info("Pushed %s | samples=%s", repo_id, len(dataset))


def load_insert_anything_dataset(repo_id: str):
    """
    Load Insert Anything dataset from Hub.

    Args:
        repo_id: HuggingFace repository ID

    Returns:
        A single Dataset merged from all available rows

    Example:
        >>> dataset = load_insert_anything_dataset('username/insert-anything')
        >>> dataset[0].keys()
        dict_keys(['id', 'ref_image', 'ref_mask', 'tar_image', 'tar_mask',
                   'prompt', 'category'])
    """
    dataset = load_editing_dataset(repo_id)
    if isinstance(dataset, DatasetDict):
        split_names = list(dataset.keys())
        return concatenate_datasets([dataset[name] for name in split_names])
    return dataset

"""Masked reference preprocessing and DINOv2 feature extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CribBuildConfig
from .io_utils import SampleRecord, load_mask, load_rgb_image, mask_area_ratio

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PreparedCrop:
    sample_uid: str
    tensor: torch.Tensor  # [3,H,W] float32 normalized
    mask_area_ratio: float
    bbox: Tuple[int, int, int, int]  # x0,y0,x1,y1 in original image
    used_full_frame_fallback: bool


def _binary_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return m >= threshold


def mask_bbox(binary_mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    height: int,
    width: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    pad_x = int(round(bw * pad_ratio))
    pad_y = int(round(bh * pad_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)
    return x0, y0, x1, y1


def prepare_masked_ref_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    image_size: int,
    mask_threshold: float = 0.5,
    bbox_pad_ratio: float = 0.15,
    min_mask_pixels: int = 16,
    fill_value: Tuple[int, int, int] = (127, 127, 127),
) -> Tuple[np.ndarray, float, Tuple[int, int, int, int], bool]:
    """Return uint8 RGB crop with non-object pixels filled.

    Fallback: if mask is empty / tiny, use full-frame image.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3, got {image_rgb.shape}")
    if mask.shape[:2] != image_rgb.shape[:2]:
        mask = cv2.resize(
            mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    binary = _binary_mask(mask, mask_threshold)
    area = float(binary.mean())
    bbox = mask_bbox(binary)
    used_fallback = False

    if bbox is None or int(binary.sum()) < min_mask_pixels:
        used_fallback = True
        crop = image_rgb.copy()
        bbox = (0, 0, image_rgb.shape[1], image_rgb.shape[0])
    else:
        bbox = expand_bbox(bbox, image_rgb.shape[0], image_rgb.shape[1], bbox_pad_ratio)
        x0, y0, x1, y1 = bbox
        crop_img = image_rgb[y0:y1, x0:x1].copy()
        crop_mask = binary[y0:y1, x0:x1]
        fill = np.full_like(crop_img, fill_value, dtype=np.uint8)
        crop = np.where(crop_mask[..., None], crop_img, fill)

    crop_resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return crop_resized, area, bbox, used_fallback


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MaskedRefDataset(Dataset):
    def __init__(
        self,
        records: Sequence[SampleRecord],
        dataset_root: Path,
        config: CribBuildConfig,
    ):
        self.records = list(records)
        self.dataset_root = Path(dataset_root)
        self.config = config
        self.transform = build_image_transform(config.image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = load_rgb_image(rec.resolve(self.dataset_root, "ref_image"))
        mask = load_mask(rec.resolve(self.dataset_root, "ref_mask"))
        crop_u8, area, bbox, fallback = prepare_masked_ref_crop(
            image,
            mask,
            image_size=self.config.image_size,
            mask_threshold=self.config.mask_threshold,
            bbox_pad_ratio=self.config.bbox_pad_ratio,
            min_mask_pixels=self.config.min_mask_pixels,
        )
        tensor = self.transform(crop_u8)
        return {
            "sample_uid": rec.sample_uid,
            "tensor": tensor,
            "mask_area_ratio": torch.tensor(area, dtype=torch.float32),
            "bbox": torch.tensor(bbox, dtype=torch.int32),
            "used_full_frame_fallback": torch.tensor(fallback, dtype=torch.bool),
            "index": torch.tensor(idx, dtype=torch.int64),
        }


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_dinov2(model_name: str = "dinov2_vits14", device: str | torch.device = "cpu"):
    """Load DINOv2 backbone via torch.hub.

    Supported names: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    """
    device = resolve_device(device) if isinstance(device, str) else device
    LOGGER.info("Loading DINOv2 model=%s device=%s", model_name, device)
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model.eval()
    model.to(device)
    return model, device


@torch.inference_mode()
def extract_embeddings(
    records: Sequence[SampleRecord],
    dataset_root: Path,
    config: CribBuildConfig,
    model=None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Extract L2-normalized DINOv2 embeddings for all records.

    Returns
    -------
    embeddings : np.ndarray, shape [N, D], float32, L2-normalized
    meta_rows  : per-sample feature metadata
    """
    if model is None or device is None:
        model, device = load_dinov2(config.model_name, config.device)
    else:
        device = resolve_device(device) if isinstance(device, str) else device

    dataset = MaskedRefDataset(records, dataset_root, config)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_emb: List[np.ndarray] = []
    meta_by_index: dict[int, dict] = {}

    for batch in loader:
        images = batch["tensor"].to(device, non_blocking=True)
        # DINOv2 forward returns [B, D] pooled features for hub models.
        feats = model(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        feats = F.normalize(feats.float(), dim=-1)
        all_emb.append(feats.cpu().numpy().astype(np.float32))

        indices = batch["index"].tolist()
        for i, idx in enumerate(indices):
            meta_by_index[int(idx)] = {
                "sample_uid": batch["sample_uid"][i],
                "mask_area_ratio": float(batch["mask_area_ratio"][i].item()),
                "bbox": [int(v) for v in batch["bbox"][i].tolist()],
                "used_full_frame_fallback": bool(batch["used_full_frame_fallback"][i].item()),
            }

    embeddings = np.concatenate(all_emb, axis=0)
    if embeddings.shape[0] != len(records):
        raise RuntimeError(
            f"embedding count mismatch: got {embeddings.shape[0]} vs {len(records)} records"
        )

    # Re-normalize numerically for safety.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    meta_rows = [meta_by_index[i] for i in range(len(records))]
    return embeddings, meta_rows


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, eps, None)

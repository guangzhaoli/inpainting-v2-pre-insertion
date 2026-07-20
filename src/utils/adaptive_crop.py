"""Strict adaptive target-space cropping shared by training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AdaptiveCropPlan:
    applied: bool
    decision_reason: str
    original_size: Tuple[int, int]
    mask_bbox: Tuple[int, int, int, int]
    crop_box: Tuple[int, int, int, int]
    mask_area_pixels: int
    mask_area_ratio: float
    bbox_area_ratio: float
    bbox_fill_ratio: float
    requested_crop_area_ratio: float
    actual_crop_area_ratio: float
    mask_threshold: float
    min_crop_side: int
    effective_zoom_x: float
    effective_zoom_y: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("original_size", "mask_bbox", "crop_box"):
            payload[key] = list(payload[key])
        x1, y1, x2, y2 = self.crop_box
        payload["enabled"] = True
        payload["crop_size"] = [x2 - x1, y2 - y1]
        return payload


def _mask_array(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(
            "adaptive crop requires a single-channel 2D mask; "
            f"convert multi-channel masks explicitly, got shape={array.shape}"
        )
    if np.issubdtype(array.dtype, np.bool_):
        return array
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or float(array.min()) < 0 or float(array.max()) > 1:
            raise ValueError("floating adaptive crop masks must contain finite values in [0, 1]")
        return array
    if array.dtype != np.uint8:
        raise TypeError(
            "integer adaptive crop masks must use uint8 values in {0,1} or [0,255], "
            f"got dtype={array.dtype}"
        )
    return array


def _binary_mask(mask: Image.Image | np.ndarray, threshold: float) -> np.ndarray:
    array = _mask_array(mask)
    max_value = float(array.max())
    threshold_value = threshold if max_value <= 1.0 else 255 * threshold
    return array > threshold_value


def build_adaptive_crop_plan(
    image_size: Tuple[int, int],
    mask: Image.Image | np.ndarray,
    config,
) -> AdaptiveCropPlan:
    """Build one deterministic crop box in original image coordinates.

    ``image_size`` follows PIL convention ``(width, height)``.
    """

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"adaptive crop requires positive image size, got {(width, height)}")

    min_crop_side = int(config.min_crop_side)
    if min_crop_side <= 0:
        raise ValueError("adaptive crop min_crop_side must be positive")
    if width < min_crop_side or height < min_crop_side:
        raise ValueError(
            "adaptive crop requires both source dimensions to be at least "
            f"min_crop_side={min_crop_side}, got {(width, height)}"
        )

    mask_array = _mask_array(mask)
    if mask_array.shape != (height, width):
        raise ValueError(
            "adaptive crop requires image and mask to have identical geometry, "
            f"got image={(width, height)} and mask={(mask_array.shape[1], mask_array.shape[0])}"
        )

    mask_threshold = float(config.mask_threshold)
    binary = _binary_mask(mask_array, mask_threshold)
    occupied_rows = np.flatnonzero(np.any(binary, axis=1))
    occupied_columns = np.flatnonzero(np.any(binary, axis=0))
    if occupied_rows.size == 0 or occupied_columns.size == 0:
        raise ValueError("adaptive crop requires a non-empty target/source mask")

    x1 = int(occupied_columns[0])
    x2 = int(occupied_columns[-1]) + 1
    y1 = int(occupied_rows[0])
    y2 = int(occupied_rows[-1]) + 1
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(f"adaptive crop produced invalid mask bbox {(x1, y1, x2, y2)}")

    image_area = width * height
    mask_area_pixels = int(binary.sum())
    bbox_area = bbox_width * bbox_height
    mask_area_ratio = mask_area_pixels / image_area
    bbox_area_ratio = bbox_area / image_area
    bbox_fill_ratio = mask_area_pixels / bbox_area

    trigger_ratio = float(config.trigger_bbox_area_ratio)
    min_crop_ratio = float(config.min_crop_area_ratio)
    if bbox_area_ratio >= trigger_ratio:
        requested_crop_ratio = 1.0
        initial_reason = "mask_bbox_large_enough"
    else:
        requested_crop_ratio = min_crop_ratio + (1.0 - min_crop_ratio) * bbox_area_ratio / trigger_ratio
        initial_reason = "adaptive_crop"

    margin = float(config.context_margin_ratio)
    expanded_bbox_width = bbox_width * (1.0 + 2.0 * margin)
    expanded_bbox_height = bbox_height * (1.0 + 2.0 * margin)
    requested_scale = math.sqrt(requested_crop_ratio)
    required_scale = max(
        expanded_bbox_width / width,
        expanded_bbox_height / height,
        min_crop_side / width,
        min_crop_side / height,
    )
    crop_scale = min(1.0, max(requested_scale, required_scale))

    crop_width = min(
        width,
        max(min_crop_side, bbox_width, int(math.ceil(width * crop_scale))),
    )
    crop_height = min(
        height,
        max(min_crop_side, bbox_height, int(math.ceil(height * crop_scale))),
    )
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(f"adaptive crop produced invalid crop size {(crop_width, crop_height)}")

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - crop_width / 2.0))
    crop_y1 = int(round(center_y - crop_height / 2.0))
    crop_x1 = min(max(crop_x1, 0), width - crop_width)
    crop_y1 = min(max(crop_y1, 0), height - crop_height)
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    if not (crop_x1 <= x1 and crop_y1 <= y1 and crop_x2 >= x2 and crop_y2 >= y2):
        raise ValueError(
            "adaptive crop failed to contain the complete target/source mask bbox: "
            f"bbox={(x1, y1, x2, y2)}, crop={(crop_x1, crop_y1, crop_x2, crop_y2)}"
        )

    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
    full_box = (0, 0, width, height)
    applied = crop_box != full_box
    if applied:
        decision_reason = initial_reason
    elif initial_reason == "adaptive_crop":
        decision_reason = "geometry_requires_full_frame"
    else:
        decision_reason = initial_reason

    actual_crop_ratio = (crop_width * crop_height) / image_area
    return AdaptiveCropPlan(
        applied=applied,
        decision_reason=decision_reason,
        original_size=(width, height),
        mask_bbox=(x1, y1, x2, y2),
        crop_box=crop_box,
        mask_area_pixels=mask_area_pixels,
        mask_area_ratio=mask_area_ratio,
        bbox_area_ratio=bbox_area_ratio,
        bbox_fill_ratio=bbox_fill_ratio,
        requested_crop_area_ratio=requested_crop_ratio,
        actual_crop_area_ratio=actual_crop_ratio,
        mask_threshold=mask_threshold,
        min_crop_side=min_crop_side,
        effective_zoom_x=width / crop_width,
        effective_zoom_y=height / crop_height,
    )


def crop_numpy_pair(
    image: np.ndarray,
    mask: np.ndarray,
    plan: AdaptiveCropPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a NumPy pair and return a binary uint8 mask using the plan threshold."""
    image_array = np.asarray(image)
    mask_array = np.asarray(mask)
    width, height = plan.original_size
    if image_array.shape[:2] != (height, width) or mask_array.shape[:2] != (height, width):
        raise ValueError(
            "adaptive crop plan geometry does not match numpy image/mask: "
            f"plan={(width, height)}, image={image_array.shape[:2][::-1]}, mask={mask_array.shape[:2][::-1]}"
        )
    x1, y1, x2, y2 = plan.crop_box
    cropped_image = image_array[y1:y2, x1:x2]
    binary_mask = _binary_mask(mask_array, plan.mask_threshold)
    cropped_mask = binary_mask[y1:y2, x1:x2].astype(np.uint8) * 255
    if cropped_image.size == 0 or cropped_mask.size == 0:
        raise ValueError(f"adaptive crop produced an empty numpy crop for box={plan.crop_box}")
    return np.ascontiguousarray(cropped_image), np.ascontiguousarray(cropped_mask)


def crop_pil_mask(
    mask: Image.Image,
    plan: AdaptiveCropPlan,
) -> Image.Image:
    """Crop a PIL mask and return a binary L mask using the plan threshold."""
    if mask.size != plan.original_size:
        raise ValueError(
            "adaptive crop plan geometry does not match PIL mask: "
            f"plan={plan.original_size}, mask={mask.size}"
        )
    binary_mask = Image.fromarray(
        (_binary_mask(mask, plan.mask_threshold).astype(np.uint8) * 255)
    )
    return binary_mask.crop(plan.crop_box)


def crop_pil_pair(
    image: Image.Image,
    mask: Image.Image,
    plan: AdaptiveCropPlan,
) -> tuple[Image.Image, Image.Image]:
    """Crop a PIL pair and return a binary L mask using the plan threshold."""
    if image.size != plan.original_size or mask.size != plan.original_size:
        raise ValueError(
            "adaptive crop plan geometry does not match PIL image/mask: "
            f"plan={plan.original_size}, image={image.size}, mask={mask.size}"
        )
    return image.crop(plan.crop_box), crop_pil_mask(mask, plan)


def restore_pil_output(
    original_source: Image.Image,
    original_mask: Image.Image,
    local_output: Image.Image,
    plan: AdaptiveCropPlan,
    blend_mode: str,
    *,
    edge_feather_px: int = 0,
) -> tuple[Image.Image, Image.Image]:
    if blend_mode != "strict_mask":
        raise ValueError(f"unsupported adaptive crop blend_mode={blend_mode!r}")

    source = original_source.convert("RGB")
    mask = original_mask.convert("L")
    if source.size != plan.original_size or mask.size != plan.original_size:
        raise ValueError(
            "adaptive crop restore geometry mismatch: "
            f"plan={plan.original_size}, source={source.size}, mask={mask.size}"
        )

    # Local import avoids a circular dependency with adaptive_crop_anchoring.
    from src.utils.adaptive_crop_anchoring import paste_local_output_to_original

    raw_restored = paste_local_output_to_original(
        source,
        local_output,
        plan,
        edge_feather_px=0,
    )
    if int(edge_feather_px) > 0:
        restored = paste_local_output_to_original(
            source,
            local_output,
            plan,
            edge_feather_px=int(edge_feather_px),
        )
    else:
        restored = raw_restored
    binary_mask = Image.fromarray(
        (_binary_mask(mask, plan.mask_threshold).astype(np.uint8) * 255)
    )
    # Strict composite always uses hard-paste restored content inside M_core.
    composite = Image.composite(raw_restored, source, binary_mask)
    return restored, composite

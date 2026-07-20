"""Training-only Hybrid Masking adapted from the official Insert Anything code.

The official implementation samples category-dependent conditioning masks:

* person: 40% Bezier augmented box, 40% expanded box, 20% dilated mask
* non-person: 50% Bezier augmented box, 50% expanded box

This local implementation preserves those mask families and probabilities while
adding two safety adaptations for the Qwen training pipeline:

1. pixel-valued parameters can scale relative to the official 768 resolution;
2. generated conditioning masks can be unioned with the GT mask so constructing
   a hole from the completed target image cannot leak GT object pixels.
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


INSERT_ANYTHING_AUGMENTED_BOX = "augmented_box"
INSERT_ANYTHING_EXPANDED_BOX = "expanded_box"
INSERT_ANYTHING_DILATED_SEGMENTATION = "dilated_segmentation"


def _binary_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Hybrid Masking expects a 2D mask, got shape={array.shape}")
    binary = array > threshold
    if not np.any(binary):
        raise ValueError("Hybrid Masking requires a non-empty target mask")
    return binary.astype(np.uint8)


def _tight_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        raise ValueError("Hybrid Masking requires a non-empty target mask")
    # Match the official code: the tight bbox maxima are inclusive.
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


def _expand_bbox(
    image_shape: Tuple[int, int],
    bbox: Tuple[int, int, int, int],
    ratio: float,
) -> Tuple[int, int, int, int]:
    height, width = image_shape
    y1, y2, x1, x2 = bbox
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    expanded_h = ratio * (y2 - y1 + 1)
    expanded_w = ratio * (x2 - x1 + 1)

    expanded_x1 = max(0, int(center_x - expanded_w * 0.5))
    expanded_x2 = min(width, int(center_x + expanded_w * 0.5))
    expanded_y1 = max(0, int(center_y - expanded_h * 0.5))
    expanded_y2 = min(height, int(center_y + expanded_h * 0.5))

    # The official slicing is end-exclusive. Keep at least one pixel for tiny
    # masks that touch the image boundary.
    expanded_x2 = min(width, max(expanded_x2, expanded_x1 + 1))
    expanded_y2 = min(height, max(expanded_y2, expanded_y1 + 1))
    return expanded_y1, expanded_y2, expanded_x1, expanded_x2


def _quadratic_bezier(
    start: Tuple[float, float],
    control: Tuple[float, float],
    end: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    one_minus_t = 1.0 - t
    x = one_minus_t**2 * start[0] + 2.0 * one_minus_t * t * control[0] + t**2 * end[0]
    y = one_minus_t**2 * start[1] + 2.0 * one_minus_t * t * control[1] + t**2 * end[1]
    return x, y


def _scaled_pixel_parameter(value: int, height: int, width: int, config) -> int:
    if value <= 0:
        return 0
    if not config.scale_pixel_parameters:
        return value
    scale = min(height, width) / float(config.reference_resolution)
    return max(1, int(round(value * scale)))


def _augmented_box_mask(
    mask: np.ndarray,
    config,
    rng,
) -> np.ndarray:
    height, width = mask.shape
    y1, y2, x1, x2 = _tight_bbox(mask)
    expanded_y1, expanded_y2, expanded_x1, expanded_x2 = _expand_bbox(
        (height, width),
        (y1, y2, x1, x2),
        config.bbox_expand_ratio,
    )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    curves = (
        ((x1, y1), (center_x, expanded_y1), (x2, y1)),
        ((x2, y1), (expanded_x2, center_y), (x2, y2)),
        ((x2, y2), (center_x, expanded_y2), (x1, y2)),
        ((x1, y2), (expanded_x1, center_y), (x1, y1)),
    )
    jitter = _scaled_pixel_parameter(config.curve_jitter_px, height, width, config)

    points: List[Tuple[float, float]] = []
    for start, control, end in curves:
        seen = set()
        # Match official samples t=0.05, 0.10, ..., 0.90.
        for index in range(1, 19):
            original_x, original_y = _quadratic_bezier(start, control, end, index * 0.05)
            offset_x = rng.randint(-jitter, jitter) if jitter > 0 else 0
            offset_y = rng.randint(-jitter, jitter) if jitter > 0 else 0
            x = original_x + offset_x
            y = original_y + offset_y

            # Preserve the official behavior: a jittered coordinate is retained
            # only when it moves outside the original tight bbox.
            if not (x < x1 or x > x2):
                x = original_x
            if not (y < y1 or y > y2):
                y = original_y

            point = (x, y)
            if point not in seen:
                points.append(point)
                seen.add(point)

    if len(points) < 3:
        raise ValueError("Hybrid Masking augmented box produced fewer than three polygon points")

    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=np.uint8)


def _expanded_box_mask(mask: np.ndarray, config) -> np.ndarray:
    height, width = mask.shape
    expanded_y1, expanded_y2, expanded_x1, expanded_x2 = _expand_bbox(
        (height, width),
        _tight_bbox(mask),
        config.bbox_expand_ratio,
    )
    output = np.zeros_like(mask, dtype=np.uint8)
    output[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = 1
    return output


def _dilated_segmentation_mask(mask: np.ndarray, config) -> np.ndarray:
    height, width = mask.shape
    kernel_size = _scaled_pixel_parameter(config.dilation_kernel_size, height, width, config)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=config.dilation_iterations).astype(np.uint8)


def sample_insert_anything_mask_mode(category: str, config, rng=None) -> str:
    """Sample the exact category-dependent mode probabilities from official code."""

    rng = rng or random
    probability = rng.random()
    if category in config.person_categories:
        if probability <= 0.4:
            return INSERT_ANYTHING_AUGMENTED_BOX
        if probability <= 0.8:
            return INSERT_ANYTHING_EXPANDED_BOX
        return INSERT_ANYTHING_DILATED_SEGMENTATION

    if probability <= 0.5:
        return INSERT_ANYTHING_AUGMENTED_BOX
    return INSERT_ANYTHING_EXPANDED_BOX


def build_insert_anything_mask_for_mode(
    mask: np.ndarray,
    mode: str,
    config,
    rng=None,
) -> np.ndarray:
    """Build one conditioning mask for a preselected official Hybrid Masking mode."""

    rng = rng or random
    gt_mask = _binary_mask(mask, config.mask_threshold)
    if mode == INSERT_ANYTHING_AUGMENTED_BOX:
        generated = _augmented_box_mask(gt_mask, config, rng)
    elif mode == INSERT_ANYTHING_EXPANDED_BOX:
        generated = _expanded_box_mask(gt_mask, config)
    elif mode == INSERT_ANYTHING_DILATED_SEGMENTATION:
        generated = _dilated_segmentation_mask(gt_mask, config)
    else:
        raise ValueError(f"Unsupported Insert Anything Hybrid Masking mode: {mode}")

    generated = (generated > 0).astype(np.uint8)
    if config.ensure_gt_coverage:
        generated = np.maximum(generated, gt_mask)
    return generated


def build_insert_anything_hybrid_masks(
    masks: torch.Tensor,
    categories: Sequence[str] | str,
    config,
    rng=None,
) -> Tuple[torch.Tensor, List[str]]:
    """Apply official Hybrid Masking independently to every sample in a batch."""

    if masks.ndim != 3:
        raise ValueError(f"Hybrid Masking expects [B, H, W], got shape={tuple(masks.shape)}")
    if isinstance(categories, str):
        category_list = [categories] * masks.shape[0]
    else:
        category_list = list(categories)
    if len(category_list) != masks.shape[0]:
        raise ValueError(
            "Hybrid Masking category count must match mask batch size: "
            f"categories={len(category_list)}, batch={masks.shape[0]}"
        )

    rng = rng or random
    output_masks = []
    modes = []
    masks_cpu = masks.detach().float().cpu().numpy()
    for mask, category in zip(masks_cpu, category_list):
        mode = sample_insert_anything_mask_mode(str(category), config, rng=rng)
        output = build_insert_anything_mask_for_mode(mask, mode, config, rng=rng)
        output_masks.append(torch.from_numpy(output))
        modes.append(mode)

    stacked = torch.stack(output_masks, dim=0).to(device=masks.device, dtype=masks.dtype)
    return stacked, modes


def apply_hybrid_masking_context_guard(
    condition_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    config,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Bound tail-risk context removal without ever exposing GT object pixels.

    The official Hybrid Masking sampler remains unchanged.  This optional local
    post-processing guard replaces an over-large sampled condition mask with the
    binary GT mask, which is the smallest leakage-safe fallback.  If the GT mask
    itself exceeds the configured limit, the fallback is still GT and the
    metadata marks the overflow as unavoidable.
    """

    if condition_masks.ndim != 3 or gt_masks.ndim != 3:
        raise ValueError(
            "Hybrid Masking context guard expects condition_masks and gt_masks "
            f"as [B, H, W], got {tuple(condition_masks.shape)} and {tuple(gt_masks.shape)}"
        )
    if condition_masks.shape != gt_masks.shape:
        raise ValueError(
            "Hybrid Masking context guard requires matching shapes, got "
            f"{tuple(condition_masks.shape)} vs {tuple(gt_masks.shape)}"
        )
    if not torch.isfinite(condition_masks).all().item():
        raise ValueError("Hybrid Masking condition masks must contain only finite values")
    if not torch.isfinite(gt_masks).all().item():
        raise ValueError("Hybrid Masking GT masks must contain only finite values")

    threshold = float(config.mask_threshold)
    condition_binary = (condition_masks > threshold).to(dtype=gt_masks.dtype)
    gt_binary = (gt_masks > threshold).to(dtype=gt_masks.dtype)
    if config.ensure_gt_coverage:
        condition_binary = torch.maximum(condition_binary, gt_binary)

    # Count binary pixels first, then divide scalar counts in float64.  Besides
    # making 80/100 compare equal to 0.80, this avoids materializing full-size
    # float64 masks on memory-tight training GPUs.
    pixel_count = condition_binary.shape[-2] * condition_binary.shape[-1]
    sampled_ratio = torch.count_nonzero(condition_binary, dim=(-2, -1)).to(torch.float64) / pixel_count
    gt_ratio = torch.count_nonzero(gt_binary, dim=(-2, -1)).to(torch.float64) / pixel_count
    guard_cfg = config.context_guard
    max_ratio = float(guard_cfg.max_condition_area_ratio)

    if guard_cfg.enabled:
        guard_triggered = sampled_ratio > max_ratio
        if guard_cfg.overflow_policy != "fallback_to_gt":
            raise ValueError(
                "Unsupported Hybrid Masking context guard overflow policy: " f"{guard_cfg.overflow_policy}"
            )
        guarded_masks = torch.where(
            guard_triggered[:, None, None],
            gt_binary,
            condition_binary,
        )
    else:
        guard_triggered = torch.zeros_like(sampled_ratio, dtype=torch.bool)
        guarded_masks = condition_binary

    final_ratio = torch.count_nonzero(guarded_masks, dim=(-2, -1)).to(torch.float64) / pixel_count
    unavoidable_large_gt = guard_triggered & (gt_ratio > max_ratio)
    metadata = {
        "gt_area_ratio": gt_ratio,
        "sampled_condition_area_ratio": sampled_ratio,
        "condition_area_ratio": final_ratio,
        "context_guard_triggered": guard_triggered,
        "fallback_applied": guard_triggered,
        "unavoidable_large_gt": unavoidable_large_gt,
    }
    return guarded_masks.to(dtype=condition_masks.dtype), metadata

"""Training-only Hybrid Masking strategies.

The official implementation samples category-dependent conditioning masks:

* person: 40% Bezier augmented box, 40% expanded box, 20% dilated mask
* non-person: 50% Bezier augmented box, 50% expanded box

The legacy ``insert_anything`` strategy preserves those mask families and
probabilities. ``maskmix_v2`` adds object-scale-aware stochastic geometry while
keeping the same leakage-safety contract. Both strategies include two local
adaptations for the Qwen training pipeline:

1. pixel-valued parameters can scale relative to the official 768 resolution;
2. generated conditioning masks can be unioned with the GT mask so constructing
   a hole from the completed target image cannot leak GT object pixels.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


INSERT_ANYTHING_AUGMENTED_BOX = "augmented_box"
INSERT_ANYTHING_EXPANDED_BOX = "expanded_box"
INSERT_ANYTHING_DILATED_SEGMENTATION = "dilated_segmentation"
MASKMIX_V2_EXACT = "exact"
MASKMIX_V2_RANDOM_MORPHOLOGY = "random_morphology"
MASKMIX_V2_ASYMMETRIC_BOX = "asymmetric_box"
LOOSE_BLOB = "loose_blob"
LOOSE_OVERSIZED_BOX = "oversized_box"

_LOOSE_MODE_ORDER = (LOOSE_BLOB, LOOSE_OVERSIZED_BOX)
_LOOSE_BLOB_SHAPE_ORDER = ("ellipse", "superellipse", "spline_blob", "brush")
_TIGHT_MODE_ORDER = (MASKMIX_V2_EXACT, MASKMIX_V2_RANDOM_MORPHOLOGY)
_LOOSE_GROWTH_STEP = 1.06
_LOOSE_MAX_GROWTH_STEPS = 120

# Loose families are appended last so zero loose weights keep the legacy
# random-number-to-mode mapping unchanged.
_MASKMIX_V2_MODE_ORDER = (
    INSERT_ANYTHING_AUGMENTED_BOX,
    MASKMIX_V2_ASYMMETRIC_BOX,
    MASKMIX_V2_EXACT,
    MASKMIX_V2_RANDOM_MORPHOLOGY,
    LOOSE_BLOB,
    LOOSE_OVERSIZED_BOX,
)
MASKMIX_LOOSE_MODES = frozenset(_LOOSE_MODE_ORDER)
_MASKMIX_V2_KERNEL_ORDER = (
    "ellipse",
    "rectangle",
    "horizontal_ellipse",
    "vertical_ellipse",
)


def _binary_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Hybrid Masking expects a 2D mask, got shape={array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Hybrid Masking target masks must contain only finite values")
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


def _round_half_up(value: float) -> int:
    """Round a non-negative pixel distance without banker's rounding."""

    return int(np.floor(value + 0.5))


def _sample_uniform(rng, minimum: float, maximum: float) -> float:
    if minimum == maximum:
        return float(minimum)
    return float(minimum) + (float(maximum) - float(minimum)) * float(rng.random())


def _weighted_choice(rng, names: Sequence[str], weights: Sequence[float]) -> str:
    """Sample a validated distribution using stable half-open intervals."""

    # Decimal-from-string makes configured decimal boundaries exact. Plain
    # float accumulation can move a documented boundary into the preceding
    # half-open interval.
    probability = Decimal(str(rng.random()))
    cumulative = Decimal("0")
    last_positive = None
    for name, weight in zip(names, weights):
        weight = float(weight)
        if weight > 0:
            last_positive = name
        cumulative += Decimal(str(weight))
        if probability < cumulative:
            return name
    if last_positive is None:
        raise ValueError("MaskMix-v2 requires at least one positive sampling weight")
    return last_positive


def _sample_asymmetric_bbox(mask: np.ndarray, config, rng) -> Tuple[int, int, int, int]:
    """Return an end-exclusive bbox with four independently sampled margins."""

    height, width = mask.shape
    y1, y2, x1, x2 = _tight_bbox(mask)
    bbox_height = y2 - y1 + 1
    bbox_width = x2 - x1 + 1
    minimum = float(config.margin_ratio_min)
    maximum = float(config.margin_ratio_max)

    top = _round_half_up(_sample_uniform(rng, minimum, maximum) * bbox_height)
    bottom = _round_half_up(_sample_uniform(rng, minimum, maximum) * bbox_height)
    left = _round_half_up(_sample_uniform(rng, minimum, maximum) * bbox_width)
    right = _round_half_up(_sample_uniform(rng, minimum, maximum) * bbox_width)
    return (
        max(0, y1 - top),
        min(height, y2 + 1 + bottom),
        max(0, x1 - left),
        min(width, x2 + 1 + right),
    )


def _sample_object_scale_parameter(
    mask: np.ndarray,
    ratio_minimum: float,
    ratio_maximum: float,
    pixel_minimum: int,
    pixel_maximum: int,
    config,
    rng,
) -> int:
    height, width = mask.shape
    minimum = _scaled_pixel_parameter(pixel_minimum, height, width, config)
    maximum = _scaled_pixel_parameter(pixel_maximum, height, width, config)
    maximum = max(minimum, maximum)
    object_scale = float(np.sqrt(np.count_nonzero(mask)))
    ratio = _sample_uniform(rng, ratio_minimum, ratio_maximum)
    value = _round_half_up(object_scale * ratio)
    return min(maximum, max(minimum, value))


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


def _maskmix_v2_asymmetric_box_mask(mask: np.ndarray, config, rng) -> np.ndarray:
    y1, y2, x1, x2 = _sample_asymmetric_bbox(
        mask,
        config.maskmix_v2.asymmetric_box,
        rng,
    )
    output = np.zeros_like(mask, dtype=np.uint8)
    output[y1:y2, x1:x2] = 1
    return output


def _maskmix_v2_augmented_box_mask(mask: np.ndarray, config, rng) -> np.ndarray:
    """Bezier box with asymmetric controls and object-relative jitter."""

    height, width = mask.shape
    y1, y2, x1, x2 = _tight_bbox(mask)
    expanded_y1, expanded_y2, expanded_x1, expanded_x2 = _sample_asymmetric_bbox(
        mask,
        config.maskmix_v2.asymmetric_box,
        rng,
    )
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    curves = (
        ((x1, y1), (center_x, expanded_y1), (x2, y1)),
        ((x2, y1), (expanded_x2 - 1, center_y), (x2, y2)),
        ((x2, y2), (center_x, expanded_y2 - 1), (x1, y2)),
        ((x1, y2), (expanded_x1, center_y), (x1, y1)),
    )
    jitter_cfg = config.maskmix_v2.augmented_box
    jitter = _sample_object_scale_parameter(
        mask,
        jitter_cfg.jitter_object_scale_min,
        jitter_cfg.jitter_object_scale_max,
        jitter_cfg.jitter_min_px,
        jitter_cfg.jitter_max_px,
        config,
        rng,
    )

    points: List[Tuple[float, float]] = []
    for start, control, end in curves:
        seen = set()
        for index in range(1, 19):
            original_x, original_y = _quadratic_bezier(start, control, end, index * 0.05)
            offset_x = rng.randint(-jitter, jitter) if jitter > 0 else 0
            offset_y = rng.randint(-jitter, jitter) if jitter > 0 else 0
            x = original_x + offset_x
            y = original_y + offset_y

            # As in the legacy policy, jitter only changes a coordinate when it
            # pushes that coordinate beyond the tight target bbox.
            if not (x < x1 or x > x2):
                x = original_x
            if not (y < y1 or y > y2):
                y = original_y
            point = (x, y)
            if point not in seen:
                points.append(point)
                seen.add(point)

    if len(points) < 3:
        raise ValueError("MaskMix-v2 augmented box produced fewer than three polygon points")
    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=np.uint8)


def _maskmix_v2_random_morphology_mask(mask: np.ndarray, config, rng) -> np.ndarray:
    morphology = config.maskmix_v2.morphology
    radius = _sample_object_scale_parameter(
        mask,
        morphology.radius_object_scale_min,
        morphology.radius_object_scale_max,
        morphology.radius_min_px,
        morphology.radius_max_px,
        config,
        rng,
    )
    kernel_weights = morphology.kernel_weights
    kernel_name = _weighted_choice(
        rng,
        _MASKMIX_V2_KERNEL_ORDER,
        tuple(getattr(kernel_weights, name) for name in _MASKMIX_V2_KERNEL_ORDER),
    )

    radius_x = radius
    radius_y = radius
    if kernel_name in {"horizontal_ellipse", "vertical_ellipse"}:
        anisotropy = _sample_uniform(
            rng,
            morphology.anisotropy_min,
            morphology.anisotropy_max,
        )
        major = max(1, _round_half_up(radius * np.sqrt(anisotropy)))
        minor = max(1, _round_half_up(radius / np.sqrt(anisotropy)))
        if kernel_name == "horizontal_ellipse":
            radius_x, radius_y = major, minor
        else:
            radius_x, radius_y = minor, major

    height, width = mask.shape
    radius_x = min(radius_x, max(0, width - 1))
    radius_y = min(radius_y, max(0, height - 1))
    shape = cv2.MORPH_RECT if kernel_name == "rectangle" else cv2.MORPH_ELLIPSE
    kernel = cv2.getStructuringElement(
        shape,
        (2 * radius_x + 1, 2 * radius_y + 1),
    )
    return cv2.dilate(mask, kernel, iterations=1).astype(np.uint8)


def maskmix_has_loose_modes(config) -> bool:
    """Whether MaskMix-v2 samples loose families, which the dataset must build before cropping."""

    if getattr(config, "strategy", "maskmix_v2") != "maskmix_v2":
        return False
    weights = config.maskmix_v2.mode_weights
    # Weight objects that predate the loose families count as zero loose weight.
    return any(getattr(weights, mode, 0.0) > 0 for mode in _LOOSE_MODE_ORDER)


def sample_maskmix_v2_mode(category: str, config, rng=None) -> str:
    """Sample MaskMix-v2; category is accepted for batch-API compatibility."""

    del category
    rng = rng or random
    weights = config.maskmix_v2.mode_weights
    return _weighted_choice(
        rng,
        _MASKMIX_V2_MODE_ORDER,
        tuple(getattr(weights, mode, 0.0) for mode in _MASKMIX_V2_MODE_ORDER),
    )


def build_maskmix_v2_mask_for_mode(
    mask: np.ndarray,
    mode: str,
    config,
    rng=None,
) -> np.ndarray:
    """Build one leakage-safe MaskMix-v2 conditioning mask."""

    rng = rng or random
    gt_mask = _binary_mask(mask, config.mask_threshold)
    if mode == INSERT_ANYTHING_AUGMENTED_BOX:
        generated = _maskmix_v2_augmented_box_mask(gt_mask, config, rng)
    elif mode == MASKMIX_V2_ASYMMETRIC_BOX:
        generated = _maskmix_v2_asymmetric_box_mask(gt_mask, config, rng)
    elif mode == MASKMIX_V2_EXACT:
        generated = gt_mask.copy()
    elif mode == MASKMIX_V2_RANDOM_MORPHOLOGY:
        generated = _maskmix_v2_random_morphology_mask(gt_mask, config, rng)
    elif mode in MASKMIX_LOOSE_MODES:
        raise ValueError(
            f"MaskMix-v2 mode {mode} must be built by build_loose_placement_mask in the dataset, "
            "before adaptive crop"
        )
    else:
        raise ValueError(f"Unsupported MaskMix-v2 mode: {mode}")

    generated = (generated > 0).astype(np.uint8)
    if config.ensure_gt_coverage:
        generated = np.maximum(generated, gt_mask)
    return generated


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


def build_maskmix_v2_hybrid_masks(
    masks: torch.Tensor,
    categories: Sequence[str] | str,
    config,
    rng=None,
) -> Tuple[torch.Tensor, List[str]]:
    """Apply MaskMix-v2 independently to every sample in a batch."""

    if maskmix_has_loose_modes(config):
        raise ValueError(
            "MaskMix-v2 loose_blob/oversized_box weights require the dataset to precompute "
            "tar_mask_cond (InsertAnythingDataset plans the crop on the loose region)"
        )
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
        mode = sample_maskmix_v2_mode(str(category), config, rng=rng)
        output = build_maskmix_v2_mask_for_mode(mask, mode, config, rng=rng)
        output_masks.append(torch.from_numpy(output))
        modes.append(mode)

    stacked = torch.stack(output_masks, dim=0).to(device=masks.device, dtype=masks.dtype)
    return stacked, modes


@dataclass(frozen=True)
class LoosePlacementMask:
    """A σ-split loose placement mask in full-image coordinates.

    ``scale_h``/``scale_w`` are the nominal per-axis expansion factors after any
    context-guard shrinking; ``coverage_growth`` is the extra uniform growth
    that made the outline cover the target.
    """

    mask: np.ndarray
    mode: str
    shape: str
    scale_h: float
    scale_w: float
    bottom_aligned: bool
    coverage_growth: float
    guard_shrinks: int
    area_ratio: float
    guard_exhausted: bool


def sample_tight_mode(config, rng=None) -> str:
    """Sample the exact/morphology family used by the low-noise segment."""

    rng = rng or random
    weights = config.loose_placement.tight_mode_weights
    return _weighted_choice(
        rng,
        _TIGHT_MODE_ORDER,
        tuple(getattr(weights, mode) for mode in _TIGHT_MODE_ORDER),
    )


def _catmull_rom_closed(points: np.ndarray, samples_per_segment: int = 16) -> np.ndarray:
    count = len(points)
    t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
    curve = []
    for index in range(count):
        p0, p1, p2, p3 = (points[(index + offset) % count] for offset in (-1, 0, 1, 2))
        curve.append(
            0.5
            * (
                2.0 * p1
                + (p2 - p0) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t**3
            )
        )
    return np.concatenate(curve, axis=0)


def _sample_loose_outline(shape: str, rng) -> Optional[np.ndarray]:
    """Closed outline whose bounding box is exactly the unit frame [-1, 1]^2."""

    if shape == "box":
        return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    if shape == "ellipse":
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if shape == "superellipse":
        power = 2.0 / _sample_uniform(rng, 2.5, 5.0)
        cos, sin = np.cos(theta), np.sin(theta)
        return np.stack([np.sign(cos) * np.abs(cos) ** power, np.sign(sin) * np.abs(sin) ** power], axis=1)
    if shape == "spline_blob":
        count = rng.randint(6, 10)
        angles = np.array(
            [2.0 * np.pi * (index + _sample_uniform(rng, -0.3, 0.3)) / count for index in range(count)]
        )
        radii = np.array([_sample_uniform(rng, 0.7, 1.0) for _ in range(count)])
        curve = _catmull_rom_closed(np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1))
        low, high = curve.min(axis=0), curve.max(axis=0)
        return 2.0 * (curve - low) / np.maximum(high - low, 1e-6) - 1.0
    if shape == "brush":
        return None
    raise ValueError(f"Unsupported loose placement shape: {shape}")


def _brush_strokes(jitter: Sequence[Tuple[float, float]], half: bool, radius_fraction: float, aspect: float):
    """Zig-zag scribble in the unit frame; consecutive rows overlap, so it has no gaps."""

    # The brush radius is relative to the shorter frame side; ``aspect`` is
    # half_height / half_width, so the same pixel radius in unit-y is smaller
    # on a tall frame.
    radius_y = radius_fraction * min(1.0, 1.0 / aspect)
    radius_x = radius_fraction * min(1.0, aspect)
    top, bottom = -1.0 + radius_y, (0.0 if half else 1.0 - radius_y)
    rows = max(2, math.ceil(max(bottom - top, 0.0) / (1.5 * radius_y)) + 1)
    points = []
    for row, y in enumerate(np.linspace(top, bottom, rows)):
        left_jitter, right_jitter = jitter[row % len(jitter)]
        left = -1.0 + radius_x + left_jitter
        right = 1.0 - radius_x - right_jitter
        if right < left:
            left = right = 0.5 * (left + right)
        ends = [(left, y), (right, y)]
        points.extend(ends if row % 2 == 0 else ends[::-1])
    return np.array(points), radius_fraction


def _rasterize_loose_shape(
    outline: Optional[np.ndarray],
    brush,
    image_shape: Tuple[int, int],
    center: Tuple[float, float],
    half_size: Tuple[float, float],
    ground_row: Optional[int],
) -> np.ndarray:
    height, width = image_shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    center_xy = np.asarray(center, dtype=np.float64)
    half_xy = np.asarray(half_size, dtype=np.float64)
    shift = 4
    scale = float(1 << shift)
    if outline is not None:
        points = np.round((center_xy + outline * half_xy) * scale).astype(np.int32)
        cv2.fillPoly(canvas, [points], 1, lineType=cv2.LINE_8, shift=shift)
    else:
        stroke_points, radius_fraction = brush
        radius = max(1, int(round(radius_fraction * min(half_xy))))
        points = np.round(center_xy + stroke_points * half_xy).astype(np.int32)
        cv2.polylines(canvas, [points], False, 1, thickness=2 * radius + 1, lineType=cv2.LINE_8)
        for x, y in points:
            cv2.circle(canvas, (int(x), int(y)), radius, 1, thickness=-1, lineType=cv2.LINE_8)
    if ground_row is not None:
        canvas[ground_row + 1 :] = 0
    return canvas


def build_loose_placement_mask(
    mask: np.ndarray,
    config,
    rng=None,
    *,
    coverage_radius_px: int = 0,
    canvas_fn: Optional[Callable[[np.ndarray], Tuple[int, int, int, int]]] = None,
    max_area_ratio: Optional[float] = None,
    mode: Optional[str] = None,
) -> LoosePlacementMask:
    """Sample a loose placement region that decouples the mask from the GT outline.

    ``mode`` fixes the loose family when the caller already sampled it (MaskMix-v2
    loose weights); otherwise ``loose_placement.mode_weights`` decides (σ-split).

    The region is a smooth blob or an oversized box whose height and width are
    independently scaled by U(scale_min, scale_max) relative to the GT box
    (``size_reference``), with the object placed at a random position inside it
    or, with ``bottom_align_prob``, standing on its bottom edge. If the outline
    does not cover ``dilate(GT, coverage_radius_px)`` it is uniformly grown
    about its anchor until it does; it is never unioned with the GT, so its
    boundary carries no object-outline information. When the area inside the
    canvas (the adaptive crop planned on this very mask) exceeds
    ``max_area_ratio``, the expansion factors are shrunk toward one instead of
    falling back to the exact GT mask.
    """

    rng = rng or random
    loose_cfg = config.loose_placement
    gt_mask = _binary_mask(mask, config.mask_threshold)
    height, width = gt_mask.shape
    gy1, gy2, gx1, gx2 = _tight_bbox(gt_mask)
    # Dilate only a window around the object; the band cannot reach further.
    radius = max(0, int(coverage_radius_px))
    wy1, wy2 = max(0, gy1 - radius), min(height, gy2 + radius + 1)
    wx1, wx2 = max(0, gx1 - radius), min(width, gx2 + radius + 1)
    target = gt_mask[wy1:wy2, wx1:wx2]
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
        target = cv2.dilate(target, kernel)
    target_ys, target_xs = np.nonzero(target)
    target_ys, target_xs = target_ys + wy1, target_xs + wx1
    object_h, object_w = gy2 - gy1 + 1, gx2 - gx1 + 1
    if loose_cfg.size_reference == "bbox_geometric_mean":
        base_h = base_w = math.sqrt(object_h * object_w)
    else:
        base_h, base_w = float(object_h), float(object_w)

    if mode is None:
        mode = _weighted_choice(
            rng,
            _LOOSE_MODE_ORDER,
            tuple(getattr(loose_cfg.mode_weights, name) for name in _LOOSE_MODE_ORDER),
        )
    elif mode not in MASKMIX_LOOSE_MODES:
        raise ValueError(f"Unsupported loose placement mode: {mode}")
    if mode == LOOSE_OVERSIZED_BOX:
        shape = "box"
    else:
        shape = _weighted_choice(
            rng,
            _LOOSE_BLOB_SHAPE_ORDER,
            tuple(getattr(loose_cfg.blob_shape_weights, name) for name in _LOOSE_BLOB_SHAPE_ORDER),
        )
    sampled_scale_h = _sample_uniform(rng, loose_cfg.scale_min, loose_cfg.scale_max)
    sampled_scale_w = _sample_uniform(rng, loose_cfg.scale_min, loose_cfg.scale_max)
    bottom_aligned = rng.random() < loose_cfg.bottom_align_prob
    offset_x, offset_y = rng.random(), rng.random()
    outline = _sample_loose_outline(shape, rng)
    if shape == "brush":
        # Stroke randomness is drawn once so guard retries only change scale.
        brush_radius = _sample_uniform(rng, 0.35, 0.5)
        brush_jitter = [(_sample_uniform(rng, 0.0, 0.3), _sample_uniform(rng, 0.0, 0.3)) for _ in range(32)]
    ground_row = int(target_ys.max()) if bottom_aligned else None

    def _origin(start: int, extent: int, size: float, offset: float) -> float:
        # cv2 vertices are pixel centres; a box covering pixels [start, start+extent) spans start-0.5.
        slack = size - extent
        if slack < 0:
            return start - 0.5 + 0.5 * extent - 0.5 * size
        return start - 0.5 - offset * slack

    for attempt in range(int(loose_cfg.guard_max_attempts) + 1):
        shrink = float(loose_cfg.guard_shrink) ** attempt
        scale_h = 1.0 + (sampled_scale_h - 1.0) * shrink
        scale_w = 1.0 + (sampled_scale_w - 1.0) * shrink
        size_h, size_w = scale_h * base_h, scale_w * base_w
        center_x = _origin(gx1, object_w, size_w, offset_x) + 0.5 * size_w
        if bottom_aligned:
            center_y, half_h = ground_row + 0.5, size_h
        else:
            center_y, half_h = _origin(gy1, object_h, size_h, offset_y) + 0.5 * size_h, 0.5 * size_h
        brush = None
        if outline is None:
            brush = _brush_strokes(brush_jitter, bottom_aligned, brush_radius, half_h / (0.5 * size_w))

        growth = 1.0
        for _ in range(_LOOSE_MAX_GROWTH_STEPS):
            loose = _rasterize_loose_shape(
                outline,
                brush,
                (height, width),
                (center_x, center_y),
                (0.5 * size_w * growth, half_h * growth),
                ground_row,
            )
            if loose[target_ys, target_xs].all():
                break
            growth *= _LOOSE_GROWTH_STEP
        else:
            raise RuntimeError("loose placement outline failed to cover the target mask")

        if canvas_fn is None:
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            x1, y1, x2, y2 = canvas_fn(loose)
        area_ratio = float(loose[y1:y2, x1:x2].sum()) / float((y2 - y1) * (x2 - x1))
        if max_area_ratio is None or area_ratio <= max_area_ratio:
            break

    return LoosePlacementMask(
        mask=loose,
        mode=mode,
        shape=shape,
        scale_h=float(scale_h),
        scale_w=float(scale_w),
        bottom_aligned=bool(bottom_aligned),
        coverage_growth=float(growth),
        guard_shrinks=int(attempt),
        area_ratio=area_ratio,
        guard_exhausted=bool(max_area_ratio is not None and area_ratio > max_area_ratio),
    )


def add_interaction_context(condition_masks, gt_masks, radius_px, threshold=0.5, mode="union"):
    """Remove object-adjacent context too, so contact/shadow pixels must be learned.

    ``mode="union"`` adds dilate(GT, radius_px) to each condition mask. ``mode="grow"``
    instead dilates the condition mask itself by the smallest Euclidean radius that
    covers dilate(GT, radius_px): the hole keeps its sampled outline, offset outward,
    rather than tracing the GT contour wherever the sampled shape fell short.
    """
    if radius_px <= 0:
        return condition_masks
    if mode not in ("union", "grow"):
        raise ValueError(f"interaction context mode must be 'union' or 'grow', got {mode!r}")
    radius_px = int(radius_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1,) * 2)
    binary = (gt_masks.detach().float().cpu().numpy() > threshold).astype(np.uint8)
    expanded = np.stack([cv2.dilate(mask, kernel) for mask in binary])
    if mode == "grow":
        condition = (condition_masks.detach().float().cpu().numpy() > threshold).astype(np.uint8)
        grown = []
        for sampled, band in zip(condition, expanded):
            if not sampled.any():
                grown.append(band)
                continue
            # Distance of every pixel to the sampled shape; grow to the farthest band pixel.
            distance = cv2.distanceTransform(1 - sampled, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            reach = float(distance[band > 0].max()) if band.any() else 0.0
            grown.append(np.maximum(sampled, (distance <= reach).astype(np.uint8)))
        return torch.from_numpy(np.stack(grown)).to(device=condition_masks.device, dtype=condition_masks.dtype)
    expanded = torch.from_numpy(expanded).to(device=condition_masks.device, dtype=condition_masks.dtype)
    return torch.maximum(condition_masks, expanded)


def build_insert_anything_hybrid_masks(
    masks: torch.Tensor,
    categories: Sequence[str] | str,
    config,
    rng=None,
) -> Tuple[torch.Tensor, List[str]]:
    """Apply the configured strategy while preserving the legacy public API."""

    strategy = getattr(config, "strategy", "insert_anything")
    if strategy == "maskmix_v2":
        return build_maskmix_v2_hybrid_masks(masks, categories, config, rng=rng)
    if strategy != "insert_anything":
        raise ValueError(f"Unsupported Hybrid Masking strategy: {strategy}")

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
    allow_fallback: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Bound tail-risk context removal without ever exposing GT object pixels.

    The official Hybrid Masking sampler remains unchanged.  This optional local
    post-processing guard replaces an over-large sampled condition mask with the
    binary GT mask (``fallback_to_gt``), the smallest leakage-safe fallback, or
    with GT plus the interaction band (``fallback_to_band``), which keeps the
    contact/shadow ring hidden as in every other mode.  If the GT mask itself
    exceeds the configured limit, the fallback is still applied and the
    metadata marks the overflow as unavoidable.

    ``allow_fallback`` ([B] bool) exempts samples whose sampler already
    enforced the limit by shrinking (σ-split loose masks); they are reported
    but never replaced by the exact GT mask.
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
        if guard_cfg.overflow_policy == "fallback_to_gt":
            fallback_masks = gt_binary
        elif guard_cfg.overflow_policy == "fallback_to_band":
            fallback_masks = add_interaction_context(
                gt_binary, gt_binary, getattr(config, "interaction_dilation_px", 0), threshold
            ).to(device=gt_binary.device, dtype=gt_binary.dtype)
        else:
            raise ValueError(
                "Unsupported Hybrid Masking context guard overflow policy: " f"{guard_cfg.overflow_policy}"
            )
        fallback_applied = guard_triggered
        if allow_fallback is not None:
            fallback_applied = guard_triggered & allow_fallback.to(device=guard_triggered.device, dtype=torch.bool)
        guarded_masks = torch.where(
            fallback_applied[:, None, None],
            fallback_masks,
            condition_binary,
        )
    else:
        guard_triggered = torch.zeros_like(sampled_ratio, dtype=torch.bool)
        fallback_applied = guard_triggered
        guarded_masks = condition_binary

    final_ratio = torch.count_nonzero(guarded_masks, dim=(-2, -1)).to(torch.float64) / pixel_count
    gt_pixel_count = torch.count_nonzero(gt_binary, dim=(-2, -1)).to(torch.float64).clamp_min(1.0)
    sampled_expansion = (
        torch.count_nonzero(condition_binary, dim=(-2, -1)).to(torch.float64)
        / gt_pixel_count
    )
    final_expansion = (
        torch.count_nonzero(guarded_masks, dim=(-2, -1)).to(torch.float64)
        / gt_pixel_count
    )
    unavoidable_large_gt = guard_triggered & (gt_ratio > max_ratio)
    metadata = {
        "gt_area_ratio": gt_ratio,
        "sampled_condition_area_ratio": sampled_ratio,
        "condition_area_ratio": final_ratio,
        "sampled_condition_expansion": sampled_expansion,
        "condition_expansion": final_expansion,
        "context_guard_triggered": guard_triggered,
        "fallback_applied": fallback_applied,
        "unavoidable_large_gt": unavoidable_large_gt,
    }
    return guarded_masks.to(dtype=condition_masks.dtype), metadata

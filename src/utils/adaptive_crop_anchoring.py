"""Compatibility helpers for Adaptive Crop + Latent Anchoring inference."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.adaptive_crop import (
    AdaptiveCropPlan,
    build_adaptive_crop_plan,
    crop_pil_mask,
)
from src.utils.latent_anchoring import (
    build_latent_anchoring_masks,
    ensure_latent_anchoring_config,
    latent_anchoring_enabled,
    normalize_latent_anchoring_mode,
)


DEFAULT_CROP_EDGE_FEATHER_PX = 8
# Extra original-space margin beyond the discrete packed-support footprint.
# Intended as a small decoder/inverse-resize buffer, not a substitute for realized-support checks.
DEFAULT_DECODER_GUARD_PX = 2
DEFAULT_VAE_SCALE_FACTOR = 8
# Match trainer lerp semantics: any positive packed weight is editable.
_SUPPORT_EPSILON = 0.0
_MIN_CORE_PIXELS = 1
# LANCZOS kernel has a multi-pixel halo; treat light ringing as support footprint.
_LANCZOS_SUPPORT_THRESHOLD = 1.0 / 255.0
# Search budget for applied-crop expansion before full-frame fallback.
DEFAULT_MAX_EXPAND_ITERS = 24


def require_matching_mask_thresholds(adaptive_crop_config, anchoring_config) -> float:
    """Fail fast when crop and anchoring would disagree on M_core."""

    crop_threshold = float(adaptive_crop_config.mask_threshold)
    anchor_threshold = float(anchoring_config.mask_threshold)
    if not math.isclose(crop_threshold, anchor_threshold, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "adaptive_crop.mask_threshold and predict.latent_anchoring.mask_threshold must match "
            f"when both features are enabled; got adaptive_crop={crop_threshold} and "
            f"latent_anchoring={anchor_threshold}"
        )
    return crop_threshold


def _processor_init_args(processor_config):
    return getattr(processor_config, "init_args", processor_config)


def build_image_processor(processor_config):
    """Construct the real ImageProcessor used by training/inference preprocessing."""

    from src.data.preprocess import ImageProcessor

    init_args = _processor_init_args(processor_config)
    return ImageProcessor(init_args)


def estimate_model_geometry(
    image_size: tuple[int, int],
    processor_config,
    *,
    processor=None,
) -> tuple[int, int]:
    """Return processor output (width, height) for a prediction-space image.

    Uses the same ImageProcessor path as training/inference, including 16-alignment
    and controls_size / controls_pixels fallbacks.
    """

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"model geometry requires positive image size, got {(width, height)}")

    image_processor = processor if processor is not None else build_image_processor(processor_config)
    probe = Image.new("RGB", (width, height), color=(0, 0, 0))
    processed = image_processor.preprocess({"image": probe})
    image_tensor = processed["image"]
    if image_tensor.ndim != 3:
        raise RuntimeError(f"unexpected processed image rank: {tuple(image_tensor.shape)}")
    model_h, model_w = int(image_tensor.shape[1]), int(image_tensor.shape[2])
    if model_h <= 0 or model_w <= 0:
        raise RuntimeError(f"processor produced invalid model geometry {(model_w, model_h)}")
    return model_w, model_h


def preprocess_crop_mask(
    crop_mask: Image.Image | np.ndarray,
    processor_config,
    *,
    processor=None,
) -> torch.Tensor:
    """Run the real mask preprocess and return a float [H,W] tensor in [0,1]."""

    image_processor = processor if processor is not None else build_image_processor(processor_config)
    if isinstance(crop_mask, Image.Image):
        mask_image = crop_mask.convert("L")
    else:
        array = np.asarray(crop_mask)
        if array.ndim != 2:
            raise ValueError(f"crop mask must be 2D, got shape={array.shape}")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        mask_image = Image.fromarray(array, mode="L")

    processed = image_processor.preprocess({"mask": mask_image})
    mask_tensor = processed["mask"]
    if mask_tensor.ndim != 2:
        raise RuntimeError(f"unexpected processed mask rank: {tuple(mask_tensor.shape)}")
    return mask_tensor.float()


def pack_edit_weight_to_latent_support(
    edit_weight: torch.Tensor,
    free_mask: torch.Tensor,
    *,
    soft_transition: bool,
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR,
) -> torch.Tensor:
    """Mirror trainer latent packing spatial rules; return binary latent support [H_lat,W_lat]."""

    if edit_weight.ndim != 2 or free_mask.ndim != 2:
        raise ValueError(
            "pack_edit_weight_to_latent_support expects [H,W] tensors, "
            f"got edit={tuple(edit_weight.shape)} free={tuple(free_mask.shape)}"
        )
    if edit_weight.shape != free_mask.shape:
        raise ValueError(
            "edit_weight and free_mask geometry mismatch: "
            f"{tuple(edit_weight.shape)} vs {tuple(free_mask.shape)}"
        )
    height_image, width_image = (int(edit_weight.shape[0]), int(edit_weight.shape[1]))
    latent_height = 2 * (height_image // (int(vae_scale_factor) * 2))
    latent_width = 2 * (width_image // (int(vae_scale_factor) * 2))
    if latent_height <= 0 or latent_width <= 0:
        raise ValueError(
            "latent anchoring target geometry is too small for VAE compression and packing: "
            f"{(height_image, width_image)} vae_scale_factor={vae_scale_factor}"
        )

    edit = edit_weight.float().unsqueeze(0).unsqueeze(0)
    free = free_mask.float().unsqueeze(0).unsqueeze(0)
    if soft_transition:
        edit = F.interpolate(
            edit,
            size=(latent_height, latent_width),
            mode="bilinear",
            align_corners=False,
        )
        free = F.adaptive_max_pool2d(free, output_size=(latent_height, latent_width))
        edit = torch.where(free > 0.5, torch.ones_like(edit), edit)
        edit = edit.clamp(0.0, 1.0)
    else:
        edit = F.adaptive_max_pool2d(edit, output_size=(latent_height, latent_width))
        edit = (edit > 0.5).to(dtype=torch.float32)
    support = (edit[0, 0] > float(_SUPPORT_EPSILON)).to(dtype=torch.float32)
    if not bool(support.any().item()):
        raise ValueError("packed latent anchoring weight must retain an editable region")
    return support


def upsample_latent_support_to_model(
    latent_support: torch.Tensor,
    model_size: tuple[int, int],
) -> np.ndarray:
    """Nearest-upsample latent support to model image geometry as a binary numpy mask."""

    model_w, model_h = (int(model_size[0]), int(model_size[1]))
    if model_w <= 0 or model_h <= 0:
        raise ValueError(f"invalid model size {model_size}")
    upsampled = F.interpolate(
        latent_support.float().unsqueeze(0).unsqueeze(0),
        size=(model_h, model_w),
        mode="nearest",
    )[0, 0]
    return (upsampled.detach().cpu().numpy() > 0.5).astype(np.uint8)


def map_model_support_to_crop(
    model_support: np.ndarray,
    plan: AdaptiveCropPlan,
    *,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
    threshold: float = _LANCZOS_SUPPORT_THRESHOLD,
) -> np.ndarray:
    """Map model-space support into crop-local original-resolution coordinates.

    Production RGB restore uses LANCZOS; safety uses the same resample so kernel
    halo is covered by the feather clearance check.
    """

    x1, y1, x2, y2 = plan.crop_box
    crop_w, crop_h = x2 - x1, y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"invalid crop box: {plan.crop_box}")
    support_img = Image.fromarray((model_support.astype(np.uint8) * 255), mode="L")
    resized = support_img.resize((crop_w, crop_h), resample)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return (arr > float(threshold)).astype(np.uint8)


def _edge_expansion_need(
    overlap: np.ndarray,
    *,
    feather: int,
    guard: int,
    left_internal: bool,
    top_internal: bool,
    right_internal: bool,
    bottom_internal: bool,
) -> tuple[int, int, int, int]:
    """Compute per-edge expansion from overlap restricted to each feather band."""

    if overlap.ndim != 2:
        raise ValueError(f"overlap must be 2D, got shape={overlap.shape}")
    crop_h, crop_w = overlap.shape
    band = max(1, int(feather)) if int(feather) > 0 else 1
    band = min(band, crop_w, crop_h)

    def _need_from_coords(coords: np.ndarray, length: int, toward_start: bool) -> int:
        if coords.size == 0:
            return 0
        if toward_start:
            # Distance from start edge; expand enough to push past feather band.
            nearest = int(coords.min())
            return int(max(0, band - nearest)) + 1 + int(guard)
        nearest = int(length - 1 - coords.max())
        return int(max(0, band - nearest)) + 1 + int(guard)

    left_band = overlap[:, :band] if left_internal else None
    right_band = overlap[:, crop_w - band :] if right_internal else None
    top_band = overlap[:band, :] if top_internal else None
    bottom_band = overlap[crop_h - band :, :] if bottom_internal else None

    need_left = 0
    need_right = 0
    need_top = 0
    need_bottom = 0

    if left_band is not None and left_band.any():
        # xs are absolute crop-local columns for left band pixels.
        ys, xs = np.nonzero(left_band)
        need_left = _need_from_coords(xs, crop_w, toward_start=True)
    if right_band is not None and right_band.any():
        ys, xs_local = np.nonzero(right_band)
        xs = xs_local + (crop_w - band)
        need_right = _need_from_coords(xs, crop_w, toward_start=False)
    if top_band is not None and top_band.any():
        ys, xs = np.nonzero(top_band)
        need_top = _need_from_coords(ys, crop_h, toward_start=True)
    if bottom_band is not None and bottom_band.any():
        ys_local, xs = np.nonzero(bottom_band)
        ys = ys_local + (crop_h - band)
        need_bottom = _need_from_coords(ys, crop_h, toward_start=False)

    return need_left, need_top, need_right, need_bottom


def _min_expand_step_px(
    plan: AdaptiveCropPlan,
    model_size: tuple[int, int],
    *,
    vae_scale_factor: int,
    feather_width_px: int,
    decoder_guard_px: int,
) -> int:
    """Minimum original-space expansion step: one packed latent cell footprint."""

    crop_w = plan.crop_box[2] - plan.crop_box[0]
    crop_h = plan.crop_box[3] - plan.crop_box[1]
    model_w, model_h = (int(model_size[0]), int(model_size[1]))
    if model_w <= 0 or model_h <= 0 or crop_w <= 0 or crop_h <= 0:
        return max(1, int(feather_width_px) + int(decoder_guard_px) + 1)
    # Packed latent spatial size uses the same formula as the trainer.
    latent_h = max(1, 2 * (model_h // (int(vae_scale_factor) * 2)))
    latent_w = max(1, 2 * (model_w // (int(vae_scale_factor) * 2)))
    cell_x = int(math.ceil(crop_w / float(latent_w)))
    cell_y = int(math.ceil(crop_h / float(latent_h)))
    return max(
        1,
        cell_x,
        cell_y,
        int(feather_width_px) + int(decoder_guard_px) + 1,
    )


def _internal_edge_flags(plan: AdaptiveCropPlan) -> tuple[bool, bool, bool, bool]:
    """Return whether left/top/right/bottom crop edges are interior to the original image."""

    width, height = plan.original_size
    x1, y1, x2, y2 = plan.crop_box
    return (x1 > 0, y1 > 0, x2 < width, y2 < height)


def build_crop_edge_alpha(
    plan: AdaptiveCropPlan,
    *,
    feather_width_px: int,
) -> np.ndarray:
    """Raised-cosine alpha over the crop rectangle in original coordinates.

    Interior crop edges feather from 1 -> 0. Edges that coincide with the original
    image border keep alpha=1 (no feather). Outside the crop, alpha is 0.
    """

    width, height = plan.original_size
    alpha = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = plan.crop_box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid crop box for edge alpha: {plan.crop_box}")

    crop_w = x2 - x1
    crop_h = y2 - y1
    left_internal, top_internal, right_internal, bottom_internal = _internal_edge_flags(plan)
    feather = max(0, int(feather_width_px))
    if feather == 0 or not plan.applied:
        alpha[y1:y2, x1:x2] = 1.0
        return alpha

    ys = np.arange(crop_h, dtype=np.float32)[:, None]
    xs = np.arange(crop_w, dtype=np.float32)[None, :]
    dist = np.full((crop_h, crop_w), np.inf, dtype=np.float32)
    if left_internal:
        dist = np.minimum(dist, xs)
    if right_internal:
        dist = np.minimum(dist, (crop_w - 1) - xs)
    if top_internal:
        dist = np.minimum(dist, ys)
    if bottom_internal:
        dist = np.minimum(dist, (crop_h - 1) - ys)

    local = np.ones((crop_h, crop_w), dtype=np.float32)
    if np.isfinite(dist).any():
        edge = dist < float(feather)
        t = np.clip(dist[edge] / float(feather), 0.0, 1.0)
        local[edge] = 0.5 * (1.0 - np.cos(np.pi * t))
        local[dist >= float(feather)] = 1.0
    alpha[y1:y2, x1:x2] = local
    return alpha


def evaluate_crop_candidate(
    plan: AdaptiveCropPlan,
    original_mask: Image.Image | np.ndarray,
    *,
    adaptive_crop_config,
    anchoring_config,
    processor_config,
    processor=None,
    feather_width_px: int = DEFAULT_CROP_EDGE_FEATHER_PX,
    decoder_guard_px: int = DEFAULT_DECODER_GUARD_PX,
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR,
) -> dict[str, Any]:
    """Evaluate one crop candidate with real processor + packed latent support."""

    mode = normalize_latent_anchoring_mode(anchoring_config.mode)
    threshold = require_matching_mask_thresholds(adaptive_crop_config, anchoring_config)
    image_processor = processor if processor is not None else build_image_processor(processor_config)

    if isinstance(original_mask, Image.Image):
        mask_image = original_mask.convert("L")
    else:
        array = np.asarray(original_mask)
        if array.ndim != 2:
            raise ValueError(f"original mask must be 2D, got shape={array.shape}")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        mask_image = Image.fromarray(array, mode="L")

    # Mask-only crop avoids allocating a full-frame RGB dummy per candidate.
    crop_mask = crop_pil_mask(mask_image, plan)
    processed_mask = preprocess_crop_mask(crop_mask, processor_config, processor=image_processor)
    core_sum = float((processed_mask > threshold).sum().item())
    model_h, model_w = int(processed_mask.shape[0]), int(processed_mask.shape[1])
    model_size = (model_w, model_h)

    result: dict[str, Any] = {
        "ok": False,
        "core_survived": core_sum >= _MIN_CORE_PIXELS,
        "processed_core_pixels": int(core_sum),
        "model_size": list(model_size),
        "support_overlap_pixels": 0,
        "need": (0, 0, 0, 0),
        "min_expand_step_px": 1,
        "min_alpha_on_support": 1.0,
        "reason": None,
        "support_epsilon": float(_SUPPORT_EPSILON),
        "support_map_resample": "lanczos",
    }
    if not result["core_survived"]:
        result["reason"] = "processed_m_core_empty"
        return result

    if not latent_anchoring_enabled(mode):
        result["ok"] = True
        result["reason"] = "legacy"
        return result

    masks = build_latent_anchoring_masks(
        processed_mask.unsqueeze(0),
        mode=mode,
        mask_threshold=threshold,
        interaction_dilation_px=int(anchoring_config.interaction_dilation_px),
        transition_width_px=int(anchoring_config.transition_width_px),
    )
    soft = mode == "expanded_soft"
    latent_support = pack_edit_weight_to_latent_support(
        masks.edit_weight[0],
        masks.free[0],
        soft_transition=soft,
        vae_scale_factor=vae_scale_factor,
    )
    model_support = upsample_latent_support_to_model(latent_support, model_size)
    crop_support = map_model_support_to_crop(
        model_support,
        plan,
        resample=Image.Resampling.LANCZOS,
        threshold=_LANCZOS_SUPPORT_THRESHOLD,
    )

    # Optional decoder guard: dilate support in crop/original pixels before feather check.
    guard = max(0, int(decoder_guard_px))
    checked_support = crop_support
    if guard > 0:
        from cv2 import dilate, getStructuringElement, MORPH_ELLIPSE

        kernel = getStructuringElement(MORPH_ELLIPSE, (2 * guard + 1, 2 * guard + 1))
        checked_support = dilate(crop_support.astype(np.uint8), kernel, iterations=1)

    alpha = build_crop_edge_alpha(plan, feather_width_px=int(feather_width_px))
    x1, y1, x2, y2 = plan.crop_box
    crop_alpha = alpha[y1:y2, x1:x2]
    if crop_alpha.shape != checked_support.shape:
        raise RuntimeError(
            f"support/alpha geometry mismatch: support={checked_support.shape} alpha={crop_alpha.shape}"
        )

    overlap = (checked_support > 0) & (crop_alpha < 1.0 - 1e-6)
    overlap_count = int(overlap.sum())
    result["support_overlap_pixels"] = overlap_count
    result["min_expand_step_px"] = _min_expand_step_px(
        plan,
        model_size,
        vae_scale_factor=int(vae_scale_factor),
        feather_width_px=int(feather_width_px),
        decoder_guard_px=int(decoder_guard_px),
    )
    if bool((checked_support > 0).any()):
        result["min_alpha_on_support"] = float(crop_alpha[checked_support > 0].min())
    else:
        result["min_alpha_on_support"] = 1.0

    if overlap_count == 0:
        result["ok"] = True
        result["reason"] = "safe"
        return result

    # Per-edge feather-band attribution: only edges that actually host overlap expand.
    left_internal, top_internal, right_internal, bottom_internal = _internal_edge_flags(plan)
    feather = max(0, int(feather_width_px))
    need_left, need_top, need_right, need_bottom = _edge_expansion_need(
        overlap,
        feather=feather,
        guard=guard,
        left_internal=left_internal,
        top_internal=top_internal,
        right_internal=right_internal,
        bottom_internal=bottom_internal,
    )
    # Grow at least one packed latent cell so small steps cannot stall on the same cell.
    min_step = int(result["min_expand_step_px"])
    if need_left > 0:
        need_left = max(need_left, min_step)
    if need_top > 0:
        need_top = max(need_top, min_step)
    if need_right > 0:
        need_right = max(need_right, min_step)
    if need_bottom > 0:
        need_bottom = max(need_bottom, min_step)

    result["need"] = (need_left, need_top, need_right, need_bottom)
    if need_left == need_top == need_right == need_bottom == 0 and plan.applied:
        result["reason"] = "support_touches_image_border_feather_unavoidable"
        result["ok"] = False
        return result

    result["reason"] = "support_intersects_feather"
    return result


def _rebuild_plan_with_crop_box(
    base_plan: AdaptiveCropPlan,
    crop_box: tuple[int, int, int, int],
    *,
    decision_reason: str,
) -> AdaptiveCropPlan:
    width, height = base_plan.original_size
    x1, y1, x2, y2 = crop_box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"expanded crop box out of bounds: {crop_box} for image={(width, height)}")
    crop_width = x2 - x1
    crop_height = y2 - y1
    applied = crop_box != (0, 0, width, height)
    return AdaptiveCropPlan(
        applied=applied,
        decision_reason=decision_reason,
        original_size=base_plan.original_size,
        mask_bbox=base_plan.mask_bbox,
        crop_box=crop_box,
        mask_area_pixels=base_plan.mask_area_pixels,
        mask_area_ratio=base_plan.mask_area_ratio,
        bbox_area_ratio=base_plan.bbox_area_ratio,
        bbox_fill_ratio=base_plan.bbox_fill_ratio,
        requested_crop_area_ratio=base_plan.requested_crop_area_ratio,
        actual_crop_area_ratio=(crop_width * crop_height) / float(width * height),
        mask_threshold=base_plan.mask_threshold,
        min_crop_side=base_plan.min_crop_side,
        effective_zoom_x=width / float(crop_width),
        effective_zoom_y=height / float(crop_height),
    )


def expand_crop_plan_for_support(
    plan: AdaptiveCropPlan,
    *,
    need_left: int,
    need_top: int,
    need_right: int,
    need_bottom: int,
) -> AdaptiveCropPlan:
    """Expand crop_box by the requested margins, clamped to the original frame."""

    width, height = plan.original_size
    x1, y1, x2, y2 = plan.crop_box
    bx1, by1, bx2, by2 = plan.mask_bbox

    new_x1 = max(0, x1 - int(need_left))
    new_y1 = max(0, y1 - int(need_top))
    new_x2 = min(width, x2 + int(need_right))
    new_y2 = min(height, y2 + int(need_bottom))

    new_x1 = min(new_x1, bx1)
    new_y1 = min(new_y1, by1)
    new_x2 = max(new_x2, bx2)
    new_y2 = max(new_y2, by2)

    target_w = min(width, (x2 - x1) + int(need_left) + int(need_right))
    target_h = min(height, (y2 - y1) + int(need_top) + int(need_bottom))
    if new_x2 - new_x1 < target_w:
        grow = target_w - (new_x2 - new_x1)
        shift_left = min(new_x1, grow)
        new_x1 -= shift_left
        grow -= shift_left
        new_x2 = min(width, new_x2 + grow)
    if new_y2 - new_y1 < target_h:
        grow = target_h - (new_y2 - new_y1)
        shift_top = min(new_y1, grow)
        new_y1 -= shift_top
        grow -= shift_top
        new_y2 = min(height, new_y2 + grow)

    new_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    if new_box == plan.crop_box:
        return plan
    reason = (
        "geometry_requires_full_frame"
        if new_box == (0, 0, width, height)
        else "expanded_for_latent_anchoring_support"
    )
    return _rebuild_plan_with_crop_box(plan, new_box, decision_reason=reason)


def ensure_crop_plan_for_latent_anchoring(
    plan: AdaptiveCropPlan,
    mask: Image.Image | np.ndarray,
    *,
    adaptive_crop_config,
    anchoring_config,
    processor_config,
    feather_width_px: int = DEFAULT_CROP_EDGE_FEATHER_PX,
    decoder_guard_px: int = DEFAULT_DECODER_GUARD_PX,
    max_expand_iters: int = DEFAULT_MAX_EXPAND_ITERS,
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR,
) -> tuple[AdaptiveCropPlan, dict[str, Any]]:
    """Expand or fall back so realized packed support stays clear of interior crop edges.

    State machine invariant: every ``current`` crop_box is evaluated before accept/reject.
    """

    mode = normalize_latent_anchoring_mode(anchoring_config.mode)
    if not latent_anchoring_enabled(mode):
        return plan, {
            "adjusted": False,
            "fallback_full_frame": False,
            "feather_width_px": int(feather_width_px),
            "decoder_guard_px": int(decoder_guard_px),
            "support_radius_model_px": 0,
            "model_size": None,
            "iterations": 0,
            "safety": "legacy_no_check",
        }

    require_matching_mask_thresholds(adaptive_crop_config, anchoring_config)
    image_processor = build_image_processor(processor_config)
    current = plan
    evaluations: list[dict[str, Any]] = []
    last_eval: dict[str, Any] | None = None
    last_core_ok: AdaptiveCropPlan | None = None
    last_core_ok_eval: dict[str, Any] | None = None
    attempted_full_frame = not plan.applied
    attempted_core_preserving_expand = False
    budget_exhausted = False

    def _evaluate(candidate: AdaptiveCropPlan, iteration: int) -> dict[str, Any]:
        evaluation = evaluate_crop_candidate(
            candidate,
            mask,
            adaptive_crop_config=adaptive_crop_config,
            anchoring_config=anchoring_config,
            processor_config=processor_config,
            processor=image_processor,
            feather_width_px=int(feather_width_px),
            decoder_guard_px=int(decoder_guard_px),
            vae_scale_factor=int(vae_scale_factor),
        )
        evaluation["iteration"] = int(iteration)
        evaluation["crop_box"] = list(candidate.crop_box)
        evaluations.append(evaluation)
        return evaluation

    def _probe_expand(
        base: AdaptiveCropPlan,
        *,
        need_left: int,
        need_top: int,
        need_right: int,
        need_bottom: int,
        reason: str,
    ) -> AdaptiveCropPlan | None:
        expanded = expand_crop_plan_for_support(
            base,
            need_left=need_left,
            need_top=need_top,
            need_right=need_right,
            need_bottom=need_bottom,
        )
        if expanded.crop_box == base.crop_box:
            return None
        return _rebuild_plan_with_crop_box(
            expanded,
            expanded.crop_box,
            decision_reason=reason,
        )

    def _core_preserving_search(
        base: AdaptiveCropPlan,
        base_eval: dict[str, Any],
    ) -> AdaptiveCropPlan | None:
        """Search larger crops when a small expand nearest-erases a thin M_core.

        Resize phase can kill single-pixel cores at intermediate widths while larger
        crops remain valid. Probe increasing per-edge budgets from the last core-ok
        crop, preferring the edges that actually need clearance.
        """

        need0 = [int(v) for v in base_eval.get("need", (0, 0, 0, 0))]
        min_step = max(1, int(base_eval.get("min_expand_step_px", 1)))
        width, height = base.original_size
        x1, y1, x2, y2 = base.crop_box
        max_left = x1
        max_top = y1
        max_right = width - x2
        max_bottom = height - y2
        # Candidate budgets: geometric growth from the measured need.
        budgets: list[tuple[int, int, int, int]] = []
        for scale in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
            budgets.append(
                (
                    min(max_left, max(need0[0], min_step) * scale if need0[0] or scale == 1 else 0),
                    min(max_top, max(need0[1], min_step) * scale if need0[1] or scale == 1 else 0),
                    min(max_right, max(need0[2], min_step) * scale if need0[2] or scale == 1 else 0),
                    min(max_bottom, max(need0[3], min_step) * scale if need0[3] or scale == 1 else 0),
                )
            )
            # Also try growing only the active edges by absolute remaining room fractions.
            budgets.append(
                (
                    min(max_left, need0[0] + min_step * scale if need0[0] else 0),
                    min(max_top, need0[1] + min_step * scale if need0[1] else 0),
                    min(max_right, need0[2] + min_step * scale if need0[2] else 0),
                    min(max_bottom, need0[3] + min_step * scale if need0[3] else 0),
                )
            )
        # Deduplicate while preserving order.
        seen: set[tuple[int, int, int, int]] = set()
        ordered: list[tuple[int, int, int, int]] = []
        for budget in budgets:
            if budget == (0, 0, 0, 0) or budget in seen:
                continue
            seen.add(budget)
            ordered.append(budget)

        for budget in ordered:
            candidate = _probe_expand(
                base,
                need_left=budget[0],
                need_top=budget[1],
                need_right=budget[2],
                need_bottom=budget[3],
                reason="expanded_for_latent_anchoring_support_core_preserving",
            )
            if candidate is None:
                continue
            probe = evaluate_crop_candidate(
                candidate,
                mask,
                adaptive_crop_config=adaptive_crop_config,
                anchoring_config=anchoring_config,
                processor_config=processor_config,
                processor=image_processor,
                feather_width_px=int(feather_width_px),
                decoder_guard_px=int(decoder_guard_px),
                vae_scale_factor=int(vae_scale_factor),
            )
            # Record probes for diagnostics but keep the main loop clean.
            probe["iteration"] = -1
            probe["crop_box"] = list(candidate.crop_box)
            probe["probe"] = "core_preserving_search"
            evaluations.append(probe)
            if probe.get("ok"):
                return candidate
            if probe.get("core_survived") and candidate.crop_box != base.crop_box:
                # Progress: a larger core-surviving crop to continue normal search from.
                return candidate
        return None

    for iteration in range(1, int(max_expand_iters) + 1):
        last_eval = _evaluate(current, iteration)

        if last_eval["ok"]:
            break

        if last_eval["core_survived"]:
            last_core_ok = current
            last_core_ok_eval = last_eval
            attempted_core_preserving_expand = False
        else:
            # Intermediate resize widths can erase thin cores while larger crops remain valid.
            if (
                last_core_ok is not None
                and last_core_ok_eval is not None
                and not attempted_core_preserving_expand
            ):
                alt = _core_preserving_search(last_core_ok, last_core_ok_eval)
                attempted_core_preserving_expand = True
                if alt is not None and alt.crop_box != current.crop_box:
                    current = alt
                    continue
            if current.applied and not attempted_full_frame:
                width, height = current.original_size
                current = _rebuild_plan_with_crop_box(
                    current,
                    (0, 0, width, height),
                    decision_reason="fallback_full_frame_after_core_loss",
                )
                attempted_full_frame = True
                continue
            raise ValueError(
                "adaptive crop + latent anchoring could not retain a non-empty processed M_core; "
                f"last_reason={last_eval.get('reason')} crop_box={current.crop_box} "
                f"processed_core_pixels={last_eval.get('processed_core_pixels')}"
            )

        need = tuple(int(v) for v in last_eval.get("need", (0, 0, 0, 0)))
        if not current.applied:
            # Full-frame already evaluated and still unsafe: no further search.
            break

        if need == (0, 0, 0, 0):
            if not attempted_full_frame:
                width, height = current.original_size
                current = _rebuild_plan_with_crop_box(
                    current,
                    (0, 0, width, height),
                    decision_reason="fallback_full_frame_for_latent_anchoring_support",
                )
                attempted_full_frame = True
                continue
            break

        expanded = expand_crop_plan_for_support(
            current,
            need_left=need[0],
            need_top=need[1],
            need_right=need[2],
            need_bottom=need[3],
        )
        if expanded.crop_box == current.crop_box:
            if not attempted_full_frame:
                width, height = current.original_size
                current = _rebuild_plan_with_crop_box(
                    current,
                    (0, 0, width, height),
                    decision_reason="fallback_full_frame_for_latent_anchoring_support",
                )
                attempted_full_frame = True
                continue
            break

        current = expanded
    else:
        budget_exhausted = True

    # Any current that differs from last evaluated box must be evaluated before return.
    if last_eval is None or tuple(last_eval.get("crop_box", ())) != tuple(current.crop_box):
        last_eval = _evaluate(current, len(evaluations) + 1)

    if last_eval is None or not last_eval.get("ok", False):
        if (
            current.applied
            and not attempted_full_frame
            and last_eval is not None
            and last_eval.get("core_survived", False)
        ):
            width, height = current.original_size
            current = _rebuild_plan_with_crop_box(
                current,
                (0, 0, width, height),
                decision_reason=(
                    "fallback_full_frame_search_budget_exhausted"
                    if budget_exhausted
                    else "fallback_full_frame_for_latent_anchoring_support"
                ),
            )
            last_eval = _evaluate(current, len(evaluations) + 1)

    if last_eval is None or tuple(last_eval.get("crop_box", ())) != tuple(current.crop_box):
        last_eval = _evaluate(current, len(evaluations) + 1)

    if last_eval is None or not last_eval.get("ok", False):
        raise ValueError(
            "adaptive crop + latent anchoring failed discrete safety validation; "
            f"mode={mode} final_crop={current.crop_box} "
            f"budget_exhausted={budget_exhausted} last={last_eval}"
        )
    if not last_eval.get("core_survived", False):
        raise ValueError(
            "adaptive crop + latent anchoring final crop lost processed M_core; "
            f"final_crop={current.crop_box} processed_core_pixels={last_eval.get('processed_core_pixels')}"
        )

    support_radius = 0
    if mode == "expanded":
        support_radius = int(anchoring_config.interaction_dilation_px)
    elif mode == "expanded_soft":
        support_radius = int(anchoring_config.interaction_dilation_px) + int(
            anchoring_config.transition_width_px
        )

    meta = {
        "adjusted": current.crop_box != plan.crop_box,
        "fallback_full_frame": (not current.applied) and plan.applied,
        "feather_width_px": int(feather_width_px),
        "decoder_guard_px": int(decoder_guard_px),
        "support_radius_model_px": int(support_radius),
        "model_size": last_eval.get("model_size"),
        "iterations": len(evaluations),
        "initial_crop_box": list(plan.crop_box),
        "final_crop_box": list(current.crop_box),
        "decision_reason": current.decision_reason,
        "safety": "packed_support_lanczos_proxy",
        "support_epsilon": float(_SUPPORT_EPSILON),
        "support_map_resample": "lanczos",
        "processed_core_pixels": last_eval.get("processed_core_pixels"),
        "support_overlap_pixels": last_eval.get("support_overlap_pixels", 0),
        "min_alpha_on_support": last_eval.get("min_alpha_on_support"),
        "budget_exhausted": bool(budget_exhausted),
        "evaluations": evaluations,
        "vae_scale_factor": int(vae_scale_factor),
    }
    if current.applied and last_eval.get("model_size"):
        model_w, model_h = last_eval["model_size"]
        crop_w = current.crop_box[2] - current.crop_box[0]
        crop_h = current.crop_box[3] - current.crop_box[1]
        meta["effective_support_radius_original_xy"] = [
            float(support_radius) * crop_w / float(model_w),
            float(support_radius) * crop_h / float(model_h),
        ]
    else:
        meta["effective_support_radius_original_xy"] = None
    return current, meta


def paste_local_output_to_original(
    original_source: Image.Image,
    local_output: Image.Image,
    plan: AdaptiveCropPlan,
    *,
    edge_feather_px: int = 0,
) -> Image.Image:
    """Map local model RGB into original coordinates, optionally feathering interior edges."""

    source = original_source.convert("RGB")
    if source.size != plan.original_size:
        raise ValueError(
            "paste geometry mismatch: "
            f"plan={plan.original_size}, source={source.size}"
        )
    x1, y1, x2, y2 = plan.crop_box
    crop_size = (x2 - x1, y2 - y1)
    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise ValueError(f"invalid crop box for paste: {plan.crop_box}")

    local = local_output.convert("RGB").resize(crop_size, Image.Resampling.LANCZOS)
    if int(edge_feather_px) <= 0 or not plan.applied:
        restored = source.copy()
        restored.paste(local, (x1, y1))
        return restored

    alpha = build_crop_edge_alpha(plan, feather_width_px=int(edge_feather_px))
    source_np = np.asarray(source, dtype=np.float32)
    restored_np = source_np.copy()
    local_np = np.asarray(local, dtype=np.float32)
    crop_alpha = alpha[y1:y2, x1:x2][..., None]
    restored_np[y1:y2, x1:x2] = crop_alpha * local_np + (1.0 - crop_alpha) * source_np[y1:y2, x1:x2]
    return Image.fromarray(np.clip(restored_np, 0, 255).astype(np.uint8), mode="RGB")


def map_model_mask_to_original(
    mask: Image.Image | np.ndarray,
    *,
    original_size: tuple[int, int],
    crop_plan: Optional[AdaptiveCropPlan],
    resample: Image.Resampling,
) -> Image.Image:
    """Map a model-space single-channel mask into original source coordinates."""

    if isinstance(mask, Image.Image):
        local = mask.convert("L")
    else:
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape={array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("mask must contain only finite values")
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(array.max()) if array.size else 0.0
            min_value = float(array.min()) if array.size else 0.0
            if min_value < 0:
                raise ValueError(f"mask values must be non-negative, got min={min_value}")
            if max_value <= 1.0:
                array = (array * 255.0).round()
            elif max_value <= 255.0:
                array = array.round()
            else:
                raise ValueError(
                    "float mask values must be in [0,1] or [0,255], "
                    f"got min={min_value} max={max_value}"
                )
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        local = Image.fromarray(array, mode="L")

    if crop_plan is not None and crop_plan.original_size != original_size:
        raise ValueError(
            "crop plan original size mismatch for mask mapping: "
            f"plan={crop_plan.original_size}, original={original_size}"
        )

    if crop_plan is None or not crop_plan.applied:
        return local.resize(original_size, resample)
    x1, y1, x2, y2 = crop_plan.crop_box
    crop_size = (x2 - x1, y2 - y1)
    resized = local.resize(crop_size, resample)
    canvas = Image.new("L", original_size, color=0)
    canvas.paste(resized, (x1, y1))
    return canvas


def prepare_adaptive_crop_for_inference(
    *,
    image_size: tuple[int, int],
    mask: Image.Image,
    adaptive_crop_config,
    config,
    processor_config=None,
    feather_width_px: int = DEFAULT_CROP_EDGE_FEATHER_PX,
    decoder_guard_px: int = DEFAULT_DECODER_GUARD_PX,
    vae_scale_factor: int = DEFAULT_VAE_SCALE_FACTOR,
) -> tuple[Optional[AdaptiveCropPlan], dict[str, Any]]:
    """Build and, if needed, adjust a crop plan for optional latent anchoring."""

    if not adaptive_crop_config.predict_enabled:
        return None, {"enabled": False}

    plan = build_adaptive_crop_plan(
        image_size=image_size,
        mask=mask,
        config=adaptive_crop_config,
    )
    anchoring = ensure_latent_anchoring_config(config)
    compatibility: dict[str, Any] = {
        "enabled": True,
        "anchoring_enabled": latent_anchoring_enabled(anchoring.mode),
        "threshold_policy": "matched_when_anchoring_enabled",
        "capability": "supported_via_crop_aware_wrapper",
    }
    if not latent_anchoring_enabled(anchoring.mode):
        compatibility.update(
            {
                "adjusted": False,
                "fallback_full_frame": False,
                "feather_width_px": int(feather_width_px),
                "safety": "legacy_no_check",
            }
        )
        return plan, compatibility

    require_matching_mask_thresholds(adaptive_crop_config, anchoring)
    if processor_config is None:
        processor_config = config.data.init_args.processor
    adjusted_plan, support_meta = ensure_crop_plan_for_latent_anchoring(
        plan,
        mask,
        adaptive_crop_config=adaptive_crop_config,
        anchoring_config=anchoring,
        processor_config=processor_config,
        feather_width_px=feather_width_px,
        decoder_guard_px=decoder_guard_px,
        vae_scale_factor=vae_scale_factor,
    )
    compatibility.update(support_meta)
    return adjusted_plan, compatibility

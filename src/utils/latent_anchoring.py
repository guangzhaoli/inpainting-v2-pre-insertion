"""Inference-only mask construction and configuration helpers for latent anchoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np
import torch

from src.models.complementary_canvas import complementary_canvas_enabled


LATENT_ANCHORING_MODES = (
    "legacy",
    "core",
    "expanded",
    "expanded_soft",
)

_MODE_ALIASES = {
    "a": "legacy",
    "b": "core",
    "c": "expanded",
    "d": "expanded_soft",
    **{mode: mode for mode in LATENT_ANCHORING_MODES},
}

_VARIANT_BY_MODE = {
    "legacy": "A",
    "core": "B",
    "expanded": "C",
    "expanded_soft": "D",
}


@dataclass(frozen=True)
class LatentAnchoringMasks:
    """Image-space masks in the model prediction geometry."""

    core: torch.Tensor
    free: torch.Tensor
    edit_weight: torch.Tensor


def normalize_latent_anchoring_mode(value: Any) -> str:
    value = getattr(value, "value", value)
    normalized = str(value).strip().lower()
    try:
        return _MODE_ALIASES[normalized]
    except KeyError as error:
        allowed = ", ".join(LATENT_ANCHORING_MODES)
        raise ValueError(
            f"latent anchoring mode must be one of {allowed} or A/B/C/D, got {value!r}"
        ) from error


def latent_anchoring_variant(mode: Any) -> str:
    return _VARIANT_BY_MODE[normalize_latent_anchoring_mode(mode)]


def latent_anchoring_enabled(mode: Any) -> bool:
    return normalize_latent_anchoring_mode(mode) != "legacy"


def ensure_latent_anchoring_config(config):
    """Attach default predict settings for lightweight legacy/test config objects."""

    from src.data.config import LatentAnchoringConfig, PredictConfig

    predict = getattr(config, "predict", None)
    if predict is None:
        predict = PredictConfig()
        config.predict = predict
    if getattr(predict, "latent_anchoring", None) is None:
        predict.latent_anchoring = LatentAnchoringConfig()
    return predict.latent_anchoring


def add_latent_anchoring_cli_args(parser) -> None:
    parser.add_argument(
        "--latent-anchoring-mode",
        type=normalize_latent_anchoring_mode,
        choices=LATENT_ANCHORING_MODES,
        default=None,
        help=(
            "Inference A/B/C/D experiment mode: legacy=A current strict composite; core=B core anchoring; "
            "expanded=C binary expanded anchoring; expanded_soft=D expanded anchoring with a soft outer transition. "
            "A/B/C/D aliases are accepted. Defaults to predict.latent_anchoring.mode from YAML."
        ),
    )
    parser.add_argument(
        "--interaction-dilation-px",
        type=int,
        default=None,
        help="M_free dilation radius in model-input pixels for C/D; overrides YAML when provided.",
    )
    parser.add_argument(
        "--transition-width-px",
        type=int,
        default=None,
        help="Soft outer transition width in model-input pixels for D; overrides YAML when provided.",
    )
    parser.add_argument(
        "--latent-anchoring-mask-threshold",
        type=float,
        default=None,
        help="Threshold used to binarize M_core before building anchoring masks; overrides YAML when provided.",
    )
    parser.add_argument(
        "--source-noise-start",
        type=float,
        default=None,
        help=(
            "Replacement tasks: start sampling at the step nearest this σ (e.g. 0.9) from the source "
            "noised inside M_free, so the old object's pose, scale and lighting carry over. Needs B/C/D; "
            "overrides predict.latent_anchoring.source_noise_start from YAML when provided."
        ),
    )
    parser.add_argument(
        "--paste-back",
        choices=("crop_feather", "change_gated"),
        default=None,
        help=(
            "Pixel paste-back of the restored primary output (anchored or complementary-canvas runs): crop_feather "
            "feathers the decoded crop over 8 px at its interior edges; change_gated keeps the output inside M_free "
            "and the exact source beyond its transition band, keeping the output in the band only where it changed. "
            "Overrides predict.latent_anchoring.paste_back from YAML when provided."
        ),
    )


def apply_latent_anchoring_cli_overrides(config, args):
    """Apply optional CLI overrides through Pydantic validation and return the effective config."""

    current = ensure_latent_anchoring_config(config)
    payload = current.model_dump(mode="python")
    overrides = {
        "mode": getattr(args, "latent_anchoring_mode", None),
        "interaction_dilation_px": getattr(args, "interaction_dilation_px", None),
        "transition_width_px": getattr(args, "transition_width_px", None),
        "mask_threshold": getattr(args, "latent_anchoring_mask_threshold", None),
        "source_noise_start": getattr(args, "source_noise_start", None),
        "paste_back": getattr(args, "paste_back", None),
    }
    payload.update({key: value for key, value in overrides.items() if value is not None})
    config.predict.latent_anchoring = type(current).model_validate(payload)
    return config.predict.latent_anchoring


def latent_anchoring_metadata(config_or_settings) -> dict[str, Any]:
    settings = getattr(config_or_settings, "latent_anchoring", config_or_settings)
    mode = normalize_latent_anchoring_mode(settings.mode)
    enabled = latent_anchoring_enabled(mode)
    if mode == "legacy":
        source_hole_mask = "M_core"
        edit_weight = None
        primary_output = "strict_M_core_composite"
    elif mode == "core":
        source_hole_mask = "M_core"
        edit_weight = "hard_M_core"
        primary_output = "latent_anchored_output"
    elif mode == "expanded":
        source_hole_mask = "M_free"
        edit_weight = "hard_M_free"
        primary_output = "latent_anchored_output"
    else:
        source_hole_mask = "M_free"
        edit_weight = "M_free_plus_soft_outer_transition"
        primary_output = "latent_anchored_output"

    return {
        "variant": latent_anchoring_variant(mode),
        "mode": mode,
        "enabled": enabled,
        "mask_threshold": float(settings.mask_threshold),
        "interaction_dilation_px": int(settings.interaction_dilation_px),
        "transition_width_px": int(settings.transition_width_px),
        "source_noise_start": getattr(settings, "source_noise_start", None),
        "paste_back": getattr(settings, "paste_back", "crop_feather"),
        "source_hole_mask": source_hole_mask,
        "placement_and_reference_mask": "M_core",
        "edit_weight": edit_weight,
        "primary_output": primary_output,
        "mvp_scope": "prediction_geometry" if enabled else "legacy_compatible",
        # Wrapper-level Adaptive Crop compatibility is recorded under adaptive_crop.compatibility.
        "adaptive_crop_capability": "supported_via_crop_aware_wrapper" if enabled else "legacy_compatible",
    }


def validate_latent_anchoring_runtime(config, *, height=None, width=None) -> None:
    from src.data.config import check_latent_anchoring_contract

    settings = ensure_latent_anchoring_config(config)
    check_latent_anchoring_contract(getattr(getattr(config, "model", None), "complementary_canvas", None), settings)
    if complementary_canvas_enabled(config):
        if height is not None or width is not None:
            raise ValueError(
                "the complementary canvas does not yet support explicit --height/--width; "
                "use the processor-derived prediction geometry size"
            )
        return
    if not latent_anchoring_enabled(settings.mode):
        return
    if height is not None or width is not None:
        raise ValueError(
            "latent anchoring modes core/expanded/expanded_soft do not yet support explicit "
            "--height/--width; use the processor-derived prediction geometry size"
        )
    adaptive_crop = config.data.init_args.adaptive_crop
    if adaptive_crop.predict_enabled:
        crop_threshold = float(adaptive_crop.mask_threshold)
        anchor_threshold = float(settings.mask_threshold)
        if not math.isclose(crop_threshold, anchor_threshold, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "adaptive_crop.mask_threshold and predict.latent_anchoring.mask_threshold must match "
                f"when both features are enabled; got adaptive_crop={crop_threshold} and "
                f"latent_anchoring={anchor_threshold}"
            )


def _normalize_core_mask(mask: torch.Tensor, threshold: float) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("latent anchoring mask must be a torch.Tensor")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError(
                "latent anchoring mask with four dimensions must have one channel, "
                f"got {tuple(mask.shape)}"
            )
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(
            "latent anchoring mask must be [H,W], [B,H,W], or [B,1,H,W], "
            f"got {tuple(mask.shape)}"
        )
    if not torch.isfinite(mask).all().item():
        raise ValueError("latent anchoring mask must contain only finite values")
    if not math.isfinite(float(threshold)) or threshold < 0 or threshold >= 1:
        raise ValueError("latent anchoring mask threshold must be finite and in [0, 1)")

    mask_float = mask.float()
    threshold_value = float(threshold)
    if float(mask_float.max().item()) > 1.0:
        threshold_value *= 255.0
    core = (mask_float > threshold_value).to(dtype=torch.float32)
    occupied = core.flatten(1).any(dim=1)
    if not bool(occupied.all().item()):
        empty_indices = torch.nonzero(~occupied, as_tuple=False).flatten().tolist()
        raise ValueError(f"latent anchoring requires a non-empty M_core for every sample; empty={empty_indices}")
    return core


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px == 0:
        return mask.copy()
    kernel_size = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1).astype(np.uint8)


def _soft_outer_transition(free_mask: np.ndarray, transition_width_px: int) -> np.ndarray:
    outside = (1 - free_mask).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    weight = np.zeros_like(distance, dtype=np.float32)
    weight[free_mask > 0] = 1.0
    transition = (distance > 0) & (distance < float(transition_width_px))
    normalized = distance[transition] / float(transition_width_px)
    weight[transition] = 0.5 * (1.0 + np.cos(np.pi * normalized))
    return weight


def build_latent_anchoring_masks(
    mask: torch.Tensor,
    *,
    mode: Any,
    mask_threshold: float = 0.5,
    interaction_dilation_px: int = 32,
    transition_width_px: int = 32,
) -> LatentAnchoringMasks:
    """Build hard M_core/M_free and continuous W_edit in image space."""

    normalized_mode = normalize_latent_anchoring_mode(mode)
    if interaction_dilation_px < 0:
        raise ValueError("interaction_dilation_px must be >= 0")
    if transition_width_px < 0:
        raise ValueError("transition_width_px must be >= 0")
    if normalized_mode in {"expanded", "expanded_soft"} and interaction_dilation_px <= 0:
        raise ValueError(f"{normalized_mode} requires interaction_dilation_px > 0")
    if normalized_mode == "expanded_soft" and transition_width_px < 2:
        raise ValueError("expanded_soft requires transition_width_px >= 2")

    core = _normalize_core_mask(mask, mask_threshold)
    device = core.device
    core_np = core.detach().cpu().numpy().astype(np.uint8)
    free_items = []
    edit_items = []

    for item in core_np:
        if normalized_mode in {"expanded", "expanded_soft"}:
            free = _dilate_mask(item, int(interaction_dilation_px))
        else:
            free = item.copy()

        if normalized_mode == "expanded_soft":
            edit_weight = _soft_outer_transition(free, int(transition_width_px))
        else:
            edit_weight = free.astype(np.float32)

        free_items.append(torch.from_numpy(free.astype(np.float32)))
        edit_items.append(torch.from_numpy(edit_weight.astype(np.float32)))

    free = torch.stack(free_items, dim=0).to(device=device)
    edit_weight = torch.stack(edit_items, dim=0).to(device=device)
    if not torch.all(free >= core).item():
        raise RuntimeError("M_free must fully cover M_core")
    if not torch.isfinite(edit_weight).all().item():
        raise RuntimeError("W_edit contains non-finite values")
    if torch.any(edit_weight < 0).item() or torch.any(edit_weight > 1).item():
        raise RuntimeError("W_edit must remain in [0, 1]")
    if not torch.all(edit_weight[free.bool()] == 1).item():
        raise RuntimeError("W_edit must equal 1 throughout M_free")

    return LatentAnchoringMasks(core=core, free=free, edit_weight=edit_weight)

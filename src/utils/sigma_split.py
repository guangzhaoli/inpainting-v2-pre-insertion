"""σ-split conditioning helpers: one LoRA, two noise segments.

High noise (σ > τ) decides position, scale and pose from a loose region and a
reference whose VAE tokens are shuffled (a bag of patches); low noise (σ <= τ)
decides identity and detail from a tight region and the ordered reference.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF


def route_high_noise(
    sigmas: torch.Tensor,
    tau: float,
    overlap: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Return ``[B]`` bools, True for the layout (σ > τ) route.

    Inside ``[τ - overlap, τ + overlap]`` each sample picks a route uniformly
    at random, so both conditions are trained around the hand-off point.
    """

    sigmas = sigmas.detach().float().reshape(-1).cpu()
    high = sigmas > tau
    if overlap > 0:
        in_band = (sigmas >= tau - overlap) & (sigmas <= tau + overlap)
        coin = torch.rand(sigmas.shape, generator=generator) < 0.5
        high = torch.where(in_band, coin, high)
    return high


def _uniform(low: float, high: float, generator: Optional[torch.Generator]) -> float:
    return float(low + (high - low) * torch.rand((), generator=generator).item())


def _composite_foreground(augmented: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Keep the white reference background; photometric jitter changes only the object."""

    keep = (mask > 0.5).unsqueeze(0)
    return torch.where(keep, augmented.clamp(0.0, 1.0), original)


def augment_reference_strong(
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flip, ±rotation and color jitter for the layout route (``[B,3,H,W]`` in [0, 1])."""

    out_images, out_masks = [], []
    for image, mask in zip(images, masks):
        if _uniform(0.0, 1.0, generator) < cfg.flip_prob:
            image, mask = image.flip(-1), mask.flip(-1)
        angle = _uniform(-cfg.rotation_deg, cfg.rotation_deg, generator)
        if angle != 0.0:
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=[1.0, 1.0, 1.0])
            mask = TF.rotate(mask[None], angle, interpolation=InterpolationMode.NEAREST, fill=[0.0])[0]
        jittered = TF.adjust_brightness(image, _uniform(1 - cfg.brightness, 1 + cfg.brightness, generator))
        jittered = TF.adjust_contrast(jittered, _uniform(1 - cfg.contrast, 1 + cfg.contrast, generator))
        jittered = TF.adjust_saturation(jittered, _uniform(1 - cfg.saturation, 1 + cfg.saturation, generator))
        jittered = TF.adjust_hue(jittered, _uniform(-cfg.hue, cfg.hue, generator))
        out_images.append(_composite_foreground(jittered, image, mask))
        out_masks.append(mask)
    return torch.stack(out_images), torch.stack(out_masks)


def augment_reference_light(
    images: torch.Tensor,
    masks: torch.Tensor,
    cfg,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Exposure, gamma, saturation and white-balance jitter for the identity route."""

    out_images = []
    for image, mask in zip(images, masks):
        jittered = image * 2.0 ** _uniform(-cfg.exposure_ev, cfg.exposure_ev, generator)
        jittered = jittered.clamp(0.0, 1.0) ** _uniform(1 - cfg.gamma, 1 + cfg.gamma, generator)
        jittered = TF.adjust_saturation(jittered, _uniform(1 - cfg.saturation, 1 + cfg.saturation, generator))
        temperature = _uniform(-cfg.temperature, cfg.temperature, generator)
        gains = jittered.new_tensor([1.0 + temperature, 1.0, 1.0 - temperature]).view(3, 1, 1)
        out_images.append(_composite_foreground(jittered * gains, image, mask))
    return torch.stack(out_images)


def shuffle_token_segment(
    tokens: torch.Tensor,
    start: int,
    length: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Permute ``tokens[:, start:start+length]`` independently per sample.

    The DiT assigns RoPE positions by sequence index, so this is the same as
    keeping the tokens and shuffling their spatial positions.
    """

    if start < 0 or length <= 0 or start + length > tokens.shape[1]:
        raise ValueError(f"token segment [{start}, {start + length}) is outside sequence {tuple(tokens.shape)}")
    shuffled = tokens.clone()
    for index in range(tokens.shape[0]):
        permutation = torch.randperm(length, generator=generator).to(tokens.device) + start
        shuffled[index, start : start + length] = tokens[index, permutation]
    return shuffled


def handoff_step_index(sigmas: Sequence[float], tau: float, num_steps: int) -> int:
    """First denoising step whose current σ is <= τ."""

    for index in range(num_steps):
        if float(sigmas[index]) <= tau:
            if index == 0:
                break
            return index
    raise ValueError(
        f"sigma hand-off τ={tau} must split the {num_steps}-step schedule "
        f"(σ from {float(sigmas[0]):.3f} to {float(sigmas[num_steps - 1]):.3f})"
    )


def alpha_region_from_logits(
    logits: torch.Tensor,
    token_grid: Tuple[int, int],
    image_size: Tuple[int, int],
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Threshold token logits and upsample them to a ``[B,H,W]`` pixel mask."""

    probabilities = torch.sigmoid(logits.float()).reshape(logits.shape[0], *token_grid)
    region = (probabilities > threshold).float().unsqueeze(1)
    region = F.interpolate(region, size=image_size, mode="nearest")[:, 0]
    return region, probabilities


def add_sigma_handoff_cli_args(parser) -> None:
    parser.add_argument(
        "--sigma-handoff",
        choices=("on", "off"),
        default=None,
        help="σ-split inference hand-off; defaults to predict.sigma_handoff.enabled from YAML.",
    )
    parser.add_argument(
        "--sigma-handoff-tau",
        type=float,
        default=None,
        help="Hand-off σ; defaults to predict.sigma_handoff.tau, then model.sigma_split.tau.",
    )
    parser.add_argument(
        "--sigma-handoff-alpha-threshold",
        type=float,
        default=None,
        help="Alpha probability threshold for the tight region; overrides YAML when provided.",
    )


def apply_sigma_handoff_cli_overrides(config, args):
    """Apply optional CLI overrides through Pydantic validation and return the effective settings."""

    from src.utils.latent_anchoring import ensure_latent_anchoring_config

    ensure_latent_anchoring_config(config)
    current = config.predict.sigma_handoff
    payload = current.model_dump(mode="python")
    switch = getattr(args, "sigma_handoff", None)
    if switch is not None:
        payload["enabled"] = switch == "on"
    for field, attribute in (("tau", "sigma_handoff_tau"), ("alpha_threshold", "sigma_handoff_alpha_threshold")):
        value = getattr(args, attribute, None)
        if value is not None:
            payload[field] = value
    settings = type(current).model_validate(payload)
    if settings.enabled and not (config.model.sigma_split.enabled and config.model.alpha_head.enabled):
        raise ValueError("--sigma-handoff on requires a config with model.sigma_split and model.alpha_head enabled")
    config.predict.sigma_handoff = settings
    return settings

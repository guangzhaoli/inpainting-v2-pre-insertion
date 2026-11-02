"""Token bookkeeping for the complementary canvas.

The target grid is split into the edit region R (condition-hole tokens) and its
complement. The DiT generates only R; the masked-source tokens outside R are
clean context, and the final latent is R scattered into the source latent.

The three-zone canvas adds a ring: the ``ring_tokens`` band of grid tokens
around R. Ring tokens are generated too (they follow R in the target stream),
but start from the noised source with the noise level capped at
``ring_sigma_max`` and are feathered into the source latent at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def complementary_canvas_enabled(config) -> bool:
    canvas = getattr(getattr(config, "model", None), "complementary_canvas", None)
    return bool(canvas is not None and canvas.enabled)


@dataclass(frozen=True)
class CanvasPlan:
    """Row-major indices into the packed ``h*w`` target token grid."""

    target_index: torch.Tensor  # generated tokens: R, then its ring
    source_index: torch.Tensor  # the remaining grid, clean source context
    num_tokens: int
    ring_count: int = 0  # trailing entries of target_index that form the ring
    keep_region_source: bool = False  # the masked-source condition keeps its whole grid, R included

    @property
    def condition_source_index(self) -> torch.Tensor:
        """Masked-source tokens in the condition stream: the complement of R, or the whole grid."""

        if self.keep_region_source:
            return torch.arange(self.num_tokens, device=self.source_index.device)
        return self.source_index

    @property
    def core_count(self) -> int:
        return int(self.target_index.numel()) - self.ring_count

    @property
    def core_index(self) -> torch.Tensor:
        return self.target_index[: self.core_count]

    @property
    def ring_index(self) -> torch.Tensor:
        return self.target_index[self.core_count :]

    @property
    def image_token_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-image token selection for [target, masked source]; later images keep their full grid."""

        return (self.target_index, self.condition_source_index)


def token_region_from_mask(mask: torch.Tensor, token_px: int, threshold: float = 0.5) -> torch.Tensor:
    """``[B,H,W]`` hole mask -> ``[B,(H/token_px)*(W/token_px)]`` bool; a token is in R if any pixel is."""

    if mask.ndim != 3:
        raise ValueError(f"canvas hole mask must be [B,H,W], got {tuple(mask.shape)}")
    height, width = mask.shape[-2:]
    if height % token_px or width % token_px:
        raise ValueError(f"canvas hole mask {(height, width)} must be divisible by the token size {token_px}")
    hole = (mask.detach().float() > threshold).float().unsqueeze(1)
    return F.max_pool2d(hole, kernel_size=token_px).flatten(1) > 0


def _dilate_tokens(region: torch.Tensor, grid: tuple[int, int], radius: int) -> torch.Tensor:
    """Square (Chebyshev) dilation of a flat token mask on its ``(h, w)`` grid."""

    height, width = grid
    pooled = F.max_pool2d(region.view(1, 1, height, width).float(), 2 * radius + 1, stride=1, padding=radius)
    return pooled.flatten() > 0


def canvas_plan_from_region(
    region: torch.Tensor,
    device=None,
    *,
    grid: tuple[int, int] | None = None,
    ring_tokens: int = 0,
    keep_region_source: bool = False,
) -> CanvasPlan:
    """One plan per batch: every sample must share R (training and inference use batch size 1).

    ``ring_tokens > 0`` (needs ``grid``) adds the band of tokens within that Chebyshev
    distance of R to the generated set, after R. ``keep_region_source`` keeps the masked
    source inside R (and the ring) in the condition stream as well.
    """

    region = region.to(device=device, dtype=torch.bool)
    if region.ndim == 2:
        if not bool((region == region[:1]).all()):
            raise ValueError("complementary canvas needs one shared edit region per batch; use batch size 1")
        region = region[0]
    if region.ndim != 1:
        raise ValueError(f"canvas region must be [N] or [B,N], got {tuple(region.shape)}")
    core_index = torch.nonzero(region, as_tuple=False).flatten()
    if core_index.numel() == 0:
        raise ValueError("complementary canvas edit region is empty")
    generated = region
    ring_index = core_index[:0]
    if ring_tokens > 0:
        if grid is None or grid[0] * grid[1] != region.numel():
            raise ValueError(f"the canvas ring needs the {region.numel()}-token grid shape, got {grid}")
        generated = _dilate_tokens(region, grid, ring_tokens)
        ring_index = torch.nonzero(generated & ~region, as_tuple=False).flatten()
    source_index = torch.nonzero(~generated, as_tuple=False).flatten()
    return CanvasPlan(
        target_index=torch.cat([core_index, ring_index]),
        source_index=source_index,
        num_tokens=int(region.numel()),
        ring_count=int(ring_index.numel()),
        keep_region_source=keep_region_source,
    )


def _distance_to_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Euclidean pixel distance to the nearest True pixel of ``[H,W]`` ``mask``, capped at ``radius + 1``."""

    height, width = mask.shape
    padded = F.pad(mask.float()[None, None], (radius,) * 4)[0, 0] > 0
    distance = torch.full(mask.shape, float(radius + 1), device=mask.device)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            step = math.hypot(dy, dx)
            if step > radius:
                continue
            hit = padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            distance = torch.where(hit, distance.clamp(max=step), distance)
    return distance


def ring_feather_weights(plan: CanvasPlan, grid: tuple[int, int], ring_tokens: int, channels: int) -> torch.Tensor:
    """Generated-content weight of each ring token in the packed ``[ring, channels*4]`` latent layout.

    The weight falls linearly with the Euclidean distance (in latent pixels, two per token
    side) from R: ``1 - d / (2 * ring_tokens + 1)``, so it reaches 0 just outside the ring.
    """

    height, width = grid
    core = torch.zeros(height * width, dtype=torch.bool, device=plan.target_index.device)
    core[plan.core_index] = True
    latent_core = core.view(height, width).repeat_interleave(2, 0).repeat_interleave(2, 1)
    span = 2 * ring_tokens + 1
    weight = (1 - _distance_to_mask(latent_core, span) / span).clamp(0, 1)
    # Packed token layout (QwenImage _pack_latents): feature = channel * 4 + dy * 2 + dx.
    per_token = weight.view(height, 2, width, 2).permute(0, 2, 1, 3).reshape(height * width, 4)
    return per_token[plan.ring_index].repeat(1, channels)


def ring_sigma(sigma: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """Ring noise level: the core's σ capped at ``sigma_max``."""

    return sigma.clamp(max=float(sigma_max))


def gather_tokens(tokens: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return tokens.index_select(1, index.to(tokens.device))


def scatter_tokens(base: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Out-of-place ``base`` with ``values`` written at ``index``; differentiable in both inputs."""

    return base.index_copy(1, index.to(base.device), values.to(dtype=base.dtype))


def compose_canvas_latents(
    source: torch.Tensor,
    plan: CanvasPlan,
    generated: torch.Tensor,
    ring_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-grid latent: generated R, the ring feathered into ``source``, ``source`` elsewhere."""

    if plan.ring_count:
        core, ring = generated[:, : plan.core_count], generated[:, plan.core_count :]
        source_ring = gather_tokens(source, plan.ring_index)
        weights = ring_weights.to(device=ring.device, dtype=ring.dtype)
        generated = torch.cat([core, torch.lerp(source_ring.to(ring.dtype), ring, weights)], dim=1)
    return scatter_tokens(source, plan.target_index, generated)

"""Token-level object-occupancy (alpha) head for the target image tokens."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ALPHA_HEAD_FILE_NAME = "alpha_head.safetensors"


class AlphaHead(nn.Module):
    """Predict per-token object occupancy logits from final DiT hidden states.

    The projection is zero-initialized: the head starts at p=0.5 everywhere and
    sends no gradient into the backbone until it has learned a direction.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map ``[B, N, D]`` target-token states to ``[B, N]`` logits."""

        return self.proj(self.norm(hidden_states)).squeeze(-1)


def alpha_occupancy_targets(mask: torch.Tensor, token_size: int) -> torch.Tensor:
    """Fraction of each ``token_size`` pixel block covered by ``mask``, row-major ``[B, N]``.

    The order matches the packed latent sequence (2x2 latent patches of an
    8x VAE, i.e. 16x16 pixels per token).
    """

    if mask.ndim != 3:
        raise ValueError(f"alpha target mask must be [B,H,W], got {tuple(mask.shape)}")
    height, width = mask.shape[-2:]
    if height % token_size or width % token_size:
        raise ValueError(f"alpha target mask {(height, width)} is not divisible by token size {token_size}")
    occupancy = F.avg_pool2d(mask.float().unsqueeze(1), kernel_size=token_size, stride=token_size)
    return occupancy.flatten(1)


def alpha_head_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bce_weight: float,
    dice_weight: float,
    eps: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(weighted total, bce, dice)`` for soft occupancy targets."""

    if logits.shape != targets.shape:
        raise ValueError(f"alpha logits {tuple(logits.shape)} must match targets {tuple(targets.shape)}")
    logits = logits.float()
    targets = targets.to(device=logits.device, dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + eps) / (probabilities.sum(dim=1) + targets.sum(dim=1) + eps)
    dice = dice.mean()
    return float(bce_weight) * bce + float(dice_weight) * dice, bce, dice

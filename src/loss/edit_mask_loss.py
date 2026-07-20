from typing import Optional

from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


MASK_LOSS_MODES = ("token_weighted", "region_balanced")


def map_mask_to_latent(
    image_mask: Tensor,
    vae_scale_factor: int,
    expected_seq_len: Optional[int] = None,
) -> Tensor:
    """
    Args:
        image_mask: [B, H, W] - Binary mask in image space
        vae_scale_factor: image-to-latent spatial downsample factor used by the VAE
        expected_seq_len: optional packed latent sequence length to validate against
    Returns:
        latent_mask: [B, seq_len] - Weights for packed latent
    """
    if image_mask.ndim != 3:
        raise ValueError(f"image_mask must be [B, H, W], got {tuple(image_mask.shape)}")
    if not isinstance(vae_scale_factor, int) or vae_scale_factor <= 0:
        raise ValueError(f"vae_scale_factor must be a positive integer, got {vae_scale_factor!r}")

    B, H, W = image_mask.shape
    packed_multiple = vae_scale_factor * 2
    if H % packed_multiple != 0 or W % packed_multiple != 0:
        raise ValueError(
            "image_mask height and width must be divisible by vae_scale_factor * 2 "
            f"for packed latents, got mask size {(H, W)} and vae_scale_factor={vae_scale_factor}"
        )

    # Step 1: VAE-aligned downsampling
    # [B, H, W] -> [B, H/vae_scale_factor, W/vae_scale_factor]
    latent_h, latent_w = H // vae_scale_factor, W // vae_scale_factor
    mask_latent = F.avg_pool2d(
        image_mask.float().unsqueeze(1),
        kernel_size=vae_scale_factor,
        stride=vae_scale_factor,
    ).squeeze(1)  # [B, latent_h, latent_w]

    # Step 2: Packing simulation
    # [B, latent_h, latent_w] -> [B, latent_h//2, latent_w//2, 4]
    # First reshape to separate 2x2 patches, then fold
    patches = mask_latent.reshape(B, latent_h//2, 2, latent_w//2, 2)
    patches = patches.permute(0, 1, 3, 2, 4).contiguous().view(B, latent_h//2, latent_w//2, 4)

    # Step 3: Patch-wise maximum (preserve text regions)
    # [B, latent_h//2, latent_w//2, 4] -> [B, latent_h//2, latent_w//2]
    packed_mask = patches.max(dim=-1)[0]

    # Step 4: Flatten to sequence
    # [B, latent_h//2, latent_w//2] -> [B, seq_len]
    seq_len = (latent_h // 2) * (latent_w // 2)
    if expected_seq_len is not None and seq_len != expected_seq_len:
        raise ValueError(
            f"Packed mask seq_len {seq_len} does not match image latent seq_len {expected_seq_len}. "
            f"Check vae_scale_factor={vae_scale_factor} and latent packing."
        )
    return packed_mask.view(B, seq_len)


class MaskEditLoss(nn.Module):
    """Mask-aware MSE with two reduction modes.

    - ``token_weighted``: global weighted mean over all tokens
      ``sum(w * e) / sum(w)`` with ``w = fg_w * m + bg_w * (1-m)``.
      Mask area still controls how much total mass the edit region gets.

    - ``region_balanced``: mean fg/bg losses separately, then mix with fixed ratio
      ``(fg_w * mean_fg + bg_w * mean_bg) / (fg_w + bg_w)``.
      Edit vs background contribution is independent of mask area.
    """

    def __init__(
        self,
        forground_weight: float = 2.0,
        background_weight: float = 1.0,
        mode: str = "token_weighted",
        eps: float = 1e-8,
    ):
        super().__init__()
        if mode not in MASK_LOSS_MODES:
            raise ValueError(f"mode must be one of {MASK_LOSS_MODES}, got {mode!r}")
        if forground_weight <= 0 or background_weight <= 0:
            raise ValueError("foreground/background weights must be > 0")
        self.forground_weight = float(forground_weight)
        self.background_weight = float(background_weight)
        self.mode = mode
        self.eps = float(eps)

    def forward(self, mask, model_pred, target, weighting=None):
        """
        计算 mask 加权的 loss
        Args:
            mask: [B, seq_len] - 掩码，1 表示修改区域，0 表示背景区域（可 soft）
            model_pred: [B, seq_len, channels] - 模型预测结果
            target: [B, seq_len, channels] - 目标值
            weighting: [B, seq_len, 1] - 可选的时间步权重
        Returns:
            torch.Tensor - scalar loss
        """
        element_loss = (model_pred.float() - target.float()) ** 2

        if weighting is not None:
            element_loss = weighting.float() * element_loss

        if self.mode == "token_weighted":
            return self._token_weighted(mask, element_loss)
        return self._region_balanced(mask, element_loss)

    def _token_weighted(self, mask: Tensor, element_loss: Tensor) -> Tensor:
        # mask: [B, seq_len] -> weight_mask: [B, seq_len, 1]
        weight_mask = mask.float() * self.forground_weight + (1.0 - mask.float()) * self.background_weight
        weight_mask = weight_mask.unsqueeze(-1)

        weighted_loss = element_loss * weight_mask
        normalizer = weight_mask.expand_as(element_loss).reshape(element_loss.shape[0], -1).sum(dim=1)
        if torch.any(normalizer <= 0):
            raise ValueError("MaskEditLoss normalizer must be positive. Check foreground/background weights.")

        per_sample_loss = weighted_loss.reshape(element_loss.shape[0], -1).sum(dim=1) / normalizer
        return per_sample_loss.mean()

    def _region_balanced(self, mask: Tensor, element_loss: Tensor) -> Tensor:
        """L = (α * mean_fg + β * mean_bg) / (α + β), independent of mask area.

        Soft masks are supported: region means are weighted by mask / (1-mask).
        If a sample has no fg or no bg mass, fall back to the non-empty region mean.
        """
        batch = element_loss.shape[0]
        mask_f = mask.float().unsqueeze(-1)  # [B, seq_len, 1]
        fg_map = mask_f.expand_as(element_loss)
        bg_map = (1.0 - mask_f).expand_as(element_loss)

        flat_loss = element_loss.reshape(batch, -1)
        flat_fg = fg_map.reshape(batch, -1)
        flat_bg = bg_map.reshape(batch, -1)

        fg_mass = flat_fg.sum(dim=1)
        bg_mass = flat_bg.sum(dim=1)

        # Avoid 0/0; invalid regions are overwritten by the branch below.
        fg_mean = (flat_loss * flat_fg).sum(dim=1) / fg_mass.clamp_min(self.eps)
        bg_mean = (flat_loss * flat_bg).sum(dim=1) / bg_mass.clamp_min(self.eps)

        has_fg = fg_mass > self.eps
        has_bg = bg_mass > self.eps
        has_both = has_fg & has_bg

        alpha = self.forground_weight
        beta = self.background_weight
        mixed = (alpha * fg_mean + beta * bg_mean) / (alpha + beta)

        per_sample = torch.where(
            has_both,
            mixed,
            torch.where(has_fg, fg_mean, bg_mean),
        )
        if torch.any(~(has_fg | has_bg)):
            raise ValueError(
                "MaskEditLoss region_balanced requires positive foreground or background mass per sample."
            )
        return per_sample.mean()

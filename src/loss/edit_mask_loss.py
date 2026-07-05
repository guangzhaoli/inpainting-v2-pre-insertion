from typing import Optional

from torch import Tensor
import torch.nn.functional as F
import torch
import torch.nn as nn


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
    def __init__(self, forground_weight=2.0, background_weight=1.0):
        super().__init__()
        self.forground_weight = forground_weight
        self.background_weight = background_weight

    def forward(self, mask, model_pred, target, weighting=None):
        """
        计算mask加权的loss
        Args:
            mask: [B, seq_len] - 二进制掩码，1表示修改区域，0表示背景区域
            model_pred: [B, seq_len, channels] - 模型预测结果
            target: [B, seq_len, channels] - 目标值
            weighting: [B, seq_len, 1] - 可选的时间步权重
        Returns:
            torch.Tensor - 加权后的loss值
        """
        # 计算基础element-wise loss
        element_loss = (model_pred.float() - target.float()) ** 2

        # 如果有weighting，应用到element_loss
        if weighting is not None:
            element_loss = weighting.float() * element_loss

        # 创建权重掩码：文本区域权重更高
        # mask: [B, seq_len] -> weight_mask: [B, seq_len, 1]
        weight_mask = (mask * self.forground_weight + (1 - mask) * self.background_weight)
        weight_mask = weight_mask.unsqueeze(-1)  # [B, seq_len, 1]

        # Apply mask weights while preserving the loss scale for different mask areas.
        weighted_loss = element_loss * weight_mask
        normalizer = weight_mask.expand_as(element_loss).reshape(target.shape[0], -1).sum(dim=1)
        if torch.any(normalizer <= 0):
            raise ValueError("MaskEditLoss normalizer must be positive. Check foreground/background weights.")

        per_sample_loss = weighted_loss.reshape(target.shape[0], -1).sum(dim=1) / normalizer
        loss = per_sample_loss.mean()
        return loss

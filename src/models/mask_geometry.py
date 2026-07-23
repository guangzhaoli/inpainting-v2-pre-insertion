"""Geometric mask encoding and lightweight tokenization for image DiTs."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_real(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if mask.is_complex():
        raise TypeError("mask must have a real-valued or boolean dtype")

    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("four-dimensional mask must have shape [B,1,H,W], " f"got {tuple(mask.shape)}")
        mask = mask[:, 0]
    elif mask.ndim != 3:
        raise ValueError("mask must have shape [B,H,W] or [B,1,H,W], " f"got {tuple(mask.shape)}")

    if any(size <= 0 for size in mask.shape):
        raise ValueError(f"mask dimensions must be non-empty, got {tuple(mask.shape)}")
    if not bool(torch.isfinite(mask).all().item()):
        raise ValueError("mask must contain only finite values")
    return mask


def _signed_distance(binary_mask: np.ndarray) -> np.ndarray:
    """Return a half-pixel-corrected signed EDT for a non-degenerate mask.

    OpenCV measures distance between pixel centres, so two samples on opposite
    sides of a one-pixel interface are one pixel apart. The actual raster-cell
    interface lies halfway between them. Subtracting half a pixel from the
    distance to the nearest opposite-class centre therefore assigns ``+0.5``
    and ``-0.5`` to the two boundary-adjacent samples instead of ``+1`` and
    ``-1``. No pixel-centre sample is assigned a synthetic zero distance.

    The boundary is relative to the finite image domain: pixels beyond an
    image edge are not treated as an additional outside class. This convention
    makes the transform exactly complement-antisymmetric,
    ``D(1 - M) == -D(M)``, including for masks that touch an image edge.
    """

    inside = cv2.distanceTransform(binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(binary_mask != 0, inside - 0.5, -(outside - 0.5)).astype(
        np.float32,
        copy=False,
    )


def build_mask_geometry(
    mask: torch.Tensor,
    sigma_px: float,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Build ``[M, tanh(D/sigma), exp(-abs(D)/sigma)]`` in image space.

    Args:
        mask: A finite tensor shaped ``[B,H,W]`` or ``[B,1,H,W]``. Values
            greater than ``threshold`` are considered inside the mask.
        sigma_px: Positive boundary falloff scale, measured in input pixels.
        threshold: Finite binarization threshold in ``[0, 1)``.

    Returns:
        A float32 tensor shaped ``[B,3,H,W]`` on the input device. The signed
        distance convention is positive inside and negative outside. For a
        non-degenerate mask, samples immediately adjacent to a raster-cell
        interface have signed distances ``+0.5`` and ``-0.5``; consequently,
        the largest sampled boundary-ring value is
        ``exp(-0.5 / sigma_px)`` rather than one. Empty and full masks use the
        limiting encodings ``[0,-1,0]`` and ``[1,+1,0]``.

    Notes:
        Binarization and OpenCV's Euclidean distance transform are intentionally
        non-differentiable. The resulting geometry conditions a trainable mask
        tokenizer; the input mask itself is not expected to receive gradients.
    """

    sigma_px = _validate_real("sigma_px", sigma_px)
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be greater than zero, got {sigma_px}")

    threshold = _validate_real("threshold", threshold)
    if threshold < 0 or threshold >= 1:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")

    mask = _normalize_mask(mask)
    binary = (mask.detach().float() > threshold).to(dtype=torch.uint8)
    binary_batch = binary.cpu().numpy()
    batch_size, height, width = binary_batch.shape
    geometry = np.empty((batch_size, 3, height, width), dtype=np.float32)

    for batch_index, binary_mask in enumerate(binary_batch):
        binary_mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
        geometry[batch_index, 0] = binary_mask

        occupied = int(binary_mask.sum())
        if occupied == 0:
            geometry[batch_index, 1].fill(-1.0)
            geometry[batch_index, 2].fill(0.0)
            continue
        if occupied == binary_mask.size:
            geometry[batch_index, 1].fill(1.0)
            geometry[batch_index, 2].fill(0.0)
            continue

        signed_distance = _signed_distance(binary_mask)
        normalized_distance = signed_distance / sigma_px
        geometry[batch_index, 1] = np.tanh(normalized_distance)
        geometry[batch_index, 2] = np.exp(-np.abs(normalized_distance))

    return torch.from_numpy(geometry).to(device=mask.device)


class MaskGeometryTokenizer(nn.Module):
    """Project three-channel mask geometry into target-grid image tokens.

    The final 1x1 projection is zero-initialized, so attaching the tokenizer to
    an existing DiT as a residual branch preserves its initial behavior.

    When reducing spatial resolution, channel zero becomes the fractional mask
    occupancy of each target token and the signed-distance channel is averaged
    over the same footprint. Thus a tiny foreground fragment is not promoted
    to a fully-interior token with an outside-valued signed channel. The ring is
    max-pooled deliberately: at token resolution it records whether any source
    sample in the footprint is close to the boundary.
    """

    _RESIZE_MODES = ("bilinear", "area")

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        resize_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        for name, value in (
            ("hidden_channels", hidden_channels),
            ("out_channels", out_channels),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if resize_mode not in self._RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {self._RESIZE_MODES}, got {resize_mode!r}")

        self.resize_mode = resize_mode
        self.conv_in = nn.Conv2d(3, int(hidden_channels), kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(int(hidden_channels), int(out_channels), kernel_size=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    @staticmethod
    def _validate_target_size(target_size: Sequence[int]) -> tuple[int, int]:
        if isinstance(target_size, (str, bytes)) or not isinstance(target_size, Sequence):
            raise TypeError("target_size must be a (height, width) sequence")
        if len(target_size) != 2:
            raise ValueError(f"target_size must contain (height, width), got {target_size!r}")
        height, width = target_size
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (height, width)):
            raise TypeError(f"target_size values must be integers, got {target_size!r}")
        if height <= 0 or width <= 0:
            raise ValueError(f"target_size values must be positive, got {target_size!r}")
        return int(height), int(width)

    def forward(
        self,
        geometry: torch.Tensor,
        target_size: Sequence[int],
    ) -> torch.Tensor:
        """Resize geometry and return row-major tokens shaped ``[B,N,C]``."""

        if not isinstance(geometry, torch.Tensor):
            raise TypeError(f"geometry must be a torch.Tensor, got {type(geometry).__name__}")
        if geometry.ndim != 4 or geometry.shape[1] != 3:
            raise ValueError("geometry must have shape [B,3,H,W], " f"got {tuple(geometry.shape)}")
        if any(size <= 0 for size in geometry.shape):
            raise ValueError(f"geometry dimensions must be non-empty, got {tuple(geometry.shape)}")
        if geometry.is_complex():
            raise TypeError("geometry must have a real-valued dtype")
        # Geometry produced by ``build_mask_geometry`` is already finite. Keep
        # the defensive check for CPU callers, but avoid a device-wide
        # synchronization on every diffusion step when the cached condition is
        # resident on GPU.
        if geometry.device.type == "cpu" and not bool(torch.isfinite(geometry).all().item()):
            raise ValueError("geometry must contain only finite values")

        target_size = self._validate_target_size(target_size)
        geometry = geometry.to(
            device=self.conv_in.weight.device,
            dtype=self.conv_in.weight.dtype,
        )
        geometry = self._resize_geometry(geometry, target_size)

        features = self.conv_out(self.activation(self.conv_in(geometry)))
        return features.flatten(2).transpose(1, 2).contiguous()

    def _resize_geometry(
        self,
        geometry: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Resize geometry while preserving consistent token-level semantics.

        Downsampling uses the same area footprint for mask occupancy and signed
        distance, while the ring uses max pooling as a boundary-presence signal.
        ``resize_mode`` controls only a subsequent upsampling stage (including
        mixed resize operations where one axis shrinks and the other grows).
        """

        source_size = tuple(int(value) for value in geometry.shape[-2:])
        if source_size == target_size:
            return geometry

        # Handle shrinking before growing so a mixed-axis resize cannot skip
        # the occupancy-preserving reduction on its shrinking axis.
        reduced_size = tuple(min(target, source) for target, source in zip(target_size, source_size))
        if reduced_size != source_size:
            occupancy = F.adaptive_avg_pool2d(
                geometry[:, 0:1],
                output_size=reduced_size,
            )
            signed_distance = F.adaptive_avg_pool2d(
                geometry[:, 1:2],
                output_size=reduced_size,
            )
            boundary_ring = F.adaptive_max_pool2d(
                geometry[:, 2:3],
                output_size=reduced_size,
            )
            geometry = torch.cat([occupancy, signed_distance, boundary_ring], dim=1)

        if reduced_size == target_size:
            return geometry
        if self.resize_mode == "bilinear":
            return F.interpolate(
                geometry,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return F.interpolate(geometry, size=target_size, mode="area")

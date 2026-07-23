"""Region-aware pixel losses for insertion training.

Objective (region terms are further scaled by w(t) outside or via sample_weights):
  total = latent_mse_weight * latent_mse          # applied outside
        + fg_lpips_weight * LPIPS(bbox-crop FG)   # anti-blur, crop-then-perceive
        + bg_l1_weight * L1(background pixels)
        + ref_dino_weight * DINO(pred crop, ref)  # identity; default off

DINO is an identity term against the reference object, not a replacement for
foreground LPIPS anti-blur supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_VALID_IMAGE_RANGES = {"0-1", "-1-1", "auto"}


class _TorchHubDinoAdapter(nn.Module):
    """Expose a local Meta DINOv2 encoder with the Transformers call signature."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Do not wrap this in no_grad: the prediction branch needs gradients with
        # respect to pixels even though the frozen encoder parameters do not train.
        return self.encoder(pixel_values)


def _ensure_bchw_mask(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Return float mask as [B, 1, H, W] in [0, 1]."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"pixel mask must be [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.float(), size=(height, width), mode="nearest")
    return mask.float().clamp(0.0, 1.0)


def _resolve_image_range(images: torch.Tensor, input_range: str = "auto") -> str:
    """Resolve image value range. Prefer an explicit contract from the caller."""
    if input_range not in _VALID_IMAGE_RANGES:
        raise ValueError(f"input_range must be one of {_VALID_IMAGE_RANGES}, got {input_range!r}")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
    if input_range != "auto":
        return input_range

    min_val = float(images.detach().amin().item())
    max_val = float(images.detach().amax().item())
    if min_val < -1.0e-3:
        if min_val < -1.0 - 1.0e-3 or max_val > 1.0 + 1.0e-3:
            raise ValueError(
                f"images look like [-1,1] but are out of range: min={min_val:.4f}, max={max_val:.4f}"
            )
        return "-1-1"
    if max_val <= 1.0 + 1.0e-3:
        # Ambiguous after masking; prefer explicit range in production paths.
        return "0-1"
    raise ValueError(
        f"images must be in [0,1] or [-1,1], got min={min_val:.4f}, max={max_val:.4f}"
    )


def _images_to_lpips_range(images: torch.Tensor, input_range: str = "auto") -> torch.Tensor:
    """Map images to LPIPS expected range [-1, 1]."""
    resolved = _resolve_image_range(images, input_range=input_range)
    images = images.float()
    if resolved == "0-1":
        return images.clamp(0.0, 1.0) * 2.0 - 1.0
    return images.clamp(-1.0, 1.0)


def _images_to_unit_range(images: torch.Tensor, input_range: str = "auto") -> torch.Tensor:
    """Map images to [0, 1] for pixel L1 / DINO prep."""
    resolved = _resolve_image_range(images, input_range=input_range)
    images = images.float()
    if resolved == "0-1":
        return images.clamp(0.0, 1.0)
    return (images.clamp(-1.0, 1.0) + 1.0) * 0.5


def _mask_bboxes(
    mask: torch.Tensor,
    *,
    mask_threshold: float = 0.5,
    pad_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute padded bboxes from a binary mask.

    Returns:
        boxes: [B, 4] int64 as (y0, y1, x0, x1) with y1/x1 exclusive
        valid: [B] bool
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask must be [B,H,W] or [B,1,H,W], got {tuple(mask.shape)}")
    binary = mask[:, 0] >= mask_threshold
    batch_size, height, width = binary.shape
    boxes = torch.zeros(batch_size, 4, device=mask.device, dtype=torch.long)
    valid = torch.zeros(batch_size, device=mask.device, dtype=torch.bool)
    for i in range(batch_size):
        ys, xs = torch.where(binary[i])
        if ys.numel() == 0:
            boxes[i] = torch.tensor([0, 1, 0, 1], device=mask.device, dtype=torch.long)
            continue
        y0 = int(ys.min().item())
        y1 = int(ys.max().item()) + 1
        x0 = int(xs.min().item())
        x1 = int(xs.max().item()) + 1
        bh = max(1, y1 - y0)
        bw = max(1, x1 - x0)
        pad_y = int(round(bh * pad_ratio))
        pad_x = int(round(bw * pad_ratio))
        y0 = max(0, y0 - pad_y)
        x0 = max(0, x0 - pad_x)
        y1 = min(height, y1 + pad_y)
        x1 = min(width, x1 + pad_x)
        boxes[i, 0] = y0
        boxes[i, 1] = y1
        boxes[i, 2] = x0
        boxes[i, 3] = x1
        valid[i] = True
    return boxes, valid


def _crop_resize_with_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    *,
    out_size: int,
    mask: Optional[torch.Tensor] = None,
    zero_outside_mask: bool = False,
) -> torch.Tensor:
    """Crop each sample with the provided boxes and resize to square [B,3,S,S]."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"images must be [B,3,H,W], got {tuple(images.shape)}")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes must be [B,4], got {tuple(boxes.shape)}")
    if images.shape[0] != boxes.shape[0]:
        raise ValueError("images/boxes batch size mismatch")
    if out_size <= 0:
        raise ValueError(f"out_size must be > 0, got {out_size}")

    if zero_outside_mask:
        if mask is None:
            raise ValueError("zero_outside_mask requires mask")
        mask_b = _ensure_bchw_mask(mask, images.shape[-2], images.shape[-1]).to(
            device=images.device, dtype=images.dtype
        )
        images = images * mask_b

    crops = []
    for i in range(images.shape[0]):
        y0, y1, x0, x1 = [int(v) for v in boxes[i].tolist()]
        y1 = max(y1, y0 + 1)
        x1 = max(x1, x0 + 1)
        crop = images[i : i + 1, :, y0:y1, x0:x1]
        crop = F.interpolate(crop, size=(out_size, out_size), mode="bilinear", align_corners=False)
        crops.append(crop[0])
    return torch.stack(crops, dim=0)


def _masked_bbox_crop(
    images: torch.Tensor,
    mask: torch.Tensor,
    *,
    out_size: int,
    pad_ratio: float = 0.1,
    mask_threshold: float = 0.5,
    input_range: str = "auto",
    output_range: str = "0-1",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop by mask bbox, resize to square.

    Returns:
        crops: [B,3,S,S] in the requested output_range
        valid: [B] bool
    """
    if output_range not in {"0-1", "-1-1"}:
        raise ValueError(f"output_range must be 0-1 or -1-1, got {output_range!r}")
    if out_size <= 0:
        raise ValueError(f"out_size must be > 0, got {out_size}")
    if pad_ratio < 0:
        raise ValueError(f"pad_ratio must be >= 0, got {pad_ratio}")

    # Convert range BEFORE masking so zeroed bg cannot corrupt auto-detect.
    if output_range == "0-1":
        images_c = _images_to_unit_range(images, input_range=input_range)
    else:
        images_c = _images_to_lpips_range(images, input_range=input_range)

    height, width = int(images_c.shape[-2]), int(images_c.shape[-1])
    mask_b = _ensure_bchw_mask(mask, height, width)
    boxes, valid = _mask_bboxes(mask_b, mask_threshold=mask_threshold, pad_ratio=pad_ratio)
    crops = _crop_resize_with_boxes(
        images_c,
        boxes,
        out_size=out_size,
        mask=mask_b,
        zero_outside_mask=True,
    )
    return crops, valid


def _imagenet_normalize(images_01: torch.Tensor) -> torch.Tensor:
    mean = images_01.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images_01.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (images_01.clamp(0.0, 1.0) - mean) / std


def region_timestep_weights(
    sigmas: torch.Tensor,
    *,
    power: float = 2.0,
    max_sigma: float = 1.0,
) -> torch.Tensor:
    """Per-sample region-loss weights that favor low noise.

    w = (1 - sigma)^power for sigma <= max_sigma, else 0.
    Higher power concentrates weight more strongly on clean timesteps.
    """
    if power < 0:
        raise ValueError(f"power must be >= 0, got {power}")
    if max_sigma < 0:
        raise ValueError(f"max_sigma must be >= 0, got {max_sigma}")
    sigma = sigmas.detach().float().reshape(sigmas.shape[0], -1)[:, 0].clamp(0.0, 1.0)
    weights = (1.0 - sigma).clamp(min=0.0).pow(float(power))
    if max_sigma < 1.0:
        weights = torch.where(sigma <= float(max_sigma), weights, torch.zeros_like(weights))
    return weights


# Default torchvision trunk filenames under $TORCH_HOME/hub/checkpoints/
_LPIPS_TRUNK_FILENAMES = {
    "alex": "alexnet-owt-7be5be79.pth",
    "vgg": "vgg16-397923af.pth",
    "vgg16": "vgg16-397923af.pth",
}


def build_lpips_module(
    net: str = "alex",
    *,
    linear_path: Optional[str] = None,
    torch_home: Optional[str] = None,
    trunk_path: Optional[str] = None,
    verbose: bool = False,
) -> nn.Module:
    """Construct LPIPS with explicit offline weight paths.

    Args:
        net: trunk name (alex / vgg / squeeze)
        linear_path: LPIPS calibrated linear layers (.pth). Maps to LPIPS(model_path=...).
        torch_home: directory used as TORCH_HOME so torchvision loads trunks from
            ``{torch_home}/hub/checkpoints/<alexnet|vgg16>.pth``.
        trunk_path: optional explicit trunk checkpoint file. When set, it is copied/linked
            into the torch hub checkpoints dir under the expected filename (if needed).
        verbose: pass through to lpips.LPIPS
    """
    import os
    from pathlib import Path

    import lpips

    net = str(net)
    if net not in {"alex", "vgg", "vgg16", "squeeze"}:
        raise ValueError(f"unsupported lpips net {net!r}")

    # Point torchvision hub at the offline trunk directory before LPIPS builds the trunk.
    if torch_home is not None:
        torch_home_path = Path(torch_home).expanduser().resolve()
        if not torch_home_path.is_dir():
            raise FileNotFoundError(f"lpips torch_home not found: {torch_home_path}")
        ckpt_dir = torch_home_path / "hub" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home_path)

        if trunk_path is not None:
            trunk_src = Path(trunk_path).expanduser().resolve()
            if not trunk_src.is_file():
                raise FileNotFoundError(f"lpips trunk_path not found: {trunk_src}")
            expected_name = _LPIPS_TRUNK_FILENAMES.get(net)
            if expected_name is None:
                raise ValueError(f"no known trunk filename mapping for net={net!r}")
            trunk_dst = ckpt_dir / expected_name
            if trunk_src != trunk_dst:
                # Prefer symlink; fall back to hard copy if cross-device symlink fails.
                if trunk_dst.exists() or trunk_dst.is_symlink():
                    trunk_dst.unlink()
                try:
                    trunk_dst.symlink_to(trunk_src)
                except OSError:
                    import shutil

                    shutil.copy2(trunk_src, trunk_dst)
        else:
            # Validate the expected trunk file is present for offline runs.
            expected_name = _LPIPS_TRUNK_FILENAMES.get(net)
            if expected_name is not None:
                expected = ckpt_dir / expected_name
                if not expected.is_file():
                    raise FileNotFoundError(
                        f"LPIPS trunk checkpoint missing for net={net!r}: {expected}. "
                        f"Set loss.lpips_trunk_path or place the file under {ckpt_dir}"
                    )

    model_path = None
    if linear_path is not None:
        linear = Path(linear_path).expanduser().resolve()
        if not linear.is_file():
            raise FileNotFoundError(f"lpips linear weights not found: {linear}")
        model_path = str(linear)

    module = lpips.LPIPS(
        net=net if net != "vgg16" else "vgg",
        model_path=model_path,
        verbose=verbose,
    )
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
    return module


class RegionPerceptualLoss(nn.Module):
    """Foreground crop-LPIPS + background L1 + optional reference DINO identity.

    Latent MSE is applied outside this module. Optional per-sample weights implement w(t).
    """

    def __init__(
        self,
        fg_metric: str = "lpips",
        lpips_net: str = "vgg",
        lpips_input_size: int = 256,
        lpips_bbox_pad_ratio: float = 0.1,
        mask_threshold: float = 0.5,
        *,
        lpips_linear_path: Optional[str] = None,
        lpips_torch_home: Optional[str] = None,
        lpips_trunk_path: Optional[str] = None,
        enable_ref_dino: bool = False,
        dino_backend: str = "transformers",
        dino_model_name: str = "facebook/dinov2-small",
        dino_repo_path: Optional[str] = None,
        dino_checkpoint_path: Optional[str] = None,
        dino_input_size: int = 224,
        dino_bbox_pad_ratio: float = 0.1,
        dino_encoder: Optional[nn.Module] = None,
        dino_embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # Backward-compatible alias used by older configs/tests.
        lpips_downsample: Optional[int] = None,
    ):
        super().__init__()
        if fg_metric not in {"lpips", "none"}:
            raise ValueError(
                f"fg_metric must be one of {{'lpips','none'}}, got {fg_metric!r}. "
                "Use ref_dino for reference identity, not fg_metric."
            )
        if lpips_net not in {"alex", "vgg", "squeeze"}:
            raise ValueError(f"lpips_net must be one of alex/vgg/squeeze, got {lpips_net!r}")
        if lpips_downsample is not None:
            lpips_input_size = int(lpips_downsample)
        if int(lpips_input_size) <= 0:
            raise ValueError(f"lpips_input_size must be positive, got {lpips_input_size}")
        if float(lpips_bbox_pad_ratio) < 0:
            raise ValueError(f"lpips_bbox_pad_ratio must be >= 0, got {lpips_bbox_pad_ratio}")
        if not 0.0 <= float(mask_threshold) <= 1.0:
            raise ValueError(f"mask_threshold must be in [0,1], got {mask_threshold}")
        if int(dino_input_size) <= 0:
            raise ValueError(f"dino_input_size must be > 0, got {dino_input_size}")
        if float(dino_bbox_pad_ratio) < 0:
            raise ValueError(f"dino_bbox_pad_ratio must be >= 0, got {dino_bbox_pad_ratio}")

        self.fg_metric = fg_metric
        self.lpips_net = lpips_net
        self.lpips_input_size = int(lpips_input_size)
        self.lpips_bbox_pad_ratio = float(lpips_bbox_pad_ratio)
        # Keep old attribute name so existing call sites/logging do not break.
        self.lpips_downsample = self.lpips_input_size
        self.mask_threshold = float(mask_threshold)
        self.lpips_linear_path = lpips_linear_path
        self.lpips_torch_home = lpips_torch_home
        self.lpips_trunk_path = lpips_trunk_path
        self.enable_ref_dino = bool(enable_ref_dino)
        self.dino_backend = dino_backend
        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_checkpoint_path = dino_checkpoint_path
        self.dino_input_size = int(dino_input_size)
        self.dino_bbox_pad_ratio = float(dino_bbox_pad_ratio)
        self.lpips: Optional[nn.Module] = None
        self.dino: Optional[nn.Module] = None
        self._dino_embed_fn = dino_embed_fn

        # LPIPS and externally loaded DINO weights are intentionally lazy. This
        # keeps construction safe for offline jobs and configurations whose
        # corresponding loss weights are zero.
        if dino_encoder is not None:
            self.dino = dino_encoder
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False

    def _ensure_lpips(self, device: torch.device) -> nn.Module:
        if self.lpips is None:
            self.lpips = build_lpips_module(
                net=self.lpips_net,
                linear_path=self.lpips_linear_path,
                torch_home=self.lpips_torch_home,
                trunk_path=self.lpips_trunk_path,
                verbose=False,
            ).to(device=device)
        return self.lpips

    def _ensure_dino(self, device: torch.device) -> nn.Module:
        if self.dino is None:
            self.dino = self._load_dinov2(
                self.dino_model_name,
                backend=self.dino_backend,
                repo_path=self.dino_repo_path,
                checkpoint_path=self.dino_checkpoint_path,
            )
            self.dino.eval()
            for param in self.dino.parameters():
                param.requires_grad = False
        return self.dino.to(device=device)

    @staticmethod
    def _load_dinov2(
        model_name: str,
        *,
        backend: str = "transformers",
        repo_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> nn.Module:
        if backend == "torch_hub":
            if repo_path is None or checkpoint_path is None:
                raise ValueError(
                    "torch_hub DINO requires both dino_repo_path and dino_checkpoint_path"
                )
            repo = Path(repo_path).expanduser().resolve()
            checkpoint = Path(checkpoint_path).expanduser().resolve()
            if not (repo / "hubconf.py").is_file():
                raise FileNotFoundError(f"local DINO torch-hub repo not found: {repo}")
            if not checkpoint.is_file():
                raise FileNotFoundError(f"local DINO checkpoint not found: {checkpoint}")

            encoder = torch.hub.load(
                str(repo),
                model_name,
                source="local",
                pretrained=True,
                weights=str(checkpoint),
            )
            return _TorchHubDinoAdapter(encoder)

        if backend != "transformers":
            raise ValueError(f"unsupported DINO backend: {backend!r}")
        try:
            from transformers import Dinov2Model
        except ImportError as exc:
            raise ImportError(
                "ref_dino requires transformers with Dinov2Model support"
            ) from exc
        return Dinov2Model.from_pretrained(model_name)

    def _reduce_per_sample(
        self,
        per_sample: torch.Tensor,
        sample_valid: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor],
        graph_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce [B] losses with optional validity mask and absolute w(t) scaling.

        Important: do NOT normalize by sum(weights). With batch_size=1 (common here),
        ``sum(w*L)/sum(w)`` would cancel w(t) entirely and make low-noise weighting a no-op.
        Instead:
          loss = sum(w_i * L_i * valid_i) / sum(valid_i)
        so w(t) keeps absolute scale while empty/invalid samples are ignored.
        """
        if per_sample.ndim != 1:
            raise ValueError(f"per_sample must be [B], got {tuple(per_sample.shape)}")
        batch = per_sample.shape[0]
        valid = torch.ones(batch, device=per_sample.device, dtype=per_sample.dtype)
        if sample_valid is not None:
            valid = sample_valid.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
            if valid.numel() != batch:
                raise ValueError(f"sample_valid length {valid.numel()} must match batch {batch}")
        weights = torch.ones(batch, device=per_sample.device, dtype=per_sample.dtype)
        if sample_weights is not None:
            weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype).reshape(-1)
            if weights.numel() != batch:
                raise ValueError(f"sample_weights length {weights.numel()} must match batch {batch}")
            weights = weights.clamp_min(0.0)

        valid_count = valid.sum()
        if float(valid_count.item()) <= 0:
            return graph_anchor.sum() * 0.0
        # If every valid sample has w=0 (e.g. all high-noise gated), region term is 0.
        if float((valid * weights).sum().item()) <= 0:
            return graph_anchor.sum() * 0.0
        return (per_sample * weights * valid).sum() / valid_count

    def _foreground_crop_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        pred_range: str,
        target_range: str,
        sample_valid: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        lpips = self._ensure_lpips(pred.device)

        # Shared bbox from the edit mask so pred/target crops align geometrically.
        height, width = int(pred.shape[-2]), int(pred.shape[-1])
        mask_b = _ensure_bchw_mask(mask, height, width)
        boxes, box_valid = _mask_bboxes(
            mask_b,
            mask_threshold=self.mask_threshold,
            pad_ratio=self.lpips_bbox_pad_ratio,
        )
        if sample_valid is None:
            sample_valid = box_valid
        else:
            sample_valid = sample_valid.to(device=box_valid.device) & box_valid

        pred_lpips = _images_to_lpips_range(pred, input_range=pred_range)
        target_lpips = _images_to_lpips_range(target, input_range=target_range)
        pred_crop = _crop_resize_with_boxes(
            pred_lpips,
            boxes,
            out_size=self.lpips_input_size,
            mask=mask_b,
            zero_outside_mask=True,
        )
        target_crop = _crop_resize_with_boxes(
            target_lpips,
            boxes,
            out_size=self.lpips_input_size,
            mask=mask_b,
            zero_outside_mask=True,
        )

        lpips_param = next(lpips.parameters())
        dist = lpips(
            pred_crop.to(device=lpips_param.device, dtype=lpips_param.dtype),
            target_crop.to(device=lpips_param.device, dtype=lpips_param.dtype),
        )
        per_sample = dist.float().reshape(dist.shape[0], -1).mean(dim=1).to(device=pred.device)
        return self._reduce_per_sample(per_sample, sample_valid, sample_weights, pred)

    def _background_l1(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bg_mask: torch.Tensor,
        *,
        pred_range: str,
        target_range: str,
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pred_u = _images_to_unit_range(pred, input_range=pred_range)
        target_u = _images_to_unit_range(target, input_range=target_range)
        abs_err = (pred_u - target_u).abs() * bg_mask
        denom = bg_mask.expand_as(abs_err).reshape(abs_err.shape[0], -1).sum(dim=1)
        per_sample = abs_err.reshape(abs_err.shape[0], -1).sum(dim=1)
        valid = denom > 0
        per_sample = torch.where(
            valid,
            per_sample / denom.clamp_min(1.0),
            per_sample.new_zeros(per_sample.shape),
        )
        return self._reduce_per_sample(per_sample, valid, sample_weights, pred)

    def _dino_embed(self, images_01: torch.Tensor) -> torch.Tensor:
        """Return [B, D] features from [B,3,S,S] images in [0,1]."""
        if self._dino_embed_fn is not None:
            feats = self._dino_embed_fn(images_01)
            if feats.ndim != 2:
                raise ValueError(f"dino_embed_fn must return [B,D], got {tuple(feats.shape)}")
            return feats

        dino = self._ensure_dino(images_01.device)

        pixel_values = _imagenet_normalize(images_01)
        try:
            param = next(dino.parameters())
            pixel_values = pixel_values.to(device=param.device, dtype=param.dtype)
        except StopIteration:
            pass

        was_training = dino.training
        dino.eval()
        outputs = dino(pixel_values=pixel_values)
        if was_training:
            dino.train(was_training)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feats = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            feats = outputs.last_hidden_state[:, 0]
        elif torch.is_tensor(outputs):
            feats = outputs
        else:
            raise TypeError(f"Unsupported DINO output type: {type(outputs)!r}")
        if feats.ndim != 2:
            raise ValueError(f"DINO features must be [B,D], got {tuple(feats.shape)}")
        return feats.float()

    def _reference_dino_identity(
        self,
        pred_images: torch.Tensor,
        pred_mask: torch.Tensor,
        ref_images: torch.Tensor,
        ref_mask: torch.Tensor,
        *,
        pred_range: str,
        ref_range: str,
        sample_valid: Optional[torch.Tensor],
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pred_crop, pred_box_valid = _masked_bbox_crop(
            pred_images,
            pred_mask,
            out_size=self.dino_input_size,
            pad_ratio=self.dino_bbox_pad_ratio,
            mask_threshold=self.mask_threshold,
            input_range=pred_range,
            output_range="0-1",
        )
        with torch.no_grad():
            ref_crop, ref_box_valid = _masked_bbox_crop(
                ref_images,
                ref_mask,
                out_size=self.dino_input_size,
                pad_ratio=self.dino_bbox_pad_ratio,
                mask_threshold=self.mask_threshold,
                input_range=ref_range,
                output_range="0-1",
            )

        # Require both sides to have a valid object mask.
        both_valid = pred_box_valid & ref_box_valid.to(device=pred_box_valid.device)
        if sample_valid is None:
            sample_valid = both_valid
        else:
            sample_valid = sample_valid.to(device=both_valid.device) & both_valid

        pred_feat = self._dino_embed(pred_crop)
        with torch.no_grad():
            ref_feat = self._dino_embed(ref_crop).to(device=pred_feat.device, dtype=pred_feat.dtype)

        cos = F.cosine_similarity(pred_feat, ref_feat, dim=-1, eps=1e-6)
        per_sample = 1.0 - cos
        return self._reduce_per_sample(per_sample, sample_valid, sample_weights, pred_images)

    def forward(
        self,
        pred_images: torch.Tensor,
        target_images: torch.Tensor,
        pixel_mask: torch.Tensor,
        *,
        fg_weight: float = 1.0,
        bg_weight: float = 1.0,
        ref_dino_weight: float = 0.0,
        ref_images: Optional[torch.Tensor] = None,
        ref_mask: Optional[torch.Tensor] = None,
        pred_range: str = "-1-1",
        target_range: str = "0-1",
        ref_range: str = "0-1",
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if pred_images.shape != target_images.shape:
            raise ValueError(
                "pred/target image shapes must match, got "
                f"{tuple(pred_images.shape)} vs {tuple(target_images.shape)}"
            )
        if pred_images.ndim != 4 or pred_images.shape[1] != 3:
            raise ValueError(f"pred_images must be [B,3,H,W], got {tuple(pred_images.shape)}")

        height, width = int(pred_images.shape[-2]), int(pred_images.shape[-1])
        mask = _ensure_bchw_mask(pixel_mask, height, width).to(
            device=pred_images.device, dtype=torch.float32
        )
        if tuple(mask.shape[-2:]) != (height, width):
            raise ValueError(
                f"pixel_mask spatial size {tuple(mask.shape[-2:])} must match "
                f"pred_images {(height, width)}"
            )

        fg_mask = (mask >= self.mask_threshold).to(dtype=pred_images.dtype)
        bg_mask = 1.0 - fg_mask
        fg_area = fg_mask.reshape(fg_mask.shape[0], -1).sum(dim=1)
        bg_area = bg_mask.reshape(bg_mask.shape[0], -1).sum(dim=1)
        fg_valid = fg_area > 0
        bg_valid = bg_area > 0

        zero = pred_images.new_zeros(())
        fg_loss = zero
        bg_loss = zero
        ref_dino_loss = zero

        if float(fg_weight) != 0.0 and self.fg_metric == "lpips":
            if not bool(fg_valid.any().item()):
                fg_loss = pred_images.sum() * 0.0
            else:
                fg_loss = self._foreground_crop_lpips(
                    pred_images,
                    target_images,
                    mask,
                    pred_range=pred_range,
                    target_range=target_range,
                    sample_valid=fg_valid,
                    sample_weights=sample_weights,
                )

        if float(bg_weight) != 0.0:
            if not bool(bg_valid.any().item()):
                bg_loss = pred_images.sum() * 0.0
            else:
                bg_loss = self._background_l1(
                    pred_images,
                    target_images,
                    bg_mask,
                    pred_range=pred_range,
                    target_range=target_range,
                    sample_weights=sample_weights,
                )

        if float(ref_dino_weight) != 0.0:
            if not self.enable_ref_dino:
                raise RuntimeError(
                    "ref_dino_weight > 0 but RegionPerceptualLoss was constructed with "
                    "enable_ref_dino=false; set loss.ref_dino_weight > 0 before setup_criterion"
                )
            if ref_images is None or ref_mask is None:
                raise ValueError(
                    "ref_dino_weight > 0 requires ref_images and ref_mask "
                    "(InsertAnything path should pass embeddings['ref_image_for_loss'] / "
                    "embeddings['ref_mask_for_loss'])"
                )
            if ref_images.shape[0] != pred_images.shape[0]:
                raise ValueError(
                    "ref_images batch size must match pred_images, got "
                    f"{ref_images.shape[0]} vs {pred_images.shape[0]}"
                )
            if not bool(fg_valid.any().item()):
                ref_dino_loss = pred_images.sum() * 0.0
            else:
                ref_dino_loss = self._reference_dino_identity(
                    pred_images=pred_images,
                    pred_mask=mask,
                    ref_images=ref_images,
                    ref_mask=ref_mask,
                    pred_range=pred_range,
                    ref_range=ref_range,
                    sample_valid=fg_valid,
                    sample_weights=sample_weights,
                )

        total = (
            float(fg_weight) * fg_loss
            + float(bg_weight) * bg_loss
            + float(ref_dino_weight) * ref_dino_loss
        )
        return {
            "total": total,
            "fg_lpips": fg_loss.detach(),
            "bg_l1": bg_loss.detach(),
            "ref_dino": ref_dino_loss.detach(),
        }

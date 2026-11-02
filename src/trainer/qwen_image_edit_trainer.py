import math
import torch
import torch.nn.functional as F  # NOQA
import PIL
import numpy as np
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Optional, Union, List, Tuple
import logging
from tqdm.auto import tqdm
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    retrieve_latents,
    randn_tensor,
)

from src.loss.edit_mask_loss import map_mask_to_latent
from src.models.complementary_canvas import (
    canvas_plan_from_region,
    compose_canvas_latents,
    gather_tokens,
    ring_feather_weights,
    ring_sigma,
)
from src.trainer.base_trainer import BaseTrainer
from src.utils.tools import infer_image_tensor
from src.utils.latent_anchoring import latent_anchoring_enabled, normalize_latent_anchoring_mode
from src.utils.images import (
    calculate_best_resolution,
    make_image_devisible,
    make_image_shape_devisible,
    resize_mask_bhw,
)


class QwenImageEditTrainer(BaseTrainer):
    """Trainer class based on QwenImageEditPipeline"""

    def __init__(self, config):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the prompt encoder should be [B,C,1,H,W] in range [-1,1]
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        super().__init__(config)
        self.config = config

        # Component attributes
        self.vae = None  # AutoencoderKLQwenImage
        self.text_encoder = None  # Qwen2_5_VLForConditionalGeneration (text_encoder)
        self.dit = None  # QwenImageTransformer2DModel
        self.tokenizer = None  # Qwen2Tokenizer
        self.scheduler = None  # FlowMatchEulerDiscreteScheduler
        self._scheduler_train_timesteps = None
        self._scheduler_train_sigmas = None
        self._scheduler_train_tensor_cache = {}

        # Parameters obtained from VAE configuration
        self.vae_scale_factor = None
        self.vae_latent_mean = None
        self.vae_latent_std = None
        self.vae_z_dim = None
        self.adapter_name = config.model.lora.adapter_name

    def get_pipeline_class(self):
        return QwenImageEditPipeline

    # Static methods: directly reference QwenImageEditPipeline methods
    _pack_latents = staticmethod(QwenImageEditPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPipeline._unpack_latents)

    @staticmethod
    def _move_tensor(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        kwargs = {"non_blocking": tensor.device.type == "cpu"}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return tensor.to(**kwargs)

    @staticmethod
    def _module_device(module: torch.nn.Module, module_name: str) -> torch.device:
        devices = {tensor.device for tensor in module.parameters()}
        devices.update(tensor.device for tensor in module.buffers())
        if len(devices) == 0:
            raise ValueError(f"{module_name} has no parameters or buffers; its device cannot be determined")
        if len(devices) != 1:
            formatted_devices = ", ".join(sorted(str(device) for device in devices))
            raise RuntimeError(f"{module_name} spans multiple devices: {formatted_devices}")
        return next(iter(devices))

    @staticmethod
    def _module_floating_dtype(module: torch.nn.Module, module_name: str) -> torch.dtype:
        dtypes = {
            tensor.dtype
            for tensor in (*module.parameters(), *module.buffers())
            if tensor.is_floating_point() or tensor.is_complex()
        }
        if len(dtypes) == 0:
            raise ValueError(
                f"{module_name} has no floating-point parameters or buffers; "
                "its input dtype cannot be determined"
            )
        if len(dtypes) != 1:
            formatted_dtypes = ", ".join(sorted(str(dtype) for dtype in dtypes))
            raise RuntimeError(f"{module_name} spans multiple floating dtypes: {formatted_dtypes}")
        return next(iter(dtypes))

    def _vae_decode_device(self) -> torch.device:
        if self.vae is None:
            raise ValueError("VAE must be loaded before determining its decode device")
        if not hasattr(self.vae, "post_quant_conv"):
            raise ValueError("Loaded VAE has no post_quant_conv module")
        return self._module_device(self.vae.post_quant_conv, "vae.post_quant_conv")

    def _vae_decode_dtype(self) -> torch.dtype:
        if self.vae is None:
            raise ValueError("VAE must be loaded before determining its decode dtype")
        if not hasattr(self.vae, "post_quant_conv"):
            raise ValueError("Loaded VAE has no post_quant_conv module")
        return self._module_floating_dtype(self.vae.post_quant_conv, "vae.post_quant_conv")

    @contextmanager
    def _loaded_model_decode_context(self):
        """Temporarily place the offloaded VAE decoder on the VAE execution device."""
        if self.vae is None:
            raise ValueError("VAE must be loaded before preparing loaded-model decode")
        if not hasattr(self.vae, "decoder"):
            raise ValueError("Loaded VAE has no decoder module")

        target_device = self._vae_decode_device()
        original_device = self._module_device(self.vae.decoder, "vae.decoder")
        if original_device == target_device:
            yield
            return

        logging.info(
            "Moving vae.decoder from %s to %s for training-time validation decode",
            original_device,
            target_device,
        )
        operation_error = None
        try:
            self.vae.decoder.to(target_device)
            yield
        except BaseException as error:
            operation_error = error
        finally:
            try:
                self.vae.decoder.to(original_device)
                logging.info(
                    "Restored vae.decoder from %s to %s after training-time validation decode",
                    target_device,
                    original_device,
                )
            except BaseException as restore_error:
                if operation_error is not None:
                    raise BaseExceptionGroup(
                        "Training-time validation decode and vae.decoder restoration both failed",
                        [operation_error, restore_error],
                    )
                raise

        if operation_error is not None:
            raise operation_error

    @staticmethod
    def _combine_reference_cfg_predictions(
        noise_pred: torch.Tensor,
        reference_cfg_noise_pred: torch.Tensor,
        reference_cfg_scale: float,
        reference_cfg_mask: Optional[torch.Tensor] = None,
        delta_clip_enabled: bool = False,
        delta_clip_max_ratio: float = 0.5,
        apg_eta: float = 1.0,
        latents: Optional[torch.Tensor] = None,
        sigma: Optional[float] = None,
    ) -> torch.Tensor:
        """``null + scale * (cond - null)`` with optional delta clip, mask-aware blend and (without clip) norm
        rescale. ``apg_eta < 1`` keeps only that share of the delta's component parallel to the conditional x0
        prediction ``latents - sigma * cond`` (projected inside the mask); that path runs in float32."""

        if noise_pred.shape != reference_cfg_noise_pred.shape:
            raise ValueError(
                "reference_cfg noise prediction shape must match positive prediction shape, "
                f"got {tuple(reference_cfg_noise_pred.shape)} and {tuple(noise_pred.shape)}"
            )
        if delta_clip_max_ratio <= 0:
            raise ValueError("reference_cfg delta_clip_max_ratio must be positive")
        if not 0.0 <= apg_eta <= 1.0:
            raise ValueError("reference_cfg apg_eta must be in [0, 1]")
        output_dtype = noise_pred.dtype
        apg = apg_eta < 1.0
        if apg:
            if latents is None or sigma is None:
                raise ValueError("reference_cfg APG needs the current latents and sigma")
            noise_pred, reference_cfg_noise_pred = noise_pred.float(), reference_cfg_noise_pred.float()

        if reference_cfg_mask is not None:
            if reference_cfg_mask.ndim == 2:
                reference_cfg_mask = reference_cfg_mask.unsqueeze(-1)
            if reference_cfg_mask.ndim != 3 or reference_cfg_mask.shape[-1] != 1:
                raise ValueError(
                    "reference_cfg_mask must be [B, seq_len] or [B, seq_len, 1], "
                    f"got {tuple(reference_cfg_mask.shape)}"
                )
            if reference_cfg_mask.shape[:2] != noise_pred.shape[:2]:
                raise ValueError(
                    "reference_cfg_mask shape must match prediction batch and sequence dimensions, "
                    f"got {tuple(reference_cfg_mask.shape)} for prediction {tuple(noise_pred.shape)}"
                )
            reference_cfg_mask = reference_cfg_mask.to(device=noise_pred.device, dtype=noise_pred.dtype)

        reference_delta = noise_pred - reference_cfg_noise_pred
        cfg_delta = reference_delta
        if delta_clip_enabled:
            delta_norm = torch.norm(reference_delta.float(), dim=-1, keepdim=True)
            positive_norm = torch.norm(noise_pred.float(), dim=-1, keepdim=True)
            max_delta_norm = positive_norm * float(delta_clip_max_ratio)
            delta_scale = max_delta_norm / torch.clamp(delta_norm, min=1e-6)
            delta_scale = torch.clamp(delta_scale, max=1.0).to(dtype=reference_delta.dtype)
            cfg_delta = reference_delta * delta_scale

        guided_pred = reference_cfg_noise_pred + reference_cfg_scale * cfg_delta
        if apg:
            # x0 = x_t - sigma * v, so the delta moves x0 by -sigma * delta; its part along x0_c is k * x0_c,
            # which is -k * x0_c / sigma in velocity.
            x0_c = latents.float() - sigma * noise_pred
            weight = torch.ones_like(noise_pred[..., :1]) if reference_cfg_mask is None else reference_cfg_mask
            k = (-sigma * cfg_delta * x0_c * weight).sum(dim=(1, 2), keepdim=True) / (
                (x0_c * x0_c * weight).sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            )
            guided_pred = guided_pred - (reference_cfg_scale - 1) * (1 - apg_eta) * (-k * x0_c / sigma)
        if reference_cfg_mask is None:
            comb_pred = guided_pred
        else:
            comb_pred = noise_pred * (1.0 - reference_cfg_mask) + guided_pred * reference_cfg_mask

        if not delta_clip_enabled:
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_norm = torch.clamp(noise_norm, min=1e-6)
            comb_pred = comb_pred * (cond_norm / noise_norm)
        return comb_pred.to(output_dtype)

    def _prepare_reference_cfg_mask(
        self,
        reference_cfg_mask: torch.Tensor,
        batch_size: int,
        height_image: int,
        width_image: int,
        image_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if reference_cfg_mask.ndim == 4:
            if reference_cfg_mask.shape[1] not in {1, 3}:
                raise ValueError(
                    f"reference_cfg_mask expects 1 or 3 channels, got {reference_cfg_mask.shape[1]}"
                )
            reference_cfg_mask = reference_cfg_mask[:, 0]
        if reference_cfg_mask.ndim != 3:
            raise ValueError(
                "reference_cfg_mask must be [B, H, W] or [B, C, H, W], "
                f"got {tuple(reference_cfg_mask.shape)}"
            )
        if reference_cfg_mask.shape[0] != batch_size:
            raise ValueError(
                "reference_cfg_mask batch size must match prediction batch size, "
                f"got {reference_cfg_mask.shape[0]} and {batch_size}"
            )

        reference_cfg_mask = reference_cfg_mask.to(device=device, dtype=torch.float32)
        reference_cfg_mask = resize_mask_bhw(reference_cfg_mask, int(height_image), int(width_image))
        reference_cfg_mask = map_mask_to_latent(
            reference_cfg_mask,
            vae_scale_factor=self.vae_scale_factor,
            expected_seq_len=image_seq_len,
        )
        return (reference_cfg_mask > 0).to(device=device, dtype=self.weight_dtype).unsqueeze(-1)

    @staticmethod
    def _validate_mask_geometry_image_grid(
        mask_geometry: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        height_image: int,
        width_image: int,
    ) -> None:
        """Require geometry to use the generated target's pixel coordinate system."""

        if not isinstance(mask_geometry, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_shape = (batch_size, 3, int(height_image), int(width_image))
        if tuple(mask_geometry.shape) != expected_shape:
            raise ValueError(
                f"{name} must match target image grid {expected_shape}, "
                f"got {tuple(mask_geometry.shape)}"
            )

    @staticmethod
    def _mask_geometries_equal(
        positive: torch.Tensor,
        reference_null: torch.Tensor,
    ) -> bool:
        """Compare CFG geometry exactly, avoiding a copy on the normal aliased path."""

        if positive is reference_null:
            return True
        if positive.shape != reference_null.shape:
            return False
        if reference_null.device != positive.device:
            reference_null = reference_null.to(positive.device)
        return torch.equal(positive, reference_null)

    def _prepare_latent_anchoring_weight(
        self,
        edit_weight: torch.Tensor,
        free_mask: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height_image: int,
        width_image: int,
        soft_transition: bool,
    ) -> torch.Tensor:
        """Map image-space W_edit to the exact packed latent feature layout."""

        for name, value in (("edit_weight", edit_weight), ("free_mask", free_mask)):
            if value.ndim != 3:
                raise ValueError(f"latent anchoring {name} must be [B,H,W], got {tuple(value.shape)}")
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"latent anchoring {name} batch size must be {batch_size}, got {value.shape[0]}"
                )
            if not torch.isfinite(value).all().item():
                raise ValueError(f"latent anchoring {name} must contain only finite values")

        latent_height = 2 * (int(height_image) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width_image) // (self.vae_scale_factor * 2))
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError(
                "latent anchoring target geometry is too small for VAE compression and packing: "
                f"{(height_image, width_image)}"
            )

        edit = edit_weight.float().unsqueeze(1)
        free = free_mask.float().unsqueeze(1)
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

        packed = self._pack_latents(
            edit.repeat(1, num_channels_latents, 1, 1),
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        if torch.any(packed < 0).item() or torch.any(packed > 1).item():
            raise RuntimeError("packed latent anchoring weights must remain in [0, 1]")
        editable_per_sample = packed.flatten(1).amax(dim=1) > 0
        if not bool(editable_per_sample.all().item()):
            empty_indices = torch.nonzero(~editable_per_sample, as_tuple=False).flatten().tolist()
            raise ValueError(
                "packed latent anchoring weight must retain an editable region for every sample; "
                f"empty={empty_indices}"
            )
        return packed

    def _apply_post_step_latent_anchoring(
        self,
        generated_next: torch.Tensor,
        anchor_source_latents: torch.Tensor,
        anchor_noise: torch.Tensor,
        edit_weight: torch.Tensor,
        timesteps: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Apply the official Qwen inpaint source trajectory after scheduler.step()."""

        expected_shape = generated_next.shape
        for name, value in (
            ("anchor_source_latents", anchor_source_latents),
            ("anchor_noise", anchor_noise),
            ("edit_weight", edit_weight),
        ):
            if value.shape != expected_shape:
                raise ValueError(
                    f"latent anchoring {name} shape must match generated latents, "
                    f"got {tuple(value.shape)} vs {tuple(expected_shape)}"
                )
            if value.device != generated_next.device:
                raise ValueError(
                    f"latent anchoring {name} device must match generated latents, "
                    f"got {value.device} vs {generated_next.device}"
                )
            if value.dtype != generated_next.dtype:
                raise ValueError(
                    f"latent anchoring {name} dtype must match generated latents, "
                    f"got {value.dtype} vs {generated_next.dtype}"
                )
        if step_index < 0 or step_index >= len(timesteps):
            raise IndexError(
                f"latent anchoring step_index must be in [0, {len(timesteps)}), got {step_index}"
            )

        if step_index < len(timesteps) - 1:
            source_next = self.scheduler.scale_noise(
                anchor_source_latents,
                timesteps[step_index + 1 : step_index + 2],
                anchor_noise,
            )
        else:
            source_next = anchor_source_latents
        source_next = source_next.to(device=generated_next.device, dtype=generated_next.dtype)
        return torch.lerp(source_next, generated_next, edit_weight)

    def _complementary_canvas_cfg(self):
        canvas = getattr(getattr(self.config, "model", None), "complementary_canvas", None)
        return canvas if canvas is not None and canvas.enabled else None

    def _canvas_plan(self, embeddings: dict, img_shapes, device):
        """Edit-region (and ring) plan on the target grid; the masked source must share that grid."""

        target_shape, source_shape = tuple(img_shapes[0][0]), tuple(img_shapes[0][1])
        if target_shape != source_shape:
            raise ValueError(
                f"complementary canvas needs the masked source on the target grid, got {source_shape} vs {target_shape}"
            )
        region = embeddings["canvas_region"]
        if region.shape[-1] != math.prod(target_shape):
            raise ValueError(
                f"canvas region has {region.shape[-1]} tokens, the target grid {target_shape} has {math.prod(target_shape)}"
            )
        ring_tokens = self._complementary_canvas_cfg().ring_tokens
        return canvas_plan_from_region(region, device=device, grid=target_shape[1:], ring_tokens=ring_tokens)

    def _canvas_ring_weights(self, plan, img_shapes, latents: torch.Tensor):
        """Feather weights of the ring tokens in the packed latent layout, or None without a ring."""

        if not plan.ring_count:
            return None
        canvas = self._complementary_canvas_cfg()
        weights = ring_feather_weights(plan, tuple(img_shapes[0][0][1:]), canvas.ring_tokens, latents.shape[-1] // 4)
        return weights.to(device=latents.device, dtype=latents.dtype)

    def _canvas_dit_kwargs(self, canvas, plan, ring_timestep=None, isolation_bias=None) -> dict:
        """DiT forward arguments of the canvas: token selection, isolation, ring group, text condition."""

        kwargs = {"image_token_index": plan.image_token_index, "isolate_conditions": canvas.isolate_conditions}
        if canvas.isolate_conditions and canvas.isolation != "image":
            kwargs["isolation"] = canvas.isolation
        if plan.ring_count:
            kwargs.update(ring_token_count=plan.ring_count, ring_timestep=ring_timestep)
        if canvas.text_as_condition:
            kwargs["text_as_condition"] = True
        if isolation_bias is not None:
            kwargs["isolation_bias"] = isolation_bias
        return kwargs

    def _canvas_loss_crop(self, plan, img_shapes, pixel_mask: torch.Tensor, margin_px: int) -> SimpleNamespace:
        """Token-aligned crop for the pixel losses: bbox(R and its ring) plus ``margin_px``, grown to
        hold the GT box with the LPIPS/DINO bbox pads and half a detail patch, so every loss term
        except the far-background L1 sees exactly the pixels it sees on the full image."""

        loss = self.config.loss
        grid_h, grid_w = (int(value) for value in img_shapes[0][0][1:])
        token_px = self.vae_scale_factor * 2
        rows, cols = plan.target_index // grid_w, plan.target_index % grid_w
        y0, y1 = int(rows.min()) * token_px - margin_px, (int(rows.max()) + 1) * token_px + margin_px
        x0, x1 = int(cols.min()) * token_px - margin_px, (int(cols.max()) + 1) * token_px + margin_px
        threshold = float(getattr(loss, "mask_threshold", 0.5))
        ys, xs = torch.where((pixel_mask >= threshold).any(dim=0))
        if ys.numel():
            pad_ratio = max(float(getattr(loss, "lpips_bbox_pad_ratio", 0.1)), float(getattr(loss, "dino_bbox_pad_ratio", 0.1)))
            detail_half = int(getattr(loss, "detail_patch_size", 256)) // 2 if float(getattr(loss, "detail_lpips_weight", 0.0)) > 0 else 0
            gy0, gy1, gx0, gx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            pad_y = max(round((gy1 - gy0) * pad_ratio), detail_half) + 1
            pad_x = max(round((gx1 - gx0) * pad_ratio), detail_half) + 1
            y0, y1 = min(y0, gy0 - pad_y), max(y1, gy1 + pad_y)
            x0, x1 = min(x0, gx0 - pad_x), max(x1, gx1 + pad_x)
        ty0, ty1 = max(0, y0 // token_px), min(grid_h, -(-y1 // token_px))
        tx0, tx1 = max(0, x0 // token_px), min(grid_w, -(-x1 // token_px))
        device = plan.target_index.device
        token_index = (
            torch.arange(ty0, ty1, device=device)[:, None] * grid_w + torch.arange(tx0, tx1, device=device)[None, :]
        ).flatten()
        rows_px, cols_px = slice(ty0 * token_px, ty1 * token_px), slice(tx0 * token_px, tx1 * token_px)
        from src.loss.local_detail import interaction_masks

        foreground = (pixel_mask >= threshold).float().unsqueeze(1)
        background = interaction_masks(foreground, int(getattr(loss, "interaction_radius_px", 0)))[1]
        total = float(background.sum())
        return SimpleNamespace(
            token_index=token_index,
            rows=rows_px,
            cols=cols_px,
            height=(ty1 - ty0) * token_px,
            width=(tx1 - tx0) * token_px,
            background_fraction=float(background[..., rows_px, cols_px].sum()) / total if total > 0 else 1.0,
        )

    def _canvas_isolation_bias(self, canvas) -> Optional[float]:
        """Annealed isolation: the logit bias ramps 0 -> anneal_max_bias, then hard isolation (None)."""

        step = int(getattr(self, "global_step", 0))
        if not canvas.isolate_conditions or step >= canvas.anneal_steps:
            return None
        return float(canvas.anneal_max_bias) * step / canvas.anneal_steps

    @staticmethod
    def _complementary_canvas_conditions(control_latents: torch.Tensor, plan) -> torch.Tensor:
        """Condition stream: masked-source tokens outside R, then the remaining condition images."""

        return torch.cat(
            [gather_tokens(control_latents[:, : plan.num_tokens], plan.source_index), control_latents[:, plan.num_tokens :]],
            dim=1,
        )

    def _unwrap_dit(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.dit)
        return self.dit

    def _dit_device(self) -> torch.device:
        return next(self.dit.parameters()).device

    def _dit_config(self):
        return self._unwrap_dit().config

    def _dit_cache_context(self, name: str):
        return self._unwrap_dit().cache_context(name)

    def _cache_scheduler_training_tensors(self):
        if self.scheduler is None:
            raise ValueError("scheduler must be loaded before caching training timesteps")
        if not hasattr(self.scheduler, "timesteps") or not hasattr(self.scheduler, "sigmas"):
            raise ValueError("scheduler must expose timesteps and sigmas for training")
        self._scheduler_train_timesteps = self.scheduler.timesteps.detach().clone()
        self._scheduler_train_sigmas = self.scheduler.sigmas.detach().clone()
        self._scheduler_train_tensor_cache = {}

    def _get_cached_scheduler_training_tensors(self, device: torch.device, dtype: torch.dtype):
        if self._scheduler_train_timesteps is None or self._scheduler_train_sigmas is None:
            self._cache_scheduler_training_tensors()

        key = (device.type, device.index, dtype)
        cached = self._scheduler_train_tensor_cache.get(key)
        if cached is None:
            cached = (
                self._scheduler_train_timesteps.to(device=device),
                self._scheduler_train_sigmas.to(device=device, dtype=dtype),
            )
            self._scheduler_train_tensor_cache[key] = cached
        return cached

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPipeline"""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        # Load complete model using pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
            text_encoder=None,
        )

        # Separate individual components

        from src.models.load_model import load_vae, load_qwenvl

        self.vae = load_vae("Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype)  # use original one
        # same to model constructed from vae self.vae = pipe.vae
        self.text_encoder = load_qwenvl(
            "Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype,
            attn_implementation=self.config.model.attention_backend,
        )
        logging.info(f"text_encoder device: {self.text_encoder.device}")
        # self.dit = pipe.transformer this is same as the following, verified
        from src.models.load_model import load_transformer

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,  # could use quantized version
            weight_dtype=self.weight_dtype,
            attention_backend=self.config.model.attention_backend,
            zero_cond_t=self.config.model.zero_cond_t,
        )
        # load_transformer is same as pipe.transformer

        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self._cache_scheduler_training_tensors()
        # Initialize image processor (for predict method)
        from diffusers.image_processor import VaeImageProcessor

        # Set VAE-related parameters
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim

        # Attributes copied from original pipeline
        self.latent_channels = self.vae.config.z_dim
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        # Set models to training/evaluation mode
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def preprocess_image_for_vae_encoder(self, image, *, input_range="auto"):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the vae encoder should be [B,C,1,H,W] in range [-1,1]
        """
        # suppose image input is [0,1] [B,C,H,W] after dataloader -> [B,C,1,H,W]
        tensor_info = infer_image_tensor(image, input_range=input_range)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "BCHW":
            image = image.unsqueeze(2)
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2).unsqueeze(2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']},{image.shape}")


        if tensor_info["range"] == "0-255":
            image = image / 255.0
            image = 2 * image - 1
        elif tensor_info["range"] == "0-1":
            image = 2 * image - 1
        elif tensor_info["range"] == "-1-1":
            pass
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}, {image.min()}, {image.max()}")

        return image

    def preprocess_image_for_text_encoder(
        self,
        image,
        best_resolution=None,
        *,
        input_range="auto",
    ) -> torch.Tensor:
        """
        the image process passed to text encoder
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        tensor_info = infer_image_tensor(image, input_range=input_range)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif tensor_info["layout"] == "BCHW":
            pass
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']}")
        if tensor_info["range"] == "0-255":
            pass
        elif tensor_info["range"] == "0-1":
            image = image * 255
        elif tensor_info["range"] == "-1-1":
            image = (image + 1) / 2 * 255
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}")
        if best_resolution is not None:
            new_width, new_height = calculate_best_resolution(image.shape[3], image.shape[2], best_resolution)
            image = F.interpolate(image, size=(new_height, new_width), mode="bilinear")
        return image

    def encode_image_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        dtype,
        generator=None,
    ):
        if image is None:
            raise ValueError("image is required for VAE latent encoding")
        if isinstance(generator, list) and len(generator) != image.shape[0]:
            raise ValueError(
                f"Generator list length {len(generator)} must match image batch size {image.shape[0]}."
            )

        device = self.vae.device
        image = self._move_tensor(image, device=device, dtype=dtype)
        if image.shape[1] != self.latent_channels:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        else:
            image_latents = image

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )

        image_latent_height, image_latent_width = image_latents.shape[3:]
        return self._pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )

    def prepare_initial_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator=None,
        latents=None,
        device=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        if device is None:
            device = next(self.dit.parameters()).device
        shape = (batch_size, 1, num_channels_latents, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = self._move_tensor(latents, device=device, dtype=dtype)
        return latents

    def setup_model_device_train_mode(self, stage="fit"):
        """Set model device allocation and train mode."""
        if stage not in ["fit", "predict"]:
            raise ValueError(f"stage must be one of ['fit', 'predict'], but got {stage}")
        if stage == "fit":
            if not hasattr(self, "accelerator"):
                raise ValueError("accelerator must be set before setting model devices")

        if stage == "fit":
            # Training mode: need all encoders
            self.resolve_memory_policy()
            self.vae.to(self.accelerator.device)
            offload = getattr(getattr(self.config, "train", None), "offload_text_encoder", False)
            encoder_device = "cpu" if offload else self.accelerator.device
            self.text_encoder.to(encoder_device)
            # self.dit.to(self.accelerator.device)
            # Region-perceptual loss decodes pred_x0 through the VAE and backprops
            # through the decoder into DiT. Keeping decoder on CPU and only moving it
            # for the forward (then restoring) breaks autograd. When region loss is on,
            # leave the decoder on the training device for the whole fit stage.
            if getattr(self.config.loss, "region_perceptual", False):
                self.vae.decoder.to(self.accelerator.device)
                logging.info(
                    "region_perceptual enabled: keeping vae.decoder on %s for training decode+backward",
                    self.accelerator.device,
                )
            else:
                self.vae.decoder.to("cpu")

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False)
            self.dit.train()
            # OPD mounts frozen Teacher LoRAs on this same backbone. Activate
            # and train only the exact Student adapter instead of every
            # parameter whose name happens to contain "lora".
            from src.distillation.backends import set_active_adapter

            set_active_adapter(
                self._unwrap_dit(),
                self.adapter_name,
                trainable_adapter=self.adapter_name,
            )

        elif stage == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.dit.to(devices.dit)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()

    def prepare_embeddings(self, batch, stage="fit"):
        """
        used in: training and predict
        Qwen-Edit
            - image_latent : target used for training
            - control_latent: concatenated version
            - height_image, width_image
            - height_control, width_control: no batch dim
            - height_control_1, width_control_1: no batch dim
            ...
        for control image, need to create two copies:
            1. processed for vae encoder
            2. processed for text encoder
        for additional control images, only pass to vae encoder
        predict mode:

            - no `image` key
            - extra: negative_prompt_embeds, negative_prompt_embeds_mask

        """

        if "image" in batch:

            batch["image"] = self.preprocess_image_for_vae_encoder(
                batch["image"], input_range="0-1"
            )  # B,3,1,H,W
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        if "control" not in batch:
            raise ValueError("prepare_embeddings requires main control image under key 'control'")

        prompt_control = self.preprocess_image_for_text_encoder(
            batch["control"], input_range="0-1"
        )  # [B,C,H,W], float, source range [0,1]
        batch["prompt_control"] = prompt_control
        batch["control"] = self.preprocess_image_for_vae_encoder(
            batch["control"], input_range="0-1"
        )  # [B,C,1,H,W], float, range [-1,1]
        batch["width_control"] = batch["control"].shape[4]
        batch["height_control"] = batch["control"].shape[3]

        if isinstance(batch["n_controls"], int):
            num_additional_controls = batch["n_controls"]
        else:
            num_additional_controls = batch["n_controls"][0]

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i+1}"
            if additional_control_key not in batch:
                raise ValueError(f"n_controls={num_additional_controls} requires missing key '{additional_control_key}'")
            batch[additional_control_key] = self.preprocess_image_for_vae_encoder(
                batch[additional_control_key], input_range="0-1"
            )
            batch[f"width_control_{i+1}"] = batch[additional_control_key].shape[4]
            batch[f"height_control_{i+1}"] = batch[additional_control_key].shape[3]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=batch["prompt"],
            image=batch["prompt_control"],
        )
        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if "negative_prompt" in batch and batch['true_cfg_scale'] > 1:
            # only for predict stage
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=batch["negative_prompt"],
                image=batch["prompt_control"],
            )
            batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            batch["negative_prompt_embeds"] = negative_prompt_embeds

        # get latents
        if "image" in batch:
            image = batch["image"]
            batch_size = image.shape[0]
            image_latents = self.encode_image_latents(
                image,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        if "control" in batch:
            control = batch["control"]
            batch_size = control.shape[0]
            control_latents = self.encode_image_latents(
                control,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["control_latents"] = [control_latents]

        for i in range(1, num_additional_controls + 1):
            control_key = f"control_{i}"
            control = batch[control_key]
            batch_size = control.shape[0]
            control_latents = self.encode_image_latents(
                control,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            batch["control_latents"].append(control_latents)

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)

        if self.config.loss.mask_loss and "mask" in batch:
            mask = batch["mask"]
            height_image, width_image = batch["height"], batch["width"]
            pixel_mask = resize_mask_bhw(mask, height_image, width_image)
            # Keep pixel-space mask for region-perceptual terms before packing to latent.
            if getattr(self.config.loss, "region_perceptual", False):
                batch["pixel_mask"] = pixel_mask
            batch["mask"] = map_mask_to_latent(
                pixel_mask,
                vae_scale_factor=self.vae_scale_factor,
                expected_seq_len=batch["image_latents"].shape[1],
            )

        # Generate img_shapes for training
        if "image_latents" in batch:
            batch_size = batch["image_latents"].shape[0]
            batch["img_shapes"] = self._get_image_shapes(batch, batch_size)

        return batch

    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Post-process output images"""
        image = image_tensor.cpu().float()
        image = image.squeeze(2).squeeze(0)  # [C,H,W]
        image = (image / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
        image = image.permute(1, 2, 0).numpy()  # [H,W,C]
        image = (image * 255).astype(np.uint8)
        return image

    def _get_image_shapes(self, embeddings: dict, batch_size: int) -> List[List[Tuple[int, int, int]]]:
        """
        embeddings comes from prepare_embeddings.
        The image shapes is made of, the first one is the target image shape, second is control, and additional controls
        ```python
        img_shapes = [
                [
                    (1, height_image // self.vae_scale_factor // 2, width_image // self.vae_scale_factor // 2),
                    (1, height_control // self.vae_scale_factor // 2, width_control // self.vae_scale_factor // 2),
                    (1, height_control_1 // self.vae_scale_factor // 2, width_control_1 // self.vae_scale_factor // 2),
                ]
            ]
        ```
        hiehgt_image, width_image, height_control, width_control, height_control_1, width_control_1
        """
        if "height" not in embeddings or "width" not in embeddings:
            raise ValueError("height_image and width_image must be in embeddings")
        img_shapes = []
        height_image = embeddings["height"]
        width_image = embeddings["width"]
        img_shapes.append((1, height_image // self.vae_scale_factor // 2, width_image // self.vae_scale_factor // 2))
        if "height_control" in embeddings and "width_control" in embeddings:
            height_control = embeddings["height_control"]
            width_control = embeddings["width_control"]
            img_shapes.append(
                (1, height_control // self.vae_scale_factor // 2, width_control // self.vae_scale_factor // 2)
            )
            num_additional_controls = (
                embeddings["n_controls"] if isinstance(embeddings["n_controls"], int) else embeddings["n_controls"][0]
            )
            for i in range(num_additional_controls):
                additional_control_key = f"control_{i+1}"
                if additional_control_key in embeddings:
                    height_control_i = embeddings[f"height_control_{i+1}"]
                    width_control_i = embeddings[f"width_control_{i+1}"]
                    img_shapes.append(
                        (
                            1,
                            height_control_i // self.vae_scale_factor // 2,
                            width_control_i // self.vae_scale_factor // 2,
                        )
                    )
        return [img_shapes] * batch_size

    def _timestep_indices_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """Map uniform samples to training-schedule indices (shared by σ-split routing)."""

        max_timestep_index = self.scheduler.config.num_train_timesteps - 1
        return (u * self.scheduler.config.num_train_timesteps).long().clamp(min=0, max=max_timestep_index)

    def _auxiliary_losses(self, embeddings: dict, dit_outputs, device) -> Tuple[Optional[torch.Tensor], dict]:
        """Alpha-head occupancy loss (every σ) and σ-split route statistics, when enabled."""

        metrics = {}
        route_high = embeddings.get("_sigma_split_route_high")
        if route_high is not None:
            metrics["train/sigma_split_high_rate"] = route_high.float().mean().to(device)
        alpha_target_mask = embeddings.get("alpha_target_mask")
        if alpha_target_mask is None:
            return None, metrics

        from src.models.alpha_head import alpha_head_loss, alpha_occupancy_targets

        mask = resize_mask_bhw(
            self._move_tensor(alpha_target_mask, device=device, dtype=torch.float32),
            int(embeddings["height"]),
            int(embeddings["width"]),
        )
        alpha_loss, bce, dice = alpha_head_loss(
            dit_outputs[1],
            alpha_occupancy_targets(mask, self.vae_scale_factor * 2),
            bce_weight=float(self.config.loss.alpha_bce_weight),
            dice_weight=float(self.config.loss.alpha_dice_weight),
        )
        metrics.update({"train/alpha_bce": bce.detach(), "train/alpha_dice": dice.detach()})
        return alpha_loss, metrics

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        device = self.accelerator.device
        image_latents = self._move_tensor(embeddings["image_latents"], device=device, dtype=self.weight_dtype)
        control_latents = self._move_tensor(embeddings["control_latents"], device=device, dtype=self.weight_dtype)
        prompt_embeds = self._move_tensor(embeddings["prompt_embeds"], device=device, dtype=self.weight_dtype)
        prompt_embeds_mask = self._move_tensor(embeddings["prompt_embeds_mask"], device=device, dtype=torch.int64)
        mask_geometry = embeddings.get("mask_geometry")
        if mask_geometry is not None:
            mask_geometry = self._move_tensor(mask_geometry, device=device, dtype=self.weight_dtype)
        img_shapes = embeddings["img_shapes"]
        batch_size = image_latents.shape[0]
        if "mask" in embeddings:
            edit_mask = self._move_tensor(embeddings["mask"], device=device, dtype=torch.float32)
        else:
            edit_mask = None

        with torch.no_grad():
            # Analysis callers may provide stateless randomness through private
            # embedding keys.  Ordinary training never sets them and therefore
            # preserves the original RNG path exactly.  The override lets
            # category-gradient diagnostics match noise seeds and timesteps
            # across semantically different samples without coupling them to
            # variable-shape VAE/text-encoder RNG consumption.
            analysis_noise_seed = embeddings.get("_analysis_noise_seed")
            if analysis_noise_seed is None:
                noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)
            else:
                noise_generator = torch.Generator(device=device)
                noise_generator.manual_seed(int(analysis_noise_seed))
                noise = torch.randn(
                    image_latents.shape,
                    generator=noise_generator,
                    device=device,
                    dtype=self.weight_dtype,
                )

            # Sample timesteps
            analysis_timestep_u = embeddings.get("_analysis_timestep_u")
            if analysis_timestep_u is None:
                # σ-split training samples σ before it builds the route's condition.
                analysis_timestep_u = embeddings.get("_sigma_split_timestep_u")
            if analysis_timestep_u is None:
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=batch_size,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
            else:
                u = torch.as_tensor(analysis_timestep_u, dtype=torch.float32).reshape(-1)
                if u.numel() == 1:
                    u = u.expand(batch_size)
                if u.numel() != batch_size:
                    raise ValueError(
                        "_analysis_timestep_u must be scalar or have one value per sample, "
                        f"got {u.numel()} values for batch_size={batch_size}"
                    )
                if not bool(torch.isfinite(u).all()) or bool(torch.any((u < 0) | (u >= 1))):
                    raise ValueError("_analysis_timestep_u values must be finite and in [0, 1)")
            indices = self._timestep_indices_from_u(u)
            schedule_timesteps, schedule_sigmas = self._get_cached_scheduler_training_tensors(device, image_latents.dtype)
            indices = indices.to(device=device)
            timesteps = schedule_timesteps[indices]
            sigmas = schedule_sigmas[indices].reshape(batch_size, *((1,) * (image_latents.ndim - 1)))
            noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise
            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
            output_token_count = image_latents.size(1)
            canvas = self._complementary_canvas_cfg()
            canvas_plan = None
            canvas_kwargs = {}
            if canvas is not None:
                # Only R (and its ring) is noised and supervised; the rest of the source is clean context.
                canvas_plan = self._canvas_plan(embeddings, img_shapes, device)
                generated_input = gather_tokens(noisy_model_input, canvas_plan.target_index)
                token_sigmas = sigmas.expand(batch_size, generated_input.shape[1], 1)
                ring_timestep = None
                if canvas_plan.ring_count:
                    # The ring starts from the noised GT, which equals the source outside the hole, with σ capped.
                    core = canvas_plan.core_count
                    ring_sigmas = ring_sigma(sigmas, canvas.ring_sigma_max)
                    ring_input = (1.0 - ring_sigmas) * gather_tokens(
                        image_latents, canvas_plan.ring_index
                    ) + ring_sigmas * gather_tokens(noise, canvas_plan.ring_index)
                    generated_input = torch.cat([generated_input[:, :core], ring_input], dim=1)
                    token_sigmas = torch.cat(
                        [token_sigmas[:, :core], ring_sigmas.expand(batch_size, canvas_plan.ring_count, 1)], dim=1
                    )
                    ring_timestep = torch.minimum(timesteps, timesteps.new_tensor(1000.0 * canvas.ring_sigma_max)) / 1000
                packed_input = torch.cat(
                    [generated_input, self._complementary_canvas_conditions(control_latents, canvas_plan)], dim=1
                )
                output_token_count = int(canvas_plan.target_index.numel())
                canvas_kwargs = self._canvas_dit_kwargs(
                    canvas, canvas_plan, ring_timestep, self._canvas_isolation_bias(canvas)
                )

        # The alpha head runs inside the (DDP-wrapped) forward so its gradients
        # take part in the normal reducer bookkeeping.
        alpha_kwargs = {"return_alpha_logits": True} if "alpha_target_mask" in embeddings else {}
        dit_outputs = self.dit(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            mask_geometry=mask_geometry,
            output_token_count=output_token_count,
            return_dict=False,
            **alpha_kwargs,
            **canvas_kwargs,
        )
        model_pred = dit_outputs[0]
        model_pred = model_pred[:, :output_token_count]
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - image_latents
        core_pred = model_pred
        if canvas_plan is not None:
            # Region-balanced over R: GT object tokens vs the rest of the edit region.
            target = gather_tokens(target, canvas_plan.target_index)
            if edit_mask is not None:
                edit_mask = gather_tokens(edit_mask, canvas_plan.core_index)
            if canvas_plan.ring_count:
                core = canvas_plan.core_count
                ring_mse = F.mse_loss(model_pred[:, core:].float(), target[:, core:].float())
                core_pred, target = model_pred[:, :core], target[:, :core]
        # pred shape [2, 4104, 64], target shape [2, 4104, 64]
        latent_mse = self.forward_loss(core_pred, target, weighting, edit_mask)
        auxiliary_loss, auxiliary_metrics = self._auxiliary_losses(embeddings, dit_outputs, device)
        if canvas_plan is not None and canvas.ring_tokens:
            # The ring's flow-matching loss joins the auxiliary terms with its own small weight. Every rank logs
            # the metric, also without ring tokens (R covering the grid): DDP reduces all ranks' keys together.
            ring_metric = latent_mse.detach().float().new_zeros(())
            if canvas_plan.ring_count:
                ring_loss = float(canvas.ring_loss_weight) * ring_mse
                auxiliary_loss = ring_loss if auxiliary_loss is None else auxiliary_loss + ring_loss
                ring_metric = ring_mse.detach()
            auxiliary_metrics["train/ring_mse"] = ring_metric

        use_region = (
            getattr(self.config.loss, "region_perceptual", False)
            and getattr(self, "region_perceptual_criterion", None) is not None
        )
        if not use_region:
            # Only total loss is used for progress bar; no region TB metrics.
            self._step_loss_metrics = {
                "train/latent_mse": latent_mse.detach().float().reshape(()),
                **auxiliary_metrics,
            }
            return latent_mse if auxiliary_loss is None else latent_mse + auxiliary_loss

        height = int(embeddings["height"])
        width = int(embeddings["width"])
        if "pixel_mask" not in embeddings:
            raise ValueError(
                "region_perceptual requires embeddings['pixel_mask']; "
                "enable loss.mask_loss and ensure tar_mask is present"
            )

        # Validate key-only contracts before the timestep gate so malformed
        # region-loss batches fail deterministically instead of only when a
        # later low-noise sample happens to activate the pixel-space path.
        ref_dino_weight = float(getattr(self.config.loss, "ref_dino_weight", 0.0))
        face_dino_weight = float(getattr(self.config.loss, "face_dino_weight", 0.0))
        face_valid_fraction = latent_mse.detach().float().new_zeros(())
        if "target_face_mask" in embeddings and "reference_face_mask" in embeddings:
            face_valid_fraction = (
                embeddings["target_face_mask"].flatten(1).gt(0).any(1)
                & embeddings["reference_face_mask"].flatten(1).gt(0).any(1)
            ).float().mean().to(device=device)
        if not embeddings.get("reference_present_for_loss", True):
            ref_dino_weight = 0.0
            face_dino_weight = 0.0
        if face_dino_weight > 0 and any(k not in embeddings for k in ("target_face_mask", "reference_face_mask")):
            raise ValueError("face identity loss requires prepared target/reference face masks")
        if (ref_dino_weight > 0.0 or face_dino_weight > 0.0) and (
            "ref_image_for_loss" not in embeddings or "ref_mask_for_loss" not in embeddings
        ):
            raise ValueError(
                "ref_dino_weight > 0 requires embeddings['ref_image_for_loss'] and "
                "embeddings['ref_mask_for_loss'] (InsertAnything prepare_embeddings "
                "attaches these when region_perceptual=true)"
            )

        # Compute w(t) before building the full-resolution VAE decoder graph.  The
        # max-sigma gate produces exact zeros, so an entirely gated local batch has
        # no region-loss contribution and can safely skip all pixel-space work.
        # Mixed batches deliberately continue through the unchanged criterion path.
        from src.loss.region_perceptual_loss import region_timestep_weights

        sample_weights = region_timestep_weights(
            sigmas,
            power=float(self.config.loss.region_sigma_power),
            max_sigma=float(self.config.loss.region_max_sigma),
        ).to(device=device)
        latent_w = float(self.config.loss.latent_mse_weight)
        if torch.count_nonzero(sample_weights).item() == 0:
            zero_metric = latent_mse.detach().float().new_zeros(())
            total = latent_w * latent_mse
            if auxiliary_loss is not None:
                total = total + auxiliary_loss
            self._step_loss_metrics = {
                "train/latent_mse": latent_mse.detach().float().reshape(()),
                "train/latent_mse_weighted": total.detach().float().reshape(()),
                "train/region_total": zero_metric,
                "train/region_fg_lpips": zero_metric.clone(),
                "train/region_bg_l1": zero_metric.clone(),
                "train/region_ref_dino": zero_metric.clone(),
                "train/region_detail_lpips": zero_metric.clone(),
                "train/region_boundary_l1": zero_metric.clone(),
                "train/region_face_dino": zero_metric.clone(),
                "train/face_pair_valid_fraction": face_valid_fraction,
                "train/region_w_t_mean": sample_weights.detach().float().mean().reshape(()),
                "train/region_w_t_max": sample_weights.detach().float().max().reshape(()),
                "train/sigma_mean": sigmas.detach().float().reshape(sigmas.shape[0], -1)[:, 0].mean().reshape(()),
                **auxiliary_metrics,
            }
            return total

        # Flow-matching clean estimate: x0 = x_t - sigma * v_pred, with v_pred ~ noise - x0.
        if canvas_plan is None:
            pred_x0 = noisy_model_input - sigmas * model_pred
        else:
            # Pixel losses see what inference decodes: the predicted R and the ring feathered into the GT latent.
            pred_x0 = compose_canvas_latents(
                image_latents,
                canvas_plan,
                generated_input - token_sigmas * model_pred,
                self._canvas_ring_weights(canvas_plan, img_shapes, image_latents),
            )
        pixel_mask = self._move_tensor(embeddings["pixel_mask"], device=device, dtype=torch.float32)
        bg_weight = float(self.config.loss.bg_l1_weight)
        crop = None
        if canvas_plan is not None and canvas.local_decode_margin_px is not None:
            # Decode only the crop holding R, its ring and every pixel-loss support; outside it the
            # latent is the exact GT. The background L1 keeps its full-image normalization.
            crop = self._canvas_loss_crop(canvas_plan, img_shapes, pixel_mask, int(canvas.local_decode_margin_px))
            bg_weight *= crop.background_fraction
            pred_x0 = gather_tokens(pred_x0, crop.token_index)
            pixel_mask = pixel_mask[..., crop.rows, crop.cols]
            height, width = crop.height, crop.width

        # Only decode the prediction. GT uses the original pixel tar_image when available
        # (cheaper and free of VAE reconstruction error). Fallback: decode image_latents.
        with self._loaded_model_decode_context():
            pred_images = self.decode_vae_latent_for_loss(pred_x0, height, width)

        # Keep pred_images in the autograd graph (no detach).
        pred_images = pred_images.to(device=device)
        if pred_images.shape[-2:] != (height, width):
            raise ValueError(
                f"decoded pred spatial size {tuple(pred_images.shape[-2:])} must match "
                f"embeddings height/width {(height, width)}"
            )
        if pixel_mask.shape[-2:] != (height, width):
            raise ValueError(
                f"pixel_mask spatial size {tuple(pixel_mask.shape[-2:])} must match "
                f"embeddings height/width {(height, width)}"
            )

        if "target_image_for_loss" in embeddings:
            target_images = self._move_tensor(
                embeddings["target_image_for_loss"], device=device, dtype=torch.float32
            )
            target_range = "0-1"
            full_size = (int(embeddings["height"]), int(embeddings["width"]))
            # Align geometry if dataset resolution differs slightly from VAE decode size.
            if target_images.shape[-2:] != full_size:
                target_images = F.interpolate(
                    target_images,
                    size=full_size,
                    mode="bilinear",
                    align_corners=False,
                )
            if crop is not None:
                target_images = target_images[..., crop.rows, crop.cols]
        else:
            gt_latents = image_latents if crop is None else gather_tokens(image_latents, crop.token_index)
            with self._loaded_model_decode_context():
                with torch.no_grad():
                    target_images = self.decode_vae_latent_for_loss(gt_latents, height, width)
            target_images = target_images.to(device=device)
            target_range = "-1-1"

        ref_images = None
        ref_mask = None
        if ref_dino_weight > 0.0 or face_dino_weight > 0.0:
            ref_images = self._move_tensor(
                embeddings["ref_image_for_loss"], device=device, dtype=torch.float32
            )
            ref_mask = self._move_tensor(
                embeddings["ref_mask_for_loss"], device=device, dtype=torch.float32
            )

        # σ-split: the reference identity term belongs to the low-noise route only.
        identity_kwargs = {}
        route_high = embeddings.get("_sigma_split_route_high")
        if route_high is not None and ref_dino_weight > 0.0:
            low_route = (~route_high).to(device=device, dtype=sample_weights.dtype)
            if not bool(low_route.any()):
                ref_dino_weight = 0.0
            identity_kwargs["ref_dino_sample_weights"] = sample_weights * low_route
        if getattr(self.config.loss, "ref_dino_mode", "cosine") == "gt_hinge":
            identity_kwargs["ref_dino_mode"] = "gt_hinge"
            identity_kwargs["ref_dino_hinge_margin"] = float(self.config.loss.ref_dino_hinge_margin)

        # Explicit value-range contracts:
        # - VAE decode pred path returns [-1,1]
        # - InsertAnything tar/ref images are [0,1]
        region_out = self.region_perceptual_criterion(
            pred_images,
            target_images,
            pixel_mask,
            fg_weight=float(self.config.loss.fg_lpips_weight),
            bg_weight=bg_weight,
            ref_dino_weight=ref_dino_weight,
            ref_images=ref_images,
            ref_mask=ref_mask,
            pred_range="-1-1",
            target_range=target_range,
            ref_range="0-1",
            sample_weights=sample_weights,
            detail_weight=float(getattr(self.config.loss, "detail_lpips_weight", 0.0)),
            detail_num_patches=int(getattr(self.config.loss, "detail_num_patches", 4)),
            detail_patch_size=int(getattr(self.config.loss, "detail_patch_size", 256)),
            boundary_weight=float(getattr(self.config.loss, "boundary_l1_weight", 0.0)),
            interaction_radius_px=int(getattr(self.config.loss, "interaction_radius_px", 0)),
            **identity_kwargs,
        )
        region_total = region_out["total"]
        face_loss = region_total.new_zeros(())
        if face_dino_weight > 0:
            target_face_mask = self._move_tensor(embeddings["target_face_mask"], device=device, dtype=torch.float32)
            if crop is not None:
                target_face_mask = target_face_mask[..., crop.rows, crop.cols]
            face_loss = self.region_perceptual_criterion._reference_dino_identity(
                pred_images, target_face_mask,
                ref_images, self._move_tensor(embeddings["reference_face_mask"], device=device, dtype=torch.float32),
                pred_range="-1-1", ref_range="0-1", sample_valid=None, sample_weights=sample_weights,
            )
            region_total = region_total + face_dino_weight * face_loss
        total = latent_w * latent_mse + region_total
        if auxiliary_loss is not None:
            total = total + auxiliary_loss

        # TensorBoard metrics only (no text spam). Progress bar still shows total loss only.
        # fg_lpips / bg_l1 / ref_dino already include w(t) from sample_weights.
        self._step_loss_metrics = {
            "train/latent_mse": latent_mse.detach().float().reshape(()),
            "train/latent_mse_weighted": (latent_w * latent_mse).detach().float().reshape(()),
            "train/region_total": region_total.detach().float().reshape(()),
            "train/region_fg_lpips": region_out["fg_lpips"].detach().float().reshape(()),
            "train/region_bg_l1": region_out["bg_l1"].detach().float().reshape(()),
            "train/region_ref_dino": region_out["ref_dino"].detach().float().reshape(()),
            "train/region_detail_lpips": region_out.get("detail_lpips", region_total.new_zeros(())).detach().float().reshape(()),
            "train/region_boundary_l1": region_out.get("boundary_l1", region_total.new_zeros(())).detach().float().reshape(()),
            "train/region_face_dino": face_loss.detach().float().reshape(()),
            "train/face_pair_valid_fraction": face_valid_fraction,
            "train/region_w_t_mean": sample_weights.detach().float().mean().reshape(()),
            "train/region_w_t_max": sample_weights.detach().float().max().reshape(()),
            "train/sigma_mean": sigmas.detach().float().reshape(sigmas.shape[0], -1)[:, 0].mean().reshape(()),
            **auxiliary_metrics,
        }
        return total

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Calculate sigma values for noise scheduler"""
        schedule_timesteps, sigmas = self._get_cached_scheduler_training_tensors(self.accelerator.device, dtype)
        timesteps = self._move_tensor(timesteps, device=self.accelerator.device)
        if sigmas.numel() < schedule_timesteps.numel():
            raise ValueError(
                f"scheduler sigmas must cover all timesteps: {sigmas.numel()} vs {schedule_timesteps.numel()}"
            )
        if schedule_timesteps[0] <= schedule_timesteps[-1]:
            step_indices = torch.searchsorted(schedule_timesteps, timesteps)
        else:
            reversed_timesteps = torch.flip(schedule_timesteps, dims=[0])
            reverse_indices = torch.searchsorted(reversed_timesteps, timesteps)
            step_indices = schedule_timesteps.numel() - 1 - reverse_indices

        if torch.any(step_indices < 0) or torch.any(step_indices >= schedule_timesteps.numel()):
            raise ValueError("timesteps contain values outside the cached scheduler range")
        matched_timesteps = schedule_timesteps[step_indices]
        if not torch.equal(matched_timesteps, timesteps):
            raise ValueError("timesteps must exactly match cached scheduler timesteps")
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
        # generator is None by default
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i: i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
    ):
        r"""
        get the embedding of prompt and image via qwen_vl. Support batch inference. For batch inference,
        will pad to largest length of prompt in batch.
        It got grad by default, lets add the inference mode first

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        num_images_per_prompt = 1  # 固定为1，支持单图像生成
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(image, torch.Tensor):
            # make sure it is identical with diffuser pipelines
            # when remove this, got relative error 0.49% in prompt embeds
            new_images = []
            for i in range(image.shape[0]):
                img = image[i].permute(1, 2, 0).cpu().numpy().astype('uint8')
                img = PIL.Image.fromarray(img)
                new_images.append(img)
            image = new_images

        with self.conditioning_context(), torch.inference_mode():
            device = self.text_encoder.device
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        encoder = getattr(self.text_encoder, "model", None)
        with torch.cuda.device(model_inputs.input_ids.device):  # 作用域内的 'cuda' 都指向同一张卡
            if encoder is not None:
                outputs = encoder(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=False,
                    return_dict=True,
                )
                hidden_states = outputs.last_hidden_state
            else:
                outputs = self.text_encoder(
                    input_ids=model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    pixel_values=model_inputs.pixel_values,
                    image_grid_thw=model_inputs.image_grid_thw,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = outputs.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def prepare_predict_batch_data(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        additional_controls: Optional[List[PIL.Image.Image]] = None,
        controls_size: Optional[List[int]] = None,  # first one is for main control
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        image_latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        Processing logic:
            if height, width is None: choose the width from prompt_image
        if controls_size is None, use same processing as that in the processor in the config
        if use_native_size = True:
            the prompt image and additional controls will use their own size. That is they will not be resized.
        """
        if prompt_image is None:
            raise ValueError("prompt_image is required")
        if prompt is None:
            raise ValueError("prompt is required")
        if additional_controls is None:
            additional_controls = []

        if isinstance(prompt_image, PIL.Image.Image):
            prompt_image = [prompt_image]
        prompt_image = [make_image_devisible(image, self.vae_scale_factor) for image in prompt_image]

        if isinstance(prompt, str):
            prompt = [prompt]

        self.weight_dtype = weight_dtype

        data = {}
        control = []
        if use_native_size:
            controls_size = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size.extend([[control.size[1], control.size[0]] for control in additional_controls[0]])
        elif best_resolution_size and controls_size is None:
            controls_size = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size.extend([[control.size[1], control.size[0]] for control in additional_controls[0]])

        if best_resolution_size:
            controls_size = [calculate_best_resolution(c_size[0], c_size[1], 1024*1024) for c_size in controls_size]
            logging.info(f'controls_size after best resolution  {controls_size}')

        logging.info(f'controls_size for processing {controls_size}')
        for img in prompt_image:
            # for each image, need to make one copy for text_encoder, another for image_encoder
            # convert to [C,H,W] in range [0,1]
            img = self.preprocessor.preprocess({"control": img}, controls_size=controls_size)["control"]
            control.append(img)
        control = torch.stack(control, dim=0)
        data["control"] = control
        data["prompt"] = prompt
        data["height"] = height
        data["width"] = width

        if height is None or width is None:
            height, width = control.shape[2], control.shape[3]
        else:
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        logging.info(f'target shape for generation {width}, {height}')

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls = {f"control_{i+1}": [] for i in range(n_controls)}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]

            for controls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": controls}, controls_size=controls_size)["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i+1}"].append(control)
            for i in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{i+1}"], dim=0)
                data[f"control_{i+1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data['n_controls'] = 0

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            if len(negative_prompt) != len(data["prompt"]):
                raise ValueError("the number of negative_prompt should be same of control")
            data["negative_prompt"] = negative_prompt
        data["num_inference_steps"] = num_inference_steps
        data["true_cfg_scale"] = true_cfg_scale
        data["guidance"] = guidance_scale
        return data

    def sampling_from_embeddings(
        self,
        embeddings: dict,
        *,
        initial_latents: Optional[torch.Tensor] = None,
        anchor_noise: Optional[torch.Tensor] = None,
        step_start: int = 0,
        step_end: Optional[int] = None,
        capture_alpha: bool = False,
    ) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.

        The keyword arguments run one segment ``[step_start, step_end)`` of the
        schedule (σ-split hand-off): ``initial_latents`` are the latents at
        ``step_start`` and ``anchor_noise`` is the shared initial noise of the
        latent-anchoring source trajectory. ``capture_alpha`` keeps the last
        conditional alpha-head logits in ``self._last_alpha_logits``.
        """
        if getattr(getattr(self.config.model, "flow_map", None), "enabled", False):
            from src.distillation.meanflow import sample_flow_map
            return sample_flow_map(self, embeddings)
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        control_latents = embeddings["control_latents"]
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        mask_geometry = embeddings.get("mask_geometry")
        batch_size = embeddings["control_latents"].shape[0]
        img_shapes = self._get_image_shapes(embeddings, batch_size)
        height_image = embeddings["height"]
        width_image = embeddings["width"]

        anchor_source_latents = embeddings.get("anchor_source_latents")
        anchor_edit_weight = embeddings.get("anchor_edit_weight")
        latent_anchoring_mode = normalize_latent_anchoring_mode(
            embeddings.get("latent_anchoring_mode", "legacy")
        )
        do_latent_anchoring = anchor_source_latents is not None or anchor_edit_weight is not None
        if (anchor_source_latents is None) != (anchor_edit_weight is None):
            raise ValueError(
                "latent anchoring requires both anchor_source_latents and anchor_edit_weight"
            )
        if do_latent_anchoring != latent_anchoring_enabled(latent_anchoring_mode):
            raise ValueError(
                "latent anchoring mode/tensor mismatch: "
                f"mode={latent_anchoring_mode}, tensors_present={do_latent_anchoring}"
            )

        negative_prompt = embeddings["negative_prompt"] if "negative_prompt" in embeddings else None

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        reference_cfg_embeddings = embeddings.get("reference_cfg_embeddings")
        reference_cfg_scale = float(embeddings.get("reference_cfg_scale", 1.0))
        reference_cfg_mask_aware = bool(embeddings.get("reference_cfg_mask_aware", False))
        reference_cfg_mask = embeddings.get("reference_cfg_mask")
        reference_cfg_delta_clip_enabled = bool(embeddings.get("reference_cfg_delta_clip_enabled", False))
        reference_cfg_delta_clip_max_ratio = float(embeddings.get("reference_cfg_delta_clip_max_ratio", 0.5))
        reference_cfg_sigma_min = float(embeddings.get("reference_cfg_sigma_min", 0.0))
        reference_cfg_apg_eta = float(embeddings.get("reference_cfg_apg_eta", 1.0))
        do_reference_cfg = reference_cfg_embeddings is not None and reference_cfg_scale > 1
        if do_true_cfg and do_reference_cfg:
            raise ValueError("text true CFG and reference_cfg cannot be enabled at the same time")
        if do_reference_cfg and reference_cfg_mask_aware and reference_cfg_mask is None:
            raise ValueError("reference_cfg.mask_aware requires reference_cfg_mask in embeddings")
        logging.info(
            f"do_true_cfg={do_true_cfg}, has_negative_prompt={has_neg_prompt}, true_cfg_scale={true_cfg_scale}"
        )
        logging.info(f"do_reference_cfg={do_reference_cfg}, reference_cfg_scale={reference_cfg_scale}")
        logging.info(f"reference_cfg_mask_aware={reference_cfg_mask_aware}")
        logging.info(
            "latent_anchoring_mode=%s, enabled=%s",
            latent_anchoring_mode,
            do_latent_anchoring,
        )
        logging.info(
            "reference_cfg_delta_clip_enabled="
            f"{reference_cfg_delta_clip_enabled}, "
            f"reference_cfg_delta_clip_max_ratio={reference_cfg_delta_clip_max_ratio}, "
            f"reference_cfg_sigma_min={reference_cfg_sigma_min}, reference_cfg_apg_eta={reference_cfg_apg_eta}"
        )
        device = self._dit_device()
        dit_config = self._dit_config()

        control_latents = control_latents.to(device=device, dtype=self.weight_dtype)
        if mask_geometry is not None:
            self._validate_mask_geometry_image_grid(
                mask_geometry,
                name="positive mask_geometry",
                batch_size=batch_size,
                height_image=height_image,
                width_image=width_image,
            )
            mask_geometry = mask_geometry.to(device=device, dtype=self.weight_dtype)

        if do_true_cfg:
            # 清理显存以确保有足够空间进行 CFG
            torch.cuda.empty_cache()
            logging.info(f"negative_prompt: {negative_prompt}")

            # 临时将 positive prompt embeddings 移到 CPU 以节省显存
            negative_prompt_embeds = embeddings["negative_prompt_embeds"]
            negative_prompt_embeds_mask = embeddings["negative_prompt_embeds_mask"]
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)

        if do_reference_cfg:
            reference_cfg_control_latents = reference_cfg_embeddings["control_latents"].to(
                self.weight_dtype
            ).to(device)
            reference_cfg_prompt_embeds = reference_cfg_embeddings["prompt_embeds"].to(
                device, dtype=self.weight_dtype
            )
            reference_cfg_prompt_embeds_mask = reference_cfg_embeddings["prompt_embeds_mask"].to(
                device, dtype=torch.int64
            )
            raw_reference_cfg_mask_geometry = reference_cfg_embeddings.get("mask_geometry")
            if (mask_geometry is None) != (raw_reference_cfg_mask_geometry is None):
                raise ValueError(
                    "reference_cfg positive and null-reference branches must use the same mask geometry contract"
                )
            reference_height = int(reference_cfg_embeddings["height"])
            reference_width = int(reference_cfg_embeddings["width"])
            if (reference_height, reference_width) != (int(height_image), int(width_image)):
                raise ValueError(
                    "reference_cfg positive and null-reference target image grids must match, "
                    f"got {(height_image, width_image)} and {(reference_height, reference_width)}"
                )
            if raw_reference_cfg_mask_geometry is not None:
                self._validate_mask_geometry_image_grid(
                    raw_reference_cfg_mask_geometry,
                    name="reference_cfg null mask_geometry",
                    batch_size=batch_size,
                    height_image=reference_height,
                    width_image=reference_width,
                )
                if not self._mask_geometries_equal(
                    embeddings["mask_geometry"],
                    raw_reference_cfg_mask_geometry,
                ):
                    raise ValueError(
                        "reference_cfg positive and null-reference mask geometry must be identical"
                    )
            if reference_cfg_control_latents.shape[0] != batch_size:
                raise ValueError("reference_cfg batch size must match the positive branch batch size")
            reference_cfg_img_shapes = self._get_image_shapes(reference_cfg_embeddings, batch_size)
            if reference_cfg_img_shapes[0][0] != img_shapes[0][0]:
                raise ValueError(
                    "reference_cfg positive and null-reference target token grids must match, "
                    f"got {img_shapes[0][0]} and {reference_cfg_img_shapes[0][0]}"
                )
            reference_cfg_txt_seq_lens = reference_cfg_prompt_embeds_mask.sum(dim=1).tolist()

        mask_geometry_tokens = None
        if mask_geometry is not None:
            with torch.inference_mode():
                mask_geometry_tokens = self._unwrap_dit().prepare_mask_geometry_tokens(
                    mask_geometry,
                    img_shapes,
                )
            if do_reference_cfg:
                reference_cfg_mask_geometry_tokens = mask_geometry_tokens
        elif do_reference_cfg:
            reference_cfg_mask_geometry_tokens = None

        logging.info(f"mask shape: {prompt_embeds_mask.shape}, dtype: {prompt_embeds_mask.dtype}")
        logging.info(f"prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

        # 5. Prepare latent variables
        num_channels_latents = dit_config.in_channels // 4

        latents = self.prepare_initial_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height_image,
            width=width_image,
            device=device,
            dtype=self.weight_dtype,
            latents=initial_latents,
        )
        image_seq_len = latents.shape[1]

        if step_start > 0 and do_latent_anchoring and anchor_noise is None:
            raise ValueError("a later sampling segment with latent anchoring needs the initial anchor_noise")
        segment_anchor_noise = anchor_noise
        anchor_noise = None
        if do_latent_anchoring:
            anchor_source_latents = anchor_source_latents.to(
                device=device,
                dtype=latents.dtype,
            )
            anchor_edit_weight = anchor_edit_weight.to(
                device=device,
                dtype=latents.dtype,
            )
            if anchor_source_latents.shape != latents.shape:
                raise ValueError(
                    "anchor source latent shape must match generated target latent shape, "
                    f"got {tuple(anchor_source_latents.shape)} vs {tuple(latents.shape)}"
                )
            if anchor_edit_weight.shape != latents.shape:
                raise ValueError(
                    "packed W_edit shape must match generated target latent shape, "
                    f"got {tuple(anchor_edit_weight.shape)} vs {tuple(latents.shape)}"
                )
            if torch.any(anchor_edit_weight < 0).item() or torch.any(anchor_edit_weight > 1).item():
                raise ValueError("packed W_edit must remain in [0, 1]")
            if segment_anchor_noise is None:
                anchor_noise = latents.detach().clone()
            else:
                anchor_noise = segment_anchor_noise.to(device=device, dtype=latents.dtype)

        logging.info(f"image latent got grad: {control_latents.requires_grad}")
        logging.info(f"latents shape: {latents.shape}")
        logging.info(f"num_channels_latents: {num_channels_latents}")
        logging.info(f"image-latent shape: {control_latents.shape}")

        logging.info(f"shape of img_shapes: {img_shapes}")
        logging.info(f"self.vae_scale_factor: {self.vae_scale_factor}")
        logging.info(f"height: {height_image}")
        logging.info(f"width: {width_image}")
        logging.info(f"image_seq_len: {image_seq_len}")

        reference_cfg_latent_mask = None
        if do_reference_cfg and reference_cfg_mask_aware:
            reference_cfg_latent_mask = self._prepare_reference_cfg_mask(
                reference_cfg_mask=reference_cfg_mask,
                batch_size=batch_size,
                height_image=height_image,
                width_image=width_image,
                image_seq_len=image_seq_len,
                device=device,
            )
            logging.info(f"reference_cfg latent mask shape: {reference_cfg_latent_mask.shape}")

        canvas = self._complementary_canvas_cfg()
        canvas_plan = None
        canvas_kwargs = reference_cfg_canvas_kwargs = {}
        if canvas is not None:
            if initial_latents is not None or step_start or step_end is not None or capture_alpha:
                raise ValueError("complementary canvas sampling runs the whole schedule from noise")
            if do_latent_anchoring and canvas.ring_tokens:
                raise ValueError("latent anchoring inside R does not combine with a canvas ring")
            if canvas.text_as_condition and do_true_cfg:
                raise ValueError("a text condition is cached per prompt; text CFG with a negative prompt is not supported")
            canvas_plan = self._canvas_plan(embeddings, img_shapes, device)
            # The full-grid draw keeps a dense run's noise at R for the same seed; only R (and its ring) is denoised.
            latents = gather_tokens(latents, canvas_plan.target_index)
            if do_latent_anchoring:
                # R's latent pixels outside M_free (and its soft transition) follow the source trajectory, as in a
                # dense anchored run; the rest of the grid is the exact source anyway.
                anchor_source_latents, anchor_edit_weight, anchor_noise = (
                    gather_tokens(value, canvas_plan.target_index)
                    for value in (anchor_source_latents, anchor_edit_weight, anchor_noise)
                )
            canvas_source_latents = embeddings["canvas_source_latents"].to(device=device, dtype=self.weight_dtype)
            control_latents = self._complementary_canvas_conditions(control_latents, canvas_plan)
            canvas_kwargs = self._canvas_dit_kwargs(canvas, canvas_plan)
            reference_cfg_canvas_kwargs = canvas_kwargs
            if do_reference_cfg:
                reference_cfg_control_latents = self._complementary_canvas_conditions(
                    reference_cfg_control_latents, canvas_plan
                )
                if reference_cfg_latent_mask is not None:
                    reference_cfg_latent_mask = gather_tokens(reference_cfg_latent_mask, canvas_plan.target_index)
            condition_cache_bytes = 0
            if canvas.cache_conditions:
                # Isolated t=0 conditions (and a text condition) are step-invariant: run them once,
                # then only the target (and a non-condition text) per step.
                dit = self._unwrap_dit()

                def build_cache(conditions, shapes, prompt, prompt_mask):
                    text = {}
                    if canvas.text_as_condition:
                        text = {
                            "encoder_hidden_states": prompt.to(device, dtype=self.weight_dtype),
                            "txt_seq_lens": prompt_mask.sum(dim=1).tolist(),
                        }
                    return dit.build_condition_cache(
                        conditions,
                        shapes,
                        canvas_plan.image_token_index,
                        isolation=canvas.isolation,
                        target_groups=2 if canvas_plan.ring_count else 1,
                        **text,
                    )

                with torch.inference_mode():
                    condition_cache = build_cache(control_latents, img_shapes, prompt_embeds, prompt_embeds_mask)
                    canvas_kwargs = {**canvas_kwargs, "condition_cache": condition_cache}
                    condition_cache_bytes = condition_cache.nbytes
                    if do_reference_cfg:
                        source_count = int(canvas_plan.source_index.numel())
                        shares_source = (
                            condition_cache.per_image
                            and condition_cache.text_layers is None
                            and reference_cfg_control_latents.shape[1] == source_count
                            and torch.equal(reference_cfg_control_latents, control_latents[:, :source_count])
                        )
                        # Per-image isolated source tokens never see the reference, so both CFG branches
                        # share them; a condition group or a text condition needs its own null pass.
                        reference_cfg_cache = (
                            condition_cache.select([0])
                            if shares_source
                            else build_cache(
                                reference_cfg_control_latents,
                                reference_cfg_img_shapes,
                                reference_cfg_prompt_embeds,
                                reference_cfg_prompt_embeds_mask,
                            )
                        )
                        if not shares_source:
                            condition_cache_bytes += reference_cfg_cache.nbytes
                        reference_cfg_canvas_kwargs = {**reference_cfg_canvas_kwargs, "condition_cache": reference_cfg_cache}
                        reference_cfg_control_latents = None
                control_latents = None
            self._last_complementary_canvas = {
                "edit_tokens": int(canvas_plan.core_count),
                "ring_tokens": int(canvas_plan.ring_count),
                "grid_tokens": int(canvas_plan.num_tokens),
                "text_tokens": int(prompt_embeds.shape[1]),
                "text_tokens_null": int(reference_cfg_prompt_embeds.shape[1]) if do_reference_cfg else None,
                "text_cached": bool(canvas.cache_conditions and canvas.text_as_condition),
                "condition_cache_bytes": int(condition_cache_bytes),
            }

        timesteps, num_inference_steps = self.prepare_predict_timesteps(num_inference_steps, image_seq_len)
        ring_sigmas = ring_weights = None
        if canvas_plan is not None and canvas_plan.ring_count:
            # The ring starts from the source noised at σ_ring = min(σ, ring_sigma_max) with its own
            # noise draw; it moves only once σ falls below the cap.
            ring_sigmas = ring_sigma(self.scheduler.sigmas.detach().float(), canvas.ring_sigma_max).tolist()
            core = canvas_plan.core_count
            ring_source = gather_tokens(canvas_source_latents, canvas_plan.ring_index)
            latents = torch.cat(
                [latents[:, :core], torch.lerp(ring_source.to(latents.dtype), latents[:, core:], ring_sigmas[0])], dim=1
            )
            ring_weights = self._canvas_ring_weights(canvas_plan, img_shapes, latents)

        if do_latent_anchoring:
            initial_sigma = self.scheduler.sigmas[0].detach().float()
            if not torch.isclose(
                initial_sigma,
                torch.ones_like(initial_sigma),
                rtol=0.0,
                atol=1e-6,
            ).item():
                raise ValueError(
                    "latent anchoring currently requires full-noise initialization with first scheduler sigma=1; "
                    f"got sigma={float(initial_sigma.cpu())}"
                )

        self._num_timesteps = len(timesteps)

        # 处理guidance
        guidance_scale = embeddings["guidance"]
        if dit_config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if do_true_cfg and negative_prompt_embeds_mask is not None
            else None
        )

        # 7. 降噪循环 (遵循原始pipeline逻辑)
        self.scheduler.set_begin_index(step_start)
        self.attention_kwargs = {}
        alpha_kwargs = {"return_alpha_logits": True} if capture_alpha else {}

        # set to proper device
        prompt_embeds = prompt_embeds.to(device, dtype=self.weight_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device, dtype=torch.int64)

        if do_true_cfg:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)
            negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.weight_dtype)

        logging.info(f"timesteps: {timesteps}")
        logging.info(f"num_inference_steps: {num_inference_steps}")
        with torch.inference_mode():
            # progress_bar = tqdm(enumerate(timesteps), total=num_inference_steps, desc="Generating")
            # for i, t in progress_bar:
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Generating"):
                # progress_bar.set_postfix({'timestep': f'{t:.1f}'})
                # progress_bar.update()
                if i < step_start:
                    continue
                if step_end is not None and i >= step_end:
                    break

                self._current_timestep = t
                latents = latents.to(device, dtype=self.weight_dtype)

                latent_model_input = latents
                if control_latents is not None:
                    latent_model_input = torch.cat([latents, control_latents], dim=1)

                # broadcast to batch dimension
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                if ring_sigmas is not None:
                    ring_timestep = torch.full_like(timestep, ring_sigmas[i])
                    canvas_kwargs = {**canvas_kwargs, "ring_timestep": ring_timestep}
                    reference_cfg_canvas_kwargs = {**reference_cfg_canvas_kwargs, "ring_timestep": ring_timestep}

                # Usecache_context (如果transformer支持)
                with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                    with self._dit_cache_context("cond"):
                        dit_outputs = self.dit(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            mask_geometry_tokens=mask_geometry_tokens,
                            attention_kwargs=self.attention_kwargs,
                            output_token_count=latents.size(1),
                            return_dict=False,
                            **alpha_kwargs,
                            **canvas_kwargs,
                        )
                        noise_pred = dit_outputs[0]
                        if capture_alpha:
                            self._last_alpha_logits = dit_outputs[1]
                        noise_pred = noise_pred[:, : latents.size(1)]
                if do_reference_cfg and float(self.scheduler.sigmas[i]) >= reference_cfg_sigma_min:
                    reference_cfg_latent_model_input = latents
                    if reference_cfg_control_latents is not None:
                        reference_cfg_latent_model_input = torch.cat(
                            [latents, reference_cfg_control_latents], dim=1
                        )

                    with torch.inference_mode():
                        with self._dit_cache_context("reference_uncond"):
                            reference_cfg_noise_pred = self.dit(
                                hidden_states=reference_cfg_latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=reference_cfg_prompt_embeds_mask,
                                encoder_hidden_states=reference_cfg_prompt_embeds,
                                img_shapes=reference_cfg_img_shapes,
                                txt_seq_lens=reference_cfg_txt_seq_lens,
                                mask_geometry_tokens=reference_cfg_mask_geometry_tokens,
                                attention_kwargs=self.attention_kwargs,
                                output_token_count=latents.size(1),
                                return_dict=False,
                                **reference_cfg_canvas_kwargs,
                            )[0]
                            reference_cfg_noise_pred = reference_cfg_noise_pred[:, : latents.size(1)]

                    noise_pred = self._combine_reference_cfg_predictions(
                        noise_pred=noise_pred,
                        reference_cfg_noise_pred=reference_cfg_noise_pred,
                        reference_cfg_scale=reference_cfg_scale,
                        reference_cfg_mask=reference_cfg_latent_mask,
                        delta_clip_enabled=reference_cfg_delta_clip_enabled,
                        delta_clip_max_ratio=reference_cfg_delta_clip_max_ratio,
                        apg_eta=reference_cfg_apg_eta,
                        latents=latents,
                        sigma=float(self.scheduler.sigmas[i]),
                    )

                    del reference_cfg_noise_pred

                if do_true_cfg:
                    # 临时释放正面推理结果的显存，避免两次推理同时占用显存
                    noise_pred_cpu = noise_pred.cpu()
                    torch.cuda.empty_cache()
                    with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                        with self._dit_cache_context("uncond"):
                            neg_noise_pred = self.dit(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                txt_seq_lens=negative_txt_seq_lens,
                                mask_geometry_tokens=mask_geometry_tokens,
                                attention_kwargs=self.attention_kwargs,
                                output_token_count=latents.size(1),
                                return_dict=False,
                                **canvas_kwargs,
                            )[0]
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    # 将正面推理结果移回 GPU 进行合并
                    noise_pred = noise_pred_cpu.to(device)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_norm = torch.clamp(noise_norm, min=1e-6)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                    # 释放中间结果显存
                    del neg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                if ring_sigmas is None:
                    generated_next = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                else:
                    # Euler on the ring's own σ schedule: no motion while σ is above the cap.
                    core = canvas_plan.core_count
                    core_next = self.scheduler.step(noise_pred[:, :core], t, latents[:, :core], return_dict=False)[0]
                    ring_next = latents[:, core:].float() + (ring_sigmas[i + 1] - ring_sigmas[i]) * noise_pred[:, core:].float()
                    generated_next = torch.cat([core_next.to(latents_dtype), ring_next.to(latents_dtype)], dim=1)
                # Flow scheduler arithmetic can promote BF16 latents to FP32.
                # The anchoring tensors are prepared in model dtype, so restore
                # the latent stream dtype before mixing the source trajectory.
                if generated_next.dtype != latents_dtype:
                    generated_next = generated_next.to(latents_dtype)
                if do_latent_anchoring:
                    latents = self._apply_post_step_latent_anchoring(
                        generated_next=generated_next,
                        anchor_source_latents=anchor_source_latents,
                        anchor_noise=anchor_noise,
                        edit_weight=anchor_edit_weight,
                        timesteps=timesteps,
                        step_index=i,
                    )
                else:
                    latents = generated_next

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)
        self._current_timestep = None
        if canvas_plan is not None:
            # The source latent stays exact outside R and its ring by construction; the ring is feathered.
            latents = compose_canvas_latents(canvas_source_latents, canvas_plan, latents, ring_weights)
        return latents

    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 8. decode final latents
        latents = latents.to(self._vae_decode_device(), dtype=self.weight_dtype)
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        final_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        # 后处理
        final_image = self.image_processor.postprocess(final_image, output_type="pt")
        return final_image

    def decode_vae_latent_for_loss(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Decode packed latents to [B,3,H,W] in [-1,1] while keeping the autograd graph.

        Unlike ``decode_vae_latent`` (inference path), this skips image_processor.postprocess
        so gradients from pixel losses can flow back into ``latents`` / the DiT prediction.
        VAE parameters remain frozen; only the input graph is used.
        """
        decode_device = self._vae_decode_device()
        decode_dtype = self._vae_decode_dtype()
        # Flow-matching arithmetic can promote pred_x0 to float32 even when the
        # frozen VAE is bf16. Match the actual first decode module instead of a
        # configured precision value. Tensor.to(dtype=...) remains differentiable,
        # so region-loss gradients still propagate back to pred_x0 / the DiT.
        latents = latents.to(device=decode_device, dtype=decode_dtype)
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=latents.device, dtype=latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=latents.device, dtype=latents.dtype)
        )
        latents = latents / latents_std + latents_mean
        # VAE weights are frozen but the forward still propagates grads to inputs.
        if getattr(getattr(getattr(self, "config", None), "train", None), "checkpoint_vae_decode", False) and latents.requires_grad:
            from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

            # Qwen VAE clears its temporal caches at decode entry/exit. Complete
            # recomputation so checkpoint early-stop cannot leave partial caches.
            with set_checkpoint_early_stop(False):
                images = checkpoint(
                    lambda z: self.vae.decode(z, return_dict=False)[0][:, :, 0],
                    latents, use_reentrant=False,
                )
        else:
            images = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        return images

"""
Trainer for the Insert Anything mask_prompt task using QwenImageEdit.
"""

import logging
import os
import random
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

import PIL.Image
import safetensors.torch
import torch
import torch.nn.functional as F
from diffusers.training_utils import compute_density_for_timestep_sampling

from src.models.alpha_head import ALPHA_HEAD_FILE_NAME, AlphaHead
from src.models.complementary_canvas import token_region_from_mask
from src.models.mask_geometry import MaskGeometryTokenizer, build_mask_geometry
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.utils.images import extract_foreground_from_image_and_mask
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_DILATED_SEGMENTATION,
    INSERT_ANYTHING_EXPANDED_BOX,
    LOOSE_BLOB,
    LOOSE_OVERSIZED_BOX,
    MASKMIX_LOOSE_MODES,
    MASKMIX_V2_ASYMMETRIC_BOX,
    MASKMIX_V2_EXACT,
    MASKMIX_V2_RANDOM_MORPHOLOGY,
    apply_hybrid_masking_context_guard,
    add_interaction_context,
    build_insert_anything_hybrid_masks,
)
from src.utils.latent_anchoring import (
    build_latent_anchoring_masks,
    ensure_latent_anchoring_config,
    latent_anchoring_enabled,
    normalize_latent_anchoring_mode,
)
from src.utils.sigma_split import (
    alpha_region_from_logits,
    augment_reference_light,
    augment_reference_strong,
    handoff_step_index,
    route_high_noise,
    shuffle_token_segment,
)


MASK_GEOMETRY_TOKENIZER_FILE_NAME = "mask_geometry_tokenizer.safetensors"


class QwenImageEditInsertAnythingTrainer(QwenImageEditPlusTrainer):
    """
    Insert Anything trainer with a single mask_prompt data contract.
    """

    @staticmethod
    def _empty_prompts(prompt, batch_size: int) -> List[str]:
        if isinstance(prompt, list):
            return [""] * len(prompt)
        return [""] * batch_size

    @staticmethod
    def _mask_to_control_image(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"mask control expects [H, W], [B, H, W], or [B, C, H, W], got {tuple(mask.shape)}")

        if mask.shape[1] == 1:
            mask = mask.repeat(1, 3, 1, 1)
        elif mask.shape[1] != 3:
            raise ValueError(f"mask control expects 1 or 3 channels, got {mask.shape[1]}")

        return mask

    @staticmethod
    def _resize_mask_to_image(mask: PIL.Image.Image, image: PIL.Image.Image) -> PIL.Image.Image:
        if mask.size == image.size:
            return mask
        return mask.resize(image.size, PIL.Image.Resampling.NEAREST)

    def _mask_control_cfg(self):
        return getattr(self.config.model, "mask_control", None)

    def _use_explicit_mask_control(self) -> bool:
        """Independent placement-mask control switch (default off = legacy)."""

        cfg = self._mask_control_cfg()
        return bool(getattr(cfg, "enabled", False))

    def _mask_control_encoding(self) -> str:
        cfg = self._mask_control_cfg()
        return str(getattr(cfg, "encoding", "rgb_vae"))

    def _use_geometric_mask_control(self) -> bool:
        return self._use_explicit_mask_control() and self._mask_control_encoding() == "geometry_tokenizer"

    @staticmethod
    def _resize_mask_for_geometry(
        mask: torch.Tensor,
        *,
        target_height: int,
        target_width: int,
        threshold: float,
    ) -> torch.Tensor:
        """Return a binary mask expressed in the target-image pixel grid."""

        if mask.ndim != 3:
            raise ValueError(
                "geometry source mask must have shape [B,H,W], "
                f"got {tuple(mask.shape)}"
            )
        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                "geometry target size must be positive, "
                f"got {(target_height, target_width)}"
            )
        if tuple(mask.shape[-2:]) == (target_height, target_width):
            return mask

        binary = (mask.detach().float() > float(threshold)).to(dtype=torch.float32)
        return F.interpolate(
            binary.unsqueeze(1),
            size=(target_height, target_width),
            mode="nearest",
        )[:, 0]

    @staticmethod
    def _mask_geometry_target_size(batch: Dict, tar_image: torch.Tensor, stage: str) -> tuple[int, int]:
        """Resolve the pixel grid on which mask geometry must be defined."""

        source_height, source_width = (int(value) for value in tar_image.shape[-2:])
        if stage != "predict":
            return source_height, source_width

        height = batch.get("height")
        width = batch.get("width")
        if height is None or width is None:
            return source_height, source_width
        if isinstance(height, torch.Tensor):
            if height.numel() != 1:
                raise ValueError("prediction height must be a scalar")
            height = height.item()
        if isinstance(width, torch.Tensor):
            if width.numel() != 1:
                raise ValueError("prediction width must be a scalar")
            width = width.item()
        return int(height), int(width)

    def _should_attach_mask_control(self) -> bool:
        """Attach legacy RGB-VAE control unless geometry replaces it."""

        return self._use_explicit_mask_control() and not self._use_geometric_mask_control()

    def _get_mask_geometry_tokenizer(self) -> Optional[MaskGeometryTokenizer]:
        return getattr(self._unwrap_dit(), "mask_geometry_tokenizer", None)

    def _get_mask_geometry_weight_path(self) -> Optional[str]:
        return self._companion_weight_path(MASK_GEOMETRY_TOKENIZER_FILE_NAME)

    def _companion_weight_path(self, file_name: str) -> Optional[str]:
        pretrained_weight = getattr(self.config.model.lora, "pretrained_weight", None)
        if pretrained_weight is None:
            return None
        weight_dir = pretrained_weight if os.path.isdir(pretrained_weight) else os.path.dirname(pretrained_weight)
        return os.path.join(weight_dir, file_name)

    def _alpha_head_enabled(self) -> bool:
        return bool(getattr(getattr(self.config.model, "alpha_head", None), "enabled", False))

    def _get_alpha_head(self) -> Optional[AlphaHead]:
        return getattr(self._unwrap_dit(), "alpha_head", None)

    def _alpha_supervised(self) -> bool:
        loss = self.config.loss
        return self._alpha_head_enabled() and (
            float(getattr(loss, "alpha_bce_weight", 0.0)) > 0 or float(getattr(loss, "alpha_dice_weight", 0.0)) > 0
        )

    def _load_alpha_head(self, stage: str, trainable_float32: bool) -> None:
        """Load the alpha-head companion exactly like the geometry tokenizer."""

        alpha_head = self._get_alpha_head()
        if alpha_head is None:
            raise ValueError("model.alpha_head is enabled but alpha_head is not attached")
        if trainable_float32:
            alpha_head.to(dtype=torch.float32)
        weight_path = self._companion_weight_path(ALPHA_HEAD_FILE_NAME)
        required = stage == "predict" or getattr(self.config, "resume", None) is not None
        if weight_path is None or not os.path.isfile(weight_path):
            if required:
                raise FileNotFoundError(f"Alpha head companion weights were not found at {weight_path}")
            if weight_path is not None:
                logging.warning(
                    "Warm-start LoRA has no alpha-head companion at %s; the zero-initialized head trains from scratch.",
                    weight_path,
                )
            return
        alpha_head.load_state_dict(safetensors.torch.load_file(weight_path), strict=True)

    def load_model(self, text_encoder_device=None):
        super().load_model(text_encoder_device=text_encoder_device)
        flow_cfg = getattr(self.config.model, "flow_map", None)
        if flow_cfg is not None and flow_cfg.enabled:
            from src.models.flow_map import FlowMapTimeEmbedding
            self.dit.flow_map_time_embedding = FlowMapTimeEmbedding(
                self.dit.time_text_embed, flow_cfg.source_time_weight
            )
        if self._alpha_head_enabled():
            self.dit.alpha_head = AlphaHead(int(self.dit.inner_dim)).to(dtype=self.weight_dtype)
        if not self._use_geometric_mask_control():
            return

        cfg = self._mask_control_cfg()
        self.dit.mask_geometry_tokenizer = MaskGeometryTokenizer(
            hidden_channels=int(cfg.tokenizer_hidden_channels),
            out_channels=int(self.dit.config.in_channels),
        ).to(dtype=self.weight_dtype)

    def setup_model_device_train_mode(self, stage="fit"):
        super().setup_model_device_train_mode(stage=stage)
        if stage == "fit" and hasattr(self._unwrap_dit(), "flow_map_time_embedding"):
            self._unwrap_dit().flow_map_time_embedding.requires_grad_(True)
        if stage == "fit" and self._alpha_head_enabled():
            self._get_alpha_head().requires_grad_(True)
        if stage != "fit" or not self._use_geometric_mask_control():
            return

        tokenizer = self._get_mask_geometry_tokenizer()
        if tokenizer is None:
            raise ValueError("geometric mask control is enabled but mask_geometry_tokenizer is not attached")
        tokenizer.requires_grad_(True)

    def save_additional_modules(self, save_folder):
        super().save_additional_modules(save_folder)
        flow = getattr(self._unwrap_dit(), "flow_map_time_embedding", None)
        if flow is not None:
            from src.models.flow_map import FLOW_MAP_FILE
            safetensors.torch.save_file(
                {k: v.detach().cpu().contiguous() for k, v in flow.state_dict().items()},
                os.path.join(save_folder, FLOW_MAP_FILE),
            )
        if self._alpha_head_enabled():
            safetensors.torch.save_file(
                {key: value.detach().cpu().contiguous() for key, value in self._get_alpha_head().state_dict().items()},
                os.path.join(save_folder, ALPHA_HEAD_FILE_NAME),
            )
        if not self._use_geometric_mask_control():
            return

        tokenizer = self._get_mask_geometry_tokenizer()
        if tokenizer is None:
            raise ValueError("geometric mask control is enabled but mask_geometry_tokenizer is not attached")
        state_dict = {key: value.detach().cpu() for key, value in tokenizer.state_dict().items()}
        safetensors.torch.save_file(
            state_dict,
            os.path.join(save_folder, MASK_GEOMETRY_TOKENIZER_FILE_NAME),
        )

    def load_additional_modules(self, stage="fit"):
        super().load_additional_modules(stage=stage)
        trainable_float32 = stage == "fit" and self.config.train.trainable_precision == "float32"
        flow = getattr(self._unwrap_dit(), "flow_map_time_embedding", None)
        if flow is not None:
            if trainable_float32:
                flow.to(dtype=torch.float32)
            from src.models.flow_map import FLOW_MAP_FILE
            weight = self.config.model.lora.pretrained_weight
            folder = weight if weight and os.path.isdir(weight) else os.path.dirname(weight or "")
            path = os.path.join(folder, FLOW_MAP_FILE)
            if os.path.isfile(path):
                flow.load_state_dict(safetensors.torch.load_file(path, device="cpu"), strict=True)
            else:
                # Warmup starts from a legacy fused checkpoint. Every other
                # flow-map load must carry its trained time-conditioning module.
                options = getattr(self.config.distillation, "meanflow", None)
                if stage != "fit" or self.config.resume or options is None or options.stage != "warmup":
                    raise FileNotFoundError(f"Missing trained flow-map companion: {path}")
        if self._alpha_head_enabled():
            self._load_alpha_head(stage, trainable_float32)
        if not self._use_geometric_mask_control():
            return

        tokenizer = self._get_mask_geometry_tokenizer()
        if tokenizer is None:
            raise ValueError("geometric mask control is enabled but mask_geometry_tokenizer is not attached")

        if trainable_float32:
            tokenizer.to(dtype=torch.float32)

        weight_path = self._get_mask_geometry_weight_path()
        is_resume = getattr(self.config, "resume", None) is not None
        if weight_path is None:
            if stage == "predict" or is_resume:
                raise FileNotFoundError(
                    "Geometric mask control requires companion weights for prediction or exact resume, "
                    "but no pretrained checkpoint is configured."
                )
            return
        if not os.path.isfile(weight_path):
            if stage == "predict" or is_resume:
                raise FileNotFoundError(
                    f"Geometric mask control companion weights were not found at {weight_path}"
                )
            logging.warning(
                "Warm-start LoRA has no geometric-mask companion at %s; "
                "the zero-initialized mask tokenizer will be trained from scratch.",
                weight_path,
            )
            return

        state_dict = safetensors.torch.load_file(weight_path)
        missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected mask geometry tokenizer keys: {unexpected}")
        if missing:
            raise ValueError(f"Missing mask geometry tokenizer keys: {missing}")

    def _reference_cfg_cfg(self):
        return self.config.model.reference_cfg

    def _use_reference_cfg(self) -> bool:
        return bool(self._reference_cfg_cfg().enabled)

    def _hybrid_masking_cfg(self, batch=None):
        data_cfg = getattr(self.config, "data", None)
        init_args = getattr(data_cfg, "init_args", None)
        profiles = getattr(init_args, "category_conditioning", {})
        if batch is not None and profiles:
            categories = batch["category"]
            category = categories if isinstance(categories, str) else categories[0]
            profile = profiles.get(category)
            if profile is not None and profile.hybrid_masking is not None:
                return profile.hybrid_masking
        return getattr(init_args, "hybrid_masking", None)

    def _hybrid_masking_rng(self) -> random.Random:
        """Derive a rank-local RNG from the checkpointed process RNG state."""

        base_seed = random.getrandbits(64)
        accelerator = getattr(self, "accelerator", None)
        process_index = int(getattr(accelerator, "process_index", 0))
        rank_offset = (process_index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
        return random.Random(base_seed ^ rank_offset)

    @staticmethod
    def _metadata_value(metadata: Dict, key: str, index: int):
        value = metadata.get(key, "")
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value.item()
            return value[index].item()
        if isinstance(value, (list, tuple)):
            return value[index] if index < len(value) else ""
        return value

    @classmethod
    def _format_sample_context(cls, batch: Dict) -> str:
        metadata = batch.get("sample_metadata")
        if not isinstance(metadata, dict):
            return f"category={batch.get('category', '<unknown>')}"

        sample_ids = metadata.get("sample_id", [])
        if isinstance(sample_ids, torch.Tensor):
            batch_size = int(sample_ids.shape[0]) if sample_ids.ndim else 1
        elif isinstance(sample_ids, (list, tuple)):
            batch_size = len(sample_ids)
        else:
            batch_size = 1

        contexts = []
        for index in range(batch_size):
            fields = []
            for key in (
                "dataset_index",
                "sample_id",
                "tar_image_path",
                "tar_mask_path",
                "ref_image_path",
                "ref_mask_path",
            ):
                value = cls._metadata_value(metadata, key, index)
                if value not in (None, ""):
                    fields.append(f"{key}={value}")
            contexts.append("{" + ", ".join(fields) + "}")
        return "samples=[" + ", ".join(contexts) + "]"

    def training_step(self, batch: dict) -> torch.Tensor:
        """Preserve the original failure while attaching actionable sample identity."""

        try:
            return super().training_step(batch)
        except Exception as exc:
            if hasattr(exc, "add_note"):
                accelerator = getattr(self, "accelerator", None)
                rank = int(getattr(accelerator, "process_index", 0))
                exc.add_note(
                    f"InsertAnything training context: rank={rank}, "
                    + self._format_sample_context(batch)
                )
            raise

    def _record_hybrid_masking_metrics(self, metadata: Dict[str, torch.Tensor], modes: List[str]) -> None:
        batch_size = int(metadata["condition_area_ratio"].shape[0])
        mode_names = (
            INSERT_ANYTHING_AUGMENTED_BOX,
            INSERT_ANYTHING_EXPANDED_BOX,
            INSERT_ANYTHING_DILATED_SEGMENTATION,
            "precomputed",
            MASKMIX_V2_EXACT,
            MASKMIX_V2_RANDOM_MORPHOLOGY,
            MASKMIX_V2_ASYMMETRIC_BOX,
            LOOSE_BLOB,
            LOOSE_OVERSIZED_BOX,
        )
        mode_indicators = torch.zeros(
            (batch_size, len(mode_names)),
            device=metadata["condition_area_ratio"].device,
            dtype=torch.float64,
        )
        mode_to_index = {mode: index for index, mode in enumerate(mode_names)}
        for sample_index, mode in enumerate(modes):
            if mode in mode_to_index:
                mode_indicators[sample_index, mode_to_index[mode]] = 1.0

        gt_area = metadata["gt_area_ratio"].to(torch.float64)
        sampled_expansion = metadata.get("sampled_condition_expansion")
        if sampled_expansion is None:
            sampled_expansion = (
                metadata["sampled_condition_area_ratio"].to(torch.float64)
                / gt_area.clamp_min(torch.finfo(torch.float64).eps)
            )
        else:
            sampled_expansion = sampled_expansion.to(torch.float64)

        condition_expansion = metadata.get("condition_expansion")
        if condition_expansion is None:
            condition_expansion = (
                metadata["condition_area_ratio"].to(torch.float64)
                / gt_area.clamp_min(torch.finfo(torch.float64).eps)
            )
        else:
            condition_expansion = condition_expansion.to(torch.float64)

        record = torch.cat(
            [
                metadata["gt_area_ratio"].reshape(-1, 1),
                metadata["sampled_condition_area_ratio"].reshape(-1, 1),
                metadata["condition_area_ratio"].reshape(-1, 1),
                metadata["context_guard_triggered"].to(torch.float64).reshape(-1, 1),
                metadata["unavoidable_large_gt"].to(torch.float64).reshape(-1, 1),
                mode_indicators,
                sampled_expansion.reshape(-1, 1),
                condition_expansion.reshape(-1, 1),
            ],
            dim=1,
        ).detach()
        self._last_hybrid_masking_metadata = {
            **{key: value.detach() for key, value in metadata.items()},
            "modes": list(modes),
        }
        metric_buffer = getattr(self, "_hybrid_masking_metric_buffer", None)
        if metric_buffer is None:
            metric_buffer = []
            self._hybrid_masking_metric_buffer = metric_buffer
        metric_buffer.append(record)

    def get_last_hybrid_masking_metadata(self):
        metadata = getattr(self, "_last_hybrid_masking_metadata", None)
        if metadata is None:
            return None
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else list(value)
            for key, value in metadata.items()
        }

    def _log_context_guard_fallbacks(
        self,
        batch: Dict,
        modes: List[str],
        metadata: Dict[str, torch.Tensor],
    ) -> None:
        triggered = metadata.get("fallback_applied", metadata["context_guard_triggered"]).detach().cpu()
        if not bool(triggered.any().item()):
            return
        sampled = metadata["sampled_condition_area_ratio"].detach().cpu()
        final = metadata["condition_area_ratio"].detach().cpu()
        gt = metadata["gt_area_ratio"].detach().cpu()
        unavoidable = metadata["unavoidable_large_gt"].detach().cpu()
        sample_metadata = batch.get("sample_metadata")
        accelerator = getattr(self, "accelerator", None)
        rank = int(getattr(accelerator, "process_index", 0))
        for index in torch.nonzero(triggered, as_tuple=False).flatten().tolist():
            sample_id = "<unknown>"
            dataset_index = "<unknown>"
            tar_mask_path = "<unknown>"
            if isinstance(sample_metadata, dict):
                sample_id = self._metadata_value(sample_metadata, "sample_id", index)
                dataset_index = self._metadata_value(sample_metadata, "dataset_index", index)
                tar_mask_path = self._metadata_value(
                    sample_metadata, "tar_mask_path", index
                )
            log = logging.warning if bool(unavoidable[index].item()) else logging.info
            log(
                "Hybrid Masking context guard fallback: rank=%s sample_id=%s dataset_index=%s "
                "tar_mask_path=%s mode=%s "
                "sampled_ratio=%.6f gt_ratio=%.6f final_ratio=%.6f unavoidable_large_gt=%s",
                rank,
                sample_id,
                dataset_index,
                tar_mask_path,
                modes[index],
                float(sampled[index]),
                float(gt[index]),
                float(final[index]),
                bool(unavoidable[index]),
            )

    @staticmethod
    def _normalize_precomputed_condition_mask(
        condition_mask: torch.Tensor,
        tar_mask: torch.Tensor,
        hybrid_cfg,
    ) -> torch.Tensor:
        if not isinstance(condition_mask, torch.Tensor):
            raise TypeError("tar_mask_cond must be a torch.Tensor")
        if condition_mask.shape != tar_mask.shape:
            raise ValueError(
                "tar_mask_cond must match tar_mask shape, got "
                f"{tuple(condition_mask.shape)} vs {tuple(tar_mask.shape)}"
            )
        if not torch.isfinite(condition_mask).all().item():
            raise ValueError("tar_mask_cond must contain only finite values")

        condition_mask = condition_mask.to(device=tar_mask.device)
        condition_mask = (condition_mask > hybrid_cfg.mask_threshold).to(dtype=tar_mask.dtype)
        if hybrid_cfg.ensure_gt_coverage:
            gt_binary = (tar_mask > hybrid_cfg.mask_threshold).to(dtype=tar_mask.dtype)
            condition_mask = torch.maximum(condition_mask, gt_binary)
        return condition_mask

    def _build_condition_mask(self, batch: Dict, tar_mask: torch.Tensor, stage: str) -> torch.Tensor:
        hybrid_cfg = self._hybrid_masking_cfg(batch)
        if stage == "opd":
            if batch.get("_opd_condition_mask_prepared", False):
                condition_mask = batch.get("tar_mask_cond")
                if condition_mask is None:
                    raise ValueError("prepared OPD batch is missing the shared tar_mask_cond")
                # prepare_opd_batch already normalized this mask, added the
                # interaction context and applied the context guard once.
                return condition_mask
            if hybrid_cfg is not None and hybrid_cfg.enabled:
                raise RuntimeError(
                    "Hybrid Masking for OPD must be sampled once through "
                    "prepare_opd_batch() before preparing policy conditions"
                )
            self._last_hybrid_masking_metadata = None
            return tar_mask

        if stage != "fit" or hybrid_cfg is None or not hybrid_cfg.enabled:
            self._last_hybrid_masking_metadata = None
            return tar_mask

        precomputed_mask = batch.get("tar_mask_cond")
        allow_fallback = None
        if precomputed_mask is not None:
            condition_mask = self._normalize_precomputed_condition_mask(
                condition_mask=precomputed_mask,
                tar_mask=tar_mask,
                hybrid_cfg=hybrid_cfg,
            )
            modes = list(
                batch.get("_condition_mask_modes")
                or batch.get("condition_mask_mode")
                or ["precomputed"] * int(tar_mask.shape[0])
            )
            # σ-split and MaskMix-v2 loose masks were already shrunk to the guard in the dataset.
            allow_fallback = batch.get("_condition_mask_allow_fallback")
            if allow_fallback is None and batch.get("condition_mask_mode") is not None:
                allow_fallback = torch.tensor([mode not in MASKMIX_LOOSE_MODES for mode in modes], dtype=torch.bool)
        else:
            condition_mask, modes = build_insert_anything_hybrid_masks(
                masks=tar_mask,
                categories=batch["category"],
                config=hybrid_cfg,
                rng=self._hybrid_masking_rng(),
            )

        condition_mask = add_interaction_context(
            condition_mask, tar_mask,
            getattr(hybrid_cfg, "interaction_dilation_px", 0), hybrid_cfg.mask_threshold,
            mode=getattr(hybrid_cfg, "interaction_context_mode", "union"),
        )
        condition_mask, guard_metadata = apply_hybrid_masking_context_guard(
            condition_masks=condition_mask,
            gt_masks=tar_mask,
            config=hybrid_cfg,
            allow_fallback=allow_fallback,
        )
        batch["hybrid_masking_modes"] = list(modes)
        self._record_hybrid_masking_metrics(guard_metadata, modes)
        self._log_context_guard_fallbacks(batch, modes, guard_metadata)
        return condition_mask

    def prepare_opd_batch(self, batch: Dict) -> Dict:
        """Sample stochastic conditioning once and share it across all policies."""

        prepared = dict(batch)
        prepared["tar_mask_cond"] = self._build_condition_mask(
            prepared,
            prepared["tar_mask"],
            stage="fit",
        )
        prepared["_opd_condition_mask_prepared"] = True
        return prepared

    def build_training_logs(
        self,
        epoch: int,
        train_dataloader,
        grad_norm: Optional[float],
        optimizer_step_time_sec: float,
    ) -> dict:
        logs = super().build_training_logs(
            epoch=epoch,
            train_dataloader=train_dataloader,
            grad_norm=grad_norm,
            optimizer_step_time_sec=optimizer_step_time_sec,
        )
        metric_buffer = getattr(self, "_hybrid_masking_metric_buffer", None)
        if not metric_buffer:
            return logs

        local_metrics = torch.cat(metric_buffer, dim=0)
        self._hybrid_masking_metric_buffer = []
        # Raw masks/augmentation statistics stay on CPU. NCCL needs CUDA
        # inputs; return the small gathered table to CPU once for all logs.
        gathered = self.accelerator.gather(
            local_metrics.to(self.accelerator.device)
        ).cpu()
        logs.update(
            {
                "hybrid/gt_area_mean": float(gathered[:, 0].mean().item()),
                "hybrid/sampled_condition_area_mean": float(gathered[:, 1].mean().item()),
                "hybrid/sampled_condition_area_p90": float(
                    torch.quantile(gathered[:, 1], 0.90).item()
                ),
                "hybrid/sampled_condition_area_max": float(gathered[:, 1].max().item()),
                "hybrid/condition_area_mean": float(gathered[:, 2].mean().item()),
                "hybrid/condition_area_p90": float(
                    torch.quantile(gathered[:, 2], 0.90).item()
                ),
                "hybrid/condition_area_max": float(gathered[:, 2].max().item()),
                "hybrid/visible_context_mean": float((1.0 - gathered[:, 2]).mean().item()),
                "hybrid/context_guard_rate": float(gathered[:, 3].mean().item()),
                "hybrid/unavoidable_large_gt_rate": float(gathered[:, 4].mean().item()),
                "hybrid/mode_augmented_box_rate": float(gathered[:, 5].mean().item()),
                "hybrid/mode_expanded_box_rate": float(gathered[:, 6].mean().item()),
                "hybrid/mode_dilated_segmentation_rate": float(gathered[:, 7].mean().item()),
                "hybrid/mode_precomputed_rate": float(gathered[:, 8].mean().item()),
                "hybrid/mode_exact_rate": float(gathered[:, 9].mean().item()),
                "hybrid/mode_random_morphology_rate": float(gathered[:, 10].mean().item()),
                "hybrid/mode_asymmetric_box_rate": float(gathered[:, 11].mean().item()),
                "hybrid/mode_loose_blob_rate": float(gathered[:, 12].mean().item()),
                "hybrid/mode_oversized_box_rate": float(gathered[:, 13].mean().item()),
                "hybrid/sampled_condition_expansion_mean": float(
                    gathered[:, 14].mean().item()
                ),
                "hybrid/sampled_condition_expansion_max": float(
                    gathered[:, 14].max().item()
                ),
                "hybrid/condition_expansion_mean": float(
                    gathered[:, 15].mean().item()
                ),
                "hybrid/condition_expansion_p90": float(
                    torch.quantile(gathered[:, 15], 0.90).item()
                ),
                "hybrid/condition_expansion_max": float(
                    gathered[:, 15].max().item()
                ),
            }
        )
        return logs

    def setup_validation(self, train_dataloader):
        if not self.config.logging.sampling.enable:
            self.validation_sampler = None
            return
        from src.trainer.insert_anything_validation_sampling import InsertAnythingValidationSampler

        self.validation_sampler = InsertAnythingValidationSampler(
            config=self.config,
            train_dataloader=train_dataloader,
            accelerator=self.accelerator,
        )

    def _drop_reference_for_training(self, stage: str) -> bool:
        if stage != "fit" or not self._use_reference_cfg():
            return False
        dropout_rate = self._reference_cfg_cfg().train_dropout_rate
        if dropout_rate <= 0:
            return False
        return bool(torch.rand((), device="cpu").item() < dropout_rate)

    def _force_empty_reference_prompt(self) -> bool:
        visual_guidance = self._use_reference_cfg() or getattr(
            getattr(self.config.model, "flow_map", None), "enabled", False
        )
        return visual_guidance and bool(self._reference_cfg_cfg().force_empty_prompt)

    def get_last_latent_anchoring_masks(self):
        artifacts = getattr(self, "_last_latent_anchoring_masks", None)
        if artifacts is None:
            return None
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in artifacts.items()
        }

    def predict(self, **kwargs):
        """Fail fast for unsupported explicit geometry before standalone model setup.

        Adaptive Crop is a wrapper concern: the trainer only sees the prediction-space
        source/mask (full-frame or already cropped) and must not blanket-reject
        adaptive_crop.predict_enabled.
        """

        flow = getattr(self.config.model, "flow_map", None)
        if flow is not None and flow.enabled:
            kwargs.setdefault("num_inference_steps", flow.inference_steps)
            kwargs.setdefault("true_cfg_scale", 1.0)
        settings = ensure_latent_anchoring_config(self.config)
        requested_mode = kwargs.get("latent_anchoring_mode")
        mode = normalize_latent_anchoring_mode(
            settings.mode if requested_mode is None else requested_mode
        )
        if latent_anchoring_enabled(mode) or self._complementary_canvas_cfg() is not None:
            if kwargs.get("height") is not None or kwargs.get("width") is not None:
                raise ValueError(
                    "latent anchoring and the complementary canvas do not yet support explicit height/width; "
                    "use the processor-derived prediction geometry size"
                )
        return super().predict(**kwargs)

    @staticmethod
    def _append_additional_control(parent_batch: Dict, control: torch.Tensor, vae_only: bool) -> str:
        next_control_index = int(parent_batch["n_controls"]) + 1
        control_key = f"control_{next_control_index}"
        parent_batch[control_key] = control
        parent_batch["n_controls"] = next_control_index
        if vae_only:
            parent_batch.setdefault("vae_only_control_keys", []).append(control_key)
        return control_key

    def _sigma_split_cfg(self):
        cfg = getattr(self.config.model, "sigma_split", None)
        return cfg if cfg is not None and cfg.enabled else None

    def _sigma_handoff_settings(self):
        handoff = getattr(getattr(self.config, "predict", None), "sigma_handoff", None)
        if handoff is None or not handoff.enabled:
            return None
        tau = self.config.model.sigma_split.tau if handoff.tau is None else handoff.tau
        return SimpleNamespace(
            tau=float(tau),
            alpha_threshold=float(handoff.alpha_threshold),
            interaction_dilation_px=handoff.interaction_dilation_px,
        )

    def _sample_sigma_route(self, batch_size: int, cfg):
        """Draw the flow-matching u early (same law as ``_compute_loss``) and route on its σ."""

        u = compute_density_for_timestep_sampling(
            weighting_scheme="none",
            batch_size=batch_size,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        _, schedule_sigmas = self._get_cached_scheduler_training_tensors(torch.device("cpu"), torch.float32)
        sigmas = schedule_sigmas[self._timestep_indices_from_u(u)]
        return u, route_high_noise(sigmas, cfg.tau, cfg.overlap)

    @staticmethod
    def _select_route_masks(batch: Dict, route_high: torch.Tensor) -> Dict:
        """σ > τ conditions on the loose mask, σ <= τ on the tight mask of the same crop."""

        loose, tight = batch.get("tar_mask_cond"), batch.get("tar_mask_cond_tight")
        if loose is None or tight is None:
            raise ValueError(
                "model.sigma_split needs the dataset's loose/tight placement masks "
                "(data.init_args.hybrid_masking.loose_placement)"
            )
        count = int(route_high.numel())
        placement = batch.get("placement_metadata") or {}
        loose_modes = list(placement.get("loose_mode", ["precomputed"] * count))
        tight_modes = list(placement.get("tight_mode", ["precomputed"] * count))
        routed = dict(batch)
        routed["tar_mask_cond"] = torch.where(
            route_high.view(-1, 1, 1).to(loose.device), loose, tight.to(device=loose.device, dtype=loose.dtype)
        )
        routed["_condition_mask_modes"] = [
            loose_modes[index] if bool(route_high[index]) else tight_modes[index] for index in range(count)
        ]
        routed["_condition_mask_allow_fallback"] = ~route_high
        return routed

    @staticmethod
    def _route_reference(ref_image: torch.Tensor, ref_mask: torch.Tensor, route_high: torch.Tensor, cfg):
        """Strong geometric + color jitter for the layout route, light photometric jitter otherwise."""

        strong = light = None
        if bool(route_high.any()):
            strong, _ = augment_reference_strong(ref_image, ref_mask, cfg.high_noise)
        if not bool(route_high.all()):
            light = augment_reference_light(ref_image, ref_mask, cfg.low_noise)
        if light is None:
            return strong
        if strong is None:
            return light
        return torch.where(route_high.view(-1, 1, 1, 1).to(ref_image.device), strong, light)

    def _shuffle_reference_tokens(self, parent_batch: Dict, rows: Optional[torch.Tensor]) -> torch.Tensor:
        """Shuffle the reference VAE tokens (``control_1``) of the selected samples."""

        step = self.vae_scale_factor * 2
        source_tokens = (int(parent_batch["height_control"]) // step) * (int(parent_batch["width_control"]) // step)
        reference_tokens = (int(parent_batch["height_control_1"]) // step) * (
            int(parent_batch["width_control_1"]) // step
        )
        control_latents = parent_batch["control_latents"]
        shuffled = shuffle_token_segment(control_latents, source_tokens, reference_tokens)
        if rows is None:
            return shuffled
        return torch.where(rows.view(-1, 1, 1).to(control_latents.device), shuffled, control_latents)

    def sampling_from_embeddings(self, embeddings: dict, **kwargs) -> torch.Tensor:
        stash = embeddings.get("_sigma_handoff")
        start_sigma = None if kwargs else self._source_noise_start()
        if start_sigma is not None:
            if stash is not None:
                raise ValueError("source_noise_start cannot be combined with the σ-split hand-off")
            return self._sample_from_noised_source(embeddings, start_sigma)
        if stash is None or kwargs:
            return super().sampling_from_embeddings(embeddings, **kwargs)
        return self._sample_with_sigma_handoff(embeddings, stash)

    def _source_noise_start(self) -> Optional[float]:
        settings = getattr(getattr(self.config, "predict", None), "latent_anchoring", None)
        return getattr(settings, "source_noise_start", None)

    def _sample_from_noised_source(self, embeddings: Dict, start_sigma: float) -> torch.Tensor:
        """Start at the step nearest ``start_sigma`` from the source latent noised by the run's noise.

        Outside M_free this equals the anchored latent of that step; inside it the old object
        replaces pure noise, so its pose, scale and lighting carry over and the remaining
        steps rewrite its appearance from the reference.
        """

        source = embeddings.get("anchor_source_latents")
        if source is None:
            raise ValueError("source_noise_start needs latent anchoring (mode core, expanded or expanded_soft)")
        height, width = int(embeddings["height"]), int(embeddings["width"])
        token = self.vae_scale_factor * 2
        timesteps, _ = self.prepare_predict_timesteps(
            int(embeddings["num_inference_steps"]), (height // token) * (width // token)
        )
        sigmas = self.scheduler.sigmas[: len(timesteps)].float()
        step = int(torch.argmin((sigmas - float(start_sigma)).abs()).item())
        # Same draw as a full-noise run with this seed: it is also the anchoring trajectory noise.
        noise = self.prepare_initial_latents(
            batch_size=int(embeddings["control_latents"].shape[0]),
            num_channels_latents=self._dit_config().in_channels // 4,
            height=height,
            width=width,
            dtype=self.weight_dtype,
            device=self._dit_device(),
        )
        source = source.to(device=noise.device, dtype=noise.dtype)
        sigma = self.scheduler.sigmas[step].to(device=noise.device, dtype=noise.dtype)
        initial = sigma * noise + (1.0 - sigma) * source
        self._last_source_noise_start = {"requested": float(start_sigma), "sigma": float(sigmas[step]), "step": step}
        logging.info("source_noise_start=%s: starting at step %d (σ=%.3f)", start_sigma, step, float(sigmas[step]))
        return super().sampling_from_embeddings(embeddings, initial_latents=initial, anchor_noise=noise, step_start=step)

    def get_last_source_noise_start(self):
        return getattr(self, "_last_source_noise_start", None)

    def _sample_with_sigma_handoff(self, high: Dict, stash: Dict) -> torch.Tensor:
        """Layout segment, alpha-head region at σ <= τ, then the identity segment."""

        batch, settings = stash["batch"], stash["settings"]
        height, width = int(high["height"]), int(high["width"])
        token = self.vae_scale_factor * 2
        timesteps, _ = self.prepare_predict_timesteps(
            int(high["num_inference_steps"]), (height // token) * (width // token)
        )
        step = handoff_step_index(self.scheduler.sigmas, settings.tau, len(timesteps))
        noise = self.prepare_initial_latents(
            batch_size=int(high["control_latents"].shape[0]),
            num_channels_latents=self._dit_config().in_channels // 4,
            height=height,
            width=width,
            dtype=self.weight_dtype,
            device=self._dit_device(),
        )
        latents = super().sampling_from_embeddings(
            high, initial_latents=noise, anchor_noise=noise, step_end=step, capture_alpha=True
        )

        placement = (batch["tar_mask"].float() > 0.5).float()
        region, probabilities = alpha_region_from_logits(
            self._last_alpha_logits, (height // token, width // token), (height, width), settings.alpha_threshold
        )
        core = region.to(placement.device) * placement
        empty = core.flatten(1).sum(dim=1) == 0
        if bool(empty.any()):
            logging.warning("alpha head found no object at the σ hand-off; keeping the placement mask")
            core = torch.where(empty.view(-1, 1, 1), placement, core)

        anchoring = ensure_latent_anchoring_config(self.config)
        mode = normalize_latent_anchoring_mode(batch.get("latent_anchoring_mode", "legacy"))
        dilation = settings.interaction_dilation_px
        if dilation is None:
            dilation = int(batch.get("latent_anchoring_interaction_dilation_px", anchoring.interaction_dilation_px))
        low_batch = {**batch, "_sigma_segment": "low", "tar_mask": core}
        anchored = latent_anchoring_enabled(mode)
        if anchored or dilation > 0:
            masks = build_latent_anchoring_masks(
                core,
                mode=mode if anchored else "expanded",
                interaction_dilation_px=dilation,
                transition_width_px=int(
                    batch.get("latent_anchoring_transition_width_px", anchoring.transition_width_px)
                ),
            )
            low_batch["source_hole_mask"] = masks.free
            if anchored:
                low_batch.update(
                    anchor_mask_core=masks.core,
                    anchor_mask_free=masks.free,
                    anchor_edit_weight=masks.edit_weight,
                )
                self._last_latent_anchoring_masks = {
                    "mode": mode,
                    "core": masks.core[0].detach().cpu(),
                    "free": masks.free[0].detach().cpu(),
                    "edit_weight": masks.edit_weight[0].detach().cpu(),
                }
        low = self.prepare_embeddings(low_batch, stage="predict")
        if anchored:
            # Re-anchor everything outside the tight region before the first identity step.
            latents = self._apply_post_step_latent_anchoring(
                generated_next=latents,
                anchor_source_latents=low["anchor_source_latents"].to(device=latents.device, dtype=latents.dtype),
                anchor_noise=noise.to(device=latents.device, dtype=latents.dtype),
                edit_weight=low["anchor_edit_weight"].to(device=latents.device, dtype=latents.dtype),
                timesteps=timesteps,
                step_index=step - 1,
            )
        self._last_sigma_handoff = {
            "step": int(step),
            "sigma": float(self.scheduler.sigmas[step]),
            "tau": float(settings.tau),
            "alpha_probabilities": probabilities.detach().cpu(),
            "core": core.detach().cpu(),
        }
        return super().sampling_from_embeddings(low, initial_latents=latents, anchor_noise=noise, step_start=step)

    def get_last_sigma_handoff(self):
        return getattr(self, "_last_sigma_handoff", None)

    def prepare_embeddings(self, batch: Dict, stage="fit") -> Dict:
        """
        Convert Insert Anything mask_prompt data to the parent Qwen-Edit conditioning format.

        Input:
            - ref_image: [B, C, H, W] in [0, 1]
            - ref_mask: [B, H, W] in [0, 1]
            - tar_image: [B, C, H, W] in [0, 1]
            - tar_mask: [B, H, W] GT/loss mask in [0, 1]
            - tar_mask_cond: optional [B, H, W] conditioning-only mask
            - prompt: List[str]
            - category: List[str]
        """
        required_keys = ["ref_image", "ref_mask", "tar_image", "tar_mask", "prompt", "category"]
        missing = [key for key in required_keys if key not in batch or batch[key] is None]
        if missing:
            raise ValueError(f"InsertAnything mask_prompt batch is missing required keys: {missing}")

        ref_image = batch["ref_image"]
        ref_mask = batch["ref_mask"]
        tar_image = batch["tar_image"]
        tar_mask = batch["tar_mask"]
        batch_size = ref_image.shape[0]

        if tar_image.shape[0] != batch_size or tar_mask.shape[0] != batch_size or ref_mask.shape[0] != batch_size:
            raise ValueError("ref_image, ref_mask, tar_image, and tar_mask batch sizes must match")

        handoff = self._sigma_handoff_settings() if stage == "predict" else None
        if handoff is not None and "_sigma_segment" not in batch:
            # Hand-off sampling starts in the layout segment (shuffled reference,
            # the user's loose mask, no reference CFG); the raw batch is kept to
            # build the identity-segment condition once σ crosses τ.
            embeddings = self.prepare_embeddings({**batch, "_sigma_segment": "high", "true_cfg_scale": 1.0}, stage)
            embeddings["_sigma_handoff"] = {"batch": batch, "settings": handoff}
            return embeddings
        segment = batch.get("_sigma_segment")
        sigma_split = self._sigma_split_cfg() if stage == "fit" else None
        route_high = None
        if sigma_split is not None:
            timestep_u, route_high = self._sample_sigma_route(batch_size, sigma_split)
            batch = self._select_route_masks(batch, route_high)

        tar_mask_cond = self._build_condition_mask(batch, tar_mask, stage)
        source_hole_mask = batch.get("source_hole_mask", tar_mask_cond)
        if source_hole_mask.shape != tar_mask.shape:
            raise ValueError(
                "source_hole_mask must match tar_mask shape, got "
                f"{tuple(source_hole_mask.shape)} vs {tuple(tar_mask.shape)}"
            )
        canvas = self._complementary_canvas_cfg()
        # R: every target token touched by the condition hole (interaction margin included).
        canvas_region = (
            token_region_from_mask(source_hole_mask, self.vae_scale_factor * 2) if canvas is not None else None
        )

        cache = batch.get("_opd_encoding_cache") if stage == "opd" else None

        def cached(key, compute):
            if cache is None:
                return compute()
            if key not in cache:
                cache[key] = compute()
            return cache[key]

        hole_key = id(batch.get("source_hole_mask", batch.get("tar_mask_cond", tar_mask)))
        control = cached(
            ("insert_source", id(tar_image), hole_key),
            lambda: tar_image * (1 - source_hole_mask.unsqueeze(1)),
        )

        # Teacher and Student guidance policies are independent of standalone
        # inference defaults. OPD can therefore distill a reference-guided
        # Teacher into an unguided Student.
        if stage == "opd" and "_opd_use_reference_cfg" in batch:
            use_reference_cfg = bool(batch["_opd_use_reference_cfg"])
        else:
            use_reference_cfg = self._use_reference_cfg()
        if use_reference_cfg and "negative_prompt" in batch:
            raise ValueError("reference_cfg cannot be combined with negative_prompt text CFG")

        drop_reference = self._drop_reference_for_training(stage)
        force_empty_reference_prompt = self._force_empty_reference_prompt()
        if stage == "opd" and "_opd_prompt_override" in batch:
            prompt = batch["_opd_prompt_override"]
        else:
            prompt = (
                self._empty_prompts(batch["prompt"], batch_size)
                if force_empty_reference_prompt
                else batch["prompt"]
            )
        mask_control = None
        if self._should_attach_mask_control():
            mask_control = cached(
                ("insert_mask_control", hole_key),
                lambda: self._mask_to_control_image(source_hole_mask),
            )
        mask_geometry = None
        if self._use_geometric_mask_control():
            mask_cfg = self._mask_control_cfg()
            geometry_height, geometry_width = self._mask_geometry_target_size(
                batch,
                tar_image,
                stage,
            )
            mask_geometry = cached(
                ("geometry", hole_key, geometry_height, geometry_width,
                 float(mask_cfg.sigma_px), float(mask_cfg.mask_threshold)),
                lambda: build_mask_geometry(
                    self._resize_mask_for_geometry(
                        source_hole_mask,
                        target_height=geometry_height,
                        target_width=geometry_width,
                        threshold=float(mask_cfg.mask_threshold),
                    ),
                    sigma_px=float(mask_cfg.sigma_px), threshold=float(mask_cfg.mask_threshold)
                ),
            )

        # The masked source always enters the DiT; the VL encoder may see only the reference.
        def vl_source():
            return {} if getattr(self.config.model, "vl_masked_source", True) else {"vae_only_control_keys": ["control"]}

        parent_batch = {
            "control": control,
            "prompt": prompt,
            "n_controls": 0,
            **vl_source(),
        }
        if cache is not None:
            parent_batch["_opd_encoding_cache"] = cache
        if mask_geometry is not None:
            parent_batch["mask_geometry"] = mask_geometry

        if not drop_reference:
            reference, reference_in_vl = ref_image, True
            if route_high is not None:
                reference = self._route_reference(ref_image, ref_mask, route_high, sigma_split)
                reference_in_vl = not (bool(route_high.any()) and sigma_split.high_noise.vl_reference == "drop")
            elif segment == "high":
                reference_in_vl = self.config.model.sigma_split.high_noise.vl_reference != "drop"
            self._append_additional_control(parent_batch, reference, vae_only=not reference_in_vl)

        if mask_control is not None:
            self._append_additional_control(parent_batch, mask_control, vae_only=True)

        if stage not in {"predict", "opd"}:
            parent_batch["reference_present_for_loss"] = not drop_reference
            for key in ("target_face_mask", "reference_face_mask"):
                if key in batch:
                    parent_batch[key] = batch[key]
            parent_batch["image"] = tar_image
            if self.config.loss.mask_loss:
                parent_batch["mask"] = tar_mask
            if getattr(self.config.loss, "region_perceptual", False):
                # Keep the unmodified pixel tensors alongside their VAE controls.
                # Region losses use the GT mask/image, never the sampled MaskMix
                # condition mask.  Reference tensors are consumed only when the
                # optional identity weight is non-zero, and are the reference
                # before lighting augmentation, so identity never pulls the
                # output toward an augmented light.
                parent_batch["target_image_for_loss"] = tar_image
                parent_batch["ref_image_for_loss"] = batch.get("ref_image_clean", ref_image)
                parent_batch["ref_mask_for_loss"] = batch.get("ref_mask_clean", ref_mask)

        if stage == "opd":
            # The GT image supplies the source canvas but must never be encoded
            # as the supervised target latent during on-policy distillation.
            parent_batch["height"] = int(tar_image.shape[-2])
            parent_batch["width"] = int(tar_image.shape[-1])

        for key in ("height", "width", "num_inference_steps", "guidance"):
            # OPD dimensions belong to the current target tensor.  Do not
            # overwrite the scalar canvas size with a collated per-sample
            # metadata tensor supplied by a custom dataset.
            if key in batch and not (stage == "opd" and key in {"height", "width"}):
                parent_batch[key] = batch[key]

        parent_batch["true_cfg_scale"] = batch.get("true_cfg_scale", 1.0)

        if "negative_prompt" in batch:
            parent_batch["negative_prompt"] = batch["negative_prompt"]

        reference_cfg_parent_batch = None
        reference_cfg_scale = float(batch.get("true_cfg_scale", 1.0))
        if stage in {"predict", "opd"} and use_reference_cfg and reference_cfg_scale > 1.0:
            # Null-reference branch keeps source/context (+ optional placement mask)
            # while dropping the reference image.
            reference_cfg_parent_batch = {
                "control": control,
                "prompt": prompt,
                "n_controls": 0,
                "true_cfg_scale": 1.0,
                **vl_source(),
            }
            if cache is not None:
                reference_cfg_parent_batch["_opd_encoding_cache"] = cache
            if mask_geometry is not None:
                reference_cfg_parent_batch["mask_geometry"] = mask_geometry
            if stage == "opd":
                reference_cfg_parent_batch["height"] = int(tar_image.shape[-2])
                reference_cfg_parent_batch["width"] = int(tar_image.shape[-1])
            if mask_control is not None:
                self._append_additional_control(reference_cfg_parent_batch, mask_control, vae_only=True)
            for key in ("height", "width", "num_inference_steps", "guidance"):
                if key in batch and not (stage == "opd" and key in {"height", "width"}):
                    reference_cfg_parent_batch[key] = batch[key]

        anchor_source_image = batch.get("anchor_source_image")
        anchor_mask_free = batch.get("anchor_mask_free")
        anchor_edit_weight = batch.get("anchor_edit_weight")
        latent_anchoring_mode = normalize_latent_anchoring_mode(
            batch.get("latent_anchoring_mode", "legacy")
        )
        anchor_fields_present = any(
            value is not None
            for value in (anchor_source_image, anchor_mask_free, anchor_edit_weight)
        )
        if anchor_fields_present and not all(
            value is not None
            for value in (anchor_source_image, anchor_mask_free, anchor_edit_weight)
        ):
            raise ValueError(
                "latent anchoring requires anchor_source_image, anchor_mask_free, and anchor_edit_weight"
            )
        if anchor_fields_present and stage != "predict":
            raise ValueError("latent anchoring is inference-only and cannot be prepared during training")

        parent_batch = super().prepare_embeddings(parent_batch, stage=stage)

        high_noise = None
        if route_high is not None and bool(route_high.any()):
            high_noise = sigma_split.high_noise
        elif segment == "high":
            high_noise = self.config.model.sigma_split.high_noise
        if high_noise is not None and high_noise.shuffle_reference_tokens and not drop_reference:
            # A bag of reference patches: appearance without the reference pose.
            parent_batch["control_latents"] = self._shuffle_reference_tokens(parent_batch, route_high)

        for key in ("height", "width", "num_inference_steps", "guidance"):
            if key in batch and not (stage == "opd" and key in {"height", "width"}):
                parent_batch[key] = batch[key]
        parent_batch["true_cfg_scale"] = batch.get(
            "true_cfg_scale", parent_batch["true_cfg_scale"]
        )

        if canvas_region is not None:
            parent_batch["canvas_region"] = canvas_region
            if stage == "predict":
                # The generated R is scattered into the unmasked source latent before decoding.
                parent_batch["canvas_source_latents"] = self.encode_image_latents(
                    self.preprocess_image_for_vae_encoder(tar_image, input_range="0-1"),
                    batch_size,
                    self.num_channels_latents,
                    self.weight_dtype,
                )

        if anchor_fields_present:
            anchor_vae_input = self.preprocess_image_for_vae_encoder(
                anchor_source_image, input_range="0-1"
            )
            target_height = int(parent_batch["height"])
            target_width = int(parent_batch["width"])
            if anchor_vae_input.shape[3:] != (target_height, target_width):
                raise ValueError(
                    "full source anchor geometry must match target generation geometry, got "
                    f"{tuple(anchor_vae_input.shape[3:])} vs {(target_height, target_width)}"
                )
            anchor_source_latents = self.encode_image_latents(
                anchor_vae_input,
                batch_size,
                self.num_channels_latents,
                self.weight_dtype,
            )
            packed_edit_weight = self._prepare_latent_anchoring_weight(
                edit_weight=anchor_edit_weight,
                free_mask=anchor_mask_free,
                batch_size=batch_size,
                num_channels_latents=self.num_channels_latents,
                height_image=target_height,
                width_image=target_width,
                soft_transition=latent_anchoring_mode == "expanded_soft",
            )
            if packed_edit_weight.shape != anchor_source_latents.shape:
                raise ValueError(
                    "packed W_edit must match full source latent shape, got "
                    f"{tuple(packed_edit_weight.shape)} vs {tuple(anchor_source_latents.shape)}"
                )
            parent_batch["anchor_source_latents"] = anchor_source_latents
            parent_batch["anchor_edit_weight"] = packed_edit_weight
            parent_batch["latent_anchoring_mode"] = latent_anchoring_mode

        if reference_cfg_parent_batch is not None:
            reference_cfg_cfg = self._reference_cfg_cfg()
            reference_cfg_embeddings = super().prepare_embeddings(reference_cfg_parent_batch, stage=stage)
            for key in ("height", "width", "num_inference_steps", "guidance"):
                if key in batch:
                    reference_cfg_embeddings[key] = batch[key]
            parent_batch["reference_cfg_embeddings"] = reference_cfg_embeddings
            parent_batch["reference_cfg_scale"] = reference_cfg_scale
            policy_cfg = batch.get("_opd_reference_cfg", {}) if stage == "opd" else {}
            parent_batch["reference_cfg_delta_clip_enabled"] = bool(
                policy_cfg.get("delta_clip_enabled", reference_cfg_cfg.delta_clip_enabled)
            )
            parent_batch["reference_cfg_delta_clip_max_ratio"] = float(
                policy_cfg.get("delta_clip_max_ratio", reference_cfg_cfg.delta_clip_max_ratio)
            )
            if policy_cfg.get("mask_aware", reference_cfg_cfg.mask_aware):
                parent_batch["reference_cfg_mask_aware"] = True
                parent_batch["reference_cfg_mask"] = tar_mask_cond

        if route_high is not None:
            parent_batch["_sigma_split_timestep_u"] = timestep_u
            parent_batch["_sigma_split_route_high"] = route_high
        if stage == "fit" and self._alpha_supervised():
            parent_batch["alpha_target_mask"] = tar_mask

        parent_batch["category"] = batch["category"]
        return parent_batch

    def prepare_predict_batch_data(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[Union[str, List[str]]] = None,
        ref_mask: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        src_mask: Optional[Union[PIL.Image.Image, List[PIL.Image.Image]]] = None,
        controls_size: Optional[List[int]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
        latent_anchoring_mode: Optional[str] = None,
        latent_anchoring_mask_threshold: Optional[float] = None,
        interaction_dilation_px: Optional[int] = None,
        transition_width_px: Optional[int] = None,
    ) -> dict:
        if image is None:
            raise ValueError("image is required")
        if not isinstance(image, list):
            image = [image]

        if len(image) < 2:
            raise ValueError("mask_prompt prediction requires [src_image, ref_image] plus src_mask")

        src_image = image[0]
        if src_mask is not None:
            src_mask_img = src_mask if not isinstance(src_mask, list) else src_mask[0]
            ref_image = image[1]
            ref_mask_index = 2
        else:
            if len(image) < 3:
                raise ValueError("mask_prompt prediction requires src_mask or image=[src_image, src_mask, ref_image]")
            src_mask_img = image[1]
            ref_image = image[2]
            ref_mask_index = 3

        if len(image) > ref_mask_index:
            ref_mask_img = image[ref_mask_index]
        elif ref_mask is not None:
            ref_mask_img = ref_mask if not isinstance(ref_mask, list) else ref_mask[0]
        elif ref_image.mode == "RGBA":
            ref_mask_img = ref_image.split()[-1]
        else:
            raise ValueError(
                "ref_mask is required for mask_prompt prediction. Provide it as an image input, ref_mask, or RGBA alpha."
            )

        use_reference_cfg = self._use_reference_cfg()
        if use_reference_cfg and negative_prompt is not None:
            raise ValueError("reference_cfg cannot be combined with negative_prompt text CFG")

        force_empty_reference_prompt = self._force_empty_reference_prompt()
        if prompt is None:
            if not force_empty_reference_prompt:
                raise ValueError("prompt is required unless reference_cfg.force_empty_prompt is enabled")
            prompt = ""
        if isinstance(prompt, str):
            prompt = [prompt]
        if len(prompt) != 1:
            raise ValueError("mask_prompt prediction expects exactly one prompt for one input image")

        latent_anchoring_config = ensure_latent_anchoring_config(self.config)
        effective_anchoring_mode = normalize_latent_anchoring_mode(
            latent_anchoring_config.mode
            if latent_anchoring_mode is None
            else latent_anchoring_mode
        )
        effective_mask_threshold = float(
            latent_anchoring_config.mask_threshold
            if latent_anchoring_mask_threshold is None
            else latent_anchoring_mask_threshold
        )
        effective_interaction_dilation_px = int(
            latent_anchoring_config.interaction_dilation_px
            if interaction_dilation_px is None
            else interaction_dilation_px
        )
        effective_transition_width_px = int(
            latent_anchoring_config.transition_width_px
            if transition_width_px is None
            else transition_width_px
        )
        anchoring_enabled = latent_anchoring_enabled(effective_anchoring_mode)
        if anchoring_enabled and (height is not None or width is not None):
            raise ValueError(
                "latent anchoring does not yet support explicit height/width; "
                "use the processor-derived prediction geometry size"
            )

        ref_image_np, ref_mask_np = extract_foreground_from_image_and_mask(ref_image, ref_mask_img)
        ref_image = PIL.Image.fromarray(ref_image_np)
        ref_mask_img = PIL.Image.fromarray(ref_mask_np, mode="L")

        src_mask_img = self._resize_mask_to_image(src_mask_img, src_image)
        ref_mask_img = self._resize_mask_to_image(ref_mask_img, ref_image)

        processed = self.preprocessor.preprocess(
            {
                "image": src_image,
                "mask": src_mask_img,
                "controls": [ref_image],
                "control_masks": [ref_mask_img],
            }
        )

        src_tensor = processed["image"].unsqueeze(0)
        src_mask_tensor = processed["mask"].unsqueeze(0)
        ref_tensor = processed["controls"][0].unsqueeze(0)
        ref_mask_tensor = processed["control_masks"][0].unsqueeze(0)

        self._last_latent_anchoring_masks = None
        anchoring_masks = None
        if anchoring_enabled:
            anchoring_masks = build_latent_anchoring_masks(
                src_mask_tensor,
                mode=effective_anchoring_mode,
                mask_threshold=effective_mask_threshold,
                interaction_dilation_px=effective_interaction_dilation_px,
                transition_width_px=effective_transition_width_px,
            )
            src_mask_tensor = anchoring_masks.core
            self._last_latent_anchoring_masks = {
                "mode": effective_anchoring_mode,
                "core": anchoring_masks.core[0].detach().cpu(),
                "free": anchoring_masks.free[0].detach().cpu(),
                "edit_weight": anchoring_masks.edit_weight[0].detach().cpu(),
            }

        canvas_hole_mask = None
        if self._complementary_canvas_cfg() is not None and effective_interaction_dilation_px > 0:
            # Training removes the interaction band around the object from the source; edit the same M_free.
            canvas_masks = build_latent_anchoring_masks(
                src_mask_tensor,
                mode="expanded",
                mask_threshold=effective_mask_threshold,
                interaction_dilation_px=effective_interaction_dilation_px,
            )
            src_mask_tensor, canvas_hole_mask = canvas_masks.core, canvas_masks.free

        data = {
            "ref_image": ref_tensor,
            "ref_mask": ref_mask_tensor,
            "tar_image": src_tensor,
            "tar_mask": src_mask_tensor,
            "prompt": [""] if force_empty_reference_prompt else prompt,
            "category": ["predict"],
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance": guidance_scale,
        }
        if canvas_hole_mask is not None:
            data["source_hole_mask"] = canvas_hole_mask

        if anchoring_masks is not None:
            data.update(
                {
                    "source_hole_mask": anchoring_masks.free,
                    "anchor_source_image": src_tensor,
                    "anchor_mask_core": anchoring_masks.core,
                    "anchor_mask_free": anchoring_masks.free,
                    "anchor_edit_weight": anchoring_masks.edit_weight,
                    "latent_anchoring_mode": effective_anchoring_mode,
                    # The σ hand-off rebuilds these masks around the alpha region.
                    "latent_anchoring_interaction_dilation_px": effective_interaction_dilation_px,
                    "latent_anchoring_transition_width_px": effective_transition_width_px,
                }
            )

        if negative_prompt is not None:
            data["negative_prompt"] = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

        if height is None or width is None:
            data["height"] = src_tensor.shape[2]
            data["width"] = src_tensor.shape[3]
        else:
            data["height"] = height
            data["width"] = width

        self.weight_dtype = weight_dtype
        return data

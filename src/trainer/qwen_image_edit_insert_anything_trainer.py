"""
Trainer for the Insert Anything mask_prompt task using QwenImageEdit.
"""

import logging
import os
import random
from typing import Dict, List, Optional, Union

import PIL.Image
import safetensors.torch
import torch
import torch.nn.functional as F

from src.models.mask_geometry import MaskGeometryTokenizer, build_mask_geometry
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.utils.images import extract_foreground_from_image_and_mask
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_DILATED_SEGMENTATION,
    INSERT_ANYTHING_EXPANDED_BOX,
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
        pretrained_weight = getattr(self.config.model.lora, "pretrained_weight", None)
        if pretrained_weight is None:
            return None
        weight_dir = pretrained_weight if os.path.isdir(pretrained_weight) else os.path.dirname(pretrained_weight)
        return os.path.join(weight_dir, MASK_GEOMETRY_TOKENIZER_FILE_NAME)

    def load_model(self, text_encoder_device=None):
        super().load_model(text_encoder_device=text_encoder_device)
        if not self._use_geometric_mask_control():
            return

        cfg = self._mask_control_cfg()
        self.dit.mask_geometry_tokenizer = MaskGeometryTokenizer(
            hidden_channels=int(cfg.tokenizer_hidden_channels),
            out_channels=int(self.dit.config.in_channels),
        ).to(dtype=self.weight_dtype)

    def setup_model_device_train_mode(self, stage="fit"):
        super().setup_model_device_train_mode(stage=stage)
        if stage != "fit" or not self._use_geometric_mask_control():
            return

        tokenizer = self._get_mask_geometry_tokenizer()
        if tokenizer is None:
            raise ValueError("geometric mask control is enabled but mask_geometry_tokenizer is not attached")
        tokenizer.requires_grad_(True)

    def save_additional_modules(self, save_folder):
        super().save_additional_modules(save_folder)
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
        if not self._use_geometric_mask_control():
            return

        tokenizer = self._get_mask_geometry_tokenizer()
        if tokenizer is None:
            raise ValueError("geometric mask control is enabled but mask_geometry_tokenizer is not attached")

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
        triggered = metadata["context_guard_triggered"].detach().cpu()
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
        if precomputed_mask is not None:
            condition_mask = self._normalize_precomputed_condition_mask(
                condition_mask=precomputed_mask,
                tar_mask=tar_mask,
                hybrid_cfg=hybrid_cfg,
            )
            modes = ["precomputed"] * int(tar_mask.shape[0])
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
        )
        condition_mask, guard_metadata = apply_hybrid_masking_context_guard(
            condition_masks=condition_mask,
            gt_masks=tar_mask,
            config=hybrid_cfg,
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
                "hybrid/sampled_condition_expansion_mean": float(
                    gathered[:, 12].mean().item()
                ),
                "hybrid/sampled_condition_expansion_max": float(
                    gathered[:, 12].max().item()
                ),
                "hybrid/condition_expansion_mean": float(
                    gathered[:, 13].mean().item()
                ),
                "hybrid/condition_expansion_p90": float(
                    torch.quantile(gathered[:, 13], 0.90).item()
                ),
                "hybrid/condition_expansion_max": float(
                    gathered[:, 13].max().item()
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
        return self._use_reference_cfg() and bool(self._reference_cfg_cfg().force_empty_prompt)

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

        settings = ensure_latent_anchoring_config(self.config)
        requested_mode = kwargs.get("latent_anchoring_mode")
        mode = normalize_latent_anchoring_mode(
            settings.mode if requested_mode is None else requested_mode
        )
        if latent_anchoring_enabled(mode):
            if kwargs.get("height") is not None or kwargs.get("width") is not None:
                raise ValueError(
                    "latent anchoring does not yet support explicit height/width; "
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

        tar_mask_cond = self._build_condition_mask(batch, tar_mask, stage)
        source_hole_mask = batch.get("source_hole_mask", tar_mask_cond)
        if source_hole_mask.shape != tar_mask.shape:
            raise ValueError(
                "source_hole_mask must match tar_mask shape, got "
                f"{tuple(source_hole_mask.shape)} vs {tuple(tar_mask.shape)}"
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

        parent_batch = {
            "control": control,
            "prompt": prompt,
            "n_controls": 0,
        }
        if cache is not None:
            parent_batch["_opd_encoding_cache"] = cache
        if mask_geometry is not None:
            parent_batch["mask_geometry"] = mask_geometry

        if not drop_reference:
            self._append_additional_control(parent_batch, ref_image, vae_only=False)

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
                # optional identity weight is non-zero.
                parent_batch["target_image_for_loss"] = tar_image
                parent_batch["ref_image_for_loss"] = ref_image
                parent_batch["ref_mask_for_loss"] = ref_mask

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

        for key in ("height", "width", "num_inference_steps", "guidance"):
            if key in batch and not (stage == "opd" and key in {"height", "width"}):
                parent_batch[key] = batch[key]
        parent_batch["true_cfg_scale"] = batch.get(
            "true_cfg_scale", parent_batch["true_cfg_scale"]
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

        if anchoring_masks is not None:
            data.update(
                {
                    "source_hole_mask": anchoring_masks.free,
                    "anchor_source_image": src_tensor,
                    "anchor_mask_core": anchoring_masks.core,
                    "anchor_mask_free": anchoring_masks.free,
                    "anchor_edit_weight": anchoring_masks.edit_weight,
                    "latent_anchoring_mode": effective_anchoring_mode,
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

"""
Configuration module for Qwen Image Fine-tuning (Pydantic BaseModel version)
按 model、data、logging、optimizer、train 分块组织配置，支持 YAML 文件加载和验证
"""

import math
import os
from typing import Dict, Any, Optional, List, Literal, Union
import re
import torch
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    model_validator,
    field_serializer,
)

from pydantic import computed_field
from enum import Enum

# ----------------------------
# Common helpers / types
# ----------------------------

DeviceLike = Union[str, torch.device]


def _normalize_path(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = os.path.expanduser(os.path.expandvars(str(v)))
    if "://" in v:  # 保留如 s3://, http://
        scheme, rest = v.split("://", 1)
        rest = re.sub(r"/+", "/", rest)
        v = f"{scheme}://{rest}"
    else:
        v = re.sub(r"/+", "/", v)  # 将 '//' '///' 压成 '/'
    # 可选：去掉尾部斜杠（非根目录）
    if len(v) > 1:
        v = v.rstrip("/")
    return v


def _normalize_device(x: Optional[DeviceLike]) -> Optional[torch.device]:
    if x is None:
        return None
    d = torch.device(x)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA not available but got device={d}.")
    if d.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS not available but got device='mps'.")
    return d


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    vae: Optional[DeviceLike] = None
    vae_encoder: Optional[DeviceLike] = None
    text_encoder: Optional[DeviceLike] = None
    text_encoder_2: Optional[DeviceLike] = None
    dit: Optional[DeviceLike] = None

    # 设备规范化（字符串/torch.device -> torch.device）
    @field_validator(
        "vae",
        "vae_encoder",
        "text_encoder",
        "text_encoder_2",
        "dit",
        mode="after",
    )
    @classmethod
    def _norm(cls, v: Optional[DeviceLike]) -> Optional[torch.device]:
        return _normalize_device(v)

    @field_serializer(
        "vae",
        "text_encoder",
        "text_encoder_2",
        "dit",
        "vae_encoder",
        when_used="always",  # 只在 JSON 导出时生效；若想在 Python dict 也转字符串，用 "always"
    )
    def _ser_dev(self, v: Optional[DeviceLike]):
        return None if v is None else str(v)

    @model_validator(mode="after")
    def _wire_vae_device(self):
        if self.vae is None and self.vae_encoder is not None:
            self.vae = self.vae_encoder
        return self

# ----------------------------
# Image Processor
# ----------------------------


class ImageProcessorInitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    process_type: str = "center_crop"  # resize, _padding, center_crop
    resize_mode: str = "bilinear"
    target_size: Optional[List[int]] = None
    controls_size: Optional[Union[List[int], List[List[int]]]] = (
        None  # None -> use target_size
    )
    controls_pixels: Optional[Union[int, List[int]]] = None
    target_pixels: Optional[int] = None

    @field_validator("process_type")
    @classmethod
    def _check_process_type(cls, v: str) -> str:
        allowed = {"resize", "center_padding", "right_padding", "center_crop", "fixed_pixels"}
        if v not in allowed:
            raise ValueError(f"process_type must be one of {allowed}")
        return v

    # 解析像素表达式，例如 "512*512" -> 262144
    @staticmethod
    def _eval_pixel_expr(expr: str) -> int:
        s = str(expr).strip()
        # 仅允许 非负整数 或 形如 a*b 的简单乘法表达式，避免执行任意代码
        if re.fullmatch(r"\d+", s):
            return int(s)
        if re.fullmatch(r"\d+\s*\*\s*\d+", s):
            a, b = re.split(r"\*", s)
            return int(a.strip()) * int(b.strip())
        raise ValueError(f"Invalid pixel expression: {expr}")

    @field_validator("target_pixels", mode="before")
    @classmethod
    def _parse_target_pixels(cls, v):
        if v is None:
            return v
        if isinstance(v, (int,)):
            return int(v)
        if isinstance(v, str):
            return cls._eval_pixel_expr(v)
        raise ValueError(f"target_pixels must be int or string expression like '512*512', got {type(v)}")

    @field_validator("controls_pixels", mode="before")
    @classmethod
    def _parse_controls_pixels(cls, v):
        if v is None:
            return v
        # 允许单个整数/表达式，或列表形式
        if isinstance(v, (int,)):
            return int(v)
        if isinstance(v, str):
            return cls._eval_pixel_expr(v)
        if isinstance(v, list):
            parsed: List[int] = []
            for item in v:
                if isinstance(item, (int,)):
                    parsed.append(int(item))
                elif isinstance(item, str):
                    parsed.append(cls._eval_pixel_expr(item))
                else:
                    raise ValueError(
                        "controls_pixels list items must be int or string expression like '512*512'"
                    )
            return parsed
        raise ValueError(
            "controls_pixels must be int, string expression, or list of them"
        )


class ImageProcessorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "src.data.preprocess.ImageProcessor"
    init_args: ImageProcessorInitArgs = Field(default_factory=ImageProcessorInitArgs)


# ----------------------------
# Predict
# ----------------------------


class LatentAnchoringMode(str, Enum):
    legacy = "legacy"
    core = "core"
    expanded = "expanded"
    expanded_soft = "expanded_soft"


class LatentAnchoringConfig(BaseModel):
    """Inference-only A/B/C/D modes for post-step source-latent anchoring."""

    model_config = ConfigDict(extra="forbid")
    mode: LatentAnchoringMode = LatentAnchoringMode.legacy
    mask_threshold: float = 0.5
    interaction_dilation_px: int = 32
    transition_width_px: int = 32

    @field_validator("mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value):
        aliases = {
            "a": "legacy",
            "b": "core",
            "c": "expanded",
            "d": "expanded_soft",
        }
        normalized = str(getattr(value, "value", value)).strip().lower()
        return aliases.get(normalized, normalized)

    @field_validator("mask_threshold")
    @classmethod
    def _check_mask_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value >= 1:
            raise ValueError("predict.latent_anchoring.mask_threshold must be finite and in [0, 1)")
        return value

    @field_validator("interaction_dilation_px", "transition_width_px")
    @classmethod
    def _check_non_negative_width(cls, value: int) -> int:
        if value < 0:
            raise ValueError("predict.latent_anchoring pixel widths must be >= 0")
        return value

    @model_validator(mode="after")
    def _check_mode_geometry(self):
        if self.mode in {LatentAnchoringMode.expanded, LatentAnchoringMode.expanded_soft}:
            if self.interaction_dilation_px <= 0:
                raise ValueError(
                    f"predict.latent_anchoring.mode={self.mode.value} requires interaction_dilation_px > 0"
                )
        if self.mode == LatentAnchoringMode.expanded_soft and self.transition_width_px < 2:
            raise ValueError(
                "predict.latent_anchoring.mode=expanded_soft requires transition_width_px >= 2"
            )
        return self


class PredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    latent_anchoring: LatentAnchoringConfig = Field(default_factory=LatentAnchoringConfig)


# ----------------------------
# LoRA / Model
# ----------------------------


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = 16
    lora_alpha: int = 16
    init_lora_weights: str = "gaussian"  # 'gaussian' | 'normal' | 'zero'
    target_modules: Union[str, List[str]] = Field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )
    pretrained_weight: Optional[str] = None
    adapter_name: str = "default"

    @field_validator("r")
    @classmethod
    def _check_r(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("r must be > 0")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def _check_alpha(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("lora_alpha must be > 0")
        return v

    @field_validator("init_lora_weights")
    @classmethod
    def _check_init(cls, v: str) -> str:
        allowed = {"gaussian", "normal", "zero"}
        if v not in allowed:
            raise ValueError(f"init_lora_weights must be in {allowed}")
        return v

    @field_validator("target_modules")
    @classmethod
    def _check_target_modules(cls, v: Union[str, List[str]]):
        if isinstance(v, str) and not v:
            raise ValueError("target_modules must be non-empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("target_modules must be non-empty")
        return v

    @field_validator("pretrained_weight")
    @classmethod
    def _check_weight_path(cls, v: Optional[str]):
        if v is not None and not os.path.exists(v):
            raise ValueError(f"pretrained_weight path not found: {v}")
        return v

    @field_validator("adapter_name")
    @classmethod
    def _check_adapter_name(cls, v: Optional[str]):
        if v is not None and not isinstance(v, str):
            raise ValueError("adapter_name must be a string")
        return v


class MaskControlConfig(BaseModel):
    """Explicit placement-mask control as an additional VAE-only RGB image."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class ReferenceCFGConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    train_dropout_rate: float = 0.0
    force_empty_prompt: bool = False
    mask_aware: bool = False
    delta_clip_enabled: bool = False
    delta_clip_max_ratio: float = 0.5

    @field_validator("train_dropout_rate")
    @classmethod
    def _check_train_dropout_rate(cls, v: float) -> float:
        if v < 0 or v >= 1:
            raise ValueError("reference_cfg.train_dropout_rate must be in [0, 1)")
        return v

    @field_validator("delta_clip_max_ratio")
    @classmethod
    def _check_delta_clip_max_ratio(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("reference_cfg.delta_clip_max_ratio must be a positive finite value")
        return v


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pretrained_model_name_or_path: str = "Qwen/Qwen-Image-Edit"
    pretrained_embeddings: Optional[dict] = None  # if want to load different embeeding model vs main model (dit)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    mask_control: MaskControlConfig = Field(default_factory=MaskControlConfig)
    reference_cfg: ReferenceCFGConfig = Field(default_factory=ReferenceCFGConfig)
    quantize: bool = False

    @model_validator(mode="before")
    @classmethod
    def _handle_split_conditioning_modes(cls, value):
        """Accept disabled legacy fields while routing enabled modes to their worktrees."""

        if not isinstance(value, dict):
            return value

        retired_modes = {
            "query_prompt": "inpainting-v2-query-prompt",
            "pre_insertion": "inpainting-v2-layout-condition",
            "layout_condition": "inpainting-v2-layout-condition",
        }
        filtered = dict(value)
        for field_name, worktree_name in retired_modes.items():
            if field_name not in filtered:
                continue
            retired_config = filtered[field_name]
            if not isinstance(retired_config, dict):
                raise ValueError(
                    f"model.{field_name} was split out; legacy values must be mappings with enabled=false"
                )
            if retired_config.get("enabled", False) not in (False, 0, None):
                raise ValueError(
                    f"model.{field_name}.enabled=true requires the {worktree_name} worktree"
                )
            filtered.pop(field_name)
        return filtered


# ----------------------------
# Data
# ----------------------------


class AdaptiveCropConfig(BaseModel):
    """Shared target-space crop geometry for training and prediction."""

    model_config = ConfigDict(extra="forbid")
    train_enabled: bool = False
    predict_enabled: bool = False
    policy: str = "bbox_area_linear"
    trigger_bbox_area_ratio: float = 0.10
    min_crop_area_ratio: float = 0.25
    context_margin_ratio: float = 0.20
    mask_threshold: float = 0.5
    min_crop_side: int = 16
    aspect_mode: str = "target"
    blend_mode: str = "strict_mask"

    @field_validator("policy")
    @classmethod
    def _check_policy(cls, v: str) -> str:
        allowed = {"bbox_area_linear"}
        if v not in allowed:
            raise ValueError(f"adaptive_crop.policy must be one of {allowed}")
        return v

    @field_validator("aspect_mode")
    @classmethod
    def _check_aspect_mode(cls, v: str) -> str:
        allowed = {"target"}
        if v not in allowed:
            raise ValueError(f"adaptive_crop.aspect_mode must be one of {allowed}")
        return v

    @field_validator("blend_mode")
    @classmethod
    def _check_blend_mode(cls, v: str) -> str:
        allowed = {"strict_mask"}
        if v not in allowed:
            raise ValueError(f"adaptive_crop.blend_mode must be one of {allowed}")
        return v

    @field_validator("trigger_bbox_area_ratio", "min_crop_area_ratio")
    @classmethod
    def _check_open_closed_unit_interval(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0 or v > 1:
            raise ValueError("adaptive crop ratio fields must be in (0, 1]")
        return v

    @field_validator("mask_threshold")
    @classmethod
    def _check_mask_threshold(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0 or v >= 1:
            raise ValueError("adaptive_crop.mask_threshold must be finite and in [0, 1)")
        return v

    @field_validator("context_margin_ratio")
    @classmethod
    def _check_context_margin_ratio(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("adaptive_crop.context_margin_ratio must be finite and >= 0")
        return v

    @field_validator("min_crop_side")
    @classmethod
    def _check_positive_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("adaptive_crop.min_crop_side must be a positive integer")
        return v


class HybridMaskingContextGuardConfig(BaseModel):
    """Optional local-data guard against conditioning masks that erase scene context."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    max_condition_area_ratio: float = 0.80
    overflow_policy: str = "fallback_to_gt"

    @field_validator("max_condition_area_ratio")
    @classmethod
    def _check_max_condition_area_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0 or value >= 1:
            raise ValueError(
                "hybrid_masking.context_guard.max_condition_area_ratio must be finite and in (0, 1)"
            )
        return value

    @field_validator("overflow_policy")
    @classmethod
    def _check_overflow_policy(cls, value: str) -> str:
        allowed = {"fallback_to_gt"}
        if value not in allowed:
            raise ValueError(
                "hybrid_masking.context_guard.overflow_policy must be one of "
                f"{allowed}"
            )
        return value


class HybridMaskingConfig(BaseModel):
    """Training-only Hybrid Masking from the official Insert Anything pipeline."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    strategy: str = "insert_anything"
    person_categories: List[str] = Field(default_factory=lambda: ["person"])
    mask_threshold: float = 0.5
    bbox_expand_ratio: float = 1.1
    curve_jitter_px: int = 40
    dilation_kernel_size: int = 7
    dilation_iterations: int = 2
    reference_resolution: int = 768
    scale_pixel_parameters: bool = True
    ensure_gt_coverage: bool = True
    context_guard: HybridMaskingContextGuardConfig = Field(
        default_factory=HybridMaskingContextGuardConfig
    )

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, value: str) -> str:
        allowed = {"insert_anything"}
        if value not in allowed:
            raise ValueError(f"hybrid_masking.strategy must be one of {allowed}")
        return value

    @field_validator("person_categories")
    @classmethod
    def _normalize_person_categories(cls, value: List[str]) -> List[str]:
        normalized = []
        for category in value:
            category = str(category).strip()
            if not category:
                raise ValueError("hybrid_masking.person_categories cannot contain empty values")
            if category not in normalized:
                normalized.append(category)
        return normalized

    @field_validator("mask_threshold")
    @classmethod
    def _check_mask_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value >= 1:
            raise ValueError("hybrid_masking.mask_threshold must be finite and in [0, 1)")
        return value

    @field_validator("bbox_expand_ratio")
    @classmethod
    def _check_bbox_expand_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value < 1:
            raise ValueError("hybrid_masking.bbox_expand_ratio must be finite and >= 1")
        return value

    @field_validator("curve_jitter_px")
    @classmethod
    def _check_curve_jitter(cls, value: int) -> int:
        if value < 0:
            raise ValueError("hybrid_masking.curve_jitter_px must be >= 0")
        return value

    @field_validator("dilation_kernel_size")
    @classmethod
    def _check_dilation_kernel_size(cls, value: int) -> int:
        if value <= 0 or value % 2 == 0:
            raise ValueError("hybrid_masking.dilation_kernel_size must be a positive odd integer")
        return value

    @field_validator("dilation_iterations", "reference_resolution")
    @classmethod
    def _check_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("hybrid_masking dilation/reference integer fields must be positive")
        return value

    @model_validator(mode="after")
    def _check_context_guard_gt_coverage(self):
        if self.context_guard.enabled and not self.ensure_gt_coverage:
            raise ValueError(
                "hybrid_masking.context_guard requires ensure_gt_coverage=true "
                "to prevent completed-target pixel leakage"
            )
        return self


class DatasetInitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_path: Union[str, List, None] = None
    manifest_path: Optional[str] = None
    split: Optional[str] = None
    categories: Optional[Union[List[str], str]] = None
    require_prompt: bool = True
    use_edit_mask: bool = False
    selected_control_indexes: Optional[List[int]] = None
    processor: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)
    adaptive_crop: AdaptiveCropConfig = Field(default_factory=AdaptiveCropConfig)
    hybrid_masking: HybridMaskingConfig = Field(default_factory=HybridMaskingConfig)

    @field_validator("selected_control_indexes")
    @classmethod
    def _check_selected_control_indexes(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        if v is not None and any(index <= 0 for index in v):
            raise ValueError("selected_control_indexes must use 1-based positive indexes")
        return v

    @model_validator(mode="after")
    def _check_adaptive_crop_processor_compatibility(self):
        process_type = self.processor.init_args.process_type
        adaptive_crop_enabled = (
            self.adaptive_crop.train_enabled or self.adaptive_crop.predict_enabled
        )
        if adaptive_crop_enabled and process_type not in {"fixed_pixels", "resize"}:
            raise ValueError(
                "adaptive crop requires processor.process_type to be 'fixed_pixels' or 'resize': "
                "crop/padding processors cannot be inverted by the original-coordinate restore step"
            )
        return self


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "torch.utils.data.Dataset"
    init_args: DatasetInitArgs = Field(default_factory=DatasetInitArgs)
    batch_size: int = 1
    num_workers: int = 1
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: Optional[int] = 4
    non_blocking: bool = True
    shuffle: bool = True
    drop_last: bool = True

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, v: str) -> str:
        if not v:
            raise ValueError("class_path must be non-empty")
        return v

    @field_validator("batch_size")
    @classmethod
    def _check_pos_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive integer")
        return v

    @field_validator("num_workers")
    @classmethod
    def _check_non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("num_workers must be >= 0")
        return v

    @field_validator("prefetch_factor")
    @classmethod
    def _check_prefetch_factor(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("prefetch_factor must be positive or None")
        return v

    @model_validator(mode="after")
    def _check_adaptive_fixed_pixels_batching(self):
        if (
            self.init_args.adaptive_crop.train_enabled
            and self.init_args.processor.init_args.process_type == "fixed_pixels"
            and self.batch_size > 1
        ):
            raise ValueError(
                "adaptive crop with fixed_pixels currently requires data.batch_size=1: "
                "dynamic H/W padding has no valid-region loss mask"
            )
        if self.init_args.hybrid_masking.context_guard.enabled and self.batch_size > 1:
            raise ValueError(
                "hybrid masking context guard currently requires data.batch_size=1: "
                "padded variable-resolution masks do not carry a valid-pixel denominator"
            )
        return self


# ----------------------------
# Logging / Sampling
# ----------------------------


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = False
    validation_steps: int = 100
    num_samples: int = 4
    seed: int = 42
    num_inference_steps: int = 20
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: Optional[str] = None
    output_dir: str = "validation"
    panel_size: int = 320
    save_composite: bool = True
    validation_data: Optional[Union[str, List[Dict[str, str]]]] = None

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, value: int) -> int:
        if value < 0 or value >= 2**32:
            raise ValueError("sampling.seed must be in [0, 2**32)")
        return value

    @model_validator(mode="after")
    def _check_when_enabled(self):
        if self.enable:
            if self.validation_steps <= 0:
                raise ValueError(
                    "validation_steps must be positive when sampling is enabled"
                )
            if self.num_samples <= 0:
                raise ValueError(
                    "num_samples must be positive when sampling is enabled"
                )
            if self.num_inference_steps <= 0:
                raise ValueError(
                    "num_inference_steps must be positive when sampling is enabled"
                )
            if self.true_cfg_scale <= 0:
                raise ValueError(
                    "true_cfg_scale must be positive when sampling is enabled"
                )
            if self.guidance_scale <= 0:
                raise ValueError(
                    "guidance_scale must be positive when sampling is enabled"
                )
            if self.panel_size <= 0:
                raise ValueError(
                    "panel_size must be positive when sampling is enabled"
                )
        return self

    @field_validator("validation_data")
    @classmethod
    def _check_validation_data(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            required = {"ref_image", "ref_mask", "tar_image", "tar_mask"}
            for index, item in enumerate(v):
                if not isinstance(item, dict):
                    raise ValueError("Each validation_data item must be a dict")
                missing = sorted(required.difference(item))
                if missing:
                    raise ValueError(
                        f"validation_data item {index} is missing required keys: {missing}"
                    )
        return v


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./output"
    report_to: str = "tensorboard"  # tensorboard, wandb, all, none
    tracker_project_name: Optional[str] = None  # will get the value from trainer
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    @field_validator("report_to")
    @classmethod
    def _check_report_to(cls, v: str) -> str:
        allowed = {"tensorboard", "wandb", "all", "none"}
        if v not in allowed:
            raise ValueError(f"report_to must be one of {allowed}")
        return v

    @field_validator("output_dir")
    @classmethod
    def _check_output_dir(cls, v: str) -> str:
        return _normalize_path(v)


# ----------------------------
# LR Scheduler
# ----------------------------


class LRSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_type: str = "constant"
    warmup_steps: int = 0
    num_cycles: float = 0.5
    power: float = 1.0

    @field_validator("scheduler_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        valid = {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }
        if v not in valid:
            raise ValueError(f"scheduler_type must be one of {valid}")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def _check_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_steps must be >= 0")
        return v

    @field_validator("num_cycles", "power")
    @classmethod
    def _check_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v


# ----------------------------
# Optimizer
# ----------------------------


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "torch.optim.AdamW"
    init_args: Dict[str, Any] = Field(
        default_factory=lambda: {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
        }
    )

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, v: str) -> str:
        if not v:
            raise ValueError("class_path must be non-empty")
        return v

    @field_validator("init_args")
    @classmethod
    def _check_init_args(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if "lr" in v:
            lr = v["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"init_args.lr must be positive, got {lr}")
        if "weight_decay" in v:
            wd = v["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                raise ValueError(f"init_args.weight_decay must be >= 0, got {wd}")
        if "betas" in v:
            betas = v["betas"]
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError("init_args.betas must be a list/tuple of length 2")
            if not all(isinstance(b, (int, float)) and 0 <= b < 1 for b in betas):
                raise ValueError("each beta must be in [0, 1)")
        return v


# ----------------------------
# Train / Loss
# ----------------------------
class TrainerKind(str, Enum):
    QwenImageEdit = "QwenImageEdit"
    QwenImageEditPlus = "QwenImageEditPlus"
    QwenImageEditInsertAnything = "QwenImageEditInsertAnything"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    num_epochs: int = 3
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = 20
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # fp16 | bf16 | no
    gradient_checkpointing: bool = True
    low_memory: bool = False
    # 指定精细设备布局（低显存模式时生效）
    fit_device: Optional[DeviceConfig] = None

    @field_validator(
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "num_epochs",
        "checkpointing_steps",
    )
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("checkpoints_total_limit")
    @classmethod
    def _check_total_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("checkpoints_total_limit must be positive or None")
        return v

    @field_validator("max_grad_norm")
    @classmethod
    def _check_grad_norm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_grad_norm must be positive")
        return v

    @field_validator("mixed_precision")
    @classmethod
    def _check_mp(cls, v: str) -> str:
        allowed = {"fp16", "bf16", "no"}
        if v not in allowed:
            raise ValueError(f"mixed_precision must be one of {allowed}")
        return v


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mask_loss: bool = False
    forground_weight: float = 2.0
    background_weight: float = 1.0
    # token_weighted: global sum(w*e)/sum(w); region_balanced: fixed-ratio mean_fg/mean_bg mix
    mask_loss_mode: Literal["token_weighted", "region_balanced"] = "token_weighted"

    @field_validator("forground_weight", "background_weight")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be > 0")
        return v

    @field_validator("mask_loss_mode")
    @classmethod
    def _mask_loss_mode(cls, v: str) -> str:
        allowed = {"token_weighted", "region_balanced"}
        if v not in allowed:
            raise ValueError(f"mask_loss_mode must be one of {allowed}")
        return v


# ----------------------------
# Root Config
# ----------------------------
class TrMode(str, Enum):
    fit = "fit"
    predict = "predict"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainer: TrainerKind = TrainerKind.QwenImageEdit
    resume: str | None = None
    mode: TrMode = TrMode.predict
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @computed_field  # type: ignore[prop-declared]
    @property
    def trainer_type(self) -> str:
        return self.trainer

    @computed_field  # type: ignore[prop-declared]
    @property
    def model_name(self) -> str:
        return self.model.pretrained_model_name_or_path

    @computed_field  # type: ignore[prop-declared]
    @property
    def lora_adapter_name(self) -> str:
        return self.model.lora.adapter_name

    @computed_field  # type: ignore[prop-declared]
    @property
    def lora_r(self) -> int:
        return self.model.lora.r

    @computed_field  # type: ignore[prop-declared]
    @property
    def lora_lora_alpha(self) -> int:
        return self.model.lora.lora_alpha

    @computed_field  # type: ignore[prop-declared]
    @property
    def target_size(self) -> Optional[List[int]]:
        return self.data.init_args.processor.init_args.target_size

    # 1) 纯计算，不读 computed 字段，不改状态
    def _compute_quantization_type(self) -> str:
        name = (self.model_name or "").lower()
        if "fp4" in name or "4bit" in name:
            return "pretrain_fp4"
        if "fp8" in name:  # 注意不要把 "8bit/int8" 当作 fp8
            return "pretrain_fp8"
        if bool(getattr(self.model, "quantize", False)):
            return "fp8_online"
        return "pretrain_fp16"

    # 2) computed_field 只读 —— 任何时候访问都不会改状态
    @computed_field  # type: ignore[prop-declared]
    @property
    def quantization_type(self) -> str:
        return self._compute_quantization_type()

    @model_validator(mode="after")
    def _wire_cross_defaults(self):
        self.train.train_batch_size = self.data.batch_size
        if self.quantization_type in {"pretrain_fp4", "pretrain_fp8", "pretrain_fp16"}:
            self.model.quantize = False
        return self


# ----------------------------
# I/O helpers
# ----------------------------


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    从 YAML 文件加载配置并验证
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        yaml_config = OmegaConf.load(yaml_path)
    except Exception as e:
        raise RuntimeError(f"Error loading YAML file {yaml_path}: {e}") from e

    data = OmegaConf.to_container(yaml_config, resolve=True) or {}

    # 直接一次性校验嵌套结构
    # 若希望保持你之前对 lora/rank 的手动对齐，这里已在 ModelConfig 的 model_validator 中处理
    config = Config.model_validate(data)
    return config

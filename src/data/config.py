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
from src.distillation.meanflow_config import FlowMapConfig, MeanFlowDMDConfig

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
        allowed = {"resize", "center_padding", "right_padding", "center_crop", "fixed_pixels", "aspect_pixels"}
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
    # Replacement tasks (e.g. head swap): start at the schedule step nearest this σ,
    # with the free region holding the source latent noised by the run's own noise,
    # so the old object's pose, scale and lighting carry over. None starts from noise.
    source_noise_start: Optional[float] = None
    # Pixel paste-back of the restored primary output of anchored or complementary-canvas runs.
    # crop_feather: the decoded crop, feathered over 8 px at its interior edges.
    # change_gated: the decoded output inside M_free, the exact source beyond transition_width_px outside it; across
    # the transition_width_px band on either side of the M_free boundary it fades to the source except where the
    # output's local mean colour changed (object, shadow).
    paste_back: Literal["crop_feather", "change_gated"] = "crop_feather"

    @field_validator("source_noise_start")
    @classmethod
    def _check_source_noise_start(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and (not math.isfinite(value) or value <= 0 or value >= 1):
            raise ValueError("predict.latent_anchoring.source_noise_start must be in (0, 1)")
        return value

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
        if self.paste_back == "change_gated" and self.transition_width_px < 2:
            raise ValueError(
                "predict.latent_anchoring.paste_back=change_gated requires transition_width_px >= 2"
            )
        if self.source_noise_start is not None and self.mode == LatentAnchoringMode.legacy:
            raise ValueError(
                "predict.latent_anchoring.source_noise_start needs the anchored source latent; "
                "use mode core, expanded or expanded_soft"
            )
        return self


def check_latent_anchoring_contract(canvas, anchoring: LatentAnchoringConfig) -> None:
    """Combinations of predict.latent_anchoring with the complementary canvas; rerun after CLI overrides."""

    canvas_enabled = canvas is not None and canvas.enabled
    if anchoring.paste_back == "change_gated" and anchoring.mode == LatentAnchoringMode.legacy and not canvas_enabled:
        raise ValueError(
            "predict.latent_anchoring.paste_back=change_gated pastes back the restored output of an anchored or "
            "complementary-canvas run; legacy runs deliver the strict M_core composite"
        )
    if not canvas_enabled or anchoring.mode == LatentAnchoringMode.legacy:
        return
    # Inside R, expanded/expanded_soft anchoring pulls the latent pixels outside M_free back to the source trajectory.
    if anchoring.mode == LatentAnchoringMode.core:
        raise ValueError(
            "model.complementary_canvas regenerates M_free (M_core dilated by interaction_dilation_px); "
            "anchor inside R with predict.latent_anchoring.mode expanded or expanded_soft"
        )
    if canvas.ring_tokens:
        raise ValueError("latent anchoring inside R does not combine with model.complementary_canvas.ring_tokens")
    if anchoring.source_noise_start is not None:
        raise ValueError(
            "model.complementary_canvas samples the whole schedule from noise; "
            "unset predict.latent_anchoring.source_noise_start"
        )


class SigmaHandoffConfig(BaseModel):
    """Inference hand-off between the σ-split layout and identity segments.

    While σ > τ the sampler uses the shuffled reference, the user's loose mask
    and no reference CFG. At the first step with σ <= τ the alpha head's
    region (thresholded, intersected with the user mask, then dilated by the
    interaction band) becomes the geometry condition, the latent-anchoring
    free region and the reference-CFG region, and the ordered reference returns.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # None uses model.sigma_split.tau.
    tau: Optional[float] = None
    alpha_threshold: float = 0.5
    # None uses predict.latent_anchoring.interaction_dilation_px.
    interaction_dilation_px: Optional[int] = Field(default=None, ge=0, le=256)

    @field_validator("tau", "alpha_threshold")
    @classmethod
    def _check_unit_interval(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and (not math.isfinite(value) or value <= 0 or value >= 1):
            raise ValueError("predict.sigma_handoff tau/alpha_threshold must be in (0, 1)")
        return value


class PredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    latent_anchoring: LatentAnchoringConfig = Field(default_factory=LatentAnchoringConfig)
    sigma_handoff: SigmaHandoffConfig = Field(default_factory=SigmaHandoffConfig)


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
        if v is not None and not os.path.isfile(v):
            raise ValueError(f"pretrained_weight path not found or is not a file: {v}")
        return v

    @field_validator("adapter_name")
    @classmethod
    def _check_adapter_name(cls, v: Optional[str]):
        if v is not None and not isinstance(v, str):
            raise ValueError("adapter_name must be a string")
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("adapter_name must be a non-empty string")
        # PEFT stores adapters in ModuleDict-like containers; dots are module
        # path separators and make add_adapter fail only much later. Keep the
        # name portable across PEFT and Diffusers serialization formats.
        if re.fullmatch(r"[A-Za-z0-9_-]+", v) is None:
            raise ValueError(
                "adapter_name may contain only letters, numbers, '_' and '-'"
            )
        return v


class MaskControlConfig(BaseModel):
    """Placement-mask conditioning with RGB-VAE or geometric token encoding."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    encoding: Literal["rgb_vae", "geometry_tokenizer"] = "rgb_vae"
    sigma_px: float = 16.0
    mask_threshold: float = 0.5
    tokenizer_hidden_channels: int = 32

    @field_validator("sigma_px")
    @classmethod
    def _check_sigma_px(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("mask_control.sigma_px must be a positive finite value")
        return value

    @field_validator("mask_threshold")
    @classmethod
    def _check_mask_threshold(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value >= 1:
            raise ValueError("mask_control.mask_threshold must be finite and in [0, 1)")
        return value

    @field_validator("tokenizer_hidden_channels")
    @classmethod
    def _check_tokenizer_hidden_channels(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("mask_control.tokenizer_hidden_channels must be positive")
        return value


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


class AlphaHeadConfig(BaseModel):
    """Token-level object-occupancy head on the final DiT target hidden states."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False


class ComplementaryCanvasConfig(BaseModel):
    """Generate only the condition-hole tokens R and keep the source outside R as clean context.

    The image stream is [x_t at R (frame 0), masked source outside R (frame 1),
    conditions...]; every token keeps its own grid RoPE coordinate. The final
    latent is the generated R scattered into the source latent; latent anchoring
    (expanded/expanded_soft, no ring) additionally pulls R's latent pixels outside
    M_free back to the source trajectory. ``isolate_conditions`` keeps the condition images from
    attending to the text and the target: with ``isolation=image`` each image
    attends only to itself, with ``isolation=group`` the conditions attend to
    each other. ``cache_conditions`` (inference only) then runs the conditions
    once and reuses their per-layer K/V at every sampling step.

    ``ring_tokens`` adds the three-zone ring: the band of grid tokens within that
    many tokens of R is generated after R with its own modulation group, starts
    from the noised source with σ capped at ``ring_sigma_max``, is supervised
    with ``ring_loss_weight`` and is feathered into the source latent at the end.
    ``text_as_condition`` modulates the text stream at t=0 and lets it attend only
    to text and conditions, so it is cached with them. ``anneal_steps`` (training)
    replaces hard isolation by a logit bias on the isolated queries' other keys,
    ramped from 0 to ``anneal_max_bias`` over that many optimizer steps.
    ``local_decode_margin_px`` makes the pixel losses decode only bbox(R and its
    ring) plus this margin; the far background there is the exact source.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    isolate_conditions: bool = False
    isolation: Literal["image", "group"] = "image"
    cache_conditions: bool = False
    ring_tokens: int = Field(default=0, ge=0, le=4)
    ring_sigma_max: float = Field(default=0.3, gt=0, le=1)
    ring_loss_weight: float = Field(default=0.1, ge=0)
    text_as_condition: bool = False
    anneal_steps: int = Field(default=0, ge=0)
    anneal_max_bias: float = Field(default=12.0, gt=0, le=100)
    local_decode_margin_px: Optional[int] = Field(default=None, ge=0, le=1024)

    @model_validator(mode="after")
    def _check_dependencies(self):
        changed = {
            "isolate_conditions": self.isolate_conditions,
            "cache_conditions": self.cache_conditions,
            "ring_tokens": self.ring_tokens > 0,
            "text_as_condition": self.text_as_condition,
            "local_decode_margin_px": self.local_decode_margin_px is not None,
        }
        if any(changed.values()) and not self.enabled:
            enabled = ", ".join(name for name, value in changed.items() if value)
            raise ValueError(f"model.complementary_canvas.{enabled} require enabled=true")
        if self.cache_conditions and not self.isolate_conditions:
            raise ValueError("model.complementary_canvas.cache_conditions requires isolate_conditions")
        if self.isolation != "image" and not self.isolate_conditions:
            raise ValueError("model.complementary_canvas.isolation=group requires isolate_conditions")
        if self.text_as_condition and not self.isolate_conditions:
            raise ValueError("model.complementary_canvas.text_as_condition requires isolate_conditions")
        if self.anneal_steps and not self.isolate_conditions:
            raise ValueError("model.complementary_canvas.anneal_steps requires isolate_conditions")
        return self


class SigmaSplitHighNoiseConfig(BaseModel):
    """Layout segment (σ > τ): the reference says what, never which pose."""

    model_config = ConfigDict(extra="forbid")
    shuffle_reference_tokens: bool = True
    flip_prob: float = 0.5
    rotation_deg: float = Field(default=15.0, ge=0, le=180)
    # Color jitter factors are sampled from [1 - x, 1 + x]; hue from [-x, x] turns.
    brightness: float = Field(default=0.3, ge=0, lt=1)
    contrast: float = Field(default=0.3, ge=0, lt=1)
    saturation: float = Field(default=0.3, ge=0, lt=1)
    hue: float = Field(default=0.05, ge=0, le=0.5)
    # "drop" keeps the reference out of the VL prompt; "augmented" shows the
    # same strongly augmented image to the VL encoder.
    vl_reference: Literal["drop", "augmented"] = "drop"

    @field_validator("flip_prob")
    @classmethod
    def _check_probability(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("sigma_split.high_noise.flip_prob must be in [0, 1]")
        return value


class SigmaSplitLowNoiseConfig(BaseModel):
    """Identity segment (σ <= τ): ordered reference with light photometric jitter."""

    model_config = ConfigDict(extra="forbid")
    exposure_ev: float = Field(default=0.25, ge=0, le=2)
    gamma: float = Field(default=0.1, ge=0, lt=1)
    saturation: float = Field(default=0.1, ge=0, lt=1)
    temperature: float = Field(default=0.05, ge=0, lt=1)


class SigmaSplitConfig(BaseModel):
    """Build the training condition from the sampled σ (one LoRA, two segments).

    σ > τ uses the loose mask and the shuffled, strongly augmented reference;
    σ <= τ uses the tight mask and the ordered reference. Inside
    [τ - overlap, τ + overlap] each sample picks one of the two at random.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    tau: float = 0.7
    overlap: float = 0.05
    high_noise: SigmaSplitHighNoiseConfig = Field(default_factory=SigmaSplitHighNoiseConfig)
    low_noise: SigmaSplitLowNoiseConfig = Field(default_factory=SigmaSplitLowNoiseConfig)

    @model_validator(mode="after")
    def _check_band(self):
        if not (math.isfinite(self.tau) and 0 < self.tau < 1):
            raise ValueError("model.sigma_split.tau must be in (0, 1)")
        if not (math.isfinite(self.overlap) and 0 <= self.overlap < min(self.tau, 1 - self.tau)):
            raise ValueError("model.sigma_split.overlap must be in [0, min(tau, 1 - tau))")
        return self


class ModelConfig(BaseModel):
    attention_backend: Literal["sdpa", "flash_attention_2"] = "flash_attention_2"
    # Legacy checkpoints used one timestep for every image token. New 2511
    # training configs explicitly enable clean-condition (t=0) modulation.
    zero_cond_t: bool = False
    model_config = ConfigDict(extra="forbid")
    pretrained_model_name_or_path: str = "Qwen/Qwen-Image-Edit"
    pretrained_embeddings: Optional[dict] = None  # if want to load different embeeding model vs main model (dit)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    mask_control: MaskControlConfig = Field(default_factory=MaskControlConfig)
    reference_cfg: ReferenceCFGConfig = Field(default_factory=ReferenceCFGConfig)
    flow_map: FlowMapConfig = Field(default_factory=FlowMapConfig)
    alpha_head: AlphaHeadConfig = Field(default_factory=AlphaHeadConfig)
    sigma_split: SigmaSplitConfig = Field(default_factory=SigmaSplitConfig)
    complementary_canvas: ComplementaryCanvasConfig = Field(default_factory=ComplementaryCanvasConfig)
    # Show the masked source image to the VL encoder (Picture 1); false keeps only the reference.
    vl_masked_source: bool = True
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
        allowed = {"fallback_to_gt", "fallback_to_band"}
        if value not in allowed:
            raise ValueError(
                "hybrid_masking.context_guard.overflow_policy must be one of "
                f"{allowed}"
            )
        return value


class MaskMixV2ModeWeightsConfig(BaseModel):
    """Sampling probabilities for the MaskMix-v2 mask families.

    ``loose_blob`` and ``oversized_box`` reuse the ``loose_placement`` geometry
    without σ-split routing: the dataset builds them before adaptive crop.
    """

    model_config = ConfigDict(extra="forbid")
    augmented_box: float = 0.10
    asymmetric_box: float = 0.10
    exact: float = 0.60
    random_morphology: float = 0.20
    loose_blob: float = 0.0
    oversized_box: float = 0.0

    @field_validator(
        "augmented_box",
        "asymmetric_box",
        "exact",
        "random_morphology",
        "loose_blob",
        "oversized_box",
    )
    @classmethod
    def _check_probability(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError("MaskMix-v2 mode weights must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def _check_probability_sum(self):
        total = (
            self.augmented_box
            + self.asymmetric_box
            + self.exact
            + self.random_morphology
            + self.loose_blob
            + self.oversized_box
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "hybrid_masking.maskmix_v2.mode_weights must sum to 1 "
                f"(got {total})"
            )
        return self


class MaskMixV2AsymmetricBoxConfig(BaseModel):
    """Independent per-side margin range relative to the tight-box extent."""

    model_config = ConfigDict(extra="forbid")
    margin_ratio_min: float = 0.0
    margin_ratio_max: float = 0.10

    @field_validator("margin_ratio_min", "margin_ratio_max")
    @classmethod
    def _check_margin_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(
                "MaskMix-v2 asymmetric-box margin ratios must be finite and in [0, 1]"
            )
        return value

    @model_validator(mode="after")
    def _check_margin_ratio_order(self):
        if self.margin_ratio_min > self.margin_ratio_max:
            raise ValueError(
                "hybrid_masking.maskmix_v2.asymmetric_box.margin_ratio_min "
                "must be <= margin_ratio_max"
            )
        return self


class MaskMixV2AugmentedBoxConfig(BaseModel):
    """Object-scale-aware Bezier jitter settings for augmented boxes."""

    model_config = ConfigDict(extra="forbid")
    jitter_object_scale_min: float = 0.05
    jitter_object_scale_max: float = 0.20
    jitter_min_px: int = 2
    jitter_max_px: int = 64

    @field_validator("jitter_object_scale_min", "jitter_object_scale_max")
    @classmethod
    def _check_jitter_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(
                "MaskMix-v2 augmented-box jitter ratios must be finite and in [0, 1]"
            )
        return value

    @field_validator("jitter_min_px", "jitter_max_px")
    @classmethod
    def _check_jitter_pixels(cls, value: int) -> int:
        if value < 0:
            raise ValueError("MaskMix-v2 augmented-box pixel clamps must be >= 0")
        return value

    @model_validator(mode="after")
    def _check_jitter_ranges(self):
        if self.jitter_object_scale_min > self.jitter_object_scale_max:
            raise ValueError(
                "hybrid_masking.maskmix_v2.augmented_box.jitter_object_scale_min "
                "must be <= jitter_object_scale_max"
            )
        if self.jitter_min_px > self.jitter_max_px:
            raise ValueError(
                "hybrid_masking.maskmix_v2.augmented_box.jitter_min_px "
                "must be <= jitter_max_px"
            )
        return self


class MaskMixV2MorphologyKernelWeightsConfig(BaseModel):
    """Sampling probabilities for morphology structuring-element shapes."""

    model_config = ConfigDict(extra="forbid")
    ellipse: float = 0.40
    rectangle: float = 0.20
    horizontal_ellipse: float = 0.20
    vertical_ellipse: float = 0.20

    @field_validator(
        "ellipse",
        "rectangle",
        "horizontal_ellipse",
        "vertical_ellipse",
    )
    @classmethod
    def _check_probability(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(
                "MaskMix-v2 morphology kernel weights must be finite and in [0, 1]"
            )
        return value

    @model_validator(mode="after")
    def _check_probability_sum(self):
        total = (
            self.ellipse
            + self.rectangle
            + self.horizontal_ellipse
            + self.vertical_ellipse
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "hybrid_masking.maskmix_v2.morphology.kernel_weights must sum "
                f"to 1 (got {total})"
            )
        return self


class MaskMixV2MorphologyConfig(BaseModel):
    """Object-scale-aware random dilation settings."""

    model_config = ConfigDict(extra="forbid")
    radius_object_scale_min: float = 0.02
    radius_object_scale_max: float = 0.12
    radius_min_px: int = 1
    radius_max_px: int = 32
    anisotropy_min: float = 1.5
    anisotropy_max: float = 3.0
    kernel_weights: MaskMixV2MorphologyKernelWeightsConfig = Field(
        default_factory=MaskMixV2MorphologyKernelWeightsConfig
    )

    @field_validator("radius_object_scale_min", "radius_object_scale_max")
    @classmethod
    def _check_radius_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(
                "MaskMix-v2 morphology radius ratios must be finite and in [0, 1]"
            )
        return value

    @field_validator("anisotropy_min", "anisotropy_max")
    @classmethod
    def _check_anisotropy(cls, value: float) -> float:
        if not math.isfinite(value) or value < 1:
            raise ValueError(
                "MaskMix-v2 morphology anisotropy values must be finite and >= 1"
            )
        return value

    @field_validator("radius_min_px", "radius_max_px")
    @classmethod
    def _check_radius_pixels(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("MaskMix-v2 morphology pixel clamps must be positive")
        return value

    @model_validator(mode="after")
    def _check_ranges(self):
        if self.radius_object_scale_min > self.radius_object_scale_max:
            raise ValueError(
                "hybrid_masking.maskmix_v2.morphology.radius_object_scale_min "
                "must be <= radius_object_scale_max"
            )
        if self.radius_min_px > self.radius_max_px:
            raise ValueError(
                "hybrid_masking.maskmix_v2.morphology.radius_min_px "
                "must be <= radius_max_px"
            )
        if self.anisotropy_min > self.anisotropy_max:
            raise ValueError(
                "hybrid_masking.maskmix_v2.morphology.anisotropy_min "
                "must be <= anisotropy_max"
            )
        return self


class MaskMixV2Config(BaseModel):
    """Configuration for the scale-aware, four-family MaskMix-v2 policy."""

    model_config = ConfigDict(extra="forbid")
    mode_weights: MaskMixV2ModeWeightsConfig = Field(
        default_factory=MaskMixV2ModeWeightsConfig
    )
    asymmetric_box: MaskMixV2AsymmetricBoxConfig = Field(
        default_factory=MaskMixV2AsymmetricBoxConfig
    )
    augmented_box: MaskMixV2AugmentedBoxConfig = Field(
        default_factory=MaskMixV2AugmentedBoxConfig
    )
    morphology: MaskMixV2MorphologyConfig = Field(
        default_factory=MaskMixV2MorphologyConfig
    )


def _check_distribution(name: str, values: Dict[str, float]) -> None:
    for key, value in values.items():
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"{name}.{key} must be finite and in [0, 1]")
    total = sum(values.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 1 (got {total})")


class LooseMaskModeWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    loose_blob: float = 0.6
    oversized_box: float = 0.4

    @model_validator(mode="after")
    def _check_weights(self):
        _check_distribution("hybrid_masking.loose_placement.mode_weights", self.model_dump())
        return self


class LooseBlobShapeWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ellipse: float = 0.25
    superellipse: float = 0.25
    spline_blob: float = 0.25
    brush: float = 0.25

    @model_validator(mode="after")
    def _check_weights(self):
        _check_distribution("hybrid_masking.loose_placement.blob_shape_weights", self.model_dump())
        return self


class TightMaskModeWeightsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exact: float = 0.5
    random_morphology: float = 0.5

    @model_validator(mode="after")
    def _check_weights(self):
        _check_distribution("hybrid_masking.loose_placement.tight_mode_weights", self.model_dump())
        return self


class LoosePlacementConfig(BaseModel):
    """σ-split placement masks sampled in the dataset, before adaptive crop.

    Each sample carries a loose region (high-noise layout segment) and, in the
    same crop, a tight exact/morphology mask (low-noise identity segment).
    The geometry fields also shape the MaskMix-v2 ``loose_blob``/``oversized_box``
    families when ``enabled`` is false.
    ``size_reference=bbox_geometric_mean`` scales both axes from sqrt(h*w) of
    the GT box so the mask aspect ratio is independent of the object's;
    ``bbox_axes`` scales each GT box side by its own factor.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    mode_weights: LooseMaskModeWeightsConfig = Field(default_factory=LooseMaskModeWeightsConfig)
    blob_shape_weights: LooseBlobShapeWeightsConfig = Field(default_factory=LooseBlobShapeWeightsConfig)
    scale_min: float = 1.2
    scale_max: float = 2.5
    size_reference: Literal["bbox_geometric_mean", "bbox_axes"] = "bbox_geometric_mean"
    bottom_align_prob: float = 0.7
    # Extra processed pixels beyond interaction_dilation_px that the outline must
    # cover, absorbing resize rounding so the trainer's interaction union is a no-op.
    coverage_margin_px: int = Field(default=2, ge=0, le=64)
    guard_shrink: float = 0.7
    guard_max_attempts: int = Field(default=8, ge=0, le=64)
    tight_mode_weights: TightMaskModeWeightsConfig = Field(default_factory=TightMaskModeWeightsConfig)

    @model_validator(mode="after")
    def _check_ranges(self):
        if not (math.isfinite(self.scale_min) and math.isfinite(self.scale_max)):
            raise ValueError("loose_placement scale range must be finite")
        if self.scale_min < 1 or self.scale_max < self.scale_min:
            raise ValueError("loose_placement requires 1 <= scale_min <= scale_max")
        if not 0 <= self.bottom_align_prob <= 1:
            raise ValueError("loose_placement.bottom_align_prob must be in [0, 1]")
        if not 0 < self.guard_shrink < 1:
            raise ValueError("loose_placement.guard_shrink must be in (0, 1)")
        return self


class HybridMaskingConfig(BaseModel):
    """Training-only Insert Anything or MaskMix-v2 mask augmentation."""

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
    maskmix_v2: MaskMixV2Config = Field(default_factory=MaskMixV2Config)
    context_guard: HybridMaskingContextGuardConfig = Field(
        default_factory=HybridMaskingContextGuardConfig
    )
    loose_placement: LoosePlacementConfig = Field(default_factory=LoosePlacementConfig)

    # Extra context removed around the supervised object, in processed pixels.
    interaction_dilation_px: int = Field(default=0, ge=0, le=128)
    # "union" adds dilate(GT, interaction_dilation_px) to the sampled mask; "grow" dilates the
    # sampled mask itself until it covers that band, so R's outline never follows the GT contour.
    interaction_context_mode: Literal["union", "grow"] = "union"

    @field_validator("strategy")
    @classmethod
    def _check_strategy(cls, value: str) -> str:
        allowed = {"insert_anything", "maskmix_v2"}
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
        if self.strategy == "maskmix_v2" and not self.ensure_gt_coverage:
            raise ValueError(
                "hybrid_masking.strategy=maskmix_v2 requires "
                "ensure_gt_coverage=true"
            )
        if self.context_guard.enabled and not self.ensure_gt_coverage:
            raise ValueError(
                "hybrid_masking.context_guard requires ensure_gt_coverage=true "
                "to prevent completed-target pixel leakage"
            )
        if self.loose_placement.enabled and not (self.enabled and self.strategy == "maskmix_v2"):
            raise ValueError(
                "hybrid_masking.loose_placement requires hybrid_masking.enabled=true "
                "and strategy=maskmix_v2 (its tight masks reuse the MaskMix-v2 families)"
            )
        weights = self.maskmix_v2.mode_weights
        if self.loose_placement.enabled and (weights.loose_blob > 0 or weights.oversized_box > 0):
            raise ValueError(
                "hybrid_masking.maskmix_v2.mode_weights.loose_blob/oversized_box are for training without "
                "σ-split; with loose_placement.enabled the σ-split routes already pair loose and tight masks"
            )
        return self


class LightingRangeConfig(BaseModel):
    """One global grade (see src/data/lighting_augmentation.py).

    Light colour moves along the Planckian locus by a Laplace(0, wb_mired_scale)
    shift in mired, truncated at wb_mired_max; 100 mired is about 21° of light
    colour. The default scale gives a median of ~4.4° and a p90 of ~15°, the
    eval-set gap between reference and target photos.
    """

    model_config = ConfigDict(extra="forbid")
    wb_mired_scale: float = Field(default=30.0, ge=0)
    wb_mired_max: float = Field(default=150.0, ge=0)
    tint_max: float = Field(default=0.03, ge=0, lt=1)
    exposure_ev_std: float = Field(default=1.0, ge=0)
    exposure_ev_max: float = Field(default=2.0, ge=0)
    gamma_max: float = Field(default=0.1, ge=0, lt=1)
    saturation_max: float = Field(default=0.1, ge=0, lt=1)


class LightingAugmentationConfig(BaseModel):
    """Break the "output lighting = reference lighting" shortcut of same-exposure pairs.

    Each sample re-lights the reference object (``reference_prob``), grades the
    whole target including GT (``scene_prob``), or keeps the pair. Edge spill,
    camera degradation and reference geometry are drawn independently. Identity
    losses keep comparing against the unaugmented reference.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    reference_prob: float = 0.5
    scene_prob: float = 0.25
    reference: LightingRangeConfig = Field(default_factory=LightingRangeConfig)
    scene: LightingRangeConfig = Field(default_factory=LightingRangeConfig)
    # Low-frequency shading inside the reference branch, as a log-gain amplitude.
    shading_prob: float = 0.5
    shading_strength: float = Field(default=0.3, ge=0, le=1)
    edge_prob: float = 0.5
    camera_reference_prob: float = 0.3
    camera_scene_prob: float = 0.2
    geometry_prob: float = 0.0
    geometry_rotation_deg: float = Field(default=8.0, ge=0, le=45)
    geometry_scale: float = Field(default=0.15, ge=0, lt=1)
    geometry_perspective: float = Field(default=0.05, ge=0, le=0.25)

    @field_validator(
        "reference_prob",
        "scene_prob",
        "shading_prob",
        "edge_prob",
        "camera_reference_prob",
        "camera_scene_prob",
        "geometry_prob",
    )
    @classmethod
    def _check_probability(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError("lighting_augmentation probabilities must be finite and in [0, 1]")
        return value

    @model_validator(mode="after")
    def _check_branch_probabilities(self):
        if self.reference_prob + self.scene_prob > 1.0 + 1e-6:
            raise ValueError("lighting_augmentation.reference_prob + scene_prob must be <= 1")
        return self


class CategoryConditioningConfig(BaseModel):
    """Preserve domain-specific teacher augmentations in a joint OPD dataset."""

    model_config = ConfigDict(extra="forbid")
    adaptive_crop: Optional[AdaptiveCropConfig] = None
    hybrid_masking: Optional[HybridMaskingConfig] = None


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
    lighting_augmentation: LightingAugmentationConfig = Field(default_factory=LightingAugmentationConfig)
    category_balanced_sampling: bool = False
    face_identity_supervision: bool = False
    category_conditioning: Dict[str, CategoryConditioningConfig] = Field(default_factory=dict)

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
        if adaptive_crop_enabled and process_type not in {"fixed_pixels", "aspect_pixels", "resize"}:
            raise ValueError(
                "adaptive crop requires processor.process_type to be 'fixed_pixels', 'aspect_pixels' or 'resize': "
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
        if self.init_args.category_conditioning and self.batch_size != 1:
            raise ValueError("category_conditioning requires data.batch_size=1")
        if (
            (self.init_args.adaptive_crop.train_enabled
             and self.init_args.processor.init_args.process_type == "fixed_pixels"
             or self.init_args.processor.init_args.process_type == "aspect_pixels")
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
    fixed_seed_across_steps: bool = False
    validate_on_start: bool = False
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
    seed: int = Field(default=42, ge=0, lt=2**32)
    rank_specific_rng: bool = True
    trainable_precision: Literal["float32", "model"] = "float32"
    offload_text_encoder: bool = False
    checkpoint_vae_decode: bool = False
    memory_policy: Literal["configured", "auto"] = "configured"
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

    # Optional pixel-space region loss.  The latent mask above always remains the
    # ground-truth edit mask; MaskMix only changes the conditioning/hole mask.
    region_perceptual: bool = False
    latent_mse_weight: float = 1.0
    fg_lpips_weight: float = 0.5
    bg_l1_weight: float = 0.5
    detail_lpips_weight: float = 0.0
    detail_num_patches: int = Field(default=4, ge=1, le=16)
    detail_patch_size: int = Field(default=256, ge=16, le=1024)
    boundary_l1_weight: float = 0.0
    interaction_radius_px: int = Field(default=0, ge=0, le=128)
    fg_metric: Literal["lpips", "none"] = "lpips"
    lpips_net: Literal["alex", "vgg", "squeeze"] = "alex"
    lpips_input_size: int = 256
    # Backward-compatible name used by the region-perceptual experiment branch.
    lpips_downsample: Optional[int] = None
    lpips_bbox_pad_ratio: float = 0.1
    mask_threshold: float = 0.5
    lpips_linear_path: Optional[str] = None
    lpips_torch_home: Optional[str] = None
    lpips_trunk_path: Optional[str] = None
    region_sigma_power: float = 2.0
    region_max_sigma: float = 1.0

    # Reference identity is deliberately independent from foreground LPIPS and
    # stays disabled by default because it needs an additional frozen encoder.
    ref_dino_weight: float = 0.0
    # cosine: 1 - cos(pred, ref). gt_hinge: relu(cos(GT, ref) - margin - cos(pred, ref)),
    # so matching the reference better than the real GT earns nothing.
    ref_dino_mode: Literal["cosine", "gt_hinge"] = "cosine"
    ref_dino_hinge_margin: float = Field(default=0.05, ge=0, le=2)
    face_dino_weight: float = 0.0
    dino_backend: Literal["transformers", "torch_hub"] = "transformers"
    dino_model_name: str = "facebook/dinov2-small"
    dino_repo_path: Optional[str] = None
    dino_checkpoint_path: Optional[str] = None
    dino_input_size: int = 224
    dino_bbox_pad_ratio: float = 0.1

    # Token-level occupancy supervision of model.alpha_head at every σ.
    alpha_bce_weight: float = 0.0
    alpha_dice_weight: float = 0.0

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

    @field_validator(
        "latent_mse_weight",
        "fg_lpips_weight",
        "bg_l1_weight",
        "ref_dino_weight",
        "face_dino_weight",
        "detail_lpips_weight",
        "boundary_l1_weight",
        "alpha_bce_weight",
        "alpha_dice_weight",
    )
    @classmethod
    def _non_negative_loss_weight(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value) or value < 0:
            raise ValueError("region loss weights must be finite and >= 0")
        return value

    @field_validator("lpips_input_size", "dino_input_size")
    @classmethod
    def _positive_region_input_size(cls, v: int) -> int:
        value = int(v)
        if value <= 0:
            raise ValueError("region metric input sizes must be positive")
        return value

    @field_validator("lpips_downsample")
    @classmethod
    def _positive_lpips_downsample(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        value = int(v)
        if value <= 0:
            raise ValueError("lpips_downsample must be positive when set")
        return value

    @field_validator("mask_threshold")
    @classmethod
    def _region_mask_threshold(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value) or value < 0 or value >= 1:
            raise ValueError("loss.mask_threshold must be finite and in [0, 1)")
        return value

    @field_validator("lpips_bbox_pad_ratio", "dino_bbox_pad_ratio")
    @classmethod
    def _non_negative_bbox_pad_ratio(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value) or value < 0:
            raise ValueError("region bbox pad ratios must be finite and >= 0")
        return value

    @field_validator("region_sigma_power")
    @classmethod
    def _non_negative_region_sigma_power(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value) or value < 0:
            raise ValueError("region_sigma_power must be finite and >= 0")
        return value

    @field_validator("region_max_sigma")
    @classmethod
    def _unit_interval_region_max_sigma(cls, v: float) -> float:
        value = float(v)
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError("region_max_sigma must be finite and in [0, 1]")
        return value

    @field_validator("dino_model_name")
    @classmethod
    def _non_empty_dino_model_name(cls, v: str) -> str:
        value = str(v).strip()
        if not value:
            raise ValueError("dino_model_name must be non-empty")
        return value

    @field_validator(
        "lpips_linear_path",
        "lpips_torch_home",
        "lpips_trunk_path",
        "dino_repo_path",
        "dino_checkpoint_path",
    )
    @classmethod
    def _normalize_region_optional_path(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_path(v)

    @model_validator(mode="after")
    def _check_region_perceptual_contract(self):
        if self.lpips_downsample is not None:
            self.lpips_input_size = int(self.lpips_downsample)

        if not self.region_perceptual:
            return self
        if not self.mask_loss:
            raise ValueError(
                "loss.region_perceptual=true requires loss.mask_loss=true"
            )

        if self.fg_metric == "none" and self.fg_lpips_weight > 0:
            raise ValueError(
                "loss.fg_lpips_weight must be 0 when loss.fg_metric='none'"
            )

        active_weights = [
            self.bg_l1_weight,
            self.ref_dino_weight,
            self.face_dino_weight,
            self.detail_lpips_weight,
            self.boundary_l1_weight,
        ]
        if self.fg_metric == "lpips":
            active_weights.append(self.fg_lpips_weight)
        if all(weight == 0 for weight in active_weights):
            raise ValueError(
                "loss.region_perceptual is enabled but all pixel-region loss weights are 0"
            )

        if self.boundary_l1_weight > 0 and self.interaction_radius_px == 0:
            raise ValueError("boundary_l1_weight requires interaction_radius_px > 0")
        lpips_active = (self.fg_metric == "lpips" and self.fg_lpips_weight > 0) or self.detail_lpips_weight > 0
        if lpips_active:
            if self.lpips_linear_path is not None and not os.path.isfile(self.lpips_linear_path):
                raise ValueError(f"lpips_linear_path not found: {self.lpips_linear_path}")
            if self.lpips_torch_home is not None and not os.path.isdir(self.lpips_torch_home):
                raise ValueError(f"lpips_torch_home not found: {self.lpips_torch_home}")
            if self.lpips_trunk_path is not None and not os.path.isfile(self.lpips_trunk_path):
                raise ValueError(f"lpips_trunk_path not found: {self.lpips_trunk_path}")
            if self.lpips_trunk_path is not None and self.lpips_torch_home is None:
                raise ValueError(
                    "loss.lpips_trunk_path requires loss.lpips_torch_home so the "
                    "torchvision trunk cache can be configured"
                )
            if self.lpips_torch_home is not None and self.lpips_trunk_path is None:
                trunk_name = {
                    "alex": "alexnet-owt-7be5be79.pth",
                    "vgg": "vgg16-397923af.pth",
                }.get(self.lpips_net)
                if trunk_name is not None:
                    trunk = os.path.join(
                        self.lpips_torch_home,
                        "hub",
                        "checkpoints",
                        trunk_name,
                    )
                    if not os.path.isfile(trunk):
                        raise ValueError(
                            f"LPIPS trunk missing for net={self.lpips_net!r}: {trunk}. "
                            "Set loss.lpips_trunk_path or install it under "
                            "loss.lpips_torch_home/hub/checkpoints/."
                        )

        if (self.ref_dino_weight > 0 or self.face_dino_weight > 0) and self.dino_backend == "torch_hub":
            hubconf = (
                os.path.join(self.dino_repo_path, "hubconf.py")
                if self.dino_repo_path is not None
                else None
            )
            if hubconf is None or not os.path.isfile(hubconf):
                raise ValueError(
                    f"dino_repo_path is not a local torch-hub repo: {self.dino_repo_path}"
                )
            if self.dino_checkpoint_path is None or not os.path.isfile(
                self.dino_checkpoint_path
            ):
                raise ValueError(
                    f"dino_checkpoint_path not found: {self.dino_checkpoint_path}"
                )
        return self


# ----------------------------
# OPD distillation
# ----------------------------


class PluginConfig(BaseModel):
    """Import path and constructor arguments for a replaceable OPD component."""

    model_config = ConfigDict(extra="forbid")
    class_path: str
    init_args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("plugin class_path must be non-empty")
        return value


class OPDGuidanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "reference_cfg", "text_cfg"] = "none"
    scale: float = 1.0
    prompt_mode: Literal["batch", "empty"] = "batch"
    # None means inherit the teacher checkpoint's policy, never the student's.
    mask_aware: Optional[bool] = None
    delta_clip_enabled: Optional[bool] = None
    delta_clip_max_ratio: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("scale")
    @classmethod
    def _check_scale(cls, value: float) -> float:
        if not math.isfinite(value) or value < 1:
            raise ValueError("OPD guidance scale must be finite and >= 1")
        return value

    @model_validator(mode="after")
    def _check_unguided_scale(self):
        if self.mode == "none" and self.scale != 1:
            raise ValueError("OPD guidance mode='none' requires scale=1")
        return self


class OPDTeacherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    route_categories: List[str]
    lora: Optional[LoraConfig] = None
    mask_geometry_pretrained_weight: Optional[str] = None
    weight: float = 1.0
    guidance: OPDGuidanceConfig = Field(default_factory=OPDGuidanceConfig)
    policy: Optional[PluginConfig] = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("OPD teacher name must be non-empty")
        return value

    @field_validator("route_categories")
    @classmethod
    def _check_route_categories(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("OPD teacher route_categories must be non-empty")

        normalized = []
        for value in values:
            value = value.strip()
            if not value:
                raise ValueError("OPD teacher route_categories cannot contain empty values")
            if value in normalized:
                raise ValueError("OPD teacher route_categories must contain unique values")
            normalized.append(value)
        return normalized

    @field_validator("mask_geometry_pretrained_weight")
    @classmethod
    def _check_mask_geometry_weight(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = _normalize_path(value)
        if not os.path.isfile(value):
            raise ValueError(f"mask geometry pretrained weight not found: {value}")
        return value

    @field_validator("weight")
    @classmethod
    def _check_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("OPD teacher weight must be finite and > 0")
        return value


class OPDRolloutConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_inference_steps: int = 28
    timestep_fraction: float = 1.0
    store_device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 0

    @field_validator("num_inference_steps")
    @classmethod
    def _check_num_inference_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("OPD rollout num_inference_steps must be > 0")
        return value

    @field_validator("timestep_fraction")
    @classmethod
    def _check_timestep_fraction(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0 or value > 1:
            raise ValueError("OPD rollout timestep_fraction must be finite and in (0, 1]")
        return value


class DistillationConfig(BaseModel):
    """Declarative OPD composition; concrete components are imported at runtime."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    meanflow: MeanFlowDMDConfig | None = None
    algorithm: PluginConfig = Field(
        default_factory=lambda: PluginConfig(class_path="src.distillation.opd.QwenImageEditOPDStrategy")
    )
    objective: PluginConfig = Field(
        default_factory=lambda: PluginConfig(class_path="src.distillation.objectives.VelocityMatchingObjective")
    )
    teacher_backend: PluginConfig = Field(
        default_factory=lambda: PluginConfig(class_path="src.distillation.backends.SharedLoRATeacherBackend")
    )
    rollout: OPDRolloutConfig = Field(default_factory=OPDRolloutConfig)
    student_guidance: OPDGuidanceConfig = Field(default_factory=OPDGuidanceConfig)
    teachers: List[OPDTeacherConfig] = Field(default_factory=list)
    fallback_teacher: Optional[str] = None
    balance_tasks: bool = True

    @field_validator("fallback_teacher")
    @classmethod
    def _check_fallback_teacher(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("fallback_teacher must be a non-empty teacher name")
        return value

    @model_validator(mode="after")
    def _check_enabled_teacher_graph(self):
        if not self.enabled:
            return self
        if not self.teachers:
            raise ValueError("enabled OPD distillation requires at least one teacher")

        teacher_names = set()
        adapter_names = set()
        routed_categories: Dict[str, str] = {}
        shared_lora_backend = self.teacher_backend.class_path == "src.distillation.backends.SharedLoRATeacherBackend"
        independent_backend = self.teacher_backend.class_path == "src.distillation.backends.IndependentTeacherBackend"
        for teacher in self.teachers:
            if teacher.name in teacher_names:
                raise ValueError(f"duplicate OPD teacher name: {teacher.name}")
            teacher_names.add(teacher.name)

            if shared_lora_backend and teacher.lora is None:
                raise ValueError(f"OPD teacher {teacher.name!r} requires lora when using SharedLoRATeacherBackend")
            if shared_lora_backend and teacher.policy is not None:
                raise ValueError(
                    f"OPD teacher {teacher.name!r} cannot use policy with "
                    "SharedLoRATeacherBackend; select IndependentTeacherBackend"
                )
            if independent_backend and teacher.policy is None:
                raise ValueError(
                    f"OPD teacher {teacher.name!r} requires policy when using IndependentTeacherBackend"
                )

            if teacher.lora is not None:
                adapter_name = teacher.lora.adapter_name.strip()
                if not adapter_name:
                    raise ValueError(f"OPD teacher {teacher.name!r} must use a non-empty LoRA adapter_name")
                if adapter_name in adapter_names:
                    raise ValueError(f"duplicate OPD teacher LoRA adapter_name: {adapter_name}")
                adapter_names.add(adapter_name)

                checkpoint = teacher.lora.pretrained_weight
                if shared_lora_backend and not checkpoint:
                    raise ValueError(f"OPD teacher {teacher.name!r} requires lora.pretrained_weight")

            for category in teacher.route_categories:
                owner = routed_categories.get(category)
                if owner is not None:
                    raise ValueError(
                        f"OPD route category {category!r} is assigned to both {owner!r} and {teacher.name!r}"
                    )
                routed_categories[category] = teacher.name

        if self.fallback_teacher is not None and self.fallback_teacher not in teacher_names:
            raise ValueError(f"fallback_teacher {self.fallback_teacher!r} does not name a configured teacher")
        return self


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
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)

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

    def _check_sigma_split_contract(self) -> None:
        insert_anything = self.trainer == TrainerKind.QwenImageEditInsertAnything
        alpha_supervised = self.loss.alpha_bce_weight > 0 or self.loss.alpha_dice_weight > 0
        if alpha_supervised and not self.model.alpha_head.enabled:
            raise ValueError("loss.alpha_*_weight > 0 requires model.alpha_head.enabled")
        if self.model.alpha_head.enabled and not insert_anything:
            raise ValueError("model.alpha_head is implemented for trainer=QwenImageEditInsertAnything")
        if self.loss.ref_dino_mode == "gt_hinge" and self.loss.ref_dino_weight > 0 and not self.loss.region_perceptual:
            raise ValueError("loss.ref_dino_mode=gt_hinge is computed by loss.region_perceptual")

        sigma_split = self.model.sigma_split
        loose_enabled = self.data.init_args.hybrid_masking.loose_placement.enabled
        if loose_enabled != sigma_split.enabled:
            raise ValueError(
                "hybrid_masking.loose_placement.enabled and model.sigma_split.enabled must match: "
                "loose masks feed the σ > τ route and only σ-split training consumes them"
            )
        if sigma_split.enabled:
            if not insert_anything:
                raise ValueError("model.sigma_split requires trainer=QwenImageEditInsertAnything")
            if self.data.batch_size != 1:
                raise ValueError("model.sigma_split routes the VL prompt per batch and requires data.batch_size=1")
            if self.distillation.enabled or self.model.flow_map.enabled:
                raise ValueError("model.sigma_split is a teacher-training path; disable distillation/flow_map")

        handoff = self.predict.sigma_handoff
        if handoff.enabled and not (sigma_split.enabled and self.model.alpha_head.enabled):
            raise ValueError("predict.sigma_handoff requires model.sigma_split and model.alpha_head")

    def _check_complementary_canvas_contract(self) -> None:
        canvas = self.model.complementary_canvas
        if not canvas.enabled:
            return
        if self.trainer != TrainerKind.QwenImageEditInsertAnything:
            raise ValueError("model.complementary_canvas requires trainer=QwenImageEditInsertAnything")
        if self.data.batch_size != 1:
            raise ValueError("model.complementary_canvas generates a per-sample token set and requires data.batch_size=1")
        if canvas.cache_conditions and not self.model.zero_cond_t:
            raise ValueError("model.complementary_canvas.cache_conditions requires model.zero_cond_t")
        if (canvas.ring_tokens or canvas.text_as_condition) and not self.model.zero_cond_t:
            raise ValueError(
                "model.complementary_canvas.ring_tokens/text_as_condition need model.zero_cond_t (per-group timesteps)"
            )
        unsupported = {
            "model.sigma_split": self.model.sigma_split.enabled,
            "model.alpha_head": self.model.alpha_head.enabled,
            "model.flow_map": self.model.flow_map.enabled,
            "distillation": self.distillation.enabled,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError(f"model.complementary_canvas is not implemented together with {', '.join(enabled)}")

    @model_validator(mode="after")
    def _wire_cross_defaults(self):
        if self.model.flow_map.enabled and self.trainer != TrainerKind.QwenImageEditInsertAnything:
            raise ValueError("Flow-map conditioning is implemented for the InsertAnything trainer")
        if self.distillation.enabled and self.distillation.meanflow is not None:
            if not self.model.flow_map.enabled:
                raise ValueError("MeanFlow/DMD requires model.flow_map.enabled")
            if self.trainer != TrainerKind.QwenImageEditInsertAnything:
                raise ValueError("MeanFlow/DMD requires the InsertAnything trainer")
            if self.distillation.algorithm.class_path != "src.distillation.meanflow.QwenImageEditMeanFlowDMDStrategy":
                raise ValueError("meanflow options require QwenImageEditMeanFlowDMDStrategy")
            if len(self.distillation.teachers) != 1:
                raise ValueError("MeanFlow/DMD requires exactly one frozen fused teacher")
            if self.distillation.teacher_backend.class_path != "src.distillation.backends.SharedLoRATeacherBackend":
                raise ValueError("MeanFlow/DMD currently requires the shared LoRA backend")
            if self.distillation.teachers[0].guidance.mode != "none":
                raise ValueError("Use a fused, guidance-absorbed teacher with guidance.mode=none")
            if self.distillation.teachers[0].guidance.prompt_mode != self.distillation.student_guidance.prompt_mode:
                raise ValueError("Teacher and generator must use the same prompt conditioning")
            if self.distillation.student_guidance.mode != "none" or self.model.reference_cfg.enabled:
                raise ValueError("MeanFlow/DMD generator uses a single positive-reference branch")
            if self.train.mixed_precision == "fp16":
                raise ValueError("MeanFlow/DMD supports bf16 or no mixed precision, not fp16")
            if self.train.trainable_precision != "float32":
                raise ValueError("MeanFlow/DMD requires FP32 trainable parameters and optimizer moments")
            if self.distillation.meanflow.fake_adapter in {
                self.model.lora.adapter_name, self.distillation.teachers[0].lora.adapter_name
            }:
                raise ValueError("fake, generator and teacher adapters must be distinct")
        if self.distillation.enabled and self.distillation.algorithm.class_path == "src.distillation.meanflow.QwenImageEditMeanFlowDMDStrategy":
            if self.distillation.meanflow is None:
                raise ValueError("MeanFlow/DMD strategy requires distillation.meanflow")
        if self.loss.face_dino_weight > 0 and not self.data.init_args.face_identity_supervision:
            raise ValueError("face_dino_weight requires data.init_args.face_identity_supervision")
        self._check_sigma_split_contract()
        self._check_complementary_canvas_contract()
        check_latent_anchoring_contract(self.model.complementary_canvas, self.predict.latent_anchoring)
        self.train.train_batch_size = self.data.batch_size
        if self.quantization_type in {"pretrain_fp4", "pretrain_fp8", "pretrain_fp16"}:
            self.model.quantize = False
        if (
            self.model.mask_control.enabled
            and self.model.mask_control.encoding == "geometry_tokenizer"
            and self.trainer != TrainerKind.QwenImageEditInsertAnything
        ):
            raise ValueError(
                "mask_control.encoding=geometry_tokenizer is only implemented for "
                "trainer=QwenImageEditInsertAnything"
            )

        if self.distillation.enabled:
            built_in_strategy = (
                self.distillation.algorithm.class_path == "src.distillation.opd.QwenImageEditOPDStrategy"
            )
            if built_in_strategy and self.trainer != TrainerKind.QwenImageEditInsertAnything:
                raise ValueError(
                    "QwenImageEditOPDStrategy requires trainer=QwenImageEditInsertAnything "
                    "so stage='opd' cannot encode the ground-truth target latent"
                )

            shared_lora_backend = (
                self.distillation.teacher_backend.class_path
                == "src.distillation.backends.SharedLoRATeacherBackend"
            )
            if shared_lora_backend:
                student_adapter = self.model.lora.adapter_name
                collisions = [
                    teacher.name
                    for teacher in self.distillation.teachers
                    if teacher.lora is not None and teacher.lora.adapter_name == student_adapter
                ]
                if collisions:
                    raise ValueError(
                        f"OPD Teacher adapters {collisions} reuse Student adapter name {student_adapter!r}"
                    )

                geometry_enabled = (
                    self.model.mask_control.enabled
                    and self.model.mask_control.encoding == "geometry_tokenizer"
                )
                if geometry_enabled:
                    missing_companions = []
                    for teacher in self.distillation.teachers:
                        companion = teacher.mask_geometry_pretrained_weight
                        if companion is None and teacher.lora is not None:
                            lora_weight = teacher.lora.pretrained_weight
                            if lora_weight:
                                weight_dir = lora_weight if os.path.isdir(lora_weight) else os.path.dirname(lora_weight)
                                candidate = os.path.join(weight_dir, "mask_geometry_tokenizer.safetensors")
                                if os.path.isfile(candidate):
                                    companion = candidate
                        if companion is None:
                            missing_companions.append(teacher.name)
                    if missing_companions:
                        raise ValueError(
                            "geometry-tokenized SharedLoRA OPD requires each Teacher checkpoint "
                            "to include mask_geometry_tokenizer.safetensors (or set "
                            "mask_geometry_pretrained_weight); missing for "
                            f"{missing_companions}"
                        )
        return self


# ----------------------------
# I/O helpers
# ----------------------------


def load_config_from_yaml(yaml_path: str, *, for_inference: bool = False) -> Config:
    """
    从 YAML 文件加载配置并验证。

    ``distillation`` is a training-only dependency graph.  Inference callers
    may opt into ``for_inference`` so a portable Student config does not try to
    validate or import moved Teacher checkpoint paths.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        yaml_config = OmegaConf.load(yaml_path)
    except Exception as e:
        raise RuntimeError(f"Error loading YAML file {yaml_path}: {e}") from e

    data = OmegaConf.to_container(yaml_config, resolve=True) or {}
    if isinstance(data, dict):
        # Earlier train_config snapshots included read-only computed fields.
        for name in Config.model_computed_fields:
            data.pop(name, None)

    if for_inference and isinstance(data, dict):
        distillation = data.get("distillation")
        if isinstance(distillation, dict) and bool(distillation.get("enabled", False)):
            data["distillation"] = {"enabled": False}

    # 直接一次性校验嵌套结构
    # 若希望保持你之前对 lora/rank 的手动对齐，这里已在 ModelConfig 的 model_validator 中处理
    config = Config.model_validate(data)
    return config

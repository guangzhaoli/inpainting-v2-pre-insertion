import copy
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.latent_anchoring import ensure_latent_anchoring_config


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _dump_model(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump_model(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            str(key): _dump_model(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return value


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _normalized_target_modules(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    return value


def validate_inference_mode_inputs(
    config,
    *,
    prompt: Optional[str],
    negative_prompt: Optional[str],
) -> None:
    query_prompt_enabled = bool(config.model.query_prompt.enabled)
    pre_insertion_enabled = bool(config.model.pre_insertion.enabled)
    enhanced_mask_prompt = query_prompt_enabled or pre_insertion_enabled
    reference_cfg_enabled = bool(config.model.reference_cfg.enabled)
    reference_cfg_force_empty_prompt = (
        reference_cfg_enabled and bool(config.model.reference_cfg.force_empty_prompt)
    )

    if reference_cfg_enabled and enhanced_mask_prompt:
        raise ValueError(
            "reference_cfg requires query_prompt.enabled=false and pre_insertion.enabled=false"
        )
    if reference_cfg_enabled and negative_prompt is not None:
        raise ValueError("reference_cfg cannot be combined with negative_prompt")
    if enhanced_mask_prompt and negative_prompt is not None:
        raise ValueError("query_prompt/pre_insertion inference cannot use negative_prompt")
    if prompt is None and not enhanced_mask_prompt and not reference_cfg_force_empty_prompt:
        raise ValueError("prompt is required when no prompt-free conditioning mode is enabled")
    # mask_control is independent and may be combined with reference_cfg; no extra input rules.


def resolve_guidance_embeds_enabled(config, trainer=None) -> Optional[bool]:
    dit = getattr(trainer, "dit", None) if trainer is not None else None
    if dit is not None:
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            dit = accelerator.unwrap_model(dit)
        dit = getattr(dit, "_orig_mod", dit)
        dit_config = getattr(dit, "config", None)
        value = getattr(dit_config, "guidance_embeds", None)
        if value is not None:
            return bool(value)

    model_path = Path(str(config.model.pretrained_model_name_or_path)).expanduser()
    transformer_config_path = model_path / "transformer" / "config.json"
    if transformer_config_path.is_file():
        try:
            with transformer_config_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None
        value = payload.get("guidance_embeds")
        if value is not None:
            return bool(value)
    return None


def build_effective_inference_metadata(
    config,
    *,
    prompt: Optional[str],
    negative_prompt: Optional[str],
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
    guidance_embeds_enabled: Optional[bool] = None,
    requested_height: Optional[int] = None,
    requested_width: Optional[int] = None,
    effective_height: Optional[int] = None,
    effective_width: Optional[int] = None,
) -> dict:
    query_prompt_enabled = bool(config.model.query_prompt.enabled)
    pre_insertion_enabled = bool(config.model.pre_insertion.enabled)
    mask_control_enabled = bool(getattr(config.model, "mask_control", None) and config.model.mask_control.enabled)
    enhanced_mask_prompt = query_prompt_enabled or pre_insertion_enabled
    # Explicit mask_control is attached independently; enhanced paths still attach it too.
    mask_control_active = enhanced_mask_prompt or mask_control_enabled
    reference_cfg_enabled = bool(config.model.reference_cfg.enabled)
    reference_cfg_force_empty_prompt = (
        reference_cfg_enabled and bool(config.model.reference_cfg.force_empty_prompt)
    )

    effective_prompt = "" if enhanced_mask_prompt or reference_cfg_force_empty_prompt else prompt
    effective_negative_prompt = (
        None if enhanced_mask_prompt or reference_cfg_enabled else negative_prompt
    )
    text_true_cfg_active = (
        not enhanced_mask_prompt
        and not reference_cfg_enabled
        and effective_negative_prompt is not None
        and float(true_cfg_scale) > 1.0
    )
    reference_cfg_active = reference_cfg_enabled and float(true_cfg_scale) > 1.0
    effective_text_true_cfg_scale = float(true_cfg_scale) if text_true_cfg_active else 1.0
    effective_reference_cfg_scale = float(true_cfg_scale) if reference_cfg_active else 1.0
    if guidance_embeds_enabled is True:
        effective_guidance_scale = float(guidance_scale)
        guidance_scale_active = True
    elif guidance_embeds_enabled is False:
        effective_guidance_scale = 1.0
        guidance_scale_active = False
    else:
        effective_guidance_scale = None
        guidance_scale_active = None

    return {
        "requested": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": int(num_inference_steps),
            "true_cfg_scale": float(true_cfg_scale),
            "guidance_scale": float(guidance_scale),
            "height": requested_height,
            "width": requested_width,
        },
        "effective": {
            "prompt": effective_prompt,
            "negative_prompt": effective_negative_prompt,
            "num_inference_steps": int(num_inference_steps),
            "text_true_cfg_scale": effective_text_true_cfg_scale,
            "reference_cfg_scale": effective_reference_cfg_scale,
            "guidance_scale": effective_guidance_scale,
            "height": effective_height,
            "width": effective_width,
        },
        "modes": {
            "query_prompt_enabled": query_prompt_enabled,
            "pre_insertion_enabled": pre_insertion_enabled,
            "mask_control_enabled": mask_control_enabled,
            "mask_control_active": mask_control_active,
            "enhanced_mask_prompt": enhanced_mask_prompt,
            "reference_cfg_enabled": reference_cfg_enabled,
            "reference_cfg_force_empty_prompt": reference_cfg_force_empty_prompt,
            "text_true_cfg_active": text_true_cfg_active,
            "reference_cfg_active": reference_cfg_active,
            "guidance_embeds_enabled": guidance_embeds_enabled,
            "guidance_scale_active": guidance_scale_active,
        },
    }


def build_inference_config_metadata(config, config_path: Optional[Path] = None) -> dict:
    return {
        "path": str(config_path.resolve()) if config_path is not None else None,
        "trainer": str(_enum_value(getattr(config, "trainer", config.__class__.__name__))),
        "pretrained_model_name_or_path": str(config.model.pretrained_model_name_or_path),
        "quantize": bool(config.model.quantize),
        "processor": _dump_model(config.data.init_args.processor.init_args),
        "adaptive_crop": _dump_model(config.data.init_args.adaptive_crop),
        "latent_anchoring": _dump_model(ensure_latent_anchoring_config(config)),
        "model_modes": {
            "query_prompt": _dump_model(config.model.query_prompt),
            "pre_insertion": _dump_model(config.model.pre_insertion),
            "layout_condition": _dump_model(config.model.layout_condition),
            "mask_control": _dump_model(getattr(config.model, "mask_control", {"enabled": False})),
            "reference_cfg": _dump_model(config.model.reference_cfg),
        },
        "lora": {
            "r": int(config.model.lora.r),
            "lora_alpha": int(config.model.lora.lora_alpha),
            "target_modules": _dump_model(config.model.lora.target_modules),
            "adapter_name": str(config.model.lora.adapter_name),
            "pretrained_weight": config.model.lora.pretrained_weight,
        },
    }


def _checkpoint_contract(config, adaptive_enabled_field: str) -> dict:
    adaptive_crop = config.data.init_args.adaptive_crop.model_dump(mode="json")
    adaptive_enabled = bool(adaptive_crop.pop(adaptive_enabled_field))
    adaptive_crop.pop("train_enabled", None)
    adaptive_crop.pop("predict_enabled", None)
    adaptive_crop["enabled"] = adaptive_enabled

    return {
        "trainer": str(_enum_value(config.trainer)),
        "model": {
            "pretrained_model_name_or_path": str(config.model.pretrained_model_name_or_path),
            "quantize": bool(config.model.quantize),
            "lora": {
                "r": int(config.model.lora.r),
                "lora_alpha": int(config.model.lora.lora_alpha),
                "target_modules": _normalized_target_modules(config.model.lora.target_modules),
                "adapter_name": str(config.model.lora.adapter_name),
            },
            "query_prompt": config.model.query_prompt.model_dump(mode="json"),
            "pre_insertion": config.model.pre_insertion.model_dump(mode="json"),
            "layout_condition": config.model.layout_condition.model_dump(mode="json"),
            "mask_control": (
                config.model.mask_control.model_dump(mode="json")
                if getattr(config.model, "mask_control", None) is not None
                else {"enabled": False}
            ),
            "reference_cfg": config.model.reference_cfg.model_dump(mode="json"),
        },
        "data": {
            "class_path": str(config.data.class_path),
            "processor": config.data.init_args.processor.model_dump(mode="json"),
            "adaptive_crop": adaptive_crop,
        },
    }


def _flatten_contract_mismatches(checkpoint_value, inference_value, prefix="") -> list[dict]:
    if isinstance(checkpoint_value, dict) and isinstance(inference_value, dict):
        mismatches = []
        for key in sorted(set(checkpoint_value) | set(inference_value)):
            child_prefix = f"{prefix}.{key}" if prefix else key
            mismatches.extend(
                _flatten_contract_mismatches(
                    checkpoint_value.get(key, "<missing>"),
                    inference_value.get(key, "<missing>"),
                    child_prefix,
                )
            )
        return mismatches
    if checkpoint_value == inference_value:
        return []
    return [
        {
            "field": prefix,
            "checkpoint": checkpoint_value,
            "inference": inference_value,
        }
    ]


def find_checkpoint_train_config(lora_path: Path, max_parent_levels: int = 1) -> Optional[Path]:
    directory = lora_path.parent if lora_path.is_file() else lora_path
    for _ in range(max_parent_levels + 1):
        candidate = directory / "train_config.yaml"
        if candidate.is_file():
            return candidate
        parent = directory.parent
        if parent == directory:
            break
        directory = parent
    return None


def _load_checkpoint_train_config(train_config_path: Path):
    from src.data.config import Config

    with train_config_path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise ValueError(f"checkpoint train config must be a mapping: {train_config_path}")

    filtered_config = {
        key: copy.deepcopy(value)
        for key, value in raw_config.items()
        if key in Config.model_fields
    }
    model_config = filtered_config.get("model")
    if isinstance(model_config, dict):
        lora_config = model_config.get("lora")
        if isinstance(lora_config, dict):
            lora_config["pretrained_weight"] = None
    # Checkpoint compatibility is device-independent. Force CPU devices so the
    # saved config can be validated on login/CI nodes without CUDA.
    filtered_config["predict"] = {"devices": {}}
    return Config.model_validate(filtered_config)


def validate_checkpoint_compatibility(
    config,
    lora_path: Optional[Path],
    *,
    allow_mismatch: bool = False,
) -> dict:
    if lora_path is None:
        return {"status": "base_model", "verified": False, "train_config": None, "mismatches": []}

    resolved_lora_path = lora_path.resolve()
    file_stat = resolved_lora_path.stat()
    result = {
        "lora_file": str(resolved_lora_path),
        "lora_size_bytes": int(file_stat.st_size),
        "lora_mtime_ns": int(file_stat.st_mtime_ns),
        "train_config": None,
        "verified": False,
        "status": "unverified",
        "mismatches": [],
    }
    train_config_path = find_checkpoint_train_config(resolved_lora_path)
    if train_config_path is None:
        if not allow_mismatch:
            raise ValueError(
                "Could not verify checkpoint compatibility because train_config.yaml was not found "
                f"near {resolved_lora_path}. Pass --allow-checkpoint-mismatch only for an intentional override."
            )
        result["status"] = "unverified_allowed"
        return result

    result["train_config"] = str(train_config_path.resolve())
    try:
        checkpoint_config = _load_checkpoint_train_config(train_config_path)
        checkpoint_contract = _checkpoint_contract(checkpoint_config, "train_enabled")
        inference_contract = _checkpoint_contract(config, "predict_enabled")
        mismatches = _flatten_contract_mismatches(checkpoint_contract, inference_contract)
    except Exception as error:
        if not allow_mismatch:
            raise ValueError(
                f"Failed to validate checkpoint train config {train_config_path}: {error}"
            ) from error
        result["status"] = "invalid_train_config_allowed"
        result["validation_error"] = str(error)
        return result

    result["mismatches"] = mismatches
    if mismatches:
        if not allow_mismatch:
            preview = "; ".join(
                f"{item['field']}: checkpoint={item['checkpoint']!r}, inference={item['inference']!r}"
                for item in mismatches[:5]
            )
            raise ValueError(
                "Checkpoint/config contract mismatch. "
                f"{preview}. Pass --allow-checkpoint-mismatch only for an intentional override."
            )
        result["status"] = "mismatch_allowed"
        return result

    result["status"] = "verified"
    result["verified"] = True
    return result

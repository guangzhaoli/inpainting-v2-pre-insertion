"""Teacher checkpoint contracts, checked before loading the large backbone."""

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.data.config import ImageProcessorConfig, MaskControlConfig, ReferenceCFGConfig
from src.utils.inference_metadata import (
    _effective_mask_control_contract,
    find_checkpoint_train_config,
)


def read_teacher_config(weight):
    path = Path(weight)
    config_path = find_checkpoint_train_config(path)
    if config_path is None:
        raise FileNotFoundError(
            f"OPD teacher {path} requires its checkpoint train_config.yaml"
        )
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"Invalid teacher metadata: {config_path}")
    return raw


def _base_identity(value):
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else str(value).rstrip("/")


def _targets(value):
    return sorted(value) if isinstance(value, list) else value


def validate_teacher_config(config, teacher):
    """Validate model/processor compatibility and resolve per-teacher CFG knobs.

    Sampling scale can intentionally differ from validation. Training-only
    augmentations are recorded separately; they do not change the latent space.
    """
    raw = read_teacher_config(teacher.lora.pretrained_weight)
    model = raw["model"]
    mismatches = []
    if getattr(config.distillation, "meanflow", None) is not None and model.get("flow_map", {}).get("enabled", False):
        mismatches.append("MeanFlow/DMD target score teacher must use the original single-time model")
    if bool(model.get("zero_cond_t", False)) != config.model.zero_cond_t:
        mismatches.append("model.zero_cond_t")
    if _base_identity(model["pretrained_model_name_or_path"]) != _base_identity(
        config.model.pretrained_model_name_or_path
    ):
        mismatches.append("model.pretrained_model_name_or_path")
    saved_lora = model.get("lora", {})
    for key in ("r", "lora_alpha", "target_modules"):
        saved, requested = saved_lora.get(key), getattr(teacher.lora, key)
        if key == "target_modules":
            saved, requested = _targets(saved), _targets(requested)
        if saved != requested:
            mismatches.append(
                f"lora.{key}: checkpoint={saved!r}, requested={requested!r}"
            )
    saved_mask = MaskControlConfig.model_validate(model.get("mask_control", {}))
    saved_contract = _effective_mask_control_contract(
        SimpleNamespace(model=SimpleNamespace(mask_control=saved_mask))
    )
    if saved_contract != _effective_mask_control_contract(config):
        mismatches.append("model.mask_control")
    saved_processor = ImageProcessorConfig.model_validate(
        raw.get("data", {}).get("init_args", {}).get("processor", {})
    )
    if saved_processor.model_dump() != config.data.init_args.processor.model_dump():
        mismatches.append("data.init_args.processor")
    reference = ReferenceCFGConfig.model_validate(model.get("reference_cfg", {}))
    if teacher.guidance.mode == "reference_cfg" and not reference.enabled:
        mismatches.append(
            "reference_cfg teacher was not trained with reference dropout"
        )
    if mismatches:
        raise ValueError(
            f"OPD teacher {teacher.name!r} checkpoint contract mismatch: "
            + "; ".join(mismatches)
        )
    for key in ("mask_aware", "delta_clip_enabled", "delta_clip_max_ratio"):
        if getattr(teacher.guidance, key) is None:
            setattr(teacher.guidance, key, getattr(reference, key))
    return raw


def preflight_teacher_checkpoints(config):
    distillation = config.distillation
    if (
        not distillation.enabled
        or distillation.teacher_backend.class_path
        != "src.distillation.backends.SharedLoRATeacherBackend"
    ):
        return {}
    return {
        teacher.name: validate_teacher_config(config, teacher)
        for teacher in distillation.teachers
    }

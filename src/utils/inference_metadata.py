import copy
import json
from pathlib import Path
from typing import Any, Optional

import yaml

from src.models.alpha_head import ALPHA_HEAD_FILE_NAME as ALPHA_HEAD_COMPANION_FILE_NAME
from src.utils.latent_anchoring import ensure_latent_anchoring_config
from src.utils.tools import calculate_sha256_file


CHECKPOINT_STATE_FILE_NAME = "state.json"
CHECKPOINT_ARTIFACT_MANIFEST_VERSION = 1
MASK_GEOMETRY_COMPANION_FILE_NAME = "mask_geometry_tokenizer.safetensors"
CHECKPOINT_COMPANION_FILE_NAMES = frozenset(
    {
        MASK_GEOMETRY_COMPANION_FILE_NAME,
        ALPHA_HEAD_COMPANION_FILE_NAME,
        "flow_map_time_embedding.safetensors",
        "fake_score_lora.safetensors",
    }
)


class UnsupportedSplitConditioningCheckpointError(ValueError):
    """Raised when a checkpoint needs conditioning code absent from this worktree."""


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


def resolve_effective_mask_control(config) -> dict:
    """Describe the placement-mask path the final trainer will execute."""

    mask_control = getattr(config.model, "mask_control", None)
    configured_enabled = bool(mask_control is not None and mask_control.enabled)
    configured_encoding = str(
        _enum_value(getattr(mask_control, "encoding", "rgb_vae"))
    )
    geometry_active = (
        configured_enabled and configured_encoding == "geometry_tokenizer"
    )
    return {
        "configured_enabled": configured_enabled,
        "configured_encoding": configured_encoding,
        "active": configured_enabled,
        "encoding": configured_encoding if configured_enabled else None,
        "geometry_active": geometry_active,
    }


def _effective_mask_control_contract(config) -> dict:
    """Return only mask-control fields that can affect the active runtime path."""

    effective = resolve_effective_mask_control(config)
    if not effective["active"]:
        return {"enabled": False}

    contract = {
        "enabled": True,
        "encoding": effective["encoding"],
    }
    if effective["geometry_active"]:
        mask_control = config.model.mask_control
        contract.update(
            {
                "sigma_px": float(mask_control.sigma_px),
                "mask_threshold": float(mask_control.mask_threshold),
                "tokenizer_hidden_channels": int(
                    mask_control.tokenizer_hidden_channels
                ),
            }
        )
    return contract


def _effective_complementary_canvas_contract(config) -> dict:
    """Fields that change what a checkpoint learned; condition caching is an exact inference-only speedup,
    and the ring loss weight, isolation annealing and local decode only shape training. Fields at their
    round-1 defaults are left out so older checkpoints keep their contract."""

    canvas = getattr(config.model, "complementary_canvas", None)
    if canvas is None or not canvas.enabled:
        return {"enabled": False}
    contract = {"enabled": True, "isolate_conditions": bool(canvas.isolate_conditions)}
    if canvas.isolate_conditions and canvas.isolation != "image":
        contract["isolation"] = str(canvas.isolation)
    if canvas.ring_tokens:
        contract.update(ring_tokens=int(canvas.ring_tokens), ring_sigma_max=float(canvas.ring_sigma_max))
    if canvas.text_as_condition:
        contract["text_as_condition"] = True
    return contract


def validate_inference_mode_inputs(
    config,
    *,
    prompt: Optional[str],
    negative_prompt: Optional[str],
) -> None:
    reference_cfg_enabled = bool(config.model.reference_cfg.enabled)
    flow_map_enabled = bool(getattr(getattr(config.model, "flow_map", None), "enabled", False))
    force_empty_prompt = (
        (reference_cfg_enabled or flow_map_enabled) and bool(config.model.reference_cfg.force_empty_prompt)
    )

    if reference_cfg_enabled and negative_prompt is not None:
        raise ValueError("reference_cfg cannot be combined with negative_prompt")
    if flow_map_enabled and negative_prompt is not None:
        raise ValueError("flow-map conditioning cannot be combined with negative_prompt")
    if prompt is None and not force_empty_prompt:
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
    effective_mask_control = resolve_effective_mask_control(config)
    mask_control_enabled = effective_mask_control["configured_enabled"]
    mask_control_active = effective_mask_control["active"]
    reference_cfg_enabled = bool(config.model.reference_cfg.enabled)
    reference_cfg_force_empty_prompt = (
        reference_cfg_enabled and bool(config.model.reference_cfg.force_empty_prompt)
    )

    flow_map_force_empty_prompt = (
        bool(getattr(getattr(config.model, "flow_map", None), "enabled", False))
        and bool(config.model.reference_cfg.force_empty_prompt)
    )
    effective_prompt = "" if reference_cfg_force_empty_prompt or flow_map_force_empty_prompt else prompt
    effective_negative_prompt = None if reference_cfg_enabled else negative_prompt
    text_true_cfg_active = (
        not reference_cfg_enabled
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
            "complementary_canvas": _effective_complementary_canvas_contract(config),
            "mask_control_enabled": mask_control_enabled,
            "mask_control_active": mask_control_active,
            "mask_control_encoding": effective_mask_control["encoding"],
            "mask_control_configured_encoding": effective_mask_control[
                "configured_encoding"
            ],
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
        "zero_cond_t": bool(getattr(config.model, "zero_cond_t", False)),
        "processor": _dump_model(config.data.init_args.processor.init_args),
        "adaptive_crop": _dump_model(config.data.init_args.adaptive_crop),
        "latent_anchoring": _dump_model(ensure_latent_anchoring_config(config)),
        "sigma_handoff": _dump_model(
            getattr(getattr(config, "predict", None), "sigma_handoff", None) or {"enabled": False}
        ),
        "model_modes": {
            "mask_control": _dump_model(getattr(config.model, "mask_control", {"enabled": False})),
            "reference_cfg": _dump_model(config.model.reference_cfg),
            "complementary_canvas": _dump_model(
                getattr(config.model, "complementary_canvas", None) or {"enabled": False}
            ),
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
            "zero_cond_t": bool(getattr(config.model, "zero_cond_t", False)),
            "flow_map": (
                {"enabled": True, "source_time_weight": config.model.flow_map.source_time_weight}
                if getattr(getattr(config.model, "flow_map", None), "enabled", False)
                else {"enabled": False}
            ),
            "lora": {
                "r": int(config.model.lora.r),
                "lora_alpha": int(config.model.lora.lora_alpha),
                "target_modules": _normalized_target_modules(config.model.lora.target_modules),
                "adapter_name": str(config.model.lora.adapter_name),
            },
            "mask_control": _effective_mask_control_contract(config),
            "reference_cfg": config.model.reference_cfg.model_dump(mode="json"),
            "alpha_head": {"enabled": bool(getattr(getattr(config.model, "alpha_head", None), "enabled", False))},
            "complementary_canvas": _effective_complementary_canvas_contract(config),
            **({} if getattr(config.model, "vl_masked_source", True) else {"vl_masked_source": False}),
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
        retired_modes = {
            "query_prompt": "inpainting-v2-query-prompt",
            "pre_insertion": "inpainting-v2-layout-condition",
            "layout_condition": "inpainting-v2-layout-condition",
        }
        for field_name, worktree_name in retired_modes.items():
            retired_config = model_config.pop(field_name, None)
            if (
                isinstance(retired_config, dict)
                and retired_config.get("enabled", False) not in (False, 0, None)
            ):
                raise UnsupportedSplitConditioningCheckpointError(
                    f"checkpoint requires retired model.{field_name}; use the {worktree_name} worktree"
                )
        lora_config = model_config.get("lora")
        if isinstance(lora_config, dict):
            lora_config["pretrained_weight"] = None
    # OPD teachers are training-time dependencies. A deployed Student
    # checkpoint must remain verifiable after those large Teacher artifacts are
    # removed or moved to another machine.
    if "distillation" in filtered_config:
        filtered_config["distillation"] = {"enabled": False}
    # Checkpoint compatibility is device-independent. Force CPU devices so the
    # saved config can be validated on login/CI nodes without CUDA.
    filtered_config["predict"] = {"devices": {}}
    return Config.model_validate(filtered_config)


def validate_resume_mask_control_compatibility(config, checkpoint_dir: Path) -> dict:
    """Require an exact effective mask-control contract for true resume."""

    checkpoint_dir = Path(checkpoint_dir)
    train_config_path = find_checkpoint_train_config(
        checkpoint_dir,
        max_parent_levels=0,
    )
    if train_config_path is None:
        raise FileNotFoundError(
            f"Resume checkpoint is missing train_config.yaml: {checkpoint_dir}"
        )

    checkpoint_config = _load_checkpoint_train_config(train_config_path)
    saved_flow = getattr(checkpoint_config.model, "flow_map", None)
    active_flow = getattr(config.model, "flow_map", None)
    if bool(getattr(saved_flow, "enabled", False)) != bool(getattr(active_flow, "enabled", False)):
        raise ValueError("Resume checkpoint model.flow_map mismatch; use a warm start for stage conversion")
    if getattr(active_flow, "enabled", False) and saved_flow.source_time_weight != active_flow.source_time_weight:
        raise ValueError("Resume checkpoint flow-map time interpolation differs")
    if bool(getattr(checkpoint_config.model, "zero_cond_t", False)) != bool(getattr(config.model, "zero_cond_t", False)):
        raise ValueError("Resume checkpoint model.zero_cond_t mismatch; retain the checkpoint's condition timestep contract")
    if _effective_complementary_canvas_contract(checkpoint_config) != _effective_complementary_canvas_contract(config):
        raise ValueError(
            "Resume checkpoint model.complementary_canvas mismatch; use model.lora.pretrained_weight "
            "for an intentional warm start into another canvas mode"
        )
    checkpoint_contract = _effective_mask_control_contract(checkpoint_config)
    current_contract = _effective_mask_control_contract(config)
    mismatches = _flatten_contract_mismatches(
        checkpoint_contract,
        current_contract,
        prefix="model.mask_control",
    )
    if mismatches:
        preview = "; ".join(
            f"{item['field']}: checkpoint={item['checkpoint']!r}, "
            f"current={item['inference']!r}"
            for item in mismatches
        )
        raise ValueError(
            "Resume checkpoint mask-control contract mismatch. "
            f"{preview}. Use model.lora.pretrained_weight instead of resume "
            "for an intentional warm start."
        )
    return {
        "train_config": str(train_config_path.resolve()),
        "mask_control": current_contract,
        "mismatches": [],
    }


def _legacy_artifact_verification(state_path: Optional[Path] = None) -> dict:
    return {
        "status": "legacy_unverified",
        "verified": False,
        "state_file": (
            str(state_path.resolve()) if state_path is not None else None
        ),
        "files": [],
    }


def _required_checkpoint_artifacts(config, resolved_lora_path: Path) -> list[dict]:
    required = [
        {
            "filename": resolved_lora_path.name,
            "roles": {"lora_peft", "lora_diffusers", "lora"},
        }
    ]
    if resolve_effective_mask_control(config)["geometry_active"]:
        required.append(
            {
                "filename": MASK_GEOMETRY_COMPANION_FILE_NAME,
                "roles": {"mask_geometry_tokenizer"},
            }
        )
    if getattr(getattr(config.model, "alpha_head", None), "enabled", False):
        required.append({"filename": ALPHA_HEAD_COMPANION_FILE_NAME, "roles": {"alpha_head"}})
    if getattr(getattr(config.model, "flow_map", None), "enabled", False):
        required.append({"filename": "flow_map_time_embedding.safetensors", "roles": {"flow_map_time_embedding"}})
        options = getattr(getattr(config, "distillation", None), "meanflow", None)
        if options is not None and config.distillation.enabled:
            required.append({"filename": "meanflow_training_state.json", "roles": {"meanflow_training_state"}})
            if options.stage == "dmd":
                required.extend([
                    {"filename": "fake_score_lora.safetensors", "roles": {"fake_score_lora"}},
                    {"filename": "fake_score_optimizer.pt", "roles": {"fake_score_optimizer"}},
                ])
    return required


def validate_checkpoint_artifacts(
    config,
    resolved_lora_path: Path,
    *,
    allow_mismatch: bool,
) -> dict:
    """Verify the selected LoRA and active geometry companion via state.json."""

    resolved_lora_path = Path(resolved_lora_path).resolve()
    checkpoint_dir = resolved_lora_path.parent
    state_path = checkpoint_dir / CHECKPOINT_STATE_FILE_NAME
    if not state_path.is_file():
        return _legacy_artifact_verification()

    try:
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        if not allow_mismatch:
            raise ValueError(
                f"Failed to read checkpoint artifact manifest {state_path}: {error}"
            ) from error
        result = _legacy_artifact_verification(state_path)
        result["status"] = "invalid_manifest_allowed"
        result["validation_error"] = str(error)
        return result

    manifest = state.get("artifacts")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        return _legacy_artifact_verification(state_path)
    if manifest.get("version") != CHECKPOINT_ARTIFACT_MANIFEST_VERSION:
        message = (
            "Unsupported checkpoint artifact manifest version: "
            f"{manifest.get('version')!r}; expected "
            f"{CHECKPOINT_ARTIFACT_MANIFEST_VERSION}"
        )
        if not allow_mismatch:
            raise ValueError(message)
        return {
            "status": "unsupported_manifest_allowed",
            "verified": False,
            "state_file": str(state_path.resolve()),
            "manifest_version": manifest.get("version"),
            "files": [],
            "errors": [message],
        }

    entries_by_filename: dict[str, list[dict]] = {}
    malformed_entries = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("filename"), str):
            malformed_entries.append(entry)
            continue
        entries_by_filename.setdefault(entry["filename"], []).append(entry)

    errors = []
    verified_files = []
    if malformed_entries:
        errors.append(f"malformed manifest entries: {malformed_entries!r}")

    for required in _required_checkpoint_artifacts(config, resolved_lora_path):
        filename = required["filename"]
        matching_entries = entries_by_filename.get(filename, [])
        if len(matching_entries) != 1:
            errors.append(
                f"{filename}: expected exactly one manifest entry, "
                f"found {len(matching_entries)}"
            )
            continue

        entry = matching_entries[0]
        role = entry.get("role")
        if role not in required["roles"]:
            errors.append(
                f"{filename}: manifest role {role!r} is not one of "
                f"{sorted(required['roles'])}"
            )
            continue

        artifact_path = checkpoint_dir / filename
        if not artifact_path.is_file():
            errors.append(f"{filename}: artifact file is missing")
            continue

        expected_size = entry.get("size_bytes")
        expected_sha256 = entry.get("sha256")
        actual_size = artifact_path.stat().st_size
        if not isinstance(expected_size, int) or expected_size != actual_size:
            errors.append(
                f"{filename}: size mismatch, manifest={expected_size!r}, "
                f"actual={actual_size}"
            )
            continue
        if not isinstance(expected_sha256, str):
            errors.append(f"{filename}: manifest sha256 is missing")
            continue
        actual_sha256 = calculate_sha256_file(str(artifact_path))
        if expected_sha256.lower() != actual_sha256.lower():
            errors.append(
                f"{filename}: sha256 mismatch, manifest={expected_sha256}, "
                f"actual={actual_sha256}"
            )
            continue
        verified_files.append(
            {
                "role": role,
                "filename": filename,
                "size_bytes": actual_size,
                "sha256": actual_sha256,
            }
        )

    if errors:
        if not allow_mismatch:
            raise ValueError(
                "Checkpoint artifact integrity validation failed. "
                + "; ".join(errors)
            )
        return {
            "status": "mismatch_allowed",
            "verified": False,
            "state_file": str(state_path.resolve()),
            "manifest_version": manifest.get("version"),
            "files": verified_files,
            "errors": errors,
        }

    return {
        "status": "verified",
        "verified": True,
        "state_file": str(state_path.resolve()),
        "manifest_version": manifest.get("version"),
        "files": verified_files,
    }


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
    artifact_verification = validate_checkpoint_artifacts(
        config,
        resolved_lora_path,
        allow_mismatch=allow_mismatch,
    )
    result["artifact_verification"] = artifact_verification
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
    except UnsupportedSplitConditioningCheckpointError as error:
        raise ValueError(
            f"Failed to validate checkpoint train config {train_config_path}: {error}"
        ) from error
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

    result["verified"] = bool(artifact_verification["verified"])
    if result["verified"]:
        result["status"] = "verified"
    elif artifact_verification["status"] == "legacy_unverified":
        result["status"] = "config_verified_artifacts_unverified"
    else:
        result["status"] = f"artifact_{artifact_verification['status']}"
    return result

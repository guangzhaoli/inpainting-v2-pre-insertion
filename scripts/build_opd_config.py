"""Build a runnable OPD config from actual teacher checkpoint metadata."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.data.config import Config, OPDGuidanceConfig
from src.distillation.checkpoints import (
    preflight_teacher_checkpoints,
    read_teacher_config,
    validate_teacher_config,
)

ROOT = Path(__file__).resolve().parents[1]


def build_config(checkpoints, *, manifest=None, student_checkpoint=None):
    if not checkpoints:
        raise ValueError("at least one teacher checkpoint is required")
    teachers, categories, profiles = [], [], {}
    first = None
    for index, (name, checkpoint) in enumerate(checkpoints.items()):
        weight = Path(checkpoint).resolve()
        if weight.is_dir():
            weight /= "model.safetensors"
        if not weight.is_file():
            raise FileNotFoundError(weight)
        raw = read_teacher_config(weight)
        first = first or raw
        data = raw["data"]["init_args"]
        routes = data.get("categories")
        if not routes:
            raise ValueError(
                f"teacher {name!r} metadata requires explicit category routes"
            )
        for category in routes:
            if category in profiles:
                raise ValueError(f"overlapping teacher category: {category}")
            categories.append(category)
            profiles[category] = {
                key: copy.deepcopy(data[key])
                for key in ("adaptive_crop", "hybrid_masking")
                if key in data
            }
        reference = raw["model"].get("reference_cfg", {})
        if not reference.get("enabled"):
            raise ValueError(f"teacher {name!r} was not trained for reference CFG")
        lora = copy.deepcopy(raw["model"]["lora"])
        lora.update(adapter_name=f"opd_teacher_{index}", pretrained_weight=str(weight))
        guidance = dict(
            mode="reference_cfg",
            prompt_mode="empty",
            scale=raw.get("logging", {}).get("sampling", {}).get("true_cfg_scale", 2.5),
        )
        for key in ("mask_aware", "delta_clip_enabled", "delta_clip_max_ratio"):
            if key in reference:
                guidance[key] = reference[key]
        teachers.append(
            dict(
                name=name,
                route_categories=routes,
                lora=lora,
                guidance=guidance,
                weight=1.0,
            )
        )
    result = {
        key: copy.deepcopy(value)
        for key, value in first.items()
        if key in Config.model_fields
    }
    result.update(mode="fit", resume=None)
    result["model"]["attention_backend"] = "flash_attention_2"
    result["model"]["lora"].update(
        adapter_name="insertfuse_opd_student", pretrained_weight=None
    )
    if student_checkpoint:
        weight = Path(student_checkpoint).resolve()
        if weight.is_dir():
            weight /= "model.safetensors"
        saved = read_teacher_config(weight)
        for key in ("r", "lora_alpha", "target_modules"):
            if saved["model"]["lora"][key] != result["model"]["lora"][key]:
                raise ValueError(
                    f"student warm-start {key} does not match the selected architecture"
                )
        result["model"]["lora"]["pretrained_weight"] = str(weight)
    result["model"]["reference_cfg"] = dict(
        enabled=False, train_dropout_rate=0.0, force_empty_prompt=True
    )
    data = result["data"]["init_args"]
    data.update(
        categories=categories,
        category_conditioning=profiles,
        face_identity_supervision=False,
        category_balanced_sampling=False,
        manifest_path=str(Path(manifest).resolve())
        if manifest
        else str(ROOT / "manifests/insert_anything_teacher_all_data.csv"),
    )
    result["data"]["batch_size"] = 1
    result["train"].update(
        num_epochs=10,
        max_train_steps=100000000,
        gradient_accumulation_steps=len(teachers),
        trainable_precision="float32",
        memory_policy="auto",
    )
    result["loss"] = dict(mask_loss=False, region_perceptual=False)
    result["logging"].update(
        output_dir="logs/opd", tracker_project_name="insertfuse_teacher_v2_opd"
    )
    result["logging"]["sampling"]["enable"] = False
    result["distillation"] = dict(
        enabled=True,
        balance_tasks=True,
        rollout=dict(
            num_inference_steps=10, timestep_fraction=1.0, store_device="cuda", seed=42
        ),
        student_guidance=dict(mode="none", scale=1.0, prompt_mode="empty"),
        teachers=teachers,
    )
    validated = Config.model_validate(result)
    preflight_teacher_checkpoints(validated)
    if student_checkpoint:
        validate_teacher_config(
            validated,
            SimpleNamespace(
                name="student warm start",
                lora=validated.model.lora,
                guidance=OPDGuidanceConfig(),
            ),
        )
        if (
            validated.model.mask_control.enabled
            and validated.model.mask_control.encoding == "geometry_tokenizer"
        ):
            companion = (
                Path(validated.model.lora.pretrained_weight).parent
                / "mask_geometry_tokenizer.safetensors"
            )
            if not companion.is_file():
                raise FileNotFoundError(companion)
    return validated.model_dump(mode="json", exclude_none=True, round_trip=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher", action="append", required=True, metavar="NAME=CHECKPOINT"
    )
    parser.add_argument("--manifest")
    parser.add_argument("--student-checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {}
    for item in args.teacher:
        name, separator, checkpoint = item.partition("=")
        if not separator or not name or name in checkpoints:
            parser.error("each --teacher must have a unique NAME=CHECKPOINT")
        checkpoints[name] = checkpoint
    config = build_config(
        checkpoints, manifest=args.manifest, student_checkpoint=args.student_checkpoint
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    print(f"Validated {len(checkpoints)} teacher checkpoints; wrote {args.output}")


if __name__ == "__main__":
    main()

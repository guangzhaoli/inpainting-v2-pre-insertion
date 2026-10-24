"""Build a runnable config from the actual fused Teacher checkpoint contract."""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

from src.data.config import Config, ImageProcessorConfig
from src.distillation.checkpoints import (
    preflight_teacher_checkpoints,
    read_teacher_config,
)
from src.models.flow_map import FLOW_MAP_FILE
from src.utils.inference_metadata import validate_checkpoint_artifacts


def lora_file(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        for name in ("model.safetensors", "pytorch_lora_weights.safetensors"):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"No LoRA weights in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_config(
    teacher,
    *,
    stage="warmup",
    generator=None,
    student_steps=8,
    teacher_steps=40,
    rollout="full",
    gradient_mode="last",
    output_dir="logs/meanflow",
):
    teacher = lora_file(teacher)
    raw = read_teacher_config(teacher)
    teacher_training = raw.get("distillation", {})
    if teacher_training.get("enabled"):
        policy = teacher_training.get("student_guidance", {})
        if policy.get("mode", "none") != "none":
            raise ValueError(
                "The fused teacher must have a guidance-absorbed single-branch policy"
            )
        prompt_mode = policy.get("prompt_mode", "empty")
    else:
        prompt_mode = (
            "empty"
            if raw["model"].get("reference_cfg", {}).get("force_empty_prompt", True)
            else "batch"
        )
    if raw["model"].get("flow_map", {}).get("enabled", False):
        raise ValueError(
            "The target score teacher must be the original single-time fused model"
        )
    source = lora_file(generator) if generator else teacher
    source_raw = read_teacher_config(source)
    if stage == "dmd" and not source_raw["model"].get("flow_map", {}).get(
        "enabled", False
    ):
        raise ValueError(
            "DMD requires --generator pointing to a trained MeanFlow warmup checkpoint"
        )
    if stage == "dmd" and not (source.parent / FLOW_MAP_FILE).is_file():
        raise FileNotFoundError(source.parent / FLOW_MAP_FILE)
    # Ignore only adapter names and checkpoint paths; architecture/processor
    # differences must never be hidden by config generation.
    for key in ("pretrained_model_name_or_path", "zero_cond_t", "mask_control"):
        if source_raw["model"].get(key) != raw["model"].get(key):
            raise ValueError(f"Generator and teacher disagree on model.{key}")
    for key in ("r", "lora_alpha", "target_modules"):
        if source_raw["model"]["lora"].get(key) != raw["model"]["lora"].get(key):
            raise ValueError(f"Generator and teacher disagree on LoRA {key}")
    source_processor = ImageProcessorConfig.model_validate(
        source_raw["data"]["init_args"].get("processor", {})
    )
    teacher_processor = ImageProcessorConfig.model_validate(
        raw["data"]["init_args"].get("processor", {})
    )
    if source_processor != teacher_processor:
        raise ValueError("Generator and teacher processors differ")
    cfg = copy.deepcopy(raw)
    for key in Config.model_computed_fields:
        cfg.pop(key, None)
    cfg.update(mode="fit", trainer="QwenImageEditInsertAnything", resume=None)
    cfg["model"]["lora"].update(
        adapter_name="meanflow_generator", pretrained_weight=str(source)
    )
    cfg["model"]["flow_map"] = {
        "enabled": True,
        "source_time_weight": source_raw["model"]
        .get("flow_map", {})
        .get("source_time_weight", 0.25),
        "inference_steps": student_steps,
    }
    cfg["model"]["reference_cfg"] = dict(
        cfg["model"].get("reference_cfg", {}),
        enabled=False,
        force_empty_prompt=prompt_mode == "empty",
    )
    cfg.setdefault("predict", {})["devices"] = dict(
        vae="cpu", text_encoder="cpu", dit="cpu"
    )
    cfg["loss"] = {"mask_loss": False, "region_perceptual": False}
    cfg.setdefault("train", {}).update(
        mixed_precision="bf16", trainable_precision="float32", offload_text_encoder=True
    )
    cfg.setdefault("logging", {}).update(
        output_dir=output_dir, tracker_project_name=f"insertfuse_meanflow_{stage}"
    )
    sampling = cfg["logging"].setdefault("sampling", {})
    sampling.update(
        num_inference_steps=student_steps, true_cfg_scale=1.0, negative_prompt=None
    )
    teacher_lora = dict(
        raw["model"]["lora"],
        adapter_name="meanflow_fused_teacher",
        pretrained_weight=str(teacher),
    )
    cfg["distillation"] = {
        "enabled": True,
        "balance_tasks": False,
        "algorithm": {
            "class_path": "src.distillation.meanflow.QwenImageEditMeanFlowDMDStrategy"
        },
        "teacher_backend": {
            "class_path": "src.distillation.backends.SharedLoRATeacherBackend"
        },
        "student_guidance": {"mode": "none", "scale": 1.0, "prompt_mode": prompt_mode},
        "teachers": [
            {
                "name": "fused",
                "route_categories": cfg["data"]["init_args"].get("categories")
                or ["__fused_fallback__"],
                "lora": teacher_lora,
                "guidance": {"mode": "none", "scale": 1.0, "prompt_mode": prompt_mode},
            }
        ],
        "fallback_teacher": "fused",
        "meanflow": {
            "stage": stage,
            "teacher_steps": teacher_steps,
            "student_steps": student_steps,
            "rollout": rollout,
            "gradient_mode": gradient_mode,
        },
    }
    parsed = Config.model_validate(cfg)
    preflight_teacher_checkpoints(parsed)
    if stage == "dmd":
        deploy_raw = copy.deepcopy(source_raw)
        for key in Config.model_computed_fields:
            deploy_raw.pop(key, None)
        deploy_raw["distillation"] = {"enabled": False}
        deploy_raw.setdefault("predict", {})["devices"] = dict(
            vae="cpu", text_encoder="cpu", dit="cpu"
        )
        deploy_config = Config.model_validate(deploy_raw)
        validate_checkpoint_artifacts(deploy_config, source, allow_mismatch=False)
    return parsed.model_dump(mode="json", exclude=set(Config.model_computed_fields))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, type=Path)
    p.add_argument("--generator", type=Path)
    p.add_argument("--stage", choices=["warmup", "dmd"], required=True)
    p.add_argument("--student-steps", type=int, default=8)
    p.add_argument("--teacher-steps", type=int, default=40)
    p.add_argument("--rollout", choices=["full", "shortcut"], default="full")
    p.add_argument("--gradient-mode", choices=["full", "last"], default="last")
    p.add_argument("--output-dir", default="logs/meanflow")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = build_config(
        args.teacher,
        stage=args.stage,
        generator=args.generator,
        student_steps=args.student_steps,
        teacher_steps=args.teacher_steps,
        rollout=args.rollout,
        gradient_mode=args.gradient_mode,
        output_dir=args.output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

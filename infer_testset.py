#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from infer import (
    enhanced_mask_prompt_enabled,
    load_rgb_image,
    resolve_lora_path,
    save_latent_anchoring_masks,
    seed_everything,
    validate_adaptive_crop_inference_args,
    validate_configured_inference,
)
from src.utils.latent_anchoring import (
    add_latent_anchoring_cli_args,
    apply_latent_anchoring_cli_overrides,
    ensure_latent_anchoring_config,
    latent_anchoring_metadata,
    normalize_latent_anchoring_mode,
    validate_latent_anchoring_runtime,
)
from src.utils.visualization import (
    binarize_mask,
    build_strict_mask_composite,
    ensure_mask,
    ensure_rgb,
    make_infer_row,
    make_mask_overlay,
    save_mask,
    stack_rows,
)
from src.utils.adaptive_crop import (
    crop_pil_pair,
    restore_pil_output,
)
from src.utils.adaptive_crop_anchoring import (
    DEFAULT_CROP_EDGE_FEATHER_PX,
    prepare_adaptive_crop_for_inference,
)
from src.utils.inference_metadata import (
    build_effective_inference_metadata,
    build_inference_config_metadata,
    resolve_guidance_embeds_enabled,
    validate_checkpoint_compatibility,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "exp" / "accessory" / "adaptive_crop.yaml"
)
DEFAULT_TESTSET_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/testset_20260120"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "testset_adaptive_crop"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch inference for Insert Anything testset folders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="LoRA file or checkpoint directory. Its train_config.yaml is verified against --config.",
    )
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow a missing or incompatible checkpoint train_config.yaml. Unsafe unless intentional.",
    )
    parser.add_argument("--testset-root", type=Path, default=DEFAULT_TESTSET_ROOT)
    parser.add_argument("--category", type=str, default="accessory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=320, help="Image area size for per-sample row previews.")
    add_latent_anchoring_cli_args(parser)
    return parser.parse_args(argv)


def natural_key(path: Path):
    parts = []
    for token in path.stem.replace("_", "-").split("-"):
        parts.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return parts


def image_files(directory: Path):
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def find_stem_file(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        upper_candidate = directory / f"{stem}{suffix.upper()}"
        if upper_candidate.exists():
            return upper_candidate
    return None


def discover_samples(testset_root: Path, category: str, limit: int | None):
    category_dir = testset_root / category
    ref_image_dir = category_dir / "ref_image"
    ref_mask_dir = category_dir / "ref_mask"
    src_image_dir = category_dir / "tar_image"
    src_mask_dir = category_dir / "tar_mask"

    required_dirs = [ref_image_dir, ref_mask_dir, src_image_dir, src_mask_dir]
    missing_dirs = [str(path) for path in required_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing testset directories: {missing_dirs}")

    samples = []
    for ref_image_path in image_files(ref_image_dir):
        stem = ref_image_path.stem
        ref_mask_path = find_stem_file(ref_mask_dir, stem)
        src_image_path = find_stem_file(src_image_dir, stem)
        src_mask_path = find_stem_file(src_mask_dir, stem)
        missing = [
            name
            for name, path in (
                ("ref_mask", ref_mask_path),
                ("source", src_image_path),
                ("source_mask", src_mask_path),
            )
            if path is None
        ]
        if missing:
            raise FileNotFoundError(f"Sample {stem} is missing: {missing}")

        samples.append(
            {
                "stem": stem,
                "ref_image": ref_image_path,
                "ref_mask": ref_mask_path,
                "source": src_image_path,
                "source_mask": src_mask_path,
            }
        )
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise ValueError(f"No samples found for category {category} under {testset_root}")
    return samples


def setup_trainer(args):
    from src.data.config import load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.true_cfg_scale <= 0 or args.guidance_scale <= 0:
        raise ValueError("--true-cfg-scale and --guidance-scale must be positive")
    if args.panel_size <= 0:
        raise ValueError("--panel-size must be positive")
    limit = getattr(args, "limit", None)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if args.height is not None and (args.height <= 0 or args.width <= 0):
        raise ValueError("--height and --width must be positive")

    config = load_config_from_yaml(str(args.config))
    apply_latent_anchoring_cli_overrides(config, args)
    validate_adaptive_crop_inference_args(config, args)
    validate_latent_anchoring_runtime(config, height=args.height, width=args.width)
    resolved_lora_path = resolve_lora_path(args.lora)
    config.model.lora.pretrained_weight = str(resolved_lora_path)

    validation_args = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        allow_base_model=False,
    )
    validate_configured_inference(config, validation_args, resolved_lora_path)
    checkpoint_compatibility = validate_checkpoint_compatibility(
        config,
        resolved_lora_path,
        allow_mismatch=bool(getattr(args, "allow_checkpoint_mismatch", False)),
    )

    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()
    return trainer, config, resolved_lora_path, checkpoint_compatibility


def run_sample(
    trainer,
    sample,
    sample_dir: Path,
    args,
    sample_seed: int,
    resolved_lora_path: Path | None = None,
    checkpoint_compatibility: dict | None = None,
):
    seed_everything(sample_seed)

    ref_image = load_rgb_image(sample["ref_image"])
    ref_mask = load_rgb_image(sample["ref_mask"])
    src_image = load_rgb_image(sample["source"])
    src_mask = load_rgb_image(sample["source_mask"])

    original_src_image = ensure_rgb(src_image)
    original_src_mask = ensure_mask(src_mask)
    if original_src_image.size != original_src_mask.size:
        raise ValueError(
            "source image and source mask must have identical geometry, "
            f"got image={original_src_image.size}, mask={original_src_mask.size}"
        )
    adaptive_crop_config = trainer.config.data.init_args.adaptive_crop
    latent_anchoring_config = ensure_latent_anchoring_config(trainer.config)
    anchoring_metadata = latent_anchoring_metadata(latent_anchoring_config)
    anchoring_is_enabled = bool(anchoring_metadata["enabled"])
    composite_mask_threshold = (
        float(latent_anchoring_config.mask_threshold)
        if anchoring_is_enabled
        else float(adaptive_crop_config.mask_threshold)
    )
    crop_plan = None
    crop_compatibility = {"enabled": False}
    predict_src_image = original_src_image
    predict_src_mask = original_src_mask
    edge_feather_px = 0
    if adaptive_crop_config.predict_enabled:
        if args.height is not None or args.width is not None:
            raise ValueError("Adaptive Crop prediction cannot be combined with explicit --height/--width")
        crop_plan, crop_compatibility = prepare_adaptive_crop_for_inference(
            image_size=original_src_image.size,
            mask=original_src_mask,
            adaptive_crop_config=adaptive_crop_config,
            config=trainer.config,
            processor_config=trainer.config.data.init_args.processor,
        )
        predict_src_image, predict_src_mask = crop_pil_pair(
            original_src_image,
            original_src_mask,
            crop_plan,
        )
        if anchoring_is_enabled and crop_plan is not None and crop_plan.applied:
            edge_feather_px = int(
                crop_compatibility.get("feather_width_px", DEFAULT_CROP_EDGE_FEATHER_PX)
            )

    predict_kwargs = {
        "image": [predict_src_image, ref_image],
        "prompt": args.prompt,
        "ref_mask": ref_mask,
        "src_mask": predict_src_mask,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "latent_anchoring_mode": normalize_latent_anchoring_mode(latent_anchoring_config.mode),
        "latent_anchoring_mask_threshold": float(latent_anchoring_config.mask_threshold),
        "interaction_dilation_px": int(latent_anchoring_config.interaction_dilation_px),
        "transition_width_px": int(latent_anchoring_config.transition_width_px),
    }
    if args.height is not None:
        predict_kwargs["height"] = args.height
    if args.width is not None:
        predict_kwargs["width"] = args.width

    result = trainer.predict(**predict_kwargs)
    local_output = ensure_rgb(result[0])
    if crop_plan is not None:
        raw_restored_output, strict_composite = restore_pil_output(
            original_source=original_src_image,
            original_mask=original_src_mask,
            local_output=local_output,
            plan=crop_plan,
            blend_mode=adaptive_crop_config.blend_mode,
            edge_feather_px=0,
        )
        if edge_feather_px > 0:
            primary_restored_output, _ = restore_pil_output(
                original_source=original_src_image,
                original_mask=original_src_mask,
                local_output=local_output,
                plan=crop_plan,
                blend_mode=adaptive_crop_config.blend_mode,
                edge_feather_px=edge_feather_px,
            )
        else:
            primary_restored_output = raw_restored_output
    else:
        raw_restored_output = local_output.resize(
            original_src_image.size,
            Image.Resampling.LANCZOS,
        )
        primary_restored_output = raw_restored_output
        strict_composite = build_strict_mask_composite(
            original_src_image,
            original_src_mask,
            raw_restored_output,
            composite_mask_threshold,
        )
    primary_output = primary_restored_output if anchoring_is_enabled else strict_composite
    if (
        raw_restored_output.size != original_src_image.size
        or strict_composite.size != original_src_image.size
        or primary_output.size != original_src_image.size
    ):
        raise ValueError("restored and final outputs must match the original source size")

    sample_dir.mkdir(parents=True, exist_ok=True)
    ensure_rgb(ref_image).save(sample_dir / "ref_image.png")
    save_mask(ref_mask, sample_dir / "ref_mask.png")
    make_mask_overlay(ref_image, ref_mask).save(sample_dir / "ref_mask_overlay.png")

    ensure_rgb(original_src_image).save(sample_dir / "source.png")
    save_mask(original_src_mask, sample_dir / "source_mask.png")
    make_mask_overlay(original_src_image, original_src_mask).save(sample_dir / "source_mask_overlay.png")

    ensure_rgb(local_output).save(sample_dir / "output_local.png")
    ensure_rgb(raw_restored_output).save(sample_dir / "output_restored_raw.png")

    if crop_plan is not None:
        ensure_rgb(predict_src_image).save(sample_dir / "crop_source.png")
        save_mask(predict_src_mask, sample_dir / "crop_mask.png")

    ensure_rgb(primary_output).save(sample_dir / "output.png")
    output_mask = binarize_mask(
        original_src_mask,
        composite_mask_threshold,
    ).resize(primary_output.size, Image.Resampling.NEAREST)
    output_mask.save(sample_dir / "output_mask.png")
    make_mask_overlay(primary_output, output_mask).save(sample_dir / "output_mask_overlay.png")

    ensure_rgb(strict_composite).save(sample_dir / "composite.png")

    anchoring_mask_paths = {}
    if anchoring_is_enabled:
        anchoring_mask_paths = save_latent_anchoring_masks(
            trainer,
            {
                "mask_core": sample_dir / "mask_core.png",
                "mask_free": sample_dir / "mask_free.png",
                "edit_weight": sample_dir / "edit_weight.png",
            },
            original_src_image.size,
            crop_plan=crop_plan,
        )

    row = make_infer_row(
        ref_image=ref_image,
        ref_mask=ref_mask,
        source=original_src_image,
        source_mask=original_src_mask,
        output=primary_output,
        composite=strict_composite,
        panel_size=args.panel_size,
        output_mask=output_mask,
    )
    row_path = sample_dir / "row.png"
    row.save(row_path)

    metadata = {
        "stem": sample["stem"],
        "seed": int(sample_seed),
        "config": build_inference_config_metadata(trainer.config, args.config),
        "checkpoint": {
            "argument": str(args.lora),
            "resolved_lora": str(resolved_lora_path) if resolved_lora_path is not None else None,
            "compatibility": checkpoint_compatibility,
        },
        "parameters": build_effective_inference_metadata(
            trainer.config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_embeds_enabled=resolve_guidance_embeds_enabled(trainer.config, trainer),
            requested_height=args.height,
            requested_width=args.width,
            effective_height=local_output.height,
            effective_width=local_output.width,
        ),
        "inputs": {key: str(value) for key, value in sample.items() if key != "stem"},
        "outputs": {
            "ref_image": str(sample_dir / "ref_image.png"),
            "ref_mask": str(sample_dir / "ref_mask.png"),
            "ref_mask_overlay": str(sample_dir / "ref_mask_overlay.png"),
            "source": str(sample_dir / "source.png"),
            "source_mask": str(sample_dir / "source_mask.png"),
            "source_mask_overlay": str(sample_dir / "source_mask_overlay.png"),
            "output": str(sample_dir / "output.png"),
            "primary_output": str(sample_dir / "output.png"),
            "final_composite": str(
                sample_dir / ("composite.png" if anchoring_is_enabled else "output.png")
            ),
            "strict_composite": str(
                sample_dir / ("composite.png" if anchoring_is_enabled else "output.png")
            ),
            "output_local": str(sample_dir / "output_local.png"),
            "output_restored_raw": str(sample_dir / "output_restored_raw.png"),
            "output_mask": str(sample_dir / "output_mask.png"),
            "output_mask_overlay": str(sample_dir / "output_mask_overlay.png"),
            "composite_compatibility_copy": str(sample_dir / "composite.png"),
            "row": str(row_path),
            **anchoring_mask_paths,
        },
        "sizes": {
            "ref_image": list(ref_image.size),
            "source": list(original_src_image.size),
            "prediction_input": list(predict_src_image.size),
            "output_local": list(local_output.size),
            "output_restored_raw": list(raw_restored_output.size),
            "output": list(primary_output.size),
            "final_composite": list(strict_composite.size),
        },
        "latent_anchoring": anchoring_metadata,
        "adaptive_crop": {
            "config": adaptive_crop_config.model_dump(mode="json"),
            "plan": crop_plan.to_dict() if crop_plan is not None else None,
            "compatibility": crop_compatibility,
        },
        "output_semantics": {
            "output": (
                "Primary latent-anchored result in original source coordinates "
                "(crop-edge-safe when adaptive crop is applied)."
                if anchoring_is_enabled
                else "Final strict-mask composite in original source coordinates."
            ),
            "output_local": "Direct model output in processor coordinates.",
            "output_restored_raw": (
                "Latent-anchored model output hard-pasted into original source coordinates "
                "before optional crop-edge feathering / strict RGB compositing."
                if anchoring_is_enabled
                else "Model output mapped to original source coordinates before mask compositing."
            ),
            "strict_composite": "Compatibility M_core strict composite for A/B comparison.",
        },
    }
    if crop_plan is not None:
        metadata["outputs"].update(
            {
                "crop_source": str(sample_dir / "crop_source.png"),
                "crop_mask": str(sample_dir / "crop_mask.png"),
            }
        )
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, row_path


def main():
    args = parse_args()
    samples = discover_samples(args.testset_root, args.category, args.limit)
    if args.seed + len(samples) - 1 >= 2**32:
        raise ValueError("per-sample seeds must remain in [0, 2**32)")
    trainer, config, resolved_lora_path, checkpoint_compatibility = setup_trainer(args)
    guidance_embeds_enabled = resolve_guidance_embeds_enabled(config, trainer)
    run_anchoring_metadata = latent_anchoring_metadata(config.predict.latent_anchoring)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "config": build_inference_config_metadata(config, args.config),
        "checkpoint": {
            "argument": str(args.lora),
            "resolved_lora": str(resolved_lora_path),
            "compatibility": checkpoint_compatibility,
        },
        "testset_root": str(args.testset_root),
        "category": args.category,
        "num_samples": len(samples),
        "parameters": build_effective_inference_metadata(
            config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_embeds_enabled=guidance_embeds_enabled,
            requested_height=args.height,
            requested_width=args.width,
        ),
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "enhanced_mask_prompt": enhanced_mask_prompt_enabled(config),
        "latent_anchoring": run_anchoring_metadata,
        "adaptive_crop": config.data.init_args.adaptive_crop.model_dump(mode="json"),
        "output_semantics": {
            "output": (
                "Primary latent-anchored result in original source coordinates."
                if run_anchoring_metadata["enabled"]
                else "Final strict-mask composite in original source coordinates."
            ),
            "output_local": "Direct model output in processor coordinates.",
            "output_restored_raw": (
                "Latent-anchored model output hard-pasted into original source coordinates "
                "before optional crop-edge feathering; primary may use feathered restore."
                if run_anchoring_metadata["enabled"]
                else "Model output mapped to original source coordinates before mask compositing."
            ),
            "strict_composite": "Compatibility M_core strict composite for A/B comparison.",
        },
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    manifest_path = args.output_dir / "manifest.jsonl"
    row_paths = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, sample in enumerate(samples):
            sample_seed = args.seed + index
            sample_dir = args.output_dir / f"{index + 1:03d}_{sample['stem']}"
            print(f"[{index + 1}/{len(samples)}] infer {sample['stem']} -> {sample_dir}", flush=True)
            metadata, row_path = run_sample(
                trainer,
                sample,
                sample_dir,
                args,
                sample_seed,
                resolved_lora_path=resolved_lora_path,
                checkpoint_compatibility=checkpoint_compatibility,
            )
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            manifest.flush()
            row_paths.append(row_path)

    stack_rows(row_paths, args.output_dir / "all_rows.png")
    print(f"Done. Results saved to {args.output_dir}", flush=True)
    print(f"Manifest saved to {manifest_path}", flush=True)
    print(f"Overview saved to {args.output_dir / 'all_rows.png'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.utils.adaptive_crop import (
    crop_pil_pair,
    restore_pil_output,
)
from src.utils.adaptive_crop_anchoring import (
    DEFAULT_CROP_EDGE_FEATHER_PX,
    map_model_mask_to_original,
    prepare_adaptive_crop_for_inference,
)
from src.utils.inference_metadata import (
    build_effective_inference_metadata,
    build_inference_config_metadata,
    resolve_guidance_embeds_enabled,
    validate_checkpoint_compatibility,
    validate_inference_mode_inputs,
)
from src.utils.latent_anchoring import (
    add_latent_anchoring_cli_args,
    apply_latent_anchoring_cli_overrides,
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
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "generated_full.png"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inference entrypoint for Insert Anything LoRA.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to config YAML.")
    parser.add_argument(
        "--lora",
        type=Path,
        default=None,
        help="Optional LoRA weight file or checkpoint directory to override config.",
    )
    parser.add_argument(
        "--allow-base-model",
        action="store_true",
        help="Run without LoRA weights. This must be explicit because trained behavior otherwise depends on a checkpoint.",
    )
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow a missing or incompatible checkpoint train_config.yaml. Unsafe unless the mismatch is intentional.",
    )
    parser.add_argument("--src-image", type=Path, required=True, help="Source/scene image path.")
    parser.add_argument("--ref-image", type=Path, required=True, help="Reference object image path.")
    parser.add_argument("--src-mask", type=Path, required=True, help="Mask on source image.")
    parser.add_argument("--ref-mask", type=Path, default=None, help="Mask for reference object. Optional for RGBA images.")
    parser.add_argument("--prompt", type=str, default=None, help="Inference prompt.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Optional negative prompt.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="Reference CFG scale when model.reference_cfg.enabled is true; otherwise native text true-CFG scale.",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic inference seed.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated image output path.")
    parser.add_argument(
        "--composite-output",
        type=Path,
        default=None,
        help="Optional compatibility copy of the final strict-mask output. Defaults to an output-scoped path.",
    )
    parser.add_argument(
        "--row-output",
        type=Path,
        default=None,
        help="Single-row preview output path. Defaults to an output-scoped path.",
    )
    parser.add_argument("--panel-size", type=int, default=320, help="Image area size for the row preview.")
    parser.add_argument(
        "--skip-composite",
        action="store_true",
        help=(
            "Do not save the extra strict-composite compatibility artifact. "
            "A keeps strict --output; B/C/D keep their anchored --output."
        ),
    )
    parser.add_argument("--skip-row", action="store_true", help="Do not save row preview and mask overlays.")
    add_latent_anchoring_cli_args(parser)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.src_mask is None:
        raise ValueError("--src-mask is required")
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
    if args.height is not None and (args.height <= 0 or args.width <= 0):
        raise ValueError("--height and --width must be positive")


def validate_adaptive_crop_inference_args(config, args) -> None:
    adaptive_crop_config = config.data.init_args.adaptive_crop
    if adaptive_crop_config.predict_enabled and (args.height is not None or args.width is not None):
        raise ValueError("Adaptive Crop prediction cannot be combined with explicit --height/--width")


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).copy()


def resolve_lora_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"LoRA weight not found: {path}")

    if path.is_file():
        return path

    preferred_files = [
        path / "pytorch_lora_weights.safetensors",
        path / "model.safetensors",
    ]
    for candidate in preferred_files:
        if candidate.exists():
            return candidate

    safetensors_files = sorted(path.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return safetensors_files[0]
    if len(safetensors_files) > 1:
        raise ValueError(
            f"Multiple safetensors files found in checkpoint directory {path}: {safetensors_files}"
        )
    raise FileNotFoundError(f"No safetensors file found in checkpoint directory: {path}")


def checkpoint_dir_for_lora(lora_path: Path) -> Path:
    return lora_path.parent if lora_path.is_file() else lora_path


def enhanced_mask_prompt_enabled(config) -> bool:
    return bool(config.model.query_prompt.enabled) or bool(config.model.pre_insertion.enabled)


def validate_configured_inference(config, args, resolved_lora_path: Path | None) -> None:
    validate_inference_mode_inputs(
        config,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
    )

    if resolved_lora_path is None and not args.allow_base_model:
        raise ValueError("LoRA weight is required. Pass --lora or set model.lora.pretrained_weight in config.")

    if config.model.query_prompt.enabled:
        if resolved_lora_path is None:
            raise ValueError("query_prompt inference requires a LoRA checkpoint directory with query_prompt.safetensors")
        query_prompt_path = checkpoint_dir_for_lora(resolved_lora_path) / "query_prompt.safetensors"
        if not query_prompt_path.exists():
            raise FileNotFoundError(f"query_prompt is enabled but companion weights were not found: {query_prompt_path}")


def resolve_ref_mask_for_visualization(ref_img: Image.Image, ref_mask: Image.Image | None) -> Image.Image:
    if ref_mask is not None:
        return ref_mask
    if ref_img.mode == "RGBA":
        return ref_img.split()[-1]
    raise ValueError("ref_mask is required for visualization when ref_image has no alpha channel")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def output_artifact_path(output_path: Path, label: str, suffix: str = ".png") -> Path:
    return output_path.with_name(f"{output_path.stem}.{label}{suffix}")


def build_output_path_plan(
    args,
    *,
    has_crop: bool,
    latent_anchoring_is_enabled: bool = False,
    protected_paths=(),
) -> dict[str, Path]:
    output_path = args.output
    paths = {
        "output": output_path,
        "output_local": output_artifact_path(output_path, "output_local"),
        "output_restored_raw": output_artifact_path(output_path, "output_restored_raw"),
        "metadata": output_artifact_path(output_path, "metadata", suffix=".json"),
    }
    if has_crop:
        paths["crop_source"] = output_artifact_path(output_path, "crop_source")
        paths["crop_mask"] = output_artifact_path(output_path, "crop_mask")
    if latent_anchoring_is_enabled:
        paths.update(
            {
                "mask_core": output_artifact_path(output_path, "mask_core"),
                "mask_free": output_artifact_path(output_path, "mask_free"),
                "edit_weight": output_artifact_path(output_path, "edit_weight"),
            }
        )
    if not args.skip_composite:
        paths["composite_compatibility_copy"] = (
            args.composite_output or output_artifact_path(output_path, "composite")
        )
    if not args.skip_row:
        paths.update(
            {
                "row": args.row_output or output_artifact_path(output_path, "row"),
                "reference_mask_overlay": output_artifact_path(output_path, "ref_mask_overlay"),
                "source_mask_overlay": output_artifact_path(output_path, "source_mask_overlay"),
                "output_mask": output_artifact_path(output_path, "output_mask"),
                "output_mask_overlay": output_artifact_path(output_path, "output_mask_overlay"),
            }
        )

    seen: dict[Path, str] = {}
    output_resolved = output_path.resolve()
    for name, path in paths.items():
        resolved = path.resolve()
        if name == "composite_compatibility_copy" and resolved == output_resolved:
            if latent_anchoring_is_enabled:
                raise ValueError(
                    "B/C/D require --output and --composite-output to be distinct because the primary "
                    "anchored output and strict M_core composite have different semantics"
                )
            continue
        previous = seen.get(resolved)
        if previous is not None:
            raise ValueError(
                f"inference output paths must be distinct: {previous} and {name} both resolve to {resolved}"
            )
        seen[resolved] = name

    protected = {Path(path).resolve() for path in protected_paths if path is not None}
    collisions = [
        (name, path.resolve())
        for name, path in paths.items()
        if path.resolve() in protected
    ]
    if collisions:
        details = ", ".join(f"{name}={path}" for name, path in collisions)
        raise ValueError(f"inference outputs must not overwrite config, checkpoint, or input files: {details}")
    return paths


def save_latent_anchoring_masks(
    trainer,
    output_paths: dict[str, Path],
    original_size,
    crop_plan=None,
) -> dict[str, str]:
    getter = getattr(trainer, "get_last_latent_anchoring_masks", None)
    artifacts = getter() if callable(getter) else None
    if artifacts is None:
        raise RuntimeError("latent anchoring was enabled but the trainer exposed no mask artifacts")

    saved = {}
    for key, resample in (
        ("core", Image.Resampling.NEAREST),
        ("free", Image.Resampling.NEAREST),
        ("edit_weight", Image.Resampling.BILINEAR),
    ):
        tensor = artifacts[key].detach().float().clamp(0, 1).cpu()
        array = (tensor.numpy() * 255.0).round().astype(np.uint8)
        local = Image.fromarray(array, mode="L")
        image = map_model_mask_to_original(
            local,
            original_size=original_size,
            crop_plan=crop_plan,
            resample=resample,
        )
        path_key = "mask_core" if key == "core" else "mask_free" if key == "free" else "edit_weight"
        path = output_paths[path_key]
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        saved[path_key] = str(path)
    return saved


def save_visualization_outputs(
    src_img: Image.Image,
    src_mask: Image.Image,
    ref_img: Image.Image,
    ref_mask: Image.Image,
    primary_output_img: Image.Image,
    final_composite_img: Image.Image,
    output_path: Path,
    row_output_path: Path,
    panel_size: int,
    mask_threshold: float,
):
    source_mask_overlay_path = output_artifact_path(output_path, "source_mask_overlay")
    ref_mask_overlay_path = output_artifact_path(output_path, "ref_mask_overlay")
    output_mask_path = output_artifact_path(output_path, "output_mask")
    output_mask_overlay_path = output_artifact_path(output_path, "output_mask_overlay")

    output_mask = binarize_mask(src_mask, mask_threshold).resize(
        ensure_rgb(primary_output_img).size,
        Image.Resampling.NEAREST,
    )
    make_mask_overlay(ref_img, ref_mask).save(ref_mask_overlay_path)
    make_mask_overlay(src_img, src_mask).save(source_mask_overlay_path)
    save_mask(output_mask, output_mask_path)
    make_mask_overlay(primary_output_img, output_mask).save(output_mask_overlay_path)

    row = make_infer_row(
        ref_image=ref_img,
        ref_mask=ref_mask,
        source=src_img,
        source_mask=src_mask,
        output=primary_output_img,
        composite=final_composite_img,
        panel_size=panel_size,
        output_mask=output_mask,
    )
    row_output_path.parent.mkdir(parents=True, exist_ok=True)
    row.save(row_output_path)
    print(f"Reference mask overlay saved to {ref_mask_overlay_path}")
    print(f"Source mask overlay saved to {source_mask_overlay_path}")
    print(f"Output mask saved to {output_mask_path}")
    print(f"Output mask overlay saved to {output_mask_overlay_path}")
    print(f"Row preview saved to {row_output_path}")
    return {
        "reference_mask_overlay": str(ref_mask_overlay_path),
        "source_mask_overlay": str(source_mask_overlay_path),
        "output_mask": str(output_mask_path),
        "output_mask_overlay": str(output_mask_overlay_path),
        "row": str(row_output_path),
    }


def main():
    args = parse_args()

    from src.data.config import load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    validate_args(args)

    config = load_config_from_yaml(str(args.config))
    latent_anchoring_config = apply_latent_anchoring_cli_overrides(config, args)
    validate_adaptive_crop_inference_args(config, args)
    validate_latent_anchoring_runtime(
        config,
        height=args.height,
        width=args.width,
    )
    anchoring_metadata = latent_anchoring_metadata(latent_anchoring_config)
    anchoring_is_enabled = bool(anchoring_metadata["enabled"])
    lora_path = args.lora
    if lora_path is None and config.model.lora.pretrained_weight is not None:
        lora_path = Path(config.model.lora.pretrained_weight)

    if lora_path is not None:
        resolved_lora_path = resolve_lora_path(lora_path)
        config.model.lora.pretrained_weight = str(resolved_lora_path)
        print(f"Using LoRA weight: {resolved_lora_path}")
    else:
        resolved_lora_path = None

    validate_configured_inference(config, args, resolved_lora_path)
    checkpoint_compatibility = validate_checkpoint_compatibility(
        config,
        resolved_lora_path,
        allow_mismatch=bool(getattr(args, "allow_checkpoint_mismatch", False)),
    )

    src_img = load_rgb_image(args.src_image)
    ref_img = load_rgb_image(args.ref_image)
    src_mask = load_rgb_image(args.src_mask) if args.src_mask is not None else None
    ref_mask = load_rgb_image(args.ref_mask) if args.ref_mask is not None else None
    ref_mask_for_visualization = resolve_ref_mask_for_visualization(ref_img, ref_mask)

    original_src_img = ensure_rgb(src_img)
    original_src_mask = ensure_mask(src_mask)
    if original_src_img.size != original_src_mask.size:
        raise ValueError(
            "source image and source mask must have identical geometry, "
            f"got image={original_src_img.size}, mask={original_src_mask.size}"
        )
    adaptive_crop_config = config.data.init_args.adaptive_crop
    composite_mask_threshold = (
        float(latent_anchoring_config.mask_threshold)
        if anchoring_is_enabled
        else float(adaptive_crop_config.mask_threshold)
    )
    crop_plan = None
    crop_compatibility = {"enabled": False}
    predict_src_img = original_src_img
    predict_src_mask = original_src_mask
    edge_feather_px = 0
    if adaptive_crop_config.predict_enabled:
        crop_plan, crop_compatibility = prepare_adaptive_crop_for_inference(
            image_size=original_src_img.size,
            mask=original_src_mask,
            adaptive_crop_config=adaptive_crop_config,
            config=config,
            processor_config=config.data.init_args.processor,
        )
        predict_src_img, predict_src_mask = crop_pil_pair(
            original_src_img,
            original_src_mask,
            crop_plan,
        )
        if anchoring_is_enabled and crop_plan is not None and crop_plan.applied:
            edge_feather_px = int(
                crop_compatibility.get("feather_width_px", DEFAULT_CROP_EDGE_FEATHER_PX)
            )

    output_path_plan = build_output_path_plan(
        args,
        has_crop=crop_plan is not None,
        latent_anchoring_is_enabled=anchoring_is_enabled,
        protected_paths=(
            args.config,
            resolved_lora_path,
            checkpoint_compatibility.get("train_config"),
            args.src_image,
            args.src_mask,
            args.ref_image,
            args.ref_mask,
        ),
    )
    local_output_path = output_path_plan["output_local"]
    raw_restored_output_path = output_path_plan["output_restored_raw"]
    metadata_output_path = output_path_plan["metadata"]
    composite_output_path = output_path_plan.get("composite_compatibility_copy")
    row_output_path = output_path_plan.get("row")
    crop_source_path = output_path_plan.get("crop_source")
    crop_mask_path = output_path_plan.get("crop_mask")

    predict_kwargs = {
        "image": [predict_src_img, ref_img],
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

    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()
    guidance_embeds_enabled = resolve_guidance_embeds_enabled(config, trainer)
    seed_everything(args.seed)
    result = trainer.predict(**predict_kwargs)
    local_generated_img = ensure_rgb(result[0])
    if crop_plan is not None:
        # Hard paste is always computed for diagnostics; optional feather is used for B/C/D primary.
        raw_restored_img, strict_composite_img = restore_pil_output(
            original_source=original_src_img,
            original_mask=original_src_mask,
            local_output=local_generated_img,
            plan=crop_plan,
            blend_mode=adaptive_crop_config.blend_mode,
            edge_feather_px=0,
        )
        if edge_feather_px > 0:
            primary_restored_img, _ = restore_pil_output(
                original_source=original_src_img,
                original_mask=original_src_mask,
                local_output=local_generated_img,
                plan=crop_plan,
                blend_mode=adaptive_crop_config.blend_mode,
                edge_feather_px=edge_feather_px,
            )
        else:
            primary_restored_img = raw_restored_img
    else:
        raw_restored_img = local_generated_img.resize(original_src_img.size, Image.Resampling.LANCZOS)
        primary_restored_img = raw_restored_img
        strict_composite_img = build_strict_mask_composite(
            original_src_img,
            original_src_mask,
            raw_restored_img,
            composite_mask_threshold,
        )
    primary_output_img = primary_restored_img if anchoring_is_enabled else strict_composite_img
    if (
        raw_restored_img.size != original_src_img.size
        or strict_composite_img.size != original_src_img.size
        or primary_output_img.size != original_src_img.size
    ):
        raise ValueError("restored and final outputs must match the original source size")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    local_generated_img.save(local_output_path)
    ensure_rgb(raw_restored_img).save(raw_restored_output_path)
    ensure_rgb(primary_output_img).save(args.output)
    print(f"Local model output saved to {local_output_path}")
    print(f"Raw restored output saved to {raw_restored_output_path}")
    if anchoring_is_enabled:
        print(f"Primary latent-anchored output saved to {args.output}")
    else:
        print(f"Final strict-mask output saved to {args.output}")

    output_paths = {
        "output": str(args.output),
        "primary_output": str(args.output),
        "output_local": str(local_output_path),
        "output_restored_raw": str(raw_restored_output_path),
        "metadata": str(metadata_output_path),
    }
    if not anchoring_is_enabled:
        output_paths["final_composite"] = str(args.output)
        output_paths["strict_composite"] = str(args.output)
    elif args.skip_composite:
        # Keep the metadata schema explicit: the compatibility image exists in
        # memory for rows/diagnostics but was intentionally not written.
        output_paths["composite_compatibility_copy"] = None
        output_paths["strict_composite"] = None
        output_paths["final_composite"] = None

    if anchoring_is_enabled:
        mask_paths = save_latent_anchoring_masks(
            trainer,
            output_path_plan,
            original_src_img.size,
            crop_plan=crop_plan,
        )
        output_paths.update(mask_paths)
        for label, path in mask_paths.items():
            print(f"Latent anchoring {label} saved to {path}")

    if crop_plan is not None:
        if crop_source_path is None or crop_mask_path is None:
            raise RuntimeError("Adaptive Crop output path plan is missing crop artifacts")
        ensure_rgb(predict_src_img).save(crop_source_path)
        save_mask(predict_src_mask, crop_mask_path)
        output_paths.update(
            {
                "crop_source": str(crop_source_path),
                "crop_mask": str(crop_mask_path),
            }
        )

    if not args.skip_composite:
        if composite_output_path is None:
            raise RuntimeError("composite compatibility output path is missing")
        composite_output_path.parent.mkdir(parents=True, exist_ok=True)
        if composite_output_path != args.output:
            ensure_rgb(strict_composite_img).save(composite_output_path)
        output_paths["composite_compatibility_copy"] = str(composite_output_path)
        if anchoring_is_enabled:
            output_paths["strict_composite"] = str(composite_output_path)
            output_paths["final_composite"] = str(composite_output_path)
        print(f"Composite compatibility copy saved to {composite_output_path}")
    if not args.skip_row:
        if row_output_path is None:
            raise RuntimeError("row output path is missing")
        visualization_paths = save_visualization_outputs(
            src_img=original_src_img,
            src_mask=original_src_mask,
            ref_img=ref_img,
            ref_mask=ref_mask_for_visualization,
            primary_output_img=primary_output_img,
            final_composite_img=strict_composite_img,
            output_path=args.output,
            row_output_path=row_output_path,
            panel_size=args.panel_size,
            mask_threshold=composite_mask_threshold,
        )
        output_paths.update(visualization_paths)

    metadata = {
        "seed": int(args.seed),
        "config": build_inference_config_metadata(config, args.config),
        "checkpoint": {
            "argument": str(args.lora) if args.lora is not None else None,
            "resolved_lora": str(resolved_lora_path) if resolved_lora_path is not None else None,
            "allow_base_model": bool(args.allow_base_model),
            "compatibility": checkpoint_compatibility,
        },
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
            effective_height=local_generated_img.height,
            effective_width=local_generated_img.width,
        ),
        "inputs": {
            "source": str(args.src_image.resolve()),
            "source_mask": str(args.src_mask.resolve()),
            "reference": str(args.ref_image.resolve()),
            "reference_mask": str(args.ref_mask.resolve()) if args.ref_mask is not None else None,
        },
        "outputs": output_paths,
        "sizes": {
            "source": list(original_src_img.size),
            "prediction_input": list(predict_src_img.size),
            "output_local": list(local_generated_img.size),
            "output_restored_raw": list(raw_restored_img.size),
            "output": list(primary_output_img.size),
            "final_composite": list(strict_composite_img.size),
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
    with metadata_output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Inference metadata saved to {metadata_output_path}")


if __name__ == "__main__":
    main()

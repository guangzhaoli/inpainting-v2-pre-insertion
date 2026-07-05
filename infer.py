#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "insert_anything_fixed_pixels_parquet.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "generated_full.png"
DEFAULT_COMPOSITE_OUTPUT = REPO_ROOT / "output" / "composite_result.png"


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
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Generated image output path.")
    parser.add_argument(
        "--composite-output",
        type=Path,
        default=DEFAULT_COMPOSITE_OUTPUT,
        help="Composite image output path.",
    )
    parser.add_argument("--skip-composite", action="store_true", help="Do not save composite image.")
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.src_mask is None:
        raise ValueError("--src-mask is required")


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).copy()


def resolve_lora_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"LoRA weight not found: {path}")

    if path.is_file():
        return path

    preferred_files = [
        path / "model.safetensors",
        path / "pytorch_lora_weights.safetensors",
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
    reference_cfg_enabled = bool(config.model.reference_cfg.enabled)
    reference_cfg_force_empty_prompt = reference_cfg_enabled and bool(config.model.reference_cfg.force_empty_prompt)

    if reference_cfg_enabled and enhanced_mask_prompt_enabled(config):
        raise ValueError("reference_cfg requires query_prompt.enabled=false and pre_insertion.enabled=false")

    if reference_cfg_enabled and args.negative_prompt is not None:
        raise ValueError("reference_cfg cannot be combined with --negative-prompt")

    if not args.prompt and not enhanced_mask_prompt_enabled(config) and not reference_cfg_force_empty_prompt:
        raise ValueError("--prompt is required for native qwen image inpainting")

    if resolved_lora_path is None and not args.allow_base_model:
        raise ValueError("LoRA weight is required. Pass --lora or set model.lora.pretrained_weight in config.")

    if config.model.query_prompt.enabled:
        if resolved_lora_path is None:
            raise ValueError("query_prompt inference requires a LoRA checkpoint directory with query_prompt.safetensors")
        query_prompt_path = checkpoint_dir_for_lora(resolved_lora_path) / "query_prompt.safetensors"
        if not query_prompt_path.exists():
            raise FileNotFoundError(f"query_prompt is enabled but companion weights were not found: {query_prompt_path}")


def build_composite(src_img: Image.Image, src_mask: Image.Image, generated_img: Image.Image) -> Image.Image:
    if generated_img.size != src_img.size:
        generated_img = generated_img.resize(src_img.size, Image.LANCZOS)

    if src_mask.size != src_img.size:
        src_mask = src_mask.resize(src_img.size, Image.NEAREST)

    src_np = np.array(src_img).astype(np.float32)
    generated_np = np.array(generated_img).astype(np.float32)
    mask_np = np.array(src_mask.convert("L")).astype(np.float32) / 255.0

    if src_np.ndim == 3:
        mask_np = mask_np[:, :, np.newaxis]

    composite_np = mask_np * generated_np + (1.0 - mask_np) * src_np
    return Image.fromarray(composite_np.astype(np.uint8))


def main():
    args = parse_args()

    from src.data.config import load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    validate_args(args)

    config = load_config_from_yaml(str(args.config))
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

    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()

    src_img = load_rgb_image(args.src_image)
    ref_img = load_rgb_image(args.ref_image)
    src_mask = load_rgb_image(args.src_mask) if args.src_mask is not None else None
    ref_mask = load_rgb_image(args.ref_mask) if args.ref_mask is not None else None

    predict_kwargs = {
        "image": [src_img, ref_img],
        "prompt": args.prompt,
        "ref_mask": ref_mask,
        "src_mask": src_mask,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
    }
    if args.height is not None:
        predict_kwargs["height"] = args.height
    if args.width is not None:
        predict_kwargs["width"] = args.width

    result = trainer.predict(**predict_kwargs)
    generated_img = result[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_img.save(args.output)
    print(f"Generated image saved to {args.output}")

    if not args.skip_composite:
        composite_img = build_composite(src_img, src_mask, generated_img)
        args.composite_output.parent.mkdir(parents=True, exist_ok=True)
        composite_img.save(args.composite_output)
        print(f"Composite image saved to {args.composite_output}")


if __name__ == "__main__":
    main()

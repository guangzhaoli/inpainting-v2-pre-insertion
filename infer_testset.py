#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from infer import (
    build_composite,
    enhanced_mask_prompt_enabled,
    load_rgb_image,
    resolve_lora_path,
    validate_configured_inference,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "insert_anything_teacher_accessory_jewelry.yaml"
DEFAULT_TESTSET_ROOT = Path("/mnt/cpfs/weiqingyan/flow/qwen-image-inpainting/testset_20260326")
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs"
    / "inpainting_v2_teacher_logs"
    / "teacher_accessory_jewelry_native_qwen_image_inpainting"
    / "v2"
    / "checkpoint-1-3000"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "testset_20260326_accessory_checkpoint-1-3000"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch inference for Insert Anything testset folders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lora", type=Path, default=DEFAULT_CHECKPOINT, help="LoRA file or checkpoint directory.")
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
    return parser.parse_args(argv)


def natural_key(path: Path):
    parts = []
    for token in path.stem.replace("_", "-").split("-"):
        parts.append(int(token) if token.isdigit() else token)
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def ensure_mask(mask: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(mask).convert("L")


def save_mask(mask: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_mask(mask).save(path)


def make_mask_overlay(image: Image.Image, mask: Image.Image, color=(255, 0, 0), alpha=0.45) -> Image.Image:
    base = ensure_rgb(image)
    mask_l = ensure_mask(mask)
    if mask_l.size != base.size:
        mask_l = mask_l.resize(base.size, Image.NEAREST)

    overlay = Image.new("RGB", base.size, color)
    mask_np = (np.array(mask_l, dtype=np.float32) / 255.0 * int(alpha * 255)).astype(np.uint8)
    alpha_mask = Image.fromarray(mask_np, mode="L")
    return Image.composite(overlay, base, alpha_mask)


def mask_to_rgb(mask: Image.Image) -> Image.Image:
    return ensure_mask(mask).convert("RGB")


def draw_centered_label(draw: ImageDraw.ImageDraw, text: str, panel_width: int, y: int):
    bbox = draw.textbbox((0, 0), text)
    x = max(0, (panel_width - (bbox[2] - bbox[0])) // 2)
    draw.text((x, y), text, fill=(20, 20, 20))


def make_panel(image: Image.Image, label: str, panel_size: int) -> Image.Image:
    label_h = 28
    panel = Image.new("RGB", (panel_size, panel_size + label_h), (245, 245, 245))
    image = ensure_rgb(image)
    image.thumbnail((panel_size, panel_size), Image.LANCZOS)
    x = (panel_size - image.width) // 2
    y = label_h + (panel_size - image.height) // 2
    panel.paste(image, (x, y))

    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size - 1, label_h - 1), fill=(235, 235, 235))
    draw_centered_label(draw, label, panel_size, 7)
    draw.rectangle((0, 0, panel_size - 1, panel_size + label_h - 1), outline=(210, 210, 210))
    return panel


def make_row(panels: list[tuple[str, Image.Image]], panel_size: int) -> Image.Image:
    gap = 8
    rendered = [make_panel(image, label, panel_size) for label, image in panels]
    width = sum(panel.width for panel in rendered) + gap * (len(rendered) - 1)
    height = max(panel.height for panel in rendered)
    row = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in rendered:
        row.paste(panel, (x, 0))
        x += panel.width + gap
    return row


def stack_rows(row_paths: list[Path], output_path: Path):
    rows = [Image.open(path).convert("RGB") for path in row_paths]
    gap = 12
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        sheet.paste(row, ((width - row.width) // 2, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    for row in rows:
        row.close()


def setup_trainer(args):
    from src.data.config import load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = load_config_from_yaml(str(args.config))
    resolved_lora_path = resolve_lora_path(args.lora)
    config.model.lora.pretrained_weight = str(resolved_lora_path)

    validation_args = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        allow_base_model=False,
    )
    validate_configured_inference(config, validation_args, resolved_lora_path)

    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()
    return trainer, config, resolved_lora_path


def run_sample(trainer, sample, sample_dir: Path, args, sample_seed: int):
    seed_everything(sample_seed)

    ref_image = load_rgb_image(sample["ref_image"])
    ref_mask = load_rgb_image(sample["ref_mask"])
    src_image = load_rgb_image(sample["source"])
    src_mask = load_rgb_image(sample["source_mask"])

    predict_kwargs = {
        "image": [src_image, ref_image],
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
    output = result[0]
    composite = build_composite(src_image, src_mask, output)

    sample_dir.mkdir(parents=True, exist_ok=True)
    ensure_rgb(ref_image).save(sample_dir / "ref_image.png")
    save_mask(ref_mask, sample_dir / "ref_mask.png")
    make_mask_overlay(ref_image, ref_mask).save(sample_dir / "ref_mask_overlay.png")

    ensure_rgb(src_image).save(sample_dir / "source.png")
    save_mask(src_mask, sample_dir / "source_mask.png")
    make_mask_overlay(src_image, src_mask).save(sample_dir / "source_mask_overlay.png")

    ensure_rgb(output).save(sample_dir / "output.png")
    output_mask = ensure_mask(src_mask).resize(output.size, Image.NEAREST)
    output_mask.save(sample_dir / "output_mask.png")
    make_mask_overlay(output, output_mask).save(sample_dir / "output_mask_overlay.png")

    ensure_rgb(composite).save(sample_dir / "composite.png")

    row = make_row(
        [
            ("ref_image", ref_image),
            ("ref_mask", mask_to_rgb(ref_mask)),
            ("source", src_image),
            ("source_mask", mask_to_rgb(src_mask)),
            ("output", output),
            ("composite", composite),
        ],
        args.panel_size,
    )
    row_path = sample_dir / "row.png"
    row.save(row_path)

    metadata = {
        "stem": sample["stem"],
        "seed": sample_seed,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "inputs": {key: str(value) for key, value in sample.items() if key != "stem"},
        "outputs": {
            "ref_image": str(sample_dir / "ref_image.png"),
            "ref_mask": str(sample_dir / "ref_mask.png"),
            "ref_mask_overlay": str(sample_dir / "ref_mask_overlay.png"),
            "source": str(sample_dir / "source.png"),
            "source_mask": str(sample_dir / "source_mask.png"),
            "source_mask_overlay": str(sample_dir / "source_mask_overlay.png"),
            "output": str(sample_dir / "output.png"),
            "output_mask": str(sample_dir / "output_mask.png"),
            "output_mask_overlay": str(sample_dir / "output_mask_overlay.png"),
            "composite": str(sample_dir / "composite.png"),
            "row": str(row_path),
        },
        "sizes": {
            "ref_image": list(ref_image.size),
            "source": list(src_image.size),
            "output": list(output.size),
            "composite": list(composite.size),
        },
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, row_path


def main():
    args = parse_args()
    samples = discover_samples(args.testset_root, args.category, args.limit)
    trainer, config, resolved_lora_path = setup_trainer(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "config": str(args.config),
        "checkpoint_arg": str(args.lora),
        "resolved_lora": str(resolved_lora_path),
        "testset_root": str(args.testset_root),
        "category": args.category,
        "num_samples": len(samples),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "enhanced_mask_prompt": enhanced_mask_prompt_enabled(config),
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
            metadata, row_path = run_sample(trainer, sample, sample_dir, args, sample_seed)
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            manifest.flush()
            row_paths.append(row_path)

    stack_rows(row_paths, args.output_dir / "all_rows.png")
    print(f"Done. Results saved to {args.output_dir}", flush=True)
    print(f"Manifest saved to {manifest_path}", flush=True)
    print(f"Overview saved to {args.output_dir / 'all_rows.png'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore an Insert Anything Hugging Face Arrow dataset to local folder structure."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a load_from_disk() directory, a split directory, or a single .arrow file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory for restored files.",
    )
    parser.add_argument(
        "--default-split",
        type=str,
        default="train",
        help="Split name to use when the input does not include split information.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".png",
        choices=[".png", ".jpg", ".jpeg", ".webp", ".bmp"],
        help="Image extension for restored files.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG/WebP save quality when using a lossy image extension.",
    )
    return parser.parse_args()


def sanitize_component(value: str | None, default_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return default_name
    return text.replace("/", "_").replace("\\", "_")


def decode_image(value) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            with Image.open(BytesIO(image_bytes)) as img:
                return img.copy()
        if image_path:
            with Image.open(image_path) as img:
                return img.copy()
    if isinstance(value, str):
        with Image.open(value) as img:
            return img.copy()
    raise TypeError(f"Unsupported image payload type: {type(value)}")


def save_image(image: Image.Image, output_path: Path, jpeg_quality: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output_path, quality=jpeg_quality)
    elif suffix == ".webp":
        image.save(output_path, quality=jpeg_quality)
    else:
        image.save(output_path)


def load_arrow_source(input_path: Path, default_split: str) -> Dict[str, Dataset]:
    if input_path.is_file():
        if input_path.suffix != ".arrow":
            raise ValueError(f"Unsupported input file: {input_path}")
        return {default_split: Dataset.from_file(str(input_path))}

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    try:
        loaded = load_from_disk(str(input_path))
        if isinstance(loaded, DatasetDict):
            return dict(loaded)
        return {default_split: loaded}
    except Exception:
        arrow_files = sorted(input_path.rglob("*.arrow"))
        if not arrow_files:
            raise
        datasets = [Dataset.from_file(str(path)) for path in arrow_files]
        return {default_split: concatenate_datasets(datasets)}


def restore_split(dataset: Dataset, split_name: str, output_root: Path, image_ext: str, jpeg_quality: int) -> Tuple[int, int]:
    restored = 0
    skipped = 0

    for idx, row in enumerate(dataset):
        try:
            category = sanitize_component(row.get("category"), "uncategorized")
            stem = sanitize_component(row.get("id"), f"{split_name}_{idx:06d}")
            prompt = row.get("prompt", "")

            ref_image = decode_image(row.get("ref_image"))
            ref_mask = decode_image(row.get("ref_mask"))
            tar_image = decode_image(row.get("tar_image"))
            tar_mask = decode_image(row.get("tar_mask"))

            if ref_image is None or ref_mask is None or tar_image is None or tar_mask is None:
                skipped += 1
                continue

            sample_root = output_root / split_name / category
            save_image(ref_image, sample_root / "ref_image" / f"{stem}{image_ext}", jpeg_quality)
            save_image(ref_mask.convert("L"), sample_root / "ref_mask" / f"{stem}{image_ext}", jpeg_quality)
            save_image(tar_image, sample_root / "tar_image" / f"{stem}{image_ext}", jpeg_quality)
            save_image(tar_mask.convert("L"), sample_root / "tar_mask" / f"{stem}{image_ext}", jpeg_quality)

            prompt_path = sample_root / "tar_image" / f"{stem}.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt or "", encoding="utf-8")

            restored += 1
        except Exception as exc:
            skipped += 1
            print(f"[warn] skip split={split_name} idx={idx} id={row.get('id')}: {exc}")

    return restored, skipped


def main():
    args = parse_args()
    datasets_by_split = load_arrow_source(args.input, args.default_split)

    args.output.mkdir(parents=True, exist_ok=True)

    total_restored = 0
    total_skipped = 0
    for split_name, dataset in datasets_by_split.items():
        restored, skipped = restore_split(
            dataset=dataset,
            split_name=split_name,
            output_root=args.output,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
        )
        total_restored += restored
        total_skipped += skipped
        print(f"[{split_name}] restored={restored} skipped={skipped}")

    print(f"[done] restored={total_restored} skipped={total_skipped} output={args.output}")


if __name__ == "__main__":
    main()

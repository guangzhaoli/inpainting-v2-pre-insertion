#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from PIL import Image

_DATASET_CACHE: dict[tuple[str, str], Dict[str, Dataset]] = {}


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
    parser.add_argument(
        "--backend",
        type=str,
        default="process",
        choices=["process", "thread", "none"],
        help="Parallel backend. 'process' is usually best for PIL decode/encode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of workers.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Number of samples handled by one worker task.",
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


def get_dataset_cached(input_path_str: str, default_split: str, split_name: str) -> Dataset:
    key = (input_path_str, default_split)
    if key not in _DATASET_CACHE:
        _DATASET_CACHE[key] = load_arrow_source(Path(input_path_str), default_split)
    return _DATASET_CACHE[key][split_name]


def restore_one_row(
    row: Dict,
    split_name: str,
    idx: int,
    output_root: Path,
    image_ext: str,
    jpeg_quality: int,
) -> bool:
    category = sanitize_component(row.get("category"), "uncategorized")
    stem = sanitize_component(row.get("id"), f"{split_name}_{idx:06d}")
    prompt = row.get("prompt", "")

    ref_image = decode_image(row.get("ref_image"))
    ref_mask = decode_image(row.get("ref_mask"))
    tar_image = decode_image(row.get("tar_image"))
    tar_mask = decode_image(row.get("tar_mask"))

    if ref_image is None or ref_mask is None or tar_image is None or tar_mask is None:
        return False

    sample_root = output_root / split_name / category

    save_image(ref_image, sample_root / "ref_image" / f"{stem}{image_ext}", jpeg_quality)
    save_image(ref_mask.convert("L"), sample_root / "ref_mask" / f"{stem}{image_ext}", jpeg_quality)
    save_image(tar_image, sample_root / "tar_image" / f"{stem}{image_ext}", jpeg_quality)
    save_image(tar_mask.convert("L"), sample_root / "tar_mask" / f"{stem}{image_ext}", jpeg_quality)

    prompt_path = sample_root / "tar_image" / f"{stem}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt or "", encoding="utf-8")

    return True


def process_chunk(task: Tuple[str, str, str, int, int, str, str, int]) -> Tuple[str, int, int]:
    input_path_str, default_split, split_name, start, end, output_root_str, image_ext, jpeg_quality = task
    dataset = get_dataset_cached(input_path_str, default_split, split_name)
    output_root = Path(output_root_str)

    batch = dataset[start:end]
    keys = list(batch.keys())
    if not keys:
        return split_name, 0, 0

    num_rows = len(batch[keys[0]])
    restored = 0
    skipped = 0

    for offset in range(num_rows):
        idx = start + offset
        row = {k: batch[k][offset] for k in keys}
        try:
            ok = restore_one_row(
                row=row,
                split_name=split_name,
                idx=idx,
                output_root=output_root,
                image_ext=image_ext,
                jpeg_quality=jpeg_quality,
            )
            if ok:
                restored += 1
            else:
                skipped += 1
        except Exception as exc:
            skipped += 1
            print(f"[warn] skip split={split_name} idx={idx} id={row.get('id')}: {exc}")

    return split_name, restored, skipped


def build_tasks(
    datasets_by_split: Dict[str, Dataset],
    input_path: Path,
    default_split: str,
    output_root: Path,
    image_ext: str,
    jpeg_quality: int,
    chunk_size: int,
):
    tasks = []
    for split_name, dataset in datasets_by_split.items():
        total = len(dataset)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            tasks.append(
                (
                    str(input_path),
                    default_split,
                    split_name,
                    start,
                    end,
                    str(output_root),
                    image_ext,
                    jpeg_quality,
                )
            )
    return tasks


def run_parallel(tasks, backend: str, workers: int):
    totals: Dict[str, list[int]] = {}

    if backend == "none" or workers <= 1:
        for task in tasks:
            split_name, restored, skipped = process_chunk(task)
            totals.setdefault(split_name, [0, 0])
            totals[split_name][0] += restored
            totals[split_name][1] += skipped
        return totals

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor

    with executor_cls(max_workers=workers) as ex:
        futures = [ex.submit(process_chunk, task) for task in tasks]
        for fut in as_completed(futures):
            split_name, restored, skipped = fut.result()
            totals.setdefault(split_name, [0, 0])
            totals[split_name][0] += restored
            totals[split_name][1] += skipped

    return totals


def main():
    args = parse_args()
    datasets_by_split = load_arrow_source(args.input, args.default_split)

    args.output.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        datasets_by_split=datasets_by_split,
        input_path=args.input,
        default_split=args.default_split,
        output_root=args.output,
        image_ext=args.image_ext,
        jpeg_quality=args.jpeg_quality,
        chunk_size=max(1, args.chunk_size),
    )

    totals = run_parallel(
        tasks=tasks,
        backend=args.backend,
        workers=max(1, args.workers),
    )

    total_restored = 0
    total_skipped = 0
    for split_name in sorted(totals):
        restored, skipped = totals[split_name]
        total_restored += restored
        total_skipped += skipped
        print(f"[{split_name}] restored={restored} skipped={skipped}")

    print(f"[done] restored={total_restored} skipped={total_skipped} output={args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""为各大类生成 manifest CSV。

按 config 中声明的 categories 扫描 dataset_path 下的子目录,
对齐 ref_image / ref_mask / tar_image / tar_mask 四路 stem 后导出 CSV。
路径为相对 dataset_path 的相对路径,与现有 manifest schema 一致:
    ref_image,ref_mask,tar_image,tar_mask,category,stem,prompt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Corresponds to categories used by configs/teacher/*.yaml
CATEGORY_GROUPS = {
    "animal": ["cat", "dog"],
    "fashion_tryon": ["clothes", "dress", "TopRef_BodyMask", "BottomRef_BodyMask", "TopRef_body_man", "pure_color"],
    "general_object": ["others"],
    "human": ["person"],
}

DATASET_PATH = "/mnt/cpfs/weiqingyan/flow/qwen-image-inpainting/datasets/anyinsertion_v2"
MANIFEST_DIR = Path("/mnt/cpfs/weiqingyan/inpainting/inpainting-v2-pre-insertion/manifests")


def find_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for ext in IMG_EXTS:
        images.extend(directory.glob(f"*{ext}"))
        images.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(images)


def find_file_with_stem(directory: Path, stem: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path
        upper_path = directory / f"{stem}{ext.upper()}"
        if upper_path.exists():
            return upper_path
    return None


def scan_category(category_dir: Path, category: str, dataset_path: Path) -> list[dict]:
    """扫描单个 category 子目录,返回 sample 行列表(相对路径)。

    假设每个 ref_image 在 ref_mask/tar_image/tar_mask 下都有同名文件,
    不做存在性校验,直接用同名拼接(已确认四路文件齐全)。
    """
    ref_image_dir = category_dir / "ref_image"
    if not ref_image_dir.exists():
        raise FileNotFoundError(f"Missing required directory: {ref_image_dir}")

    ref_images = find_images(ref_image_dir)
    rows: list[dict] = []
    rel_prefix = category_dir.relative_to(dataset_path)

    for ref_img_path in ref_images:
        name = ref_img_path.name
        stem = ref_img_path.stem
        rows.append({
            "ref_image": f"{rel_prefix}/ref_image/{name}",
            "ref_mask": f"{rel_prefix}/ref_mask/{name}",
            "tar_image": f"{rel_prefix}/tar_image/{name}",
            "tar_mask": f"{rel_prefix}/tar_mask/{name}",
            "category": category,
            "stem": stem,
            "prompt": "",
        })

    print(f"  [{category}] samples={len(rows)}", file=sys.stderr)
    return rows


def build_manifest(group: str, categories: list[str], dataset_path: Path, out_path: Path) -> int:
    rows: list[dict] = []
    print(f"Building manifest for '{group}' (categories={categories})", file=sys.stderr)
    for category in categories:
        category_dir = dataset_path / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")
        rows.extend(scan_category(category_dir, category, dataset_path))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ref_image", "ref_mask", "tar_image", "tar_mask", "category", "stem", "prompt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"  -> wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-path", default=DATASET_PATH, help="数据集根目录")
    ap.add_argument("--manifest-dir", default=str(MANIFEST_DIR), help="manifest 输出目录")
    ap.add_argument(
        "--groups", nargs="*", default=list(CATEGORY_GROUPS.keys()),
        help="要生成的大类(默认全部: %(default)s)",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    manifest_dir = Path(args.manifest_dir)

    totals = {}
    for group in args.groups:
        if group not in CATEGORY_GROUPS:
            raise ValueError(f"Unknown group '{group}', known: {list(CATEGORY_GROUPS)}")
        out_path = manifest_dir / f"insert_anything_teacher_{group}.csv"
        n = build_manifest(group, CATEGORY_GROUPS[group], dataset_path, out_path)
        totals[group] = n

    print("\nSummary:", file=sys.stderr)
    for g, n in totals.items():
        print(f"  {g:20s} {n:>8d}", file=sys.stderr)


if __name__ == "__main__":
    main()

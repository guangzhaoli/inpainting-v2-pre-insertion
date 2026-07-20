#!/usr/bin/env python3
"""Audit and visualize Hybrid Masking on real Insert Anything training samples.

The script scans the configured manifest, selects representative geometry cases
for every category, applies the actual fixed-pixels preprocessing, and renders all
three Hybrid Masking families on the processed target mask:

* Bezier augmented box;
* centered expanded box;
* dilated segmentation.

For non-person categories the dilation panel is diagnostic only: the official
Insert Anything sampling policy never selects it.  The generated artifacts are a
geometry/data-augmentation audit and do not run model inference.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import HybridMaskingConfig, ImageProcessorInitArgs
from src.data.preprocess import ImageProcessor
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_DILATED_SEGMENTATION,
    INSERT_ANYTHING_EXPANDED_BOX,
    build_insert_anything_mask_for_mode,
    sample_insert_anything_mask_mode,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/teacher/all.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/hybrid_masking_review_20260711_final"
MODES = (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_EXPANDED_BOX,
    INSERT_ANYTHING_DILATED_SEGMENTATION,
)
MODE_COLORS = {
    INSERT_ANYTHING_AUGMENTED_BOX: (0, 190, 255),
    INSERT_ANYTHING_EXPANDED_BOX: (255, 165, 0),
    INSERT_ANYTHING_DILATED_SEGMENTATION: (0, 210, 105),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--bezier-trials", type=int, default=16)
    parser.add_argument("--audit-samples-per-category", type=int, default=100)
    parser.add_argument("--panel-size", type=int, default=300)
    return parser.parse_args()


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_settings(args: argparse.Namespace):
    with args.config.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    init_args = raw["data"]["init_args"]
    adaptive_crop = init_args.get("adaptive_crop", {}) or {}
    if adaptive_crop.get("train_enabled", False):
        raise ValueError(
            "this visualization currently expects adaptive_crop.train_enabled=false; "
            "otherwise it would not reproduce the complete training geometry"
        )
    data_root = (args.data_root or Path(init_args["dataset_path"])).resolve()
    manifest = args.manifest or Path(init_args["manifest_path"])
    if not manifest.is_absolute():
        manifest = data_root / manifest

    processor_config = ImageProcessorInitArgs.model_validate(init_args["processor"]["init_args"])
    hybrid_config = HybridMaskingConfig.model_validate(init_args.get("hybrid_masking", {})).model_copy(
        update={"enabled": True}
    )
    configured_categories = [str(item) for item in init_args.get("categories", [])]
    return (
        data_root,
        manifest.resolve(),
        processor_config,
        hybrid_config,
        configured_categories,
    )


def _load_manifest(
    manifest: Path,
    configured_categories: list[str],
    max_samples: int,
) -> list[dict[str, str]]:
    allowed = set(configured_categories)
    rows: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if allowed and row.get("category", "") not in allowed:
                continue
            rows.append(dict(row))
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def _scan_mask(index: int, row: dict[str, str], data_root: Path):
    mask_path = _resolve_path(data_root, row["tar_mask"])
    image_path = _resolve_path(data_root, row["tar_image"])
    try:
        with Image.open(mask_path) as opened:
            mask_array = np.asarray(ImageOps.exif_transpose(opened).convert("L"))
        threshold = 0 if int(mask_array.max()) <= 1 else 127
        binary = mask_array > threshold
        area = int(binary.sum())
        if area <= 0:
            raise ValueError("empty target mask")

        height, width = binary.shape
        ys, xs = np.where(binary)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox_area = (x2 - x1) * (y2 - y1)
        image_area = width * height
        edge_gap_px = min(x1, y1, width - x2, height - y2)
        soft_pixels = int(np.count_nonzero((mask_array > 0) & (mask_array < 255)))
        component_count = int(cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)[0] - 1)
        return {
            "index": index,
            "category": row.get("category", ""),
            "stem": row.get("stem", Path(row["tar_mask"]).stem),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "width": width,
            "height": height,
            "mask_area_pixels": area,
            "mask_area_ratio": area / image_area,
            "bbox_area_ratio": bbox_area / image_area,
            "bbox_fill_ratio": area / bbox_area,
            "bbox_aspect_ratio": (x2 - x1) / (y2 - y1),
            "edge_gap_px": edge_gap_px,
            "edge_gap_ratio": edge_gap_px / min(width, height),
            "touches_border": edge_gap_px == 0,
            "soft_pixels": soft_pixels,
            "component_count": component_count,
        }, None
    except Exception as exc:
        return None, {
            "index": index,
            "category": row.get("category", ""),
            "stem": row.get("stem", ""),
            "mask_path": str(mask_path),
            "error": repr(exc),
        }


def scan_manifest(
    rows: list[dict[str, str]],
    data_root: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def task(item):
        index, row = item
        return _scan_mask(index, row, data_root)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, (record, error) in enumerate(
            executor.map(task, enumerate(rows)),
            start=1,
        ):
            if record is not None:
                records.append(record)
            if error is not None:
                errors.append(error)
            if completed % 5000 == 0 or completed == len(rows):
                print(
                    f"scanned={completed}/{len(rows)} valid={len(records)} errors={len(errors)}",
                    flush=True,
                )
    return records, errors


def _closest_unused(
    records: Iterable[dict[str, Any]],
    score,
    used: set[tuple[str, str]],
) -> dict[str, Any] | None:
    for record in sorted(records, key=score):
        key = (record["category"], record["stem"])
        if key not in used:
            return record
    return None


def select_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)

    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        category_records = grouped[category]
        mask_areas = np.asarray([item["mask_area_ratio"] for item in category_records])
        fill_ratios = np.asarray([item["bbox_fill_ratio"] for item in category_records])
        targets = {
            "small_p05": float(np.percentile(mask_areas, 5)),
            "medium_p50": float(np.percentile(mask_areas, 50)),
            "large_p95": float(np.percentile(mask_areas, 95)),
        }
        used: set[tuple[str, str]] = set()

        for case_name in ("small_p05", "medium_p50", "large_p95"):
            target = targets[case_name]
            chosen = _closest_unused(
                category_records,
                lambda item, target=target: abs(item["mask_area_ratio"] - target),
                used,
            )
            if chosen is not None:
                copied = dict(chosen)
                copied["case"] = case_name
                selected.append(copied)
                used.add((chosen["category"], chosen["stem"]))

        median_area = targets["medium_p50"]
        edge = _closest_unused(
            category_records,
            lambda item: (
                item["edge_gap_ratio"],
                abs(math.log(max(item["mask_area_ratio"], 1e-12)) - math.log(max(median_area, 1e-12))),
            ),
            used,
        )
        if edge is not None:
            copied = dict(edge)
            copied["case"] = "nearest_edge"
            selected.append(copied)
            used.add((edge["category"], edge["stem"]))

        fill_target = float(np.percentile(fill_ratios, 5))
        low_fill = _closest_unused(
            category_records,
            lambda item: (
                abs(item["bbox_fill_ratio"] - fill_target),
                abs(math.log(max(item["mask_area_ratio"], 1e-12)) - math.log(max(median_area, 1e-12))),
            ),
            used,
        )
        if low_fill is not None:
            copied = dict(low_fill)
            copied["case"] = "low_fill_p05"
            selected.append(copied)
            used.add((low_fill["category"], low_fill["stem"]))

        component_counts = [item["component_count"] for item in category_records]
        component_target = float(np.percentile(component_counts, 95))
        if max(component_counts) >= 2:
            component_target = max(component_target, 2.0)
        fragmented = _closest_unused(
            category_records,
            lambda item: (
                abs(item["component_count"] - component_target),
                -item["component_count"],
                abs(math.log(max(item["mask_area_ratio"], 1e-12)) - math.log(max(median_area, 1e-12))),
            ),
            used,
        )
        if fragmented is not None:
            copied = dict(fragmented)
            copied["case"] = "fragmented_p95"
            selected.append(copied)
    return selected


def _to_processed_pil(processed: dict[str, Any]) -> tuple[Image.Image, np.ndarray]:
    image_value = processed["image"]
    if isinstance(image_value, torch.Tensor):
        image_array = image_value.detach().float().cpu().numpy()
        if image_array.ndim == 3 and image_array.shape[0] in {1, 3, 4}:
            image_array = np.transpose(image_array, (1, 2, 0))
    else:
        image_array = np.asarray(image_value)
    if image_array.dtype.kind == "f" and float(image_array.max()) <= 1.0:
        image_array = image_array * 255.0
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[..., None], 3, axis=-1)
    image = Image.fromarray(image_array[..., :3], mode="RGB")

    mask_value = processed["mask"]
    if isinstance(mask_value, torch.Tensor):
        mask_array = mask_value.detach().float().cpu().numpy()
    else:
        mask_array = np.asarray(mask_value, dtype=np.float32)
    mask_array = np.squeeze(mask_array)
    threshold = 0.5 if float(mask_array.max()) <= 1.0 else 127.0
    return image, (mask_array > threshold)


def _stable_seed(record: dict[str, Any], suffix: str = "") -> int:
    key = f"{record['category']}/{record['stem']}/{suffix}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _mode_probability(category: str, mode: str, config: HybridMaskingConfig) -> float:
    is_person = category in config.person_categories
    if is_person:
        return {
            INSERT_ANYTHING_AUGMENTED_BOX: 0.4,
            INSERT_ANYTHING_EXPANDED_BOX: 0.4,
            INSERT_ANYTHING_DILATED_SEGMENTATION: 0.2,
        }[mode]
    return {
        INSERT_ANYTHING_AUGMENTED_BOX: 0.5,
        INSERT_ANYTHING_EXPANDED_BOX: 0.5,
        INSERT_ANYTHING_DILATED_SEGMENTATION: 0.0,
    }[mode]


def _effective_pixel_parameter(
    value: int,
    height: int,
    width: int,
    config: HybridMaskingConfig,
) -> int:
    if not config.scale_pixel_parameters:
        return value
    scale = min(height, width) / float(config.reference_resolution)
    return max(1, int(round(value * scale)))


def _effective_hybrid_parameters(
    height: int,
    width: int,
    config: HybridMaskingConfig,
) -> dict[str, int]:
    jitter = _effective_pixel_parameter(config.curve_jitter_px, height, width, config)
    kernel = _effective_pixel_parameter(
        config.dilation_kernel_size,
        height,
        width,
        config,
    )
    if kernel % 2 == 0:
        kernel += 1
    return {
        "curve_jitter_px": jitter,
        "dilation_kernel_size": kernel,
        "dilation_iterations": config.dilation_iterations,
    }


def _measure_masks(
    gt: np.ndarray,
    raw: np.ndarray,
    safe: np.ndarray,
) -> dict[str, Any]:
    gt = gt.astype(bool)
    raw = raw.astype(bool)
    safe = safe.astype(bool)
    gt_area = int(gt.sum())
    raw_area = int(raw.sum())
    safe_area = int(safe.sum())
    image_area = int(gt.size)
    raw_intersection = int(np.logical_and(raw, gt).sum())
    safe_intersection = int(np.logical_and(safe, gt).sum())
    safe_union = int(np.logical_or(safe, gt).sum())
    rescued = int(np.logical_and(gt, ~raw).sum())
    return {
        "gt_area_pixels": gt_area,
        "raw_area_pixels": raw_area,
        "condition_area_pixels": safe_area,
        "condition_area_ratio": safe_area / image_area,
        "area_expansion_ratio": safe_area / gt_area,
        "raw_gt_coverage": raw_intersection / gt_area,
        "safe_gt_coverage": safe_intersection / gt_area,
        "safe_iou_with_gt": safe_intersection / safe_union,
        "added_outside_gt_pixels": int(np.logical_and(safe, ~gt).sum()),
        "safety_union_rescued_pixels": rescued,
        "condition_touches_border": bool(
            safe[0, :].any() or safe[-1, :].any() or safe[:, 0].any() or safe[:, -1].any()
        ),
        "binary_output": bool(np.all(np.isin(np.unique(safe.astype(np.uint8)), [0, 1]))),
    }


def _blend(array: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    if not np.any(mask):
        return
    color_array = np.asarray(color, dtype=np.float32)
    array[mask] = array[mask] * (1.0 - alpha) + color_array * alpha


def _overlay_mode(
    image: Image.Image,
    gt: np.ndarray,
    raw: np.ndarray,
    safe: np.ndarray,
    mode: str,
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    gt = gt.astype(bool)
    raw = raw.astype(bool)
    safe = safe.astype(bool)
    _blend(array, gt, (255, 40, 40), 0.28)
    _blend(array, np.logical_and(safe, ~gt), MODE_COLORS[mode], 0.52)
    _blend(array, np.logical_and(gt, ~raw), (255, 0, 255), 0.72)
    boundary = cv2.morphologyEx(
        gt.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    array[boundary] = np.asarray((255, 255, 255), dtype=np.float32)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _overlay_gt(image: Image.Image, gt: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    _blend(array, gt.astype(bool), (255, 40, 40), 0.48)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _condition_hole(image: Image.Image, condition: np.ndarray) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    array[condition.astype(bool)] = 0
    return Image.fromarray(array, mode="RGB")


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _panel(image: Image.Image, label: str, panel_size: int) -> Image.Image:
    label_height = 88
    panel = Image.new("RGB", (panel_size, panel_size + label_height), (246, 246, 246))
    rendered = image.copy().convert("RGB")
    rendered.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
    x = (panel_size - rendered.width) // 2
    y = label_height + (panel_size - rendered.height) // 2
    panel.paste(rendered, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel_size, label_height), fill=(235, 235, 235))
    draw.multiline_text((6, 5), label, fill=(20, 20, 20), font=_font(12), spacing=2)
    draw.rectangle((0, 0, panel.width - 1, panel.height - 1), outline=(205, 205, 205))
    return panel


def _row(panels: list[tuple[str, Image.Image]], panel_size: int) -> Image.Image:
    gap = 8
    rendered = [_panel(image, label, panel_size) for label, image in panels]
    width = sum(item.width for item in rendered) + gap * (len(rendered) - 1)
    height = max(item.height for item in rendered)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for item in rendered:
        canvas.paste(item, (x, 0))
        x += item.width + gap
    return canvas


def _stack_with_header(
    rows: list[Image.Image],
    title: str,
    output_path: Path,
) -> None:
    gap = 10
    header_height = 82
    width = max(row.width for row in rows)
    height = header_height + sum(row.height for row in rows) + gap * max(0, len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((8, 8), title, fill=(15, 15, 15), font=_font(14), spacing=3)
    y = header_height
    for row in rows:
        canvas.paste(row, ((width - row.width) // 2, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def prepare_case(
    record: dict[str, Any],
    processor: ImageProcessor,
    hybrid_config: HybridMaskingConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with Image.open(record["image_path"]) as opened:
        raw_image = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(record["mask_path"]) as opened:
        raw_mask = ImageOps.exif_transpose(opened).convert("L")

    processed = processor.preprocess({"image": raw_image.copy(), "mask": raw_mask.copy()})
    image, gt = _to_processed_pil(processed)
    if not np.any(gt):
        raise ValueError(f"processed mask became empty: {record['category']}/{record['stem']}")

    raw_config = hybrid_config.model_copy(update={"ensure_gt_coverage": False})
    mode_outputs: dict[str, dict[str, Any]] = {}
    bezier_variants: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    effective_parameters = _effective_hybrid_parameters(
        image.height,
        image.width,
        hybrid_config,
    )
    for mode in MODES:
        variant_count = 3 if mode == INSERT_ANYTHING_AUGMENTED_BOX else 1
        for variant_index in range(variant_count):
            seed = _stable_seed(record, f"{mode}_visual_{variant_index}")
            raw_condition = build_insert_anything_mask_for_mode(
                gt.astype(np.float32),
                mode,
                raw_config,
                rng=random.Random(seed),
            ).astype(bool)
            safe_condition = build_insert_anything_mask_for_mode(
                gt.astype(np.float32),
                mode,
                hybrid_config,
                rng=random.Random(seed),
            ).astype(bool)
            metrics = _measure_masks(gt, raw_condition, safe_condition)
            metrics.update(
                {
                    "category": record["category"],
                    "stem": record["stem"],
                    "case": record["case"],
                    "mode": mode,
                    "visual_variant": variant_index,
                    "seed": seed,
                    "official_probability": _mode_probability(record["category"], mode, hybrid_config),
                    "official_for_category": _mode_probability(record["category"], mode, hybrid_config) > 0,
                    "processed_width": image.width,
                    "processed_height": image.height,
                    **effective_parameters,
                }
            )
            metric_rows.append(metrics)
            output = {
                "raw": raw_condition,
                "safe": safe_condition,
                "metrics": metrics,
                "overlay": _overlay_mode(
                    image,
                    gt,
                    raw_condition,
                    safe_condition,
                    mode,
                ),
                "hole": _condition_hole(image, safe_condition),
            }
            if mode == INSERT_ANYTHING_AUGMENTED_BOX:
                bezier_variants.append(output)
            if variant_index == 0:
                mode_outputs[mode] = output

    prepared = {
        "image": image,
        "gt": gt,
        "gt_overlay": _overlay_gt(image, gt),
        "mode_outputs": mode_outputs,
        "bezier_variants": bezier_variants,
        "effective_parameters": effective_parameters,
        "original_size": raw_image.size,
        "processed_size": image.size,
    }
    return prepared, metric_rows


def _case_row(
    record: dict[str, Any],
    prepared: dict[str, Any],
    panel_size: int,
) -> Image.Image:
    gt_ratio = prepared["gt"].mean() * 100.0
    panels: list[tuple[str, Image.Image]] = [
        (
            f"{record['case']} | {record['category']}/{record['stem']}\n"
            f"raw {record['width']}x{record['height']} -> proc "
            f"{prepared['processed_size'][0]}x{prepared['processed_size'][1]}\n"
            f"GT={gt_ratio:.3f}% fill={record['bbox_fill_ratio'] * 100:.1f}% "
            f"edge={record['edge_gap_px']}px comps={record['component_count']}",
            prepared["gt_overlay"],
        )
    ]

    for variant_index, output in enumerate(prepared["bezier_variants"]):
        metrics = output["metrics"]
        applicability = (
            f"official p={metrics['official_probability']:.1f}"
            if metrics["official_for_category"]
            else "diagnostic only (official p=0)"
        )
        label = (
            f"Bezier seed {variant_index} | {applicability}\n"
            f"jitter={metrics['curve_jitter_px']}px seed={metrics['seed'] & 0xFFFFFF:06x}\n"
            f"area x{metrics['area_expansion_ratio']:.2f}, "
            f"cond={metrics['condition_area_ratio'] * 100:.2f}%\n"
            f"rawCov={metrics['raw_gt_coverage'] * 100:.2f}% "
            f"rescue={metrics['safety_union_rescued_pixels']}px"
        )
        panels.append((label, output["overlay"]))

    for mode, display_name in (
        (INSERT_ANYTHING_EXPANDED_BOX, "Expanded box 1.1x"),
        (INSERT_ANYTHING_DILATED_SEGMENTATION, "Dilation"),
    ):
        output = prepared["mode_outputs"][mode]
        metrics = output["metrics"]
        applicability = (
            f"official p={metrics['official_probability']:.1f}"
            if metrics["official_for_category"]
            else "diagnostic only (official p=0)"
        )
        parameter_label = (
            "bbox ratio=1.1"
            if mode == INSERT_ANYTHING_EXPANDED_BOX
            else (
                f"kernel={metrics['dilation_kernel_size']}x"
                f"{metrics['dilation_kernel_size']} iter={metrics['dilation_iterations']}"
            )
        )
        label = (
            f"{display_name} | {applicability}\n"
            f"{parameter_label}\n"
            f"area x{metrics['area_expansion_ratio']:.2f}, "
            f"cond={metrics['condition_area_ratio'] * 100:.2f}%\n"
            f"rawCov={metrics['raw_gt_coverage'] * 100:.2f}% "
            f"rescue={metrics['safety_union_rescued_pixels']}px"
        )
        panels.append((label, output["overlay"]))
    return _row(panels, panel_size)


def _case_hole_row(
    record: dict[str, Any],
    prepared: dict[str, Any],
    panel_size: int,
) -> Image.Image:
    panels: list[tuple[str, Image.Image]] = [
        (
            f"Actual masked-target conditioning\n"
            f"{record['category']}/{record['stem']} | {record['case']}\n"
            "black pixels are removed from tar_image",
            prepared["image"],
        )
    ]
    for mode, display_name in (
        (INSERT_ANYTHING_AUGMENTED_BOX, "Bezier seed 0 hole"),
        (INSERT_ANYTHING_EXPANDED_BOX, "Expanded-box hole"),
        (INSERT_ANYTHING_DILATED_SEGMENTATION, "Dilation hole"),
    ):
        output = prepared["mode_outputs"][mode]
        metrics = output["metrics"]
        applicability = (
            f"official p={metrics['official_probability']:.1f}"
            if metrics["official_for_category"]
            else "forced diagnostic p=0"
        )
        panels.append(
            (
                f"{display_name} | {applicability}\n"
                f"removed={metrics['condition_area_ratio'] * 100:.2f}% "
                f"context={100 * (1 - metrics['condition_area_ratio']):.2f}%\n"
                f"area x{metrics['area_expansion_ratio']:.2f}",
                output["hole"],
            )
        )
    return _row(panels, panel_size)


def render_selected(
    selected: list[dict[str, Any]],
    processor_config: ImageProcessorInitArgs,
    hybrid_config: HybridMaskingConfig,
    output_dir: Path,
    panel_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    processor = ImageProcessor(processor_config)
    prepared_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    preparation_errors: list[dict[str, Any]] = []
    category_rows: dict[str, list[Image.Image]] = defaultdict(list)
    category_hole_rows: dict[str, list[Image.Image]] = defaultdict(list)
    category_medium_rows: list[tuple[str, Image.Image]] = []
    category_medium_hole_rows: list[tuple[str, Image.Image]] = []
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    for completed, record in enumerate(selected, start=1):
        try:
            prepared, case_metrics = prepare_case(record, processor, hybrid_config)
            key = (record["category"], record["stem"])
            prepared_by_key[key] = prepared
            metric_rows.extend(case_metrics)
            overlay_row = _case_row(record, prepared, panel_size)
            hole_row = _case_hole_row(record, prepared, panel_size)
            category_rows[record["category"]].append(overlay_row)
            category_hole_rows[record["category"]].append(hole_row)
            _stack_with_header(
                [overlay_row, hole_row],
                "Hybrid Masking mask geometry + actual masked-target control",
                cases_dir / f"{record['category']}__{record['case']}__{record['stem']}.png",
            )
            if record["case"] == "medium_p50":
                category_medium_rows.append((record["category"], overlay_row))
                category_medium_hole_rows.append((record["category"], hole_row))
        except Exception as exc:
            preparation_errors.append(
                {
                    "category": record["category"],
                    "stem": record["stem"],
                    "case": record["case"],
                    "error": repr(exc),
                }
            )
        if completed % 10 == 0 or completed == len(selected):
            print(
                f"rendered={completed}/{len(selected)} errors={len(preparation_errors)}",
                flush=True,
            )

    legend = (
        "Hybrid Masking on processed training inputs (no model inference)\n"
        "red=GT, white=GT boundary, mode color=conditioning area outside GT, "
        "magenta=GT pixels recovered by safety union"
    )
    for category, rows in sorted(category_rows.items()):
        _stack_with_header(
            rows,
            f"Category: {category}\n{legend}",
            output_dir / "categories" / f"{category}.png",
        )
        _stack_with_header(
            category_hole_rows[category],
            f"Category: {category} | actual tar_image * (1 - tar_mask_cond)\n"
            "Dilation is forced diagnostic for categories with official p=0",
            output_dir / "categories_holes" / f"{category}.png",
        )
    if category_medium_rows:
        _stack_with_header(
            [row for _, row in sorted(category_medium_rows)],
            "All categories, median-area representative per category\n" + legend,
            output_dir / "overview.png",
        )
        _stack_with_header(
            [row for _, row in sorted(category_medium_hole_rows)],
            "All categories, median-area actual masked-target controls\n"
            "Black region is removed from the completed target before conditioning",
            output_dir / "overview_holes.png",
        )
    return metric_rows, preparation_errors, prepared_by_key


def run_bezier_trials(
    selected: list[dict[str, Any]],
    prepared_by_key: dict[tuple[str, str], dict[str, Any]],
    hybrid_config: HybridMaskingConfig,
    trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_config = hybrid_config.model_copy(update={"ensure_gt_coverage": False})
    rows: list[dict[str, Any]] = []
    variability_rows: list[dict[str, Any]] = []
    for record in selected:
        prepared = prepared_by_key.get((record["category"], record["stem"]))
        if prepared is None:
            continue
        gt = prepared["gt"]
        safe_masks: list[np.ndarray] = []
        case_rows: list[dict[str, Any]] = []
        for trial in range(trials):
            seed = _stable_seed(record, f"bezier_trial_{trial}")
            raw = build_insert_anything_mask_for_mode(
                gt.astype(np.float32),
                INSERT_ANYTHING_AUGMENTED_BOX,
                raw_config,
                rng=random.Random(seed),
            ).astype(bool)
            safe = build_insert_anything_mask_for_mode(
                gt.astype(np.float32),
                INSERT_ANYTHING_AUGMENTED_BOX,
                hybrid_config,
                rng=random.Random(seed),
            ).astype(bool)
            metrics = _measure_masks(gt, raw, safe)
            metrics.update(
                {
                    "category": record["category"],
                    "stem": record["stem"],
                    "case": record["case"],
                    "trial": trial,
                    "seed": seed,
                }
            )
            rows.append(metrics)
            case_rows.append(metrics)
            safe_masks.append(safe)

        pairwise_ious = []
        for first_index in range(len(safe_masks)):
            for second_index in range(first_index + 1, len(safe_masks)):
                first = safe_masks[first_index]
                second = safe_masks[second_index]
                union = int(np.logical_or(first, second).sum())
                intersection = int(np.logical_and(first, second).sum())
                pairwise_ious.append(intersection / union if union else 1.0)
        pairwise = _quantiles(pairwise_ious)
        area = _quantiles(row["area_expansion_ratio"] for row in case_rows)
        raw_coverage = _quantiles(row["raw_gt_coverage"] for row in case_rows)
        variability_rows.append(
            {
                "category": record["category"],
                "stem": record["stem"],
                "case": record["case"],
                "trials": trials,
                "pairwise_iou_min": pairwise.get("min", 1.0),
                "pairwise_iou_median": pairwise.get("median", 1.0),
                "pairwise_iou_p95": pairwise.get("p95", 1.0),
                "area_expansion_min": area["min"],
                "area_expansion_median": area["median"],
                "area_expansion_max": area["max"],
                "raw_gt_coverage_min": raw_coverage["min"],
                "rescued_trial_fraction": sum(row["safety_union_rescued_pixels"] > 0 for row in case_rows)
                / len(case_rows),
            }
        )
    return rows, variability_rows


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _select_audit_records(
    records: list[dict[str, Any]],
    per_category: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    selected: list[dict[str, Any]] = []
    for category in sorted(grouped):
        items = sorted(grouped[category], key=lambda item: item["index"])
        if len(items) <= per_category:
            selected.extend(items)
            continue
        indices = np.linspace(0, len(items) - 1, per_category, dtype=int)
        selected.extend(items[int(index)] for index in indices)
    return selected


def run_stratified_audit(
    records: list[dict[str, Any]],
    processor_config: ImageProcessorInitArgs,
    hybrid_config: HybridMaskingConfig,
    per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processor = ImageProcessor(processor_config)
    raw_config = hybrid_config.model_copy(update={"ensure_gt_coverage": False})
    audit_records = _select_audit_records(records, per_category)
    metric_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for completed, record in enumerate(audit_records, start=1):
        try:
            with Image.open(record["mask_path"]) as opened:
                raw_mask = ImageOps.exif_transpose(opened).convert("L")
            placeholder = Image.new("RGB", raw_mask.size, (0, 0, 0))
            processed = processor.preprocess({"image": placeholder, "mask": raw_mask.copy()})
            _, gt = _to_processed_pil(processed)
            effective = _effective_hybrid_parameters(
                gt.shape[0],
                gt.shape[1],
                hybrid_config,
            )
            for mode in MODES:
                seed = _stable_seed(record, f"audit_{mode}")
                raw = build_insert_anything_mask_for_mode(
                    gt.astype(np.float32),
                    mode,
                    raw_config,
                    rng=random.Random(seed),
                ).astype(bool)
                safe = build_insert_anything_mask_for_mode(
                    gt.astype(np.float32),
                    mode,
                    hybrid_config,
                    rng=random.Random(seed),
                ).astype(bool)
                metrics = _measure_masks(gt, raw, safe)
                metrics.update(
                    {
                        "category": record["category"],
                        "stem": record["stem"],
                        "mode": mode,
                        "official_probability": _mode_probability(record["category"], mode, hybrid_config),
                        "official_for_category": _mode_probability(record["category"], mode, hybrid_config) > 0,
                        "processed_width": gt.shape[1],
                        "processed_height": gt.shape[0],
                        **effective,
                    }
                )
                metric_rows.append(metrics)
        except Exception as exc:
            errors.append(
                {
                    "category": record["category"],
                    "stem": record["stem"],
                    "error": repr(exc),
                }
            )
        if completed % 100 == 0 or completed == len(audit_records):
            print(
                f"audit_processed={completed}/{len(audit_records)} errors={len(errors)}",
                flush=True,
            )
    return metric_rows, errors


def summarize_category_modes(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["mode"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (category, mode), items in sorted(grouped.items()):
        condition = _quantiles(item["condition_area_ratio"] for item in items)
        expansion = _quantiles(item["area_expansion_ratio"] for item in items)
        raw_coverage = _quantiles(item["raw_gt_coverage"] for item in items)
        summary_rows.append(
            {
                "category": category,
                "mode": mode,
                "count": len(items),
                "official_probability": items[0]["official_probability"],
                "condition_fraction_p05": condition["p05"],
                "condition_fraction_median": condition["median"],
                "condition_fraction_p95": condition["p95"],
                "condition_fraction_max": condition["max"],
                "area_expansion_p05": expansion["p05"],
                "area_expansion_median": expansion["median"],
                "area_expansion_p95": expansion["p95"],
                "area_expansion_max": expansion["max"],
                "raw_gt_coverage_min": raw_coverage["min"],
                "safe_gt_coverage_min": min(item["safe_gt_coverage"] for item in items),
                "condition_fraction_gt_0_60": sum(item["condition_area_ratio"] > 0.60 for item in items),
                "condition_fraction_gt_0_80": sum(item["condition_area_ratio"] > 0.80 for item in items),
                "area_expansion_gt_10": sum(item["area_expansion_ratio"] > 10.0 for item in items),
                "safety_union_rescued_samples": sum(item["safety_union_rescued_pixels"] > 0 for item in items),
            }
        )
    return summary_rows


def simulate_mode_frequencies(
    hybrid_config: HybridMaskingConfig,
    draws: int = 100000,
) -> dict[str, Any]:
    output = {}
    for profile, category in (("person", "person"), ("nonperson", "accessory")):
        rng = random.Random(20260711)
        counts = Counter(sample_insert_anything_mask_mode(category, hybrid_config, rng=rng) for _ in range(draws))
        output[profile] = {
            "draws": draws,
            "counts": dict(sorted(counts.items())),
            "fractions": {mode: counts.get(mode, 0) / draws for mode in MODES},
        }
    return output


def render_dataset_statistics(
    records: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    categories = sorted(grouped)
    summary_rows: list[dict[str, Any]] = []
    for category in categories:
        items = grouped[category]
        mask_q = _quantiles(item["mask_area_ratio"] for item in items)
        fill_q = _quantiles(item["bbox_fill_ratio"] for item in items)
        component_q = _quantiles(item["component_count"] for item in items)
        summary_rows.append(
            {
                "category": category,
                "count": len(items),
                "mask_area_p05": mask_q["p05"],
                "mask_area_median": mask_q["median"],
                "mask_area_p95": mask_q["p95"],
                "bbox_fill_p05": fill_q["p05"],
                "bbox_fill_median": fill_q["median"],
                "bbox_fill_p95": fill_q["p95"],
                "border_touch_rate": sum(item["touches_border"] for item in items) / len(items),
                "soft_mask_rate": sum(item["soft_pixels"] > 0 for item in items) / len(items),
                "multi_component_rate": sum(item["component_count"] > 1 for item in items) / len(items),
                "component_count_p95": component_q["p95"],
                "component_count_max": component_q["max"],
            }
        )

    x = np.arange(len(categories))
    figure, axes = plt.subplots(2, 2, figsize=(18, 12))
    counts = [row["count"] for row in summary_rows]
    axes[0, 0].bar(x, counts, color="#4C78A8")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Manifest samples per category (log scale)")
    axes[0, 0].set_xticks(x, categories, rotation=45, ha="right")

    for row, color, label in (
        ("mask_area_p05", "#72B7B2", "p05"),
        ("mask_area_median", "#F58518", "median"),
        ("mask_area_p95", "#E45756", "p95"),
    ):
        axes[0, 1].plot(
            x,
            [item[row] * 100 for item in summary_rows],
            marker="o",
            label=label,
            color=color,
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("GT mask area percentiles (%)")
    axes[0, 1].set_xticks(x, categories, rotation=45, ha="right")
    axes[0, 1].legend()

    for row, color, label in (
        ("bbox_fill_p05", "#72B7B2", "p05"),
        ("bbox_fill_median", "#F58518", "median"),
        ("bbox_fill_p95", "#E45756", "p95"),
    ):
        axes[1, 0].plot(
            x,
            [item[row] * 100 for item in summary_rows],
            marker="o",
            label=label,
            color=color,
        )
    axes[1, 0].set_title("GT mask fill inside tight bbox (%)")
    axes[1, 0].set_xticks(x, categories, rotation=45, ha="right")
    axes[1, 0].legend()

    bar_width = 0.42
    axes[1, 1].bar(
        x - bar_width / 2,
        [item["border_touch_rate"] * 100 for item in summary_rows],
        width=bar_width,
        color="#B279A2",
        label="touches border",
    )
    axes[1, 1].bar(
        x + bar_width / 2,
        [item["multi_component_rate"] * 100 for item in summary_rows],
        width=bar_width,
        color="#59A14F",
        label="multiple components",
    )
    axes[1, 1].set_title("Border-touching and multi-component GT masks (%)")
    axes[1, 1].set_xticks(x, categories, rotation=45, ha="right")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].legend()

    figure.suptitle(
        "Hybrid Masking audit: source-mask geometry across the full manifest",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "dataset_statistics.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return summary_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_mode_metrics(
    metric_rows: list[dict[str, Any]],
    bezier_trials: list[dict[str, Any]],
    bezier_variability: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_mode[row["mode"]].append(row)
    mode_summary = {}
    for mode, rows in sorted(by_mode.items()):
        mode_summary[mode] = {
            "count": len(rows),
            "area_expansion_ratio": _quantiles(row["area_expansion_ratio"] for row in rows),
            "condition_area_ratio": _quantiles(row["condition_area_ratio"] for row in rows),
            "raw_gt_coverage": _quantiles(row["raw_gt_coverage"] for row in rows),
            "safe_gt_coverage_min": min(row["safe_gt_coverage"] for row in rows),
            "safety_union_rescued_samples": sum(row["safety_union_rescued_pixels"] > 0 for row in rows),
            "border_touch_samples": sum(row["condition_touches_border"] for row in rows),
        }

    all_checked_rows = [*metric_rows, *audit_rows]
    return {
        "selected_mode_summary": mode_summary,
        "global_checks": {
            "selected_metric_rows": len(metric_rows),
            "audit_metric_rows": len(audit_rows),
            "safe_coverage_min": min(
                (row["safe_gt_coverage"] for row in all_checked_rows),
                default=None,
            ),
            "binary_failures": sum(not row["binary_output"] for row in all_checked_rows),
            "area_shrink_failures": sum(
                row["condition_area_pixels"] < row["gt_area_pixels"] for row in all_checked_rows
            ),
            "bezier_trial_rows": len(bezier_trials),
            "bezier_trial_raw_coverage": _quantiles(row["raw_gt_coverage"] for row in bezier_trials),
            "bezier_trial_area_expansion": _quantiles(row["area_expansion_ratio"] for row in bezier_trials),
            "bezier_trial_rescued_fraction": (
                sum(row["safety_union_rescued_pixels"] > 0 for row in bezier_trials) / len(bezier_trials)
                if bezier_trials
                else 0.0
            ),
            "bezier_pairwise_iou": _quantiles(row["pairwise_iou_median"] for row in bezier_variability),
            "official_policy_condition_fraction_gt_0_60": sum(
                row["official_for_category"] and row["condition_area_ratio"] > 0.60 for row in audit_rows
            ),
            "official_policy_condition_fraction_gt_0_80": sum(
                row["official_for_category"] and row["condition_area_ratio"] > 0.80 for row in audit_rows
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.bezier_trials <= 0:
        raise ValueError("--bezier-trials must be positive")
    if args.audit_samples_per_category <= 0:
        raise ValueError("--audit-samples-per-category must be positive")
    if args.panel_size < 128:
        raise ValueError("--panel-size must be at least 128")

    (
        data_root,
        manifest,
        processor_config,
        hybrid_config,
        configured_categories,
    ) = load_settings(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"config={args.config.resolve()}")
    print(f"manifest={manifest}")
    print(f"data_root={data_root}")
    print(f"output_dir={output_dir}")
    rows = _load_manifest(manifest, configured_categories, args.max_samples)
    print(f"manifest_rows={len(rows)} categories={sorted(set(r['category'] for r in rows))}")

    records, scan_errors = scan_manifest(rows, data_root, args.workers)
    if not records:
        raise RuntimeError("no valid target masks were scanned")
    selected = select_cases(records)
    if not selected:
        raise RuntimeError("no representative cases could be selected")

    dataset_summary = render_dataset_statistics(records, output_dir)
    metric_rows, preparation_errors, prepared_by_key = render_selected(
        selected,
        processor_config,
        hybrid_config,
        output_dir,
        args.panel_size,
    )
    bezier_trials, bezier_variability = run_bezier_trials(
        selected,
        prepared_by_key,
        hybrid_config,
        args.bezier_trials,
    )
    audit_rows, audit_errors = run_stratified_audit(
        records,
        processor_config,
        hybrid_config,
        args.audit_samples_per_category,
    )
    category_mode_summary = summarize_category_modes(audit_rows)
    mode_frequencies = simulate_mode_frequencies(hybrid_config)

    write_csv(selected, output_dir / "selected_cases.csv")
    write_csv(metric_rows, output_dir / "mode_metrics.csv")
    write_csv(bezier_trials, output_dir / "bezier_trials.csv")
    write_csv(bezier_variability, output_dir / "bezier_variability.csv")
    write_csv(audit_rows, output_dir / "audit_mode_metrics.csv")
    write_csv(category_mode_summary, output_dir / "category_mode_summary.csv")
    write_csv(dataset_summary, output_dir / "category_dataset_summary.csv")
    write_csv(scan_errors, output_dir / "scan_errors.csv")
    write_csv(preparation_errors, output_dir / "preparation_errors.csv")
    write_csv(audit_errors, output_dir / "audit_errors.csv")

    summary = {
        "config": str(args.config.resolve()),
        "manifest": str(manifest),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "manifest_rows": len(rows),
        "valid_records": len(records),
        "scan_errors": len(scan_errors),
        "category_counts": dict(sorted(Counter(r["category"] for r in records).items())),
        "selected_cases": len(selected),
        "preparation_errors": len(preparation_errors),
        "audit_errors": len(audit_errors),
        "hybrid_config": hybrid_config.model_dump(),
        "dataset_summary": dataset_summary,
        "category_mode_summary": category_mode_summary,
        "mode_frequency_simulation": mode_frequencies,
        **summarize_mode_metrics(
            metric_rows,
            bezier_trials,
            bezier_variability,
            audit_rows,
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"selected_cases={len(selected)}")
    print(f"mode_metrics={len(metric_rows)}")
    print(f"bezier_trials={len(bezier_trials)}")
    print(f"audit_mode_metrics={len(audit_rows)}")
    print(
        f"scan_errors={len(scan_errors)} preparation_errors={len(preparation_errors)} "
        f"audit_errors={len(audit_errors)}"
    )
    print(json.dumps(summary["global_checks"], ensure_ascii=False, indent=2))
    print(f"overview={output_dir / 'overview.png'}")
    print(f"summary={output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

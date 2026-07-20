#!/usr/bin/env python3
"""Visualize Adaptive Crop geometry on real Insert Anything samples.

The script scans a manifest, selects representative geometry cases, and writes:

- ``overview.png``: original target, crop, processed target/mask, mask-scale comparison;
- ``dataset_statistics.png``: dataset-level crop/zoom distributions;
- ``cases/*.png``: one detailed row for every selected sample;
- ``selected_cases.csv`` and ``summary.json``: reproducible paths and metrics.

This visualizes preprocessing geometry only. It does not claim model-generation quality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.config import AdaptiveCropConfig, ImageProcessorInitArgs
from src.data.preprocess import ImageProcessor
from src.utils.adaptive_crop import AdaptiveCropPlan, build_adaptive_crop_plan, crop_pil_pair


DEFAULT_CONFIG = REPO_ROOT / "configs/exp/accessory/adaptive_crop.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "output/adaptive_crop_review_20260711"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--crop-curve-gamma",
        type=float,
        default=1.0,
        help=(
            "Visualization-only exponent for the bbox-area crop curve. "
            "1.0 reproduces the current implementation; 2.0 simulates the proposed ablation."
        ),
    )
    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=None,
        help="Optional gamma=1 summary.json whose selected cases are reused for a paired comparison.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum manifest rows to scan; 0 scans the complete manifest.",
    )
    return parser.parse_args()


def load_settings(args: argparse.Namespace):
    with args.config.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    init_args = raw["data"]["init_args"]
    crop_config = AdaptiveCropConfig.model_validate(init_args.get("adaptive_crop", {}))
    processor_config = ImageProcessorInitArgs.model_validate(init_args["processor"]["init_args"])

    data_root = args.data_root or Path(init_args["dataset_path"])
    manifest = args.manifest or Path(init_args["manifest_path"])
    if not manifest.is_absolute():
        manifest = data_root / manifest
    return crop_config, processor_config, data_root.resolve(), manifest.resolve()


def _resolve_sample_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _build_visualization_crop_plan(
    image_size,
    mask,
    crop_config: AdaptiveCropConfig,
    crop_curve_gamma: float,
) -> AdaptiveCropPlan:
    """Build the current plan or a visualization-only gamma-curve variant.

    The production implementation is untouched. Gamma only changes the requested
    crop-area interpolation; thresholding, bbox extraction, margin constraints,
    target aspect, boundary shifting, and metadata semantics stay identical.
    """
    base_plan = build_adaptive_crop_plan(image_size, mask, crop_config)
    if math.isclose(crop_curve_gamma, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return base_plan

    width, height = base_plan.original_size
    x1, y1, x2, y2 = base_plan.mask_bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    trigger_ratio = float(crop_config.trigger_bbox_area_ratio)
    min_crop_ratio = float(crop_config.min_crop_area_ratio)

    if base_plan.bbox_area_ratio >= trigger_ratio:
        requested_crop_ratio = 1.0
        initial_reason = "mask_bbox_large_enough"
    else:
        normalized_bbox_ratio = base_plan.bbox_area_ratio / trigger_ratio
        requested_crop_ratio = min_crop_ratio + (1.0 - min_crop_ratio) * (
            normalized_bbox_ratio ** crop_curve_gamma
        )
        initial_reason = "adaptive_crop"

    margin = float(crop_config.context_margin_ratio)
    min_crop_side = int(crop_config.min_crop_side)
    expanded_bbox_width = bbox_width * (1.0 + 2.0 * margin)
    expanded_bbox_height = bbox_height * (1.0 + 2.0 * margin)
    requested_scale = math.sqrt(requested_crop_ratio)
    required_scale = max(
        expanded_bbox_width / width,
        expanded_bbox_height / height,
        min_crop_side / width,
        min_crop_side / height,
    )
    crop_scale = min(1.0, max(requested_scale, required_scale))
    crop_width = min(
        width,
        max(min_crop_side, bbox_width, int(math.ceil(width * crop_scale))),
    )
    crop_height = min(
        height,
        max(min_crop_side, bbox_height, int(math.ceil(height * crop_scale))),
    )

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    crop_x1 = int(round(center_x - crop_width / 2.0))
    crop_y1 = int(round(center_y - crop_height / 2.0))
    crop_x1 = min(max(crop_x1, 0), width - crop_width)
    crop_y1 = min(max(crop_y1, 0), height - crop_height)
    crop_x2 = crop_x1 + crop_width
    crop_y2 = crop_y1 + crop_height

    if not (crop_x1 <= x1 and crop_y1 <= y1 and crop_x2 >= x2 and crop_y2 >= y2):
        raise ValueError(
            "gamma crop failed to contain the complete target mask bbox: "
            f"bbox={(x1, y1, x2, y2)}, crop={(crop_x1, crop_y1, crop_x2, crop_y2)}"
        )

    crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
    applied = crop_box != (0, 0, width, height)
    if applied:
        decision_reason = initial_reason
    elif initial_reason == "adaptive_crop":
        decision_reason = "geometry_requires_full_frame"
    else:
        decision_reason = initial_reason

    image_area = width * height
    actual_crop_ratio = crop_width * crop_height / image_area
    return AdaptiveCropPlan(
        applied=applied,
        decision_reason=decision_reason,
        original_size=base_plan.original_size,
        mask_bbox=base_plan.mask_bbox,
        crop_box=crop_box,
        mask_area_pixels=base_plan.mask_area_pixels,
        mask_area_ratio=base_plan.mask_area_ratio,
        bbox_area_ratio=base_plan.bbox_area_ratio,
        bbox_fill_ratio=base_plan.bbox_fill_ratio,
        requested_crop_area_ratio=requested_crop_ratio,
        actual_crop_area_ratio=actual_crop_ratio,
        mask_threshold=base_plan.mask_threshold,
        min_crop_side=base_plan.min_crop_side,
        effective_zoom_x=width / crop_width,
        effective_zoom_y=height / crop_height,
    )


def _scan_manifest(
    manifest: Path,
    data_root: Path,
    crop_config: AdaptiveCropConfig,
    crop_curve_gamma: float,
    max_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if max_samples > 0 and index >= max_samples:
                break
            mask_path = _resolve_sample_path(data_root, row["tar_mask"])
            image_path = _resolve_sample_path(data_root, row["tar_image"])
            try:
                with Image.open(mask_path) as opened_mask:
                    mask = opened_mask.convert("L")
                    mask_array = np.asarray(mask)
                    plan = _build_visualization_crop_plan(
                        mask.size,
                        mask_array,
                        crop_config,
                        crop_curve_gamma,
                    )
            except Exception as exc:  # report bad data without hiding it
                errors.append(
                    {
                        "index": str(index),
                        "category": row.get("category", ""),
                        "stem": row.get("stem", ""),
                        "mask_path": str(mask_path),
                        "error": repr(exc),
                    }
                )
                continue

            width, height = plan.original_size
            x1, y1, x2, y2 = plan.mask_bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            crop_x1, crop_y1, crop_x2, crop_y2 = plan.crop_box
            crop_area = (crop_x2 - crop_x1) * (crop_y2 - crop_y1)
            edge_gap_ratio = min(
                x1 / width,
                y1 / height,
                (width - x2) / width,
                (height - y2) / height,
            )
            soft_pixels = int(np.count_nonzero((mask_array > 0) & (mask_array < 255)))
            total_pixels = width * height
            post_crop_mask_ratio = plan.mask_area_pixels / crop_area
            records.append(
                {
                    "index": index,
                    "category": row.get("category", ""),
                    "stem": row.get("stem", ""),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "width": width,
                    "height": height,
                    "bbox_aspect_ratio": bbox_width / bbox_height,
                    "edge_gap_ratio": edge_gap_ratio,
                    "soft_pixels": soft_pixels,
                    "soft_pixel_ratio": soft_pixels / total_pixels,
                    "post_crop_mask_area_ratio": post_crop_mask_ratio,
                    "mask_area_gain": post_crop_mask_ratio / plan.mask_area_ratio,
                    "plan": plan.to_dict(),
                }
            )
            if (index + 1) % 2000 == 0:
                print(f"scanned={index + 1} valid={len(records)} errors={len(errors)}", flush=True)
    return records, errors


def _ranked(records: list[dict[str, Any]], predicate, score) -> list[dict[str, Any]]:
    return sorted((record for record in records if predicate(record)), key=score)


def _select_cases(
    records: list[dict[str, Any]], crop_config: AdaptiveCropConfig
) -> list[dict[str, Any]]:
    trigger = crop_config.trigger_bbox_area_ratio
    applied = lambda record: bool(record["plan"]["applied"])
    not_applied = lambda record: not applied(record)

    specifications: list[
        tuple[str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], Any]]
    ] = [
        (
            "tiny_center_max_zoom",
            lambda r: applied(r) and r["edge_gap_ratio"] >= 0.08,
            lambda r: (-r["plan"]["effective_zoom_x"], r["plan"]["bbox_area_ratio"]),
        ),
        (
            "tiny_near_edge",
            lambda r: applied(r) and r["edge_gap_ratio"] <= 0.01,
            lambda r: (r["plan"]["bbox_area_ratio"], r["edge_gap_ratio"]),
        ),
        (
            "wide_thin_bbox",
            lambda r: applied(r) and r["bbox_aspect_ratio"] >= 2.0,
            lambda r: (-r["bbox_aspect_ratio"], r["plan"]["bbox_fill_ratio"]),
        ),
        (
            "tall_thin_bbox",
            lambda r: applied(r) and r["bbox_aspect_ratio"] <= 0.5,
            lambda r: (r["bbox_aspect_ratio"], r["plan"]["bbox_fill_ratio"]),
        ),
        (
            "low_fill_fragmented_or_outline",
            lambda r: applied(r) and r["plan"]["bbox_area_ratio"] >= 0.002,
            lambda r: (r["plan"]["bbox_fill_ratio"], r["plan"]["bbox_area_ratio"]),
        ),
        (
            "medium_crop",
            lambda r: applied(r),
            lambda r: abs(r["plan"]["bbox_area_ratio"] - trigger * 0.5),
        ),
        (
            "near_trigger_still_cropped",
            lambda r: applied(r) and r["plan"]["bbox_area_ratio"] < trigger,
            lambda r: trigger - r["plan"]["bbox_area_ratio"],
        ),
        (
            "just_over_trigger_no_crop",
            lambda r: not_applied(r) and r["plan"]["bbox_area_ratio"] >= trigger,
            lambda r: r["plan"]["bbox_area_ratio"] - trigger,
        ),
        (
            "large_mask_no_crop",
            lambda r: not_applied(r),
            lambda r: -r["plan"]["bbox_area_ratio"],
        ),
    ]

    selected: list[dict[str, Any]] = []
    used: set[tuple[str, str]] = set()
    for label, predicate, score in specifications:
        candidates = _ranked(records, predicate, score)
        chosen = next(
            (
                record
                for record in candidates
                if (record["category"], record["stem"]) not in used
            ),
            None,
        )
        if chosen is None:
            continue
        chosen = dict(chosen)
        chosen["case"] = label
        selected.append(chosen)
        used.add((chosen["category"], chosen["stem"]))
    return selected


def _mask_overlay(mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    binary = np.asarray(mask) > 127
    overlay = np.zeros((*binary.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = binary.astype(np.float32) * alpha
    return overlay


def _show_with_mask(ax, image: Image.Image, mask: Image.Image) -> None:
    ax.imshow(image.convert("RGB"))
    ax.imshow(_mask_overlay(np.asarray(mask.convert("L"))))
    ax.set_axis_off()


def _draw_original(ax, image: Image.Image, mask: Image.Image, record: dict[str, Any]) -> None:
    _show_with_mask(ax, image, mask)
    bbox = record["plan"]["mask_bbox"]
    crop = record["plan"]["crop_box"]
    ax.add_patch(
        Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            edgecolor="#FFD400",
            linewidth=2.0,
            linestyle="--",
        )
    )
    ax.add_patch(
        Rectangle(
            (crop[0], crop[1]),
            crop[2] - crop[0],
            crop[3] - crop[1],
            fill=False,
            edgecolor="#00E5FF",
            linewidth=2.5,
        )
    )


def _mask_scale_panel(original_mask: Image.Image, crop_mask: Image.Image) -> Image.Image:
    size = (320, 320)
    left = original_mask.convert("L").resize(size, Image.Resampling.NEAREST)
    right = crop_mask.convert("L").resize(size, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size[0] * 2, size[1] + 30), "black")
    red_left = Image.new("RGB", size, (220, 30, 30))
    red_right = Image.new("RGB", size, (220, 30, 30))
    canvas.paste(red_left, (0, 30), left)
    canvas.paste(red_right, (size[0], 30), right)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), "full-frame mask", fill="white", font=font)
    draw.text((size[0] + 8, 8), "cropped mask", fill="white", font=font)
    return canvas


def _prepare_case(
    record: dict[str, Any],
    crop_config: AdaptiveCropConfig,
    processor: ImageProcessor,
    crop_curve_gamma: float,
) -> dict[str, Any]:
    with Image.open(record["image_path"]) as opened_image:
        image = opened_image.convert("RGB")
    with Image.open(record["mask_path"]) as opened_mask:
        mask = opened_mask.convert("L")
    plan = _build_visualization_crop_plan(
        image.size,
        mask,
        crop_config,
        crop_curve_gamma,
    )
    crop_image, crop_mask = crop_pil_pair(image, mask, plan)
    processed = processor.preprocess({"image": crop_image.copy(), "mask": crop_mask.copy()})
    processed_image = Image.fromarray(
        np.clip(
            processed["image"].permute(1, 2, 0).cpu().numpy() * 255.0,
            0,
            255,
        ).astype(np.uint8)
    )
    processed_mask = Image.fromarray(
        (processed["mask"].cpu().numpy() > 0.5).astype(np.uint8) * 255
    )
    return {
        "image": image,
        "mask": mask,
        "crop_image": crop_image,
        "crop_mask": crop_mask,
        "processed_image": processed_image,
        "processed_mask": processed_mask,
        "mask_scale": _mask_scale_panel(mask, crop_mask),
        "plan": plan,
    }


def _case_title(record: dict[str, Any], prepared: dict[str, Any]) -> str:
    plan = record["plan"]
    return (
        f"{record['case']} | {record['category']}/{record['stem']}\n"
        f"bbox={plan['bbox_area_ratio'] * 100:.3f}%  "
        f"mask={plan['mask_area_ratio'] * 100:.3f}%  "
        f"crop={plan['actual_crop_area_ratio'] * 100:.1f}%  "
        f"zoom={plan['effective_zoom_x']:.2f}x  "
        f"fill={plan['bbox_fill_ratio'] * 100:.1f}%"
    )


def _render_row(
    axes,
    record: dict[str, Any],
    prepared: dict[str, Any],
) -> None:
    _draw_original(axes[0], prepared["image"], prepared["mask"], record)
    axes[0].set_title(_case_title(record, prepared), fontsize=9, loc="left")

    _show_with_mask(axes[1], prepared["crop_image"], prepared["crop_mask"])
    axes[1].set_title(
        f"Raw crop {prepared['crop_image'].width}x{prepared['crop_image'].height}",
        fontsize=9,
    )

    _show_with_mask(axes[2], prepared["processed_image"], prepared["processed_mask"])
    axes[2].set_title(
        f"Processed target + mask {prepared['processed_image'].width}x{prepared['processed_image'].height}",
        fontsize=9,
    )

    axes[3].imshow(prepared["mask_scale"])
    axes[3].set_axis_off()
    axes[3].set_title(
        "Mask occupancy "
        f"{record['plan']['mask_area_ratio'] * 100:.3f}% -> "
        f"{record['post_crop_mask_area_ratio'] * 100:.3f}%",
        fontsize=9,
    )


def _render_cases(
    selected: list[dict[str, Any]],
    crop_config: AdaptiveCropConfig,
    processor_config: ImageProcessorInitArgs,
    output_dir: Path,
    crop_curve_gamma: float,
) -> None:
    processor = ImageProcessor(processor_config)
    prepared_cases = [
        (
            record,
            _prepare_case(record, crop_config, processor, crop_curve_gamma),
        )
        for record in selected
    ]

    figure, axes = plt.subplots(
        len(prepared_cases),
        4,
        figsize=(20, 4.6 * len(prepared_cases)),
        squeeze=False,
    )
    for row_index, (record, prepared) in enumerate(prepared_cases):
        _render_row(axes[row_index], record, prepared)
    figure.suptitle(
        f"Adaptive Crop gamma={crop_curve_gamma:g} on real accessory/jewel_filter data\n"
        "Preprocessing/geometry simulation only — ground-truth target image, no model inference\n"
        "red=target mask, yellow dashed=mask bbox, cyan=crop box; gamma candidate does not change production code",
        fontsize=16,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.988))
    figure.savefig(output_dir / "overview.png", dpi=150, bbox_inches="tight")
    plt.close(figure)

    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    for record, prepared in prepared_cases:
        figure, axes = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)
        _render_row(axes[0], record, prepared)
        figure.tight_layout()
        filename = f"{record['case']}__{record['category']}__{record['stem']}.png"
        figure.savefig(cases_dir / filename, dpi=160, bbox_inches="tight")
        plt.close(figure)


def _render_gamma_comparison(
    selected: list[dict[str, Any]],
    crop_config: AdaptiveCropConfig,
    output_dir: Path,
    candidate_gamma: float,
) -> list[dict[str, Any]]:
    """Render paired gamma=1 and candidate-gamma crops for identical samples."""
    candidate_file_label = f"{candidate_gamma:g}".replace(".", "p")
    figure, axes = plt.subplots(
        len(selected),
        4,
        figsize=(20, 4.6 * len(selected)),
        squeeze=False,
    )
    comparison_records: list[dict[str, Any]] = []

    for row_index, record in enumerate(selected):
        with Image.open(record["image_path"]) as opened_image:
            image = opened_image.convert("RGB")
        with Image.open(record["mask_path"]) as opened_mask:
            mask = opened_mask.convert("L")

        gamma1_plan = _build_visualization_crop_plan(image.size, mask, crop_config, 1.0)
        candidate_plan = _build_visualization_crop_plan(
            image.size,
            mask,
            crop_config,
            candidate_gamma,
        )
        gamma1_crop, gamma1_mask = crop_pil_pair(image, mask, gamma1_plan)
        candidate_crop, candidate_mask = crop_pil_pair(image, mask, candidate_plan)

        original_ax = axes[row_index, 0]
        _show_with_mask(original_ax, image, mask)
        bbox = gamma1_plan.mask_bbox
        original_ax.add_patch(
            Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor="#FFD400",
                linewidth=2.0,
                linestyle="--",
            )
        )
        for plan, color, linestyle in (
            (gamma1_plan, "#00E5FF", "-"),
            (candidate_plan, "#FF3CAC", "-"),
        ):
            x1, y1, x2, y2 = plan.crop_box
            original_ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.5,
                    linestyle=linestyle,
                )
            )
        original_ax.set_title(
            f"{record['case']} | {record['category']}/{record['stem']}\n"
            f"yellow=bbox, cyan=gamma1, magenta=gamma{candidate_gamma:g}",
            fontsize=9,
            loc="left",
        )

        _show_with_mask(axes[row_index, 1], gamma1_crop, gamma1_mask)
        gamma1_occupancy = gamma1_plan.mask_area_pixels / (
            gamma1_crop.width * gamma1_crop.height
        )
        axes[row_index, 1].set_title(
            "gamma=1 target crop (production current)\n"
            f"crop={gamma1_plan.actual_crop_area_ratio * 100:.1f}%  "
            f"zoom={gamma1_plan.effective_zoom_x:.2f}x  "
            f"occupancy={gamma1_occupancy * 100:.3f}%",
            fontsize=9,
        )

        _show_with_mask(axes[row_index, 2], candidate_crop, candidate_mask)
        candidate_occupancy = candidate_plan.mask_area_pixels / (
            candidate_crop.width * candidate_crop.height
        )
        axes[row_index, 2].set_title(
            f"gamma={candidate_gamma:g} target crop (visualization candidate)\n"
            f"crop={candidate_plan.actual_crop_area_ratio * 100:.1f}%  "
            f"zoom={candidate_plan.effective_zoom_x:.2f}x  "
            f"occupancy={candidate_occupancy * 100:.3f}%",
            fontsize=9,
        )

        metric_ax = axes[row_index, 3]
        metric_ax.set_axis_off()
        metric_ax.text(
            0.02,
            0.95,
            "Paired geometry delta\n\n"
            f"bbox area: {gamma1_plan.bbox_area_ratio * 100:.4f}%\n"
            f"mask area: {gamma1_plan.mask_area_ratio * 100:.4f}%\n\n"
            f"crop area: {gamma1_plan.actual_crop_area_ratio * 100:.2f}%"
            f" -> {candidate_plan.actual_crop_area_ratio * 100:.2f}%\n"
            f"zoom: {gamma1_plan.effective_zoom_x:.3f}x"
            f" -> {candidate_plan.effective_zoom_x:.3f}x\n"
            f"mask occupancy: {gamma1_occupancy * 100:.4f}%"
            f" -> {candidate_occupancy * 100:.4f}%\n\n"
            f"gamma1 box: {list(gamma1_plan.crop_box)}\n"
            f"gamma{candidate_gamma:g} box: {list(candidate_plan.crop_box)}",
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

        comparison_records.append(
            {
                "case": record["case"],
                "category": record["category"],
                "stem": record["stem"],
                "gamma_1": gamma1_plan.to_dict(),
                f"gamma_{candidate_gamma:g}": candidate_plan.to_dict(),
                "gamma_1_mask_occupancy": gamma1_occupancy,
                f"gamma_{candidate_gamma:g}_mask_occupancy": candidate_occupancy,
            }
        )

    figure.suptitle(
        f"Adaptive Crop paired comparison: gamma=1 vs gamma={candidate_gamma:g}\n"
        "Preprocessing/geometry simulation only — ground-truth target, no model inference\n"
        "same image/mask/constraints; gamma candidate does not change production code",
        fontsize=16,
        y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.988))
    comparison_stem = f"gamma1_vs_gamma{candidate_file_label}"
    figure.savefig(output_dir / f"{comparison_stem}.png", dpi=155, bbox_inches="tight")
    plt.close(figure)

    with (output_dir / f"{comparison_stem}.json").open("w", encoding="utf-8") as file:
        json.dump(comparison_records, file, ensure_ascii=False, indent=2)
    return comparison_records


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _render_statistics(
    records: list[dict[str, Any]],
    crop_config: AdaptiveCropConfig,
    output_dir: Path,
    crop_curve_gamma: float,
) -> dict[str, Any]:
    bbox_ratio = np.asarray([r["plan"]["bbox_area_ratio"] for r in records])
    crop_ratio = np.asarray([r["plan"]["actual_crop_area_ratio"] for r in records])
    zoom = np.asarray([r["plan"]["effective_zoom_x"] for r in records])
    mask_before = np.asarray([r["plan"]["mask_area_ratio"] for r in records])
    mask_after = np.asarray([r["post_crop_mask_area_ratio"] for r in records])
    applied = np.asarray([r["plan"]["applied"] for r in records], dtype=bool)

    figure, axes = plt.subplots(2, 2, figsize=(15, 11))

    axes[0, 0].hist(bbox_ratio * 100, bins=60, color="#4C78A8", alpha=0.9)
    axes[0, 0].axvline(
        crop_config.trigger_bbox_area_ratio * 100,
        color="red",
        linestyle="--",
        label=f"trigger={crop_config.trigger_bbox_area_ratio * 100:.1f}%",
    )
    axes[0, 0].set_title("Target mask bbox area distribution")
    axes[0, 0].set_xlabel("bbox / full image area (%)")
    axes[0, 0].set_ylabel("samples")
    axes[0, 0].legend()

    sample_step = max(1, len(records) // 5000)
    sample_indices = np.arange(0, len(records), sample_step)
    axes[0, 1].scatter(
        bbox_ratio[sample_indices] * 100,
        crop_ratio[sample_indices] * 100,
        c=np.where(applied[sample_indices], "#00A6A6", "#888888"),
        s=9,
        alpha=0.45,
        linewidths=0,
    )
    x = np.linspace(0, crop_config.trigger_bbox_area_ratio, 300)
    policy = crop_config.min_crop_area_ratio + (
        1.0 - crop_config.min_crop_area_ratio
    ) * np.power(x / crop_config.trigger_bbox_area_ratio, crop_curve_gamma)
    axes[0, 1].plot(x * 100, policy * 100, color="black", linewidth=2, label="configured policy")
    axes[0, 1].set_title("bbox area -> actual crop area")
    axes[0, 1].set_xlabel("bbox / full image area (%)")
    axes[0, 1].set_ylabel("crop / full image area (%)")
    axes[0, 1].legend()

    axes[1, 0].hist(zoom[applied], bins=45, color="#F58518", alpha=0.9)
    axes[1, 0].set_title(
        f"Effective linear zoom (applied {applied.sum():,}/{len(records):,})"
    )
    axes[1, 0].set_xlabel("zoom x")
    axes[1, 0].set_ylabel("samples")

    positive = (mask_before > 0) & (mask_after > 0)
    axes[1, 1].scatter(
        mask_before[positive] * 100,
        mask_after[positive] * 100,
        c=np.where(applied[positive], "#E45756", "#888888"),
        s=9,
        alpha=0.4,
        linewidths=0,
    )
    max_value = max(float((mask_before * 100).max()), float((mask_after * 100).max()))
    axes[1, 1].plot([0, max_value], [0, max_value], color="black", linestyle="--")
    axes[1, 1].set_title("Foreground occupancy before vs after crop")
    axes[1, 1].set_xlabel("before (%)")
    axes[1, 1].set_ylabel("after (%)")

    figure.suptitle(
        "Adaptive Crop dataset statistics\n"
        f"policy=bbox_area_gamma, gamma={crop_curve_gamma:g}, "
        f"min crop={crop_config.min_crop_area_ratio * 100:.0f}%, "
        f"trigger={crop_config.trigger_bbox_area_ratio * 100:.0f}%, "
        f"margin={crop_config.context_margin_ratio * 100:.0f}%",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "dataset_statistics.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    reasons = Counter(record["plan"]["decision_reason"] for record in records)
    soft_samples = sum(record["soft_pixels"] > 0 for record in records)
    return {
        "sample_count": len(records),
        "applied_count": int(applied.sum()),
        "not_applied_count": int((~applied).sum()),
        "applied_ratio": float(applied.mean()),
        "decision_reasons": dict(sorted(reasons.items())),
        "bbox_area_ratio": _percentiles(bbox_ratio),
        "actual_crop_area_ratio": _percentiles(crop_ratio),
        "effective_zoom_x": _percentiles(zoom),
        "mask_area_ratio_before": _percentiles(mask_before),
        "mask_area_ratio_after": _percentiles(mask_after),
        "soft_mask_sample_count": soft_samples,
        "soft_mask_sample_ratio": soft_samples / len(records),
    }


def _render_policy_options(
    crop_config: AdaptiveCropConfig,
    output_dir: Path,
) -> None:
    """Plot analytical policy alternatives before margin/geometry constraints."""
    trigger = float(crop_config.trigger_bbox_area_ratio)
    current_min = float(crop_config.min_crop_area_ratio)
    bbox_ratio = np.linspace(0.0, trigger, 500)
    normalized = bbox_ratio / trigger
    options = [
        ("current: linear, max zoom 2.0x", current_min, 1.0, "#4C78A8"),
        ("recommended ablation: gamma=2, max zoom 2.0x", current_min, 2.0, "#E45756"),
        ("aggressive: gamma=2, max zoom 2.5x", 0.16, 2.0, "#F58518"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for label, min_area, gamma, color in options:
        crop_ratio = min_area + (1.0 - min_area) * np.power(normalized, gamma)
        zoom = 1.0 / np.sqrt(crop_ratio)
        axes[0].plot(bbox_ratio * 100, crop_ratio * 100, label=label, color=color, linewidth=2.5)
        axes[1].plot(bbox_ratio * 100, zoom, label=label, color=color, linewidth=2.5)

    axes[0].set_title("Requested crop area curve")
    axes[0].set_xlabel("mask bbox / full image area (%)")
    axes[0].set_ylabel("requested crop / full image area (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Implied linear zoom (before geometry constraints)")
    axes[1].set_xlabel("mask bbox / full image area (%)")
    axes[1].set_ylabel("zoom x")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(bottom=0.95)

    figure.suptitle(
        "Adaptive Crop policy options\n"
        "gamma=2 gives more zoom to small/medium targets without moving the 10% no-crop threshold",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_dir / "policy_options.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_selected_csv(
    selected: list[dict[str, Any]],
    path: Path,
    crop_curve_gamma: float,
) -> None:
    fields = [
        "crop_curve_gamma",
        "case",
        "category",
        "stem",
        "image_path",
        "mask_path",
        "width",
        "height",
        "bbox_aspect_ratio",
        "edge_gap_ratio",
        "soft_pixel_ratio",
        "mask_area_ratio",
        "bbox_area_ratio",
        "bbox_fill_ratio",
        "actual_crop_area_ratio",
        "post_crop_mask_area_ratio",
        "effective_zoom_x",
        "decision_reason",
        "mask_bbox",
        "crop_box",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            plan = record["plan"]
            original_width, original_height = plan["original_size"]
            writer.writerow(
                {
                    "crop_curve_gamma": crop_curve_gamma,
                    "case": record["case"],
                    "category": record["category"],
                    "stem": record["stem"],
                    "image_path": record["image_path"],
                    "mask_path": record["mask_path"],
                    "width": record.get("width", original_width),
                    "height": record.get("height", original_height),
                    "bbox_aspect_ratio": record["bbox_aspect_ratio"],
                    "edge_gap_ratio": record["edge_gap_ratio"],
                    "soft_pixel_ratio": record["soft_pixel_ratio"],
                    "mask_area_ratio": plan["mask_area_ratio"],
                    "bbox_area_ratio": plan["bbox_area_ratio"],
                    "bbox_fill_ratio": plan["bbox_fill_ratio"],
                    "actual_crop_area_ratio": plan["actual_crop_area_ratio"],
                    "post_crop_mask_area_ratio": record["post_crop_mask_area_ratio"],
                    "effective_zoom_x": plan["effective_zoom_x"],
                    "decision_reason": plan["decision_reason"],
                    "mask_bbox": json.dumps(plan["mask_bbox"]),
                    "crop_box": json.dumps(plan["crop_box"]),
                }
            )


def main() -> None:
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if not math.isfinite(args.crop_curve_gamma) or args.crop_curve_gamma <= 0:
        raise ValueError("--crop-curve-gamma must be finite and > 0")
    crop_config, processor_config, data_root, manifest = load_settings(args)
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, errors = _scan_manifest(
        manifest,
        data_root,
        crop_config,
        args.crop_curve_gamma,
        args.max_samples,
    )
    if not records:
        raise RuntimeError("no valid samples were scanned")
    selected = _select_cases(records, crop_config)
    if not selected:
        raise RuntimeError("could not select any representative Adaptive Crop cases")

    statistics = _render_statistics(
        records,
        crop_config,
        output_dir,
        args.crop_curve_gamma,
    )
    _render_policy_options(crop_config, output_dir)
    _render_cases(
        selected,
        crop_config,
        processor_config,
        output_dir,
        args.crop_curve_gamma,
    )
    _write_selected_csv(
        selected,
        output_dir / "selected_cases.csv",
        args.crop_curve_gamma,
    )

    comparison_records = None
    if args.comparison_summary is not None:
        comparison_path = args.comparison_summary.resolve()
        if not comparison_path.is_file():
            raise FileNotFoundError(f"comparison summary not found: {comparison_path}")
        comparison_payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        comparison_selected = comparison_payload.get("selected_cases")
        if not isinstance(comparison_selected, list) or not comparison_selected:
            raise ValueError(
                f"comparison summary has no selected_cases list: {comparison_path}"
            )
        comparison_records = _render_gamma_comparison(
            comparison_selected,
            crop_config,
            output_dir,
            args.crop_curve_gamma,
        )

    summary = {
        "config": str(args.config.resolve()),
        "manifest": str(manifest),
        "data_root": str(data_root),
        "visualization_crop_curve_gamma": args.crop_curve_gamma,
        "adaptive_crop": crop_config.model_dump(mode="json"),
        "processor": processor_config.model_dump(mode="json"),
        "statistics": statistics,
        "scan_errors": errors,
        "comparison_summary": (
            None if args.comparison_summary is None else str(args.comparison_summary.resolve())
        ),
        "comparison_case_count": 0 if comparison_records is None else len(comparison_records),
        "selected_cases": [
            {
                "case": record["case"],
                "category": record["category"],
                "stem": record["stem"],
                "image_path": record["image_path"],
                "mask_path": record["mask_path"],
                "bbox_aspect_ratio": record["bbox_aspect_ratio"],
                "edge_gap_ratio": record["edge_gap_ratio"],
                "soft_pixel_ratio": record["soft_pixel_ratio"],
                "post_crop_mask_area_ratio": record["post_crop_mask_area_ratio"],
                "plan": record["plan"],
            }
            for record in selected
        ],
        "note": (
            "Preprocessing geometry visualization only; no model checkpoint was run. "
            "crop_curve_gamma is simulated in this script and does not change production crop code."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(statistics, indent=2))
    print(f"selected_cases={len(selected)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()

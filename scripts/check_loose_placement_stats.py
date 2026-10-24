"""Check σ-split loose placement masks on real training samples.

Each sample goes through the training dataset (loose-mask sampling, adaptive
crop planned on the loose mask, processor resize), then the trainer's
interaction band is applied. Reported per configuration:

* fill: object bbox / mask bbox per axis and ``min(h, w)`` per sample
  (expected median roughly 0.5-0.65);
* aspect coupling: Pearson r between mask and object aspect ratios, raw and
  log (expected near 0);
* leakage: samples where the trainer's GT union or interaction band would add
  pixels to the loose mask (must be 0);
* context guard shrink/exhaustion rates and the mask-family mix.

Usage:
    uv run --offline --no-sync python scripts/check_loose_placement_stats.py \
        --config configs/teacher_v2/animal_sigma_split.yaml --num-samples 400
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import load_config_from_yaml  # noqa: E402
from src.data.insert_anything_dataset import InsertAnythingDataset  # noqa: E402
from src.utils.hybrid_masking import add_interaction_context  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--size-reference",
        choices=("bbox_geometric_mean", "bbox_axes"),
        default=None,
        help="Override hybrid_masking.loose_placement.size_reference for an A/B check.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON with the summary and per-sample rows.")
    return parser.parse_args(argv)


def _bbox_extent(mask: torch.Tensor):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def _first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def measure(sample, interaction_px: int) -> dict:
    gt = sample["tar_mask"][0]
    loose = sample["tar_mask_cond"][0]
    banded = add_interaction_context(loose[None], gt[None], interaction_px)[0]
    object_h, object_w = _bbox_extent(gt)
    mask_h, mask_w = _bbox_extent(banded)
    metadata = {key: _first(value) for key, value in sample["placement_metadata"].items()}
    return {
        "sample_id": _first(sample["sample_metadata"]["sample_id"]),
        "h_cover": object_h / mask_h,
        "w_cover": object_w / mask_w,
        "min_cover": min(object_h / mask_h, object_w / mask_w),
        "area_fill": float(gt.sum() / banded.sum()),
        "condition_area_ratio": float(banded.mean()),
        "object_aspect": object_h / object_w,
        "mask_aspect": mask_h / mask_w,
        "touches_border": bool(loose[0].any() or loose[-1].any() or loose[:, 0].any() or loose[:, -1].any()),
        "union_leak": bool((torch.maximum(loose, gt) != loose).any()),
        "band_leak": bool((banded != loose).any()),
        **{key: (value.item() if isinstance(value, torch.Tensor) else value) for key, value in metadata.items()},
    }


def summarize(rows) -> dict:
    def percentiles(key):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        return {f"p{q}": round(float(np.percentile(values, q)), 3) for q in (5, 25, 50, 75, 95)}

    object_aspect = np.array([row["object_aspect"] for row in rows])
    mask_aspect = np.array([row["mask_aspect"] for row in rows])
    inside = np.array([not row["touches_border"] for row in rows])
    count = len(rows)
    return {
        "samples": count,
        "min_cover": percentiles("min_cover"),
        "h_cover": percentiles("h_cover"),
        "w_cover": percentiles("w_cover"),
        "area_fill": percentiles("area_fill"),
        "condition_area_ratio": percentiles("condition_area_ratio"),
        "aspect_pearson_r": round(float(np.corrcoef(mask_aspect, object_aspect)[0, 1]), 3),
        "log_aspect_pearson_r": round(float(np.corrcoef(np.log(mask_aspect), np.log(object_aspect))[0, 1]), 3),
        # Masks clipped by the canvas inherit the frame's aspect, not the sampled one.
        "border_clipped_rate": float(1.0 - inside.mean()),
        "log_aspect_pearson_r_unclipped": (
            round(float(np.corrcoef(np.log(mask_aspect[inside]), np.log(object_aspect[inside]))[0, 1]), 3)
            if inside.sum() > 2
            else None
        ),
        "union_leak_rate": sum(row["union_leak"] for row in rows) / count,
        "band_leak_rate": sum(row["band_leak"] for row in rows) / count,
        "guard_shrink_rate": sum(row["guard_shrinks"] > 0 for row in rows) / count,
        "guard_exhausted_rate": sum(bool(row["guard_exhausted"]) for row in rows) / count,
        "bottom_aligned_rate": sum(bool(row["bottom_aligned"]) for row in rows) / count,
        "loose_shapes": dict(collections.Counter(row["loose_shape"] for row in rows)),
        "tight_modes": dict(collections.Counter(row["tight_mode"] for row in rows)),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config_from_yaml(str(args.config))
    init_args = config.data.init_args
    loose_cfg = init_args.hybrid_masking.loose_placement
    if not loose_cfg.enabled:
        raise SystemExit(f"{args.config} does not enable hybrid_masking.loose_placement")
    if args.size_reference is not None:
        loose_cfg.size_reference = args.size_reference

    dataset = InsertAnythingDataset(init_args)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    torch.manual_seed(args.seed)
    loader = DataLoader(
        Subset(dataset, indices[: args.num_samples]),
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda items: {
            key: [item[key] for item in items] if not isinstance(items[0][key], torch.Tensor)
            else torch.stack([item[key] for item in items])
            for key in items[0]
        },
    )
    interaction_px = int(init_args.hybrid_masking.interaction_dilation_px)
    rows = []
    for sample in loader:
        sample["placement_metadata"] = sample["placement_metadata"][0]
        sample["sample_metadata"] = sample["sample_metadata"][0]
        rows.append(measure(sample, interaction_px))

    summary = summarize(rows)
    summary["size_reference"] = loose_cfg.size_reference
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    median = summary["min_cover"]["p50"]
    print(
        f"median min(h,w) cover = {median} (expected ~0.5-0.65); "
        f"aspect r = {summary['aspect_pearson_r']} / log r = {summary['log_aspect_pearson_r']} (expected ~0); "
        f"union/band leaks = {summary['union_leak_rate']}/{summary['band_leak_rate']} (must be 0)"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

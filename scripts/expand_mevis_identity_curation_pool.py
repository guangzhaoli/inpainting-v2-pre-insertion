#!/usr/bin/env python3
"""Expand the pinned MeViS identity-pair pool from the same raw snapshot.

The original hand-curated pool is deliberately preserved.  This utility adds
only previously unseen official ``(video_id, anno_id)`` tracks which pass the
builder's cross-frame gates or a recorded, bounded geometry fallback.  The
foreground-difference and sharpness floors are unchanged.  The resulting JSON
contains upstream IDs and deterministic summary statistics only; images and
masks remain in the pinned MeViS source snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import imagehash
import numpy as np
from PIL import Image, ImageOps

try:
    import prepare_identity_paired_insertion_dataset as builder
except ModuleNotFoundError:  # Imported as ``scripts.*`` from the repository root.
    from scripts import prepare_identity_paired_insertion_dataset as builder


BASE_POOL_SHA256 = "04ade013a827a26923c44eae919e998a2b41956ed8ee600705d95309915a7392"

_TRACKS: list[builder.MevisTrack] = []
_IMAGE_ROOT: Path | None = None
_SEED = builder.DEFAULT_SEED


def _frame_geometry(info: builder.MevisFrameInfo) -> dict[str, Any]:
    x, y, width, height = info.bbox_xywh
    return {
        "height": info.height,
        "width": info.width,
        "mask_area_pixels": float(round(info.area_ratio * info.width * info.height)),
        "mask_area_ratio": info.area_ratio,
        "bbox_xywh": [float(x), float(y), float(width), float(height)],
        "bbox_short_side_ratio": min(width, height) / min(info.width, info.height),
        "mask_bbox_solidity": info.solidity,
        "border_margin_ratio": info.margin_ratio,
    }


def _white_foreground_canvas(image: Image.Image, rle: Mapping[str, Any], size: int = 128) -> Image.Image:
    mask = builder.decode_coco_rle(rle)
    clean, clean_mask, _ = builder.crop_clean_reference(image, mask, margin_ratio=0.0)
    fitted_image = ImageOps.contain(clean, (size, size), Image.Resampling.BILINEAR)
    fitted_mask = ImageOps.contain(clean_mask, (size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), "white")
    offset = ((size - fitted_image.width) // 2, (size - fitted_image.height) // 2)
    canvas.paste(fitted_image, offset, fitted_mask)
    return canvas


def _pixel_delta(
    src_image: Image.Image,
    tar_image: Image.Image,
    src_info: builder.MevisFrameInfo,
    tar_info: builder.MevisFrameInfo,
) -> dict[str, Any]:
    src_crop = _white_foreground_canvas(src_image, src_info.rle)
    tar_crop = _white_foreground_canvas(tar_image, tar_info.rle)
    src_crop_values = np.asarray(src_crop, dtype=np.float32)
    tar_crop_values = np.asarray(tar_crop, dtype=np.float32)
    src_frame = np.asarray(src_image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    tar_frame = np.asarray(tar_image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    return {
        "crop_mae": float(np.mean(np.abs(src_crop_values - tar_crop_values)) / 255.0),
        "crop_phash": int(
            imagehash.phash(src_crop, hash_size=8, highfreq_factor=4)
            - imagehash.phash(tar_crop, hash_size=8, highfreq_factor=4)
        ),
        "crop_dhash": int(imagehash.dhash(src_crop, hash_size=8) - imagehash.dhash(tar_crop, hash_size=8)),
        "frame_mae": float(np.mean(np.abs(src_frame - tar_frame)) / 255.0),
        "frame_phash": int(
            imagehash.phash(src_image, hash_size=8, highfreq_factor=4)
            - imagehash.phash(tar_image, hash_size=8, highfreq_factor=4)
        ),
    }


def _pixel_tier(pixel: Mapping[str, Any]) -> str:
    crop_mae = float(pixel["crop_mae"])
    crop_phash = int(pixel["crop_phash"])
    if crop_mae >= 0.04 and crop_phash >= 6:
        return "strong"
    if crop_mae >= 0.03 and crop_phash >= 4:
        return "medium"
    return "base"


def _relaxed_frame_info(
    index: int,
    frame_id: str,
    rle: Mapping[str, Any],
    *,
    relaxed_daily_object: bool = False,
) -> builder.MevisFrameInfo | None:
    """A bounded fallback for clean tracks just outside the original geometry gate."""

    del relaxed_daily_object
    normalized = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(normalized["counts"], str):
        normalized["counts"] = normalized["counts"].encode("ascii")
    height, width = map(int, normalized["size"])
    area = float(builder.mask_utils.area(normalized))
    x, y, bbox_width, bbox_height = map(float, builder.mask_utils.toBbox(normalized))
    if bbox_width <= 0 or bbox_height <= 0:
        return None
    area_ratio = area / float(width * height)
    short_side_ratio = min(bbox_width, bbox_height) / float(min(width, height))
    solidity = area / float(bbox_width * bbox_height)
    margin_ratio = min(
        x / width,
        y / height,
        max(0.0, width - (x + bbox_width)) / width,
        max(0.0, height - (y + bbox_height)) / height,
    )
    if not (0.018 <= area_ratio <= 0.60 and short_side_ratio >= 0.04 and solidity >= 0.045 and margin_ratio >= 0.0):
        return None
    ideal = 0.16
    area_score = max(0.0, 1.0 - abs(math.log(area_ratio / ideal)) / 2.2)
    geometry_score = (
        2.2 * area_score + min(1.0, short_side_ratio / 0.22) + min(1.0, solidity / 0.45) + min(1.0, margin_ratio / 0.04)
    )
    return builder.MevisFrameInfo(
        index=index,
        frame_id=frame_id,
        rle=dict(rle),
        width=width,
        height=height,
        area_ratio=area_ratio,
        bbox_xywh=(x, y, bbox_width, bbox_height),
        aspect_ratio=bbox_width / bbox_height,
        solidity=solidity,
        margin_ratio=margin_ratio,
        geometry_score=geometry_score,
    )


def _select_supplemental_pair(
    track: builder.MevisTrack,
    classifications: Mapping[tuple[str, str], float],
) -> tuple[
    tuple[float, builder.MevisFrameInfo, builder.MevisFrameInfo, dict[str, Any]] | None,
    str,
]:
    if _IMAGE_ROOT is None:
        raise RuntimeError("Worker image root was not initialized")
    daily = ("object", "daily_sports_objects") in classifications
    pair = builder.select_frame_pair(
        track,
        _IMAGE_ROOT,
        _SEED,
        relaxed_daily_object=daily,
    )
    if pair is not None:
        return pair, "daily_sports_custom" if daily else "moderate"
    original = builder._mevis_frame_info
    builder._mevis_frame_info = _relaxed_frame_info
    try:
        pair = builder.select_frame_pair(
            track,
            _IMAGE_ROOT,
            _SEED,
            relaxed_daily_object=True,
        )
    finally:
        builder._mevis_frame_info = original
    return pair, "supplemental_relaxed_geometry_v1"


def _supplemental_rows(track_index: int) -> list[dict[str, Any]]:
    if _IMAGE_ROOT is None:
        raise RuntimeError("Worker image root was not initialized")
    track = _TRACKS[track_index]
    classifications = builder.classify_mevis_track(track)
    if not classifications:
        return []
    pair, geometry_rule = _select_supplemental_pair(track, classifications)
    if pair is None:
        return []
    pair_score, src, tar, evidence = pair
    with Image.open(_IMAGE_ROOT / track.video_id / f"{src.frame_id}.jpg") as handle:
        src_image = handle.convert("RGB")
    with Image.open(_IMAGE_ROOT / track.video_id / f"{tar.frame_id}.jpg") as handle:
        tar_image = handle.convert("RGB")
    pixel = _pixel_delta(src_image, tar_image, src, tar)
    if float(pixel["crop_mae"]) < 0.02 or int(pixel["crop_phash"]) < 2:
        return []
    tier = _pixel_tier(pixel)
    votes = builder.mevis_subject_votes(track)
    vote_rows = votes.most_common(2)
    winner = vote_rows[0] if vote_rows else (None, 0)
    runner_up = vote_rows[1] if len(vote_rows) > 1 else (None, 0)
    normalized_geometry_score = min(1.0, max(0.0, (float(pair_score) - 6.0) / 8.0))
    rows: list[dict[str, Any]] = []
    for (major, minor), _classification_score in sorted(classifications.items()):
        candidate_id = f"mevis_v2_train_{track.video_id}_{track.anno_id}" f"__{major}_{minor}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_source": (
                    "generic_supplement_relaxed_v1"
                    if geometry_rule == "supplemental_relaxed_geometry_v1"
                    else "generic_supplement_v1"
                ),
                "selection_priority_penalty": (5.0 if geometry_rule == "supplemental_relaxed_geometry_v1" else 4.0),
                "major": major,
                "minor": minor,
                "video_id": track.video_id,
                "anno_id": int(track.anno_id),
                "identity_key": track.identity_id,
                "src_frame_index": src.index,
                "src_frame_name": src.frame_id,
                "tar_frame_index": tar.index,
                "tar_frame_name": tar.frame_id,
                "expressions": list(track.expressions),
                "frame_count": len(track.frame_ids),
                "semantic_primary": winner[0],
                "semantic_winner_score": int(winner[1]),
                "semantic_runner_up": runner_up[0],
                "semantic_runner_up_score": int(runner_up[1]),
                "geometry_rule": geometry_rule,
                "geometry": {
                    "score": normalized_geometry_score,
                    "temporal_gap_frames": int(evidence["temporal_gap_frames"]),
                    "temporal_gap_fraction": float(evidence["temporal_gap_ratio"]),
                    "mask_area_ratio_change": float(evidence["area_ratio_change"]),
                    "bbox_aspect_ratio_change": float(evidence["aspect_ratio_change"]),
                    "src": _frame_geometry(src),
                    "tar": _frame_geometry(tar),
                },
                "pixel_delta": pixel,
                "pixel_delta_tier": tier,
                "requires_visual_qc": True,
            }
        )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--mevis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=builder.DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    global _TRACKS, _IMAGE_ROOT, _SEED
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.base_pool.resolve() == args.output.resolve():
        raise ValueError("Output must not overwrite the preserved base pool")
    if builder.file_sha256(args.base_pool) != BASE_POOL_SHA256:
        raise ValueError(f"Base-pool SHA256 mismatch: {args.base_pool}")
    source_pins = {
        "JPEGImages.tar": builder.MEVIS_ARCHIVE_SHA256,
        "meta_expressions_v2.json": builder.MEVIS_META_SHA256,
        "mask_dict.json": builder.MEVIS_MASK_DICT_SHA256,
    }
    for filename, expected_sha256 in source_pins.items():
        source_path = args.mevis_root / filename
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"Missing regular pinned MeViS source: {source_path}")
        if builder.file_sha256(source_path) != expected_sha256:
            raise ValueError(f"Pinned MeViS source SHA256 mismatch: {source_path}")
    base = json.loads(args.base_pool.read_text(encoding="utf-8"))
    if base.get("schema_version") != "mevis_diverse_candidate_pool_v1":
        raise ValueError("Unexpected base-pool schema")
    base_rows = list(base["candidates"])
    base_identity_ids = {str(row["identity_key"]) for row in base_rows}
    _TRACKS = builder.load_mevis_tracks(
        args.mevis_root / "meta_expressions_v2.json",
        args.mevis_root / "mask_dict.json",
    )
    _IMAGE_ROOT = args.mevis_root / "JPEGImages"
    if not _IMAGE_ROOT.is_dir() or _IMAGE_ROOT.is_symlink():
        raise FileNotFoundError(f"Missing regular extracted MeViS image tree: {_IMAGE_ROOT}")
    _SEED = args.seed
    eligible_indices = [
        index
        for index, track in enumerate(_TRACKS)
        if track.identity_id not in base_identity_ids and builder.classify_mevis_track(track)
    ]
    supplemental: list[dict[str, Any]] = []
    if args.workers == 1:
        iterator = map(_supplemental_rows, eligible_indices)
        for completed, rows in enumerate(iterator, start=1):
            supplemental.extend(rows)
            if completed % 100 == 0:
                print(
                    f"MeViS expansion: {completed}/{len(eligible_indices)} " f"supplemental_rows={len(supplemental)}",
                    flush=True,
                )
    else:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            iterator = executor.map(_supplemental_rows, eligible_indices, chunksize=1)
            for completed, rows in enumerate(iterator, start=1):
                supplemental.extend(rows)
                if completed % 100 == 0:
                    print(
                        f"MeViS expansion: {completed}/{len(eligible_indices)} "
                        f"supplemental_rows={len(supplemental)}",
                        flush=True,
                    )
    normalized_base = [
        {
            **row,
            "candidate_source": "curated_v1",
            "selection_priority_penalty": 0.0,
        }
        for row in base_rows
    ]
    category_order = {
        (major, minor): (major_index, minor_index)
        for major_index, major in enumerate(("object", "person"))
        for minor_index, minor in enumerate(builder.MAJOR_MINOR_QUOTAS[major])
    }
    combined = sorted(
        [*normalized_base, *supplemental],
        key=lambda row: (
            category_order[(str(row["major"]), str(row["minor"]))],
            str(row["candidate_id"]),
        ),
    )
    candidate_ids = [str(row["candidate_id"]) for row in combined]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Expanded pool contains duplicate candidate_id values")
    counts = Counter(f"{row['major']}/{row['minor']}" for row in combined)
    tier_counts: dict[str, dict[str, int]] = {}
    for category in sorted(counts):
        tier_counts[category] = dict(
            sorted(
                Counter(
                    str(row["pixel_delta_tier"]) for row in combined if f"{row['major']}/{row['minor']}" == category
                ).items()
            )
        )
    payload = {
        "schema_version": "mevis_diverse_candidate_pool_v5",
        "source": base["source"],
        "contract": {
            **base["contract"],
            "base_pool_sha256": BASE_POOL_SHA256,
            "supplemental_candidate_source": ("deterministic_builder_gates_with_bounded_relaxed_geometry_fallback"),
            "supplemental_relaxed_geometry_fallback": {
                "mask_area_ratio": [0.018, 0.60],
                "bbox_short_side_ratio_min": 0.04,
                "solidity_min": 0.045,
                "border_margin_ratio_min": 0.0,
                "max_area_ratio_change": 2.5,
                "max_aspect_ratio_change": 2.5,
            },
            "supplemental_tracks_are_absent_from_base_pool": True,
            "supplemental_visual_qc_required": True,
            "seed": args.seed,
            "source_pins": {
                "jpeg_archive_sha256": builder.MEVIS_ARCHIVE_SHA256,
                "meta_sha256": builder.MEVIS_META_SHA256,
                "mask_dict_sha256": builder.MEVIS_MASK_DICT_SHA256,
            },
        },
        "eligibility": {
            **base["eligibility"],
            "supplemental_relaxed_geometry_v1": {
                "mask_area_ratio": [0.018, 0.60],
                "bbox_short_side_ratio_min": 0.04,
                "solidity_min": 0.045,
                "border_margin_ratio_min": 0.0,
                "temporal_gap_fraction_min": 0.12,
                "temporal_gap_frames_min": 4,
                "max_area_ratio_change": 2.5,
                "max_aspect_ratio_change": 2.5,
                "foreground_crop_phash_min": 2,
                "foreground_crop_rgb_mae_min": 0.02,
            },
        },
        "counts": {
            "membership_rows": len(combined),
            "unique_tracks": len({str(row["identity_key"]) for row in combined}),
            "unique_videos": len({str(row["video_id"]) for row in combined}),
            "base_membership_rows": len(normalized_base),
            "supplemental_membership_rows": len(supplemental),
            "by_minor": dict(sorted(counts.items())),
            "pixel_tiers_by_minor": tier_counts,
        },
        "candidates": combined,
    }
    builder.atomic_write_text(
        args.output,
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": builder.file_sha256(args.output),
                **payload["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

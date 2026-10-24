#!/usr/bin/env python3
"""Render unused MeViS daily/sports-object identities for visual review.

The renderer is deliberately read-only with respect to the candidate cache,
curation pool, exclusion list, selected benchmark, and raw source snapshot.  A
backup is a candidate which is neither identity-excluded nor already selected
anywhere in the supplied benchmark manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

try:
    import prepare_identity_paired_insertion_dataset as builder
    import render_identity_replacement_review as replacement_renderer
except ModuleNotFoundError:  # Imported as ``scripts.*`` from the repository root.
    from scripts import prepare_identity_paired_insertion_dataset as builder
    from scripts import render_identity_replacement_review as replacement_renderer


MAJOR = "object"
MINOR = "daily_sports_objects"
TILE_SIZE = (840, 304)
PANEL_SIZE = (196, 188)


def _candidate_id(plan: builder.PairPlan) -> str:
    value = plan.extra.get("curation_candidate_id")
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"MeViS candidate lacks curation_candidate_id: {plan.identity_id}"
        )
    return value


def select_daily_backup_plans(
    plans: Sequence[builder.PairPlan],
    *,
    excluded_identities: set[str],
    selected_identities: set[str],
) -> tuple[list[builder.PairPlan], dict[str, int]]:
    """Return identity-unique, non-excluded, currently unused daily candidates."""

    category = [plan for plan in plans if plan.major == MAJOR and plan.minor == MINOR]
    identity_counts = Counter(plan.identity_id for plan in category)
    duplicate_identities = sorted(
        identity for identity, count in identity_counts.items() if count != 1
    )
    if duplicate_identities:
        raise ValueError(
            f"Daily pool is not identity-unique: {duplicate_identities[:5]}"
        )
    candidate_counts = Counter(_candidate_id(plan) for plan in category)
    duplicate_candidates = sorted(
        candidate for candidate, count in candidate_counts.items() if count != 1
    )
    if duplicate_candidates:
        raise ValueError(
            f"Daily pool has duplicate candidate IDs: {duplicate_candidates[:5]}"
        )

    nonexcluded = [
        plan for plan in category if plan.identity_id not in excluded_identities
    ]
    selected = [plan for plan in nonexcluded if plan.identity_id in selected_identities]
    backups = [
        plan for plan in nonexcluded if plan.identity_id not in selected_identities
    ]
    backups.sort(
        key=lambda plan: (
            float(plan.extra.get("selection_priority_penalty", 0.0)),
            -float(plan.selection_score),
            _candidate_id(plan),
        )
    )
    counts = {
        "category_candidate_count": len(category),
        "excluded_category_candidate_count": len(category) - len(nonexcluded),
        "nonexcluded_category_candidate_count": len(nonexcluded),
        "currently_selected_category_candidate_count": len(selected),
        "backup_candidate_count": len(backups),
    }
    if sum(
        counts[key]
        for key in (
            "excluded_category_candidate_count",
            "nonexcluded_category_candidate_count",
        )
    ) != len(category):
        raise AssertionError("Daily category accounting mismatch")
    if len(selected) + len(backups) != len(nonexcluded):
        raise AssertionError("Daily non-excluded accounting mismatch")
    return backups, counts


def _selected_context(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]]]:
    identities: set[str] = set()
    cases_by_video: dict[str, list[str]] = defaultdict(list)
    cases_by_frame: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        identity_id = str(row.get("identity_id", ""))
        case_id = str(row.get("case_id", ""))
        if not identity_id or not case_id:
            raise ValueError("Selected manifest row lacks identity_id or case_id")
        if identity_id in identities:
            raise ValueError(f"Selected manifest repeats identity: {identity_id}")
        identities.add(identity_id)
        if row.get("source_dataset") != "mevis_v2":
            continue
        for role in ("src", "tar"):
            upstream = row.get(f"{role}_upstream")
            image_id = row.get(f"{role}_image_id")
            if not isinstance(upstream, Mapping) or not isinstance(image_id, str):
                raise ValueError(
                    f"Selected MeViS row lacks {role} provenance: {case_id}"
                )
            video_id = str(upstream.get("video_id", ""))
            if not video_id:
                raise ValueError(f"Selected MeViS row lacks video ID: {case_id}")
            if case_id not in cases_by_video[video_id]:
                cases_by_video[video_id].append(case_id)
            cases_by_frame[image_id].append(case_id)
    return identities, dict(cases_by_video), dict(cases_by_frame)


def _plan_review_row(
    plan: builder.PairPlan,
    *,
    rank: int,
    selected_cases_by_video: Mapping[str, Sequence[str]],
    selected_cases_by_frame: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    video_id = str(plan.src_upstream["video_id"])
    if video_id != str(plan.tar_upstream["video_id"]):
        raise ValueError(f"Cross-video MeViS candidate: {_candidate_id(plan)}")
    same_video_cases = sorted(set(map(str, selected_cases_by_video.get(video_id, ()))))
    same_frame_cases = sorted(
        set(map(str, selected_cases_by_frame.get(plan.src_image_id, ())))
        | set(map(str, selected_cases_by_frame.get(plan.tar_image_id, ())))
    )
    return {
        "rank": rank,
        "candidate_id": _candidate_id(plan),
        "identity_id": plan.identity_id,
        "candidate_source": str(plan.extra.get("curation_candidate_source", "")),
        "raw_source_dataset": plan.source_dataset,
        "difficulty": plan.difficulty,
        "selection_score": float(plan.selection_score),
        "selection_priority_penalty": float(
            plan.extra.get("selection_priority_penalty", 0.0)
        ),
        "video_id": video_id,
        "src_frame": {
            "frame_id": str(plan.src_upstream["frame_id"]),
            "frame_index": int(plan.src_upstream["frame_index"]),
            "image_id": plan.src_image_id,
        },
        "tar_frame": {
            "frame_id": str(plan.tar_upstream["frame_id"]),
            "frame_index": int(plan.tar_upstream["frame_index"]),
            "image_id": plan.tar_image_id,
        },
        "selected_same_video_case_ids": same_video_cases,
        "selected_same_frame_case_ids": same_frame_cases,
        "selected_video_overlap": bool(same_video_cases),
        "selected_frame_overlap": bool(same_frame_cases),
    }


def _review_tile(plan: builder.PairPlan, row: Mapping[str, Any]) -> Image.Image:
    with Image.open(plan.src_image_path) as handle:
        src_image = handle.convert("RGB")
    with Image.open(plan.tar_image_path) as handle:
        tar_image = handle.convert("RGB")
    src_mask = builder.decode_coco_rle(plan.src_mask_payload)
    tar_mask = builder.decode_coco_rle(plan.tar_mask_payload)
    if src_mask.size != src_image.size or tar_mask.size != tar_image.size:
        raise ValueError(f"Native image/mask size mismatch: {_candidate_id(plan)}")
    clean_reference, clean_mask, _ = builder.crop_clean_reference(src_image, src_mask)
    panels = (
        builder._fit_panel(builder._mask_outline(src_image, src_mask), PANEL_SIZE),
        builder._fit_panel(
            builder._mask_outline(clean_reference, clean_mask), PANEL_SIZE
        ),
        builder._fit_panel(builder._masked_target(tar_image, tar_mask), PANEL_SIZE),
        builder._fit_panel(builder._mask_outline(tar_image, tar_mask), PANEL_SIZE),
    )

    tile = Image.new("RGB", TILE_SIZE, (248, 248, 248))
    panel_x = (8, 216, 424, 632)
    for x, panel in zip(panel_x, panels):
        tile.paste(panel, (x, 88))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (8, 4), str(row["candidate_id"]), font=builder._font(14), fill=(15, 15, 15)
    )
    draw.text(
        (8, 23), str(row["identity_id"]), font=builder._font(12), fill=(35, 35, 35)
    )
    overlap = "yes" if row["selected_video_overlap"] else "no"
    source_line = (
        f"source={row['candidate_source']} / {row['raw_source_dataset']} | "
        f"score={row['selection_score']:.3f} | penalty={row['selection_priority_penalty']:.1f} | "
        f"selected-video-overlap={overlap}"
    )
    draw.text((8, 42), source_line, font=builder._font(11), fill=(55, 55, 55))
    src_frame, tar_frame = row["src_frame"], row["tar_frame"]
    frame_line = (
        f"frames: src={src_frame['frame_id']}[idx={src_frame['frame_index']}] -> "
        f"tar={tar_frame['frame_id']}[idx={tar_frame['frame_index']}] | video={row['video_id']}"
    )
    draw.text((8, 59), frame_line, font=builder._font(11), fill=(55, 55, 55))
    if row["selected_frame_overlap"]:
        conflict = "selected-frame-overlap: " + ", ".join(
            row["selected_same_frame_case_ids"]
        )
        draw.text((8, 74), conflict[:120], font=builder._font(10), fill=(180, 45, 25))
    for x, label in zip(
        panel_x,
        ("SRC NATIVE + MASK", "REFERENCE CROP", "MASKED TARGET", "TARGET GT + MASK"),
    ):
        draw.text((x + 2, 282), label, font=builder._font(10), fill=(35, 35, 35))
    return tile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--plan-cache", type=Path, required=True)
    parser.add_argument("--curation-pool", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=builder.DEFAULT_SEED)
    parser.add_argument("--cases-per-sheet", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cases_per_sheet < 1:
        raise ValueError("cases-per-sheet must be positive")
    paths = {
        "plan_cache": args.plan_cache.resolve(),
        "curation_pool": args.curation_pool.resolve(),
        "exclusions": args.exclusions.resolve(),
        "selected_manifest": args.selected_manifest.resolve(),
        "builder": Path(builder.__file__).resolve(),
        "renderer": Path(__file__).resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing review input(s): {missing}")
    input_sha256 = {name: builder.file_sha256(path) for name, path in paths.items()}

    mevis_plans, _ = builder.load_plan_cache(
        paths["plan_cache"],
        seed=args.seed,
        raw_root=args.raw_root.resolve(),
        mevis_curation_pool_path=paths["curation_pool"],
    )
    excluded, exclusions_sha256 = builder.load_excluded_identities(paths["exclusions"])
    if exclusions_sha256 != input_sha256["exclusions"]:
        raise RuntimeError("Exclusion file changed while loading")
    selected_rows, selected_manifest_sha256 = replacement_renderer._jsonl_snapshot(
        paths["selected_manifest"]
    )
    if selected_manifest_sha256 != input_sha256["selected_manifest"]:
        raise RuntimeError("Selected manifest changed while loading")
    selected_identities, selected_cases_by_video, selected_cases_by_frame = (
        _selected_context(selected_rows)
    )
    backups, counts = select_daily_backup_plans(
        mevis_plans,
        excluded_identities=excluded,
        selected_identities=selected_identities,
    )
    review_rows = [
        _plan_review_row(
            plan,
            rank=index + 1,
            selected_cases_by_video=selected_cases_by_video,
            selected_cases_by_frame=selected_cases_by_frame,
        )
        for index, plan in enumerate(backups)
    ]

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Review output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}.")
    )
    try:
        sheets: list[dict[str, Any]] = []
        for sheet_index, start in enumerate(
            range(0, len(backups), args.cases_per_sheet)
        ):
            subset = backups[start : start + args.cases_per_sheet]
            subset_rows = review_rows[start : start + args.cases_per_sheet]
            columns = 2
            rows = math.ceil(len(subset) / columns)
            canvas = Image.new(
                "RGB", (columns * TILE_SIZE[0], rows * TILE_SIZE[1]), (225, 225, 225)
            )
            for position, (plan, row) in enumerate(zip(subset, subset_rows)):
                tile = _review_tile(plan, row)
                canvas.paste(
                    tile,
                    (
                        (position % columns) * TILE_SIZE[0],
                        (position // columns) * TILE_SIZE[1],
                    ),
                )
            relative = Path(f"sheet_{sheet_index:02d}.jpg")
            sheet_path = staging / relative
            replacement_renderer._save_jpeg_atomic(canvas, sheet_path)
            sheets.append(
                {
                    "sheet": relative.as_posix(),
                    "sha256": builder.file_sha256(sheet_path),
                    "candidate_ids": [str(row["candidate_id"]) for row in subset_rows],
                    "identity_ids": [str(row["identity_id"]) for row in subset_rows],
                    "review_status": "pending",
                }
            )

        source_counts = Counter(str(row["candidate_source"]) for row in review_rows)
        report = {
            "schema_version": 1,
            "status": "agent_identity_pool_visual_review_pending_not_human_certified",
            "selection_mode": "nonexcluded_and_not_selected_identity_backups",
            "major": MAJOR,
            "minor": MINOR,
            "seed": args.seed,
            "cases_per_sheet": args.cases_per_sheet,
            **counts,
            "backup_candidate_source_counts": dict(sorted(source_counts.items())),
            "backup_selected_video_overlap_count": sum(
                bool(row["selected_video_overlap"]) for row in review_rows
            ),
            "backup_selected_frame_overlap_count": sum(
                bool(row["selected_frame_overlap"]) for row in review_rows
            ),
            "inputs": {
                name: {"path": str(paths[name]), "sha256": input_sha256[name]}
                for name in (
                    "plan_cache",
                    "curation_pool",
                    "exclusions",
                    "selected_manifest",
                    "builder",
                    "renderer",
                )
            },
            "sheet_count": len(sheets),
            "sheets": sheets,
            "candidates": review_rows,
        }
        changed = [
            str(path)
            for name, path in paths.items()
            if builder.file_sha256(path) != input_sha256[name]
        ]
        if changed:
            raise RuntimeError(f"Review inputs changed during rendering: {changed}")
        builder.atomic_write_text(
            staging / "index.json",
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Review output appeared during rendering: {output_dir}"
            )
        os.rename(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

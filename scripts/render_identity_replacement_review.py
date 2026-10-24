#!/usr/bin/env python3
"""Render only identity-paired cases not already approved in a baseline review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

try:
    import prepare_identity_paired_insertion_dataset as builder
except ModuleNotFoundError:  # Imported as ``scripts.*`` from the repository root.
    from scripts import prepare_identity_paired_insertion_dataset as builder


SIGNATURE_FIELDS = (
    "identity_id",
    "major",
    "minor",
    "src_pixel_sha256",
    "src_mask_pixel_sha256",
    "tar_pixel_sha256",
    "tar_mask_pixel_sha256",
)


def review_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(record[field]) for field in SIGNATURE_FIELDS)


def _jsonl_snapshot(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    return rows, hashlib.sha256(payload).hexdigest()


def _json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8")), hashlib.sha256(payload).hexdigest()


def _save_jpeg_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            image.save(handle, format="JPEG", quality=94, subsampling=0)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases-per-sheet", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    renderer_path = Path(__file__).resolve()
    renderer_sha256 = builder.file_sha256(renderer_path)
    if args.cases_per_sheet < 1:
        raise ValueError("cases-per-sheet must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Replacement review output already exists: {args.output_dir}")
    baseline_manifest = args.baseline_root / "manifests" / "paperinsert_900.jsonl"
    candidate_manifest = args.candidate_root / "manifests" / "paperinsert_900.jsonl"
    baseline, baseline_manifest_sha256 = _jsonl_snapshot(baseline_manifest)
    candidate, candidate_manifest_sha256 = _jsonl_snapshot(candidate_manifest)
    exclusion_payload, exclusion_config_sha256 = _json_snapshot(args.exclusions)
    rejected = {
        str(row if isinstance(row, str) else row["identity_id"]) for row in exclusion_payload["excluded_identities"]
    }
    approved_signatures = {
        review_signature(record) for record in baseline if str(record["identity_id"]) not in rejected
    }
    pending = sorted(
        (record for record in candidate if review_signature(record) not in approved_signatures),
        key=lambda record: (
            str(record["major"]),
            str(record["minor"]),
            str(record["case_id"]),
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sheets: list[dict[str, Any]] = []
    for sheet_index, start in enumerate(range(0, len(pending), args.cases_per_sheet)):
        subset = pending[start : start + args.cases_per_sheet]
        columns = 2
        rows = (len(subset) + columns - 1) // columns
        canvas = Image.new("RGB", (columns * 620, rows * 248), "white")
        for position, record in enumerate(subset):
            tile = builder._case_review_tile(args.candidate_root, record)
            canvas.paste(tile, ((position % columns) * 620, (position // columns) * 248))
        relative = Path(f"sheet_{sheet_index:02d}.jpg")
        _save_jpeg_atomic(canvas, args.output_dir / relative)
        sheets.append(
            {
                "sheet": relative.as_posix(),
                "case_ids": [str(record["case_id"]) for record in subset],
                "identity_ids": [str(record["identity_id"]) for record in subset],
                "review_status": "pending",
            }
        )
    report = {
        "schema_version": 1,
        "status": "agent_replacement_visual_review_pending_not_human_certified",
        "signature_fields": list(SIGNATURE_FIELDS),
        "renderer": str(renderer_path),
        "renderer_sha256": renderer_sha256,
        "cases_per_sheet": args.cases_per_sheet,
        "baseline_root": str(args.baseline_root.resolve()),
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "candidate_root": str(args.candidate_root.resolve()),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "exclusion_config": str(args.exclusions.resolve()),
        "exclusion_config_sha256": exclusion_config_sha256,
        "baseline_case_count": len(baseline),
        "candidate_case_count": len(candidate),
        "approved_baseline_signature_count": len(approved_signatures),
        "pending_case_count": len(pending),
        "sheet_count": len(sheets),
        "sheets": sheets,
    }
    current_inputs = {
        baseline_manifest: baseline_manifest_sha256,
        candidate_manifest: candidate_manifest_sha256,
        args.exclusions: exclusion_config_sha256,
        renderer_path: renderer_sha256,
    }
    changed = [
        str(path) for path, expected_sha256 in current_inputs.items() if builder.file_sha256(path) != expected_sha256
    ]
    if changed:
        raise RuntimeError(f"Replacement-review inputs changed during rendering: {changed}")
    builder.atomic_write_text(
        args.output_dir / "index.json",
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

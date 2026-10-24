#!/usr/bin/env python3
"""Export review-gallery ``keep`` cases into a flat, post-processing-friendly tree.

Default layout::

    output/review_keep_export/
      <suite>/<case_id>/
        ref_image.png
        ref_mask.png
        tar_image.png
        tar_mask.png
        results/
          ours.png
          anydoor.png
          ...
        meta.json
      manifest.jsonl
      summary.json

Usage::

    python scripts/export_review_keeps.py
    python scripts/export_review_keeps.py --suites anyinsertion,paperinsert_900
    python scripts/export_review_keeps.py --decision keep --output-dir output/review_keep_export
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse gallery path resolution
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "review_gallery"))
from server import GalleryIndex, METHOD_ORDER  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS = REPO / "output" / "review_gallery_decisions"
DEFAULT_OUT = REPO / "output" / "review_keep_export"


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["case_id"])] = row
    return out


def safe_case_dirname(case_id: str) -> str:
    # Keep nested ids readable: accessory/1 -> accessory__1
    return case_id.replace("/", "__")


def copy_file(src: Path, dst: Path) -> bool:
    if not src.is_file() or src.stat().st_size <= 0:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def export_suite(
    suite: str,
    decision: str,
    decisions_dir: Path,
    index: GalleryIndex,
    out_root: Path,
) -> dict[str, Any]:
    dec_path = decisions_dir / f"{suite}.jsonl"
    decisions = load_decisions(dec_path)
    keep_ids = sorted(
        cid for cid, row in decisions.items() if row.get("decision") == decision
    )
    cases = index.cases.get(suite, {})
    exported = 0
    missing_case = 0
    rows_out: list[dict[str, Any]] = []

    for cid in keep_ids:
        case = cases.get(cid)
        if case is None:
            missing_case += 1
            rows_out.append(
                {
                    "suite": suite,
                    "case_id": cid,
                    "status": "missing_case_in_index",
                    "decision": decisions[cid],
                }
            )
            continue

        case_dir = out_root / suite / safe_case_dirname(cid)
        results_dir = case_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        copied: dict[str, str] = {}
        for key, src in [
            ("ref_image", Path(case.ref_image)),
            ("ref_mask", Path(case.ref_mask)),
            ("tar_image", Path(case.tar_image)),
            ("tar_mask", Path(case.tar_mask)),
        ]:
            # preserve extension
            ext = src.suffix.lower() or ".png"
            dst = case_dir / f"{key}{ext}"
            if copy_file(src, dst):
                copied[key] = str(dst.relative_to(out_root))

        method_copied: dict[str, str] = {}
        for method in METHOD_ORDER:
            src_s = case.results.get(method)
            if not src_s:
                continue
            src = Path(src_s)
            # Prefer restored_raw basename awareness; always save as method.png
            ext = src.suffix.lower() or ".png"
            dst = results_dir / f"{method}{ext}"
            if copy_file(src, dst):
                method_copied[method] = str(dst.relative_to(out_root))

        meta = {
            "suite": suite,
            "case_id": cid,
            "major": case.major,
            "minor": case.minor,
            "decision": decisions[cid],
            "source_paths": {
                "ref_image": case.ref_image,
                "ref_mask": case.ref_mask,
                "tar_image": case.tar_image,
                "tar_mask": case.tar_mask,
                "results": case.results,
            },
            "copied": {"inputs": copied, "results": method_copied},
            "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (case_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        exported += 1
        rows_out.append(
            {
                "suite": suite,
                "case_id": cid,
                "status": "ok",
                "dir": str((case_dir).relative_to(out_root)),
                "n_methods": len(method_copied),
                "methods": sorted(method_copied),
                "decision": decisions[cid].get("decision"),
                "note": decisions[cid].get("note", ""),
                "stars": decisions[cid].get("stars", 0),
            }
        )

    return {
        "suite": suite,
        "decision": decision,
        "n_keep": len(keep_ids),
        "exported": exported,
        "missing_case": missing_case,
        "rows": rows_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--decisions-dir",
        type=Path,
        default=DEFAULT_DECISIONS,
        help="review_gallery decisions jsonl dir",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="destination root for keep export",
    )
    ap.add_argument(
        "--suites",
        type=str,
        default="anyinsertion,paperinsert_900,testset_20260120",
        help="comma-separated suites",
    )
    ap.add_argument("--decision", type=str, default="keep")
    ap.add_argument(
        "--clean",
        action="store_true",
        help="remove output-dir before export",
    )
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    out_root = args.output_dir.resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("[export] building gallery index …", flush=True)
    index = GalleryIndex()
    index.build()

    summary: dict[str, Any] = {
        "decision": args.decision,
        "output_dir": str(out_root),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "suites": {},
    }
    manifest_path = out_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as mf:
        for suite in suites:
            print(f"[export] suite={suite} decision={args.decision}", flush=True)
            rep = export_suite(
                suite=suite,
                decision=args.decision,
                decisions_dir=args.decisions_dir,
                index=index,
                out_root=out_root,
            )
            summary["suites"][suite] = {
                k: v for k, v in rep.items() if k != "rows"
            }
            for row in rep["rows"]:
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"  keep={rep['n_keep']} exported={rep['exported']} "
                f"missing_case={rep['missing_case']}",
                flush=True,
            )

    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # also dump raw decisions snapshot
    snap = out_root / "decisions_snapshot"
    snap.mkdir(exist_ok=True)
    for suite in suites:
        src = args.decisions_dir / f"{suite}.jsonl"
        if src.is_file():
            shutil.copy2(src, snap / f"{suite}.jsonl")

    print(f"\n[done] {out_root}", flush=True)
    print(f"  manifest: {manifest_path}", flush=True)
    print(f"  summary:  {out_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

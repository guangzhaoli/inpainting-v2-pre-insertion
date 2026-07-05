#!/usr/bin/env python3
"""
Scan an Insert Anything local folder dataset for structural and file-level issues.

Checks include:
- missing required directories
- missing paired files per sample stem
- orphan files not referenced by the canonical ref_image stem set
- empty files
- unreadable images or masks
- dimension mismatches for paired image/mask files
- labels.json coverage mismatches
- unexpected directories and files under category roots

Examples:
  python script/scan_insert_anything_dataset.py \
    --root /path/to/data \
    --json-out logs/dataset_scan/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image, UnidentifiedImageError


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PRIMARY_DIRS = ("ref_image", "ref_mask", "tar_image", "tar_mask")
MASK_DIR_NAMES = {
    "ref_mask",
    "ref_mask_1",
    "tar_mask",
    "tar_mask_1",
    "tar_mask_2",
    "tar_mask_erode",
}
KNOWN_DIRS = {
    "ref_image",
    "ref_image_1",
    "ref_mask",
    "ref_mask_1",
    "tar_image",
    "tar_image_1",
    "tar_mask",
    "tar_mask_1",
    "tar_mask_2",
    "tar_mask_erode",
}
KNOWN_ROOT_FILES = {
    "labels.json",
    "ref_vs_output_check_qwen3vl.json",
    "solid_color_check.json",
    "solid_color_check_qwen3vl.json",
    "solid_color_check_qwen3vl_2.json",
    "solid_color_check_with_reason.json",
    "ref_mask.zip",
    "tar_mask.zip",
}
JSON_ARTIFACT_SUFFIXES = (".json",)
ZIP_ARTIFACT_SUFFIXES = (".zip",)


@dataclass
class FileInspection:
    path: str
    category: str
    dir_name: str
    stem: str
    is_mask: bool
    exists: bool
    size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    image_mode: Optional[str] = None
    fmt: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    all_zero_mask: bool = False
    all_full_mask: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Insert Anything dataset for bad or inconsistent cases.")
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root directory. Example: /path/to/datasets/inpainting/data",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional JSON output path for the full report.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, max(4, (os.cpu_count() or 4))),
        help="Thread workers for file inspection.",
    )
    parser.add_argument(
        "--check-mask-content",
        action="store_true",
        help="Flag all-zero and all-255 masks as suspicious.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="How many example paths to print per issue type in terminal summary.",
    )
    parser.add_argument(
        "--tar-image-only",
        action="store_true",
        help="Only inspect primary tar_image files and labels.json coverage.",
    )
    parser.add_argument(
        "--ignore-name-suffixes",
        nargs="*",
        default=[],
        help="Ignore category or directory names ending with these suffixes. Example: --ignore-name-suffixes _1 _2",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def has_ignored_suffix(name: str, suffixes: List[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def normalize_base_dir_name(dir_name: str) -> Optional[str]:
    if dir_name in {"ref_image", "ref_mask", "tar_image", "tar_mask"}:
        return dir_name
    if dir_name.startswith("ref_image_"):
        return "ref_image"
    if dir_name.startswith("ref_mask_"):
        return "ref_mask"
    if dir_name.startswith("tar_image_"):
        return "tar_image"
    if dir_name.startswith("tar_mask_") or dir_name == "tar_mask_erode":
        return "tar_mask"
    return None


def find_image_files(directory: Path) -> List[Path]:
    return sorted(path for path in directory.iterdir() if is_image_file(path))


def inspect_file(path: Path, category: str, dir_name: str, check_mask_content: bool) -> FileInspection:
    is_mask = dir_name in MASK_DIR_NAMES or "mask" in dir_name
    info = FileInspection(
        path=str(path),
        category=category,
        dir_name=dir_name,
        stem=path.stem,
        is_mask=is_mask,
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else -1,
    )
    if not info.exists:
        info.error_type = "missing_file"
        info.error_message = "path does not exist"
        return info
    if info.size_bytes == 0:
        info.error_type = "empty_file"
        info.error_message = "file size is 0 bytes"
        return info

    try:
        with Image.open(path) as img:
            img.load()
            info.width, info.height = img.size
            info.image_mode = img.mode
            info.fmt = img.format
            if check_mask_content and is_mask:
                gray = img.convert("L")
                min_v, max_v = gray.getextrema()
                info.all_zero_mask = max_v == 0
                info.all_full_mask = min_v == 255
    except UnidentifiedImageError as exc:
        info.error_type = "unidentified_image"
        info.error_message = str(exc)
    except OSError as exc:
        info.error_type = "oserror_image"
        info.error_message = str(exc)
    except Exception as exc:  # pragma: no cover - keep the scanner resilient
        info.error_type = type(exc).__name__
        info.error_message = str(exc)
    return info


def add_issue(issues: Dict[str, List[Dict[str, Any]]], issue_type: str, **payload: Any) -> None:
    issues[issue_type].append(payload)


def collect_dir_stems(files: Iterable[Path]) -> Dict[str, List[Path]]:
    stems: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        stems[path.stem].append(path)
    return stems


def summarize_issue_counts(issues: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    return {issue_type: len(items) for issue_type, items in sorted(issues.items()) if items}


def scan_category(
    category_dir: Path,
    root: Path,
    check_mask_content: bool,
    workers: int,
    tar_image_only: bool,
    ignore_name_suffixes: List[str],
) -> Dict[str, Any]:
    category = category_dir.name
    issues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not category_dir.exists():
        add_issue(
            issues,
            "missing_category_dir",
            category=category,
            path=relpath(category_dir, root),
            detail="category directory disappeared before scanning",
        )
        return {
            "category": category,
            "total_image_files": 0,
            "canonical_ref_stems": 0,
            "issue_counts": summarize_issue_counts(issues),
            "issues": {issue_type: items for issue_type, items in sorted(issues.items()) if items},
        }

    entries = sorted(category_dir.iterdir(), key=lambda p: p.name)
    child_dirs = {entry.name: entry for entry in entries if entry.is_dir()}
    child_files = {entry.name: entry for entry in entries if entry.is_file()}

    required_dirs_missing = (
        ["tar_image"] if tar_image_only and "tar_image" not in child_dirs
        else [name for name in PRIMARY_DIRS if not tar_image_only and name not in child_dirs]
    )
    for dir_name in required_dirs_missing:
        add_issue(
            issues,
            "missing_required_dir",
            category=category,
            path=relpath(category_dir / dir_name, root),
            detail=f"required directory '{dir_name}' is missing",
        )

    for dir_name, path in child_dirs.items():
        if has_ignored_suffix(dir_name, ignore_name_suffixes):
            continue
        if tar_image_only and dir_name != "tar_image":
            continue
        if dir_name not in KNOWN_DIRS:
            add_issue(
                issues,
                "unexpected_dir",
                category=category,
                path=relpath(path, root),
                detail=f"unexpected directory '{dir_name}' under category root",
            )

    for file_name, path in child_files.items():
        if file_name in KNOWN_ROOT_FILES:
            continue
        if path.suffix.lower() in JSON_ARTIFACT_SUFFIXES or path.suffix.lower() in ZIP_ARTIFACT_SUFFIXES:
            add_issue(
                issues,
                "unexpected_root_file",
                category=category,
                path=relpath(path, root),
                detail=f"unexpected root artifact '{file_name}'",
            )
        else:
            add_issue(
                issues,
                "unexpected_root_file",
                category=category,
                path=relpath(path, root),
                detail=f"unexpected file '{file_name}' under category root",
            )

    dir_files: Dict[str, List[Path]] = {}
    dir_stems: Dict[str, Dict[str, List[Path]]] = {}
    all_image_files: List[Path] = []

    for dir_name, dir_path in child_dirs.items():
        if has_ignored_suffix(dir_name, ignore_name_suffixes):
            continue
        if tar_image_only and dir_name != "tar_image":
            continue
        files = find_image_files(dir_path)
        dir_files[dir_name] = files
        dir_stems[dir_name] = collect_dir_stems(files)
        all_image_files.extend(files)

    inspection_by_key: Dict[tuple[str, str], FileInspection] = {}
    total_files = len(all_image_files)

    if total_files > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(inspect_file, path, category, path.parent.name, check_mask_content): path
                for path in all_image_files
            }
            processed = 0
            for future in as_completed(futures):
                inspected = future.result()
                inspection_by_key[(inspected.dir_name, inspected.stem)] = inspected
                processed += 1
                if processed % 5000 == 0:
                    print(
                        f"[scan] {category}: validated {processed}/{total_files} files",
                        file=sys.stderr,
                        flush=True,
                    )

    for dir_name, stem_map in dir_stems.items():
        for stem, paths in stem_map.items():
            if len(paths) > 1:
                add_issue(
                    issues,
                    "duplicate_stem_in_dir",
                    category=category,
                    stem=stem,
                    path=relpath(paths[0].parent, root),
                    files=[relpath(path, root) for path in paths],
                    detail=f"multiple files with the same stem '{stem}' in '{dir_name}'",
                )

    for inspected in inspection_by_key.values():
        if inspected.error_type:
            add_issue(
                issues,
                inspected.error_type,
                category=category,
                stem=inspected.stem,
                path=relpath(Path(inspected.path), root),
                detail=inspected.error_message,
            )
        elif inspected.is_mask and check_mask_content and not tar_image_only:
            if inspected.all_zero_mask:
                add_issue(
                    issues,
                    "all_zero_mask",
                    category=category,
                    stem=inspected.stem,
                    path=relpath(Path(inspected.path), root),
                    detail="mask pixels are all 0",
                )
            if inspected.all_full_mask:
                add_issue(
                    issues,
                    "all_full_mask",
                    category=category,
                    stem=inspected.stem,
                    path=relpath(Path(inspected.path), root),
                    detail="mask pixels are all 255",
                )

    canonical_ref_stems = set(dir_stems.get("ref_image", {}).keys())
    ref_mask_stems = set(dir_stems.get("ref_mask", {}).keys())
    tar_image_stems = set(dir_stems.get("tar_image", {}).keys())
    tar_mask_stems = set(dir_stems.get("tar_mask", {}).keys())

    if not tar_image_only:
        for stem in sorted(canonical_ref_stems - ref_mask_stems):
            add_issue(
                issues,
                "missing_ref_mask",
                category=category,
                stem=stem,
                path=relpath(category_dir / "ref_mask", root),
                detail="ref_mask missing for ref_image stem",
            )
        for stem in sorted(canonical_ref_stems - tar_image_stems):
            add_issue(
                issues,
                "missing_tar_image",
                category=category,
                stem=stem,
                path=relpath(category_dir / "tar_image", root),
                detail="tar_image missing for ref_image stem",
            )
        for stem in sorted(canonical_ref_stems - tar_mask_stems):
            add_issue(
                issues,
                "missing_tar_mask",
                category=category,
                stem=stem,
                path=relpath(category_dir / "tar_mask", root),
                detail="tar_mask missing for ref_image stem",
            )
        for stem in sorted(ref_mask_stems - canonical_ref_stems):
            add_issue(
                issues,
                "orphan_ref_mask",
                category=category,
                stem=stem,
                path=relpath(category_dir / "ref_mask", root),
                detail="ref_mask exists without matching ref_image",
            )
        for stem in sorted(tar_image_stems - canonical_ref_stems):
            add_issue(
                issues,
                "orphan_tar_image",
                category=category,
                stem=stem,
                path=relpath(category_dir / "tar_image", root),
                detail="tar_image exists without matching ref_image",
            )
        for stem in sorted(tar_mask_stems - canonical_ref_stems):
            add_issue(
                issues,
                "orphan_tar_mask",
                category=category,
                stem=stem,
                path=relpath(category_dir / "tar_mask", root),
                detail="tar_mask exists without matching ref_image",
            )

        for dir_name, stem_map in dir_stems.items():
            base_dir = normalize_base_dir_name(dir_name)
            if base_dir is None or dir_name == base_dir:
                continue
            base_stems = set(dir_stems.get(base_dir, {}).keys())
            current_stems = set(stem_map.keys())
            for stem in sorted(base_stems - current_stems):
                add_issue(
                    issues,
                    "missing_variant_file",
                    category=category,
                    stem=stem,
                    path=relpath(category_dir / dir_name, root),
                    detail=f"variant directory '{dir_name}' is missing stem present in '{base_dir}'",
                )
            for stem in sorted(current_stems - base_stems):
                add_issue(
                    issues,
                    "orphan_variant_file",
                    category=category,
                    stem=stem,
                    path=relpath(category_dir / dir_name, root),
                    detail=f"variant directory '{dir_name}' contains stem not present in '{base_dir}'",
                )

        def compare_pair(left_dir: str, right_dir: str, issue_type: str) -> None:
            common = sorted(set(dir_stems.get(left_dir, {}).keys()) & set(dir_stems.get(right_dir, {}).keys()))
            for stem in common:
                left = inspection_by_key.get((left_dir, stem))
                right = inspection_by_key.get((right_dir, stem))
                if left is None or right is None:
                    continue
                if left.error_type or right.error_type:
                    continue
                if left.width != right.width or left.height != right.height:
                    add_issue(
                        issues,
                        issue_type,
                        category=category,
                        stem=stem,
                        path_left=relpath(Path(left.path), root),
                        path_right=relpath(Path(right.path), root),
                        detail=f"size mismatch: {left.width}x{left.height} vs {right.width}x{right.height}",
                    )

        compare_pair("ref_image", "ref_mask", "ref_image_ref_mask_size_mismatch")
        compare_pair("ref_image_1", "ref_mask_1", "ref_image1_ref_mask1_size_mismatch")
        compare_pair("tar_image", "tar_mask", "tar_image_tar_mask_size_mismatch")
        compare_pair("tar_image", "tar_mask_1", "tar_image_tar_mask1_size_mismatch")
        compare_pair("tar_image", "tar_mask_2", "tar_image_tar_mask2_size_mismatch")
        compare_pair("tar_image", "tar_mask_erode", "tar_image_tar_mask_erode_size_mismatch")

    labels_path = category_dir / "labels.json"
    if labels_path.exists():
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            if not isinstance(labels, dict):
                add_issue(
                    issues,
                    "invalid_labels_json",
                    category=category,
                    path=relpath(labels_path, root),
                    detail="labels.json is not a JSON object",
                )
            else:
                tar_filenames = {path.name for path in dir_files.get("tar_image", [])}
                label_keys = set(labels.keys())
                for filename in sorted(tar_filenames - label_keys):
                    add_issue(
                        issues,
                        "missing_label_entry",
                        category=category,
                        path=relpath(labels_path, root),
                        detail=f"tar_image file '{filename}' missing from labels.json",
                    )
                for filename in sorted(label_keys - tar_filenames):
                    add_issue(
                        issues,
                        "orphan_label_entry",
                        category=category,
                        path=relpath(labels_path, root),
                        detail=f"labels.json entry '{filename}' has no matching tar_image file",
                    )
        except Exception as exc:  # pragma: no cover - keep report generation robust
            add_issue(
                issues,
                "invalid_labels_json",
                category=category,
                path=relpath(labels_path, root),
                detail=f"failed to parse labels.json: {exc}",
            )

    return {
        "category": category,
        "total_image_files": total_files,
        "canonical_ref_stems": len(canonical_ref_stems),
        "issue_counts": summarize_issue_counts(issues),
        "issues": {issue_type: items for issue_type, items in sorted(issues.items()) if items},
    }


def build_terminal_summary(report: Dict[str, Any], print_limit: int) -> str:
    lines: List[str] = []
    summary = report["summary"]
    lines.append("Dataset Scan Summary")
    lines.append(f"Root: {report['root']}")
    lines.append(f"Categories: {summary['total_categories']}")
    lines.append(f"Image files scanned: {summary['total_image_files_scanned']}")
    lines.append(f"Issue instances: {summary['total_issue_instances']}")
    lines.append("")
    lines.append("Issue counts:")
    for issue_type, count in summary["issue_counts"].items():
        lines.append(f"  {issue_type}: {count}")

    issue_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for category_result in report["categories"]:
        for issue_type, items in category_result["issues"].items():
            if len(issue_examples[issue_type]) >= print_limit:
                continue
            remaining = print_limit - len(issue_examples[issue_type])
            issue_examples[issue_type].extend(items[:remaining])

    if issue_examples:
        lines.append("")
        lines.append("Examples:")
        for issue_type, items in sorted(issue_examples.items()):
            lines.append(f"  {issue_type}:")
            for item in items:
                location = item.get("path") or item.get("path_left") or item.get("category", "<unknown>")
                detail = item.get("detail", "")
                stem = item.get("stem")
                if stem:
                    lines.append(f"    - {location} | stem={stem} | {detail}")
                else:
                    lines.append(f"    - {location} | {detail}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"Dataset root does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    category_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and not has_ignored_suffix(path.name, args.ignore_name_suffixes)
    )
    report_categories: List[Dict[str, Any]] = []
    aggregate_issue_counter: Counter[str] = Counter()
    total_image_files_scanned = 0

    print(f"[scan] root={root}", file=sys.stderr, flush=True)
    print(f"[scan] categories={len(category_dirs)}", file=sys.stderr, flush=True)

    for index, category_dir in enumerate(category_dirs, start=1):
        print(f"[scan] category {index}/{len(category_dirs)}: {category_dir.name}", file=sys.stderr, flush=True)
        category_result = scan_category(
            category_dir=category_dir,
            root=root,
            check_mask_content=args.check_mask_content and not args.tar_image_only,
            workers=args.workers,
            tar_image_only=args.tar_image_only,
            ignore_name_suffixes=args.ignore_name_suffixes,
        )
        report_categories.append(category_result)
        total_image_files_scanned += category_result["total_image_files"]
        aggregate_issue_counter.update(category_result["issue_counts"])

    report = {
        "root": str(root),
        "summary": {
            "total_categories": len(category_dirs),
            "total_image_files_scanned": total_image_files_scanned,
            "total_issue_instances": sum(aggregate_issue_counter.values()),
            "issue_counts": dict(sorted(aggregate_issue_counter.items())),
        },
        "categories": report_categories,
    }

    print(build_terminal_summary(report, args.print_limit))

    if args.json_out:
        json_out = Path(args.json_out).expanduser().resolve()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report written to: {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check PaperInsert candidates against the model training image pool.

The scanner uses decoded-pixel SHA256 for exact matches and a 64-bit pHash
BK-tree for near-duplicate candidates.  Results are cached as JSONL so a long
training-pool scan can be resumed safely in a zmx session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import stat
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import imagehash
import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CACHE_SCHEMA_VERSION = 2
FINGERPRINT_ALGORITHM = "decoded-rgb-sha256+imagehash-phash8-hf4+dhash8"
FULL_SOURCE_FINGERPRINT_FIELDS = (
    "src_full_pixel_sha256",
    "src_full_phash",
    "src_full_dhash",
    "src_full_width",
    "src_full_height",
)
DEFAULT_DATASET_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
    "a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900"
)
DEFAULT_TRAIN_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data"
)
PRIMARY_MANIFEST_NAMES = ("paperinsert_900.jsonl", "anyinsertion_official.jsonl")
CHECKSUM_MANIFEST_NAME = "CHECKSUMS.sha256"
CANONICAL_FIELDS = ("src_image", "src_mask", "tar_image", "tar_mask")
CORE_AUTHORITATIVE_PATHS = (
    "manifests/paperinsert_900.jsonl",
    "manifests/paperinsert_900.csv",
    "manifests/object.jsonl",
    "manifests/person.jsonl",
    "manifests/garment.jsonl",
    "reports/sources.json",
)
ANYINSERTION_AUTHORITATIVE_PATHS = (
    "manifests/anyinsertion_official.jsonl",
    "manifests/anyinsertion_clean67_for_current_model.jsonl",
    "manifests/anyinsertion_clean67_case_ids.txt",
    "manifests/anyinsertion_current_model_overlap_excluded33.json",
    "reports/conversion.json",
)
SUPPORTED_PROTOCOLS = {
    "paired_reconstruction",
    "cross_instance_replace",
    "official_anyinsertion",
}


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(np.asarray(rgb, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_fingerprints(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {
            "width": rgb.width,
            "height": rgb.height,
            "pixel_sha256": decoded_pixel_sha256(rgb),
            "phash": str(imagehash.phash(rgb, hash_size=8, highfreq_factor=4)),
            "dhash": str(imagehash.dhash(rgb, hash_size=8)),
        }


def fingerprint_cache_row(path_key: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "path": path_key,
    }
    try:
        path = Path(path_key)
        stat = path.stat()
        row.update(
            {
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                **image_fingerprints(path),
            }
        )
    except Exception as error:
        row["error"] = repr(error)
    return row


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass
class BKNode:
    value: int
    keys: list[str] = field(default_factory=list)
    children: dict[int, "BKNode"] = field(default_factory=dict)


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    def add(self, value: int, key: str) -> None:
        if self.root is None:
            self.root = BKNode(value=value, keys=[key])
            return
        node = self.root
        while True:
            distance = hamming(value, node.value)
            if distance == 0:
                node.keys.append(key)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value=value, keys=[key])
                return
            node = child

    def query(self, value: int, radius: int) -> Iterator[tuple[int, str]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node.value)
            if distance <= radius:
                for key in node.keys:
                    yield distance, key
            lower = distance - radius
            upper = distance + radius
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    stack.append(child)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_safe_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one safe path component: {value!r}")


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute, normalized path without following symlinks."""

    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(path: Path, label: str, *, include_leaf: bool) -> None:
    """Reject an existing symlink in an output/input path before resolution."""

    absolute = _absolute_lexical(path)
    parts = absolute.parts[:-1] if not include_leaf else absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} uses an unsafe symlink component: {current}")


def _validate_contained_regular_file(
    root: Path,
    path: Path,
    label: str,
) -> Path:
    root = root.resolve()
    lexical = _absolute_lexical(path)
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} is outside dataset root: {lexical}") from error
    _assert_no_symlink_components(lexical, label, include_leaf=True)
    try:
        mode = os.lstat(lexical).st_mode
    except FileNotFoundError as error:
        raise ValueError(f"Missing {label}: {lexical}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"{label} is not a regular file: {lexical}")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} resolves outside dataset root: {resolved}") from error
    return resolved


def _safe_checksum_relative_path(value: str, line_number: int) -> str:
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in value
        or "\x00" in value
        or relative.as_posix() != value
    ):
        raise ValueError(
            f"Unsafe checksum path at row {line_number}: {value!r}"
        )
    return value


def _read_authoritative_checksums(dataset_root: Path) -> tuple[Path, dict[str, str]]:
    checksum_path = _validate_contained_regular_file(
        dataset_root,
        dataset_root / CHECKSUM_MANIFEST_NAME,
        "checksum manifest",
    )
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read checksum manifest: {error}") from error
    if not lines:
        raise ValueError(f"Checksum manifest is empty: {checksum_path}")
    rows: dict[str, str] = {}
    order: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ValueError(f"Invalid checksum row {line_number}: {line!r}")
        digest, relative = match.groups()
        relative = _safe_checksum_relative_path(relative, line_number)
        if relative == CHECKSUM_MANIFEST_NAME:
            raise ValueError("Checksum manifest must not checksum itself")
        if relative in rows:
            raise ValueError(f"Duplicate checksum path: {relative}")
        rows[relative] = digest
        order.append(relative)
    if order != sorted(order):
        raise ValueError("Checksum rows must be sorted by canonical relative path")
    return checksum_path, rows


def _release_primary_manifest(dataset_root: Path) -> Path:
    manifest_dir = dataset_root / "manifests"
    primary_candidates = [
        _validate_contained_regular_file(
            dataset_root,
            manifest_dir / name,
            f"primary manifest {name}",
        )
        for name in PRIMARY_MANIFEST_NAMES
        if (manifest_dir / name).is_file()
    ]
    if len(primary_candidates) != 1:
        raise ValueError(
            "Expected exactly one supported primary manifest under "
            f"{manifest_dir}, found {[path.name for path in primary_candidates]}"
        )
    return primary_candidates[0]


def verify_release_integrity(
    dataset_root: Path,
    selected_manifest: Path,
) -> dict[str, Any]:
    """Authenticate the complete released asset inventory before fingerprinting.

    ``CHECKSUMS.sha256`` is authoritative only when its path set exactly matches
    the release schema.  This binds all four canonical assets for every primary
    case, the primary and selected manifests, and the release provenance file.
    """

    dataset_root = dataset_root.resolve()
    checksum_path, checksums = _read_authoritative_checksums(dataset_root)
    primary_manifest = _release_primary_manifest(dataset_root)
    selected_manifest = _validate_contained_regular_file(
        dataset_root,
        selected_manifest,
        "selected manifest",
    )
    primary_name = primary_manifest.name
    if primary_name == "paperinsert_900.jsonl":
        release_type = "paperinsert_900"
        authoritative_paths = CORE_AUTHORITATIVE_PATHS
        provenance_relative = "reports/sources.json"
    elif primary_name == "anyinsertion_official.jsonl":
        release_type = "anyinsertion_official"
        authoritative_paths = ANYINSERTION_AUTHORITATIVE_PATHS
        provenance_relative = "reports/conversion.json"
    else:  # guarded by _release_primary_manifest; retained as fail-closed defense
        raise ValueError(f"Unsupported primary release manifest: {primary_name}")

    try:
        primary_records = read_jsonl(primary_manifest)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid primary release manifest: {error}") from error
    if not primary_records:
        raise ValueError("Primary release manifest is empty")

    expected_paths = set(authoritative_paths)
    canonical_paths: set[str] = set()
    canonical_directories: set[Path] = set()
    seen_case_ids: set[str] = set()
    for row_index, record in enumerate(primary_records):
        if not isinstance(record, dict):
            raise ValueError(f"Primary manifest row {row_index} is not an object")
        case_id = str(record.get("case_id", ""))
        major = str(record.get("major", ""))
        validate_safe_component(case_id, f"Primary manifest row {row_index} case_id")
        validate_safe_component(major, f"{case_id}: major")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate benchmark case_id: {case_id}")
        seen_case_ids.add(case_id)
        for field in CANONICAL_FIELDS:
            relative_value = record.get(field)
            if not isinstance(relative_value, str) or not relative_value:
                raise ValueError(f"{case_id}: primary manifest is missing {field}")
            expected = (Path(major) / field / f"{case_id}.png").as_posix()
            if relative_value != expected:
                raise ValueError(
                    f"{case_id}: non-canonical {field} path: "
                    f"actual={relative_value!r} expected={expected!r}"
                )
            if relative_value in canonical_paths:
                raise ValueError(f"Duplicate canonical release path: {relative_value}")
            canonical_paths.add(relative_value)
            expected_paths.add(relative_value)
            canonical_directories.add((dataset_root / major / field).resolve())

    try:
        selected_relative = selected_manifest.relative_to(dataset_root).as_posix()
    except ValueError as error:  # defensive; containment was checked above
        raise ValueError("Selected manifest is outside dataset root") from error
    if selected_manifest.suffix != ".jsonl":
        raise ValueError(f"Selected manifest must be JSONL: {selected_manifest}")
    if selected_relative not in expected_paths:
        raise ValueError(
            "Selected manifest is not an authoritative checksummed release file: "
            f"{selected_relative}"
        )

    recorded_paths = set(checksums)
    if recorded_paths != expected_paths:
        raise ValueError(
            "Checksum path set mismatch: "
            f"missing={sorted(expected_paths - recorded_paths)[:10]} "
            f"extra={sorted(recorded_paths - expected_paths)[:10]}"
        )

    protected_paths: set[Path] = {checksum_path}
    for relative in sorted(expected_paths):
        protected = _validate_contained_regular_file(
            dataset_root,
            dataset_root / relative,
            f"checksummed release file {relative}",
        )
        actual_digest = file_sha256(protected)
        if actual_digest != checksums[relative]:
            raise ValueError(
                f"Checksum mismatch for {relative}: "
                f"recorded={checksums[relative]} actual={actual_digest}"
            )
        protected_paths.add(protected)

    provenance_path = (dataset_root / provenance_relative).resolve()
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Invalid release provenance {provenance_relative}: {error}") from error
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 3:
        raise ValueError(
            f"Release provenance {provenance_relative} must be a schema-3 JSON object"
        )

    return {
        "release_type": release_type,
        "primary_manifest": primary_manifest,
        "selected_manifest": selected_manifest,
        "checksum_manifest": checksum_path,
        "checksum_manifest_sha256": file_sha256(checksum_path),
        "checksum_entry_count": len(checksums),
        "provenance_file": provenance_path,
        "provenance_sha256": checksums[provenance_relative],
        "protected_paths": protected_paths,
        "canonical_directories": canonical_directories,
    }


def resolve_manifest_path(dataset_root: Path, manifest_path: Path | None = None) -> Path:
    dataset_root = dataset_root.resolve()
    if manifest_path is not None:
        resolved = (
            manifest_path if manifest_path.is_absolute() else dataset_root / manifest_path
        ).resolve()
        try:
            resolved.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Benchmark manifest resolves outside dataset root: {resolved}"
            ) from error
        if not resolved.is_file():
            raise ValueError(f"Benchmark manifest does not exist: {resolved}")
        return resolved
    manifest_dir = dataset_root / "manifests"
    existing = [
        manifest_dir / name
        for name in PRIMARY_MANIFEST_NAMES
        if (manifest_dir / name).is_file()
    ]
    if len(existing) != 1:
        raise ValueError(
            "Expected exactly one supported benchmark manifest under "
            f"{manifest_dir}, found {[path.name for path in existing]}"
        )
    return existing[0].resolve()


def validate_explicit_manifest_subset(
    dataset_root: Path,
    manifest_path: Path,
    records: Sequence[dict[str, Any]],
) -> Path:
    """Require an explicit protocol split to be an unchanged primary subset."""

    primary_candidates = [
        (dataset_root / "manifests" / name).resolve()
        for name in PRIMARY_MANIFEST_NAMES
        if (dataset_root / "manifests" / name).is_file()
    ]
    if len(primary_candidates) != 1:
        raise ValueError(
            "Explicit benchmark manifests require exactly one primary manifest; "
            f"found {[path.name for path in primary_candidates]}"
        )
    primary_path = primary_candidates[0]
    if manifest_path == primary_path:
        return primary_path
    primary_records = read_jsonl(primary_path)
    primary_index: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for position, record in enumerate(primary_records):
        key = (str(record.get("major", "")), str(record.get("case_id", "")))
        if key in primary_index:
            raise ValueError(f"Primary benchmark manifest has duplicate key: {key}")
        primary_index[key] = (position, record)
    positions: list[int] = []
    for record in records:
        key = (str(record.get("major", "")), str(record.get("case_id", "")))
        primary_entry = primary_index.get(key)
        if primary_entry is None:
            raise ValueError(f"Explicit benchmark manifest contains non-primary case: {key}")
        position, primary_record = primary_entry
        if record != primary_record:
            raise ValueError(
                f"Explicit benchmark manifest altered primary metadata for case: {key}"
            )
        positions.append(position)
    if positions != sorted(positions):
        raise ValueError("Explicit benchmark manifest must preserve primary row order")
    return primary_path


def full_source_fingerprints(record: dict[str, Any]) -> dict[str, Any]:
    """Read and validate a full source image's manifest fingerprints.

    The canonical ``src_image`` is an object crop, so it cannot prove that the
    original full source image was absent from training.  Records may therefore
    carry fingerprints for that full image even though the image is
    intentionally not duplicated inside the benchmark directory.  These fields
    remain mandatory for cross-instance records and optional (all-or-none) for
    legacy-compatible paired-reconstruction records.
    """

    case_id = str(record.get("case_id", "<unknown>"))
    protocol = str(record.get("protocol", "<unknown>"))
    missing = [
        field
        for field in FULL_SOURCE_FINGERPRINT_FIELDS
        if field not in record or record[field] is None or record[field] == ""
    ]
    if missing:
        raise ValueError(
            f"{case_id}: {protocol} record is missing required full "
            f"donor fingerprint fields: {', '.join(missing)}"
        )

    expected_hex_lengths = {
        "src_full_pixel_sha256": 64,
        "src_full_phash": 16,
        "src_full_dhash": 16,
    }
    for field, expected_length in expected_hex_lengths.items():
        value = record[field]
        if not isinstance(value, str) or len(value) != expected_length:
            raise ValueError(
                f"{case_id}: {field} must be a {expected_length}-character hex string"
            )
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{case_id}: {field} must be hexadecimal") from error

    for field in ("src_full_width", "src_full_height"):
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{case_id}: {field} must be a positive integer")

    return {
        "pixel_sha256": record["src_full_pixel_sha256"].lower(),
        "phash": record["src_full_phash"].lower(),
        "dhash": record["src_full_dhash"].lower(),
        "width": record["src_full_width"],
        "height": record["src_full_height"],
    }


def load_candidate_index(
    dataset_root: Path,
    manifest_path: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], BKTree, dict[str, list[str]]]:
    manifest_path = resolve_manifest_path(dataset_root, manifest_path)
    records = read_jsonl(manifest_path)
    validate_explicit_manifest_subset(dataset_root.resolve(), manifest_path, records)
    candidates: dict[str, dict[str, Any]] = {}
    tree = BKTree()
    exact_index: dict[str, list[str]] = defaultdict(list)
    seen_case_ids: set[str] = set()
    seen_canonical_paths: set[Path] = set()
    dataset_root = dataset_root.resolve()

    def register(candidate: dict[str, Any]) -> None:
        key = str(candidate["key"])
        if key in candidates:
            raise ValueError(f"Duplicate leakage candidate key: {key}")
        candidate_path = candidate.get("path")
        if candidate_path is not None:
            resolved_path = Path(str(candidate_path)).resolve()
            if resolved_path in seen_canonical_paths:
                raise ValueError(f"Duplicate leakage candidate path: {resolved_path}")
            seen_canonical_paths.add(resolved_path)
        fingerprints = {
            field: candidate[field]
            for field in ("pixel_sha256", "phash", "dhash", "width", "height")
        }
        candidates[key] = candidate
        tree.add(int(fingerprints["phash"], 16), key)
        exact_index[str(fingerprints["pixel_sha256"])].append(key)

    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("Benchmark manifest record is missing case_id")
        validate_safe_component(case_id, "Benchmark case_id")
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate benchmark case_id: {case_id}")
        seen_case_ids.add(case_id)
        for required in (
            "major",
            "minor",
            "source_dataset",
            "protocol",
            "src_image",
            "tar_image",
        ):
            if required not in record or record[required] in (None, ""):
                raise ValueError(f"{case_id}: benchmark record is missing {required}")
        protocol = str(record["protocol"])
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"{case_id}: unsupported benchmark protocol: {protocol!r}")
        validate_safe_component(str(record["major"]), f"{case_id}: major")
        for role in ("src_image", "tar_image"):
            key = f"{case_id}:{role}"
            relative_path = Path(str(record[role]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"{case_id}: unsafe canonical {role} path: {relative_path}")
            path = (dataset_root / relative_path).resolve()
            try:
                path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"{case_id}: canonical {role} resolves outside dataset root: {path}"
                ) from error
            expected_path = Path(str(record["major"])) / role / f"{case_id}.png"
            if relative_path != expected_path:
                raise ValueError(
                    f"{case_id}: non-canonical {role} path: "
                    f"actual={relative_path} expected={expected_path}"
                )
            if not path.is_file():
                raise ValueError(f"{case_id}: missing canonical {role}: {path}")
            fingerprints = image_fingerprints(path)
            candidate = {
                "key": key,
                "case_id": record["case_id"],
                "role": role,
                "path": str(path),
                "major": record["major"],
                "minor": record["minor"],
                "source_dataset": (
                    record.get("src_source_dataset", record["source_dataset"])
                    if role == "src_image"
                    else record["source_dataset"]
                ),
                "source_image_id": (
                    record.get(
                        "src_image_id",
                        f"{record.get('original_group', 'unknown')}/"
                        f"{record.get('original_stem', record['case_id'])}:src",
                    )
                    if role == "src_image"
                    else record.get(
                        "tar_image_id",
                        f"{record.get('original_group', 'unknown')}/"
                        f"{record.get('original_stem', record['case_id'])}:tar",
                    )
                ),
                **fingerprints,
            }
            register(candidate)

        present_full_source_fields = [
            field
            for field in FULL_SOURCE_FINGERPRINT_FIELDS
            if field in record and record[field] is not None and record[field] != ""
        ]
        should_register_full_source = protocol == "cross_instance_replace" or (
            protocol == "paired_reconstruction" and bool(present_full_source_fields)
        )
        if should_register_full_source:
            fingerprints = full_source_fingerprints(record)
            register(
                {
                    "key": f"{record['case_id']}:src_full_image",
                    "case_id": record["case_id"],
                    "role": "src_full_image",
                    # The full source is represented by provenance fingerprints
                    # in the manifest; only the canonical crop is distributed.
                    "path": None,
                    "fingerprint_origin": "manifest_full_source_image",
                    "major": record["major"],
                    "minor": record["minor"],
                    "source_dataset": record.get(
                        "src_source_dataset", record["source_dataset"]
                    ),
                    "source_image_id": record.get(
                        "src_image_id",
                        f"{record.get('original_group', 'unknown')}/"
                        f"{record.get('original_stem', record['case_id'])}:src",
                    ),
                    **fingerprints,
                }
            )
    return candidates, tree, exact_index


def _images_by_stem(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate training image stem in {directory}: {path.stem}")
        result[path.stem] = path.resolve()
    return result


def training_image_inventory(train_root: Path) -> tuple[list[Path], dict[str, Any]]:
    train_root = train_root.resolve()
    if not train_root.is_dir():
        raise ValueError(f"Training root is not a directory: {train_root}")
    resolved_root = train_root / "mask_prompt"
    if not resolved_root.is_dir():
        resolved_root = train_root
    required_names = ("ref_image", "ref_mask", "tar_image", "tar_mask")
    paths: list[Path] = []
    category_rows: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for category in sorted(path for path in resolved_root.iterdir() if path.is_dir()):
        present = [name for name in required_names if (category / name).is_dir()]
        if not present:
            continue
        missing = sorted(set(required_names).difference(present))
        if missing:
            raise ValueError(
                f"Partial training category {category}: missing required directories {missing}"
            )
        by_role = {
            name: _images_by_stem(category / name)
            for name in required_names
        }
        ref_stems = set(by_role["ref_image"])
        missing_by_role = {
            name: sorted(ref_stems.difference(by_role[name]))[:10]
            for name in ("ref_mask", "tar_image", "tar_mask")
            if ref_stems.difference(by_role[name])
        }
        if missing_by_role:
            raise ValueError(
                f"Incomplete training samples in category {category.name}: {missing_by_role}"
            )
        for stem in sorted(ref_stems):
            for role in ("ref_image", "tar_image"):
                path = by_role[role][stem]
                if path in seen_paths:
                    raise ValueError(f"Training image path is reused: {path}")
                seen_paths.add(path)
                paths.append(path)
        category_rows.append({"category": category.name, "sample_count": len(ref_stems)})
    sample_count = sum(int(row["sample_count"]) for row in category_rows)
    if sample_count <= 0 or not paths:
        raise ValueError(
            f"No complete mask_prompt training samples found under {resolved_root}"
        )
    inventory_digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        inventory_digest.update(
            (
                f"{path.relative_to(resolved_root)}\0{stat.st_size}\0"
                f"{stat.st_mtime_ns}\0{stat.st_ctime_ns}\n"
            ).encode("utf-8")
        )
    return sorted(paths), {
        "input_root": str(train_root),
        "resolved_mask_prompt_root": str(resolved_root),
        "category_count": len(category_rows),
        "sample_count": sample_count,
        "path_state_sha256": inventory_digest.hexdigest(),
        "categories": category_rows,
    }


def training_image_paths(train_root: Path) -> list[Path]:
    return training_image_inventory(train_root)[0]


def load_cache(cache_path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    if not cache_path.is_file():
        return {}, 0
    result: dict[str, dict[str, Any]] = {}
    ignored = 0
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                ignored += 1
                continue
            if (
                not isinstance(row, dict)
                or
                row.get("cache_schema_version") != CACHE_SCHEMA_VERSION
                or row.get("fingerprint_algorithm") != FINGERPRINT_ALGORITHM
                or not isinstance(row.get("path"), str)
            ):
                ignored += 1
                continue
            if "error" not in row:
                valid_fingerprint = (
                    isinstance(row.get("file_size"), int)
                    and isinstance(row.get("mtime_ns"), int)
                    and isinstance(row.get("ctime_ns"), int)
                    and isinstance(row.get("width"), int)
                    and isinstance(row.get("height"), int)
                    and isinstance(row.get("pixel_sha256"), str)
                    and len(row["pixel_sha256"]) == 64
                    and isinstance(row.get("phash"), str)
                    and len(row["phash"]) == 16
                    and isinstance(row.get("dhash"), str)
                    and len(row["dhash"]) == 16
                )
                try:
                    if valid_fingerprint:
                        int(row["pixel_sha256"], 16)
                        int(row["phash"], 16)
                        int(row["dhash"], 16)
                except (TypeError, ValueError):
                    valid_fingerprint = False
                if not valid_fingerprint:
                    ignored += 1
                    continue
            result[row["path"]] = row
    return result, ignored


def _same_existing_file(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_file_path(path: Path, label: str) -> Path:
    """Validate an output path without creating its parent or following aliases."""

    lexical = _absolute_lexical(path)
    _assert_no_symlink_components(lexical, label, include_leaf=True)
    if lexical.exists():
        mode = os.lstat(lexical).st_mode
        if not stat.S_ISREG(mode):
            raise ValueError(f"{label} must be a regular file path: {lexical}")
    parent = lexical.parent
    _assert_no_symlink_components(parent, f"{label} parent", include_leaf=True)
    existing_parent = parent
    while not existing_parent.exists():
        if existing_parent == existing_parent.parent:
            break
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise ValueError(
            f"{label} has a non-directory existing parent: {existing_parent}"
        )
    return lexical.resolve(strict=False)


def validate_scan_output_paths(
    *,
    cache_path: Path,
    report_path: Path,
    dataset_root: Path,
    train_root: Path,
    training_paths: Sequence[Path],
    release_integrity: dict[str, Any],
) -> tuple[Path, Path]:
    """Reject output aliases/collisions before the first scanner write."""

    resolved_cache = _validate_output_file_path(cache_path, "Leakage cache")
    resolved_report = _validate_output_file_path(report_path, "Leakage report")
    if (
        resolved_cache == resolved_report
        or _same_existing_file(resolved_cache, resolved_report)
        or _is_relative_to(resolved_cache, resolved_report)
        or _is_relative_to(resolved_report, resolved_cache)
    ):
        raise ValueError("Leakage cache and report paths must be distinct and non-overlapping")

    protected_paths = {
        Path(path).resolve() for path in release_integrity["protected_paths"]
    }
    protected_paths.update(Path(path).resolve() for path in training_paths)
    protected_labels = {
        Path(release_integrity["checksum_manifest"]).resolve(): "CHECKSUMS.sha256",
        Path(release_integrity["primary_manifest"]).resolve(): "primary manifest",
        Path(release_integrity["selected_manifest"]).resolve(): "selected manifest",
        Path(release_integrity["provenance_file"]).resolve(): "release provenance",
    }
    for output, output_label in (
        (resolved_cache, "Leakage cache"),
        (resolved_report, "Leakage report"),
    ):
        for protected in protected_paths:
            if output == protected or _same_existing_file(output, protected):
                protected_label = protected_labels.get(protected, "release/training input")
                raise ValueError(
                    f"{output_label} must not alias {protected_label}: {output}"
                )

        resolved_train_root = train_root.resolve()
        if _is_relative_to(output, resolved_train_root):
            raise ValueError(
                f"{output_label} must not be written inside the training input root: {output}"
            )
        for canonical_directory in release_integrity["canonical_directories"]:
            if _is_relative_to(output, Path(canonical_directory).resolve()):
                raise ValueError(
                    f"{output_label} must not be written in a canonical asset directory: "
                    f"{output}"
                )
        manifest_directory = dataset_root.resolve() / "manifests"
        if _is_relative_to(output, manifest_directory):
            raise ValueError(
                f"{output_label} must not be written in the authoritative manifest directory: "
                f"{output}"
            )
    return resolved_cache, resolved_report


def scan(
    dataset_root: Path,
    train_root: Path,
    cache_path: Path,
    report_path: Path,
    phash_radius: int,
    workers: int = 1,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    dataset_root = dataset_root.resolve()
    resolved_manifest = resolve_manifest_path(dataset_root, manifest_path)
    release_integrity = verify_release_integrity(dataset_root, resolved_manifest)
    primary_manifest = validate_explicit_manifest_subset(
        dataset_root,
        resolved_manifest,
        read_jsonl(resolved_manifest),
    )
    candidates, tree, exact_index = load_candidate_index(dataset_root, resolved_manifest)
    training_paths, training_inventory = training_image_inventory(train_root)
    resolved_cache, resolved_report = validate_scan_output_paths(
        cache_path=cache_path,
        report_path=report_path,
        dataset_root=dataset_root,
        train_root=train_root,
        training_paths=training_paths,
        release_integrity=release_integrity,
    )
    cache_path = resolved_cache
    report_path = resolved_report
    cached, ignored_cache_rows = load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    exact_matches: list[dict[str, Any]] = []
    near_matches: list[dict[str, Any]] = []
    decode_errors: list[dict[str, str]] = []

    processed = 0

    def match_row(row: dict[str, Any]) -> None:
        nonlocal processed
        processed += 1
        path_key = str(row["path"])
        if "error" in row:
            decode_errors.append({"path": path_key, "error": str(row["error"])})
        else:
            for candidate_key in exact_index.get(row["pixel_sha256"], []):
                exact_matches.append(
                    {
                        "candidate": candidates[candidate_key],
                        "training_path": path_key,
                        "pixel_sha256": row["pixel_sha256"],
                    }
                )
            training_phash = int(row["phash"], 16)
            for distance, candidate_key in tree.query(training_phash, phash_radius):
                candidate = candidates[candidate_key]
                dhash_distance = hamming(int(row["dhash"], 16), int(candidate["dhash"], 16))
                near_matches.append(
                    {
                        "candidate": candidate,
                        "training_path": path_key,
                        "phash_distance": distance,
                        "dhash_distance": dhash_distance,
                    }
                )
        if processed % 5000 == 0:
            print(
                f"scanned={processed}/{len(training_paths)} exact={len(exact_matches)} near={len(near_matches)}",
                flush=True,
            )

    missing_paths: list[str] = []
    for path in training_paths:
        path_key = str(path.resolve())
        row = cached.get(path_key)
        stat = path.stat()
        if (
            row is None
            or row.get("file_size") != stat.st_size
            or row.get("mtime_ns") != stat.st_mtime_ns
            or row.get("ctime_ns") != stat.st_ctime_ns
        ):
            missing_paths.append(path_key)
        else:
            match_row(row)

    mode = "a" if cache_path.is_file() else "w"
    with cache_path.open(mode, encoding="utf-8") as cache_handle:
        if workers == 1:
            generated_rows: Iterable[dict[str, Any]] = map(fingerprint_cache_row, missing_paths)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            generated_rows = executor.map(fingerprint_cache_row, missing_paths, chunksize=16)
        try:
            for generated_index, row in enumerate(generated_rows, start=1):
                cache_handle.write(json.dumps(row, sort_keys=True) + "\n")
                if generated_index % 100 == 0:
                    cache_handle.flush()
                match_row(row)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    exact_unique = {
        (row["candidate"]["key"], row["training_path"]): row for row in exact_matches
    }
    near_unique = {
        (row["candidate"]["key"], row["training_path"]): row for row in near_matches
    }
    exact_case_ids = sorted(
        {row["candidate"]["case_id"] for row in exact_unique.values()}
    )
    near_case_ids = sorted(
        {row["candidate"]["case_id"] for row in near_unique.values()}
    )
    all_case_ids = sorted({row["case_id"] for row in candidates.values()})
    report = {
        "dataset_root": str(dataset_root),
        "manifest": str(resolved_manifest.resolve()),
        "manifest_sha256": file_sha256(resolved_manifest),
        "primary_manifest": str(primary_manifest),
        "primary_manifest_sha256": file_sha256(primary_manifest),
        "release_type": release_integrity["release_type"],
        "checksum_manifest": str(release_integrity["checksum_manifest"]),
        "checksum_manifest_sha256": release_integrity["checksum_manifest_sha256"],
        "checksum_entry_count": release_integrity["checksum_entry_count"],
        "release_provenance": str(release_integrity["provenance_file"]),
        "release_provenance_sha256": release_integrity["provenance_sha256"],
        "release_integrity_verified": True,
        "manifest_case_inventory_sha256": hashlib.sha256(
            "\n".join(sorted(candidates)).encode("utf-8")
        ).hexdigest(),
        "train_root": str(train_root.resolve()),
        "candidate_image_count": len(candidates),
        "training_image_count": len(training_paths),
        "training_inventory": training_inventory,
        "training_cache_hits": len(training_paths) - len(missing_paths),
        "training_fingerprints_computed": len(missing_paths),
        "ignored_invalid_cache_rows": ignored_cache_rows,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "workers": workers,
        "phash_radius": phash_radius,
        "exact_match_count": len(exact_unique),
        "near_match_count": len(near_unique),
        "exact_case_count": len(exact_case_ids),
        "exact_case_ids": exact_case_ids,
        "near_case_count": len(near_case_ids),
        "near_case_ids": near_case_ids,
        "no_exact_match_case_ids": sorted(set(all_case_ids) - set(exact_case_ids)),
        "no_exact_or_near_match_case_ids": sorted(
            set(all_case_ids) - set(exact_case_ids).union(near_case_ids)
        ),
        "exact_matches": sorted(exact_unique.values(), key=lambda row: (row["candidate"]["key"], row["training_path"])),
        "near_matches": sorted(
            near_unique.values(),
            key=lambda row: (row["phash_distance"], row["dhash_distance"], row["candidate"]["key"]),
        ),
        "decode_errors": decode_errors,
        "leakage_free_exact": not exact_unique,
        "near_matches_require_review": bool(near_unique),
        "scan_complete": not decode_errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
        os.chmod(report_path, 0o644)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path (absolute or relative to dataset root)",
    )
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--phash-radius", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 <= args.phash_radius <= 16:
        raise ValueError("--phash-radius must be in [0, 16]")
    if not 1 <= args.workers <= 64:
        raise ValueError("--workers must be in [1, 64]")
    cache = args.cache or args.dataset_root.parent / "_cache" / "training_image_fingerprints.jsonl"
    report = args.report or args.dataset_root / "reports" / "training_leakage.json"
    result = scan(
        args.dataset_root,
        args.train_root,
        cache,
        report,
        args.phash_radius,
        workers=args.workers,
        manifest_path=args.manifest,
    )
    print(
        json.dumps(
            {
                "exact_match_count": result["exact_match_count"],
                "near_match_count": result["near_match_count"],
                "decode_errors": len(result["decode_errors"]),
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    if result["decode_errors"]:
        # A corrupted/unreadable training image means the scan is incomplete;
        # never let automation report that run as leakage-free success.
        return 2
    return 1 if result["exact_match_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

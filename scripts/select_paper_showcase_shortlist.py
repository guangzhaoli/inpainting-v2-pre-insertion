#!/usr/bin/env python3
"""Build a visually reviewed, release-bound PaperInsert showcase shortlist.

``audit`` performs the same read-only strict release checks used by both stages.
``seed`` selects six candidates per minor category and renders four-view review
sheets.  After an agent marks every decision accepted or rejected, ``finalize``
requires exactly three accepted cases per category and emits derived JSON/CSV.

The primary manifest and release files are never modified, and this tool never
sets ``paper_ok``.  Final artifacts remain pending human content/rights review.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import hashlib
import io
import json
import math
import os
import signal
import tempfile
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageOps

try:
    import prepare_identity_paired_insertion_dataset as builder
except ModuleNotFoundError:  # Imported as ``scripts.*`` from the repository root.
    from scripts import prepare_identity_paired_insertion_dataset as builder


SCHEMA_VERSION = 2
CANDIDATES_PER_MINOR = 6
ACCEPTED_PER_MINOR = 3
DECISION_ARTIFACT_TYPE = "paper_showcase_agent_decisions"
REVIEW_ARTIFACT_TYPE = "paper_showcase_four_view_review"
SHORTLIST_ARTIFACT_TYPE = "paper_showcase_shortlist"
FINAL_STATUS = "agent_shortlisted_pending_human_content_and_rights_review"
NATIVE_MASK_ORIGIN = "native_instance_annotation"
DECISION_VALUES = {"pending", "accepted", "rejected"}
CONTROL_FILENAMES = (
    "manifests/paperinsert_900.jsonl",
    "CHECKSUMS.sha256",
    "reports/audit.json",
)

TAXONOMY: tuple[tuple[str, str], ...] = tuple(
    (major, minor)
    for major, minors in builder.MAJOR_MINOR_QUOTAS.items()
    for minor in minors
)

CANDIDATE_BINDING_FIELDS = (
    "case_id",
    "major",
    "minor",
    "source_dataset",
    "leaf_label",
    "identity_id",
    "src_image_id",
    "tar_image_id",
    "src_image",
    "src_mask",
    "tar_image",
    "tar_mask",
    "src_pixel_sha256",
    "src_mask_pixel_sha256",
    "tar_pixel_sha256",
    "tar_mask_pixel_sha256",
    "candidate_rank",
    "selection_score",
)

SHORTLIST_CASE_FIELDS = (
    "case_id",
    "major",
    "minor",
    "leaf_label",
    "identity_id",
    "identity_scope",
    "pair_kind",
    "source_dataset",
    "src_image_id",
    "tar_image_id",
    "src_image",
    "src_mask",
    "tar_image",
    "tar_mask",
    "difficulty",
    "selection_score",
    "mask_origin_src",
    "mask_origin_tar",
    "native_category_id",
)


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    data: bytes
    sha256: str


@dataclass(frozen=True)
class ReleaseSnapshot:
    root: Path
    manifest: FileSnapshot
    checksums: FileSnapshot
    raw_audit: FileSnapshot
    records: tuple[Mapping[str, Any], ...]
    raw_audit_report: Mapping[str, Any]
    strict_audit_report: Mapping[str, Any]


@dataclass(frozen=True)
class RendererSnapshot:
    script: FileSnapshot
    builder_script: FileSnapshot
    sha256: str


@dataclass(frozen=True)
class ReviewBundle:
    root: Path
    index: FileSnapshot
    payload: Mapping[str, Any]
    sheets: tuple[FileSnapshot, ...]
    annotations: Mapping[str, Mapping[str, Any]]


def _snapshot_file(path: Path, label: str) -> FileSnapshot:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    data = path.read_bytes()
    return FileSnapshot(path=path.resolve(), data=data, sha256=hashlib.sha256(data).hexdigest())


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is forbidden: {value}")


def _parse_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Cannot parse {label}: {error}") from error


def _parse_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"Cannot decode {label} as UTF-8: {error}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = _parse_json_bytes(line.encode("utf-8"), f"{label} line {line_number}")
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        records.append(parsed)
    return records


def _assert_snapshots_unchanged(snapshots: Iterable[FileSnapshot]) -> None:
    for snapshot in snapshots:
        current = _snapshot_file(snapshot.path, "snapshotted input")
        if current.sha256 != snapshot.sha256:
            raise RuntimeError(
                f"Input changed during showcase operation: {snapshot.path} "
                f"start={snapshot.sha256} end={current.sha256}"
            )


def _resolve_release(path: Path) -> tuple[Path, Path]:
    if path.is_symlink():
        raise ValueError(f"Release input may not be a symlink: {path}")
    if path.is_dir():
        root = path.resolve()
        manifest = root / "manifests" / "paperinsert_900.jsonl"
    elif path.is_file():
        if path.name != "paperinsert_900.jsonl" or path.parent.name != "manifests":
            raise ValueError(
                "Manifest input must be <dataset-root>/manifests/paperinsert_900.jsonl"
            )
        manifest = path.resolve()
        root = manifest.parent.parent
    else:
        raise FileNotFoundError(f"Release root or manifest does not exist: {path}")
    return root, manifest


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_manifest_records(records: Sequence[Mapping[str, Any]]) -> None:
    expected_counts = {
        (major, minor): quota
        for major, minors in builder.MAJOR_MINOR_QUOTAS.items()
        for minor, quota in minors.items()
    }
    observed = Counter((row.get("major"), row.get("minor")) for row in records)
    if observed != Counter(expected_counts):
        wrong = {
            f"{major}/{minor}": {"expected": quota, "observed": observed.get((major, minor), 0)}
            for (major, minor), quota in expected_counts.items()
            if observed.get((major, minor), 0) != quota
        }
        unexpected = {
            f"{major}/{minor}": count
            for (major, minor), count in observed.items()
            if (major, minor) not in expected_counts
        }
        raise ValueError(
            "Manifest is not the balanced 18-minor PaperInsert-900 manifest: "
            f"wrong_counts={wrong} unexpected={unexpected} total={len(records)}"
        )

    required_ids = ("case_id", "identity_id", "src_image_id", "tar_image_id")
    for index, record in enumerate(records):
        for field in required_ids:
            if not _nonempty_string(record.get(field)):
                raise ValueError(f"Manifest row {index} has an empty/non-string {field}")
    case_ids = [str(row["case_id"]) for row in records]
    duplicates = sorted(case_id for case_id, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Manifest contains duplicate case_ids: {duplicates[:10]}")


def _release_evidence(release: ReleaseSnapshot) -> dict[str, Any]:
    return {
        "dataset_root": str(release.root),
        "manifest_path": str(release.manifest.path),
        "manifest_sha256": release.manifest.sha256,
        "checksum_manifest_path": str(release.checksums.path),
        "checksum_manifest_sha256": release.checksums.sha256,
        "raw_backed_audit_path": str(release.raw_audit.path),
        "raw_backed_audit_sha256": release.raw_audit.sha256,
        "raw_backed_audit_manifest_sha256": release.raw_audit_report["manifest_sha256"],
        "strict_read_only_audit": True,
        "strict_audit_valid": True,
    }


def strict_release_snapshot(path: Path) -> ReleaseSnapshot:
    """Snapshot control bytes, then perform the builder's non-writing strict audit."""

    root, manifest_path = _resolve_release(path)
    manifest = _snapshot_file(manifest_path, "primary manifest")
    checksums = _snapshot_file(root / "CHECKSUMS.sha256", "checksum manifest")
    raw_audit = _snapshot_file(root / "reports" / "audit.json", "raw-backed audit")
    records = _parse_jsonl_bytes(manifest.data, "primary manifest")
    _validate_manifest_records(records)
    raw_report = _parse_json_bytes(raw_audit.data, "raw-backed audit")
    if not isinstance(raw_report, Mapping):
        raise ValueError("Raw-backed audit root must be an object")

    required_raw_evidence = {
        "schema_version": 1,
        "valid": True,
        "errors": [],
        "warnings": [],
        "record_count": len(records),
        "identity_count": len(records),
        "different_source_image_count": len(records),
        "target_derived_reference_count": 0,
        "raw_backed_audit": True,
        "manifest_sha256": manifest.sha256,
        "checksum_manifest_sha256": checksums.sha256,
    }
    for field, expected in required_raw_evidence.items():
        if _canonical_json_bytes(raw_report.get(field)) != _canonical_json_bytes(expected):
            raise ValueError(
                f"Existing raw-backed audit is not release-binding: {field}="
                f"{raw_report.get(field)!r}, expected={expected!r}"
            )

    strict_report = builder.audit_dataset(root, verify_raw=False, write_report=False)
    if not isinstance(strict_report, Mapping) or strict_report.get("valid") is not True:
        errors = strict_report.get("errors") if isinstance(strict_report, Mapping) else strict_report
        raise ValueError(f"Read-only strict dataset audit failed: {errors}")
    if strict_report.get("errors") != []:
        raise ValueError(f"Read-only strict dataset audit returned errors: {strict_report.get('errors')}")
    if strict_report.get("manifest_sha256") != manifest.sha256:
        raise ValueError("Strict audit manifest SHA does not match the manifest byte snapshot")
    if strict_report.get("checksum_manifest_sha256") != checksums.sha256:
        raise ValueError("Strict audit CHECKSUMS SHA does not match the checksum byte snapshot")
    _assert_snapshots_unchanged((manifest, checksums, raw_audit))
    return ReleaseSnapshot(
        root=root,
        manifest=manifest,
        checksums=checksums,
        raw_audit=raw_audit,
        records=tuple(records),
        raw_audit_report=raw_report,
        strict_audit_report=strict_report,
    )


def _assert_release_unchanged(release: ReleaseSnapshot) -> None:
    checksum_errors = builder.verify_checksums(release.root, release.records)
    if checksum_errors:
        raise RuntimeError(f"Release changed after strict audit: {checksum_errors[:10]}")
    _assert_snapshots_unchanged((release.manifest, release.checksums, release.raw_audit))


def _renderer_snapshot() -> RendererSnapshot:
    script = _snapshot_file(Path(__file__).resolve(), "showcase renderer script")
    builder_script = _snapshot_file(builder.LOADED_BUILDER_PATH, "builder renderer dependency")
    combined = hashlib.sha256(
        f"{script.sha256}\n{builder_script.sha256}\n".encode("ascii")
    ).hexdigest()
    return RendererSnapshot(script=script, builder_script=builder_script, sha256=combined)


def _renderer_evidence(renderer: RendererSnapshot) -> dict[str, Any]:
    return {
        "renderer_sha256": renderer.sha256,
        "renderer_script_path": str(renderer.script.path),
        "renderer_script_sha256": renderer.script.sha256,
        "builder_renderer_path": str(renderer.builder_script.path),
        "builder_renderer_sha256": renderer.builder_script.sha256,
        "layout": "four_view_ref_outline_ref_mask_masked_target_target_gt_v1",
    }


def _assert_renderer_unchanged(renderer: RendererSnapshot) -> None:
    _assert_snapshots_unchanged((renderer.script, renderer.builder_script))


def _eligible(record: Mapping[str, Any]) -> bool:
    """Return whether a row meets the conservative automatic seed gate."""

    identity_id = record.get("identity_id")
    src_image_id = record.get("src_image_id")
    tar_image_id = record.get("tar_image_id")
    canonical_paths_ok = all(
        _nonempty_string(record.get(field))
        for field in ("src_image", "src_mask", "tar_image", "tar_mask")
    )
    pixel_signatures_ok = all(
        _nonempty_string(record.get(field))
        for field in (
            "src_pixel_sha256",
            "src_mask_pixel_sha256",
            "tar_pixel_sha256",
            "tar_mask_pixel_sha256",
        )
    )
    native_category_valid = True
    if record.get("major") == "garment":
        native_category_id = record.get("native_category_id")
        native_category_valid = (
            type(native_category_id) is int
            and builder.GARMENT_CATEGORY_TO_MINOR.get(native_category_id) == record.get("minor")
            and builder.GARMENT_CATEGORY_NAMES.get(native_category_id) == record.get("leaf_label")
        )
    return (
        record.get("difficulty") == "easy"
        and record.get("same_identity") is True
        and record.get("gt_available") is True
        and record.get("case_type") == "paired_gt"
        and record.get("reference_derived_from_target") is False
        and record.get("mask_origin_src") == NATIVE_MASK_ORIGIN
        and record.get("mask_origin_tar") == NATIVE_MASK_ORIGIN
        and record.get("auto_quality_pass") is True
        and record.get("agent_paper_avoid") is not True
        and _nonempty_string(record.get("source_dataset"))
        and _nonempty_string(identity_id)
        and _nonempty_string(src_image_id)
        and _nonempty_string(tar_image_id)
        and src_image_id != tar_image_id
        and canonical_paths_ok
        and pixel_signatures_ok
        and native_category_valid
    )


def _finite_score(record: Mapping[str, Any]) -> float:
    try:
        score = float(record.get("selection_score", -1e300))
    except (TypeError, ValueError):
        return -1e300
    return score if math.isfinite(score) else -1e300


def _rank_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("showcase_candidate") is not True,
        -_finite_score(record),
        str(record["case_id"]),
    )


def _ordered_group_candidates(
    major: str,
    minor: str,
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    if major != "garment":
        yield from sorted(rows, key=_rank_key)
        return

    native_categories = tuple(
        sorted(
            builder.GARMENT_NATIVE_SUBQUOTAS[minor],
            key=lambda category_id: (
                -builder.GARMENT_NATIVE_SUBQUOTAS[minor][category_id],
                category_id,
            ),
        )
    )
    queues: dict[int, deque[Mapping[str, Any]]] = {}
    for category_id in native_categories:
        queues[category_id] = deque(
            sorted(
                (row for row in rows if row.get("native_category_id") == category_id),
                key=_rank_key,
            )
        )
    missing = [category_id for category_id, queue in queues.items() if not queue]
    if missing:
        raise ValueError(
            f"Garment seed cannot cover every native category for {minor}; missing={missing}"
        )
    while any(queues.values()):
        for category_id in native_categories:
            if queues[category_id]:
                yield queues[category_id].popleft()


def _candidate_binding(row: Mapping[str, Any]) -> dict[str, Any]:
    binding = {field: row[field] for field in CANDIDATE_BINDING_FIELDS}
    if row.get("major") == "garment":
        binding["native_category_id"] = row["native_category_id"]
    return binding


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _candidate_set_sha256(candidates: Sequence[Mapping[str, Any]]) -> str:
    bindings = [_candidate_binding(row) for row in candidates]
    return hashlib.sha256(_canonical_json_bytes(bindings)).hexdigest()


def select_seed_candidates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if _eligible(record):
            grouped[(str(record["major"]), str(record["minor"]))].append(record)

    selected: list[dict[str, Any]] = []
    used_identities: set[str] = set()
    used_images: set[tuple[str, str]] = set()
    for major, minor in TAXONOMY:
        accepted_for_group: list[Mapping[str, Any]] = []
        for record in _ordered_group_candidates(major, minor, grouped[(major, minor)]):
            identity_id = str(record["identity_id"])
            source_dataset = str(record["source_dataset"])
            image_ids = {
                (source_dataset, str(record["src_image_id"])),
                (source_dataset, str(record["tar_image_id"])),
            }
            if identity_id in used_identities or image_ids & used_images:
                continue
            accepted_for_group.append(record)
            used_identities.add(identity_id)
            used_images.update(image_ids)
            if len(accepted_for_group) == CANDIDATES_PER_MINOR:
                break
        if len(accepted_for_group) != CANDIDATES_PER_MINOR:
            raise ValueError(
                f"Insufficient unique eligible seed candidates for {major}/{minor}: "
                f"needed={CANDIDATES_PER_MINOR} found={len(accepted_for_group)}"
            )
        if major == "garment":
            expected_native = set(builder.GARMENT_NATIVE_SUBQUOTAS[minor])
            selected_native = {int(row["native_category_id"]) for row in accepted_for_group}
            if not expected_native.issubset(selected_native):
                raise ValueError(
                    f"Garment round-robin failed native diversity for {minor}: "
                    f"selected={sorted(selected_native)} expected={sorted(expected_native)}"
                )
        for rank, record in enumerate(accepted_for_group, start=1):
            candidate = {
                "case_id": str(record["case_id"]),
                "major": major,
                "minor": minor,
                "source_dataset": str(record["source_dataset"]),
                "leaf_label": str(record["leaf_label"]),
                "identity_id": str(record["identity_id"]),
                "src_image_id": str(record["src_image_id"]),
                "tar_image_id": str(record["tar_image_id"]),
                "src_image": str(record["src_image"]),
                "src_mask": str(record["src_mask"]),
                "tar_image": str(record["tar_image"]),
                "tar_mask": str(record["tar_mask"]),
                "src_pixel_sha256": str(record["src_pixel_sha256"]),
                "src_mask_pixel_sha256": str(record["src_mask_pixel_sha256"]),
                "tar_pixel_sha256": str(record["tar_pixel_sha256"]),
                "tar_mask_pixel_sha256": str(record["tar_mask_pixel_sha256"]),
                "candidate_rank": rank,
                "selection_score": _finite_score(record),
                "decision": "pending",
                "agent_note": "",
            }
            if major == "garment":
                candidate["native_category_id"] = int(record["native_category_id"])
            selected.append(candidate)
    return selected


def _save_jpeg_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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


def render_review_tile(
    dataset_root: Path,
    record: Mapping[str, Any],
    *,
    tile_size: tuple[int, int] = (816, 248),
) -> Image.Image:
    """Render all four canonical inputs with the insertion context visible."""

    with Image.open(dataset_root / str(record["src_image"])) as handle:
        src_image = handle.convert("RGB")
    with Image.open(dataset_root / str(record["src_mask"])) as handle:
        src_mask = handle.convert("L")
    with Image.open(dataset_root / str(record["tar_image"])) as handle:
        tar_image = handle.convert("RGB")
    with Image.open(dataset_root / str(record["tar_mask"])) as handle:
        tar_mask = handle.convert("L")
    panel_size = (190, 190)
    src_mask_visual = ImageOps.colorize(src_mask, black=(24, 24, 24), white=(245, 245, 245))
    panels = (
        builder._fit_panel(builder._mask_outline(src_image, src_mask), panel_size),
        builder._fit_panel(src_mask_visual, panel_size),
        builder._fit_panel(builder._masked_target(tar_image, tar_mask), panel_size),
        builder._fit_panel(builder._mask_outline(tar_image, tar_mask), panel_size),
    )
    tile = Image.new("RGB", tile_size, (248, 248, 248))
    for index, panel in enumerate(panels):
        tile.paste(panel, (8 + index * 202, 46))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 4), str(record["case_id"]), font=builder._font(16), fill=(15, 15, 15))
    native = (
        f" | native={record['native_category_id']}"
        if record.get("major") == "garment"
        else ""
    )
    subtitle = (
        f"{record['source_dataset']} | {record['leaf_label']} | {record['difficulty']}"
        f"{native} | {record['identity_id']}"
    )
    draw.text((8, 24), subtitle[:124], font=builder._font(11), fill=(55, 55, 55))
    labels = ("REF + OUTLINE", "REF NATIVE MASK", "MASKED INPUT", "TARGET GT + OUTLINE")
    for index, label in enumerate(labels):
        draw.text((10 + index * 202, 226), label, font=builder._font(11), fill=(35, 35, 35))
    return tile


def _require_fresh_output(path: Path, protected: Sequence[Path]) -> None:
    resolved = path.resolve()
    if any(resolved == item.resolve() for item in protected):
        raise ValueError(f"Refusing to overwrite an input artifact: {path}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a staged directory without replacing a racer."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise OSError(
            errno.ENOSYS,
            "libc does not expose renameat2",
            destination,
        ) from error
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _mkdir_owned(
    path: Path,
    relative: Path,
    identities: list[tuple[Path, bool, int, int]],
) -> None:
    """Create one directory and record ownership before SIGINT is unblocked."""

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        path.mkdir(mode=0o755)
        stat_result = path.lstat()
        identities.append((relative, True, stat_result.st_dev, stat_result.st_ino))
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _copy_file_atomic_noreplace(
    source: Path,
    destination: Path,
    destination_root: Path,
    identities: list[tuple[Path, bool, int, int]],
) -> None:
    """Copy through an owned temp inode, then commit with no-replace link(2)."""

    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.publish-tmp"
    temporary_relative = temporary.relative_to(destination_root)
    destination_relative = destination.relative_to(destination_root)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    file_descriptor: int | None = None
    temporary_identity: tuple[int, int] | None = None
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        file_descriptor = os.open(temporary, flags, 0o644)
        stat_result = os.fstat(file_descriptor)
        temporary_identity = (stat_result.st_dev, stat_result.st_ino)
        identities.append(
            (temporary_relative, False, stat_result.st_dev, stat_result.st_ino)
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    if file_descriptor is None:
        raise AssertionError("Exclusive review temp creation returned no file descriptor")

    try:
        with source.open("rb") as source_handle, os.fdopen(file_descriptor, "wb") as output_handle:
            file_descriptor = None
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        identities.append(
            (destination_relative, False, stat_result.st_dev, stat_result.st_ino)
        )
        os.link(temporary, destination, follow_symlinks=False)
        _remove_owned_file(temporary, temporary_identity)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _publish_directory_noreplace(
    source: Path,
    destination: Path,
    identities: list[tuple[Path, bool, int, int]],
) -> str:
    """Publish a review tree using renameat2 or a portable index-last commit.

    Some mounted filesystems return ``EINVAL`` for ``RENAME_NOREPLACE``.  The
    fallback first claims the destination directory with mkdir(2), copies every
    sheet through exclusive temp inodes, and links ``index.json`` last.  Thus a
    consumer never sees a committed index until the complete sheet set exists.
    """

    source_identities = list(_snapshot_tree_identities(source))
    identities.extend(source_identities)
    try:
        _rename_noreplace(source, destination)
        return "renameat2_noreplace"
    except OSError as error:
        unsupported = {
            errno.EINVAL,
            errno.ENOSYS,
            errno.EXDEV,
            errno.EOPNOTSUPP,
        }
        if error.errno not in unsupported:
            raise
    identities.clear()

    _mkdir_owned(destination, Path("."), identities)
    source_directories = sorted(
        (path for path in source.rglob("*") if path.is_dir()),
        key=lambda path: (len(path.relative_to(source).parts), path.as_posix()),
    )
    for source_directory in source_directories:
        relative = source_directory.relative_to(source)
        _mkdir_owned(destination / relative, relative, identities)

    source_files = [path for path in source.rglob("*") if path.is_file()]
    source_files.sort(
        key=lambda path: (
            path.relative_to(source) == Path("index.json"),
            path.relative_to(source).as_posix(),
        )
    )
    if not source_files or source_files[-1].relative_to(source) != Path("index.json"):
        raise ValueError("Staged review bundle has no index.json commit point")
    for source_file in source_files:
        relative = source_file.relative_to(source)
        _copy_file_atomic_noreplace(
            source_file,
            destination / relative,
            destination,
            identities,
        )
    return "portable_index_last_noreplace"


def _atomic_create_text(
    path: Path,
    text: str,
    authorized_identities: dict[Path, tuple[int, int]],
) -> None:
    """Publish complete text with link(2), which has no overwrite mode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    temporary_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        stat_result = temporary.lstat()
        temporary_identity = (stat_result.st_dev, stat_result.st_ino)
        # Authorize cleanup before link(2), so even an asynchronous exception
        # immediately after a successful link cannot leak our output inode.
        authorized_identities[path] = (stat_result.st_dev, stat_result.st_ino)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary is not None:
            _remove_owned_file(temporary, temporary_identity)


def _remove_owned_file(path: Path, identity: tuple[int, int] | None) -> None:
    if identity is None:
        return
    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return
    if (stat_result.st_dev, stat_result.st_ino) == identity and not path.is_dir():
        path.unlink()


def _snapshot_tree_identities(root: Path) -> tuple[tuple[Path, bool, int, int], ...]:
    rows: list[tuple[Path, bool, int, int]] = []
    for path in (root, *sorted(root.rglob("*"))):
        if path.is_symlink():
            raise ValueError(f"Generated review bundle unexpectedly contains a symlink: {path}")
        stat_result = path.lstat()
        rows.append((path.relative_to(root), path.is_dir(), stat_result.st_dev, stat_result.st_ino))
    return tuple(rows)


def _remove_owned_tree(
    root: Path,
    identities: Sequence[tuple[Path, bool, int, int]] | None,
) -> None:
    """Remove only entries whose inode was created by this invocation."""

    if identities is None:
        return
    for relative, was_directory, device, inode in sorted(
        identities,
        key=lambda row: len(row[0].parts),
        reverse=True,
    ):
        path = root / relative
        try:
            stat_result = path.lstat()
        except FileNotFoundError:
            continue
        if path.is_symlink() or (stat_result.st_dev, stat_result.st_ino) != (device, inode):
            continue
        try:
            path.rmdir() if was_directory else path.unlink()
        except OSError as error:
            if not (was_directory and error.errno in {errno.ENOTEMPTY, errno.EEXIST}):
                raise


def _review_index_payload(
    release: ReleaseSnapshot,
    renderer: RendererSnapshot,
    candidates: Sequence[Mapping[str, Any]],
    review_root: Path,
    sheets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_sha = _candidate_set_sha256(candidates)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REVIEW_ARTIFACT_TYPE,
        "status": "agent_visual_decisions_pending_not_human_certified",
        **_release_evidence(release),
        **_renderer_evidence(renderer),
        "review_root": str(review_root.resolve()),
        "candidate_set_sha256": candidate_sha,
        "candidate_count": len(candidates),
        "sheet_count": len(sheets),
        "four_view_panels": [
            "independent_reference_with_native_mask_outline",
            "reference_native_mask",
            "target_masked_input",
            "target_ground_truth_with_native_mask_outline",
        ],
        "candidate_bindings": [_candidate_binding(row) for row in candidates],
        "sheets": list(sheets),
    }


def render_review_sheets(
    release: ReleaseSnapshot,
    renderer: RendererSnapshot,
    candidates: Sequence[Mapping[str, Any]],
    review_root: Path,
    *,
    published_review_root: Path | None = None,
) -> ReviewBundle:
    _require_fresh_output(review_root, [release.root, *(release.root / name for name in CONTROL_FILENAMES)])
    review_root.parent.mkdir(parents=True, exist_ok=True)
    records_by_case = {str(row["case_id"]): row for row in release.records}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(str(candidate["major"]), str(candidate["minor"]))].append(candidate)

    with tempfile.TemporaryDirectory(
        dir=review_root.parent,
        prefix=f".{review_root.name}.staging.",
    ) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        sheets: list[dict[str, Any]] = []
        for major, minor in TAXONOMY:
            subset = sorted(grouped[(major, minor)], key=lambda row: int(row["candidate_rank"]))
            tile_width, tile_height, columns = 816, 248, 2
            rows = math.ceil(len(subset) / columns)
            canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), (230, 230, 230))
            case_rows: list[dict[str, Any]] = []
            for position, candidate in enumerate(subset):
                record = records_by_case[str(candidate["case_id"])]
                tile = render_review_tile(release.root, record, tile_size=(tile_width, tile_height))
                column, row = position % columns, position // columns
                xy = (column * tile_width, row * tile_height)
                canvas.paste(tile, xy)
                case_rows.append(
                    {
                        "case_id": str(candidate["case_id"]),
                        "candidate_rank": int(candidate["candidate_rank"]),
                        "tile_index": position,
                        "tile_xywh": [xy[0], xy[1], tile_width, tile_height],
                        "canonical_paths": {
                            role: str(candidate[role])
                            for role in ("src_image", "src_mask", "tar_image", "tar_mask")
                        },
                    }
                )
            relative = Path("sheets") / major / minor / "sheet_00.jpg"
            sheet_path = staging / relative
            _save_jpeg_atomic(canvas, sheet_path)
            sheets.append(
                {
                    "sheet": relative.as_posix(),
                    "sheet_sha256": builder.file_sha256(sheet_path),
                    "major": major,
                    "minor": minor,
                    "case_count": len(case_rows),
                    "cases": case_rows,
                }
            )
        index_payload = _review_index_payload(
            release,
            renderer,
            candidates,
            published_review_root or review_root,
            sheets,
        )
        builder.atomic_write_text(
            staging / "index.json",
            json.dumps(index_payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )
        os.replace(staging, review_root)

    return _load_review_bundle(review_root / "index.json")


def _safe_relative_path(value: Any, label: str) -> Path:
    if not _nonempty_string(value):
        raise ValueError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return relative


def _load_review_bundle(index_path: Path) -> ReviewBundle:
    index = _snapshot_file(index_path, "review index")
    payload = _parse_json_bytes(index.data, "review index")
    if not isinstance(payload, Mapping):
        raise ValueError("Review index root must be an object")
    root = index.path.parent
    sheets_payload = payload.get("sheets")
    if not isinstance(sheets_payload, list):
        raise ValueError("Review index sheets must be a list")
    sheets: list[FileSnapshot] = []
    annotations: dict[str, Mapping[str, Any]] = {}
    for sheet_row in sheets_payload:
        if not isinstance(sheet_row, Mapping):
            raise ValueError("Every review sheet row must be an object")
        relative = _safe_relative_path(sheet_row.get("sheet"), "review sheet path")
        sheet = _snapshot_file(root / relative, "review sheet")
        if sheet.sha256 != sheet_row.get("sheet_sha256"):
            raise ValueError(f"Review sheet SHA mismatch: {relative}")
        sheets.append(sheet)
        cases = sheet_row.get("cases")
        if not isinstance(cases, list):
            raise ValueError(f"Review sheet cases must be a list: {relative}")
        for case_row in cases:
            if not isinstance(case_row, Mapping) or not _nonempty_string(case_row.get("case_id")):
                raise ValueError(f"Invalid review case annotation in {relative}")
            case_id = str(case_row["case_id"])
            if case_id in annotations:
                raise ValueError(f"Duplicate case in review index: {case_id}")
            annotations[case_id] = {
                "review_sheet": relative.as_posix(),
                "review_sheet_sha256": sheet.sha256,
                "review_tile_index": case_row.get("tile_index"),
                "review_tile_xywh": case_row.get("tile_xywh"),
            }
    return ReviewBundle(
        root=root,
        index=index,
        payload=payload,
        sheets=tuple(sheets),
        annotations=annotations,
    )


def _assert_review_unchanged(review: ReviewBundle) -> None:
    _assert_snapshots_unchanged((review.index, *review.sheets))


def _expected_review_structure(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[(str(candidate["major"]), str(candidate["minor"]))].append(candidate)
    result: list[dict[str, Any]] = []
    tile_width, tile_height, columns = 816, 248, 2
    for major, minor in TAXONOMY:
        subset = sorted(grouped[(major, minor)], key=lambda row: int(row["candidate_rank"]))
        case_rows = []
        for position, candidate in enumerate(subset):
            column, row = position % columns, position // columns
            case_rows.append(
                {
                    "case_id": str(candidate["case_id"]),
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "tile_index": position,
                    "tile_xywh": [column * tile_width, row * tile_height, tile_width, tile_height],
                    "canonical_paths": {
                        role: str(candidate[role])
                        for role in ("src_image", "src_mask", "tar_image", "tar_mask")
                    },
                }
            )
        result.append(
            {
                "sheet": (Path("sheets") / major / minor / "sheet_00.jpg").as_posix(),
                "major": major,
                "minor": minor,
                "case_count": CANDIDATES_PER_MINOR,
                "cases": case_rows,
            }
        )
    return result


def _validate_review_bundle(
    review: ReviewBundle,
    release: ReleaseSnapshot,
    renderer: RendererSnapshot,
    candidates: Sequence[Mapping[str, Any]],
    *,
    expected_review_root: Path | None = None,
) -> None:
    payload = review.payload
    required = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": REVIEW_ARTIFACT_TYPE,
        "status": "agent_visual_decisions_pending_not_human_certified",
        "manifest_sha256": release.manifest.sha256,
        "checksum_manifest_sha256": release.checksums.sha256,
        "raw_backed_audit_sha256": release.raw_audit.sha256,
        "renderer_sha256": renderer.sha256,
        "candidate_set_sha256": _candidate_set_sha256(candidates),
        "candidate_count": len(candidates),
        "sheet_count": len(TAXONOMY),
        "candidate_bindings": [_candidate_binding(row) for row in candidates],
        "review_root": str((expected_review_root or review.root).resolve()),
    }
    for field, expected in required.items():
        matches = (
            _canonical_json_bytes(payload.get(field)) == _canonical_json_bytes(expected)
            if field == "candidate_bindings"
            else payload.get(field) == expected
        )
        if not matches:
            raise ValueError(f"Review index binding mismatch for {field}")
    expected_cases = {str(row["case_id"]) for row in candidates}
    if set(review.annotations) != expected_cases:
        raise ValueError("Review index case set differs from the deterministic candidate set")
    actual_sheets = payload.get("sheets")
    expected_sheets = _expected_review_structure(candidates)
    if not isinstance(actual_sheets, list) or len(actual_sheets) != len(expected_sheets):
        raise ValueError("Review index must contain exactly one sheet for each of the 18 minors")
    if len(review.sheets) != len(expected_sheets):
        raise ValueError("Loaded review sheet count differs from the 18-minor layout")
    seen_sheet_paths: set[str] = set()
    for actual, expected, snapshot in zip(actual_sheets, expected_sheets, review.sheets):
        if not isinstance(actual, Mapping):
            raise ValueError("Review sheet metadata must be an object")
        sheet_path = actual.get("sheet")
        if sheet_path in seen_sheet_paths:
            raise ValueError(f"Review index reuses a sheet path: {sheet_path!r}")
        seen_sheet_paths.add(str(sheet_path))
        for field, expected_value in expected.items():
            if actual.get(field) != expected_value:
                raise ValueError(
                    f"Review sheet structure mismatch for {expected['major']}/{expected['minor']} field {field}"
                )
        if actual.get("sheet_sha256") != snapshot.sha256:
            raise ValueError(f"Review sheet snapshot binding mismatch: {sheet_path}")
        try:
            with Image.open(io.BytesIO(snapshot.data)) as image:
                if image.format != "JPEG" or image.size != (1632, 744):
                    raise ValueError(
                        f"Review sheet has wrong format/size: {sheet_path} "
                        f"format={image.format} size={image.size}"
                    )
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot validate review sheet pixels: {sheet_path}: {error}") from error


def _build_seed_payload(
    release: ReleaseSnapshot,
    renderer: RendererSnapshot,
    candidates: Sequence[Mapping[str, Any]],
    review: ReviewBundle,
    *,
    published_review_index: Path | None = None,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        case_id = str(candidate["case_id"])
        decisions.append({**candidate, **review.annotations[case_id]})
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "status": "agent_decisions_pending",
        **_release_evidence(release),
        **_renderer_evidence(renderer),
        "candidate_set_sha256": _candidate_set_sha256(candidates),
        "review_index_path": str((published_review_index or review.index.path).resolve()),
        "review_index_sha256": review.index.sha256,
        "candidates_per_minor": CANDIDATES_PER_MINOR,
        "required_accepted_per_minor": ACCEPTED_PER_MINOR,
        "candidate_count": len(candidates),
        "ranking_policy": (
            "eligible easy same-identity paired-GT non-target-derived native-mask auto-quality "
            "non-agent-avoid rows; builder showcase then score; garment native-category round-robin"
        ),
        "allowed_decisions": sorted(DECISION_VALUES),
        "taxonomy": {major: list(minors) for major, minors in builder.MAJOR_MINOR_QUOTAS.items()},
        "decisions": decisions,
    }


def seed_decision_file(
    release_path: Path,
    output_path: Path,
    review_root: Path,
) -> dict[str, Any]:
    release = strict_release_snapshot(release_path)
    renderer = _renderer_snapshot()
    protected = [release.manifest.path, release.checksums.path, release.raw_audit.path]
    _require_fresh_output(output_path, protected)
    _require_fresh_output(review_root, [*protected, output_path, release.root])
    candidates = select_seed_candidates(release.records)
    review_root.parent.mkdir(parents=True, exist_ok=True)
    review_identities: list[tuple[Path, bool, int, int]] = []
    created_files: dict[Path, tuple[int, int]] = {}
    with tempfile.TemporaryDirectory(
        dir=review_root.parent,
        prefix=f".{review_root.name}.publish-staging.",
    ) as temporary:
        staged_review_root = Path(temporary) / "review"
        review = render_review_sheets(
            release,
            renderer,
            candidates,
            staged_review_root,
            published_review_root=review_root,
        )
        _validate_review_bundle(
            review,
            release,
            renderer,
            candidates,
            expected_review_root=review_root,
        )
        payload = _build_seed_payload(
            release,
            renderer,
            candidates,
            review,
            published_review_index=review_root / "index.json",
        )
        decision_text = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        # This is the end-of-input check.  Nothing is published before it passes.
        _assert_release_unchanged(release)
        _assert_renderer_unchanged(renderer)
        _assert_review_unchanged(review)
        try:
            _publish_directory_noreplace(
                staged_review_root,
                review_root,
                review_identities,
            )
            published_review = _load_review_bundle(review_root / "index.json")
            _validate_review_bundle(
                published_review,
                release,
                renderer,
                candidates,
                expected_review_root=review_root,
            )
            # Recheck source controls after the possibly non-rename fallback,
            # before decisions make the review bundle discoverable as a pair.
            _assert_release_unchanged(release)
            _assert_renderer_unchanged(renderer)
            _assert_review_unchanged(published_review)
            _atomic_create_text(output_path, decision_text, created_files)
        except BaseException:
            _remove_owned_file(output_path, created_files.get(output_path))
            _remove_owned_tree(review_root, review_identities)
            raise
    return payload


def _snapshot_decisions(path: Path) -> tuple[FileSnapshot, Mapping[str, Any]]:
    snapshot = _snapshot_file(path, "agent decision file")
    payload = _parse_json_bytes(snapshot.data, "agent decision file")
    if not isinstance(payload, Mapping):
        raise ValueError("Decision file root must be an object")
    return snapshot, payload


def _validate_decisions(
    decision_payload: Mapping[str, Any],
    release: ReleaseSnapshot,
    renderer: RendererSnapshot,
    review: ReviewBundle,
) -> tuple[list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    candidates = select_seed_candidates(release.records)
    expected_by_case = {str(row["case_id"]): row for row in candidates}
    required = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": DECISION_ARTIFACT_TYPE,
        "manifest_sha256": release.manifest.sha256,
        "checksum_manifest_sha256": release.checksums.sha256,
        "raw_backed_audit_sha256": release.raw_audit.sha256,
        "raw_backed_audit_manifest_sha256": release.manifest.sha256,
        "renderer_sha256": renderer.sha256,
        "candidate_set_sha256": _candidate_set_sha256(candidates),
        "review_index_sha256": review.index.sha256,
        "candidates_per_minor": CANDIDATES_PER_MINOR,
        "required_accepted_per_minor": ACCEPTED_PER_MINOR,
        "candidate_count": len(candidates),
    }
    for field, expected in required.items():
        if decision_payload.get(field) != expected:
            raise ValueError(f"Decision release/review binding mismatch for {field}")
    _validate_review_bundle(review, release, renderer, candidates)

    raw_decisions = decision_payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Decision payload field 'decisions' must be a list")
    manifest_by_case = {str(record["case_id"]): record for record in release.records}
    decisions: list[Mapping[str, Any]] = []
    seen_cases: set[str] = set()
    for index, decision in enumerate(raw_decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"Decision {index} must be an object")
        case_id_value = decision.get("case_id")
        if not _nonempty_string(case_id_value):
            raise ValueError(f"Decision {index} has an empty/non-string case_id")
        case_id = str(case_id_value)
        if case_id in seen_cases:
            raise ValueError(f"Duplicate decision for case_id: {case_id}")
        seen_cases.add(case_id)
        expected = expected_by_case.get(case_id)
        if expected is None:
            raise ValueError(f"Decision contains a case outside the deterministic seed: {case_id}")
        if _canonical_json_bytes(_candidate_binding(decision)) != _canonical_json_bytes(
            _candidate_binding(expected)
        ):
            raise ValueError(f"Decision candidate binding mismatch for {case_id}")
        for field, expected_value in review.annotations[case_id].items():
            if decision.get(field) != expected_value:
                raise ValueError(f"Decision review binding mismatch for {case_id} field {field}")
        value = decision.get("decision")
        if value not in DECISION_VALUES:
            raise ValueError(f"Invalid decision for {case_id}: {value!r}")
        if not isinstance(decision.get("agent_note", ""), str):
            raise ValueError(f"Agent note must be a string for {case_id}")
        decisions.append(decision)

    if seen_cases != set(expected_by_case):
        missing = sorted(set(expected_by_case) - seen_cases)
        raise ValueError(f"Decision file does not cover the complete seed; missing={missing[:10]}")
    pending = sorted(str(row["case_id"]) for row in decisions if row["decision"] == "pending")
    if pending:
        raise ValueError(f"All agent decisions must be explicit before finalize; pending={pending[:10]}")
    accepted_counts = Counter(
        (str(row["major"]), str(row["minor"]))
        for row in decisions
        if row["decision"] == "accepted"
    )
    wrong = {
        f"{major}/{minor}": accepted_counts.get((major, minor), 0)
        for major, minor in TAXONOMY
        if accepted_counts.get((major, minor), 0) != ACCEPTED_PER_MINOR
    }
    if wrong:
        raise ValueError(f"Finalize requires exactly {ACCEPTED_PER_MINOR} accepted cases per minor: {wrong}")
    for minor, native_subquotas in builder.GARMENT_NATIVE_SUBQUOTAS.items():
        accepted_native = {
            int(row["native_category_id"])
            for row in decisions
            if row["decision"] == "accepted"
            and row["major"] == "garment"
            and row["minor"] == minor
        }
        required_distinct = min(ACCEPTED_PER_MINOR, len(native_subquotas))
        if len(accepted_native) != required_distinct:
            raise ValueError(
                f"Accepted garment shortlist must maximize native-category diversity for {minor}: "
                f"actual={sorted(accepted_native)} required_distinct={required_distinct}"
            )
    return decisions, manifest_by_case


def _shortlist_case(record: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: record.get(field) for field in SHORTLIST_CASE_FIELDS}
    result["candidate_rank"] = int(decision["candidate_rank"])
    result["agent_note"] = str(decision.get("agent_note", ""))
    for field in ("review_sheet", "review_sheet_sha256", "review_tile_index", "review_tile_xywh"):
        result[field] = decision[field]
    result["content_review_status"] = "pending_human_review"
    result["rights_review_status"] = "pending_human_review"
    return result


def _prepare_shortlist(
    release_path: Path,
    decision_path: Path,
    *,
    end_recheck: bool = True,
) -> tuple[dict[str, Any], ReleaseSnapshot, RendererSnapshot, FileSnapshot, ReviewBundle]:
    release = strict_release_snapshot(release_path)
    renderer = _renderer_snapshot()
    decision_snapshot, decision_payload = _snapshot_decisions(decision_path)
    review_path = decision_payload.get("review_index_path")
    if not _nonempty_string(review_path):
        raise ValueError("Decision file is missing review_index_path")
    review = _load_review_bundle(Path(review_path))
    if decision_payload.get("review_index_sha256") != review.index.sha256:
        raise ValueError("Decision review index SHA does not match the review byte snapshot")
    decisions, manifest_by_case = _validate_decisions(decision_payload, release, renderer, review)
    accepted = [row for row in decisions if row["decision"] == "accepted"]
    order = {key: index for index, key in enumerate(TAXONOMY)}
    accepted.sort(
        key=lambda row: (
            order[(str(row["major"]), str(row["minor"]))],
            int(row["candidate_rank"]),
            str(row["case_id"]),
        )
    )
    identities = [str(row["identity_id"]) for row in accepted]
    duplicate_identities = sorted(key for key, count in Counter(identities).items() if count > 1)
    if duplicate_identities:
        raise ValueError(f"Accepted shortlist contains duplicate identities: {duplicate_identities}")
    image_ids = [
        (str(row["source_dataset"]), str(row[field]))
        for row in accepted
        for field in ("src_image_id", "tar_image_id")
    ]
    duplicate_images = sorted(key for key, count in Counter(image_ids).items() if count > 1)
    if duplicate_images:
        raise ValueError(f"Accepted shortlist contains duplicate upstream images: {duplicate_images}")
    image_paths = [str(row[field]) for row in accepted for field in ("src_image", "tar_image")]
    duplicate_paths = sorted(key for key, count in Counter(image_paths).items() if count > 1)
    if duplicate_paths:
        raise ValueError(f"Accepted shortlist contains duplicate canonical image paths: {duplicate_paths}")

    cases = [_shortlist_case(manifest_by_case[str(row["case_id"])], row) for row in accepted]
    counts = Counter((str(row["major"]), str(row["minor"])) for row in cases)
    native_counts = Counter(
        int(row["native_category_id"])
        for row in cases
        if row["major"] == "garment"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SHORTLIST_ARTIFACT_TYPE,
        "status": FINAL_STATUS,
        **_release_evidence(release),
        **_renderer_evidence(renderer),
        "candidate_set_sha256": decision_payload["candidate_set_sha256"],
        "review_index_path": str(review.index.path),
        "review_index_sha256": review.index.sha256,
        "decision_path": str(decision_snapshot.path),
        "decision_sha256": decision_snapshot.sha256,
        "case_count": len(cases),
        "accepted_per_minor": ACCEPTED_PER_MINOR,
        "minor_count": len(TAXONOMY),
        "counts_by_minor": {
            major: {minor: counts[(major, minor)] for minor in minors}
            for major, minors in builder.MAJOR_MINOR_QUOTAS.items()
        },
        "garment_native_category_counts": {
            str(category_id): native_counts[category_id]
            for category_id in sorted(builder.GARMENT_CATEGORY_NAMES)
        },
        "content_review_status": "pending_human_review",
        "rights_review_status": "pending_human_review",
        "cases": cases,
    }
    if end_recheck:
        _assert_release_unchanged(release)
        _assert_renderer_unchanged(renderer)
        _assert_review_unchanged(review)
        _assert_snapshots_unchanged((decision_snapshot,))
    return payload, release, renderer, decision_snapshot, review


def build_shortlist_payload(release_path: Path, decision_path: Path) -> dict[str, Any]:
    payload, _, _, _, _ = _prepare_shortlist(release_path, decision_path)
    return payload


def _shortlist_csv(payload: Mapping[str, Any]) -> str:
    fieldnames = [
        "shortlist_status",
        "manifest_sha256",
        "checksum_manifest_sha256",
        "raw_backed_audit_sha256",
        "decision_sha256",
        "review_index_sha256",
        *SHORTLIST_CASE_FIELDS,
        "candidate_rank",
        "agent_note",
        "review_sheet",
        "review_sheet_sha256",
        "review_tile_index",
        "review_tile_xywh",
        "content_review_status",
        "rights_review_status",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for case in payload["cases"]:
        row = dict(case)
        row["review_tile_xywh"] = json.dumps(row["review_tile_xywh"], separators=(",", ":"))
        writer.writerow(
            {
                "shortlist_status": payload["status"],
                "manifest_sha256": payload["manifest_sha256"],
                "checksum_manifest_sha256": payload["checksum_manifest_sha256"],
                "raw_backed_audit_sha256": payload["raw_backed_audit_sha256"],
                "decision_sha256": payload["decision_sha256"],
                "review_index_sha256": payload["review_index_sha256"],
                **row,
            }
        )
    return buffer.getvalue()


def finalize_shortlist(
    release_path: Path,
    decision_path: Path,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    payload, release, renderer, decision_snapshot, review = _prepare_shortlist(
        release_path,
        decision_path,
        end_recheck=False,
    )
    protected = [
        release.manifest.path,
        release.checksums.path,
        release.raw_audit.path,
        decision_snapshot.path,
        review.index.path,
        *(sheet.path for sheet in review.sheets),
    ]
    _require_fresh_output(output_json, protected)
    _require_fresh_output(output_csv, [*protected, output_json])
    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    csv_text = _shortlist_csv(payload)
    # End-of-input recheck precedes all publication.
    _assert_release_unchanged(release)
    _assert_renderer_unchanged(renderer)
    _assert_review_unchanged(review)
    _assert_snapshots_unchanged((decision_snapshot,))
    created_files: dict[Path, tuple[int, int]] = {}
    try:
        _atomic_create_text(output_json, json_text, created_files)
        _atomic_create_text(output_csv, csv_text, created_files)
    except BaseException:
        _remove_owned_file(output_csv, created_files.get(output_csv))
        _remove_owned_file(output_json, created_files.get(output_json))
        raise
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="run the non-writing strict release audit")
    audit_parser.add_argument("--manifest", type=Path, required=True, help="dataset root or primary manifest")

    seed_parser = subparsers.add_parser("seed", help="render sheets and write 6-per-minor decisions")
    seed_parser.add_argument("--manifest", type=Path, required=True, help="dataset root or primary manifest")
    seed_parser.add_argument("--output-decisions", type=Path, required=True)
    seed_parser.add_argument("--review-dir", type=Path, required=True)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="validate visual decisions and write a 3-per-minor derived shortlist",
    )
    finalize_parser.add_argument("--manifest", type=Path, required=True, help="dataset root or primary manifest")
    finalize_parser.add_argument("--decisions", type=Path, required=True)
    finalize_parser.add_argument("--output-json", type=Path, required=True)
    finalize_parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "audit":
        release = strict_release_snapshot(args.manifest)
        _assert_release_unchanged(release)
        payload: Mapping[str, Any] = {
            "status": "strict_read_only_audit_passed",
            **_release_evidence(release),
        }
    elif args.command == "seed":
        payload = seed_decision_file(args.manifest, args.output_decisions, args.review_dir)
    else:
        payload = finalize_shortlist(
            args.manifest,
            args.decisions,
            args.output_json,
            args.output_csv,
        )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

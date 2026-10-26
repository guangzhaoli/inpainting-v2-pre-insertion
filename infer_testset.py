#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from infer import (
    add_fast_canvas_sampler_cli_arg,
    apply_flow_map_defaults,
    install_fast_canvas_sampler,
    record_sampling_overrides,
    load_rgb_image,
    resolve_lora_path,
    save_latent_anchoring_masks,
    seed_everything,
    validate_adaptive_crop_inference_args,
    validate_configured_inference,
)
from src.models.complementary_canvas import complementary_canvas_enabled
from src.utils.latent_anchoring import (
    add_latent_anchoring_cli_args,
    apply_latent_anchoring_cli_overrides,
    ensure_latent_anchoring_config,
    latent_anchoring_metadata,
    normalize_latent_anchoring_mode,
    validate_latent_anchoring_runtime,
)
from src.utils.sigma_split import add_sigma_handoff_cli_args, apply_sigma_handoff_cli_overrides
from src.utils.visualization import (
    binarize_mask,
    build_strict_mask_composite,
    ensure_mask,
    ensure_rgb,
    make_infer_row,
    make_mask_overlay,
    save_mask,
    stack_rows,
)
from src.utils.adaptive_crop import (
    crop_pil_pair,
    restore_pil_output,
)
from src.utils.adaptive_crop_anchoring import (
    DEFAULT_CROP_EDGE_FEATHER_PX,
    change_gated_paste_back,
    prepare_adaptive_crop_for_inference,
)
from src.utils.inference_metadata import (
    build_effective_inference_metadata,
    build_inference_config_metadata,
    resolve_guidance_embeds_enabled,
    validate_checkpoint_compatibility,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "exp" / "accessory" / "adaptive_crop.yaml"
)
DEFAULT_TESTSET_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/testset_20260120"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "testset_adaptive_crop"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
BENCHMARK_MANIFEST_NAMES = ("paperinsert_900.jsonl", "anyinsertion_official.jsonl")
RELEASE_AUTHORITATIVE_PATHS = {
    "paperinsert_900.jsonl": (
        "manifests/paperinsert_900.jsonl",
        "manifests/paperinsert_900.csv",
        "manifests/object.jsonl",
        "manifests/person.jsonl",
        "manifests/garment.jsonl",
        "reports/sources.json",
    ),
    "anyinsertion_official.jsonl": (
        "manifests/anyinsertion_official.jsonl",
        "manifests/anyinsertion_clean67_for_current_model.jsonl",
        "manifests/anyinsertion_clean67_case_ids.txt",
        "manifests/anyinsertion_current_model_overlap_excluded33.json",
        "reports/conversion.json",
    ),
}
CANONICAL_MANIFEST_PATH_FIELDS = ("src_image", "src_mask", "tar_image", "tar_mask")
PROMOTED_BENCHMARK_FIELDS = (
    "major",
    "minor",
    "leaf_label",
    "protocol",
    "pair_kind",
    "case_type",
    "gt_available",
    "same_identity",
    "identity_id",
    "identity_scope",
    "identity_evidence_status",
    "reference_derived_from_target",
    "difficulty",
    "showcase_candidate",
    "paper_ok",
    "manual_qc_status",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch inference for Insert Anything testset folders.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--lora",
        type=Path,
        required=True,
        help="LoRA file or checkpoint directory. Its train_config.yaml is verified against --config.",
    )
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow a missing or incompatible checkpoint train_config.yaml. Unsafe unless intentional.",
    )
    parser.add_argument("--testset-root", type=Path, default=DEFAULT_TESTSET_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional benchmark subset JSONL. Relative paths are resolved under "
            "--testset-root; absolute paths must also remain inside it."
        ),
    )
    parser.add_argument(
        "--category",
        type=str,
        default="accessory",
        help="Category leaf under --testset-root (supports nested paths like pets/cats).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override all model devices, e.g. cuda:0.")
    parser.add_argument("--panel-size", type=int, default=320, help="Image area size for per-sample row previews.")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Data-parallel rank. Defaults to env RANK (0 if unset).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Data-parallel world size. Defaults to env WORLD_SIZE (1 if unset).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose sample_dir/output.png already exists; allow non-empty --output-dir.",
    )
    add_fast_canvas_sampler_cli_arg(parser)
    add_latent_anchoring_cli_args(parser)
    add_sigma_handoff_cli_args(parser)
    return record_sampling_overrides(parser.parse_args(argv), argv)


def natural_key(path: Path):
    parts = []
    for token in path.stem.replace("_", "-").split("-"):
        parts.append((0, int(token)) if token.isdigit() else (1, token.lower()))
    return parts


def image_files(directory: Path):
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )


def find_stem_file(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
        upper_candidate = directory / f"{stem}{suffix.upper()}"
        if upper_candidate.exists():
            return upper_candidate
    return None


def _validate_safe_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Benchmark manifest {label} must be one safe path component: {value!r}")


def _validate_category_path(category: str) -> None:
    """Allow nested category paths like pets/cats while rejecting escapes."""
    if not category or category in {".", ".."}:
        raise ValueError(f"Invalid category path: {category!r}")
    parts = Path(category).parts
    if not parts:
        raise ValueError(f"Invalid category path: {category!r}")
    for part in parts:
        if part in {".", ".."} or "/" in part or "\\" in part:
            raise ValueError(
                f"Benchmark category path component must be a safe name: {category!r}"
            )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_normalized_release_path(value: str, label: str) -> str:
    relative_path = Path(value)
    if (
        not value
        or "\x00" in value
        or relative_path.is_absolute()
        or ".." in relative_path.parts
        or "\\" in value
        or relative_path.as_posix() != value
    ):
        raise ValueError(f"{label} must be a normalized safe POSIX relative path: {value!r}")
    return value


def _require_contained_path_without_symlinks(
    testset_root: Path,
    path: Path,
    label: str,
    *,
    kind: str,
) -> Path:
    """Return a lexical path below root after rejecting every symlink component."""

    root = testset_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain inside testset root {root}: {candidate}"
        ) from exc

    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise FileNotFoundError(f"{label} does not exist: {candidate}") from exc
    if not _is_relative_to(resolved, root):
        raise ValueError(
            f"{label} must resolve inside testset root {root}: {candidate} -> {resolved}"
        )
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {candidate}")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {candidate}")
    if kind not in {"file", "directory"}:
        raise ValueError(f"Unsupported contained-path kind: {kind!r}")
    return candidate


def _supported_primary_manifest_paths(testset_root: Path) -> list[Path]:
    root = testset_root.resolve(strict=True)
    manifest_root = root / "manifests"
    if manifest_root.is_symlink():
        raise ValueError(f"Benchmark manifest directory contains a symlink: {manifest_root}")
    return [
        manifest_root / name
        for name in BENCHMARK_MANIFEST_NAMES
        if (manifest_root / name).is_symlink() or (manifest_root / name).exists()
    ]


def resolve_benchmark_manifest_path(testset_root: Path, manifest: Path) -> Path:
    """Resolve an explicitly selected manifest without root escapes or aliases."""

    root = testset_root.resolve(strict=True)
    if ".." in manifest.parts:
        raise ValueError(f"Benchmark manifest path must not contain '..': {manifest}")
    candidate = manifest if manifest.is_absolute() else root / manifest
    return _require_contained_path_without_symlinks(
        root,
        candidate,
        "Benchmark manifest",
        kind="file",
    )


def _validate_manifest_record_paths(record: dict, manifest_path: Path, line_number: int) -> None:
    major = str(record["major"])
    case_id = str(record["case_id"])
    _validate_safe_component(major, f"major at {manifest_path}:{line_number}")
    _validate_safe_component(case_id, f"case_id at {manifest_path}:{line_number}")
    canonical_directories = {
        "src_image": "src_image",
        "src_mask": "src_mask",
        "tar_image": "tar_image",
        "tar_mask": "tar_mask",
    }
    for field, directory in canonical_directories.items():
        raw_path = record[field]
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(
                f"Benchmark manifest {manifest_path}:{line_number} field {field} "
                "must be a non-empty relative path"
            )
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Benchmark manifest {manifest_path}:{line_number} field {field} "
                f"must not be absolute or contain '..': {raw_path!r}"
            )
        if relative_path.as_posix() != raw_path:
            raise ValueError(
                f"Benchmark manifest {manifest_path}:{line_number} field {field} "
                f"is not a normalized POSIX path: {raw_path!r}"
            )
        if (
            len(relative_path.parts) != 3
            or relative_path.parts[0] != major
            or relative_path.parts[1] != directory
            or relative_path.stem != case_id
            or relative_path.suffix.lower() not in IMAGE_EXTS
        ):
            raise ValueError(
                f"Benchmark manifest {manifest_path}:{line_number} field {field} does not "
                f"match canonical {major}/{directory}/{case_id}.<image-ext>: {raw_path!r}"
            )


def load_benchmark_manifest_index(
    testset_root: Path,
    manifest: Path | None = None,
) -> tuple[dict[tuple[str, str], dict], Path | None]:
    if manifest is not None:
        manifest_path = resolve_benchmark_manifest_path(testset_root, manifest)
    else:
        existing = _supported_primary_manifest_paths(testset_root)
        if not existing:
            return {}, None
        if len(existing) > 1:
            raise ValueError(f"Multiple primary benchmark manifests found: {existing}")
        manifest_path = resolve_benchmark_manifest_path(testset_root, existing[0])

    index: dict[tuple[str, str], dict] = {}
    case_owners: dict[str, tuple[str, int]] = {}
    path_owners: dict[str, tuple[str, int]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Benchmark manifest {manifest_path}:{line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Benchmark manifest {manifest_path}:{line_number} must contain a JSON object"
                )
            missing = {"case_id", "major", "src_image", "src_mask", "tar_image", "tar_mask"}.difference(record)
            if missing:
                raise ValueError(
                    f"Benchmark manifest {manifest_path}:{line_number} is missing {sorted(missing)}"
                )
            key = (str(record["major"]), str(record["case_id"]))
            if key in index:
                raise ValueError(f"Duplicate benchmark manifest key {key} in {manifest_path}")
            for field in CANONICAL_MANIFEST_PATH_FIELDS:
                raw_path = record[field]
                if isinstance(raw_path, str) and raw_path in path_owners:
                    owner_field, owner_line = path_owners[raw_path]
                    raise ValueError(
                        f"Duplicate benchmark path {raw_path!r} in {manifest_path}:{line_number} "
                        f"field {field}; first used by {owner_field} on line {owner_line}"
                    )
            _validate_manifest_record_paths(record, manifest_path, line_number)
            case_id = key[1]
            if case_id in case_owners:
                owner_major, owner_line = case_owners[case_id]
                raise ValueError(
                    f"Duplicate benchmark case_id {case_id!r} in {manifest_path}:{line_number}; "
                    f"first used by major {owner_major!r} on line {owner_line}"
                )
            case_owners[case_id] = (key[0], line_number)
            for field in CANONICAL_MANIFEST_PATH_FIELDS:
                relative_path = str(record[field])
                path_owners[relative_path] = (field, line_number)
            index[key] = record
    if manifest is not None:
        validate_explicit_manifest_against_primary(
            testset_root,
            manifest_path,
            index,
        )
    else:
        _verify_supported_release_integrity(testset_root, manifest_path, index)
    return index, manifest_path


def validate_explicit_manifest_against_primary(
    testset_root: Path,
    explicit_path: Path,
    explicit_index: dict[tuple[str, str], dict],
) -> Path:
    """Prove an explicit metric split is an unchanged, ordered primary subset."""

    primary_paths = [
        resolve_benchmark_manifest_path(testset_root, path)
        for path in _supported_primary_manifest_paths(testset_root)
    ]
    if len(primary_paths) != 1:
        raise ValueError(
            "Explicit benchmark manifests require exactly one primary manifest; "
            f"found {primary_paths}"
        )
    primary_path = primary_paths[0]
    if explicit_path == primary_path:
        _verify_supported_release_integrity(testset_root, primary_path, explicit_index)
        return primary_path
    primary_index, loaded_primary_path = load_benchmark_manifest_index(testset_root, None)
    if loaded_primary_path != primary_path:
        raise AssertionError("Primary benchmark manifest resolution changed during validation")
    primary_positions = {key: position for position, key in enumerate(primary_index)}
    explicit_positions: list[int] = []
    for key, record in explicit_index.items():
        if key not in primary_index:
            raise ValueError(f"Explicit benchmark manifest contains non-primary case: {key}")
        if record != primary_index[key]:
            raise ValueError(f"Explicit benchmark manifest altered primary metadata: {key}")
        explicit_positions.append(primary_positions[key])
    if explicit_positions != sorted(explicit_positions):
        raise ValueError("Explicit benchmark manifest must preserve primary row order")
    return primary_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_supported_release_integrity(
    testset_root: Path,
    primary_manifest_path: Path,
    manifest_index: dict[tuple[str, str], dict],
) -> None:
    """Verify the complete release inventory before any sample is discovered."""

    root = testset_root.resolve(strict=True)
    primary_name = primary_manifest_path.name
    authoritative_paths = RELEASE_AUTHORITATIVE_PATHS.get(primary_name)
    if authoritative_paths is None:
        raise ValueError(f"Unsupported primary benchmark manifest: {primary_manifest_path}")
    expected_primary_relative = f"manifests/{primary_name}"
    try:
        actual_primary_relative = primary_manifest_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Primary benchmark manifest is outside testset root: {primary_manifest_path}"
        ) from exc
    if actual_primary_relative != expected_primary_relative:
        raise ValueError(
            "Primary benchmark manifest is not at its authoritative release path: "
            f"actual={actual_primary_relative!r} expected={expected_primary_relative!r}"
        )

    expected_paths = set(authoritative_paths)
    for record in manifest_index.values():
        for field in CANONICAL_MANIFEST_PATH_FIELDS:
            expected_paths.add(
                _validate_normalized_release_path(
                    str(record[field]),
                    f"Benchmark manifest field {field}",
                )
            )

    checksum_path = _require_contained_path_without_symlinks(
        root,
        root / "CHECKSUMS.sha256",
        "Benchmark release checksum manifest",
        kind="file",
    )
    errors: list[str] = []
    recorded: dict[str, str] = {}
    try:
        checksum_lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Cannot read benchmark release checksum manifest: {checksum_path}") from exc
    if not checksum_lines:
        errors.append("Checksum manifest is empty")
    for line_number, line in enumerate(checksum_lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            errors.append(f"Invalid checksum row {line_number}: {line!r}")
            continue
        digest, relative = match.groups()
        try:
            _validate_normalized_release_path(
                relative,
                f"Checksum row {line_number} path",
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if relative in recorded:
            errors.append(f"Duplicate checksum path at row {line_number}: {relative}")
            continue
        recorded[relative] = digest

    recorded_paths = set(recorded)
    if recorded_paths != expected_paths:
        errors.append(
            "Checksum path set mismatch: "
            f"missing={sorted(expected_paths - recorded_paths)[:10]} "
            f"extra={sorted(recorded_paths - expected_paths)[:10]}"
        )
    if errors:
        raise ValueError(
            "Benchmark release integrity verification failed before sample discovery:\n- "
            + "\n- ".join(errors[:50])
        )

    missing_errors: list[str] = []
    for relative in sorted(expected_paths):
        candidate = root / relative
        try:
            regular_path = _require_contained_path_without_symlinks(
                root,
                candidate,
                f"Checksummed release file {relative}",
                kind="file",
            )
        except FileNotFoundError as exc:
            missing_errors.append(str(exc))
            continue
        except ValueError as exc:
            errors.append(str(exc))
            continue
        actual_digest = _sha256_file(regular_path)
        if recorded[relative] != actual_digest:
            errors.append(
                f"Checksum mismatch {relative}: "
                f"recorded={recorded[relative]} actual={actual_digest}"
            )
    if missing_errors:
        raise FileNotFoundError(
            "Benchmark release integrity verification found missing files before sample discovery:\n- "
            + "\n- ".join(missing_errors[:50])
        )
    if errors:
        raise ValueError(
            "Benchmark release integrity verification failed before sample discovery:\n- "
            + "\n- ".join(errors[:50])
        )


def _selected_manifest_sha256(samples: list[dict]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        record = sample["benchmark_metadata"]
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def benchmark_manifest_provenance(sample: dict) -> dict | None:
    manifest_path = sample.get("benchmark_manifest")
    if manifest_path is None:
        return None
    result = {
        "path": str(Path(manifest_path).resolve()),
        "sha256": str(sample["benchmark_manifest_sha256"]),
        "selection_sha256": str(sample["benchmark_manifest_selection_sha256"]),
        "selected_rows": int(sample["benchmark_manifest_selected_rows"]),
        "mode": str(sample["benchmark_manifest_mode"]),
    }
    primary_path = sample.get("benchmark_primary_manifest")
    if primary_path is not None:
        result["primary_path"] = str(Path(primary_path).resolve())
        result["primary_sha256"] = str(sample["benchmark_primary_manifest_sha256"])
    return result


def benchmark_output_fields(sample: dict) -> dict:
    benchmark = sample.get("benchmark_metadata")
    fields = {
        "case_id": str(benchmark.get("case_id", sample["stem"])) if benchmark else sample["stem"],
        # Programmatic callers such as infer_ref_sensitivity predate the
        # canonical loader and pass the legacy four paths without this field.
        "input_schema": sample.get("schema", "legacy_ref_tar"),
    }
    if benchmark:
        fields["benchmark"] = dict(benchmark)
        for key in PROMOTED_BENCHMARK_FIELDS:
            if key in benchmark:
                fields[key] = benchmark[key]
    manifest_provenance = benchmark_manifest_provenance(sample)
    if manifest_provenance is not None:
        fields["benchmark_manifest"] = manifest_provenance
    return fields


def prepare_fresh_output_dir(output_dir: Path, *, allow_existing: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Refusing to mix a new run with existing outputs in {output_dir}. "
            "Choose a fresh --output-dir, or pass --skip-existing to resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _reject_canonical_role_symlinks(testset_root: Path, role_dirs: dict[str, Path]) -> None:
    for role, directory in role_dirs.items():
        _require_contained_path_without_symlinks(
            testset_root,
            directory,
            f"Canonical {role} directory",
            kind="directory",
        )
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise ValueError(
                    f"Canonical {role} file contains a symlink alias: {entry}"
                )


def discover_samples(
    testset_root: Path,
    category: str,
    limit: int | None,
    manifest: Path | None = None,
):
    _validate_category_path(category)
    testset_root = testset_root.resolve(strict=True)
    manifest_index, manifest_path = load_benchmark_manifest_index(testset_root, manifest)
    explicit_manifest = manifest is not None
    manifest_sha256 = _sha256_file(manifest_path) if manifest_path is not None else None
    if explicit_manifest and manifest_path is not None:
        primary_paths = [
            resolve_benchmark_manifest_path(testset_root, path)
            for path in _supported_primary_manifest_paths(testset_root)
        ]
        if len(primary_paths) != 1:
            raise AssertionError(
                "Explicit manifest validation completed without exactly one primary manifest"
            )
        primary_manifest_path = primary_paths[0]
    else:
        primary_manifest_path = manifest_path
    primary_manifest_sha256 = (
        _sha256_file(primary_manifest_path)
        if primary_manifest_path is not None
        else None
    )

    category_dir = testset_root / category
    canonical_dirs = {
        "ref_image": category_dir / "src_image",
        "ref_mask": category_dir / "src_mask",
        "source": category_dir / "tar_image",
        "source_mask": category_dir / "tar_mask",
    }
    legacy_dirs = {
        "ref_image": category_dir / "ref_image",
        "ref_mask": category_dir / "ref_mask",
        "source": category_dir / "tar_image",
        "source_mask": category_dir / "tar_mask",
    }
    if all(path.is_dir() for path in canonical_dirs.values()):
        schema = "src_tar_v1"
        selected_dirs = canonical_dirs
        _reject_canonical_role_symlinks(testset_root, canonical_dirs)
    elif all(path.is_dir() for path in legacy_dirs.values()):
        schema = "legacy_ref_tar"
        selected_dirs = legacy_dirs
    else:
        canonical_missing = [str(path) for path in canonical_dirs.values() if not path.is_dir()]
        legacy_missing = [str(path) for path in legacy_dirs.values() if not path.is_dir()]
        raise FileNotFoundError(
            "Testset category must use either canonical "
            "{src_image,src_mask,tar_image,tar_mask} or legacy "
            "{ref_image,ref_mask,tar_image,tar_mask} directories. "
            f"Canonical missing: {canonical_missing}; legacy missing: {legacy_missing}"
        )

    ref_image_dir = selected_dirs["ref_image"]
    ref_mask_dir = selected_dirs["ref_mask"]
    src_image_dir = selected_dirs["source"]
    src_mask_dir = selected_dirs["source_mask"]
    samples = []
    discovered_stems: set[str] = set()
    for ref_image_path in image_files(ref_image_dir):
        stem = ref_image_path.stem
        if stem in discovered_stems:
            raise ValueError(
                f"Duplicate canonical src_image stem {category}/{stem} under {ref_image_dir}"
            )
        discovered_stems.add(stem)
        benchmark = manifest_index.get((category, stem)) if manifest_path is not None else None
        if explicit_manifest and benchmark is None:
            # An explicit manifest is an intentional subset selector. Canonical
            # cases outside that selector do not participate in this run.
            continue
        ref_mask_path = find_stem_file(ref_mask_dir, stem)
        src_image_path = find_stem_file(src_image_dir, stem)
        src_mask_path = find_stem_file(src_mask_dir, stem)
        missing = [
            name
            for name, path in (
                ("ref_mask", ref_mask_path),
                ("source", src_image_path),
                ("source_mask", src_mask_path),
            )
            if path is None
        ]
        if missing:
            raise FileNotFoundError(f"Sample {stem} is missing: {missing}")

        sample = {
            "stem": stem,
            "schema": schema,
            "ref_image": ref_image_path,
            "ref_mask": ref_mask_path,
            "source": src_image_path,
            "source_mask": src_mask_path,
        }
        if manifest_path is not None:
            if benchmark is None:
                raise ValueError(
                    f"Canonical sample {category}/{stem} has no row in primary manifest {manifest_path}"
                )
            expected_paths = {
                "ref_image": testset_root / str(benchmark["src_image"]),
                "ref_mask": testset_root / str(benchmark["src_mask"]),
                "source": testset_root / str(benchmark["tar_image"]),
                "source_mask": testset_root / str(benchmark["tar_mask"]),
            }
            for key, expected_path in expected_paths.items():
                try:
                    expected_regular = _require_contained_path_without_symlinks(
                        testset_root,
                        expected_path,
                        f"Manifest-selected sample {category}/{stem} {key}",
                        kind="file",
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"Manifest-selected sample {category}/{stem} is missing {key}: "
                        f"{expected_path}"
                    ) from exc
                if sample[key] != expected_regular:
                    raise ValueError(
                        f"Manifest/filesystem mismatch for {category}/{stem} {key}: "
                        f"discovered={sample[key]} manifest={expected_path}"
                    )
            sample["benchmark_metadata"] = benchmark
            sample["benchmark_manifest"] = manifest_path
        samples.append(sample)

    if manifest_path is not None:
        expected_keys = {key for key in manifest_index if key[0] == category}
        discovered_keys = {(category, str(sample["stem"])) for sample in samples}
        missing_filesystem_cases = sorted(expected_keys.difference(discovered_keys))
        if missing_filesystem_cases:
            raise FileNotFoundError(
                f"Primary manifest {manifest_path} has {len(missing_filesystem_cases)} "
                f"{category} cases missing from canonical src_image discovery: "
                f"{missing_filesystem_cases[:10]}"
            )

        # Manifest row order defines seed assignment and --limit semantics.
        # Filesystem enumeration is used only for strict presence checking.
        samples_by_key = {
            (category, str(sample["stem"])): sample
            for sample in samples
        }
        ordered_keys = [key for key in manifest_index if key[0] == category]
        samples = [samples_by_key[key] for key in ordered_keys]

    if not samples:
        raise ValueError(f"No samples found for category {category} under {testset_root}")
    selected_samples = samples[:limit] if limit is not None else samples
    if manifest_path is not None:
        selected_sha256 = _selected_manifest_sha256(selected_samples)
        for sample in selected_samples:
            sample["benchmark_manifest_sha256"] = manifest_sha256
            sample["benchmark_manifest_selection_sha256"] = selected_sha256
            sample["benchmark_manifest_selected_rows"] = len(selected_samples)
            sample["benchmark_manifest_mode"] = "explicit_subset" if explicit_manifest else "primary"
            sample["benchmark_primary_manifest"] = primary_manifest_path
            sample["benchmark_primary_manifest_sha256"] = primary_manifest_sha256
    return selected_samples


def setup_trainer(args):
    from src.data.config import load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be provided together")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.true_cfg_scale <= 0 or args.guidance_scale <= 0:
        raise ValueError("--true-cfg-scale and --guidance-scale must be positive")
    if args.panel_size <= 0:
        raise ValueError("--panel-size must be positive")
    limit = getattr(args, "limit", None)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    if args.height is not None and (args.height <= 0 or args.width <= 0):
        raise ValueError("--height and --width must be positive")

    config = load_config_from_yaml(str(args.config), for_inference=True)
    apply_flow_map_defaults(config, args)
    if getattr(args, "device", None) is not None:
        from src.data.config import DeviceConfig

        config.predict.devices = DeviceConfig(vae=args.device, text_encoder=args.device, dit=args.device)
    apply_latent_anchoring_cli_overrides(config, args)
    apply_sigma_handoff_cli_overrides(config, args)
    validate_adaptive_crop_inference_args(config, args)
    validate_latent_anchoring_runtime(config, height=args.height, width=args.width)
    if getattr(args, "fast_canvas_sampler", False) and not complementary_canvas_enabled(config):
        raise ValueError("--fast-canvas-sampler needs model.complementary_canvas.enabled")
    resolved_lora_path = resolve_lora_path(args.lora)
    config.model.lora.pretrained_weight = str(resolved_lora_path)

    validation_args = SimpleNamespace(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        allow_base_model=False,
    )
    validate_configured_inference(config, validation_args, resolved_lora_path)
    checkpoint_compatibility = validate_checkpoint_compatibility(
        config,
        resolved_lora_path,
        allow_mismatch=bool(getattr(args, "allow_checkpoint_mismatch", False)),
    )

    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()
    install_fast_canvas_sampler(trainer, args)
    return trainer, config, resolved_lora_path, checkpoint_compatibility


def run_sample(
    trainer,
    sample,
    sample_dir: Path,
    args,
    sample_seed: int,
    resolved_lora_path: Path | None = None,
    checkpoint_compatibility: dict | None = None,
):
    seed_everything(sample_seed)

    ref_image = load_rgb_image(sample["ref_image"])
    ref_mask = load_rgb_image(sample["ref_mask"])
    src_image = load_rgb_image(sample["source"])
    src_mask = load_rgb_image(sample["source_mask"])

    original_src_image = ensure_rgb(src_image)
    original_src_mask = ensure_mask(src_mask)
    if original_src_image.size != original_src_mask.size:
        raise ValueError(
            "source image and source mask must have identical geometry, "
            f"got image={original_src_image.size}, mask={original_src_mask.size}"
        )
    adaptive_crop_config = trainer.config.data.init_args.adaptive_crop
    latent_anchoring_config = ensure_latent_anchoring_config(trainer.config)
    anchoring_metadata = latent_anchoring_metadata(latent_anchoring_config)
    anchoring_is_enabled = bool(anchoring_metadata["enabled"])
    canvas_is_enabled = complementary_canvas_enabled(trainer.config)
    # Both keep the source latent outside the edit region, so the raw output is primary.
    latent_preserving = anchoring_is_enabled or canvas_is_enabled
    composite_mask_threshold = (
        float(latent_anchoring_config.mask_threshold)
        if anchoring_is_enabled
        else float(adaptive_crop_config.mask_threshold)
    )
    crop_plan = None
    crop_compatibility = {"enabled": False}
    predict_src_image = original_src_image
    predict_src_mask = original_src_mask
    edge_feather_px = 0
    if adaptive_crop_config.predict_enabled:
        if args.height is not None or args.width is not None:
            raise ValueError("Adaptive Crop prediction cannot be combined with explicit --height/--width")
        crop_plan, crop_compatibility = prepare_adaptive_crop_for_inference(
            image_size=original_src_image.size,
            mask=original_src_mask,
            adaptive_crop_config=adaptive_crop_config,
            config=trainer.config,
            processor_config=trainer.config.data.init_args.processor,
        )
        predict_src_image, predict_src_mask = crop_pil_pair(
            original_src_image,
            original_src_mask,
            crop_plan,
        )
        if (
            latent_preserving
            and crop_plan is not None
            and crop_plan.applied
            and latent_anchoring_config.paste_back == "crop_feather"
        ):
            edge_feather_px = int(
                crop_compatibility.get("feather_width_px", DEFAULT_CROP_EDGE_FEATHER_PX)
            )

    predict_kwargs = {
        "image": [predict_src_image, ref_image],
        "prompt": args.prompt,
        "ref_mask": ref_mask,
        "src_mask": predict_src_mask,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "latent_anchoring_mode": normalize_latent_anchoring_mode(latent_anchoring_config.mode),
        "latent_anchoring_mask_threshold": float(latent_anchoring_config.mask_threshold),
        "interaction_dilation_px": int(latent_anchoring_config.interaction_dilation_px),
        "transition_width_px": int(latent_anchoring_config.transition_width_px),
    }
    if args.height is not None:
        predict_kwargs["height"] = args.height
    if args.width is not None:
        predict_kwargs["width"] = args.width

    result = trainer.predict(**predict_kwargs)
    local_output = ensure_rgb(result[0])
    if crop_plan is not None:
        raw_restored_output, strict_composite = restore_pil_output(
            original_source=original_src_image,
            original_mask=original_src_mask,
            local_output=local_output,
            plan=crop_plan,
            blend_mode=adaptive_crop_config.blend_mode,
            edge_feather_px=0,
        )
        if edge_feather_px > 0:
            primary_restored_output, _ = restore_pil_output(
                original_source=original_src_image,
                original_mask=original_src_mask,
                local_output=local_output,
                plan=crop_plan,
                blend_mode=adaptive_crop_config.blend_mode,
                edge_feather_px=edge_feather_px,
            )
        else:
            primary_restored_output = raw_restored_output
    else:
        raw_restored_output = local_output.resize(
            original_src_image.size,
            Image.Resampling.LANCZOS,
        )
        primary_restored_output = raw_restored_output
        strict_composite = build_strict_mask_composite(
            original_src_image,
            original_src_mask,
            raw_restored_output,
            composite_mask_threshold,
        )
    if latent_anchoring_config.paste_back == "change_gated":
        primary_restored_output = change_gated_paste_back(
            original_src_image,
            original_src_mask,
            raw_restored_output,
            local_size=local_output.size,
            crop_plan=crop_plan,
            settings=latent_anchoring_config,
        )
    primary_output = primary_restored_output if latent_preserving else strict_composite
    if (
        raw_restored_output.size != original_src_image.size
        or strict_composite.size != original_src_image.size
        or primary_output.size != original_src_image.size
    ):
        raise ValueError("restored and final outputs must match the original source size")

    sample_dir.mkdir(parents=True, exist_ok=True)
    ensure_rgb(ref_image).save(sample_dir / "ref_image.png")
    save_mask(ref_mask, sample_dir / "ref_mask.png")
    make_mask_overlay(ref_image, ref_mask).save(sample_dir / "ref_mask_overlay.png")

    ensure_rgb(original_src_image).save(sample_dir / "source.png")
    save_mask(original_src_mask, sample_dir / "source_mask.png")
    make_mask_overlay(original_src_image, original_src_mask).save(sample_dir / "source_mask_overlay.png")

    ensure_rgb(local_output).save(sample_dir / "output_local.png")
    ensure_rgb(raw_restored_output).save(sample_dir / "output_restored_raw.png")

    if crop_plan is not None:
        ensure_rgb(predict_src_image).save(sample_dir / "crop_source.png")
        save_mask(predict_src_mask, sample_dir / "crop_mask.png")

    ensure_rgb(primary_output).save(sample_dir / "output.png")
    output_mask = binarize_mask(
        original_src_mask,
        composite_mask_threshold,
    ).resize(primary_output.size, Image.Resampling.NEAREST)
    output_mask.save(sample_dir / "output_mask.png")
    make_mask_overlay(primary_output, output_mask).save(sample_dir / "output_mask_overlay.png")

    ensure_rgb(strict_composite).save(sample_dir / "composite.png")

    anchoring_mask_paths = {}
    if anchoring_is_enabled:
        anchoring_mask_paths = save_latent_anchoring_masks(
            trainer,
            {
                "mask_core": sample_dir / "mask_core.png",
                "mask_free": sample_dir / "mask_free.png",
                "edit_weight": sample_dir / "edit_weight.png",
            },
            original_src_image.size,
            crop_plan=crop_plan,
        )

    row = make_infer_row(
        ref_image=ref_image,
        ref_mask=ref_mask,
        source=original_src_image,
        source_mask=original_src_mask,
        output=primary_output,
        composite=strict_composite,
        panel_size=args.panel_size,
        output_mask=output_mask,
    )
    row_path = sample_dir / "row.png"
    row.save(row_path)

    metadata = {
        "stem": sample["stem"],
        **benchmark_output_fields(sample),
        "seed": int(sample_seed),
        "config": build_inference_config_metadata(trainer.config, args.config),
        "checkpoint": {
            "argument": str(args.lora),
            "resolved_lora": str(resolved_lora_path) if resolved_lora_path is not None else None,
            "compatibility": checkpoint_compatibility,
        },
        "parameters": build_effective_inference_metadata(
            trainer.config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_embeds_enabled=resolve_guidance_embeds_enabled(trainer.config, trainer),
            requested_height=args.height,
            requested_width=args.width,
            effective_height=local_output.height,
            effective_width=local_output.width,
        ),
        "inputs": {
            "schema": str(sample.get("schema", "legacy_ref_tar")),
            **{
                key: str(sample[key])
                for key in ("ref_image", "ref_mask", "source", "source_mask")
            },
        },
        "outputs": {
            "ref_image": str(sample_dir / "ref_image.png"),
            "ref_mask": str(sample_dir / "ref_mask.png"),
            "ref_mask_overlay": str(sample_dir / "ref_mask_overlay.png"),
            "source": str(sample_dir / "source.png"),
            "source_mask": str(sample_dir / "source_mask.png"),
            "source_mask_overlay": str(sample_dir / "source_mask_overlay.png"),
            "output": str(sample_dir / "output.png"),
            "primary_output": str(sample_dir / "output.png"),
            "final_composite": str(
                sample_dir / ("composite.png" if latent_preserving else "output.png")
            ),
            "strict_composite": str(
                sample_dir / ("composite.png" if latent_preserving else "output.png")
            ),
            "output_local": str(sample_dir / "output_local.png"),
            "output_restored_raw": str(sample_dir / "output_restored_raw.png"),
            "output_mask": str(sample_dir / "output_mask.png"),
            "output_mask_overlay": str(sample_dir / "output_mask_overlay.png"),
            "composite_compatibility_copy": str(sample_dir / "composite.png"),
            "row": str(row_path),
            **anchoring_mask_paths,
        },
        "sizes": {
            "ref_image": list(ref_image.size),
            "source": list(original_src_image.size),
            "prediction_input": list(predict_src_image.size),
            "output_local": list(local_output.size),
            "output_restored_raw": list(raw_restored_output.size),
            "output": list(primary_output.size),
            "final_composite": list(strict_composite.size),
        },
        "latent_anchoring": anchoring_metadata,
        "complementary_canvas": getattr(trainer, "_last_complementary_canvas", None) if canvas_is_enabled else None,
        "adaptive_crop": {
            "config": adaptive_crop_config.model_dump(mode="json"),
            "plan": crop_plan.to_dict() if crop_plan is not None else None,
            "compatibility": crop_compatibility,
        },
        "output_semantics": {
            "output": (
                "Primary latent-anchored result in original source coordinates "
                "(crop-edge-safe when adaptive crop is applied)."
                if anchoring_is_enabled
                else "Primary complementary-canvas result (generated M_free tokens in the source latent) "
                "in original source coordinates (crop-edge-safe when adaptive crop is applied)."
                if canvas_is_enabled
                else "Final strict-mask composite in original source coordinates."
            ),
            "output_local": "Direct model output in processor coordinates.",
            "output_restored_raw": (
                "Latent-anchored model output hard-pasted into original source coordinates "
                "before optional crop-edge feathering / strict RGB compositing."
                if anchoring_is_enabled
                else "Complementary-canvas model output hard-pasted into original source coordinates "
                "before optional crop-edge feathering / strict RGB compositing."
                if canvas_is_enabled
                else "Model output mapped to original source coordinates before mask compositing."
            ),
            "strict_composite": "Compatibility M_core strict composite for A/B comparison.",
        },
    }
    if crop_plan is not None:
        metadata["outputs"].update(
            {
                "crop_source": str(sample_dir / "crop_source.png"),
                "crop_mask": str(sample_dir / "crop_mask.png"),
            }
        )
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata, row_path


def _resolve_parallelism(args) -> tuple[int, int]:
    rank = args.rank if args.rank is not None else int(os.environ.get("RANK", "0"))
    world_size = (
        args.world_size
        if args.world_size is not None
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size <= 0:
        raise ValueError("--world-size / WORLD_SIZE must be positive")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank out of range: {rank}/{world_size}")
    return rank, world_size


def main():
    args = parse_args()
    rank, world_size = _resolve_parallelism(args)
    all_samples = discover_samples(
        args.testset_root, args.category, args.limit, manifest=args.manifest
    )
    # Keep global indices so per-sample seeds stay stable under sharding.
    indexed = list(enumerate(all_samples))
    shard = [(global_i, sample) for global_i, sample in indexed if global_i % world_size == rank]
    if args.seed + max((len(all_samples) - 1), 0) >= 2**32:
        raise ValueError("per-sample seeds must remain in [0, 2**32)")
    prepare_fresh_output_dir(args.output_dir, allow_existing=bool(args.skip_existing) or world_size > 1)
    trainer, config, resolved_lora_path, checkpoint_compatibility = setup_trainer(args)
    guidance_embeds_enabled = resolve_guidance_embeds_enabled(config, trainer)
    run_anchoring_metadata = latent_anchoring_metadata(config.predict.latent_anchoring)

    run_config = {
        "config": build_inference_config_metadata(config, args.config),
        "checkpoint": {
            "argument": str(args.lora),
            "resolved_lora": str(resolved_lora_path),
            "compatibility": checkpoint_compatibility,
        },
        "testset_root": str(args.testset_root),
        "category": args.category,
        "num_samples_global": len(all_samples),
        "num_samples_rank": len(shard),
        "rank": rank,
        "world_size": world_size,
        "skip_existing": bool(args.skip_existing),
        "parameters": build_effective_inference_metadata(
            config,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            guidance_scale=args.guidance_scale,
            guidance_embeds_enabled=guidance_embeds_enabled,
            requested_height=args.height,
            requested_width=args.width,
        ),
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "latent_anchoring": run_anchoring_metadata,
        "adaptive_crop": config.data.init_args.adaptive_crop.model_dump(mode="json"),
        "output_semantics": {
            "output": (
                "Primary latent-anchored result in original source coordinates."
                if run_anchoring_metadata["enabled"]
                else "Final strict-mask composite in original source coordinates."
            ),
            "output_local": "Direct model output in processor coordinates.",
            "output_restored_raw": (
                "Latent-anchored model output hard-pasted into original source coordinates "
                "before optional crop-edge feathering; primary may use feathered restore."
                if run_anchoring_metadata["enabled"]
                else "Model output mapped to original source coordinates before mask compositing."
            ),
            "strict_composite": "Compatibility M_core strict composite for A/B comparison.",
        },
    }
    if all_samples:
        manifest_provenance = benchmark_manifest_provenance(all_samples[0])
        if manifest_provenance is not None:
            run_config["benchmark_manifest"] = manifest_provenance
    run_config_name = (
        f"run_config_rank{rank:02d}.json" if world_size > 1 else "run_config.json"
    )
    with (args.output_dir / run_config_name).open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    manifest_path = (
        args.output_dir / f"manifest_rank{rank:02d}.jsonl"
        if world_size > 1
        else args.output_dir / "manifest.jsonl"
    )
    row_paths = []
    n_ok = 0
    n_skip = 0
    write_mode = "a" if args.skip_existing and manifest_path.exists() else "w"
    with manifest_path.open(write_mode, encoding="utf-8") as manifest:
        for local_i, (global_i, sample) in enumerate(shard):
            sample_seed = args.seed + global_i
            sample_dir = args.output_dir / f"{global_i + 1:03d}_{sample['stem']}"
            if args.skip_existing and (sample_dir / "output.png").is_file():
                n_skip += 1
                print(
                    f"[rank{rank} {local_i + 1}/{len(shard)}] skip existing "
                    f"{sample['stem']} -> {sample_dir}",
                    flush=True,
                )
                continue
            print(
                f"[rank{rank} {local_i + 1}/{len(shard)} global {global_i + 1}/{len(all_samples)}] "
                f"infer {sample['stem']} -> {sample_dir}",
                flush=True,
            )
            metadata, row_path = run_sample(
                trainer,
                sample,
                sample_dir,
                args,
                sample_seed,
                resolved_lora_path=resolved_lora_path,
                checkpoint_compatibility=checkpoint_compatibility,
            )
            manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            manifest.flush()
            row_paths.append(row_path)
            n_ok += 1

    if row_paths:
        overview_name = (
            f"all_rows_rank{rank:02d}.png" if world_size > 1 else "all_rows.png"
        )
        stack_rows(row_paths, args.output_dir / overview_name)
        print(f"Overview saved to {args.output_dir / overview_name}", flush=True)
    print(
        f"Done rank={rank}/{world_size}. ok={n_ok} skip={n_skip} "
        f"assigned={len(shard)} global={len(all_samples)} out={args.output_dir}",
        flush=True,
    )
    print(f"Manifest saved to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

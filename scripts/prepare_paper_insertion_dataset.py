#!/usr/bin/env python3
"""Build and audit the legacy cross-instance replacement benchmark.

This module is retained for reproducibility of the superseded COCO/Fashionpedia
release and the AnyInsertion supplement.  It does **not** satisfy a normal
same-instance insertion protocol because its curated donor and target are
different identities.  New primary releases must use
``prepare_identity_paired_insertion_dataset.py``.

The canonical four-file contract used by this script is deliberately explicit:

    src_image  donor/reference image
    src_mask   exact donor instance mask
    tar_image  target scene (the masked model input is derived at inference time)
    tar_mask   exact target instance/edit mask

The curated core has three major categories and six minor categories per major.
Each minor category contains 50 cases, for 900 cases in total.  Every curated
case is a geometry-matched, same-leaf cross-instance replacement whose donor
and target come from different source images.  Identity/realism metrics are
meaningful; pixel GT is not available for this track.

AnyInsertion is converted into a separate official-benchmark supplement.  It is
never silently mixed into the curated core because its protocol and GT status
differ from the COCO/Fashionpedia-derived tracks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import imagehash
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
CROSS_MODERATE_COST = 0.6
CROSS_HARD_COST = 1.2
MASK_THRESHOLD = 127
DEFAULT_SEED = 20260723
DEFAULT_WORK_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
    "a2edit/eval/paper_insertion_benchmark_v1"
)
DEFAULT_EXCLUSION_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "paper_insertion_exclusions.json"
)

# High-resolution reference/target mask-crop review found one complete official
# four-file case whose product configuration differs: the reference is a
# two-seat sofa while the target is a three-seat sofa.  Its masks remain exact,
# so it is useful for open insertion, but the target pixels are not GT for the
# requested reference identity.  Key this override by the upstream identifiers,
# never by the generated case index (which depends on filename sort order).
ANYINSERTION_OPEN_CASES: dict[tuple[str, str], str] = {
    ("object", "5"): "reference_two_seat_sofa_target_three_seat_sofa",
}
ANYINSERTION_LABEL_OVERRIDES: dict[tuple[str, str], tuple[str, str]] = {
    ("object", "28"): ("beverage_can", "official_bottle_label_is_visually_a_can"),
    ("object", "34"): ("backpack", "official_shoe_label_is_visually_a_backpack"),
    ("object", "4"): ("side_table", "official_desk_label_is_visually_a_side_table"),
}
ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS: frozenset[tuple[str, str]] = frozenset(
    {
        *(('garment', stem) for stem in ('24', '6', '8')),
        *(('object', stem) for stem in ('19', '21')),
        *(
            ('person/simple_scene', str(stem))
            for stem in range(30)
            if stem not in {15, 27}
        ),
    }
)
ANYINSERTION_CLEAN_MANIFEST_NAME = "anyinsertion_clean67_for_current_model.jsonl"
ANYINSERTION_CLEAN_IDS_NAME = "anyinsertion_clean67_case_ids.txt"
ANYINSERTION_EXCLUSIONS_NAME = "anyinsertion_current_model_overlap_excluded33.json"
ANYINSERTION_LEGACY_OVERLAP_REVIEW = {
    "review_date": "2026-07-23",
    "review_status": "agent_assisted_visual_review_not_human_certified",
    "visual_review_report_sha256": "0c2a13cf09bdcc936457bf58007856f83db9a0c9cfc1c3afd7d829597dabef75",
    "source_manifest_sha256": "f1c1d637d69fa79408f58ece76e9bffae21f13a16f938de67c480ec316a659e7",
    "scanner_report_sha256": "da2169195158685850a10577180d70f58e0c55909aeb70595626aa08579e89a1",
}
ANYINSERTION_SOURCE_SNAPSHOT = {
    "expected_file_count": 562,
    "drive_tree_sha256": "c2303ddb287cf191907d8ef8bf6489a7dfa94632f9f33cdd339cec1b19c6f1ea",
    "content_tree_sha256": "f6c56823fb13ab53e503773e2291b6e06d1b4edf09276d2465d872b0838906e5",
}
CURATED_SOURCE_SNAPSHOT = {
    "coco_val_images": {
        "bytes": 815585330,
        "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    },
    "coco_annotations": {
        "bytes": 252907541,
        "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    },
    "fashionpedia_exact_masks": {
        "bytes": 93526258,
        "sha256": "9d2376c8acbbbb7f23c9aee2b2a1cb4d760d0538732d2bb8a617946cbedaeb31",
    },
}

MAJOR_MINOR_QUOTAS: dict[str, dict[str, int]] = {
    "object": {
        "animal": 50,
        "vehicle": 50,
        "furniture_appliance": 50,
        "food_plant": 50,
        "daily_container_electronics": 50,
        "sport_tool_toy": 50,
    },
    "person": {
        "standing_fullbody": 50,
        "walking_running": 50,
        "sitting_kneeling": 50,
        "upperbody_portrait": 50,
        "sport_dance_work": 50,
        "mild_interaction": 50,
    },
    "garment": {
        "tops": 50,
        "outerwear_vest": 50,
        "dress_onepiece": 50,
        "bottoms": 50,
        "footwear": 50,
        "bag_hat_glasses_accessory": 50,
    },
}

COCO_OBJECT_GROUPS: dict[str, set[str]] = {
    "animal": {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    },
    "vehicle": {
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    },
    "furniture_appliance": {
        "bench",
        "chair",
        "couch",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
    },
    "food_plant": {
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "potted plant",
    },
    "daily_container_electronics": {
        "bottle",
        "wine glass",
        "cup",
        "bowl",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "book",
        "clock",
        "vase",
    },
    "sport_tool_toy": {
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "fork",
        "knife",
        "spoon",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    },
}

FASHION_GROUP_IDS: dict[str, set[int]] = {
    "tops": {0, 1, 2, 3},
    "outerwear_vest": {4, 5, 9, 12},
    "dress_onepiece": {10, 11},
    "bottoms": {6, 7, 8},
    "footwear": {22, 23},
    "bag_hat_glasses_accessory": {13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26},
}

FASHION_CATEGORY_NAMES = {
    0: "shirt_blouse",
    1: "top_tshirt_sweatshirt",
    2: "sweater",
    3: "cardigan",
    4: "jacket",
    5: "vest",
    6: "pants",
    7: "shorts",
    8: "skirt",
    9: "coat",
    10: "dress",
    11: "jumpsuit",
    12: "cape",
    13: "glasses",
    14: "hat",
    15: "headband_hair_accessory",
    16: "tie",
    17: "glove",
    18: "watch",
    19: "belt",
    20: "leg_warmer",
    21: "tights_stockings",
    22: "sock",
    23: "shoe",
    24: "bag_wallet",
    25: "scarf",
    26: "umbrella",
}

PERSON_KEYWORDS: dict[str, tuple[str, ...]] = {
    "walking_running": (
        " walking ",
        " walks ",
        " running ",
        " runs ",
        " jogging ",
        " crossing ",
    ),
    "sitting_kneeling": (
        " sitting ",
        " seated ",
        " sits ",
        " kneeling ",
        " crouching ",
        " squatting ",
    ),
    "sport_dance_work": (
        " skiing ",
        " snowboard",
        " surf",
        " skateboard",
        " baseball",
        " tennis",
        " soccer",
        " football",
        " basketball",
        " volleyball",
        " frisbee",
        " dancing ",
        " dancer ",
        " working ",
        " cooking ",
        " cutting ",
        " painting ",
    ),
    "mild_interaction": (
        " holding ",
        " carrying ",
        " riding ",
        " using ",
        " playing with ",
        " talking ",
        " feeding ",
        " hugging ",
        " petting ",
        " looking at ",
    ),
    "standing_fullbody": (
        " standing ",
        " stands ",
        " posing ",
        " pose ",
    ),
}

SOURCE_LICENSES = {
    "coco2017": "COCO annotations CC BY 4.0; each Flickr image retains its original license",
    "fashionpedia": "Fashionpedia annotations/images released under CC BY 4.0 by the dataset authors",
    "anyinsertion": "Research benchmark; upstream image rights are not represented by a single uniform license",
}


@dataclass
class Candidate:
    dataset: str
    split: str
    major: str
    minor: str
    leaf_label: str
    image_id: str
    ann_id: str
    width: int
    height: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    payload: dict[str, Any] = field(repr=False)
    quality_score: float = 0.0
    caption: str = ""
    classification_evidence: str = "native_category"
    shape_profile: tuple[float, ...] = ()

    @property
    def area_ratio(self) -> float:
        return float(self.area) / float(self.width * self.height)

    @property
    def bbox_area_ratio(self) -> float:
        _, _, width, height = self.bbox_xywh
        return float(width * height) / float(self.width * self.height)

    @property
    def aspect_ratio(self) -> float:
        _, _, width, height = self.bbox_xywh
        return float(width) / max(float(height), 1.0)


def stable_fraction(*parts: Any, seed: int = DEFAULT_SEED) -> float:
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def load_exclusion_config(path: Path | None = None) -> dict[str, Any]:
    default_required = path is None
    path = DEFAULT_EXCLUSION_CONFIG if path is None else path
    if not path.is_file():
        # The repository-owned default is release policy, not an optional
        # convenience file.  Silently replacing it with empty sets would let
        # both a build and its strict audit pass after all reviewed hard and
        # paper-only exclusions disappeared.  Explicit alternate paths retain
        # the empty fallback for small library/test callers.
        if default_required or path.resolve(strict=False) == DEFAULT_EXCLUSION_CONFIG.resolve(
            strict=False
        ):
            raise FileNotFoundError(
                f"Missing required release exclusion config: {path}"
            )
        return {
            "schema_version": 1,
            "coco2017_image_ids": [],
            "fashionpedia_image_ids": [],
            "paper_avoid_coco2017_image_ids": [],
            "paper_avoid_fashionpedia_image_ids": [],
            "source_case_ids": [],
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported exclusion config schema: {path}")
    image_fields = (
        "coco2017_image_ids",
        "fashionpedia_image_ids",
        "paper_avoid_coco2017_image_ids",
        "paper_avoid_fashionpedia_image_ids",
        "v9_qc_coco2017_image_ids",
        "v9_qc_fashionpedia_image_ids",
        "v9_qc_paper_avoid_coco2017_image_ids",
        "v9_qc_paper_avoid_fashionpedia_image_ids",
    )
    for field_name in image_fields:
        values = data.get(field_name, [])
        if (
            not isinstance(values, list)
            or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9]+", value)
                for value in values
            )
            or len(values) != len(set(values))
            or values != sorted(values, key=int)
        ):
            raise ValueError(
                f"Exclusion config field must contain unique numeric strings in order: {field_name}"
            )
    for hard_fields, paper_fields in (
        (
            ("coco2017_image_ids", "v9_qc_coco2017_image_ids"),
            (
                "paper_avoid_coco2017_image_ids",
                "v9_qc_paper_avoid_coco2017_image_ids",
            ),
        ),
        (
            ("fashionpedia_image_ids", "v9_qc_fashionpedia_image_ids"),
            (
                "paper_avoid_fashionpedia_image_ids",
                "v9_qc_paper_avoid_fashionpedia_image_ids",
            ),
        ),
    ):
        hard_values = [value for field_name in hard_fields for value in data.get(field_name, [])]
        paper_values = [value for field_name in paper_fields for value in data.get(field_name, [])]
        duplicate_hard = len(hard_values) != len(set(hard_values))
        duplicate_paper = len(paper_values) != len(set(paper_values))
        if duplicate_hard or duplicate_paper:
            raise ValueError("Base and v9 QC exclusion lists must not contain duplicate IDs")
        overlap = set(hard_values).intersection(paper_values)
        if overlap:
            raise ValueError(
                f"Hard and paper-only exclusion lists overlap for {hard_fields[0]}: "
                f"{sorted(overlap, key=int)[:10]}"
            )
    source_case_ids = data.get("source_case_ids")
    if (
        not isinstance(source_case_ids, list)
        or not all(isinstance(value, str) and value for value in source_case_ids)
        or len(source_case_ids) != len(set(source_case_ids))
        or source_case_ids != sorted(source_case_ids)
    ):
        raise ValueError("Exclusion config source_case_ids must be unique and sorted")
    v9_source_case_ids = data.get("v9_qc_source_case_ids", [])
    if (
        not isinstance(v9_source_case_ids, list)
        or not all(isinstance(value, str) and value for value in v9_source_case_ids)
        or len(v9_source_case_ids) != len(set(v9_source_case_ids))
        or v9_source_case_ids != sorted(v9_source_case_ids)
        or set(v9_source_case_ids).intersection(source_case_ids)
    ):
        raise ValueError("Exclusion config v9_qc_source_case_ids must be new, unique, and sorted")
    if not isinstance(data.get("reason"), str) or not isinstance(
        data.get("visual_qc_basis"), Mapping
    ):
        raise ValueError("Exclusion config must document reason and visual_qc_basis")
    return data


def load_excluded_image_ids(
    path: Path = DEFAULT_EXCLUSION_CONFIG,
) -> set[tuple[str, str]]:
    data = load_exclusion_config(path)
    excluded: set[tuple[str, str]] = set()
    for dataset, field_name in (
        ("coco2017", "coco2017_image_ids"),
        ("fashionpedia", "fashionpedia_image_ids"),
        ("coco2017", "v9_qc_coco2017_image_ids"),
        ("fashionpedia", "v9_qc_fashionpedia_image_ids"),
    ):
        for image_id in data.get(field_name, []):
            excluded.add((dataset, str(image_id)))
    return excluded


def load_paper_avoid_image_ids(
    path: Path = DEFAULT_EXCLUSION_CONFIG,
) -> set[tuple[str, str]]:
    data = load_exclusion_config(path)
    avoided: set[tuple[str, str]] = set()
    for dataset, field_name in (
        ("coco2017", "paper_avoid_coco2017_image_ids"),
        ("fashionpedia", "paper_avoid_fashionpedia_image_ids"),
        ("coco2017", "v9_qc_paper_avoid_coco2017_image_ids"),
        ("fashionpedia", "v9_qc_paper_avoid_fashionpedia_image_ids"),
    ):
        for image_id in data.get(field_name, []):
            avoided.add((dataset, str(image_id)))
    return avoided


def slugify(value: str) -> str:
    result = []
    for char in value.lower():
        if char.isalnum():
            result.append(char)
        elif not result or result[-1] != "_":
            result.append("_")
    return "".join(result).strip("_") or "unknown"


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def find_stem_file(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        for actual_suffix in (suffix, suffix.upper()):
            path = directory / f"{stem}{actual_suffix}"
            if path.is_file():
                return path
    return None


def decode_image_value(value: Any, mode: str = "RGB") -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert(mode)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert(mode)
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if raw is not None:
            return Image.open(io.BytesIO(bytes(raw))).convert(mode)
        path = value.get("path")
        if path:
            return Image.open(path).convert(mode)
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert(mode)
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def binary_mask(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array > MASK_THRESHOLD, 255, 0).astype(np.uint8), mode="L")


def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    array = np.asarray(binary_mask(mask), dtype=np.uint8) > 0
    ys, xs = np.nonzero(array)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_area(mask: Image.Image) -> int:
    return int(np.count_nonzero(np.asarray(binary_mask(mask), dtype=np.uint8)))


def mask_shape_profile(mask: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[float, ...]:
    """Compact scale-normalized silhouette profile for donor matching."""

    crop = binary_mask(mask).crop(bbox).resize((16, 16), Image.Resampling.NEAREST)
    values = np.asarray(crop, dtype=np.float32) / 255.0
    return tuple(float(value) for value in np.concatenate((values.mean(axis=1), values.mean(axis=0))))


def crop_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    margin_ratio: float = 0.15,
) -> tuple[Image.Image, Image.Image]:
    bbox = mask_bbox(mask)
    if bbox is None:
        raise ValueError("Cannot crop an empty mask")
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    margin_x = max(8, int(round(width * margin_ratio)))
    margin_y = max(8, int(round(height * margin_ratio)))
    crop = (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    )
    return image.crop(crop).convert("RGB"), binary_mask(mask.crop(crop))


def mask_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = image.convert("RGB")
    red = Image.new("RGB", base.size, (255, 45, 45))
    alpha = binary_mask(mask).point(lambda value: 92 if value else 0)
    return Image.composite(red, base, alpha)


def decoded_pixel_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(np.asarray(rgb, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def decoded_mask_sha256(mask: Image.Image) -> str:
    binary = binary_mask(mask)
    digest = hashlib.sha256()
    digest.update(f"L:{binary.width}x{binary.height}:".encode("ascii"))
    digest.update(np.asarray(binary, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def source_image_fingerprints(image: Image.Image) -> dict[str, Any]:
    """Return full-image provenance fingerprints used by leakage scanning."""

    rgb = image.convert("RGB")
    return {
        "src_full_pixel_sha256": decoded_pixel_sha256(rgb),
        "src_full_phash": str(imagehash.phash(rgb, hash_size=8, highfreq_factor=4)),
        "src_full_dhash": str(imagehash.dhash(rgb, hash_size=8)),
        "src_full_width": rgb.width,
        "src_full_height": rgb.height,
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_dependency_versions() -> dict[str, str | None]:
    """Record the packages that affect decoding, masks, matching, or hashes."""

    result: dict[str, str | None] = {}
    for package in ("numpy", "Pillow", "pycocotools", "scipy", "pyarrow", "ImageHash"):
        try:
            result[package] = package_version(package)
        except PackageNotFoundError:  # pragma: no cover - environment provenance
            result[package] = None
    return result


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o644)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def checksum_verification_errors(
    dataset_root: Path,
    expected_rows: Sequence[tuple[str, str]],
) -> list[str]:
    checksum_path = dataset_root / "CHECKSUMS.sha256"
    if not checksum_path.is_file():
        return [f"Missing checksum manifest: {checksum_path}"]
    try:
        validate_contained_regular_file(
            dataset_root,
            checksum_path,
            "checksum manifest",
        )
    except ValueError as error:
        return [str(error)]
    errors: list[str] = []
    expected: dict[str, str] = {}
    for digest, relative_value in expected_rows:
        relative = str(relative_value)
        relative_path = Path(relative)
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative_path.as_posix() != relative
        ):
            errors.append(f"Unsafe expected checksum row: {digest!r} {relative!r}")
            continue
        if relative in expected:
            errors.append(f"Duplicate expected checksum path: {relative}")
            continue
        expected[relative] = str(digest)
    recorded: dict[str, str] = {}
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or re.fullmatch(r"[0-9a-f]{64}", parts[0]) is None:
            errors.append(f"Invalid checksum row {line_number}: {line!r}")
            continue
        digest, relative = parts
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative_path.as_posix() != relative
        ):
            errors.append(f"Unsafe checksum path at row {line_number}: {relative}")
            continue
        if relative in recorded:
            errors.append(f"Duplicate checksum path: {relative}")
            continue
        recorded[relative] = digest

    expected_paths = set(expected)
    recorded_paths = set(recorded)
    if recorded_paths != expected_paths:
        errors.append(
            "Checksum path set mismatch: "
            f"missing={sorted(expected_paths - recorded_paths)[:10]} "
            f"extra={sorted(recorded_paths - expected_paths)[:10]}"
        )
    for relative in sorted(recorded_paths.intersection(expected_paths)):
        if recorded[relative] != expected[relative]:
            errors.append(
                f"Checksum mismatch {relative}: recorded={recorded[relative]} actual={expected[relative]}"
            )
    return errors


def write_checksums_atomic(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    seen_paths: set[str] = set()
    validated_rows: list[tuple[str, str]] = []
    for digest, relative in rows:
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise ValueError(f"Invalid checksum digest for {relative!r}: {digest!r}")
        relative = str(relative)
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative_path.as_posix() != relative
        ):
            raise ValueError(f"Unsafe checksum path: {relative!r}")
        if relative in seen_paths:
            raise ValueError(f"Duplicate checksum path: {relative}")
        seen_paths.add(relative)
        validated_rows.append((str(digest), relative))
    content = "".join(
        f"{digest}  {relative}\n"
        for digest, relative in sorted(validated_rows, key=lambda item: item[1])
    )
    atomic_write_text(path, content)


def validate_safe_component(value: Any, field_name: str) -> str:
    """Return one portable path component or raise before any filesystem use."""

    component = str(value)
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or Path(component).name != component
    ):
        raise ValueError(f"Unsafe {field_name} component: {component!r}")
    return component


def validate_canonical_relative_path(value: Any, field_name: str) -> str:
    """Reject absolute, traversal, backslash, and non-normalized paths."""

    relative = str(value)
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in relative
        or path.as_posix() != relative
    ):
        raise ValueError(f"Unsafe {field_name} path: {relative!r}")
    return relative


def validate_contained_regular_file(
    root: Path,
    path: Path,
    field_name: str,
) -> Path:
    """Require a real regular file below root with no symlinked component."""

    try:
        lexical_relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} is not lexically below root: {path}") from error
    current = root
    for component in lexical_relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{field_name} contains a symlink: {current}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{field_name} is missing or escapes root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{field_name} is not a regular file: {path}")
    return resolved


def validate_contained_directory(
    root: Path,
    path: Path,
    field_name: str,
) -> Path:
    """Require a real directory below root with no symlinked component."""

    try:
        lexical_relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field_name} is not lexically below root: {path}") from error
    current = root
    for component in lexical_relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{field_name} contains a symlink: {current}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{field_name} is missing or escapes root: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{field_name} is not a directory: {path}")
    return resolved


def atomic_save_image(
    image: Image.Image,
    path: Path,
    *,
    image_format: str,
    **save_options: Any,
) -> None:
    """Atomically publish an image with release-readable permissions."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing to overwrite image symlink: {path}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            image.save(handle, format=image_format, **save_options)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o644)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def canonical_pairs_sha256(rows: Iterable[tuple[str, str]]) -> str:
    payload = json.dumps(
        sorted((str(left), str(right)) for left, right in rows),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_anyinsertion_source_snapshot(raw_root: Path) -> dict[str, Any]:
    """Verify the exact pinned Drive tree and every source file byte hash."""

    any_root = (raw_root / "anyinsertion").resolve()
    manifest_path = any_root / "google_drive_files.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pinned AnyInsertion Drive manifest: {manifest_path}")
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Invalid AnyInsertion Drive manifest: {manifest_path}")
    expected_count = int(ANYINSERTION_SOURCE_SNAPSHOT["expected_file_count"])
    if len(rows) != expected_count:
        raise ValueError(
            f"AnyInsertion source file count mismatch: actual={len(rows)} expected={expected_count}"
        )
    remote_paths: set[str] = set()
    file_ids: set[str] = set()
    tree_rows: list[tuple[str, str]] = []
    content_rows: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("AnyInsertion Drive manifest contains a non-object row")
        remote_path = str(row.get("remote_path", ""))
        file_id = str(row.get("id", ""))
        if not remote_path or not file_id:
            raise ValueError("AnyInsertion Drive manifest row is missing remote_path/id")
        if remote_path in remote_paths or file_id in file_ids:
            raise ValueError(f"Duplicate AnyInsertion source path or ID: {remote_path}")
        remote_paths.add(remote_path)
        file_ids.add(file_id)
        source_path = (any_root / remote_path).resolve()
        try:
            source_path.relative_to(any_root)
        except ValueError as error:
            raise ValueError(f"AnyInsertion source path escapes raw root: {remote_path}") from error
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing AnyInsertion source file: {source_path}")
        tree_rows.append((remote_path, file_id))
        content_rows.append((remote_path, file_sha256(source_path)))
    tree_sha256 = canonical_pairs_sha256(tree_rows)
    content_sha256 = canonical_pairs_sha256(content_rows)
    expected_tree = str(ANYINSERTION_SOURCE_SNAPSHOT["drive_tree_sha256"])
    expected_content = str(ANYINSERTION_SOURCE_SNAPSHOT["content_tree_sha256"])
    if tree_sha256 != expected_tree or content_sha256 != expected_content:
        raise ValueError(
            "AnyInsertion source snapshot mismatch: "
            f"tree={tree_sha256} expected_tree={expected_tree} "
            f"content={content_sha256} expected_content={expected_content}"
        )
    return {
        **ANYINSERTION_SOURCE_SNAPSHOT,
        "verified": True,
        "manifest": str(manifest_path),
    }


def safe_extract_zip(archive: Path, destination: Path) -> None:
    def member_matches(path: Path, info: zipfile.ZipInfo) -> bool:
        if not path.is_file() or path.stat().st_size != info.file_size:
            return False
        crc = 0
        with path.open("rb") as existing:
            for chunk in iter(lambda: existing.read(1024 * 1024), b""):
                crc = zlib.crc32(chunk, crc)
        return crc & 0xFFFFFFFF == info.CRC

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        corrupt_member = handle.testzip()
        if corrupt_member is not None:
            raise ValueError(f"Corrupt zip member in {archive}: {corrupt_member}")
        members: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in handle.infolist():
            target = (destination / info.filename).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe zip member outside destination: {info.filename}")
            members.append((info, target))
        # Resume incomplete extraction by rewriting only missing or
        # CRC-mismatched members.  A pre-existing directory alone is never
        # treated as proof that a large archive finished extracting.
        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member_matches(target, info):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_path: Path | None = None
            try:
                with handle.open(info) as source, tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    shutil.copyfileobj(source, temporary, length=1024 * 1024)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                if not member_matches(temporary_path, info):
                    raise ValueError(f"Extracted zip member failed CRC verification: {info.filename}")
                os.replace(temporary_path, target)
                os.chmod(target, 0o644)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        incomplete = [
            info.filename
            for info, target in members
            if not info.is_dir()
            and not member_matches(target, info)
        ]
        if incomplete:
            raise ValueError(
                f"Incomplete zip extraction for {archive}: {incomplete[:10]}"
            )


def extract_raw_archives(raw_root: Path) -> None:
    coco_root = raw_root / "coco"
    val_zip = coco_root / "val2017.zip"
    annotations_zip = coco_root / "annotations_trainval2017.zip"
    if not val_zip.is_file():
        raise FileNotFoundError(f"Missing COCO archive: {val_zip}")
    safe_extract_zip(val_zip, coco_root)
    if not annotations_zip.is_file():
        raise FileNotFoundError(f"Missing COCO annotations archive: {annotations_zip}")
    safe_extract_zip(annotations_zip, coco_root)


def coco_segmentation_to_mask(segmentation: Any, height: int, width: int) -> Image.Image:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("pycocotools is required to rasterize native COCO polygons") from error

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise TypeError(f"Unsupported COCO segmentation type: {type(segmentation)!r}")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return Image.fromarray(np.where(decoded, 255, 0).astype(np.uint8), mode="L")


def candidate_passes_geometry(
    width: int,
    height: int,
    bbox_xywh: Sequence[float],
    area: float,
    *,
    major: str,
    min_area_ratio: float | None = None,
    max_area_ratio: float | None = None,
    min_scaled_bbox_side: float | None = None,
    border_ratio: float = 0.0,
) -> bool:
    x, y, bbox_width, bbox_height = (float(value) for value in bbox_xywh)
    if min(width, height) < 320 or bbox_width <= 0 or bbox_height <= 0:
        return False
    area_ratio = float(area) / float(width * height)
    min_area = min_area_ratio if min_area_ratio is not None else (0.035 if major != "person" else 0.045)
    max_area = max_area_ratio if max_area_ratio is not None else (0.42 if major != "person" else 0.48)
    if not min_area <= area_ratio <= max_area:
        return False
    scaled_bbox_side = min(bbox_width, bbox_height) * 1024.0 / max(width, height)
    required_scaled_side = min_scaled_bbox_side if min_scaled_bbox_side is not None else 96.0
    if scaled_bbox_side < required_scaled_side:
        return False
    if max(bbox_width / bbox_height, bbox_height / bbox_width) > 4.5:
        return False
    if float(area) / max(bbox_width * bbox_height, 1.0) < 0.18:
        return False
    margin_x = max(2.0, width * border_ratio)
    margin_y = max(2.0, height * border_ratio)
    if x <= margin_x or y <= margin_y or x + bbox_width >= width - margin_x or y + bbox_height >= height - margin_y:
        return False
    return True


def candidate_quality(
    width: int,
    height: int,
    bbox_xywh: Sequence[float],
    area: float,
    key: str,
    seed: int,
) -> float:
    x, y, bbox_width, bbox_height = (float(value) for value in bbox_xywh)
    ratio = float(area) / float(width * height)
    target_ratio = 0.16
    area_score = max(0.0, 1.0 - abs(math.log(max(ratio, 1e-6) / target_ratio)) / math.log(8.0))
    fill = float(area) / max(bbox_width * bbox_height, 1.0)
    border = min(x, y, width - (x + bbox_width), height - (y + bbox_height)) / max(min(width, height), 1)
    size_score = min(min(bbox_width, bbox_height) / 180.0, 1.0)
    jitter = stable_fraction(key, seed=seed) * 0.02
    return 0.42 * area_score + 0.23 * min(fill / 0.7, 1.0) + 0.25 * size_score + 0.10 * min(border / 0.08, 1.0) + jitter


def coco_minor_for_category(name: str) -> str | None:
    for minor, names in COCO_OBJECT_GROUPS.items():
        if name in names:
            return minor
    return None


def normalize_caption(texts: Iterable[str]) -> str:
    return " " + " ".join(text.lower().strip() for text in texts) + " "


def visible_keypoints(keypoints: Sequence[float] | None) -> dict[int, tuple[float, float]]:
    if not keypoints:
        return {}
    result: dict[int, tuple[float, float]] = {}
    for index in range(0, len(keypoints), 3):
        x, y, visibility = keypoints[index : index + 3]
        if visibility >= 2:
            result[index // 3] = (float(x), float(y))
    return result


def person_minor_scores(
    caption: str,
    bbox_xywh: Sequence[float],
    keypoints: Sequence[float] | None,
    nearby_object_count: int,
) -> dict[str, float]:
    caption = normalize_caption([caption])
    _, _, width, height = (float(value) for value in bbox_xywh)
    points = visible_keypoints(keypoints)
    shoulders = int(5 in points) + int(6 in points)
    hips = int(11 in points) + int(12 in points)
    knees = int(13 in points) + int(14 in points)
    ankles = int(15 in points) + int(16 in points)
    fullbody = shoulders + hips + knees + ankles
    aspect = height / max(width, 1.0)

    scores = {minor: 0.0 for minor in MAJOR_MINOR_QUOTAS["person"]}
    for minor, keywords in PERSON_KEYWORDS.items():
        scores[minor] += 4.0 * sum(keyword in caption for keyword in keywords)

    scores["standing_fullbody"] += 0.55 * fullbody + 1.2 * (ankles == 2) + min(aspect / 2.0, 1.2)
    scores["walking_running"] += 0.35 * fullbody + 0.8 * (ankles == 2)
    scores["sitting_kneeling"] += 0.5 * (hips >= 1 and knees >= 1) + 0.4 * (aspect < 1.8)
    scores["upperbody_portrait"] += 1.5 * (shoulders >= 1) + 1.0 * (knees == 0) + 0.8 * (ankles == 0)
    scores["upperbody_portrait"] += 0.5 * (0.7 <= aspect <= 2.2)
    scores["sport_dance_work"] += 0.2 * fullbody
    scores["mild_interaction"] += min(float(nearby_object_count), 3.0) * 0.35

    # Strongly contradictory caption evidence should not win through geometry.
    sitting_words = any(word in caption for word in PERSON_KEYWORDS["sitting_kneeling"])
    sport_words = any(word in caption for word in PERSON_KEYWORDS["sport_dance_work"])
    if sitting_words:
        scores["standing_fullbody"] -= 3.0
        scores["walking_running"] -= 2.0
    if sport_words:
        scores["mild_interaction"] -= 1.0
    return scores


PERSON_SITTING_WORDS = (
    "sitting",
    "seated",
    "sits",
    "kneeling",
    "kneels",
    "crouching",
    "squatting",
)
PERSON_WALKING_WORDS = (
    "walking",
    "walks",
    "running",
    "runs",
    "jogging",
    "jogs",
    "crossing",
)
PERSON_RIDING_WORDS = (
    "riding",
    "rides",
    "rider",
    "driving",
    "drives",
)
PERSON_SPORT_WORK_WORDS = (
    "ski",
    "skiing",
    "skier",
    "snowboard",
    "snowboarding",
    "surf",
    "surfing",
    "skateboard",
    "skateboarding",
    "baseball",
    "tennis",
    "soccer",
    "football",
    "basketball",
    "volleyball",
    "frisbee",
    "dance",
    "dancing",
    "dancer",
    "work",
    "working",
    "cooking",
    "cutting",
    "painting",
)
PERSON_INTERACTION_WORDS = (
    "holding",
    "holds",
    "carrying",
    "carries",
    "using",
    "uses",
    "talking",
    "phone",
    "cellphone",
    "reading",
    "drinking",
    "feeding",
    "hugging",
    "petting",
    "looking at",
)
PERSON_MINOR_SCENE_WORDS = (
    "child",
    "children",
    "kid",
    "kids",
    "toddler",
    "toddlers",
    "baby",
    "babies",
    "little boy",
    "little girl",
    "young boy",
    "young girl",
)
PAPER_AVOID_REASON_CODES = frozenset(
    {
        "agent_visual_qc_image_flag",
        "caption_indicates_possible_minor_in_scene",
    }
)


def caption_has_any(caption: str, words: Sequence[str]) -> bool:
    lowered = caption.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in words)


def candidate_paper_avoid_reasons(
    candidate: Candidate,
    configured_image_ids: set[tuple[str, str]],
) -> list[str]:
    reasons: list[str] = []
    if (candidate.dataset, candidate.image_id) in configured_image_ids:
        reasons.append("agent_visual_qc_image_flag")
    # Keep possible minors in the metric track, but never auto-promote their
    # scenes for a paper figure without a separate human privacy review.
    if candidate.major == "person" and caption_has_any(candidate.caption, PERSON_MINOR_SCENE_WORDS):
        reasons.append("caption_indicates_possible_minor_in_scene")
    return reasons


def keypoint_angle(
    first: tuple[float, float],
    middle: tuple[float, float],
    last: tuple[float, float],
) -> float | None:
    left = (first[0] - middle[0], first[1] - middle[1])
    right = (last[0] - middle[0], last[1] - middle[1])
    denominator = math.hypot(*left) * math.hypot(*right)
    if denominator <= 1e-8:
        return None
    cosine = (left[0] * right[0] + left[1] * right[1]) / denominator
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def person_minor_eligibility(
    caption: str,
    bbox_xywh: Sequence[float],
    area_ratio: float,
    keypoints: Sequence[float] | None,
) -> dict[str, bool]:
    """Return hard, instance-level gates for the six person strata.

    COCO captions describe the whole image, so a caption keyword alone is not
    enough: pose strata also require visible-keypoint geometry.  The caller
    separately restricts candidates to a dominant person in the scene.
    """

    _, _, bbox_width, bbox_height = (float(value) for value in bbox_xywh)
    points = visible_keypoints(keypoints)
    visible_count = len(points)
    face_count = sum(index in points for index in range(5))
    shoulder_count = sum(index in points for index in (5, 6))
    wrist_count = sum(index in points for index in (9, 10))
    hip_count = sum(index in points for index in (11, 12))
    knee_count = sum(index in points for index in (13, 14))
    ankle_count = sum(index in points for index in (15, 16))
    aspect = bbox_height / max(bbox_width, 1.0)

    complete_chains = [
        chain
        for chain in ((11, 13, 15), (12, 14, 16))
        if all(index in points for index in chain)
    ]
    knee_angles = [
        keypoint_angle(points[hip], points[knee], points[ankle])
        for hip, knee, ankle in complete_chains
    ]
    knee_angles = [angle for angle in knee_angles if angle is not None]
    ankle_separation = (
        abs(points[15][0] - points[16][0]) / max(bbox_height, 1.0)
        if 15 in points and 16 in points
        else 0.0
    )
    weak_motion = ankle_separation >= 0.08 or any(angle <= 155.0 for angle in knee_angles)
    strong_motion = ankle_separation >= 0.15 or any(angle <= 130.0 for angle in knee_angles)

    sitting_words = caption_has_any(caption, PERSON_SITTING_WORDS)
    walking_words = caption_has_any(caption, PERSON_WALKING_WORDS)
    riding_words = caption_has_any(caption, PERSON_RIDING_WORDS)
    sport_words = caption_has_any(caption, PERSON_SPORT_WORK_WORDS)
    interaction_words = caption_has_any(caption, PERSON_INTERACTION_WORDS)

    strict_standing = (
        visible_count >= 8
        and shoulder_count >= 1
        and hip_count >= 1
        and knee_count >= 1
        and ankle_count >= 1
        and aspect >= 1.4
        # COCO often annotates the visible hip/knee/ankle on alternating
        # sides, leaving no complete three-joint chain even for an otherwise
        # clear full-body pose.  Require a straight leg when a complete chain
        # exists, while allowing the cross-side evidence to stand on its own.
        and (not knee_angles or any(angle >= 150.0 for angle in knee_angles))
        and not (sitting_words or walking_words or riding_words or sport_words)
    )
    explicit_standing = caption_has_any(
        caption,
        ("standing", "stands", "stand", "posing", "poses", "posed"),
    )
    # Keep a low-priority reserve for this scarce stratum.  Static sports/work
    # scenes are still valid full-body standing poses, but the strict pool is
    # preferred so they cannot dominate the taxonomy.  The caller applies a
    # large ranking penalty and records the fallback evidence explicitly.
    standing_fallback = (
        not strict_standing
        and explicit_standing
        and not (sitting_words or walking_words)
        and visible_count >= 6
        and shoulder_count >= 1
        and hip_count >= 1
        and knee_count + ankle_count >= 2
        and aspect >= 1.25
    )
    standing = strict_standing or standing_fallback
    walking_pose = (
        visible_count >= 6
        and bool(complete_chains)
        and ((walking_words and weak_motion) or strong_motion)
        and not (sitting_words or riding_words)
    )
    explicit_walking = walking_words and not sitting_words and visible_count >= 3
    walking_fallback = walking_pose and not explicit_walking
    walking = explicit_walking or walking_fallback
    sitting = (
        sitting_words
        and visible_count >= 5
        and hip_count >= 1
        and knee_count >= 1
        and (any(angle <= 150.0 for angle in knee_angles) or aspect <= 1.8)
    )
    upperbody = (
        area_ratio >= 0.08
        and face_count >= 2
        and shoulder_count >= 1
        and not complete_chains
        and knee_count + ankle_count <= 2
        and visible_count >= 5
        and 0.65 <= aspect <= 2.7
        and not (sitting_words or walking_words or sport_words)
    )
    sport = sport_words and visible_count >= 5
    mild_interaction = (
        interaction_words
        and visible_count >= 5
        and wrist_count >= 1
        and not sport_words
    )
    return {
        "standing_fullbody": standing,
        "walking_running": walking,
        "sitting_kneeling": sitting,
        "upperbody_portrait": upperbody,
        "sport_dance_work": sport,
        "mild_interaction": mild_interaction,
        "_standing_fullbody_fallback": standing_fallback,
        "_walking_running_fallback": walking_fallback,
    }


def load_coco_candidates(raw_root: Path, seed: int) -> dict[str, list[Candidate]]:
    coco_root = raw_root / "coco"
    annotation_root = coco_root / "annotations"
    instances_path = annotation_root / "instances_val2017.json"
    captions_path = annotation_root / "captions_val2017.json"
    keypoints_path = annotation_root / "person_keypoints_val2017.json"
    if not instances_path.is_file():
        raise FileNotFoundError(f"Missing extracted COCO annotations: {instances_path}")
    for required_path in (captions_path, keypoints_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Person selection requires COCO annotation file: {required_path}")
    if not (coco_root / "val2017").is_dir():
        raise FileNotFoundError(f"Missing extracted COCO images: {coco_root / 'val2017'}")

    instances_data = json.loads(instances_path.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in instances_data["images"]}
    categories = {int(row["id"]): row["name"] for row in instances_data["categories"]}
    excluded_image_ids = load_excluded_image_ids()

    captions: dict[int, list[str]] = defaultdict(list)
    captions_data = json.loads(captions_path.read_text(encoding="utf-8"))
    for row in captions_data.get("annotations", []):
        captions[int(row["image_id"])].append(str(row.get("caption", "")))

    keypoint_by_ann: dict[int, Sequence[float]] = {}
    keypoints_data = json.loads(keypoints_path.read_text(encoding="utf-8"))
    for row in keypoints_data.get("annotations", []):
        values = row.get("keypoints") or []
        keypoint_by_ann[int(row["id"])] = values

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in instances_data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    pools: dict[str, list[Candidate]] = defaultdict(list)
    person_rows: list[tuple[Candidate, dict[str, float], dict[str, bool]]] = []
    for annotation in instances_data["annotations"]:
        if annotation.get("iscrowd") or not annotation.get("segmentation"):
            continue
        image_id = int(annotation["image_id"])
        if ("coco2017", str(image_id)) in excluded_image_ids:
            continue
        image_info = images[image_id]
        category_name = categories[int(annotation["category_id"])]
        major = "person" if category_name == "person" else "object"
        if major == "object":
            minor = coco_minor_for_category(category_name)
            if minor is None:
                continue
        else:
            minor = ""
        width = int(image_info["width"])
        height = int(image_info["height"])
        bbox = tuple(float(value) for value in annotation["bbox"])
        area = float(annotation["area"])
        if not candidate_passes_geometry(width, height, bbox, area, major=major):
            continue
        image_path = coco_root / "val2017" / image_info["file_name"]
        if not image_path.is_file():
            continue
        native_mask = coco_segmentation_to_mask(annotation["segmentation"], height, width)
        native_bbox = mask_bbox(native_mask)
        if native_bbox is None:
            continue
        left, top, right, bottom = native_bbox
        bbox = (float(left), float(top), float(right - left), float(bottom - top))
        area = float(mask_area(native_mask))
        if not candidate_passes_geometry(width, height, bbox, area, major=major):
            continue
        caption = " ".join(captions.get(image_id, []))
        candidate = Candidate(
            dataset="coco2017",
            split="val2017",
            major=major,
            minor=minor,
            leaf_label=category_name,
            image_id=str(image_id),
            ann_id=str(annotation["id"]),
            width=width,
            height=height,
            bbox_xywh=bbox,
            area=area,
            payload={
                "image_path": image_path,
                "segmentation": annotation["segmentation"],
            },
            caption=caption,
            quality_score=candidate_quality(width, height, bbox, area, str(annotation["id"]), seed),
            shape_profile=mask_shape_profile(native_mask, native_bbox),
        )
        if major == "object":
            pools[f"object/{minor}"].append(candidate)
        else:
            nearby = sum(
                int(other["id"] != annotation["id"])
                for other in annotations_by_image[image_id]
                if not other.get("iscrowd")
            )
            other_person_areas = [
                float(other["area"])
                for other in annotations_by_image[image_id]
                if (
                    other["id"] != annotation["id"]
                    and not other.get("iscrowd")
                    and categories[int(other["category_id"])] == "person"
                )
            ]
            # Scene captions are far more likely to describe the dominant
            # person.  This also removes crowd/by-stander cases that are harder
            # to display and whose action label is fundamentally ambiguous.
            if other_person_areas and max(other_person_areas) > 0.5 * area:
                continue
            keypoints = keypoint_by_ann.get(int(annotation["id"]))
            scores = person_minor_scores(
                caption,
                bbox,
                keypoints,
                nearby,
            )
            eligibility = person_minor_eligibility(
                caption,
                bbox,
                candidate.area_ratio,
                keypoints,
            )
            candidate.payload["keypoints"] = list(keypoints or [])
            person_rows.append((candidate, scores, eligibility))

    # A single person instance has scores for every person minor.  We clone it
    # into each ranked pool; target-image uniqueness is enforced during global
    # selection so an image cannot leak across action/pose strata.
    for candidate, scores, eligibility in person_rows:
        for minor, evidence_score in scores.items():
            if not eligibility[minor]:
                continue
            clone = Candidate(**{**asdict(candidate), "minor": minor})
            clone.payload = candidate.payload
            standing_fallback = bool(
                minor == "standing_fullbody"
                and eligibility.get("_standing_fullbody_fallback", False)
            )
            walking_fallback = bool(
                minor == "walking_running"
                and eligibility.get("_walking_running_fallback", False)
            )
            clone.quality_score = (
                candidate.quality_score
                + 0.12 * evidence_score
                - (4.0 if standing_fallback or walking_fallback else 0.0)
            )
            clone.classification_evidence = (
                f"hard_dominant_instance_pose;caption_keypoint_score={evidence_score:.3f}"
                + (";standing_explicit_caption_fallback" if standing_fallback else "")
                + (";walking_keypoint_pose_fallback" if walking_fallback else "")
            )
            pools[f"person/{minor}"].append(clone)

    for values in pools.values():
        values.sort(
            key=lambda item: (
                -item.quality_score,
                stable_fraction(item.dataset, item.image_id, item.ann_id, seed=seed),
            )
        )
    return pools


def _parallel_field(objects: Mapping[str, Any], names: Sequence[str]) -> list[Any] | None:
    for name in names:
        value = objects.get(name)
        if value is not None:
            return list(value)
    return None


def load_fashion_candidates(raw_root: Path, seed: int) -> tuple[dict[str, list[Candidate]], Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("pyarrow is required to read Fashionpedia parquet") from error

    parquet_path = raw_root / "fashionpedia" / "test_with_masks.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"Missing exact-mask Fashionpedia parquet: {parquet_path}. "
            "The detection-datasets/fashionpedia parquet is not sufficient because it contains bboxes only."
        )
    table = parquet.read_table(parquet_path)
    required = {
        "image",
        "object_masks",
        "object_bboxes",
        "object_category_ids",
        "object_attribute_ids",
    }
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"Fashionpedia exact-mask parquet is missing columns: {sorted(missing)}")

    pools: dict[str, list[Candidate]] = defaultdict(list)
    excluded_image_ids = load_excluded_image_ids()
    for row_index, row in enumerate(table.to_pylist()):
        image_value = row["image"]
        width = int(row.get("image_width") or row.get("width") or 0)
        height = int(row.get("image_height") or row.get("height") or 0)
        if width <= 0 or height <= 0:
            image = decode_image_value(image_value)
            width, height = image.size
        masks = list(row.get("object_masks") or [])
        bboxes = list(row.get("object_bboxes") or [])
        category_ids = list(row.get("object_category_ids") or [])
        object_attribute_ids = list(row.get("object_attribute_ids") or [])
        if not (len(masks) == len(bboxes) == len(category_ids) == len(object_attribute_ids)):
            continue
        image_id = str(row.get("image_id", row_index))
        if ("fashionpedia", image_id) in excluded_image_ids:
            continue
        for object_index, (mask_value, bbox_value, category_value, attribute_values) in enumerate(
            zip(masks, bboxes, category_ids, object_attribute_ids)
        ):
            category_id = int(category_value)
            minor = next((name for name, ids in FASHION_GROUP_IDS.items() if category_id in ids), None)
            if minor is None:
                continue
            mask = binary_mask(decode_image_value(mask_value, mode="L"))
            if mask.size != (width, height):
                raise ValueError(
                    f"Fashionpedia mask/image geometry mismatch at row={row_index} "
                    f"object={object_index}: mask={mask.size} image={(width, height)}"
                )
            bbox_native = mask_bbox(mask)
            if bbox_native is None:
                continue
            left, top, right, bottom = bbox_native
            bbox_xywh = (float(left), float(top), float(right - left), float(bottom - top))
            area = float(mask_area(mask))
            small_item = minor in {"footwear", "bag_hat_glasses_accessory"}
            min_scaled_side = 80.0 if minor == "footwear" else (64.0 if small_item else 96.0)
            if not candidate_passes_geometry(
                width,
                height,
                bbox_xywh,
                area,
                major="garment",
                min_area_ratio=0.005 if small_item else 0.03,
                max_area_ratio=0.25 if small_item else 0.45,
                min_scaled_bbox_side=min_scaled_side,
                border_ratio=0.01,
            ):
                continue
            ann_id = f"{image_id}:{object_index}"
            candidate = Candidate(
                dataset="fashionpedia",
                split="val2020_hf_exact_masks",
                major="garment",
                minor=minor,
                leaf_label=FASHION_CATEGORY_NAMES.get(category_id, f"category_{category_id}"),
                image_id=image_id,
                ann_id=ann_id,
                width=width,
                height=height,
                bbox_xywh=bbox_xywh,
                area=area,
                payload={
                    "row_index": row_index,
                    "object_index": object_index,
                    "mask": mask,
                    "attribute_ids": sorted({int(value) for value in (attribute_values or [])}),
                },
                quality_score=candidate_quality(width, height, bbox_xywh, area, ann_id, seed),
                shape_profile=mask_shape_profile(mask, bbox_native),
            )
            pools[f"garment/{minor}"].append(candidate)

    for values in pools.values():
        values.sort(
            key=lambda item: (
                -item.quality_score,
                stable_fraction(item.dataset, item.image_id, item.ann_id, seed=seed),
            )
        )
    return pools, table


def load_candidate_image(candidate: Candidate, fashion_table: Any | None = None) -> Image.Image:
    if candidate.dataset == "coco2017":
        return Image.open(candidate.payload["image_path"]).convert("RGB")
    if candidate.dataset == "fashionpedia":
        if fashion_table is None:
            raise ValueError("Fashionpedia table is required")
        row = fashion_table.slice(candidate.payload["row_index"], 1).to_pylist()[0]
        return decode_image_value(row["image"], mode="RGB")
    raise ValueError(f"Unsupported candidate dataset: {candidate.dataset}")


def load_candidate_mask(candidate: Candidate) -> Image.Image:
    if candidate.dataset == "coco2017":
        return coco_segmentation_to_mask(candidate.payload["segmentation"], candidate.height, candidate.width)
    if candidate.dataset == "fashionpedia":
        return binary_mask(candidate.payload["mask"])
    raise ValueError(f"Unsupported candidate dataset: {candidate.dataset}")


def select_targets(
    pool: Sequence[Candidate],
    count: int,
    used_target_images: set[tuple[str, str]],
    forbidden_images: set[tuple[str, str]],
    *,
    max_per_leaf: int,
    deprioritized_images: set[tuple[str, str]] | None = None,
) -> list[Candidate]:
    deprioritized_images = deprioritized_images or set()
    # Keep the loader's quality order inside each partition, but consume
    # agent-flagged paper-only images only when a stratum would otherwise be
    # infeasible.  They remain valid metric candidates and are therefore not
    # conflated with hard annotation exclusions.
    ordered_pool = [
        candidate
        for priority in (False, True)
        for candidate in pool
        if ((candidate.dataset, candidate.image_id) in deprioritized_images) == priority
    ]
    # Every case uses a cross-image donor, so a target is only feasible when
    # its leaf appears in at least two distinct pool images, and the number
    # of selected targets per leaf can never exceed that image count: each
    # case needs a unique donor image different from its own target image.
    leaf_images: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for candidate in pool:
        leaf_images[candidate.leaf_label].add((candidate.dataset, candidate.image_id))
    leaf_donor_capacity = {
        leaf: len(images) for leaf, images in leaf_images.items() if len(images) >= 2
    }
    selected: list[Candidate] = []
    leaf_counts: Counter[str] = Counter()
    for candidate in ordered_pool:
        image_key = (candidate.dataset, candidate.image_id)
        if image_key in used_target_images or image_key in forbidden_images:
            continue
        donor_capacity = leaf_donor_capacity.get(candidate.leaf_label)
        if donor_capacity is None:
            continue
        if leaf_counts[candidate.leaf_label] >= min(max_per_leaf, donor_capacity):
            continue
        selected.append(candidate)
        used_target_images.add(image_key)
        leaf_counts[candidate.leaf_label] += 1
        if len(selected) == count:
            return selected

    # Relax the quality-driven leaf cap only after every available leaf has
    # had a chance (needed for groups such as dress/onepiece with only two
    # native labels), but never beyond the donor-image capacity: relaxing
    # past the distinct-image count would make the cross-image donor
    # assignment infeasible by construction.
    for candidate in ordered_pool:
        image_key = (candidate.dataset, candidate.image_id)
        if image_key in used_target_images or image_key in forbidden_images:
            continue
        donor_capacity = leaf_donor_capacity.get(candidate.leaf_label)
        if donor_capacity is None or leaf_counts[candidate.leaf_label] >= donor_capacity:
            continue
        selected.append(candidate)
        used_target_images.add(image_key)
        leaf_counts[candidate.leaf_label] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"Only selected {len(selected)}/{count} unique targets from a pool of {len(pool)}")


def select_person_targets_globally(
    pools: Mapping[str, Sequence[Candidate]],
    minor_quotas: Mapping[str, int],
    used_target_images: set[tuple[str, str]],
    paper_avoid_image_ids: set[tuple[str, str]],
    seed: int,
) -> dict[str, list[Candidate]]:
    """Allocate all person strata jointly under global target-image uniqueness."""

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("scipy is required for global person-target matching") from error

    candidates_by_minor: dict[str, dict[tuple[str, str], Candidate]] = {}
    all_image_keys: set[tuple[str, str]] = set()
    for minor in minor_quotas:
        best_by_image: dict[tuple[str, str], Candidate] = {}
        for candidate in pools[f"person/{minor}"]:
            image_key = (candidate.dataset, candidate.image_id)
            if image_key in used_target_images:
                continue
            previous = best_by_image.get(image_key)
            if previous is None or candidate.quality_score > previous.quality_score:
                best_by_image[image_key] = candidate
        if len(best_by_image) < int(minor_quotas[minor]):
            raise ValueError(
                f"Only {len(best_by_image)}/{minor_quotas[minor]} available person targets "
                f"for {minor} before global allocation"
            )
        candidates_by_minor[minor] = best_by_image
        all_image_keys.update(best_by_image)

    slots = [
        (minor, slot_index)
        for minor, quota in minor_quotas.items()
        for slot_index in range(int(quota))
    ]
    image_keys = sorted(all_image_keys)
    if len(image_keys) < len(slots):
        raise ValueError(
            f"Only {len(image_keys)} unique person images for {len(slots)} target slots"
        )
    image_columns = {key: index for index, key in enumerate(image_keys)}
    unavailable_cost = 1e12
    cost_matrix = np.full(
        (len(slots), len(image_keys)),
        unavailable_cost,
        dtype=np.float64,
    )
    for row_index, (minor, slot_index) in enumerate(slots):
        for image_key, candidate in candidates_by_minor[minor].items():
            paper_priority = int(bool(candidate_paper_avoid_reasons(candidate, paper_avoid_image_ids)))
            fallback_priority = int("_fallback" in candidate.classification_evidence)
            # Paper avoidance is lexicographically dominant; inside that
            # partition minimize semantic fallback use before maximizing the
            # native quality/evidence score.  A tiny stable slot-specific term
            # makes ties reproducible.
            cost_matrix[row_index, image_columns[image_key]] = (
                1000.0 * paper_priority
                + 100.0 * fallback_priority
                - candidate.quality_score
                + 1e-6 * stable_fraction(minor, slot_index, candidate.ann_id, seed=seed)
            )

    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    if len(row_indices) != len(slots):
        raise ValueError(
            f"Global person target assignment covered {len(row_indices)}/{len(slots)} slots"
        )
    result: dict[str, list[Candidate]] = defaultdict(list)
    selected_image_keys: list[tuple[str, str]] = []
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        if cost_matrix[row_index, column_index] >= unavailable_cost / 2:
            minor, _ = slots[row_index]
            raise ValueError(f"No feasible globally unique person target for {minor}")
        minor, _ = slots[row_index]
        image_key = image_keys[column_index]
        candidate = candidates_by_minor[minor][image_key]
        result[minor].append(candidate)
        selected_image_keys.append(image_key)

    for minor, quota in minor_quotas.items():
        values = result[minor]
        values.sort(key=lambda candidate: (-candidate.quality_score, candidate.ann_id))
        if len(values) != int(quota):
            raise ValueError(f"Person target quota mismatch for {minor}: {len(values)}/{quota}")
    # Commit caller-visible state only after every assignment edge and quota
    # has been validated.  Infeasible Hall components must leave the input set
    # unchanged so a caller can retry with relaxed constraints.
    used_target_images.update(selected_image_keys)
    return dict(result)


def choose_cross_donor(
    target: Candidate,
    pool: Sequence[Candidate],
    donor_use_counts: Counter[tuple[str, str]],
    seed: int,
) -> Candidate:
    choices = []
    for donor in pool:
        if donor.leaf_label != target.leaf_label or donor.image_id == target.image_id:
            continue
        donor_key = (donor.dataset, donor.image_id)
        if donor_use_counts[donor_key] >= 1:
            continue
        match_score = cross_donor_match_score(target, donor, seed)
        choices.append((match_score, donor))
    if not choices:
        raise ValueError(
            f"No cross-instance donor for target {target.dataset}/{target.ann_id} ({target.leaf_label})"
        )
    donor = min(choices, key=lambda item: item[0])[1]
    donor_use_counts[(donor.dataset, donor.image_id)] += 1
    return donor


def cross_donor_match_score(target: Candidate, donor: Candidate, seed: int) -> float:
    aspect_gap = abs(math.log(max(donor.aspect_ratio, 1e-6) / max(target.aspect_ratio, 1e-6)))
    area_gap = abs(math.log(max(donor.area_ratio, 1e-6) / max(target.area_ratio, 1e-6)))
    match_score = 1.4 * aspect_gap + 0.35 * area_gap - 0.25 * donor.quality_score
    if target.shape_profile and len(target.shape_profile) == len(donor.shape_profile):
        profile_gap = float(
            np.mean(
                np.abs(
                    np.asarray(target.shape_profile, dtype=np.float32)
                    - np.asarray(donor.shape_profile, dtype=np.float32)
                )
            )
        )
        match_score += (1.1 if target.major == "garment" else 0.3) * profile_gap
    if target.major == "garment":
        target_attributes = {int(value) for value in target.payload.get("attribute_ids", [])}
        donor_attributes = {int(value) for value in donor.payload.get("attribute_ids", [])}
        attribute_union = target_attributes | donor_attributes
        if attribute_union:
            attribute_gap = 1.0 - len(target_attributes & donor_attributes) / len(attribute_union)
            match_score += 1.25 * attribute_gap
    if target.major == "person":
        match_score += 1.8 * person_pose_gap(donor, target)
    return match_score + 0.02 * stable_fraction(target.ann_id, donor.ann_id, seed=seed)


def assign_cross_donors(
    targets: Sequence[Candidate],
    pool: Sequence[Candidate],
    donor_use_counts: Counter[tuple[str, str]],
    seed: int,
    forbidden_images: set[tuple[str, str]] | None = None,
    deprioritized_images: set[tuple[str, str]] | None = None,
    paper_avoid_image_ids: set[tuple[str, str]] | None = None,
) -> list[Candidate]:
    return assign_cross_donors_from_pools(
        targets,
        [pool] * len(targets),
        donor_use_counts,
        seed,
        forbidden_images_by_target=[forbidden_images or set()] * len(targets),
        deprioritized_images=deprioritized_images,
        paper_avoid_image_ids=paper_avoid_image_ids,
    )


def assign_cross_donors_from_pools(
    targets: Sequence[Candidate],
    pools: Sequence[Sequence[Candidate]],
    donor_use_counts: Counter[tuple[str, str]],
    seed: int,
    forbidden_images_by_target: Sequence[set[tuple[str, str]]] | None = None,
    deprioritized_images: set[tuple[str, str]] | None = None,
    paper_avoid_image_ids: set[tuple[str, str]] | None = None,
) -> list[Candidate]:
    """Find a global minimum-cost, unique-image donor matching.

    Connected components of the target/donor graph are solved independently.
    This keeps the assignment matrices compact while preserving global source
    uniqueness wherever image candidates overlap across strata.  Unlike a
    greedy or pairwise-swap matcher, the solver can make multi-way exchanges
    needed for coherent pose and garment-silhouette matches.
    """

    if len(pools) != len(targets):
        raise ValueError(f"Donor pool count mismatch: pools={len(pools)} targets={len(targets)}")
    if forbidden_images_by_target is None:
        forbidden_images_by_target = [set()] * len(targets)
    if len(forbidden_images_by_target) != len(targets):
        raise ValueError(
            "Forbidden-image set count must equal the number of cross targets"
        )
    # Severe pair mismatch, paper-QC risk, and target-image reuse are separate
    # lexicographic tiers.  First avoid turning an otherwise moderate benchmark
    # into a hard geometry-gap case; then minimize paper-avoid donors, then
    # target reuse, and finally raw geometry/appearance cost.  This permits a
    # paper-flagged metric-only donor when every clean alternative is grossly
    # mismatched, while showcase selection still excludes the flagged case.
    # Dynamic caption flags (for example, possible minors) are evaluated here,
    # before the donor is selected.
    deprioritized_images = deprioritized_images or set()
    paper_avoid_image_ids = paper_avoid_image_ids or set()
    options: list[list[tuple[int, int, int, float, tuple[str, str], Candidate]]] = []
    for target, pool, forbidden_images in zip(targets, pools, forbidden_images_by_target):
        best_by_image: dict[tuple[str, str], tuple[float, Candidate]] = {}
        target_key = (target.dataset, target.image_id)
        for donor in pool:
            if donor.leaf_label != target.leaf_label:
                continue
            donor_key = (donor.dataset, donor.image_id)
            if (
                donor_key == target_key
                or donor_key in forbidden_images
                or donor_use_counts[donor_key] >= 1
            ):
                continue
            score = cross_donor_match_score(target, donor, seed)
            if not math.isfinite(score):
                raise ValueError(
                    "Cross-instance donor score must be finite: "
                    f"target={target.dataset}/{target.ann_id} "
                    f"donor={donor.dataset}/{donor.ann_id} score={score}"
                )
            previous = best_by_image.get(donor_key)
            if previous is None or score < previous[0]:
                best_by_image[donor_key] = (score, donor)
        target_options = sorted(
            (
                int(max(score, 0.0) > CROSS_HARD_COST),
                int(bool(candidate_paper_avoid_reasons(donor, paper_avoid_image_ids))),
                int(donor_key in deprioritized_images),
                score,
                donor_key,
                donor,
            )
            for donor_key, (score, donor) in best_by_image.items()
        )
        if not target_options:
            raise ValueError(
                f"No cross-instance donor candidates for {target.dataset}/{target.ann_id} "
                f"({target.leaf_label})"
            )
        options.append(target_options)

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError("scipy is required for global cross-instance matching") from error

    donor_to_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for target_index, target_options in enumerate(options):
        for _, _, _, _, donor_key, _ in target_options:
            donor_to_targets[donor_key].add(target_index)

    selected: dict[int, Candidate] = {}
    remaining_targets = set(range(len(targets)))
    while remaining_targets:
        seed_target = min(remaining_targets)
        component_targets: set[int] = set()
        component_donors: set[tuple[str, str]] = set()
        target_stack = [seed_target]
        while target_stack:
            target_index = target_stack.pop()
            if target_index in component_targets:
                continue
            component_targets.add(target_index)
            for _, _, _, _, donor_key, _ in options[target_index]:
                if donor_key in component_donors:
                    continue
                component_donors.add(donor_key)
                target_stack.extend(donor_to_targets[donor_key] - component_targets)

        target_indices = sorted(component_targets)
        donor_keys = sorted(component_donors)
        if len(donor_keys) < len(target_indices):
            raise ValueError(
                "No complete unique-donor matching for component: "
                f"targets={len(target_indices)} donors={len(donor_keys)}"
            )
        donor_columns = {key: index for index, key in enumerate(donor_keys)}
        cost_matrix = np.full(
            (len(target_indices), len(donor_keys)),
            np.inf,
            dtype=np.float64,
        )
        component_scores = [
            score
            for target_index in target_indices
            for _, _, _, score, _, _ in options[target_index]
        ]
        score_span = max(component_scores) - min(component_scores)
        raw_cost_bound = len(target_indices) * score_span + 1.0
        reuse_penalty = raw_cost_bound
        paper_avoid_penalty = (len(target_indices) + 1) * raw_cost_bound
        hard_gap_penalty = (len(target_indices) + 1) * paper_avoid_penalty
        candidate_matrix: dict[tuple[int, int], Candidate] = {}
        for row_index, target_index in enumerate(target_indices):
            for (
                hard_gap_priority,
                paper_priority,
                reuse_priority,
                score,
                donor_key,
                donor,
            ) in options[target_index]:
                column_index = donor_columns[donor_key]
                cost = (
                    hard_gap_penalty * hard_gap_priority
                    + paper_avoid_penalty * paper_priority
                    + reuse_penalty * reuse_priority
                    + score
                )
                if cost < cost_matrix[row_index, column_index]:
                    cost_matrix[row_index, column_index] = cost
                    candidate_matrix[(row_index, column_index)] = donor

        try:
            row_indices, column_indices = linear_sum_assignment(cost_matrix)
        except ValueError as error:
            raise ValueError(
                "No complete unique-donor matching for component: "
                f"targets={len(target_indices)} donors={len(donor_keys)}"
            ) from error
        if len(row_indices) != len(target_indices):
            raise ValueError(
                "Cross-instance assignment did not cover every target in component: "
                f"assigned={len(row_indices)} targets={len(target_indices)}"
        )
        for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
            if (row_index, column_index) not in candidate_matrix:
                raise ValueError(
                    "No complete unique-donor matching for component: "
                    f"targets={len(target_indices)} donors={len(donor_keys)}"
                )
            selected[target_indices[row_index]] = candidate_matrix[(row_index, column_index)]
        remaining_targets.difference_update(component_targets)

    result = [selected[index] for index in range(len(targets))]
    for donor in result:
        donor_use_counts[(donor.dataset, donor.image_id)] += 1
    return result


def person_pose_gap(left: Candidate, right: Candidate) -> float:
    def normalized(candidate: Candidate) -> dict[int, tuple[float, float]]:
        x, y, width, height = candidate.bbox_xywh
        points = visible_keypoints(candidate.payload.get("keypoints"))
        return {
            index: ((point[0] - x) / max(width, 1.0), (point[1] - y) / max(height, 1.0))
            for index, point in points.items()
        }

    left_points = normalized(left)
    right_points = normalized(right)
    common = sorted(set(left_points).intersection(right_points))
    if len(common) < 4:
        return 0.75
    distances = [
        math.hypot(
            left_points[index][0] - right_points[index][0],
            left_points[index][1] - right_points[index][1],
        )
        for index in common
    ]
    visibility_gap = len(set(left_points).symmetric_difference(right_points)) / 17.0
    return float(np.mean(distances)) + 0.25 * visibility_gap


def save_png(image: Image.Image, path: Path, *, mask: bool = False) -> None:
    value = binary_mask(image) if mask else image.convert("RGB")
    atomic_save_image(value, path, image_format="PNG", optimize=True)


def write_case(
    output_root: Path,
    case_id: str,
    donor: Candidate,
    target: Candidate,
    protocol: str,
    fashion_table: Any | None,
    seed: int,
    paper_avoid_image_ids: set[tuple[str, str]],
) -> dict[str, Any]:
    donor_image = load_candidate_image(donor, fashion_table)
    donor_mask = load_candidate_mask(donor)
    target_image = load_candidate_image(target, fashion_table)
    target_mask = load_candidate_mask(target)
    if donor_image.size != donor_mask.size:
        raise ValueError(f"Donor image/mask mismatch for {case_id}: {donor_image.size} vs {donor_mask.size}")
    if target_image.size != target_mask.size:
        raise ValueError(f"Target image/mask mismatch for {case_id}: {target_image.size} vs {target_mask.size}")
    source_crop, source_mask_crop = crop_image_and_mask(donor_image, donor_mask)
    full_source_fingerprints = source_image_fingerprints(donor_image)

    major_root = output_root / target.major
    paths = {
        "src_image": major_root / "src_image" / f"{case_id}.png",
        "src_mask": major_root / "src_mask" / f"{case_id}.png",
        "tar_image": major_root / "tar_image" / f"{case_id}.png",
        "tar_mask": major_root / "tar_mask" / f"{case_id}.png",
    }
    save_png(source_crop, paths["src_image"])
    save_png(source_mask_crop, paths["src_mask"], mask=True)
    save_png(target_image, paths["tar_image"])
    save_png(target_mask, paths["tar_mask"], mask=True)

    src_area = mask_area(source_mask_crop) / float(source_mask_crop.width * source_mask_crop.height)
    tar_area = mask_area(target_mask) / float(target_mask.width * target_mask.height)
    same_instance = donor.dataset == target.dataset and donor.image_id == target.image_id and donor.ann_id == target.ann_id
    cross_match_cost = (
        cross_donor_match_score(target, donor, seed)
        if protocol == "cross_instance_replace"
        else 0.0
    )
    positive_match_cost = max(cross_match_cost, 0.0)
    # A paper example must have both a clean target and a coherent donor/target
    # geometry.  Match cost therefore receives full weight instead of being a
    # weak tie-breaker behind target quality.
    showcase_score = target.quality_score - positive_match_cost
    challenge_flags = (
        ["large_cross_match_gap"]
        if protocol == "cross_instance_replace" and positive_match_cost > CROSS_HARD_COST
        else []
    )
    if challenge_flags:
        difficulty = "hard"
    elif protocol == "cross_instance_replace" and positive_match_cost > CROSS_MODERATE_COST:
        difficulty = "moderate"
    else:
        difficulty = "easy" if target.quality_score >= 0.68 else "moderate"
    paper_avoid_reasons = {
        role: reasons
        for role, reasons in (
            ("src", candidate_paper_avoid_reasons(donor, paper_avoid_image_ids)),
            ("tar", candidate_paper_avoid_reasons(target, paper_avoid_image_ids)),
        )
        if reasons
    }
    paper_avoid_roles = list(paper_avoid_reasons)
    relative = {name: str(path.relative_to(output_root)) for name, path in paths.items()}
    return {
        "case_id": case_id,
        "major": target.major,
        "minor": target.minor,
        "leaf_label": target.leaf_label,
        "src_leaf_label": donor.leaf_label,
        "src_source_dataset": donor.dataset,
        "source_dataset": target.dataset,
        "source_split": target.split,
        "protocol": protocol,
        "same_identity": same_instance,
        "gt_available": bool(protocol == "paired_reconstruction"),
        "mask_origin_src": "native_instance",
        "mask_origin_tar": "native_instance",
        "difficulty": difficulty,
        "challenge_flags": challenge_flags,
        "src_image_id": donor.image_id,
        "src_ann_id": donor.ann_id,
        "tar_image_id": target.image_id,
        "tar_ann_id": target.ann_id,
        "src_attribute_ids": list(donor.payload.get("attribute_ids", [])),
        "tar_attribute_ids": list(target.payload.get("attribute_ids", [])),
        "src_area_ratio": round(src_area, 8),
        "tar_area_ratio": round(tar_area, 8),
        "selection_score": round(target.quality_score, 8),
        "cross_match_cost": round(cross_match_cost, 8),
        "showcase_score": round(showcase_score, 8),
        "classification_evidence": target.classification_evidence,
        "license_note": SOURCE_LICENSES[target.dataset],
        "auto_quality_pass": True,
        "agent_paper_avoid": bool(paper_avoid_roles),
        "agent_paper_avoid_roles": paper_avoid_roles,
        "agent_paper_avoid_reasons": paper_avoid_reasons,
        "manual_qc_status": "pending",
        "paper_ok": False,
        "showcase_candidate": False,
        "src_pixel_sha256": decoded_pixel_sha256(source_crop),
        "tar_pixel_sha256": decoded_pixel_sha256(target_image),
        "src_mask_pixel_sha256": decoded_mask_sha256(source_mask_crop),
        "tar_mask_pixel_sha256": decoded_mask_sha256(target_mask),
        **full_source_fingerprints,
        **relative,
    }


def build_curated_core(raw_root: Path, output_root: Path, seed: int) -> list[dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty curated output: {output_root}. "
            "Choose a fresh --output-root or move the previous build aside."
        )
    coco_pools = load_coco_candidates(raw_root, seed)
    fashion_pools, fashion_table = load_fashion_candidates(raw_root, seed)
    pools = {**coco_pools, **fashion_pools}
    paper_avoid_image_ids = load_paper_avoid_image_ids()
    used_target_images: set[tuple[str, str]] = set()
    # Reserve the truly scarce pose pools before selecting COCO object cases.
    # Otherwise an object in the same photo can consume one of only ~50 valid
    # standing/sitting scenes before the person strata are planned.
    protected_person_images: set[tuple[str, str]] = set()
    for minor, quota in MAJOR_MINOR_QUOTAS["person"].items():
        person_pool = pools[f"person/{minor}"]
        unique_images = {(row.dataset, row.image_id) for row in person_pool}
        preferred_images = {
            (row.dataset, row.image_id)
            for row in person_pool
            if "_fallback" not in row.classification_evidence
        }
        if len(preferred_images) < quota * 2:
            # Every case now consumes both a unique target image and a unique
            # cross-image donor, so reserve the entire feasible pool whenever
            # it cannot obviously supply 2x the quota.  Otherwise object
            # selection can consume scarce person scenes before the person
            # strata are planned, making a feasible allocation appear short.
            protected_person_images.update(unique_images)

    plans: list[dict[str, Any]] = []
    person_targets_by_minor: dict[str, list[Candidate]] | None = None
    for major, minor_quotas in MAJOR_MINOR_QUOTAS.items():
        if major == "person":
            person_targets_by_minor = select_person_targets_globally(
                pools,
                minor_quotas,
                used_target_images,
                paper_avoid_image_ids,
                seed,
            )
        for minor, quota in minor_quotas.items():
            pool_key = f"{major}/{minor}"
            pool = pools.get(pool_key, [])
            max_per_leaf = 10 if major == "object" else (30 if major == "garment" else quota)
            if major == "person":
                if person_targets_by_minor is None:  # pragma: no cover - defensive
                    raise AssertionError("Person targets were not globally planned")
                targets = list(person_targets_by_minor[minor])
            else:
                targets = select_targets(
                    pool,
                    quota,
                    used_target_images,
                    protected_person_images if major == "object" else set(),
                    max_per_leaf=max_per_leaf,
                    deprioritized_images=paper_avoid_image_ids,
                )
            # All cases are cross-instance replace: the donor always comes
            # from a different image than the target, so the reference and
            # the edited region never share identical shape and structure.
            # There is no same-image paired reconstruction track.
            plans.append(
                {
                    "major": major,
                    "minor": minor,
                    "pool": pool,
                    "targets": targets,
                }
            )

    # Plan every target before assigning donors so the global matcher can
    # enforce cross-image feasibility and donor-role uniqueness before any
    # files are materialized.
    donor_use_counts: Counter[tuple[str, str]] = Counter()
    all_cross_targets: list[Candidate] = []
    all_cross_pools: list[Sequence[Candidate]] = []
    for plan in plans:
        plan["cross_start"] = len(all_cross_targets)
        all_cross_targets.extend(plan["targets"])
        all_cross_pools.extend([plan["pool"]] * len(plan["targets"]))

    all_cross_donors = assign_cross_donors_from_pools(
        all_cross_targets,
        all_cross_pools,
        donor_use_counts,
        seed,
        deprioritized_images=used_target_images,
        paper_avoid_image_ids=paper_avoid_image_ids,
    )
    for plan in plans:
        start = plan["cross_start"]
        plan["donors"] = all_cross_donors[start : start + len(plan["targets"])]

    # Do not materialize even an empty output tree until the complete target
    # and donor plan is known to be feasible.
    output_root.mkdir(parents=True, exist_ok=True)
    for major in MAJOR_MINOR_QUOTAS:
        for directory in ("src_image", "src_mask", "tar_image", "tar_mask"):
            (output_root / major / directory).mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for plan in plans:
        major = plan["major"]
        minor = plan["minor"]
        for index, (target, donor) in enumerate(zip(plan["targets"], plan["donors"])):
            case_id = f"{major[:3]}_{slugify(minor)}_{index:03d}"
            records.append(
                write_case(
                    output_root,
                    case_id,
                    donor,
                    target,
                    "cross_instance_replace",
                    fashion_table,
                    seed,
                    paper_avoid_image_ids,
                )
            )

    freeze_showcase_candidates(records, per_minor=10)
    write_manifests(output_root, records)
    write_source_record(raw_root, output_root, seed)
    generate_previews(output_root, records)
    return records


def freeze_showcase_candidates(records: list[dict[str, Any]], per_minor: int) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["showcase_candidate"] = False
        groups[(record["major"], record["minor"])].append(record)
    for values in groups.values():
        # Every case is cross-instance replace (the donor always comes from
        # a different image than the target), so the qualitative queue draws
        # up to per_minor representatives from that single protocol and
        # selects leaf labels round-robin for diversity.
        protocol_quotas = {
            "cross_instance_replace": per_minor,
        }
        selected: list[dict[str, Any]] = []
        for protocol, quota in protocol_quotas.items():
            protocol_values_all = [
                row
                for row in values
                if row["protocol"] == protocol and not row.get("agent_paper_avoid", False)
            ]
            # Large-gap cases remain useful for challenge metrics, but should
            # not enter the automatic qualitative pool when coherent options
            # exist in the same stratum/protocol.
            protocol_values = [
                row
                for row in protocol_values_all
                if "large_cross_match_gap" not in row.get("challenge_flags", [])
            ]
            protocol_selected = select_diverse_showcases(protocol_values, quota)
            # Never promote a paper-avoided case merely to satisfy a visual
            # queue quota, and never promote a large-gap challenge case into
            # the automatic paper queue.  Some narrow strata genuinely have
            # fewer than five clean/coherent candidates.  The strict audit
            # records that shortfall as a warning while the 900-case metric
            # protocol remains complete and unchanged.
            selected.extend(protocol_selected)
        for row in selected:
            row["showcase_candidate"] = True


def refresh_paper_avoid_metadata(
    records: Sequence[dict[str, Any]],
    configured_image_ids: set[tuple[str, str]],
) -> None:
    """Reapply paper-only image flags without changing canonical assignments.

    Agent visual QC can discover watermark, privacy, or composition concerns
    after a candidate build has completed.  Those images remain valid metric
    cases, so updating the manifest metadata and qualitative queue is safer
    than silently resampling the benchmark.  Hard semantic/mask exclusions
    still require a fresh build and are deliberately not handled here.

    Caption-derived possible-minor flags cannot be reconstructed from the
    compact manifest, so valid non-config reasons already recorded per role
    are retained verbatim.
    """

    for record in records:
        existing = record.get("agent_paper_avoid_reasons", {})
        if not isinstance(existing, Mapping):
            raise ValueError(
                f"{record.get('case_id', '<unknown>')}: invalid paper-avoid reasons"
            )
        role_keys = {
            "src": (
                str(record["src_source_dataset"]),
                str(record["src_image_id"]),
            ),
            "tar": (
                str(record["source_dataset"]),
                str(record["tar_image_id"]),
            ),
        }
        refreshed: dict[str, list[str]] = {}
        for role, image_key in role_keys.items():
            previous = existing.get(role, [])
            if not isinstance(previous, list) or not all(
                isinstance(reason, str) for reason in previous
            ):
                raise ValueError(
                    f"{record.get('case_id', '<unknown>')}: invalid {role} paper-avoid reasons"
                )
            unknown = set(previous).difference(PAPER_AVOID_REASON_CODES)
            if unknown:
                raise ValueError(
                    f"{record.get('case_id', '<unknown>')}: unknown paper-avoid reasons "
                    f"for {role}: {sorted(unknown)}"
                )
            reasons: list[str] = []
            if image_key in configured_image_ids:
                reasons.append("agent_visual_qc_image_flag")
            if "caption_indicates_possible_minor_in_scene" in previous:
                reasons.append("caption_indicates_possible_minor_in_scene")
            if reasons:
                refreshed[role] = reasons
        record["agent_paper_avoid_reasons"] = refreshed
        record["agent_paper_avoid_roles"] = list(refreshed)
        record["agent_paper_avoid"] = bool(refreshed)


def select_diverse_showcases(
    records: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_leaf[str(record["leaf_label"])].append(record)
    for values in by_leaf.values():
        values.sort(
            key=lambda row: (
                -float(row.get("showcase_score", row["selection_score"])),
                str(row["case_id"]),
            )
        )

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        available = [
            (leaf, values)
            for leaf, values in by_leaf.items()
            if values
        ]
        if not available:
            break
        # Take at most one sample from each leaf per round.  Higher-quality
        # leaves win only when the requested quota is smaller than the number
        # of available labels.
        available.sort(
            key=lambda item: (
                -float(
                    item[1][0].get(
                        "showcase_score",
                        item[1][0]["selection_score"],
                    )
                ),
                item[0],
            )
        )
        for _, values in available:
            if len(selected) >= count:
                break
            selected.append(values.pop(0))
    return selected


def write_manifests(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    primary_jsonl = "".join(
        json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    fieldnames = list(records[0].keys()) if records else []
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        row = dict(record)
        for key in (
            "challenge_flags",
            "agent_paper_avoid_roles",
            "agent_paper_avoid_reasons",
            "src_attribute_ids",
            "tar_attribute_ids",
        ):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        writer.writerow(row)
    rendered = {
        manifest_root / "paperinsert_900.jsonl": primary_jsonl,
        manifest_root / "paperinsert_900.csv": csv_buffer.getvalue(),
    }
    for major in MAJOR_MINOR_QUOTAS:
        major_records = [record for record in records if record["major"] == major]
        rendered[manifest_root / f"{major}.jsonl"] = "".join(
            json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n"
            for record in major_records
        )
    # Render every authoritative view before replacing any one of them.  Each
    # replace is atomic; the checksum-backed strict audit catches an interrupted
    # multi-file refresh instead of silently accepting mixed manifest versions.
    for path, content in rendered.items():
        atomic_write_text(path, content)


def write_source_record(raw_root: Path, output_root: Path, seed: int) -> None:
    source_files = {
        "coco_val_images": raw_root / "coco" / "val2017.zip",
        "coco_annotations": raw_root / "coco" / "annotations_trainval2017.zip",
        "fashionpedia_exact_masks": raw_root / "fashionpedia" / "test_with_masks.parquet",
    }
    dependency_versions = runtime_dependency_versions()
    generator_path = Path(__file__).resolve()
    data = {
        "dataset_name": "PaperInsert-900",
        "schema_version": 3,
        "selection_seed": seed,
        "curation_status": "automatic selection complete; visual QC pending",
        "generator": {
            "path": str(generator_path),
            "sha256": file_sha256(generator_path),
            "python_version": sys.version.split()[0],
            "dependency_versions": dependency_versions,
            "command": [
                sys.executable,
                str(generator_path),
                "build",
                "--raw-root",
                str(raw_root.resolve()),
                "--output-root",
                str(output_root.resolve()),
                "--seed",
                str(seed),
            ],
        },
        "quotas": MAJOR_MINOR_QUOTAS,
        "protocol_policy": {
            "per_minor": {
                "cross_instance_replace": 50,
            },
            "cross_gt_available": False,
            "donor_target_same_image": "forbidden",
            "source_image_reuse_cap": 1,
            "target_image_reuse_cap": 1,
        },
        "semantics": {
            "src_image": "donor/reference image crop",
            "src_mask": "exact native donor instance mask",
            "tar_image": "target scene; inference derives the masked control using tar_mask",
            "tar_mask": "exact native target instance/edit mask",
        },
        "exclusions": {
            "path": str(DEFAULT_EXCLUSION_CONFIG),
            "sha256": (
                file_sha256(DEFAULT_EXCLUSION_CONFIG)
                if DEFAULT_EXCLUSION_CONFIG.is_file()
                else None
            ),
            "excluded_image_counts": dict(
                sorted(
                    Counter(dataset for dataset, _ in load_excluded_image_ids()).items()
                )
            ),
            "paper_avoid_image_counts": dict(
                sorted(
                    Counter(dataset for dataset, _ in load_paper_avoid_image_ids()).items()
                )
            ),
        },
        "sources": {},
    }
    for name, path in source_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned curated source: {path}")
        actual_size = path.stat().st_size
        actual_sha256 = file_sha256(path)
        expected_source = CURATED_SOURCE_SNAPSHOT[name]
        if (
            actual_size != expected_source["bytes"]
            or actual_sha256 != expected_source["sha256"]
        ):
            raise ValueError(
                f"Pinned curated source mismatch {name}: "
                f"bytes={actual_size} expected_bytes={expected_source['bytes']} "
                f"sha256={actual_sha256} expected_sha256={expected_source['sha256']}"
            )
        data["sources"][name] = {
            "path": str(path),
            "bytes": actual_size,
            "sha256": actual_sha256,
        }
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        report_root / "sources.json",
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=(245, 245, 245))


def prepare_contained_output_directory(output_root: Path, relative: str) -> Path:
    """Create a child directory without following a pre-planted symlink."""

    root_resolved = output_root.resolve()
    relative = validate_canonical_relative_path(relative, "output directory")
    current = output_root
    for component in Path(relative).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Refusing symlinked output directory: {current}")
    current.mkdir(parents=True, exist_ok=True)
    try:
        current.resolve().relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"Output directory escapes dataset root: {current}") from error
    return current


def canonical_record_asset(
    output_root: Path,
    record: Mapping[str, Any],
    path_key: str,
) -> Path:
    """Resolve one exact canonical case asset and keep it inside the release."""

    case_id = validate_safe_component(record.get("case_id", ""), "case_id")
    major = validate_safe_component(record.get("major", ""), "major")
    relative = validate_canonical_relative_path(record.get(path_key, ""), path_key)
    expected = f"{major}/{path_key}/{case_id}.png"
    if relative != expected:
        raise ValueError(
            f"{case_id}: non-canonical {path_key}: actual={relative} expected={expected}"
        )
    path = output_root / relative
    return validate_contained_regular_file(
        output_root,
        path,
        f"{case_id} canonical asset {relative}",
    )


def generate_previews(output_root: Path, records: Sequence[Mapping[str, Any]], per_minor: int = 10) -> None:
    preview_root = prepare_contained_output_directory(output_root, "previews")
    font = _load_font(16)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["showcase_candidate"]:
            groups[(str(record["major"]), str(record["minor"]))].append(record)
    for (major, minor), values in groups.items():
        values = sorted(values, key=lambda row: str(row["case_id"]))[:per_minor]
        cell_width, cell_height = 512, 216
        canvas = Image.new("RGB", (cell_width * 2, cell_height * len(values)), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for row_index, record in enumerate(values):
            src_image = Image.open(canonical_record_asset(output_root, record, "src_image")).convert("RGB")
            src_mask = Image.open(canonical_record_asset(output_root, record, "src_mask")).convert("L")
            tar_image = Image.open(canonical_record_asset(output_root, record, "tar_image")).convert("RGB")
            tar_mask = Image.open(canonical_record_asset(output_root, record, "tar_mask")).convert("L")
            donor_panel = fit_panel(mask_overlay(src_image, src_mask), (cell_width, cell_height - 24))
            target_panel = fit_panel(mask_overlay(tar_image, tar_mask), (cell_width, cell_height - 24))
            y = row_index * cell_height
            canvas.paste(donor_panel, (0, y + 24))
            canvas.paste(target_panel, (cell_width, y + 24))
            title = f"{record['case_id']} | {record['leaf_label']} | {record['protocol']}"
            draw.text((6, y + 3), title, fill=(20, 20, 20), font=font)
        filename = f"{slugify(major)}__{slugify(minor)}.jpg"
        atomic_save_image(
            canvas,
            preview_root / filename,
            image_format="JPEG",
            quality=92,
            subsampling=0,
        )


def generate_qc_contact_sheets(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    per_sheet: int = 10,
) -> dict[str, Any]:
    """Materialize deterministic overlays for visual review of every case."""

    if per_sheet <= 0:
        raise ValueError("per_sheet must be positive")
    sheet_root = prepare_contained_output_directory(
        output_root,
        "reports/qc_contact_sheets",
    )
    if sheet_root.exists() and any(sheet_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty QC sheet directory: {sheet_root}")
    font = _load_font(16)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["major"]), str(record["minor"]))].append(record)

    sheet_rows: list[dict[str, Any]] = []
    cell_width, cell_height = 512, 216
    for (major, minor), values in sorted(groups.items()):
        values = sorted(values, key=lambda row: str(row["case_id"]))
        for sheet_index, start in enumerate(range(0, len(values), per_sheet)):
            chunk = values[start : start + per_sheet]
            canvas = Image.new(
                "RGB",
                (cell_width * 2, cell_height * len(chunk)),
                (255, 255, 255),
            )
            draw = ImageDraw.Draw(canvas)
            for row_index, record in enumerate(chunk):
                src_image = Image.open(canonical_record_asset(output_root, record, "src_image")).convert("RGB")
                src_mask = Image.open(canonical_record_asset(output_root, record, "src_mask")).convert("L")
                tar_image = Image.open(canonical_record_asset(output_root, record, "tar_image")).convert("RGB")
                tar_mask = Image.open(canonical_record_asset(output_root, record, "tar_mask")).convert("L")
                donor_panel = fit_panel(mask_overlay(src_image, src_mask), (cell_width, cell_height - 24))
                target_panel = fit_panel(mask_overlay(tar_image, tar_mask), (cell_width, cell_height - 24))
                y = row_index * cell_height
                canvas.paste(donor_panel, (0, y + 24))
                canvas.paste(target_panel, (cell_width, y + 24))
                title = (
                    f"{record['case_id']} | {record.get('leaf_label', '')} | "
                    f"{record.get('protocol', '')}"
                )
                draw.text((6, y + 3), title, fill=(20, 20, 20), font=font)
            filename = f"{slugify(major)}__{slugify(minor)}__{sheet_index:02d}.jpg"
            sheet_path = sheet_root / filename
            atomic_save_image(
                canvas,
                sheet_path,
                image_format="JPEG",
                quality=92,
                subsampling=0,
            )
            sheet_rows.append(
                {
                    "path": str(sheet_path.relative_to(output_root)),
                    "sha256": file_sha256(sheet_path),
                    "major": major,
                    "minor": minor,
                    "case_ids": [str(record["case_id"]) for record in chunk],
                }
            )

    report = {
        "schema_version": 1,
        "review_artifact_only": True,
        "review_status": "not_human_reviewed",
        "record_count": len(records),
        "sheet_count": len(sheet_rows),
        "per_sheet": per_sheet,
        "sheets": sheet_rows,
    }
    atomic_write_text(
        output_root / "reports" / "qc_contact_sheets.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def discover_anyinsertion_groups(raw_root: Path) -> Iterator[tuple[str, Path]]:
    any_root = raw_root / "anyinsertion"
    if not any_root.is_dir():
        return
    for ref_dir in sorted(any_root.rglob("ref_image")):
        parent = ref_dir.parent
        if not (parent / "ref_mask").is_dir() or not (parent / "tar_image").is_dir() or not (parent / "tar_mask").is_dir():
            continue
        lowered = [part.lower() for part in parent.relative_to(any_root).parts]
        if any("garment" in part or "cloth" in part for part in lowered):
            major = "garment"
        elif any("person" in part for part in lowered):
            major = "person"
        elif any("object" in part for part in lowered):
            major = "object"
        else:
            continue
        yield major, parent


def convert_anyinsertion(
    raw_root: Path,
    output_root: Path,
    *,
    verify_source_snapshot: bool = True,
) -> list[dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty AnyInsertion output: {output_root}")
    source_snapshot = (
        validate_anyinsertion_source_snapshot(raw_root)
        if verify_source_snapshot
        else {**ANYINSERTION_SOURCE_SNAPSHOT, "verified": False, "test_fixture_bypass": True}
    )
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for major, group_root in discover_anyinsertion_groups(raw_root):
        labels: dict[str, str] = {}
        label_path = group_root / "label.json"
        if label_path.is_file():
            loaded_labels = json.loads(label_path.read_text(encoding="utf-8"))
            if isinstance(loaded_labels, dict):
                labels = {str(key): str(value) for key, value in loaded_labels.items()}
        for ref_path in image_files(group_root / "ref_image"):
            stem = ref_path.stem
            original_group = str(group_root.relative_to(raw_root / "anyinsertion"))
            ref_mask_path = find_stem_file(group_root / "ref_mask", stem)
            tar_path = find_stem_file(group_root / "tar_image", stem)
            tar_mask_path = find_stem_file(group_root / "tar_mask", stem)
            if None in (ref_mask_path, tar_path, tar_mask_path):
                continue
            ref_image = Image.open(ref_path).convert("RGB")
            ref_mask = binary_mask(Image.open(ref_mask_path))
            tar_image = Image.open(tar_path).convert("RGB")
            tar_mask = binary_mask(Image.open(tar_mask_path))
            if ref_image.size != ref_mask.size or tar_image.size != tar_mask.size:
                continue
            if mask_area(ref_mask) == 0 or mask_area(tar_mask) == 0:
                continue
            index = counters[major]
            counters[major] += 1
            relative_group = slugify("_".join(group_root.relative_to(raw_root / "anyinsertion").parts))
            case_id = f"ai_{major[:3]}_{relative_group}_{index:03d}"
            major_root = output_root / major
            paths = {
                "src_image": major_root / "src_image" / f"{case_id}.png",
                "src_mask": major_root / "src_mask" / f"{case_id}.png",
                "tar_image": major_root / "tar_image" / f"{case_id}.png",
                "tar_mask": major_root / "tar_mask" / f"{case_id}.png",
            }
            save_png(ref_image, paths["src_image"])
            save_png(ref_mask, paths["src_mask"], mask=True)
            save_png(tar_image, paths["tar_image"])
            save_png(tar_mask, paths["tar_mask"], mask=True)
            relative = {name: str(path.relative_to(output_root)) for name, path in paths.items()}
            identity_override = ANYINSERTION_OPEN_CASES.get((original_group, stem))
            official_label = labels.get(
                ref_path.name,
                "person" if major == "person" else "unlabeled",
            )
            label_override = ANYINSERTION_LABEL_OVERRIDES.get((original_group, stem))
            paired_gt = identity_override is None
            clean_source_available = find_stem_file(group_root / "src_image", stem) is not None
            records.append(
                {
                    "case_id": case_id,
                    "major": major,
                    "minor": relative_group,
                    "leaf_label": label_override[0] if label_override else official_label,
                    "original_leaf_label": official_label,
                    "label_override_reason": label_override[1] if label_override else None,
                    "source_dataset": "anyinsertion",
                    "source_split": "official_heldout_mask_test",
                    "protocol": "official_anyinsertion",
                    # A complete target image/mask only proves that exact target
                    # pixels exist; pixel GT also requires the reference and
                    # masked target to depict the same concrete identity.
                    "gt_available": paired_gt,
                    "case_type": "paired_gt" if paired_gt else "open_insertion",
                    "same_identity": paired_gt,
                    "identity_review_status": (
                        "verified_same_instance_visual"
                        if paired_gt
                        else "verified_different_instance_visual"
                    ),
                    "identity_override_reason": identity_override,
                    # Some official groups also contain a clean pre-edit scene
                    # named src_image.  It is useful provenance, but is not our
                    # canonical src_image (which is the reference/donor image)
                    # and does not determine GT availability.
                    "official_clean_source_available": clean_source_available,
                    "mask_origin_src": "official_native",
                    "mask_origin_tar": "official_native",
                    "license_note": SOURCE_LICENSES["anyinsertion"],
                    "original_group": original_group,
                    "original_stem": stem,
                    "src_pixel_sha256": decoded_pixel_sha256(ref_image),
                    "tar_pixel_sha256": decoded_pixel_sha256(tar_image),
                    "src_mask_pixel_sha256": decoded_mask_sha256(ref_mask),
                    "tar_mask_pixel_sha256": decoded_mask_sha256(tar_mask),
                    **relative,
                }
            )
    if not records:
        raise ValueError(f"No complete AnyInsertion samples found below {raw_root / 'anyinsertion'}")
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        manifest_root / "anyinsertion_official.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
    )
    admitted_groups = sorted(
        {
            str(group_root.relative_to(raw_root / "anyinsertion"))
            for _, group_root in discover_anyinsertion_groups(raw_root)
        }
    )
    excluded_groups = []
    any_root = raw_root / "anyinsertion"
    for ref_dir in sorted(any_root.rglob("ref_image")):
        group_root = ref_dir.parent
        relative_group = str(group_root.relative_to(any_root))
        if relative_group in admitted_groups:
            continue
        missing = [
            name
            for name in ("ref_mask", "tar_image", "tar_mask")
            if not (group_root / name).is_dir()
        ]
        excluded_groups.append(
            {
                "group": relative_group,
                "reason": "missing_exact_four_file_contract",
                "missing_directories": missing,
                "reference_count": len(image_files(ref_dir)),
            }
        )
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    conversion_report = {
        "schema_version": 3,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
            "python_version": sys.version.split()[0],
            "dependency_versions": runtime_dependency_versions(),
        },
        "record_count": len(records),
        "counts_major": dict(sorted(Counter(row["major"] for row in records).items())),
        "counts_case_type": dict(sorted(Counter(row["case_type"] for row in records).items())),
        "official_clean_source_count": sum(
            bool(row["official_clean_source_available"]) for row in records
        ),
        "identity_review_method": "agent_assisted_high_resolution_reference_target_mask_crop_review",
        "source_snapshot": source_snapshot,
        "identity_overrides": [
            {"original_group": group, "original_stem": stem, "reason": reason}
            for (group, stem), reason in sorted(ANYINSERTION_OPEN_CASES.items())
        ],
        "label_overrides": [
            {
                "original_group": group,
                "original_stem": stem,
                "corrected_label": corrected,
                "reason": reason,
            }
            for (group, stem), (corrected, reason) in sorted(ANYINSERTION_LABEL_OVERRIDES.items())
        ],
        "admitted_groups": admitted_groups,
        "excluded_groups": excluded_groups,
        "admission_rule": (
            "official ref_image/ref_mask/tar_image/tar_mask all exist with matching "
            "geometry and non-empty exact binary masks"
        ),
    }
    atomic_write_text(
        report_root / "conversion.json",
        json.dumps(conversion_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL row must be an object: {path}:{line_number}"
                )
            records.append(record)
        return records


def write_anyinsertion_clean_split(output_root: Path) -> list[dict[str, Any]]:
    """Materialize the current-model leakage-clean protocol as authoritative files."""

    manifest_root = output_root / "manifests"
    primary_path = manifest_root / "anyinsertion_official.jsonl"
    if not primary_path.is_file():
        raise FileNotFoundError(f"Missing AnyInsertion primary manifest: {primary_path}")
    records = read_jsonl(primary_path)
    if len(records) != 100:
        raise ValueError(
            f"AnyInsertion clean split requires the official 100 rows, found {len(records)}"
        )
    upstream_owners: dict[tuple[str, str], str] = {}
    case_ids: set[str] = set()
    for record in records:
        case_id = str(record.get("case_id", ""))
        upstream_key = (
            str(record.get("original_group", "")),
            str(record.get("original_stem", "")),
        )
        if not case_id or case_id in case_ids:
            raise ValueError(f"Missing or duplicate AnyInsertion case_id: {case_id!r}")
        if upstream_key in upstream_owners:
            raise ValueError(
                f"Duplicate AnyInsertion upstream key {upstream_key}: "
                f"{upstream_owners[upstream_key]} and {case_id}"
            )
        case_ids.add(case_id)
        upstream_owners[upstream_key] = case_id

    missing_exclusions = ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS.difference(
        upstream_owners
    )
    if missing_exclusions:
        raise ValueError(
            f"Current-model AnyInsertion exclusions are absent: {sorted(missing_exclusions)}"
        )
    excluded_records = [
        record
        for record in records
        if (
            str(record["original_group"]),
            str(record["original_stem"]),
        )
        in ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS
    ]
    clean_records = [record for record in records if record not in excluded_records]
    if len(excluded_records) != 33 or len(clean_records) != 67:
        raise ValueError(
            f"AnyInsertion clean split cardinality mismatch: "
            f"excluded={len(excluded_records)} clean={len(clean_records)}"
        )
    expected_major_counts = {"garment": 27, "object": 38, "person": 2}
    actual_major_counts = dict(
        sorted(Counter(str(record["major"]) for record in clean_records).items())
    )
    expected_case_types = {"open_insertion": 1, "paired_gt": 66}
    actual_case_types = dict(
        sorted(Counter(str(record["case_type"]) for record in clean_records).items())
    )
    if actual_major_counts != expected_major_counts or actual_case_types != expected_case_types:
        raise ValueError(
            "AnyInsertion clean split statistics mismatch: "
            f"major={actual_major_counts} case_types={actual_case_types}"
        )

    clean_path = manifest_root / ANYINSERTION_CLEAN_MANIFEST_NAME
    ids_path = manifest_root / ANYINSERTION_CLEAN_IDS_NAME
    exclusions_path = manifest_root / ANYINSERTION_EXCLUSIONS_NAME
    clean_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in clean_records
    )
    included_ids_text = "".join(f"{record['case_id']}\n" for record in clean_records)
    atomic_write_text(clean_path, clean_text)
    atomic_write_text(ids_path, included_ids_text)
    exclusions_report = {
        "schema_version": 1,
        "split_name": "AnyInsertion-clean67-for-current-model",
        "scope": "training-overlap exclusion for the current model; not a universal source-quality filter",
        "source_manifest": str(primary_path.relative_to(output_root)),
        "source_manifest_sha256": file_sha256(primary_path),
        "clean_manifest": str(clean_path.relative_to(output_root)),
        "clean_manifest_sha256": file_sha256(clean_path),
        "included_case_ids_file": str(ids_path.relative_to(output_root)),
        "included_case_ids_sha256": file_sha256(ids_path),
        "included_case_ids": [str(record["case_id"]) for record in clean_records],
        "excluded_case_ids": [str(record["case_id"]) for record in excluded_records],
        "excluded_upstream_keys": [
            {"original_group": group, "original_stem": stem}
            for group, stem in sorted(ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS)
        ],
        "counts_major": actual_major_counts,
        "counts_case_type": actual_case_types,
        "legacy_decision_provenance": ANYINSERTION_LEGACY_OVERLAP_REVIEW,
    }
    atomic_write_text(
        exclusions_path,
        json.dumps(exclusions_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return clean_records


def refresh_anyinsertion_metadata(
    output_root: Path,
    raw_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Atomically apply reviewed identity/label metadata to a converted copy."""

    manifest_path = output_root / "manifests" / "anyinsertion_official.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing AnyInsertion manifest: {manifest_path}")
    validate_contained_regular_file(
        output_root,
        manifest_path,
        "AnyInsertion manifest",
    )
    records = read_jsonl(manifest_path)
    pre_refresh_rows: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for record in records:
        for path_key in ("src_image", "src_mask", "tar_image", "tar_mask"):
            canonical_path = canonical_record_asset(output_root, record, path_key)
            relative = validate_canonical_relative_path(record[path_key], path_key)
            if relative in seen_paths:
                raise ValueError(f"Repeated AnyInsertion canonical path before refresh: {relative}")
            seen_paths.add(relative)
            pre_refresh_rows.append((file_sha256(canonical_path), relative))
    pre_refresh_rows.append(
        (file_sha256(manifest_path), str(manifest_path.relative_to(output_root)))
    )
    conversion_path = output_root / "reports" / "conversion.json"
    if not conversion_path.is_file():
        raise FileNotFoundError(f"Missing AnyInsertion conversion provenance: {conversion_path}")
    validate_contained_regular_file(
        output_root,
        conversion_path,
        "AnyInsertion conversion provenance",
    )
    pre_refresh_rows.append(
        (file_sha256(conversion_path), str(conversion_path.relative_to(output_root)))
    )
    for authoritative_name in (
        ANYINSERTION_CLEAN_MANIFEST_NAME,
        ANYINSERTION_CLEAN_IDS_NAME,
        ANYINSERTION_EXCLUSIONS_NAME,
    ):
        authoritative_path = output_root / "manifests" / authoritative_name
        if authoritative_path.is_file():
            validate_contained_regular_file(
                output_root,
                authoritative_path,
                f"AnyInsertion authoritative file {authoritative_name}",
            )
            pre_refresh_rows.append(
                (
                    file_sha256(authoritative_path),
                    str(authoritative_path.relative_to(output_root)),
                )
            )
    checksum_errors = checksum_verification_errors(output_root, pre_refresh_rows)
    if checksum_errors:
        raise ValueError(
            "Refusing to refresh AnyInsertion metadata before checksum verification succeeds:\n- "
            + "\n- ".join(checksum_errors[:50])
        )
    # Complete every fallible read, parse, and source-snapshot validation
    # before replacing any authoritative file.  A validation error must leave
    # the old release and its checksum evidence byte-for-byte intact.
    try:
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Invalid AnyInsertion conversion provenance: {error}") from error
    if not isinstance(conversion, dict):
        raise ValueError("Invalid AnyInsertion conversion provenance: expected a JSON object")
    validated_source_snapshot = (
        validate_anyinsertion_source_snapshot(raw_root)
        if raw_root is not None
        else conversion.get(
            "source_snapshot",
            {**ANYINSERTION_SOURCE_SNAPSHOT, "verified": False},
        )
    )
    for record in records:
        upstream_key = (
            str(record.get("original_group", "")),
            str(record.get("original_stem", "")),
        )
        identity_override = ANYINSERTION_OPEN_CASES.get(upstream_key)
        paired_gt = identity_override is None
        record.update(
            {
                "gt_available": paired_gt,
                "case_type": "paired_gt" if paired_gt else "open_insertion",
                "same_identity": paired_gt,
                "identity_review_status": (
                    "verified_same_instance_visual"
                    if paired_gt
                    else "verified_different_instance_visual"
                ),
                "identity_override_reason": identity_override,
            }
        )
        label_override = ANYINSERTION_LABEL_OVERRIDES.get(upstream_key)
        original_label = str(
            record.get("original_leaf_label", record.get("leaf_label", "unlabeled"))
        )
        record["original_leaf_label"] = original_label
        record["leaf_label"] = label_override[0] if label_override else original_label
        record["label_override_reason"] = label_override[1] if label_override else None
        canonical_images = {
            "src": Image.open(canonical_record_asset(output_root, record, "src_image")).convert("RGB"),
            "tar": Image.open(canonical_record_asset(output_root, record, "tar_image")).convert("RGB"),
        }
        canonical_masks = {
            "src": binary_mask(Image.open(canonical_record_asset(output_root, record, "src_mask"))),
            "tar": binary_mask(Image.open(canonical_record_asset(output_root, record, "tar_mask"))),
        }
        record.update(
            {
                "src_pixel_sha256": decoded_pixel_sha256(canonical_images["src"]),
                "tar_pixel_sha256": decoded_pixel_sha256(canonical_images["tar"]),
                "src_mask_pixel_sha256": decoded_mask_sha256(canonical_masks["src"]),
                "tar_mask_pixel_sha256": decoded_mask_sha256(canonical_masks["tar"]),
            }
        )
    expected_upstream_keys = {
        *(('object', str(index)) for index in range(40)),
        *(('garment', str(index)) for index in range(30)),
        *(('person/simple_scene', str(index)) for index in range(30)),
    }
    actual_upstream_keys = {
        (
            str(record.get("original_group", "")),
            str(record.get("original_stem", "")),
        )
        for record in records
    }
    source_snapshot = conversion.get("source_snapshot")
    official_snapshot_claimed = (
        isinstance(source_snapshot, Mapping)
        and source_snapshot.get("verified") is True
    )
    if official_snapshot_claimed and (
        len(records) != 100 or actual_upstream_keys != expected_upstream_keys
    ):
        raise ValueError(
            "Refusing to refresh a non-official AnyInsertion key set: "
            f"rows={len(records)} missing={sorted(expected_upstream_keys - actual_upstream_keys)[:10]} "
            f"extra={sorted(actual_upstream_keys - expected_upstream_keys)[:10]}"
        )
    conversion["schema_version"] = 3
    conversion["counts_case_type"] = dict(
        sorted(Counter(record["case_type"] for record in records).items())
    )
    conversion["identity_override_count"] = sum(
        record.get("identity_override_reason") is not None for record in records
    )
    conversion["label_override_count"] = sum(
        record.get("label_override_reason") is not None for record in records
    )
    conversion["identity_review_method"] = (
        "agent_assisted_high_resolution_reference_target_mask_crop_review"
    )
    conversion["source_snapshot"] = validated_source_snapshot
    conversion["identity_overrides"] = [
        {"original_group": group, "original_stem": stem, "reason": reason}
        for (group, stem), reason in sorted(ANYINSERTION_OPEN_CASES.items())
    ]
    conversion["label_overrides"] = [
        {
            "original_group": group,
            "original_stem": stem,
            "corrected_label": corrected,
            "reason": reason,
        }
        for (group, stem), (corrected, reason) in sorted(
            ANYINSERTION_LABEL_OVERRIDES.items()
        )
    ]
    conversion["generator"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
        "python_version": sys.version.split()[0],
        "dependency_versions": runtime_dependency_versions(),
    }
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    conversion_text = json.dumps(
        conversion,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    atomic_write_text(manifest_path, manifest_text)
    atomic_write_text(conversion_path, conversion_text)
    return records


def anyinsertion_clean_split_errors(
    dataset_root: Path,
    primary_records: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Validate clean67 as an exact, ordered, provenance-pinned primary subset."""

    manifest_root = dataset_root / "manifests"
    clean_path = manifest_root / ANYINSERTION_CLEAN_MANIFEST_NAME
    ids_path = manifest_root / ANYINSERTION_CLEAN_IDS_NAME
    exclusions_path = manifest_root / ANYINSERTION_EXCLUSIONS_NAME
    errors: list[str] = []
    for path in (clean_path, ids_path, exclusions_path):
        if not path.is_file():
            errors.append(f"Missing authoritative AnyInsertion clean-split file: {path}")
    if errors:
        return errors

    try:
        clean_records = read_jsonl(clean_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"Invalid AnyInsertion clean manifest: {error}"]
    expected_clean = [
        dict(record)
        for record in primary_records
        if (
            str(record.get("original_group", "")),
            str(record.get("original_stem", "")),
        )
        not in ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS
    ]
    expected_excluded = [
        dict(record)
        for record in primary_records
        if (
            str(record.get("original_group", "")),
            str(record.get("original_stem", "")),
        )
        in ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS
    ]
    if clean_records != expected_clean:
        errors.append(
            "AnyInsertion clean manifest is not the exact ordered primary subset "
            "defined by the pinned 33 upstream exclusions"
        )
    clean_case_ids = [str(record.get("case_id", "")) for record in expected_clean]
    excluded_case_ids = [str(record.get("case_id", "")) for record in expected_excluded]
    included_id_rows = [
        line
        for line in ids_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if included_id_rows != clean_case_ids:
        errors.append("AnyInsertion clean67 case-ID sidecar differs from the clean manifest")
    clean_major_counts = dict(
        sorted(Counter(str(record.get("major", "")) for record in expected_clean).items())
    )
    clean_case_types = dict(
        sorted(Counter(str(record.get("case_type", "")) for record in expected_clean).items())
    )
    if clean_major_counts != {"garment": 27, "object": 38, "person": 2}:
        errors.append(f"AnyInsertion clean67 major counts mismatch: {clean_major_counts}")
    if clean_case_types != {"open_insertion": 1, "paired_gt": 66}:
        errors.append(f"AnyInsertion clean67 case-type counts mismatch: {clean_case_types}")

    try:
        exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append(f"Invalid AnyInsertion overlap-exclusion sidecar: {error}")
        return errors
    if not isinstance(exclusions, Mapping):
        errors.append("AnyInsertion overlap-exclusion sidecar must be a JSON object")
        return errors
    expected_upstream_rows = [
        {"original_group": group, "original_stem": stem}
        for group, stem in sorted(ANYINSERTION_CURRENT_MODEL_OVERLAP_EXCLUSIONS)
    ]
    expected_sidecar_fields = {
        "schema_version": 1,
        "split_name": "AnyInsertion-clean67-for-current-model",
        "scope": (
            "training-overlap exclusion for the current model; "
            "not a universal source-quality filter"
        ),
        "source_manifest": "manifests/anyinsertion_official.jsonl",
        "source_manifest_sha256": file_sha256(
            manifest_root / "anyinsertion_official.jsonl"
        ),
        "clean_manifest": f"manifests/{ANYINSERTION_CLEAN_MANIFEST_NAME}",
        "clean_manifest_sha256": file_sha256(clean_path),
        "included_case_ids_file": f"manifests/{ANYINSERTION_CLEAN_IDS_NAME}",
        "included_case_ids_sha256": file_sha256(ids_path),
        "included_case_ids": clean_case_ids,
        "excluded_case_ids": excluded_case_ids,
        "excluded_upstream_keys": expected_upstream_rows,
        "counts_major": clean_major_counts,
        "counts_case_type": clean_case_types,
        "legacy_decision_provenance": ANYINSERTION_LEGACY_OVERLAP_REVIEW,
    }
    for field, expected in expected_sidecar_fields.items():
        if exclusions.get(field) != expected:
            errors.append(
                f"AnyInsertion overlap-exclusion sidecar mismatch {field}: "
                f"actual={exclusions.get(field)} expected={expected}"
            )
    return errors


def audit_dataset(
    dataset_root: Path,
    *,
    strict_900: bool = True,
    strict_anyinsertion: bool = False,
    finalize_checksums: bool = False,
) -> dict[str, Any]:
    manifest_name = "paperinsert_900.jsonl" if strict_900 else "anyinsertion_official.jsonl"
    manifest_path = dataset_root / "manifests" / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    validate_contained_regular_file(
        dataset_root,
        manifest_path,
        "primary manifest",
    )
    records = read_jsonl(manifest_path)
    required_manifest_fields = {
        "case_id",
        "major",
        "minor",
        "protocol",
        "src_image",
        "src_mask",
        "tar_image",
        "tar_mask",
    }
    for row_index, record in enumerate(records):
        missing_fields = sorted(required_manifest_fields.difference(record))
        if missing_fields:
            raise ValueError(
                f"Manifest row {row_index} is missing required fields: {missing_fields}"
            )
    errors: list[str] = []
    warnings: list[str] = []
    counts_major = Counter(record["major"] for record in records)
    counts_minor = Counter((record["major"], record["minor"]) for record in records)
    protocols = Counter(record["protocol"] for record in records)
    tar_image_ids: set[tuple[str, str]] = set()
    src_image_ids: set[tuple[str, str]] = set()
    case_ids: set[str] = set()
    manifest_paths: set[str] = set()
    cross_source_image_ids: set[tuple[str, str]] = set()
    any_upstream_keys: set[tuple[str, str]] = set()
    validated_majors: set[str] = set()
    configured_excluded_ids = load_excluded_image_ids() if strict_900 else set()
    configured_paper_avoid_ids = load_paper_avoid_image_ids() if strict_900 else set()
    checksum_rows: list[tuple[str, str]] = []

    if strict_900:
        source_report_path = dataset_root / "reports" / "sources.json"
        source_report_safe = False
        if not source_report_path.is_file():
            errors.append(f"Missing curated source provenance: {source_report_path}")
        else:
            try:
                validate_contained_regular_file(
                    dataset_root,
                    source_report_path,
                    "curated source provenance",
                )
            except ValueError as error:
                errors.append(str(error))
            else:
                source_report_safe = True
        if source_report_safe:
            try:
                source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                errors.append(f"Invalid curated source provenance: {error}")
            else:
                if not isinstance(source_report, Mapping):
                    errors.append("Curated source provenance must be a JSON object")
                    source_report = {}
                if source_report.get("schema_version") != 3:
                    errors.append(
                        "Curated source provenance schema mismatch: "
                        f"actual={source_report.get('schema_version')} expected=3"
                    )
                generator = source_report.get("generator")
                dependency_versions = (
                    generator.get("dependency_versions")
                    if isinstance(generator, Mapping)
                    else None
                )
                if (
                    not isinstance(generator, Mapping)
                    or re.fullmatch(r"[0-9a-f]{64}", str(generator.get("sha256", ""))) is None
                    or not isinstance(dependency_versions, Mapping)
                    or not str(dependency_versions.get("ImageHash", ""))
                ):
                    errors.append("Curated source provenance lacks generator/ImageHash version")
                else:
                    current_generator = Path(__file__).resolve()
                    if (
                        generator.get("path") != str(current_generator)
                        or generator.get("sha256") != file_sha256(current_generator)
                    ):
                        errors.append("Curated source provenance generator does not match this release builder")
                if source_report.get("dataset_name") != "PaperInsert-900":
                    errors.append("Curated source provenance dataset_name mismatch")
                selection_seed = source_report.get("selection_seed")
                if (
                    not isinstance(selection_seed, int)
                    or isinstance(selection_seed, bool)
                    or not 0 <= selection_seed < 2**32
                ):
                    errors.append("Curated source provenance has an invalid selection_seed")
                if source_report.get("quotas") != MAJOR_MINOR_QUOTAS:
                    errors.append("Curated source provenance quota policy mismatch")
                expected_protocol_policy = {
                    "per_minor": {
                        "cross_instance_replace": 50,
                    },
                    "cross_gt_available": False,
                    "donor_target_same_image": "forbidden",
                    "source_image_reuse_cap": 1,
                    "target_image_reuse_cap": 1,
                }
                if source_report.get("protocol_policy") != expected_protocol_policy:
                    errors.append("Curated source provenance protocol policy mismatch")
                exclusions = source_report.get("exclusions")
                expected_exclusion_sha = (
                    file_sha256(DEFAULT_EXCLUSION_CONFIG)
                    if DEFAULT_EXCLUSION_CONFIG.is_file()
                    else None
                )
                if (
                    not isinstance(exclusions, Mapping)
                    or exclusions.get("sha256") != expected_exclusion_sha
                ):
                    errors.append(
                        "Curated exclusion-config provenance mismatch: "
                        f"actual={exclusions.get('sha256') if isinstance(exclusions, Mapping) else None} "
                        f"expected={expected_exclusion_sha}"
                    )
                else:
                    expected_excluded_counts = dict(
                        sorted(Counter(dataset for dataset, _ in configured_excluded_ids).items())
                    )
                    expected_paper_counts = dict(
                        sorted(Counter(dataset for dataset, _ in configured_paper_avoid_ids).items())
                    )
                    if exclusions.get("excluded_image_counts") != expected_excluded_counts:
                        errors.append("Curated hard-exclusion provenance counts mismatch")
                    if exclusions.get("paper_avoid_image_counts") != expected_paper_counts:
                        errors.append("Curated paper-avoid provenance counts mismatch")
                source_entries = source_report.get("sources")
                expected_source_names = {
                    "coco_val_images",
                    "coco_annotations",
                    "fashionpedia_exact_masks",
                }
                if (
                    not isinstance(source_entries, Mapping)
                    or set(source_entries) != expected_source_names
                ):
                    errors.append("Curated source provenance inventory mismatch")
                else:
                    for source_name in sorted(expected_source_names):
                        source_entry = source_entries[source_name]
                        if not isinstance(source_entry, Mapping):
                            errors.append(f"Curated source entry is not an object: {source_name}")
                            continue
                        source_path = Path(str(source_entry.get("path", "")))
                        recorded_size = source_entry.get("bytes")
                        recorded_sha = str(source_entry.get("sha256", ""))
                        if (
                            not source_path.is_file()
                            or not isinstance(recorded_size, int)
                            or isinstance(recorded_size, bool)
                            or re.fullmatch(r"[0-9a-f]{64}", recorded_sha) is None
                        ):
                            errors.append(f"Curated source entry is incomplete: {source_name}")
                            continue
                        expected_source = CURATED_SOURCE_SNAPSHOT[source_name]
                        if (
                            recorded_size != expected_source["bytes"]
                            or recorded_sha != expected_source["sha256"]
                        ):
                            errors.append(f"Curated pinned source provenance mismatch: {source_name}")
                            continue
                        if source_path.stat().st_size != recorded_size:
                            errors.append(f"Curated source byte count mismatch: {source_name}")
                            continue
                        if file_sha256(source_path) != recorded_sha:
                            errors.append(f"Curated source hash mismatch: {source_name}")
        expected_major = {major: sum(quotas.values()) for major, quotas in MAJOR_MINOR_QUOTAS.items()}
        if dict(counts_major) != expected_major:
            errors.append(f"Major counts mismatch: actual={dict(counts_major)} expected={expected_major}")
        for major, minor_quotas in MAJOR_MINOR_QUOTAS.items():
            for minor, expected in minor_quotas.items():
                actual = counts_minor[(major, minor)]
                if actual != expected:
                    errors.append(f"Minor count mismatch {major}/{minor}: actual={actual} expected={expected}")
    elif strict_anyinsertion:
        expected_any_counts = {"garment": 30, "object": 40, "person": 30}
        if dict(counts_major) != expected_any_counts:
            errors.append(
                f"AnyInsertion counts mismatch: actual={dict(counts_major)} expected={expected_any_counts}"
            )
        conversion_path = dataset_root / "reports" / "conversion.json"
        conversion_safe = False
        if not conversion_path.is_file():
            errors.append(f"Missing AnyInsertion conversion provenance: {conversion_path}")
        else:
            try:
                validate_contained_regular_file(
                    dataset_root,
                    conversion_path,
                    "AnyInsertion conversion provenance",
                )
            except ValueError as error:
                errors.append(str(error))
            else:
                conversion_safe = True
        if conversion_safe:
            try:
                conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                errors.append(f"Invalid AnyInsertion conversion provenance: {error}")
            else:
                if not isinstance(conversion, Mapping):
                    errors.append("AnyInsertion conversion provenance must be a JSON object")
                    conversion = {}
                if conversion.get("schema_version") != 3:
                    errors.append(
                        "AnyInsertion conversion schema mismatch: "
                        f"actual={conversion.get('schema_version')} expected=3"
                    )
                generator = conversion.get("generator")
                dependency_versions = (
                    generator.get("dependency_versions")
                    if isinstance(generator, Mapping)
                    else None
                )
                if (
                    not isinstance(generator, Mapping)
                    or re.fullmatch(r"[0-9a-f]{64}", str(generator.get("sha256", ""))) is None
                    or not isinstance(dependency_versions, Mapping)
                    or not str(dependency_versions.get("ImageHash", ""))
                ):
                    errors.append("AnyInsertion conversion lacks generator/ImageHash version")
                else:
                    current_generator = Path(__file__).resolve()
                    if (
                        generator.get("path") != str(current_generator)
                        or generator.get("sha256") != file_sha256(current_generator)
                    ):
                        errors.append("AnyInsertion conversion generator does not match this release builder")
                expected_snapshot = {**ANYINSERTION_SOURCE_SNAPSHOT, "verified": True}
                actual_snapshot = conversion.get("source_snapshot", {})
                if not isinstance(actual_snapshot, Mapping):
                    errors.append("AnyInsertion source_snapshot provenance must be an object")
                else:
                    for key, expected_value in expected_snapshot.items():
                        if actual_snapshot.get(key) != expected_value:
                            errors.append(
                                f"AnyInsertion source snapshot field mismatch {key}: "
                                f"actual={actual_snapshot.get(key)} expected={expected_value}"
                            )
                expected_identity_overrides = [
                    {"original_group": group, "original_stem": stem, "reason": reason}
                    for (group, stem), reason in sorted(ANYINSERTION_OPEN_CASES.items())
                ]
                expected_label_overrides = [
                    {
                        "original_group": group,
                        "original_stem": stem,
                        "corrected_label": corrected,
                        "reason": reason,
                    }
                    for (group, stem), (corrected, reason) in sorted(
                        ANYINSERTION_LABEL_OVERRIDES.items()
                    )
                ]
                expected_excluded_groups = [
                    {
                        "group": "person/complex_scene",
                        "missing_directories": ["tar_image", "tar_mask"],
                        "reason": "missing_exact_four_file_contract",
                        "reference_count": 30,
                    }
                ]
                expected_conversion_fields = {
                    "record_count": 100,
                    "counts_major": {"garment": 30, "object": 40, "person": 30},
                    "counts_case_type": {"open_insertion": 1, "paired_gt": 99},
                    "official_clean_source_count": 38,
                    "identity_review_method": (
                        "agent_assisted_high_resolution_reference_target_mask_crop_review"
                    ),
                    "identity_overrides": expected_identity_overrides,
                    "label_overrides": expected_label_overrides,
                    "admitted_groups": ["garment", "object", "person/simple_scene"],
                    "excluded_groups": expected_excluded_groups,
                    "admission_rule": (
                        "official ref_image/ref_mask/tar_image/tar_mask all exist with "
                        "matching geometry and non-empty exact binary masks"
                    ),
                }
                for field_name, expected_value in expected_conversion_fields.items():
                    if conversion.get(field_name) != expected_value:
                        errors.append(
                            f"AnyInsertion conversion provenance mismatch {field_name}: "
                            f"actual={conversion.get(field_name)} expected={expected_value}"
                        )

    for record in records:
        try:
            case_id = validate_safe_component(record.get("case_id", ""), "case_id")
            major = validate_safe_component(record.get("major", ""), "major")
        except ValueError as error:
            errors.append(str(error))
            continue
        validated_majors.add(major)
        if case_id in case_ids:
            errors.append(f"Duplicate case_id: {case_id}")
        case_ids.add(case_id)
        for image_key, mask_key in (("src_image", "src_mask"), ("tar_image", "tar_mask")):
            pair_paths_valid = True
            for path_key in (image_key, mask_key):
                try:
                    relative_path = validate_canonical_relative_path(
                        record.get(path_key, ""),
                        path_key,
                    )
                except ValueError as error:
                    errors.append(f"{case_id}: {error}")
                    pair_paths_valid = False
                    continue
                expected_path = f"{major}/{path_key}/{case_id}.png"
                if relative_path != expected_path:
                    errors.append(
                        f"{case_id}: non-canonical {path_key} path: "
                        f"actual={relative_path} expected={expected_path}"
                    )
                    pair_paths_valid = False
                if relative_path in manifest_paths:
                    errors.append(f"{case_id}: manifest path reused: {relative_path}")
                manifest_paths.add(relative_path)
            if not pair_paths_valid:
                continue
            image_path = dataset_root / str(record[image_key])
            mask_path = dataset_root / str(record[mask_key])
            contained = True
            for candidate_path, path_key in ((image_path, image_key), (mask_path, mask_key)):
                try:
                    validate_contained_regular_file(
                        dataset_root,
                        candidate_path,
                        f"{case_id} {path_key}",
                    )
                except ValueError as error:
                    errors.append(str(error))
                    contained = False
            if not contained:
                continue
            if not image_path.is_file() or not mask_path.is_file():
                errors.append(f"{case_id}: missing {image_key}/{mask_key}")
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
            except Exception as error:
                errors.append(f"{case_id}: decode failure {image_key}/{mask_key}: {error}")
                continue
            if image.size != mask.size:
                errors.append(f"{case_id}: geometry mismatch {image_key}={image.size} {mask_key}={mask.size}")
            values = set(np.unique(np.asarray(mask, dtype=np.uint8)).tolist())
            if not values.issubset({0, 255}):
                errors.append(f"{case_id}: non-binary {mask_key} values={sorted(values)[:16]}")
            area = int(np.count_nonzero(np.asarray(mask, dtype=np.uint8)))
            if area <= 0 or area >= mask.width * mask.height:
                errors.append(f"{case_id}: empty/full {mask_key} area={area}")
            if strict_900:
                area_field = "src_area_ratio" if mask_key == "src_mask" else "tar_area_ratio"
                area_ratio = area / float(mask.width * mask.height)
                try:
                    recorded_area_ratio = float(record[area_field])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{case_id}: invalid or missing {area_field}")
                else:
                    if abs(recorded_area_ratio - area_ratio) > 1e-8:
                        errors.append(
                            f"{case_id}: {area_field} mismatch "
                            f"actual={recorded_area_ratio} decoded={area_ratio:.10f}"
                        )
            if strict_900:
                mask_array = np.asarray(mask, dtype=np.uint8)
                if np.any(mask_array[0]) or np.any(mask_array[-1]) or np.any(mask_array[:, 0]) or np.any(mask_array[:, -1]):
                    errors.append(f"{case_id}: {mask_key} touches image border")
            checksum_rows.append((file_sha256(image_path), str(image_path.relative_to(dataset_root))))
            checksum_rows.append((file_sha256(mask_path), str(mask_path.relative_to(dataset_root))))
            pixel_hash_field = "src_pixel_sha256" if image_key == "src_image" else "tar_pixel_sha256"
            if (strict_900 or strict_anyinsertion) and decoded_pixel_sha256(image) != str(
                record.get(pixel_hash_field, "")
            ):
                errors.append(f"{case_id}: decoded hash mismatch for {image_key}")
            mask_hash_field = (
                "src_mask_pixel_sha256" if mask_key == "src_mask" else "tar_mask_pixel_sha256"
            )
            if (strict_900 or strict_anyinsertion) and decoded_mask_sha256(mask) != str(
                record.get(mask_hash_field, "")
            ):
                errors.append(f"{case_id}: decoded hash mismatch for {mask_key}")
        if strict_900:
            target_key = (str(record["source_dataset"]), str(record["tar_image_id"]))
            if target_key in tar_image_ids:
                errors.append(f"{case_id}: repeated target image {target_key}")
            tar_image_ids.add(target_key)
            source_key = (str(record["src_source_dataset"]), str(record["src_image_id"]))
            if source_key in src_image_ids:
                errors.append(f"{case_id}: repeated source image {source_key}")
            src_image_ids.add(source_key)
            for role, image_key in (("src", source_key), ("tar", target_key)):
                if image_key in configured_excluded_ids:
                    errors.append(
                        f"{case_id}: hard-excluded {role} image remains in manifest: {image_key}"
                    )

            full_hash = record.get("src_full_pixel_sha256")
            full_phash = record.get("src_full_phash")
            full_dhash = record.get("src_full_dhash")
            full_width = record.get("src_full_width")
            full_height = record.get("src_full_height")
            if not isinstance(full_hash, str) or re.fullmatch(r"[0-9a-f]{64}", full_hash) is None:
                errors.append(f"{case_id}: invalid src_full_pixel_sha256")
            if not isinstance(full_phash, str) or re.fullmatch(r"[0-9a-f]{16}", full_phash) is None:
                errors.append(f"{case_id}: invalid src_full_phash")
            if not isinstance(full_dhash, str) or re.fullmatch(r"[0-9a-f]{16}", full_dhash) is None:
                errors.append(f"{case_id}: invalid src_full_dhash")
            if not isinstance(full_width, int) or isinstance(full_width, bool) or full_width <= 0:
                errors.append(f"{case_id}: invalid src_full_width")
            if not isinstance(full_height, int) or isinstance(full_height, bool) or full_height <= 0:
                errors.append(f"{case_id}: invalid src_full_height")

            protocol = str(record["protocol"])
            try:
                recorded_match_cost = float(record["cross_match_cost"])
                recorded_showcase_score = float(record["showcase_score"])
                recorded_selection_score = float(record["selection_score"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{case_id}: missing numeric pair/showcase score")
                recorded_match_cost = math.nan
                recorded_showcase_score = math.nan
                recorded_selection_score = math.nan
            if not all(
                math.isfinite(value)
                for value in (
                    recorded_match_cost,
                    recorded_showcase_score,
                    recorded_selection_score,
                )
            ):
                errors.append(f"{case_id}: non-finite pair/showcase score")
            else:
                expected_showcase_score = recorded_selection_score - max(recorded_match_cost, 0.0)
                if abs(recorded_showcase_score - expected_showcase_score) > 2e-8:
                    errors.append(
                        f"{case_id}: showcase_score mismatch "
                        f"actual={recorded_showcase_score} expected={expected_showcase_score:.8f}"
                    )
                expected_challenge_flags = (
                    ["large_cross_match_gap"]
                    if protocol == "cross_instance_replace"
                    and max(recorded_match_cost, 0.0) > CROSS_HARD_COST
                    else []
                )
                if record.get("challenge_flags") != expected_challenge_flags:
                    errors.append(f"{case_id}: stale challenge_flags")
                if expected_challenge_flags:
                    expected_difficulty = "hard"
                elif (
                    protocol == "cross_instance_replace"
                    and max(recorded_match_cost, 0.0) > CROSS_MODERATE_COST
                ):
                    expected_difficulty = "moderate"
                else:
                    expected_difficulty = (
                        "easy" if recorded_selection_score >= 0.68 else "moderate"
                    )
                if record.get("difficulty") != expected_difficulty:
                    errors.append(f"{case_id}: stale difficulty metadata")
            avoid_roles = record.get("agent_paper_avoid_roles")
            avoid_reasons = record.get("agent_paper_avoid_reasons")
            avoid_flag = record.get("agent_paper_avoid")
            roles_are_strings = isinstance(avoid_roles, list) and all(
                isinstance(role, str) for role in avoid_roles
            )
            role_set = set(avoid_roles) if roles_are_strings else set()
            valid_avoid_roles = (
                roles_are_strings
                and len(avoid_roles) == len(role_set)
                and role_set.issubset({"src", "tar"})
            )
            flag_matches_roles = (
                isinstance(avoid_flag, bool) and avoid_flag == bool(avoid_roles)
            )
            if (
                not isinstance(avoid_flag, bool)
                or not valid_avoid_roles
                or not flag_matches_roles
            ):
                errors.append(f"{case_id}: invalid agent_paper_avoid metadata")
            reasons_valid = isinstance(avoid_reasons, Mapping)
            if reasons_valid:
                reason_keys = list(avoid_reasons)
                reasons_valid = (
                    all(isinstance(role, str) for role in reason_keys)
                    and set(reason_keys) == role_set
                )
            if reasons_valid:
                for reasons in avoid_reasons.values():
                    if (
                        not isinstance(reasons, list)
                        or not reasons
                        or not all(isinstance(reason, str) for reason in reasons)
                        or len(reasons) != len(set(reasons))
                        or not set(reasons).issubset(PAPER_AVOID_REASON_CODES)
                    ):
                        reasons_valid = False
                        break
            if not reasons_valid:
                errors.append(f"{case_id}: invalid agent_paper_avoid reasons")
            else:
                role_keys = {
                    "src": (
                        str(record["src_source_dataset"]),
                        str(record["src_image_id"]),
                    ),
                    "tar": (
                        str(record["source_dataset"]),
                        str(record["tar_image_id"]),
                    ),
                }
                for role, image_key in role_keys.items():
                    has_config_reason = "agent_visual_qc_image_flag" in avoid_reasons.get(role, [])
                    if has_config_reason != (image_key in configured_paper_avoid_ids):
                        errors.append(
                            f"{case_id}: stale configured paper-avoid reason for {role} {image_key}"
                        )
                if any(
                    "caption_indicates_possible_minor_in_scene" in reasons
                    for reasons in avoid_reasons.values()
                ) and str(record.get("major")) != "person":
                    errors.append(f"{case_id}: possible-minor paper flag outside person track")
            same_instance = (
                str(record["src_source_dataset"]) == str(record["source_dataset"])
                and str(record["src_image_id"]) == str(record["tar_image_id"])
                and str(record["src_ann_id"]) == str(record["tar_ann_id"])
            )
            same_image = (
                str(record["src_source_dataset"]) == str(record["source_dataset"])
                and str(record["src_image_id"]) == str(record["tar_image_id"])
            )
            same_leaf = str(record["src_leaf_label"]) == str(record["leaf_label"])
            if record.get("mask_origin_src") != "native_instance" or record.get("mask_origin_tar") != "native_instance":
                errors.append(f"{case_id}: curated masks must have native_instance provenance")
            if protocol == "cross_instance_replace":
                cross_source_image_ids.add(source_key)
                if same_image or same_instance or record.get("same_identity") is not False:
                    errors.append(f"{case_id}: cross protocol must use a different source image")
                if not same_leaf:
                    errors.append(f"{case_id}: cross protocol leaf mismatch")
                if record.get("gt_available") is not False:
                    errors.append(f"{case_id}: cross protocol must have gt_available=false")
            else:
                errors.append(f"{case_id}: unsupported curated protocol {protocol}")
        elif strict_anyinsertion:
            if record.get("protocol") != "official_anyinsertion":
                errors.append(f"{case_id}: unexpected AnyInsertion protocol {record.get('protocol')}")
            upstream_key = (
                str(record.get("original_group", "")),
                str(record.get("original_stem", "")),
            )
            if upstream_key in any_upstream_keys:
                errors.append(f"{case_id}: duplicate upstream AnyInsertion key {upstream_key}")
            any_upstream_keys.add(upstream_key)
            expected_open_reason = ANYINSERTION_OPEN_CASES.get(upstream_key)
            expected_label_override = ANYINSERTION_LABEL_OVERRIDES.get(upstream_key)
            if expected_open_reason is None:
                if (
                    record.get("gt_available") is not True
                    or record.get("case_type") != "paired_gt"
                    or record.get("same_identity") is not True
                    or record.get("identity_review_status") != "verified_same_instance_visual"
                ):
                    errors.append(f"{case_id}: verified paired official case must expose pixel GT")
            elif (
                record.get("gt_available") is not False
                or record.get("case_type") != "open_insertion"
                or record.get("same_identity") is not False
                or record.get("identity_review_status") != "verified_different_instance_visual"
                or record.get("identity_override_reason") != expected_open_reason
            ):
                errors.append(f"{case_id}: known identity-mismatch case must be open/no-GT")
            if not isinstance(record.get("official_clean_source_available"), bool):
                errors.append(f"{case_id}: missing official_clean_source_available boolean")
            if expected_label_override is not None and (
                record.get("leaf_label") != expected_label_override[0]
                or record.get("label_override_reason") != expected_label_override[1]
            ):
                errors.append(f"{case_id}: known official label error was not corrected")
            if not str(record.get("original_leaf_label", "")):
                errors.append(f"{case_id}: missing original official label provenance")
            if expected_label_override is None and (
                record.get("leaf_label") != record.get("original_leaf_label")
                or record.get("label_override_reason") is not None
            ):
                errors.append(f"{case_id}: unreviewed AnyInsertion label was changed")
            if record.get("mask_origin_src") != "official_native" or record.get("mask_origin_tar") != "official_native":
                errors.append(f"{case_id}: official mask provenance is missing")

    if strict_900:
        # Every case is cross-instance replace: the donor always comes from
        # a different image than the target.  A same-image paired track no
        # longer exists, so any paired record is a protocol violation.
        if protocols["paired_reconstruction"] != 0:
            errors.append(
                f"Protocol count mismatch paired_reconstruction: actual={protocols['paired_reconstruction']} expected=0"
            )
        if protocols["cross_instance_replace"] != len(records):
            errors.append(
                f"Protocol count mismatch cross_instance_replace: actual={protocols['cross_instance_replace']} expected={len(records)}"
            )
        protocol_minor_counts = Counter(
            (record["major"], record["minor"], record["protocol"])
            for record in records
        )
        for major, minor_quotas in MAJOR_MINOR_QUOTAS.items():
            for minor, expected in minor_quotas.items():
                for protocol, quota in (
                    ("paired_reconstruction", 0),
                    ("cross_instance_replace", expected),
                ):
                    actual = protocol_minor_counts[(major, minor, protocol)]
                    if actual != quota:
                        errors.append(
                            f"Minor protocol mismatch {major}/{minor}/{protocol}: "
                            f"actual={actual} expected={quota}"
                        )

    if strict_900:
        avoided_showcases = [
            str(record["case_id"])
            for record in records
            if record.get("showcase_candidate") and record.get("agent_paper_avoid")
        ]
        if avoided_showcases:
            errors.append(f"Paper-avoided cases entered showcase pool: {avoided_showcases[:10]}")
        showcase_protocol_counts = Counter(
            (record["major"], record["minor"], record["protocol"])
            for record in records
            if record.get("showcase_candidate")
        )
        for major, minor_quotas in MAJOR_MINOR_QUOTAS.items():
            for minor in minor_quotas:
                # With the single cross-instance protocol the qualitative
                # queue draws up to ten clean/coherent cases per minor.
                protocol = "cross_instance_replace"
                actual = showcase_protocol_counts[(major, minor, protocol)]
                available = sum(
                    1
                    for record in records
                    if record["major"] == major
                    and record["minor"] == minor
                    and record["protocol"] == protocol
                    and not record.get("agent_paper_avoid", False)
                    and "large_cross_match_gap"
                    not in record.get("challenge_flags", [])
                )
                expected = min(10, available)
                if actual != expected:
                    errors.append(
                        f"Showcase protocol mismatch {major}/{minor}/{protocol}: "
                        f"actual={actual} expected={expected}"
                    )
                if expected < 10:
                    warnings.append(
                        f"Showcase clean/coherent-candidate shortfall {major}/{minor}/{protocol}: "
                        f"selected={actual} preferred=10"
                    )

    if strict_anyinsertion:
        expected_upstream_keys = {
            *(('object', str(index)) for index in range(40)),
            *(('garment', str(index)) for index in range(30)),
            *(('person/simple_scene', str(index)) for index in range(30)),
        }
        if any_upstream_keys != expected_upstream_keys:
            errors.append(
                "AnyInsertion upstream key set mismatch: "
                f"missing={sorted(expected_upstream_keys - any_upstream_keys)[:10]} "
                f"extra={sorted(any_upstream_keys - expected_upstream_keys)[:10]}"
            )
        expected_groups = {
            "garment": 30,
            "object": 40,
            "person/simple_scene": 30,
        }
        actual_groups = Counter(str(record.get("original_group", "")) for record in records)
        if dict(actual_groups) != expected_groups:
            errors.append(
                f"AnyInsertion group counts mismatch: "
                f"actual={dict(actual_groups)} expected={expected_groups}"
            )
        expected_case_types = {"open_insertion": 1, "paired_gt": 99}
        actual_case_types = Counter(str(record.get("case_type", "")) for record in records)
        if dict(actual_case_types) != expected_case_types:
            errors.append(
                f"AnyInsertion case-type counts mismatch: "
                f"actual={dict(actual_case_types)} expected={expected_case_types}"
            )
        expected_clean_source_counts = {"garment": 22, "object": 16}
        actual_clean_source_counts = Counter(
            str(record.get("original_group", ""))
            for record in records
            if record.get("official_clean_source_available") is True
        )
        if dict(actual_clean_source_counts) != expected_clean_source_counts:
            errors.append(
                f"AnyInsertion official-clean-source counts mismatch: "
                f"actual={dict(actual_clean_source_counts)} expected={expected_clean_source_counts}"
            )
        errors.extend(anyinsertion_clean_split_errors(dataset_root, records))

    canonical_dirs = [
        dataset_root / major / field
        for major in validated_majors
        for field in ("src_image", "src_mask", "tar_image", "tar_mask")
    ]
    actual_paths: set[str] = set()
    for directory in canonical_dirs:
        try:
            validate_contained_directory(
                dataset_root,
                directory,
                f"canonical directory {directory.relative_to(dataset_root)}",
            )
        except ValueError as error:
            errors.append(str(error))
            continue
        for path in directory.iterdir():
            relative = str(path.relative_to(dataset_root))
            # Canonical leaves are deliberately flat.  A nested directory,
            # symlink (including dangling links), FIFO, socket, or device is
            # unsigned package content and must fail both verification and
            # checksum finalization instead of disappearing from inventory.
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"Unsigned non-regular or nested canonical content: {relative}"
                )
                continue
            actual_paths.add(relative)
    orphan_paths = sorted(actual_paths.difference(manifest_paths))
    if orphan_paths:
        errors.append(f"Orphan canonical files not present in manifest: {orphan_paths[:10]}")

    if strict_900:
        authoritative_paths = (
            "manifests/paperinsert_900.jsonl",
            "manifests/paperinsert_900.csv",
            "manifests/object.jsonl",
            "manifests/person.jsonl",
            "manifests/garment.jsonl",
            "reports/sources.json",
        )
    elif strict_anyinsertion:
        authoritative_paths = (
            "manifests/anyinsertion_official.jsonl",
            f"manifests/{ANYINSERTION_CLEAN_MANIFEST_NAME}",
            f"manifests/{ANYINSERTION_CLEAN_IDS_NAME}",
            f"manifests/{ANYINSERTION_EXCLUSIONS_NAME}",
            "reports/conversion.json",
        )
    else:
        authoritative_paths = ["manifests/anyinsertion_official.jsonl"]
        for optional_name in (
            ANYINSERTION_CLEAN_MANIFEST_NAME,
            ANYINSERTION_CLEAN_IDS_NAME,
            ANYINSERTION_EXCLUSIONS_NAME,
        ):
            if (dataset_root / "manifests" / optional_name).is_file():
                authoritative_paths.append(f"manifests/{optional_name}")
        if (dataset_root / "reports" / "conversion.json").is_file():
            authoritative_paths.append("reports/conversion.json")
    for relative_authoritative_path in authoritative_paths:
        authoritative_path = dataset_root / relative_authoritative_path
        if not authoritative_path.is_file():
            errors.append(f"Missing authoritative manifest: {authoritative_path}")
            continue
        try:
            validate_contained_regular_file(
                dataset_root,
                authoritative_path,
                f"authoritative file {relative_authoritative_path}",
            )
        except ValueError as error:
            errors.append(str(error))
            continue
        checksum_rows.append(
            (file_sha256(authoritative_path), str(authoritative_path.relative_to(dataset_root)))
        )

    checksum_path = dataset_root / "CHECKSUMS.sha256"
    if finalize_checksums:
        checksum_errors: list[str] = []
        checksum_verified = not errors
        if not errors:
            write_checksums_atomic(checksum_path, checksum_rows)
    else:
        checksum_errors = checksum_verification_errors(dataset_root, checksum_rows)
        errors.extend(checksum_errors)
        checksum_verified = not checksum_errors

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "manifest": str(manifest_path),
        "valid": not errors,
        "record_count": len(records),
        "counts_major": {str(key): value for key, value in sorted(counts_major.items())},
        "counts_minor": {f"{major}/{minor}": value for (major, minor), value in sorted(counts_minor.items())},
        "protocols": dict(sorted(protocols.items())),
        "case_types": dict(
            sorted(Counter(str(record.get("case_type", "")) for record in records).items())
        ),
        "protocol_leaf_counts": {
            f"{major}/{minor}/{leaf}/{protocol}": count
            for (major, minor, leaf, protocol), count in sorted(
                Counter(
                    (
                        str(record["major"]),
                        str(record["minor"]),
                        str(record.get("leaf_label", "")),
                        str(record["protocol"]),
                    )
                    for record in records
                ).items()
            )
        },
        "unique_case_ids": len(case_ids),
        "canonical_file_count": len(actual_paths),
        "manifest_path_count": len(manifest_paths),
        "checksum_mode": "finalize" if finalize_checksums else "verify",
        "checksum_entry_count": len(checksum_rows),
        "checksum_verified": checksum_verified,
        "source_target_overlap_count": len(src_image_ids.intersection(tar_image_ids)) if strict_900 else None,
        "cross_source_target_overlap_count": (
            len(cross_source_image_ids.intersection(tar_image_ids)) if strict_900 else None
        ),
        "errors": errors,
        "warnings": warnings,
    }
    report_root = prepare_contained_output_directory(dataset_root, "reports")
    atomic_write_text(
        report_root / "audit.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if errors:
        raise ValueError("Dataset audit failed:\n- " + "\n- ".join(errors[:50]))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "extract",
            "build",
            "refresh-showcase",
            "qc-previews",
            "convert-anyinsertion",
            "refresh-anyinsertion-metadata",
            "qc-previews-anyinsertion",
            "audit",
            "audit-anyinsertion",
            "all",
        ),
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--anyinsertion-output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw_root = (args.raw_root or args.work_root / "_raw").resolve()
    curated_root = (args.output_root or args.work_root / "paperinsert_900").resolve()
    if args.command in (
        "convert-anyinsertion",
        "refresh-anyinsertion-metadata",
        "qc-previews-anyinsertion",
        "audit-anyinsertion",
    ):
        anyinsertion_root = (
            args.anyinsertion_output_root or args.output_root or args.work_root / "anyinsertion_official"
        ).resolve()
    else:
        anyinsertion_root = (
            args.anyinsertion_output_root or args.work_root / "anyinsertion_official"
        ).resolve()
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("--seed must be in [0, 2**32)")

    if args.command in ("extract", "all"):
        extract_raw_archives(raw_root)
    if args.command == "build":
        # A standalone build must consume members verified against the pinned
        # archives, not trust a pre-existing extracted tree.
        extract_raw_archives(raw_root)
    if args.command in ("build", "all"):
        records = build_curated_core(raw_root, curated_root, args.seed)
        print(f"Built curated core: {len(records)} cases at {curated_root}")
        report = audit_dataset(curated_root, strict_900=True, finalize_checksums=True)
        print(json.dumps(report["counts_major"], sort_keys=True))
    if args.command == "refresh-showcase":
        # Never turn unverified canonical/manifest edits into a fresh checksum
        # set.  A refresh is metadata-only and must begin from a known release.
        audit_dataset(curated_root, strict_900=True)
        manifest_path = curated_root / "manifests" / "paperinsert_900.jsonl"
        records = read_jsonl(manifest_path)
        refresh_paper_avoid_metadata(records, load_paper_avoid_image_ids())
        freeze_showcase_candidates(records, per_minor=10)
        write_manifests(curated_root, records)
        generate_previews(curated_root, records)
        report = audit_dataset(curated_root, strict_900=True, finalize_checksums=True)
        print(json.dumps(report["counts_major"], sort_keys=True))
    if args.command == "qc-previews":
        records = read_jsonl(curated_root / "manifests" / "paperinsert_900.jsonl")
        report = generate_qc_contact_sheets(curated_root, records)
        print(json.dumps({"record_count": report["record_count"], "sheet_count": report["sheet_count"]}))
    if args.command == "convert-anyinsertion" or args.command == "all":
        records = convert_anyinsertion(raw_root, anyinsertion_root)
        clean_records = write_anyinsertion_clean_split(anyinsertion_root)
        print(f"Converted AnyInsertion: {len(records)} cases at {anyinsertion_root}")
        print(f"Materialized current-model clean split: {len(clean_records)} cases")
        report = audit_dataset(
            anyinsertion_root,
            strict_900=False,
            strict_anyinsertion=True,
            finalize_checksums=True,
        )
        print(json.dumps(report["counts_major"], sort_keys=True))
    if args.command == "refresh-anyinsertion-metadata":
        records = refresh_anyinsertion_metadata(anyinsertion_root, raw_root=raw_root)
        write_anyinsertion_clean_split(anyinsertion_root)
        report = audit_dataset(
            anyinsertion_root,
            strict_900=False,
            strict_anyinsertion=True,
            finalize_checksums=True,
        )
        print(
            json.dumps(
                {"record_count": len(records), "case_types": report["case_types"]},
                sort_keys=True,
            )
        )
    if args.command == "audit":
        report = audit_dataset(curated_root, strict_900=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "audit-anyinsertion":
        report = audit_dataset(anyinsertion_root, strict_900=False, strict_anyinsertion=True)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "qc-previews-anyinsertion":
        records = read_jsonl(anyinsertion_root / "manifests" / "anyinsertion_official.jsonl")
        report = generate_qc_contact_sheets(anyinsertion_root, records)
        print(json.dumps({"record_count": report["record_count"], "sheet_count": report["sheet_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

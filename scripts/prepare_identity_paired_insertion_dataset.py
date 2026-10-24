#!/usr/bin/env python3
"""Build the identity-paired PaperInsert-900 benchmark.

This builder implements the normal reference-insertion evaluation protocol:
the reference and target are different source images, while the foreground is
the same tracked physical instance (MeViS v2) or the same product identity
(DeepFashion2).  It never derives a reference from target pixels.

Canonical files keep the existing inference contract::

    src_image  clean reference crop made only from the reference source image
    src_mask   exact native reference-instance mask after the same crop
    tar_image  complete target ground-truth image
    tar_mask   exact native target-instance mask

The target hole used by the model is derived from ``tar_mask`` at inference.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import imagehash
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools import mask as mask_utils


DEFAULT_SEED = 20260725
SCHEMA_VERSION = 4
PROTOCOL = "paired_reconstruction"
MASK_THRESHOLD = 127
DEFAULT_WORK_ROOT = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/" "a2edit/eval/paper_insertion_benchmark_v1"
)

MEVIS_ARCHIVE_SHA256 = "bdbd8bd6deb17b10705e43ecba66772e9be6a702ca10306e52aff76f3d4178d2"
MEVIS_META_SHA256 = "134b6e6ca013ea648b3bd71c7f65662d0b660ece932e4c0e9e9c2d1c93fc1d3d"
MEVIS_MASK_DICT_SHA256 = "f9d9b01a2a6d7a004e2e7a664ce2b9b1576218e975030f8e4df4f66111fb253f"
MEVIS_CURATION_POOL_FILENAME = "mevis_diverse_candidate_pool_v5.json"
MEVIS_CURATION_POOL_SCHEMA = "mevis_diverse_candidate_pool_v5"
MEVIS_CURATION_POOL_SHA256 = "7414e68c16f71bd11c0bc37763465146bc4593728e145e1befe04d9eac13a993"
DEEPFASHION_ARCHIVE_SHA256 = "0da4c5eabd8693606a26b8a5d95542d4e467655cb3c7331c85ece1f45a75ca51"
DEEPFASHION_QUERY_SHA256 = "f822c94c77d3b1b2277203700ea16b6f75ac359eb9e66112de9798c69c66be9b"
DEEPFASHION_GALLERY_SHA256 = "dd06bac3eedb8c956df2b12e0f9dd0d28354140fa3f55306b56f0a492c21f11b"

# The MeViS object taxonomy is expression-derived and is finalized only after
# source-capacity analysis.  Every published row also carries its native
# expression evidence, so classification remains reviewable.
MAJOR_MINOR_QUOTAS: dict[str, dict[str, int]] = {
    "object": {
        "companion_animals": 50,
        "wild_or_livestock": 50,
        "birds": 50,
        "aquatic_reptiles": 50,
        "vehicles": 50,
        "daily_sports_objects": 50,
    },
    "person": {
        "sport_dance_exercise": 50,
        "sitting_kneeling_lying": 50,
        "walking_running": 50,
        "social_interaction_care": 50,
        "work_object_handling": 50,
        "standing_gesturing": 50,
    },
    "garment": {
        "short_sleeve_top": 50,
        "long_sleeve_top": 50,
        "outerwear_vest_sling": 50,
        "shorts_trousers": 50,
        "skirt": 50,
        "dress_onepiece": 50,
    },
}

GARMENT_CATEGORY_NAMES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

GARMENT_CATEGORY_TO_MINOR = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "outerwear_vest_sling",
    4: "outerwear_vest_sling",
    5: "outerwear_vest_sling",
    6: "outerwear_vest_sling",
    7: "shorts_trousers",
    8: "shorts_trousers",
    9: "skirt",
    10: "dress_onepiece",
    11: "dress_onepiece",
    12: "dress_onepiece",
    13: "dress_onepiece",
}

# The sub-quotas force all 13 native DeepFashion2 classes to appear while the
# externally visible benchmark remains six balanced garment groups.
GARMENT_NATIVE_SUBQUOTAS: dict[str, dict[int, int]] = {
    "short_sleeve_top": {1: 50},
    "long_sleeve_top": {2: 50},
    "outerwear_vest_sling": {3: 4, 4: 18, 5: 18, 6: 10},
    "shorts_trousers": {7: 25, 8: 25},
    "skirt": {9: 50},
    "dress_onepiece": {10: 13, 11: 12, 12: 13, 13: 12},
}

CASE_PREFIXES = {
    "object": "obj",
    "person": "per",
    "garment": "gar",
}


@dataclass(frozen=True)
class MaskStats:
    width: int
    height: int
    area: int
    area_ratio: float
    bbox_xyxy: tuple[int, int, int, int]
    border_fraction: float
    min_bbox_side: int


@dataclass(frozen=True)
class MevisTrack:
    video_id: str
    anno_id: str
    frame_ids: tuple[str, ...]
    rles: tuple[dict[str, Any] | None, ...]
    expressions: tuple[str, ...]

    @property
    def identity_id(self) -> str:
        return f"mevis_v2/train/{self.video_id}/{self.anno_id}"


@dataclass(frozen=True)
class DeepFashionItem:
    image_id: str
    item_key: str
    source: str
    pair_id: int
    style: int
    category_id: int
    category_name: str
    bbox_xyxy: tuple[int, int, int, int]
    segmentation: tuple[tuple[float, ...], ...]
    scale: int
    occlusion: int
    zoom_in: int
    viewpoint: int
    image_path: Path

    @property
    def identity_id(self) -> str:
        # Official retrieval identity is pair_id + non-zero style.  Category is
        # validated separately rather than silently changing the official key.
        return f"deepfashion2/validation/{self.pair_id}/{self.style}"

    @property
    def annotation_id(self) -> str:
        return f"{self.image_id}/{self.item_key}"


@dataclass
class PairPlan:
    major: str
    minor: str
    leaf_label: str
    source_dataset: str
    pair_kind: str
    identity_scope: str
    identity_id: str
    src_image_path: Path
    tar_image_path: Path
    src_mask_payload: Any
    tar_mask_payload: Any
    src_image_id: str
    tar_image_id: str
    src_ann_id: str
    tar_ann_id: str
    src_upstream: dict[str, Any]
    tar_upstream: dict[str, Any]
    identity_evidence: dict[str, Any]
    classification_evidence: str
    selection_score: float
    difficulty: str
    license_note: str
    extra: dict[str, Any] = field(default_factory=dict)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Python compiles the full module before executing it.  Capturing the source
# digest during module initialization therefore binds caches and reports to the
# code loaded by this process, rather than whatever bytes happen to be on disk
# several minutes later.
LOADED_BUILDER_PATH = Path(__file__).resolve()
LOADED_BUILDER_SHA256 = file_sha256(LOADED_BUILDER_PATH)


def assert_builder_source_unchanged() -> None:
    current = file_sha256(LOADED_BUILDER_PATH)
    if current != LOADED_BUILDER_SHA256:
        raise RuntimeError(
            "Builder source changed while this process was running: "
            f"loaded={LOADED_BUILDER_SHA256} current={current}"
        )


@contextmanager
def exclusive_build_lock(output_root: Path):
    """Prevent concurrent builders from mixing files in one release path."""

    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_root.parent / f".{output_root.name}.build.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise FileExistsError(f"Another build owns {output_root}; lock={lock_path} owner={owner!r}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


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


def image_fingerprints(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    return {
        "src_full_pixel_sha256": decoded_pixel_sha256(rgb),
        "src_full_phash": str(imagehash.phash(rgb, hash_size=8, highfreq_factor=4)),
        "src_full_dhash": str(imagehash.dhash(rgb, hash_size=8)),
        "src_full_width": rgb.width,
        "src_full_height": rgb.height,
    }


def binary_mask(mask: Image.Image) -> Image.Image:
    array = np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(np.where(array > MASK_THRESHOLD, 255, 0).astype(np.uint8), mode="L")


def decode_coco_rle(rle: Mapping[str, Any]) -> Image.Image:
    normalized = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(normalized["counts"], str):
        normalized["counts"] = normalized["counts"].encode("ascii")
    decoded = mask_utils.decode(normalized)
    if decoded.ndim == 3:
        decoded = np.any(decoded > 0, axis=2)
    return Image.fromarray((decoded > 0).astype(np.uint8) * 255, mode="L")


def deepfashion_polygon_to_mask(segmentation: Sequence[Sequence[float]], width: int, height: int) -> Image.Image:
    polygons = [list(map(float, polygon)) for polygon in segmentation]
    if not polygons or any(len(polygon) < 6 or len(polygon) % 2 for polygon in polygons):
        raise ValueError("DeepFashion2 segmentation must contain valid polygons")
    rles = mask_utils.frPyObjects(polygons, height, width)
    decoded = mask_utils.decode(mask_utils.merge(rles))
    return Image.fromarray((decoded > 0).astype(np.uint8) * 255, mode="L")


def mask_stats(mask: Image.Image) -> MaskStats:
    values = np.asarray(binary_mask(mask), dtype=np.uint8) > 0
    ys, xs = np.nonzero(values)
    if not len(xs):
        raise ValueError("Mask is empty")
    left, top = int(xs.min()), int(ys.min())
    right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
    border = int(values[0].sum() + values[-1].sum() + values[:, 0].sum() + values[:, -1].sum())
    area = int(values.sum())
    return MaskStats(
        width=mask.width,
        height=mask.height,
        area=area,
        area_ratio=area / float(mask.width * mask.height),
        bbox_xyxy=(left, top, right, bottom),
        border_fraction=border / float(max(1, area)),
        min_bbox_side=min(right - left, bottom - top),
    )


def crop_clean_reference(
    image: Image.Image,
    mask: Image.Image,
    *,
    margin_ratio: float = 0.15,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop a reference source and whiten only its background.

    This function receives no target data by design.  Foreground RGB and mask
    pixels remain native; there is no resize or synthetic augmentation.
    """

    image = image.convert("RGB")
    mask = binary_mask(mask)
    if image.size != mask.size:
        raise ValueError(f"Reference image/mask size mismatch: {image.size} vs {mask.size}")
    stats = mask_stats(mask)
    left, top, right, bottom = stats.bbox_xyxy
    width, height = right - left, bottom - top
    margin_x = max(8, int(round(width * margin_ratio)))
    margin_y = max(8, int(round(height * margin_ratio)))
    crop_xyxy = (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    )
    crop_image = image.crop(crop_xyxy)
    crop_mask = binary_mask(mask.crop(crop_xyxy))
    clean = Image.composite(crop_image, Image.new("RGB", crop_image.size, "white"), crop_mask)
    return clean, crop_mask, crop_xyxy


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            image.save(handle, format="PNG", optimize=False, compress_level=6)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def runtime_dependency_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("numpy", "Pillow", "pycocotools", "ImageHash", "networkx"):
        try:
            result[package] = package_version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_mevis_tracks(meta_path: Path, mask_dict_path: Path) -> list[MevisTrack]:
    """Load one deduplicated entry per official single-object track."""

    metadata = read_json(meta_path)
    masks = read_json(mask_dict_path)
    tracks: dict[tuple[str, str], dict[str, Any]] = {}
    for video_id, video in metadata["videos"].items():
        frame_ids = tuple(map(str, video["frames"]))
        for expression in video["expressions"].values():
            anno_ids = expression.get("anno_id", [])
            if len(anno_ids) != 1:
                continue
            anno_id = str(anno_ids[0])
            key = (str(video_id), anno_id)
            entry = tracks.setdefault(key, {"frame_ids": frame_ids, "expressions": set()})
            if tuple(entry["frame_ids"]) != frame_ids:
                raise ValueError(f"Conflicting frame list for MeViS track {key}")
            text = " ".join(str(expression.get("exp", "")).split())
            if text:
                entry["expressions"].add(text)
    result: list[MevisTrack] = []
    for (video_id, anno_id), entry in sorted(tracks.items()):
        raw_rles = masks.get(anno_id)
        if raw_rles is None:
            raise ValueError(f"MeViS mask_dict is missing anno_id={anno_id}")
        if len(raw_rles) != len(entry["frame_ids"]):
            raise ValueError(
                f"MeViS RLE/frame length mismatch for {video_id}/{anno_id}: "
                f"{len(raw_rles)} vs {len(entry['frame_ids'])}"
            )
        result.append(
            MevisTrack(
                video_id=video_id,
                anno_id=anno_id,
                frame_ids=tuple(entry["frame_ids"]),
                rles=tuple(raw_rles),
                expressions=tuple(sorted(entry["expressions"])),
            )
        )
    return result


def _bbox_tuple(value: Sequence[Any]) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError(f"Expected four bbox values, got {value!r}")
    return tuple(int(round(float(part))) for part in value)  # type: ignore[return-value]


def _deepfashion_item_from_annotation(
    *, image_id: str, item_key: str, annotation: Mapping[str, Any], root: Path, source: str, pair_id: int
) -> DeepFashionItem:
    return DeepFashionItem(
        image_id=image_id,
        item_key=item_key,
        source=source,
        pair_id=pair_id,
        style=int(annotation["style"]),
        category_id=int(annotation["category_id"]),
        category_name=str(annotation.get("category_name", "")),
        bbox_xyxy=_bbox_tuple(annotation["bounding_box"]),
        segmentation=tuple(tuple(map(float, polygon)) for polygon in annotation["segmentation"]),
        scale=int(annotation.get("scale", 0)),
        occlusion=int(annotation.get("occlusion", 0)),
        zoom_in=int(annotation.get("zoom_in", 0)),
        viewpoint=int(annotation.get("viewpoint", 0)),
        image_path=root / "validation" / "image" / f"{image_id}.jpg",
    )


def load_deepfashion_items(root: Path) -> dict[str, list[DeepFashionItem]]:
    """Index every validation annotation by zero-padded image ID."""

    annotation_root = root / "validation" / "annos"
    image_root = root / "validation" / "image"
    if not annotation_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError(f"DeepFashion2 validation tree is incomplete under {root}")
    result: dict[str, list[DeepFashionItem]] = {}
    for annotation_path in sorted(annotation_root.glob("*.json")):
        image_id = annotation_path.stem
        image_path = image_root / f"{image_id}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing DeepFashion2 image: {image_path}")
        payload = read_json(annotation_path)
        source = str(payload.get("source", ""))
        pair_id = int(payload.get("pair_id", -1))
        items = []
        for item_key in sorted(key for key in payload if key.startswith("item")):
            items.append(
                _deepfashion_item_from_annotation(
                    image_id=image_id,
                    item_key=item_key,
                    annotation=payload[item_key],
                    root=root,
                    source=source,
                    pair_id=pair_id,
                )
            )
        result[image_id] = items
    return result


def _match_deepfashion_metadata_item(
    candidates: Sequence[DeepFashionItem],
    *,
    pair_id: int,
    style: int,
    bbox: Sequence[Any],
    category_id: int | None = None,
) -> DeepFashionItem:
    expected_bbox = _bbox_tuple(bbox)
    matches = [
        item
        for item in candidates
        if item.pair_id == pair_id
        and item.style == style
        and item.bbox_xyxy == expected_bbox
        and (category_id is None or item.category_id == category_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            "DeepFashion2 metadata did not resolve to exactly one item: "
            f"pair={pair_id} style={style} bbox={expected_bbox} category={category_id} "
            f"matches={len(matches)}"
        )
    return matches[0]


def load_deepfashion_pairs(root: Path) -> dict[str, tuple[list[DeepFashionItem], list[DeepFashionItem]]]:
    """Return official user/shop positive-pair pools keyed by pair_id/style.

    Query/gallery metadata is used as the authority.  ``matching_pairs.json``
    is intentionally ignored because that local file references the train split.
    """

    items_by_image = load_deepfashion_items(root)
    metadata_root = root / "json_for_validation"
    query_rows = read_json(metadata_root / "val_query.json")
    gallery_rows = read_json(metadata_root / "val_gallery.json")
    grouped: dict[str, tuple[list[DeepFashionItem], list[DeepFashionItem]]] = {}

    def pools(key: str) -> tuple[list[DeepFashionItem], list[DeepFashionItem]]:
        return grouped.setdefault(key, ([], []))

    for row in query_rows:
        pair_id, style = int(row["pair_id"]), int(row["style"])
        if style <= 0:
            continue
        image_id = f"{int(row['query_image_id']):06d}"
        item = _match_deepfashion_metadata_item(
            items_by_image[image_id], pair_id=pair_id, style=style, bbox=row["bbox"], category_id=int(row["cls"])
        )
        if item.source != "user":
            raise ValueError(f"DeepFashion query is not user-domain: {item}")
        pools(f"{pair_id}/{style}")[0].append(item)

    for row in gallery_rows:
        pair_id, style = int(row["pair_id"]), int(row["style"])
        if style <= 0:
            continue
        image_id = f"{int(row['gallery_image_id']):06d}"
        item = _match_deepfashion_metadata_item(
            items_by_image[image_id], pair_id=pair_id, style=style, bbox=row["bbox"]
        )
        if item.source != "shop":
            raise ValueError(f"DeepFashion gallery is not shop-domain: {item}")
        pools(f"{pair_id}/{style}")[1].append(item)

    return {key: (users, shops) for key, (users, shops) in grouped.items() if users and shops}


def validate_identity_evidence(record: Mapping[str, Any]) -> list[str]:
    """Recompute protocol identity invariants instead of trusting booleans."""

    errors: list[str] = []
    case_id = str(record.get("case_id", "<unknown>"))
    if record.get("manifest_schema_version") != SCHEMA_VERSION:
        errors.append(f"{case_id}: manifest schema must be {SCHEMA_VERSION}")
    if record.get("protocol") != PROTOCOL:
        errors.append(f"{case_id}: protocol must be {PROTOCOL}")
    if record.get("case_type") != "paired_gt":
        errors.append(f"{case_id}: case_type must be paired_gt")
    if record.get("same_identity") is not True or record.get("gt_available") is not True:
        errors.append(f"{case_id}: same_identity and gt_available must both be true")
    if record.get("reference_derived_from_target") is not False:
        errors.append(f"{case_id}: reference_derived_from_target must be false")
    src = record.get("src_upstream")
    tar = record.get("tar_upstream")
    if not isinstance(src, Mapping) or not isinstance(tar, Mapping):
        return errors + [f"{case_id}: src_upstream/tar_upstream must be objects"]
    if str(record.get("src_image_id", "")) == str(record.get("tar_image_id", "")):
        errors.append(f"{case_id}: src and target must be different source images")
    pair_kind = record.get("pair_kind")
    if pair_kind == "same_track_cross_frame":
        keys = ("video_id", "anno_id")
        if any(str(src.get(key)) != str(tar.get(key)) for key in keys):
            errors.append(f"{case_id}: MeViS video_id/anno_id identity mismatch")
        if str(src.get("frame_id")) == str(tar.get("frame_id")):
            errors.append(f"{case_id}: MeViS reference and target frames must differ")
        expected = f"mevis_v2/train/{src.get('video_id')}/{src.get('anno_id')}"
        expected_fields = {
            "source_dataset": "mevis_v2",
            "src_source_dataset": "mevis_v2",
            "source_split": "train",
            "src_source_split": "train",
            "identity_scope": "physical_instance",
            "identity_evidence_status": "verified_upstream_annotation",
            "src_image_id": f"train/{src.get('video_id')}/{src.get('frame_id')}",
            "tar_image_id": f"train/{tar.get('video_id')}/{tar.get('frame_id')}",
            "src_ann_id": str(src.get("anno_id")),
            "tar_ann_id": str(tar.get("anno_id")),
        }
        expected_evidence_type = "official_tracked_instance_id"
    elif pair_kind == "same_product_cross_domain":
        if src.get("domain") != "shop" or tar.get("domain") != "user":
            errors.append(f"{case_id}: DeepFashion direction must be shop -> user")
        if int(src.get("style", 0)) <= 0 or int(tar.get("style", 0)) <= 0:
            errors.append(f"{case_id}: DeepFashion style must be non-zero")
        if any(str(src.get(key)) != str(tar.get(key)) for key in ("pair_id", "style")):
            errors.append(f"{case_id}: DeepFashion pair_id/style identity mismatch")
        if int(src.get("category_id", -1)) != int(tar.get("category_id", -2)):
            errors.append(f"{case_id}: DeepFashion category_id sanity check failed")
        expected = f"deepfashion2/validation/{src.get('pair_id')}/{src.get('style')}"
        category_id = int(src.get("category_id", -1))
        expected_fields = {
            "source_dataset": "deepfashion2",
            "src_source_dataset": "deepfashion2",
            "source_split": "validation",
            "src_source_split": "validation",
            "identity_scope": "product_identity",
            "identity_evidence_status": "verified_upstream_annotation",
            "src_image_id": f"validation/{src.get('image_id')}",
            "tar_image_id": f"validation/{tar.get('image_id')}",
            "src_ann_id": f"{src.get('image_id')}/{src.get('item_key')}",
            "tar_ann_id": f"{tar.get('image_id')}/{tar.get('item_key')}",
            "native_category_id": category_id,
            "minor": GARMENT_CATEGORY_TO_MINOR.get(category_id),
        }
        expected_evidence_type = "official_consumer_shop_product_pair"
    else:
        errors.append(f"{case_id}: unsupported pair_kind={pair_kind!r}")
        return errors
    if record.get("identity_id") != expected:
        errors.append(f"{case_id}: identity_id mismatch actual={record.get('identity_id')!r} " f"expected={expected!r}")
    if record.get("src_identity_id") != expected or record.get("tar_identity_id") != expected:
        errors.append(f"{case_id}: src/tar identity IDs do not match upstream evidence")
    for field_name, expected_value in expected_fields.items():
        if record.get(field_name) != expected_value:
            errors.append(
                f"{case_id}: {field_name} mismatch actual={record.get(field_name)!r} " f"expected={expected_value!r}"
            )
    evidence = record.get("identity_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("type") != expected_evidence_type:
        errors.append(f"{case_id}: identity evidence type mismatch")
    return errors


def _stable_tiebreak(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _deepfashion_metadata_quality(item: DeepFashionItem) -> tuple[bool, float]:
    left, top, right, bottom = item.bbox_xyxy
    min_side = min(right - left, bottom - top)
    hard_pass = (
        item.style > 0
        and item.occlusion == 1
        and item.scale >= 2
        and 0 < item.zoom_in <= 2
        and min_side >= 64
        and item.category_id in GARMENT_CATEGORY_TO_MINOR
    )
    score = (
        2.0 * float(item.occlusion == 1)
        + 1.2 * float(item.scale >= 2)
        + 0.8 * float(item.zoom_in == 1)
        + 0.5 * float(item.viewpoint == 2)
        + min(1.0, min_side / 240.0)
    )
    return hard_pass, score


def _deepfashion_item_geometry(item: DeepFashionItem) -> tuple[Image.Image, MaskStats]:
    with Image.open(item.image_path) as handle:
        width, height = handle.size
    mask = deepfashion_polygon_to_mask(item.segmentation, width, height)
    return mask, mask_stats(mask)


def _deepfashion_geometry_score(stats: MaskStats, *, target: bool) -> float:
    ideal = 0.16 if target else 0.28
    area_term = max(0.0, 1.0 - abs(math.log(max(stats.area_ratio, 1e-8) / ideal)) / 2.2)
    side_term = min(1.0, stats.min_bbox_side / 220.0)
    border_term = min(1.0, stats.border_fraction * 50.0)
    return 2.0 * area_term + side_term - 1.2 * border_term


def deepfashion_positive_pair(src: DeepFashionItem, tar: DeepFashionItem) -> bool:
    return (
        src.source == "shop"
        and tar.source == "user"
        and src.image_id != tar.image_id
        and src.pair_id == tar.pair_id
        and src.style == tar.style
        and src.style > 0
        and src.category_id == tar.category_id
    )


def build_deepfashion_candidate_plans(root: Path, seed: int) -> list[PairPlan]:
    """Resolve the best conservative shop/user pair for every usable identity/item."""

    grouped = load_deepfashion_pairs(root)
    geometry_cache: dict[str, tuple[Image.Image, MaskStats]] = {}

    def geometry(item: DeepFashionItem) -> tuple[Image.Image, MaskStats]:
        key = item.annotation_id
        cached = geometry_cache.get(key)
        if cached is None:
            cached = _deepfashion_item_geometry(item)
            geometry_cache[key] = cached
        return cached

    result: list[PairPlan] = []
    for identity_key, (users, shops) in sorted(grouped.items()):
        users_by_category: dict[int, list[DeepFashionItem]] = defaultdict(list)
        shops_by_category: dict[int, list[DeepFashionItem]] = defaultdict(list)
        for item in users:
            if _deepfashion_metadata_quality(item)[0]:
                users_by_category[item.category_id].append(item)
        for item in shops:
            if _deepfashion_metadata_quality(item)[0]:
                shops_by_category[item.category_id].append(item)
        for category_id in sorted(set(users_by_category) & set(shops_by_category)):
            # Metadata quality narrows each identity to a small set before the
            # more expensive polygon rasterization and image-size checks.
            user_pool = sorted(
                users_by_category[category_id],
                key=lambda item: (
                    -_deepfashion_metadata_quality(item)[1],
                    _stable_tiebreak(seed, identity_key, item.annotation_id, "user"),
                ),
            )[:4]
            shop_pool = sorted(
                shops_by_category[category_id],
                key=lambda item: (
                    -_deepfashion_metadata_quality(item)[1],
                    _stable_tiebreak(seed, identity_key, item.annotation_id, "shop"),
                ),
            )[:4]
            best: tuple[float, DeepFashionItem, DeepFashionItem, MaskStats, MaskStats] | None = None
            for tar in user_pool:
                tar_mask, tar_stats = geometry(tar)
                if not (0.018 <= tar_stats.area_ratio <= 0.62 and tar_stats.min_bbox_side >= 64):
                    continue
                for src in shop_pool:
                    if not deepfashion_positive_pair(src, tar):
                        continue
                    src_mask, src_stats = geometry(src)
                    if not (0.02 <= src_stats.area_ratio <= 0.80 and src_stats.min_bbox_side >= 64):
                        continue
                    score = (
                        _deepfashion_metadata_quality(src)[1]
                        + _deepfashion_metadata_quality(tar)[1]
                        + _deepfashion_geometry_score(src_stats, target=False)
                        + _deepfashion_geometry_score(tar_stats, target=True)
                        - 0.25 * abs(math.log(max(src_stats.area_ratio, 1e-8) / tar_stats.area_ratio))
                    )
                    candidate = (score, src, tar, src_stats, tar_stats)
                    if best is None or (score, src.annotation_id, tar.annotation_id) > (
                        best[0],
                        best[1].annotation_id,
                        best[2].annotation_id,
                    ):
                        best = candidate
            if best is None:
                continue
            score, src, tar, src_stats, tar_stats = best
            identity_id = src.identity_id
            if identity_id != tar.identity_id:
                raise AssertionError("DeepFashion identity mismatch after positive-pair validation")
            minor = GARMENT_CATEGORY_TO_MINOR[category_id]
            result.append(
                PairPlan(
                    major="garment",
                    minor=minor,
                    leaf_label=GARMENT_CATEGORY_NAMES[category_id],
                    source_dataset="deepfashion2",
                    pair_kind="same_product_cross_domain",
                    identity_scope="product_identity",
                    identity_id=identity_id,
                    src_image_path=src.image_path,
                    tar_image_path=tar.image_path,
                    src_mask_payload=src.segmentation,
                    tar_mask_payload=tar.segmentation,
                    src_image_id=f"validation/{src.image_id}",
                    tar_image_id=f"validation/{tar.image_id}",
                    src_ann_id=src.annotation_id,
                    tar_ann_id=tar.annotation_id,
                    src_upstream={
                        "image_id": src.image_id,
                        "item_key": src.item_key,
                        "pair_id": src.pair_id,
                        "style": src.style,
                        "category_id": src.category_id,
                        "domain": "shop",
                    },
                    tar_upstream={
                        "image_id": tar.image_id,
                        "item_key": tar.item_key,
                        "pair_id": tar.pair_id,
                        "style": tar.style,
                        "category_id": tar.category_id,
                        "domain": "user",
                    },
                    identity_evidence={
                        "type": "official_consumer_shop_product_pair",
                        "key_fields": ["pair_id", "style"],
                        "category_id_sanity_check": True,
                    },
                    classification_evidence=f"native_category_id={category_id}",
                    selection_score=round(score, 8),
                    difficulty=(
                        "easy" if src_stats.border_fraction == 0 and tar_stats.border_fraction == 0 else "moderate"
                    ),
                    license_note=(
                        "DeepFashion2 research dataset; official access terms and original "
                        "image rights apply, with no open redistribution grant assumed"
                    ),
                    extra={
                        "native_category_id": category_id,
                        "src_native_bbox_xyxy": list(src.bbox_xyxy),
                        "tar_native_bbox_xyxy": list(tar.bbox_xyxy),
                        "src_native_area_ratio": round(src_stats.area_ratio, 8),
                        "tar_native_area_ratio": round(tar_stats.area_ratio, 8),
                        "src_quality": {
                            "occlusion": src.occlusion,
                            "scale": src.scale,
                            "zoom_in": src.zoom_in,
                            "viewpoint": src.viewpoint,
                        },
                        "tar_quality": {
                            "occlusion": tar.occlusion,
                            "scale": tar.scale,
                            "zoom_in": tar.zoom_in,
                            "viewpoint": tar.viewpoint,
                        },
                    },
                )
            )
    return result


def select_deepfashion_plans(candidates: Sequence[PairPlan], seed: int) -> list[PairPlan]:
    by_category: dict[int, list[PairPlan]] = defaultdict(list)
    for plan in candidates:
        by_category[int(plan.extra["native_category_id"])].append(plan)
    for category_id in by_category:
        by_category[category_id].sort(
            key=lambda plan: (
                -plan.selection_score,
                _stable_tiebreak(seed, plan.identity_id, plan.src_image_id, plan.tar_image_id),
            )
        )

    selected: list[PairPlan] = []
    used_identities: set[str] = set()
    used_src_images: set[str] = set()
    used_tar_images: set[str] = set()
    for minor in MAJOR_MINOR_QUOTAS["garment"]:
        subquotas = GARMENT_NATIVE_SUBQUOTAS[minor]
        for category_id, quota in subquotas.items():
            accepted = 0
            for plan in by_category.get(category_id, []):
                if (
                    plan.identity_id in used_identities
                    or plan.src_image_id in used_src_images
                    or plan.tar_image_id in used_tar_images
                ):
                    continue
                selected.append(plan)
                used_identities.add(plan.identity_id)
                used_src_images.add(plan.src_image_id)
                used_tar_images.add(plan.tar_image_id)
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise ValueError(
                    f"DeepFashion2 native category {category_id} underfilled: "
                    f"selected={accepted} required={quota} candidates={len(by_category.get(category_id, []))}"
                )
    counts = Counter(plan.minor for plan in selected)
    expected = MAJOR_MINOR_QUOTAS["garment"]
    if counts != Counter(expected):
        raise AssertionError(f"DeepFashion garment quota mismatch: {counts} vs {expected}")
    return selected


def _mask_from_plan_payload(plan: PairPlan, *, source: bool, image_size: tuple[int, int]) -> Image.Image:
    payload = plan.src_mask_payload if source else plan.tar_mask_payload
    if plan.source_dataset == "mevis_v2":
        return decode_coco_rle(payload)
    if plan.source_dataset == "deepfashion2":
        return deepfashion_polygon_to_mask(payload, image_size[0], image_size[1])
    raise ValueError(f"Unsupported source dataset: {plan.source_dataset}")


def materialize_case(plan: PairPlan, case_id: str, output_root: Path) -> dict[str, Any]:
    """Write one four-file case, reading source and target independently."""

    if plan.src_image_path.resolve() == plan.tar_image_path.resolve():
        raise ValueError(f"{case_id}: source and target paths must differ")
    with Image.open(plan.src_image_path) as handle:
        src_full = handle.convert("RGB")
    with Image.open(plan.tar_image_path) as handle:
        tar_image = handle.convert("RGB")
    src_native_mask = _mask_from_plan_payload(plan, source=True, image_size=src_full.size)
    tar_mask = _mask_from_plan_payload(plan, source=False, image_size=tar_image.size)
    if src_native_mask.size != src_full.size or tar_mask.size != tar_image.size:
        raise ValueError(f"{case_id}: native image/mask geometry mismatch")
    src_image, src_mask, crop_xyxy = crop_clean_reference(src_full, src_native_mask)
    src_stats = mask_stats(src_mask)
    tar_stats = mask_stats(tar_mask)
    if src_stats.area != mask_stats(src_native_mask).area:
        raise AssertionError(f"{case_id}: reference crop changed native mask area")
    if not (0.01 <= tar_stats.area_ratio <= 0.70):
        raise ValueError(f"{case_id}: target mask area ratio is unsuitable: {tar_stats.area_ratio}")

    relative_paths = {
        role: (Path(plan.major) / role / f"{case_id}.png").as_posix()
        for role in ("src_image", "src_mask", "tar_image", "tar_mask")
    }
    for role, image in (
        ("src_image", src_image),
        ("src_mask", src_mask),
        ("tar_image", tar_image),
        ("tar_mask", tar_mask),
    ):
        atomic_save_png(image, output_root / relative_paths[role])

    record: dict[str, Any] = {
        "manifest_schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "major": plan.major,
        "minor": plan.minor,
        "leaf_label": plan.leaf_label,
        "protocol": PROTOCOL,
        "pair_kind": plan.pair_kind,
        "case_type": "paired_gt",
        "same_identity": True,
        "gt_available": True,
        "reference_derived_from_target": False,
        "identity_scope": plan.identity_scope,
        "identity_id": plan.identity_id,
        "src_identity_id": plan.identity_id,
        "tar_identity_id": plan.identity_id,
        "identity_evidence_status": "verified_upstream_annotation",
        "identity_evidence": plan.identity_evidence,
        "src_upstream": plan.src_upstream,
        "tar_upstream": plan.tar_upstream,
        "src_source_dataset": plan.source_dataset,
        "source_dataset": plan.source_dataset,
        "src_source_split": "train" if plan.source_dataset == "mevis_v2" else "validation",
        "source_split": "train" if plan.source_dataset == "mevis_v2" else "validation",
        "src_image_id": plan.src_image_id,
        "tar_image_id": plan.tar_image_id,
        "src_ann_id": plan.src_ann_id,
        "tar_ann_id": plan.tar_ann_id,
        "classification_evidence": plan.classification_evidence,
        "selection_score": plan.selection_score,
        "difficulty": plan.difficulty,
        "mask_origin_src": "native_instance_annotation",
        "mask_origin_tar": "native_instance_annotation",
        "src_is_derived_crop": True,
        "src_crop_xyxy": list(crop_xyxy),
        "src_crop_context_ratio": 0.15,
        "src_background_policy": "white_outside_native_mask",
        "src_transform_version": "native_bbox_crop_white_v1",
        "src_area_ratio": round(src_stats.area_ratio, 8),
        "tar_area_ratio": round(tar_stats.area_ratio, 8),
        "src_native_area": mask_stats(src_native_mask).area,
        "tar_native_area": tar_stats.area,
        "src_pixel_sha256": decoded_pixel_sha256(src_image),
        "src_mask_pixel_sha256": decoded_mask_sha256(src_mask),
        "tar_pixel_sha256": decoded_pixel_sha256(tar_image),
        "tar_mask_pixel_sha256": decoded_mask_sha256(tar_mask),
        **image_fingerprints(src_full),
        **relative_paths,
        "auto_quality_pass": True,
        "showcase_candidate": False,
        "agent_paper_avoid": False,
        "agent_paper_avoid_roles": [],
        "agent_paper_avoid_reasons": {},
        "agent_visual_qc_status": "pending",
        "manual_qc_status": "pending",
        "paper_ok": False,
        "license_note": plan.license_note,
    }
    reserved_extra_fields = sorted(set(record).intersection(plan.extra))
    if reserved_extra_fields:
        raise ValueError(
            f"{case_id}: plan.extra attempted to override reserved manifest fields: " f"{reserved_extra_fields}"
        )
    record.update(plan.extra)
    errors = validate_identity_evidence(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


PERSON_SUBJECT_WORDS = {
    "person",
    "human",
    "man",
    "woman",
    "lady",
    "guy",
    "boy",
    "girl",
    "baby",
    "child",
    "kid",
    "adult",
    "player",
    "worker",
    "driver",
    "rider",
    "dancer",
    "athlete",
    "skier",
    "surfer",
    "individual",
    "gentleman",
    "female",
    "male",
    "people",
}

OBJECT_SUBJECT_WORDS: dict[str, set[str]] = {
    "companion_animals": {
        "dog",
        "puppy",
        "cat",
        "kitten",
        "rabbit",
        "bunny",
        "hamster",
    },
    "wild_or_livestock": {
        "horse",
        "pony",
        "donkey",
        "cow",
        "cattle",
        "bull",
        "calf",
        "sheep",
        "lamb",
        "goat",
        "pig",
        "hog",
        "camel",
        "llama",
        "alpaca",
        "bear",
        "panda",
        "monkey",
        "ape",
        "gorilla",
        "chimpanzee",
        "tiger",
        "lion",
        "leopard",
        "cheetah",
        "elephant",
        "giraffe",
        "zebra",
        "deer",
        "kangaroo",
        "fox",
        "wolf",
        "raccoon",
        "squirrel",
        "hippo",
        "hippopotamus",
        "rhino",
        "rhinoceros",
        "koala",
        "sloth",
        "otter",
    },
    "birds": {
        "bird",
        "avian",
        "duck",
        "goose",
        "chicken",
        "hen",
        "rooster",
        "pigeon",
        "parrot",
        "eagle",
        "owl",
        "swan",
        "penguin",
        "ostrich",
        "peacock",
        "flamingo",
        "seagull",
        "gull",
        "turkey",
        "pelican",
        "falcon",
    },
    "aquatic_reptiles": {
        "fish",
        "goldfish",
        "salmon",
        "trout",
        "koi",
        "pufferfish",
        "shark",
        "whale",
        "dolphin",
        "seal",
        "walrus",
        "turtle",
        "tortoise",
        "lizard",
        "snake",
        "crocodile",
        "alligator",
        "frog",
        "toad",
        "octopus",
        "crab",
        "lobster",
        "shrimp",
        "ray",
        "manta",
        "stingray",
        "eel",
        "jellyfish",
        "starfish",
        "seahorse",
        "salamander",
        "reptile",
    },
    "vehicles": {
        "car",
        "automobile",
        "truck",
        "bus",
        "van",
        "jeep",
        "bicycle",
        "bike",
        "motorcycle",
        "motorbike",
        "scooter",
        "boat",
        "ship",
        "ferry",
        "kayak",
        "canoe",
        "aircraft",
        "airplane",
        "plane",
        "helicopter",
        "train",
        "tractor",
    },
    "daily_sports_objects": {
        "ball",
        "football",
        "basketball",
        "volleyball",
        "frisbee",
        "racket",
        "skateboard",
        "surfboard",
        "snowboard",
        "ski",
        "skis",
        "kite",
        "barbell",
        "dumbbell",
        "bat",
        "club",
        "goal",
        "chair",
        "bench",
        "stool",
        "sofa",
        "couch",
        "table",
        "backpack",
        "bag",
        "handbag",
        "suitcase",
        "umbrella",
        "bottle",
        "cup",
        "mug",
        "bowl",
        "phone",
        "cellphone",
        "camera",
        "laptop",
        "computer",
        "monitor",
        "drone",
        "robot",
        "toy",
        "doll",
        "wheelchair",
        "stroller",
        "cart",
        "guitar",
        "violin",
        "drum",
        "instrument",
        "piano",
        "shovel",
        "hammer",
        "tool",
        "parachute",
        "flag",
        "book",
        "clock",
    },
}

PERSON_ACTION_WORDS: dict[str, set[str]] = {
    "sport_dance_exercise": {
        "dance",
        "dancing",
        "football",
        "soccer",
        "basketball",
        "tennis",
        "baseball",
        "volleyball",
        "ski",
        "skiing",
        "snowboard",
        "surf",
        "surfing",
        "skate",
        "skating",
        "exercise",
        "exercising",
        "gym",
        "workout",
        "acrobat",
        "golf",
        "boxing",
        "swimming",
        "swim",
        "sport",
        "athlete",
        "player",
    },
    "sitting_kneeling_lying": {
        "sit",
        "sits",
        "sitting",
        "seated",
        "kneel",
        "kneeling",
        "crouch",
        "crouching",
        "squat",
        "squatting",
        "lying",
        "laying",
        "prone",
        "reclining",
    },
    "walking_running": {
        "walk",
        "walks",
        "walking",
        "run",
        "runs",
        "running",
        "jog",
        "jogging",
        "climb",
        "climbing",
        "hike",
        "hiking",
        "march",
        "marching",
        "strolling",
    },
    "social_interaction_care": {
        "hug",
        "hugging",
        "kiss",
        "kissing",
        "talk",
        "talking",
        "handshake",
        "help",
        "helping",
        "feeding",
        "caring",
        "together",
        "greeting",
        "embracing",
        "chatting",
        "conversation",
        "leading",
        "following",
        "playing",
        "interacting",
        "touching",
        "watching",
        "fighting",
        "battling",
        "chasing",
        "hands",
    },
    "work_object_handling": {
        "holding",
        "hold",
        "carrying",
        "carry",
        "lifting",
        "pushing",
        "pulling",
        "operating",
        "working",
        "work",
        "cooking",
        "cleaning",
        "driving",
        "painting",
        "cutting",
        "using",
        "throwing",
        "catching",
        "digging",
        "washing",
    },
    "standing_gesturing": {
        "standing",
        "stand",
        "stands",
        "gesturing",
        "gesture",
        "waving",
        "wave",
        "pointing",
        "posing",
        "pose",
        "turning",
        "looking",
        "facing",
        "bowing",
    },
}


def _tokens(text: str) -> list[str]:
    token = []
    result = []
    for character in text.lower():
        if "a" <= character <= "z":
            token.append(character)
        elif token:
            result.append("".join(token))
            token = []
    if token:
        result.append("".join(token))
    return result


def mevis_subject_votes(track: MevisTrack) -> Counter[str]:
    """Vote by the first explicit subject noun in each paraphrased expression."""

    word_to_group = {word: group for group, words in OBJECT_SUBJECT_WORDS.items() for word in words}
    votes: Counter[str] = Counter()
    for expression in track.expressions:
        candidates: list[tuple[int, str]] = []
        for index, word in enumerate(_tokens(expression)):
            variants = [word]
            if word.endswith("ies") and len(word) > 3:
                variants.append(word[:-3] + "y")
            if word.endswith("es") and len(word) > 3:
                variants.extend((word[:-2], word[:-1]))
            if word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
                variants.append(word[:-1])
            if any(variant in PERSON_SUBJECT_WORDS for variant in variants):
                candidates.append((index, "person"))
            object_group = next(
                (word_to_group[variant] for variant in variants if variant in word_to_group),
                None,
            )
            if object_group is not None:
                candidates.append((index, object_group))
        if candidates:
            candidates.sort(key=lambda row: (row[0], row[1]))
            votes[candidates[0][1]] += 1
    return votes


def classify_mevis_track(track: MevisTrack) -> dict[tuple[str, str], float]:
    """Return reviewable major/minor eligibility scores for a track."""

    subject_votes = mevis_subject_votes(track)
    if not subject_votes:
        return {}
    subject, vote_count = subject_votes.most_common(1)[0]
    recognized = sum(subject_votes.values())
    second = subject_votes.most_common(2)[1][1] if len(subject_votes) > 1 else 0
    if vote_count < 2 or vote_count / recognized < 0.55 or vote_count <= second:
        return {}
    subject_confidence = vote_count / max(1, len(track.expressions))
    if subject != "person":
        return {("object", subject): 5.0 + 3.0 * subject_confidence}

    action_counts: Counter[str] = Counter()
    for expression in track.expressions:
        words = set(_tokens(expression))
        for minor, keywords in PERSON_ACTION_WORDS.items():
            if words & keywords:
                action_counts[minor] += 1
    eligible: dict[tuple[str, str], float] = {}
    for minor, count in action_counts.items():
        if count >= 2:
            eligible[("person", minor)] = 4.0 + 3.0 * subject_confidence + 2.0 * (count / len(track.expressions))
    return eligible


@dataclass(frozen=True)
class MevisFrameInfo:
    index: int
    frame_id: str
    rle: dict[str, Any]
    width: int
    height: int
    area_ratio: float
    bbox_xywh: tuple[float, float, float, float]
    aspect_ratio: float
    solidity: float
    margin_ratio: float
    geometry_score: float


def _mevis_frame_info(
    index: int,
    frame_id: str,
    rle: Mapping[str, Any],
    *,
    relaxed_daily_object: bool = False,
) -> MevisFrameInfo | None:
    normalized = {"size": list(rle["size"]), "counts": rle["counts"]}
    if isinstance(normalized["counts"], str):
        normalized["counts"] = normalized["counts"].encode("ascii")
    height, width = map(int, normalized["size"])
    area = float(mask_utils.area(normalized))
    x, y, bbox_width, bbox_height = map(float, mask_utils.toBbox(normalized))
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
    minimum_short_side = 0.05 if relaxed_daily_object else 0.06
    minimum_solidity = 0.06 if relaxed_daily_object else 0.07
    minimum_margin = 0.0 if relaxed_daily_object else 0.003
    if not (
        0.025 <= area_ratio <= 0.55
        and short_side_ratio >= minimum_short_side
        and solidity >= minimum_solidity
        and margin_ratio >= minimum_margin
    ):
        return None
    ideal = 0.16
    area_score = max(0.0, 1.0 - abs(math.log(area_ratio / ideal)) / 2.2)
    geometry_score = (
        2.2 * area_score + min(1.0, short_side_ratio / 0.22) + min(1.0, solidity / 0.45) + min(1.0, margin_ratio / 0.04)
    )
    return MevisFrameInfo(
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


def _image_difference(left: Image.Image, right: Image.Image) -> float:
    left_small = np.asarray(left.convert("RGB").resize((64, 64)), dtype=np.float32)
    right_small = np.asarray(right.convert("RGB").resize((64, 64)), dtype=np.float32)
    return float(np.mean(np.abs(left_small - right_small)))


def _image_sharpness(image: Image.Image) -> float:
    values = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    gradient = np.concatenate((np.diff(values, axis=0).ravel(), np.diff(values, axis=1).ravel()))
    return float(np.var(gradient))


def select_frame_pair(
    track: MevisTrack,
    image_root: Path,
    seed: int,
    *,
    relaxed_daily_object: bool = False,
) -> tuple[float, MevisFrameInfo, MevisFrameInfo, dict[str, Any]] | None:
    frame_infos = [
        info
        for index, (frame_id, rle) in enumerate(zip(track.frame_ids, track.rles))
        if rle is not None
        for info in [
            _mevis_frame_info(
                index,
                frame_id,
                rle,
                relaxed_daily_object=relaxed_daily_object,
            )
        ]
        if info is not None
    ]
    if len(frame_infos) < 2:
        return None
    # Keep high-quality frames but also sample all temporal regions so that a
    # static middle segment cannot crowd out a genuinely different view.
    frame_infos.sort(
        key=lambda info: (
            -info.geometry_score,
            _stable_tiebreak(seed, track.identity_id, info.frame_id),
        )
    )
    shortlisted = frame_infos[:32]
    if len(frame_infos) > 32:
        temporal = sorted(frame_infos, key=lambda info: info.index)
        for position in np.linspace(0, len(temporal) - 1, num=16, dtype=int):
            candidate = temporal[int(position)]
            if candidate not in shortlisted:
                shortlisted.append(candidate)

    image_cache: dict[int, Image.Image] = {}
    phash_cache: dict[int, imagehash.ImageHash] = {}
    sharpness_cache: dict[int, float] = {}
    small_cache: dict[int, np.ndarray] = {}
    foreground_small_cache: dict[int, np.ndarray] = {}
    foreground_phash_cache: dict[int, imagehash.ImageHash] = {}

    def image_for(info: MevisFrameInfo) -> Image.Image:
        if info.index not in image_cache:
            path = image_root / track.video_id / f"{info.frame_id}.jpg"
            if not path.is_file():
                raise FileNotFoundError(f"Missing MeViS frame: {path}")
            with Image.open(path) as handle:
                image_cache[info.index] = handle.convert("RGB")
        return image_cache[info.index]

    def phash_for(info: MevisFrameInfo) -> imagehash.ImageHash:
        if info.index not in phash_cache:
            phash_cache[info.index] = imagehash.phash(image_for(info), hash_size=8, highfreq_factor=4)
        return phash_cache[info.index]

    def sharpness_for(info: MevisFrameInfo) -> float:
        if info.index not in sharpness_cache:
            sharpness_cache[info.index] = _image_sharpness(image_for(info))
        return sharpness_cache[info.index]

    def small_for(info: MevisFrameInfo) -> np.ndarray:
        if info.index not in small_cache:
            small_cache[info.index] = np.asarray(
                image_for(info).resize((64, 64), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        return small_cache[info.index]

    def foreground_small_for(info: MevisFrameInfo) -> np.ndarray:
        if info.index not in foreground_small_cache:
            mask = decode_coco_rle(info.rle)
            crop, _, _ = crop_clean_reference(image_for(info), mask)
            fitted = ImageOps.contain(crop, (64, 64), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (64, 64), "white")
            canvas.paste(
                fitted,
                ((64 - fitted.width) // 2, (64 - fitted.height) // 2),
            )
            foreground_small_cache[info.index] = np.asarray(canvas, dtype=np.float32)
            foreground_phash_cache[info.index] = imagehash.phash(canvas, hash_size=8, highfreq_factor=4)
        return foreground_small_cache[info.index]

    minimum_gap = max(4, int(math.ceil(len(track.frame_ids) * 0.12)))
    best: tuple[float, MevisFrameInfo, MevisFrameInfo, dict[str, Any]] | None = None
    for left_position, left in enumerate(shortlisted):
        for right in shortlisted[left_position + 1 :]:
            temporal_gap = abs(left.index - right.index)
            if temporal_gap < minimum_gap:
                continue
            area_ratio_change = max(left.area_ratio, right.area_ratio) / min(left.area_ratio, right.area_ratio)
            aspect_change = max(left.aspect_ratio, right.aspect_ratio) / min(left.aspect_ratio, right.aspect_ratio)
            maximum_geometry_change = 2.5 if relaxed_daily_object else 2.25
            if area_ratio_change > maximum_geometry_change or aspect_change > maximum_geometry_change:
                continue
            phash_distance = int(phash_for(left) - phash_for(right))
            if phash_distance < 2:
                continue
            difference = float(np.mean(np.abs(small_for(left) - small_for(right))))
            if difference < 4.0:
                continue
            left_foreground = foreground_small_for(left)
            right_foreground = foreground_small_for(right)
            foreground_difference = float(np.mean(np.abs(left_foreground - right_foreground)))
            foreground_phash_distance = int(foreground_phash_cache[left.index] - foreground_phash_cache[right.index])
            if foreground_difference < 5.1 or foreground_phash_distance < 2:
                continue
            left_sharpness, right_sharpness = sharpness_for(left), sharpness_for(right)
            if min(left_sharpness, right_sharpness) < 8.0:
                continue
            # Put the cleaner/larger instance on the reference side while the
            # target retains the frame with more useful surrounding context.
            left_ref_score = left.geometry_score + 0.12 * math.log1p(left_sharpness)
            right_ref_score = right.geometry_score + 0.12 * math.log1p(right_sharpness)
            if right_ref_score > left_ref_score:
                src, tar = right, left
            else:
                src, tar = left, right
            normalized_gap = temporal_gap / max(1, len(track.frame_ids) - 1)
            score = (
                src.geometry_score
                + tar.geometry_score
                + min(1.5, phash_distance / 12.0)
                + min(1.0, difference / 35.0)
                + min(1.0, foreground_difference / 30.0)
                + min(1.0, normalized_gap / 0.35)
                - 0.45 * abs(math.log(area_ratio_change))
                - 0.35 * abs(math.log(aspect_change))
            )
            evidence = {
                "temporal_gap_frames": temporal_gap,
                "temporal_gap_ratio": round(normalized_gap, 8),
                "phash_distance": phash_distance,
                "mean_rgb_difference_64": round(difference, 8),
                "foreground_mean_rgb_difference_64": round(foreground_difference, 8),
                "foreground_phash_distance": foreground_phash_distance,
                "src_sharpness": round(sharpness_for(src), 8),
                "tar_sharpness": round(sharpness_for(tar), 8),
                "area_ratio_change": round(area_ratio_change, 8),
                "aspect_ratio_change": round(aspect_change, 8),
            }
            candidate = (score, src, tar, evidence)
            if best is None or (score, src.frame_id, tar.frame_id) > (best[0], best[1].frame_id, best[2].frame_id):
                best = candidate
    return best


def build_mevis_candidate_plans(root: Path, seed: int) -> list[PairPlan]:
    tracks = load_mevis_tracks(root / "meta_expressions_v2.json", root / "mask_dict.json")
    image_root = root / "JPEGImages"
    if not image_root.is_dir():
        raise FileNotFoundError(f"MeViS JPEGImages has not been extracted: {image_root}")
    plans: list[PairPlan] = []
    for track_index, track in enumerate(tracks, start=1):
        classifications = classify_mevis_track(track)
        if not classifications:
            continue
        selected_pair = select_frame_pair(
            track,
            image_root,
            seed,
            relaxed_daily_object=(("object", "daily_sports_objects") in classifications),
        )
        if selected_pair is None:
            continue
        pair_score, src, tar, pair_evidence = selected_pair
        subject_votes = mevis_subject_votes(track)
        leaf_label = subject_votes.most_common(1)[0][0]
        expression_evidence = " | ".join(track.expressions)
        for (major, minor), classification_score in classifications.items():
            src_path = image_root / track.video_id / f"{src.frame_id}.jpg"
            tar_path = image_root / track.video_id / f"{tar.frame_id}.jpg"
            identity_id = track.identity_id
            plans.append(
                PairPlan(
                    major=major,
                    minor=minor,
                    leaf_label=leaf_label,
                    source_dataset="mevis_v2",
                    pair_kind="same_track_cross_frame",
                    identity_scope="physical_instance",
                    identity_id=identity_id,
                    src_image_path=src_path,
                    tar_image_path=tar_path,
                    src_mask_payload=src.rle,
                    tar_mask_payload=tar.rle,
                    src_image_id=f"train/{track.video_id}/{src.frame_id}",
                    tar_image_id=f"train/{track.video_id}/{tar.frame_id}",
                    src_ann_id=track.anno_id,
                    tar_ann_id=track.anno_id,
                    src_upstream={
                        "video_id": track.video_id,
                        "anno_id": track.anno_id,
                        "frame_id": src.frame_id,
                        "frame_index": src.index,
                    },
                    tar_upstream={
                        "video_id": track.video_id,
                        "anno_id": track.anno_id,
                        "frame_id": tar.frame_id,
                        "frame_index": tar.index,
                    },
                    identity_evidence={
                        "type": "official_tracked_instance_id",
                        "key_fields": ["video_id", "anno_id"],
                    },
                    classification_evidence=expression_evidence,
                    selection_score=round(pair_score + classification_score, 8),
                    difficulty=(
                        "easy"
                        if pair_evidence["area_ratio_change"] <= 1.65 and pair_evidence["aspect_ratio_change"] <= 1.65
                        else "moderate"
                    ),
                    license_note=(
                        "MeViS v2: CC BY-NC-SA 4.0 and non-commercial research only; "
                        "underlying video/image rights also apply"
                    ),
                    extra={
                        "mevis_expressions": list(track.expressions),
                        "subject_votes": dict(subject_votes),
                        "src_native_bbox_xywh": [round(value, 4) for value in src.bbox_xywh],
                        "tar_native_bbox_xywh": [round(value, 4) for value in tar.bbox_xywh],
                        "src_native_area_ratio": round(src.area_ratio, 8),
                        "tar_native_area_ratio": round(tar.area_ratio, 8),
                        **pair_evidence,
                    },
                )
            )
        if track_index % 250 == 0:
            print(
                f"MeViS planning: tracks={track_index}/{len(tracks)} plans={len(plans)}",
                flush=True,
            )
    return plans


def load_curated_mevis_candidate_plans(root: Path, pool_path: Path) -> list[PairPlan]:
    """Load the pinned, source-derived MeViS visual-QC candidate pool.

    The pool stores only upstream IDs, paired frame indices, deterministic
    geometry/pixel statistics, and expression evidence.  Native pixels and
    masks are always read again from the pinned official source snapshot.
    """

    if file_sha256(pool_path) != MEVIS_CURATION_POOL_SHA256:
        raise ValueError(f"MeViS curation-pool SHA256 mismatch: {pool_path}")
    payload = read_json(pool_path)
    if payload.get("schema_version") != MEVIS_CURATION_POOL_SCHEMA:
        raise ValueError(f"Unsupported MeViS curation-pool schema: {pool_path}")
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise ValueError("MeViS curation-pool candidates must be a non-empty list")
    membership_rows = payload.get("counts", {}).get("membership_rows")
    if isinstance(membership_rows, bool) or not isinstance(membership_rows, int) or membership_rows != len(rows):
        raise ValueError("MeViS curation-pool membership count disagrees with rows")
    metadata = read_json(root / "meta_expressions_v2.json")["videos"]
    masks = read_json(root / "mask_dict.json")
    image_root = root / "JPEGImages"
    tier_bonus = {"strong": 3.0, "medium": 1.5, "base": 0.0}
    plans: list[PairPlan] = []
    seen_candidate_ids: set[str] = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate MeViS curation candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)
        major, minor = str(row["major"]), str(row["minor"])
        if major not in {"object", "person"} or minor not in MAJOR_MINOR_QUOTAS[major]:
            raise ValueError(f"Unexpected MeViS curation category: {major}/{minor}")
        video_id, anno_id = str(row["video_id"]), str(row["anno_id"])
        identity_id = f"mevis_v2/train/{video_id}/{anno_id}"
        if row.get("identity_key") != identity_id:
            raise ValueError(f"MeViS curation identity mismatch: {candidate_id}")
        video = metadata.get(video_id)
        rles = masks.get(anno_id)
        if video is None or rles is None:
            raise ValueError(f"MeViS curation upstream identity missing: {identity_id}")
        src_index, tar_index = int(row["src_frame_index"]), int(row["tar_frame_index"])
        if src_index == tar_index:
            raise ValueError(f"MeViS curation reused one frame: {candidate_id}")
        frame_ids = list(map(str, video["frames"]))
        src_frame, tar_frame = frame_ids[src_index], frame_ids[tar_index]
        if src_frame != str(row["src_frame_name"]) or tar_frame != str(row["tar_frame_name"]):
            raise ValueError(f"MeViS curation frame-name mismatch: {candidate_id}")
        if rles[src_index] is None or rles[tar_index] is None:
            raise ValueError(f"MeViS curation selected an invisible frame: {candidate_id}")
        src_path = image_root / video_id / f"{src_frame}.jpg"
        tar_path = image_root / video_id / f"{tar_frame}.jpg"
        if not src_path.is_file() or not tar_path.is_file():
            raise FileNotFoundError(f"MeViS curated frame is missing: {candidate_id}")
        geometry = row["geometry"]
        pixel = row["pixel_delta"]
        tier = str(row["pixel_delta_tier"])
        if tier not in tier_bonus:
            raise ValueError(f"Unexpected MeViS pixel tier: {tier}")
        candidate_source = str(row.get("candidate_source", ""))
        if candidate_source not in {
            "curated_v1",
            "generic_supplement_v1",
            "generic_supplement_relaxed_v1",
        }:
            raise ValueError(f"Unexpected MeViS candidate source: {candidate_id}")
        penalty_value = row.get("selection_priority_penalty")
        if isinstance(penalty_value, bool) or not isinstance(penalty_value, (int, float)):
            raise ValueError(f"Invalid MeViS selection priority penalty: {candidate_id}")
        selection_priority_penalty = float(penalty_value)
        if not math.isfinite(selection_priority_penalty) or selection_priority_penalty < 0.0:
            raise ValueError(f"Invalid MeViS selection priority penalty: {candidate_id}")
        if float(pixel["crop_mae"]) < 0.02 or int(pixel["crop_phash"]) < 2:
            raise ValueError(f"MeViS curated pair failed pixel-difference floor: {candidate_id}")
        score = (
            7.0
            + tier_bonus[tier]
            + min(2.0, int(pixel["crop_phash"]) / 12.0)
            + min(2.0, float(pixel["crop_mae"]) / 0.06)
            + float(geometry.get("score", 0.0))
            + min(1.0, float(row.get("semantic_winner_score") or 0) / 12.0)
            - selection_priority_penalty
        )
        src_geometry, tar_geometry = geometry["src"], geometry["tar"]
        min_margin = min(
            float(src_geometry["border_margin_ratio"]),
            float(tar_geometry["border_margin_ratio"]),
        )
        easy = (
            tier == "strong"
            and min_margin >= 0.003
            and float(geometry["mask_area_ratio_change"]) <= 1.8
            and float(geometry["bbox_aspect_ratio_change"]) <= 1.8
        )
        expressions = [str(expression) for expression in row["expressions"]]
        plans.append(
            PairPlan(
                major=major,
                minor=minor,
                leaf_label=str(row.get("semantic_primary") or minor),
                source_dataset="mevis_v2",
                pair_kind="same_track_cross_frame",
                identity_scope="physical_instance",
                identity_id=identity_id,
                src_image_path=src_path,
                tar_image_path=tar_path,
                src_mask_payload=rles[src_index],
                tar_mask_payload=rles[tar_index],
                src_image_id=f"train/{video_id}/{src_frame}",
                tar_image_id=f"train/{video_id}/{tar_frame}",
                src_ann_id=anno_id,
                tar_ann_id=anno_id,
                src_upstream={
                    "video_id": video_id,
                    "anno_id": anno_id,
                    "frame_id": src_frame,
                    "frame_index": src_index,
                },
                tar_upstream={
                    "video_id": video_id,
                    "anno_id": anno_id,
                    "frame_id": tar_frame,
                    "frame_index": tar_index,
                },
                identity_evidence={
                    "type": "official_tracked_instance_id",
                    "key_fields": ["video_id", "anno_id"],
                },
                classification_evidence=" | ".join(expressions),
                selection_score=round(score, 8),
                difficulty="easy" if easy else "moderate",
                license_note=(
                    "MeViS v2: CC BY-NC-SA 4.0 and non-commercial research only; "
                    "underlying video/image rights also apply"
                ),
                extra={
                    "curation_candidate_id": candidate_id,
                    "curation_candidate_source": candidate_source,
                    "selection_priority_penalty": selection_priority_penalty,
                    "mevis_expressions": expressions,
                    "semantic_primary": row.get("semantic_primary"),
                    "semantic_winner_score": row.get("semantic_winner_score"),
                    "semantic_runner_up": row.get("semantic_runner_up"),
                    "semantic_runner_up_score": row.get("semantic_runner_up_score"),
                    "geometry_rule": row.get("geometry_rule"),
                    "pixel_delta_tier": tier,
                    "src_native_bbox_xywh": src_geometry["bbox_xywh"],
                    "tar_native_bbox_xywh": tar_geometry["bbox_xywh"],
                    "src_native_area_ratio": round(float(src_geometry["mask_area_ratio"]), 8),
                    "tar_native_area_ratio": round(float(tar_geometry["mask_area_ratio"]), 8),
                    "temporal_gap_frames": int(geometry["temporal_gap_frames"]),
                    "temporal_gap_ratio": round(float(geometry["temporal_gap_fraction"]), 8),
                    "area_ratio_change": round(float(geometry["mask_area_ratio_change"]), 8),
                    "aspect_ratio_change": round(float(geometry["bbox_aspect_ratio_change"]), 8),
                    "foreground_mean_rgb_difference_128": round(float(pixel["crop_mae"]) * 255.0, 8),
                    "foreground_phash_distance": int(pixel["crop_phash"]),
                    "foreground_dhash_distance": int(pixel["crop_dhash"]),
                    "frame_mean_rgb_difference_128": round(float(pixel["frame_mae"]) * 255.0, 8),
                    "frame_phash_distance": int(pixel["frame_phash"]),
                },
            )
        )
    actual_counts = Counter(f"{plan.major}/{plan.minor}" for plan in plans)
    expected_counts = Counter(payload["counts"]["by_minor"])
    if actual_counts != expected_counts:
        raise ValueError(f"MeViS curation-pool category counts mismatch: {actual_counts}")
    return plans


def select_mevis_plans(candidates: Sequence[PairPlan], seed: int) -> list[PairPlan]:
    """Fill quotas with unique tracks, source frames, and target frames.

    Capacity analysis proves that the 600 required tracks span at least 596
    videos, not 600.  A min-cost flow therefore gives every video one free slot
    and a heavily penalized second slot.  This deterministically uses only four
    repeated videos while never reusing an official ``(video_id, anno_id)``.

    Different tracks can coexist in the same video and therefore propose the
    same frame.  A plain video/identity flow cannot express that pairwise frame
    constraint.  After each flow solve, duplicate source/target frames cause the
    least-preferred conflicting plan edge to be forbidden and the flow is solved
    again.  The loop is deterministic, fails closed if quotas become infeasible,
    and also prevents a frame from appearing once as a reference and elsewhere
    as a target.
    """

    import networkx as nx

    required = sum(
        MAJOR_MINOR_QUOTAS[major][minor] for major in ("object", "person") for minor in MAJOR_MINOR_QUOTAS[major]
    )
    best_by_identity_minor: dict[tuple[str, str, str], PairPlan] = {}
    identity_video: dict[str, str] = {}
    for plan in candidates:
        video_id = str(plan.src_upstream["video_id"])
        if plan.identity_id in identity_video and identity_video[plan.identity_id] != video_id:
            raise ValueError(f"MeViS identity appears in multiple videos: {plan.identity_id}")
        identity_video[plan.identity_id] = video_id
        key = (plan.identity_id, plan.major, plan.minor)
        existing = best_by_identity_minor.get(key)
        if existing is None or (plan.selection_score, plan.src_image_id, plan.tar_image_id) > (
            existing.selection_score,
            existing.src_image_id,
            existing.tar_image_id,
        ):
            best_by_identity_minor[key] = plan

    identities_by_video: dict[str, list[str]] = defaultdict(list)
    for identity_id, video_id in identity_video.items():
        identities_by_video[video_id].append(identity_id)
    repeat_penalty = 10**12
    forbidden_plan_keys: set[tuple[str, str, str]] = set()
    selected: list[PairPlan] = []
    for solve_index in range(len(best_by_identity_minor) + 1):
        graph = nx.DiGraph()
        source, sink = "source", "sink"
        graph.add_node(source, demand=-required)
        graph.add_node(sink, demand=required)
        for video_id, identities in identities_by_video.items():
            primary = f"video-primary:{video_id}"
            repeat = f"video-repeat:{video_id}"
            graph.add_edge(source, primary, capacity=1, weight=0)
            graph.add_edge(source, repeat, capacity=1, weight=repeat_penalty)
            for identity_id in identities:
                identity_in = f"identity-in:{identity_id}"
                identity_out = f"identity-out:{identity_id}"
                graph.add_edge(primary, identity_in, capacity=1, weight=0)
                graph.add_edge(repeat, identity_in, capacity=1, weight=0)
                graph.add_edge(identity_in, identity_out, capacity=1, weight=0)

        for major in ("object", "person"):
            for minor, quota in MAJOR_MINOR_QUOTAS[major].items():
                minor_node = f"minor:{major}/{minor}"
                graph.add_edge(minor_node, sink, capacity=quota, weight=0)
        plan_by_edge: dict[tuple[str, str], PairPlan] = {}
        for plan_key, plan in best_by_identity_minor.items():
            if plan_key in forbidden_plan_keys:
                continue
            identity_id, major, minor = plan_key
            identity_out = f"identity-out:{identity_id}"
            minor_node = f"minor:{major}/{minor}"
            tie = _stable_tiebreak(seed, identity_id, major, minor) % 1000
            weight = -int(round(plan.selection_score * 1_000_000)) + int(tie)
            graph.add_edge(identity_out, minor_node, capacity=1, weight=weight)
            plan_by_edge[(identity_out, minor_node)] = plan
        try:
            flow = nx.min_cost_flow(graph)
        except nx.NetworkXUnfeasible as error:
            capacity = Counter((plan.major, plan.minor) for plan in candidates)
            raise ValueError(
                "MeViS min-cost quota flow became infeasible while enforcing "
                f"global frame uniqueness; forbidden_edges={len(forbidden_plan_keys)} "
                f"capacity={capacity}"
            ) from error
        selected = [plan for edge, plan in plan_by_edge.items() if flow[edge[0]].get(edge[1], 0) == 1]
        if len(selected) != required:
            raise AssertionError(f"MeViS flow selected {len(selected)} plans, expected {required}")

        plans_by_frame: dict[str, list[PairPlan]] = defaultdict(list)
        for plan in selected:
            plans_by_frame[plan.src_image_id].append(plan)
            plans_by_frame[plan.tar_image_id].append(plan)
        duplicate_groups = [plans for plans in plans_by_frame.values() if len(plans) > 1]
        if not duplicate_groups:
            break

        newly_forbidden: set[tuple[str, str, str]] = set()
        for plans in duplicate_groups:
            loser = min(
                plans,
                key=lambda plan: (
                    int(round(plan.selection_score * 1_000_000)),
                    -(_stable_tiebreak(seed, plan.identity_id, plan.major, plan.minor) % 1000),
                    plan.identity_id,
                    plan.major,
                    plan.minor,
                ),
            )
            newly_forbidden.add((loser.identity_id, loser.major, loser.minor))
        newly_forbidden -= forbidden_plan_keys
        if not newly_forbidden:
            raise AssertionError("MeViS frame-uniqueness solver made no progress with duplicate frames")
        forbidden_plan_keys.update(newly_forbidden)
    else:
        raise AssertionError("MeViS frame-uniqueness solver exceeded its finite edge bound")
    selected.sort(
        key=lambda plan: (
            ("object", "person").index(plan.major),
            list(MAJOR_MINOR_QUOTAS[plan.major]).index(plan.minor),
            -plan.selection_score,
            plan.identity_id,
        )
    )
    video_counts = Counter(str(plan.src_upstream["video_id"]) for plan in selected)
    if max(video_counts.values(), default=0) > 2:
        raise AssertionError(f"MeViS selection reused a video more than twice: {video_counts}")
    if len(video_counts) < len(selected) - 4:
        raise AssertionError(
            f"MeViS selection used too many repeated videos: cases={len(selected)} "
            f"unique_videos={len(video_counts)}"
        )
    if len({plan.identity_id for plan in selected}) != len(selected):
        raise AssertionError("MeViS selection reused an official track identity")
    all_frame_ids = [image_id for plan in selected for image_id in (plan.src_image_id, plan.tar_image_id)]
    if len(set(all_frame_ids)) != len(all_frame_ids):
        raise AssertionError("MeViS selection reused a source frame in the final result")
    counts = Counter((plan.major, plan.minor) for plan in selected)
    expected = Counter(
        {(major, minor): quota for major in ("object", "person") for minor, quota in MAJOR_MINOR_QUOTAS[major].items()}
    )
    if counts != expected:
        raise AssertionError(f"MeViS quota mismatch: {counts} vs {expected}")
    return selected


def load_excluded_identities(path: Path | None) -> tuple[set[str], str | None]:
    if path is None:
        return set(), None
    if not path.is_file():
        raise FileNotFoundError(f"Missing required identity exclusion config: {path}")
    payload = read_json(path)
    rows = payload.get("excluded_identities", [])
    if not isinstance(rows, list):
        raise ValueError("excluded_identities must be a list")
    identities: set[str] = set()
    for row in rows:
        identity = row if isinstance(row, str) else row.get("identity_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"Invalid excluded identity row: {row!r}")
        if identity in identities:
            raise ValueError(f"Duplicate excluded identity: {identity}")
        identities.add(identity)
    return identities, file_sha256(path)


def _plan_to_json(plan: PairPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["src_image_path"] = str(plan.src_image_path)
    payload["tar_image_path"] = str(plan.tar_image_path)
    return payload


def _plan_from_json(payload: Mapping[str, Any]) -> PairPlan:
    values = dict(payload)
    values["src_image_path"] = Path(str(values["src_image_path"]))
    values["tar_image_path"] = Path(str(values["tar_image_path"]))
    return PairPlan(**values)


def _plan_cache_context(
    raw_root: Path,
    mevis_curation_pool_path: Path | None,
) -> dict[str, Any]:
    pool_path = mevis_curation_pool_path.resolve() if mevis_curation_pool_path else None
    pool_sha256 = file_sha256(pool_path) if pool_path is not None else None
    if pool_sha256 is not None and pool_sha256 != MEVIS_CURATION_POOL_SHA256:
        raise ValueError(f"MeViS curation-pool SHA256 mismatch: {pool_path}")
    return {
        "raw_root": str(raw_root.resolve()),
        "mevis_planner_mode": "curated_pool" if pool_path is not None else "generic",
        "mevis_curation_pool": str(pool_path) if pool_path is not None else None,
        "mevis_curation_pool_sha256": pool_sha256,
        "dependency_versions": runtime_dependency_versions(),
    }


def _validate_cached_plan_paths(
    plans: Sequence[PairPlan],
    raw_root: Path,
) -> None:
    raw_root = raw_root.resolve()
    for plan in plans:
        for source, path, upstream in (
            (True, plan.src_image_path, plan.src_upstream),
            (False, plan.tar_image_path, plan.tar_upstream),
        ):
            if plan.source_dataset == "mevis_v2":
                expected = (
                    raw_root
                    / "mevis_v2"
                    / "train"
                    / "JPEGImages"
                    / str(upstream["video_id"])
                    / f"{upstream['frame_id']}.jpg"
                )
            elif plan.source_dataset == "deepfashion2":
                expected = raw_root / "deepfashion2_validation" / "validation" / "image" / f"{upstream['image_id']}.jpg"
            else:
                raise ValueError(f"Unsupported cached source dataset: {plan.source_dataset}")
            if path.resolve() != expected.resolve():
                role = "src" if source else "tar"
                raise ValueError(
                    f"Cached {role} path is not bound to the requested raw root: " f"actual={path} expected={expected}"
                )
            if not expected.is_file():
                raise FileNotFoundError(f"Cached candidate image is missing: {expected}")


def write_plan_cache(
    path: Path,
    *,
    seed: int,
    raw_root: Path,
    mevis_curation_pool_path: Path | None,
    mevis_candidates: Sequence[PairPlan],
    fashion_candidates: Sequence[PairPlan],
) -> None:
    assert_builder_source_unchanged()
    payload = {
        "schema_version": 2,
        "seed": seed,
        "builder_sha256": LOADED_BUILDER_SHA256,
        "cache_context": _plan_cache_context(raw_root, mevis_curation_pool_path),
        "source_pins": {
            "mevis_archive_sha256": MEVIS_ARCHIVE_SHA256,
            "mevis_meta_sha256": MEVIS_META_SHA256,
            "mevis_mask_dict_sha256": MEVIS_MASK_DICT_SHA256,
            "mevis_curation_pool_sha256": MEVIS_CURATION_POOL_SHA256,
            "deepfashion_archive_sha256": DEEPFASHION_ARCHIVE_SHA256,
            "deepfashion_query_sha256": DEEPFASHION_QUERY_SHA256,
            "deepfashion_gallery_sha256": DEEPFASHION_GALLERY_SHA256,
        },
        "mevis_candidates": [_plan_to_json(plan) for plan in mevis_candidates],
        "fashion_candidates": [_plan_to_json(plan) for plan in fashion_candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
                compressed.write(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    assert_builder_source_unchanged()


def load_plan_cache(
    path: Path,
    *,
    seed: int,
    raw_root: Path,
    mevis_curation_pool_path: Path | None,
) -> tuple[list[PairPlan], list[PairPlan]]:
    assert_builder_source_unchanged()
    if path.is_symlink():
        raise ValueError(f"Candidate-plan cache must not be a symlink: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 2 or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"Plan cache schema/seed mismatch: {path}")
    if payload.get("builder_sha256") != LOADED_BUILDER_SHA256:
        raise ValueError(f"Plan cache was generated by a different builder: {path}; remove it and rebuild")
    expected_context = _plan_cache_context(raw_root, mevis_curation_pool_path)
    if payload.get("cache_context") != expected_context:
        raise ValueError(f"Plan cache raw-root/planner/dependency context mismatch: {path}")
    expected_pins = {
        "mevis_archive_sha256": MEVIS_ARCHIVE_SHA256,
        "mevis_meta_sha256": MEVIS_META_SHA256,
        "mevis_mask_dict_sha256": MEVIS_MASK_DICT_SHA256,
        "mevis_curation_pool_sha256": MEVIS_CURATION_POOL_SHA256,
        "deepfashion_archive_sha256": DEEPFASHION_ARCHIVE_SHA256,
        "deepfashion_query_sha256": DEEPFASHION_QUERY_SHA256,
        "deepfashion_gallery_sha256": DEEPFASHION_GALLERY_SHA256,
    }
    if payload.get("source_pins") != expected_pins:
        raise ValueError(f"Plan cache source pins mismatch: {path}")
    mevis = [_plan_from_json(row) for row in payload["mevis_candidates"]]
    fashion = [_plan_from_json(row) for row in payload["fashion_candidates"]]
    _validate_cached_plan_paths([*mevis, *fashion], raw_root)
    assert_builder_source_unchanged()
    return mevis, fashion


def validate_source_snapshot(raw_root: Path, *, verify_large_archives: bool = True) -> dict[str, Any]:
    mevis_root = raw_root / "mevis_v2" / "train"
    deepfashion_root = raw_root / "deepfashion2_validation"
    expected = {
        "mevis_jpeg_archive": (
            mevis_root / "JPEGImages.tar",
            15483000320,
            MEVIS_ARCHIVE_SHA256,
            verify_large_archives,
        ),
        "mevis_metadata": (
            mevis_root / "meta_expressions_v2.json",
            3728474,
            MEVIS_META_SHA256,
            True,
        ),
        "mevis_mask_dict": (
            mevis_root / "mask_dict.json",
            410960092,
            MEVIS_MASK_DICT_SHA256,
            True,
        ),
        "deepfashion_validation_archive": (
            deepfashion_root / "validation.zip",
            1814423256,
            DEEPFASHION_ARCHIVE_SHA256,
            verify_large_archives,
        ),
        "deepfashion_query_metadata": (
            deepfashion_root / "json_for_validation" / "val_query.json",
            None,
            DEEPFASHION_QUERY_SHA256,
            True,
        ),
        "deepfashion_gallery_metadata": (
            deepfashion_root / "json_for_validation" / "val_gallery.json",
            None,
            DEEPFASHION_GALLERY_SHA256,
            True,
        ),
    }
    result: dict[str, Any] = {}
    for name, (path, expected_size, expected_hash, should_hash) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing pinned source file: {path}")
        size = path.stat().st_size
        if expected_size is not None and size != expected_size:
            raise ValueError(f"Pinned source size mismatch for {name}: actual={size} expected={expected_size}")
        digest = file_sha256(path) if should_hash else expected_hash
        if digest != expected_hash:
            raise ValueError(f"Pinned source SHA256 mismatch for {name}: actual={digest} expected={expected_hash}")
        result[name] = {
            "path": str(path),
            "bytes": size,
            "sha256": expected_hash,
            "verified": bool(should_hash),
        }
    mevis_images = mevis_root / "JPEGImages"
    deepfashion_images = deepfashion_root / "validation" / "image"
    deepfashion_annos = deepfashion_root / "validation" / "annos"
    if not mevis_images.is_dir():
        raise FileNotFoundError(f"Missing extracted MeViS images: {mevis_images}")
    if not deepfashion_images.is_dir() or not deepfashion_annos.is_dir():
        raise FileNotFoundError(f"Missing extracted DeepFashion2 validation tree: {deepfashion_root}")
    deepfashion_image_count = sum(1 for path in deepfashion_images.glob("*.jpg") if path.is_file())
    deepfashion_annotation_count = sum(1 for path in deepfashion_annos.glob("*.json") if path.is_file())
    if (deepfashion_image_count, deepfashion_annotation_count) != (32153, 32153):
        raise ValueError(
            "DeepFashion2 extracted inventory mismatch: "
            f"images={deepfashion_image_count} annos={deepfashion_annotation_count}"
        )
    result["extracted_inventory"] = {
        "deepfashion_image_count": deepfashion_image_count,
        "deepfashion_annotation_count": deepfashion_annotation_count,
        "mevis_video_directory_count": sum(1 for path in mevis_images.iterdir() if path.is_dir()),
    }
    return result


def verify_selected_archive_members(
    plans: Sequence[PairPlan],
    raw_root: Path,
) -> dict[str, Any]:
    """Match every selected extracted source byte-for-byte to pinned archives."""

    raw_root = raw_root.resolve()
    mevis_members: dict[str, Path] = {}
    fashion_members: dict[str, Path] = {}
    for plan in plans:
        if plan.source_dataset == "mevis_v2":
            for path, upstream in (
                (plan.src_image_path, plan.src_upstream),
                (plan.tar_image_path, plan.tar_upstream),
            ):
                member = f"JPEGImages/{upstream['video_id']}/{upstream['frame_id']}.jpg"
                mevis_members[member] = path
        elif plan.source_dataset == "deepfashion2":
            for path, upstream in (
                (plan.src_image_path, plan.src_upstream),
                (plan.tar_image_path, plan.tar_upstream),
            ):
                image_id = str(upstream["image_id"])
                fashion_members[f"validation/image/{image_id}.jpg"] = path
                fashion_members[f"validation/annos/{image_id}.json"] = (
                    raw_root / "deepfashion2_validation" / "validation" / "annos" / f"{image_id}.json"
                )
        else:
            raise ValueError(f"Unsupported selected source dataset: {plan.source_dataset}")

    verified_rows: list[str] = []
    mevis_archive = raw_root / "mevis_v2" / "train" / "JPEGImages.tar"
    wanted_mevis = set(mevis_members)
    with tarfile.open(mevis_archive, mode="r:") as archive:
        infos = {info.name: info for info in archive if info.isfile() and info.name in wanted_mevis}
        missing = sorted(wanted_mevis.difference(infos))
        if missing:
            raise ValueError(f"Selected MeViS archive members are missing: {missing[:5]}")
        for member, info in sorted(infos.items(), key=lambda item: item[1].offset_data):
            archived_handle = archive.extractfile(info)
            if archived_handle is None:
                raise ValueError(f"Cannot read selected MeViS archive member: {member}")
            archived = archived_handle.read()
            extracted = mevis_members[member].read_bytes()
            if archived != extracted:
                raise ValueError(f"Extracted MeViS source differs from pinned archive member: {member}")
            verified_rows.append(f"mevis_v2:{member}:{hashlib.sha256(archived).hexdigest()}")

    fashion_archive = raw_root / "deepfashion2_validation" / "validation.zip"
    with zipfile.ZipFile(fashion_archive) as archive:
        names = set(archive.namelist())
        missing = sorted(set(fashion_members).difference(names))
        if missing:
            raise ValueError(f"Selected DeepFashion2 archive members are missing: {missing[:5]}")
        for member in sorted(fashion_members):
            archived = archive.read(member)
            extracted = fashion_members[member].read_bytes()
            if archived != extracted:
                raise ValueError(f"Extracted DeepFashion2 source differs from pinned archive member: {member}")
            verified_rows.append(f"deepfashion2:{member}:{hashlib.sha256(archived).hexdigest()}")

    canonical = "\n".join(sorted(verified_rows)).encode("utf-8")
    return {
        "status": "verified_byte_exact_against_pinned_archives",
        "mevis_image_member_count": len(mevis_members),
        "deepfashion_image_member_count": sum(member.startswith("validation/image/") for member in fashion_members),
        "deepfashion_annotation_member_count": sum(
            member.startswith("validation/annos/") for member in fashion_members
        ),
        "verified_member_count": len(verified_rows),
        "verified_member_set_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def assign_case_ids(plans: Sequence[PairPlan]) -> list[tuple[str, PairPlan]]:
    result: list[tuple[str, PairPlan]] = []
    for major in ("object", "person", "garment"):
        for minor in MAJOR_MINOR_QUOTAS[major]:
            subset = sorted(
                (plan for plan in plans if plan.major == major and plan.minor == minor),
                key=lambda plan: (-plan.selection_score, plan.identity_id),
            )
            quota = MAJOR_MINOR_QUOTAS[major][minor]
            if len(subset) != quota:
                raise ValueError(f"Plan quota mismatch for {major}/{minor}: {len(subset)} vs {quota}")
            for index, plan in enumerate(subset):
                case_id = f"{CASE_PREFIXES[major]}_{minor}_{index:03d}"
                result.append((case_id, plan))
    if len(result) != 900:
        raise AssertionError(f"Expected 900 assigned plans, got {len(result)}")
    return result


def freeze_showcase_candidates(records: list[dict[str, Any]], per_minor: int = 10) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        record["showcase_candidate"] = False
        grouped[(record["major"], record["minor"])].append(record)
    for key, rows in grouped.items():
        rows.sort(
            key=lambda record: (
                record["difficulty"] != "easy",
                -float(record["selection_score"]),
                record["case_id"],
            )
        )
        for record in rows[:per_minor]:
            record["showcase_candidate"] = True


def _json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def write_manifests(output_root: Path, records: Sequence[dict[str, Any]]) -> None:
    manifest_root = output_root / "manifests"
    primary = "".join(_json_line(record) for record in records)
    atomic_write_text(manifest_root / "paperinsert_900.jsonl", primary)
    for major in ("object", "person", "garment"):
        atomic_write_text(
            manifest_root / f"{major}.jsonl",
            "".join(_json_line(record) for record in records if record["major"] == major),
        )
    fieldnames = sorted({key for record in records for key in record})
    buffer = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in record.items()
                }
            )
        buffer.seek(0)
        atomic_write_text(manifest_root / "paperinsert_900.csv", buffer.read())
    finally:
        buffer.close()


def write_source_report(
    output_root: Path,
    records: Sequence[dict[str, Any]],
    source_snapshot: Mapping[str, Any],
    selected_archive_verification: Mapping[str, Any],
    *,
    seed: int,
    exclusion_path: Path | None,
    exclusion_sha256: str | None,
    plan_cache_path: Path | None,
    plan_cache_sha256: str | None,
    mevis_curation_pool_path: Path | None,
    mevis_curation_pool_sha256: str | None,
    dependency_versions: Mapping[str, str | None],
) -> None:
    assert_builder_source_unchanged()
    if exclusion_path is not None and file_sha256(exclusion_path) != exclusion_sha256:
        raise RuntimeError(f"Exclusion config changed during build: {exclusion_path}")
    if plan_cache_path is not None and file_sha256(plan_cache_path) != plan_cache_sha256:
        raise RuntimeError(f"Candidate-plan cache changed during build: {plan_cache_path}")
    if mevis_curation_pool_path is not None and file_sha256(mevis_curation_pool_path) != mevis_curation_pool_sha256:
        raise RuntimeError(f"MeViS curation pool changed during build: {mevis_curation_pool_path}")
    report = {
        # Kept at schema 3 for compatibility with the authenticated leakage
        # scanner; the manifest itself is explicitly schema 4.
        "schema_version": 3,
        "dataset": "PaperInsert-900 identity-paired benchmark",
        "manifest_schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "protocol_policy": {
            "pair_kinds": ["same_track_cross_frame", "same_product_cross_domain"],
            "same_identity_required": True,
            "gt_available_required": True,
            "different_source_images_required": True,
            "target_derived_reference_forbidden": True,
            "identity_evidence": "recomputed_from_pinned_upstream_annotations",
            "mevis_track_reuse_max": 1,
            "mevis_video_reuse_max": 2,
            "mevis_minimum_unique_videos_for_600_cases": 596,
        },
        "canonical_contract": ["src_image", "src_mask", "tar_image", "tar_mask"],
        "quotas": MAJOR_MINOR_QUOTAS,
        "garment_native_subquotas": GARMENT_NATIVE_SUBQUOTAS,
        "record_count": len(records),
        "identity_count": len({record["identity_id"] for record in records}),
        "source_counts": dict(Counter(record["source_dataset"] for record in records)),
        "source_snapshot": source_snapshot,
        "selected_archive_verification": dict(selected_archive_verification),
        "generator": {
            "script": str(LOADED_BUILDER_PATH),
            "script_sha256": LOADED_BUILDER_SHA256,
            "seed": seed,
            "dependency_versions": dict(dependency_versions),
            "exclusion_config": str(exclusion_path.resolve()) if exclusion_path else None,
            "exclusion_config_sha256": exclusion_sha256,
            "candidate_plan_cache": str(plan_cache_path.resolve()) if plan_cache_path else None,
            "candidate_plan_cache_sha256": plan_cache_sha256,
            "mevis_curation_pool": (str(mevis_curation_pool_path.resolve()) if mevis_curation_pool_path else None),
            "mevis_curation_pool_sha256": mevis_curation_pool_sha256,
        },
        "licenses": {
            "mevis_v2": (
                "CC BY-NC-SA 4.0; the official release states non-commercial research "
                "use only, and underlying video/image rights also apply."
            ),
            "deepfashion2": (
                "Official access is research-oriented and requires accepting the dataset "
                "access terms; no open redistribution grant is assumed."
            ),
        },
        "official_source_urls": {
            "mevis_v2": "https://github.com/henghuiding/MeViS",
            "deepfashion2": "https://github.com/switchablenorms/DeepFashion2",
        },
    }
    atomic_write_text(
        output_root / "reports" / "sources.json",
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def authoritative_release_paths(records: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = {
        "manifests/paperinsert_900.jsonl",
        "manifests/paperinsert_900.csv",
        "manifests/object.jsonl",
        "manifests/person.jsonl",
        "manifests/garment.jsonl",
        "reports/sources.json",
    }
    for record in records:
        paths.update(str(record[field]) for field in ("src_image", "src_mask", "tar_image", "tar_mask"))
    return sorted(paths)


def write_checksums(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    rows = []
    for relative in authoritative_release_paths(records):
        path = output_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Cannot checksum missing release file: {path}")
        rows.append(f"{file_sha256(path)}  {relative}\n")
    atomic_write_text(output_root / "CHECKSUMS.sha256", "".join(rows))


def _materialize_job(arguments: tuple[PairPlan, str, Path]) -> dict[str, Any]:
    plan, case_id, output_root = arguments
    return materialize_case(plan, case_id, output_root)


def build_dataset(
    raw_root: Path,
    output_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    workers: int = 8,
    exclusion_path: Path | None = None,
    plan_cache_path: Path | None = None,
    mevis_curation_pool_path: Path | None = None,
    verify_large_archives: bool = True,
) -> list[dict[str, Any]]:
    with exclusive_build_lock(output_root):
        return _build_dataset_locked(
            raw_root,
            output_root,
            seed=seed,
            workers=workers,
            exclusion_path=exclusion_path,
            plan_cache_path=plan_cache_path,
            mevis_curation_pool_path=mevis_curation_pool_path,
            verify_large_archives=verify_large_archives,
        )


def _build_dataset_locked(
    raw_root: Path,
    output_root: Path,
    *,
    seed: int = DEFAULT_SEED,
    workers: int = 8,
    exclusion_path: Path | None = None,
    plan_cache_path: Path | None = None,
    mevis_curation_pool_path: Path | None = None,
    verify_large_archives: bool = True,
) -> list[dict[str, Any]]:
    """Plan all quotas before creating the output directory, then build once."""

    from concurrent.futures import ProcessPoolExecutor

    assert_builder_source_unchanged()
    raw_root = raw_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_root}")
    source_snapshot = validate_source_snapshot(raw_root, verify_large_archives=verify_large_archives)
    excluded, exclusion_sha256 = load_excluded_identities(exclusion_path)
    dependency_versions = runtime_dependency_versions()
    mevis_curation_pool_sha256 = file_sha256(mevis_curation_pool_path) if mevis_curation_pool_path is not None else None
    mevis_root = raw_root / "mevis_v2" / "train"
    deepfashion_root = raw_root / "deepfashion2_validation"

    plan_cache_sha256: str | None = None
    if plan_cache_path is not None and plan_cache_path.is_file():
        print(f"Loading pinned candidate-plan cache: {plan_cache_path}", flush=True)
        plan_cache_sha256 = file_sha256(plan_cache_path)
        mevis_candidates, fashion_candidates = load_plan_cache(
            plan_cache_path,
            seed=seed,
            raw_root=raw_root,
            mevis_curation_pool_path=mevis_curation_pool_path,
        )
        if file_sha256(plan_cache_path) != plan_cache_sha256:
            raise RuntimeError(f"Candidate-plan cache changed while loading: {plan_cache_path}")
    else:
        print("Planning MeViS same-track pairs...", flush=True)
        mevis_candidates = (
            load_curated_mevis_candidate_plans(mevis_root, mevis_curation_pool_path)
            if mevis_curation_pool_path is not None
            else build_mevis_candidate_plans(mevis_root, seed)
        )
        print("Planning DeepFashion2 shop/user pairs...", flush=True)
        fashion_candidates = build_deepfashion_candidate_plans(deepfashion_root, seed)
        if plan_cache_path is not None:
            write_plan_cache(
                plan_cache_path,
                seed=seed,
                raw_root=raw_root,
                mevis_curation_pool_path=mevis_curation_pool_path,
                mevis_candidates=mevis_candidates,
                fashion_candidates=fashion_candidates,
            )
            plan_cache_sha256 = file_sha256(plan_cache_path)
            print(f"Wrote candidate-plan cache: {plan_cache_path}", flush=True)
    mevis_candidates = [plan for plan in mevis_candidates if plan.identity_id not in excluded]
    fashion_candidates = [plan for plan in fashion_candidates if plan.identity_id not in excluded]
    print(f"MeViS candidate plans: {len(mevis_candidates)}", flush=True)
    mevis_selected = select_mevis_plans(mevis_candidates, seed)
    print(f"DeepFashion2 candidate plans: {len(fashion_candidates)}", flush=True)
    fashion_selected = select_deepfashion_plans(fashion_candidates, seed)
    assigned = assign_case_ids([*mevis_selected, *fashion_selected])
    print("Verifying selected files against pinned source archives...", flush=True)
    selected_archive_verification = verify_selected_archive_members([plan for _, plan in assigned], raw_root)

    # The output tree becomes visible only after every quota has a valid plan.
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [(plan, case_id, output_root) for case_id, plan in assigned]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        iterator = map(_materialize_job, jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_materialize_job, jobs, chunksize=1)
    try:
        for index, record in enumerate(iterator, start=1):
            records.append(record)
            if index % 50 == 0:
                print(f"Materialized cases: {index}/900", flush=True)
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    freeze_showcase_candidates(records)
    write_manifests(output_root, records)
    write_source_report(
        output_root,
        records,
        source_snapshot,
        selected_archive_verification,
        seed=seed,
        exclusion_path=exclusion_path,
        exclusion_sha256=exclusion_sha256,
        plan_cache_path=plan_cache_path,
        plan_cache_sha256=plan_cache_sha256,
        mevis_curation_pool_path=mevis_curation_pool_path,
        mevis_curation_pool_sha256=mevis_curation_pool_sha256,
        dependency_versions=dependency_versions,
    )
    write_checksums(output_root, records)
    return records


def verify_checksums(dataset_root: Path, records: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    checksum_path = dataset_root / "CHECKSUMS.sha256"
    if not checksum_path.is_file():
        return [f"Missing checksum manifest: {checksum_path}"]
    expected_paths = authoritative_release_paths(records)
    parsed: dict[str, str] = {}
    order: list[str] = []
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) < 67 or line[64:66] != "  ":
            errors.append(f"Invalid checksum row {line_number}: {line!r}")
            continue
        digest, relative = line[:64], line[66:]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            errors.append(f"Invalid checksum digest at row {line_number}")
            continue
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.as_posix() != relative:
            errors.append(f"Unsafe checksum path at row {line_number}: {relative!r}")
            continue
        if relative in parsed:
            errors.append(f"Duplicate checksum path: {relative}")
            continue
        parsed[relative] = digest
        order.append(relative)
    if order != sorted(order):
        errors.append("Checksum rows are not sorted")
    if set(parsed) != set(expected_paths):
        errors.append(
            "Checksum path set mismatch: "
            f"missing={sorted(set(expected_paths) - set(parsed))[:10]} "
            f"extra={sorted(set(parsed) - set(expected_paths))[:10]}"
        )
    for relative, expected_digest in parsed.items():
        path = dataset_root / relative
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing or unsafe checksummed file: {relative}")
            continue
        actual_digest = file_sha256(path)
        if actual_digest != expected_digest:
            errors.append(f"Checksum mismatch for {relative}: {actual_digest} vs {expected_digest}")
    return errors


def _audit_record_files(dataset_root: Path, record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    case_id, major = str(record.get("case_id", "")), str(record.get("major", ""))
    loaded: dict[str, Image.Image] = {}
    for role in ("src_image", "src_mask", "tar_image", "tar_mask"):
        relative = record.get(role)
        expected = (Path(major) / role / f"{case_id}.png").as_posix()
        if relative != expected:
            errors.append(f"{case_id}: non-canonical {role} path {relative!r} expected {expected!r}")
            continue
        path = dataset_root / expected
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(dataset_root.resolve())
        except (FileNotFoundError, ValueError):
            errors.append(f"{case_id}: missing or escaping {role}: {path}")
            continue
        if path.is_symlink() or not path.is_file():
            errors.append(f"{case_id}: {role} is not a regular non-symlink file")
            continue
        try:
            with Image.open(path) as handle:
                loaded[role] = handle.convert("L" if role.endswith("mask") else "RGB")
        except Exception as error:
            errors.append(f"{case_id}: cannot decode {role}: {error}")
    if len(loaded) != 4:
        return errors
    if loaded["src_image"].size != loaded["src_mask"].size:
        errors.append(f"{case_id}: src image/mask size mismatch")
    if loaded["tar_image"].size != loaded["tar_mask"].size:
        errors.append(f"{case_id}: target image/mask size mismatch")
    for role in ("src_mask", "tar_mask"):
        values = set(np.unique(np.asarray(loaded[role], dtype=np.uint8)).tolist())
        if not values.issubset({0, 255}) or 255 not in values:
            errors.append(f"{case_id}: {role} is not non-empty binary 0/255")
        if values == {255}:
            errors.append(f"{case_id}: {role} is a full-frame mask")
    hash_fields = {
        "src_image": "src_pixel_sha256",
        "src_mask": "src_mask_pixel_sha256",
        "tar_image": "tar_pixel_sha256",
        "tar_mask": "tar_mask_pixel_sha256",
    }
    for role, field_name in hash_fields.items():
        digest = decoded_mask_sha256(loaded[role]) if role.endswith("mask") else decoded_pixel_sha256(loaded[role])
        if record.get(field_name) != digest:
            errors.append(f"{case_id}: decoded hash mismatch for {role}")
    src_stats, tar_stats = mask_stats(loaded["src_mask"]), mask_stats(loaded["tar_mask"])
    if abs(float(record.get("src_area_ratio", -1)) - src_stats.area_ratio) > 1e-8:
        errors.append(f"{case_id}: src_area_ratio mismatch")
    if abs(float(record.get("tar_area_ratio", -1)) - tar_stats.area_ratio) > 1e-8:
        errors.append(f"{case_id}: tar_area_ratio mismatch")
    src_rgb = np.asarray(loaded["src_image"], dtype=np.uint8)
    src_mask_values = np.asarray(loaded["src_mask"], dtype=np.uint8) > 0
    if np.any(src_rgb[~src_mask_values] != 255):
        errors.append(f"{case_id}: canonical reference background is not pure white")
    if record.get("src_background_policy") != "white_outside_native_mask":
        errors.append(f"{case_id}: unexpected reference background policy")
    return errors


def _mevis_identity_index(meta_path: Path) -> set[tuple[str, str]]:
    metadata = read_json(meta_path)
    result: set[tuple[str, str]] = set()
    for video_id, video in metadata["videos"].items():
        for expression in video["expressions"].values():
            for anno_id in expression.get("anno_id", []):
                result.add((str(video_id), str(anno_id)))
    return result


def _raw_reconstruction_errors(
    record: Mapping[str, Any],
    raw_root: Path,
    *,
    mevis_masks: Mapping[str, Sequence[Mapping[str, Any] | None]] | None,
    mevis_identities: set[tuple[str, str]] | None,
) -> list[str]:
    errors: list[str] = []
    case_id = str(record["case_id"])
    src_upstream = record["src_upstream"]
    tar_upstream = record["tar_upstream"]
    if record["source_dataset"] == "mevis_v2":
        if mevis_masks is None or mevis_identities is None:
            return [f"{case_id}: MeViS raw indices were not loaded"]
        video_id, anno_id = str(src_upstream["video_id"]), str(src_upstream["anno_id"])
        if (video_id, anno_id) not in mevis_identities:
            errors.append(f"{case_id}: MeViS identity absent from official metadata")
        rles = mevis_masks.get(anno_id)
        if rles is None:
            return errors + [f"{case_id}: MeViS anno_id missing from mask_dict"]
        src_index, tar_index = int(src_upstream["frame_index"]), int(tar_upstream["frame_index"])
        try:
            src_rle, tar_rle = rles[src_index], rles[tar_index]
        except IndexError:
            return errors + [f"{case_id}: MeViS frame index outside mask list"]
        if src_rle is None or tar_rle is None:
            return errors + [f"{case_id}: MeViS selected an invisible frame"]
        image_root = raw_root / "mevis_v2" / "train" / "JPEGImages" / video_id
        src_path = image_root / f"{src_upstream['frame_id']}.jpg"
        tar_path = image_root / f"{tar_upstream['frame_id']}.jpg"
        src_mask_native = decode_coco_rle(src_rle)
        tar_mask_native = decode_coco_rle(tar_rle)
    elif record["source_dataset"] == "deepfashion2":
        root = raw_root / "deepfashion2_validation" / "validation"
        src_image_id, tar_image_id = str(src_upstream["image_id"]), str(tar_upstream["image_id"])
        src_path = root / "image" / f"{src_image_id}.jpg"
        tar_path = root / "image" / f"{tar_image_id}.jpg"
        src_annotation = read_json(root / "annos" / f"{src_image_id}.json")
        tar_annotation = read_json(root / "annos" / f"{tar_image_id}.json")
        src_item = src_annotation[str(src_upstream["item_key"])]
        tar_item = tar_annotation[str(tar_upstream["item_key"])]
        for side, annotation, item, upstream, expected_domain in (
            ("src", src_annotation, src_item, src_upstream, "shop"),
            ("tar", tar_annotation, tar_item, tar_upstream, "user"),
        ):
            if annotation.get("source") != expected_domain:
                errors.append(f"{case_id}: raw DeepFashion {side} domain mismatch")
            if int(annotation.get("pair_id", -1)) != int(upstream["pair_id"]):
                errors.append(f"{case_id}: raw DeepFashion {side} pair_id mismatch")
            for key in ("style", "category_id"):
                if int(item.get(key, -1)) != int(upstream[key]):
                    errors.append(f"{case_id}: raw DeepFashion {side} {key} mismatch")
            if list(_bbox_tuple(item.get("bounding_box", []))) != list(record.get(f"{side}_native_bbox_xyxy", [])):
                errors.append(f"{case_id}: raw DeepFashion {side} bounding-box mismatch")
            recorded_quality = record.get(f"{side}_quality", {})
            for key in ("occlusion", "scale", "zoom_in", "viewpoint"):
                if int(item.get(key, -1)) != int(recorded_quality.get(key, -2)):
                    errors.append(f"{case_id}: raw DeepFashion {side} {key} quality mismatch")
            if (
                int(item.get("occlusion", -1)) != 1
                or int(item.get("scale", -1)) < 2
                or int(item.get("zoom_in", 0)) <= 0
                or int(item.get("zoom_in", 99)) > 2
            ):
                errors.append(f"{case_id}: raw DeepFashion {side} quality gate failed")
        with Image.open(src_path) as handle:
            src_size = handle.size
        with Image.open(tar_path) as handle:
            tar_size = handle.size
        src_mask_native = deepfashion_polygon_to_mask(src_item["segmentation"], src_size[0], src_size[1])
        tar_mask_native = deepfashion_polygon_to_mask(tar_item["segmentation"], tar_size[0], tar_size[1])
    else:
        return [f"{case_id}: unsupported raw source dataset"]

    if src_path.resolve() == tar_path.resolve():
        errors.append(f"{case_id}: raw source and target paths are identical")
    with Image.open(src_path) as handle:
        src_full = handle.convert("RGB")
    with Image.open(tar_path) as handle:
        tar_full = handle.convert("RGB")
    rebuilt_src, rebuilt_src_mask, crop_xyxy = crop_clean_reference(
        src_full,
        src_mask_native,
        margin_ratio=float(record.get("src_crop_context_ratio", 0.15)),
    )
    expected = {
        "src_pixel_sha256": decoded_pixel_sha256(rebuilt_src),
        "src_mask_pixel_sha256": decoded_mask_sha256(rebuilt_src_mask),
        "tar_pixel_sha256": decoded_pixel_sha256(tar_full),
        "tar_mask_pixel_sha256": decoded_mask_sha256(tar_mask_native),
    }
    for field_name, digest in expected.items():
        if record.get(field_name) != digest:
            errors.append(f"{case_id}: raw-backed reconstruction mismatch for {field_name}")
    if list(crop_xyxy) != list(record.get("src_crop_xyxy", [])):
        errors.append(f"{case_id}: raw-backed source crop metadata mismatch")
    for field_name, value in image_fingerprints(src_full).items():
        if record.get(field_name) != value:
            errors.append(f"{case_id}: raw full-reference fingerprint mismatch for {field_name}")
    if decoded_pixel_sha256(src_full) == decoded_pixel_sha256(tar_full):
        errors.append(f"{case_id}: raw source and target decoded images are identical")
    return errors


def audit_dataset(
    dataset_root: Path,
    *,
    raw_root: Path | None = None,
    expected_quotas: Mapping[str, Mapping[str, int]] = MAJOR_MINOR_QUOTAS,
    verify_raw: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "manifests" / "paperinsert_900.jsonl"
    records = read_jsonl(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    expected_count = sum(sum(minors.values()) for minors in expected_quotas.values())
    if len(records) != expected_count:
        errors.append(f"Record count mismatch: actual={len(records)} expected={expected_count}")
    counts = Counter((str(record.get("major")), str(record.get("minor"))) for record in records)
    expected_counts = Counter(
        {(major, minor): quota for major, minors in expected_quotas.items() for minor, quota in minors.items()}
    )
    if counts != expected_counts:
        errors.append(f"Quota matrix mismatch: actual={counts} expected={expected_counts}")
    case_ids = [str(record.get("case_id", "")) for record in records]
    identities = [str(record.get("identity_id", "")) for record in records]
    if len(set(case_ids)) != len(case_ids):
        errors.append("Duplicate case_id detected")
    if len(set(identities)) != len(identities):
        errors.append("Identity IDs are not globally unique")
    src_ids = [(record.get("source_dataset"), record.get("src_image_id")) for record in records]
    tar_ids = [(record.get("source_dataset"), record.get("tar_image_id")) for record in records]
    if len(set(src_ids)) != len(src_ids):
        errors.append("Reference source images are not globally unique")
    if len(set(tar_ids)) != len(tar_ids):
        errors.append("Target source images are not globally unique")
    all_source_ids = [*src_ids, *tar_ids]
    if len(set(all_source_ids)) != len(all_source_ids):
        errors.append("A source image is reused across reference/target roles")
    mevis_videos = [
        str(record["src_upstream"]["video_id"]) for record in records if record.get("source_dataset") == "mevis_v2"
    ]
    mevis_video_counts = Counter(mevis_videos)
    if max(mevis_video_counts.values(), default=0) > 2:
        errors.append("A MeViS video is used more than twice")
    if len(mevis_video_counts) < len(mevis_videos) - 4:
        errors.append("Too many repeated MeViS videos: " f"cases={len(mevis_videos)} unique={len(mevis_video_counts)}")
    for record in records:
        errors.extend(validate_identity_evidence(record))
        if record.get("difficulty") not in {"easy", "moderate"}:
            errors.append(f"{record.get('case_id')}: hard/unknown difficulty is forbidden")
        if record.get("auto_quality_pass") is not True:
            errors.append(f"{record.get('case_id')}: auto quality gate did not pass")
        errors.extend(_audit_record_files(dataset_root, record))

    for major in expected_quotas:
        major_rows = read_jsonl(dataset_root / "manifests" / f"{major}.jsonl")
        expected_rows = [record for record in records if record["major"] == major]
        if major_rows != expected_rows:
            errors.append(f"{major} manifest is not an exact primary-manifest subset")
    csv_path = dataset_root / "manifests" / "paperinsert_900.csv"
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
            expected_fieldnames = sorted({key for record in records for key in record})
            if reader.fieldnames != expected_fieldnames:
                errors.append("CSV manifest columns disagree with the JSONL manifest")
            if len(csv_rows) != len(records):
                errors.append("CSV manifest row count disagrees with the JSONL manifest")
            else:
                for index, (csv_row, record) in enumerate(zip(csv_rows, records)):
                    expected_csv_row: dict[str, str] = {}
                    for key in expected_fieldnames:
                        value = record.get(key)
                        if isinstance(value, (dict, list)):
                            expected_csv_row[key] = json.dumps(
                                value,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        else:
                            expected_csv_row[key] = "" if value is None else str(value)
                    if csv_row != expected_csv_row:
                        errors.append(f"CSV manifest row {index + 1} disagrees with the JSONL manifest")
                        break
    except (FileNotFoundError, csv.Error, UnicodeDecodeError) as error:
        errors.append(f"Cannot validate CSV manifest: {error}")
    errors.extend(verify_checksums(dataset_root, records))
    source_report: Mapping[str, Any] = {}
    try:
        source_report = read_json(dataset_root / "reports" / "sources.json")
        expected_source_report_fields = {
            "schema_version": 3,
            "manifest_schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "record_count": len(records),
            "identity_count": len(set(identities)),
            "quotas": MAJOR_MINOR_QUOTAS,
            "source_counts": dict(Counter(record["source_dataset"] for record in records)),
        }
        for field_name, expected_value in expected_source_report_fields.items():
            if source_report.get(field_name) != expected_value:
                errors.append(f"Source report {field_name} disagrees with the authenticated manifest")
        generator = source_report.get("generator", {})
        if generator.get("dependency_versions") != runtime_dependency_versions():
            errors.append("Source report dependency versions differ from the audit environment")
        for path_field, digest_field in (
            ("exclusion_config", "exclusion_config_sha256"),
            ("candidate_plan_cache", "candidate_plan_cache_sha256"),
            ("mevis_curation_pool", "mevis_curation_pool_sha256"),
        ):
            recorded_path = generator.get(path_field)
            recorded_digest = generator.get(digest_field)
            if not isinstance(recorded_path, str) or not isinstance(recorded_digest, str):
                errors.append(f"Source report is missing {path_field} provenance")
                continue
            provenance_path = Path(recorded_path)
            if (
                not provenance_path.is_file()
                or provenance_path.is_symlink()
                or file_sha256(provenance_path) != recorded_digest
            ):
                errors.append(f"Source report {path_field} no longer matches its recorded SHA256")
        script_digest = generator.get("script_sha256")
        if not isinstance(script_digest, str) or len(script_digest) != 64:
            errors.append("Source report builder SHA256 is missing or malformed")
        snapshot = source_report.get("source_snapshot", {})
        pinned_snapshot_names = {
            "mevis_jpeg_archive",
            "mevis_metadata",
            "mevis_mask_dict",
            "deepfashion_validation_archive",
            "deepfashion_query_metadata",
            "deepfashion_gallery_metadata",
        }
        if expected_count == 900 and any(
            snapshot.get(name, {}).get("verified") is not True for name in pinned_snapshot_names
        ):
            errors.append("Source report does not prove full pinned-source verification")
        archive_verification = source_report["selected_archive_verification"]
        if expected_count == 900:
            expected_archive_counts = {
                "mevis_image_member_count": 1200,
                "deepfashion_image_member_count": 600,
                "deepfashion_annotation_member_count": 600,
                "verified_member_count": 2400,
            }
            for field_name, expected_value in expected_archive_counts.items():
                if archive_verification.get(field_name) != expected_value:
                    errors.append(
                        f"Selected archive verification {field_name} mismatch: "
                        f"actual={archive_verification.get(field_name)!r} "
                        f"expected={expected_value}"
                    )
        if archive_verification.get("status") != ("verified_byte_exact_against_pinned_archives"):
            errors.append("Selected sources were not verified against pinned archives")
        verification_digest = archive_verification.get("verified_member_set_sha256")
        if not isinstance(verification_digest, str) or len(verification_digest) != 64:
            errors.append("Selected archive verification digest is missing or malformed")
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"Invalid selected archive verification evidence: {error}")

    if verify_raw:
        if raw_root is None:
            errors.append("Raw-backed audit requested without raw_root")
        else:
            try:
                current_source_snapshot = validate_source_snapshot(raw_root, verify_large_archives=True)
                if source_report.get("source_snapshot") != current_source_snapshot:
                    errors.append("Current pinned raw-source snapshot differs from the build source report")
            except (FileNotFoundError, ValueError) as error:
                errors.append(f"Pinned raw-source snapshot verification failed: {error}")
            mevis_root = raw_root / "mevis_v2" / "train"
            mevis_masks = read_json(mevis_root / "mask_dict.json")
            mevis_identities = _mevis_identity_index(mevis_root / "meta_expressions_v2.json")
            curation_pool_path = raw_root / "mevis_v2" / "curation" / MEVIS_CURATION_POOL_FILENAME
            if file_sha256(curation_pool_path) != MEVIS_CURATION_POOL_SHA256:
                errors.append("MeViS curation pool failed its pinned SHA256 during audit")
                curation_rows: dict[str, Mapping[str, Any]] = {}
            else:
                curation_payload = read_json(curation_pool_path)
                raw_curation_rows = curation_payload.get("candidates")
                membership_rows = curation_payload.get("counts", {}).get("membership_rows")
                if curation_payload.get("schema_version") != MEVIS_CURATION_POOL_SCHEMA:
                    errors.append("MeViS curation pool schema differs from the pinned schema")
                if not isinstance(raw_curation_rows, list) or not raw_curation_rows:
                    errors.append("MeViS curation pool has no candidate rows")
                    raw_curation_rows = []
                if (
                    isinstance(membership_rows, bool)
                    or not isinstance(membership_rows, int)
                    or membership_rows != len(raw_curation_rows)
                ):
                    errors.append("MeViS curation pool membership count is inconsistent")
                curation_rows = {str(row["candidate_id"]): row for row in raw_curation_rows}
                if len(curation_rows) != len(raw_curation_rows):
                    errors.append("MeViS curation pool contains duplicate candidate IDs")
            fashion_metadata_root = raw_root / "deepfashion2_validation" / "json_for_validation"
            fashion_query_members = {
                (
                    f"{int(row['query_image_id']):06d}",
                    int(row["pair_id"]),
                    int(row["style"]),
                    int(row["cls"]),
                    _bbox_tuple(row["bbox"]),
                )
                for row in read_json(fashion_metadata_root / "val_query.json")
                if int(row["style"]) > 0
            }
            fashion_gallery_members = {
                (
                    f"{int(row['gallery_image_id']):06d}",
                    int(row["pair_id"]),
                    int(row["style"]),
                    _bbox_tuple(row["bbox"]),
                )
                for row in read_json(fashion_metadata_root / "val_gallery.json")
                if int(row["style"]) > 0
            }
            for index, record in enumerate(records, start=1):
                if record.get("source_dataset") == "mevis_v2":
                    candidate_id = str(record.get("curation_candidate_id", ""))
                    curated = curation_rows.get(candidate_id)
                    if curated is None:
                        errors.append(f"{record['case_id']}: selected MeViS pair is absent from pinned curation pool")
                    else:
                        expected_membership = {
                            "identity_id": str(curated["identity_key"]),
                            "major": str(curated["major"]),
                            "minor": str(curated["minor"]),
                            "curation_candidate_source": str(curated["candidate_source"]),
                        }
                        for field_name, expected_value in expected_membership.items():
                            if str(record.get(field_name)) != expected_value:
                                errors.append(f"{record['case_id']}: MeViS curation {field_name} mismatch")
                        expected_penalty = float(curated["selection_priority_penalty"])
                        if float(record.get("selection_priority_penalty", -1.0)) != expected_penalty:
                            errors.append(f"{record['case_id']}: MeViS curation selection penalty mismatch")
                        for side in ("src", "tar"):
                            upstream = record[f"{side}_upstream"]
                            expected_upstream = {
                                "video_id": str(curated["video_id"]),
                                "anno_id": str(curated["anno_id"]),
                                "frame_id": str(curated[f"{side}_frame_name"]),
                                "frame_index": int(curated[f"{side}_frame_index"]),
                            }
                            for field_name, expected_value in expected_upstream.items():
                                actual_value = upstream.get(field_name)
                                if field_name == "frame_index":
                                    matches = int(actual_value) == expected_value
                                else:
                                    matches = str(actual_value) == expected_value
                                if not matches:
                                    errors.append(
                                        f"{record['case_id']}: MeViS curation {side} " f"{field_name} mismatch"
                                    )
                elif record.get("source_dataset") == "deepfashion2":
                    src_upstream = record["src_upstream"]
                    tar_upstream = record["tar_upstream"]
                    gallery_key = (
                        str(src_upstream["image_id"]),
                        int(src_upstream["pair_id"]),
                        int(src_upstream["style"]),
                        tuple(record["src_native_bbox_xyxy"]),
                    )
                    query_key = (
                        str(tar_upstream["image_id"]),
                        int(tar_upstream["pair_id"]),
                        int(tar_upstream["style"]),
                        int(tar_upstream["category_id"]),
                        tuple(record["tar_native_bbox_xyxy"]),
                    )
                    if gallery_key not in fashion_gallery_members:
                        errors.append(
                            f"{record['case_id']}: source item is absent from official " "DeepFashion2 gallery metadata"
                        )
                    if query_key not in fashion_query_members:
                        errors.append(
                            f"{record['case_id']}: target item is absent from official " "DeepFashion2 query metadata"
                        )
                errors.extend(
                    _raw_reconstruction_errors(
                        record,
                        raw_root,
                        mevis_masks=mevis_masks,
                        mevis_identities=mevis_identities,
                    )
                )
                if index % 100 == 0:
                    print(f"Raw-backed audit: {index}/{len(records)}", flush=True)

    garment_native_counts = Counter(
        int(record["native_category_id"]) for record in records if record.get("major") == "garment"
    )
    if expected_count == 900:
        expected_native = Counter(
            {
                category_id: quota
                for subquotas in GARMENT_NATIVE_SUBQUOTAS.values()
                for category_id, quota in subquotas.items()
            }
        )
        if garment_native_counts != expected_native:
            errors.append(
                f"DeepFashion2 native category coverage mismatch: {garment_native_counts} " f"vs {expected_native}"
            )
    report = {
        "schema_version": 1,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(records),
        "major_counts": dict(Counter(record["major"] for record in records)),
        "minor_counts": {f"{major}/{minor}": count for (major, minor), count in sorted(counts.items())},
        "source_counts": dict(Counter(record["source_dataset"] for record in records)),
        "pair_kind_counts": dict(Counter(record["pair_kind"] for record in records)),
        "difficulty_counts": dict(Counter(record["difficulty"] for record in records)),
        "garment_native_category_counts": dict(sorted(garment_native_counts.items())),
        "identity_count": len(set(identities)),
        "mevis_unique_video_count": len(mevis_video_counts),
        "mevis_repeated_video_count": sum(count > 1 for count in mevis_video_counts.values()),
        "different_source_image_count": sum(
            record.get("src_image_id") != record.get("tar_image_id") for record in records
        ),
        "target_derived_reference_count": sum(
            record.get("reference_derived_from_target") is not False for record in records
        ),
        "raw_backed_audit": bool(verify_raw and raw_root is not None),
        "manifest_sha256": file_sha256(manifest_path),
        "checksum_manifest_sha256": (
            file_sha256(dataset_root / "CHECKSUMS.sha256") if (dataset_root / "CHECKSUMS.sha256").is_file() else None
        ),
    }
    if write_report:
        atomic_write_text(
            dataset_root / "reports" / "audit.json",
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    return report


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _fit_panel(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGB", size, "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    panel.paste(fitted, offset)
    return panel


def _mask_outline(image: Image.Image, mask: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    mask = binary_mask(mask)
    values = np.asarray(mask, dtype=np.uint8) > 0
    padded = np.pad(values, 1, mode="constant")
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )
    outline = values & ~eroded
    result = np.asarray(image, dtype=np.uint8).copy()
    result[outline] = (255, 35, 35)
    return Image.fromarray(result, mode="RGB")


def _masked_target(image: Image.Image, mask: Image.Image) -> Image.Image:
    values = np.asarray(binary_mask(mask), dtype=np.uint8) > 0
    result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    yy, xx = np.indices(values.shape)
    checker = np.where(((xx // 12 + yy // 12) % 2)[..., None], 214, 240).astype(np.uint8)
    checker = np.repeat(checker, 3, axis=2)
    result[values] = checker[values]
    return _mask_outline(Image.fromarray(result, mode="RGB"), mask)


def _case_review_tile(
    dataset_root: Path,
    record: Mapping[str, Any],
    *,
    tile_size: tuple[int, int] = (620, 248),
) -> Image.Image:
    panel_size = (190, 190)
    with Image.open(dataset_root / str(record["src_image"])) as handle:
        src_image = handle.convert("RGB")
    with Image.open(dataset_root / str(record["src_mask"])) as handle:
        src_mask = handle.convert("L")
    with Image.open(dataset_root / str(record["tar_image"])) as handle:
        tar_image = handle.convert("RGB")
    with Image.open(dataset_root / str(record["tar_mask"])) as handle:
        tar_mask = handle.convert("L")
    panels = (
        _fit_panel(_mask_outline(src_image, src_mask), panel_size),
        _fit_panel(_masked_target(tar_image, tar_mask), panel_size),
        _fit_panel(_mask_outline(tar_image, tar_mask), panel_size),
    )
    tile = Image.new("RGB", tile_size, (248, 248, 248))
    for index, panel in enumerate(panels):
        tile.paste(panel, (8 + index * 202, 46))
    draw = ImageDraw.Draw(tile)
    draw.text((8, 4), str(record["case_id"]), font=_font(16), fill=(15, 15, 15))
    subtitle = (
        f"{record['source_dataset']} | {record['leaf_label']} | {record['difficulty']} | " f"{record['identity_id']}"
    )
    draw.text((8, 24), subtitle[:92], font=_font(11), fill=(55, 55, 55))
    for index, label in enumerate(("REF (independent)", "MASKED INPUT", "TARGET GT")):
        draw.text((10 + index * 202, 226), label, font=_font(11), fill=(35, 35, 35))
    return tile


def generate_qc_sheets(
    dataset_root: Path,
    *,
    cases_per_sheet: int = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    records = read_jsonl(dataset_root / "manifests" / "paperinsert_900.jsonl")
    output_root = dataset_root / "qc" / "full_review"
    if output_root.exists() and any(output_root.rglob("*.jpg")) and not overwrite:
        raise FileExistsError(f"QC sheets already exist: {output_root}")
    if overwrite and output_root.exists():
        # Only this derived, reproducible review directory is replaced.
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["major"], record["minor"])].append(record)
    for (major, minor), rows in grouped.items():
        rows.sort(key=lambda record: record["case_id"])
        for sheet_index, start in enumerate(range(0, len(rows), cases_per_sheet)):
            subset = rows[start : start + cases_per_sheet]
            columns = 2
            tile_width, tile_height = 620, 248
            canvas = Image.new(
                "RGB",
                (columns * tile_width, math.ceil(len(subset) / columns) * tile_height),
                (230, 230, 230),
            )
            for position, record in enumerate(subset):
                tile = _case_review_tile(dataset_root, record, tile_size=(tile_width, tile_height))
                canvas.paste(
                    tile,
                    ((position % columns) * tile_width, (position // columns) * tile_height),
                )
            relative = Path(major) / minor / f"sheet_{sheet_index:02d}.jpg"
            path = output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(path, format="JPEG", quality=92, subsampling=0)
            index_rows.append(
                {
                    "sheet": relative.as_posix(),
                    "major": major,
                    "minor": minor,
                    "case_ids": [record["case_id"] for record in subset],
                    "review_status": "pending",
                }
            )
    report = {
        "schema_version": 1,
        "sheet_count": len(index_rows),
        "case_count": sum(len(row["case_ids"]) for row in index_rows),
        "cases_per_sheet": cases_per_sheet,
        "status": "agent_visual_review_pending_not_human_certified",
        "sheets": index_rows,
    }
    atomic_write_text(
        dataset_root / "qc" / "index.json",
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build", "audit", "qc-previews", "analyze-sources"),
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--plan-cache", type=Path)
    parser.add_argument("--mevis-curation-pool", type=Path)
    parser.add_argument("--skip-large-archive-hash", action="store_true")
    parser.add_argument("--no-raw-audit", action="store_true")
    parser.add_argument("--overwrite-qc", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    work_root = args.work_root.resolve()
    raw_root = (args.raw_root or work_root / "_raw").resolve()
    output_root = (args.output_root or work_root / "paperinsert_900").resolve()
    plan_cache = (
        args.plan_cache or work_root / "_cache" / f"identity_paired_candidate_plans_seed{args.seed}.json.gz"
    ).resolve()
    mevis_curation_pool = (
        args.mevis_curation_pool or raw_root / "mevis_v2" / "curation" / MEVIS_CURATION_POOL_FILENAME
    ).resolve()
    if args.command == "analyze-sources":
        snapshot = validate_source_snapshot(raw_root, verify_large_archives=not args.skip_large_archive_hash)
        mevis = load_curated_mevis_candidate_plans(raw_root / "mevis_v2" / "train", mevis_curation_pool)
        fashion = build_deepfashion_candidate_plans(raw_root / "deepfashion2_validation", args.seed)
        report = {
            "source_snapshot": snapshot,
            "mevis_candidate_counts": {
                f"{major}/{minor}": count
                for (major, minor), count in sorted(Counter((plan.major, plan.minor) for plan in mevis).items())
            },
            "deepfashion_candidate_counts": {
                f"{minor}/category_{category}": count
                for (minor, category), count in sorted(
                    Counter((plan.minor, int(plan.extra["native_category_id"])) for plan in fashion).items()
                )
            },
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if args.command == "build":
        records = build_dataset(
            raw_root,
            output_root,
            seed=args.seed,
            workers=args.workers,
            exclusion_path=args.exclusions,
            plan_cache_path=plan_cache,
            mevis_curation_pool_path=mevis_curation_pool,
            verify_large_archives=not args.skip_large_archive_hash,
        )
        report = audit_dataset(
            output_root,
            raw_root=raw_root,
            verify_raw=not args.no_raw_audit,
        )
        if not report["valid"]:
            raise ValueError(
                f"Built candidate failed audit with {len(report['errors'])} errors: " + "; ".join(report["errors"][:10])
            )
        generate_qc_sheets(output_root)
        print(
            json.dumps(
                {
                    "output_root": str(output_root),
                    "record_count": len(records),
                    "manifest_sha256": report["manifest_sha256"],
                    "audit_valid": True,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "audit":
        report = audit_dataset(
            output_root,
            raw_root=raw_root,
            verify_raw=not args.no_raw_audit,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if report["valid"] else 1
    report = generate_qc_sheets(output_root, overwrite=args.overwrite_qc)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

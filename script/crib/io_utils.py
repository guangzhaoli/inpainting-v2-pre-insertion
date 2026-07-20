"""Manifest / image IO helpers for the CRIB sidecar builder."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np


REQUIRED_MANIFEST_FIELDS = (
    "ref_image",
    "ref_mask",
    "tar_image",
    "tar_mask",
    "category",
    "stem",
)


@dataclass(frozen=True)
class SampleRecord:
    sample_uid: str
    category: str
    stem: str
    ref_image: str
    ref_mask: str
    tar_image: str
    tar_mask: str
    prompt: str = ""
    row_index: int = -1

    def resolve(self, dataset_root: Path, key: str) -> Path:
        rel = getattr(self, key)
        path = Path(rel)
        if path.is_absolute():
            return path
        return dataset_root / path


def make_sample_uid(category: str, stem: str) -> str:
    return f"{category}/{stem}"


def load_manifest(
    manifest_path: Path,
    *,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    shard_index: Optional[int] = None,
    num_shards: Optional[int] = None,
) -> List[SampleRecord]:
    """Load manifest rows with optional offset/limit or strided sharding.

    Sharding (for multi-GPU embedding extraction):
      keep rows where kept_index % num_shards == shard_index
    applied after category filter, before offset/limit is not mixed with shards:
      if shard_index/num_shards set, offset/limit are applied on the shard subset.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if shard_index is not None or num_shards is not None:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards must be provided together")
        if num_shards < 1:
            raise ValueError("num_shards must be >= 1")
        if not (0 <= shard_index < num_shards):
            raise ValueError(f"shard_index must be in [0, {num_shards})")

    allowed = set(categories) if categories else None
    records: List[SampleRecord] = []
    kept_index = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty manifest: {manifest_path}")
        missing = [c for c in REQUIRED_MANIFEST_FIELDS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"manifest missing columns {missing}: {manifest_path}")

        for row_index, row in enumerate(reader):
            category = (row.get("category") or "").strip()
            if allowed is not None and category not in allowed:
                continue
            # kept_index counts category-filtered rows for strided sharding
            if num_shards is not None and (kept_index % num_shards) != shard_index:
                kept_index += 1
                continue
            kept_index += 1

            stem = (row.get("stem") or "").strip()
            if not stem:
                stem = Path(row["ref_image"]).stem
            uid = make_sample_uid(category, stem)
            records.append(
                SampleRecord(
                    sample_uid=uid,
                    category=category,
                    stem=stem,
                    ref_image=row["ref_image"],
                    ref_mask=row["ref_mask"],
                    tar_image=row["tar_image"],
                    tar_mask=row["tar_mask"],
                    prompt=(row.get("prompt") or "").strip(),
                    row_index=row_index,
                )
            )

    # offset/limit apply after category + shard filters
    if offset:
        records = records[offset:]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"no rows loaded from manifest: {manifest_path}")
    return records


def load_rgb_image(path: Path) -> np.ndarray:
    path = Path(path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path) -> np.ndarray:
    path = Path(path)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"failed to read mask: {path}")
    return mask


def mask_area_ratio(mask: np.ndarray, threshold: float = 0.5) -> float:
    if mask.size == 0:
        return 0.0
    binary = mask.astype(np.float32)
    if binary.max() > 1.0:
        binary = binary / 255.0
    return float((binary >= threshold).mean())


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

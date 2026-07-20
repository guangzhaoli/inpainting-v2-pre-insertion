"""Similarity stats + lightweight human review pack export."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from .features import prepare_masked_ref_crop
from .io_utils import SampleRecord, load_mask, load_rgb_image, write_json


def summarize_similarity_stats(neighbor_scores: np.ndarray) -> Dict[str, Any]:
    """Summarize top-1 / top-k neighbor cosine similarities."""
    if neighbor_scores.size == 0:
        return {
            "top1_mean": None,
            "top1_median": None,
            "top1_p90": None,
            "frac_top1_ge_0.90": None,
            "frac_top1_ge_0.95": None,
            "frac_top1_ge_0.98": None,
        }
    # only finite scores
    top1 = neighbor_scores[:, 0]
    top1 = top1[np.isfinite(top1)]
    if top1.size == 0:
        return {
            "top1_mean": None,
            "top1_median": None,
            "top1_p90": None,
            "frac_top1_ge_0.90": None,
            "frac_top1_ge_0.95": None,
            "frac_top1_ge_0.98": None,
        }
    return {
        "top1_mean": float(np.mean(top1)),
        "top1_median": float(np.median(top1)),
        "top1_p90": float(np.quantile(top1, 0.90)),
        "frac_top1_ge_0.90": float(np.mean(top1 >= 0.90)),
        "frac_top1_ge_0.95": float(np.mean(top1 >= 0.95)),
        "frac_top1_ge_0.98": float(np.mean(top1 >= 0.98)),
        "all_neighbor_mean": float(np.mean(neighbor_scores[np.isfinite(neighbor_scores)])),
    }


def _save_side_by_side(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    out_path: Path,
    label: str,
) -> None:
    h = max(left_rgb.shape[0], right_rgb.shape[0])
    w = left_rgb.shape[1] + right_rgb.shape[1] + 8
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    xoff = left_rgb.shape[1] + 8
    canvas[: right_rgb.shape[0], xoff : xoff + right_rgb.shape[1]] = right_rgb
    # draw label bar
    bar = np.full((28, w, 3), 20, dtype=np.uint8)
    out = np.concatenate([bar, canvas], axis=0)
    cv2.putText(
        out,
        label[:120],
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _load_preview_crop(record: SampleRecord, dataset_root: Path, image_size: int = 256) -> np.ndarray:
    image = load_rgb_image(record.resolve(dataset_root, "ref_image"))
    mask = load_mask(record.resolve(dataset_root, "ref_mask"))
    crop, _, _, _ = prepare_masked_ref_crop(
        image,
        mask,
        image_size=image_size,
        bbox_pad_ratio=0.15,
    )
    return crop


def export_review_packs(
    *,
    records: Sequence[SampleRecord],
    dataset_root: Path,
    sidecar_rows: Sequence[dict],
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    sample_uids: Sequence[str],
    output_dir: Path,
    n_hard_pairs: int = 120,
    n_same_id_clusters: int = 40,
    seed: int = 0,
) -> Dict[str, Any]:
    """Export CSV + image pairs for manual QC."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    uid_to_rec = {r.sample_uid: r for r in records}
    rng = random.Random(seed)

    # --- hard-neg pair review ---
    hard_candidates = []
    for row in sidecar_rows:
        for neg_uid, sc in zip(row.get("hard_neg_uids", []), row.get("hard_neg_scores", [])):
            hard_candidates.append((row["sample_uid"], neg_uid, float(sc), row["instance_id"]))
    rng.shuffle(hard_candidates)
    hard_selected = hard_candidates[:n_hard_pairs]

    hard_csv = output_dir / "hard_neg_pairs.csv"
    hard_img_dir = output_dir / "hard_neg_images"
    hard_img_dir.mkdir(parents=True, exist_ok=True)
    with hard_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anchor_uid",
                "neg_uid",
                "sim",
                "anchor_instance_id",
                "image_path",
                "human_ok",
                "notes",
            ],
        )
        writer.writeheader()
        for i, (a_uid, n_uid, sc, iid) in enumerate(hard_selected):
            img_rel = f"hard_neg_images/{i:04d}_{a_uid.replace('/', '_')}__{n_uid.replace('/', '_')}.jpg"
            img_path = output_dir / img_rel
            try:
                left = _load_preview_crop(uid_to_rec[a_uid], dataset_root)
                right = _load_preview_crop(uid_to_rec[n_uid], dataset_root)
                _save_side_by_side(left, right, img_path, f"HARD {a_uid} vs {n_uid} sim={sc:.3f}")
            except Exception as exc:  # pragma: no cover
                img_rel = f"ERROR: {exc}"
            writer.writerow(
                {
                    "anchor_uid": a_uid,
                    "neg_uid": n_uid,
                    "sim": f"{sc:.6f}",
                    "anchor_instance_id": iid,
                    "image_path": img_rel,
                    "human_ok": "",
                    "notes": "",
                }
            )

    # --- same-id cluster review ---
    clusters: Dict[str, List[str]] = {}
    for row in sidecar_rows:
        clusters.setdefault(row["instance_id"], []).append(row["sample_uid"])
    multi = [(iid, uids) for iid, uids in clusters.items() if len(uids) >= 2]
    multi.sort(key=lambda x: -len(x[1]))
    # sample some multi clusters
    if len(multi) > n_same_id_clusters:
        multi = rng.sample(multi, n_same_id_clusters)

    same_csv = output_dir / "same_id_clusters.csv"
    same_img_dir = output_dir / "same_id_images"
    same_img_dir.mkdir(parents=True, exist_ok=True)
    with same_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id",
                "size",
                "member_uids",
                "pair_image_path",
                "human_ok",
                "notes",
            ],
        )
        writer.writeheader()
        for i, (iid, uids) in enumerate(multi):
            a_uid, b_uid = uids[0], uids[1]
            img_rel = f"same_id_images/{i:04d}_{iid}.jpg"
            img_path = output_dir / img_rel
            try:
                left = _load_preview_crop(uid_to_rec[a_uid], dataset_root)
                right = _load_preview_crop(uid_to_rec[b_uid], dataset_root)
                _save_side_by_side(left, right, img_path, f"SAME {iid} size={len(uids)}")
            except Exception as exc:  # pragma: no cover
                img_rel = f"ERROR: {exc}"
            writer.writerow(
                {
                    "instance_id": iid,
                    "size": len(uids),
                    "member_uids": "|".join(uids),
                    "pair_image_path": img_rel,
                    "human_ok": "",
                    "notes": "",
                }
            )

    # --- top1 neighbor dump for threshold tuning ---
    top1_csv = output_dir / "top1_neighbors.csv"
    with top1_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_uid", "neighbor_uid", "sim", "same_category"],
        )
        writer.writeheader()
        uid_to_cat = {r.sample_uid: r.category for r in records}
        for i, uid in enumerate(sample_uids):
            if neighbor_indices.shape[1] == 0:
                continue
            j = int(neighbor_indices[i, 0])
            if j < 0:
                continue
            n_uid = sample_uids[j]
            writer.writerow(
                {
                    "sample_uid": uid,
                    "neighbor_uid": n_uid,
                    "sim": f"{float(neighbor_scores[i, 0]):.6f}",
                    "same_category": uid_to_cat[uid] == uid_to_cat[n_uid],
                }
            )

    meta = {
        "n_hard_pairs_exported": len(hard_selected),
        "n_same_id_clusters_exported": len(multi),
        "hard_neg_csv": str(hard_csv),
        "same_id_csv": str(same_csv),
        "top1_csv": str(top1_csv),
    }
    write_json(output_dir / "review_meta.json", meta)
    return meta

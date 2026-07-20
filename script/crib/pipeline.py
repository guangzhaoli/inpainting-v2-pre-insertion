"""End-to-end CRIB sidecar construction pipeline."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cluster import ClusterResult, cluster_same_id
from .config import CribBuildConfig
from .features import extract_embeddings, l2_normalize
from .hard_neg import HardNegResult, build_instance_index, mine_hard_negatives
from .io_utils import SampleRecord, load_manifest, write_json, write_jsonl
from .neighbors import knn_search
from .review import export_review_packs, summarize_similarity_stats

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineArraysResult:
    sample_uids: List[str]
    categories: List[str]
    instance_ids: List[str]
    hard_neg_uids: List[List[str]]
    hard_neg_scores: List[List[float]]
    neighbor_indices: np.ndarray
    neighbor_scores: np.ndarray
    cluster: ClusterResult
    hard_neg: HardNegResult
    stats: Dict[str, Any]
    sidecar_rows: List[Dict[str, Any]]


def run_pipeline_from_arrays(
    embeddings: np.ndarray,
    sample_uids: Sequence[str],
    categories: Sequence[str],
    config: CribBuildConfig,
    *,
    feature_meta: Optional[Sequence[dict]] = None,
    stems: Optional[Sequence[str]] = None,
) -> PipelineArraysResult:
    """Pure array pipeline used by tests and by the full builder."""
    config.validate()
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    n = embeddings.shape[0]
    if n != len(sample_uids) or n != len(categories):
        raise ValueError("embeddings / sample_uids / categories length mismatch")

    neighbor_indices, neighbor_scores = knn_search(
        embeddings,
        topk=config.neighbor_topk,
        use_faiss=config.use_faiss,
    )

    cluster = cluster_same_id(
        neighbor_indices,
        neighbor_scores,
        sample_uids,
        categories,
        sim_threshold=config.same_id_sim_threshold,
        require_mutual=config.require_mutual_nearest_for_same_id,
    )

    hard_neg = mine_hard_negatives(
        neighbor_indices,
        neighbor_scores,
        sample_uids,
        categories,
        cluster.instance_ids,
        topk=config.hard_neg_topk,
        min_sim=config.hard_neg_min_sim,
        max_sim=config.hard_neg_max_sim,
        same_category_only=config.hard_neg_same_category_only,
    )

    instance_index = build_instance_index(sample_uids, cluster.instance_ids)
    sim_stats = summarize_similarity_stats(neighbor_scores)

    stems = list(stems) if stems is not None else [uid.split("/", 1)[-1] for uid in sample_uids]
    feature_meta = list(feature_meta) if feature_meta is not None else [{} for _ in range(n)]

    sidecar_rows: List[Dict[str, Any]] = []
    for i, uid in enumerate(sample_uids):
        iid = cluster.instance_ids[i]
        meta = feature_meta[i] if i < len(feature_meta) else {}
        row = {
            "sample_uid": uid,
            "category": categories[i],
            "stem": stems[i],
            "instance_id": iid,
            "instance_conf": 1.0 if cluster.cluster_sizes[iid] > 1 else 0.5,
            "same_id_size": int(cluster.cluster_sizes[iid]),
            "same_id_group": list(instance_index[iid]),
            "hard_neg_uids": hard_neg.hard_neg_uids[i],
            "hard_neg_scores": [round(float(s), 6) for s in hard_neg.hard_neg_scores[i]],
            "mask_area_ratio": float(meta.get("mask_area_ratio", -1.0)),
            "bbox": meta.get("bbox"),
            "used_full_frame_fallback": bool(meta.get("used_full_frame_fallback", False)),
            "top1_neighbor_uid": (
                sample_uids[int(neighbor_indices[i, 0])]
                if neighbor_indices.shape[1] > 0 and int(neighbor_indices[i, 0]) >= 0
                else None
            ),
            "top1_neighbor_sim": (
                float(neighbor_scores[i, 0])
                if neighbor_scores.shape[1] > 0 and np.isfinite(neighbor_scores[i, 0])
                else None
            ),
        }
        sidecar_rows.append(row)

    stats = {
        "n_samples": n,
        "embedding_dim": int(embeddings.shape[1]),
        "neighbor_topk": int(neighbor_indices.shape[1]),
        "same_id": {
            "threshold": config.same_id_sim_threshold,
            "edge_count": cluster.edge_count,
            "n_clusters": len(cluster.cluster_sizes),
            "n_singleton": cluster.n_singleton,
            "n_multi": cluster.n_multi,
            "singleton_ratio": cluster.n_singleton / max(len(cluster.cluster_sizes), 1),
            "size_hist": _size_hist(cluster.cluster_sizes),
        },
        "hard_neg": {
            "topk": config.hard_neg_topk,
            "min_sim": config.hard_neg_min_sim,
            "max_sim": config.hard_neg_max_sim,
            "n_with_any": hard_neg.n_with_any,
            "n_full": hard_neg.n_full,
            "coverage": hard_neg.n_with_any / max(n, 1),
            "full_coverage": hard_neg.n_full / max(n, 1),
            "mean_count": hard_neg.mean_count,
        },
        "similarity": sim_stats,
        "config": config.to_dict(),
    }

    return PipelineArraysResult(
        sample_uids=list(sample_uids),
        categories=list(categories),
        instance_ids=list(cluster.instance_ids),
        hard_neg_uids=hard_neg.hard_neg_uids,
        hard_neg_scores=hard_neg.hard_neg_scores,
        neighbor_indices=neighbor_indices,
        neighbor_scores=neighbor_scores,
        cluster=cluster,
        hard_neg=hard_neg,
        stats=stats,
        sidecar_rows=sidecar_rows,
    )


def _size_hist(cluster_sizes: Dict[str, int]) -> Dict[str, int]:
    hist = {"1": 0, "2": 0, "3-5": 0, "6-10": 0, "11+": 0}
    for size in cluster_sizes.values():
        if size == 1:
            hist["1"] += 1
        elif size == 2:
            hist["2"] += 1
        elif size <= 5:
            hist["3-5"] += 1
        elif size <= 10:
            hist["6-10"] += 1
        else:
            hist["11+"] += 1
    return hist


def build_crib_sidecar(
    *,
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    config: Optional[CribBuildConfig] = None,
    categories: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Full offline builder: features -> knn -> cluster -> hard neg -> files."""
    config = config or CribBuildConfig()
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)

    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_manifest(
        manifest_path,
        categories=categories,
        limit=config.limit,
        offset=config.offset,
        shard_index=config.shard_index,
        num_shards=config.num_shards,
    )
    LOGGER.info(
        "Loaded %d manifest rows from %s (offset=%s limit=%s shard=%s/%s)",
        len(records),
        manifest_path,
        config.offset,
        config.limit,
        config.shard_index,
        config.num_shards,
    )

    embeddings, feature_meta = extract_embeddings(records, dataset_root, config)
    sample_uids = [r.sample_uid for r in records]
    categories_list = [r.category for r in records]
    stems = [r.stem for r in records]

    # Persist feature artifacts first (useful for multi-GPU shard jobs)
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings)
    write_json(output_dir / "sample_uids.json", sample_uids)
    write_json(output_dir / "categories.json", categories_list)
    write_json(output_dir / "stems.json", stems)
    write_json(output_dir / "feature_meta.json", list(feature_meta))
    write_json(output_dir / "config.json", config.to_dict())

    if config.embeddings_only:
        summary = {
            "output_dir": str(output_dir),
            "n_samples": len(sample_uids),
            "embeddings_path": str(emb_path),
            "embeddings_only": True,
            "shard_index": config.shard_index,
            "num_shards": config.num_shards,
        }
        write_json(output_dir / "summary.json", summary)
        LOGGER.info("Embeddings-only shard done: n=%d -> %s", len(sample_uids), emb_path)
        return summary

    result = run_pipeline_from_arrays(
        embeddings,
        sample_uids,
        categories_list,
        config,
        feature_meta=feature_meta,
        stems=stems,
    )

    write_json(output_dir / "stats.json", result.stats)
    n_side = write_jsonl(output_dir / "sidecar.jsonl", result.sidecar_rows)
    write_json(
        output_dir / "instance_index.json",
        build_instance_index(sample_uids, result.instance_ids),
    )

    if config.save_neighbor_matrix:
        np.savez_compressed(
            output_dir / "neighbors.npz",
            indices=result.neighbor_indices,
            scores=result.neighbor_scores,
        )

    review_dir = output_dir / "review"
    export_review_packs(
        records=records,
        dataset_root=dataset_root,
        sidecar_rows=result.sidecar_rows,
        neighbor_indices=result.neighbor_indices,
        neighbor_scores=result.neighbor_scores,
        sample_uids=sample_uids,
        output_dir=review_dir,
        n_hard_pairs=config.review_pairs,
        n_same_id_clusters=config.review_same_id_clusters,
        seed=config.seed,
    )

    summary = {
        "output_dir": str(output_dir),
        "n_sidecar_rows": n_side,
        "embeddings_path": str(emb_path),
        "stats": result.stats,
    }
    write_json(output_dir / "summary.json", summary)
    LOGGER.info(
        "CRIB sidecar done: n=%d singleton_clusters=%d multi=%d hard_cov=%.3f",
        result.stats["n_samples"],
        result.stats["same_id"]["n_singleton"],
        result.stats["same_id"]["n_multi"],
        result.stats["hard_neg"]["coverage"],
    )
    return summary

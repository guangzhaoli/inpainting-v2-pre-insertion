#!/usr/bin/env python3
"""Merge multi-GPU embedding shards and finish CRIB clustering/hard-neg mining.

Stage-1 (per GPU):
  python -m script.crib.build_crib_sidecar ... --shard-index i --num-shards 8 --embeddings-only

Stage-2 (single process, CPU/GPU irrelevant for KNN on 15k):
  python -m script.crib.merge_embedding_shards \\
    --shard-dirs outputs/crib/full/shards/shard_{0..7} \\
    --output-dir outputs/crib/full \\
    --dataset-root ... --manifest ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from script.crib.config import CribBuildConfig  # noqa: E402
from script.crib.hard_neg import build_instance_index  # noqa: E402
from script.crib.io_utils import load_manifest, write_json, write_jsonl  # noqa: E402
from script.crib.pipeline import run_pipeline_from_arrays  # noqa: E402
from script.crib.review import export_review_packs  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _load_shard(shard_dir: Path):
    emb = np.load(shard_dir / "embeddings.npy")
    uids = json.loads((shard_dir / "sample_uids.json").read_text())
    cats = json.loads((shard_dir / "categories.json").read_text())
    stems = json.loads((shard_dir / "stems.json").read_text())
    meta = json.loads((shard_dir / "feature_meta.json").read_text())
    if not (len(uids) == len(cats) == len(stems) == len(meta) == emb.shape[0]):
        raise ValueError(f"shard length mismatch in {shard_dir}")
    return emb, uids, cats, stems, meta


def merge_shards(shard_dirs: List[Path]):
    all_emb, all_uids, all_cats, all_stems, all_meta = [], [], [], [], []
    seen = set()
    for d in shard_dirs:
        emb, uids, cats, stems, meta = _load_shard(Path(d))
        for i, uid in enumerate(uids):
            if uid in seen:
                raise ValueError(f"duplicate sample_uid across shards: {uid}")
            seen.add(uid)
            all_emb.append(emb[i])
            all_uids.append(uid)
            all_cats.append(cats[i])
            all_stems.append(stems[i])
            all_meta.append(meta[i])
    embeddings = np.stack(all_emb, axis=0).astype(np.float32)
    return embeddings, all_uids, all_cats, all_stems, all_meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--same-id-threshold", type=float, default=0.97)
    p.add_argument("--hard-neg-min-sim", type=float, default=0.70)
    p.add_argument("--hard-neg-max-sim", type=float, default=0.97)
    p.add_argument("--hard-neg-topk", type=int, default=8)
    p.add_argument("--neighbor-topk", type=int, default=50)
    p.add_argument(
        "--require-mutual-same-id",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--review-pairs", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-faiss", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    embeddings, uids, cats, stems, meta = merge_shards(args.shard_dirs)
    LOGGER.info("Merged %d samples from %d shards", len(uids), len(args.shard_dirs))

    cfg = CribBuildConfig(
        same_id_sim_threshold=args.same_id_threshold,
        require_mutual_nearest_for_same_id=args.require_mutual_same_id,
        hard_neg_min_sim=args.hard_neg_min_sim,
        hard_neg_max_sim=args.hard_neg_max_sim,
        hard_neg_topk=args.hard_neg_topk,
        neighbor_topk=args.neighbor_topk,
        use_faiss=not args.no_faiss,
        review_pairs=args.review_pairs,
        seed=args.seed,
    )
    result = run_pipeline_from_arrays(
        embeddings, uids, cats, cfg, feature_meta=meta, stems=stems
    )

    np.save(out / "embeddings.npy", embeddings)
    write_json(out / "sample_uids.json", uids)
    write_json(out / "categories.json", cats)
    write_json(out / "stems.json", stems)
    write_json(out / "feature_meta.json", meta)
    write_json(out / "config.json", cfg.to_dict())
    write_json(out / "stats.json", result.stats)
    write_jsonl(out / "sidecar.jsonl", result.sidecar_rows)
    write_json(out / "instance_index.json", build_instance_index(uids, result.instance_ids))
    np.savez_compressed(
        out / "neighbors.npz",
        indices=result.neighbor_indices,
        scores=result.neighbor_scores,
    )

    # rebuild records for review export (full manifest, then filter to uids)
    all_records = load_manifest(args.manifest, categories=args.categories)
    uid_set = set(uids)
    records = [r for r in all_records if r.sample_uid in uid_set]
    # preserve embedding order
    uid2rec = {r.sample_uid: r for r in records}
    ordered = [uid2rec[u] for u in uids if u in uid2rec]
    export_review_packs(
        records=ordered,
        dataset_root=Path(args.dataset_root),
        sidecar_rows=result.sidecar_rows,
        neighbor_indices=result.neighbor_indices,
        neighbor_scores=result.neighbor_scores,
        sample_uids=uids,
        output_dir=out / "review",
        n_hard_pairs=args.review_pairs,
        seed=args.seed,
    )

    summary = {
        "output_dir": str(out),
        "n_samples": len(uids),
        "stats": result.stats,
        "shard_dirs": [str(p) for p in args.shard_dirs],
    }
    write_json(out / "summary.json", summary)
    print("=== merge complete ===")
    print(json.dumps(result.stats, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

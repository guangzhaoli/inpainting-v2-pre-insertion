#!/usr/bin/env python3
"""Build CRIB identity + hard-negative sidecar for InsertAnything manifests.

This script is intentionally standalone:
- does NOT modify training code / Dataset
- only reads manifest + images
- writes artifacts under --output-dir

Examples
--------
Smoke (CPU, 64 samples):

  python -m script.crib.build_crib_sidecar \\
    --dataset-root /path/to/a2edit/data \\
    --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \\
    --output-dir outputs/crib/smoke64 --limit 64 --device cpu

8-GPU stage-1 (embeddings only, one process per GPU):

  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i python -m script.crib.build_crib_sidecar \\
      --dataset-root /path/to/a2edit/data \\
      --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \\
      --output-dir outputs/crib/full/shards/shard_$i \\
      --shard-index $i --num-shards 8 \\
      --embeddings-only --device cuda --batch-size 64 &
  done
  wait
  python -m script.crib.merge_embedding_shards \\
    --shard-dirs outputs/crib/full/shards/shard_{0..7} \\
    --output-dir outputs/crib/full \\
    --dataset-root /path/to/a2edit/data \\
    --manifest manifests/insert_anything_teacher_accessory_jewelry.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from script.crib.config import CribBuildConfig  # noqa: E402
from script.crib.pipeline import build_crib_sidecar  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "manifests/insert_anything_teacher_accessory_jewelry.csv",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Only extract DINOv2 embeddings (for multi-GPU stage-1)",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--model-name", type=str, default="dinov2_vits14")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--neighbor-topk", type=int, default=50)
    p.add_argument("--hard-neg-topk", type=int, default=8)
    p.add_argument("--same-id-threshold", type=float, default=0.97)
    p.add_argument("--hard-neg-min-sim", type=float, default=0.70)
    p.add_argument("--hard-neg-max-sim", type=float, default=0.97)
    p.add_argument(
        "--require-mutual-same-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require mutual nearest-neighbor edge for same-id merge (default: false)",
    )
    p.add_argument("--no-faiss", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--review-pairs", type=int, default=120)
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = CribBuildConfig(
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        neighbor_topk=args.neighbor_topk,
        use_faiss=not args.no_faiss,
        same_id_sim_threshold=args.same_id_threshold,
        require_mutual_nearest_for_same_id=args.require_mutual_same_id,
        hard_neg_topk=args.hard_neg_topk,
        hard_neg_min_sim=args.hard_neg_min_sim,
        hard_neg_max_sim=args.hard_neg_max_sim,
        review_pairs=args.review_pairs,
        seed=args.seed,
        limit=args.limit,
        offset=args.offset,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        embeddings_only=args.embeddings_only,
    )

    summary = build_crib_sidecar(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        config=config,
        categories=args.categories,
    )
    print("=== CRIB sidecar build complete ===")
    print(f"output_dir: {summary['output_dir']}")
    if summary.get("embeddings_only"):
        print(f"n_samples:  {summary['n_samples']} (embeddings only)")
    else:
        print(f"n_samples:  {summary['stats']['n_samples']}")
        print(f"same_id:    {summary['stats']['same_id']}")
        print(f"hard_neg:   {summary['stats']['hard_neg']}")
        print(f"similarity: {summary['stats']['similarity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

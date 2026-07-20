#!/usr/bin/env python3
"""Search CRIB thresholds on one or more finished embedding runs.

Scores each (same_id_thr, mutual, hard_min, hard_max) config by:
  + hard_neg coverage / mean_count
  - presence of huge same-id clusters (11+)
  - excessive multi-merge edges
  - same-id under-merge penalty if multi is extremely low

Example:
  python -m script.crib.search_best_params \\
    --run-dirs outputs/crib/accessory_jewelry_validate1024 \\
               outputs/crib/accessory_jewelry_tune2048
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from script.crib.config import CribBuildConfig  # noqa: E402
from script.crib.pipeline import run_pipeline_from_arrays  # noqa: E402


@dataclass
class CandScore:
    same_id_thr: float
    require_mutual: bool
    hard_min: float
    hard_max: float
    hard_topk: int
    score: float
    hard_coverage: float
    hard_mean_count: float
    n_multi: int
    n_11plus: int
    n_6_10: int
    edge_count: int
    singleton_ratio: float
    notes: str = ""


def load_run(run_dir: Path):
    emb = np.load(run_dir / "embeddings.npy")
    uids = json.loads((run_dir / "sample_uids.json").read_text())
    # prefer categories.json if present
    cat_path = run_dir / "categories.json"
    if cat_path.is_file():
        cats = json.loads(cat_path.read_text())
    else:
        cats = [json.loads(l)["category"] for l in open(run_dir / "sidecar.jsonl")]
    return emb, uids, cats


def score_config(
    emb,
    uids,
    cats,
    *,
    same_id_thr: float,
    require_mutual: bool,
    hard_min: float,
    hard_max: float,
    hard_topk: int = 8,
) -> CandScore:
    if hard_max > same_id_thr:
        hard_max = same_id_thr
    cfg = CribBuildConfig(
        same_id_sim_threshold=same_id_thr,
        require_mutual_nearest_for_same_id=require_mutual,
        hard_neg_min_sim=hard_min,
        hard_neg_max_sim=hard_max,
        hard_neg_topk=hard_topk,
        neighbor_topk=50,
        use_faiss=True,
        hard_neg_same_category_only=False,
    )
    r = run_pipeline_from_arrays(emb, uids, cats, cfg)
    s = r.stats["same_id"]
    h = r.stats["hard_neg"]
    hist = s["size_hist"]
    n_11 = int(hist.get("11+", 0))
    n_6 = int(hist.get("6-10", 0))

    # Score design (higher better):
    # - reward hard-neg usability
    # - strongly punish giant clusters (likely wrong merges)
    # - mild punish medium clusters
    # - mild punish too few multi clusters only if coverage of near-dups is high
    score = 0.0
    score += 2.5 * h["coverage"]
    score += 0.08 * min(h["mean_count"], 8.0)
    score += 0.5 * h["full_coverage"]
    score -= 3.0 * n_11
    score -= 0.6 * n_6
    # prefer some multi clusters (true near-dups exist), but not too many edges
    score += 0.01 * min(s["n_multi"], 80)
    if s["edge_count"] > 0.2 * len(uids):
        score -= 0.5

    notes = []
    if n_11:
        notes.append(f"has {n_11} size>=11 clusters")
    if h["coverage"] < 0.7:
        notes.append("low hard-neg coverage")
    if s["n_multi"] == 0:
        notes.append("no multi same-id clusters")

    return CandScore(
        same_id_thr=same_id_thr,
        require_mutual=require_mutual,
        hard_min=hard_min,
        hard_max=hard_max,
        hard_topk=hard_topk,
        score=float(score),
        hard_coverage=float(h["coverage"]),
        hard_mean_count=float(h["mean_count"]),
        n_multi=int(s["n_multi"]),
        n_11plus=n_11,
        n_6_10=n_6,
        edge_count=int(s["edge_count"]),
        singleton_ratio=float(s["singleton_ratio"]),
        notes="; ".join(notes),
    )


def search_one_run(run_dir: Path) -> Dict[str, Any]:
    emb, uids, cats = load_run(run_dir)
    cands: List[CandScore] = []
    same_thrs = [0.94, 0.95, 0.96, 0.97, 0.98]
    hard_mins = [0.65, 0.68, 0.70, 0.72, 0.75]
    mutual_opts = [False, True]
    for thr in same_thrs:
        for mutual in mutual_opts:
            for mn in hard_mins:
                mx = thr
                cands.append(
                    score_config(
                        emb,
                        uids,
                        cats,
                        same_id_thr=thr,
                        require_mutual=mutual,
                        hard_min=mn,
                        hard_max=mx,
                        hard_topk=8,
                    )
                )
    cands.sort(key=lambda c: c.score, reverse=True)
    top = cands[:15]
    best = cands[0]
    # also report constrained best with n_11plus==0
    safe = [c for c in cands if c.n_11plus == 0]
    safe.sort(key=lambda c: c.score, reverse=True)
    best_safe = safe[0] if safe else best
    return {
        "run_dir": str(run_dir),
        "n_samples": len(uids),
        "best": asdict(best),
        "best_safe_no_11plus": asdict(best_safe),
        "top15": [asdict(c) for c in top],
        "all": [asdict(c) for c in cands],
    }


def aggregate_recommendation(per_run: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick params that frequently rank high across runs."""
    # vote on best_safe configs
    votes: Dict[tuple, List[float]] = {}
    for run in per_run:
        b = run["best_safe_no_11plus"]
        key = (
            b["same_id_thr"],
            b["require_mutual"],
            b["hard_min"],
            b["hard_max"],
            b["hard_topk"],
        )
        votes.setdefault(key, []).append(b["score"])
    ranked = sorted(
        votes.items(),
        key=lambda kv: (len(kv[1]), float(np.mean(kv[1]))),
        reverse=True,
    )
    best_key, scores = ranked[0]
    return {
        "same_id_thr": best_key[0],
        "require_mutual": best_key[1],
        "hard_min": best_key[2],
        "hard_max": best_key[3],
        "hard_topk": best_key[4],
        "votes": len(scores),
        "mean_score": float(np.mean(scores)),
        "runner_ups": [
            {
                "same_id_thr": k[0],
                "require_mutual": k[1],
                "hard_min": k[2],
                "hard_max": k[3],
                "hard_topk": k[4],
                "votes": len(v),
                "mean_score": float(np.mean(v)),
            }
            for k, v in ranked[1:5]
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)

    per_run = [search_one_run(p) for p in args.run_dirs]
    rec = aggregate_recommendation(per_run)
    payload = {"runs": per_run, "aggregate_recommendation": rec}

    out_json = args.out_json or Path("outputs/crib/param_search_summary.json")
    out_md = args.out_md or Path("outputs/crib/param_search_summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# CRIB parameter search summary", ""]
    for run in per_run:
        b = run["best_safe_no_11plus"]
        lines.append(f"## {run['run_dir']} (n={run['n_samples']})")
        lines.append(
            f"- best_safe: same_id={b['same_id_thr']}, mutual={b['require_mutual']}, "
            f"hard=[{b['hard_min']},{b['hard_max']}), topk={b['hard_topk']}"
        )
        lines.append(
            f"  score={b['score']:.3f}, cov={b['hard_coverage']:.3f}, "
            f"mean={b['hard_mean_count']:.2f}, multi={b['n_multi']}, "
            f"11+={b['n_11plus']}, 6-10={b['n_6_10']}"
        )
        lines.append("- top5:")
        for c in run["top15"][:5]:
            lines.append(
                f"  - thr={c['same_id_thr']}/mutual={c['require_mutual']}/"
                f"min={c['hard_min']}: score={c['score']:.3f}, "
                f"cov={c['hard_coverage']:.3f}, 11+={c['n_11plus']}, multi={c['n_multi']}"
            )
        lines.append("")
    lines.append("## Aggregate recommendation")
    lines.append(f"- **same_id_thr = {rec['same_id_thr']}**")
    lines.append(f"- **require_mutual = {rec['require_mutual']}**")
    lines.append(f"- **hard_neg = [{rec['hard_min']}, {rec['hard_max']})**")
    lines.append(f"- **hard_neg_topk = {rec['hard_topk']}**")
    lines.append(f"- votes={rec['votes']}, mean_score={rec['mean_score']:.3f}")
    lines.append("")
    lines.append("### 8-GPU full-run command sketch")
    lines.append("```bash")
    lines.append("for i in 0 1 2 3 4 5 6 7; do")
    lines.append("  CUDA_VISIBLE_DEVICES=$i python -m script.crib.build_crib_sidecar \\")
    lines.append("    --dataset-root $DATA_ROOT \\")
    lines.append("    --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \\")
    lines.append("    --output-dir outputs/crib/full/shards/shard_$i \\")
    lines.append("    --shard-index $i --num-shards 8 --embeddings-only \\")
    lines.append("    --device cuda --batch-size 64 &")
    lines.append("done")
    lines.append("wait")
    lines.append("python -m script.crib.merge_embedding_shards \\")
    lines.append("  --shard-dirs outputs/crib/full/shards/shard_{0..7} \\")
    lines.append("  --output-dir outputs/crib/full \\")
    lines.append("  --dataset-root $DATA_ROOT \\")
    lines.append("  --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \\")
    lines.append(
        f"  --same-id-threshold {rec['same_id_thr']} "
        f"--hard-neg-min-sim {rec['hard_min']} "
        f"--hard-neg-max-sim {rec['hard_max']} "
        f"{'--require-mutual-same-id' if rec['require_mutual'] else '--no-require-mutual-same-id'}"
    )
    lines.append("```")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

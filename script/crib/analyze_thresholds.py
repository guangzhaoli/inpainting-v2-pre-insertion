#!/usr/bin/env python3
"""Offline threshold sensitivity analysis on a finished CRIB run.

Reads embeddings.npy + sidecar.jsonl from an output dir and sweeps:
  - same_id_sim_threshold
  - hard_neg_min_sim / hard_neg_max_sim
  - hard_neg_topk

Does not re-run DINOv2.

Example:
  python -m script.crib.analyze_thresholds \\
    --run-dir outputs/crib/accessory_jewelry_validate1024
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from script.crib.config import CribBuildConfig  # noqa: E402
from script.crib.neighbors import knn_search  # noqa: E402
from script.crib.pipeline import run_pipeline_from_arrays  # noqa: E402


def load_run(run_dir: Path):
    run_dir = Path(run_dir)
    emb = np.load(run_dir / "embeddings.npy")
    uids = json.loads((run_dir / "sample_uids.json").read_text())
    cats: List[str] = []
    with (run_dir / "sidecar.jsonl").open() as f:
        for line in f:
            cats.append(json.loads(line)["category"])
    if len(uids) != emb.shape[0] or len(cats) != emb.shape[0]:
        raise ValueError("length mismatch among embeddings / uids / sidecar")
    return emb, uids, cats


def top1_profile(emb: np.ndarray) -> Dict[str, Any]:
    _, sc = knn_search(emb, topk=min(20, max(emb.shape[0] - 1, 1)), use_faiss=True)
    top1 = sc[:, 0]
    top1 = top1[np.isfinite(top1)]
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "n": int(top1.size),
        "mean": float(np.mean(top1)),
        "std": float(np.std(top1)),
        "quantiles": {str(q): float(np.quantile(top1, q)) for q in qs},
        "frac_ge": {
            str(t): float(np.mean(top1 >= t))
            for t in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        },
    }


def sweep_same_id(emb, uids, cats) -> List[Dict[str, Any]]:
    rows = []
    for thr in [0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
        cfg = CribBuildConfig(
            same_id_sim_threshold=thr,
            hard_neg_max_sim=thr,
            hard_neg_min_sim=0.70,
            hard_neg_topk=8,
            neighbor_topk=50,
            use_faiss=True,
            hard_neg_same_category_only=False,
        )
        r = run_pipeline_from_arrays(emb, uids, cats, cfg)
        s = r.stats["same_id"]
        rows.append(
            {
                "same_id_thr": thr,
                "n_clusters": s["n_clusters"],
                "n_multi": s["n_multi"],
                "n_singleton": s["n_singleton"],
                "edge_count": s["edge_count"],
                "singleton_ratio": s["singleton_ratio"],
                "size_hist": s["size_hist"],
                "largest_cluster": max(s["size_hist"].keys(), key=lambda k: (
                    0 if k == "1" else (2 if k == "2" else (4 if k == "3-5" else (8 if k == "6-10" else 20)))
                )),
            }
        )
    return rows


def sweep_hard_neg(emb, uids, cats, same_id_thr: float = 0.95) -> List[Dict[str, Any]]:
    rows = []
    for mn in [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85]:
        for mx in [0.92, 0.94, 0.95, 0.96]:
            if mx <= mn or mx < same_id_thr - 1e-9 and False:
                pass
            if mx < mn:
                continue
            # hard_neg_max must be <= same_id_thr by config validation
            if mx > same_id_thr:
                continue
            cfg = CribBuildConfig(
                same_id_sim_threshold=same_id_thr,
                hard_neg_min_sim=mn,
                hard_neg_max_sim=mx,
                hard_neg_topk=8,
                neighbor_topk=50,
                use_faiss=True,
                hard_neg_same_category_only=False,
            )
            r = run_pipeline_from_arrays(emb, uids, cats, cfg)
            h = r.stats["hard_neg"]
            rows.append(
                {
                    "min_sim": mn,
                    "max_sim": mx,
                    "coverage": h["coverage"],
                    "full_coverage": h["full_coverage"],
                    "mean_count": h["mean_count"],
                    "n_with_any": h["n_with_any"],
                    "n_full": h["n_full"],
                }
            )
    return rows


def sweep_topk(emb, uids, cats, same_id_thr=0.95, min_sim=0.70, max_sim=0.95) -> List[Dict[str, Any]]:
    rows = []
    for k in [4, 8, 12, 16]:
        cfg = CribBuildConfig(
            same_id_sim_threshold=same_id_thr,
            hard_neg_min_sim=min_sim,
            hard_neg_max_sim=max_sim,
            hard_neg_topk=k,
            neighbor_topk=max(50, k + 10),
            use_faiss=True,
            hard_neg_same_category_only=False,
        )
        r = run_pipeline_from_arrays(emb, uids, cats, cfg)
        h = r.stats["hard_neg"]
        rows.append(
            {
                "hard_neg_topk": k,
                "coverage": h["coverage"],
                "full_coverage": h["full_coverage"],
                "mean_count": h["mean_count"],
            }
        )
    return rows


def recommend(top1: Dict[str, Any], same_rows: List[Dict], hard_rows: List[Dict]) -> Dict[str, Any]:
    """Heuristic recommendations based on sweep tables."""
    # Prefer same_id thr where multi exists but size_hist has no 11+
    safe = []
    for row in same_rows:
        hist = row["size_hist"]
        if hist.get("11+", 0) == 0 and hist.get("6-10", 0) <= 2:
            safe.append(row)
    # among safe, pick thr closest to 0.95 with some multi clusters
    if safe:
        # prefer higher thr if multi still > 0
        with_multi = [r for r in safe if r["n_multi"] > 0]
        pool = with_multi or safe
        # choose thr that maximizes multi under thr>=0.94, else best multi
        preferred = [r for r in pool if r["same_id_thr"] >= 0.94]
        if preferred:
            # among preferred, take the one with most multi but thr not below 0.94
            # actually conservative: take 0.95 if present else max thr with multi
            thr_pref = sorted(preferred, key=lambda r: (abs(r["same_id_thr"] - 0.95), -r["n_multi"]))
            same_rec = thr_pref[0]["same_id_thr"]
        else:
            same_rec = sorted(pool, key=lambda r: -r["same_id_thr"])[0]["same_id_thr"]
    else:
        same_rec = 0.97

    # hard-neg: want coverage >= 0.75 and mean_count >= 3, prefer higher min_sim
    hard_at_thr = [r for r in hard_rows if abs(r["max_sim"] - same_rec) < 1e-9 or r["max_sim"] == min(same_rec, 0.95)]
    if not hard_at_thr:
        hard_at_thr = [r for r in hard_rows if r["max_sim"] <= same_rec]
    good = [r for r in hard_at_thr if r["coverage"] >= 0.75 and r["mean_count"] >= 3.0]
    if good:
        # highest min_sim among good
        hard_rec = sorted(good, key=lambda r: (-r["min_sim"], -r["coverage"]))[0]
    else:
        # best coverage
        hard_rec = sorted(hard_at_thr, key=lambda r: (-r["coverage"], -r["min_sim"]))[0]

    notes = []
    frac95 = top1["frac_ge"].get("0.95", 0.0)
    frac70 = top1["frac_ge"].get("0.7", top1["frac_ge"].get("0.70", 0.0))
    if frac95 < 0.02:
        notes.append("Very few near-dups at 0.95; same-id clusters will be sparse (OK for conservative MVP).")
    if frac95 > 0.15:
        notes.append("Many top1>=0.95 pairs; inspect same_id review carefully for over-merge.")
    if hard_rec["coverage"] < 0.7:
        notes.append("Hard-neg coverage <70% on this pool; full dataset or lower min_sim will help.")
    if hard_rec["coverage"] >= 0.85 and hard_rec["min_sim"] >= 0.70:
        notes.append("Hard-neg coverage healthy at min_sim>=0.70; defaults look solid.")

    return {
        "recommended_same_id_thr": same_rec,
        "recommended_hard_neg_min_sim": hard_rec["min_sim"],
        "recommended_hard_neg_max_sim": hard_rec["max_sim"],
        "expected_hard_neg_coverage": hard_rec["coverage"],
        "expected_hard_neg_mean_count": hard_rec["mean_count"],
        "notes": notes,
        "default_config_ok": (
            abs(same_rec - 0.95) <= 0.01
            and abs(hard_rec["min_sim"] - 0.70) <= 0.05
            and hard_rec["coverage"] >= 0.65
        ),
    }


def format_report(run_dir: Path, payload: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# CRIB threshold analysis: {run_dir}")
    lines.append("")
    lines.append(f"n_samples = {payload['n_samples']}, dim = {payload['embedding_dim']}")
    lines.append("")
    lines.append("## Top-1 neighbor similarity")
    t = payload["top1"]
    lines.append(f"- mean={t['mean']:.4f}  std={t['std']:.4f}")
    lines.append(f"- quantiles={ {k: round(v,4) for k,v in t['quantiles'].items()} }")
    lines.append(f"- frac_ge={ {k: round(v,4) for k,v in t['frac_ge'].items()} }")
    lines.append("")
    lines.append("## same_id threshold sweep")
    lines.append("thr | multi | edges | singleton_ratio | size_hist")
    lines.append("---|---:|---:|---:|---")
    for r in payload["same_id_sweep"]:
        lines.append(
            f"{r['same_id_thr']:.2f} | {r['n_multi']} | {r['edge_count']} | "
            f"{r['singleton_ratio']:.3f} | {r['size_hist']}"
        )
    lines.append("")
    lines.append("## hard_neg sweep (selected max_sim <= same_id thr)")
    lines.append("min | max | coverage | full_cov | mean_count")
    lines.append("---|---|---:|---:|---:")
    # compact: only max_sim in {0.95, recommended}
    for r in payload["hard_neg_sweep"]:
        if r["max_sim"] in {0.94, 0.95, 0.96, 0.92}:
            lines.append(
                f"{r['min_sim']:.2f} | {r['max_sim']:.2f} | {r['coverage']:.3f} | "
                f"{r['full_coverage']:.3f} | {r['mean_count']:.2f}"
            )
    lines.append("")
    lines.append("## hard_neg topk sweep")
    for r in payload["topk_sweep"]:
        lines.append(
            f"- topk={r['hard_neg_topk']}: cov={r['coverage']:.3f}, "
            f"full={r['full_coverage']:.3f}, mean={r['mean_count']:.2f}"
        )
    lines.append("")
    rec = payload["recommendation"]
    lines.append("## Recommendation")
    lines.append(f"- same_id_thr = **{rec['recommended_same_id_thr']}**")
    lines.append(
        f"- hard_neg band = **[{rec['recommended_hard_neg_min_sim']}, "
        f"{rec['recommended_hard_neg_max_sim']})**"
    )
    lines.append(
        f"- expected coverage ≈ {rec['expected_hard_neg_coverage']:.3f}, "
        f"mean_count ≈ {rec['expected_hard_neg_mean_count']:.2f}"
    )
    lines.append(f"- default_config_ok = **{rec['default_config_ok']}**")
    for n in rec["notes"]:
        lines.append(f"- note: {n}")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)

    emb, uids, cats = load_run(args.run_dir)
    top1 = top1_profile(emb)
    same_rows = sweep_same_id(emb, uids, cats)
    # use default 0.95 and also recommended later; sweep hard with max<=0.95 primarily
    hard_rows = sweep_hard_neg(emb, uids, cats, same_id_thr=0.95)
    # also attach sweeps for 0.94/0.96 for completeness in json
    hard_rows += sweep_hard_neg(emb, uids, cats, same_id_thr=0.94)
    hard_rows += sweep_hard_neg(emb, uids, cats, same_id_thr=0.96)
    # dedupe
    seen = set()
    hard_dedup = []
    for r in hard_rows:
        key = (r["min_sim"], r["max_sim"])
        if key in seen:
            continue
        seen.add(key)
        hard_dedup.append(r)
    hard_rows = hard_dedup

    topk_rows = sweep_topk(emb, uids, cats)
    rec = recommend(top1, same_rows, [r for r in hard_rows if r["max_sim"] <= 0.95])

    payload = {
        "run_dir": str(args.run_dir),
        "n_samples": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "top1": top1,
        "same_id_sweep": same_rows,
        "hard_neg_sweep": hard_rows,
        "topk_sweep": topk_rows,
        "recommendation": rec,
        "current_defaults": {
            "same_id_sim_threshold": 0.95,
            "hard_neg_min_sim": 0.70,
            "hard_neg_max_sim": 0.95,
            "hard_neg_topk": 8,
        },
    }

    out_json = args.out_json or (args.run_dir / "threshold_analysis.json")
    out_md = args.out_md or (args.run_dir / "threshold_analysis.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = format_report(args.run_dir, payload)
    out_md.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

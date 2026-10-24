#!/usr/bin/env python3
"""Summary, HTML page and Orbit record for scripts/probe_lighting_leak.py.

Per checkpoint and perturbation, with dOut = output mean log-RGB (perturbed - clean run, GT object
mask eroded 3 px) and g = effective log gain on the reference object (perturbed - clean reference):
  leak_total  = sum <dOut, g> / sum |g|^2
  leak_chroma = the same on the channel-mean-removed parts (white balance)
  leak_lum    = sum mean(dOut) mean(g) / sum mean(g)^2 (exposure)
pooled over cases; 1 = the output copies the reference change, 0 = it ignores the reference.
95% intervals: percentile bootstrap over cases. The clean_seed2 run gives the same projections
for a pure seed change (noise floor).

  python scripts/probe_lighting_leak_report.py --run DIR [--no-orbit]
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_lighting_leak as probe  # noqa: E402

ORBIT_SDK = Path("/inspire/hdd/global_user/250041225002/orbit-release/sdk")
PERTURBED = ("warm", "cool", "ev_plus", "ev_minus", "warm_dim")
CHROMA = ("warm", "cool", "warm_dim")
LUM = ("ev_plus", "ev_minus", "warm_dim")
BOOT = 4000
STEP = {"s250": 250, "s750": 750, "s1500": 1500}
LABEL = {
    "animal_new_s250": "宠物 新 run · 250",
    "animal_new_s750": "宠物 新 run · 750",
    "animal_new_s1500": "宠物 新 run · 1500",
    "animal_old_s250": "宠物 旧 run · 250",
    "animal_old_s750": "宠物 旧 run · 750",
    "animal_old_s1500": "宠物 旧 run · 1500",
    "human_s1500": "人像 · 1500",
    "general_s1500": "通用物体 · 1500",
    "base": "底座 2511（无 LoRA）",
}
VARIANT_LABEL = {
    "clean": "干净",
    "clean_seed2": "换 seed",
    "warm": "偏暖 +100 mired",
    "cool": "偏冷 −100 mired",
    "ev_plus": "+1 EV",
    "ev_minus": "−1 EV",
    "warm_dim": "偏暖 + −1 EV",
}
SHOWCASE = {"animal": "animal_new_s1500", "human": "human_s1500", "general_object": "general_s1500"}


def center(v: np.ndarray) -> np.ndarray:
    return v - v.mean(-1, keepdims=True)


def load_runs(run: Path) -> dict:
    runs = {}
    for path in (run / "runs").glob("*/*/*.json"):
        meta = json.loads(path.read_text())
        runs[(meta["checkpoint"], meta["variant"], meta["case"]["prefix"])] = meta
    return runs


def ratio_ci(num: np.ndarray, den: np.ndarray, rng: np.random.Generator) -> dict:
    """Pooled ratio sum(num) / sum(den) with a percentile bootstrap over cases."""
    n = len(num)
    if n == 0 or den.sum() <= 0:
        return {"est": None, "lo": None, "hi": None, "n": n}
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"est": float(num.sum() / den.sum()), "lo": float(lo), "hi": float(hi), "n": n}


def mean_ci(values: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(values)
    if n == 0:
        return {"est": None, "lo": None, "hi": None, "n": 0}
    idx = rng.integers(0, n, size=(BOOT, n))
    lo, hi = np.percentile(values[idx].mean(1), [2.5, 97.5])
    return {"est": float(values.mean()), "lo": float(lo), "hi": float(hi), "median": float(np.median(values)), "n": n}


def projections(runs: dict, key: str, prefixes: list[str], variant: str, delta_from: str = None) -> dict:
    """Per-case numerators / denominators of the three leak coefficients.

    ``delta_from='clean_seed2'`` projects the seed-change difference onto the variant's gain instead.
    """
    rows = []
    for prefix in prefixes:
        clean = runs.get((key, "clean", prefix))
        pert = runs.get((key, variant, prefix))
        other = pert if delta_from is None else runs.get((key, delta_from, prefix))
        if clean is None or pert is None or other is None:
            continue
        d = np.subtract(other["output"]["mean_log_rgb"], clean["output"]["mean_log_rgb"])
        g = np.subtract(pert["reference"]["mean_log_rgb"], pert["reference_clean"]["mean_log_rgb"])
        rows.append((prefix, d, g))
    if not rows:
        return {"prefixes": [], "d": np.zeros((0, 3)), "g": np.zeros((0, 3))}
    return {"prefixes": [r[0] for r in rows], "d": np.stack([r[1] for r in rows]), "g": np.stack([r[2] for r in rows])}


def leak_stats(p: dict, rng) -> dict:
    d, g = p["d"], p["g"]
    out = {
        "n": len(d),
        "g_eff_mean": g.mean(0).tolist() if len(g) else None,
        "g_eff_lum_mean": float(g.mean(1).mean()) if len(g) else None,
        "g_eff_chroma_norm_mean": float(np.linalg.norm(center(g), axis=1).mean()) if len(g) else None,
        "leak_total": ratio_ci((d * g).sum(1), (g * g).sum(1), rng),
        "leak_chroma": ratio_ci((center(d) * center(g)).sum(1), (center(g) ** 2).sum(1), rng),
        "leak_lum": ratio_ci(d.mean(1) * g.mean(1), g.mean(1) ** 2, rng),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        per_c = (center(d) * center(g)).sum(1) / (center(g) ** 2).sum(1)
        per_l = d.mean(1) / g.mean(1)
    out["per_case_chroma_median"] = float(np.nanmedian(per_c)) if len(d) else None
    out["per_case_lum_median"] = float(np.nanmedian(per_l)) if len(d) else None
    return out


def summarize(plan: dict, runs: dict) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for key, ck in plan["checkpoints"].items():
        prefixes = [c["prefix"] for c in ck["cases"]]
        categories = {c["prefix"]: c["category"] for c in ck["cases"]}
        domains = {c["prefix"]: c["domain"] for c in ck["cases"]}
        entry = {
            "label": LABEL.get(key, key),
            "domain": ck["domain"],
            "path": ck["path"],
            "train_step": STEP.get(key.rsplit("_", 1)[-1]),
            "n_cases": len(prefixes),
            "runs_done": sum(1 for p in prefixes for v in plan["variants"] if (key, v, p) in runs),
            "variants": {},
            "seed_noise": {},
            "by_category": {},
            "by_domain": {},
        }
        for variant in PERTURBED:
            p = projections(runs, key, prefixes, variant)
            entry["variants"][variant] = leak_stats(p, rng)
            noise = projections(runs, key, prefixes, variant, delta_from="clean_seed2")
            s = leak_stats(noise, rng)
            entry["seed_noise"][variant] = {k: s[k] for k in ("n", "leak_total", "leak_chroma", "leak_lum")}
        groups = [("by_category", categories)] + ([("by_domain", domains)] if key == "base" else [])
        for field, labels in groups:
            for name in sorted(set(labels.values())):
                members = [q for q in prefixes if labels[q] == name]
                entry[field][name] = {
                    v: {
                        k: leak_stats(projections(runs, key, members, v), rng)[k]
                        for k in ("n", "leak_chroma", "leak_lum", "leak_total")
                    }
                    for v in PERTURBED
                }
        # Colour fidelity of the clean run against the GT object.
        clean = [runs[(key, "clean", q)] for q in prefixes if (key, "clean", q) in runs]
        seed2 = [
            (runs[(key, "clean", q)], runs[(key, "clean_seed2", q)])
            for q in prefixes
            if (key, "clean", q) in runs and (key, "clean_seed2", q) in runs
        ]
        entry["de00"] = {
            "mean_color_vs_gt": mean_ci(np.array([m["de00_mean_color_vs_gt"] for m in clean]), rng),
            "pixel_mean_vs_gt": mean_ci(np.array([m["de00_pixel_mean_vs_gt"] for m in clean]), rng),
            "reference_vs_gt_mean_color": mean_ci(
                np.array([float(probe.ciede2000(np.array(m["reference_clean"]["mean_lab"]), np.array(m["gt"]["mean_lab"])))
                          for m in clean]), rng),  # fmt: skip
            "seed_change_mean_color": mean_ci(
                np.array([float(probe.ciede2000(np.array(a["output"]["mean_lab"]), np.array(b["output"]["mean_lab"])))
                          for a, b in seed2]), rng),  # fmt: skip
        }
        if key == "base":
            for name in sorted(set(domains.values())):
                members = [m for m in clean if m["case"]["domain"] == name]
                entry["by_domain"].setdefault(name, {})["de00_mean_color_vs_gt"] = mean_ci(
                    np.array([m["de00_mean_color_vs_gt"] for m in members]), rng
                )
        seconds = [m["sample_seconds"] for v in plan["variants"] for q in prefixes if (m := runs.get((key, v, q)))]
        entry["sample_seconds_median"] = float(np.median(seconds)) if seconds else None
        out[key] = entry
    return out


def fmt(ci: dict, digits: int = 2) -> str:
    if not ci or ci.get("est") is None:
        return "–"
    return f"{ci['est']:.{digits}f} [{ci['lo']:.{digits}f}, {ci['hi']:.{digits}f}]"


def headline(summary: dict) -> str:
    lines = ["泄漏系数（1=照抄参考，0=按场景），色度取偏暖/偏冷均值，亮度取 ±1EV 均值："]
    for key, e in summary.items():
        v = e["variants"]
        chroma = np.mean([v[x]["leak_chroma"]["est"] for x in ("warm", "cool") if v[x]["leak_chroma"]["est"] is not None] or [np.nan])
        lum = np.mean([v[x]["leak_lum"]["est"] for x in ("ev_plus", "ev_minus") if v[x]["leak_lum"]["est"] is not None] or [np.nan])
        de = e["de00"]["mean_color_vs_gt"]["est"]
        lines.append(f"{e['label']}: 色度 {chroma:.2f}，亮度 {lum:.2f}，干净 ΔE00 {de if de is None else round(de, 2)}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- images


def gt_mask_local(meta: dict, case: dict) -> np.ndarray:
    mask = probe.read_mask(case["tar_mask"])
    box = (meta.get("crop_plan") or {}).get("crop_box")  # the worker crops with it even when not applied
    if box:
        mask = mask.crop(tuple(box))
    return probe.binary_mask(mask, tuple(meta["sizes"]["local"]))


def bbox(mask: np.ndarray, margin: float, shape) -> tuple:
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    return max(0, x0 - mx), max(0, y0 - my), min(shape[1], x1 + mx), min(shape[0], y1 + my)


def thumb(image: Image.Image, box, path: Path, height: int = 220):
    crop = image.crop(box)
    width = max(1, round(crop.width * height / crop.height))
    crop.resize((width, height), Image.Resampling.LANCZOS).save(path, quality=88)


def reference_thumbs(case: dict, img_dir: Path) -> dict:
    from src.utils.images import extract_foreground_from_image_and_mask

    ref, mask = probe.read_rgb(case["ref_image"]), probe.read_mask(case["ref_mask"])
    names = {}
    for variant in ("clean",) + PERTURBED:
        name = f"{case['prefix']}_ref_{variant}.jpg"
        if not (img_dir / name).exists():
            fg, fg_mask = extract_foreground_from_image_and_mask(probe.perturb_reference(ref, mask, variant), mask)
            image = Image.fromarray(fg)
            thumb(image, (0, 0, image.width, image.height), img_dir / name)
        names[variant] = name
    return names


def output_thumbs(run: Path, key: str, case: dict, runs: dict, img_dir: Path) -> dict:
    clean = runs.get((key, "clean", case["prefix"]))
    if clean is None:
        return {}
    mask = gt_mask_local(clean, case)
    w, h = clean["sizes"]["local"]
    box = bbox(mask, 0.25, (h, w))
    names = {}
    for variant in ("clean",) + PERTURBED:
        src = run / "runs" / key / variant / f"{case['prefix']}.jpg"
        if not src.exists():
            continue
        name = f"{case['prefix']}_{key}_{variant}.jpg"
        if not (img_dir / name).exists():
            thumb(Image.open(src).convert("RGB"), box, img_dir / name)
        names[variant] = name
    gt = sorted((run / "cases").glob(f"{case['prefix']}_gt_local_*.jpg"))
    if gt:
        name = f"{case['prefix']}_gt.jpg"
        if not (img_dir / name).exists():
            thumb(Image.open(gt[0]).convert("RGB"), box, img_dir / name)
        names["gt"] = name
    return names


def pick_showcase(plan: dict, runs: dict, key: str, k: int = 3) -> list[dict]:
    """Cases with the highest, median and lowest per-case warm chroma leak."""
    cases = plan["checkpoints"][key]["cases"]
    p = projections(runs, key, [c["prefix"] for c in cases], "warm")
    if not p["prefixes"]:
        return []
    per = (center(p["d"]) * center(p["g"])).sum(1) / (center(p["g"]) ** 2).sum(1)
    order = np.argsort(per)
    picks = [order[-1], order[len(order) // 2], order[0]][:k]
    by_prefix = {c["prefix"]: c for c in cases}
    return [{**by_prefix[p["prefixes"][i]], "warm_leak": float(per[i])} for i in picks]


# --------------------------------------------------------------------------- HTML

CSS = """
:root{color-scheme:light;--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ref:#b9b8b2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#1a1a19;--panel:#222220;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#8f8e86;--rule:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ref:#5c5b56}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#1a1a19;--panel:#222220;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;
--rule:#3a3a37;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ref:#5c5b56}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,"PingFang SC","Noto Sans CJK SC",
"Microsoft YaHei",sans-serif}main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 6px}h2{font-size:18px;margin:36px 0 10px;border-bottom:1px solid var(--rule);padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px}p,li{color:var(--ink2)}.lead{color:var(--ink2);max-width:860px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}.scroll{overflow-x:auto}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.kpi{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px 14px}.kpi b{display:block;font-size:22px;color:var(--ink)}
.kpi span{color:var(--ink2);font-size:12px}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.chart{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px}.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--ink2);margin:4px 0 8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
.case{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:12px;margin:14px 0}
.grid{display:grid;grid-template-columns:110px repeat(6,minmax(96px,1fr));gap:6px;align-items:center;overflow-x:auto}
.grid img{width:100%;height:auto;border-radius:4px;display:block;background:#fff}.grid .h{font-size:12px;color:var(--ink2);text-align:center}
.grid .r{font-size:12px;color:var(--ink2)}.note{font-size:12px;color:var(--muted)}code{font-size:12px}
@media (max-width:640px){.grid{grid-template-columns:70px repeat(6,96px)}}
"""


def forest_svg(summary: dict, keys: list[str], component: str, variants: tuple, title: str) -> str:
    """Dot + 95% CI per checkpoint (rows) and variant (colour), axis -0.2 .. 1.2."""
    lo_x, hi_x = -0.2, 1.2
    row_h, top, left, right = 26, 28, 150, 16
    width, height = 520, top + row_h * len(keys) + 30
    plot_w = width - left - right

    def x(v):
        return left + (min(max(v, lo_x), hi_x) - lo_x) / (hi_x - lo_x) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        tx = x(tick)
        stroke = "var(--ref)" if tick in (0, 1.0) else "var(--rule)"
        parts.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{top - 6}" y2="{height - 26}" stroke="{stroke}" stroke-width="1"/>')
        parts.append(f'<text x="{tx:.1f}" y="{height - 10}" font-size="11" text-anchor="middle" fill="var(--ink2)">{tick:g}</text>')
    parts.append(f'<text x="{x(0):.1f}" y="{top - 12}" font-size="10" text-anchor="middle" fill="var(--muted)">按场景</text>')
    parts.append(f'<text x="{x(1):.1f}" y="{top - 12}" font-size="10" text-anchor="middle" fill="var(--muted)">照抄参考</text>')
    colours = ("var(--s1)", "var(--s2)", "var(--s3)")
    for r, key in enumerate(keys):
        cy = top + r * row_h + row_h / 2
        parts.append(f'<text x="{left - 8}" y="{cy + 4:.1f}" font-size="12" text-anchor="end" fill="var(--ink)">{html.escape(summary[key]["label"])}</text>')
        for j, variant in enumerate(variants):
            ci = summary[key]["variants"][variant][component]
            if ci["est"] is None:
                continue
            yy = cy + (j - (len(variants) - 1) / 2) * 7
            tip = f"{summary[key]['label']} · {VARIANT_LABEL[variant]}: {fmt(ci)} (n={ci['n']})"
            parts.append(f'<g><title>{html.escape(tip)}</title>'
                         f'<rect x="{x(ci["lo"]) - 2:.1f}" y="{yy - 5:.1f}" width="{x(ci["hi"]) - x(ci["lo"]) + 4:.1f}" height="10" fill="transparent"/>'
                         f'<line x1="{x(ci["lo"]):.1f}" x2="{x(ci["hi"]):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{colours[j]}" stroke-width="2" stroke-linecap="round"/>'
                         f'<circle cx="{x(ci["est"]):.1f}" cy="{yy:.1f}" r="4" fill="{colours[j]}" stroke="var(--panel)" stroke-width="2"/></g>')
    parts.append("</svg>")
    legend = "".join(f'<span><i style="background:{c}"></i>{VARIANT_LABEL[v]}</span>' for v, c in zip(variants, colours))
    return f'<div class="chart"><h3>{html.escape(title)}</h3><div class="legend">{legend}</div>{"".join(parts)}</div>'


def build_html(run: Path, plan: dict, runs: dict, summary: dict) -> str:
    report = run / "report"
    img_dir = report / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    keys = [k for k in plan["checkpoints"] if summary[k]["runs_done"]]
    animal = [k for k in keys if k.startswith("animal")]

    def row(key):
        e = summary[key]
        v = e["variants"]
        cells = [html.escape(e["label"]), str(e["n_cases"])]
        cells += [fmt(v[x]["leak_chroma"]) for x in CHROMA] + [fmt(v[x]["leak_lum"]) for x in LUM]
        cells += [fmt(e["de00"]["mean_color_vs_gt"], 1), fmt(e["de00"]["pixel_mean_vs_gt"], 1)]
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    head = ("<tr><th>Checkpoint</th><th>n</th>" + "".join(f"<th>色度 · {VARIANT_LABEL[x]}</th>" for x in CHROMA)
            + "".join(f"<th>亮度 · {VARIANT_LABEL[x]}</th>" for x in LUM)
            + "<th>ΔE00 均色 vs GT</th><th>ΔE00 逐像素 vs GT</th></tr>")  # fmt: skip
    noise_rows = "".join(
        "<tr><td>" + html.escape(summary[k]["label"]) + "</td>"
        + "".join(f"<td>{fmt(summary[k]['seed_noise'][x]['leak_chroma'])}</td>" for x in ("warm", "cool"))
        + "".join(f"<td>{fmt(summary[k]['seed_noise'][x]['leak_lum'])}</td>" for x in ("ev_plus", "ev_minus"))
        + f"<td>{fmt(summary[k]['de00']['seed_change_mean_color'], 2)}</td>"
        + f"<td>{fmt(summary[k]['de00']['reference_vs_gt_mean_color'], 2)}</td></tr>"
        for k in keys
    )
    gain_rows = "".join(
        f"<tr><td>{VARIANT_LABEL[v]}</td><td>{', '.join(f'{x:.3f}' for x in plan['gains'][v])}</td>"
        + "".join(f"<td>{summary[k]['variants'][v]['g_eff_chroma_norm_mean']:.3f} / {summary[k]['variants'][v]['g_eff_lum_mean']:+.3f}</td>"
                  for k in keys[:1] if summary[k]['variants'][v]['n'])
        + "</tr>"
        for v in PERTURBED
    )
    cat_rows = []
    for k in keys:
        for name, groups in list(summary[k]["by_category"].items()) + [(f"[{d}]", g) for d, g in summary[k]["by_domain"].items()]:
            if not isinstance(groups.get("warm"), dict) or not groups["warm"]["n"]:
                continue
            cat_rows.append(
                f"<tr><td>{html.escape(summary[k]['label'])}</td><td>{html.escape(name)}</td><td>{groups['warm']['n']}</td>"
                + "".join(f"<td>{fmt(groups[x]['leak_chroma'])}</td>" for x in ("warm", "cool"))
                + "".join(f"<td>{fmt(groups[x]['leak_lum'])}</td>" for x in ("ev_plus", "ev_minus"))
                + "</tr>"
            )

    showcase = []
    for domain, key in SHOWCASE.items():
        if key not in keys:
            continue
        showcase.append(f"<h3>{html.escape(summary[key]['label'])}（与底座同一组样本）</h3>")
        for case in pick_showcase(plan, runs, key):
            refs = reference_thumbs(case, img_dir)
            teacher = output_thumbs(run, key, case, runs, img_dir)
            base = output_thumbs(run, "base", case, runs, img_dir) if "base" in keys else {}
            columns = ("clean",) + PERTURBED
            cells = ['<div></div>'] + [f'<div class="h">{VARIANT_LABEL[v]}</div>' for v in columns]

            def line(label, names):
                out = [f'<div class="r">{label}</div>']
                for v in columns:
                    out.append(f'<img loading="lazy" src="img/{names[v]}" alt="{label} {VARIANT_LABEL[v]}">' if v in names else "<div></div>")
                return out

            cells += line("参考图", refs) + line("teacher 输出", teacher)
            if base:
                cells += line("底座输出", base)
            gt = f'<img loading="lazy" src="img/{teacher["gt"]}" alt="GT" style="max-width:150px">' if "gt" in teacher else ""
            showcase.append(
                f'<div class="case"><div class="note">{html.escape(case["prefix"])} · 偏暖色度泄漏 {case["warm_leak"]:.2f}'
                f' · seed {case["seed"]}</div><div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start">'
                f'<div style="flex:1;min-width:320px"><div class="grid">{"".join(cells)}</div></div>'
                f'<div><div class="h" style="font-size:12px;color:var(--ink2)">GT</div>{gt}</div></div></div>'
            )

    def kpi(key, label):
        if key not in summary:
            return ""
        v = summary[key]["variants"]
        c = np.mean([v[x]["leak_chroma"]["est"] for x in ("warm", "cool")])
        lum = np.mean([v[x]["leak_lum"]["est"] for x in ("ev_plus", "ev_minus")])
        return f'<div class="kpi"><span>{html.escape(label)}</span><b>{c:.2f} / {lum:.2f}</b><span>色度 / 亮度泄漏（暖冷、±1EV 均值）</span></div>'

    done = sum(e["runs_done"] for e in summary.values())
    total = sum(len(c["cases"]) for c in plan["checkpoints"].values()) * len(plan["variants"])
    charts = forest_svg(summary, keys, "leak_chroma", CHROMA, "色度泄漏（去均值的 log-RGB 投影）") + forest_svg(
        summary, keys, "leak_lum", LUM, "亮度泄漏（三通道均值）"
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>参考光照泄漏诊断</title><style>{CSS}</style></head><body><main>
<h1>InsertFuse teacher v2 · 参考光照泄漏</h1>
<p class="lead">只改参考图物体像素的光照（线性 RGB 增益，白底不变），看插入结果跟着参考变（泄漏≈1）还是按场景保持不变（≈0）。
条件 mask 固定为 GT + 32 px 交互带，各变体同 seed，40 步，true_cfg_scale 2.5。指标在 GT 物体 mask（腐蚀 3 px）内取输出的平均 log-RGB；
g 用参考物体实际得到的 log 增益（含截断）。区间为按样本 bootstrap 的 95% 置信区间。完成 {done}/{total} 次采样，{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}。</p>
<div class="kpis">{kpi('animal_new_s1500', '宠物新 run · 1500 步')}{kpi('human_s1500', '人像 · 1500 步')}{kpi('general_s1500', '通用物体 · 1500 步')}{kpi('base', '底座 2511（无 LoRA）')}</div>
<h2>泄漏系数</h2><div class="charts">{charts}</div>
<div class="scroll"><table>{head}{''.join(row(k) for k in keys)}</table></div>
<p class="note">色度泄漏 = Σ⟨PΔout, Pg⟩ / Σ|Pg|²，P 去掉三通道均值；亮度泄漏 = Σ mean(Δout)·mean(g) / Σ mean(g)²。偏暖/偏冷的 g 几何均值为 1，只有色度分量；±1EV 只有亮度分量；组合档两者都有。</p>
<h2>宠物：随训练步数</h2><div class="charts">{forest_svg(summary, animal, 'leak_chroma', ('warm', 'cool'), '宠物 色度泄漏 vs 步数') if animal else ''}{forest_svg(summary, animal, 'leak_lum', ('ev_plus', 'ev_minus'), '宠物 亮度泄漏 vs 步数') if animal else ''}</div>
<h2>噪声底与色彩还原基线</h2>
<p>同一 checkpoint 只换 seed（干净参考）时，输出差在同一 g 方向上的投影；真实泄漏需明显超出这个范围。ΔE00 均色 = 物体区域平均颜色之间的 CIEDE2000。</p>
<div class="scroll"><table><tr><th>Checkpoint</th><th>seed 噪声 · 偏暖</th><th>seed 噪声 · 偏冷</th><th>seed 噪声 · +1EV</th><th>seed 噪声 · −1EV</th><th>换 seed 均色 ΔE00</th><th>参考 vs GT 均色 ΔE00</th></tr>{noise_rows}</table></div>
<h3>施加的增益</h3><div class="scroll"><table><tr><th>扰动</th><th>名义线性增益 R,G,B</th><th>实际 |Pg| / mean(g)（样本均值，{html.escape(summary[keys[0]]['label']) if keys else ''}）</th></tr>{gain_rows}</table></div>
<p class="note">±100 mired 相当于约 21° 的光源色角，远大于评测集参考与场景之间 4–6° 的差，用来得到清楚的线性响应；+1EV 在亮物体上会截断，所以按实际 g 归一。</p>
<h2>按类别</h2><div class="scroll"><table><tr><th>Checkpoint</th><th>类别</th><th>n</th><th>色度 · 偏暖</th><th>色度 · 偏冷</th><th>亮度 · +1EV</th><th>亮度 · −1EV</th></tr>{''.join(cat_rows)}</table></div>
<h2>样例</h2><p>每类按偏暖色度泄漏取最高、中位、最低三例；物体区域裁剪，列为参考图的扰动档，行依次为参考图、teacher 输出、底座输出。</p>
{''.join(showcase)}
<h2>设置</h2><p class="note">底座：{html.escape(plan['base_model']['changes'])}。样本：每个 manifest 固定 seed 抽 32 对；宠物 checkpoint 和人像、通用物体 checkpoint 各用本域样本，底座用全部 96 对。
数据：<code>{html.escape(str(run))}</code>，汇总 <code>summary.json</code>，脚本 <code>scripts/probe_lighting_leak.py</code> / <code>probe_lighting_leak_report.py</code>。</p>
</main></body></html>"""


# --------------------------------------------------------------------------- Orbit


def log_orbit(run: Path, plan: dict, runs: dict, summary: dict):
    sys.path.insert(0, str(ORBIT_SDK))
    import orbit

    img_dir = run / "report" / "img"
    for key, e in summary.items():
        if not e["runs_done"]:
            continue
        orbit.init(
            workspace="qwen-inpainting",
            project="lighting-leak",
            name=f"leak-{key}",
            id=f"lighting-leak-20261024-{key}",
            group="lighting-leak-20261024",
            tags=["agent", "inference-only", e["domain"]],
            description="参考图光照扰动的泄漏系数（1=照抄参考，0=按场景），作为参考光照增强前的基线",
            config={
                "checkpoint": key,
                "checkpoint_path": e["path"],
                "train_step": e["train_step"],
                "domain": e["domain"],
                "n_cases": e["n_cases"],
                "num_inference_steps": plan["num_inference_steps"],
                "true_cfg_scale": plan["true_cfg_scale"],
                "gains": plan["gains"],
                "out_dir": str(run),
            },
            console=False,
            system=False,
        )
        if e["path"]:
            orbit.use_artifact(e["path"])
        metrics = {}
        for v, s in e["variants"].items():
            for comp in ("leak_chroma", "leak_lum", "leak_total"):
                if s[comp]["est"] is not None:
                    metrics[f"test/{comp}/{v}"] = s[comp]["est"]
                    metrics[f"test/{comp}/{v}_lo"] = s[comp]["lo"]
                    metrics[f"test/{comp}/{v}_hi"] = s[comp]["hi"]
        metrics["test/de00_mean_color_vs_gt"] = e["de00"]["mean_color_vs_gt"]["est"]
        metrics["test/de00_pixel_mean_vs_gt"] = e["de00"]["pixel_mean_vs_gt"]["est"]
        orbit.log(metrics, step=0)
        rows = []
        for case in plan["checkpoints"][key]["cases"]:
            prefix = case["prefix"]
            clean = runs.get((key, "clean", prefix))
            if clean is None:
                continue
            row = [prefix, case["category"], clean["de00_mean_color_vs_gt"]]
            for v in PERTURBED:
                p = projections(runs, key, [prefix], v)
                if p["prefixes"]:
                    d, g = p["d"][0], p["g"][0]
                    c = float(center(d) @ center(g) / max(center(g) @ center(g), 1e-12))
                    lum = float(d.mean() / g.mean()) if abs(g.mean()) > 1e-6 else None
                    row += [round(c, 4) if v in CHROMA else None, round(lum, 4) if lum is not None and v in LUM else None]
                else:
                    row += [None, None]
            thumbs = [img_dir / f"{prefix}_{key}_{v}.jpg" for v in ("clean", "warm")]
            row += [orbit.Image(str(t)) if t.exists() else None for t in thumbs]
            rows.append(row)
        columns = ["id", "category", "de00_mean_color_vs_gt"]
        for v in PERTURBED:
            columns += [f"chroma_{v}", f"lum_{v}"]
        columns += ["output_clean", "output_warm"]
        orbit.log({"test/cases": orbit.Table(columns=columns, data=rows, key="id")}, step=0)
        orbit.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--no-orbit", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    plan = json.loads((run / "plan.json").read_text())
    runs = load_runs(run)
    summary = summarize(plan, runs)
    page = build_html(run, plan, runs, summary)
    (run / "report" / "index.html").write_text(page)
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "definition": __doc__.split("\n\n")[1],
        "settings": {k: plan[k] for k in ("variants", "gains", "num_inference_steps", "true_cfg_scale", "seed_rule", "erode_px", "base_model")},
        "runs_done": len(runs),
        "checkpoints": summary,
        "headline": headline(summary),
    }
    (run / "summary.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(result["headline"])
    if not args.no_orbit:
        try:
            log_orbit(run, plan, runs, summary)
        except Exception as error:  # the summary and page stand on their own
            print(f"orbit logging failed: {error!r}")


if __name__ == "__main__":
    main()

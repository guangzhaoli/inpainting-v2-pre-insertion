#!/usr/bin/env python3
"""Metrics and HTML report for scripts/probe_reference_rope.py outputs (CPU only).

  python scripts/probe_reference_rope_report.py --out DIR [--procs 64] [--conclusion FILE]

Writes DIR/report/metrics.json (per-run rows + aggregates) and DIR/report/index.html.
Fill ratio / bbox-fill use failure_analysis_20261019/scripts/fill_ratio.py as is;
reference DINO follows failure_analysis_20261019/scripts/dino_baseline.py.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# One compute thread per pool worker: ~100 workers each running full-size
# OpenCV/BLAS thread pools oversubscribe the node, and forked children deadlock.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from probe_reference_rope import CHECKPOINTS, VARIANTS, detect_faces  # noqa: E402

cv2.setNumThreads(1)

RUNS = REPO / "output" / "teacher_v2_runs"
FILL_RATIO = RUNS / "failure_analysis_20261019" / "scripts" / "fill_ratio.py"
DINO_PATH = REPO.parent / "pretrained_weights" / "metrics_offline" / "dinov2-small"
VALIDATION_DIRS = {
    "animal_s250": RUNS / "20261014_135247/training/insertfuse_teacher_v2_animal/v2/validation/step-00000250",
    "animal_s3250": RUNS / "20261014_135247/training/insertfuse_teacher_v2_animal/v2/validation/step-00003250",
    "human_s1500": RUNS / "20261015_194628/training/insertfuse_teacher_v2_human/v0/validation/step-00001500",
    "jewelry_s3250": RUNS / "20261014_135247/training/insertfuse_teacher_v2_accessory_jewelry/v2/validation/step-00003250",
}
VARIANT_LABELS = {
    "native": "(0) native",
    "native_seed2": "(0′) native, seed+1000",
    "translate": "(1) translation",
    "affine_full": "(2) full-band affine",
    "affine_low5": "(3) low-band affine, k≥5",
    "affine_low7": "(3) low-band affine, k≥7",
    "polyptych": "(4) virtual polyptych",
}
# PIL's default bitmap font has no glyphs for the primes and >= used above.
STRIP_LABELS = {
    "native": "(0) native",
    "native_seed2": "(0') seed+1000",
    "translate": "(1) translate",
    "affine_full": "(2) full affine",
    "affine_low5": "(3) low k>=5",
    "affine_low7": "(3) low k>=7",
    "polyptych": "(4) polyptych",
}
CKPT_LABELS = {
    "animal_s250": "animal v2 · step 250",
    "animal_s3250": "animal v2 · step 3250",
    "human_s1500": "human · step 1500",
    "jewelry_s3250": "accessory_jewelry · step 3250",
}


def load_fill_ratio():
    spec = importlib.util.spec_from_file_location("fill_ratio", FILL_RATIO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rgb(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def gray(path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


# --------------------------------------------------------------------------- per-run metrics


def object_mask(src: np.ndarray, out: np.ndarray, mask: np.ndarray, thr: float = 18.0) -> np.ndarray:
    """fill_ratio.py's object extraction (blurred Lab difference inside the mask, largest component)."""
    scale = max(src.shape[:2]) / 1024.0
    k = int(round(7 * scale)) | 1
    a = cv2.cvtColor(cv2.GaussianBlur(src, (k, k), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(out, (k, k), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    obj = (np.sqrt(((a - b) ** 2).sum(-1)) > thr) & mask
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obj)
    if n > 1:
        return lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return obj > 0


def box_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def region_box(geom: dict, shape, margin: float = 0.25):
    """Placement region (core bbox + anchor box) with a margin, in processed pixels."""
    boxes = [geom["core_box_px"], geom["anchor_box_px"]]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(shape[1], int(x1 + mx)), min(shape[0], int(y1 + my)))


def run_geometry_metrics(item: dict) -> dict:
    """Processed-space metrics: object bbox vs anchor box, face scale for head swaps."""
    meta, case_dir, run_dir = item["meta"], Path(item["case_dir"]), Path(item["run_dir"])
    prefix, geom = meta["case"]["prefix"], meta["geometry"]
    src = rgb(case_dir / f"{prefix}_proc_source.jpg")
    core = gray(case_dir / f"{prefix}_proc_core.png") > 127
    out = rgb(run_dir / f"{prefix}_output_local.png")
    if out.shape[:2] != src.shape[:2]:
        out = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    obj = object_mask(src, out, core)
    row = {"object_found": bool(obj.any())}
    anchor = geom["anchor_box_px"]
    if obj.any():
        ys, xs = np.nonzero(obj)
        ob = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        aw, ah = anchor[2] - anchor[0], anchor[3] - anchor[1]
        row.update(
            object_box_px=list(ob),
            anchor_iou=box_iou(ob, anchor),
            object_to_anchor_area=((ob[2] - ob[0]) * (ob[3] - ob[1])) / max(aw * ah, 1),
            object_to_anchor_center=math.hypot(
                ((ob[0] + ob[2]) - (anchor[0] + anchor[2])) / 2 / aw, ((ob[1] + ob[3]) - (anchor[1] + anchor[3])) / 2 / ah
            ),
        )
    if geom["head_swap"] and geom["target_face_px"] is not None:
        old = geom["target_face_px"]
        cx0, cy0, cx1, cy1 = geom["core_box_px"]
        best = None
        for face in detect_faces(out):
            fx, fy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
            area = (face[2] - face[0]) * (face[3] - face[1])
            if cx0 <= fx <= cx1 and cy0 <= fy <= cy1 and (best is None or area > best[0]):
                best = (area, face)
        row["face_found"] = best is not None
        if best is not None:
            face = best[1]
            old_area = (old[2] - old[0]) * (old[3] - old[1])
            row.update(
                face_box_px=list(face),
                face_log_scale=0.5 * math.log(best[0] / old_area),
                face_center_offset=math.hypot(
                    (face[0] + face[2] - old[0] - old[2]) / 2, (face[1] + face[3] - old[1] - old[3]) / 2
                )
                / (old[3] - old[1]),
                face_iou=box_iou(face, old),
            )
    return row


def run_fill_ratio(item: dict) -> dict:
    fill_ratio = load_fill_ratio()
    run_dir, prefix = Path(item["run_dir"]), item["meta"]["case"]["prefix"]
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("row", "source", "source_mask", "output_restored_raw"):
            os.symlink(os.path.realpath(run_dir / f"{prefix}_{name}.png"), Path(tmp) / f"{prefix}_{name}.png")
        rows = fill_ratio.analyze(tmp)
    row = rows[0]
    return {
        "area_fill": row["obj_over_mask_area"],
        "h_cover": row["h_cover"],
        "w_cover": row["w_cover"],
        "mask_aspect": row["mask_aspect"],
        "bbox_full": bool(row["h_cover"] >= 0.95 and row["w_cover"] >= 0.95),
    }


_DINO = None


def _dino():
    global _DINO
    if _DINO is None:
        import torch
        from transformers import AutoModel

        torch.set_num_threads(1)
        _DINO = AutoModel.from_pretrained(str(DINO_PATH)).eval()
    return _DINO


def _embed(img01, mask):
    import torch
    from src.loss.region_perceptual_loss import _imagenet_normalize, _masked_bbox_crop

    crop, _ = _masked_bbox_crop(img01[None], mask[None], out_size=224, pad_ratio=0.1, input_range="0-1")
    with torch.no_grad():
        return _dino()(pixel_values=_imagenet_normalize(crop)).pooler_output[0].numpy()


def _t(img):
    import torch

    return torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)


def run_dino(item: dict) -> dict:
    """dino_baseline.py 'val outputs' mode: output (placement-mask crop) vs foreground reference."""
    import torch
    from src.utils.images import extract_foreground_from_image_and_mask

    run_dir, prefix = Path(item["run_dir"]), item["meta"]["case"]["prefix"]
    src_mask = gray(run_dir / f"{prefix}_source_mask.png")
    out = rgb(run_dir / f"{prefix}_output.png")
    if out.shape[:2] != src_mask.shape:
        out = cv2.resize(out, (src_mask.shape[1], src_mask.shape[0]))
    # Pre-crop beyond _masked_bbox_crop's 10% padding (same crop, far less memory).
    ys, xs = np.nonzero(src_mask > 127)
    my, mx = int(0.15 * (ys.max() - ys.min() + 1)) + 2, int(0.15 * (xs.max() - xs.min() + 1)) + 2
    y0, y1 = max(0, ys.min() - my), min(src_mask.shape[0], ys.max() + 1 + my)
    x0, x1 = max(0, xs.min() - mx), min(src_mask.shape[1], xs.max() + 1 + mx)
    out, src_mask = out[y0:y1, x0:x1], src_mask[y0:y1, x0:x1]
    out_emb = _embed(_t(out), torch.from_numpy((src_mask > 127).astype(np.float32)))
    ref_crop, ref_crop_mask = extract_foreground_from_image_and_mask(
        rgb(run_dir / f"{prefix}_reference.png"), gray(run_dir / f"{prefix}_reference_mask.png")
    )
    ref_emb = _embed(_t(ref_crop), torch.from_numpy((ref_crop_mask > 127).astype(np.float32)))
    cos = float(np.dot(out_emb, ref_emb) / (np.linalg.norm(out_emb) * np.linalg.norm(ref_emb)))
    return {"dino_ref": cos, "_out_emb": out_emb.tolist()}


def run_item_metrics(item: dict) -> dict:
    row = {"key": item["key"]}
    for fn in (run_fill_ratio, run_dino, run_geometry_metrics):
        try:
            row.update(fn(item))
        except Exception as error:  # keep the rest of the report alive; record the failure
            row.setdefault("errors", []).append(f"{fn.__name__}: {type(error).__name__}: {error}")
    return row


def native_change(pair: dict) -> dict:
    """Mean |variant - native| (0-255) in the placement region, processed space."""
    a = rgb(pair["variant"])
    b = rgb(pair["native"])
    x0, y0, x1, y1 = pair["region"]
    return {"key": pair["key"], "mae_vs_native": float(np.abs(a[y0:y1, x0:x1].astype(np.float32) - b[y0:y1, x0:x1]).mean())}


def validation_psnr(pair: dict) -> dict:
    a = rgb(pair["mine"]).astype(np.float32)
    b = rgb(pair["validation"]).astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    return {"key": pair["key"], "psnr": 99.0 if mse == 0 else 10 * math.log10(255**2 / mse)}


# --------------------------------------------------------------------------- grids


def draw_box(draw, box, color, width):
    draw.rectangle([box[0], box[1], box[2] - 1, box[3] - 1], outline=color, width=width)


def make_strip(task: dict) -> str:
    case_dir, prefix, geom = Path(task["case_dir"]), task["prefix"], task["geometry"]
    tile_h = 230
    ref = Image.open(case_dir / f"{prefix}_proc_reference.jpg").convert("RGB")
    rm = gray(case_dir / f"{prefix}_proc_ref_mask.png") > 127
    ref_np = np.array(ref)
    ref_np[~rm] = (0.35 * ref_np[~rm] + 0.65 * 255).astype(np.uint8)
    ref = Image.fromarray(ref_np)
    lw = max(3, ref.width // 160)
    draw_box(ImageDraw.Draw(ref), geom["key_box_px"], (0, 200, 0), lw)
    src = rgb(case_dir / f"{prefix}_proc_source.jpg")
    core = gray(case_dir / f"{prefix}_proc_core.png") > 127
    region = region_box(geom, src.shape)
    ctx = src.copy()
    ctx[core] = (0.6 * ctx[core] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)
    ctx = Image.fromarray(ctx)
    d = ImageDraw.Draw(ctx)
    draw_box(d, geom["anchor_box_px"], (0, 220, 0), max(3, ctx.width // 200))
    draw_box(d, region, (255, 255, 255), max(2, ctx.width // 300))
    tiles = [("reference / key box", ref), ("target / mask / anchor", ctx)]
    for variant in task["variants"]:
        path = Path(task["run_root"]) / variant / f"{prefix}_output_local.png"
        if not path.exists():
            tiles.append((STRIP_LABELS[variant], Image.new("RGB", (64, 64), (230, 230, 230))))
            continue
        out = Image.open(path).convert("RGB")
        if out.size != (src.shape[1], src.shape[0]):
            out = out.resize((src.shape[1], src.shape[0]), Image.Resampling.LANCZOS)
        crop = out.crop(region)
        cd = ImageDraw.Draw(crop)
        a = geom["anchor_box_px"]
        draw_box(cd, (a[0] - region[0], a[1] - region[1], a[2] - region[0], a[3] - region[1]), (0, 220, 0), 2)
        tiles.append((STRIP_LABELS[variant], crop))
    resized = []
    for label, image in tiles:
        w = max(1, round(image.width * tile_h / image.height))
        image = image.resize((min(w, 2 * tile_h), tile_h), Image.Resampling.LANCZOS)
        if image.width < 120:  # keep room for the label above narrow (full-body) crops
            padded = Image.new("RGB", (120, tile_h), (255, 255, 255))
            padded.paste(image, ((120 - image.width) // 2, 0))
            image = padded
        resized.append((label, image))
    width = sum(img.width for _, img in resized) + 6 * (len(resized) - 1)
    strip = Image.new("RGB", (width, tile_h + 20), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    x = 0
    for label, image in resized:
        strip.paste(image, (x, 20))
        draw.text((x + 2, 3), label, fill=(40, 40, 40))
        x += image.width + 6
    path = Path(task["img_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    strip.save(path, quality=84)
    return str(path)


# --------------------------------------------------------------------------- aggregation


def mean(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def aggregate(rows: list[dict], plan: dict) -> dict:
    by = defaultdict(dict)
    for row in rows:
        by[(row["checkpoint"], row["variant"])][row["prefix"]] = row
    summary = {}
    for ckpt, info in plan["checkpoints"].items():
        prefixes = [c["prefix"] for c in info["cases"]]
        native = by.get((ckpt, "native"), {})
        summary[ckpt] = {}
        for variant in plan["variants"]:
            runs = by.get((ckpt, variant), {})
            got = [runs[p] for p in prefixes if p in runs]
            if not got:
                continue
            paired = [
                runs[p]["dino_ref"] - native[p]["dino_ref"]
                for p in prefixes
                if p in runs and p in native and "dino_ref" in runs[p] and "dino_ref" in native[p]
            ]
            tall = [r for r in got if r.get("mask_aspect", 0) > 1.0]
            heads = [r for r in got if r.get("head_swap")]
            free = [r for r in got if not r.get("head_swap")]
            entry = {
                "n": len(got),
                "dino_ref_mean": mean([r.get("dino_ref") for r in got]),
                "dino_ref_delta_mean": mean(paired),
                "dino_ref_delta_wins": int(sum(d > 0 for d in paired)),
                "area_fill_median": median([r.get("area_fill") for r in got]),
                "n_bbox_full": int(sum(bool(r.get("bbox_full")) for r in got)),
                "tall_area_fill_median": median([r.get("area_fill") for r in tall]),
                "n_tall": len(tall),
                "anchor_iou_median": median([r.get("anchor_iou") for r in free]),
                "object_to_anchor_area_median": median([r.get("object_to_anchor_area") for r in free]),
                "mae_vs_native_mean": mean([r.get("mae_vs_native") for r in got]),
                "dino_to_native_mean": mean([r.get("dino_to_native") for r in got]),
                "n_errors": int(sum(bool(r.get("errors")) for r in got)),
            }
            if heads:
                found = [r for r in heads if r.get("face_found")]
                entry.update(
                    {
                        "n_head_swap": len(heads),
                        "face_detect_rate": len(found) / len(heads),
                        "face_abs_log_scale_median": median([abs(r["face_log_scale"]) for r in found]),
                        "face_log_scale_mean": mean([r["face_log_scale"] for r in found]),
                        "face_center_offset_median": median([r["face_center_offset"] for r in found]),
                        "face_iou_median": median([r["face_iou"] for r in found]),
                    }
                )
            attention = defaultdict(list)
            for r in got:
                for key, stats in (r.get("diagnostics") or {}).items():
                    attention[key].append(stats)
            entry["attention"] = {
                key: {
                    "sigma_mean": mean([s["sigma"] for s in values]),
                    "ref_obj_mass": mean([s["mass"]["ref_obj"] for s in values]),
                    "ref_bg_mass": mean([s["mass"]["ref_bg"] for s in values]),
                    "ref_obj_share": mean([s["ref_obj_share"] for s in values]),
                    "ref_obj_density_ratio": median([s["ref_obj_density_ratio"] for s in values]),
                    "target_mass": mean([s["mass"]["target"] for s in values]),
                    "source_mass": mean([s["mass"]["source"] for s in values]),
                    "text_mass": mean([s["mass"]["text"] for s in values]),
                    "centroid_dist": mean([s.get("centroid_dist") for s in values]),
                    "centroid_dist_uniform": mean([s.get("centroid_dist_uniform") for s in values]),
                    "centroid_dist_best_head": mean([s.get("centroid_dist_best_head") for s in values]),
                    "n": len(values),
                }
                for key, values in sorted(attention.items())
            }
            summary[ckpt][variant] = entry
    return summary


# --------------------------------------------------------------------------- HTML

CSS = """
:root { color-scheme: light; --bg:#fcfcfb; --surface:#fff; --surface-2:#f3f2ef; --border:#e2e0da; --text:#1b1b1a;
  --text-2:#52514e; --text-3:#7a7974; --accent:#2a78d6; --warn-bg:#fff4e5; --warn-border:#eda100; --ok-bg:#eaf6f0;
  --ok-border:#1baf7a; --pos:#1a7f4b; --neg:#b42318; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --bg:#161615; --surface:#1d1d1c;
  --surface-2:#252524; --border:#3a3936; --text:#f0efec; --text-2:#c3c2b7; --text-3:#93928a; --accent:#5598e7;
  --warn-bg:#33270f; --warn-border:#c98500; --ok-bg:#102a20; --ok-border:#199e70; --pos:#4cc38a; --neg:#f97066; } }
:root[data-theme="dark"] { color-scheme: dark; --bg:#161615; --surface:#1d1d1c; --surface-2:#252524; --border:#3a3936;
  --text:#f0efec; --text-2:#c3c2b7; --text-3:#93928a; --accent:#5598e7; --warn-bg:#33270f; --warn-border:#c98500;
  --ok-bg:#102a20; --ok-border:#199e70; --pos:#4cc38a; --neg:#f97066; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--bg); color:var(--text);
  font:15px/1.7 -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 28px 16px 64px; }
h1 { font-size: 26px; line-height: 1.35; margin: 0 0 6px; }
h2 { font-size: 20px; margin: 44px 0 10px; padding-top: 6px; border-top: 1px solid var(--border); }
h3 { font-size: 16px; margin: 26px 0 8px; }
.meta { color: var(--text-2); font-size: 13px; margin-bottom: 20px; }
a { color: var(--accent); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; background: var(--surface-2);
  padding: 1px 5px; border-radius: 4px; overflow-wrap: anywhere; }
.box { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin: 14px 0; }
.box.warn { background: var(--warn-bg); border-color: var(--warn-border); }
.box.ok { background: var(--ok-bg); border-color: var(--ok-border); }
.table-wrap { overflow-x: auto; margin: 12px 0 18px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; background: var(--surface); }
th, td { border: 1px solid var(--border); padding: 6px 8px; text-align: left; vertical-align: top; }
th { background: var(--surface-2); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.pos { color: var(--pos); } .neg { color: var(--neg); } .dim { color: var(--text-3); }
figure { margin: 10px 0 16px; }
figure img { max-width: 100%; height: auto; display: block; border: 1px solid var(--border); border-radius: 6px; background: #fff; }
figcaption { color: var(--text-2); font-size: 12.5px; margin-top: 4px; }
details { margin: 8px 0 14px; } summary { cursor: pointer; color: var(--text-2); font-size: 14px; }
.small { font-size: 13px; color: var(--text-2); }
.scroll { overflow-x: auto; }
"""


def fmt(value, digits=3, signed=False):
    if value is None:
        return '<span class="dim">–</span>'
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    if signed and abs(value) >= 10 ** -digits:
        return f'<span class="{"pos" if value > 0 else "neg"}">{text}</span>'
    return text


def render_html(plan, summary, rows, sanity, strips, conclusion_html, out: Path) -> str:
    esc = html.escape
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Reference RoPE 探针</title>",
        f"<style>{CSS}</style></head><body><main>",
        "<h1>参考图 RoPE 几何探针：跨图位置先验在 Teacher v2 里起不起作用</h1>",
        "<div class='meta'>"
        + esc(plan["created"][:10])
        + " · 纯推理，不训练。4 个 checkpoint（animal v2 step 250 / 3250、human 1500、accessory_jewelry 3250），"
        f"各自 validation 集全部用例 × 7 个变体；{plan['num_inference_steps']} 步，true_cfg {plan['true_cfg_scale']}，"
        "mask-aware 参考 CFG，expanded_soft 锚定；seed 规则与训练期验证一致（42 + 验证顺序）。"
        "原始数据：<a href='metrics.json'>metrics.json</a>。</div>",
    ]
    if conclusion_html:
        parts.append(f"<div class='box ok'>{conclusion_html}</div>")

    parts.append("<h2>1. 做了什么</h2>")
    parts.append(
        "<p>DiT（<code>src/models/transformer_qwenimage.py</code> 的 <code>QwenEmbedRope</code>）给每张图居中的 h/w 坐标，"
        "帧号按图序：0 目标、1 挖洞后的 source、2 参考。也就是说参考物体与放置区域在位置编码上没有任何对应关系。"
        "探针在推理时只替换参考那一段的 RoPE 表（<code>scripts/probe_reference_rope.py</code> 在运行时覆盖 "
        "<code>pos_embed.forward</code>），目标 / source token、去参考的 CFG 分支和全部权重都不动。"
        "角度由浮点坐标经 <code>torch.polar</code> 算出；h/w 第 k 个频带 ω<sub>k</sub> = 10000<sup>−2k/56</sup>，"
        "波长 2π/ω<sub>k</sub>（k=5 约 32 token，k=7 约 63 token）。</p>"
    )
    parts.append(
        "<div class='table-wrap'><table><tr><th>变体</th><th>参考 token 的坐标</th></tr>"
        "<tr><td>(0) native</td><td>原样（居中坐标，帧号 2）</td></tr>"
        "<tr><td>(0′) native, seed+1000</td><td>原样，只换 seed：同一 checkpoint 下“换个 seed 输出会变多少”的噪声基线</td></tr>"
        "<tr><td>(1) translation</td><td>整数平移，使参考 key box 中心落到 anchor box 中心；参考自注意力完全不变</td></tr>"
        "<tr><td>(2) full-band affine</td><td>key box → anchor box 的逐轴缩放 + 平移，作用于全部 h/w 频带（DreamFuse 式）</td></tr>"
        "<tr><td>(3) low-band affine</td><td>频带 k ≥ k<sub>split</sub> 用仿射坐标，k &lt; k<sub>split</sub> 保持原坐标；k<sub>split</sub> ∈ {5, 7}</td></tr>"
        "<tr><td>(4) virtual polyptych</td><td>帧号改为 0，w 平移成紧贴目标左侧的一块面板（h 不变）</td></tr></table></div>"
    )
    parts.append(
        "<p><strong>Anchor box</strong> 在裁剪、缩放后的目标 token 网格上计算。自由插入：先取放置 mask 的凸包，"
        "在其中找宽高比等于参考物体 bbox 的最大框（框内 ≥90% 落在凸包里），缩到 95% 后放置——animal 贴底、其余居中。"
        "换头（human 中 category=person、旧脸 ≥80% 落在 mask 内、mask 高 ≤ 4 倍旧脸高——这一条也排除了歪头时正脸检测器在脸颊上的小误检；"
        "参考里检到脸时还要求参考是头像或半身：参考物体高 ≤ 4 倍参考脸高，整身参考的这一比例是 6–11）：anchor 是在未挖洞原图上用 OpenCV 正脸检测器"
        "（与 <code>src/data/face_regions.py</code> 同一设置）找到的旧脸框。<strong>参考侧的 key box</strong> 默认是参考物体 bbox；"
        "换头时改用参考图里检测到的脸框，这样是“脸对脸”的对应——如果用整颗头的 bbox 对旧脸框，仿射会把头缩小约 1.5 倍。"
        "参考没检测到脸的换头用例回退到“物体 bbox ↔ 内接框”。</p>"
    )

    parts.append("<h3>完整性检查</h3><ul class='small'>")
    for line in sanity:
        parts.append(f"<li>{line}</li>")
    parts.append("</ul>")

    parts.append("<h2>2. 逐变体指标</h2>")
    parts.append(
        "<p class='small'>参考 DINO：输出在放置 mask 外框内的 crop 与参考前景的 DINOv2-small CLS 余弦（同 "
        "<code>dino_baseline.py</code>），Δ 是与 native 的逐例配对差，括号内是变好的例数。填充率与“外框贴满”沿用 "
        "<code>fill_ratio.py</code>（贴满 = 物体外框在高、宽两方向都 ≥95% 的 mask 外框）。anchor IoU：同一套物体提取在"
        "预测空间给出的物体外框与 anchor box 的 IoU（只算非换头用例）；面积比 = 物体外框面积 / anchor 面积。"
        "与 native 的差异：放置区域内 |变体 − native| 的像素均值（0–255）和两者输出 crop 的 DINO 余弦，"
        "(0′) 那一行就是 seed 噪声的量级。</p>"
    )
    for ckpt, entries in summary.items():
        if not entries:
            continue
        parts.append(f"<h3>{esc(CKPT_LABELS.get(ckpt, ckpt))}</h3><div class='table-wrap'><table>")
        head = (
            "<tr><th>变体</th><th class='num'>n</th><th class='num'>参考 DINO</th><th class='num'>Δ vs native</th>"
            "<th class='num'>面积填充率（中位）</th><th class='num'>外框贴满</th>"
        )
        if ckpt.startswith("animal"):
            head += "<th class='num'>高 blob 填充率</th>"
        head += (
            "<th class='num'>anchor IoU</th><th class='num'>物体/anchor 面积</th>"
            "<th class='num'>|Δ| vs native</th><th class='num'>DINO→native</th>"
        )
        has_face = any("n_head_swap" in e for e in entries.values())
        if has_face:
            head += "<th class='num'>换头：检出脸</th><th class='num'>|log 尺度|</th><th class='num'>log 尺度均值</th><th class='num'>中心偏移/脸高</th>"
        parts.append(head + "</tr>")
        for variant, e in entries.items():
            cells = [
                f"<td>{esc(VARIANT_LABELS[variant])}</td>",
                f"<td class='num'>{e['n']}</td>",
                f"<td class='num'>{fmt(e['dino_ref_mean'])}</td>",
                "<td class='num'>"
                + ("<span class='dim'>–</span>" if variant == "native" else f"{fmt(e['dino_ref_delta_mean'], signed=True)} ({e['dino_ref_delta_wins']}/{e['n']})")
                + "</td>",
                f"<td class='num'>{fmt(e['area_fill_median'])}</td>",
                f"<td class='num'>{e['n_bbox_full']}/{e['n']}</td>",
            ]
            if ckpt.startswith("animal"):
                cells.append(f"<td class='num'>{fmt(e['tall_area_fill_median'])} <span class='dim'>({e['n_tall']})</span></td>")
            cells += [
                f"<td class='num'>{fmt(e['anchor_iou_median'])}</td>",
                f"<td class='num'>{fmt(e['object_to_anchor_area_median'], 2)}</td>",
                f"<td class='num'>{fmt(e['mae_vs_native_mean'], 1) if variant != 'native' else '–'}</td>",
                f"<td class='num'>{fmt(e['dino_to_native_mean']) if variant != 'native' else '–'}</td>",
            ]
            if has_face:
                if "n_head_swap" in e:
                    cells += [
                        f"<td class='num'>{e['face_detect_rate'] * e['n_head_swap']:.0f}/{e['n_head_swap']}</td>",
                        f"<td class='num'>{fmt(e['face_abs_log_scale_median'])}</td>",
                        f"<td class='num'>{fmt(e['face_log_scale_mean'], signed=True)}</td>",
                        f"<td class='num'>{fmt(e['face_center_offset_median'], 2)}</td>",
                    ]
                else:
                    cells += ["<td></td>"] * 4
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table></div>")

    jewelry = defaultdict(dict)
    for row in rows:
        if row["checkpoint"] == "jewelry_s3250":
            jewelry[row["prefix"]][row["variant"]] = row
    if jewelry:
        compare = ["native", "affine_full", "affine_low5", "affine_low7"]
        parts.append(
            "<h3>accessory_jewelry 逐例：(2) 与 (3) 的对比</h3><p class='small'>按 anchor box 占预测画面的面积从小到大排序；"
            "每格是参考 DINO / anchor IoU。</p><div class='table-wrap'><table><tr><th>用例</th><th class='num'>anchor 面积占比</th>"
            + "".join(f"<th class='num'>{esc(VARIANT_LABELS[v])}</th>" for v in compare)
            + "</tr>"
        )

        def area_fraction(runs):
            geom = next(iter(runs.values()))["geometry"]
            x0, y0, x1, y1 = geom["anchor_box_px"]
            grid_h, grid_w = geom["target_grid"]
            return (x1 - x0) * (y1 - y0) / (grid_h * grid_w * 256)

        for prefix, runs in sorted(jewelry.items(), key=lambda item: area_fraction(item[1])):
            cells = [f"<td>{esc(prefix)}</td>", f"<td class='num'>{area_fraction(runs):.3f}</td>"]
            for variant in compare:
                r = runs.get(variant, {})
                cells.append(f"<td class='num'>{fmt(r.get('dino_ref'))} / {fmt(r.get('anchor_iou'), 2)}</td>")
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table></div>")

    parts.append("<h2>3. 注意力诊断</h2>")
    parts.append(
        "<p class='small'>条件分支（含参考）在第 20 / 30 / 40 层、σ 首次降到 0.9 / 0.7 / 0.4 以下的那一步，"
        "取放置 mask 内的目标 token 作 query（最多 768 个，等距抽样），对全部 key（文本 + 三张图）做 softmax，"
        "24 个头平均。ref obj / ref bg：落在参考物体 token 与参考白底 token 上的注意力质量；obj 占比 = obj /(obj+bg)；"
        "密度比 = 每个物体 token 的质量 / 每个白底 token 的质量（中位）。质心距离：只在 anchor box 内的 query 上，"
        "用对参考物体 token 的注意力（头平均后）求加权质心，与该 query 经 anchor 映射回参考图的位置比较，"
        "以参考 key box 对角线归一化；“均匀”是注意力在物体 token 上均匀分布时的同一距离，“最佳头”是 24 个头里最小的平均距离。"
        "距离越接近 0，说明 query 越在看参考里“对应的那一块”。</p>"
    )
    for ckpt, entries in summary.items():
        if not entries:
            continue
        keys = sorted({k for e in entries.values() for k in e["attention"]}, key=lambda k: (-float(k.split("_")[0][1:]), int(k.split("_l")[1])))
        parts.append(f"<details {'open' if ckpt == 'animal_s3250' else ''}><summary>{esc(CKPT_LABELS.get(ckpt, ckpt))}</summary><div class='table-wrap'><table>")
        parts.append(
            "<tr><th>σ / 层</th><th>变体</th><th class='num'>ref obj</th><th class='num'>ref bg</th><th class='num'>obj 占比</th>"
            "<th class='num'>密度比</th><th class='num'>目标</th><th class='num'>source</th><th class='num'>文本</th>"
            "<th class='num'>质心距离</th><th class='num'>均匀</th><th class='num'>最佳头</th></tr>"
        )
        for key in keys:
            first = True
            for variant, e in entries.items():
                a = e["attention"].get(key)
                if not a:
                    continue
                label = f"σ≈{a['sigma_mean']:.2f} · L{key.split('_l')[1]}" if first else ""
                first = False
                parts.append(
                    f"<tr><td>{label}</td><td>{esc(VARIANT_LABELS[variant])}</td>"
                    f"<td class='num'>{fmt(a['ref_obj_mass'])}</td><td class='num'>{fmt(a['ref_bg_mass'])}</td>"
                    f"<td class='num'>{fmt(a['ref_obj_share'], 2)}</td><td class='num'>{fmt(a['ref_obj_density_ratio'], 2)}</td>"
                    f"<td class='num'>{fmt(a['target_mass'])}</td><td class='num'>{fmt(a['source_mass'])}</td>"
                    f"<td class='num'>{fmt(a['text_mass'])}</td><td class='num'>{fmt(a['centroid_dist'])}</td>"
                    f"<td class='num'>{fmt(a['centroid_dist_uniform'])}</td><td class='num'>{fmt(a['centroid_dist_best_head'])}</td></tr>"
                )
        parts.append("</table></div></details>")

    parts.append("<h2>4. 逐例对比</h2>")
    parts.append(
        "<p class='small'>每行：参考（灰掉的是白底以外的部分，绿框 = key box）、预测空间的目标（红 = 放置 mask，绿框 = anchor box，"
        "白框 = 右侧各图的取景范围），然后是 7 个变体在放置区域的直接模型输出（<code>output_local</code>，未贴回原图），绿框同为 anchor box。"
        "同一行所有变体用同一个 seed（0′ 除外）。</p>"
    )
    by_case = defaultdict(dict)
    for row in rows:
        by_case[(row["checkpoint"], row["prefix"])][row["variant"]] = row
    for ckpt, info in plan["checkpoints"].items():
        parts.append(f"<details {'open' if ckpt == 'animal_s3250' else ''}><summary>{esc(CKPT_LABELS.get(ckpt, ckpt))}（{len(info['cases'])} 例）</summary>")
        for case in info["cases"]:
            img = strips.get((ckpt, case["prefix"]))
            if not img:
                continue
            runs = by_case.get((ckpt, case["prefix"]), {})
            native = runs.get("native", {})
            notes = []
            for variant in plan["variants"]:
                r = runs.get(variant)
                if r and r.get("dino_ref") is not None:
                    notes.append(f"{variant} {r['dino_ref']:.2f}")
            geom = native.get("geometry_kind", "")
            parts.append(
                f"<figure><div class='scroll'><img loading='lazy' src='{esc(os.path.relpath(img, out / 'report'))}' alt='{esc(case['prefix'])}'></div>"
                f"<figcaption>{esc(case['prefix'])} · anchor: {esc(geom)} · 参考 DINO: {esc(' / '.join(notes))}</figcaption></figure>"
            )
        parts.append("</details>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- main


def Pool(processes):
    return multiprocessing.get_context("spawn").Pool(processes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--procs", type=int, default=96)
    parser.add_argument("--conclusion", default=None, help="HTML snippet placed at the top of the report")
    parser.add_argument("--skip-strips", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="re-render index.html from report/metrics.json")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    conclusion = Path(args.conclusion).read_text() if args.conclusion else ""
    if args.render_only:
        payload = json.loads((out / "report" / "metrics.json").read_text())
        strips = {}
        for ckpt, info in plan["checkpoints"].items():
            for case in info["cases"]:
                path = out / "report" / "img" / ckpt / f"{case['prefix']}.jpg"
                if path.exists():
                    strips[(ckpt, case["prefix"])] = str(path)
        html_text = render_html(plan, payload["summary"], payload["runs"], payload["sanity"], strips, conclusion, out)
        (out / "report" / "index.html").write_text(html_text)
        print("re-rendered", out / "report" / "index.html")
        return

    items = []
    for ckpt, info in plan["checkpoints"].items():
        for case in info["cases"]:
            for variant in plan["variants"]:
                run_dir = out / "runs" / ckpt / variant
                meta_path = run_dir / f"{case['prefix']}.probe.json"
                if meta_path.exists():
                    items.append(
                        {
                            "key": f"{ckpt}|{variant}|{case['prefix']}",
                            "meta": json.loads(meta_path.read_text()),
                            "run_dir": str(run_dir),
                            "case_dir": str(out / "cases" / info["domain"]),
                        }
                    )
    print(f"{len(items)} runs with outputs", flush=True)
    with Pool(args.procs) as pool:
        metric_rows = {r["key"]: r for r in pool.map(run_item_metrics, items, chunksize=2)}

    rows = []
    for item in items:
        meta = item["meta"]
        geom = meta["geometry"]
        row = {
            "checkpoint": meta["checkpoint"],
            "variant": meta["variant"],
            "prefix": meta["case"]["prefix"],
            "category": meta["case"]["category"],
            "seed": meta["seed"],
            "head_swap": geom["head_swap"],
            "geometry_kind": geom["kind"],
            "geometry": geom,
            "rope_params": meta["rope_params"],
            "rope_overrides": meta["rope_overrides"],
            "cond_forwards": meta["cond_forwards"],
            "diagnostics": meta["diagnostics"],
            "sample_seconds": meta["sample_seconds"],
            **{k: v for k, v in metric_rows[item["key"]].items() if k != "key"},
        }
        rows.append(row)

    # Change relative to native (same checkpoint and case).
    index = {(r["checkpoint"], r["variant"], r["prefix"]): r for r in rows}
    pairs = []
    for r in rows:
        native = index.get((r["checkpoint"], "native", r["prefix"]))
        if native is None or r["variant"] == "native":
            continue
        if "_out_emb" in r and "_out_emb" in native:
            a, b = np.array(r["_out_emb"]), np.array(native["_out_emb"])
            r["dino_to_native"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        domain = plan["checkpoints"][r["checkpoint"]]["domain"]
        src_shape = gray(out / "cases" / domain / f"{r['prefix']}_proc_core.png").shape
        pairs.append(
            {
                "key": (r["checkpoint"], r["variant"], r["prefix"]),
                "variant": str(out / "runs" / r["checkpoint"] / r["variant"] / f"{r['prefix']}_output_local.png"),
                "native": str(out / "runs" / r["checkpoint"] / "native" / f"{r['prefix']}_output_local.png"),
                "region": region_box(r["geometry"], src_shape, margin=0.1),
            }
        )
    with Pool(args.procs) as pool:
        for result in pool.map(native_change, pairs, chunksize=4):
            index[result["key"]]["mae_vs_native"] = result["mae_vs_native"]

    # Native vs the training-time validation images (same seed, same order).
    val_pairs = []
    for ckpt, vdir in VALIDATION_DIRS.items():
        for case in plan["checkpoints"].get(ckpt, {}).get("cases", []):
            mine = out / "runs" / ckpt / "native" / f"{case['prefix']}_output_local.png"
            theirs = vdir / f"{case['prefix']}_output_local.png"
            if mine.exists() and theirs.exists():
                val_pairs.append({"key": (ckpt, case["prefix"]), "mine": str(mine), "validation": str(theirs)})
    with Pool(args.procs) as pool:
        psnr = pool.map(validation_psnr, val_pairs)
    sanity = []
    for ckpt in VALIDATION_DIRS:
        values = [p["psnr"] for p in psnr if p["key"][0] == ckpt]
        if values:
            sanity.append(
                f"{esc_text(CKPT_LABELS[ckpt])}：native 与训练期同 seed 验证图（<code>output_local</code>）的 PSNR 中位 "
                f"{np.median(values):.1f} dB，最低 {min(values):.1f} dB（{len(values)} 例）。"
            )
    overrides = [r for r in rows if r["variant"] not in ("native", "native_seed2")]
    bad = [r for r in overrides if r["rope_overrides"] != r["cond_forwards"] or r["rope_overrides"] == 0]
    sanity.append(
        f"RoPE 覆盖：{len(overrides) - len(bad)}/{len(overrides)} 个变体运行里，每次含参考的条件前向都替换了参考段"
        f"（每例 {overrides[0]['cond_forwards'] if overrides else 0} 次）；去参考的 CFG 分支只有两段图像，不受影响。"
    )
    kinds = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["variant"] == "native":
            kinds[r["checkpoint"]][r["geometry_kind"]] += 1
    sanity.append(
        "anchor 来源："
        + "；".join(f"{CKPT_LABELS[c]} " + "、".join(f"{k} {v}" for k, v in d.items()) for c, d in kinds.items())
        + "（face = 脸对脸，inscribed = 物体 bbox ↔ 内接框）。"
    )
    errors = [r for r in rows if r.get("errors")]
    if errors:
        sanity.append(f"指标计算报错 {len(errors)} 例（见 metrics.json 的 errors 字段）。")
    seconds = [r["sample_seconds"] for r in rows]
    sanity.append(f"单次采样（含编码与解码）耗时中位 {np.median(seconds):.0f} s，共 {len(rows)} 次。")

    summary = aggregate(rows, plan)

    strips = {}
    if not args.skip_strips:
        tasks = []
        for ckpt, info in plan["checkpoints"].items():
            for case in info["cases"]:
                native = index.get((ckpt, "native", case["prefix"]))
                if native is None:
                    continue
                tasks.append(
                    {
                        "key": (ckpt, case["prefix"]),
                        "case_dir": str(out / "cases" / info["domain"]),
                        "prefix": case["prefix"],
                        "geometry": native["geometry"],
                        "variants": plan["variants"],
                        "run_root": str(out / "runs" / ckpt),
                        "img_path": str(out / "report" / "img" / ckpt / f"{case['prefix']}.jpg"),
                    }
                )
        with Pool(args.procs) as pool:
            for task, path in zip(tasks, pool.map(make_strip, tasks, chunksize=1)):
                strips[task["key"]] = path

    for r in rows:
        r.pop("_out_emb", None)
    payload = {
        "plan": {k: v for k, v in plan.items() if k != "workers"},
        "sanity": sanity,
        "validation_psnr": [{"checkpoint": p["key"][0], "prefix": p["key"][1], "psnr": p["psnr"]} for p in psnr],
        "summary": summary,
        "runs": rows,
    }
    report = out / "report"
    report.mkdir(exist_ok=True)
    (report / "metrics.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    (report / "index.html").write_text(render_html(plan, summary, rows, sanity, strips, conclusion, out))
    print("wrote", report / "metrics.json", report / "index.html")


def esc_text(text: str) -> str:
    return html.escape(text)


if __name__ == "__main__":
    main()

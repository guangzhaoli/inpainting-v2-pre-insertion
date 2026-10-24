#!/usr/bin/env python3
"""Inference-only probe: does cross-image RoPE geometry change InsertFuse Teacher v2 outputs?

The DiT gives every image centred (h, w) RoPE coordinates plus a frame index
(0 target, 1 masked source, 2 reference), so the reference object's positional
relation to the placement region is arbitrary. This probe overrides only the
reference segment of the rotary table at inference time. Target and source
tokens, the null-reference CFG branch and all weights are untouched.

Variants (reference tokens only, same seed for all except ``native_seed2``):
  native        unchanged
  native_seed2  unchanged, seed + 1000 (seed-noise control)
  translate     integer shift so the reference key-box centre lands on the
                anchor-box centre; reference self-attention is unchanged
  affine_full   per-axis scale + shift (key box -> anchor box) in every h/w band
  affine_low5   affine coordinates in h/w bands k >= 5, native in k < 5
  affine_low7   affine coordinates in h/w bands k >= 7, native in k < 7
  polyptych     frame index 0 and w shifted so the reference is a left panel
                adjacent to the target

h/w band k has w_k = 10000^(-2k/56) (wavelength 2*pi/w_k tokens: k=5 ~ 32,
k=7 ~ 63). The key box is the reference object bbox, or the reference face for
head swaps; the anchor box is computed on the processed (post-crop) target.

  python scripts/probe_reference_rope.py plan --out DIR --workers 64
  python scripts/probe_reference_rope.py run --plan DIR/plan.json --worker 0          # static share
  python scripts/probe_reference_rope.py work --plan DIR/plan.json --name n0-g0      # shared queue

Metrics and the HTML report: scripts/probe_reference_rope_report.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUNS = REPO / "output" / "teacher_v2_runs"
CHECKPOINTS = {
    "animal_s250": ("animal", RUNS / "20261014_135247/training/insertfuse_teacher_v2_animal/v2/checkpoint-0-250"),
    "animal_s3250": ("animal", RUNS / "20261014_135247/training/insertfuse_teacher_v2_animal/v2/checkpoint-1-3250"),
    "human_s1500": ("human", RUNS / "20261015_194628/training/insertfuse_teacher_v2_human/v0/checkpoint-0-1500"),
    "jewelry_s3250": (
        "accessory_jewelry",
        RUNS / "20261014_135247/training/insertfuse_teacher_v2_accessory_jewelry/v2/checkpoint-1-3250",
    ),
}
VARIANTS = ("native", "native_seed2", "translate", "affine_full", "affine_low5", "affine_low7", "polyptych")
BOTTOM_ALIGNED_DOMAINS = {"animal", "general_object"}
SEED_BASE = 42
SEED2_OFFSET = 1000
NUM_STEPS = 40
TRUE_CFG = 2.5
PATCH = 16  # processed pixels per DiT token (VAE x8, patch 2)
DIAG_LAYERS = (20, 30, 40)
DIAG_SIGMAS = (0.9, 0.7, 0.4)
DIAG_MAX_QUERIES = 768


# --------------------------------------------------------------------------- geometry


@lru_cache(maxsize=1)
def _face_detector():
    # Same packaged detector as src/data/face_regions.py.
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("OpenCV's packaged frontal face detector is unavailable")
    return detector


def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Frontal faces as (x0, y0, x1, y1) with the face_regions.py settings."""
    height, width = rgb.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))))
    faces = _face_detector().detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    boxes = []
    for x, y, w, h in faces:
        x0, y0 = max(0, int(x / scale)), max(0, int(y / scale))
        x1, y1 = min(width, int((x + w) / scale)), min(height, int((y + h) / scale))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def largest_face_inside(rgb: np.ndarray, mask: np.ndarray, min_inside: float = 0.8):
    """Largest face with >= min_inside of its box in ``mask`` (face_regions.py rule)."""
    best = None
    for x0, y0, x1, y1 in detect_faces(rgb):
        inside = float(mask[y0:y1, x0:x1].mean())
        area = (x1 - x0) * (y1 - y0)
        if inside >= min_inside and (best is None or area > best[0]):
            best = (area, (x0, y0, x1, y1))
    return None if best is None else best[1]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        raise ValueError("empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def filled_hull(mask: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    hull = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    return hull.astype(bool) | mask


def inscribed_box(
    mask: np.ndarray,
    aspect_hw: float,
    bottom: bool,
    coverage: float = 0.9,
    slack: float = 0.95,
    cell: int = 4,
) -> tuple[int, int, int, int]:
    """Box with height/width ``aspect_hw`` inscribed in the placement mask.

    The mask is replaced by its filled convex hull, so thin or concave masks
    (necklace bands, silhouettes) behave like the region they enclose. The box
    is the largest one with >= ``coverage`` of its area inside the hull; it is
    then shrunk by ``slack`` and placed bottom-most (``bottom``) or closest to
    the hull centroid.
    """
    hull = filled_hull(mask)
    height, width = hull.shape
    rows, cols = math.ceil(height / cell), math.ceil(width / cell)
    padded = np.zeros((rows * cell, cols * cell), np.float64)
    padded[:height, :width] = hull
    frac = padded.reshape(rows, cell, cols, cell).mean(axis=(1, 3))
    integral = np.zeros((rows + 1, cols + 1))
    integral[1:, 1:] = frac.cumsum(0).cumsum(1)

    def cover(box_h: int, box_w: int) -> np.ndarray:
        total = (
            integral[box_h:, box_w:] - integral[:-box_h, box_w:] - integral[box_h:, :-box_w] + integral[:-box_h, :-box_w]
        )
        return total / (box_h * box_w)

    def size(box_h: int) -> tuple[int, int]:
        return box_h, max(1, round(box_h / aspect_hw))

    def feasible(box_h: int) -> bool:
        box_h, box_w = size(box_h)
        return box_h <= rows and box_w <= cols and bool((cover(box_h, box_w) >= coverage).any())

    lo, hi = 1, rows
    if not feasible(lo):
        x0, y0, x1, y1 = mask_bbox(mask)
        return x0, y0, x1, y1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    box_h, box_w = size(max(1, int(slack * lo)))
    if not (cover(box_h, box_w) >= coverage).any():
        box_h, box_w = size(lo)
    ys, xs = np.nonzero(cover(box_h, box_w) >= coverage)
    hy, hx = np.nonzero(hull)
    cy, cx = hy.mean() / cell, hx.mean() / cell
    if bottom:
        keep = ys == ys.max()
        ys, xs = ys[keep], xs[keep]
        pick = int(np.argmin(np.abs(xs + box_w / 2 - cx)))
    else:
        pick = int(np.argmin((ys + box_h / 2 - cy) ** 2 + (xs + box_w / 2 - cx) ** 2))
    y, x = int(ys[pick]), int(xs[pick])
    return x * cell, y * cell, min(width, (x + box_w) * cell), min(height, (y + box_h) * cell)


def token_box(box_px) -> dict:
    """Pixel box [x0, x1) -> centre and size in token-index units (token i centre = i)."""
    x0, y0, x1, y1 = box_px
    return {
        "cy": (y0 + y1) / (2 * PATCH) - 0.5,
        "cx": (x0 + x1) / (2 * PATCH) - 0.5,
        "h": (y1 - y0) / PATCH,
        "w": (x1 - x0) / PATCH,
    }


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def token_mask(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    rows, cols = grid
    pooled = mask[: rows * PATCH, : cols * PATCH].reshape(rows, PATCH, cols, PATCH).mean(axis=(1, 3))
    return pooled >= 0.5


def compute_geometry(data: dict, domain: str, category: str) -> dict:
    """Key box (reference), anchor box (target) and token masks from a predict batch."""
    target = tensor_to_uint8(data["tar_image"][0])
    core = data["tar_mask"][0].detach().float().cpu().numpy() > 0.5
    reference = tensor_to_uint8(data["ref_image"][0])
    ref_mask = data["ref_mask"][0].detach().float().cpu().numpy() > 0.5
    target_grid = (int(data["height"]) // PATCH, int(data["width"]) // PATCH)
    ref_grid = (reference.shape[0] // PATCH, reference.shape[1] // PATCH)
    if target.shape[:2] != (target_grid[0] * PATCH, target_grid[1] * PATCH):
        raise ValueError(f"target {target.shape} is not on the {PATCH}px token grid {target_grid}")
    if reference.shape[:2] != (ref_grid[0] * PATCH, ref_grid[1] * PATCH):
        raise ValueError(f"reference {reference.shape} is not on the {PATCH}px token grid {ref_grid}")

    object_box = mask_bbox(ref_mask)
    core_box = mask_bbox(core)
    target_face = largest_face_inside(target, core) if domain == "human" else None
    ref_face = largest_face_inside(reference, ref_mask) if domain == "human" else None
    # Head swap: the old face lies in a head-sized mask (<= 4 face heights; this
    # also rejects small false detections on tilted faces) and the reference,
    # when its face is found, is a head/bust crop (full-body references are ~6-11).
    head_swap = False
    if domain == "human" and category == "person" and target_face is not None:
        head_swap = core_box[3] - core_box[1] <= 4.0 * (target_face[3] - target_face[1])
        if head_swap and ref_face is not None:
            head_swap = object_box[3] - object_box[1] <= 4.0 * (ref_face[3] - ref_face[1])
    if head_swap and ref_face is not None:
        kind, key_box, anchor_box = "face", ref_face, target_face
    else:
        kind, key_box = "inscribed", object_box
        aspect = (key_box[3] - key_box[1]) / (key_box[2] - key_box[0])
        anchor_box = inscribed_box(core, aspect, bottom=domain in BOTTOM_ALIGNED_DOMAINS)
    return {
        "kind": kind,
        "head_swap": bool(head_swap),
        "target_grid": list(target_grid),
        "ref_grid": list(ref_grid),
        "core_box_px": list(core_box),
        "ref_object_box_px": list(object_box),
        "target_face_px": None if target_face is None else list(target_face),
        "ref_face_px": None if ref_face is None else list(ref_face),
        "key_box_px": list(key_box),
        "anchor_box_px": list(anchor_box),
        "key": token_box(key_box),
        "anchor": token_box(anchor_box),
        "_core_tokens": token_mask(core, target_grid),
        "_ref_object_tokens": token_mask(ref_mask, ref_grid),
        "_images": {"target": target, "core": core, "reference": reference, "ref_mask": ref_mask},
    }


# --------------------------------------------------------------------------- RoPE override


def _centred(n: int) -> np.ndarray:
    # QwenEmbedRope(scale_rope=True): index i -> i - (n - n // 2)
    return np.arange(n, dtype=np.float64) - (n - n // 2)


def reference_positions(variant: str, geom: dict, h_bands: int) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Per-token, per-band reference coordinates: (frame, h [N, bands], w [N, bands], params)."""
    ref_h, ref_w = geom["ref_grid"]
    tgt_h, tgt_w = geom["target_grid"]
    hh, ww = np.meshgrid(_centred(ref_h), _centred(ref_w), indexing="ij")
    hn, wn = hh.reshape(-1, 1), ww.reshape(-1, 1)
    key, anchor = geom["key"], geom["anchor"]
    # Box centres in RoPE coordinates of their own image.
    key_cy, key_cx = key["cy"] - (ref_h - ref_h // 2), key["cx"] - (ref_w - ref_w // 2)
    anc_cy, anc_cx = anchor["cy"] - (tgt_h - tgt_h // 2), anchor["cx"] - (tgt_w - tgt_w // 2)
    frame, params = 2.0, {}
    if variant == "translate":
        dy, dx = round(anc_cy - key_cy), round(anc_cx - key_cx)
        h, w, params = hn + dy, wn + dx, {"dy": dy, "dx": dx}
    elif variant.startswith("affine_"):
        sy, sx = anchor["h"] / key["h"], anchor["w"] / key["w"]
        ah, aw = anc_cy + sy * (hn - key_cy), anc_cx + sx * (wn - key_cx)
        split = 0 if variant == "affine_full" else int(variant.removeprefix("affine_low"))
        low = np.arange(h_bands)[None, :] >= split
        h, w = np.where(low, ah, hn), np.where(low, aw, wn)
        params = {"sy": sy, "sx": sx, "k_split": split}
    elif variant == "polyptych":
        dx = -(tgt_w - tgt_w // 2) - ref_w // 2
        frame, h, w, params = 0.0, hn, wn + dx, {"dx": dx, "frame": 0}
    else:
        raise ValueError(f"no reference override for variant {variant!r}")
    return frame, np.broadcast_to(h, (len(hn), h_bands)), np.broadcast_to(w, (len(wn), h_bands)), params


def rope_table(theta: float, axes_dim, frame: float, h: np.ndarray, w: np.ndarray) -> torch.Tensor:
    """Complex rotary table [N, sum(axes)/2] from float positions (per band for h/w)."""
    f_dim, h_dim, w_dim = axes_dim

    def omega(dim):
        return 1.0 / np.power(float(theta), np.arange(0, dim, 2, dtype=np.float64) / dim)

    n = h.shape[0]
    angles = np.concatenate(
        [np.full((n, f_dim // 2), frame) * omega(f_dim), h * omega(h_dim), w * omega(w_dim)], axis=1
    )
    angles = torch.from_numpy(angles)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


# --------------------------------------------------------------------------- probe hooks


class ReferenceRopeProbe:
    """RoPE override for the reference segment plus hole-query attention diagnostics."""

    def __init__(self, trainer, domain: str):
        from src.models.transformer_qwenimage import apply_rotary_emb_qwen

        self.apply_rope = apply_rotary_emb_qwen
        self.domain = domain
        self.trainer = trainer
        self.dit = trainer._unwrap_dit()
        self.rope = self.dit.pos_embed
        self.variant = "native"
        self.category = ""
        self.geom = None
        self.rope_params = None
        self._ref_table = None
        self._original_rope_forward = self.rope.forward
        self.rope.forward = self._rope_forward

        original_prepare = trainer.prepare_predict_batch_data

        def prepare(*args, **kwargs):
            data = original_prepare(*args, **kwargs)
            self._on_batch(data)
            return data

        trainer.prepare_predict_batch_data = prepare
        self.dit.register_forward_pre_hook(self._dit_pre_hook, with_kwargs=True)
        for layer in DIAG_LAYERS:
            block = self.dit.transformer_blocks[layer]
            block.attn.register_forward_pre_hook(self._make_attn_hook(layer), with_kwargs=True)
        self.reset("native", "")

    def reset(self, variant: str, category: str):
        self.variant, self.category = variant, category
        self.geom, self.rope_params, self._ref_table = None, None, None
        self.diagnostics, self._sigmas_done, self._current = {}, set(), None
        self.cond_calls, self.rope_overrides = 0, 0

    # -- batch geometry
    def _on_batch(self, data: dict):
        self.geom = compute_geometry(data, self.domain, self.category)
        if self.variant not in ("native", "native_seed2"):
            h_bands = self.rope.axes_dim[1] // 2
            frame, h, w, params = reference_positions(self.variant, self.geom, h_bands)
            self._ref_table = rope_table(self.rope.theta, self.rope.axes_dim, frame, h, w)
            self.rope_params = params
        core = torch.from_numpy(self.geom["_core_tokens"].reshape(-1))
        queries = torch.nonzero(core).flatten()
        if len(queries) > DIAG_MAX_QUERIES:
            queries = queries[torch.linspace(0, len(queries) - 1, DIAG_MAX_QUERIES).round().long()]
        self._queries = queries
        ref_h, ref_w = self.geom["ref_grid"]
        obj = torch.from_numpy(self.geom["_ref_object_tokens"].reshape(-1))
        self._ref_obj = obj
        yy, xx = torch.meshgrid(torch.arange(ref_h), torch.arange(ref_w), indexing="ij")
        self._ref_pos = torch.stack([yy.reshape(-1), xx.reshape(-1)], 1).float()

    # -- RoPE
    def _rope_forward(self, video_fhw, txt_seq_lens, device):
        vid_freqs, txt_freqs = self._original_rope_forward(video_fhw, txt_seq_lens, device)
        shapes = video_fhw[0] if isinstance(video_fhw, list) else video_fhw
        shapes = shapes if isinstance(shapes, list) else [shapes]
        if len(shapes) == 3:
            self.cond_calls += 1
        elif len(shapes) != 2:
            raise ValueError(f"probe expects [target, source(, reference)] image shapes, got {shapes}")
        if self._ref_table is None or len(shapes) != 3:
            return vid_freqs, txt_freqs
        frame, height, width = (int(v) for v in shapes[2])
        if (frame, height, width) != (1, *self.geom["ref_grid"]):
            raise ValueError(f"reference grid {shapes[2]} != probe geometry {self.geom['ref_grid']}")
        start = math.prod(shapes[0]) + math.prod(shapes[1])
        count = frame * height * width
        table = self._ref_table.to(device=vid_freqs.device, dtype=vid_freqs.dtype)
        self.rope_overrides += 1
        return torch.cat([vid_freqs[:start], table, vid_freqs[start + count :]], 0), txt_freqs

    # -- attention diagnostics
    def _dit_pre_hook(self, module, args, kwargs):
        self._current = None
        shapes = kwargs.get("img_shapes")
        if self.geom is None or not shapes or len(shapes[0]) != 3:
            return
        sigma = float(kwargs["timestep"].flatten()[0])
        pending = [s for s in DIAG_SIGMAS if s not in self._sigmas_done and sigma <= s + 1e-6]
        if pending:
            target = max(pending)
            self._sigmas_done.update(pending)
            self._current = {"target": target, "sigma": sigma, "shapes": [list(map(int, s)) for s in shapes[0]]}

    def _make_attn_hook(self, layer: int):
        def hook(module, args, kwargs):
            if self._current is not None:
                with torch.no_grad():
                    key = f"s{self._current['target']}_l{layer}"
                    self.diagnostics[key] = self._attention_stats(module, kwargs, layer)

        return hook

    def _attention_stats(self, attn, kwargs, layer: int) -> dict:
        hidden, text = kwargs["hidden_states"], kwargs["encoder_hidden_states"]
        img_freqs, txt_freqs = kwargs["image_rotary_emb"]
        shapes = self._current["shapes"]
        n_tgt, n_src, n_ref = (math.prod(s) for s in shapes)
        n_txt = text.shape[1]
        device = hidden.device
        q_idx = self._queries.to(device)
        heads = attn.heads

        def project(linear, norm, x, freqs):
            y = linear(x).unflatten(-1, (heads, -1))
            if norm is not None:
                y = norm(y)
            return self.apply_rope(y, freqs, use_real=False)[0].float()

        q = project(attn.to_q, attn.norm_q, hidden[:1, q_idx], img_freqs[q_idx])
        k = torch.cat(
            [
                project(attn.add_k_proj, attn.norm_added_k, text[:1], txt_freqs),
                project(attn.to_k, attn.norm_k, hidden[:1], img_freqs),
            ],
            0,
        )
        scale = q.shape[-1] ** -0.5
        ref0 = n_txt + n_tgt + n_src
        obj = self._ref_obj.to(device)
        pos_obj = self._ref_pos.to(device)[obj]
        groups = {"text": [], "target": [], "source": [], "ref_obj": [], "ref_bg": []}
        obj_weight_sum = torch.zeros(len(q_idx), int(obj.sum()), device=device)
        head_centroids = []
        for h0 in range(0, heads, 6):
            logits = torch.einsum("qhd,khd->hqk", q[:, h0 : h0 + 6], k[:, h0 : h0 + 6]) * scale
            probs = logits.softmax(-1)
            ref = probs[..., ref0 : ref0 + n_ref]
            ref_obj = ref[..., obj]
            groups["text"].append(probs[..., :n_txt].sum(-1))
            groups["target"].append(probs[..., n_txt : n_txt + n_tgt].sum(-1))
            groups["source"].append(probs[..., n_txt + n_tgt : ref0].sum(-1))
            groups["ref_obj"].append(ref_obj.sum(-1))
            groups["ref_bg"].append(ref.sum(-1) - ref_obj.sum(-1))
            obj_weight_sum += ref_obj.sum(0)
            head_centroids.append(ref_obj @ pos_obj / ref_obj.sum(-1, keepdim=True).clamp_min(1e-12))
            del logits, probs, ref, ref_obj
        masses = {name: torch.cat(parts, 0) for name, parts in groups.items()}  # [heads, nq]
        centroid = obj_weight_sum @ pos_obj / obj_weight_sum.sum(-1, keepdim=True).clamp_min(1e-12)
        head_centroids = torch.cat(head_centroids, 0)  # [heads, nq, 2]

        # Anchor-mapped reference position of each query (key box <-> anchor box).
        tgt_w = shapes[0][2]
        qy, qx = (q_idx // tgt_w).float(), (q_idx % tgt_w).float()
        key, anchor = self.geom["key"], self.geom["anchor"]
        sy, sx = anchor["h"] / key["h"], anchor["w"] / key["w"]
        expected = torch.stack([key["cy"] + (qy - anchor["cy"]) / sy, key["cx"] + (qx - anchor["cx"]) / sx], 1)
        inside = ((qy - anchor["cy"]).abs() <= anchor["h"] / 2) & ((qx - anchor["cx"]).abs() <= anchor["w"] / 2)
        diag = math.hypot(key["h"], key["w"])

        def dist(c):
            return ((c - expected).norm(dim=-1) / diag)[..., inside]

        n_obj, n_bg = int(obj.sum()), int(n_ref - obj.sum())
        mean = {name: float(v.mean()) for name, v in masses.items()}
        stats = {
            "sigma": self._current["sigma"],
            "layer": layer,
            "n_queries": int(len(q_idx)),
            "n_queries_in_anchor": int(inside.sum()),
            "mass": mean,
            "ref_obj_share": mean["ref_obj"] / max(mean["ref_obj"] + mean["ref_bg"], 1e-12),
            "ref_obj_density_ratio": (mean["ref_obj"] / max(n_obj, 1)) / max(mean["ref_bg"] / max(n_bg, 1), 1e-12),
            "head_ref_obj_mass": [round(float(v), 5) for v in masses["ref_obj"].mean(1)],
        }
        if int(inside.sum()) > 0:
            per_head = dist(head_centroids).mean(-1)
            stats.update(
                {
                    "centroid_dist": float(dist(centroid).mean()),
                    "centroid_dist_uniform": float(dist(pos_obj.mean(0, keepdim=True).expand_as(centroid)).mean()),
                    "centroid_dist_best_head": float(per_head.min()),
                    "head_centroid_dist": [round(float(v), 4) for v in per_head],
                }
            )
        return stats


# --------------------------------------------------------------------------- plan


def case_prefix(order: int, index: int, category: str, stem: str) -> str:
    def safe(value):
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))[:80] or "sample"

    return f"{order:03d}-idx{index:06d}-{safe(category)}-{safe(stem)}"


def load_cases(domain: str) -> list[dict]:
    rows = json.loads((REPO / "configs" / "teacher_v2" / f"validation_{domain}.json").read_text())
    # Training-time validation order and seeds: randperm(seed 42), seed = 42 + order.
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(SEED_BASE)).tolist()
    cases = []
    for order, index in enumerate(perm):
        row = rows[index]
        cases.append(
            {
                "order": order,
                "index": index,
                "category": row["category"],
                "stem": str(row["stem"]),
                "prompt": row.get("prompt") or "",
                "seed": SEED_BASE + order,
                "prefix": case_prefix(order, index, row["category"], row["stem"]),
                **{key: row[key] for key in ("ref_image", "ref_mask", "tar_image", "tar_mask")},
            }
        )
    return cases


def cmd_plan(args):
    out = Path(args.out).resolve()
    checkpoints = {}
    for key, (domain, path) in CHECKPOINTS.items():
        if not (path / "train_config.yaml").exists():
            raise FileNotFoundError(path / "train_config.yaml")
        checkpoints[key] = {"domain": domain, "path": str(path), "cases": load_cases(domain)}
    work = {key: [[c["order"], v] for c in ck["cases"] for v in VARIANTS] for key, ck in checkpoints.items()}
    total = sum(len(items) for items in work.values())
    # Largest-remainder share of workers per checkpoint, at least one each.
    raw = {key: args.workers * len(items) / total for key, items in work.items()}
    share = {key: max(1, int(value)) for key, value in raw.items()}
    for key in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if sum(share.values()) >= args.workers:
            break
        share[key] += 1
    while sum(share.values()) > args.workers:
        key = max(share, key=lambda k: share[k])
        share[key] -= 1
    workers = []
    for key, items in work.items():
        for part in np.array_split(np.arange(len(items)), share[key]):
            workers.append({"checkpoint": key, "items": [items[i] for i in part]})
    plan = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": str(out),
        "variants": list(VARIANTS),
        "num_inference_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG,
        "seed_rule": f"{SEED_BASE} + validation order (native_seed2: +{SEED2_OFFSET})",
        "diagnostics": {"layers": list(DIAG_LAYERS), "sigmas": list(DIAG_SIGMAS), "max_queries": DIAG_MAX_QUERIES},
        "checkpoints": checkpoints,
        "workers": workers,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    sizes = [len(w["items"]) for w in workers]
    print(f"{len(workers)} workers, {total} runs, per-worker {min(sizes)}-{max(sizes)}; share {share}")


# --------------------------------------------------------------------------- run


def read_rgb(path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def read_mask(path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("L").copy()


def atomic_save(image: Image.Image, path: Path, **kwargs):
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    image.save(tmp, format="PNG" if path.suffix == ".png" else "JPEG", **kwargs)
    os.replace(tmp, path)


def link(target: Path, path: Path):
    if not path.exists():
        tmp = path.with_name(f".{path.name}.{os.getpid()}.lnk")
        tmp.unlink(missing_ok=True)
        os.symlink(os.path.relpath(target, path.parent), tmp)
        os.replace(tmp, path)


def setup_trainer(checkpoint: Path):
    import infer_testset

    argv = [
        "--config", str(checkpoint / "train_config.yaml"),
        "--lora", str(checkpoint),
        "--num-inference-steps", str(NUM_STEPS),
        "--true-cfg-scale", str(TRUE_CFG),
        "--guidance-scale", "1.0",
        "--device", "cuda:0",
    ]  # fmt: skip
    trainer, _, _, _ = infer_testset.setup_trainer(infer_testset.parse_args(argv))
    return trainer


def run_item(trainer, probe: ReferenceRopeProbe, case: dict, variant: str, out: Path, ckpt_key: str, domain: str, saver):
    from infer import seed_everything
    from src.utils.adaptive_crop import crop_pil_pair, restore_pil_output
    from src.utils.adaptive_crop_anchoring import prepare_adaptive_crop_for_inference
    from src.utils.latent_anchoring import ensure_latent_anchoring_config
    from src.utils.visualization import build_strict_mask_composite, ensure_mask, ensure_rgb

    started = time.time()
    seed = case["seed"] + (SEED2_OFFSET if variant == "native_seed2" else 0)
    seed_everything(seed)
    ref_image, ref_mask = read_rgb(case["ref_image"]), read_mask(case["ref_mask"])
    source, source_mask = ensure_rgb(read_rgb(case["tar_image"])), ensure_mask(read_mask(case["tar_mask"]))

    # Same wrapper as InsertAnythingValidationSampler.run_validation_loop.
    crop_cfg = trainer.config.data.init_args.adaptive_crop
    anchoring = ensure_latent_anchoring_config(trainer.config)
    anchored = anchoring.mode.value != "legacy"
    crop_plan, edge_feather_px = None, 0
    predict_source, predict_mask = source, source_mask
    if crop_cfg.predict_enabled:
        crop_plan, crop_compat = prepare_adaptive_crop_for_inference(
            image_size=source.size,
            mask=source_mask,
            adaptive_crop_config=crop_cfg,
            config=trainer.config,
            processor_config=trainer.config.data.init_args.processor,
        )
        if anchored and crop_plan.applied:
            edge_feather_px = int(crop_compat.get("feather_width_px", 8))
        predict_source, predict_mask = crop_pil_pair(source, source_mask, crop_plan)

    probe.reset(variant, case["category"])
    local = ensure_rgb(
        trainer.predict(
            image=[predict_source, ref_image],
            prompt=case["prompt"],
            ref_mask=ref_mask,
            src_mask=predict_mask,
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG,
            guidance_scale=1.0,
            negative_prompt=None,
            weight_dtype=torch.bfloat16,
            latent_anchoring_mode=anchoring.mode.value,
        )[0]
    )
    sample_seconds = time.time() - started
    if variant not in ("native", "native_seed2") and probe.rope_overrides != probe.cond_calls:
        raise RuntimeError(f"RoPE override hit {probe.rope_overrides}/{probe.cond_calls} conditional forwards")
    if crop_plan is not None:
        raw_restored, composite = restore_pil_output(
            original_source=source, original_mask=source_mask, local_output=local, plan=crop_plan,
            blend_mode=crop_cfg.blend_mode,
        )  # fmt: skip
    else:
        raw_restored = local.resize(source.size, Image.Resampling.LANCZOS)
        composite = build_strict_mask_composite(source, source_mask, raw_restored, crop_cfg.mask_threshold)
    primary = composite
    if anchored:
        primary = raw_restored
        if edge_feather_px:
            primary, _ = restore_pil_output(
                original_source=source, original_mask=source_mask, local_output=local, plan=crop_plan,
                blend_mode=crop_cfg.blend_mode, edge_feather_px=edge_feather_px,
            )  # fmt: skip

    geom = probe.geom
    images = geom.pop("_images")
    meta = {
        "checkpoint": ckpt_key,
        "domain": domain,
        "variant": variant,
        "seed": seed,
        "case": {k: case[k] for k in ("order", "index", "category", "stem", "prefix", "ref_image", "tar_image")},
        "geometry": {k: v for k, v in geom.items() if not k.startswith("_")},
        "rope_params": probe.rope_params,
        "rope_overrides": probe.rope_overrides,
        "cond_forwards": probe.cond_calls,
        "diagnostics": probe.diagnostics,
        "crop_plan": crop_plan.to_dict() if crop_plan is not None else None,
        "sizes": {"source": list(source.size), "local": list(local.size), "reference": list(ref_image.size)},
        "sample_seconds": round(sample_seconds, 1),
    }
    case_dir = out / "cases" / domain
    run_dir = out / "runs" / ckpt_key / variant
    prefix = case["prefix"]

    def save():
        case_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        fast = {"compress_level": 1}
        shared = {
            "source": source,
            "source_mask": source_mask,
            "reference": ensure_rgb(ref_image),
            "reference_mask": ensure_mask(ref_mask),
        }
        for name, image in shared.items():
            atomic_save(image, case_dir / f"{prefix}_{name}.png", **fast)
            link(case_dir / f"{prefix}_{name}.png", run_dir / f"{prefix}_{name}.png")
        processed = {
            "proc_source": (Image.fromarray(images["target"]), ".jpg"),
            "proc_core": (Image.fromarray(images["core"].astype(np.uint8) * 255), ".png"),
            "proc_reference": (Image.fromarray(images["reference"]), ".jpg"),
            "proc_ref_mask": (Image.fromarray(images["ref_mask"].astype(np.uint8) * 255), ".png"),
        }
        for name, (image, suffix) in processed.items():
            atomic_save(image, case_dir / f"{prefix}_{name}{suffix}", **({"quality": 92} if suffix == ".jpg" else fast))
        atomic_save(local, run_dir / f"{prefix}_output_local.png", **fast)
        atomic_save(raw_restored, run_dir / f"{prefix}_output_restored_raw.png", **fast)
        atomic_save(primary, run_dir / f"{prefix}_output.png", **fast)
        # fill_ratio.py enumerates cases by *_row.png; the row itself is not used.
        link(run_dir / f"{prefix}_output_local.png", run_dir / f"{prefix}_row.png")
        tmp = run_dir / f".{prefix}.probe.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, run_dir / f"{prefix}.probe.json")

    saver.submit(save)
    return meta


def item_done(out: Path, ckpt_key: str, case: dict, variant: str) -> bool:
    return (out / "runs" / ckpt_key / variant / f"{case['prefix']}.probe.json").exists()


def cmd_run(args):
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan["out_dir"])
    worker = plan["workers"][args.worker]
    ckpt_key = worker["checkpoint"]
    ckpt = plan["checkpoints"][ckpt_key]
    cases = {c["order"]: c for c in ckpt["cases"]}
    todo = [(cases[o], v) for o, v in worker["items"] if not item_done(out, ckpt_key, cases[o], v)]
    status_path = out / "status" / f"worker{args.worker:03d}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"worker": args.worker, "checkpoint": ckpt_key, "total": len(worker["items"]), "todo": len(todo)}
    status.update({"done": 0, "failed": [], "host": os.uname().nodename, "started": time.time()})

    def write_status(**extra):
        status.update(extra, updated=time.time())
        status_path.write_text(json.dumps(status, indent=1))

    write_status(state="loading")
    if not todo:
        write_status(state="finished")
        return
    trainer = setup_trainer(Path(ckpt["path"]))
    probe = ReferenceRopeProbe(trainer, ckpt["domain"])
    write_status(state="running", loaded=time.time())
    with ThreadPoolExecutor(max_workers=2) as saver:
        for case, variant in todo:
            try:
                meta = run_item(trainer, probe, case, variant, out, ckpt_key, ckpt["domain"], saver)
                status["done"] += 1
                print(f"[worker {args.worker}] {ckpt_key} {variant} {case['prefix']} {meta['sample_seconds']}s", flush=True)
            except Exception:
                traceback.print_exc()
                status["failed"].append([case["prefix"], variant, traceback.format_exc()[-2000:]])
            write_status(state="running")
            torch.cuda.empty_cache()
    write_status(state="finished" if not status["failed"] else "finished_with_errors")
    if status["failed"]:
        sys.exit(1)


# --------------------------------------------------------------------------- dynamic work queue
#
# Independent single-node jobs share the plan's run list through claim files on
# the shared disk, so any number of nodes (started whenever the project GPU cap
# allows) completes the full set. Each worker keeps one checkpoint loaded and
# switches only when its share of another checkpoint's remaining runs outweighs
# a reload.

CLAIM_STALE_SECONDS = 900
HEARTBEAT_SECONDS = 30
ITEM_SECONDS, LOAD_SECONDS = 80.0, 600.0


def _claim(path: Path, name: str, stale: bool = False) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"worker": name, "time": time.time()})
    if stale:
        tmp = path.with_name(f".{path.name}.{name}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)
        return True
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as stream:
        stream.write(payload)
    return True


class WorkQueue:
    def __init__(self, plan: dict, name: str):
        self.plan, self.name = plan, name
        self.out = Path(plan["out_dir"])
        self.items = {
            key: [(case, variant) for case in ck["cases"] for variant in plan["variants"]]
            for key, ck in plan["checkpoints"].items()
        }
        self.active_dir = self.out / "active"
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = None
        self._stop = threading.Event()
        threading.Thread(target=self._heartbeat, daemon=True).start()

    def _heartbeat(self):
        while not self._stop.is_set():
            self.beat()
            self._stop.wait(HEARTBEAT_SECONDS)

    def beat(self):
        path = self.active_dir / f"{self.name}.json"
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({"checkpoint": self.checkpoint, "time": time.time(), "host": os.uname().nodename}))
        os.replace(tmp, path)

    def stop(self):
        self._stop.set()
        (self.active_dir / f"{self.name}.json").unlink(missing_ok=True)

    def _listing(self, root: str, key: str, variant: str) -> set:
        try:
            return set(os.listdir(self.out / root / key / variant))
        except FileNotFoundError:
            return set()

    def remaining(self) -> dict:
        counts = {}
        for key, items in self.items.items():
            done, claimed = {}, {}
            for variant in self.plan["variants"]:
                done[variant] = self._listing("runs", key, variant)
                claimed[variant] = self._listing("claims", key, variant)
            counts[key] = sum(
                1
                for case, variant in items
                if f"{case['prefix']}.probe.json" not in done[variant] and case["prefix"] not in claimed[variant]
            )
        return counts

    def active(self) -> dict:
        counts = {key: 0 for key in self.items}
        now = time.time()
        for path in self.active_dir.glob("*.json"):
            if path.stem == self.name:
                continue
            try:
                info = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if now - info.get("time", 0) < 3 * HEARTBEAT_SECONDS and info.get("checkpoint") in counts:
                counts[info["checkpoint"]] += 1
        return counts

    def choose(self):
        """Keep the loaded checkpoint while it has unclaimed runs; otherwise pick the largest share."""
        remaining = self.remaining()
        if self.checkpoint is not None and remaining.get(self.checkpoint, 0) > 0:
            return self.checkpoint
        active = self.active()
        share = {key: remaining[key] / (active[key] + 1) for key in remaining if remaining[key] > 0}
        if not share:
            return None
        best = max(share, key=share.get)
        # A reload pays off only if this worker's expected share outlasts it.
        minimum = 2 if self.checkpoint is None else LOAD_SECONDS / ITEM_SECONDS
        return best if share[best] >= minimum else None

    def claim_next(self, key: str):
        for case, variant in self.items[key]:
            if item_done(self.out, key, case, variant):
                continue
            if _claim(self.out / "claims" / key / variant / case["prefix"], self.name):
                return case, variant
        return None

    def claim_stale(self, key: str):
        now = time.time()
        for case, variant in self.items[key]:
            path = self.out / "claims" / key / variant / case["prefix"]
            if item_done(self.out, key, case, variant):
                continue
            try:
                age = now - path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > CLAIM_STALE_SECONDS and _claim(path, self.name, stale=True):
                return case, variant
        return None


def cmd_work(args):
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan["out_dir"])
    time.sleep(args.stagger)
    queue = WorkQueue(plan, args.name)
    status_path = out / "status" / f"{args.name}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"worker": args.name, "host": os.uname().nodename, "started": time.time(), "done": 0, "failed": []}
    status["loads"] = []

    def write_status(**extra):
        status.update(extra, updated=time.time())
        status_path.write_text(json.dumps(status, indent=1))

    trainer = probe = None
    try:
        with ThreadPoolExecutor(max_workers=2) as saver:
            while True:
                key = queue.choose()
                if key is None:
                    break
                if key != queue.checkpoint:
                    queue.checkpoint = key
                    queue.beat()
                    write_status(state="loading", checkpoint=key)
                    trainer = probe = None
                    torch.cuda.empty_cache()
                    started = time.time()
                    trainer = setup_trainer(Path(plan["checkpoints"][key]["path"]))
                    probe = ReferenceRopeProbe(trainer, plan["checkpoints"][key]["domain"])
                    status["loads"].append([key, round(time.time() - started)])
                    write_status(state="running")
                while True:
                    item = queue.claim_next(key) or queue.claim_stale(key)
                    if item is None:
                        break
                    case, variant = item
                    try:
                        meta = run_item(trainer, probe, case, variant, out, key, plan["checkpoints"][key]["domain"], saver)
                        status["done"] += 1
                        print(f"[{args.name}] {key} {variant} {case['prefix']} {meta['sample_seconds']}s", flush=True)
                    except Exception:
                        traceback.print_exc()
                        status["failed"].append([key, case["prefix"], variant, traceback.format_exc()[-2000:]])
                    write_status(state="running")
                    torch.cuda.empty_cache()
    finally:
        queue.stop()
    write_status(state="finished" if not status["failed"] else "finished_with_errors")
    if status["failed"]:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--worker", type=int, required=True)
    r.set_defaults(func=cmd_run)
    w = sub.add_parser("work", help="dynamic worker: claim runs from the shared queue until none are left")
    w.add_argument("--plan", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--stagger", type=float, default=0.0, help="seconds to wait before choosing a checkpoint")
    w.set_defaults(func=cmd_work)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

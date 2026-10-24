#!/usr/bin/env python3
"""Evaluate insertion results: DINO-I↑ CLIP-I↑ PSNR↑ SSIM↑ LPIPS↓ FID↓.

Suites
------
- paperinsert_900  (our 900 benchmark; GT = tar_image when available)
- anyinsertion     (Insert Anything official; identity metrics always;
                    reconstruction metrics only when GT path exists)

Methods
-------
ours, anydoor, insert_anything, qwen_edit_2511, qwen_image_2_0, flux_kontext, a2edit_moe

Metric definitions (insertion-standard)
--------------------------------------
Identity (region crops):
  CLIP-I / DINO-I = cosine sim of embeddings between
      (ref object crop from src_image⊗src_mask) and
      (pred object crop from pred⊗tar_mask)

Reconstruction (full image, resized to GT size):
  PSNR / SSIM / LPIPS between pred and GT (tar_image).
  LPIPS uses alex backbone; SSIM is windowed greyscale; PSNR is RGB.

FID:
  Frechet distance between Inception-v3 pool3 features of GT set vs pred set
  (only cases with both GT and pred).

Usage
-----
  python scripts/eval_insertion_metrics.py \\
      --suites paperinsert_900,anyinsertion \\
      --methods ours,anydoor,insert_anything,qwen_edit_2511,qwen_image_2_0,flux_kontext,a2edit_moe \\
      --device cuda:0 \\
      --output-dir output/metrics

  # one-click wrapper:
  bash scripts/run_eval_metrics_all.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
EVAL = Path(
    "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval"
)
PI_ROOT = EVAL / "paper_insertion_benchmark_v1" / "paperinsert_900"
AI_ROOT = EVAL / "paper_insertion_benchmark_v1" / "anyinsertion_official"
PI_BASELINES = EVAL / "paper_insertion_benchmark_v1" / "baselines_infer"
AI_BASELINES = EVAL / "anyinsertion_official_baselines_infer"
OURS_ROOT = REPO / "output/inference/ours_ckpt_two_testsets"
OURS_PI = (
    OURS_ROOT
    / "paperinsert_900/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
)
OURS_AI = (
    OURS_ROOT
    / "anyinsertion_official/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
)

# Offline metric backbones (GPU nodes cannot reach huggingface.co / pytorch.org).
# Prepared under pretrained_weights/metrics_offline/.
METRICS_OFFLINE = Path(
    os.environ.get(
        "METRICS_OFFLINE_ROOT",
        "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
        "pretrained_weights/metrics_offline",
    )
)
DEFAULT_CLIP_PATH = METRICS_OFFLINE / "clip-vit-base-patch32"
DEFAULT_DINO_PATH = METRICS_OFFLINE / "dinov2-small"
DEFAULT_TORCH_HOME = METRICS_OFFLINE / "torch_home"
DEFAULT_LPIPS_ALEX = METRICS_OFFLINE / "lpips" / "alex.pth"


def _configure_offline_torch_home(torch_home: Path) -> None:
    """Force torchvision/LPIPS trunk downloads to local TORCH_HOME."""
    torch_home = Path(torch_home)
    ckpt = torch_home / "hub" / "checkpoints"
    if not (ckpt / "inception_v3_google-0cc3c7bd.pth").is_file():
        raise FileNotFoundError(
            f"Missing Inception weights under {ckpt}. "
            "Run scripts/prepare_metrics_offline_weights.sh first."
        )
    os.environ["TORCH_HOME"] = str(torch_home)
    # Also used by some torchvision versions
    os.environ.setdefault("TORCH_MODEL_ZOO", str(ckpt))

METHOD_ORDER = [
    "ours",
    "anydoor",
    "insert_anything",
    "qwen_edit_2511",
    "qwen_image_2_0",
    "flux_kontext",
    "a2edit_moe",
]

OURS_NAMES = ("output_restored_raw.png", "output.png", "output_local.png")


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


@dataclass
class Case:
    case_id: str
    suite: str
    major: str
    ref_image: Path
    ref_mask: Path
    tar_image: Path  # also GT when gt_available
    tar_mask: Path
    gt_available: bool
    results: dict[str, Path] = field(default_factory=dict)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    m = Image.open(path).convert("L")
    if size_hw is not None:
        h, w = size_hw
        if m.size != (w, h):
            m = m.resize((w, h), Image.NEAREST)
    arr = np.asarray(m, dtype=np.uint8)
    return (arr > 128).astype(np.uint8)


def _bbox(mask: np.ndarray, pad_ratio: float = 0.1) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    h, w = mask.shape
    ph = int((y2 - y1) * pad_ratio)
    pw = int((x2 - x1) * pad_ratio)
    y1 = max(0, y1 - ph)
    x1 = max(0, x1 - pw)
    y2 = min(h, y2 + ph)
    x2 = min(w, x2 + pw)
    if y2 <= y1 or x2 <= x1:
        return None
    return y1, y2, x1, x2


def _crop_object(image: np.ndarray, mask: np.ndarray, pad_ratio: float = 0.1) -> np.ndarray | None:
    box = _bbox(mask, pad_ratio=pad_ratio)
    if box is None:
        return None
    y1, y2, x1, x2 = box
    crop = image[y1:y2, x1:x2].copy()
    m = mask[y1:y2, x1:x2]
    # white background outside object (common for identity metrics)
    out = np.ones_like(crop) * 255
    out[m > 0] = crop[m > 0]
    return out


def _find_baseline(root: Path, method: str, case_id: str, major: str) -> Path | None:
    cands = [
        root / method / f"{case_id}.png",
        root / method / major / f"{case_id}.png",
        root / method / f"{case_id}.jpg",
    ]
    for p in cands:
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _find_ours(ours_root: Path, major: str, case_id: str) -> Path | None:
    if not ours_root.is_dir():
        return None
    major_dir = ours_root / major
    if not major_dir.is_dir():
        return None
    suffix = f"_{case_id}"
    for d in major_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name == case_id or d.name.endswith(suffix) or d.name.endswith(case_id):
            for name in OURS_NAMES:
                p = d / name
                if p.is_file() and p.stat().st_size > 0:
                    return p
    for name in OURS_NAMES:
        for p in major_dir.rglob(name):
            if case_id in p.parent.name and p.is_file() and p.stat().st_size > 0:
                return p
    return None


def load_manifest_suite(
    suite: str,
    data_root: Path,
    manifest: Path,
    baseline_root: Path,
    ours_root: Path,
    methods: list[str],
) -> list[Case]:
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest missing: {manifest}")
    cases: list[Case] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row["case_id"])
            major = str(row.get("major", ""))
            results: dict[str, Path] = {}
            for m in methods:
                if m == "ours":
                    p = _find_ours(ours_root, major, case_id)
                else:
                    p = _find_baseline(baseline_root, m, case_id, major)
                if p is not None:
                    results[m] = p
            cases.append(
                Case(
                    case_id=case_id,
                    suite=suite,
                    major=major,
                    ref_image=(data_root / row["src_image"]).resolve(),
                    ref_mask=(data_root / row["src_mask"]).resolve(),
                    tar_image=(data_root / row["tar_image"]).resolve(),
                    tar_mask=(data_root / row["tar_mask"]).resolve(),
                    gt_available=bool(row.get("gt_available", True)),
                    results=results,
                )
            )
    return cases


# ---------------------------------------------------------------------------
# Metrics backends
# ---------------------------------------------------------------------------


class IdentityEncoders:
    def __init__(
        self,
        device: torch.device,
        clip_path: Path | str = DEFAULT_CLIP_PATH,
        dino_path: Path | str = DEFAULT_DINO_PATH,
    ) -> None:
        self.device = device
        from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

        clip_path = Path(os.environ.get("CLIP_MODEL_PATH", str(clip_path)))
        dino_path = Path(os.environ.get("DINO_MODEL_PATH", str(dino_path)))
        if not clip_path.is_dir():
            raise FileNotFoundError(
                f"CLIP model dir not found: {clip_path}\n"
                "Prepare offline weights: bash scripts/prepare_metrics_offline_weights.sh"
            )
        if not dino_path.is_dir():
            raise FileNotFoundError(
                f"DINO model dir not found: {dino_path}\n"
                "Prepare offline weights: bash scripts/prepare_metrics_offline_weights.sh"
            )

        # Never hit the network on air-gapped GPU nodes.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        print(f"[metrics] loading CLIP from {clip_path} …", flush=True)
        self.clip_proc = CLIPProcessor.from_pretrained(
            str(clip_path), local_files_only=True
        )
        self.clip = (
            CLIPModel.from_pretrained(str(clip_path), local_files_only=True)
            .to(device)
            .eval()
        )

        print(f"[metrics] loading DINOv2 from {dino_path} …", flush=True)
        self.dino_proc = AutoImageProcessor.from_pretrained(
            str(dino_path), local_files_only=True
        )
        self.dino = (
            AutoModel.from_pretrained(str(dino_path), local_files_only=True)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def clip_embed(self, images: list[Image.Image]) -> torch.Tensor:
        # batch
        inputs = self.clip_proc(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "text"}
        # CLIPProcessor may include text keys None
        pixel = inputs.get("pixel_values")
        feats = self.clip.get_image_features(pixel_values=pixel)
        return F.normalize(feats.float(), dim=-1)

    @torch.inference_mode()
    def dino_embed(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.dino_proc(images=images, return_tensors="pt")
        pixel = inputs["pixel_values"].to(self.device)
        out = self.dino(pixel_values=pixel)
        # CLS token
        feats = out.last_hidden_state[:, 0]
        return F.normalize(feats.float(), dim=-1)


class LPIPSMetric:
    def __init__(
        self,
        device: torch.device,
        torch_home: Path | str = DEFAULT_TORCH_HOME,
        lpips_alex: Path | str = DEFAULT_LPIPS_ALEX,
    ) -> None:
        import lpips

        self.device = device
        _configure_offline_torch_home(Path(os.environ.get("TORCH_HOME", str(torch_home))))
        print("[metrics] loading LPIPS(alex) …", flush=True)
        # Prefer local linear weights if present (lpips still needs alex trunk via TORCH_HOME).
        lpips_alex = Path(os.environ.get("LPIPS_ALEX_PATH", str(lpips_alex)))
        self.fn = lpips.LPIPS(net="alex").to(device).eval()
        if lpips_alex.is_file():
            # Optional: load official linear head if package weights missing on node.
            try:
                state = torch.load(lpips_alex, map_location=device)
                # Only apply if shapes match current module linear layers
                own = self.fn.state_dict()
                filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
                if filtered:
                    own.update(filtered)
                    self.fn.load_state_dict(own, strict=False)
                    print(f"[metrics] LPIPS linear weights from {lpips_alex}", flush=True)
            except Exception as e:
                print(f"[metrics] LPIPS linear load skipped: {e}", flush=True)

    @torch.inference_mode()
    def __call__(self, pred: np.ndarray, gt: np.ndarray) -> float:
        # resize pred to gt
        if pred.shape[:2] != gt.shape[:2]:
            pred = np.asarray(
                Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
            )
        t_pred = torch.from_numpy(np.ascontiguousarray(pred)).permute(2, 0, 1).float() / 127.5 - 1.0
        t_gt = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float() / 127.5 - 1.0
        # LPIPS likes ~64+ resolution
        t_pred = t_pred.unsqueeze(0).to(self.device)
        t_gt = t_gt.unsqueeze(0).to(self.device)
        # optional downscale for speed/memory if huge
        h, w = t_pred.shape[-2:]
        max_side = 512
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
            t_pred = F.interpolate(t_pred, size=(nh, nw), mode="bilinear", align_corners=False)
            t_gt = F.interpolate(t_gt, size=(nh, nw), mode="bilinear", align_corners=False)
        val = self.fn(t_pred, t_gt)
        return float(val.item())


class FIDAccumulator:
    """Online mean/cov of Inception-v3 pool3 features."""

    _shared_net: nn.Module | None = None
    _shared_device: torch.device | None = None
    _shared_tf = None

    def __init__(self, device: torch.device) -> None:
        self.device = device
        if (
            FIDAccumulator._shared_net is None
            or FIDAccumulator._shared_device != device
        ):
            torch_home = Path(os.environ.get("TORCH_HOME", str(DEFAULT_TORCH_HOME)))
            _configure_offline_torch_home(torch_home)
            print(f"[metrics] loading Inception-v3 from TORCH_HOME={torch_home} …", flush=True)
            # Offline: load state_dict from local file (no network).
            ckpt = torch_home / "hub" / "checkpoints" / "inception_v3_google-0cc3c7bd.pth"
            try:
                net = models.inception_v3(weights=None, aux_logits=True, init_weights=False)
            except TypeError:
                net = models.inception_v3(weights=None, aux_logits=True)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = net.load_state_dict(state, strict=False)
            if missing:
                print(f"[metrics] inception missing keys (ok if aux): {len(missing)}", flush=True)
            net.fc = nn.Identity()
            FIDAccumulator._shared_net = net.to(device).eval()
            FIDAccumulator._shared_device = device
            FIDAccumulator._shared_tf = transforms.Compose(
                [
                    transforms.Resize((299, 299), antialias=True),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        self.net = FIDAccumulator._shared_net
        self.tf = FIDAccumulator._shared_tf
        self.reset()

    def reset(self) -> None:
        self.sum = None
        self.sum_sq = None
        self.n = 0
        self.feats: list[np.ndarray] = []

    @torch.inference_mode()
    def add(self, images: list[Image.Image]) -> None:
        if not images:
            return
        batch = torch.stack([self.tf(im) for im in images], dim=0).to(self.device)
        # inception expects NCHW
        feats = self.net(batch).float()
        # handle tuple if any
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        arr = feats.cpu().numpy()
        self.feats.append(arr)
        self.n += arr.shape[0]

    def _stack(self) -> np.ndarray:
        if not self.feats:
            return np.zeros((0, 2048), dtype=np.float64)
        return np.concatenate(self.feats, axis=0).astype(np.float64)

    @staticmethod
    def frechet(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
        from numpy.linalg import eigvals

        diff = mu1 - mu2
        # product might be almost singular
        covmean_sq = sigma1.dot(sigma2)
        # sqrtm via eigen of product (more stable for moderate d)
        # Use scipy if available else eigenvalue trick
        try:
            from scipy import linalg

            covmean, _ = linalg.sqrtm(covmean_sq, disp=False)
            if not np.isfinite(covmean).all():
                offset = np.eye(sigma1.shape[0]) * eps
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
        except Exception:
            # fallback diagonal approx
            covmean = np.diag(np.sqrt(np.maximum(np.diag(sigma1) * np.diag(sigma2), 0)))
        tr = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        return float(diff.dot(diff) + tr)

    def stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        x = self._stack()
        if x.shape[0] < 2:
            return None
        mu = np.mean(x, axis=0)
        sigma = np.cov(x, rowvar=False)
        return mu, sigma


def psnr_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape[:2] != gt.shape[:2]:
        pred = np.asarray(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
        )
    pred_f = pred.astype(np.float64)
    gt_f = gt.astype(np.float64)
    mse = np.mean((pred_f - gt_f) ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def ssim_gray(pred: np.ndarray, gt: np.ndarray) -> float:
    """Simple SSIM on luminance (Gaussian approx via average pool)."""
    if pred.shape[:2] != gt.shape[:2]:
        pred = np.asarray(
            Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BICUBIC)
        )
    # to gray
    def gray(x):
        x = x.astype(np.float64)
        return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]

    x = gray(pred)
    y = gray(gt)
    h, w = x.shape
    max_side = 512
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        nh, nw = max(16, int(h * scale)), max(16, int(w * scale))
        x = np.asarray(Image.fromarray(x).resize((nw, nh), Image.BICUBIC), dtype=np.float64)
        y = np.asarray(Image.fromarray(y).resize((nw, nh), Image.BICUBIC), dtype=np.float64)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k = 11
    pad = k // 2
    xt = torch.from_numpy(x)[None, None].float()
    yt = torch.from_numpy(y)[None, None].float()
    ker = torch.ones(1, 1, k, k) / (k * k)
    mu_x = F.conv2d(xt, ker, padding=pad)
    mu_y = F.conv2d(yt, ker, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(xt * xt, ker, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(yt * yt, ker, padding=pad) - mu_y2
    sigma_xy = F.conv2d(xt * yt, ker, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return float(ssim_map.mean().item())


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


@dataclass
class MethodStats:
    n_identity: int = 0
    n_recon: int = 0
    clip_i: list[float] = field(default_factory=list)
    dino_i: list[float] = field(default_factory=list)
    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)
    lpips: list[float] = field(default_factory=list)
    fid_pred: FIDAccumulator | None = None
    fid_gt: FIDAccumulator | None = None
    missing: int = 0
    errors: int = 0

    def summary(self) -> dict[str, Any]:
        def mean(xs):
            return float(np.mean(xs)) if xs else None

        out = {
            "n_identity": self.n_identity,
            "n_recon": self.n_recon,
            "missing": self.missing,
            "errors": self.errors,
            "DINO-I": mean(self.dino_i),
            "CLIP-I": mean(self.clip_i),
            "PSNR": mean(self.psnr),
            "SSIM": mean(self.ssim),
            "LPIPS": mean(self.lpips),
            "FID": None,
        }
        if self.fid_pred is not None and self.fid_gt is not None:
            sp = self.fid_pred.stats()
            sg = self.fid_gt.stats()
            if sp is not None and sg is not None:
                out["FID"] = FIDAccumulator.frechet(sg[0], sg[1], sp[0], sp[1])
        return out


def evaluate_suite(
    suite: str,
    cases: list[Case],
    methods: list[str],
    device: torch.device,
    batch_size: int = 16,
    limit: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    shard_dir: Path | None = None,
) -> dict[str, Any]:
    if limit is not None:
        cases = cases[:limit]
    # Deterministic shard across ranks
    cases = cases[rank::world_size]
    print(
        f"[{suite}] rank={rank}/{world_size} local_cases={len(cases)} device={device}",
        flush=True,
    )
    if not cases:
        empty = {
            "suite": suite,
            "rank": rank,
            "world_size": world_size,
            "n_cases_local": 0,
            "methods": {
                m: {
                    "n_identity": 0,
                    "n_recon": 0,
                    "missing": 0,
                    "errors": 0,
                    "DINO-I": None,
                    "CLIP-I": None,
                    "PSNR": None,
                    "SSIM": None,
                    "LPIPS": None,
                    "FID": None,
                    "clip_i": [],
                    "dino_i": [],
                    "psnr": [],
                    "ssim": [],
                    "lpips": [],
                }
                for m in methods
            },
            "per_case": [],
            "seconds": 0.0,
        }
        return empty

    enc = IdentityEncoders(device)
    lpips_m = LPIPSMetric(device)  # uses TORCH_HOME / LPIPS_ALEX_PATH env

    stats: dict[str, MethodStats] = {m: MethodStats() for m in methods}
    for m in methods:
        stats[m].fid_pred = FIDAccumulator(device)
        stats[m].fid_gt = FIDAccumulator(device)

    per_case: list[dict[str, Any]] = []

    t0 = time.time()
    for i, case in enumerate(cases):
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "major": case.major,
            "gt_available": case.gt_available,
            "methods": {},
        }
        try:
            ref = _load_rgb(case.ref_image)
            ref_mask = _load_mask(case.ref_mask, ref.shape[:2])
            tar = _load_rgb(case.tar_image)
            tar_mask = _load_mask(case.tar_mask, tar.shape[:2])
            ref_crop = _crop_object(ref, ref_mask)
            if ref_crop is None:
                raise RuntimeError("empty ref mask")
            ref_pil = Image.fromarray(ref_crop)
        except Exception as e:
            row["error"] = f"load inputs: {e}"
            per_case.append(row)
            continue

        for m in methods:
            pred_path = case.results.get(m)
            if pred_path is None:
                stats[m].missing += 1
                row["methods"][m] = {"status": "missing"}
                continue
            try:
                pred = _load_rgb(pred_path)
                if pred.shape[:2] != tar.shape[:2]:
                    pred_r = np.asarray(
                        Image.fromarray(pred).resize(
                            (tar.shape[1], tar.shape[0]), Image.BICUBIC
                        )
                    )
                else:
                    pred_r = pred
                pred_crop = _crop_object(pred_r, tar_mask)
                if pred_crop is None:
                    raise RuntimeError("empty tar mask on pred")
                pred_pil = Image.fromarray(pred_crop)

                clip_e = enc.clip_embed([ref_pil, pred_pil])
                dino_e = enc.dino_embed([ref_pil, pred_pil])
                clip_i = float((clip_e[0] * clip_e[1]).sum().item())
                dino_i = float((dino_e[0] * dino_e[1]).sum().item())
                stats[m].clip_i.append(clip_i)
                stats[m].dino_i.append(dino_i)
                stats[m].n_identity += 1

                mrow: dict[str, Any] = {
                    "status": "ok",
                    "pred": str(pred_path),
                    "CLIP-I": clip_i,
                    "DINO-I": dino_i,
                }

                if case.gt_available and case.tar_image.is_file():
                    gt = tar
                    psnr_v = psnr_rgb(pred_r, gt)
                    ssim_v = ssim_gray(pred_r, gt)
                    lpips_v = lpips_m(pred_r, gt)
                    stats[m].psnr.append(psnr_v)
                    stats[m].ssim.append(ssim_v)
                    stats[m].lpips.append(lpips_v)
                    stats[m].n_recon += 1
                    stats[m].fid_pred.add([Image.fromarray(pred_r)])
                    stats[m].fid_gt.add([Image.fromarray(gt)])
                    mrow.update({"PSNR": psnr_v, "SSIM": ssim_v, "LPIPS": lpips_v})
                row["methods"][m] = mrow
            except Exception as e:
                stats[m].errors += 1
                row["methods"][m] = {"status": "error", "error": str(e)}

        per_case.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(cases):
            dt = time.time() - t0
            print(
                f"[{suite}:r{rank}] {i+1}/{len(cases)}  "
                f"({dt:.1f}s, {(i+1)/max(dt,1e-6):.2f} it/s)",
                flush=True,
            )

    # Local summary + raw lists for multi-rank merge
    methods_out: dict[str, Any] = {}
    for m in methods:
        s = stats[m]
        summ = s.summary()
        summ.update(
            {
                "clip_i": s.clip_i,
                "dino_i": s.dino_i,
                "psnr": s.psnr,
                "ssim": s.ssim,
                "lpips": s.lpips,
            }
        )
        methods_out[m] = summ
        # FID features for merge
        if shard_dir is not None:
            shard_dir.mkdir(parents=True, exist_ok=True)
            if s.fid_pred is not None:
                np.save(
                    shard_dir / f"{suite}__{m}__r{rank:02d}__fid_pred.npy",
                    s.fid_pred._stack(),
                )
            if s.fid_gt is not None:
                np.save(
                    shard_dir / f"{suite}__{m}__r{rank:02d}__fid_gt.npy",
                    s.fid_gt._stack(),
                )

    return {
        "suite": suite,
        "rank": rank,
        "world_size": world_size,
        "n_cases_local": len(cases),
        "methods": methods_out,
        "per_case": per_case,
        "seconds": time.time() - t0,
    }


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def merge_rank_shards(
    output_dir: Path,
    suites: list[str],
    methods: list[str],
    world_size: int,
    skip_per_case: bool = False,
) -> dict[str, Any]:
    """Merge rank partial JSON + FID .npy into final tables."""
    shard_dir = output_dir / "shards"
    all_results: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "world_size": world_size,
        "methods": methods,
        "suites": {},
    }

    for suite in suites:
        merged_methods: dict[str, Any] = {}
        per_case: list[dict[str, Any]] = []
        n_local_sum = 0
        seconds = 0.0
        for rank in range(world_size):
            path = shard_dir / f"{suite}__rank{rank:02d}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing shard: {path}")
            part = json.loads(path.read_text(encoding="utf-8"))
            n_local_sum += int(part.get("n_cases_local", 0))
            seconds = max(seconds, float(part.get("seconds", 0.0)))
            per_case.extend(part.get("per_case") or [])
            for m in methods:
                ms = part["methods"][m]
                bucket = merged_methods.setdefault(
                    m,
                    {
                        "n_identity": 0,
                        "n_recon": 0,
                        "missing": 0,
                        "errors": 0,
                        "clip_i": [],
                        "dino_i": [],
                        "psnr": [],
                        "ssim": [],
                        "lpips": [],
                    },
                )
                for k in ("n_identity", "n_recon", "missing", "errors"):
                    bucket[k] += int(ms.get(k, 0))
                for k in ("clip_i", "dino_i", "psnr", "ssim", "lpips"):
                    bucket[k].extend(ms.get(k) or [])

        # FID: concat features across ranks
        for m in methods:
            pred_feats = []
            gt_feats = []
            for rank in range(world_size):
                pp = shard_dir / f"{suite}__{m}__r{rank:02d}__fid_pred.npy"
                gp = shard_dir / f"{suite}__{m}__r{rank:02d}__fid_gt.npy"
                if pp.is_file():
                    arr = np.load(pp)
                    if arr.size:
                        pred_feats.append(arr)
                if gp.is_file():
                    arr = np.load(gp)
                    if arr.size:
                        gt_feats.append(arr)
            fid_v = None
            if pred_feats and gt_feats:
                pred = np.concatenate(pred_feats, axis=0).astype(np.float64)
                gt = np.concatenate(gt_feats, axis=0).astype(np.float64)
                if pred.shape[0] >= 2 and gt.shape[0] >= 2:
                    mu_p, sig_p = np.mean(pred, 0), np.cov(pred, rowvar=False)
                    mu_g, sig_g = np.mean(gt, 0), np.cov(gt, rowvar=False)
                    fid_v = FIDAccumulator.frechet(mu_g, sig_g, mu_p, sig_p)
            b = merged_methods[m]
            b["DINO-I"] = _mean(b["dino_i"])
            b["CLIP-I"] = _mean(b["clip_i"])
            b["PSNR"] = _mean(b["psnr"])
            b["SSIM"] = _mean(b["ssim"])
            b["LPIPS"] = _mean(b["lpips"])
            b["FID"] = fid_v
            # drop raw lists from summary
            for k in ("clip_i", "dino_i", "psnr", "ssim", "lpips"):
                b.pop(k, None)

        suite_res = {
            "suite": suite,
            "n_cases": n_local_sum,
            "methods": merged_methods,
            "seconds_max_rank": seconds,
            "world_size": world_size,
        }
        all_results["suites"][suite] = suite_res

        suite_path = output_dir / f"{suite}_metrics.json"
        payload = dict(suite_res)
        if not skip_per_case:
            # stable order by case_id
            payload["per_case"] = sorted(per_case, key=lambda r: str(r.get("case_id", "")))
        suite_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[merge] wrote {suite_path}", flush=True)

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = format_markdown(all_results)
    md_path = output_dir / "metrics_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    print(f"[merge] wrote {summary_path}", flush=True)
    print(f"[merge] wrote {md_path}", flush=True)
    return all_results


def format_markdown(all_results: dict[str, Any]) -> str:
    lines = [
        "# Insertion Metrics",
        "",
        "Higher is better: **DINO-I↑ CLIP-I↑ PSNR↑ SSIM↑**. Lower is better: **LPIPS↓ FID↓**.",
        "",
    ]
    for suite, res in all_results["suites"].items():
        lines.append(f"## {suite} (n={res['n_cases']})")
        lines.append("")
        header = [
            "method",
            "n_id",
            "n_recon",
            "DINO-I↑",
            "CLIP-I↑",
            "PSNR↑",
            "SSIM↑",
            "LPIPS↓",
            "FID↓",
            "miss",
            "err",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for m, s in res["methods"].items():

            def fmt(v, nd=4):
                if v is None:
                    return "—"
                if isinstance(v, float):
                    return f"{v:.{nd}f}"
                return str(v)

            row = [
                m,
                str(s["n_identity"]),
                str(s["n_recon"]),
                fmt(s["DINO-I"]),
                fmt(s["CLIP-I"]),
                fmt(s["PSNR"], 2),
                fmt(s["SSIM"]),
                fmt(s["LPIPS"]),
                fmt(s["FID"], 2),
                str(s["missing"]),
                str(s["errors"]),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--suites",
        type=str,
        default="paperinsert_900,anyinsertion",
        help="Comma list: paperinsert_900,anyinsertion",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(METHOD_ORDER),
        help="Comma list of methods",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=None, help="Cap cases per suite (debug)")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "output" / "metrics",
    )
    p.add_argument("--skip-per-case", action="store_true", help="Do not write huge per-case json")
    p.add_argument("--rank", type=int, default=int(__import__("os").environ.get("RANK", "0")))
    p.add_argument(
        "--world-size",
        type=int,
        default=int(__import__("os").environ.get("WORLD_SIZE", "1")),
    )
    p.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge shards under output-dir/shards (after multi-GPU workers finish).",
    )
    p.add_argument(
        "--clip-path",
        type=Path,
        default=Path(os.environ.get("CLIP_MODEL_PATH", str(DEFAULT_CLIP_PATH))),
    )
    p.add_argument(
        "--dino-path",
        type=Path,
        default=Path(os.environ.get("DINO_MODEL_PATH", str(DEFAULT_DINO_PATH))),
    )
    p.add_argument(
        "--torch-home",
        type=Path,
        default=Path(os.environ.get("TORCH_HOME", str(DEFAULT_TORCH_HOME))),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Offline env as early as possible (before any HF/torch hub touch).
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CLIP_MODEL_PATH"] = str(args.clip_path)
    os.environ["DINO_MODEL_PATH"] = str(args.dino_path)
    _configure_offline_torch_home(args.torch_home)
    print(f"[offline] CLIP={args.clip_path}", flush=True)
    print(f"[offline] DINO={args.dino_path}", flush=True)
    print(f"[offline] TORCH_HOME={args.torch_home}", flush=True)

    if args.merge_only:
        merge_rank_shards(
            output_dir=args.output_dir,
            suites=suites,
            methods=methods,
            world_size=args.world_size,
            skip_per_case=args.skip_per_case,
        )
        return 0

    # When launched with CUDA_VISIBLE_DEVICES=<one gpu>, always use cuda:0 inside.
    if torch.cuda.is_available():
        device = torch.device(args.device if args.device.startswith("cuda") else "cuda:0")
        # Prefer local cuda:0 under single-visible-GPU launches.
        if "CUDA_VISIBLE_DEVICES" in __import__("os").environ:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("[warn] CUDA not available; running on CPU (slow)", flush=True)

    # Speed knobs for fat GPUs
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    suite_loaders = {
        "paperinsert_900": lambda: load_manifest_suite(
            "paperinsert_900",
            PI_ROOT,
            PI_ROOT / "manifests/paperinsert_900.jsonl",
            PI_BASELINES,
            OURS_PI,
            methods,
        ),
        "anyinsertion": lambda: load_manifest_suite(
            "anyinsertion",
            AI_ROOT,
            AI_ROOT / "manifests/anyinsertion_official.jsonl",
            AI_BASELINES,
            OURS_AI,
            methods,
        ),
    }

    rank = int(args.rank)
    world_size = int(args.world_size)
    print(
        f"[worker] rank={rank}/{world_size} device={device} "
        f"suites={suites} methods={methods}",
        flush=True,
    )

    for suite in suites:
        if suite not in suite_loaders:
            raise SystemExit(f"unknown suite: {suite}")
        print(f"\n======== suite={suite} rank={rank} ========", flush=True)
        cases = suite_loaders[suite]()
        cov = {m: sum(1 for c in cases if m in c.results) for m in methods}
        print(f"[coverage global] {cov}", flush=True)
        res = evaluate_suite(
            suite=suite,
            cases=cases,
            methods=methods,
            device=device,
            limit=args.limit,
            rank=rank,
            world_size=world_size,
            shard_dir=shard_dir,
        )
        shard_path = shard_dir / f"{suite}__rank{rank:02d}.json"
        # Keep raw metric lists for merge; drop per-case if requested
        if args.skip_per_case:
            res = {k: v for k, v in res.items() if k != "per_case"}
            res["per_case"] = []
        shard_path.write_text(json.dumps(res, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[wrote shard] {shard_path}", flush=True)

    # Single-process convenience: auto-merge
    if world_size == 1:
        merge_rank_shards(
            output_dir=args.output_dir,
            suites=suites,
            methods=methods,
            world_size=1,
            skip_per_case=args.skip_per_case,
        )
    else:
        print(
            f"[rank {rank}] done. After all ranks finish, run merge "
            f"(run_eval_metrics_all.sh does this automatically).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

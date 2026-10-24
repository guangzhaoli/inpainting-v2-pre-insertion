#!/usr/bin/env python3
"""Inference-only probe: does InsertFuse Teacher v2 copy the reference's lighting or adapt to the scene?

Only the reference object's pixels are changed (the white background stays white),
by a known gain in linear RGB:
  clean        unchanged
  clean_seed2  unchanged, seed + 1000 (seed-noise control)
  warm / cool  white balance along the Planckian locus, +100 / -100 mired from 6500 K
               (gains normalised to geometric mean 1: chroma only)
  ev_plus / ev_minus   exposure +1 / -1 EV
  warm_dim     warm + -1 EV
Every variant of a case uses the same seed, the GT mask as the condition (the
expanded_soft anchoring adds the 32 px interaction band) and the validation
sampler settings (40 steps, true_cfg_scale 2.5).

Per run the worker stores the mean log-linear-RGB of the output inside the GT
object mask (eroded 3 px, local crop space) and of the reference object (eroded
3 px). The leak coefficient <dOut, g> / |g|^2 (g = effective log gain applied to
the reference, after clipping) is computed by scripts/probe_lighting_leak_report.py.

  python scripts/probe_lighting_leak.py plan --out DIR
  python scripts/probe_lighting_leak.py work --plan DIR/plan.json --name n0-g0
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DATA = Path("/inspire/hdd3/project/agent-driven-world-model/250041225002/qwen-inpainting/a2edit/data")
PROJECT = Path("/inspire/hdd3/project/agent-driven-world-model/250041225002/qwen-inpainting/inpainting-v2-final-architecture")
RUNS = PROJECT / "output" / "teacher_v2_runs"
NEW = RUNS / "20261020_122256/training/insertfuse_teacher_v2_animal/v0"
OLD = RUNS / "20261014_135247/training/insertfuse_teacher_v2_animal/v2"
CHECKPOINTS = {
    "animal_new_s250": ("animal", NEW / "checkpoint-0-250"),
    "animal_new_s750": ("animal", NEW / "checkpoint-0-750"),
    "animal_new_s1500": ("animal", NEW / "checkpoint-1-1500"),
    "animal_old_s250": ("animal", OLD / "checkpoint-0-250"),
    "animal_old_s750": ("animal", OLD / "checkpoint-0-750"),
    "animal_old_s1500": ("animal", OLD / "checkpoint-0-1500"),
    "human_s1500": ("human", RUNS / "20261015_194628/training/insertfuse_teacher_v2_human/v0/checkpoint-0-1500"),
    "general_s1500": (
        "general_object",
        RUNS / "20261015_194628/training/insertfuse_teacher_v2_general_object/v0/checkpoint-0-1500",
    ),
    "base": ("all", None),
}
BASE_CONFIG = NEW / "train_config.yaml"
BASE_NOUN = {"cat": "cat", "dog": "dog", "person": "person", "person_hair": "hair", "others": "object"}
DOMAINS = ("animal", "human", "general_object")
CASES_PER_DOMAIN = 32
SAMPLE_SEED = 20261024
SEED_BASE = 42
SEED2_OFFSET = 1000
NUM_STEPS = 40
TRUE_CFG = 2.5
ERODE_PX = 3
LOG_EPS = 1e-3
VARIANTS = ("clean", "clean_seed2", "warm", "cool", "ev_plus", "ev_minus", "warm_dim")
REF_KELVIN = 6500.0
MIRED_STEP = 100.0
SHM = Path(os.environ.get("LIGHT_LEAK_SHM", "/dev/shm/light_leak"))


# --------------------------------------------------------------------------- colour


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def planckian_linear_rgb(kelvin: float) -> np.ndarray:
    """Linear sRGB of the Planckian locus point (Kim et al. 2002 cubic fit, 1667-25000 K), Y = 1."""
    t = float(kelvin)
    if not 1667 <= t <= 25000:
        raise ValueError(f"{t} K outside the 1667-25000 K fit")
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    xyz = np.array([x / y, 1.0, (1 - x - y) / y])
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    return m @ xyz


def white_balance_gains(mired_shift: float) -> np.ndarray:
    """Per-channel gains moving a 6500 K illuminant by ``mired_shift`` along the locus; geometric mean 1."""
    kelvin = 1e6 / (1e6 / REF_KELVIN + mired_shift)
    gains = planckian_linear_rgb(kelvin) / planckian_linear_rgb(REF_KELVIN)
    return gains / np.exp(np.log(gains).mean())


def variant_gains(variant: str) -> np.ndarray:
    ones = np.ones(3)
    return {
        "clean": ones,
        "clean_seed2": ones,
        "warm": white_balance_gains(+MIRED_STEP),
        "cool": white_balance_gains(-MIRED_STEP),
        "ev_plus": ones * 2.0,
        "ev_minus": ones * 0.5,
        "warm_dim": white_balance_gains(+MIRED_STEP) * 0.5,
    }[variant]


def perturb_reference(image: Image.Image, mask: Image.Image, variant: str) -> Image.Image:
    """Apply the variant's linear-RGB gain to the object pixels (mask > 127) only."""
    gains = variant_gains(variant)
    if np.allclose(gains, 1.0):
        return image
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    keep = binary_mask(mask, image.size)
    out = linear_to_srgb(srgb_to_linear(rgb) * gains)
    out = np.where(keep[..., None], np.round(out * 255.0), rgb * 255.0).astype(np.uint8)
    return Image.fromarray(out)


def binary_mask(mask: Image.Image, size) -> np.ndarray:
    mask = mask.convert("L")
    if mask.size != tuple(size):
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def linear_to_lab(lin: np.ndarray) -> np.ndarray:
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (kL = kC = kH = 1), broadcast over leading axes."""
    l1, a1, b1 = np.moveaxis(lab1, -1, 0)
    l2, a2, b2 = np.moveaxis(lab2, -1, 0)
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dlp, dcp = l2 - l1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dhp = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh) / 2)
    lp_bar, cp_bar = (l1 + l2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    hp_bar = np.where(
        c1p * c2p == 0, h_sum,
        np.where(np.abs(h1p - h2p) <= 180, h_sum / 2, np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)),
    )  # fmt: skip
    t = (
        1 - 0.17 * np.cos(np.radians(hp_bar - 30)) + 0.24 * np.cos(np.radians(2 * hp_bar))
        + 0.32 * np.cos(np.radians(3 * hp_bar + 6)) - 0.20 * np.cos(np.radians(4 * hp_bar - 63))
    )  # fmt: skip
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7))
    sl = 1 + 0.015 * (lp_bar - 50) ** 2 / np.sqrt(20 + (lp_bar - 50) ** 2)
    sc, sh = 1 + 0.045 * cp_bar, 1 + 0.015 * cp_bar * t
    rt = -np.sin(np.radians(2 * d_theta)) * rc
    return np.sqrt((dlp / sl) ** 2 + (dcp / sc) ** 2 + (dhp / sh) ** 2 + rt * (dcp / sc) * (dhp / sh))


def region_stats(image: Image.Image, region: np.ndarray) -> dict:
    lin = srgb_to_linear(np.asarray(image.convert("RGB"), dtype=np.float64)[region] / 255.0)
    mean_lin = lin.mean(0)
    return {
        "n_px": int(region.sum()),
        "mean_log_rgb": np.log(lin + LOG_EPS).mean(0).tolist(),
        "mean_lin_rgb": mean_lin.tolist(),
        "mean_lab": linear_to_lab(mean_lin).tolist(),
        "clip_hi_frac": float((lin.max(1) >= 0.999).mean()),
        "_lab": linear_to_lab(lin),
    }


def public(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- plan


def case_prefix(domain: str, order: int, category: str, stem: str) -> str:
    def safe(value):
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value))[:60] or "sample"

    return f"{domain}-{order:02d}-{safe(category)}-{safe(stem)}"


def load_cases(domain: str) -> list[dict]:
    rows = list(csv.DictReader(open(PROJECT / "manifests" / f"insert_anything_teacher_{domain}.csv")))
    random.Random(f"{SAMPLE_SEED}-{domain}").shuffle(rows)
    cases = []
    for row in rows:
        paths = {k: str(DATA / row[k]) for k in ("ref_image", "ref_mask", "tar_image", "tar_mask")}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        masks_ok = True
        for key in ("ref_mask", "tar_mask"):
            with Image.open(paths[key]) as m:
                masks_ok &= erode(np.asarray(m.convert("L")) > 127, ERODE_PX).sum() >= 400
        if not masks_ok:
            continue
        order = len(cases)
        cases.append(
            {
                "domain": domain,
                "order": order,
                "category": row["category"],
                "stem": str(row["stem"]),
                "prompt": row.get("prompt") or "",
                "seed": SEED_BASE + order,
                "prefix": case_prefix(domain, order, row["category"], row["stem"]),
                **paths,
            }
        )
        if len(cases) == CASES_PER_DOMAIN:
            break
    return cases


def cmd_plan(args):
    out = Path(args.out).resolve()
    cases = {domain: load_cases(domain) for domain in DOMAINS}
    checkpoints = {}
    for key, (domain, path) in CHECKPOINTS.items():
        if path is not None and not (path / "pytorch_lora_weights.safetensors").exists():
            raise FileNotFoundError(path)
        members = [c for d in DOMAINS for c in cases[d]] if domain == "all" else cases[domain]
        checkpoints[key] = {"domain": domain, "path": None if path is None else str(path), "cases": members}
    plan = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "out_dir": str(out),
        "variants": list(VARIANTS),
        "gains": {v: variant_gains(v).tolist() for v in VARIANTS},
        "num_inference_steps": NUM_STEPS,
        "true_cfg_scale": TRUE_CFG,
        "seed_rule": f"{SEED_BASE} + order within domain (clean_seed2: +{SEED2_OFFSET})",
        "erode_px": ERODE_PX,
        "base_model": {
            "config": str(BASE_CONFIG),
            "changes": "no LoRA, mask_control off (geometry tokenizer is trained with the LoRA), reference_cfg off; "
            "text prompt 'Place the <noun> from Picture 2 into the black region of Picture 1.' with text CFG "
            "(negative ' ') at the same true_cfg_scale; same expanded_soft latent anchoring",
        },
        "checkpoints": checkpoints,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    total = sum(len(ck["cases"]) for ck in checkpoints.values()) * len(VARIANTS)
    print(f"{total} runs; cases per domain {[len(cases[d]) for d in DOMAINS]}")


# --------------------------------------------------------------------------- model


def read_rgb(path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def read_mask(path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("L").copy()


def _locked_copy(sources: list[Path], dest_dir: Path):
    """Copy files into a /dev/shm directory once per node (flock serialises the 8 local workers)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for src in sources:
            dst = dest_dir / src.name
            if not dst.exists():
                tmp = dest_dir / f".{src.name}.tmp"
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)


def staged_model_root() -> str:
    model = SHM / "model"
    if (model / ".complete").exists():
        return str(model)
    return "/inspire/hdd3/project/agent-driven-world-model/250041225002/qwen-inpainting/pretrained_weights/Qwen-Image-Edit-2511"


def setup_trainer(key: str, ckpt: dict):
    import yaml

    from src.data.config import DeviceConfig, load_config_from_yaml
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer

    model_root = staged_model_root()
    if ckpt["path"] is None:
        config = load_config_from_yaml(str(BASE_CONFIG), for_inference=True)
        config.model.pretrained_model_name_or_path = model_root
        config.model.lora.pretrained_weight = None
        config.model.mask_control.enabled = False
        config.model.reference_cfg.enabled = False
        config.predict.devices = DeviceConfig(vae="cuda:0", text_encoder="cuda:0", dit="cuda:0")
        trainer = QwenImageEditInsertAnythingTrainer(config)
        trainer.setup_predict()
        return trainer

    import infer_testset

    src = Path(ckpt["path"])
    stage = SHM / "ckpt" / key
    _locked_copy([src / "pytorch_lora_weights.safetensors", src / "mask_geometry_tokenizer.safetensors"], stage)
    raw = yaml.safe_load((src / "train_config.yaml").read_text())
    raw["model"]["pretrained_model_name_or_path"] = model_root
    config_path = stage / "train_config.yaml"
    tmp = stage / f".train_config.{os.getpid()}.yaml"
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False))
    os.replace(tmp, config_path)
    argv = [
        "--config", str(config_path),
        "--lora", str(stage),
        "--num-inference-steps", str(NUM_STEPS),
        "--true-cfg-scale", str(TRUE_CFG),
        "--guidance-scale", "1.0",
        "--device", "cuda:0",
    ]  # fmt: skip
    trainer, _, _, _ = infer_testset.setup_trainer(infer_testset.parse_args(argv))
    return trainer


# --------------------------------------------------------------------------- run


def run_item(trainer, key: str, case: dict, variant: str, out: Path, saver) -> dict:
    from infer import seed_everything
    from src.utils.adaptive_crop import crop_pil_mask, crop_pil_pair
    from src.utils.adaptive_crop_anchoring import prepare_adaptive_crop_for_inference
    from src.utils.latent_anchoring import ensure_latent_anchoring_config
    from src.utils.visualization import ensure_mask, ensure_rgb

    started = time.time()
    seed = case["seed"] + (SEED2_OFFSET if variant == "clean_seed2" else 0)
    ref_clean, ref_mask = read_rgb(case["ref_image"]), read_mask(case["ref_mask"])
    ref_image = perturb_reference(ref_clean, ref_mask, variant)
    target, target_mask = ensure_rgb(read_rgb(case["tar_image"])), ensure_mask(read_mask(case["tar_mask"]))

    # Same wrapper as InsertAnythingValidationSampler.run_validation_loop: adaptive crop of the target,
    # GT mask as the condition (expanded_soft anchoring adds the interaction band).
    crop_cfg = trainer.config.data.init_args.adaptive_crop
    anchoring = ensure_latent_anchoring_config(trainer.config)
    crop_plan = None
    predict_source, predict_mask = target, target_mask
    if crop_cfg.predict_enabled:
        crop_plan, _ = prepare_adaptive_crop_for_inference(
            image_size=target.size,
            mask=target_mask,
            adaptive_crop_config=crop_cfg,
            config=trainer.config,
            processor_config=trainer.config.data.init_args.processor,
        )
        predict_source, predict_mask = crop_pil_pair(target, target_mask, crop_plan)

    kwargs = {}
    prompt = case["prompt"]
    if key == "base":
        prompt = f"Place the {BASE_NOUN.get(case['category'], 'object')} from Picture 2 into the black region of Picture 1."
        kwargs["negative_prompt"] = " "
    seed_everything(seed)
    local = ensure_rgb(
        trainer.predict(
            image=[predict_source, ref_image],
            prompt=prompt,
            ref_mask=ref_mask,
            src_mask=predict_mask,
            num_inference_steps=NUM_STEPS,
            true_cfg_scale=TRUE_CFG,
            guidance_scale=1.0,
            weight_dtype=torch.bfloat16,
            latent_anchoring_mode=anchoring.mode.value,
            **kwargs,
        )[0]
    )
    sample_seconds = time.time() - started

    # Metrics in local (processed crop) space: GT crop resized to the output size.
    gt_local = predict_source.resize(local.size, Image.Resampling.LANCZOS)
    gt_mask_local = binary_mask(predict_mask, local.size)
    region = erode(gt_mask_local, ERODE_PX)
    out_stats = region_stats(local, region)
    gt_stats = region_stats(gt_local, region)
    de_px = ciede2000(out_stats["_lab"], gt_stats["_lab"])
    ref_region = erode(binary_mask(ref_mask, ref_clean.size), ERODE_PX)
    ref_stats = region_stats(ref_image, ref_region)
    ref_clean_stats = region_stats(ref_clean, ref_region)
    meta = {
        "checkpoint": key,
        "variant": variant,
        "seed": seed,
        "prompt": prompt,
        "case": {k: case[k] for k in ("domain", "order", "category", "stem", "prefix")},
        "gains_nominal": variant_gains(variant).tolist(),
        "output": public(out_stats),
        "gt": public(gt_stats),
        "reference": public(ref_stats),
        "reference_clean": public(ref_clean_stats),
        "de00_pixel_mean_vs_gt": float(de_px.mean()),
        "de00_pixel_median_vs_gt": float(np.median(de_px)),
        "de00_mean_color_vs_gt": float(ciede2000(np.array(out_stats["mean_lab"]), np.array(gt_stats["mean_lab"]))),
        "crop_plan": crop_plan.to_dict() if crop_plan is not None else None,
        "sizes": {"target": list(target.size), "local": list(local.size), "reference": list(ref_clean.size)},
        "sample_seconds": round(sample_seconds, 1),
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0),
    }
    run_dir = out / "runs" / key / variant
    case_dir = out / "cases"

    def save():
        run_dir.mkdir(parents=True, exist_ok=True)
        case_dir.mkdir(parents=True, exist_ok=True)
        local.save(run_dir / f"{case['prefix']}.jpg", quality=93)
        gt_path = case_dir / f"{case['prefix']}_gt_local_{local.size[0]}x{local.size[1]}.jpg"
        if not gt_path.exists():
            tmp_gt = gt_path.with_name(f".{gt_path.name}.{os.getpid()}.tmp")
            gt_local.save(tmp_gt, format="JPEG", quality=93)
            os.replace(tmp_gt, gt_path)
        tmp = run_dir / f".{case['prefix']}.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, run_dir / f"{case['prefix']}.json")

    saver.submit(save)
    return meta


def item_done(out: Path, key: str, case: dict, variant: str) -> bool:
    return (out / "runs" / key / variant / f"{case['prefix']}.json").exists()


# --------------------------------------------------------------------------- shared-disk work queue
# Same scheme as scripts/probe_reference_rope.py work: atomic O_EXCL claim files, stale-claim
# reclaim, and a worker keeps its loaded checkpoint until that checkpoint has no unclaimed runs.

CLAIM_STALE_SECONDS = 600
HEARTBEAT_SECONDS = 30
ITEM_SECONDS, LOAD_SECONDS = 20.0, 150.0


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
        # Variant-major per case: the clean run of a case lands before its perturbed runs.
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
            done = {v: self._listing("runs", key, v) for v in self.plan["variants"]}
            claimed = {v: self._listing("claims", key, v) for v in self.plan["variants"]}
            counts[key] = sum(
                1
                for case, v in items
                if f"{case['prefix']}.json" not in done[v] and case["prefix"] not in claimed[v]
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
        remaining = self.remaining()
        if self.checkpoint is not None and remaining.get(self.checkpoint, 0) > 0:
            return self.checkpoint
        active = self.active()
        share = {key: remaining[key] / (active[key] + 1) for key in remaining if remaining[key] > 0}
        if not share:
            return None
        best = max(share, key=share.get)
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

    trainer = None
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
                    trainer = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    started = time.time()
                    trainer = setup_trainer(key, plan["checkpoints"][key])
                    status["loads"].append([key, round(time.time() - started)])
                    write_status(state="running")
                while True:
                    item = queue.claim_next(key) or queue.claim_stale(key)
                    if item is None:
                        break
                    case, variant = item
                    try:
                        meta = run_item(trainer, key, case, variant, out, saver)
                        status["done"] += 1
                        print(f"[{args.name}] {key} {variant} {case['prefix']} {meta['sample_seconds']}s", flush=True)
                    except Exception:
                        traceback.print_exc()
                        status["failed"].append([key, case["prefix"], variant, traceback.format_exc()[-2000:]])
                    write_status(state="running")
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
    p.set_defaults(func=cmd_plan)
    w = sub.add_parser("work", help="claim runs from the shared queue until none are left")
    w.add_argument("--plan", required=True)
    w.add_argument("--name", required=True)
    w.add_argument("--stagger", type=float, default=0.0)
    w.set_defaults(func=cmd_work)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

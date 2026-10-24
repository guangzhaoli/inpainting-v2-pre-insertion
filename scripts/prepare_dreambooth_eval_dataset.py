#!/usr/bin/env python3
"""Convert AnyDoor-style DreamBooth FG/BG dirs into a canonical insertion testset.

AnyDoor stores evaluation subjects as RGBA (or per-subject folders with RGBA)
under ``FG`` / ``AnyDoor_DreamBooth``, and backgrounds as paired
``{id}_GT.png`` + ``{id}_mask.png`` under ``BG`` / ``v1_800``.

This script materializes a folder layout our baseline + ours runners already
understand:

```text
<output_root>/
  subject/
    src_image/   # reference subject RGB
    src_mask/    # reference subject mask
    tar_image/   # background scene (GT)
    tar_mask/    # insertion hole on the background
  manifests/dreambooth_eval.jsonl
  reports/summary.json
```

Default pairing is the cartesian product FG × BG (fixed, deterministic).
Optional ``--max-pairs-per-fg`` / ``--limit`` cap size for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MASK_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_mask$", re.IGNORECASE)
_GT_SUFFIX_RE = re.compile(r"^(?P<stem>.+)_GT$", re.IGNORECASE)


@dataclass(frozen=True)
class Foreground:
    fg_id: str
    image_path: Path


@dataclass(frozen=True)
class Background:
    bg_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class Pair:
    case_id: str
    fg: Foreground
    bg: Background


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def discover_foregrounds(fg_dir: Path) -> list[Foreground]:
    """Accept flat RGBA images or per-subject subfolders (AnyDoor full set)."""
    fg_dir = fg_dir.resolve()
    if not fg_dir.is_dir():
        raise FileNotFoundError(f"FG dir not found: {fg_dir}")

    out: list[Foreground] = []
    # Subfolder layout: fg_dir/<subject_id>/*.png (RGBA preferred)
    subdirs = sorted([p for p in fg_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if subdirs:
        for sub in subdirs:
            images = sorted([p for p in sub.iterdir() if _is_image(p)], key=lambda p: p.name)
            if not images:
                continue
            # Prefer RGBA if present.
            rgba: list[Path] = []
            for p in images:
                with Image.open(p) as im:
                    if im.mode == "RGBA":
                        rgba.append(p)
            chosen = rgba[0] if rgba else images[0]
            out.append(Foreground(fg_id=sub.name, image_path=chosen))
        if out:
            return out

    # Flat layout: fg_dir/*.png (TestDreamBooth style)
    for path in sorted(fg_dir.iterdir(), key=lambda p: p.name):
        if not _is_image(path):
            continue
        out.append(Foreground(fg_id=path.stem, image_path=path))
    if not out:
        raise RuntimeError(f"no foreground images under {fg_dir}")
    return out


def discover_backgrounds(bg_dir: Path) -> list[Background]:
    """Pair ``*_mask`` with ``*_GT`` (or bare stem image) under bg_dir."""
    bg_dir = bg_dir.resolve()
    if not bg_dir.is_dir():
        raise FileNotFoundError(f"BG dir not found: {bg_dir}")

    by_stem: dict[str, dict[str, Path]] = {}
    for path in bg_dir.iterdir():
        if not _is_image(path):
            continue
        stem = path.stem
        m_mask = _MASK_SUFFIX_RE.match(stem)
        m_gt = _GT_SUFFIX_RE.match(stem)
        if m_mask:
            by_stem.setdefault(m_mask.group("stem"), {})["mask"] = path
        elif m_gt:
            by_stem.setdefault(m_gt.group("stem"), {})["image"] = path
        else:
            # Bare image stem; mask may still be stem_mask.
            by_stem.setdefault(stem, {}).setdefault("image", path)

    out: list[Background] = []
    for stem in sorted(by_stem):
        slots = by_stem[stem]
        mask = slots.get("mask")
        image = slots.get("image")
        if mask is None or image is None:
            continue
        out.append(Background(bg_id=stem, image_path=image, mask_path=mask))
    if not out:
        raise RuntimeError(
            f"no complete BG pairs (*_GT + *_mask, or stem + stem_mask) under {bg_dir}"
        )
    return out


def build_pairs(
    fgs: list[Foreground],
    bgs: list[Background],
    *,
    max_pairs_per_fg: int | None = None,
    limit: int | None = None,
) -> list[Pair]:
    pairs: list[Pair] = []
    for fg in fgs:
        used = 0
        for bg in bgs:
            case_id = f"{fg.fg_id}__{bg.bg_id}"
            pairs.append(Pair(case_id=case_id, fg=fg, bg=bg))
            used += 1
            if max_pairs_per_fg is not None and used >= max_pairs_per_fg:
                break
            if limit is not None and len(pairs) >= limit:
                return pairs
    if not pairs:
        raise RuntimeError("no pairs built")
    return pairs


def _load_rgb_and_mask_from_fg(path: Path) -> tuple[Image.Image, Image.Image]:
    img = Image.open(path)
    if img.mode in {"LA", "PA"}:
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        rgb = img.convert("RGB")
        alpha = img.getchannel("A")
        mask = alpha.point(lambda v: 255 if v > 128 else 0)
        return rgb, mask
    # No alpha: full-image mask (degraded fallback for rare RGB-only FG).
    rgb = img.convert("RGB")
    mask = Image.new("L", rgb.size, 255)
    return rgb, mask


def _load_bg(image_path: Path, mask_path: Path) -> tuple[Image.Image, Image.Image]:
    rgb = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path)
    if mask_img.mode == "RGBA":
        mask = mask_img.getchannel("A").point(lambda v: 255 if v > 128 else 0)
    else:
        mask = mask_img.convert("L").point(lambda v: 255 if v > 128 else 0)
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.NEAREST)
    return rgb, mask


def materialize(
    pairs: list[Pair],
    output_root: Path,
    *,
    major: str = "subject",
    overwrite: bool = False,
) -> dict:
    output_root = output_root.resolve()
    leaf = output_root / major
    dirs = {
        "src_image": leaf / "src_image",
        "src_mask": leaf / "src_mask",
        "tar_image": leaf / "tar_image",
        "tar_mask": leaf / "tar_mask",
    }
    if leaf.exists() and any(d.exists() and any(d.iterdir()) for d in dirs.values()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty DreamBooth output: {leaf} "
                f"(pass --overwrite)"
            )
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pair in pairs:
        stem = pair.case_id
        src_rgb, src_mask = _load_rgb_and_mask_from_fg(pair.fg.image_path)
        tar_rgb, tar_mask = _load_bg(pair.bg.image_path, pair.bg.mask_path)

        src_image_rel = f"{major}/src_image/{stem}.png"
        src_mask_rel = f"{major}/src_mask/{stem}.png"
        tar_image_rel = f"{major}/tar_image/{stem}.png"
        tar_mask_rel = f"{major}/tar_mask/{stem}.png"

        src_rgb.save(output_root / src_image_rel)
        src_mask.save(output_root / src_mask_rel)
        tar_rgb.save(output_root / tar_image_rel)
        tar_mask.save(output_root / tar_mask_rel)

        rows.append(
            {
                "case_id": stem,
                "major": major,
                "minor": "subject",
                "protocol": "anydoor_dreambooth_eval",
                "same_identity": False,
                "gt_available": True,
                "src_image": src_image_rel,
                "src_mask": src_mask_rel,
                "tar_image": tar_image_rel,
                "tar_mask": tar_mask_rel,
                "fg_id": pair.fg.fg_id,
                "bg_id": pair.bg.bg_id,
                "source_fg_path": str(pair.fg.image_path),
                "source_bg_image": str(pair.bg.image_path),
                "source_bg_mask": str(pair.bg.mask_path),
            }
        )

    manifest_path = output_root / "manifests" / "dreambooth_eval.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "n_cases": len(rows),
        "n_fg": len({r["fg_id"] for r in rows}),
        "n_bg": len({r["bg_id"] for r in rows}),
        "major": major,
        "manifest": str(manifest_path.relative_to(output_root)),
        "output_root": str(output_root),
    }
    (output_root / "reports" / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fg-dir",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
            "baselines/AnyDoor/examples/TestDreamBooth/FG"
        ),
        help="AnyDoor FG subjects (RGBA flat images or per-subject folders).",
    )
    p.add_argument(
        "--bg-dir",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
            "baselines/AnyDoor/examples/TestDreamBooth/BG"
        ),
        help="AnyDoor BG dir with *_GT.png + *_mask.png pairs.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/"
            "a2edit/eval/dreambooth_eval"
        ),
    )
    p.add_argument("--major", type=str, default="subject")
    p.add_argument(
        "--max-pairs-per-fg",
        type=int,
        default=None,
        help="Optional cap of backgrounds paired to each FG subject.",
    )
    p.add_argument("--limit", type=int, default=None, help="Global pair cap.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    fgs = discover_foregrounds(args.fg_dir)
    bgs = discover_backgrounds(args.bg_dir)
    pairs = build_pairs(
        fgs,
        bgs,
        max_pairs_per_fg=args.max_pairs_per_fg,
        limit=args.limit,
    )
    summary = materialize(
        pairs,
        args.output_root,
        major=args.major,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

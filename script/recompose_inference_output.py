#!/usr/bin/env python3
"""Recompose saved inference artifacts with an expanded soft edit mask.

This is a non-destructive repair utility for legacy strict-mask outputs. It uses
``source.png``, ``source_mask.png``, and ``output_restored_raw.png`` already
stored in each sample directory, so it does not load a model or require a GPU.
The original ``output.png`` is never overwritten by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.visualization import build_expanded_soft_mask_composite


DEFAULT_OUTPUT_NAME = "output_expanded_soft.png"
DEFAULT_MASK_OUTPUT_NAME = "output_expanded_soft_mask.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sample_dirs",
        type=Path,
        nargs="+",
        help=(
            "Inference sample directories containing source.png, source_mask.png, "
            "and output_restored_raw.png."
        ),
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--dilation-px",
        type=int,
        default=16,
        help="Radius of the fully generated replacement region outside the original mask.",
    )
    parser.add_argument(
        "--transition-width-px",
        type=int,
        default=16,
        help="Width of the cosine-decayed source/generated transition outside the dilation.",
    )
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--mask-output-name", default=DEFAULT_MASK_OUTPUT_NAME)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace derived expanded-soft files if they already exist.",
    )
    return parser.parse_args(argv)


def _validate_output_name(value: str, argument: str) -> str:
    path = Path(value)
    if not value or path.name != value or value in {".", ".."}:
        raise ValueError(f"{argument} must be a plain file name, got {value!r}")
    return value


def recompose_sample(
    sample_dir: Path,
    *,
    mask_threshold: float,
    dilation_px: int,
    transition_width_px: int,
    output_name: str = DEFAULT_OUTPUT_NAME,
    mask_output_name: str = DEFAULT_MASK_OUTPUT_NAME,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    sample_dir = sample_dir.resolve()
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"sample directory does not exist: {sample_dir}")

    output_name = _validate_output_name(output_name, "output_name")
    mask_output_name = _validate_output_name(mask_output_name, "mask_output_name")
    output_path = sample_dir / output_name
    mask_output_path = sample_dir / mask_output_name
    existing = [path for path in (output_path, mask_output_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "derived output already exists; pass --overwrite to replace it: "
            + ", ".join(str(path) for path in existing)
        )

    source_path = sample_dir / "source.png"
    source_mask_path = sample_dir / "source_mask.png"
    raw_output_path = sample_dir / "output_restored_raw.png"
    missing = [
        path
        for path in (source_path, source_mask_path, raw_output_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "sample directory is missing required inference artifacts: "
            + ", ".join(str(path) for path in missing)
        )

    with (
        Image.open(source_path) as source_image,
        Image.open(source_mask_path) as source_mask_image,
        Image.open(raw_output_path) as raw_output_image,
    ):
        composite, soft_mask = build_expanded_soft_mask_composite(
            source=source_image,
            mask=source_mask_image,
            output=raw_output_image,
            mask_threshold=mask_threshold,
            dilation_px=dilation_px,
            transition_width_px=transition_width_px,
        )
        composite.save(output_path)
        soft_mask.save(mask_output_path)

    return output_path, mask_output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for sample_dir in args.sample_dirs:
        output_path, mask_output_path = recompose_sample(
            sample_dir,
            mask_threshold=args.mask_threshold,
            dilation_px=args.dilation_px,
            transition_width_px=args.transition_width_px,
            output_name=args.output_name,
            mask_output_name=args.mask_output_name,
            overwrite=args.overwrite,
        )
        print(f"saved composite: {output_path}", flush=True)
        print(f"saved soft mask: {mask_output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

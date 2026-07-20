#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from infer_testset import (
    DEFAULT_CONFIG,
    DEFAULT_TESTSET_ROOT,
    discover_samples,
    run_sample,
    setup_trainer,
    stack_rows,
)
from src.utils.latent_anchoring import (
    add_latent_anchoring_cli_args,
    ensure_latent_anchoring_config,
    latent_anchoring_metadata,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "ref_sensitivity_ablation"


def parse_stems(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None

    stems = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                stems.append(token)
    return stems


def build_sample_by_stem(samples: list[dict]) -> dict[str, dict]:
    sample_by_stem = {}
    for sample in samples:
        stem = sample["stem"]
        if stem in sample_by_stem:
            raise ValueError(f"Duplicate sample stem in testset: {stem}")
        sample_by_stem[stem] = sample
    return sample_by_stem


def require_stems(sample_by_stem: dict[str, dict], stems: list[str], kind: str) -> None:
    missing = [stem for stem in stems if stem not in sample_by_stem]
    if missing:
        available = ", ".join(sample_by_stem.keys())
        raise ValueError(f"Unknown {kind} stems: {missing}. Available stems: {available}")


def choose_ref_stems(samples: list[dict], source_stems: list[str], requested_ref_stems: list[str] | None, max_refs: int):
    if requested_ref_stems is not None:
        return requested_ref_stems

    selected = []
    for stem in source_stems:
        if stem not in selected:
            selected.append(stem)

    for sample in samples:
        stem = sample["stem"]
        if stem not in selected:
            selected.append(stem)
        if len(selected) >= max_refs:
            break
    return selected


def masked_mae(image_a: Path, image_b: Path, mask_path: Path) -> float:
    a = Image.open(image_a).convert("RGB")
    b = Image.open(image_b).convert("RGB")
    if b.size != a.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)

    mask = Image.open(mask_path).convert("L")
    if mask.size != a.size:
        mask = mask.resize(a.size, Image.Resampling.NEAREST)

    mask_np = np.array(mask) > 127
    if not mask_np.any():
        return float("nan")

    a_np = np.array(a, dtype=np.float32)
    b_np = np.array(b, dtype=np.float32)
    return float(np.abs(a_np[mask_np] - b_np[mask_np]).mean())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run reference sensitivity ablations by swapping references for fixed source samples."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--lora", type=Path, required=True, help="LoRA file or checkpoint directory.")
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help="Allow a missing or incompatible checkpoint train_config.yaml. Unsafe unless intentional.",
    )
    parser.add_argument("--testset-root", type=Path, default=DEFAULT_TESTSET_ROOT)
    parser.add_argument("--category", type=str, default="accessory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source-stems",
        nargs="+",
        default=["8", "9"],
        help="Source sample stems to keep fixed. Comma-separated values are also accepted.",
    )
    parser.add_argument(
        "--ref-stems",
        nargs="+",
        default=None,
        help="Reference stems to swap in. If omitted, uses original source refs plus the first --max-ref-variants samples.",
    )
    parser.add_argument("--max-ref-variants", type=int, default=6)
    parser.add_argument("--prompt", type=str, default="Insert the reference jewelry into the masked region.")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-size", type=int, default=320)
    add_latent_anchoring_cli_args(parser)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    source_stems = parse_stems(args.source_stems)
    requested_ref_stems = parse_stems(args.ref_stems)
    if not source_stems:
        raise ValueError("--source-stems must contain at least one stem")
    if args.max_ref_variants <= 0:
        raise ValueError("--max-ref-variants must be positive")

    samples = discover_samples(args.testset_root, args.category, limit=None)
    sample_by_stem = build_sample_by_stem(samples)
    require_stems(sample_by_stem, source_stems, "source")

    ref_stems = choose_ref_stems(samples, source_stems, requested_ref_stems, args.max_ref_variants)
    if not ref_stems:
        raise ValueError("reference stem selection must not be empty")
    require_stems(sample_by_stem, ref_stems, "reference")
    if args.seed + len(source_stems) - 1 >= 2**32:
        raise ValueError("per-source seeds must remain in [0, 2**32)")

    trainer, config, resolved_lora_path, checkpoint_compatibility = setup_trainer(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "config": str(args.config),
        "checkpoint_arg": str(args.lora),
        "resolved_lora": str(resolved_lora_path),
        "checkpoint_compatibility": checkpoint_compatibility,
        "testset_root": str(args.testset_root),
        "category": args.category,
        "source_stems": source_stems,
        "ref_stems": ref_stems,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "reference_cfg_enabled": bool(config.model.reference_cfg.enabled),
        "query_prompt_enabled": bool(config.model.query_prompt.enabled),
        "pre_insertion_enabled": bool(config.model.pre_insertion.enabled),
        "latent_anchoring": latent_anchoring_metadata(ensure_latent_anchoring_config(config)),
    }
    write_json(args.output_dir / "run_config.json", run_config)

    manifest_path = args.output_dir / "manifest.jsonl"
    summary_rows = []
    all_row_paths = []

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for source_index, source_stem in enumerate(source_stems):
            source_sample = sample_by_stem[source_stem]
            source_dir = args.output_dir / f"src_{source_stem}"
            source_row_paths = []
            baseline_output_path = None
            baseline_ref_stem = source_stem if source_stem in ref_stems else ref_stems[0]
            source_seed = args.seed + source_index

            for ref_stem in ref_stems:
                ref_sample = sample_by_stem[ref_stem]
                sample = {
                    "stem": f"src{source_stem}_ref{ref_stem}",
                    "ref_image": ref_sample["ref_image"],
                    "ref_mask": ref_sample["ref_mask"],
                    "source": source_sample["source"],
                    "source_mask": source_sample["source_mask"],
                }
                sample_dir = source_dir / f"ref_{ref_stem}"
                print(
                    f"[src {source_stem}] ref {ref_stem}, seed {source_seed} -> {sample_dir}",
                    flush=True,
                )
                metadata, row_path = run_sample(
                    trainer,
                    sample,
                    sample_dir,
                    args,
                    source_seed,
                    resolved_lora_path=resolved_lora_path,
                    checkpoint_compatibility=checkpoint_compatibility,
                )
                metadata["ablation"] = {
                    "source_stem": source_stem,
                    "ref_stem": ref_stem,
                    "baseline_ref_stem": baseline_ref_stem,
                }
                with (sample_dir / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                manifest.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                manifest.flush()

                source_row_paths.append(row_path)
                all_row_paths.append(row_path)

                if ref_stem == baseline_ref_stem:
                    baseline_output_path = sample_dir / "output.png"

            if baseline_output_path is None:
                raise RuntimeError(f"Missing baseline output for source {source_stem}, ref {baseline_ref_stem}")

            for ref_stem in ref_stems:
                sample_dir = source_dir / f"ref_{ref_stem}"
                mae = masked_mae(
                    baseline_output_path,
                    sample_dir / "output.png",
                    sample_dir / "output_mask.png",
                )
                summary_rows.append(
                    {
                        "source_stem": source_stem,
                        "ref_stem": ref_stem,
                        "baseline_ref_stem": baseline_ref_stem,
                        "seed": source_seed,
                        "masked_output_mae_vs_baseline": mae,
                        "output": str(sample_dir / "output.png"),
                        "composite": str(sample_dir / "composite.png"),
                        "row": str(sample_dir / "row.png"),
                    }
                )

            stack_rows(source_row_paths, source_dir / "all_rows.png")

    stack_rows(all_row_paths, args.output_dir / "all_rows.png")

    summary_path = args.output_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_stem",
                "ref_stem",
                "baseline_ref_stem",
                "seed",
                "masked_output_mae_vs_baseline",
                "output",
                "composite",
                "row",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Done. Results saved to {args.output_dir}", flush=True)
    print(f"Overview saved to {args.output_dir / 'all_rows.png'}", flush=True)
    print(f"Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

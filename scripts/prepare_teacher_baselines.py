"""Evaluate stored v1 outputs on the same validation cases used by Teacher v2."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.data.config import load_config_from_yaml
from src.loss.region_perceptual_loss import RegionPerceptualLoss
from src.utils.teacher_validation import teacher_validation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    base = (
        root
        / "output/inference/testset_20260120/all_ckpts_final_architecture_8gpu/runs"
    )
    torch.set_num_threads(2)
    output = {
        "comparison": "Historical v1 final checkpoints versus v2 validation; seeds, CFG and output modes differ. Not a controlled architecture-only ablation.",
        "domains": {},
    }
    criterion = None
    for domain, job in manifest["jobs"].items():
        cfg = load_config_from_yaml(job["config"])
        if criterion is None:
            criterion = RegionPerceptualLoss(
                fg_metric="none",
                enable_ref_dino=True,
                dino_model_name=cfg.loss.dino_model_name,
            )
        rows = json.loads(Path(cfg.logging.sampling.validation_data).read_text())
        records = {}
        for row in rows:
            if domain == "accessory_jewelry":
                parent = (
                    base
                    / "accessory/final_architecture_maskmix_geometry_region_perceptual/checkpoint-last-4-9800-last"
                )
            else:
                parent = (
                    base
                    / f"pets/{row['category']}/final_architecture_animal/checkpoint-last-4-14490-last"
                )
            candidates = list(parent.glob(f"*_{row['stem']}/metadata.json"))
            if len(candidates) != 1:
                continue
            directory = candidates[0].parent
            metadata = json.loads(candidates[0].read_text())
            raw_path = directory / "output_restored_raw.png"
            sample = SimpleNamespace(
                tar_image=Image.open(row["tar_image"]).convert("RGB"),
                tar_mask=Image.open(row["tar_mask"]).convert("L"),
                ref_image=Image.open(row["ref_image"]).convert("RGB"),
                ref_mask=Image.open(row["ref_mask"]).convert("L"),
            )
            raw = Image.open(raw_path).convert("RGB")
            sizes = metadata["sizes"]
            ratio = sizes["prediction_input"][0] / sizes["output_local"][0]
            metrics = teacher_validation_metrics(
                sample,
                raw,
                criterion,
                exclusion_radius=cfg.loss.interaction_radius_px * ratio,
            )
            key = f"{row['category']}/{row['stem']}"
            records[key] = {
                "metrics": metrics,
                "primary": str(directory / "output.png"),
                "raw": str(raw_path),
                "metadata": str(candidates[0]),
            }
            print(domain, key, metrics, flush=True)
        output["domains"][domain] = records
    path = args.manifest.parent / "historical_baselines.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()

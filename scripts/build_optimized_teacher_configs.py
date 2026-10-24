"""Build the five integrated teacher configs using this checkout's local assets.

Run with uv run scripts/build_optimized_teacher_configs.py. Existing baseline
configs and checkpoint snapshots are not rewritten. No files are downloaded.
"""

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEST = ROOT / "configs/teacher_v2"
DOMAINS = ["accessory_jewelry", "animal", "fashion_tryon", "general_object", "human"]

# MaskMix without `exact`: no hole traces the GT outline. Loose families
# (loose_blob, oversized_box) are drawn in the dataset before the crop and
# leave part of the hole to real background, so the object no longer has to
# fill the mask. Fashion keeps tight families only: a loose region around a
# garment would also hide the face and hands that the GT keeps.
MASK_MODE_WEIGHTS = {
    "animal": dict(random_morphology=0.10, augmented_box=0.15, asymmetric_box=0.15,
                   loose_blob=0.35, oversized_box=0.25),
    "general_object": dict(random_morphology=0.20, augmented_box=0.20, asymmetric_box=0.20,
                           loose_blob=0.25, oversized_box=0.15),
    "human": dict(random_morphology=0.20, augmented_box=0.20, asymmetric_box=0.20,
                  loose_blob=0.25, oversized_box=0.15),
    "accessory_jewelry": dict(random_morphology=0.30, augmented_box=0.25, asymmetric_box=0.25,
                              loose_blob=0.15, oversized_box=0.05),
    "fashion_tryon": dict(random_morphology=0.40, augmented_box=0.30, asymmetric_box=0.30),
}
LOOSE_GEOMETRY = {
    "animal": dict(scale_min=1.2, scale_max=2.5, bottom_align_prob=0.7),
    "general_object": dict(scale_min=1.2, scale_max=2.2, bottom_align_prob=0.5),
    "human": dict(scale_min=1.2, scale_max=2.0, bottom_align_prob=0.7),
    "accessory_jewelry": dict(scale_min=1.2, scale_max=2.0, bottom_align_prob=0.0),
}
# Hair masks wrap the face; a loose region would hide the face the GT keeps.
HAIR_MODE_WEIGHTS = dict(random_morphology=0.50, augmented_box=0.25, asymmetric_box=0.25)


def mode_weights(weights):
    return {name: weights.get(name, 0.0) for name in
            ("exact", "random_morphology", "augmented_box", "asymmetric_box", "loose_blob", "oversized_box")}


def validation_rows(domain):
    testset = WORKSPACE / "a2edit/eval/testset_20260120"
    leaves = {
        "accessory_jewelry": [testset / "accessory"],
        "animal": [testset / "pets/cats", testset / "pets/dogs"],
        "fashion_tryon": [testset / "clothes", testset / "clothes/pure_color"],
        "human": [testset / "person", testset / "hair"],
        "general_object": [WORKSPACE / "a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900/object"],
    }[domain]
    rows = []
    for leaf in leaves:
        ref_key = "ref_image" if (leaf / "ref_image").is_dir() else "src_image"
        mask_key = "ref_mask" if ref_key == "ref_image" else "src_mask"
        for ref in sorted((leaf / ref_key).glob("*")):
            if ref.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            paths = {"ref_image": ref}
            for key, folder in [("ref_mask", mask_key), ("tar_image", "tar_image"), ("tar_mask", "tar_mask")]:
                matches = sorted(p for p in (leaf / folder).glob(ref.stem + ".*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
                if matches:
                    paths[key] = matches[0]
            if len(paths) == 4:
                rows.append({**{k: str(v) for k, v in paths.items()}, "stem": ref.stem,
                             "category": leaf.name, "prompt": ""})
    if not rows:
        raise FileNotFoundError(f"No complete external validation cases for {domain}")
    return rows


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        old_name = "final_architecture.yaml" if domain == "accessory_jewelry" else f"final_architecture_{domain}.yaml"
        cfg = yaml.safe_load((ROOT / "configs/exp" / old_name).read_text())
        model = cfg["model"]
        model["pretrained_model_name_or_path"] = str(WORKSPACE / "pretrained_weights/Qwen-Image-Edit-2511")
        model["attention_backend"] = "flash_attention_2"
        model["zero_cond_t"] = True
        # Keep attention and visual MLP adaptation; avoid updating the text MLP.
        model["lora"].update(r=128, lora_alpha=128, adapter_name=f"insertfuse_teacher_v2_{domain}_r128")
        model["lora"]["target_modules"] = [x for x in model["lora"]["target_modules"] if not x.startswith("txt_mlp.")]
        model["reference_cfg"].update(mask_aware=True, delta_clip_enabled=True, delta_clip_max_ratio=1.0)
        data = cfg["data"]["init_args"]
        data["dataset_path"] = str(WORKSPACE / "a2edit/data")
        data["manifest_path"] = str(ROOT / "manifests" / f"insert_anything_teacher_{domain}.csv")
        data["category_balanced_sampling"] = True
        data["face_identity_supervision"] = domain == "human"
        data["processor"]["init_args"].update(process_type="aspect_pixels", resize_mode="bicubic")
        small = domain == "accessory_jewelry"
        data["adaptive_crop"].update(trigger_bbox_area_ratio=0.2,
                                     min_crop_area_ratio=0.0625 if small else 0.125,
                                     context_margin_ratio=0.35)
        radius = 16 if small else 32
        hybrid = data["hybrid_masking"]
        hybrid["interaction_dilation_px"] = radius
        hybrid["maskmix_v2"]["mode_weights"] = mode_weights(MASK_MODE_WEIGHTS[domain])
        # Every box side keeps a margin, and morphology radii exceed the interaction
        # band, so neither family reproduces the outline.
        hybrid["maskmix_v2"]["asymmetric_box"].update(margin_ratio_min=0.05, margin_ratio_max=0.30)
        hybrid["maskmix_v2"]["morphology"].update(
            radius_object_scale_min=0.06, radius_object_scale_max=0.20, radius_min_px=8, radius_max_px=96
        )
        hybrid["context_guard"]["overflow_policy"] = "fallback_to_band"
        if domain in LOOSE_GEOMETRY:
            hybrid["loose_placement"] = dict(enabled=False, **LOOSE_GEOMETRY[domain])
        if domain == "human":
            hair = copy.deepcopy(hybrid)
            hair["maskmix_v2"]["mode_weights"] = mode_weights(HAIR_MODE_WEIGHTS)
            data["category_conditioning"] = {"person_hair": {"hybrid_masking": hair}}
        cfg["train"].update(trainable_precision="float32", rank_specific_rng=True, seed=42, memory_policy="auto")
        cfg["optimizer"]["init_args"]["lr"] = 5e-5
        loss = cfg["loss"]
        weights = WORKSPACE / "pretrained_weights"
        loss.update(ref_dino_weight=0.2, dino_backend="transformers",
                    face_dino_weight=0.2 if domain == "human" else 0.0,
                    dino_model_name=str(weights / "metrics_offline/dinov2-small"),
                    detail_lpips_weight=0.25, detail_num_patches=4,
                    detail_patch_size=128 if small else 256,
                    boundary_l1_weight=0.25, interaction_radius_px=radius,
                    lpips_linear_path=str(weights / "lpips/linear/vgg.pth"),
                    lpips_torch_home=str(weights / "torch"),
                    lpips_trunk_path=str(weights / "torch/hub/checkpoints/vgg16-397923af.pth"))
        cfg["predict"] = {"devices": {"vae": "cpu", "text_encoder": "cpu", "dit": "cpu"},
                          "latent_anchoring": {"mode": "expanded_soft", "mask_threshold": 0.5,
                                                "interaction_dilation_px": radius, "transition_width_px": 16}}
        rows = validation_rows(domain)
        val_path = DEST / f"validation_{domain}.json"
        val_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        cfg["logging"].update(output_dir="logs/teacher_v2", tracker_project_name=f"insertfuse_teacher_v2_{domain}")
        cfg["logging"]["sampling"].update(enable=True, validation_steps=1000, num_samples=min(16, len(rows)),
                                          fixed_seed_across_steps=True, true_cfg_scale=2.5,
                                          validation_data=str(val_path))
        cfg["resume"] = None
        (DEST / f"{domain}.yaml").write_text(
            "# Integrated Teacher v2. Regenerate paths with scripts/build_optimized_teacher_configs.py.\n"
            + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print(f"{domain}: {len(rows)} external validation cases; full {cfg['train']['num_epochs']} epochs retained")


if __name__ == "__main__":
    main()

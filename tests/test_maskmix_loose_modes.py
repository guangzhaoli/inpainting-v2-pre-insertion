"""MaskMix-v2 loose families without σ-split, and teacher configs without `exact`."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pydantic import ValidationError

from src.data.config import Config, DatasetInitArgs, HybridMaskingConfig
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    LOOSE_BLOB,
    LOOSE_OVERSIZED_BOX,
    MASKMIX_LOOSE_MODES,
    MASKMIX_V2_ASYMMETRIC_BOX,
    MASKMIX_V2_EXACT,
    MASKMIX_V2_RANDOM_MORPHOLOGY,
    _weighted_choice,
    add_interaction_context,
    apply_hybrid_masking_context_guard,
    build_loose_placement_mask,
    build_maskmix_v2_hybrid_masks,
    build_maskmix_v2_mask_for_mode,
    sample_maskmix_v2_mode,
)

ROOT = Path(__file__).resolve().parents[1]
NO_EXACT = {
    "exact": 0.0,
    "random_morphology": 0.2,
    "augmented_box": 0.2,
    "asymmetric_box": 0.2,
    "loose_blob": 0.25,
    "oversized_box": 0.15,
}


def hybrid(weights=None, **overrides) -> HybridMaskingConfig:
    payload = {
        "enabled": True,
        "strategy": "maskmix_v2",
        "interaction_dilation_px": 8,
        "context_guard": {"enabled": True, "overflow_policy": "fallback_to_band"},
        "maskmix_v2": {"mode_weights": dict(NO_EXACT if weights is None else weights)},
    }
    payload.update(overrides)
    return HybridMaskingConfig(**payload)


def blob_mask(height=256, width=384) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (190, 150), (50, 30), 0, 0, 360, 1, -1)
    return mask


class MaskMixLooseConfigTest(unittest.TestCase):
    def test_loose_weights_join_the_distribution(self):
        config = hybrid()
        self.assertEqual(config.maskmix_v2.mode_weights.loose_blob, 0.25)
        self.assertEqual(config.context_guard.overflow_policy, "fallback_to_band")
        with self.assertRaises(ValidationError):
            hybrid({**NO_EXACT, "oversized_box": 0.25})

    def test_loose_weights_are_rejected_with_sigma_split_masks(self):
        with self.assertRaisesRegex(ValidationError, "σ-split"):
            hybrid(loose_placement={"enabled": True})

    def test_teacher_configs_never_sample_exact(self):
        for domain in ("accessory_jewelry", "animal", "fashion_tryon", "general_object", "human"):
            cfg = Config.model_validate(yaml.safe_load((ROOT / "configs/teacher_v2" / f"{domain}.yaml").read_text()))
            hybrid_cfg = cfg.data.init_args.hybrid_masking
            profiles = [hybrid_cfg] + [p.hybrid_masking for p in cfg.data.init_args.category_conditioning.values()]
            for config in profiles:
                with self.subTest(domain=domain):
                    self.assertEqual(config.maskmix_v2.mode_weights.exact, 0.0)
                    self.assertEqual(config.context_guard.overflow_policy, "fallback_to_band")
                    self.assertFalse(config.loose_placement.enabled)
                    morphology = config.maskmix_v2.morphology
                    self.assertGreater(morphology.radius_max_px, config.interaction_dilation_px)
            weights = hybrid_cfg.maskmix_v2.mode_weights
            loose = weights.loose_blob + weights.oversized_box
            if domain == "fashion_tryon":
                self.assertEqual(loose, 0.0)
            else:
                self.assertGreater(loose, 0.0)
            if domain == "human":
                hair = cfg.data.init_args.category_conditioning["person_hair"].hybrid_masking
                self.assertEqual(
                    hair.maskmix_v2.mode_weights.loose_blob + hair.maskmix_v2.mode_weights.oversized_box, 0.0
                )


class MaskMixLooseSamplingTest(unittest.TestCase):
    def test_zero_loose_weights_keep_the_legacy_mode_mapping(self):
        legacy = {"exact": 0.2, "random_morphology": 0.2, "augmented_box": 0.3, "asymmetric_box": 0.3}
        config = hybrid({**legacy, "loose_blob": 0.0, "oversized_box": 0.0})
        order = (
            INSERT_ANYTHING_AUGMENTED_BOX,
            MASKMIX_V2_ASYMMETRIC_BOX,
            MASKMIX_V2_EXACT,
            MASKMIX_V2_RANDOM_MORPHOLOGY,
        )
        for seed in range(200):
            expected = _weighted_choice(random.Random(seed), order, tuple(legacy[name] for name in order))
            self.assertEqual(sample_maskmix_v2_mode("cat", config, random.Random(seed)), expected)

    def test_no_exact_distribution_samples_every_family_but_exact(self):
        config = hybrid()
        rng = random.Random(0)
        seen = {sample_maskmix_v2_mode("cat", config, rng) for _ in range(500)}
        self.assertEqual(
            seen,
            {
                INSERT_ANYTHING_AUGMENTED_BOX,
                MASKMIX_V2_ASYMMETRIC_BOX,
                MASKMIX_V2_RANDOM_MORPHOLOGY,
                LOOSE_BLOB,
                LOOSE_OVERSIZED_BOX,
            },
        )

    def test_loose_families_cannot_be_built_after_the_crop(self):
        config = hybrid()
        for mode in MASKMIX_LOOSE_MODES:
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "before adaptive crop"):
                build_maskmix_v2_mask_for_mode(blob_mask(), mode, config, random.Random(0))
        with self.assertRaisesRegex(ValueError, "precompute"):
            build_maskmix_v2_hybrid_masks(torch.from_numpy(blob_mask()).float()[None], ["cat"], config)

    def test_caller_fixes_the_loose_family(self):
        config = hybrid()
        boxes = build_loose_placement_mask(blob_mask(), config, random.Random(1), mode=LOOSE_OVERSIZED_BOX)
        self.assertEqual((boxes.mode, boxes.shape), (LOOSE_OVERSIZED_BOX, "box"))
        blobs = build_loose_placement_mask(blob_mask(), config, random.Random(1), mode=LOOSE_BLOB)
        self.assertEqual(blobs.mode, LOOSE_BLOB)
        self.assertNotEqual(blobs.shape, "box")
        with self.assertRaisesRegex(ValueError, "Unsupported loose placement mode"):
            build_loose_placement_mask(blob_mask(), config, random.Random(1), mode=MASKMIX_V2_EXACT)

    def test_guard_falls_back_to_the_banded_outline(self):
        config = hybrid(
            context_guard={"enabled": True, "max_condition_area_ratio": 0.3, "overflow_policy": "fallback_to_band"}
        )
        gt = torch.zeros(3, 40, 40)
        gt[:, 18:22, 18:22] = 1
        condition = torch.ones(3, 40, 40)
        guarded, metadata = apply_hybrid_masking_context_guard(
            condition, gt, config, allow_fallback=torch.tensor([True, False, True])
        )
        banded = add_interaction_context(gt, gt, 8)
        self.assertTrue(torch.equal(guarded[0], banded[0]))
        self.assertTrue(torch.equal(guarded[1], condition[1]))
        self.assertGreater(float(guarded[2].sum()), float(gt[2].sum()))
        self.assertEqual(metadata["fallback_applied"].tolist(), [True, False, True])


class MaskMixLooseDatasetTest(unittest.TestCase):
    def _build(self, root: Path) -> None:
        category = root / "cat"
        for name in ("ref_image", "ref_mask", "tar_image", "tar_mask"):
            (category / name).mkdir(parents=True)
        height, width = 384, 512
        image = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (260, 230), (36, 24), 0, 0, 360, 255, -1)
        ref_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ref_mask, (250, 200), (80, 60), 0, 0, 360, 255, -1)
        Image.fromarray(image).save(category / "ref_image" / "1.png")
        Image.fromarray(ref_mask).save(category / "ref_mask" / "1.png")
        Image.fromarray(image).save(category / "tar_image" / "1.png")
        Image.fromarray(mask).save(category / "tar_mask" / "1.png")

    def _config(self, root: Path) -> DatasetInitArgs:
        return DatasetInitArgs(
            dataset_path=str(root),
            categories=["cat"],
            require_prompt=False,
            processor={
                "class_path": "src.data.preprocess.ImageProcessor",
                "init_args": {
                    "process_type": "aspect_pixels",
                    "resize_mode": "bilinear",
                    "target_pixels": 256 * 256,
                    "controls_pixels": 128 * 128,
                },
            },
            adaptive_crop={
                "train_enabled": True,
                "trigger_bbox_area_ratio": 0.2,
                "min_crop_area_ratio": 0.125,
                "context_margin_ratio": 0.35,
            },
            hybrid_masking=hybrid().model_dump(),
        )

    def test_every_sample_carries_one_precomputed_condition_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            dataset = InsertAnythingDataset(self._config(root))
            modes = set()
            for _ in range(40):
                sample = dataset[0]
                gt, condition, mode = sample["tar_mask"], sample["tar_mask_cond"], sample["condition_mask_mode"]
                modes.add(mode)
                self.assertNotEqual(mode, MASKMIX_V2_EXACT)
                self.assertNotIn("tar_mask_cond_tight", sample)
                self.assertNotIn("placement_metadata", sample)
                self.assertEqual(condition.shape, gt.shape)
                self.assertTrue(torch.equal(torch.maximum(condition, gt), condition))
                if mode in MASKMIX_LOOSE_MODES:
                    # Planned on the loose region: the crop border never cuts it,
                    # and the interaction band adds nothing.
                    border = condition[0].sum() + condition[-1].sum() + condition[:, 0].sum() + condition[:, -1].sum()
                    self.assertEqual(float(border), 0.0)
                    self.assertTrue(torch.equal(add_interaction_context(condition[None], gt[None], 8)[0], condition))
                    self.assertGreater(float(condition.sum()), 1.5 * float(gt.sum()))
            self.assertTrue(modes & MASKMIX_LOOSE_MODES)
            self.assertTrue(modes - MASKMIX_LOOSE_MODES)


class MaskMixLooseTrainerTest(unittest.TestCase):
    def _trainer(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        config = hybrid(
            context_guard={"enabled": True, "max_condition_area_ratio": 0.3, "overflow_policy": "fallback_to_band"}
        )
        trainer.config = SimpleNamespace(data=SimpleNamespace(init_args=SimpleNamespace(hybrid_masking=config)))
        return trainer

    def test_dataset_loose_masks_skip_the_fallback_and_tight_ones_use_the_band(self):
        gt = torch.zeros(1, 40, 40)
        gt[:, 18:22, 18:22] = 1
        for mode, expect_fallback in ((LOOSE_BLOB, False), (MASKMIX_V2_RANDOM_MORPHOLOGY, True)):
            trainer = self._trainer()
            batch = {"tar_mask_cond": torch.ones(1, 40, 40), "condition_mask_mode": [mode], "category": ["cat"]}
            condition = trainer._build_condition_mask(batch, gt, stage="fit")
            expected = add_interaction_context(gt, gt, 8) if expect_fallback else torch.ones(1, 40, 40)
            with self.subTest(mode=mode):
                self.assertTrue(torch.equal(condition, expected))
                self.assertEqual(batch["hybrid_masking_modes"], [mode])
                record = trainer._hybrid_masking_metric_buffer[-1]
                self.assertEqual(record.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()

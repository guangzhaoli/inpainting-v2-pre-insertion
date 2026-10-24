"""Lighting augmentation: calibration, branch semantics, dataset and trainer wiring."""

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from pydantic import ValidationError

from src.data.config import (
    Config,
    DatasetInitArgs,
    HybridMaskingConfig,
    LightingAugmentationConfig,
    LightingRangeConfig,
    MaskControlConfig,
    ReferenceCFGConfig,
)
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.data.lighting_augmentation import (
    LUMA,
    Grade,
    LightingPlan,
    add_edge_spill,
    apply_grade,
    augment_reference,
    jitter_mask_edge,
    sample_grade,
    sample_lighting_plan,
    soft_shoulder,
    srgb_to_linear,
    warp_reference,
    white_balance_gains,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

ROOT = Path(__file__).resolve().parents[1]


def light_angle(gains) -> float:
    gains = np.asarray(gains, dtype=np.float64)
    return math.degrees(math.acos(min(1.0, gains.sum() / math.sqrt(3) / np.linalg.norm(gains))))


def reference_canvas(size=96):
    """A grey-brown object on the white background the reference cut-out leaves."""
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.ellipse(mask, (size // 2, size // 2), (size // 3, size // 4), 0, 0, 360, 1.0, -1)
    image = np.ones((size, size, 3), dtype=np.float32)
    image[mask > 0.5] = (0.45, 0.35, 0.25)
    return image, mask


def plan(branch="none", **flags) -> LightingPlan:
    grade = None if branch == "none" else Grade(delta_mired=60.0, tint=0.0, exposure_ev=0.5, gamma=1.0, saturation=1.0)
    values = dict(shading=False, edge=False, camera_reference=False, camera_scene=False, geometry=False)
    values.update(flags)
    return LightingPlan(branch=branch, grade=grade, **values)


class GradeTest(unittest.TestCase):
    def test_white_balance_follows_the_planckian_locus(self):
        np.testing.assert_allclose(white_balance_gains(0.0), np.ones(3), atol=1e-6)
        warm, cool = white_balance_gains(100.0), white_balance_gains(-100.0)
        self.assertTrue(warm[0] > 1.0 > warm[2] and cool[2] > 1.0 > cool[0])
        self.assertAlmostEqual(light_angle(warm), 20.8, delta=1.0)
        for gains in (warm, cool, white_balance_gains(40.0, tint=0.03)):
            self.assertAlmostEqual(float(LUMA @ gains), 1.0, places=5)

    def test_default_range_matches_the_eval_gap(self):
        rng = np.random.default_rng(0)
        grades = [sample_grade(LightingRangeConfig(), rng) for _ in range(4000)]
        angles = [light_angle(white_balance_gains(g.delta_mired, g.tint)) for g in grades]
        self.assertAlmostEqual(float(np.median(angles)), 4.8, delta=0.7)
        self.assertAlmostEqual(float(np.percentile(angles, 90)), 15.0, delta=1.5)
        self.assertLessEqual(max(abs(g.exposure_ev) for g in grades), 2.0)

    def test_grade_is_one_global_gain_in_linear_light(self):
        image = np.random.default_rng(1).uniform(0.05, 0.4, (16, 16, 3)).astype(np.float32)
        grade = Grade(delta_mired=50.0, tint=0.0, exposure_ev=-0.5, gamma=1.0, saturation=1.0)
        ratio = srgb_to_linear(apply_grade(image, grade)) / srgb_to_linear(image)
        np.testing.assert_allclose(ratio.reshape(-1, 3), np.broadcast_to(grade.gains, (256, 3)), rtol=2e-3)

    def test_brightening_compresses_highlights_instead_of_clipping(self):
        values = np.linspace(0.0, 4.0, 401, dtype=np.float32)
        shoulder = soft_shoulder(values)
        self.assertTrue(np.all(np.diff(shoulder) >= 0))
        self.assertTrue(np.all(np.diff(shoulder[values <= 1.5]) > 0))
        self.assertTrue(np.all(shoulder <= 1.0))
        np.testing.assert_array_equal(shoulder[values <= 0.8], values[values <= 0.8])


class ReferenceAugmentationTest(unittest.TestCase):
    def test_reference_branch_relights_only_the_object(self):
        image, mask = reference_canvas()
        out, out_mask = augment_reference(
            image, mask, plan("reference", shading=True), LightingAugmentationConfig(), np.random.default_rng(0)
        )
        np.testing.assert_array_equal(out_mask, mask)
        np.testing.assert_array_equal(out[mask < 0.5], 1.0)
        self.assertGreater(float(np.abs(out - image)[mask > 0.5].mean()), 0.02)

    def test_edge_jitter_moves_the_outline_by_a_few_pixels(self):
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.circle(mask, (200, 200), 150, 255, -1)
        area = float((mask > 127).sum())
        for seed in range(20):
            jittered = jitter_mask_edge(mask, np.random.default_rng(seed))
            change = abs(float((jittered > 127).sum()) - area)
            # Radius is 0.2–0.8% of the 301 px long side: 1–2 px of a 942 px rim.
            self.assertGreater(change, 0.0)
            self.assertLess(change, 3 * 942)

    def test_edges_spare_thin_parts(self):
        # A pendant on a 4 px chain loop: erosion trims the pendant and keeps the chain
        # attached; spill tints the pendant's rim, not the chain.
        pendant = np.zeros((600, 600), dtype=np.uint8)
        cv2.circle(pendant, (300, 420), 60, 255, -1)
        mask = pendant.copy()
        loop = np.array([[300, 360], [190, 250], [190, 90], [410, 90], [410, 250], [300, 360]], np.int32)
        cv2.polylines(mask, [loop], False, 255, 4)
        chain = (mask > 127) & (cv2.dilate(pendant, np.ones((21, 21), np.uint8)) == 0)
        eroded = 0
        for seed in range(20):
            jittered = jitter_mask_edge(mask, np.random.default_rng(seed)) > 127
            if jittered.sum() < (mask > 127).sum():
                eroded += 1
                self.assertTrue(jittered[chain].all())
                self.assertEqual(cv2.connectedComponents(jittered.astype(np.uint8))[0], 2)
                self.assertLess(int(jittered[pendant > 127].sum()), int((pendant > 127).sum()))
        self.assertGreater(eroded, 0)
        image = np.ones((600, 600, 3), dtype=np.float32)
        image[mask > 127] = (0.8, 0.6, 0.3)
        for seed in range(5):
            spilled = add_edge_spill(image, (mask > 127).astype(np.float32), np.random.default_rng(seed))
            np.testing.assert_array_equal(spilled[chain], image[chain])
            self.assertGreater(float(np.abs(spilled - image)[pendant > 127].max()), 0.02)

    def test_geometry_keeps_pixels_and_mask_aligned(self):
        image, mask = reference_canvas()
        warped, warped_mask = warp_reference(image, mask, np.random.default_rng(3), 8.0, 0.15, 0.05)
        self.assertFalse(np.array_equal(warped_mask, mask))
        # Bilinear pixels blend with white on the rim; two pixels in they are the object's.
        inside = cv2.erode((warped_mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.assertGreater(int(inside.sum()), 1000)
        np.testing.assert_allclose(warped[inside], np.broadcast_to((0.45, 0.35, 0.25), warped[inside].shape), atol=0.02)
        far = cv2.dilate((warped_mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
        np.testing.assert_allclose(warped[far], 1.0, atol=1e-6)

    def test_plans_follow_the_branch_probabilities(self):
        cfg = LightingAugmentationConfig(enabled=True, reference_prob=0.5, scene_prob=0.25)
        rng = np.random.default_rng(0)
        plans = [sample_lighting_plan(cfg, rng) for _ in range(4000)]
        rates = {branch: np.mean([p.branch == branch for p in plans]) for branch in ("reference", "scene", "none")}
        self.assertAlmostEqual(rates["reference"], 0.5, delta=0.03)
        self.assertAlmostEqual(rates["scene"], 0.25, delta=0.03)
        self.assertFalse(any(p.shading for p in plans if p.branch != "reference"))
        self.assertTrue(all(p.grade is None for p in plans if p.branch == "none"))

    def test_branch_probabilities_cannot_exceed_one(self):
        with self.assertRaises(ValidationError):
            LightingAugmentationConfig(reference_prob=0.8, scene_prob=0.3)


class LightingDatasetTest(unittest.TestCase):
    def _build(self, root: Path) -> None:
        category = root / "cat"
        for name in ("ref_image", "ref_mask", "tar_image", "tar_mask"):
            (category / name).mkdir(parents=True)
        height, width = 256, 320
        image = np.random.default_rng(0).integers(40, 200, (height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (170, 140), (60, 45), 0, 0, 360, 255, -1)
        Image.fromarray(image).save(category / "ref_image" / "1.png")
        Image.fromarray(mask).save(category / "ref_mask" / "1.png")
        Image.fromarray(image).save(category / "tar_image" / "1.png")
        Image.fromarray(mask).save(category / "tar_mask" / "1.png")

    def _dataset(self, root: Path, **lighting) -> InsertAnythingDataset:
        return InsertAnythingDataset(
            DatasetInitArgs(
                dataset_path=str(root),
                categories=["cat"],
                require_prompt=False,
                processor={
                    "class_path": "src.data.preprocess.ImageProcessor",
                    "init_args": {
                        "process_type": "aspect_pixels",
                        "resize_mode": "bilinear",
                        "target_pixels": 192 * 192,
                        "controls_pixels": 128 * 128,
                    },
                },
                lighting_augmentation=lighting,
            )
        )

    def _samples(self, root: Path, seed=0, **lighting):
        torch.manual_seed(seed)
        return self._dataset(root, **lighting)[0]

    def test_disabled_augmentation_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            sample = self._samples(root)
            for key in ("ref_image_clean", "ref_mask_clean", "lighting_metadata"):
                self.assertNotIn(key, sample)

    def test_reference_branch_keeps_a_clean_reference_and_the_target(self):
        quiet = dict(edge_prob=0.0, camera_reference_prob=0.0, camera_scene_prob=0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            base = self._samples(root, seed=5)
            sample = self._samples(root, seed=5, enabled=True, reference_prob=1.0, scene_prob=0.0, **quiet)
            self.assertEqual(sample["lighting_metadata"]["branch"], "reference")
            self.assertTrue(torch.equal(sample["ref_image_clean"], base["ref_image"]))
            self.assertTrue(torch.equal(sample["ref_mask_clean"], base["ref_mask"]))
            self.assertTrue(torch.equal(sample["tar_image"], base["tar_image"]))
            inside = sample["ref_mask"] > 0.5
            changed = (sample["ref_image"] - base["ref_image"]).abs().amax(0)
            self.assertGreater(float(changed[inside].mean()), 0.01)
            self.assertEqual(float(changed[~inside].max()), 0.0)

    def test_scene_branch_grades_the_whole_target_and_spares_the_reference(self):
        quiet = dict(edge_prob=0.0, camera_reference_prob=0.0, camera_scene_prob=0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            base = self._samples(root, seed=7)
            sample = self._samples(root, seed=7, enabled=True, reference_prob=0.0, scene_prob=1.0, **quiet)
            self.assertEqual(sample["lighting_metadata"]["branch"], "scene")
            self.assertTrue(torch.equal(sample["ref_image"], base["ref_image"]))
            self.assertTrue(torch.equal(sample["tar_mask"], base["tar_mask"]))
            changed = (sample["tar_image"] - base["tar_image"]).abs().amax(0)
            object_px = sample["tar_mask"] > 0.5
            self.assertGreater(float(changed[object_px].mean()), 0.005)
            self.assertGreater(float(changed[~object_px].mean()), 0.005)

    def test_edge_jitter_cuts_the_clean_reference_from_the_original_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            base = self._samples(root, seed=11)
            sample = self._samples(
                root, seed=11, enabled=True, reference_prob=0.0, scene_prob=0.0, edge_prob=1.0,
                camera_reference_prob=0.0, camera_scene_prob=0.0,
            )
            self.assertTrue(sample["lighting_metadata"]["edge"])
            self.assertTrue(torch.equal(sample["ref_image_clean"], base["ref_image"]))
            self.assertFalse(torch.equal(sample["ref_mask"], base["ref_mask"]))


def _passthrough(_self, batch, stage="fit"):
    return dict(batch)


class LightingTrainerTest(unittest.TestCase):
    def test_identity_losses_compare_against_the_clean_reference(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = SimpleNamespace(
            model=SimpleNamespace(
                mask_control=MaskControlConfig(enabled=False),
                reference_cfg=ReferenceCFGConfig(enabled=False, train_dropout_rate=0.0, force_empty_prompt=False),
            ),
            loss=SimpleNamespace(mask_loss=False, region_perceptual=True),
            data=SimpleNamespace(init_args=SimpleNamespace(hybrid_masking=HybridMaskingConfig(enabled=False))),
        )
        trainer.vae_scale_factor = 8
        height, width = 16, 16
        batch = {
            "ref_image": torch.rand(1, 3, height, width),
            "ref_mask": torch.ones(1, height, width),
            "ref_image_clean": torch.rand(1, 3, height, width),
            "ref_mask_clean": torch.ones(1, height, width),
            "tar_image": torch.rand(1, 3, height, width),
            "tar_mask": torch.ones(1, height, width),
            "prompt": ["object"],
            "category": ["cat"],
        }
        with patch.object(QwenImageEditPlusTrainer, "prepare_embeddings", autospec=True, side_effect=_passthrough):
            output = trainer.prepare_embeddings(batch, stage="fit")
        self.assertIs(output["ref_image_for_loss"], batch["ref_image_clean"])
        self.assertIs(output["ref_mask_for_loss"], batch["ref_mask_clean"])
        self.assertIs(output["control_1"], batch["ref_image"])


class LightingConfigTest(unittest.TestCase):
    def test_teacher_configs_enable_lighting_by_domain(self):
        settings = {}
        for domain in ("accessory_jewelry", "animal", "fashion_tryon", "general_object", "human"):
            raw = yaml.safe_load((ROOT / "configs/teacher_v2" / f"{domain}.yaml").read_text())
            settings[domain] = Config.model_validate(raw).data.init_args.lighting_augmentation
        self.assertTrue(all(cfg.enabled for cfg in settings.values()))
        for domain in ("animal", "general_object", "human"):
            self.assertGreater(settings[domain].reference_prob, settings[domain].scene_prob)
        for domain in ("fashion_tryon", "accessory_jewelry"):
            # Colour is the product: re-lighting comes mostly from the scene side.
            self.assertLess(settings[domain].reference_prob, settings[domain].scene_prob)
            self.assertLess(settings[domain].reference.wb_mired_scale, LightingRangeConfig().wb_mired_scale)
        self.assertGreater(settings["human"].geometry_prob, 0.0)
        self.assertEqual(sum(cfg.geometry_prob for name, cfg in settings.items() if name != "human"), 0.0)


if __name__ == "__main__":
    unittest.main()

import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from src.data.config import Config, DatasetInitArgs, HybridMaskingConfig
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.utils.hybrid_masking import (
    LOOSE_BLOB,
    LOOSE_OVERSIZED_BOX,
    MASKMIX_V2_EXACT,
    MASKMIX_V2_RANDOM_MORPHOLOGY,
    add_interaction_context,
    apply_hybrid_masking_context_guard,
    build_loose_placement_mask,
)


def loose_config(**loose_overrides) -> HybridMaskingConfig:
    return HybridMaskingConfig(
        enabled=True,
        strategy="maskmix_v2",
        interaction_dilation_px=8,
        context_guard={"enabled": True, "max_condition_area_ratio": 0.8},
        loose_placement={"enabled": True, **loose_overrides},
    )


def cat_mask(height=256, width=384, center=(190, 150), axes=(50, 30)) -> np.ndarray:
    """An ellipse body with a thin, tall ear: a union with it would leave a spike."""

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
    cx, cy = center
    mask[cy - axes[1] - 25 : cy, cx - 3 : cx + 3] = 1
    return mask


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(mask, kernel)


def bbox_hw(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return ys.max() - ys.min() + 1, xs.max() - xs.min() + 1


class LoosePlacementMaskTest(unittest.TestCase):
    def test_every_family_covers_the_band_without_tracing_the_gt(self):
        gt = cat_mask()
        target = dilate(gt, 8)
        seen = set()
        for seed in range(80):
            for reference in ("bbox_geometric_mean", "bbox_axes"):
                result = build_loose_placement_mask(
                    gt, loose_config(size_reference=reference), random.Random(seed), coverage_radius_px=8
                )
                seen.add(result.shape)
                self.assertTrue(np.all(result.mask >= target), (seed, reference, result.shape))
                self.assertIn(result.mode, {LOOSE_BLOB, LOOSE_OVERSIZED_BOX})
                # A union with the GT would leave the 6 px ear as a thin appendage;
                # opening removes thin parts, so the thick outline alone must cover it.
                opened = cv2.morphologyEx(result.mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
                self.assertTrue(np.all(opened >= gt), (seed, reference, result.shape))
                # Each axis is at least as large as the object (expansion is never < 1).
                mask_h, mask_w = bbox_hw(result.mask)
                gt_h, gt_w = bbox_hw(gt)
                self.assertGreaterEqual(mask_h, gt_h)
                self.assertGreaterEqual(mask_w, gt_w)
        self.assertEqual(seen, {"box", "ellipse", "superellipse", "spline_blob", "brush"})

    def test_oversized_box_is_an_exact_rectangle(self):
        gt = cat_mask()
        for seed in range(20):
            result = build_loose_placement_mask(
                gt,
                loose_config(mode_weights={"loose_blob": 0.0, "oversized_box": 1.0}),
                random.Random(seed),
                coverage_radius_px=8,
            )
            ys, xs = np.nonzero(result.mask)
            box = np.zeros_like(result.mask)
            box[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = 1
            np.testing.assert_array_equal(result.mask, box)

    def test_bottom_alignment_puts_the_flat_edge_on_the_contact_band(self):
        gt = cat_mask()
        target_bottom = np.nonzero(dilate(gt, 8))[0].max()
        for seed in range(30):
            result = build_loose_placement_mask(
                gt, loose_config(bottom_align_prob=1.0), random.Random(seed), coverage_radius_px=8
            )
            self.assertTrue(result.bottom_aligned)
            self.assertEqual(np.nonzero(result.mask)[0].max(), target_bottom)
        bottoms = {
            np.nonzero(
                build_loose_placement_mask(
                    gt, loose_config(bottom_align_prob=0.0), random.Random(seed), coverage_radius_px=8
                ).mask
            )[0].max()
            for seed in range(30)
        }
        self.assertGreater(len(bottoms), 5)

    def test_geometric_mean_reference_decouples_mask_and_object_aspect(self):
        rng = np.random.default_rng(0)
        correlations = {}
        for reference in ("bbox_geometric_mean", "bbox_axes"):
            object_aspects, mask_aspects = [], []
            for seed in range(160):
                aspect = float(np.exp(rng.normal(0.0, 0.5)))
                half_h, half_w = 60 * np.sqrt(aspect), 60 / np.sqrt(aspect)
                gt = np.zeros((1024, 1024), dtype=np.uint8)
                cv2.ellipse(gt, (512, 512), (int(half_w), int(half_h)), 0, 0, 360, 1, -1)
                result = build_loose_placement_mask(
                    gt,
                    loose_config(size_reference=reference, mode_weights={"loose_blob": 0.0, "oversized_box": 1.0}),
                    random.Random(seed),
                )
                gt_h, gt_w = bbox_hw(gt)
                mask_h, mask_w = bbox_hw(result.mask)
                object_aspects.append(np.log(gt_h / gt_w))
                mask_aspects.append(np.log(mask_h / mask_w))
            correlations[reference] = float(np.corrcoef(object_aspects, mask_aspects)[0, 1])
        self.assertLess(abs(correlations["bbox_geometric_mean"]), 0.2, correlations)
        self.assertGreater(correlations["bbox_axes"], 0.6, correlations)

    def test_guard_shrinks_the_expansion_instead_of_falling_back_to_exact(self):
        gt = np.zeros((480, 640), dtype=np.uint8)
        cv2.ellipse(gt, (320, 240), (150, 100), 0, 0, 360, 1, -1)
        shrunk = 0
        for seed in range(30):
            result = build_loose_placement_mask(gt, loose_config(), random.Random(seed), max_area_ratio=0.45)
            self.assertTrue(np.all(result.mask >= gt))
            self.assertGreater(result.mask.sum(), 1.05 * gt.sum())
            self.assertFalse(result.guard_exhausted)
            self.assertLessEqual(result.area_ratio, 0.45)
            shrunk += result.guard_shrinks > 0
        self.assertGreater(shrunk, 0)

        huge = np.zeros((200, 200), dtype=np.uint8)
        huge[10:190, 10:190] = 1
        result = build_loose_placement_mask(huge, loose_config(), random.Random(0), max_area_ratio=0.5)
        self.assertTrue(result.guard_exhausted)
        self.assertEqual(result.guard_shrinks, 8)
        self.assertTrue(np.all(result.mask >= huge))
        self.assertGreater(result.mask.sum(), huge.sum())

    def test_guard_area_is_measured_in_the_planned_canvas(self):
        gt = np.zeros((400, 400), dtype=np.uint8)
        gt[180:220, 180:220] = 1
        canvas = (150, 150, 250, 250)
        result = build_loose_placement_mask(
            gt, loose_config(), random.Random(3), canvas_fn=lambda _mask: canvas, max_area_ratio=0.6
        )
        x1, y1, x2, y2 = canvas
        expected = result.mask[y1:y2, x1:x2].sum() / ((y2 - y1) * (x2 - x1))
        self.assertAlmostEqual(result.area_ratio, float(expected))

    def test_trainer_guard_keeps_loose_masks_that_opt_out_of_fallback(self):
        config = HybridMaskingConfig(
            enabled=True,
            strategy="maskmix_v2",
            context_guard={"enabled": True, "max_condition_area_ratio": 0.5},
            loose_placement={"enabled": True},
        )
        gt = torch.zeros(2, 10, 10)
        gt[:, 4:6, 4:6] = 1
        loose = torch.ones(2, 10, 10)
        guarded, metadata = apply_hybrid_masking_context_guard(
            loose, gt, config, allow_fallback=torch.tensor([False, True])
        )
        self.assertTrue(torch.equal(guarded[0], loose[0]))
        self.assertTrue(torch.equal(guarded[1], gt[1]))
        self.assertTrue(bool(metadata["context_guard_triggered"].all()))
        self.assertEqual(metadata["fallback_applied"].tolist(), [False, True])


class LoosePlacementDatasetTest(unittest.TestCase):
    def _build(self, root: Path) -> None:
        category = root / "cat"
        for name in ("ref_image", "ref_mask", "tar_image", "tar_mask"):
            (category / name).mkdir(parents=True)
        height, width = 384, 512
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (260, 230), (36, 24), 0, 0, 360, 255, -1)
        ref_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(ref_mask, (250, 200), (80, 60), 0, 0, 360, 255, -1)
        Image.fromarray(image).save(category / "ref_image" / "1.png")
        Image.fromarray(ref_mask).save(category / "ref_mask" / "1.png")
        Image.fromarray(image).save(category / "tar_image" / "1.png")
        Image.fromarray(mask).save(category / "tar_mask" / "1.png")

    def _config(self, root: Path, loose_enabled: bool) -> DatasetInitArgs:
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
            hybrid_masking={
                "enabled": True,
                "strategy": "maskmix_v2",
                "interaction_dilation_px": 8,
                "context_guard": {"enabled": True},
                "loose_placement": {"enabled": loose_enabled},
            },
        )

    def test_dataset_emits_loose_and_tight_masks_in_one_crop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            dataset = InsertAnythingDataset(self._config(root, loose_enabled=True))
            for _ in range(12):
                sample = dataset[0]
                gt = sample["tar_mask"]
                loose = sample["tar_mask_cond"]
                tight = sample["tar_mask_cond_tight"]
                self.assertEqual(loose.shape, gt.shape)
                self.assertEqual(tight.shape, gt.shape)
                # The trainer's coverage union and 8 px interaction band add nothing.
                self.assertTrue(torch.equal(torch.maximum(loose, gt), loose))
                banded = add_interaction_context(loose[None], gt[None], 8)[0]
                self.assertTrue(torch.equal(banded, loose))
                self.assertTrue(torch.all(tight >= gt))
                self.assertIn(sample["placement_metadata"]["tight_mode"], {MASKMIX_V2_EXACT, MASKMIX_V2_RANDOM_MORPHOLOGY})
                # The crop was planned on the loose region, so it is not cut by the crop border.
                self.assertEqual(float(loose[0].sum() + loose[-1].sum() + loose[:, 0].sum() + loose[:, -1].sum()), 0.0)
                self.assertGreater(float(loose.sum()), 1.5 * float(gt.sum()))

    def test_disabled_loose_placement_keeps_the_legacy_sample_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            sample = InsertAnythingDataset(self._config(root, loose_enabled=False))[0]
            for key in ("tar_mask_cond", "tar_mask_cond_tight", "placement_metadata"):
                self.assertNotIn(key, sample)


class SigmaSplitConfigTest(unittest.TestCase):
    def _payload(self, **overrides):
        payload = {
            "trainer": "QwenImageEditInsertAnything",
            "model": {"alpha_head": {"enabled": True}, "sigma_split": {"enabled": True}},
            "data": {
                "batch_size": 1,
                "init_args": {
                    "hybrid_masking": {
                        "enabled": True,
                        "strategy": "maskmix_v2",
                        "loose_placement": {"enabled": True},
                    }
                },
            },
            "loss": {"alpha_bce_weight": 0.1},
            "predict": {"sigma_handoff": {"enabled": True}},
        }
        for path, value in overrides.items():
            node = payload
            keys = path.split(".")
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value
        return payload

    def test_valid_sigma_split_contract(self):
        config = Config.model_validate(self._payload())
        self.assertTrue(config.model.sigma_split.enabled)
        self.assertEqual(config.model.sigma_split.tau, 0.7)
        self.assertEqual(config.data.init_args.hybrid_masking.loose_placement.scale_max, 2.5)

    def test_defaults_are_disabled(self):
        config = Config()
        self.assertFalse(config.model.sigma_split.enabled)
        self.assertFalse(config.model.alpha_head.enabled)
        self.assertFalse(config.data.init_args.hybrid_masking.loose_placement.enabled)
        self.assertFalse(config.predict.sigma_handoff.enabled)
        self.assertEqual(config.loss.ref_dino_mode, "cosine")

    def test_inconsistent_contracts_are_rejected(self):
        invalid = {
            "loose without routing": {"model.sigma_split": {"enabled": False}, "predict.sigma_handoff": {"enabled": False}},
            "routing without loose": {"data.init_args.hybrid_masking.loose_placement": {"enabled": False}},
            "alpha loss without head": {"model.alpha_head": {"enabled": False}, "predict.sigma_handoff": {"enabled": False}},
            "multi-sample batch": {"data.batch_size": 2},
            "band wider than tau": {"model.sigma_split": {"enabled": True, "tau": 0.7, "overlap": 0.4}},
            "bad scale range": {
                "data.init_args.hybrid_masking.loose_placement": {"enabled": True, "scale_min": 2.0, "scale_max": 1.5}
            },
        }
        for name, overrides in invalid.items():
            with self.subTest(name), self.assertRaises(ValidationError):
                Config.model_validate(self._payload(**overrides))


if __name__ == "__main__":
    unittest.main()

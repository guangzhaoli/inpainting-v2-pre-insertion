"""Opt-in regression for the four observed full-frame Hybrid Masking samples."""

import os
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from src.data.config import HybridMaskingConfig
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_EXPANDED_BOX,
    MASKMIX_V2_ASYMMETRIC_BOX,
    MASKMIX_V2_EXACT,
    MASKMIX_V2_RANDOM_MORPHOLOGY,
    apply_hybrid_masking_context_guard,
    build_insert_anything_mask_for_mode,
    build_maskmix_v2_mask_for_mode,
)


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "a2edit/data"
RUN_REAL_DATA_TESTS = os.environ.get("RUN_A2EDIT_REAL_DATA_TESTS") == "1"


def make_maskmix_v2_config():
    return SimpleNamespace(
        strategy="maskmix_v2",
        mask_threshold=0.5,
        ensure_gt_coverage=True,
        scale_pixel_parameters=True,
        reference_resolution=768,
        context_guard=SimpleNamespace(
            enabled=True,
            max_condition_area_ratio=0.80,
            overflow_policy="fallback_to_gt",
        ),
        maskmix_v2=SimpleNamespace(
            mode_weights=SimpleNamespace(
                augmented_box=0.10,
                asymmetric_box=0.10,
                exact=0.60,
                random_morphology=0.20,
            ),
            asymmetric_box=SimpleNamespace(margin_ratio_min=0.0, margin_ratio_max=0.10),
            augmented_box=SimpleNamespace(
                jitter_object_scale_min=0.05,
                jitter_object_scale_max=0.20,
                jitter_min_px=2,
                jitter_max_px=64,
            ),
            morphology=SimpleNamespace(
                radius_object_scale_min=0.02,
                radius_object_scale_max=0.12,
                radius_min_px=1,
                radius_max_px=32,
                anisotropy_min=1.5,
                anisotropy_max=3.0,
                kernel_weights=SimpleNamespace(
                    ellipse=0.40,
                    rectangle=0.20,
                    horizontal_ellipse=0.20,
                    vertical_ellipse=0.20,
                ),
            ),
        ),
    )


@unittest.skipUnless(
    RUN_REAL_DATA_TESTS,
    "set RUN_A2EDIT_REAL_DATA_TESTS=1 to exercise local A2Edit masks",
)
class HybridMaskingRealDataRegressionTest(unittest.TestCase):
    SAMPLE_STEMS = ("2591", "3972", "4082", "4250", "969")

    def test_high_risk_expanded_boxes_fall_back_to_gt(self):
        data_root = Path(os.environ.get("A2EDIT_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        config = HybridMaskingConfig(
            enabled=True,
            context_guard={
                "enabled": True,
                "max_condition_area_ratio": 0.80,
                "overflow_policy": "fallback_to_gt",
            },
        )
        expected_ratios = {
            # 2591 is the second real sample whose GT itself exceeds the guard.
            "2591": (0.8009936556927297, 0.9155092592592593, True),
            "3972": (0.8431420940170941, 1.0, True),
            "4082": (0.481142735042735, 1.0, False),
            "4250": (0.6705228632478633, 1.0, False),
            "969": (0.5273673076923077, 1.0, False),
        }

        for stem, (expected_gt_ratio, expected_expanded_ratio, unavoidable) in expected_ratios.items():
            with self.subTest(sample_id=f"accessory/{stem}"):
                mask_path = data_root / "accessory" / "tar_mask" / f"{stem}.png"
                gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                self.assertIsNotNone(gt_mask, f"failed to read {mask_path}")

                expanded = build_insert_anything_mask_for_mode(
                    gt_mask,
                    INSERT_ANYTHING_EXPANDED_BOX,
                    config,
                )
                self.assertAlmostEqual(
                    float(expanded.mean()),
                    expected_expanded_ratio,
                    places=12,
                )

                gt_tensor = torch.from_numpy((gt_mask > 127).astype("float32")).unsqueeze(0)
                expanded_tensor = torch.from_numpy(expanded.astype("float32")).unsqueeze(0)
                guarded, metadata = apply_hybrid_masking_context_guard(
                    expanded_tensor,
                    gt_tensor,
                    config,
                )

                torch.testing.assert_close(guarded, gt_tensor)
                self.assertTrue(bool(metadata["context_guard_triggered"][0]))
                self.assertEqual(
                    bool(metadata["unavoidable_large_gt"][0]),
                    unavoidable,
                )
                self.assertAlmostEqual(
                    float(metadata["gt_area_ratio"][0]),
                    expected_gt_ratio,
                    places=12,
                )
                self.assertAlmostEqual(
                    float(metadata["condition_area_ratio"][0]),
                    expected_gt_ratio,
                    places=12,
                )

    def test_maskmix_v2_families_are_leakage_safe_on_real_masks(self):
        data_root = Path(os.environ.get("A2EDIT_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        config = make_maskmix_v2_config()
        modes = (
            INSERT_ANYTHING_AUGMENTED_BOX,
            MASKMIX_V2_ASYMMETRIC_BOX,
            MASKMIX_V2_EXACT,
            MASKMIX_V2_RANDOM_MORPHOLOGY,
        )

        for stem in self.SAMPLE_STEMS:
            mask_path = data_root / "accessory" / "tar_mask" / f"{stem}.png"
            gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(gt_mask, f"failed to read {mask_path}")
            gt_binary = (gt_mask > 127).astype(np.uint8)
            gt_tensor = torch.from_numpy(gt_binary.astype("float32")).unsqueeze(0)

            for mode_index, mode in enumerate(modes):
                with self.subTest(sample_id=f"accessory/{stem}", mode=mode):
                    condition = build_maskmix_v2_mask_for_mode(
                        gt_mask,
                        mode,
                        config,
                        rng=random.Random(1000 + int(stem) + mode_index),
                    )
                    self.assertTrue(set(np.unique(condition)).issubset({0, 1}))
                    self.assertTrue(np.all(condition[gt_binary > 0] == 1))

                    guarded, metadata = apply_hybrid_masking_context_guard(
                        torch.from_numpy(condition.astype("float32")).unsqueeze(0),
                        gt_tensor,
                        config,
                    )
                    self.assertTrue(torch.all(guarded >= gt_tensor).item())
                    self.assertGreaterEqual(
                        float(metadata["sampled_condition_expansion"][0]), 1.0
                    )
                    self.assertGreaterEqual(
                        float(metadata["condition_expansion"][0]), 1.0
                    )


if __name__ == "__main__":
    unittest.main()

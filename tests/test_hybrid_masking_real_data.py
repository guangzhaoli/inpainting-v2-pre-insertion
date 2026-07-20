"""Opt-in regression for the four observed full-frame Hybrid Masking samples."""

import os
import unittest
from pathlib import Path

import cv2
import torch

from src.data.config import HybridMaskingConfig
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_EXPANDED_BOX,
    apply_hybrid_masking_context_guard,
    build_insert_anything_mask_for_mode,
)


DEFAULT_DATA_ROOT = Path("/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data")
RUN_REAL_DATA_TESTS = os.environ.get("RUN_A2EDIT_REAL_DATA_TESTS") == "1"


@unittest.skipUnless(
    RUN_REAL_DATA_TESTS,
    "set RUN_A2EDIT_REAL_DATA_TESTS=1 to exercise local A2Edit masks",
)
class HybridMaskingRealDataRegressionTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

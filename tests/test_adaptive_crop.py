import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from pydantic import ValidationError

from infer import validate_adaptive_crop_inference_args
from src.data.config import AdaptiveCropConfig
from src.utils.adaptive_crop import (
    build_adaptive_crop_plan,
    crop_numpy_pair,
    crop_pil_pair,
    restore_pil_output,
)


class AdaptiveCropTests(unittest.TestCase):
    def setUp(self):
        self.config = AdaptiveCropConfig(
            train_enabled=True,
            predict_enabled=True,
            trigger_bbox_area_ratio=0.10,
            min_crop_area_ratio=0.25,
            context_margin_ratio=0.20,
        )

    def test_small_center_mask_produces_aspect_preserving_crop(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[35:45, 45:55] = 255
        plan = build_adaptive_crop_plan((100, 80), mask, self.config)

        self.assertTrue(plan.applied)
        x1, y1, x2, y2 = plan.crop_box
        bx1, by1, bx2, by2 = plan.mask_bbox
        self.assertLessEqual(x1, bx1)
        self.assertLessEqual(y1, by1)
        self.assertGreaterEqual(x2, bx2)
        self.assertGreaterEqual(y2, by2)
        self.assertAlmostEqual((x2 - x1) / (y2 - y1), 100 / 80, delta=0.04)
        self.assertGreater(plan.effective_zoom_x, 1.0)

    def test_edge_mask_shifts_crop_inside_image(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[0:8, 0:8] = 255
        plan = build_adaptive_crop_plan((100, 80), mask, self.config)
        self.assertEqual(plan.crop_box[0], 0)
        self.assertEqual(plan.crop_box[1], 0)
        self.assertGreaterEqual(plan.crop_box[2], 8)
        self.assertGreaterEqual(plan.crop_box[3], 8)

    def test_large_bbox_uses_explicit_full_frame_branch(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[10:70, 10:90] = 255
        plan = build_adaptive_crop_plan((100, 80), mask, self.config)
        self.assertFalse(plan.applied)
        self.assertEqual(plan.decision_reason, "mask_bbox_large_enough")
        self.assertEqual(plan.crop_box, (0, 0, 100, 80))

    def test_empty_mask_raises(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_adaptive_crop_plan((100, 80), mask, self.config)

    def test_bbox_detection_does_not_materialize_all_foreground_coordinates(self):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[10:70, 10:90] = 255
        with patch(
            "src.utils.adaptive_crop.np.where",
            side_effect=AssertionError("np.where must not be used for bbox extraction"),
        ):
            plan = build_adaptive_crop_plan((100, 80), mask, self.config)
        self.assertEqual(plan.mask_bbox, (10, 10, 90, 70))

    def test_minimum_crop_side_is_enforced(self):
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[11:13, 11:13] = 255
        plan = build_adaptive_crop_plan((24, 24), mask, self.config)
        x1, y1, x2, y2 = plan.crop_box
        self.assertGreaterEqual(x2 - x1, self.config.min_crop_side)
        self.assertGreaterEqual(y2 - y1, self.config.min_crop_side)

    def test_source_smaller_than_minimum_crop_side_raises(self):
        mask = np.ones((15, 24), dtype=np.uint8) * 255
        with self.assertRaisesRegex(ValueError, "min_crop_side=16"):
            build_adaptive_crop_plan((24, 15), mask, self.config)

    def test_minimum_crop_side_must_be_positive(self):
        with self.assertRaises(ValidationError):
            AdaptiveCropConfig(min_crop_side=0)

    def test_mask_threshold_one_is_rejected(self):
        with self.assertRaises(ValidationError):
            AdaptiveCropConfig(mask_threshold=1.0)

    def test_predict_crop_rejects_explicit_output_size_before_model_setup(self):
        config = SimpleNamespace(
            data=SimpleNamespace(
                init_args=SimpleNamespace(
                    adaptive_crop=AdaptiveCropConfig(predict_enabled=True),
                )
            )
        )
        args = SimpleNamespace(height=768, width=None)
        with self.assertRaisesRegex(ValueError, "explicit --height/--width"):
            validate_adaptive_crop_inference_args(config, args)

    def test_geometry_mismatch_raises(self):
        mask = np.zeros((79, 100), dtype=np.uint8)
        mask[30:40, 45:55] = 255
        with self.assertRaisesRegex(ValueError, "identical geometry"):
            build_adaptive_crop_plan((100, 80), mask, self.config)

    def test_numpy_and_pil_crop_use_identical_box(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[35:45, 45:55] = 255
        plan = build_adaptive_crop_plan((100, 80), mask, self.config)

        np_image, np_mask = crop_numpy_pair(image, mask, plan)
        pil_image, pil_mask = crop_pil_pair(Image.fromarray(image), Image.fromarray(mask), plan)
        self.assertEqual(np_image.shape[1::-1], pil_image.size)
        self.assertEqual(np_mask.shape[1::-1], pil_mask.size)
        np.testing.assert_array_equal(np_mask, np.asarray(pil_mask))

    def test_numpy_and_pil_crop_masks_use_plan_threshold(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[35:37, 45:47] = 64
        mask[39:41, 49:51] = 192
        plan = build_adaptive_crop_plan((100, 80), mask, self.config)

        _, np_mask = crop_numpy_pair(image, mask, plan)
        _, pil_mask = crop_pil_pair(Image.fromarray(image), Image.fromarray(mask), plan)

        np.testing.assert_array_equal(np_mask, np.asarray(pil_mask))
        self.assertEqual(set(np.unique(np_mask).tolist()), {0, 255})

    def test_restore_strict_composite_preserves_mask_outside(self):
        source_array = np.zeros((80, 100, 3), dtype=np.uint8)
        source_array[:, :] = (255, 0, 0)
        mask_array = np.zeros((80, 100), dtype=np.uint8)
        mask_array[35:45, 45:55] = 255
        source = Image.fromarray(source_array)
        mask = Image.fromarray(mask_array)
        plan = build_adaptive_crop_plan(source.size, mask, self.config)
        local_output = Image.new("RGB", (64, 64), (0, 255, 0))

        restored, composite = restore_pil_output(
            source,
            mask,
            local_output,
            plan,
            blend_mode=self.config.blend_mode,
        )
        self.assertEqual(restored.size, source.size)
        composite_array = np.asarray(composite)
        self.assertTrue(np.all(composite_array[0, 0] == np.array([255, 0, 0])))
        self.assertTrue(np.all(composite_array[40, 50] == np.array([0, 255, 0])))

    def test_restore_strict_composite_uses_binary_crop_threshold(self):
        source = Image.new("RGB", (100, 80), (255, 0, 0))
        mask_array = np.zeros((80, 100), dtype=np.uint8)
        mask_array[35:37, 45:47] = 64
        mask_array[39:41, 49:51] = 192
        mask = Image.fromarray(mask_array)
        plan = build_adaptive_crop_plan(source.size, mask, self.config)
        local_output = Image.new("RGB", (64, 64), (0, 255, 0))

        restored, composite = restore_pil_output(
            source,
            mask,
            local_output,
            plan,
            blend_mode=self.config.blend_mode,
        )

        restored_array = np.asarray(restored)
        composite_array = np.asarray(composite)
        self.assertTrue(np.all(restored_array[35, 45] == np.array([0, 255, 0])))
        self.assertTrue(np.all(composite_array[35, 45] == np.array([255, 0, 0])))
        self.assertTrue(np.all(composite_array[39, 49] == np.array([0, 255, 0])))

    def test_multichannel_numpy_mask_is_rejected(self):
        mask = np.zeros((80, 100, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "single-channel 2D mask"):
            build_adaptive_crop_plan((100, 80), mask, self.config)

    def test_out_of_range_float_mask_is_rejected(self):
        mask = np.zeros((80, 100), dtype=np.float32)
        mask[35:45, 45:55] = 255.0
        with self.assertRaisesRegex(ValueError, "finite values in \\[0, 1\\]"):
            build_adaptive_crop_plan((100, 80), mask, self.config)


if __name__ == "__main__":
    unittest.main()

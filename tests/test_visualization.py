import unittest

import numpy as np
from PIL import Image

from src.utils.visualization import (
    binarize_mask,
    build_expanded_soft_mask,
    build_expanded_soft_mask_composite,
    build_strict_mask_composite,
)


class StrictMaskCompositeTest(unittest.TestCase):
    def test_binarize_mask_matches_the_applied_strict_threshold(self):
        mask = Image.fromarray(np.array([[0, 127, 128]], dtype=np.uint8), mode="L")

        binary = binarize_mask(mask, mask_threshold=0.5)

        np.testing.assert_array_equal(
            np.asarray(binary),
            np.array([[0, 0, 255]], dtype=np.uint8),
        )

    def test_binary_zero_one_mask_uses_the_same_threshold_semantics_as_adaptive_crop(self):
        source = Image.new("RGB", (2, 1), color=(255, 0, 0))
        output = Image.new("RGB", (2, 1), color=(0, 255, 0))
        mask = Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="L")

        composite = build_strict_mask_composite(source, mask, output, mask_threshold=0.5)

        np.testing.assert_array_equal(
            np.asarray(composite),
            np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8),
        )

    def test_grayscale_mask_is_binarized_without_alpha_blending(self):
        source = Image.new("RGB", (3, 1), color=(255, 0, 0))
        output = Image.new("RGB", (3, 1), color=(0, 255, 0))
        mask = Image.fromarray(np.array([[0, 127, 128]], dtype=np.uint8), mode="L")

        composite = build_strict_mask_composite(source, mask, output, mask_threshold=0.5)

        np.testing.assert_array_equal(
            np.asarray(composite),
            np.array(
                [[[255, 0, 0], [255, 0, 0], [0, 255, 0]]],
                dtype=np.uint8,
            ),
        )


class ExpandedSoftMaskCompositeTest(unittest.TestCase):
    def test_dilation_fills_a_small_hole(self):
        mask_array = np.zeros((11, 11), dtype=np.uint8)
        mask_array[3:8, 3:8] = 255
        mask_array[5, 5] = 0

        soft_mask = build_expanded_soft_mask(
            Image.fromarray(mask_array, mode="L"),
            mask_threshold=0.5,
            dilation_px=1,
            transition_width_px=3,
        )

        self.assertEqual(int(np.asarray(soft_mask)[5, 5]), 255)

    def test_protected_region_is_unchanged(self):
        source = Image.new("RGB", (17, 17), color=(10, 20, 30))
        output = Image.new("RGB", (17, 17), color=(210, 220, 230))
        mask_array = np.zeros((17, 17), dtype=np.uint8)
        mask_array[8, 8] = 255

        composite, soft_mask = build_expanded_soft_mask_composite(
            source,
            Image.fromarray(mask_array, mode="L"),
            output,
            mask_threshold=0.5,
            dilation_px=1,
            transition_width_px=3,
        )

        np.testing.assert_array_equal(np.asarray(composite)[0, 0], np.array([10, 20, 30]))
        self.assertEqual(int(np.asarray(soft_mask)[0, 0]), 0)

    def test_cosine_transition_contains_mixed_weights(self):
        source = Image.new("RGB", (15, 15), color=(255, 0, 0))
        output = Image.new("RGB", (15, 15), color=(0, 255, 0))
        mask_array = np.zeros((15, 15), dtype=np.uint8)
        mask_array[7, 7] = 255

        composite, soft_mask = build_expanded_soft_mask_composite(
            source,
            Image.fromarray(mask_array, mode="L"),
            output,
            mask_threshold=0.5,
            dilation_px=0,
            transition_width_px=4,
        )

        soft_array = np.asarray(soft_mask)
        mixed = (soft_array > 0) & (soft_array < 255)
        self.assertTrue(bool(mixed.any()))
        mixed_pixel = np.asarray(composite)[mixed][0]
        self.assertGreater(int(mixed_pixel[0]), 0)
        self.assertLess(int(mixed_pixel[0]), 255)
        self.assertGreater(int(mixed_pixel[1]), 0)
        self.assertLess(int(mixed_pixel[1]), 255)
        self.assertEqual(int(mixed_pixel[2]), 0)

    def test_parameters_are_validated(self):
        mask = Image.new("L", (5, 5), color=255)

        with self.assertRaisesRegex(ValueError, "dilation_px must be >= 0"):
            build_expanded_soft_mask(mask, 0.5, dilation_px=-1, transition_width_px=2)
        with self.assertRaisesRegex(ValueError, "transition_width_px must be >= 2"):
            build_expanded_soft_mask(mask, 0.5, dilation_px=0, transition_width_px=1)
        with self.assertRaisesRegex(TypeError, "dilation_px must be an integer"):
            build_expanded_soft_mask(mask, 0.5, dilation_px=1.5, transition_width_px=2)
        with self.assertRaisesRegex(ValueError, "mask_threshold must be in"):
            build_expanded_soft_mask(mask, 1.0, dilation_px=0, transition_width_px=2)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import torch

from src.models.mask_geometry import MaskGeometryTokenizer, build_mask_geometry


class BuildMaskGeometryTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(2, 9, 9)
        self.mask[:, 2:7, 2:7] = 1

    def test_bhw_and_b1hw_have_identical_three_channel_geometry(self):
        from_bhw = build_mask_geometry(self.mask, sigma_px=2.0)
        from_b1hw = build_mask_geometry(self.mask.unsqueeze(1), sigma_px=2.0)

        self.assertEqual(tuple(from_bhw.shape), (2, 3, 9, 9))
        self.assertEqual(from_bhw.dtype, torch.float32)
        torch.testing.assert_close(from_bhw, from_b1hw, rtol=0, atol=0)
        torch.testing.assert_close(from_bhw[:, 0], self.mask, rtol=0, atol=0)

    def test_signed_distance_and_boundary_ring_encode_geometry(self):
        geometry = build_mask_geometry(self.mask[:1], sigma_px=2.0)
        signed = geometry[0, 1]
        ring = geometry[0, 2]

        self.assertGreater(float(signed[4, 4]), float(signed[2, 2]))
        self.assertGreater(float(signed[2, 2]), 0.0)
        self.assertLess(float(signed[1, 2]), 0.0)
        self.assertLess(float(signed[0, 0]), float(signed[1, 2]))
        self.assertGreater(float(ring[2, 2]), float(ring[4, 4]))
        self.assertGreater(float(ring[1, 2]), float(ring[0, 0]))
        torch.testing.assert_close(ring[2, 2], ring[1, 2], rtol=0, atol=1e-6)

        expected_boundary_signed = math.tanh(0.5 / 2.0)
        expected_boundary_ring = math.exp(-0.5 / 2.0)
        self.assertAlmostEqual(float(signed[2, 2]), expected_boundary_signed, places=6)
        self.assertAlmostEqual(float(signed[1, 2]), -expected_boundary_signed, places=6)
        self.assertAlmostEqual(float(ring[2, 2]), expected_boundary_ring, places=6)
        self.assertLess(float(ring.max()), 1.0)

    def test_single_pixel_and_thin_line_keep_half_pixel_inside_distance(self):
        masks = torch.zeros(2, 7, 7)
        masks[0, 3, 3] = 1
        masks[1, :, 3] = 1

        geometry = build_mask_geometry(masks, sigma_px=1.0)
        expected_signed = math.tanh(0.5)
        expected_ring = math.exp(-0.5)

        self.assertAlmostEqual(float(geometry[0, 1, 3, 3]), expected_signed, places=6)
        self.assertAlmostEqual(float(geometry[0, 1, 3, 2]), -expected_signed, places=6)
        self.assertAlmostEqual(float(geometry[0, 2, 3, 3]), expected_ring, places=6)
        self.assertAlmostEqual(float(geometry[0, 2, 3, 2]), expected_ring, places=6)
        torch.testing.assert_close(
            geometry[1, 1, :, 3],
            torch.full((7,), expected_signed),
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            geometry[1, 1, :, 2],
            torch.full((7,), -expected_signed),
            rtol=0,
            atol=1e-6,
        )

    def test_small_region_has_deeper_positive_interior(self):
        mask = torch.zeros(1, 7, 7)
        mask[:, 2:5, 2:5] = 1
        geometry = build_mask_geometry(mask, sigma_px=1.0)

        expected_center = math.tanh(1.5)
        expected_boundary = math.tanh(0.5)
        self.assertAlmostEqual(float(geometry[0, 1, 3, 3]), expected_center, places=6)
        self.assertAlmostEqual(float(geometry[0, 1, 2, 2]), expected_boundary, places=6)
        self.assertGreater(float(geometry[0, 1, 3, 3]), float(geometry[0, 1, 2, 2]))

    def test_complement_is_antisymmetric_including_edge_touching_mask(self):
        mask = torch.zeros(1, 7, 8)
        mask[:, :, :3] = 1
        mask[:, 2:5, 3:6] = 1

        geometry = build_mask_geometry(mask, sigma_px=2.5)
        complement = build_mask_geometry(1 - mask, sigma_px=2.5)

        torch.testing.assert_close(complement[:, 0], 1 - geometry[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(complement[:, 1], -geometry[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(complement[:, 2], geometry[:, 2], rtol=0, atol=0)

        torch.testing.assert_close(geometry[0, 1, 0, 0], geometry[0, 1, 1, 0], rtol=0, atol=0)

    def test_sigma_controls_distance_saturation_and_ring_width(self):
        narrow = build_mask_geometry(self.mask[:1], sigma_px=1.0)
        wide = build_mask_geometry(self.mask[:1], sigma_px=4.0)
        location = (0, 2, 2)

        narrow_signed = abs(float(narrow[location[0], 1, location[1], location[2]]))
        wide_signed = abs(float(wide[location[0], 1, location[1], location[2]]))
        narrow_ring = float(narrow[location[0], 2, location[1], location[2]])
        wide_ring = float(wide[location[0], 2, location[1], location[2]])
        self.assertGreater(narrow_signed, wide_signed)
        self.assertLess(narrow_ring, wide_ring)

    def test_empty_and_full_masks_use_explicit_limiting_encodings(self):
        masks = torch.stack([torch.zeros(6, 7), torch.ones(6, 7)])
        geometry = build_mask_geometry(masks, sigma_px=3.0)

        torch.testing.assert_close(geometry[0, 0], torch.zeros(6, 7), rtol=0, atol=0)
        torch.testing.assert_close(geometry[0, 1], -torch.ones(6, 7), rtol=0, atol=0)
        torch.testing.assert_close(geometry[0, 2], torch.zeros(6, 7), rtol=0, atol=0)
        torch.testing.assert_close(geometry[1, 0], torch.ones(6, 7), rtol=0, atol=0)
        torch.testing.assert_close(geometry[1, 1], torch.ones(6, 7), rtol=0, atol=0)
        torch.testing.assert_close(geometry[1, 2], torch.zeros(6, 7), rtol=0, atol=0)

    def test_threshold_binarizes_soft_masks(self):
        mask = torch.tensor([[[0.49, 0.5, 0.51]]])
        geometry = build_mask_geometry(mask, sigma_px=1.0, threshold=0.5)
        torch.testing.assert_close(
            geometry[:, 0],
            torch.tensor([[[0.0, 0.0, 1.0]]]),
            rtol=0,
            atol=0,
        )

    def test_rejects_invalid_input(self):
        invalid_calls = (
            (TypeError, lambda: build_mask_geometry([[0, 1]], sigma_px=1.0)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(5, 5), sigma_px=1.0)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 2, 5, 5), sigma_px=1.0)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 0, 5), sigma_px=1.0)),
            (ValueError, lambda: build_mask_geometry(torch.tensor([[[float("nan")]]]), sigma_px=1.0)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 5, 5), sigma_px=0.0)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 5, 5), sigma_px=float("inf"))),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 5, 5), sigma_px=1.0, threshold=-0.1)),
            (ValueError, lambda: build_mask_geometry(torch.zeros(1, 5, 5), sigma_px=1.0, threshold=1.0)),
        )
        for expected_error, call in invalid_calls:
            with self.subTest(expected_error=expected_error), self.assertRaises(expected_error):
                call()


class MaskGeometryTokenizerTest(unittest.TestCase):
    def test_zero_initialization_preserves_baseline_and_only_zeros_final_layer(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=8, out_channels=12)
        geometry = build_mask_geometry(torch.ones(2, 8, 10), sigma_px=2.0)
        tokens = tokenizer(geometry, target_size=(4, 5))

        self.assertEqual(tuple(tokens.shape), (2, 20, 12))
        torch.testing.assert_close(tokens, torch.zeros_like(tokens), rtol=0, atol=0)
        self.assertGreater(int(torch.count_nonzero(tokenizer.conv_in.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(tokenizer.conv_out.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(tokenizer.conv_out.bias)), 0)

    def test_output_projection_gets_gradient_at_zero_initialization(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=6, out_channels=4)
        geometry = build_mask_geometry(
            torch.nn.functional.pad(torch.ones(1, 3, 3), (2, 2, 2, 2)),
            sigma_px=2.0,
        )
        tokenizer(geometry, target_size=(5, 4)).sum().backward()

        self.assertIsNotNone(tokenizer.conv_out.weight.grad)
        self.assertGreater(float(tokenizer.conv_out.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(tokenizer.conv_out.bias.grad)
        self.assertGreater(float(tokenizer.conv_out.bias.grad.abs().sum()), 0.0)
        self.assertIsNotNone(tokenizer.conv_in.weight.grad)
        self.assertEqual(float(tokenizer.conv_in.weight.grad.abs().sum()), 0.0)

    def test_gradient_reaches_earlier_layers_after_output_projection_opens(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=5, out_channels=3, resize_mode="area")
        torch.nn.init.normal_(tokenizer.conv_out.weight, std=0.02)
        geometry = torch.randn(2, 3, 8, 10, requires_grad=True)
        tokens = tokenizer(geometry, target_size=(4, 5))
        tokens.square().mean().backward()

        self.assertEqual(tuple(tokens.shape), (2, 20, 3))
        self.assertIsNotNone(geometry.grad)
        self.assertGreater(float(geometry.grad.abs().sum()), 0.0)
        self.assertIsNotNone(tokenizer.conv_in.weight.grad)
        self.assertGreater(float(tokenizer.conv_in.weight.grad.abs().sum()), 0.0)

    def test_downsampling_uses_occupancy_and_preserves_boundary_presence(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=4)
        mask = torch.zeros(1, 16, 16)
        mask[:, 0, 0] = 1
        geometry = build_mask_geometry(mask, sigma_px=2.0)

        resized = tokenizer._resize_geometry(geometry, target_size=(1, 1))

        self.assertEqual(tuple(resized.shape), (1, 3, 1, 1))
        self.assertAlmostEqual(float(resized[0, 0, 0, 0]), 1.0 / 256.0, places=7)
        self.assertLess(float(resized[0, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(resized[0, 2, 0, 0]),
            math.exp(-0.5 / 2.0),
            places=6,
        )

    def test_downsampling_keeps_definite_interior_and_outside_signs_consistent(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=4)
        mask = torch.zeros(1, 8, 8)
        mask[:, :, :4] = 1
        geometry = build_mask_geometry(mask, sigma_px=2.0)

        resized = tokenizer._resize_geometry(geometry, target_size=(2, 2))

        torch.testing.assert_close(resized[0, 0, :, 0], torch.ones(2), rtol=0, atol=0)
        torch.testing.assert_close(resized[0, 0, :, 1], torch.zeros(2), rtol=0, atol=0)
        self.assertTrue(bool((resized[0, 1, :, 0] > 0).all()))
        self.assertTrue(bool((resized[0, 1, :, 1] < 0).all()))

    def test_downsampling_thin_line_and_small_region_produce_fractional_occupancy(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=4)
        masks = torch.zeros(2, 16, 16)
        masks[0, :, 0] = 1
        masks[1, 1:3, 1:3] = 1
        geometry = build_mask_geometry(masks, sigma_px=2.0)

        resized = tokenizer._resize_geometry(geometry, target_size=(2, 2))

        torch.testing.assert_close(
            resized[0, 0],
            torch.tensor([[1.0 / 8.0, 0.0], [1.0 / 8.0, 0.0]]),
            rtol=0,
            atol=0,
        )
        self.assertAlmostEqual(float(resized[1, 0, 0, 0]), 4.0 / 64.0, places=7)
        self.assertTrue(bool((resized[:, 0] >= 0).all()))
        self.assertTrue(bool((resized[:, 0] <= 1).all()))

    def test_mixed_axis_resize_reduces_to_occupancy_before_upsampling(self):
        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=4)
        geometry = torch.zeros(1, 3, 8, 4)
        geometry[:, 1].fill_(-1.0)
        geometry[:, 0, 0, 0] = 1.0
        geometry[:, 2, 0, 0] = 0.75

        resized = tokenizer._resize_geometry(geometry, target_size=(4, 8))

        self.assertEqual(tuple(resized.shape), (1, 3, 4, 8))
        self.assertLessEqual(float(resized[:, 0].max()), 0.5)
        self.assertAlmostEqual(float(resized[:, 2].max()), 0.75, places=6)

    def test_dtype_contract_is_preserved_by_builder_and_tokenizer(self):
        mask = torch.zeros(1, 5, 6, dtype=torch.float64)
        mask[:, 1:4, 2:5] = 1
        geometry = build_mask_geometry(mask, sigma_px=2.0)
        self.assertEqual(geometry.dtype, torch.float32)
        self.assertEqual(geometry.device, mask.device)

        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=3).double()
        tokens = tokenizer(geometry, target_size=(3, 4))
        self.assertEqual(tokens.dtype, torch.float64)
        self.assertEqual(tokens.device, tokenizer.conv_in.weight.device)

    def test_rejects_invalid_shapes_sizes_and_constructor_values(self):
        with self.assertRaises(ValueError):
            MaskGeometryTokenizer(hidden_channels=0, out_channels=4)
        with self.assertRaises(ValueError):
            MaskGeometryTokenizer(hidden_channels=4, out_channels=4, resize_mode="nearest")

        tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=4)
        with self.assertRaisesRegex(ValueError, r"\[B,3,H,W\]"):
            tokenizer(torch.zeros(1, 1, 8, 8), target_size=(4, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            tokenizer(torch.zeros(1, 3, 8, 8), target_size=(0, 4))
        with self.assertRaisesRegex(ValueError, "contain"):
            tokenizer(torch.zeros(1, 3, 8, 8), target_size=(4,))


if __name__ == "__main__":
    unittest.main()

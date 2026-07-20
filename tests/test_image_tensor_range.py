import unittest

import torch

from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.tools import infer_image_tensor


class ImageTensorRangeInferenceTest(unittest.TestCase):
    def test_auto_accepts_constant_and_low_dynamic_range_images(self):
        self.assertEqual(infer_image_tensor(torch.zeros(1, 3, 8, 8))["range"], "0-1")
        self.assertEqual(
            infer_image_tensor(torch.full((1, 3, 8, 8), 0.05))["range"],
            "0-1",
        )
        self.assertEqual(
            infer_image_tensor(torch.full((1, 3, 8, 8), -1.0))["range"],
            "-1-1",
        )
        self.assertEqual(
            infer_image_tensor(torch.zeros(1, 3, 8, 8, dtype=torch.uint8))["range"],
            "0-255",
        )

    def test_auto_is_deterministic_and_does_not_advance_torch_rng(self):
        large_image = torch.zeros(2_000_001)
        torch.manual_seed(1234)
        expected = torch.rand(4)
        torch.manual_seed(1234)
        self.assertEqual(infer_image_tensor(large_image)["range"], "0-1")
        actual = torch.rand(4)
        self.assertTrue(torch.equal(actual, expected))

    def test_explicit_range_accepts_black_and_rejects_out_of_bounds(self):
        image = torch.zeros(1, 3, 8, 8)
        self.assertEqual(
            infer_image_tensor(image, input_range="0-1")["range"],
            "0-1",
        )
        with self.assertRaisesRegex(ValueError, "outside its declared range"):
            infer_image_tensor(torch.full_like(image, 1.1), input_range="0-1")

    def test_non_finite_empty_complex_and_unknown_contract_fail_clearly(self):
        for bad_value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=bad_value), self.assertRaisesRegex(ValueError, "NaN or Inf"):
                infer_image_tensor(torch.full((1, 3, 8, 8), bad_value))
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            infer_image_tensor(torch.empty(0), input_range="0-1")
        with self.assertRaisesRegex(TypeError, "Complex image tensors"):
            infer_image_tensor(torch.zeros(1, 3, 8, 8, dtype=torch.complex64))
        with self.assertRaisesRegex(ValueError, "input_range must be one of"):
            infer_image_tensor(torch.zeros(1, 3, 8, 8), input_range="bad")


class ImageTensorPreprocessContractTest(unittest.TestCase):
    def setUp(self):
        self.trainer = object.__new__(QwenImageEditTrainer)

    def test_black_zero_one_control_is_valid_for_text_and_vae(self):
        control = torch.zeros(1, 3, 16, 16)
        text = self.trainer.preprocess_image_for_text_encoder(
            control,
            input_range="0-1",
        )
        vae = self.trainer.preprocess_image_for_vae_encoder(
            control,
            input_range="0-1",
        )
        self.assertEqual(tuple(text.shape), (1, 3, 16, 16))
        self.assertTrue(torch.equal(text, torch.zeros_like(text)))
        self.assertEqual(tuple(vae.shape), (1, 3, 1, 16, 16))
        self.assertTrue(torch.equal(vae, torch.full_like(vae, -1.0)))

    def test_low_dynamic_zero_one_control_scales_by_declared_contract(self):
        control = torch.full((1, 3, 16, 16), 0.05)
        text = self.trainer.preprocess_image_for_text_encoder(
            control,
            input_range="0-1",
        )
        vae = self.trainer.preprocess_image_for_vae_encoder(
            control,
            input_range="0-1",
        )
        self.assertTrue(torch.allclose(text, torch.full_like(text, 12.75)))
        self.assertTrue(torch.allclose(vae, torch.full_like(vae, -0.9)))

    def test_other_explicit_contracts_keep_previous_conversion_semantics(self):
        uint8_image = torch.full((1, 3, 16, 16), 255, dtype=torch.uint8)
        text_255 = self.trainer.preprocess_image_for_text_encoder(
            uint8_image,
            input_range="0-255",
        )
        vae_255 = self.trainer.preprocess_image_for_vae_encoder(
            uint8_image,
            input_range="0-255",
        )
        self.assertTrue(torch.equal(text_255, uint8_image))
        self.assertTrue(torch.allclose(vae_255, torch.ones_like(vae_255)))

        normalized = torch.zeros(1, 3, 16, 16)
        text_normalized = self.trainer.preprocess_image_for_text_encoder(
            normalized,
            input_range="-1-1",
        )
        self.assertTrue(torch.allclose(text_normalized, torch.full_like(text_normalized, 127.5)))


if __name__ == "__main__":
    unittest.main()

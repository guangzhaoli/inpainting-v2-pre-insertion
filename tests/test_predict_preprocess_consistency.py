import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.data.config import ImageProcessorInitArgs, ModelConfig
from src.data.preprocess import ImageProcessor
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    QwenImageEditInsertAnythingTrainer,
)
from src.utils.images import extract_foreground_from_image_and_mask


class PredictPreprocessConsistencyTest(unittest.TestCase):
    def _assert_predict_matches_direct_preprocessor(self, processor_args):
        processor = ImageProcessor(ImageProcessorInitArgs(**processor_args))
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = SimpleNamespace(model=ModelConfig())
        trainer.preprocessor = processor
        trainer.vae_scale_factor = 8

        src_array = np.arange(83 * 101 * 3, dtype=np.uint32).reshape(83, 101, 3)
        src_array = (src_array % 256).astype(np.uint8)
        src_image = Image.fromarray(src_array, mode="RGB")
        src_mask_array = np.zeros((83, 101), dtype=np.uint8)
        src_mask_array[23:61, 37:76] = 255
        src_mask = Image.fromarray(src_mask_array, mode="L")

        ref_array = np.flip(src_array[:67, :79], axis=1).copy()
        ref_image = Image.fromarray(ref_array, mode="RGB")
        ref_mask_array = np.zeros((67, 79), dtype=np.uint8)
        ref_mask_array[12:55, 18:64] = 255
        ref_mask = Image.fromarray(ref_mask_array, mode="L")

        ref_foreground, ref_foreground_mask = extract_foreground_from_image_and_mask(
            ref_image,
            ref_mask,
        )
        expected = processor.preprocess(
            {
                "image": src_image.copy(),
                "mask": src_mask.copy(),
                "controls": [Image.fromarray(ref_foreground, mode="RGB")],
                "control_masks": [Image.fromarray(ref_foreground_mask, mode="L")],
            }
        )

        actual = trainer.prepare_predict_batch_data(
            image=[src_image, ref_image],
            prompt="insert the reference",
            ref_mask=ref_mask,
            src_mask=src_mask,
            num_inference_steps=2,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            weight_dtype=torch.float32,
        )

        torch.testing.assert_close(actual["tar_image"][0], expected["image"], rtol=0, atol=0)
        torch.testing.assert_close(actual["tar_mask"][0], expected["mask"], rtol=0, atol=0)
        torch.testing.assert_close(actual["ref_image"][0], expected["controls"][0], rtol=0, atol=0)
        torch.testing.assert_close(
            actual["ref_mask"][0],
            expected["control_masks"][0],
            rtol=0,
            atol=0,
        )

    def test_predict_matches_training_preprocessor_for_every_supported_geometry_mode(self):
        cases = {
            "aspect_pixels": {
                "process_type": "aspect_pixels",
                "target_pixels": 256 * 256,
                "controls_pixels": [256 * 256],
            },
            "fixed_pixels": {
                "process_type": "fixed_pixels",
                "target_pixels": 256 * 256,
                "controls_pixels": [256 * 256],
            },
            "resize": {
                "process_type": "resize",
                "target_size": [64, 96],
                "controls_size": [[48, 64]],
            },
            "center_padding": {
                "process_type": "center_padding",
                "target_size": [64, 96],
                "controls_size": [[48, 64]],
            },
            "right_padding": {
                "process_type": "right_padding",
                "target_size": [64, 96],
                "controls_size": [[48, 64]],
            },
        }
        for process_type, processor_args in cases.items():
            with self.subTest(process_type=process_type):
                self._assert_predict_matches_direct_preprocessor(processor_args)


if __name__ == "__main__":
    unittest.main()

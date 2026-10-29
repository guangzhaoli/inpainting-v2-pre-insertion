"""Reference downscale augmentation: resampling, dataset wiring and config."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from src.data.config import DatasetInitArgs, ReferenceDownscaleConfig
from src.data.insert_anything_dataset import InsertAnythingDataset, downscale_reference


def detail(image) -> float:
    """Mean finest-octave energy of an HWC or CHW image."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    image = image.astype(np.float32)
    return float(np.abs(image - cv2.GaussianBlur(image, (0, 0), 0.8)).mean())


class DownscaleReferenceTest(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)

    def test_keeps_the_size_and_removes_detail(self):
        out = downscale_reference(self.image, 3.0, 120 * 160)
        self.assertEqual(out.shape, self.image.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertLess(detail(out), 0.5 * detail(self.image))

    def test_factor_one_below_the_model_size_is_a_no_op(self):
        out = downscale_reference(self.image, 1.0, 2 * 120 * 160)
        self.assertIs(out, self.image)

    def test_factor_is_measured_against_the_model_resolution(self):
        # A cut-out four times the model's pixels already loses half its linear
        # resolution in the processor, so factor 2 must cut it to a quarter.
        out = downscale_reference(self.image, 2.0, 120 * 160 // 4)
        reference = cv2.resize(
            cv2.resize(self.image, (40, 30), interpolation=cv2.INTER_AREA), (160, 120), interpolation=cv2.INTER_CUBIC
        )
        self.assertTrue(np.array_equal(out, reference))


class ReferenceDownscaleDatasetTest(unittest.TestCase):
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

    def _sample(self, root: Path, seed=0, **downscale):
        torch.manual_seed(seed)
        dataset = InsertAnythingDataset(
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
                reference_downscale=downscale,
            )
        )
        return dataset[0]

    def test_disabled_augmentation_adds_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            sample = self._sample(root)
            for key in ("ref_image_clean", "ref_mask_clean", "reference_downscale_factor"):
                self.assertNotIn(key, sample)

    def test_downscaled_reference_keeps_a_clean_copy_and_the_target(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            base = self._sample(root, seed=3)
            sample = self._sample(root, seed=3, enabled=True, prob=1.0, factor_min=3.0, factor_max=3.0)
            self.assertAlmostEqual(sample["reference_downscale_factor"], 3.0)
            self.assertTrue(torch.equal(sample["ref_image_clean"], base["ref_image"]))
            self.assertTrue(torch.equal(sample["ref_mask_clean"], base["ref_mask"]))
            self.assertTrue(torch.equal(sample["ref_mask"], base["ref_mask"]))
            self.assertTrue(torch.equal(sample["tar_image"], base["tar_image"]))
            self.assertLess(detail(sample["ref_image"]), 0.6 * detail(base["ref_image"]))

    def test_skipped_draw_keeps_the_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build(root)
            base = self._sample(root, seed=3)
            sample = self._sample(root, seed=3, enabled=True, prob=0.0)
            self.assertEqual(sample["reference_downscale_factor"], 1.0)
            self.assertNotIn("ref_image_clean", sample)
            self.assertTrue(torch.equal(sample["ref_image"], base["ref_image"]))


class ReferenceDownscaleConfigTest(unittest.TestCase):
    def test_rejects_an_inverted_range(self):
        with self.assertRaises(ValidationError):
            ReferenceDownscaleConfig(factor_min=2.0, factor_max=1.5)

    def test_rejects_upscaling(self):
        with self.assertRaises(ValidationError):
            ReferenceDownscaleConfig(factor_min=0.5)


if __name__ == "__main__":
    unittest.main()

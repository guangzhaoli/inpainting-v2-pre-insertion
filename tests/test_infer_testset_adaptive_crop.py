import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from infer_testset import run_sample
from src.data.config import (
    AdaptiveCropConfig,
    ImageProcessorConfig,
    LatentAnchoringConfig,
    ModelConfig,
    PredictConfig,
)


class FakePredictTrainer:
    def __init__(self, adaptive_crop: AdaptiveCropConfig, anchoring_mode: str = "legacy"):
        self.config = SimpleNamespace(
            model=ModelConfig(),
            predict=PredictConfig(
                latent_anchoring=LatentAnchoringConfig(mode=anchoring_mode)
            ),
            data=SimpleNamespace(
                init_args=SimpleNamespace(
                    adaptive_crop=adaptive_crop,
                    processor=ImageProcessorConfig(),
                ),
            )
        )
        self.last_predict_kwargs = None
        self.anchor_masks = None

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        if kwargs.get("latent_anchoring_mode", "legacy") != "legacy":
            core = torch.zeros(64, 64)
            core[24:40, 24:40] = 1
            self.anchor_masks = {
                "mode": kwargs["latent_anchoring_mode"],
                "core": core,
                "free": core.clone(),
                "edit_weight": core.clone(),
            }
        return [Image.new("RGB", (64, 64), color=(0, 255, 0))]

    def get_last_latent_anchoring_masks(self):
        return self.anchor_masks


class InferTestsetAdaptiveCropTest(unittest.TestCase):
    def test_run_sample_crops_predict_input_and_restores_full_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = Image.new("RGB", (100, 80), color=(255, 0, 0))
            source_mask_array = np.zeros((80, 100), dtype=np.uint8)
            source_mask_array[35:45, 45:55] = 255
            source_mask = Image.fromarray(source_mask_array, mode="L")
            reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
            reference_mask = Image.new("L", (40, 40), color=255)

            paths = {
                "source": root / "source.png",
                "source_mask": root / "source_mask.png",
                "ref_image": root / "ref_image.png",
                "ref_mask": root / "ref_mask.png",
            }
            source.save(paths["source"])
            source_mask.save(paths["source_mask"])
            reference.save(paths["ref_image"])
            reference_mask.save(paths["ref_mask"])

            trainer = FakePredictTrainer(
                AdaptiveCropConfig(
                    predict_enabled=True,
                    trigger_bbox_area_ratio=0.10,
                    min_crop_area_ratio=0.25,
                    context_margin_ratio=0.20,
                )
            )
            args = SimpleNamespace(
                height=None,
                width=None,
                prompt="",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                panel_size=64,
                config=root / "config.yaml",
                lora=root / "checkpoint",
            )
            sample = {"stem": "sample", **paths}
            sample_dir = root / "result"

            metadata, _ = run_sample(trainer, sample, sample_dir, args, sample_seed=7)

            predict_source = trainer.last_predict_kwargs["image"][0]
            predict_mask = trainer.last_predict_kwargs["src_mask"]
            self.assertLess(predict_source.width, source.width)
            self.assertLess(predict_source.height, source.height)
            self.assertEqual(predict_source.size, predict_mask.size)

            with Image.open(sample_dir / "output.png") as output:
                self.assertEqual(output.size, source.size)
            with Image.open(sample_dir / "composite.png") as composite:
                composite_array = np.asarray(composite.convert("RGB"))
            self.assertTrue(np.all(composite_array[0, 0] == np.array([255, 0, 0])))
            self.assertTrue(np.all(composite_array[40, 50] == np.array([0, 255, 0])))
            self.assertTrue(metadata["adaptive_crop"]["config"]["predict_enabled"])
            self.assertTrue(metadata["adaptive_crop"]["plan"]["applied"])
            self.assertEqual(metadata["seed"], 7)
            self.assertEqual(metadata["parameters"]["effective"]["prompt"], "")
            self.assertTrue((sample_dir / "output_local.png").is_file())
            self.assertTrue((sample_dir / "output_restored_raw.png").is_file())

    def test_run_sample_without_crop_still_saves_full_size_consistent_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = Image.new("RGB", (100, 80), color=(255, 0, 0))
            source_mask_array = np.zeros((80, 100), dtype=np.uint8)
            source_mask_array[35:45, 45:55] = 255
            source_mask_array[0, 0] = 127
            source_mask_array[0, 1] = 128
            source_mask = Image.fromarray(source_mask_array, mode="L")
            reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
            reference_mask = Image.new("L", (40, 40), color=255)

            paths = {
                "source": root / "source.png",
                "source_mask": root / "source_mask.png",
                "ref_image": root / "ref_image.png",
                "ref_mask": root / "ref_mask.png",
            }
            source.save(paths["source"])
            source_mask.save(paths["source_mask"])
            reference.save(paths["ref_image"])
            reference_mask.save(paths["ref_mask"])

            trainer = FakePredictTrainer(AdaptiveCropConfig(predict_enabled=False))
            args = SimpleNamespace(
                height=None,
                width=None,
                prompt="native prompt",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                panel_size=64,
                config=root / "config.yaml",
                lora=root / "checkpoint",
            )
            sample_dir = root / "result"

            metadata, _ = run_sample(
                trainer,
                {"stem": "sample", **paths},
                sample_dir,
                args,
                sample_seed=9,
            )

            self.assertEqual(trainer.last_predict_kwargs["image"][0].size, source.size)
            for name in ("output.png", "output_restored_raw.png", "composite.png"):
                with Image.open(sample_dir / name) as image:
                    self.assertEqual(image.size, source.size)
            with Image.open(sample_dir / "output_local.png") as image:
                self.assertEqual(image.size, (64, 64))
            with Image.open(sample_dir / "output_mask.png") as image:
                output_mask = np.asarray(image.convert("L"))
            self.assertEqual(int(output_mask[0, 0]), 0)
            self.assertEqual(int(output_mask[0, 1]), 255)
            self.assertIsNone(metadata["adaptive_crop"]["plan"])
            self.assertEqual(metadata["sizes"]["output"], [100, 80])

    def test_run_sample_core_anchoring_keeps_raw_primary_and_strict_composite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = Image.new("RGB", (80, 80), color=(255, 0, 0))
            source_mask_array = np.zeros((80, 80), dtype=np.uint8)
            source_mask_array[30:50, 30:50] = 255
            source_mask = Image.fromarray(source_mask_array, mode="L")
            reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
            reference_mask = Image.new("L", (40, 40), color=255)

            paths = {
                "source": root / "source.png",
                "source_mask": root / "source_mask.png",
                "ref_image": root / "ref_image.png",
                "ref_mask": root / "ref_mask.png",
            }
            source.save(paths["source"])
            source_mask.save(paths["source_mask"])
            reference.save(paths["ref_image"])
            reference_mask.save(paths["ref_mask"])

            trainer = FakePredictTrainer(
                AdaptiveCropConfig(predict_enabled=False),
                anchoring_mode="core",
            )
            args = SimpleNamespace(
                height=None,
                width=None,
                prompt="insert",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                panel_size=64,
                config=root / "config.yaml",
                lora=root / "checkpoint",
            )
            sample_dir = root / "result"

            metadata, _ = run_sample(
                trainer,
                {"stem": "sample", **paths},
                sample_dir,
                args,
                sample_seed=11,
            )

            with Image.open(sample_dir / "output.png") as output:
                output_array = np.asarray(output.convert("RGB"))
            self.assertTrue(np.all(output_array == np.array([0, 255, 0])))
            with Image.open(sample_dir / "composite.png") as composite:
                composite_array = np.asarray(composite.convert("RGB"))
            self.assertTrue(np.all(composite_array[0, 0] == np.array([255, 0, 0])))
            self.assertTrue(np.all(composite_array[40, 40] == np.array([0, 255, 0])))
            self.assertEqual(metadata["latent_anchoring"]["variant"], "B")
            for name in ("mask_core.png", "mask_free.png", "edit_weight.png"):
                self.assertTrue((sample_dir / name).is_file())


if __name__ == "__main__":
    unittest.main()

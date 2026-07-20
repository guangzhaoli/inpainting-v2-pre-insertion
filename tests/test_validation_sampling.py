import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from pydantic import ValidationError

from src.data.config import (
    AdaptiveCropConfig,
    ImageProcessorConfig,
    LatentAnchoringConfig,
    ModelConfig,
    PredictConfig,
    SamplingConfig,
)
from src.trainer.insert_anything_validation_sampling import (
    InsertAnythingValidationSampler,
    ManifestValidationSource,
)


class FakeAccelerator:
    is_main_process = True
    trackers = []

    def wait_for_everyone(self):
        return None


class FakeDataset:
    def __len__(self):
        return 4

    def _get_sample_info(self, index):
        return {
            "ref_image": f"ref-{index}",
            "ref_mask": f"ref-mask-{index}",
            "tar_image": f"tar-{index}",
            "tar_mask": f"tar-mask-{index}",
            "prompt": f"prompt {index}",
            "category": "object",
            "stem": f"sample-{index}",
        }

    def _read_image(self, value):
        if value.startswith("ref"):
            return np.full((32, 32, 3), fill_value=128, dtype=np.uint8)
        return np.full((32, 32, 3), fill_value=32, dtype=np.uint8)

    def _read_mask(self, value):
        return np.full((32, 32), fill_value=255, dtype=np.uint8)


class FakeSmallMaskDataset(FakeDataset):
    def _read_mask(self, value):
        if value.startswith("tar-mask"):
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[14:18, 14:18] = 255
            return mask
        return super()._read_mask(value)


class FakeTrainer:
    def __init__(self):
        self.weight_dtype = torch.float32
        self.accelerator = FakeAccelerator()
        self.calls = []

    def predict_loaded_model(self, **kwargs):
        self.calls.append(kwargs)
        return [Image.new("RGB", (32, 32), color=(10, 20, 30))]


def make_config(
    output_dir: str,
    sampling: SamplingConfig,
    adaptive_crop: AdaptiveCropConfig | None = None,
):
    return SimpleNamespace(
        logging=SimpleNamespace(output_dir=output_dir, sampling=sampling),
        model=ModelConfig(),
        predict=PredictConfig(),
        data=SimpleNamespace(
            class_path="",
            init_args=SimpleNamespace(
                adaptive_crop=adaptive_crop or AdaptiveCropConfig(),
                processor=ImageProcessorConfig(),
            ),
        ),
    )


class ValidationSamplingTest(unittest.TestCase):
    def test_sampling_seed_rejects_values_outside_uint32_range(self):
        for seed in (-1, 2**32):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValidationError, "sampling.seed"):
                    SamplingConfig(seed=seed)

    def test_sampler_rejects_negative_prompt_for_prompt_free_modes_at_setup(self):
        for mode in ("reference_cfg", "query_prompt"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmpdir:
                sampling = SamplingConfig(
                    enable=True,
                    validation_steps=1,
                    num_samples=1,
                    negative_prompt="must fail early",
                )
                config = make_config(tmpdir, sampling)
                getattr(config.model, mode).enabled = True
                with self.assertRaisesRegex(ValueError, "negative_prompt"):
                    InsertAnythingValidationSampler(
                        config,
                        SimpleNamespace(dataset=FakeDataset()),
                        FakeAccelerator(),
                    )

    def test_manifest_source_reads_image_and_mask_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            Image.new("RGB", (12, 8), color=(10, 20, 30)).save(base_dir / "image.png")
            Image.new("L", (12, 8), color=255).save(base_dir / "mask.png")
            source = ManifestValidationSource(
                [
                    {
                        "ref_image": "image.png",
                        "ref_mask": "mask.png",
                        "tar_image": "image.png",
                        "tar_mask": "mask.png",
                    }
                ],
                base_dir,
            )

            sample = source.get_sample(0)

            self.assertEqual(sample.ref_image.size, (12, 8))
            self.assertEqual(sample.ref_mask.size, (12, 8))
            self.assertEqual(sample.tar_image.size, (12, 8))
            self.assertEqual(sample.tar_mask.size, (12, 8))

    def test_sampling_config_rejects_legacy_validation_data_shape(self):
        with self.assertRaises(ValidationError):
            SamplingConfig(
                enable=True,
                validation_data=[
                    {
                        "control": "old.png",
                        "prompt": "legacy",
                    }
                ],
            )

    def test_sampler_uses_deterministic_indices_and_step_gate(self):
        sampling = SamplingConfig(enable=True, validation_steps=2, num_samples=2, seed=7)
        config = make_config("/tmp/out", sampling)
        dataloader = SimpleNamespace(dataset=FakeDataset())

        sampler_a = InsertAnythingValidationSampler(config, dataloader, FakeAccelerator())
        sampler_b = InsertAnythingValidationSampler(config, dataloader, FakeAccelerator())

        self.assertEqual(sampler_a.sample_indices, sampler_b.sample_indices)
        self.assertEqual(len(sampler_a.sample_indices), 2)
        self.assertFalse(sampler_a.should_run_validation(0))
        self.assertFalse(sampler_a.should_run_validation(1))
        self.assertTrue(sampler_a.should_run_validation(2))

    def test_validation_loop_saves_outputs_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampling = SamplingConfig(
                enable=True,
                validation_steps=1,
                num_samples=1,
                seed=11,
                num_inference_steps=3,
                true_cfg_scale=2.5,
                output_dir="validation",
            )
            config = make_config(tmpdir, sampling)
            config.predict.latent_anchoring = LatentAnchoringConfig(
                mode="expanded_soft",
                interaction_dilation_px=24,
                transition_width_px=16,
            )
            sampler = InsertAnythingValidationSampler(
                config,
                SimpleNamespace(dataset=FakeDataset()),
                FakeAccelerator(),
            )
            trainer = FakeTrainer()

            sampler.run_validation_loop(global_step=4, trainer=trainer)

            step_dir = Path(tmpdir) / "validation" / "step-00000004"
            self.assertTrue((step_dir / "grid.png").is_file())
            self.assertTrue((step_dir / "metadata.json").is_file())
            generated = list(step_dir.glob("*_generated.png"))
            outputs = list(step_dir.glob("*_output.png"))
            rows = list(step_dir.glob("*_row.png"))
            source_overlays = list(step_dir.glob("*_source_mask_overlay.png"))
            output_overlays = list(step_dir.glob("*_output_mask_overlay.png"))
            self.assertEqual(len(generated), 0)
            self.assertEqual(len(outputs), 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(source_overlays), 1)
            self.assertEqual(len(output_overlays), 1)
            self.assertTrue((step_dir / "all_rows.png").is_file())

            with (step_dir / "metadata.json").open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["global_step"], 4)
            self.assertEqual(metadata["requested_sampling"]["num_inference_steps"], 3)
            self.assertEqual(metadata["requested_sampling"]["true_cfg_scale"], 2.5)
            self.assertEqual(metadata["panel_size"], 320)
            self.assertEqual(metadata["latent_anchoring"]["configured"]["variant"], "D")
            self.assertEqual(metadata["latent_anchoring"]["effective"]["variant"], "A")
            self.assertTrue(metadata["latent_anchoring"]["forced_legacy_for_validation"])
            self.assertTrue(metadata["latent_anchoring"]["configured_mode_overridden"])
            self.assertEqual(len(metadata["samples"]), 1)
            self.assertEqual(
                metadata["samples"][0]["latent_anchoring"]["effective"]["mode"],
                "legacy",
            )
            sample_paths = metadata["samples"][0]["paths"]
            self.assertIn("row", sample_paths)
            self.assertIn("output", sample_paths)
            self.assertIn("output_local", sample_paths)
            self.assertIn("output_restored_raw", sample_paths)
            self.assertIn("source_mask_overlay", sample_paths)
            self.assertIn("output_mask_overlay", sample_paths)
            self.assertEqual(sample_paths["generated"], sample_paths["output"])
            self.assertEqual(len(trainer.calls), 1)
            self.assertEqual(trainer.calls[0]["num_inference_steps"], 3)
            self.assertEqual(trainer.calls[0]["true_cfg_scale"], 2.5)
            self.assertEqual(trainer.calls[0]["latent_anchoring_mode"], "legacy")

    def test_validation_loop_restores_adaptive_crop_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampling = SamplingConfig(enable=True, validation_steps=1, num_samples=1, seed=5)
            adaptive_crop = AdaptiveCropConfig(
                predict_enabled=True,
                trigger_bbox_area_ratio=0.20,
                min_crop_area_ratio=0.25,
                context_margin_ratio=0.0,
            )
            config = make_config(tmpdir, sampling, adaptive_crop=adaptive_crop)
            sampler = InsertAnythingValidationSampler(
                config,
                SimpleNamespace(dataset=FakeSmallMaskDataset()),
                FakeAccelerator(),
            )
            trainer = FakeTrainer()

            sampler.run_validation_loop(global_step=1, trainer=trainer)

            step_dir = Path(tmpdir) / "validation" / "step-00000001"
            self.assertEqual(len(list(step_dir.glob("*_crop_source.png"))), 1)
            self.assertEqual(len(list(step_dir.glob("*_crop_mask.png"))), 1)
            self.assertEqual(len(list(step_dir.glob("*_output_local.png"))), 1)
            self.assertEqual(len(list(step_dir.glob("*_output_restored_raw.png"))), 1)
            output_path = next(step_dir.glob("*_output.png"))
            with Image.open(output_path) as output_image:
                self.assertEqual(output_image.size, (32, 32))
            with (step_dir / "metadata.json").open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertTrue(metadata["samples"][0]["adaptive_crop"]["plan"]["applied"])


if __name__ == "__main__":
    unittest.main()

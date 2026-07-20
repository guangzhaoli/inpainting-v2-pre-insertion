import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import infer
from src.data.config import AdaptiveCropConfig, ImageProcessorConfig, ModelConfig


class FakeCliTrainer:
    instances = []

    def __init__(self, config):
        self.config = config
        self.last_predict_kwargs = None
        self.anchor_masks = None
        self.__class__.instances.append(self)

    def setup_predict(self):
        return None

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


def make_config(adaptive_crop: AdaptiveCropConfig):
    return SimpleNamespace(
        model=ModelConfig(),
        data=SimpleNamespace(
            init_args=SimpleNamespace(
                adaptive_crop=adaptive_crop,
                processor=ImageProcessorConfig(),
            )
        ),
    )


class InferCliTest(unittest.TestCase):
    def test_seed_everything_reproduces_python_numpy_and_torch_rng(self):
        infer.seed_everything(1234)
        first = (random.random(), np.random.rand(), torch.randn(4))
        infer.seed_everything(1234)
        second = (random.random(), np.random.rand(), torch.randn(4))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_output_path_plan_rejects_derived_artifact_and_input_collisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / "result.png"
            args = SimpleNamespace(
                output=output,
                composite_output=None,
                row_output=root / "result.output_mask.png",
                skip_composite=False,
                skip_row=False,
            )
            with self.assertRaisesRegex(ValueError, "output paths must be distinct"):
                infer.build_output_path_plan(args, has_crop=True)

            args.row_output = None
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                infer.build_output_path_plan(
                    args,
                    has_crop=True,
                    protected_paths=[output],
                )

            args.composite_output = output
            with self.assertRaisesRegex(ValueError, "different semantics"):
                infer.build_output_path_plan(
                    args,
                    has_crop=False,
                    latent_anchoring_is_enabled=True,
                )

    def test_main_saves_output_scoped_artifacts_and_effective_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = Image.new("RGB", (100, 80), color=(255, 0, 0))
            source_mask_array = np.zeros((80, 100), dtype=np.uint8)
            source_mask_array[35:45, 45:55] = 255
            source_mask = Image.fromarray(source_mask_array, mode="L")
            reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
            reference_mask = Image.new("L", (40, 40), color=255)

            source_path = root / "source.png"
            source_mask_path = root / "source_mask.png"
            reference_path = root / "reference.png"
            reference_mask_path = root / "reference_mask.png"
            config_path = root / "config.yaml"
            lora_path = root / "weights.safetensors"
            source.save(source_path)
            source_mask.save(source_mask_path)
            reference.save(reference_path)
            reference_mask.save(reference_mask_path)
            config_path.write_text("test: true\n", encoding="utf-8")
            lora_path.write_bytes(b"test")

            config = make_config(
                AdaptiveCropConfig(
                    predict_enabled=True,
                    trigger_bbox_area_ratio=0.10,
                    min_crop_area_ratio=0.25,
                    context_margin_ratio=0.20,
                )
            )
            output_path = root / "first.png"
            args = SimpleNamespace(
                config=config_path,
                lora=lora_path,
                allow_base_model=False,
                allow_checkpoint_mismatch=True,
                src_image=source_path,
                ref_image=reference_path,
                src_mask=source_mask_path,
                ref_mask=reference_mask_path,
                prompt="insert",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                seed=123,
                height=None,
                width=None,
                output=output_path,
                composite_output=None,
                row_output=None,
                panel_size=64,
                skip_composite=False,
                skip_row=True,
            )

            import src.data.config as config_module
            import src.trainer.qwen_image_edit_insert_anything_trainer as trainer_module

            FakeCliTrainer.instances.clear()
            with (
                patch.object(infer, "parse_args", return_value=args),
                patch.object(config_module, "load_config_from_yaml", return_value=config),
                patch.object(
                    trainer_module,
                    "QwenImageEditInsertAnythingTrainer",
                    FakeCliTrainer,
                ),
            ):
                infer.main()

            expected_paths = {
                "output": output_path,
                "local": root / "first.output_local.png",
                "raw": root / "first.output_restored_raw.png",
                "crop_source": root / "first.crop_source.png",
                "crop_mask": root / "first.crop_mask.png",
                "composite": root / "first.composite.png",
                "metadata": root / "first.metadata.json",
            }
            for path in expected_paths.values():
                self.assertTrue(path.is_file(), path)

            with Image.open(output_path) as output:
                self.assertEqual(output.size, source.size)
                output_array = np.asarray(output.convert("RGB"))
            self.assertTrue(np.all(output_array[0, 0] == np.array([255, 0, 0])))
            self.assertTrue(np.all(output_array[40, 50] == np.array([0, 255, 0])))
            with Image.open(expected_paths["raw"]) as raw:
                self.assertEqual(raw.size, source.size)

            metadata = json.loads(expected_paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(metadata["seed"], 123)
            self.assertEqual(metadata["parameters"]["requested"]["prompt"], "insert")
            self.assertEqual(metadata["parameters"]["effective"]["prompt"], "insert")
            self.assertEqual(metadata["sizes"]["final_composite"], [100, 80])
            self.assertEqual(metadata["checkpoint"]["resolved_lora"], str(lora_path))

            predict_input = FakeCliTrainer.instances[-1].last_predict_kwargs["image"][0]
            self.assertLess(predict_input.width, source.width)
            self.assertLess(predict_input.height, source.height)

    def test_core_anchoring_uses_anchored_primary_and_keeps_strict_composite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = Image.new("RGB", (80, 80), color=(255, 0, 0))
            source_mask_array = np.zeros((80, 80), dtype=np.uint8)
            source_mask_array[30:50, 30:50] = 255
            source_mask = Image.fromarray(source_mask_array, mode="L")
            reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
            reference_mask = Image.new("L", (40, 40), color=255)

            source_path = root / "source.png"
            source_mask_path = root / "source_mask.png"
            reference_path = root / "reference.png"
            reference_mask_path = root / "reference_mask.png"
            config_path = root / "config.yaml"
            lora_path = root / "weights.safetensors"
            source.save(source_path)
            source_mask.save(source_mask_path)
            reference.save(reference_path)
            reference_mask.save(reference_mask_path)
            config_path.write_text("test: true\n", encoding="utf-8")
            lora_path.write_bytes(b"test")

            config = make_config(AdaptiveCropConfig(predict_enabled=False))
            output_path = root / "anchored.png"
            args = SimpleNamespace(
                config=config_path,
                lora=lora_path,
                allow_base_model=False,
                allow_checkpoint_mismatch=True,
                src_image=source_path,
                ref_image=reference_path,
                src_mask=source_mask_path,
                ref_mask=reference_mask_path,
                prompt="insert",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=4.0,
                guidance_scale=1.0,
                seed=123,
                height=None,
                width=None,
                output=output_path,
                composite_output=None,
                row_output=None,
                panel_size=64,
                skip_composite=False,
                skip_row=True,
                latent_anchoring_mode="core",
                interaction_dilation_px=None,
                transition_width_px=None,
                latent_anchoring_mask_threshold=None,
            )

            import src.data.config as config_module
            import src.trainer.qwen_image_edit_insert_anything_trainer as trainer_module

            FakeCliTrainer.instances.clear()
            with (
                patch.object(infer, "parse_args", return_value=args),
                patch.object(config_module, "load_config_from_yaml", return_value=config),
                patch.object(
                    trainer_module,
                    "QwenImageEditInsertAnythingTrainer",
                    FakeCliTrainer,
                ),
            ):
                infer.main()

            with Image.open(output_path) as output:
                output_array = np.asarray(output.convert("RGB"))
            self.assertTrue(np.all(output_array == np.array([0, 255, 0])))

            composite_path = root / "anchored.composite.png"
            with Image.open(composite_path) as composite:
                composite_array = np.asarray(composite.convert("RGB"))
            self.assertTrue(np.all(composite_array[0, 0] == np.array([255, 0, 0])))
            self.assertTrue(np.all(composite_array[40, 40] == np.array([0, 255, 0])))

            for name in ("mask_core", "mask_free", "edit_weight"):
                self.assertTrue((root / f"anchored.{name}.png").is_file())

            metadata = json.loads((root / "anchored.metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["latent_anchoring"]["variant"], "B")
            self.assertEqual(metadata["outputs"]["output"], str(output_path))
            self.assertEqual(metadata["outputs"]["strict_composite"], str(composite_path))

            skip_output_path = root / "anchored_skip.png"
            args.output = skip_output_path
            args.skip_composite = True
            FakeCliTrainer.instances.clear()
            with (
                patch.object(infer, "parse_args", return_value=args),
                patch.object(config_module, "load_config_from_yaml", return_value=config),
                patch.object(
                    trainer_module,
                    "QwenImageEditInsertAnythingTrainer",
                    FakeCliTrainer,
                ),
            ):
                infer.main()

            self.assertTrue(skip_output_path.is_file())
            self.assertFalse((root / "anchored_skip.composite.png").exists())
            skip_metadata = json.loads(
                (root / "anchored_skip.metadata.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(skip_metadata["outputs"]["strict_composite"])
            self.assertIsNone(skip_metadata["outputs"]["final_composite"])
            self.assertIsNone(skip_metadata["outputs"]["composite_compatibility_copy"])


if __name__ == "__main__":
    unittest.main()

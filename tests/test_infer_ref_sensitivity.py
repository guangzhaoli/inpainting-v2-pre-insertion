import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import infer_ref_sensitivity
import infer_testset
from src.data.config import AdaptiveCropConfig, ImageProcessorConfig, ModelConfig


def make_config():
    return SimpleNamespace(
        model=ModelConfig(),
        data=SimpleNamespace(
            init_args=SimpleNamespace(
                adaptive_crop=AdaptiveCropConfig(),
                processor=ImageProcessorConfig(),
            )
        ),
    )


class FakeTrainer:
    def __init__(self, config):
        self.config = config

    def setup_predict(self):
        return None


class InferRefSensitivityTest(unittest.TestCase):
    def test_shared_setup_trainer_accepts_sensitivity_args_without_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.yaml"
            lora_path = root / "model.safetensors"
            config_path.write_text("test: true\n", encoding="utf-8")
            lora_path.write_bytes(b"checkpoint")
            args = SimpleNamespace(
                config=config_path,
                lora=lora_path,
                prompt="insert",
                negative_prompt=None,
                allow_checkpoint_mismatch=True,
                height=None,
                width=None,
                seed=42,
                num_inference_steps=2,
                true_cfg_scale=1.0,
                guidance_scale=1.0,
                panel_size=64,
            )
            config = make_config()

            import src.data.config as config_module
            import src.trainer.qwen_image_edit_insert_anything_trainer as trainer_module

            with (
                patch.object(config_module, "load_config_from_yaml", return_value=config),
                patch.object(
                    trainer_module,
                    "QwenImageEditInsertAnythingTrainer",
                    FakeTrainer,
                ),
            ):
                trainer, loaded_config, resolved_lora, compatibility = infer_testset.setup_trainer(args)

            self.assertIsInstance(trainer, FakeTrainer)
            self.assertIs(loaded_config, config)
            self.assertEqual(resolved_lora, lora_path)
            self.assertEqual(compatibility["status"], "unverified_allowed")

    def test_sensitivity_main_forwards_checkpoint_compatibility_to_every_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = SimpleNamespace(
                config=root / "config.yaml",
                lora=root / "model.safetensors",
                allow_checkpoint_mismatch=False,
                testset_root=root / "testset",
                category="accessory",
                output_dir=root / "output",
                source_stems=["8"],
                ref_stems=["8"],
                max_ref_variants=1,
                prompt="insert",
                negative_prompt=None,
                num_inference_steps=2,
                true_cfg_scale=1.0,
                guidance_scale=1.0,
                height=None,
                width=None,
                seed=42,
                panel_size=64,
            )
            sample = {
                "stem": "8",
                "ref_image": root / "ref.png",
                "ref_mask": root / "ref_mask.png",
                "source": root / "source.png",
                "source_mask": root / "source_mask.png",
            }
            config = make_config()
            compatibility = {"status": "verified", "verified": True}
            calls = []

            def fake_run_sample(trainer, sample, sample_dir, args, sample_seed, **kwargs):
                sample_dir.mkdir(parents=True, exist_ok=True)
                calls.append(kwargs)
                return {"seed": sample_seed}, sample_dir / "row.png"

            with (
                patch.object(infer_ref_sensitivity, "parse_args", return_value=args),
                patch.object(infer_ref_sensitivity, "discover_samples", return_value=[sample]),
                patch.object(
                    infer_ref_sensitivity,
                    "setup_trainer",
                    return_value=(object(), config, args.lora, compatibility),
                ),
                patch.object(infer_ref_sensitivity, "run_sample", side_effect=fake_run_sample),
                patch.object(infer_ref_sensitivity, "stack_rows"),
                patch.object(infer_ref_sensitivity, "masked_mae", return_value=0.0),
            ):
                infer_ref_sensitivity.main()

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["resolved_lora_path"], args.lora)
            self.assertIs(calls[0]["checkpoint_compatibility"], compatibility)


if __name__ == "__main__":
    unittest.main()

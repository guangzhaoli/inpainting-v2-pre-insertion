import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.config import (
    AdaptiveCropConfig,
    Config,
    DataConfig,
    DatasetInitArgs,
    ImageProcessorConfig,
    ImageProcessorInitArgs,
    LatentAnchoringConfig,
    LoraConfig,
    MaskControlConfig,
    ModelConfig,
)
from src.utils.inference_metadata import (
    MASK_GEOMETRY_COMPANION_FILE_NAME,
    build_effective_inference_metadata,
    resolve_effective_mask_control,
    validate_checkpoint_compatibility,
    validate_resume_mask_control_compatibility,
)
from src.utils.tools import calculate_sha256_file


def make_adaptive_config(*, train_enabled=True, predict_enabled=True, adapter_name="adaptive"):
    return Config(
        trainer="QwenImageEditInsertAnything",
        model=ModelConfig(
            lora=LoraConfig(
                r=8,
                lora_alpha=8,
                target_modules=["to_q", "to_v"],
                adapter_name=adapter_name,
            )
        ),
        data=DataConfig(
            init_args=DatasetInitArgs(
                processor=ImageProcessorConfig(
                    init_args=ImageProcessorInitArgs(
                        process_type="fixed_pixels",
                        target_pixels=256 * 256,
                        controls_pixels=[256 * 256],
                    )
                ),
                adaptive_crop=AdaptiveCropConfig(
                    train_enabled=train_enabled,
                    predict_enabled=predict_enabled,
                ),
            )
        ),
    )


class SplitConditioningConfigTest(unittest.TestCase):
    def test_disabled_legacy_modes_are_ignored(self):
        config = ModelConfig.model_validate(
            {
                "query_prompt": {"enabled": False},
                "pre_insertion": {"enabled": False},
                "layout_condition": {"enabled": False},
            }
        )

        self.assertFalse(hasattr(config, "query_prompt"))
        self.assertFalse(hasattr(config, "pre_insertion"))
        self.assertFalse(hasattr(config, "layout_condition"))

    def test_enabled_legacy_mode_requires_split_worktree(self):
        with self.assertRaisesRegex(ValueError, "inpainting-v2-query-prompt"):
            ModelConfig.model_validate({"query_prompt": {"enabled": True}})


class CheckpointCompatibilityTest(unittest.TestCase):
    def _write_checkpoint(self, root: Path, config: Config) -> Path:
        run_dir = root / "run"
        checkpoint_dir = run_dir / "checkpoint-0-1"
        checkpoint_dir.mkdir(parents=True)
        train_config_path = run_dir / "train_config.yaml"
        train_config_path.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", exclude_none=True),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        lora_path = checkpoint_dir / "model.safetensors"
        lora_path.write_bytes(b"test checkpoint")
        artifact_specs = [("lora_peft", lora_path)]
        if resolve_effective_mask_control(config)["geometry_active"]:
            geometry_path = checkpoint_dir / MASK_GEOMETRY_COMPANION_FILE_NAME
            geometry_path.write_bytes(b"geometry tokenizer")
            artifact_specs.append(("mask_geometry_tokenizer", geometry_path))
        (checkpoint_dir / "state.json").write_text(
            json.dumps(
                {
                    "artifacts": {
                        "version": 1,
                        "files": [
                            {
                                "role": role,
                                "filename": path.name,
                                "size_bytes": path.stat().st_size,
                                "sha256": calculate_sha256_file(str(path)),
                            }
                            for role, path in artifact_specs
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        return lora_path

    def test_matching_training_and_prediction_contract_is_verified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            lora_path = self._write_checkpoint(Path(tmpdir), config)

            result = validate_checkpoint_compatibility(config, lora_path)

            self.assertTrue(result["verified"])
            self.assertEqual(result["status"], "verified")
            self.assertEqual(result["mismatches"], [])

    def test_nonadaptive_checkpoint_is_rejected_for_adaptive_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inference_config = make_adaptive_config()
            training_config = make_adaptive_config(
                train_enabled=False,
                predict_enabled=False,
                adapter_name="legacy",
            )
            lora_path = self._write_checkpoint(Path(tmpdir), training_config)

            with self.assertRaisesRegex(ValueError, "Checkpoint/config contract mismatch"):
                validate_checkpoint_compatibility(inference_config, lora_path)

            allowed = validate_checkpoint_compatibility(
                inference_config,
                lora_path,
                allow_mismatch=True,
            )
            self.assertEqual(allowed["status"], "mismatch_allowed")
            fields = {item["field"] for item in allowed["mismatches"]}
            self.assertIn("data.adaptive_crop.enabled", fields)
            self.assertIn("model.lora.adapter_name", fields)

    def test_missing_train_config_requires_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "model.safetensors"
            lora_path.write_bytes(b"test checkpoint")
            config = make_adaptive_config()

            with self.assertRaisesRegex(ValueError, "train_config.yaml was not found"):
                validate_checkpoint_compatibility(config, lora_path)

            result = validate_checkpoint_compatibility(
                config,
                lora_path,
                allow_mismatch=True,
            )
            self.assertEqual(result["status"], "unverified_allowed")

    def test_legacy_checkpoint_without_artifact_manifest_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            lora_path = self._write_checkpoint(Path(tmpdir), config)
            (lora_path.parent / "state.json").unlink()

            result = validate_checkpoint_compatibility(config, lora_path)

            self.assertFalse(result["verified"])
            self.assertEqual(
                result["status"],
                "config_verified_artifacts_unverified",
            )
            self.assertEqual(
                result["artifact_verification"]["status"],
                "legacy_unverified",
            )

    def test_artifact_hash_mismatch_is_rejected_or_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            lora_path = self._write_checkpoint(Path(tmpdir), config)
            lora_path.write_bytes(b"tampered checkpoint")

            with self.assertRaisesRegex(ValueError, "artifact integrity"):
                validate_checkpoint_compatibility(config, lora_path)

            result = validate_checkpoint_compatibility(
                config,
                lora_path,
                allow_mismatch=True,
            )
            self.assertFalse(result["verified"])
            self.assertEqual(result["status"], "artifact_mismatch_allowed")
            self.assertEqual(
                result["artifact_verification"]["status"],
                "mismatch_allowed",
            )

    def test_inference_only_latent_anchoring_does_not_change_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_adaptive_config(train_enabled=False, predict_enabled=False)
            lora_path = self._write_checkpoint(Path(tmpdir), training_config)
            inference_config = make_adaptive_config(train_enabled=False, predict_enabled=False)
            inference_config.predict.latent_anchoring = LatentAnchoringConfig(
                mode="expanded_soft",
                interaction_dilation_px=32,
                transition_width_px=32,
            )

            result = validate_checkpoint_compatibility(inference_config, lora_path)

            self.assertTrue(result["verified"])
            self.assertEqual(result["mismatches"], [])

    def test_disabled_retired_modes_are_ignored_for_legacy_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            lora_path = self._write_checkpoint(Path(tmpdir), config)
            train_config_path = lora_path.parent.parent / "train_config.yaml"
            payload = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
            payload["model"].update(
                {
                    "query_prompt": {"enabled": False},
                    "pre_insertion": {"enabled": False},
                    "layout_condition": {"enabled": False},
                }
            )
            train_config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            result = validate_checkpoint_compatibility(config, lora_path)

            self.assertTrue(result["verified"])

    def test_enabled_retired_mode_points_to_its_worktree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            lora_path = self._write_checkpoint(Path(tmpdir), config)
            train_config_path = lora_path.parent.parent / "train_config.yaml"
            payload = yaml.safe_load(train_config_path.read_text(encoding="utf-8"))
            payload["model"]["layout_condition"] = {"enabled": True}
            train_config_path.write_text(
                yaml.safe_dump(payload, sort_keys=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "inpainting-v2-layout-condition"):
                validate_checkpoint_compatibility(config, lora_path)
            with self.assertRaisesRegex(ValueError, "inpainting-v2-layout-condition"):
                validate_checkpoint_compatibility(config, lora_path, allow_mismatch=True)

    def test_mask_geometry_encoding_is_part_of_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_adaptive_config()
            training_config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="geometry_tokenizer",
                sigma_px=24.0,
            )
            lora_path = self._write_checkpoint(Path(tmpdir), training_config)

            inference_config = make_adaptive_config()
            inference_config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="rgb_vae",
            )

            with self.assertRaisesRegex(ValueError, "model.mask_control.encoding"):
                validate_checkpoint_compatibility(inference_config, lora_path)

    def test_mask_geometry_semantic_fields_are_part_of_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_adaptive_config()
            training_config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="geometry_tokenizer",
                sigma_px=16.0,
                mask_threshold=0.5,
                tokenizer_hidden_channels=32,
            )
            lora_path = self._write_checkpoint(Path(tmpdir), training_config)

            overrides = {
                "sigma_px": 32.0,
                "mask_threshold": 0.25,
                "tokenizer_hidden_channels": 64,
            }
            for field, value in overrides.items():
                with self.subTest(field=field):
                    kwargs = {
                        "enabled": True,
                        "encoding": "geometry_tokenizer",
                        "sigma_px": 16.0,
                        "mask_threshold": 0.5,
                        "tokenizer_hidden_channels": 32,
                    }
                    kwargs[field] = value
                    inference_config = make_adaptive_config()
                    inference_config.model.mask_control = MaskControlConfig(
                        **kwargs
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        f"model.mask_control.{field}",
                    ):
                        validate_checkpoint_compatibility(
                            inference_config,
                            lora_path,
                        )

    def test_inactive_mask_control_ignores_geometry_only_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_adaptive_config()
            training_config.model.mask_control = MaskControlConfig(
                enabled=False,
                encoding="geometry_tokenizer",
                sigma_px=24.0,
                mask_threshold=0.2,
                tokenizer_hidden_channels=64,
            )
            lora_path = self._write_checkpoint(Path(tmpdir), training_config)

            inference_config = make_adaptive_config()
            inference_config.model.mask_control = MaskControlConfig(
                enabled=False,
                encoding="rgb_vae",
                sigma_px=3.0,
                mask_threshold=0.8,
                tokenizer_hidden_channels=4,
            )

            result = validate_checkpoint_compatibility(
                inference_config,
                lora_path,
            )

            self.assertTrue(result["verified"])
            self.assertEqual(result["mismatches"], [])

    def test_artifact_manifest_verifies_geometry_companion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_adaptive_config()
            config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="geometry_tokenizer",
            )
            lora_path = self._write_checkpoint(Path(tmpdir), config)

            result = validate_checkpoint_compatibility(config, lora_path)

            roles = {
                item["role"]
                for item in result["artifact_verification"]["files"]
            }
            self.assertTrue(result["verified"])
            self.assertEqual(roles, {"lora_peft", "mask_geometry_tokenizer"})

    def test_resume_rejects_mask_geometry_semantic_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_config = make_adaptive_config()
            checkpoint_config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="geometry_tokenizer",
                sigma_px=16.0,
            )
            lora_path = self._write_checkpoint(Path(tmpdir), checkpoint_config)
            checkpoint_train_config = lora_path.parent / "train_config.yaml"
            checkpoint_train_config.write_text(
                yaml.safe_dump(
                    checkpoint_config.model_dump(mode="json", exclude_none=True),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            current_config = make_adaptive_config()
            current_config.model.mask_control = MaskControlConfig(
                enabled=True,
                encoding="geometry_tokenizer",
                sigma_px=32.0,
            )

            with self.assertRaisesRegex(ValueError, "mask-control contract mismatch"):
                validate_resume_mask_control_compatibility(
                    current_config,
                    lora_path.parent,
                )


class EffectiveMetadataTest(unittest.TestCase):
    def test_geometry_encoding_is_recorded_as_the_effective_mask_path(self):
        config = make_adaptive_config()
        config.model.mask_control = MaskControlConfig(
            enabled=True,
            encoding="geometry_tokenizer",
        )

        metadata = build_effective_inference_metadata(
            config,
            prompt="insert",
            negative_prompt=None,
            num_inference_steps=20,
            true_cfg_scale=1.0,
            guidance_scale=1.0,
        )

        self.assertTrue(metadata["modes"]["mask_control_active"])
        self.assertEqual(
            metadata["modes"]["mask_control_encoding"],
            "geometry_tokenizer",
        )

    def test_disabled_guidance_embedding_is_recorded_as_inactive(self):
        config = make_adaptive_config()

        metadata = build_effective_inference_metadata(
            config,
            prompt="insert",
            negative_prompt=None,
            num_inference_steps=20,
            true_cfg_scale=4.0,
            guidance_scale=7.0,
            guidance_embeds_enabled=False,
            requested_height=768,
            requested_width=512,
            effective_height=1024,
            effective_width=672,
        )

        self.assertFalse(metadata["modes"]["guidance_scale_active"])
        self.assertEqual(metadata["effective"]["guidance_scale"], 1.0)
        self.assertEqual(metadata["requested"]["guidance_scale"], 7.0)
        self.assertEqual(metadata["requested"]["height"], 768)
        self.assertEqual(metadata["effective"]["width"], 672)

    def test_text_and_reference_cfg_modes_are_distinguished(self):
        native_config = make_adaptive_config()
        native = build_effective_inference_metadata(
            native_config,
            prompt="insert",
            negative_prompt="bad result",
            num_inference_steps=20,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
        )
        self.assertTrue(native["modes"]["text_true_cfg_active"])
        self.assertEqual(native["effective"]["text_true_cfg_scale"], 4.0)
        self.assertFalse(native["modes"]["reference_cfg_active"])

        reference_config = make_adaptive_config()
        reference_config.model.reference_cfg.enabled = True
        reference_config.model.reference_cfg.force_empty_prompt = True
        reference = build_effective_inference_metadata(
            reference_config,
            prompt="ignored",
            negative_prompt=None,
            num_inference_steps=20,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
        )
        self.assertEqual(reference["effective"]["prompt"], "")
        self.assertTrue(reference["modes"]["reference_cfg_active"])
        self.assertEqual(reference["effective"]["reference_cfg_scale"], 4.0)
        self.assertEqual(reference["effective"]["text_true_cfg_scale"], 1.0)

if __name__ == "__main__":
    unittest.main()

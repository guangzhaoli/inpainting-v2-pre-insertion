import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from pydantic import ValidationError

from src.data.config import (
    Config,
    HybridMaskingConfig,
    LossConfig,
    MaskControlConfig,
    ReferenceCFGConfig,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    QwenImageEditInsertAnythingTrainer,
)
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer


FINAL_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "exp"
    / "final_architecture.yaml"
)


def passthrough_prepare_embeddings(_self, batch, stage="fit"):
    return dict(batch)


class FinalArchitectureConfigTest(unittest.TestCase):
    def test_combined_yaml_selects_every_requested_component(self):
        raw = OmegaConf.to_container(OmegaConf.load(FINAL_CONFIG), resolve=True)
        # Validate the portable schema without requiring CUDA or this cluster's
        # optional LPIPS cache.  The checked-in paths are validated in real runs.
        raw["predict"]["devices"] = {}
        raw["loss"]["lpips_linear_path"] = None
        raw["loss"]["lpips_torch_home"] = None
        raw["loss"]["lpips_trunk_path"] = None

        config = Config.model_validate(raw)

        self.assertEqual(config.trainer.value, "QwenImageEditInsertAnything")
        self.assertTrue(config.model.reference_cfg.enabled)
        self.assertTrue(config.model.mask_control.enabled)
        self.assertEqual(config.model.mask_control.encoding, "geometry_tokenizer")
        self.assertTrue(config.data.init_args.adaptive_crop.train_enabled)
        self.assertTrue(config.data.init_args.adaptive_crop.predict_enabled)
        self.assertTrue(config.data.init_args.hybrid_masking.enabled)
        self.assertEqual(config.data.init_args.hybrid_masking.strategy, "maskmix_v2")
        self.assertTrue(config.data.init_args.hybrid_masking.ensure_gt_coverage)
        self.assertEqual(config.loss.mask_loss_mode, "region_balanced")
        self.assertTrue(config.loss.region_perceptual)
        self.assertGreater(config.loss.fg_lpips_weight, 0)
        self.assertGreater(config.loss.bg_l1_weight, 0)
        self.assertEqual(config.loss.region_max_sigma, 0.5)
        self.assertEqual(config.loss.ref_dino_weight, 0)

    def test_region_perceptual_requires_the_gt_mask_path(self):
        with self.assertRaisesRegex(ValidationError, "requires loss.mask_loss=true"):
            LossConfig(region_perceptual=True, mask_loss=False)

    def test_disabled_lpips_term_does_not_touch_external_weight_paths(self):
        config = LossConfig(
            mask_loss=True,
            region_perceptual=True,
            fg_metric="none",
            fg_lpips_weight=0.0,
            bg_l1_weight=0.5,
            lpips_linear_path="/definitely/missing/lpips-linear.pth",
            lpips_torch_home="/definitely/missing/torch-home",
            lpips_trunk_path="/definitely/missing/lpips-trunk.pth",
        )
        self.assertEqual(config.fg_metric, "none")

    def test_region_switch_requires_an_actual_pixel_region_term(self):
        with self.assertRaisesRegex(ValidationError, "pixel-region loss weights"):
            LossConfig(
                mask_loss=True,
                region_perceptual=True,
                latent_mse_weight=1.0,
                fg_metric="none",
                fg_lpips_weight=0.0,
                bg_l1_weight=0.0,
                ref_dino_weight=0.0,
            )

    def test_none_foreground_metric_cannot_silently_ignore_weight(self):
        with self.assertRaisesRegex(ValidationError, "fg_lpips_weight must be 0"):
            LossConfig(
                mask_loss=True,
                region_perceptual=True,
                fg_metric="none",
                fg_lpips_weight=0.5,
                bg_l1_weight=0.5,
            )

    def test_geometry_tokenizer_is_restricted_to_insert_anything(self):
        with self.assertRaisesRegex(ValidationError, "QwenImageEditInsertAnything"):
            Config.model_validate(
                {
                    "trainer": "QwenImageEdit",
                    "model": {
                        "mask_control": {
                            "enabled": True,
                            "encoding": "geometry_tokenizer",
                        }
                    },
                }
            )


class FinalArchitectureMaskRoutingTest(unittest.TestCase):
    def _trainer(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = SimpleNamespace(
            data=SimpleNamespace(
                init_args=SimpleNamespace(
                    hybrid_masking=HybridMaskingConfig(
                        enabled=True,
                        strategy="maskmix_v2",
                        ensure_gt_coverage=True,
                    )
                )
            ),
            model=SimpleNamespace(
                mask_control=MaskControlConfig(
                    enabled=True,
                    encoding="geometry_tokenizer",
                    sigma_px=2.0,
                ),
                reference_cfg=ReferenceCFGConfig(enabled=False),
            ),
            loss=LossConfig(
                mask_loss=True,
                mask_loss_mode="region_balanced",
                region_perceptual=True,
                fg_metric="none",
                fg_lpips_weight=0.0,
                bg_l1_weight=0.5,
            ),
        )
        return trainer

    @staticmethod
    def _batch():
        gt_mask = torch.zeros(1, 12, 12)
        gt_mask[:, 5:8, 5:8] = 1.0
        condition_mask = torch.zeros_like(gt_mask)
        condition_mask[:, 3:10, 2:11] = 1.0
        ref_mask = torch.zeros_like(gt_mask)
        ref_mask[:, 2:9, 4:8] = 1.0
        return {
            "ref_image": torch.full((1, 3, 12, 12), 0.25),
            "ref_mask": ref_mask,
            "tar_image": torch.ones(1, 3, 12, 12),
            "tar_mask": gt_mask,
            "tar_mask_cond": condition_mask,
            "prompt": [""],
            "category": ["accessory"],
        }

    def test_condition_geometry_and_gt_losses_use_their_own_masks(self):
        trainer = self._trainer()
        batch = self._batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="fit")

        condition = batch["tar_mask_cond"]
        expected_hole = batch["tar_image"] * (1 - condition.unsqueeze(1))
        torch.testing.assert_close(output["control"], expected_hole)
        torch.testing.assert_close(output["mask"], batch["tar_mask"])
        torch.testing.assert_close(output["mask_geometry"][:, 0], condition)

        # Geometry replaces the legacy RGB-VAE mask control: reference is the
        # only additional VAE control in the positive branch.
        self.assertEqual(output["n_controls"], 1)
        torch.testing.assert_close(output["control_1"], batch["ref_image"])
        self.assertNotIn("vae_only_control_keys", output)

        # Region-perceptual supervision keeps original GT/reference pixels.
        self.assertIs(output["target_image_for_loss"], batch["tar_image"])
        self.assertIs(output["ref_image_for_loss"], batch["ref_image"])
        self.assertIs(output["ref_mask_for_loss"], batch["ref_mask"])

    def test_real_plus_prepare_keeps_pixel_mask_before_latent_packing(self):
        """Lock the InsertAnything MRO path, not the passthrough routing mock."""

        trainer = object.__new__(QwenImageEditPlusTrainer)
        trainer.config = SimpleNamespace(
            loss=SimpleNamespace(mask_loss=True, region_perceptual=True)
        )
        trainer.vae_scale_factor = 2
        trainer.num_channels_latents = 2
        trainer.weight_dtype = torch.float32
        trainer.preprocess_image_for_vae_encoder = (
            lambda image, **_kwargs: image.unsqueeze(2)
        )
        trainer.preprocess_image_for_text_encoder = lambda image, **_kwargs: image
        trainer.encode_prompt = lambda prompt, image=None: (
            torch.zeros(len(prompt), 2, 8),
            torch.ones(len(prompt), 2, dtype=torch.long),
        )
        trainer.encode_image_latents = lambda image, batch_size, *_args: torch.zeros(
            batch_size,
            16,
            8,
        )
        trainer._get_image_shapes = lambda _batch, batch_size: [
            [(1, 4, 4)]
        ] * batch_size

        gt_mask = torch.zeros(1, 16, 16)
        gt_mask[:, 4:10, 5:11] = 1.0
        output = QwenImageEditPlusTrainer.prepare_embeddings(
            trainer,
            {
                "image": torch.ones(1, 3, 16, 16),
                "control": torch.zeros(1, 3, 16, 16),
                "mask": gt_mask,
                "prompt": [""],
                "n_controls": 0,
                "true_cfg_scale": 1.0,
            },
            stage="fit",
        )

        torch.testing.assert_close(output["pixel_mask"], gt_mask)
        self.assertEqual(tuple(output["mask"].shape), (1, 16))


if __name__ == "__main__":
    unittest.main()

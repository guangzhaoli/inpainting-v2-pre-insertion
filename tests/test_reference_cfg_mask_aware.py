import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.config import (
    ReferenceCFGConfig,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


def make_batch():
    tar_mask = torch.zeros(1, 8, 8)
    tar_mask[:, 2:6, 2:6] = 1.0
    return {
        "ref_image": torch.zeros(1, 3, 8, 8),
        "ref_mask": torch.ones(1, 8, 8),
        "tar_image": torch.ones(1, 3, 8, 8),
        "tar_mask": tar_mask,
        "prompt": ["insert object"],
        "category": ["object"],
        "true_cfg_scale": 4.0,
    }


def make_trainer(mask_aware, delta_clip_enabled=False, delta_clip_max_ratio=0.5):
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            reference_cfg=ReferenceCFGConfig(
                enabled=True,
                train_dropout_rate=0.0,
                force_empty_prompt=True,
                mask_aware=mask_aware,
                delta_clip_enabled=delta_clip_enabled,
                delta_clip_max_ratio=delta_clip_max_ratio,
            ),
        ),
        loss=SimpleNamespace(mask_loss=False),
    )
    return trainer


def passthrough_prepare_embeddings(_self, batch, stage="fit"):
    return dict(batch)


class ReferenceCFGMaskAwareTest(unittest.TestCase):
    def test_mask_aware_combination_only_guides_masked_tokens(self):
        positive = torch.tensor([[[2.0, 0.0], [3.0, 4.0]]])
        null_reference = torch.tensor([[[1.0, 0.0], [0.0, 4.0]]])
        mask = torch.tensor([[1.0, 0.0]])

        masked = QwenImageEditTrainer._combine_reference_cfg_predictions(
            noise_pred=positive,
            reference_cfg_noise_pred=null_reference,
            reference_cfg_scale=3.0,
            reference_cfg_mask=mask,
        )
        unmasked = QwenImageEditTrainer._combine_reference_cfg_predictions(
            noise_pred=positive,
            reference_cfg_noise_pred=null_reference,
            reference_cfg_scale=3.0,
        )

        self.assertTrue(torch.allclose(masked[:, 0], unmasked[:, 0]))
        self.assertTrue(torch.allclose(masked[:, 1], positive[:, 1]))

    def test_delta_clip_limits_delta_without_final_norm_rescale(self):
        positive = torch.tensor([[[11.0, 0.0]]])
        null_reference = torch.tensor([[[1.0, 0.0]]])

        output = QwenImageEditTrainer._combine_reference_cfg_predictions(
            noise_pred=positive,
            reference_cfg_noise_pred=null_reference,
            reference_cfg_scale=3.0,
            delta_clip_enabled=True,
            delta_clip_max_ratio=0.2,
        )

        self.assertTrue(torch.allclose(output, torch.tensor([[[7.6, 0.0]]]), atol=1e-5))

    def test_delta_clip_mask_aware_keeps_unmasked_positive_prediction(self):
        positive = torch.tensor([[[11.0, 0.0], [21.0, 0.0]]])
        null_reference = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        mask = torch.tensor([[1.0, 0.0]])

        output = QwenImageEditTrainer._combine_reference_cfg_predictions(
            noise_pred=positive,
            reference_cfg_noise_pred=null_reference,
            reference_cfg_scale=3.0,
            reference_cfg_mask=mask,
            delta_clip_enabled=True,
            delta_clip_max_ratio=0.2,
        )

        self.assertTrue(torch.allclose(output[:, 0], torch.tensor([[7.6, 0.0]]), atol=1e-5))
        self.assertTrue(torch.allclose(output[:, 1], positive[:, 1]))

    def test_prepare_embeddings_carries_mask_when_mask_aware_enabled(self):
        trainer = make_trainer(mask_aware=True)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        self.assertTrue(output["reference_cfg_mask_aware"])
        self.assertTrue(torch.equal(output["reference_cfg_mask"], batch["tar_mask"]))
        self.assertIn("reference_cfg_embeddings", output)

    def test_prepare_embeddings_omits_mask_when_mask_aware_disabled(self):
        trainer = make_trainer(mask_aware=False)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        self.assertNotIn("reference_cfg_mask_aware", output)
        self.assertNotIn("reference_cfg_mask", output)
        self.assertIn("reference_cfg_embeddings", output)

    def test_prepare_embeddings_carries_delta_clip_config(self):
        trainer = make_trainer(mask_aware=False, delta_clip_enabled=True, delta_clip_max_ratio=0.25)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        self.assertTrue(output["reference_cfg_delta_clip_enabled"])
        self.assertEqual(output["reference_cfg_delta_clip_max_ratio"], 0.25)

    def test_prepare_embeddings_carries_interval_and_apg_defaults(self):
        trainer = make_trainer(mask_aware=False)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        self.assertEqual(output["reference_cfg_sigma_min"], 0.75)
        self.assertEqual(output["reference_cfg_apg_eta"], 0.0)


def apg_inputs():
    generator = torch.Generator().manual_seed(0)
    positive, null_reference, latents = (torch.randn(2, 6, 4, generator=generator) for _ in range(3))
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]])
    return positive, null_reference, latents, mask


def combine_apg(apg_eta, positive, null_reference, latents, mask, sigma=0.8, **kwargs):
    return QwenImageEditTrainer._combine_reference_cfg_predictions(
        noise_pred=positive, reference_cfg_noise_pred=null_reference, reference_cfg_scale=3.0,
        reference_cfg_mask=mask, apg_eta=apg_eta, latents=latents, sigma=sigma, **kwargs,
    )


class ReferenceCFGAPGTest(unittest.TestCase):
    def test_defaults_are_scale_3_interval_and_full_apg(self):
        config = ReferenceCFGConfig()
        self.assertEqual((config.scale, config.sigma_min, config.apg_eta), (3.0, 0.75, 0.0))

    def test_eta_one_is_plain_cfg(self):
        positive, null_reference, latents, mask = apg_inputs()
        plain = QwenImageEditTrainer._combine_reference_cfg_predictions(
            noise_pred=positive, reference_cfg_noise_pred=null_reference, reference_cfg_scale=3.0,
            reference_cfg_mask=mask,
        )
        self.assertTrue(torch.equal(combine_apg(1.0, positive, null_reference, latents, mask), plain))

    def test_eta_zero_guidance_is_orthogonal_to_conditional_x0_inside_mask(self):
        positive, null_reference, latents, mask = apg_inputs()
        sigma = 0.8
        output = combine_apg(0.0, positive, null_reference, latents, mask, sigma=sigma,
                             delta_clip_enabled=True, delta_clip_max_ratio=100.0)
        x0_c = latents - sigma * positive
        inner = ((output - positive) * x0_c * mask.unsqueeze(-1)).sum(dim=(1, 2))
        self.assertTrue(torch.allclose(inner, torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.equal(output[mask == 0], positive[mask == 0]))

    def test_eta_half_is_midpoint_of_zero_and_one(self):
        positive, null_reference, latents, mask = apg_inputs()
        kwargs = dict(delta_clip_enabled=True, delta_clip_max_ratio=1.0)
        half, zero, one = (combine_apg(eta, positive, null_reference, latents, mask, **kwargs) for eta in (0.5, 0.0, 1.0))
        self.assertTrue(torch.allclose(half, (zero + one) / 2, atol=1e-6))

    def test_apg_keeps_the_prediction_dtype(self):
        positive, null_reference, latents, mask = (t.to(torch.bfloat16) for t in apg_inputs())
        self.assertEqual(combine_apg(0.0, positive, null_reference, latents, mask).dtype, torch.bfloat16)

    def test_apg_needs_latents_and_sigma(self):
        positive, null_reference, _, mask = apg_inputs()
        with self.assertRaises(ValueError):
            combine_apg(0.0, positive, null_reference, None, mask)

    def test_cli_defaults_to_config_scale_unless_given(self):
        from infer import apply_reference_cfg_defaults

        config = SimpleNamespace(model=SimpleNamespace(reference_cfg=ReferenceCFGConfig(enabled=True)))
        implicit = SimpleNamespace(true_cfg_scale=4.0, _true_cfg_scale_explicit=False)
        explicit = SimpleNamespace(true_cfg_scale=2.5, _true_cfg_scale_explicit=True)
        apply_reference_cfg_defaults(config, implicit)
        apply_reference_cfg_defaults(config, explicit)
        self.assertEqual((implicit.true_cfg_scale, explicit.true_cfg_scale), (3.0, 2.5))


if __name__ == "__main__":
    unittest.main()

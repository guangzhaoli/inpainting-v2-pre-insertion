import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


class _FakeSamplingDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(in_channels=16, guidance_embeds=False)
        self.forward_calls = 0

    def cache_context(self, _name):
        return nullcontext()

    def forward(self, hidden_states, **_kwargs):
        self.forward_calls += 1
        values = (
            hidden_states.new_tensor([2.0, 0.0, 0.0, 0.0])
            if self.forward_calls == 1
            else hidden_states.new_tensor([0.0, 1.0, 0.0, 0.0])
        )
        prediction = values.view(1, 1, 4).expand_as(hidden_states) + self.device_anchor * 0
        return (prediction,)


class _FakeScheduler:
    def __init__(self):
        self.last_noise_pred = None
        self.begin_index = None

    def set_begin_index(self, index):
        self.begin_index = index

    def step(self, noise_pred, _timestep, latents, return_dict=False):
        self.last_noise_pred = noise_pred
        return (latents - 0.1 * noise_pred,)


class RuntimeHotspotRegressionTest(unittest.TestCase):
    def test_rotary_cache_created_in_validation_is_safe_for_training(self):
        from src.models.transformer_qwenimage import QwenEmbedRope, apply_rotary_emb_qwen

        rope = QwenEmbedRope(theta=10000, axes_dim=[2, 2, 4])
        with torch.inference_mode():
            rope([[(1, 2, 2)]], [3], torch.device("cpu"))
        freqs, _ = rope([[(1, 2, 2)]], [3], torch.device("cpu"))
        self.assertFalse(torch.is_inference(freqs))
        self.assertTrue(all(not torch.is_inference(v) for v in rope.rope_cache.values()))
        x = torch.randn(1, 4, 2, 8, requires_grad=True)
        result = apply_rotary_emb_qwen(x, freqs, use_real=False)
        result.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_fp32_scheduler_output_preserves_bf16_anchoring_stream(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = None
        trainer.dit = _FakeSamplingDit()
        trainer.scheduler = _FakeScheduler()
        trainer.scheduler.sigmas = torch.tensor([1.0, 0.0])
        original_step = trainer.scheduler.step
        trainer.scheduler.step = lambda *a, **kw: (original_step(*a, **kw)[0].float(),)
        trainer.weight_dtype = torch.bfloat16
        trainer.vae_scale_factor = 8
        trainer.prepare_initial_latents = Mock(return_value=torch.zeros(1, 2, 4, dtype=torch.bfloat16))
        trainer.prepare_predict_timesteps = Mock(return_value=(torch.tensor([1000.0]), 1))
        source = torch.full((1, 2, 4), .5, dtype=torch.bfloat16)
        weight = torch.zeros_like(source)
        weight[:, 1] = 1
        embeddings = {
            "control_latents": torch.zeros_like(source),
            "prompt_embeds": torch.zeros(1, 1, 4, dtype=torch.bfloat16),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "height": 16, "width": 16, "height_control": 16, "width_control": 16,
            "n_controls": 0, "num_inference_steps": 1, "true_cfg_scale": 1.0,
            "guidance": 1.0, "anchor_source_latents": source,
            "anchor_edit_weight": weight, "latent_anchoring_mode": "expanded_soft",
        }
        with patch("src.trainer.qwen_image_edit_trainer.tqdm", side_effect=lambda iterable, **kw: iterable):
            result = trainer.sampling_from_embeddings(embeddings)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result[:, 0], source[:, 0])
        self.assertFalse(torch.equal(result[:, 1], source[:, 1]))

    def test_all_zero_region_weights_skip_decode_and_criterion(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.weight_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1)
        )
        trainer.config = SimpleNamespace(
            loss=SimpleNamespace(
                region_perceptual=True,
                region_sigma_power=2.0,
                region_max_sigma=0.4,
                latent_mse_weight=2.0,
                fg_lpips_weight=1.0,
                bg_l1_weight=1.0,
                ref_dino_weight=1.0,
            )
        )
        trainer.dit = Mock(
            side_effect=lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        )
        trainer.forward_loss = Mock(
            return_value=torch.tensor(3.0, requires_grad=True)
        )
        trainer.region_perceptual_criterion = Mock(
            side_effect=AssertionError("gated region criterion must not run")
        )
        trainer.decode_vae_latent_for_loss = Mock(
            side_effect=AssertionError("gated VAE decode must not run")
        )
        trainer._get_cached_scheduler_training_tensors = Mock(
            return_value=(torch.tensor([1000.0]), torch.tensor([0.9]))
        )
        trainer._get_sigmas = Mock(return_value=torch.full((1, 1, 1), 0.9))

        embeddings = {
            "image_latents": torch.zeros(1, 2, 4),
            "control_latents": torch.zeros(1, 2, 4),
            "prompt_embeds": torch.zeros(1, 1, 4),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "img_shapes": [[(1, 1, 2), (1, 1, 2)]],
            "height": 16,
            "width": 16,
            "pixel_mask": torch.ones(1, 16, 16),
            "ref_image_for_loss": torch.zeros(1, 3, 16, 16),
            "ref_mask_for_loss": torch.ones(1, 16, 16),
        }

        loss = trainer._compute_loss(embeddings)

        self.assertEqual(float(loss.detach()), 6.0)
        loss.backward()
        trainer.decode_vae_latent_for_loss.assert_not_called()
        trainer.region_perceptual_criterion.assert_not_called()
        self.assertEqual(
            set(trainer._step_loss_metrics),
            {
                "train/latent_mse",
                "train/latent_mse_weighted",
                "train/region_total",
                "train/region_fg_lpips",
                "train/region_bg_l1",
                "train/region_ref_dino",
                "train/region_detail_lpips",
                "train/region_boundary_l1",
                "train/region_face_dino",
                "train/face_pair_valid_fraction",
                "train/region_w_t_mean",
                "train/region_w_t_max",
                "train/sigma_mean",
            },
        )
        self.assertEqual(float(trainer._step_loss_metrics["train/latent_mse_weighted"]), 6.0)
        for key in (
            "train/region_total",
            "train/region_fg_lpips",
            "train/region_bg_l1",
            "train/region_ref_dino",
            "train/region_w_t_mean",
            "train/region_w_t_max",
        ):
            self.assertEqual(float(trainer._step_loss_metrics[key]), 0.0)

    def test_mixed_region_weights_keep_full_batch_and_original_weighting(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.weight_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=2)
        )
        trainer.config = SimpleNamespace(
            loss=SimpleNamespace(
                region_perceptual=True,
                region_sigma_power=2.0,
                region_max_sigma=0.5,
                latent_mse_weight=1.0,
                fg_lpips_weight=0.5,
                bg_l1_weight=0.5,
                ref_dino_weight=0.0,
            )
        )
        trainer.dit = Mock(
            side_effect=lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        )
        trainer.forward_loss = Mock(
            return_value=torch.tensor(3.0, requires_grad=True)
        )
        trainer.decode_vae_latent_for_loss = Mock(
            return_value=torch.zeros(2, 3, 16, 16, requires_grad=True)
        )
        trainer._loaded_model_decode_context = Mock(return_value=nullcontext())
        trainer._get_cached_scheduler_training_tensors = Mock(
            return_value=(torch.tensor([1000.0, 0.0]), torch.tensor([0.9, 0.1]))
        )
        trainer.region_perceptual_criterion = Mock(
            return_value={
                "total": torch.tensor(2.0, requires_grad=True),
                "fg_lpips": torch.tensor(1.0),
                "bg_l1": torch.tensor(1.0),
                "ref_dino": torch.tensor(0.0),
            }
        )

        embeddings = {
            "image_latents": torch.zeros(2, 2, 4),
            "_analysis_timestep_u": torch.tensor([0.0, 0.5]),
            "control_latents": torch.zeros(2, 2, 4),
            "prompt_embeds": torch.zeros(2, 1, 4),
            "prompt_embeds_mask": torch.ones(2, 1, dtype=torch.long),
            "img_shapes": [[(1, 1, 2), (1, 1, 2)]] * 2,
            "height": 16,
            "width": 16,
            "pixel_mask": torch.ones(2, 16, 16),
            "target_image_for_loss": torch.zeros(2, 3, 16, 16),
        }

        loss = trainer._compute_loss(embeddings)

        self.assertEqual(float(loss.detach()), 5.0)
        trainer.decode_vae_latent_for_loss.assert_called_once()
        decoded_latents = trainer.decode_vae_latent_for_loss.call_args.args[0]
        self.assertEqual(decoded_latents.shape[0], 2)
        trainer.region_perceptual_criterion.assert_called_once()
        criterion_kwargs = trainer.region_perceptual_criterion.call_args.kwargs
        torch.testing.assert_close(
            criterion_kwargs["sample_weights"],
            torch.tensor([0.0, 0.81]),
        )

    def test_gated_region_step_still_validates_reference_dino_inputs(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.weight_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1)
        )
        trainer.config = SimpleNamespace(
            loss=SimpleNamespace(
                region_perceptual=True,
                region_sigma_power=2.0,
                region_max_sigma=0.4,
                latent_mse_weight=1.0,
                fg_lpips_weight=1.0,
                bg_l1_weight=1.0,
                ref_dino_weight=1.0,
            )
        )
        trainer.dit = Mock(
            side_effect=lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]),)
        )
        trainer.forward_loss = Mock(return_value=torch.tensor(1.0))
        trainer.region_perceptual_criterion = Mock()
        trainer._get_cached_scheduler_training_tensors = Mock(
            return_value=(torch.tensor([1000.0]), torch.tensor([0.9]))
        )
        trainer._get_sigmas = Mock(return_value=torch.full((1, 1, 1), 0.9))
        embeddings = {
            "image_latents": torch.zeros(1, 2, 4),
            "control_latents": torch.zeros(1, 2, 4),
            "prompt_embeds": torch.zeros(1, 1, 4),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "img_shapes": [[(1, 1, 2), (1, 1, 2)]],
            "height": 16,
            "width": 16,
            "pixel_mask": torch.ones(1, 16, 16),
        }

        with self.assertRaisesRegex(ValueError, "ref_dino_weight > 0"):
            trainer._compute_loss(embeddings)

        trainer.region_perceptual_criterion.assert_not_called()

    def test_reference_cfg_keeps_predictions_on_device_without_allocator_flush(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = None
        trainer.dit = _FakeSamplingDit()
        trainer.scheduler = _FakeScheduler()
        trainer.weight_dtype = torch.float32
        trainer.vae_scale_factor = 8
        trainer.prepare_initial_latents = Mock(
            return_value=torch.zeros(1, 2, 4)
        )
        trainer.prepare_predict_timesteps = Mock(
            return_value=(torch.tensor([1000.0]), 1)
        )

        branch_embeddings = {
            "control_latents": torch.zeros(1, 2, 4),
            "prompt_embeds": torch.zeros(1, 1, 4),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "height": 16,
            "width": 16,
            "height_control": 16,
            "width_control": 16,
            "n_controls": 0,
        }
        embeddings = {
            **branch_embeddings,
            "num_inference_steps": 1,
            "true_cfg_scale": 1.0,
            "guidance": 1.0,
            "reference_cfg_scale": 2.0,
            "reference_cfg_embeddings": dict(branch_embeddings),
        }

        with (
            patch.object(
                torch.Tensor,
                "cpu",
                side_effect=AssertionError("reference CFG must not round-trip through CPU"),
            ) as tensor_cpu,
            patch("torch.cuda.empty_cache") as empty_cache,
            patch(
                "src.trainer.qwen_image_edit_trainer.tqdm",
                side_effect=lambda iterable, **_kwargs: iterable,
            ),
        ):
            output = trainer.sampling_from_embeddings(embeddings)

        tensor_cpu.assert_not_called()
        empty_cache.assert_not_called()
        self.assertEqual(trainer.dit.forward_calls, 2)
        self.assertEqual(trainer.scheduler.begin_index, 0)
        self.assertEqual(tuple(output.shape), (1, 2, 4))
        self.assertIsNotNone(trainer.scheduler.last_noise_pred)
        self.assertEqual(
            trainer.scheduler.last_noise_pred.device,
            trainer.dit.device_anchor.device,
        )
        expected = torch.tensor([4.0, -1.0, 0.0, 0.0])
        expected = expected * (2.0 / torch.linalg.vector_norm(expected))
        torch.testing.assert_close(
            trainer.scheduler.last_noise_pred,
            expected.view(1, 1, 4).expand(1, 2, 4),
        )


if __name__ == "__main__":
    unittest.main()

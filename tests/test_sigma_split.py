import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import safetensors.torch
import torch
import torch.nn.functional as F

from src.data.config import (
    AlphaHeadConfig,
    HybridMaskingConfig,
    PredictConfig,
    ReferenceCFGConfig,
    SigmaSplitConfig,
)
from src.loss.region_perceptual_loss import RegionPerceptualLoss
from src.models.alpha_head import ALPHA_HEAD_FILE_NAME, AlphaHead, alpha_head_loss, alpha_occupancy_targets
from src.models.transformer_qwenimage import QwenImageTransformer2DModel
from src.trainer.base_trainer import BaseTrainer
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.inference_metadata import validate_checkpoint_artifacts
from src.utils.sigma_split import (
    alpha_region_from_logits,
    augment_reference_light,
    augment_reference_strong,
    handoff_step_index,
    route_high_noise,
    shuffle_token_segment,
)

TRAINER_MODULE = "src.trainer.qwen_image_edit_insert_anything_trainer"


def tiny_dit(num_layers: int = 2) -> QwenImageTransformer2DModel:
    torch.manual_seed(0)
    return QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=num_layers,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
    )


def dit_kwargs(batch_size: int, img_shapes):
    return {
        "encoder_hidden_states": torch.randn(batch_size, 3, 8),
        "encoder_hidden_states_mask": torch.ones(batch_size, 3, dtype=torch.long),
        "timestep": torch.full((batch_size,), 0.5),
        "img_shapes": img_shapes,
        "txt_seq_lens": [3] * batch_size,
        "return_dict": False,
    }


def make_trainer(*, handoff: bool = False, vl_reference: str = "drop", anchoring: str = "legacy"):
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    hybrid = HybridMaskingConfig(
        enabled=True,
        strategy="maskmix_v2",
        context_guard={"enabled": True, "max_condition_area_ratio": 0.5},
        loose_placement={"enabled": True},
    )
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            reference_cfg=ReferenceCFGConfig(
                enabled=True, train_dropout_rate=0.0, force_empty_prompt=True, mask_aware=True
            ),
            sigma_split=SigmaSplitConfig(enabled=True, high_noise={"vl_reference": vl_reference}),
            alpha_head=AlphaHeadConfig(enabled=True),
            lora=SimpleNamespace(pretrained_weight=None),
        ),
        data=SimpleNamespace(init_args=SimpleNamespace(hybrid_masking=hybrid)),
        loss=SimpleNamespace(mask_loss=False, alpha_bce_weight=0.1, alpha_dice_weight=0.1),
        predict=PredictConfig(
            sigma_handoff={"enabled": handoff},
            latent_anchoring={"mode": anchoring, "interaction_dilation_px": 4, "transition_width_px": 4},
        ),
    )
    trainer.accelerator = None
    trainer.vae_scale_factor = 8
    trainer.weight_dtype = torch.float32
    schedule_sigmas = torch.linspace(1.0, 0.001, 1000)
    trainer.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    trainer._get_cached_scheduler_training_tensors = Mock(return_value=(schedule_sigmas * 1000, schedule_sigmas))
    return trainer


def fake_parent_prepare(_self, batch, stage="fit"):
    """Stand-in for Qwen encoding: one latent token per 16x16 pixels, numbered in order."""

    out = dict(batch)
    height, width = batch["control"].shape[-2:]
    out["height_control"], out["width_control"] = height, width
    tokens = (height // 16) * (width // 16)
    if batch["n_controls"] >= 1:
        ref_height, ref_width = batch["control_1"].shape[-2:]
        out["height_control_1"], out["width_control_1"] = ref_height, ref_width
        tokens += (ref_height // 16) * (ref_width // 16)
    out["control_latents"] = torch.arange(tokens, dtype=torch.float32).view(1, tokens, 1).repeat(
        batch["control"].shape[0], 1, 4
    )
    return out


def make_batch():
    tar_mask = torch.zeros(1, 32, 48)
    tar_mask[:, 12:20, 18:30] = 1
    loose = torch.zeros(1, 32, 48)
    loose[:, 2:30, 4:44] = 1  # 73% of the canvas: over the 50% guard
    ref_mask = torch.zeros(1, 48, 64)
    ref_mask[:, 8:40, 8:56] = 1
    ref_image = torch.rand(1, 3, 48, 64) * ref_mask.unsqueeze(1) + (1 - ref_mask.unsqueeze(1))
    return {
        "ref_image": ref_image,
        "ref_mask": ref_mask,
        "tar_image": torch.rand(1, 3, 32, 48),
        "tar_mask": tar_mask,
        "tar_mask_cond": loose,
        "tar_mask_cond_tight": tar_mask.clone(),
        "placement_metadata": {"loose_mode": ["loose_blob"], "tight_mode": ["exact"]},
        "prompt": [""],
        "category": ["cat"],
    }


def run_fit(trainer, batch, u: float):
    with (
        patch(f"{TRAINER_MODULE}.compute_density_for_timestep_sampling", return_value=torch.tensor([u])),
        patch.object(QwenImageEditPlusTrainer, "prepare_embeddings", autospec=True, side_effect=fake_parent_prepare),
    ):
        return trainer.prepare_embeddings(batch, stage="fit")


class TokenShuffleRopeTest(unittest.TestCase):
    def test_shuffling_reference_tokens_equals_shuffling_their_rope_positions(self):
        model = tiny_dit().eval()
        img_shapes = [[(1, 2, 2), (1, 2, 2), (1, 2, 3)]]
        start, length = 8, 6
        hidden = torch.randn(1, 14, 8)
        kwargs = dit_kwargs(1, img_shapes)

        shuffled = shuffle_token_segment(hidden, start, length, generator=torch.Generator().manual_seed(7))
        permutation = torch.randperm(length, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(shuffled[0, :start], hidden[0, :start])
        torch.testing.assert_close(shuffled[0, start:], hidden[0, start + permutation])

        inverse = torch.argsort(permutation)
        original_rope = model.pos_embed.forward

        def permuted_rope(video_fhw, txt_seq_lens, device):
            video, text = original_rope(video_fhw, txt_seq_lens, device)
            video = video.clone()
            video[start : start + length] = video[start : start + length][inverse]
            return video, text

        with torch.no_grad():
            shuffled_tokens = model(hidden_states=shuffled, **kwargs)[0][:, :4]
            plain = model(hidden_states=hidden, **kwargs)[0][:, :4]
            with patch.object(model.pos_embed, "forward", side_effect=permuted_rope):
                shuffled_positions = model(hidden_states=hidden, **kwargs)[0][:, :4]

        torch.testing.assert_close(shuffled_tokens, shuffled_positions, rtol=1e-5, atol=1e-5)
        self.assertFalse(torch.allclose(shuffled_tokens, plain, atol=1e-4))


class SigmaRoutingTest(unittest.TestCase):
    def test_route_is_fixed_outside_the_overlap_band_and_random_inside(self):
        sigmas = torch.tensor([0.95, 0.76, 0.64, 0.2, 0.7, 0.72, 0.68])
        outcomes = torch.stack(
            [route_high_noise(sigmas, 0.7, 0.05, generator=torch.Generator().manual_seed(seed)) for seed in range(300)]
        )
        self.assertTrue(bool(outcomes[:, 0].all()) and bool(outcomes[:, 1].all()))
        self.assertFalse(bool(outcomes[:, 2].any()) or bool(outcomes[:, 3].any()))
        for column in (4, 5, 6):
            rate = float(outcomes[:, column].float().mean())
            self.assertGreater(rate, 0.35)
            self.assertLess(rate, 0.65)
        self.assertEqual(route_high_noise(torch.tensor([0.7, 0.7001, 0.69]), 0.7, 0.0).tolist(), [False, True, False])

    def test_high_noise_route_uses_loose_mask_and_a_bag_of_reference_patches(self):
        trainer = make_trainer()
        batch = make_batch()
        out = run_fit(trainer, batch, u=0.1)  # sigma ~ 0.9

        self.assertEqual(out["_sigma_split_route_high"].tolist(), [True])
        torch.testing.assert_close(out["_sigma_split_timestep_u"], torch.tensor([0.1]))
        # Over the guard, but loose masks are never replaced by the exact GT.
        self.assertTrue(torch.equal(out["control"], batch["tar_image"] * (1 - batch["tar_mask_cond"].unsqueeze(1))))
        metadata = trainer.get_last_hybrid_masking_metadata()
        self.assertEqual(metadata["modes"], ["loose_blob"])
        self.assertTrue(bool(metadata["context_guard_triggered"][0]))
        self.assertFalse(bool(metadata["fallback_applied"][0]))
        # The augmented reference reaches only the VAE path ("drop").
        self.assertEqual(out["vae_only_control_keys"], ["control_1"])
        self.assertFalse(torch.equal(out["control_1"], batch["ref_image"]))
        latents = out["control_latents"][0, :, 0]
        torch.testing.assert_close(latents[:6], torch.arange(6.0))
        self.assertTrue(torch.equal(latents[6:].sort().values, torch.arange(6.0, 18.0)))
        self.assertFalse(torch.equal(latents[6:], torch.arange(6.0, 18.0)))
        self.assertTrue(torch.equal(out["alpha_target_mask"], batch["tar_mask"]))

    def test_augmented_vl_policy_keeps_the_reference_in_the_prompt(self):
        out = run_fit(make_trainer(vl_reference="augmented"), make_batch(), u=0.1)
        self.assertNotIn("vae_only_control_keys", out)

    def test_low_noise_route_uses_tight_mask_and_ordered_reference(self):
        trainer = make_trainer()
        batch = make_batch()
        out = run_fit(trainer, batch, u=0.8)  # sigma ~ 0.2

        self.assertEqual(out["_sigma_split_route_high"].tolist(), [False])
        self.assertTrue(torch.equal(out["control"], batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))))
        self.assertEqual(trainer.get_last_hybrid_masking_metadata()["modes"], ["exact"])
        self.assertNotIn("vae_only_control_keys", out)
        self.assertTrue(torch.equal(out["control_latents"][0, :, 0], torch.arange(18.0)))
        background = batch["ref_mask"][0] == 0
        self.assertTrue(torch.equal(out["control_1"][0][:, background], batch["ref_image"][0][:, background]))
        self.assertFalse(torch.equal(out["control_1"], batch["ref_image"]))

    def test_routing_requires_dataset_placement_masks(self):
        batch = make_batch()
        del batch["tar_mask_cond_tight"]
        with self.assertRaisesRegex(ValueError, "loose/tight placement masks"):
            run_fit(make_trainer(), batch, u=0.1)

    def test_reference_augmentations_keep_the_white_background(self):
        ref_mask = torch.zeros(2, 32, 32)
        ref_mask[:, 8:24, 8:24] = 1
        images = torch.rand(2, 3, 32, 32) * ref_mask.unsqueeze(1) + (1 - ref_mask.unsqueeze(1))
        high_cfg = SigmaSplitConfig().high_noise.model_copy(update={"rotation_deg": 0.0, "flip_prob": 0.0})
        strong, masks = augment_reference_strong(images, ref_mask, high_cfg)
        light = augment_reference_light(images, ref_mask, SigmaSplitConfig().low_noise)
        background = ref_mask.unsqueeze(1).expand_as(images) == 0
        self.assertTrue(torch.equal(strong[background], images[background]))
        self.assertTrue(torch.equal(light[background], images[background]))
        self.assertTrue(torch.equal(masks, ref_mask))
        identity_cfg = high_cfg.model_copy(update={"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0})
        torch.testing.assert_close(augment_reference_strong(images, ref_mask, identity_cfg)[0], images)


class AlphaHeadTest(unittest.TestCase):
    def test_dit_returns_token_logits_from_a_zero_initialized_head(self):
        model = tiny_dit(num_layers=1)
        model.alpha_head = AlphaHead(model.inner_dim)
        img_shapes = [[(1, 2, 2), (1, 2, 3)]] * 2
        output, logits = model(
            hidden_states=torch.randn(2, 10, 8),
            output_token_count=4,
            return_alpha_logits=True,
            **dit_kwargs(2, img_shapes),
        )
        self.assertEqual(tuple(output.shape), (2, 4, 8))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.equal(logits, torch.zeros(2, 4)))

        loss, _, _ = alpha_head_loss(logits, torch.tensor([[1.0, 0.0, 0.0, 1.0]] * 2), bce_weight=1.0, dice_weight=1.0)
        loss.backward()
        self.assertGreater(float(model.alpha_head.proj.weight.grad.abs().sum()), 0.0)
        # The zero projection keeps the backbone untouched until the head has learned.
        self.assertEqual(float(model.img_in.weight.grad.abs().sum()), 0.0)

        with self.assertRaisesRegex(ValueError, "return_dict=False"):
            model(hidden_states=torch.randn(2, 10, 8), return_alpha_logits=True, **{**dit_kwargs(2, img_shapes), "return_dict": True})
        del model.alpha_head
        with self.assertRaisesRegex(ValueError, "alpha_head"):
            model(hidden_states=torch.randn(2, 10, 8), return_alpha_logits=True, **dit_kwargs(2, img_shapes))

    def test_occupancy_targets_follow_the_packed_token_order(self):
        mask = torch.zeros(1, 32, 48)
        mask[:, 0:8, 0:16] = 1
        mask[:, 16:32, 32:48] = 1
        torch.testing.assert_close(alpha_occupancy_targets(mask, 16), torch.tensor([[0.5, 0, 0, 0, 0, 1.0]]))
        good = torch.tensor([[0.0, -8, -8, -8, -8, 8.0]])
        bad = -good
        targets = alpha_occupancy_targets(mask, 16)
        self.assertLess(
            float(alpha_head_loss(good, targets, bce_weight=1.0, dice_weight=1.0)[0]),
            float(alpha_head_loss(bad, targets, bce_weight=1.0, dice_weight=1.0)[0]),
        )

    @staticmethod
    def _companion_trainer(pretrained_weight):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = SimpleNamespace(
            model=SimpleNamespace(
                alpha_head=AlphaHeadConfig(enabled=True),
                lora=SimpleNamespace(pretrained_weight=pretrained_weight),
            ),
            train=SimpleNamespace(trainable_precision="float32"),
            resume=None,
        )
        trainer.accelerator = None
        trainer.dit = torch.nn.Module()
        trainer.dit.alpha_head = AlphaHead(8)
        return trainer

    def test_companion_is_saved_verified_by_state_json_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir)
            lora_path = checkpoint / "model.safetensors"
            safetensors.torch.save_file({"lora": torch.zeros(1)}, str(lora_path))
            saved = self._companion_trainer(None)
            with torch.no_grad():
                saved.dit.alpha_head.proj.weight.normal_()
                saved.dit.alpha_head.proj.bias.fill_(0.3)
            saved.save_additional_modules(str(checkpoint))
            manifest = BaseTrainer._build_checkpoint_artifact_manifest(str(checkpoint))
            (checkpoint / "state.json").write_text(json.dumps({"artifacts": manifest}))
            roles = {entry["filename"]: entry["role"] for entry in manifest["files"]}
            self.assertEqual(roles[ALPHA_HEAD_FILE_NAME], "alpha_head")

            config = SimpleNamespace(model=SimpleNamespace(alpha_head=AlphaHeadConfig(enabled=True)))
            result = validate_checkpoint_artifacts(config, lora_path, allow_mismatch=False)
            self.assertTrue(result["verified"])
            self.assertIn(ALPHA_HEAD_FILE_NAME, {item["filename"] for item in result["files"]})

            loaded = self._companion_trainer(str(lora_path))
            loaded.load_additional_modules(stage="predict")
            for key, value in saved.dit.alpha_head.state_dict().items():
                self.assertTrue(torch.equal(loaded.dit.alpha_head.state_dict()[key], value))

            safetensors.torch.save_file(
                {key: torch.zeros_like(value) for key, value in saved.dit.alpha_head.state_dict().items()},
                str(checkpoint / ALPHA_HEAD_FILE_NAME),
            )
            with self.assertRaisesRegex(ValueError, "alpha_head.safetensors: (size|sha256) mismatch"):
                validate_checkpoint_artifacts(config, lora_path, allow_mismatch=False)

    def test_missing_companion_fails_prediction_but_allows_warm_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_path = Path(tmpdir) / "model.safetensors"
            safetensors.torch.save_file({"lora": torch.zeros(1)}, str(lora_path))
            with self.assertRaises(FileNotFoundError):
                self._companion_trainer(str(lora_path)).load_additional_modules(stage="predict")
            warm = self._companion_trainer(str(lora_path))
            with self.assertLogs(level="WARNING"):
                warm.load_additional_modules(stage="fit")
            self.assertEqual(warm.dit.alpha_head.proj.weight.dtype, torch.float32)


class ComputeLossIntegrationTest(unittest.TestCase):
    def _trainer(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.weight_dtype = torch.float32
        trainer.vae_scale_factor = 8
        trainer.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=2))
        trainer.config = SimpleNamespace(
            loss=SimpleNamespace(
                region_perceptual=True,
                region_sigma_power=2.0,
                region_max_sigma=1.0,
                latent_mse_weight=1.0,
                fg_lpips_weight=0.5,
                bg_l1_weight=0.5,
                ref_dino_weight=0.2,
                ref_dino_mode="gt_hinge",
                ref_dino_hinge_margin=0.05,
                alpha_bce_weight=1.0,
                alpha_dice_weight=0.0,
            )
        )
        self.alpha_logits = torch.zeros(1, 6, requires_grad=True)
        trainer.dit = Mock(side_effect=lambda **kwargs: (torch.zeros_like(kwargs["hidden_states"]), self.alpha_logits))
        trainer.forward_loss = Mock(return_value=torch.tensor(1.0, requires_grad=True))
        trainer.decode_vae_latent_for_loss = Mock(return_value=torch.zeros(1, 3, 32, 48, requires_grad=True))
        trainer._loaded_model_decode_context = Mock(return_value=nullcontext())
        trainer._get_cached_scheduler_training_tensors = Mock(
            return_value=(torch.tensor([1000.0, 0.0]), torch.tensor([0.9, 0.1]))
        )
        trainer.region_perceptual_criterion = Mock(
            return_value={
                "total": torch.tensor(0.5, requires_grad=True),
                "fg_lpips": torch.tensor(0.0),
                "bg_l1": torch.tensor(0.0),
                "ref_dino": torch.tensor(0.0),
            }
        )
        return trainer

    def _embeddings(self, route_high: bool):
        mask = torch.zeros(1, 32, 48)
        mask[:, :16, :16] = 1
        return {
            "image_latents": torch.zeros(1, 6, 64),
            "control_latents": torch.zeros(1, 6, 64),
            "prompt_embeds": torch.zeros(1, 1, 4),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "img_shapes": [[(1, 2, 3), (1, 2, 3)]],
            "height": 32,
            "width": 48,
            "pixel_mask": mask,
            "target_image_for_loss": torch.zeros(1, 3, 32, 48),
            "ref_image_for_loss": torch.zeros(1, 3, 32, 48),
            "ref_mask_for_loss": torch.ones(1, 32, 48),
            "alpha_target_mask": mask,
            "_sigma_split_timestep_u": torch.tensor([0.0 if route_high else 0.75]),
            "_sigma_split_route_high": torch.tensor([route_high]),
        }

    def test_presampled_sigma_alpha_loss_and_identity_gate(self):
        for route_high, sigma in ((True, 0.9), (False, 0.1)):
            with self.subTest(route_high=route_high):
                trainer = self._trainer()
                loss = trainer._compute_loss(self._embeddings(route_high))
                bce = F.binary_cross_entropy_with_logits(torch.zeros(6), torch.tensor([1.0, 0, 0, 0, 0, 0]))
                self.assertAlmostEqual(loss.item(), 1.0 + 0.5 + bce.item(), places=5)
                self.assertTrue(trainer.dit.call_args.kwargs["return_alpha_logits"])
                kwargs = trainer.region_perceptual_criterion.call_args.kwargs
                weight = (1.0 - sigma) ** 2
                torch.testing.assert_close(kwargs["sample_weights"], torch.tensor([weight]))
                self.assertEqual(kwargs["ref_dino_mode"], "gt_hinge")
                self.assertEqual(kwargs["ref_dino_weight"], 0.0 if route_high else 0.2)
                torch.testing.assert_close(
                    kwargs["ref_dino_sample_weights"], torch.tensor([0.0 if route_high else weight])
                )
                metrics = trainer._step_loss_metrics
                self.assertEqual(float(metrics["train/sigma_split_high_rate"]), float(route_high))
                self.assertAlmostEqual(float(metrics["train/alpha_bce"]), float(bce), places=5)
                loss.backward()
                self.assertIsNotNone(self.alpha_logits.grad)


class GtHingeIdentityTest(unittest.TestCase):
    @staticmethod
    def _run(pred_color, gt_color, ref_color, *, mode="gt_hinge", weights=None):
        criterion = RegionPerceptualLoss(
            enable_ref_dino=True, dino_input_size=8, dino_embed_fn=lambda images: images.mean(dim=(2, 3))
        )

        def solid(color):
            return torch.tensor(color).view(1, 3, 1, 1).expand(1, 3, 16, 16).clone()

        pred = solid(pred_color).requires_grad_(True)
        target = solid(gt_color).requires_grad_(True)
        mask = torch.ones(1, 16, 16)
        out = criterion(
            pred,
            target,
            mask,
            fg_weight=0.0,
            bg_weight=0.0,
            ref_dino_weight=1.0,
            ref_images=solid(ref_color),
            ref_mask=mask,
            pred_range="0-1",
            target_range="0-1",
            ref_range="0-1",
            ref_dino_mode=mode,
            ref_dino_hinge_margin=0.05,
            ref_dino_sample_weights=weights,
        )
        return out["total"], pred, target

    def test_hinge_pulls_toward_the_reference_only_when_behind_the_gt(self):
        ref, gt = (0.9, 0.1, 0.1), (0.8, 0.2, 0.1)
        loss, pred, target = self._run((0.1, 0.1, 0.9), gt, ref)
        cos = lambda a, b: float(F.cosine_similarity(torch.tensor(a), torch.tensor(b), dim=0))
        self.assertAlmostEqual(loss.item(), cos(gt, ref) - 0.05 - cos((0.1, 0.1, 0.9), ref), places=4)
        loss.backward()
        self.assertIsNone(target.grad)  # cos(GT, ref) is a no-grad anchor
        stepped = (pred - 0.5 * pred.grad).detach().mean(dim=(2, 3))[0]
        self.assertGreater(cos(stepped.tolist(), ref), cos((0.1, 0.1, 0.9), ref))

    def test_matching_the_reference_better_than_gt_earns_nothing(self):
        loss, pred, _ = self._run((0.9, 0.1, 0.1), (0.5, 0.4, 0.3), (0.9, 0.1, 0.1))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(float(pred.grad.abs().sum()), 0.0)
        cosine_loss, _, _ = self._run((0.9, 0.1, 0.1), (0.5, 0.4, 0.3), (0.1, 0.1, 0.9), mode="cosine")
        self.assertGreater(float(cosine_loss), 0.0)

    def test_route_weights_switch_the_identity_term_off(self):
        loss, _, _ = self._run((0.1, 0.1, 0.9), (0.8, 0.2, 0.1), (0.9, 0.1, 0.1), weights=torch.tensor([0.0]))
        self.assertEqual(float(loss), 0.0)


class SigmaHandoffTest(unittest.TestCase):
    @staticmethod
    def _predict_batch(true_cfg_scale=2.5):
        batch = make_batch()
        for key in ("tar_mask_cond", "tar_mask_cond_tight", "placement_metadata"):
            del batch[key]
        user_mask = torch.zeros(1, 32, 48)
        user_mask[:, :, :32] = 1
        batch.update(
            tar_mask=user_mask,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=8,
            guidance=1.0,
            height=32,
            width=48,
            category=["predict"],
        )
        return batch

    def test_prediction_starts_in_the_layout_segment(self):
        trainer = make_trainer(handoff=True)
        batch = self._predict_batch()
        with patch.object(QwenImageEditPlusTrainer, "prepare_embeddings", autospec=True, side_effect=fake_parent_prepare):
            high = trainer.prepare_embeddings(batch, stage="predict")
        self.assertIs(high["_sigma_handoff"]["batch"], batch)
        self.assertEqual(high["true_cfg_scale"], 1.0)
        self.assertNotIn("reference_cfg_embeddings", high)
        self.assertEqual(high["vae_only_control_keys"], ["control_1"])
        self.assertTrue(torch.equal(high["control"], batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))))
        self.assertTrue(torch.equal(high["control_1"], batch["ref_image"]))  # no random jitter at inference
        latents = high["control_latents"][0, :, 0]
        self.assertTrue(torch.equal(latents[6:].sort().values, torch.arange(6.0, 18.0)))
        self.assertFalse(torch.equal(latents[6:], torch.arange(6.0, 18.0)))

    def test_identity_segment_guides_inside_the_alpha_region_with_the_ordered_reference(self):
        trainer = make_trainer(handoff=True)
        batch = self._predict_batch()
        core = torch.zeros(1, 32, 48)
        core[:, :16, :32] = 1
        free = torch.zeros(1, 32, 48)
        free[:, :20, :36] = 1
        low_batch = {**batch, "_sigma_segment": "low", "tar_mask": core, "source_hole_mask": free}
        with patch.object(QwenImageEditPlusTrainer, "prepare_embeddings", autospec=True, side_effect=fake_parent_prepare):
            low = trainer.prepare_embeddings(low_batch, stage="predict")
        self.assertNotIn("_sigma_handoff", low)
        self.assertTrue(low["reference_cfg_mask_aware"])
        self.assertTrue(torch.equal(low["reference_cfg_mask"], core))
        self.assertIn("reference_cfg_embeddings", low)
        self.assertNotIn("vae_only_control_keys", low)
        self.assertTrue(torch.equal(low["control_latents"][0, :, 0], torch.arange(18.0)))
        self.assertTrue(torch.equal(low["control"], batch["tar_image"] * (1 - free.unsqueeze(1))))

    def test_handoff_switches_condition_and_anchoring_at_tau(self):
        trainer = make_trainer(handoff=True, anchoring="expanded_soft")
        trainer.scheduler.sigmas = torch.tensor([1.0, 0.9, 0.8, 0.72, 0.68, 0.5, 0.3, 0.1, 0.0])
        timesteps = trainer.scheduler.sigmas[:-1] * 1000
        trainer.prepare_predict_timesteps = Mock(return_value=(timesteps, 8))
        noise = torch.randn(1, 6, 64)
        trainer.prepare_initial_latents = Mock(return_value=noise)
        trainer._dit_config = lambda: SimpleNamespace(in_channels=64)
        trainer._dit_device = lambda: torch.device("cpu")
        batch = self._predict_batch()
        batch.update(
            latent_anchoring_mode="expanded_soft",
            latent_anchoring_interaction_dilation_px=4,
            latent_anchoring_transition_width_px=4,
        )
        # Token grid 2x3: object in the top row (left two tokens) and bottom-left token;
        # the top-right token is outside the user's mask and must be dropped.
        alpha_logits = torch.tensor([[5.0, 5.0, 5.0, 5.0, -5.0, -5.0]])
        expected_core = torch.zeros(1, 32, 48)
        expected_core[:, :16, :32] = 1
        expected_core[:, 16:, :16] = 1

        calls = []
        after_high = torch.full((1, 6, 64), 0.5)
        reanchored = torch.full((1, 6, 64), 0.75)
        final = torch.full((1, 6, 64), 0.25)

        def fake_segment(self_, embeddings, **kwargs):
            calls.append((embeddings, kwargs))
            if kwargs.get("capture_alpha"):
                self_._last_alpha_logits = alpha_logits
                return after_high
            return final

        low_embeddings = {
            "anchor_source_latents": torch.zeros(1, 6, 64),
            "anchor_edit_weight": torch.ones(1, 6, 64),
        }
        trainer.prepare_embeddings = Mock(return_value=low_embeddings)
        trainer._apply_post_step_latent_anchoring = Mock(return_value=reanchored)
        high = {
            "height": 32,
            "width": 48,
            "num_inference_steps": 8,
            "control_latents": torch.zeros(1, 18, 64),
            "_sigma_handoff": {"batch": batch, "settings": trainer._sigma_handoff_settings()},
        }
        with patch.object(QwenImageEditTrainer, "sampling_from_embeddings", autospec=True, side_effect=fake_segment):
            result = trainer.sampling_from_embeddings(high)

        self.assertIs(result, final)
        (high_call, high_kwargs), (low_call, low_kwargs) = calls
        self.assertIs(high_call, high)
        self.assertEqual(high_kwargs["step_end"], 4)  # first step with sigma <= 0.7
        self.assertTrue(high_kwargs["capture_alpha"])
        self.assertIs(high_kwargs["initial_latents"], noise)
        self.assertIs(high_kwargs["anchor_noise"], noise)
        self.assertIs(low_call, low_embeddings)
        self.assertEqual(low_kwargs["step_start"], 4)
        self.assertIs(low_kwargs["initial_latents"], reanchored)
        self.assertIs(low_kwargs["anchor_noise"], noise)

        low_batch = trainer.prepare_embeddings.call_args.args[0]
        self.assertEqual(low_batch["_sigma_segment"], "low")
        self.assertTrue(torch.equal(low_batch["tar_mask"], expected_core))
        free = low_batch["anchor_mask_free"]
        self.assertTrue(torch.equal(low_batch["source_hole_mask"], free))
        self.assertTrue(bool(torch.all(free >= expected_core)))
        self.assertGreater(float(free.sum()), float(expected_core.sum()))
        self.assertLess(float(free.sum()), float(batch["tar_mask"].sum()) + 4 * 48)
        anchoring = trainer._apply_post_step_latent_anchoring.call_args.kwargs
        self.assertEqual(anchoring["step_index"], 3)
        self.assertIs(anchoring["generated_next"], after_high)
        handoff = trainer.get_last_sigma_handoff()
        self.assertEqual(handoff["step"], 4)
        self.assertAlmostEqual(handoff["sigma"], 0.68, places=5)

    def test_helpers(self):
        self.assertEqual(handoff_step_index(torch.tensor([1.0, 0.8, 0.7, 0.3]), 0.7, 4), 2)
        with self.assertRaises(ValueError):
            handoff_step_index(torch.tensor([0.6, 0.5]), 0.7, 2)
        with self.assertRaises(ValueError):
            handoff_step_index(torch.tensor([1.0, 0.9]), 0.7, 2)
        region, probabilities = alpha_region_from_logits(torch.tensor([[3.0, -3.0]]), (1, 2), (16, 32), 0.5)
        self.assertEqual(tuple(probabilities.shape), (1, 1, 2))
        self.assertTrue(torch.equal(region[0, :, :16], torch.ones(16, 16)))
        self.assertTrue(torch.equal(region[0, :, 16:], torch.zeros(16, 16)))


if __name__ == "__main__":
    unittest.main()

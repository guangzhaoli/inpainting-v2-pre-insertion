import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from src.data.config import Config, LatentAnchoringConfig, LatentAnchoringMode, PredictConfig
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    QwenImageEditInsertAnythingTrainer,
)
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.latent_anchoring import (
    apply_latent_anchoring_cli_overrides,
    build_latent_anchoring_masks,
    latent_anchoring_metadata,
    normalize_latent_anchoring_mode,
    validate_latent_anchoring_runtime,
)


class DummyPredictPreprocessor:
    def preprocess(self, _data):
        source = torch.ones(3, 24, 24)
        core = torch.zeros(24, 24)
        core[10:14, 10:14] = 1
        reference = torch.ones(3, 12, 12)
        reference_mask = torch.ones(12, 12)
        return {
            "image": source,
            "mask": core,
            "controls": [reference],
            "control_masks": [reference_mask],
        }


class FakeScaleNoiseScheduler:
    def __init__(self):
        self.calls = []

    def scale_noise(self, source, timestep, noise):
        self.calls.append(
            {
                "timestep": timestep.detach().clone(),
                "noise_data_ptr": noise.data_ptr(),
            }
        )
        return source * 0.25 + noise * 0.75


class LatentAnchoringConfigTest(unittest.TestCase):
    def test_default_and_ab_aliases(self):
        config = Config()
        self.assertEqual(config.predict.latent_anchoring.mode, LatentAnchoringMode.legacy)
        self.assertEqual(normalize_latent_anchoring_mode("A"), "legacy")
        self.assertEqual(normalize_latent_anchoring_mode("B"), "core")
        self.assertEqual(normalize_latent_anchoring_mode("C"), "expanded")
        self.assertEqual(normalize_latent_anchoring_mode("D"), "expanded_soft")
        self.assertEqual(LatentAnchoringConfig(mode="D").mode, LatentAnchoringMode.expanded_soft)

    def test_mode_specific_geometry_validation(self):
        with self.assertRaisesRegex(ValueError, "interaction_dilation_px > 0"):
            LatentAnchoringConfig(mode="expanded", interaction_dilation_px=0)
        for width in (0, 1):
            with self.subTest(width=width), self.assertRaisesRegex(
                ValueError,
                "transition_width_px >= 2",
            ):
                LatentAnchoringConfig(mode="expanded_soft", transition_width_px=width)
        with self.assertRaisesRegex(ValueError, "mask_threshold"):
            LatentAnchoringConfig(mask_threshold=float("nan"))

    def test_metadata_maps_modes_to_ab_variants(self):
        metadata = latent_anchoring_metadata(
            LatentAnchoringConfig(
                mode="expanded_soft",
                interaction_dilation_px=24,
                transition_width_px=16,
            )
        )
        self.assertEqual(metadata["variant"], "D")
        self.assertTrue(metadata["enabled"])
        self.assertEqual(metadata["source_hole_mask"], "M_free")
        self.assertEqual(metadata["placement_and_reference_mask"], "M_core")

    def test_cli_none_preserves_yaml_and_explicit_values_override_it(self):
        config = Config()
        config.predict.latent_anchoring = LatentAnchoringConfig(
            mode="expanded_soft",
            mask_threshold=0.4,
            interaction_dilation_px=48,
            transition_width_px=24,
        )
        args = SimpleNamespace(
            latent_anchoring_mode=None,
            latent_anchoring_mask_threshold=None,
            interaction_dilation_px=None,
            transition_width_px=None,
        )

        preserved = apply_latent_anchoring_cli_overrides(config, args)
        self.assertEqual(preserved.mode, LatentAnchoringMode.expanded_soft)
        self.assertEqual(preserved.mask_threshold, 0.4)
        self.assertEqual(preserved.interaction_dilation_px, 48)
        self.assertEqual(preserved.transition_width_px, 24)

        args.latent_anchoring_mode = "expanded"
        args.interaction_dilation_px = 16
        overridden = apply_latent_anchoring_cli_overrides(config, args)
        self.assertEqual(overridden.mode, LatentAnchoringMode.expanded)
        self.assertEqual(overridden.interaction_dilation_px, 16)
        self.assertEqual(overridden.transition_width_px, 24)
        self.assertEqual(overridden.paste_back, "crop_feather")

        args.paste_back = "change_gated"
        gated = apply_latent_anchoring_cli_overrides(config, args)
        self.assertEqual(gated.paste_back, "change_gated")
        self.assertEqual(latent_anchoring_metadata(gated)["paste_back"], "change_gated")
        with self.assertRaisesRegex(ValueError, "transition_width_px >= 2"):
            LatentAnchoringConfig(mode="expanded", transition_width_px=1, paste_back="change_gated")

    def test_runtime_allows_adaptive_crop_when_thresholds_match_and_rejects_explicit_size(self):
        config = Config()
        config.predict.latent_anchoring = LatentAnchoringConfig(mode="core", mask_threshold=0.5)
        config.data.init_args.adaptive_crop.predict_enabled = True
        config.data.init_args.adaptive_crop.mask_threshold = 0.5
        validate_latent_anchoring_runtime(config)

        config.data.init_args.adaptive_crop.mask_threshold = 0.4
        with self.assertRaisesRegex(ValueError, "mask_threshold"):
            validate_latent_anchoring_runtime(config)

        config.data.init_args.adaptive_crop.predict_enabled = False
        with self.assertRaisesRegex(ValueError, "explicit --height/--width"):
            validate_latent_anchoring_runtime(config, height=768, width=768)


class LatentAnchoringMaskTest(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(1, 33, 33)
        self.mask[:, 15:18, 15:18] = 1

    def test_b_uses_exact_binary_core(self):
        masks = build_latent_anchoring_masks(self.mask, mode="core")
        torch.testing.assert_close(masks.core, masks.free, rtol=0, atol=0)
        torch.testing.assert_close(masks.core, masks.edit_weight, rtol=0, atol=0)
        self.assertEqual(set(torch.unique(masks.edit_weight).tolist()), {0.0, 1.0})

    def test_c_expands_binary_free_region(self):
        masks = build_latent_anchoring_masks(
            self.mask,
            mode="expanded",
            interaction_dilation_px=4,
        )
        self.assertGreater(int(masks.free.sum()), int(masks.core.sum()))
        torch.testing.assert_close(masks.free, masks.edit_weight, rtol=0, atol=0)
        self.assertTrue(torch.all(masks.free >= masks.core).item())

    def test_d_keeps_free_region_hard_and_softens_only_outer_band(self):
        masks = build_latent_anchoring_masks(
            self.mask,
            mode="expanded_soft",
            interaction_dilation_px=4,
            transition_width_px=6,
        )
        self.assertTrue(torch.all(masks.edit_weight[masks.free.bool()] == 1).item())
        transition = (masks.edit_weight > 0) & (masks.edit_weight < 1)
        self.assertTrue(bool(transition.any().item()))
        self.assertEqual(float(masks.edit_weight[0, 0, 0]), 0.0)

    def test_d_builder_rejects_transition_too_narrow_to_be_soft(self):
        with self.assertRaisesRegex(ValueError, "transition_width_px >= 2"):
            build_latent_anchoring_masks(
                self.mask,
                mode="expanded_soft",
                interaction_dilation_px=4,
                transition_width_px=1,
            )

    def test_edge_dilation_is_clipped_safely_and_empty_core_is_rejected(self):
        edge = torch.zeros(1, 16, 16)
        edge[:, :2, :2] = 1
        masks = build_latent_anchoring_masks(
            edge,
            mode="expanded",
            interaction_dilation_px=5,
        )
        self.assertEqual(tuple(masks.free.shape), (1, 16, 16))
        self.assertTrue(torch.all(masks.free >= masks.core).item())
        with self.assertRaisesRegex(ValueError, "non-empty M_core"):
            build_latent_anchoring_masks(torch.zeros(1, 16, 16), mode="core")


class LatentAnchoringPackingAndSchedulerTest(unittest.TestCase):
    def test_packed_weight_matches_per_feature_latent_layout(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.vae_scale_factor = 2

        edit_weight = torch.zeros(1, 8, 8)
        edit_weight[:, :, 2:] = 1
        free_mask = edit_weight.clone()
        packed = trainer._prepare_latent_anchoring_weight(
            edit_weight=edit_weight,
            free_mask=free_mask,
            batch_size=1,
            num_channels_latents=1,
            height_image=8,
            width_image=8,
            soft_transition=False,
        )

        self.assertEqual(tuple(packed.shape), (1, 4, 4))
        self.assertEqual(set(torch.unique(packed).tolist()), {0.0, 1.0})
        self.assertTrue(any(torch.unique(token).numel() > 1 for token in packed[0]))

    def test_hard_packing_preserves_single_pixel_thin_core(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.vae_scale_factor = 8
        edit_weight = torch.zeros(1, 32, 32)
        edit_weight[:, 3, 3] = 1

        packed = trainer._prepare_latent_anchoring_weight(
            edit_weight=edit_weight,
            free_mask=edit_weight,
            batch_size=1,
            num_channels_latents=1,
            height_image=32,
            width_image=32,
            soft_transition=False,
        )

        self.assertGreater(float(packed.sum()), 0.0)

    def test_soft_outer_transition_survives_latent_packing(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.vae_scale_factor = 8
        core = torch.zeros(1, 64, 64)
        core[:, 28:36, 28:36] = 1
        masks = build_latent_anchoring_masks(
            core,
            mode="expanded_soft",
            interaction_dilation_px=8,
            transition_width_px=16,
        )

        packed = trainer._prepare_latent_anchoring_weight(
            edit_weight=masks.edit_weight,
            free_mask=masks.free,
            batch_size=1,
            num_channels_latents=1,
            height_image=64,
            width_image=64,
            soft_transition=True,
        )

        self.assertTrue(bool((packed == 1).any().item()))
        self.assertTrue(bool(((packed > 0) & (packed < 1)).any().item()))

    def test_post_step_uses_next_timestep_and_clean_source_at_terminal(self):
        trainer = object.__new__(QwenImageEditTrainer)
        trainer.scheduler = FakeScaleNoiseScheduler()
        source = torch.full((1, 2, 2), 2.0)
        noise = torch.full((1, 2, 2), 10.0)
        generated = torch.full((1, 2, 2), 6.0)
        weight = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]])
        timesteps = torch.tensor([1000.0, 700.0])

        first = trainer._apply_post_step_latent_anchoring(
            generated_next=generated,
            anchor_source_latents=source,
            anchor_noise=noise,
            edit_weight=weight,
            timesteps=timesteps,
            step_index=0,
        )
        source_next = source * 0.25 + noise * 0.75
        torch.testing.assert_close(first, torch.lerp(source_next, generated, weight))
        self.assertEqual(trainer.scheduler.calls[0]["timestep"].tolist(), [700.0])
        self.assertEqual(trainer.scheduler.calls[0]["noise_data_ptr"], noise.data_ptr())

        terminal = trainer._apply_post_step_latent_anchoring(
            generated_next=generated,
            anchor_source_latents=source,
            anchor_noise=noise,
            edit_weight=weight,
            timesteps=timesteps,
            step_index=1,
        )
        torch.testing.assert_close(terminal, torch.lerp(source, generated, weight))
        self.assertEqual(len(trainer.scheduler.calls), 1)

    def test_real_flowmatch_scheduler_uses_internal_next_sigma_after_step(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=1.15,
            base_image_seq_len=256,
            max_image_seq_len=4096,
        )
        scheduler.set_timesteps(
            sigmas=np.linspace(1.0, 0.25, 4),
            mu=1.15,
            device="cpu",
        )
        scheduler.set_begin_index(0)

        trainer = object.__new__(QwenImageEditTrainer)
        trainer.scheduler = scheduler
        source = torch.full((1, 2, 4), 2.0)
        noise = torch.full_like(source, 10.0)
        current = noise.clone()
        generated_next = scheduler.step(
            torch.zeros_like(current),
            scheduler.timesteps[0],
            current,
            return_dict=False,
        )[0]
        weight = torch.zeros_like(source)

        anchored = trainer._apply_post_step_latent_anchoring(
            generated_next=generated_next,
            anchor_source_latents=source,
            anchor_noise=noise,
            edit_weight=weight,
            timesteps=scheduler.timesteps,
            step_index=0,
        )
        sigma_next = scheduler.sigmas[1].to(dtype=source.dtype)
        expected = (1.0 - sigma_next) * source + sigma_next * noise

        self.assertEqual(scheduler.step_index, 1)
        torch.testing.assert_close(anchored, expected)


class InsertAnythingLatentAnchoringRoutingTest(unittest.TestCase):
    def test_standalone_predict_rejects_explicit_size_but_allows_adaptive_crop_flag(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = Config()
        trainer.config.predict.latent_anchoring = LatentAnchoringConfig(mode="core")
        trainer.config.data.init_args.adaptive_crop.predict_enabled = True

        with patch.object(
            QwenImageEditInsertAnythingTrainer.__mro__[1],
            "predict",
            return_value=["ok"],
        ) as parent_predict:
            result = trainer.predict(latent_anchoring_mode=None)
        self.assertEqual(result, ["ok"])
        parent_predict.assert_called_once()

        with patch.object(trainer, "setup_predict") as setup_predict:
            with self.assertRaisesRegex(ValueError, "explicit height/width"):
                trainer.predict(
                    latent_anchoring_mode="expanded",
                    height=768,
                    width=768,
                )
        setup_predict.assert_not_called()

    def test_predict_builds_independent_core_free_and_anchor_inputs(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = Config()
        trainer.preprocessor = DummyPredictPreprocessor()

        source = Image.new("RGB", (24, 24), color="white")
        source_mask = Image.new("L", (24, 24), color=255)
        reference = Image.new("RGB", (12, 12), color="blue")
        reference_mask = Image.new("L", (12, 12), color=255)
        data = trainer.prepare_predict_batch_data(
            image=[source, reference],
            src_mask=source_mask,
            ref_mask=reference_mask,
            prompt="insert",
            latent_anchoring_mode="expanded",
            interaction_dilation_px=3,
        )

        self.assertEqual(data["latent_anchoring_mode"], "expanded")
        self.assertGreater(int(data["source_hole_mask"].sum()), int(data["tar_mask"].sum()))
        torch.testing.assert_close(data["anchor_source_image"], data["tar_image"])
        torch.testing.assert_close(data["anchor_mask_core"], data["tar_mask"])

        legacy = trainer.prepare_predict_batch_data(
            image=[source, reference],
            src_mask=source_mask,
            ref_mask=reference_mask,
            prompt="insert",
            latent_anchoring_mode="legacy",
        )
        self.assertNotIn("anchor_source_image", legacy)
        self.assertNotIn("source_hole_mask", legacy)

    def test_expanded_source_hole_does_not_expand_reference_cfg_core_route(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = Config()
        trainer.config.model.reference_cfg.enabled = True
        trainer.config.model.reference_cfg.force_empty_prompt = True
        trainer.config.model.reference_cfg.mask_aware = True

        core = torch.zeros(1, 8, 8)
        core[:, 3:5, 3:5] = 1
        free = torch.zeros(1, 8, 8)
        free[:, 1:7, 1:7] = 1
        batch = {
            "ref_image": torch.ones(1, 3, 8, 8),
            "ref_mask": torch.ones(1, 8, 8),
            "tar_image": torch.ones(1, 3, 8, 8),
            "tar_mask": core,
            "source_hole_mask": free,
            "prompt": [""],
            "category": ["predict"],
            "height": 8,
            "width": 8,
            "num_inference_steps": 2,
            "guidance": 1.0,
            "true_cfg_scale": 4.0,
        }

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=lambda _self, parent_batch, stage="fit": parent_batch,
        ):
            embeddings = trainer.prepare_embeddings(batch, stage="predict")

        control = embeddings["control"]
        self.assertTrue(torch.all(control[:, :, free.bool()[0]] == 0).item())
        torch.testing.assert_close(embeddings["reference_cfg_mask"], core)

    def test_anchor_source_is_encoded_independently_only_when_enabled(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = Config()
        trainer.num_channels_latents = 1
        trainer.vae_scale_factor = 2
        trainer.weight_dtype = torch.float32
        trainer.preprocess_image_for_vae_encoder = (
            lambda image, **_kwargs: image.unsqueeze(2)
        )

        core = torch.zeros(1, 8, 8)
        core[:, 3:5, 3:5] = 1
        anchor_source = torch.full((1, 3, 8, 8), 0.75)
        batch = {
            "ref_image": torch.ones(1, 3, 8, 8),
            "ref_mask": torch.ones(1, 8, 8),
            "tar_image": torch.ones(1, 3, 8, 8),
            "tar_mask": core,
            "prompt": ["insert"],
            "category": ["predict"],
            "height": 8,
            "width": 8,
            "num_inference_steps": 2,
            "guidance": 1.0,
            "true_cfg_scale": 1.0,
            "anchor_source_image": anchor_source,
            "anchor_mask_free": core,
            "anchor_edit_weight": core,
            "latent_anchoring_mode": "core",
        }
        encoded_anchor = torch.full((1, 4, 4), 7.0)

        with (
            patch.object(
                QwenImageEditPlusTrainer,
                "prepare_embeddings",
                autospec=True,
                side_effect=lambda _self, parent_batch, stage="fit": parent_batch,
            ),
            patch.object(trainer, "encode_image_latents", return_value=encoded_anchor) as encode_mock,
        ):
            embeddings = trainer.prepare_embeddings(batch, stage="predict")

        encode_mock.assert_called_once()
        torch.testing.assert_close(encode_mock.call_args.args[0], anchor_source.unsqueeze(2))
        torch.testing.assert_close(embeddings["anchor_source_latents"], encoded_anchor)
        self.assertEqual(tuple(embeddings["anchor_edit_weight"].shape), (1, 4, 4))

        legacy_batch = {key: value for key, value in batch.items() if not key.startswith("anchor_")}
        legacy_batch["latent_anchoring_mode"] = "legacy"
        with (
            patch.object(
                QwenImageEditPlusTrainer,
                "prepare_embeddings",
                autospec=True,
                side_effect=lambda _self, parent_batch, stage="fit": parent_batch,
            ),
            patch.object(trainer, "encode_image_latents") as legacy_encode_mock,
        ):
            legacy_embeddings = trainer.prepare_embeddings(legacy_batch, stage="predict")
        legacy_encode_mock.assert_not_called()
        self.assertNotIn("anchor_source_latents", legacy_embeddings)


class SourceNoiseStartTest(unittest.TestCase):
    @staticmethod
    def _trainer(start):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        settings = {"mode": "expanded_soft", "transition_width_px": 4, "source_noise_start": start}
        trainer.config = SimpleNamespace(predict=PredictConfig(latent_anchoring=settings))
        trainer.vae_scale_factor = 8
        trainer.weight_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(sigmas=torch.tensor([1.0, 0.95, 0.88, 0.8, 0.6, 0.3, 0.1, 0.0]))
        trainer.prepare_predict_timesteps = lambda steps, seq_len: (trainer.scheduler.sigmas[:-1] * 1000, steps)
        trainer._dit_config = lambda: SimpleNamespace(in_channels=64)
        trainer._dit_device = lambda: torch.device("cpu")
        return trainer

    @staticmethod
    def _embeddings():
        return {
            "height": 32,
            "width": 48,
            "num_inference_steps": 7,
            "control_latents": torch.zeros(1, 18, 64),
            "anchor_source_latents": torch.full((1, 6, 64), 2.0),
        }

    def test_config_cli_and_metadata(self):
        with self.assertRaisesRegex(ValueError, r"in \(0, 1\)"):
            LatentAnchoringConfig(mode="expanded_soft", source_noise_start=1.0)
        with self.assertRaisesRegex(ValueError, "anchored source latent"):
            LatentAnchoringConfig(source_noise_start=0.9)
        config = Config()
        config.predict.latent_anchoring = LatentAnchoringConfig(mode="expanded_soft")
        args = SimpleNamespace(latent_anchoring_mode=None, latent_anchoring_mask_threshold=None,
                               interaction_dilation_px=None, transition_width_px=None, source_noise_start=None)
        self.assertIsNone(apply_latent_anchoring_cli_overrides(config, args).source_noise_start)
        args.source_noise_start = 0.9
        settings = apply_latent_anchoring_cli_overrides(config, args)
        self.assertEqual(settings.source_noise_start, 0.9)
        self.assertEqual(latent_anchoring_metadata(settings)["source_noise_start"], 0.9)

    def test_starts_from_noised_source_at_nearest_step(self):
        trainer = self._trainer(0.9)
        noise = torch.randn(1, 6, 64)
        trainer.prepare_initial_latents = lambda **_kwargs: noise
        calls = []

        def fake_segment(_self, embeddings, **kwargs):
            calls.append(kwargs)
            return embeddings["anchor_source_latents"]

        with patch.object(QwenImageEditTrainer, "sampling_from_embeddings", autospec=True, side_effect=fake_segment):
            trainer.sampling_from_embeddings(self._embeddings())
            # Segment calls (σ-split hand-off internals) keep their own start.
            trainer.sampling_from_embeddings(self._embeddings(), step_start=3)

        started, segment = calls
        self.assertEqual(started["step_start"], 2)  # σ=0.88 is nearest to 0.9
        self.assertIs(started["anchor_noise"], noise)
        torch.testing.assert_close(started["initial_latents"], 0.88 * noise + 0.12 * 2.0)
        self.assertEqual(segment, {"step_start": 3})
        last = trainer.get_last_source_noise_start()
        self.assertEqual((last["requested"], last["step"]), (0.9, 2))
        self.assertAlmostEqual(last["sigma"], 0.88, places=5)

    def test_requires_anchoring_and_excludes_sigma_handoff(self):
        trainer = self._trainer(0.9)
        trainer.prepare_initial_latents = lambda **_kwargs: torch.zeros(1, 6, 64)
        embeddings = self._embeddings()
        del embeddings["anchor_source_latents"]
        with self.assertRaisesRegex(ValueError, "needs latent anchoring"):
            trainer.sampling_from_embeddings(embeddings)
        with self.assertRaisesRegex(ValueError, "σ-split hand-off"):
            trainer.sampling_from_embeddings({**self._embeddings(), "_sigma_handoff": {}})


if __name__ == "__main__":
    unittest.main()

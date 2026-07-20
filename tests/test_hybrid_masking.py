import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from pydantic import ValidationError

from src.data.config import (
    DataConfig,
    HybridMaskingConfig,
    ReferenceCFGConfig,
    load_config_from_yaml,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.utils.hybrid_masking import (
    INSERT_ANYTHING_AUGMENTED_BOX,
    INSERT_ANYTHING_DILATED_SEGMENTATION,
    INSERT_ANYTHING_EXPANDED_BOX,
    apply_hybrid_masking_context_guard,
    build_insert_anything_hybrid_masks,
    build_insert_anything_mask_for_mode,
    sample_insert_anything_mask_mode,
)


class FixedProbabilityRng:
    def __init__(self, probability: float):
        self.probability = probability

    def random(self):
        return self.probability

    @staticmethod
    def randint(low, high):
        if low <= 0 <= high:
            return 0
        return low


def make_mask(size=32):
    mask = np.zeros((size, size), dtype=np.float32)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = 1.0
    return mask


def make_batch():
    tar_mask = torch.zeros(1, 12, 12)
    tar_mask[:, 4:8, 4:8] = 1.0
    tar_mask_cond = torch.zeros_like(tar_mask)
    tar_mask_cond[:, 2:10, 2:10] = 1.0

    ref_image = torch.zeros(1, 3, 12, 12)
    ref_image[:, :, 3:9, 4:8] = 0.75
    ref_mask = torch.zeros(1, 12, 12)
    ref_mask[:, 3:9, 4:8] = 1.0
    return {
        "ref_image": ref_image,
        "ref_mask": ref_mask,
        "tar_image": torch.ones(1, 3, 12, 12),
        "tar_mask": tar_mask,
        "tar_mask_cond": tar_mask_cond,
        "prompt": ["insert object"],
        "category": ["object"],
        "true_cfg_scale": 4.0,
    }


def make_trainer(
    *,
    hybrid_enabled=False,
    context_guard_enabled=False,
    max_condition_area_ratio=0.80,
    reference_cfg=False,
    mask_aware=False,
    mask_loss=True,
):
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(
            init_args=SimpleNamespace(
                hybrid_masking=HybridMaskingConfig(
                    enabled=hybrid_enabled,
                    scale_pixel_parameters=False,
                    curve_jitter_px=0,
                    context_guard={
                        "enabled": context_guard_enabled,
                        "max_condition_area_ratio": max_condition_area_ratio,
                        "overflow_policy": "fallback_to_gt",
                    },
                )
            )
        ),
        model=SimpleNamespace(
            reference_cfg=ReferenceCFGConfig(
                enabled=reference_cfg,
                train_dropout_rate=0.0,
                force_empty_prompt=reference_cfg,
                mask_aware=mask_aware,
            ),
        ),
        loss=SimpleNamespace(mask_loss=mask_loss),
    )
    return trainer


def passthrough_prepare_embeddings(_self, batch, stage="fit"):
    return dict(batch)


class HybridMaskingConfigTest(unittest.TestCase):
    def test_defaults_match_official_parameters_and_are_disabled(self):
        config = HybridMaskingConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.strategy, "insert_anything")
        self.assertEqual(config.person_categories, ["person"])
        self.assertEqual(config.bbox_expand_ratio, 1.1)
        self.assertEqual(config.curve_jitter_px, 40)
        self.assertEqual(config.dilation_kernel_size, 7)
        self.assertEqual(config.dilation_iterations, 2)
        self.assertTrue(config.ensure_gt_coverage)
        self.assertFalse(config.context_guard.enabled)
        self.assertEqual(config.context_guard.max_condition_area_ratio, 0.80)
        self.assertEqual(config.context_guard.overflow_policy, "fallback_to_gt")

    def test_invalid_parameters_are_rejected(self):
        invalid_kwargs = (
            {"strategy": "unknown"},
            {"mask_threshold": 1.0},
            {"bbox_expand_ratio": 0.9},
            {"curve_jitter_px": -1},
            {"dilation_kernel_size": 4},
            {"dilation_iterations": 0},
            {"reference_resolution": 0},
            {"person_categories": [""]},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                HybridMaskingConfig(**kwargs)

    def test_invalid_context_guard_parameters_are_rejected(self):
        invalid_context_guards = (
            {"max_condition_area_ratio": 0},
            {"max_condition_area_ratio": 1},
            {"max_condition_area_ratio": float("nan")},
            {"max_condition_area_ratio": float("inf")},
            {"overflow_policy": "clip"},
            {"unexpected": True},
        )
        for context_guard in invalid_context_guards:
            with self.subTest(context_guard=context_guard), self.assertRaises(ValidationError):
                HybridMaskingConfig(context_guard=context_guard)

    def test_context_guard_rejects_unsafe_padded_multi_sample_batches(self):
        with self.assertRaisesRegex(ValidationError, "valid-pixel denominator"):
            DataConfig(
                batch_size=2,
                init_args={
                    "hybrid_masking": {
                        "context_guard": {"enabled": True},
                    }
                },
            )

    def test_context_guard_requires_gt_coverage(self):
        with self.assertRaisesRegex(ValidationError, "ensure_gt_coverage=true"):
            HybridMaskingConfig(
                ensure_gt_coverage=False,
                context_guard={"enabled": True},
            )

    def test_target_b_config_explicitly_enables_context_guard(self):
        config_path = Path(__file__).resolve().parents[1] / "configs/exp/accessory/hybrid_masking_r256_fg3.yaml"
        with patch("torch.cuda.is_available", return_value=True):
            config = load_config_from_yaml(str(config_path))
        guard = config.data.init_args.hybrid_masking.context_guard
        self.assertTrue(guard.enabled)
        self.assertEqual(guard.max_condition_area_ratio, 0.80)
        self.assertEqual(guard.overflow_policy, "fallback_to_gt")
        self.assertEqual(config.data.batch_size, 1)


class HybridMaskingAlgorithmTest(unittest.TestCase):
    def setUp(self):
        self.config = HybridMaskingConfig(
            enabled=True,
            curve_jitter_px=0,
            scale_pixel_parameters=False,
        )
        self.mask = make_mask()

    def test_category_probabilities_match_official_code(self):
        person_expected = (
            (0.2, INSERT_ANYTHING_AUGMENTED_BOX),
            (0.6, INSERT_ANYTHING_EXPANDED_BOX),
            (0.9, INSERT_ANYTHING_DILATED_SEGMENTATION),
        )
        nonperson_expected = (
            (0.2, INSERT_ANYTHING_AUGMENTED_BOX),
            (0.8, INSERT_ANYTHING_EXPANDED_BOX),
        )
        for probability, expected in person_expected:
            self.assertEqual(
                sample_insert_anything_mask_mode("person", self.config, FixedProbabilityRng(probability)),
                expected,
            )
        for probability, expected in nonperson_expected:
            self.assertEqual(
                sample_insert_anything_mask_mode("accessory", self.config, FixedProbabilityRng(probability)),
                expected,
            )

    def test_all_modes_are_binary_and_cover_gt(self):
        for mode in (
            INSERT_ANYTHING_AUGMENTED_BOX,
            INSERT_ANYTHING_EXPANDED_BOX,
            INSERT_ANYTHING_DILATED_SEGMENTATION,
        ):
            with self.subTest(mode=mode):
                output = build_insert_anything_mask_for_mode(
                    self.mask,
                    mode,
                    self.config,
                    rng=random.Random(7),
                )
                self.assertEqual(output.shape, self.mask.shape)
                self.assertTrue(set(np.unique(output)).issubset({0, 1}))
                self.assertTrue(np.all(output[self.mask > 0] == 1))
                self.assertGreaterEqual(int(output.sum()), int(self.mask.sum()))

    def test_dilation_and_expanded_box_grow_the_mask(self):
        for mode in (INSERT_ANYTHING_DILATED_SEGMENTATION, INSERT_ANYTHING_EXPANDED_BOX):
            output = build_insert_anything_mask_for_mode(self.mask, mode, self.config)
            self.assertGreater(int(output.sum()), int(self.mask.sum()))

    def test_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_insert_anything_mask_for_mode(
                np.zeros((16, 16), dtype=np.float32),
                INSERT_ANYTHING_EXPANDED_BOX,
                self.config,
            )

    def test_batch_builder_preserves_shape_dtype_and_is_reproducible(self):
        masks = torch.from_numpy(np.stack([self.mask, self.mask])).to(torch.bfloat16)
        first, first_modes = build_insert_anything_hybrid_masks(
            masks,
            ["person", "object"],
            self.config,
            rng=random.Random(123),
        )
        second, second_modes = build_insert_anything_hybrid_masks(
            masks,
            ["person", "object"],
            self.config,
            rng=random.Random(123),
        )
        self.assertEqual(first.shape, masks.shape)
        self.assertEqual(first.dtype, masks.dtype)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_modes, second_modes)


class HybridMaskingContextGuardTest(unittest.TestCase):
    @staticmethod
    def _config(*, enabled=True, max_ratio=0.80):
        return HybridMaskingConfig(
            enabled=True,
            context_guard={
                "enabled": enabled,
                "max_condition_area_ratio": max_ratio,
                "overflow_policy": "fallback_to_gt",
            },
        )

    @staticmethod
    def _mask_with_area(area: int) -> torch.Tensor:
        mask = torch.zeros(1, 10, 10, dtype=torch.bfloat16)
        mask.reshape(-1)[:area] = 1
        return mask

    def test_disabled_guard_preserves_full_frame_candidate(self):
        gt_mask = self._mask_with_area(4)
        condition_mask = torch.ones_like(gt_mask)

        guarded, metadata = apply_hybrid_masking_context_guard(
            condition_mask,
            gt_mask,
            self._config(enabled=False),
        )

        self.assertTrue(torch.equal(guarded, condition_mask))
        self.assertFalse(bool(metadata["context_guard_triggered"][0]))
        self.assertEqual(float(metadata["condition_area_ratio"][0]), 1.0)

    def test_strict_threshold_keeps_exact_eighty_percent_and_falls_back_above_it(self):
        gt_mask = self._mask_with_area(4)
        at_limit = self._mask_with_area(80)
        above_limit = self._mask_with_area(81)
        config = self._config()

        guarded_at_limit, metadata_at_limit = apply_hybrid_masking_context_guard(
            at_limit,
            gt_mask,
            config,
        )
        guarded_above_limit, metadata_above_limit = apply_hybrid_masking_context_guard(
            above_limit,
            gt_mask,
            config,
        )

        self.assertTrue(torch.equal(guarded_at_limit, at_limit))
        self.assertFalse(bool(metadata_at_limit["context_guard_triggered"][0]))
        self.assertEqual(float(metadata_at_limit["condition_area_ratio"][0]), 0.80)
        self.assertTrue(torch.equal(guarded_above_limit, gt_mask))
        self.assertTrue(bool(metadata_above_limit["context_guard_triggered"][0]))
        self.assertFalse(bool(metadata_above_limit["unavoidable_large_gt"][0]))

    def test_full_frame_candidate_falls_back_to_smallest_leakage_safe_gt(self):
        gt_mask = self._mask_with_area(4)
        condition_mask = torch.ones_like(gt_mask)

        guarded, metadata = apply_hybrid_masking_context_guard(
            condition_mask,
            gt_mask,
            self._config(),
        )

        self.assertEqual(guarded.dtype, condition_mask.dtype)
        self.assertEqual(guarded.device, condition_mask.device)
        self.assertEqual(guarded.shape, condition_mask.shape)
        self.assertTrue(torch.equal(guarded, gt_mask))
        self.assertTrue(torch.all(guarded >= gt_mask).item())
        self.assertEqual(set(torch.unique(guarded).tolist()), {0.0, 1.0})
        self.assertTrue(bool(metadata["context_guard_triggered"][0]))
        self.assertTrue(bool(metadata["fallback_applied"][0]))
        self.assertFalse(bool(metadata["unavoidable_large_gt"][0]))
        self.assertEqual(float(metadata["sampled_condition_area_ratio"][0]), 1.0)
        self.assertEqual(float(metadata["condition_area_ratio"][0]), 0.04)

    def test_large_gt_is_preserved_and_marked_unavoidable(self):
        gt_mask = self._mask_with_area(90)
        condition_mask = torch.ones_like(gt_mask)

        guarded, metadata = apply_hybrid_masking_context_guard(
            condition_mask,
            gt_mask,
            self._config(),
        )

        self.assertTrue(torch.equal(guarded, gt_mask))
        self.assertTrue(bool(metadata["context_guard_triggered"][0]))
        self.assertTrue(bool(metadata["unavoidable_large_gt"][0]))
        self.assertEqual(float(metadata["condition_area_ratio"][0]), 0.90)


class HybridMaskingTrainerRoutingTest(unittest.TestCase):
    def test_enabled_fit_precomputed_mask_drives_hole_but_loss_keeps_gt(self):
        trainer = make_trainer(hybrid_enabled=True, mask_loss=True)
        batch = make_batch()
        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="fit")

        expected_control = batch["tar_image"] * (1 - batch["tar_mask_cond"].unsqueeze(1))
        self.assertTrue(torch.equal(output["control"], expected_control))
        self.assertTrue(torch.equal(output["mask"], batch["tar_mask"]))

    def test_precomputed_mask_is_fit_only_binary_and_gt_covering(self):
        trainer = make_trainer(hybrid_enabled=True, mask_loss=False)
        batch = make_batch()
        batch["tar_mask_cond"] = torch.zeros_like(batch["tar_mask"])
        batch["tar_mask_cond"][:, 1:3, 1:3] = 255.0

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            fit_output = trainer.prepare_embeddings(batch, stage="fit")
            predict_output = trainer.prepare_embeddings(batch, stage="predict")

        expected_fit_mask = torch.maximum(
            (batch["tar_mask_cond"] > 0.5).to(batch["tar_mask"].dtype),
            batch["tar_mask"],
        )
        expected_fit_control = batch["tar_image"] * (1 - expected_fit_mask.unsqueeze(1))
        expected_predict_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        self.assertTrue(torch.equal(fit_output["control"], expected_fit_control))
        self.assertTrue(torch.equal(predict_output["control"], expected_predict_control))
        self.assertGreaterEqual(float(fit_output["control"].min()), 0.0)
        self.assertLessEqual(float(fit_output["control"].max()), 1.0)

    def test_rank_local_rng_is_reproducible_and_differs_across_ranks(self):
        rank_zero = make_trainer(hybrid_enabled=True, mask_loss=False)
        rank_one = make_trainer(hybrid_enabled=True, mask_loss=False)
        rank_zero.accelerator = SimpleNamespace(process_index=0)
        rank_one.accelerator = SimpleNamespace(process_index=1)

        with patch(
            "src.trainer.qwen_image_edit_insert_anything_trainer.random.getrandbits",
            return_value=123456789,
        ):
            rank_zero_values = [rank_zero._hybrid_masking_rng().random() for _ in range(2)]
            rank_one_values = [rank_one._hybrid_masking_rng().random() for _ in range(2)]

        self.assertEqual(rank_zero_values[0], rank_zero_values[1])
        self.assertEqual(rank_one_values[0], rank_one_values[1])
        self.assertNotEqual(rank_zero_values[0], rank_one_values[0])

    def test_enabled_config_samples_only_during_fit(self):
        trainer = make_trainer(hybrid_enabled=True, mask_loss=False)
        batch = make_batch()
        del batch["tar_mask_cond"]
        sampled = torch.ones_like(batch["tar_mask"])

        with patch(
            "src.trainer.qwen_image_edit_insert_anything_trainer.build_insert_anything_hybrid_masks",
            return_value=(sampled, [INSERT_ANYTHING_EXPANDED_BOX]),
        ) as build_mock, patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            fit_output = trainer.prepare_embeddings(batch, stage="fit")
            self.assertEqual(build_mock.call_count, 1)
            predict_output = trainer.prepare_embeddings(batch, stage="predict")
            self.assertEqual(build_mock.call_count, 1)

        expected_fit_control = batch["tar_image"] * (1 - sampled.unsqueeze(1))
        expected_predict_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        self.assertTrue(torch.equal(fit_output["control"], expected_fit_control))
        self.assertTrue(torch.equal(predict_output["control"], expected_predict_control))

    def test_precomputed_full_frame_mask_is_guarded_during_fit(self):
        trainer = make_trainer(
            hybrid_enabled=True,
            context_guard_enabled=True,
            mask_loss=True,
        )
        batch = make_batch()
        batch["tar_mask_cond"] = torch.ones_like(batch["tar_mask"])

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="fit")

        expected_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        self.assertTrue(torch.equal(output["control"], expected_control))
        self.assertTrue(torch.equal(output["mask"], batch["tar_mask"]))
        metadata = trainer.get_last_hybrid_masking_metadata()
        self.assertEqual(metadata["modes"], ["precomputed"])
        self.assertTrue(bool(metadata["context_guard_triggered"][0]))
        self.assertEqual(
            float(metadata["condition_area_ratio"][0]),
            float(batch["tar_mask"].to(torch.float64).mean()),
        )

    def test_sampled_mode_identity_survives_context_guard_fallback(self):
        trainer = make_trainer(
            hybrid_enabled=True,
            context_guard_enabled=True,
            mask_loss=False,
        )
        batch = make_batch()
        del batch["tar_mask_cond"]
        sampled = torch.ones_like(batch["tar_mask"])

        with patch(
            "src.trainer.qwen_image_edit_insert_anything_trainer.build_insert_anything_hybrid_masks",
            return_value=(sampled, [INSERT_ANYTHING_EXPANDED_BOX]),
        ), patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="fit")

        expected_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        self.assertTrue(torch.equal(output["control"], expected_control))
        metadata = trainer.get_last_hybrid_masking_metadata()
        self.assertEqual(metadata["modes"], [INSERT_ANYTHING_EXPANDED_BOX])
        self.assertTrue(bool(metadata["context_guard_triggered"][0]))

    def test_metric_buffer_aggregates_microbatches_and_clears_after_logging(self):
        trainer = make_trainer(
            hybrid_enabled=True,
            context_guard_enabled=True,
            mask_loss=False,
        )
        trainer.accelerator = SimpleNamespace(gather=lambda value: value)
        config = trainer._hybrid_masking_cfg()
        gt_mask = torch.zeros(1, 10, 10)
        gt_mask[:, :2, :2] = 1
        full_mask = torch.ones_like(gt_mask)
        safe_mask = torch.zeros_like(gt_mask)
        safe_mask[:, :5, :5] = 1

        _, fallback_metadata = apply_hybrid_masking_context_guard(
            full_mask,
            gt_mask,
            config,
        )
        _, safe_metadata = apply_hybrid_masking_context_guard(
            safe_mask,
            gt_mask,
            config,
        )
        trainer._record_hybrid_masking_metrics(
            fallback_metadata,
            [INSERT_ANYTHING_EXPANDED_BOX],
        )
        trainer._record_hybrid_masking_metrics(
            safe_metadata,
            [INSERT_ANYTHING_AUGMENTED_BOX],
        )

        with patch.object(
            QwenImageEditPlusTrainer,
            "build_training_logs",
            autospec=True,
            return_value={"loss": 1.0},
        ):
            logs = trainer.build_training_logs(
                epoch=0,
                train_dataloader=[0],
                grad_norm=None,
                optimizer_step_time_sec=1.0,
            )

        self.assertEqual(logs["hybrid/context_guard_rate"], 0.5)
        self.assertEqual(logs["hybrid/mode_augmented_box_rate"], 0.5)
        self.assertEqual(logs["hybrid/mode_expanded_box_rate"], 0.5)
        self.assertEqual(logs["hybrid/unavoidable_large_gt_rate"], 0.0)
        self.assertEqual(trainer._hybrid_masking_metric_buffer, [])

    def test_training_error_keeps_root_exception_and_adds_sample_context(self):
        trainer = make_trainer(hybrid_enabled=True)
        trainer.accelerator = SimpleNamespace(process_index=3)
        batch = {
            "sample_metadata": {
                "dataset_index": [3614],
                "sample_id": ["accessory/4250"],
                "tar_image_path": ["/data/accessory/tar_image/4250.png"],
                "tar_mask_path": ["/data/accessory/tar_mask/4250.png"],
                "ref_image_path": ["/data/accessory/ref_image/4250.png"],
                "ref_mask_path": ["/data/accessory/ref_mask/4250.png"],
            }
        }

        with patch.object(
            QwenImageEditPlusTrainer,
            "training_step",
            autospec=True,
            side_effect=ValueError("root failure"),
        ), self.assertRaisesRegex(ValueError, "root failure") as caught:
            trainer.training_step(batch)

        notes = "\n".join(getattr(caught.exception, "__notes__", []))
        self.assertIn("dataset_index=3614", notes)
        self.assertIn("sample_id=accessory/4250", notes)
        self.assertIn("tar_mask_path=/data/accessory/tar_mask/4250.png", notes)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.config import (
    LayoutConditionConfig,
    MaskControlConfig,
    PreInsertionConfig,
    QueryPromptConfig,
    ReferenceCFGConfig,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer


def make_batch(true_cfg_scale=1.0):
    tar_image = torch.ones(1, 3, 6, 6)
    tar_mask = torch.zeros(1, 6, 6)
    tar_mask[:, 2:5, 1:5] = 1.0

    ref_image = torch.zeros(1, 3, 6, 6)
    ref_image[:, :, 1:5, 2:4] = 0.75
    ref_mask = torch.zeros(1, 6, 6)
    ref_mask[:, 1:5, 2:4] = 1.0

    return {
        "ref_image": ref_image,
        "ref_mask": ref_mask,
        "tar_image": tar_image,
        "tar_mask": tar_mask,
        "prompt": ["insert object"],
        "category": ["object"],
        "true_cfg_scale": true_cfg_scale,
    }


def make_trainer(
    *,
    mask_control_enabled=False,
    reference_cfg_enabled=False,
    force_empty_prompt=False,
    layout_enabled=False,
    pre_insertion=False,
    query_prompt=False,
):
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            query_prompt=QueryPromptConfig(enabled=query_prompt),
            pre_insertion=PreInsertionConfig(
                enabled=pre_insertion,
                train_scale_jitter_ratio=0.0,
                train_offset_jitter_ratio=0.0,
            ),
            layout_condition=LayoutConditionConfig(
                enabled=layout_enabled,
                train_scale_jitter_ratio=0.0,
                train_offset_jitter_ratio=0.0,
            ),
            mask_control=MaskControlConfig(enabled=mask_control_enabled),
            reference_cfg=ReferenceCFGConfig(
                enabled=reference_cfg_enabled,
                train_dropout_rate=0.0,
                force_empty_prompt=force_empty_prompt,
            ),
        ),
        loss=SimpleNamespace(mask_loss=False),
    )
    return trainer


def passthrough_prepare_embeddings(_self, batch, stage="fit"):
    return dict(batch)


class MaskControlTest(unittest.TestCase):
    def test_default_off_preserves_legacy_control_contract(self):
        trainer = make_trainer(mask_control_enabled=False, reference_cfg_enabled=True, force_empty_prompt=True)
        batch = make_batch(true_cfg_scale=4.0)

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        self.assertTrue(torch.equal(output["control"], expected_control))
        self.assertTrue(torch.equal(output["control_1"], batch["ref_image"]))
        self.assertEqual(output["n_controls"], 1)
        self.assertNotIn("control_2", output)
        self.assertNotIn("vae_only_control_keys", output)
        self.assertEqual(output["reference_cfg_scale"], 4.0)

        null_branch = output["reference_cfg_embeddings"]
        self.assertEqual(null_branch["n_controls"], 0)
        self.assertNotIn("control_1", null_branch)
        self.assertNotIn("vae_only_control_keys", null_branch)

    def test_explicit_mask_control_adds_vae_only_control(self):
        trainer = make_trainer(mask_control_enabled=True)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_mask = batch["tar_mask"].unsqueeze(1).repeat(1, 3, 1, 1)
        self.assertTrue(torch.equal(output["control_1"], batch["ref_image"]))
        self.assertTrue(torch.equal(output["control_2"], expected_mask))
        self.assertEqual(output["n_controls"], 2)
        self.assertEqual(output["vae_only_control_keys"], ["control_2"])
        # Explicit mask_control alone must not force CFG scale to 1.
        self.assertEqual(output["true_cfg_scale"], 1.0)
        self.assertEqual(output["prompt"], ["insert object"])

    def test_explicit_mask_control_with_reference_cfg_keeps_scale_and_null_mask(self):
        trainer = make_trainer(
            mask_control_enabled=True,
            reference_cfg_enabled=True,
            force_empty_prompt=True,
        )
        batch = make_batch(true_cfg_scale=4.0)

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_mask = batch["tar_mask"].unsqueeze(1).repeat(1, 3, 1, 1)
        self.assertEqual(output["n_controls"], 2)
        self.assertTrue(torch.equal(output["control_1"], batch["ref_image"]))
        self.assertTrue(torch.equal(output["control_2"], expected_mask))
        self.assertEqual(output["vae_only_control_keys"], ["control_2"])
        self.assertEqual(output["prompt"], [""])
        self.assertEqual(output["reference_cfg_scale"], 4.0)
        # Positive parent_batch true_cfg_scale is 1 for the DiT call itself;
        # reference scale is carried separately.
        self.assertEqual(output["true_cfg_scale"], 4.0)

        null_branch = output["reference_cfg_embeddings"]
        self.assertEqual(null_branch["n_controls"], 1)
        self.assertTrue(torch.equal(null_branch["control_1"], expected_mask))
        self.assertEqual(null_branch["vae_only_control_keys"], ["control_1"])
        self.assertNotIn("control_2", null_branch)
        self.assertEqual(null_branch["prompt"], [""])
        self.assertEqual(null_branch["true_cfg_scale"], 1.0)

    def test_layout_plus_mask_control_null_keeps_only_mask(self):
        trainer = make_trainer(
            mask_control_enabled=True,
            layout_enabled=True,
            reference_cfg_enabled=True,
            force_empty_prompt=True,
        )
        batch = make_batch(true_cfg_scale=3.0)

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_mask = batch["tar_mask"].unsqueeze(1).repeat(1, 3, 1, 1)
        # control_1=ref, control_2=layout, control_3=mask
        self.assertEqual(output["n_controls"], 3)
        self.assertTrue(torch.equal(output["control_3"], expected_mask))
        self.assertEqual(output["vae_only_control_keys"], ["control_2", "control_3"])

        null_branch = output["reference_cfg_embeddings"]
        self.assertEqual(null_branch["n_controls"], 1)
        self.assertTrue(torch.equal(null_branch["control_1"], expected_mask))
        self.assertEqual(null_branch["vae_only_control_keys"], ["control_1"])

    def test_enhanced_mask_path_still_attaches_mask_without_explicit_switch(self):
        trainer = make_trainer(pre_insertion=True, mask_control_enabled=False)
        batch = make_batch(true_cfg_scale=4.0)

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_mask = batch["tar_mask"].unsqueeze(1).repeat(1, 3, 1, 1)
        self.assertTrue(torch.equal(output["control_2"], expected_mask))
        self.assertEqual(output["true_cfg_scale"], 1.0)
        self.assertEqual(output["prompt"], [""])

    def test_missing_mask_control_config_defaults_to_legacy_off(self):
        trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
        trainer.config = SimpleNamespace(
            model=SimpleNamespace(
                query_prompt=QueryPromptConfig(enabled=False),
                pre_insertion=PreInsertionConfig(enabled=False),
                layout_condition=LayoutConditionConfig(enabled=False),
                # no mask_control attribute
                reference_cfg=ReferenceCFGConfig(enabled=False),
            ),
            loss=SimpleNamespace(mask_loss=False),
        )
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        self.assertEqual(output["n_controls"], 1)
        self.assertNotIn("vae_only_control_keys", output)


if __name__ == "__main__":
    unittest.main()

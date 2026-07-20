import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.config import (
    LayoutConditionConfig,
    PreInsertionConfig,
    QueryPromptConfig,
    ReferenceCFGConfig,
)
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from src.utils.images import build_preinserted_composite


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


def make_trainer(layout_enabled=False, reference_cfg_enabled=False, force_empty_prompt=False):
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            query_prompt=QueryPromptConfig(enabled=False),
            pre_insertion=PreInsertionConfig(enabled=False),
            layout_condition=LayoutConditionConfig(
                enabled=layout_enabled,
                train_scale_jitter_ratio=0.0,
                train_offset_jitter_ratio=0.0,
            ),
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


class LayoutConditionTest(unittest.TestCase):
    def test_disabled_layout_condition_preserves_control_contract(self):
        trainer = make_trainer(layout_enabled=False)
        batch = make_batch()

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
        self.assertEqual(output["prompt"], ["insert object"])

    def test_layout_condition_adds_vae_only_control_without_replacing_main_control(self):
        trainer = make_trainer(layout_enabled=True)
        batch = make_batch()

        with patch.object(
            QwenImageEditPlusTrainer,
            "prepare_embeddings",
            autospec=True,
            side_effect=passthrough_prepare_embeddings,
        ):
            output = trainer.prepare_embeddings(batch, stage="predict")

        expected_control = batch["tar_image"] * (1 - batch["tar_mask"].unsqueeze(1))
        expected_layout = build_preinserted_composite(
            target_image=batch["tar_image"],
            target_mask=batch["tar_mask"],
            ref_image=batch["ref_image"],
            ref_mask=batch["ref_mask"],
            training=False,
            scale_jitter_ratio=0.0,
            offset_jitter_ratio=0.0,
            paste_mode="bbox_fit_keep_aspect",
        )

        self.assertTrue(torch.equal(output["control"], expected_control))
        self.assertTrue(torch.equal(output["control_1"], batch["ref_image"]))
        self.assertTrue(torch.equal(output["control_2"], expected_layout))
        self.assertEqual(output["n_controls"], 2)
        self.assertEqual(output["vae_only_control_keys"], ["control_2"])

    def test_reference_cfg_null_branch_excludes_layout_condition(self):
        trainer = make_trainer(
            layout_enabled=True,
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

        self.assertEqual(output["n_controls"], 2)
        self.assertEqual(output["vae_only_control_keys"], ["control_2"])
        self.assertEqual(output["prompt"], [""])
        self.assertEqual(output["reference_cfg_scale"], 4.0)

        null_branch = output["reference_cfg_embeddings"]
        self.assertEqual(null_branch["n_controls"], 0)
        self.assertNotIn("control_1", null_branch)
        self.assertNotIn("control_2", null_branch)
        self.assertNotIn("vae_only_control_keys", null_branch)
        self.assertEqual(null_branch["prompt"], [""])


if __name__ == "__main__":
    unittest.main()

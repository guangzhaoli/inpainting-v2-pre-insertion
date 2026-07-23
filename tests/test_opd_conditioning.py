from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.data.config import MaskControlConfig, ReferenceCFGConfig
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    QwenImageEditInsertAnythingTrainer,
)
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer


def _passthrough_prepare_embeddings(_self, batch, stage="fit"):
    assert stage == "opd"
    return dict(batch)


def test_opd_conditioning_excludes_ground_truth_and_preserves_cfg_geometry():
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            mask_control=MaskControlConfig(enabled=False),
            reference_cfg=ReferenceCFGConfig(
                enabled=False,
                train_dropout_rate=0.0,
                force_empty_prompt=False,
            ),
        ),
        loss=SimpleNamespace(mask_loss=False),
    )
    height, width = 12, 20
    target_mask = torch.zeros(2, height, width)
    target_mask[:, 3:9, 5:15] = 1.0
    batch = {
        "ref_image": torch.rand(2, 3, height, width),
        "ref_mask": torch.ones(2, height, width),
        "tar_image": torch.rand(2, 3, height, width),
        "tar_mask": target_mask,
        "prompt": ["first", "second"],
        "category": ["cat", "others"],
        "true_cfg_scale": 4.0,
        # OPD policy overrides are independent from the trainer's inference
        # defaults and allow a guided teacher with an unguided student.
        "_opd_use_reference_cfg": True,
        "_opd_prompt_override": ["", ""],
    }

    with patch.object(
        QwenImageEditPlusTrainer,
        "prepare_embeddings",
        autospec=True,
        side_effect=_passthrough_prepare_embeddings,
    ):
        output = trainer.prepare_embeddings(batch, stage="opd")

    assert "image" not in output
    assert "image_latents" not in output
    assert (output["height"], output["width"]) == (height, width)
    assert output["prompt"] == ["", ""]

    null_reference = output["reference_cfg_embeddings"]
    assert "image" not in null_reference
    assert "image_latents" not in null_reference
    assert (null_reference["height"], null_reference["width"]) == (height, width)
    assert null_reference["n_controls"] == 0
    assert "control_1" not in null_reference


def test_opd_geometry_condition_is_kept_but_gt_region_fields_are_excluded():
    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = SimpleNamespace(
        model=SimpleNamespace(
            mask_control=MaskControlConfig(enabled=True, encoding="geometry_tokenizer"),
            reference_cfg=ReferenceCFGConfig(
                enabled=False,
                train_dropout_rate=0.0,
                force_empty_prompt=False,
            ),
        ),
        # These are intentionally enabled to prove stage='opd' does not leak
        # supervised target/region tensors into either policy condition.
        loss=SimpleNamespace(mask_loss=True, region_perceptual=True),
    )
    height, width = 12, 20
    target_mask = torch.zeros(1, height, width)
    target_mask[:, 3:9, 5:15] = 1.0
    batch = {
        "ref_image": torch.rand(1, 3, height, width),
        "ref_mask": torch.ones(1, height, width),
        "tar_image": torch.rand(1, 3, height, width),
        "tar_mask": target_mask,
        "prompt": ["object"],
        "category": ["cat"],
        "true_cfg_scale": 4.0,
        "_opd_use_reference_cfg": True,
        "_opd_prompt_override": [""],
    }

    with patch.object(
        QwenImageEditPlusTrainer,
        "prepare_embeddings",
        autospec=True,
        side_effect=_passthrough_prepare_embeddings,
    ):
        output = trainer.prepare_embeddings(batch, stage="opd")

    assert "mask_geometry" in output
    assert output["mask_geometry"].shape == (1, 3, height, width)
    assert "image" not in output
    assert "image_latents" not in output
    assert "target_image_for_loss" not in output
    assert "ref_image_for_loss" not in output
    assert "ref_mask_for_loss" not in output

    null_reference = output["reference_cfg_embeddings"]
    assert torch.equal(null_reference["mask_geometry"], output["mask_geometry"])
    assert "target_image_for_loss" not in null_reference

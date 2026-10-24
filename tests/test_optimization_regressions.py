from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import safetensors.torch
import torch
import torch.nn.functional as F
import yaml
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader, SequentialSampler

from src.data.resume import EpochOffsetSampler, resume_dataloader
from src.distillation.objectives import VelocityMatchingObjective
from src.loss.local_detail import interaction_masks
from src.trainer.base_trainer import BaseTrainer


@pytest.mark.parametrize("radius", [1, 4, 16, 32])
def test_separable_morphology_is_exact_including_image_edges(radius):
    torch.manual_seed(9)
    mask = torch.rand(2, 1, 19, 37).gt(0.7).float()
    kernel = 2 * radius + 1
    dilated = F.max_pool2d(mask, kernel, 1, radius)
    eroded = -F.max_pool2d(-mask, kernel, 1, radius)
    ring, background = interaction_masks(mask, radius)
    assert torch.equal(ring, (dilated - eroded).clamp(0, 1))
    assert torch.equal(background, 1 - dilated)


def test_teacher_weights_have_identical_gradients_when_batch_is_sharded():
    objective = VelocityMatchingObjective()

    def gradient(targets, weights):
        p = torch.tensor(0.0, requires_grad=True)
        loss = objective(
            p.expand(len(targets), 1),
            torch.tensor(targets).view(-1, 1),
            teacher_weight=torch.tensor(weights),
        ).loss
        loss.backward()
        return p.grad

    joint = gradient([1.0, 2.0], [1.0, 9.0])
    shards = (gradient([1.0], [1.0]) + gradient([2.0], [9.0])) / 2
    torch.testing.assert_close(joint, shards)
    torch.testing.assert_close(gradient([1.0], [10.0]), 10 * gradient([1.0], [1.0]))


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_resume_cursor_is_sliced_before_sharding_without_missing_tail(world):
    dataset = list(range(35))
    base = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=1, drop_last=True
    )
    seen = []
    for rank in range(world):
        resumed = resume_dataloader(base, epoch=2, offset=13)
        sharded = prepare_data_loader(
            resumed, device=torch.device("cpu"), num_processes=world, process_index=rank
        )
        sharded.set_epoch(2)
        seen.extend(int(x.item()) for x in sharded)
    assert set(seen) == set(range(13, 35))
    assert min(seen) == 13
    sampler = EpochOffsetSampler(SequentialSampler(dataset), 2, 13)
    sampler.set_epoch(3)
    assert list(sampler) == dataset


def test_expansion_preserves_optimizer_step_budget_and_extends_epochs():
    trainer = NS(
        config=NS(
            train=NS(gradient_accumulation_steps=1, num_epochs=5, max_train_steps=1000)
        ),
        global_step=10,
        start_epoch=1,
        _full_epoch_num_batches=8,
        _resume_state={"effective_max_train_steps": 80},
    )
    BaseTrainer.compute_training_schedule(trainer, [None] * 6)
    assert trainer.effective_max_train_steps == 80
    assert trainer.num_epochs == 10


@pytest.mark.parametrize("saved_drop_last", [True, False])
def test_same_world_resume_keeps_tail_policy_and_idle_checkpoint_cursor(saved_drop_last):
    dataset = list(range(35))
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=1, drop_last=True)
    trainer = NS(config=NS(train=NS(seed=42), data=NS(batch_size=1)),
                 accelerator=NS(num_processes=2), start_epoch=1, resume_batch_in_epoch=6,
                 _resume_state=dict(num_processes=2, data_batch_size=1, sampler_num_samples=35,
                                    global_samples_in_epoch=12, effective_max_train_steps=85,
                                    dataloader_drop_last=saved_drop_last))
    resumed = BaseTrainer.prepare_resume_dataloader(trainer, loader)
    assert resumed.drop_last == saved_drop_last
    assert trainer._epoch_sample_offset == 12 and trainer.batch_in_epoch == 0
    assert trainer._full_epoch_num_batches == (17 if saved_drop_last else 18)
    seen = []
    for rank in range(2):
        shards = prepare_data_loader(resumed, device=torch.device('cpu'),
                                    num_processes=2, process_index=rank)
        shards.set_epoch(1)
        seen.extend(int(x.item()) for x in shards)
    assert set(seen) == set(range(12, 34 if saved_drop_last else 35))


def test_conditioning_offload_is_exception_safe():
    stream = Mock()
    trainer = NS(
        config=NS(train=NS(offload_text_encoder=True)),
        _stream_text_encoder=stream,
        accelerator=NS(device=torch.device("cpu")),
    )
    with pytest.raises(RuntimeError):
        with BaseTrainer.conditioning_context(trainer):
            raise RuntimeError("encoding failed")
    assert [call.args for call in stream.call_args_list] == [(torch.device("cpu"),), (None,)]


@pytest.mark.parametrize("gib,policy,expected", [(80, "auto", True), (141, "auto", False), (80, "configured", False)])
def test_memory_policy_selects_h100_safety_and_h200_throughput(monkeypatch, gib, policy, expected):
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda device: NS(total_memory=gib*2**30))
    train = NS(memory_policy=policy, offload_text_encoder=False, checkpoint_vae_decode=False)
    trainer = NS(config=NS(train=train, loss=NS(region_perceptual=True)), accelerator=NS(device=torch.device('cuda:0')))
    BaseTrainer.resolve_memory_policy(trainer)
    assert train.offload_text_encoder == expected
    assert train.checkpoint_vae_decode == expected


def test_checkpoint_builder_preserves_teacher_specific_conditioning(tmp_path):
    from scripts.build_opd_config import build_config
    from src.data.config import Config
    from src.distillation.checkpoints import preflight_teacher_checkpoints

    root = Path(__file__).resolve().parents[1]
    checkpoints = {}
    for domain in ("accessory_jewelry", "human"):
        folder = tmp_path / domain
        folder.mkdir()
        raw = yaml.safe_load(
            (root / "configs/teacher_v2" / f"{domain}.yaml").read_text()
        )
        (folder / "train_config.yaml").write_text(yaml.safe_dump(raw))
        for filename in ("model.safetensors", "mask_geometry_tokenizer.safetensors"):
            safetensors.torch.save_file(
                {"probe": torch.zeros(1)}, str(folder / filename)
            )
        checkpoints[domain] = str(folder)
    result = build_config(checkpoints)
    cfg = Config.model_validate(result)
    assert (
        cfg.model.lora.r == 128 and cfg.model.attention_backend == "flash_attention_2"
    )
    assert all(
        t.guidance.mask_aware and t.guidance.delta_clip_enabled
        for t in cfg.distillation.teachers
    )
    profiles = cfg.data.init_args.category_conditioning
    assert profiles["accessory"].hybrid_masking.interaction_dilation_px == 16
    assert profiles["person"].hybrid_masking.interaction_dilation_px == 32
    # The human teacher's own person_hair profile (no loose families) survives.
    assert profiles["person"].hybrid_masking.maskmix_v2.mode_weights.loose_blob > 0
    assert profiles["person_hair"].hybrid_masking.maskmix_v2.mode_weights.loose_blob == 0
    # Teachers train with lighting augmentation; OPD keeps the pairs as photographed.
    assert not cfg.data.init_args.lighting_augmentation.enabled
    cfg.distillation.teachers[0].lora.r = 256
    with pytest.raises(ValueError, match="lora.r"):
        preflight_teacher_checkpoints(cfg)
    cfg.distillation.teachers[0].lora.r = 128
    cfg.model.zero_cond_t = not cfg.model.zero_cond_t
    with pytest.raises(ValueError, match="model.zero_cond_t"):
        preflight_teacher_checkpoints(cfg)


def test_opd_reuses_actual_positive_and_null_frozen_encoders():
    from src.data.config import (
        HybridMaskingConfig,
        MaskControlConfig,
        ReferenceCFGConfig,
    )
    from src.trainer.qwen_image_edit_insert_anything_trainer import (
        QwenImageEditInsertAnythingTrainer,
    )
    from src.distillation.opd import QwenImageEditOPDStrategy

    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = NS(
        model=NS(
            mask_control=MaskControlConfig(enabled=True, encoding="geometry_tokenizer"),
            reference_cfg=ReferenceCFGConfig(enabled=False),
        ),
        loss=NS(mask_loss=False),
        data=NS(init_args=NS(hybrid_masking=HybridMaskingConfig(enabled=True))),
    )
    trainer.weight_dtype = torch.float32
    trainer.num_channels_latents = 16
    trainer.preprocess_image_for_text_encoder = lambda x, **kw: x * 255
    trainer.preprocess_image_for_vae_encoder = lambda x, **kw: x.unsqueeze(2)
    trainer.encode_prompt = Mock(
        side_effect=lambda **kw: (
            torch.full((1, 1, 4), float(len(kw["image"]))),
            torch.ones(1, 1),
        )
    )
    trainer.encode_image_latents = Mock(
        side_effect=lambda x, *a: torch.ones(1, 4, 64) * x.mean()
    )
    student = NS(mode="none", scale=1.0, prompt_mode="empty")
    teacher = NS(mode="reference_cfg", scale=2.5, prompt_mode="empty")
    strategy = object.__new__(QwenImageEditOPDStrategy)
    strategy.config = NS(student_guidance=student)
    strategy.teachers = {"expert": NS(guidance=teacher)}
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1
    batch = dict(
        ref_image=torch.rand(1, 3, 32, 32),
        ref_mask=torch.ones_like(mask),
        tar_image=torch.rand(1, 3, 32, 32),
        tar_mask=mask,
        prompt=[""],
        category=["accessory"],
        tar_mask_cond=mask.clone(),
        _opd_condition_mask_prepared=True,
    )
    conditions = strategy._prepare_conditions(trainer, batch)
    assert trainer.encode_prompt.call_count == 2
    assert trainer.encode_image_latents.call_count == 2
    a, b = (
        conditions[strategy._condition_key(student)],
        conditions[strategy._condition_key(teacher)],
    )
    assert a["prompt_embeds"] is b["prompt_embeds"]
    assert a["mask_geometry"] is b["mask_geometry"]
    assert b["reference_cfg_embeddings"]["prompt_embeds"].mean() == 1

"""Regressions for correctness and avoidable work found in the runtime review."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.build_meanflow_config import build_config
from src.data.config import Config
from src.distillation.meanflow import (
    QwenImageEditMeanFlowDMDStrategy,
    forward_map,
    sample_flow_map,
    sigma_schedule,
)
from src.distillation.meanflow_math import flow_step, meanflow_loss
from tests.test_meanflow_dmd import FitTrainer, TinyTrainer, make_case


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_diagonal_time_does_not_round_to_invalid_interval(tmp_path):
    trainer, strategy, batch = make_case(tmp_path)
    _, condition, _ = strategy._prepare(trainer, batch)
    t = torch.tensor([0.020101508125662804])
    assert bool((t * 1000 / 1000 < t).all())  # Regression trigger in FP32.
    observed = []
    handle = trainer.dit.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(
            (kwargs["timestep"].clone(), kwargs["end_timestep"].clone())
        ),
        with_kwargs=True,
    )
    try:
        output = forward_map(trainer, torch.randn(1, 4, 8), t, t, condition)
        output.square().mean().backward()
    finally:
        handle.remove()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(observed[0][0], observed[0][1], rtol=0, atol=0)


def test_diagonal_meanflow_uses_one_forward_and_same_gradient():
    parameter = torch.nn.Parameter(torch.tensor(0.3))
    clean, noise = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
    t = torch.tensor([0.2, 0.6])
    calls = []

    def predict(x, start, end):
        calls.append(torch.is_grad_enabled())
        return parameter * x

    loss = meanflow_loss(predict, clean, noise, t, t)
    loss.backward()
    x = (1 - t[:, None, None]) * clean + t[:, None, None] * noise
    expected = (2 * (parameter.detach() * x - (noise - clean)) * x).mean()
    torch.testing.assert_close(parameter.grad, expected)
    assert calls == [True]


def test_schedule_cache_is_safe_after_validation_and_reuses_work(tmp_path):
    trainer, strategy, batch = make_case(tmp_path)
    with patch.object(
        trainer, "prepare_predict_timesteps", wraps=trainer.prepare_predict_timesteps
    ) as prepare:
        with torch.inference_mode():
            first = sigma_schedule(trainer, 2, 4)
        second = sigma_schedule(trainer, 2, 4)
        assert prepare.call_count == 1
        assert first is second and not second.is_inference()
        state = torch.ones(1, 4, 8, requires_grad=True)
        flow_step(state, state, second[0], second[1]).sum().backward()
        sigma_schedule(trainer, 3, 4)
        sigma_schedule(trainer, 2, 16)
        assert prepare.call_count == 3
    sample = sample_flow_map(trainer, batch["condition"])
    assert not sample.requires_grad


def test_real_qwen_scheduler_contract_and_cache(tmp_path):
    from types import MethodType
    from diffusers import FlowMatchEulerDiscreteScheduler
    from src.trainer.base_trainer import BaseTrainer

    trainer, _, _ = make_case(tmp_path)
    trainer.scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        shift_terminal=0.02,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        base_shift=0.5,
        max_shift=0.9,
    )
    trainer.prepare_predict_timesteps = MethodType(
        BaseTrainer.prepare_predict_timesteps, trainer
    )
    expected = {}
    for steps in (40, 8, 4, 1):
        sigma = sigma_schedule(trainer, steps, 4096)
        assert sigma.shape == (steps + 1,)
        assert sigma[0] == 1 and sigma[-1] == 0
        assert torch.all(sigma[:-1] > sigma[1:])
        expected[steps] = sigma.clone()
    # Reusing the 40-step cache is safe after other sampling budgets.
    torch.testing.assert_close(sigma_schedule(trainer, 40, 4096), expected[40])


def test_keep_loss_reuses_teacher_prediction(tmp_path):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    strategy.options.forward_weight = 0
    strategy.options.keep_weight = 0.1
    roles = []
    original = strategy._instantaneous

    def counted(*args):
        roles.append(strategy._active_role)
        return original(*args)

    strategy._instantaneous = counted
    strategy.backward_step(trainer, batch)
    assert roles.count("teacher") == 1
    assert roles.count("fake") == strategy.options.fake_updates + 1


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_cached_rollout_geometry_preserves_outputs_and_gradients(tmp_path, precision):
    trainer, strategy, batch = make_case(tmp_path)
    if precision == "bf16":
        from src.utils.seed import prepare_trainable_precision

        trainer.weight_dtype = torch.bfloat16
        trainer.dit.to(dtype=torch.bfloat16)
        prepare_trainable_precision(trainer.dit, "float32")
        trainer.accelerator.autocast = lambda: torch.autocast(
            "cpu", dtype=torch.bfloat16
        )
    _, condition, _ = strategy._prepare(trainer, batch)
    strategy.options.student_steps = 4
    torch.manual_seed(51)
    with patch.object(
        trainer.dit.mask_geometry_tokenizer,
        "forward",
        wraps=trainer.dit.mask_geometry_tokenizer.forward,
    ) as tokenize:
        output = strategy._generate(condition, with_grad=True)
        output.square().mean().backward()
        assert tokenize.call_count == 2  # One no-grad prefix cache + live final step.
    expected = [
        p.grad.clone() if p.grad is not None else None
        for p in strategy.generator_parameters(trainer)
    ]
    trainer.dit.zero_grad(set_to_none=True)
    torch.manual_seed(51)
    with patch.object(
        strategy,
        "_cache_student_geometry",
        side_effect=lambda trainer, condition: condition,
    ):
        reference = strategy._generate(condition, with_grad=True)
        reference.square().mean().backward()
    torch.testing.assert_close(reference, output, rtol=0, atol=0)
    for parameter, expected_grad in zip(
        strategy.generator_parameters(trainer), expected
    ):
        if expected_grad is not None:
            torch.testing.assert_close(parameter.grad, expected_grad)


def test_role_switch_uses_cached_parameters_and_keeps_frozen_roles(tmp_path):
    trainer, strategy, _ = make_case(tmp_path, "dmd")
    with patch.object(
        trainer.dit,
        "named_parameters",
        side_effect=AssertionError("hot-path full scan"),
    ):
        for role in ("generator", "teacher", "fake", "generator"):
            strategy._activate(role)
            strategy.fake_parameters(trainer)
            strategy.generator_parameters(trainer)
    assert trainer.dit.active_adapters() == [trainer.adapter_name]
    strategy.backend.assert_frozen()


def test_condition_is_encoded_once_despite_unused_cfg_defaults(tmp_path):
    trainer, strategy, batch = make_case(tmp_path)
    strategy.teacher.guidance.mask_aware = True
    strategy.teacher.guidance.delta_clip_enabled = True
    with patch.object(
        trainer, "prepare_embeddings", wraps=trainer.prepare_embeddings
    ) as encode:
        strategy._prepare(trainer, batch)
    assert encode.call_count == 1


@pytest.mark.parametrize("independent_source", [False, True])
def test_real_conditioning_reuses_masked_vae_source_without_gt_leak(
    tmp_path, independent_source
):
    from types import MethodType
    from src.trainer.qwen_image_edit_insert_anything_trainer import (
        QwenImageEditInsertAnythingTrainer,
    )

    trainer, strategy, _ = make_case(tmp_path)
    strategy.options.anchoring = "deployment"
    trainer.config.predict.latent_anchoring.mode = "core"
    trainer.prepare_embeddings = MethodType(
        QwenImageEditInsertAnythingTrainer.prepare_embeddings, trainer
    )
    trainer.preprocess_image_for_text_encoder = lambda image, **kwargs: image * 255
    trainer.encode_prompt = lambda **kwargs: (
        torch.ones(1, 3, 8),
        torch.ones(1, 3, dtype=torch.long),
    )
    encoded = []

    def vae(image, batch_size, channels, dtype):
        encoded.append(image.clone())
        latent = torch.nn.functional.avg_pool2d(image[:, :2, 0], 8)
        return trainer._pack_latents(latent, batch_size, channels, 4, 4)

    trainer.encode_image_latents = vae
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1
    batch = dict(
        tar_image=torch.ones(1, 3, 32, 32),
        ref_image=torch.full((1, 3, 32, 32), 0.5),
        tar_mask=mask,
        tar_mask_cond=mask,
        ref_mask=torch.ones_like(mask),
        prompt=[""],
        category=["accessory"],
        _opd_condition_mask_prepared=True,
    )
    if independent_source:
        batch["source_image"] = torch.full_like(batch["tar_image"], 0.2)
    _, condition, frozen = strategy._prepare(trainer, batch)
    assert len(encoded) == (3 if independent_source else 2)
    torch.testing.assert_close(
        encoded[0][:, :, :, 8:24, 8:24], torch.full((1, 3, 1, 16, 16), -1.0)
    )
    torch.testing.assert_close(condition["control_latents"], frozen["control_latents"])
    if independent_source:
        torch.testing.assert_close(encoded[-1], torch.full((1, 3, 1, 32, 32), -0.6))
    else:
        torch.testing.assert_close(
            condition["anchor_source_latents"], condition["control_latents"][:, :4]
        )


def test_meanflow_setup_does_not_apply_routed_opd_batch_limit(tmp_path):
    prototype, _, _ = make_case(tmp_path)
    config = prototype.config.model_copy(deep=True)
    config.data.batch_size = 2
    trainer = TinyTrainer(config)
    trainer.accelerator.num_processes = 2
    strategy = QwenImageEditMeanFlowDMDStrategy(config.distillation)
    strategy.setup(trainer)
    assert not hasattr(strategy, "objective")
    assert not hasattr(strategy, "router")


def test_testset_setup_accepts_flow_checkpoint_and_sampling_defaults(tmp_path):
    from infer_testset import parse_args, setup_trainer

    trainer, _, _ = make_case(tmp_path)
    folder = tmp_path / "flow"
    folder.mkdir()
    trainer.save_lora(str(folder))
    trainer.save_additional_modules(str(folder))
    config_path = folder / "train_config.yaml"
    config_path.write_text(yaml.safe_dump(trainer.config.model_dump(mode="json")))
    args = parse_args(
        ["--config", str(config_path), "--lora", str(folder), "--device", "cpu"]
    )
    with patch(
        "src.trainer.qwen_image_edit_insert_anything_trainer.QwenImageEditInsertAnythingTrainer"
    ) as cls:
        _, config, _, _ = setup_trainer(args)
        cls.return_value.setup_predict.assert_called_once()
    assert config.model.flow_map.enabled
    assert args.num_inference_steps == 8 and args.true_cfg_scale == 1


def test_validation_accepts_missing_prompts_for_absorbed_visual_conditioning(tmp_path):
    from types import SimpleNamespace
    from src.trainer.insert_anything_validation_sampling import (
        InsertAnythingValidationSampler,
    )
    from src.utils.inference_metadata import build_effective_inference_metadata
    from tests.test_validation_sampling import FakeAccelerator, FakeDataset

    trainer, _, _ = make_case(tmp_path)
    dataset = FakeDataset()
    original = dataset._get_sample_info
    dataset._get_sample_info = lambda index: dict(original(index), prompt=None)
    trainer.config.logging.sampling.enable = True
    trainer.config.logging.sampling.true_cfg_scale = 1
    sampler = InsertAnythingValidationSampler(
        trainer.config, SimpleNamespace(dataset=dataset), FakeAccelerator()
    )
    assert len(sampler.sample_indices) == 4
    metadata = build_effective_inference_metadata(
        trainer.config,
        prompt="ignored text",
        negative_prompt=None,
        num_inference_steps=8,
        true_cfg_scale=1,
        guidance_scale=1,
    )
    assert metadata["requested"]["prompt"] == "ignored text"
    assert metadata["effective"]["prompt"] == ""


def test_builder_uses_single_branch_validation_defaults(tmp_path):
    make_case(tmp_path)
    path = tmp_path / "teacher" / "train_config.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["logging"]["sampling"].update(
        enable=True, negative_prompt="blur", true_cfg_scale=4
    )
    path.write_text(yaml.safe_dump(raw))
    generated = Config.model_validate(build_config(tmp_path / "teacher"))
    assert generated.logging.sampling.negative_prompt is None
    assert generated.logging.sampling.true_cfg_scale == 1
    assert generated.model.reference_cfg.force_empty_prompt == (
        generated.distillation.student_guidance.prompt_mode == "empty"
    )


@pytest.mark.parametrize("diagonal,expected", [(1.0, 60), (0.0, 62)])
def test_default_dmd_forward_budget_has_no_duplicate_work(tmp_path, diagonal, expected):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    strategy.options.teacher_steps = 40
    strategy.options.student_steps = 8
    strategy.options.fake_updates = 1
    strategy.options.diagonal_probability = diagonal
    calls = []
    handle = trainer.dit.register_forward_pre_hook(lambda *args: calls.append(True))
    try:
        strategy.backward_step(trainer, batch)
    finally:
        handle.remove()
    assert len(calls) == expected


def _collate_synthetic(rows):
    first = rows[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(rows, dim=0)
    if isinstance(first, dict):
        return {key: _collate_synthetic([row[key] for row in rows]) for key in first}
    if isinstance(first, list):
        return [item for row in rows for item in row]
    return first


def _distributed_fit_worker(rank, folder, port, precision):
    import os
    import torch.distributed as dist

    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        ACCELERATE_USE_CPU="true",
    )
    torch.set_num_threads(1)
    prototype, _, batch = make_case(Path(folder), "dmd")
    config = prototype.config.model_copy(deep=True)
    config.data.batch_size = 2
    config.data.num_workers = 0
    config.data.drop_last = False
    config.train.gradient_accumulation_steps = 2
    config.train.max_train_steps = 1
    config.train.num_epochs = 1
    config.train.mixed_precision = precision
    config.train.checkpointing_steps = 99
    config.logging.output_dir = str(Path(folder) / "fit")
    config.logging.sampling.enable = False
    trainer = FitTrainer(
        Config.model_validate(
            config.model_dump(mode="json", exclude=set(Config.model_computed_fields))
        )
    )
    original_save = trainer.save_checkpoint

    def checked_save(epoch, step, is_last=False):
        original_save(epoch, step, is_last=is_last)
        if is_last:
            assert trainer.accelerator.num_processes == 2
            for parameters in (
                trainer.training_strategy.generator_parameters(trainer),
                trainer.training_strategy.fake_parameters(trainer),
            ):
                flat = torch.cat(
                    [parameter.detach().flatten() for parameter in parameters]
                )
                replicas = [torch.empty_like(flat) for _ in range(2)]
                dist.all_gather(replicas, flat)
                torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)

    trainer.save_checkpoint = checked_save
    try:
        trainer.fit(
            DataLoader([batch] * 8, batch_size=2, collate_fn=_collate_synthetic)
        )
        assert trainer.global_step == 1
        assert trainer.training_strategy.fake_update_count == 4
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_real_two_rank_fit_with_microbatch_two(tmp_path, precision):
    import socket
    import torch.multiprocessing as mp

    make_case(tmp_path, "dmd")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(
        _distributed_fit_worker,
        args=(str(tmp_path), port, precision),
        nprocs=2,
        join=True,
    )

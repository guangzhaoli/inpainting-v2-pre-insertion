import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
import yaml
from peft import LoraConfig, get_peft_model_state_dict

from src.data.config import Config, load_config_from_yaml
from src.distillation.backends import is_adapter_parameter
from src.distillation.meanflow import (
    QwenImageEditMeanFlowDMDStrategy,
    forward_map,
    sample_flow_map,
)
from src.distillation.meanflow_math import (
    add_noise,
    dmd_surrogate,
    meanflow_loss,
    velocity_to_score,
)
from src.models.flow_map import FlowMapTimeEmbedding, FLOW_MAP_FILE
from src.models.mask_geometry import MaskGeometryTokenizer
from src.models.transformer_qwenimage import QwenImageTransformer2DModel
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    QwenImageEditInsertAnythingTrainer,
)
from src.utils.inference_metadata import validate_checkpoint_artifacts


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_score_conversion_and_dmd_gradient_sign():
    z = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn_like(z, requires_grad=True)
    fake = torch.randn_like(z, requires_grad=True)
    tau = torch.tensor([0.2, 0.7])
    noisy = add_noise(z.detach(), torch.randn_like(z), tau)
    loss, direction = dmd_surrogate(z, noisy, teacher, fake, tau, normalize=False)
    loss.backward()
    score_gap = velocity_to_score(noisy, fake.detach(), tau) - velocity_to_score(
        noisy, teacher.detach(), tau
    )
    expected = score_gap * (tau.square() / (1 - tau))[:, None, None]
    torch.testing.assert_close(direction, expected)
    torch.testing.assert_close(z.grad, expected / z.numel())
    assert teacher.grad is None and fake.grad is None
    with pytest.raises(ValueError, match="interior"):
        dmd_surrogate(z, noisy, teacher, fake, 0)


def test_meanflow_finite_difference_has_no_teacher_or_jvp_gradient():
    a = torch.nn.Parameter(torch.tensor(0.3))
    clean = torch.randn(2, 3, 4, requires_grad=True)
    noise = torch.randn_like(clean)
    t, r = torch.tensor([0.6, 0.4]), torch.tensor([0.2, 0.4])
    calls = []

    def predict(x, start, end):
        calls.append(torch.is_grad_enabled())
        return a * x + start[:, None, None]

    loss = meanflow_loss(predict, clean, noise, t, r, epsilon=0.001)
    x = add_noise(clean.detach(), noise, t)
    v = noise - clean.detach()
    target = v - (t - r)[:, None, None] * (a.detach() * v + 1)
    expected_grad = (2 * (a.detach() * x + t[:, None, None] - target) * x).mean()
    loss.backward()
    torch.testing.assert_close(a.grad, expected_grad, atol=2e-4, rtol=2e-4)
    assert clean.grad is None
    assert calls == [False, False, True]


def tiny_model():
    model = QwenImageTransformer2DModel(
        in_channels=8,
        out_channels=2,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )
    model.mask_geometry_tokenizer = MaskGeometryTokenizer(4, 8)
    return model


class TinyAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0
    is_main_process = True
    mixed_precision = "no"
    sync_gradients = True

    def __init__(self, accumulation=1):
        self.accumulation = accumulation

    def backward(self, loss):
        (loss / self.accumulation).backward()

    def autocast(self):
        return nullcontext()

    def unwrap_model(self, model):
        return model


class TinyTrainer(QwenImageEditInsertAnythingTrainer):
    """Real Qwen transformer/PEFT/backend; replace only frozen image encoders."""

    def __init__(self, config):
        self.config = config
        self.adapter_name = config.model.lora.adapter_name
        self.weight_dtype = torch.float32
        self.accelerator = TinyAccelerator(config.train.gradient_accumulation_steps)
        self.global_step = 0
        self.num_channels_latents = 2
        self.vae_scale_factor = 8
        torch.manual_seed(123)
        self.dit = tiny_model()
        self.dit.requires_grad_(False)
        self.load_pretrain_lora_model(self.dit, config, self.adapter_name)
        self.dit.flow_map_time_embedding = FlowMapTimeEmbedding(
            self.dit.time_text_embed
        )
        self.load_additional_modules(stage="fit")
        self.dit.train()
        self.dit.enable_gradient_checkpointing()
        from diffusers import QwenImageEditPlusPipeline

        self.pipeline_class = QwenImageEditPlusPipeline

    def _unwrap_dit(self):
        return self.dit

    def _dit_device(self):
        return torch.device("cpu")

    def _dit_config(self):
        return self.dit.config

    def _get_image_shapes(self, batch, batch_size):
        return [[(1, 2, 2), (1, 2, 2), (1, 2, 2)] for _ in range(batch_size)]

    def prepare_initial_latents(self, batch_size, **kwargs):
        return torch.randn(batch_size, 4, 8)

    def prepare_predict_timesteps(self, steps, length):
        self.scheduler = SimpleNamespace(sigmas=torch.linspace(1, 0, steps + 1))
        return self.scheduler.sigmas[:-1] * 1000, steps

    def prepare_opd_batch(self, batch):
        return batch

    def prepare_embeddings(self, batch, stage="opd"):
        assert stage == "opd"
        return dict(batch["condition"])

    def conditioning_context(self):
        return nullcontext()


def make_case(tmp_path, stage="warmup", *, resume=None, checkpoint=None):
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir(exist_ok=True)
    teacher_weight = teacher_dir / "model.safetensors"
    if not teacher_weight.exists():
        torch.manual_seed(123)
        teacher = tiny_model()
        teacher.add_adapter(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v"]),
            adapter_name="original",
        )
        for name, p in teacher.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(p, std=0.1)
        safetensors.torch.save_file(
            get_peft_model_state_dict(teacher, adapter_name="original"),
            str(teacher_weight),
        )
        safetensors.torch.save_file(
            teacher.mask_geometry_tokenizer.state_dict(),
            str(teacher_dir / "mask_geometry_tokenizer.safetensors"),
        )
    raw = {
        "mode": "fit",
        "trainer": "QwenImageEditInsertAnything",
        "model": {
            "pretrained_model_name_or_path": str(tmp_path / "base"),
            "attention_backend": "sdpa",
            "zero_cond_t": True,
            "mask_control": {
                "enabled": True,
                "encoding": "geometry_tokenizer",
                "tokenizer_hidden_channels": 4,
            },
            "reference_cfg": {"enabled": False, "force_empty_prompt": True},
            "lora": {
                "r": 2,
                "lora_alpha": 2,
                "target_modules": ["to_q", "to_k", "to_v"],
                "adapter_name": "generator",
            },
        },
        "train": {"mixed_precision": "no", "gradient_checkpointing": True},
        "predict": {"devices": {"vae": "cpu", "dit": "cpu", "text_encoder": "cpu"}},
        "loss": {"mask_loss": False, "region_perceptual": False},
    }
    teacher_cfg = Config.model_validate(raw)
    if not (teacher_dir / "train_config.yaml").exists():
        (teacher_dir / "train_config.yaml").write_text(
            yaml.safe_dump(teacher_cfg.model_dump(mode="json"))
        )
    raw["model"]["flow_map"] = {"enabled": True}
    raw["model"]["lora"]["pretrained_weight"] = str(checkpoint or teacher_weight)
    raw["resume"] = str(resume) if resume else None
    raw["distillation"] = {
        "enabled": True,
        "balance_tasks": False,
        "algorithm": {
            "class_path": "src.distillation.meanflow.QwenImageEditMeanFlowDMDStrategy"
        },
        "student_guidance": {"mode": "none", "scale": 1, "prompt_mode": "empty"},
        "fallback_teacher": "fused",
        "teachers": [
            {
                "name": "fused",
                "route_categories": ["accessory"],
                "guidance": {"mode": "none", "scale": 1, "prompt_mode": "empty"},
                "lora": dict(
                    raw["model"]["lora"],
                    adapter_name="fused_teacher",
                    pretrained_weight=str(teacher_weight),
                ),
            }
        ],
        "meanflow": {
            "stage": stage,
            "teacher_steps": 2,
            "student_steps": 2,
            "fake_updates": 2,
            "anchoring": "disabled",
        },
    }
    config = Config.model_validate(raw)
    if stage == "dmd" and not checkpoint:
        # A valid fresh endpoint module is the warmup fixture; production CLI
        # requires the trained warmup checkpoint and checks metadata/artifacts.
        torch.manual_seed(123)
        embed = FlowMapTimeEmbedding(tiny_model().time_text_embed)
        safetensors.torch.save_file(
            embed.state_dict(), str(teacher_dir / FLOW_MAP_FILE)
        )
    trainer = TinyTrainer(config)
    strategy = QwenImageEditMeanFlowDMDStrategy(config.distillation)
    trainer.training_strategy = strategy
    strategy.setup(trainer)
    strategy.after_prepare(trainer)
    trainer.optimizer = torch.optim.AdamW(
        strategy.generator_parameters(trainer), lr=1e-3
    )
    generator = torch.Generator().manual_seed(8)
    condition = {
        "control_latents": torch.randn(1, 8, 8, generator=generator),
        "prompt_embeds": torch.randn(1, 3, 8, generator=generator),
        "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
        "mask_geometry": torch.randn(1, 3, 32, 32, generator=generator),
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "true_cfg_scale": 1.0,
    }
    batch = {
        "condition": condition,
        "category": ["accessory"],
        "prompt": [""],
        "tar_mask": torch.ones(1, 32, 32),
    }
    return trainer, strategy, batch


@pytest.mark.parametrize("stage", ["warmup", "dmd"])
def test_real_qwen_updates_correct_roles_and_preserves_teacher(tmp_path, stage):
    trainer, strategy, batch = make_case(tmp_path, stage)
    before = {n: p.detach().clone() for n, p in trainer.dit.named_parameters()}
    loss = strategy.backward_step(trainer, batch)
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in strategy.generator_parameters(trainer)
    )
    assert all(p.grad is None for p in strategy.fake_parameters(trainer))
    trainer.optimizer.step()
    changes = [
        n for n, p in trainer.dit.named_parameters() if not torch.equal(before[n], p)
    ]
    assert any(is_adapter_parameter(n, "generator") for n in changes)
    assert not any(is_adapter_parameter(n, "fused_teacher") for n in changes)
    assert all(
        "lora_" in n or "flow_map_time_embedding" in n or "mask_geometry_tokenizer" in n
        for n in changes
    )
    assert not any(
        p.requires_grad
        for n, p in trainer.dit.named_parameters()
        if is_adapter_parameter(n, "fused_teacher")
    )
    if stage == "dmd":
        assert any(is_adapter_parameter(n, strategy.fake_adapter) for n in changes)
        assert strategy.fake_update_count == 2
        assert trainer._step_loss_metrics["meanflow/dmd_gradient_rms"] > 0


def test_checkpoint_recomputation_keeps_generator_adapter(tmp_path):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    _, condition, frozen = strategy._prepare(trainer, batch)
    x = torch.randn(1, 4, 8)
    reference = forward_map(trainer, x, 0.6, 0.2, condition).square().mean()
    reference.backward()
    expected = {
        n: p.grad.clone()
        for n, p in trainer.dit.named_parameters()
        if p.grad is not None
    }
    trainer.dit.zero_grad(set_to_none=True)
    live = forward_map(trainer, x, 0.6, 0.2, condition).square().mean()
    with torch.no_grad(), strategy.role("teacher"):
        strategy._instantaneous(x, torch.tensor(0.6), frozen)
    with torch.no_grad(), strategy.role("fake"):
        strategy._instantaneous(x, torch.tensor(0.6), frozen)
    live.backward()
    for n, p in trainer.dit.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n])


def test_optional_region_keep_preserves_gradient_isolation(tmp_path):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    strategy.options.keep_weight = 0.1
    loss = strategy.backward_step(trainer, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(trainer._step_loss_metrics["meanflow/keep_loss"])
    assert all(p.grad is None for p in strategy.fake_parameters(trainer))
    assert any(p.grad is not None for p in strategy.generator_parameters(trainer))


def test_training_anchor_uses_masked_source_and_accepts_independent_source(tmp_path):
    trainer, strategy, batch = make_case(tmp_path)
    strategy.options.anchoring = "deployment"
    trainer.config.predict.latent_anchoring.mode = "expanded_soft"
    trainer.config.predict.latent_anchoring.interaction_dilation_px = 2
    trainer.config.predict.latent_anchoring.transition_width_px = 2
    mask = torch.zeros(1, 32, 32)
    mask[:, 12:20, 12:20] = 1
    batch.update(tar_mask=mask, tar_mask_cond=mask, tar_image=torch.ones(1, 3, 32, 32))
    encoded = []
    trainer.preprocess_image_for_vae_encoder = lambda image, **kwargs: image

    def encode(image, *args):
        encoded.append(image.clone())
        return torch.zeros(1, 4, 8)

    trainer.encode_image_latents = encode
    _, condition, frozen = strategy._prepare(trainer, batch)
    assert encoded == []  # Reuse the already encoded, masked source control.
    torch.testing.assert_close(
        condition["anchor_source_latents"], batch["condition"]["control_latents"][:, :4]
    )
    torch.testing.assert_close(
        condition["anchor_edit_weight"], frozen["anchor_edit_weight"]
    )
    clean = torch.full_like(batch["tar_image"], 0.2)
    batch["source_image"] = clean
    prepared, _, _ = strategy._prepare(trainer, batch)
    torch.testing.assert_close(prepared["tar_image"], clean)
    assert len(encoded) == 1
    torch.testing.assert_close(encoded[-1], clean)


def test_deployment_sampler_preserves_hard_background_and_changes_with_reference(
    tmp_path,
):
    trainer, strategy, batch = make_case(tmp_path)
    condition = batch["condition"]
    source = torch.randn(1, 4, 8)
    mask = torch.ones_like(source)
    mask[:, :2] = 0
    condition = dict(condition, anchor_source_latents=source, anchor_edit_weight=mask)
    torch.manual_seed(44)
    out = sample_flow_map(trainer, condition)
    torch.testing.assert_close(out[:, :2], source[:, :2])
    altered = dict(condition, control_latents=condition["control_latents"] * -2)
    torch.manual_seed(44)
    other = sample_flow_map(trainer, altered)
    assert not torch.allclose(out[:, 2:], other[:, 2:])


@pytest.mark.parametrize(
    "rollout,grad", [("full", "full"), ("full", "last"), ("shortcut", "full")]
)
def test_rollout_gradient_modes_keep_live_output(tmp_path, rollout, grad):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    strategy.options.rollout = rollout
    strategy.options.gradient_mode = grad
    _, condition, _ = strategy._prepare(trainer, batch)
    generated = strategy._generate(condition, with_grad=True)
    assert generated.requires_grad
    generated.square().mean().backward()
    assert any(p.grad is not None for p in strategy.generator_parameters(trainer))
    assert all(p.grad is None for p in strategy.fake_parameters(trainer))


def test_resume_fake_optimizer_and_generator_exactly(tmp_path):
    trainer, strategy, batch = make_case(tmp_path, "dmd")
    strategy.backward_step(trainer, batch)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    trainer.save_lora(str(folder))
    trainer.save_additional_modules(str(folder))
    strategy.save_state(trainer, folder)
    generator_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    rng = torch.get_rng_state()
    state_before = {n: p.detach().clone() for n, p in trainer.dit.named_parameters()}
    strategy.backward_step(trainer, batch)
    trainer.optimizer.step()
    expected = {n: p.detach().clone() for n, p in trainer.dit.named_parameters()}
    other, restored, same_batch = make_case(
        tmp_path, "dmd", resume=folder, checkpoint=folder / "model.safetensors"
    )
    restored.restore_state(other, folder)
    other.optimizer.load_state_dict(generator_optimizer)
    for n, p in other.dit.named_parameters():
        torch.testing.assert_close(p, state_before[n])
    torch.set_rng_state(rng)
    restored.backward_step(other, same_batch)
    other.optimizer.step()
    for n, p in other.dit.named_parameters():
        torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
    assert restored.fake_update_count == strategy.fake_update_count


def test_flow_companion_required_and_inference_drops_teacher_dependencies(tmp_path):
    trainer, strategy, _ = make_case(tmp_path)
    folder = tmp_path / "ckpt"
    folder.mkdir()
    trainer.save_lora(str(folder))
    trainer.save_additional_modules(str(folder))
    strategy.save_state(trainer, folder)
    (folder / "state.json").write_text(
        __import__("json").dumps(
            {
                "artifacts": trainer._build_checkpoint_artifact_manifest(str(folder)),
            }
        )
    )
    cfg = trainer.config.model_copy(deep=True)
    cfg.distillation.enabled = False
    check = validate_checkpoint_artifacts(
        cfg, folder / "model.safetensors", allow_mismatch=False
    )
    assert check["verified"]
    (folder / FLOW_MAP_FILE).write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        validate_checkpoint_artifacts(
            cfg, folder / "model.safetensors", allow_mismatch=False
        )
    raw = trainer.config.model_dump(mode="json")
    raw["distillation"]["teachers"][0]["lora"]["pretrained_weight"] = (
        "/missing/teacher/model.safetensors"
    )
    path = tmp_path / "inference.yaml"
    path.write_text(yaml.safe_dump(raw))
    parsed = load_config_from_yaml(str(path), for_inference=True)
    assert parsed.model.flow_map.enabled and not parsed.distillation.enabled


def test_config_builder_and_cli_defaults(tmp_path):
    from scripts.build_meanflow_config import build_config
    from infer import apply_flow_map_defaults, parse_args

    trainer, strategy, _ = make_case(tmp_path)
    warm = Config.model_validate(build_config(tmp_path / "teacher"))
    assert warm.distillation.meanflow.stage == "warmup"
    assert warm.model.flow_map.enabled
    folder = tmp_path / "warmup"
    folder.mkdir()
    trainer.save_lora(str(folder))
    trainer.save_additional_modules(str(folder))
    strategy.save_state(trainer, folder)
    (folder / "train_config.yaml").write_text(
        yaml.safe_dump(trainer.config.model_dump(mode="json"))
    )
    (folder / "state.json").write_text(
        __import__("json").dumps(
            {
                "artifacts": trainer._build_checkpoint_artifact_manifest(str(folder)),
            }
        )
    )
    dmd = Config.model_validate(
        build_config(tmp_path / "teacher", stage="dmd", generator=folder)
    )
    assert dmd.distillation.meanflow.stage == "dmd"
    args = parse_args(
        ["--src-image", "a.png", "--ref-image", "b.png", "--src-mask", "m.png"]
    )
    apply_flow_map_defaults(dmd, args)
    assert args.num_inference_steps == 8 and args.true_cfg_scale == 1
    args = parse_args(
        [
            "--src-image",
            "a.png",
            "--ref-image",
            "b.png",
            "--src-mask",
            "m.png",
            "--num-inference-steps",
            "4",
        ]
    )
    apply_flow_map_defaults(dmd, args)
    assert args.num_inference_steps == 4
    with pytest.raises(ValueError, match="trained MeanFlow"):
        build_config(tmp_path / "teacher", stage="dmd")


def _distributed_worker(rank, folder, rendezvous):
    import torch.distributed as dist
    from pathlib import Path
    from src.distillation.meanflow_math import synchronize_gradients

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method="file://" + rendezvous, rank=rank, world_size=2
    )
    try:
        unused = torch.nn.Parameter(torch.ones(2))
        partial = torch.nn.Parameter(torch.ones(2))
        partial.grad = torch.ones(2) if rank == 0 else None
        synchronize_gradients([unused, partial])
        assert unused.grad is None
        torch.testing.assert_close(partial.grad, torch.full((2,), 0.5))
        trainer, strategy, batch = make_case(Path(folder), "dmd")
        trainer.accelerator.num_processes = 2
        trainer.accelerator.process_index = rank
        batch["condition"]["control_latents"] += rank * 0.3
        torch.manual_seed(99 + rank)
        strategy.backward_step(trainer, batch)
        synchronize_gradients(strategy.generator_parameters(trainer))
        trainer.optimizer.step()
        for params in [
            strategy.generator_parameters(trainer),
            strategy.fake_parameters(trainer),
        ]:
            flat = torch.cat([p.detach().flatten() for p in params])
            replicas = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(replicas, flat)
            torch.testing.assert_close(replicas[0], replicas[1], rtol=0, atol=0)
        assert all(
            p.grad is None
            for n, p in trainer.dit.named_parameters()
            if is_adapter_parameter(n, "fused_teacher")
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_generator_and_fake_updates_are_synchronized(tmp_path):
    import torch.multiprocessing as mp

    make_case(tmp_path, "dmd")  # Prepare files once before concurrent readers.
    mp.spawn(
        _distributed_worker,
        args=(str(tmp_path), str(tmp_path / "gloo")),
        nprocs=2,
        join=True,
    )


def test_gradient_accumulation_does_not_rescale_fake_updates(tmp_path):
    (tmp_path / "one").mkdir()
    one, s1, batch1 = make_case(tmp_path / "one", "dmd")
    one.config.train.gradient_accumulation_steps = 1
    one.accelerator.accumulation = 1
    (tmp_path / "three").mkdir()
    three, s3, batch3 = make_case(tmp_path / "three", "dmd")
    three.config.train.gradient_accumulation_steps = 3
    three.accelerator.accumulation = 3
    torch.manual_seed(17)
    s1.backward_step(one, batch1)
    torch.manual_seed(17)
    s3.backward_step(three, batch3)
    for a, b in zip(s1.fake_parameters(one), s3.fake_parameters(three)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(s1.generator_parameters(one), s3.generator_parameters(three)):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad * 3, rtol=2e-4, atol=1e-6)


class FitTrainer(TinyTrainer):
    """Run the real BaseTrainer lifecycle, using synthetic frozen conditions."""

    def __init__(self, config):
        from src.trainer.base_trainer import BaseTrainer

        BaseTrainer.__init__(self, config)
        self.num_channels_latents = 2
        self.vae_scale_factor = 8

    def load_preprocessor(self):
        pass

    def setup_signal_handlers(self):
        pass

    def load_model(self, text_encoder_device=None):
        torch.manual_seed(123)
        self.dit = tiny_model().to(dtype=self.weight_dtype)
        self.dit.flow_map_time_embedding = FlowMapTimeEmbedding(
            self.dit.time_text_embed
        )

    def setup_model_device_train_mode(self, stage="fit"):
        self.dit.requires_grad_(False)
        self.dit.train()

    def setup_criterion(self):
        pass

    def setup_validation(self, train_dataloader):
        self.validation_sampler = None


@pytest.fixture
def clean_accelerator_state():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()
    yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.mark.parametrize("precision", ["no", "bf16"])
def test_real_trainer_lifecycle_saves_both_optimizers(
    tmp_path, precision, clean_accelerator_state
):
    from torch.utils.data import DataLoader

    _, _, batch = make_case(tmp_path, "dmd")
    prototype, _, _ = make_case(tmp_path, "dmd")
    cfg = prototype.config.model_copy(deep=True)
    cfg.train.gradient_accumulation_steps = 2
    cfg.train.mixed_precision = precision
    cfg.distillation.meanflow.teacher_score_probe_interval = 1
    cfg.train.max_train_steps = 2
    cfg.train.num_epochs = 1
    cfg.train.checkpointing_steps = 1
    cfg.logging.output_dir = str(tmp_path / "fit")
    cfg.logging.tracker_project_name = "meanflow_lifecycle"
    cfg.data.num_workers = 0
    cfg.data.drop_last = False
    cfg.logging.sampling.enable = False
    trainer = FitTrainer(cfg)
    trainer.fit(
        DataLoader([batch, batch, batch], batch_size=1, collate_fn=lambda rows: rows[0])
    )
    checkpoints = list(
        __import__("pathlib")
        .Path(trainer.config.logging.output_dir)
        .glob("checkpoint-last-*")
    )
    assert len(checkpoints) == 1
    folder = checkpoints[0]
    for name in [
        "model.safetensors",
        FLOW_MAP_FILE,
        "fake_score_lora.safetensors",
        "fake_score_optimizer.pt",
        "optimizer.bin",
        "meanflow_training_state.json",
    ]:
        assert (folder / name).is_file(), name
    assert trainer.global_step == 2

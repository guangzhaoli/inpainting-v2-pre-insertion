from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.distillation.opd import QwenImageEditOPDStrategy
from src.distillation.types import FlowTrajectory


class FakeBackend:
    def __init__(self):
        self.trainer = None
        self.routed_batches = []

    def setup(self, trainer, teachers):
        self.trainer = trainer

    def activate_student(self):
        self.trainer.dit.active_policy = "student"

    @contextmanager
    def teacher(self, teacher_name):
        self.trainer.dit.active_policy = teacher_name
        try:
            yield
        finally:
            self.activate_student()

    def assert_frozen(self):
        return None

    def predict_velocity(
        self,
        teacher,
        *,
        trainer,
        state,
        timestep,
        condition,
        batch,
        guidance,
        shared_predictor,
    ):
        assert not torch.is_grad_enabled()
        self.routed_batches.append((teacher.name, tuple(batch["category"])))
        with self.teacher(teacher.name):
            return shared_predictor(
                trainer,
                state,
                timestep,
                condition,
                guidance,
            )


class DistributedFakeBackend(FakeBackend):
    uses_student_distributed_model = True


class FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0

    def __init__(self):
        self.no_sync_depth = 0
        self.no_sync_models = []
        self.backward_sync_states = []

    @contextmanager
    def no_sync(self, model):
        self.no_sync_models.append(model)
        self.no_sync_depth += 1
        try:
            yield
        finally:
            self.no_sync_depth -= 1

    def backward(self, loss):
        self.backward_sync_states.append(self.no_sync_depth == 0)
        loss.backward()

    @staticmethod
    def reduce(value, reduction="sum"):
        assert reduction == "sum"
        return value


class FakeDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.student_velocity = nn.Parameter(torch.tensor(0.0))
        self.active_policy = "student"
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=False)
        self.accelerator = None
        self.student_forward_sync_states = []

    def forward(self, hidden_states, **kwargs):
        values = {
            "teacher_animal": 1.0,
            "teacher_object": 2.0,
        }
        if self.active_policy == "student":
            velocity = self.student_velocity
            if torch.is_grad_enabled() and self.accelerator is not None:
                self.student_forward_sync_states.append(
                    self.accelerator.no_sync_depth == 0
                )
        else:
            velocity = hidden_states.new_tensor(values[self.active_policy])
        return (torch.ones_like(hidden_states) * velocity,)


class FakeScheduler:
    def __init__(self):
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])


class FakeTrainer:
    def __init__(self):
        self.dit = FakeDit()
        self.scheduler = FakeScheduler()
        self.weight_dtype = torch.float32
        self.global_step = 0
        self.accelerator = FakeAccelerator()
        self.dit.accelerator = self.accelerator
        self.opd_batch_prepared = False

    def _dit_device(self):
        return torch.device("cpu")

    def _dit_config(self):
        return self.dit.config

    def prepare_opd_batch(self, batch):
        self.opd_batch_prepared = True
        batch["shared_condition"] = torch.arange(len(batch["category"]))
        return batch

    def prepare_embeddings(self, batch, stage):
        assert stage == "opd"
        assert self.opd_batch_prepared
        assert "shared_condition" in batch
        batch_size = len(batch["category"])
        return {
            "control_latents": torch.zeros(batch_size, 1, 3),
            "prompt_embeds": torch.zeros(batch_size, 1, 3),
            "prompt_embeds_mask": torch.ones(batch_size, 1, dtype=torch.long),
            "height": 16,
            "width": 16,
            "height_control": 16,
            "width_control": 16,
            "n_controls": 0,
            "guidance": 1.0,
        }

    def _get_image_shapes(self, condition, batch_size):
        return [[(1, 1, 1), (1, 1, 1)] for _ in range(batch_size)]

    def prepare_initial_latents(self, batch_size, **kwargs):
        return torch.zeros(batch_size, 2, 3)

    def prepare_predict_timesteps(self, num_inference_steps, image_seq_len):
        assert num_inference_steps == 2
        assert image_seq_len == 2
        self.scheduler.sigmas = torch.tensor([1.0, 0.5, 0.0])
        return torch.tensor([1000.0, 500.0]), 2


def _plugin(path, **init_args):
    return SimpleNamespace(class_path=path, init_args=init_args)


def _guidance(mode="none", scale=1.0):
    return SimpleNamespace(mode=mode, scale=scale, prompt_mode="batch")


def _teacher(name, category, weight=1.0):
    return SimpleNamespace(
        name=name,
        route_categories=[category],
        weight=weight,
        guidance=_guidance(),
    )


def test_opd_strategy_routes_mixed_batch_and_micro_backpropagates():
    teachers = [
        _teacher("teacher_animal", "cat"),
        _teacher("teacher_object", "object"),
    ]
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.FakeBackend"),
        teachers=teachers,
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    strategy = QwenImageEditOPDStrategy(config)
    strategy.setup(trainer)

    loss = strategy.backward_step(
        trainer,
        {
            "category": ["cat", "object"],
            "prompt": ["cat", "object"],
        },
    )

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert trainer.dit.student_velocity.grad is not None
    assert trainer.dit.student_velocity.grad.item() == pytest.approx(-1.5)
    assert trainer.dit.active_policy == "student"
    assert trainer.opd_batch_prepared
    assert strategy.backend.routed_batches == [
        ("teacher_animal", ("cat",)),
        ("teacher_animal", ("cat",)),
        ("teacher_object", ("object",)),
        ("teacher_object", ("object",)),
    ]
    assert trainer.accelerator.no_sync_models == [trainer.dit]
    assert trainer.dit.student_forward_sync_states == [False, True]
    assert trainer.accelerator.backward_sync_states == [False, True]
    assert trainer.accelerator.no_sync_depth == 0

    metrics = strategy.consume_metrics(trainer.accelerator)
    assert metrics["opd/rollout_steps_per_batch"] == 2.0
    assert metrics["opd/trained_steps_per_batch"] == 2.0
    assert metrics["opd/teacher/teacher_animal/route_fraction"] == 0.5
    assert metrics["opd/teacher/teacher_object/route_fraction"] == 0.5


def test_opd_timestep_no_sync_nests_with_outer_accumulation_no_sync():
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.FakeBackend"),
        teachers=[_teacher("teacher_animal", "cat")],
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    strategy = QwenImageEditOPDStrategy(config)
    strategy.setup(trainer)

    with trainer.accelerator.no_sync(trainer.dit):
        strategy.backward_step(
            trainer,
            {"category": ["cat"], "prompt": ["cat"]},
        )

    assert trainer.dit.student_forward_sync_states == [False, False]
    assert trainer.accelerator.backward_sync_states == [False, False]
    assert trainer.accelerator.no_sync_models == [trainer.dit, trainer.dit]
    assert trainer.accelerator.no_sync_depth == 0
    assert trainer.dit.student_velocity.grad.item() == pytest.approx(-1.0)


def test_opd_single_selected_timestep_does_not_enter_internal_no_sync():
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.FakeBackend"),
        teachers=[_teacher("teacher_animal", "cat")],
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=0.5,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    strategy = QwenImageEditOPDStrategy(config)
    strategy.setup(trainer)

    strategy.backward_step(
        trainer,
        {"category": ["cat"], "prompt": ["cat"]},
    )

    assert trainer.accelerator.no_sync_models == []
    assert trainer.dit.student_forward_sync_states == [True]
    assert trainer.accelerator.backward_sync_states == [True]
    assert trainer.accelerator.no_sync_depth == 0
    assert trainer.dit.student_velocity.grad.item() == pytest.approx(-1.0)


def test_distributed_shared_backend_requires_local_batch_size_one():
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.DistributedFakeBackend"),
        teachers=[_teacher("teacher_animal", "cat")],
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    trainer.accelerator.num_processes = 2
    trainer.config = SimpleNamespace(data=SimpleNamespace(batch_size=2))

    with pytest.raises(ValueError, match="batch_size=1"):
        QwenImageEditOPDStrategy(config).setup(trainer)


def test_distributed_shared_backend_requires_equal_guidance_forward_counts():
    teacher_unguided = _teacher("teacher_animal", "cat")
    teacher_guided = _teacher("teacher_object", "object")
    teacher_guided.guidance = _guidance(mode="reference_cfg", scale=4.0)
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.DistributedFakeBackend"),
        teachers=[teacher_unguided, teacher_guided],
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    trainer.accelerator.num_processes = 2
    trainer.config = SimpleNamespace(data=SimpleNamespace(batch_size=1))

    with pytest.raises(ValueError, match="all guided or all unguided"):
        QwenImageEditOPDStrategy(config).setup(trainer)


def test_condition_cache_key_includes_guidance_scale():
    student = _guidance(mode="reference_cfg", scale=1.0)
    teacher = _teacher("teacher_animal", "cat")
    teacher.guidance = _guidance(mode="reference_cfg", scale=4.0)
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.FakeBackend"),
        teachers=[teacher],
        fallback_teacher=None,
        student_guidance=student,
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    strategy = QwenImageEditOPDStrategy(config)
    policies = strategy._required_condition_policies()
    assert len(policies) == 2
    assert strategy._condition_key(student) != strategy._condition_key(teacher.guidance)


def test_opd_routes_per_sample_timesteps_with_teacher_groups():
    teachers = [
        _teacher("teacher_animal", "cat"),
        _teacher("teacher_object", "object"),
    ]
    config = SimpleNamespace(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("tests.test_opd_strategy.FakeBackend"),
        teachers=teachers,
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=SimpleNamespace(
            num_inference_steps=2,
            timestep_fraction=1.0,
            store_device="cpu",
            seed=7,
        ),
    )
    trainer = FakeTrainer()
    strategy = QwenImageEditOPDStrategy(config)
    strategy.setup(trainer)

    # A custom rollout may use a timestep per sample.  The two local Teacher
    # groups must receive [1] values rather than the full [B] vector.
    trajectory = FlowTrajectory(
        states=torch.zeros(2, 3, 2, 3),
        timesteps=torch.tensor([[1000.0, 500.0], [900.0, 400.0]]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    strategy._rollout = lambda _trainer, _condition: trajectory
    loss = strategy.backward_step(
        trainer,
        {"category": ["cat", "object"], "prompt": ["cat", "object"]},
    )

    assert loss.ndim == 0
    assert trainer.dit.student_velocity.grad is not None


class _GeometryDit(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_scale = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(in_channels=4, guidance_embeds=False)
        self.seen_geometry = None

    def forward(self, hidden_states, *, mask_geometry=None, **kwargs):
        del kwargs
        self.seen_geometry = mask_geometry
        value = self.geometry_scale * mask_geometry.mean()
        return (torch.ones_like(hidden_states) * value,)


def test_default_opd_forward_passes_geometry_and_keeps_student_gradient():
    trainer = FakeTrainer()
    trainer.dit = _GeometryDit()
    strategy = object.__new__(QwenImageEditOPDStrategy)
    state = torch.zeros(1, 2, 3)
    condition = {
        "control_latents": torch.zeros(1, 1, 3),
        "prompt_embeds": torch.zeros(1, 1, 3),
        "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
        "height": 16,
        "width": 16,
        "mask_geometry": torch.ones(1, 3, 2, 2),
        "guidance": 1.0,
    }
    velocity = strategy._predict_raw_velocity(
        trainer,
        state,
        torch.tensor(500.0),
        condition,
    )
    velocity.sum().backward()

    assert trainer.dit.seen_geometry is not None
    assert trainer.dit.geometry_scale.grad is not None

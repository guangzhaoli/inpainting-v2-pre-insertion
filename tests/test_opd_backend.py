from types import SimpleNamespace

from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
import pytest
import safetensors.torch
import torch
from torch import nn

from src.distillation.backends import (
    IndependentTeacherBackend,
    SharedLoRATeacherBackend,
    set_active_adapter,
)
from src.models.mask_geometry import MaskGeometryTokenizer
from src.trainer.base_trainer import BaseTrainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, value):
        return self.proj(value)


class _IndependentPolicy:
    def __init__(self, velocity):
        self.velocity = float(velocity)
        self.model = nn.Linear(1, 1)
        self.seen_teacher = None
        self.seen_category = None

    def setup(self, trainer, teacher):
        self.seen_teacher = teacher.name

    def predict_velocity(self, *, state, timestep, condition, batch, guidance):
        del timestep, condition, guidance
        self.seen_category = tuple(batch["category"])
        return torch.full_like(state, self.velocity)


def test_set_active_adapter_trains_only_exact_student_adapter():
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    model = get_peft_model(_TinyModel(), config, adapter_name="student")
    model.add_adapter("teacher", config)

    set_active_adapter(model, "student", trainable_adapter="student")
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(".student." in name for name in trainable)

    set_active_adapter(model, "teacher", trainable_adapter=None)
    assert not any(param.requires_grad for param in model.parameters())

    set_active_adapter(model, "student", trainable_adapter="student")
    restored = [name for name, param in model.named_parameters() if param.requires_grad]
    assert restored == trainable


def test_set_active_adapter_rejects_missing_trainable_adapter():
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    model = get_peft_model(_TinyModel(), config, adapter_name="student")

    try:
        set_active_adapter(model, "student", trainable_adapter="missing")
    except ValueError as exc:
        assert "No LoRA parameters matched" in str(exc)
    else:
        raise AssertionError("expected a missing-adapter error")


def test_independent_backend_loads_freezes_and_routes_policy():
    config = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    student = get_peft_model(_TinyModel(), config, adapter_name="student")
    trainer = SimpleNamespace(
        dit=student,
        adapter_name="student",
    )
    teacher = SimpleNamespace(
        name="external",
        policy=SimpleNamespace(
            class_path="tests.test_opd_backend._IndependentPolicy",
            init_args={"velocity": 3.0},
        ),
    )
    backend = IndependentTeacherBackend()

    backend.setup(trainer, [teacher])
    with torch.no_grad():
        velocity = backend.predict_velocity(
            teacher,
            trainer=trainer,
            state=torch.zeros(1, 2, 3),
            timestep=torch.tensor(500.0),
            condition={},
            batch={"category": ["cat"]},
            guidance=SimpleNamespace(mode="none", scale=1.0),
            shared_predictor=lambda *_args: None,
        )

    policy = backend._policies["external"]
    assert policy.seen_teacher == "external"
    assert policy.seen_category == ("cat",)
    assert torch.equal(velocity, torch.full((1, 2, 3), 3.0))
    assert not any(parameter.requires_grad for parameter in policy.model.parameters())
    assert all(".student." in name for name, parameter in student.named_parameters() if parameter.requires_grad)


def test_loading_multiple_peft_adapters_ignores_unrelated_missing_keys():
    """A shared backbone reports prior adapters as missing under strict=False."""

    config = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    model = get_peft_model(_TinyModel(), config, adapter_name="student")
    model.add_adapter("teacher_a", config)
    model.add_adapter("teacher_b", config)
    state = get_peft_model_state_dict(model, adapter_name="student")

    # The loader must validate teacher_a's keys without treating student and
    # teacher_b keys in the same model as missing teacher_a weights.
    BaseTrainer._load_peft_lora_state_dict(
        transformer=model,
        state_dict=state,
        adapter_name="teacher_a",
        stage="teacher",
    )
    BaseTrainer._load_peft_lora_state_dict(
        transformer=model,
        state_dict=state,
        adapter_name="teacher_b",
        stage="teacher",
    )


class _GeometryTrainer:
    def __init__(self, tokenizer):
        self.dit = nn.Module()
        self.dit.mask_geometry_tokenizer = tokenizer
        self.weight_dtype = torch.float32

    def _get_mask_geometry_tokenizer(self):
        return self.dit.mask_geometry_tokenizer

    def _dit_device(self):
        return torch.device("cpu")

    def _get_image_shapes(self, condition, batch_size):
        del condition
        return [[(1, 2, 2)] for _ in range(batch_size)]


def test_shared_backend_loads_and_uses_teacher_geometry_companion(tmp_path):
    student_tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=3)
    teacher_tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=3)
    with torch.no_grad():
        teacher_tokenizer.conv_out.bias.fill_(0.25)
    companion = tmp_path / "mask_geometry_tokenizer.safetensors"
    safetensors.torch.save_file(teacher_tokenizer.state_dict(), str(companion))

    trainer = _GeometryTrainer(student_tokenizer)
    teacher = SimpleNamespace(
        name="animal",
        mask_geometry_pretrained_weight=str(companion),
        lora=None,
    )
    backend = SharedLoRATeacherBackend()
    backend._trainer = trainer
    backend._load_teacher_geometry_tokenizer(trainer, teacher)

    condition = {
        "height": 16,
        "width": 16,
        "n_controls": 0,
        "mask_geometry": torch.zeros(1, 3, 8, 8),
        "reference_cfg_embeddings": {
            "height": 16,
            "width": 16,
            "n_controls": 0,
            "mask_geometry": torch.zeros(1, 3, 8, 8),
        },
    }
    prepared = backend._tokenize_teacher_geometry("animal", trainer, condition)

    assert "mask_geometry" not in prepared
    assert prepared["mask_geometry_tokens"].shape == (1, 4, 3)
    assert torch.allclose(prepared["mask_geometry_tokens"], torch.full((1, 4, 3), 0.25))
    assert "mask_geometry" not in prepared["reference_cfg_embeddings"]
    assert not any(parameter.requires_grad for parameter in backend._teacher_geometry_tokenizers["animal"].parameters())


def test_shared_backend_rejects_foreign_or_missing_teacher_geometry_condition(tmp_path):
    student_tokenizer = MaskGeometryTokenizer(hidden_channels=4, out_channels=3)
    companion = tmp_path / "mask_geometry_tokenizer.safetensors"
    safetensors.torch.save_file(student_tokenizer.state_dict(), str(companion))

    trainer = _GeometryTrainer(student_tokenizer)
    teacher = SimpleNamespace(
        name="animal",
        mask_geometry_pretrained_weight=str(companion),
        lora=None,
    )
    backend = SharedLoRATeacherBackend()
    backend._trainer = trainer
    backend._load_teacher_geometry_tokenizer(trainer, teacher)

    with pytest.raises(ValueError, match="precomputed mask_geometry_tokens"):
        backend._tokenize_teacher_geometry(
            "animal",
            trainer,
            {"mask_geometry_tokens": torch.zeros(1, 4, 3)},
        )
    with pytest.raises(ValueError, match="missing mask_geometry"):
        backend._tokenize_teacher_geometry("animal", trainer, {})

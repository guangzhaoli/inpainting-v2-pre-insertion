from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.config import (
    Config,
    DistillationConfig,
    LoraConfig,
    OPDGuidanceConfig,
    OPDRolloutConfig,
    OPDTeacherConfig,
    PluginConfig,
    load_config_from_yaml,
)


def _teacher(
    checkpoint: Path,
    *,
    name: str = "subject",
    adapter_name: str = "subject_adapter",
    categories: list[str] | None = None,
) -> OPDTeacherConfig:
    return OPDTeacherConfig(
        name=name,
        route_categories=categories or ["subject"],
        lora=LoraConfig(
            adapter_name=adapter_name,
            pretrained_weight=str(checkpoint),
        ),
    )


def test_distillation_is_backward_compatible_and_plugins_have_defaults():
    config = Config()

    assert config.distillation.enabled is False
    assert config.distillation.teachers == []
    assert config.distillation.algorithm.class_path == "src.distillation.opd.QwenImageEditOPDStrategy"
    assert config.distillation.objective.class_path == "src.distillation.objectives.VelocityMatchingObjective"
    assert config.distillation.teacher_backend.class_path == "src.distillation.backends.SharedLoRATeacherBackend"
    assert config.distillation.student_guidance == OPDGuidanceConfig()
    assert config.distillation.student_guidance.prompt_mode == "batch"
    assert config.distillation.balance_tasks is True


def test_enabled_distillation_accepts_valid_teacher_graph(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()
    teacher = _teacher(checkpoint)

    config = DistillationConfig(
        enabled=True,
        teachers=[teacher],
        fallback_teacher="subject",
        algorithm=PluginConfig(class_path="custom.Strategy", init_args={"temperature": 2.0}),
    )

    assert config.teachers[0].lora.pretrained_weight == str(checkpoint)
    assert config.fallback_teacher == "subject"
    assert config.algorithm.init_args == {"temperature": 2.0}


def test_independent_backend_accepts_policy_without_lora():
    teacher = OPDTeacherConfig(
        name="heterogeneous",
        route_categories=["subject"],
        policy=PluginConfig(class_path="ideas.CustomTeacherPolicy"),
    )

    config = DistillationConfig(
        enabled=True,
        teacher_backend=PluginConfig(class_path="src.distillation.backends.IndependentTeacherBackend"),
        teachers=[teacher],
    )

    assert config.teachers[0].lora is None
    assert config.teachers[0].policy.class_path == "ideas.CustomTeacherPolicy"


def test_independent_backend_requires_teacher_policy():
    teacher = OPDTeacherConfig(name="heterogeneous", route_categories=["subject"])

    with pytest.raises(ValidationError, match="requires policy"):
        DistillationConfig(
            enabled=True,
            teacher_backend=PluginConfig(class_path="src.distillation.backends.IndependentTeacherBackend"),
            teachers=[teacher],
        )


def test_shared_backend_rejects_custom_teacher_policy(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()
    teacher = _teacher(checkpoint)
    teacher.policy = PluginConfig(class_path="ideas.CustomTeacherPolicy")

    with pytest.raises(ValidationError, match="IndependentTeacherBackend"):
        DistillationConfig(enabled=True, teachers=[teacher])


def test_built_in_strategy_requires_insert_anything_trainer(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()

    with pytest.raises(ValidationError, match="QwenImageEditInsertAnything"):
        Config(
            trainer="QwenImageEdit",
            distillation=DistillationConfig(
                enabled=True,
                teachers=[_teacher(checkpoint)],
            ),
        )


def test_shared_backend_rejects_student_teacher_adapter_collision(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()

    with pytest.raises(ValidationError, match="reuse Student adapter"):
        Config(
            trainer="QwenImageEditInsertAnything",
            model={"lora": {"adapter_name": "same"}},
            distillation=DistillationConfig(
                enabled=True,
                teachers=[
                    _teacher(
                        checkpoint,
                        adapter_name="same",
                    )
                ],
            ),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PluginConfig(class_path=" "), "class_path"),
        (lambda: OPDGuidanceConfig(scale=0.99), "scale"),
        (lambda: OPDGuidanceConfig(mode="none", scale=4.0), "requires scale=1"),
        (lambda: OPDRolloutConfig(num_inference_steps=0), "num_inference_steps"),
        (lambda: OPDRolloutConfig(timestep_fraction=0), "timestep_fraction"),
        (lambda: OPDRolloutConfig(timestep_fraction=1.01), "timestep_fraction"),
        (
            lambda: OPDTeacherConfig(
                name="teacher",
                route_categories=[],
                lora=LoraConfig(),
            ),
            "route_categories",
        ),
        (
            lambda: OPDTeacherConfig(
                name="teacher",
                route_categories=["person", "person"],
                lora=LoraConfig(),
            ),
            "route_categories",
        ),
        (
            lambda: OPDTeacherConfig(
                name="teacher",
                route_categories=["person"],
                lora=LoraConfig(),
                weight=0,
            ),
            "weight",
        ),
        (lambda: LoraConfig(adapter_name=""), "adapter_name"),
        (lambda: LoraConfig(adapter_name="teacher.v1"), "adapter_name"),
    ],
)
def test_opd_leaf_validation_is_strict(factory, message: str):
    with pytest.raises(ValidationError, match=message):
        factory()


def test_enabled_distillation_requires_teachers():
    with pytest.raises(ValidationError, match="at least one teacher"):
        DistillationConfig(enabled=True)


def test_enabled_distillation_requires_teacher_checkpoint():
    teacher = OPDTeacherConfig(
        name="subject",
        route_categories=["subject"],
        lora=LoraConfig(adapter_name="subject_adapter"),
    )

    with pytest.raises(ValidationError, match="lora.pretrained_weight"):
        DistillationConfig(enabled=True, teachers=[teacher])


@pytest.mark.parametrize("duplicate", ["name", "adapter", "category"])
def test_enabled_distillation_rejects_ambiguous_teacher_graph(tmp_path: Path, duplicate: str):
    checkpoint_a = tmp_path / "a.safetensors"
    checkpoint_b = tmp_path / "b.safetensors"
    checkpoint_a.touch()
    checkpoint_b.touch()
    first = _teacher(
        checkpoint_a,
        name="first",
        adapter_name="first_adapter",
        categories=["person"],
    )
    second = _teacher(
        checkpoint_b,
        name="first" if duplicate == "name" else "second",
        adapter_name="first_adapter" if duplicate == "adapter" else "second_adapter",
        categories=["person"] if duplicate == "category" else ["product"],
    )

    with pytest.raises(ValidationError, match="duplicate|route category"):
        DistillationConfig(enabled=True, teachers=[first, second])


def test_enabled_distillation_fallback_must_reference_teacher(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()

    with pytest.raises(ValidationError, match="fallback_teacher"):
        DistillationConfig(
            enabled=True,
            teachers=[_teacher(checkpoint)],
            fallback_teacher="missing",
        )


def test_lora_rejects_missing_teacher_checkpoint(tmp_path: Path):
    with pytest.raises(ValidationError, match="pretrained_weight path not found"):
        _teacher(tmp_path / "missing.safetensors")


def test_unknown_opd_fields_are_forbidden():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DistillationConfig.model_validate({"unknown": True})


def test_inference_config_loader_strips_training_only_teacher_graph(tmp_path: Path):
    checkpoint = tmp_path / "teacher.safetensors"
    checkpoint.touch()
    config_path = tmp_path / "opd.yaml"
    config_path.write_text(
        (
            "trainer: QwenImageEditInsertAnything\n"
            "mode: predict\n"
            "distillation:\n"
            "  enabled: true\n"
            "  teachers:\n"
            "    - name: subject\n"
            "      route_categories: [subject]\n"
            "      lora:\n"
            "        adapter_name: subject_teacher\n"
            f"        pretrained_weight: {checkpoint}\n"
        ),
        encoding="utf-8",
    )

    training_config = load_config_from_yaml(str(config_path))
    assert training_config.distillation.enabled is True
    inference_config = load_config_from_yaml(str(config_path), for_inference=True)
    assert inference_config.distillation.enabled is False
    assert inference_config.distillation.teachers == []

"""Teacher execution backends for on-policy distillation.

The default backend keeps one frozen Qwen transformer backbone and mounts every
task teacher as a named LoRA adapter.  This is the memory-efficient path for the
teachers currently produced by this repository.  The strategy only depends on
the small backend interface below, so a future backend can instead load full,
heterogeneous teacher models and offload inactive models to CPU.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import importlib
import logging
import os
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Protocol

import safetensors.torch
import torch


logger = logging.getLogger(__name__)

# Keep the distillation package independent from the inference/config stack.
# This filename is part of the final checkpoint artifact contract.
MASK_GEOMETRY_COMPANION_FILE_NAME = "mask_geometry_tokenizer.safetensors"

_LORA_PARAMETER_MARKERS = (
    ".lora_A.",
    ".lora_B.",
    ".lora_embedding_A.",
    ".lora_embedding_B.",
    ".lora_magnitude_vector.",
)


class TeacherBackend(Protocol):
    """Runtime contract used by the OPD strategy.

    A backend may share the student's model (the built-in LoRA path) or own
    completely independent teacher models. Independent backends can ignore
    ``shared_predictor`` and use ``batch``/``condition`` to run their own
    policy, but must return a velocity tensor in the student's state space.
    """

    def setup(self, trainer, teachers: Iterable[object]) -> None: ...

    def activate_student(self) -> None: ...

    def predict_velocity(
        self,
        teacher: object,
        *,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        condition: Mapping[str, Any],
        batch: Mapping[str, Any],
        guidance: object,
        shared_predictor: Callable[..., torch.Tensor],
    ) -> torch.Tensor: ...

    def assert_frozen(self) -> None: ...


def is_lora_parameter(name: str) -> bool:
    """Return whether ``name`` is a PEFT LoRA parameter name."""

    wrapped = f".{name}."
    return any(marker in wrapped for marker in _LORA_PARAMETER_MARKERS)


def is_adapter_parameter(name: str, adapter_name: str) -> bool:
    """Match an exact PEFT adapter segment instead of a substring."""

    wrapped = f".{name}."
    return any(f"{marker}{adapter_name}." in wrapped for marker in _LORA_PARAMETER_MARKERS)


def set_active_adapter(
    model: torch.nn.Module,
    adapter_name: str,
    *,
    trainable_adapter: str | None,
) -> None:
    """Activate one adapter and set an explicit, safe trainability mask.

    PEFT's ``set_adapter`` may mark the selected adapter trainable.  Teacher
    forwards happen inside ``no_grad``, but relying on that side effect would
    still let teacher tensors leak into optimizer construction or checkpoint
    accounting.  We therefore reset every LoRA parameter after each switch.
    """

    setter = getattr(model, "set_adapter", None)
    if setter is None:
        raise TypeError(
            f"{type(model).__name__} does not expose set_adapter(); "
            "SharedLoRATeacherBackend requires a PEFT-compatible model"
        )
    setter(adapter_name)

    found_trainable = False
    for name, parameter in model.named_parameters():
        if not is_lora_parameter(name):
            continue
        should_train = bool(trainable_adapter is not None and is_adapter_parameter(name, trainable_adapter))
        parameter.requires_grad_(should_train)
        found_trainable = found_trainable or should_train

    if trainable_adapter is not None and not found_trainable:
        raise ValueError(f"No LoRA parameters matched trainable adapter {trainable_adapter!r}")


class SharedLoRATeacherBackend:
    """Run all compatible teachers as frozen adapters on the student backbone."""

    # Teacher inference goes through trainer.dit. Under DDP/FSDP every rank
    # therefore needs the same number of teacher forwards per training batch.
    uses_student_distributed_model = True

    def __init__(self, **_: object):
        self._trainer = None
        self._teacher_to_adapter: Dict[str, str] = {}
        self._teacher_geometry_tokenizers: Dict[str, torch.nn.Module] = {}
        self._student_adapter: str | None = None

    @property
    def teacher_names(self) -> tuple[str, ...]:
        return tuple(self._teacher_to_adapter)

    def setup(self, trainer, teachers: Iterable[object]) -> None:
        """Load teacher LoRAs before optimizer/DDP/FSDP construction."""

        self._trainer = trainer
        self._student_adapter = trainer.adapter_name
        if not self._student_adapter:
            raise ValueError("OPD requires a non-empty student LoRA adapter name")

        for teacher in teachers:
            if getattr(teacher, "policy", None) is not None:
                raise ValueError(
                    f"Teacher {teacher.name!r} requests a custom policy, but "
                    "SharedLoRATeacherBackend only supports architecture-compatible LoRAs. "
                    "Configure an independent teacher backend for heterogeneous ideas."
                )

            if teacher.lora is None:
                raise ValueError(
                    f"Teacher {teacher.name!r} has no LoRA configuration; "
                    "use IndependentTeacherBackend for a full-model policy"
                )
            adapter_name = teacher.lora.adapter_name
            if adapter_name == self._student_adapter:
                raise ValueError(f"Teacher {teacher.name!r} reuses student adapter name {adapter_name!r}")

            teacher_config = trainer.config.model_copy(deep=True)
            teacher_config.model.lora = teacher.lora.model_copy(deep=True)
            trainer.__class__.load_pretrain_lora_model(
                trainer.dit,
                teacher_config,
                adapter_name,
                stage="teacher",
            )
            self._teacher_to_adapter[teacher.name] = adapter_name
            self._load_teacher_geometry_tokenizer(trainer, teacher)
            logger.info(
                "Loaded OPD teacher %s as shared LoRA adapter %s from %s",
                teacher.name,
                adapter_name,
                teacher.lora.pretrained_weight,
            )

        self.activate_student()

    @staticmethod
    def _geometry_weight_path(teacher: object) -> str | None:
        explicit = getattr(teacher, "mask_geometry_pretrained_weight", None)
        if explicit:
            explicit = str(explicit)
            if not os.path.isfile(explicit):
                raise FileNotFoundError(
                    f"Teacher {getattr(teacher, 'name', '<unknown>')!r} geometry companion "
                    f"is not a file: {explicit}"
                )
            return explicit
        lora = getattr(teacher, "lora", None)
        lora_weight = getattr(lora, "pretrained_weight", None)
        if not lora_weight:
            return None
        weight_dir = lora_weight if os.path.isdir(lora_weight) else os.path.dirname(lora_weight)
        candidate = os.path.join(weight_dir, MASK_GEOMETRY_COMPANION_FILE_NAME)
        return candidate if os.path.isfile(candidate) else None

    def _load_teacher_geometry_tokenizer(self, trainer, teacher: object) -> None:
        """Load the frozen companion module for a geometry-aware Teacher."""

        student_tokenizer_getter = getattr(trainer, "_get_mask_geometry_tokenizer", None)
        student_tokenizer = student_tokenizer_getter() if callable(student_tokenizer_getter) else None
        weight_path = self._geometry_weight_path(teacher)
        if student_tokenizer is None:
            if weight_path is not None:
                raise ValueError(
                    f"Teacher {teacher.name!r} configures mask geometry weights, but "
                    "the Student transformer has no mask_geometry_tokenizer"
                )
            return
        if weight_path is None:
            raise FileNotFoundError(
                f"Geometry-aware OPD Teacher {teacher.name!r} is missing "
                f"{MASK_GEOMETRY_COMPANION_FILE_NAME} next to its LoRA checkpoint"
            )
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Geometry-aware OPD Teacher {teacher.name!r} companion is not a file: {weight_path}"
            )

        tokenizer = copy.deepcopy(student_tokenizer).to("cpu")
        state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(
                f"Teacher {teacher.name!r} geometry tokenizer has unexpected keys: {list(unexpected)}"
            )
        if missing:
            raise ValueError(
                f"Teacher {teacher.name!r} geometry tokenizer is missing keys: {list(missing)}"
            )
        tokenizer.requires_grad_(False).eval()
        self._teacher_geometry_tokenizers[teacher.name] = tokenizer
        logger.info(
            "Loaded OPD teacher %s geometry tokenizer from %s",
            teacher.name,
            weight_path,
        )

    def _model(self) -> torch.nn.Module:
        if self._trainer is None:
            raise RuntimeError("Teacher backend has not been set up")
        unwrap = getattr(self._trainer, "_unwrap_dit", None)
        return unwrap() if unwrap is not None else self._trainer.dit

    def activate_student(self) -> None:
        if self._student_adapter is None:
            raise RuntimeError("Teacher backend has not been set up")
        set_active_adapter(
            self._model(),
            self._student_adapter,
            trainable_adapter=self._student_adapter,
        )

    def _tokenize_teacher_geometry(
        self,
        teacher_name: str,
        trainer,
        condition: Mapping[str, Any],
    ) -> Dict[str, Any]:
        tokenizer = self._teacher_geometry_tokenizers.get(teacher_name)
        if tokenizer is None:
            return dict(condition)

        prepared = dict(condition)
        geometry = prepared.pop("mask_geometry", None)
        existing_tokens = prepared.get("mask_geometry_tokens")
        if geometry is not None and existing_tokens is not None:
            raise ValueError("Teacher condition contains both mask_geometry and mask_geometry_tokens")
        if geometry is None and existing_tokens is not None:
            raise ValueError(
                "Geometry-aware Teacher condition contains precomputed mask_geometry_tokens "
                "without raw mask_geometry; refusing to use tokens that may belong to the Student "
                "instead of this Teacher's companion tokenizer"
            )
        if geometry is None:
            raise ValueError(
                "Geometry-aware Teacher condition is missing mask_geometry; "
                "the selected Teacher companion cannot be applied"
            )
        if geometry is not None:
            batch_size = int(geometry.shape[0])
            img_shapes = trainer._get_image_shapes(prepared, batch_size)
            target_grid = getattr(self._model(), "_mask_geometry_target_grid", None)
            if callable(target_grid):
                target_size = target_grid(img_shapes, batch_size)
            else:
                # Keep the backend usable with compatible idea branches that
                # expose the same img_shapes contract but not final's private
                # transformer helper.
                if not img_shapes or not img_shapes[0] or len(img_shapes[0][0]) != 3:
                    raise TypeError(
                        "Geometry-aware SharedLoRATeacherBackend requires a valid "
                        "target entry in img_shapes"
                    )
                target_size = (int(img_shapes[0][0][1]), int(img_shapes[0][0][2]))
            device = trainer._dit_device()
            tokenizer.to(device=device, dtype=trainer.weight_dtype)
            with torch.no_grad():
                prepared["mask_geometry_tokens"] = tokenizer(
                    geometry.to(device=device, dtype=trainer.weight_dtype),
                    target_size,
                ).detach()

        null_condition = prepared.get("reference_cfg_embeddings")
        if isinstance(null_condition, Mapping):
            prepared["reference_cfg_embeddings"] = self._tokenize_teacher_geometry(
                teacher_name,
                trainer,
                null_condition,
            )
        return prepared

    @contextmanager
    def teacher(self, teacher_name: str) -> Iterator[None]:
        """Activate one frozen teacher and always restore the student."""

        try:
            adapter_name = self._teacher_to_adapter[teacher_name]
        except KeyError as exc:
            raise KeyError(f"Unknown OPD teacher {teacher_name!r}") from exc

        model = self._model()
        set_active_adapter(model, adapter_name, trainable_adapter=None)
        try:
            yield
        finally:
            self.activate_student()

    def predict_velocity(
        self,
        teacher: object,
        *,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        condition: Mapping[str, Any],
        batch: Mapping[str, Any],
        guidance: object,
        shared_predictor: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        """Run a compatible teacher through the shared Qwen forward path."""

        del batch  # Raw inputs are reserved for independent teacher backends.
        condition = self._tokenize_teacher_geometry(teacher.name, trainer, condition)
        with self.teacher(teacher.name):
            return shared_predictor(
                trainer,
                state,
                timestep,
                condition,
                guidance,
            )

    def assert_frozen(self) -> None:
        """Fail fast if a teacher adapter was accidentally made trainable."""

        model = self._model()
        violations = []
        teacher_adapters = set(self._teacher_to_adapter.values())
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(is_adapter_parameter(name, adapter) for adapter in teacher_adapters):
                violations.append(name)
        if violations:
            raise RuntimeError("Frozen OPD teacher parameters became trainable; first violation: " f"{violations[0]}")
        for teacher_name, tokenizer in self._teacher_geometry_tokenizers.items():
            violation = next(
                (name for name, parameter in tokenizer.named_parameters() if parameter.requires_grad),
                None,
            )
            if violation is not None:
                raise RuntimeError(
                    f"Frozen OPD teacher {teacher_name!r} geometry tokenizer parameter "
                    f"{violation!r} became trainable"
                )


class IndependentTeacherBackend:
    """Host independently loaded teacher policies behind one stable contract.

    Each teacher must configure ``policy.class_path``. The policy is constructed
    from ``policy.init_args``, then receives ``setup(trainer=..., teacher=...)``.
    Its ``predict_velocity`` method gets both raw routed batch fields and the
    standard Qwen condition. This lets an idea branch use its own model and
    preprocessing while keeping routing, rollout, objective, and metrics shared.
    """

    uses_student_distributed_model = False

    def __init__(self, **_: object):
        self._trainer = None
        self._policies: Dict[str, object] = {}

    @staticmethod
    def _policy_module(policy: object) -> torch.nn.Module | None:
        if isinstance(policy, torch.nn.Module):
            return policy
        model = getattr(policy, "model", None)
        return model if isinstance(model, torch.nn.Module) else None

    def setup(self, trainer, teachers: Iterable[object]) -> None:
        self._trainer = trainer
        for teacher in teachers:
            plugin = getattr(teacher, "policy", None)
            if plugin is None:
                raise ValueError(
                    f"Teacher {teacher.name!r} requires policy.class_path for " "IndependentTeacherBackend"
                )
            module_name, class_name = plugin.class_path.rsplit(".", 1)
            policy_class = getattr(importlib.import_module(module_name), class_name)
            policy = policy_class(**dict(plugin.init_args))
            setup = getattr(policy, "setup", None)
            if not callable(setup):
                raise TypeError(f"Independent teacher policy {plugin.class_path} must define setup()")
            setup(trainer=trainer, teacher=teacher)

            policy_module = self._policy_module(policy)
            if policy_module is not None:
                policy_module.requires_grad_(False).eval()
            elif not callable(getattr(policy, "assert_frozen", None)):
                raise TypeError(
                    f"Independent teacher policy {plugin.class_path} must be an nn.Module, "
                    "expose .model, or implement assert_frozen()"
                )
            self._policies[teacher.name] = policy

        self.activate_student()
        self.assert_frozen()

    def _student_model(self) -> torch.nn.Module:
        if self._trainer is None:
            raise RuntimeError("Teacher backend has not been set up")
        unwrap = getattr(self._trainer, "_unwrap_dit", None)
        return unwrap() if unwrap is not None else self._trainer.dit

    def activate_student(self) -> None:
        if self._trainer is None:
            raise RuntimeError("Teacher backend has not been set up")
        adapter_name = getattr(self._trainer, "adapter_name", None)
        model = self._student_model()
        if adapter_name and callable(getattr(model, "set_adapter", None)):
            set_active_adapter(
                model,
                adapter_name,
                trainable_adapter=adapter_name,
            )

    def predict_velocity(
        self,
        teacher: object,
        *,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        condition: Mapping[str, Any],
        batch: Mapping[str, Any],
        guidance: object,
        shared_predictor: Callable[..., torch.Tensor],
    ) -> torch.Tensor:
        del trainer, shared_predictor
        try:
            policy = self._policies[teacher.name]
        except KeyError as exc:
            raise KeyError(f"Unknown independent OPD teacher {teacher.name!r}") from exc
        predict = getattr(policy, "predict_velocity", None)
        if not callable(predict):
            raise TypeError(f"Independent teacher policy {type(policy).__name__} must define " "predict_velocity()")
        return predict(
            state=state,
            timestep=timestep,
            condition=condition,
            batch=batch,
            guidance=guidance,
        )

    def assert_frozen(self) -> None:
        for teacher_name, policy in self._policies.items():
            custom_assert = getattr(policy, "assert_frozen", None)
            if callable(custom_assert):
                custom_assert()
            policy_module = self._policy_module(policy)
            if policy_module is None:
                continue
            violation = next(
                (name for name, parameter in policy_module.named_parameters() if parameter.requires_grad),
                None,
            )
            if violation is not None:
                raise RuntimeError(f"Independent teacher {teacher_name!r} parameter {violation!r} " "is trainable")

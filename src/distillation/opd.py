"""Qwen Image Edit on-policy distillation strategy.

The implementation follows the deterministic-ODE recipe used for Qwen image
OPD: sample a trajectory from the *current student*, detach it, route each
sample to exactly one task teacher, and match teacher/student velocities at the
states visited by the student.  A transition-mean objective can be selected by
configuration without changing the training loop.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import logging
import math
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

import torch

from src.distillation.routing import CategoryTeacherRouter
from src.distillation.types import FlowTrajectory


logger = logging.getLogger(__name__)


def _load_component(plugin_config, **extra_kwargs):
    module_name, class_name = plugin_config.class_path.rsplit(".", 1)
    component_class = getattr(importlib.import_module(module_name), class_name)
    kwargs = dict(plugin_config.init_args)
    kwargs.update(extra_kwargs)
    return component_class(**kwargs)


def _normalize_categories(categories: object) -> list[str]:
    if isinstance(categories, str):
        return [categories]
    if isinstance(categories, Sequence):
        normalized = [str(value) for value in categories]
        if not normalized:
            raise ValueError("OPD cannot route an empty batch")
        return normalized
    raise TypeError("OPD expects batch['category'] to be a string or a sequence of strings")


def _index_batch_value(value: Any, indices: torch.Tensor, batch_size: int) -> Any:
    """Select batch items from a nested condition mapping.

    Scalars and metadata shared by the batch are retained.  Tensor/list values
    whose leading length equals ``batch_size`` are indexed.  This keeps the
    strategy independent of optional conditioning fields introduced by idea
    branches, provided those fields follow the usual leading-batch convention.
    """

    if isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value.index_select(0, indices.to(value.device))
        return value
    if isinstance(value, Mapping):
        return {key: _index_batch_value(item, indices, batch_size) for key, item in value.items()}
    if isinstance(value, list) and len(value) == batch_size:
        selected = indices.detach().cpu().tolist()
        return [value[index] for index in selected]
    if isinstance(value, tuple) and len(value) == batch_size:
        selected = indices.detach().cpu().tolist()
        return tuple(value[index] for index in selected)
    return value


def _select_condition(condition: Mapping[str, Any], indices: torch.Tensor, batch_size: int) -> Dict[str, Any]:
    return {key: _index_batch_value(value, indices, batch_size) for key, value in condition.items()}


@contextmanager
def _temporary_eval(module: torch.nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class QwenImageEditOPDStrategy:
    """Composable OPD algorithm plugged into the existing trainer lifecycle."""

    def __init__(self, config, **_: object):
        self.config = config
        self.objective = _load_component(config.objective)
        self.backend = _load_component(config.teacher_backend)
        self.teachers = {teacher.name: teacher for teacher in config.teachers}
        self.router = CategoryTeacherRouter(
            {teacher.name: tuple(teacher.route_categories) for teacher in config.teachers},
            fallback=config.fallback_teacher,
        )
        self._is_setup = False
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self._metric_sums = {
            "loss": 0.0,
            "loss_count": 0.0,
            "rollout_steps": 0.0,
            "trained_steps": 0.0,
            "batches": 0.0,
            "samples": 0.0,
        }
        for teacher_name in self.teachers:
            self._metric_sums[f"teacher/{teacher_name}/loss"] = 0.0
            self._metric_sums[f"teacher/{teacher_name}/loss_count"] = 0.0
            self._metric_sums[f"teacher/{teacher_name}/samples"] = 0.0

    def setup(self, trainer) -> None:
        self.backend.setup(trainer, self.config.teachers)
        self.backend.assert_frozen()
        self._validate_distributed_teacher_forwards(trainer)
        self._is_setup = True
        logger.info(
            "Configured Qwen OPD with teachers=%s, objective=%s, rollout_steps=%d",
            list(self.teachers),
            type(self.objective).__name__,
            self.config.rollout.num_inference_steps,
        )

    @staticmethod
    def _guidance_forward_count(guidance) -> int:
        guided = guidance.mode != "none" and float(guidance.scale) > 1.0
        return 2 if guided else 1

    def _validate_distributed_teacher_forwards(self, trainer) -> None:
        """Prevent mismatched DDP/FSDP teacher collectives across ranks."""

        if not bool(getattr(self.backend, "uses_student_distributed_model", False)):
            return
        if int(trainer.accelerator.num_processes) <= 1:
            return

        batch_size = int(trainer.config.data.batch_size)
        if batch_size != 1:
            raise ValueError(
                "Distributed SharedLoRA OPD currently requires data.batch_size=1; "
                "mixed per-rank Teacher groups can execute different DDP/FSDP "
                "forward counts"
            )

        forward_counts = {self._guidance_forward_count(teacher.guidance) for teacher in self.teachers.values()}
        if len(forward_counts) != 1:
            raise ValueError(
                "Distributed SharedLoRA OPD requires every Teacher guidance policy "
                "to use the same number of model forwards (all guided or all unguided)"
            )

    @staticmethod
    def _condition_key(guidance) -> Tuple[str, str, float]:
        # Scale affects whether CFG auxiliary embeddings are prepared. Keeping
        # it in the key prevents a scale=1 policy from being reused by a guided
        # policy with otherwise identical prompt/mode fields.
        return guidance.prompt_mode, guidance.mode, float(guidance.scale)

    def _required_condition_policies(self) -> Dict[Tuple[str, str, float], object]:
        policies = {self._condition_key(self.config.student_guidance): self.config.student_guidance}
        for teacher in self.teachers.values():
            policies.setdefault(self._condition_key(teacher.guidance), teacher.guidance)
        return policies

    @staticmethod
    def _empty_prompts(batch_size: int) -> list[str]:
        return [""] * batch_size

    def _prepare_conditions(
        self,
        trainer,
        batch: Mapping[str, Any],
    ) -> Dict[Tuple[str, str, float], Dict[str, Any]]:
        categories = _normalize_categories(batch.get("category"))
        batch_size = len(categories)
        conditions: Dict[Tuple[str, str, float], Dict[str, Any]] = {}

        for key, guidance in self._required_condition_policies().items():
            prompt_mode, guidance_mode, _ = key
            local_batch: MutableMapping[str, Any] = dict(batch)
            if prompt_mode == "empty":
                local_batch["_opd_prompt_override"] = self._empty_prompts(batch_size)
            else:
                local_batch["_opd_prompt_override"] = batch["prompt"]

            local_batch["_opd_use_reference_cfg"] = guidance_mode == "reference_cfg"
            local_batch["true_cfg_scale"] = float(guidance.scale) if guidance_mode != "none" else 1.0
            local_batch["guidance"] = 1.0

            if guidance_mode == "text_cfg":
                negative_prompt = batch.get("negative_prompt")
                if negative_prompt is None:
                    negative_prompt = self._empty_prompts(batch_size)
                local_batch["negative_prompt"] = negative_prompt
            else:
                local_batch.pop("negative_prompt", None)

            with torch.no_grad():
                conditions[key] = trainer.prepare_embeddings(dict(local_batch), stage="opd")

        return conditions

    @staticmethod
    def _module_device(trainer) -> torch.device:
        return trainer._dit_device()

    def _predict_raw_velocity(
        self,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        condition: Mapping[str, Any],
    ) -> torch.Tensor:
        custom_forward = getattr(trainer, "opd_forward_velocity", None)
        if callable(custom_forward):
            return custom_forward(
                state=state,
                timestep=timestep,
                condition=condition,
            )

        device = self._module_device(trainer)
        state = state.to(device=device, dtype=trainer.weight_dtype)
        batch_size = state.shape[0]

        control_latents = condition.get("control_latents")
        if control_latents is not None:
            control_latents = control_latents.to(device=device, dtype=trainer.weight_dtype)
            hidden_states = torch.cat([state, control_latents], dim=1)
        else:
            hidden_states = state

        prompt_embeds = condition["prompt_embeds"].to(device=device, dtype=trainer.weight_dtype)
        prompt_mask = condition["prompt_embeds_mask"].to(device=device, dtype=torch.int64)
        img_shapes = trainer._get_image_shapes(condition, batch_size)
        txt_seq_lens = prompt_mask.sum(dim=1).tolist()

        if not isinstance(timestep, torch.Tensor):
            timestep = torch.as_tensor(timestep, device=device)
        timestep = timestep.to(device=device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.shape != (batch_size,):
            raise ValueError(f"OPD timestep must be scalar or [B], got {tuple(timestep.shape)}")
        timestep = timestep.to(dtype=state.dtype)

        dit_config = trainer._dit_config()
        if dit_config.guidance_embeds:
            embedded_scale = condition.get("guidance", 1.0)
            if isinstance(embedded_scale, torch.Tensor):
                guidance = embedded_scale.to(device=device, dtype=torch.float32)
                if guidance.ndim == 0:
                    guidance = guidance.expand(batch_size)
            else:
                guidance = torch.full(
                    (batch_size,),
                    float(embedded_scale),
                    device=device,
                    dtype=torch.float32,
                )
        else:
            guidance = None

        geometry_kwargs = {}
        mask_geometry = condition.get("mask_geometry")
        mask_geometry_tokens = condition.get("mask_geometry_tokens")
        if mask_geometry is not None and mask_geometry_tokens is not None:
            raise ValueError("OPD condition cannot contain both mask_geometry and mask_geometry_tokens")
        if mask_geometry is not None:
            geometry_kwargs["mask_geometry"] = mask_geometry.to(
                device=device,
                dtype=trainer.weight_dtype,
            )
        if mask_geometry_tokens is not None:
            geometry_kwargs["mask_geometry_tokens"] = mask_geometry_tokens.to(
                device=device,
                dtype=trainer.weight_dtype,
            )

        prediction = trainer.dit(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            **geometry_kwargs,
            return_dict=False,
        )[0]
        return prediction[:, : state.shape[1]]

    def _predict_guided_velocity(
        self,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        condition: Mapping[str, Any],
        guidance,
    ) -> torch.Tensor:
        positive = self._predict_raw_velocity(trainer, state, timestep, condition)
        if guidance.mode == "none" or float(guidance.scale) == 1.0:
            return positive

        if guidance.mode == "reference_cfg":
            null_condition = condition.get("reference_cfg_embeddings")
            if null_condition is None:
                raise ValueError("reference_cfg OPD policy requires prepared null-reference embeddings")
            null_prediction = self._predict_raw_velocity(trainer, state, timestep, null_condition)
            reference_mask = None
            if condition.get("reference_cfg_mask_aware", False):
                raw_mask = condition.get("reference_cfg_mask")
                if raw_mask is not None:
                    reference_mask = trainer._prepare_reference_cfg_mask(
                        reference_cfg_mask=raw_mask,
                        batch_size=state.shape[0],
                        height_image=int(condition["height"]),
                        width_image=int(condition["width"]),
                        image_seq_len=state.shape[1],
                        device=state.device,
                    )
            return trainer._combine_reference_cfg_predictions(
                noise_pred=positive,
                reference_cfg_noise_pred=null_prediction,
                reference_cfg_scale=float(guidance.scale),
                reference_cfg_mask=reference_mask,
                delta_clip_enabled=bool(condition.get("reference_cfg_delta_clip_enabled", False)),
                delta_clip_max_ratio=float(condition.get("reference_cfg_delta_clip_max_ratio", 0.5)),
            )

        if guidance.mode == "text_cfg":
            if "negative_prompt_embeds" not in condition:
                raise ValueError("text_cfg OPD policy requires negative prompt embeddings")
            negative_condition = dict(condition)
            negative_condition["prompt_embeds"] = condition["negative_prompt_embeds"]
            negative_condition["prompt_embeds_mask"] = condition["negative_prompt_embeds_mask"]
            negative = self._predict_raw_velocity(trainer, state, timestep, negative_condition)
            combined = negative + float(guidance.scale) * (positive - negative)
            positive_norm = torch.norm(positive, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True).clamp(min=1e-6)
            return combined * (positive_norm / combined_norm)

        raise ValueError(f"Unsupported OPD guidance mode: {guidance.mode!r}")

    def _rollout(
        self,
        trainer,
        condition: Mapping[str, Any],
    ) -> FlowTrajectory:
        device = self._module_device(trainer)
        batch_size = condition["control_latents"].shape[0]
        height = int(condition["height"])
        width = int(condition["width"])
        num_channels = trainer._dit_config().in_channels // 4

        seed = (
            int(self.config.rollout.seed)
            + int(trainer.global_step) * max(1, trainer.accelerator.num_processes)
            + int(trainer.accelerator.process_index)
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        latents = trainer.prepare_initial_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels,
            height=height,
            width=width,
            dtype=trainer.weight_dtype,
            generator=generator,
            device=device,
        )
        timesteps, _ = trainer.prepare_predict_timesteps(int(self.config.rollout.num_inference_steps), latents.shape[1])
        sigmas = trainer.scheduler.sigmas
        if sigmas.numel() < timesteps.numel() + 1:
            raise ValueError(
                "Flow scheduler must expose T+1 sigmas for an OPD rollout, got "
                f"{sigmas.numel()} sigmas for {timesteps.numel()} timesteps"
            )

        store_device = device if self.config.rollout.store_device == "cuda" else torch.device("cpu")
        states = [latents.detach().to(store_device)]
        self.backend.activate_student()
        with _temporary_eval(trainer.dit), torch.no_grad():
            for step_index, timestep in enumerate(timesteps):
                velocity = self._predict_guided_velocity(
                    trainer,
                    latents,
                    timestep,
                    condition,
                    self.config.student_guidance,
                )
                delta_sigma = (sigmas[step_index + 1] - sigmas[step_index]).to(device=device, dtype=torch.float32)
                latents = (latents.float() + delta_sigma * velocity.float()).to(dtype=trainer.weight_dtype)
                states.append(latents.detach().to(store_device))

        return FlowTrajectory(
            states=torch.stack(states, dim=1),
            timesteps=timesteps.detach().to(store_device),
            sigmas=sigmas[: timesteps.numel() + 1].detach().to(store_device),
        )

    def _selected_timesteps(self, trainer, num_steps: int) -> torch.Tensor:
        fraction = float(self.config.rollout.timestep_fraction)
        selected_count = max(1, math.ceil(num_steps * fraction))
        if selected_count >= num_steps:
            return torch.arange(num_steps, dtype=torch.long)
        generator = torch.Generator(device="cpu").manual_seed(int(self.config.rollout.seed) + int(trainer.global_step))
        return torch.randperm(num_steps, generator=generator)[:selected_count].sort().values

    def _teacher_targets(
        self,
        trainer,
        state: torch.Tensor,
        timestep: torch.Tensor,
        batch: Mapping[str, Any],
        conditions: Mapping[Tuple[str, str, float], Mapping[str, Any]],
        groups: Mapping[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.empty_like(state, device=self._module_device(trainer))
        teacher_weight = torch.empty((batch_size,), device=target.device, dtype=torch.float32)

        with _temporary_eval(trainer.dit):
            for teacher_name, indices in groups.items():
                teacher = self.teachers[teacher_name]
                selected_state = state.index_select(0, indices.to(state.device))
                # FlowTrajectory also permits per-sample timesteps [B, T].
                # Route those values together with the state subset; a shared
                # scalar/step vector is intentionally passed through unchanged.
                selected_timestep = timestep
                if (
                    isinstance(timestep, torch.Tensor)
                    and timestep.ndim == 1
                    and timestep.shape[0] == batch_size
                ):
                    selected_timestep = timestep.index_select(0, indices.to(timestep.device))
                selected_condition = _select_condition(
                    conditions[self._condition_key(teacher.guidance)],
                    indices,
                    batch_size,
                )
                selected_batch = _select_condition(batch, indices, batch_size)
                backend_predictor = getattr(self.backend, "predict_velocity", None)
                with torch.no_grad():
                    if callable(backend_predictor):
                        prediction = backend_predictor(
                            teacher,
                            trainer=trainer,
                            state=selected_state,
                            timestep=selected_timestep,
                            condition=selected_condition,
                            batch=selected_batch,
                            guidance=teacher.guidance,
                            shared_predictor=self._predict_guided_velocity,
                        )
                    else:
                        # Compatibility path for third-party backends written
                        # before the independent-teacher forward contract.
                        with self.backend.teacher(teacher_name):
                            prediction = self._predict_guided_velocity(
                                trainer,
                                selected_state,
                                selected_timestep,
                                selected_condition,
                                teacher.guidance,
                            )
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError(f"Teacher backend {type(self.backend).__name__} must return a tensor")
                if prediction.shape != selected_state.shape:
                    raise ValueError(
                        f"Teacher {teacher_name!r} returned velocity shape "
                        f"{tuple(prediction.shape)}, expected {tuple(selected_state.shape)}"
                    )
                target.index_copy_(
                    0,
                    indices.to(target.device),
                    prediction.detach().to(device=target.device, dtype=target.dtype),
                )
                teacher_weight.index_fill_(0, indices.to(teacher_weight.device), float(teacher.weight))

        self.backend.activate_student()
        return target, teacher_weight

    def _objective_result(
        self,
        state: torch.Tensor,
        step_index: int,
        trajectory: FlowTrajectory,
        student_velocity: torch.Tensor,
        teacher_velocity: torch.Tensor,
        teacher_weight: torch.Tensor,
    ):
        prediction_space = getattr(self.objective, "prediction_space", None)
        if prediction_space == "velocity":
            return self.objective(
                student_velocity,
                teacher_velocity,
                teacher_weight=teacher_weight,
            )
        if prediction_space == "transition_mean":
            if getattr(self.objective, "mode", "ode") != "ode":
                raise NotImplementedError(
                    "The Qwen OPD rollout is deterministic ODE. Configure an SDE "
                    "rollout backend before selecting an SDE transition objective."
                )
            if trajectory.sigmas is None:
                raise ValueError(
                    "transition_mean objective requires FlowTrajectory.sigmas with shape [T+1]"
                )
            delta_sigma = (trajectory.sigmas[step_index + 1] - trajectory.sigmas[step_index]).to(
                device=state.device, dtype=torch.float32
            )
            student_mean = state.float() + delta_sigma * student_velocity.float()
            teacher_mean = state.float() + delta_sigma * teacher_velocity.float()
            return self.objective(
                student_mean,
                teacher_mean,
                teacher_weight=teacher_weight,
            )
        raise TypeError(
            f"Objective {type(self.objective).__name__} must declare "
            "prediction_space='velocity' or 'transition_mean'"
        )

    def backward_step(self, trainer, batch: Mapping[str, Any]) -> torch.Tensor:
        """Run one rollout and backpropagate timestep micro-losses immediately."""

        if not self._is_setup:
            raise RuntimeError("OPD strategy must be set up before training")

        prepare_opd_batch = getattr(trainer, "prepare_opd_batch", None)
        if callable(prepare_opd_batch):
            batch = prepare_opd_batch(dict(batch))
            if not isinstance(batch, Mapping):
                raise TypeError("prepare_opd_batch() must return a mapping")

        categories = _normalize_categories(batch.get("category"))
        batch_size = len(categories)
        groups = self.router.route_groups(categories)
        if (
            bool(getattr(self.backend, "uses_student_distributed_model", False))
            and int(trainer.accelerator.num_processes) > 1
            and len(groups) != 1
        ):
            raise RuntimeError(
                "Distributed SharedLoRA OPD received a mixed-Teacher local batch; "
                "use data.batch_size=1 or a synchronized homogeneous batch sampler"
            )
        conditions = self._prepare_conditions(trainer, batch)
        student_condition = conditions[self._condition_key(self.config.student_guidance)]
        trajectory = self._rollout(trainer, student_condition)
        if not isinstance(trajectory, FlowTrajectory):
            raise TypeError(
                "OPD rollout must return a FlowTrajectory, "
                f"got {type(trajectory).__name__}"
            )
        if trajectory.batch_size != batch_size:
            raise ValueError(
                "OPD rollout batch size must match the routed input batch: "
                f"got trajectory B={trajectory.batch_size}, input B={batch_size}"
            )
        selected_steps = self._selected_timesteps(trainer, trajectory.num_steps)

        total_loss = torch.zeros((), device=self._module_device(trainer), dtype=torch.float32)
        teacher_loss_sums = {name: 0.0 for name in self.teachers}
        teacher_loss_counts = {name: 0.0 for name in self.teachers}

        trainer.dit.train()
        self.backend.activate_student()
        selected_step_count = int(selected_steps.numel())
        for selected_position, step_tensor in enumerate(selected_steps):
            step_index = int(step_tensor.item())
            state = trajectory.states[:, step_index].to(device=self._module_device(trainer), dtype=trainer.weight_dtype)
            if trajectory.timesteps.ndim == 1:
                timestep = trajectory.timesteps[step_index].to(device=self._module_device(trainer))
            else:
                timestep = trajectory.timesteps[:, step_index].to(device=self._module_device(trainer))
            teacher_velocity, teacher_weight = self._teacher_targets(
                trainer,
                state,
                timestep,
                batch,
                conditions,
                groups,
                batch_size,
            )
            # DDP's no_sync context must cover both the Student forward and its
            # backward.  Accumulate the first T-1 timestep gradients locally,
            # then let only the final timestep participate in the outer
            # Accelerate accumulation policy.  If the outer microbatch is
            # already under no_sync, the nested context restores that state and
            # the final timestep correctly remains unsynchronized as well.
            sync_context = (
                trainer.accelerator.no_sync(trainer.dit)
                if selected_position < selected_step_count - 1
                else nullcontext()
            )
            with sync_context:
                student_velocity = self._predict_guided_velocity(
                    trainer,
                    state,
                    timestep,
                    student_condition,
                    self.config.student_guidance,
                )
                result = self._objective_result(
                    state,
                    step_index,
                    trajectory,
                    student_velocity,
                    teacher_velocity,
                    teacher_weight,
                )
                trainer.accelerator.backward(result.loss / selected_step_count)
            total_loss = total_loss + result.loss.detach().float()

            per_sample = result.per_sample.detach().float()
            for teacher_name, indices in groups.items():
                selected = per_sample.index_select(0, indices.to(per_sample.device))
                teacher_loss_sums[teacher_name] += float(selected.sum().item())
                teacher_loss_counts[teacher_name] += float(selected.numel())

        mean_loss = total_loss / selected_steps.numel()
        self._metric_sums["loss"] += float(total_loss.item())
        self._metric_sums["loss_count"] += float(selected_steps.numel())
        self._metric_sums["rollout_steps"] += float(trajectory.num_steps)
        self._metric_sums["trained_steps"] += float(selected_steps.numel())
        self._metric_sums["batches"] += 1.0
        self._metric_sums["samples"] += float(batch_size)
        for teacher_name, indices in groups.items():
            self._metric_sums[f"teacher/{teacher_name}/samples"] += float(indices.numel())
        for teacher_name in self.teachers:
            self._metric_sums[f"teacher/{teacher_name}/loss"] += teacher_loss_sums[teacher_name]
            self._metric_sums[f"teacher/{teacher_name}/loss_count"] += teacher_loss_counts[teacher_name]

        self.backend.assert_frozen()
        return mean_loss.detach()

    def consume_metrics(self, accelerator) -> Dict[str, float]:
        """Reduce a stable metric vector on every rank and clear local state."""

        keys = tuple(self._metric_sums)
        values = torch.tensor(
            [self._metric_sums[key] for key in keys],
            device=accelerator.device,
            dtype=torch.float64,
        )
        reduced = accelerator.reduce(values, reduction="sum")
        totals = {key: float(reduced[index].item()) for index, key in enumerate(keys)}
        self._reset_metrics()

        loss_count = max(totals["loss_count"], 1.0)
        batches = max(totals["batches"], 1.0)
        samples = max(totals["samples"], 1.0)
        metrics = {
            "opd/loss": totals["loss"] / loss_count,
            "opd/rollout_steps_per_batch": totals["rollout_steps"] / batches,
            "opd/trained_steps_per_batch": totals["trained_steps"] / batches,
        }
        for teacher_name in self.teachers:
            count = totals[f"teacher/{teacher_name}/loss_count"]
            if count > 0:
                metrics[f"opd/teacher/{teacher_name}/loss"] = totals[f"teacher/{teacher_name}/loss"] / count
            metrics[f"opd/teacher/{teacher_name}/route_fraction"] = totals[f"teacher/{teacher_name}/samples"] / samples
        return metrics

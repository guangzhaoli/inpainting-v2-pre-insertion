"""Composable distillation objectives for deterministic and stochastic flows."""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Union

import torch
from torch import Tensor, nn

from .types import DistillationResult


TensorOrScalar = Union[Tensor, float]


def _validate_prediction_pair(student: Tensor, teacher: Tensor, *, student_name: str, teacher_name: str) -> None:
    if not isinstance(student, Tensor):
        raise TypeError(f"{student_name} must be a torch.Tensor, got {type(student).__name__}")
    if not isinstance(teacher, Tensor):
        raise TypeError(f"{teacher_name} must be a torch.Tensor, got {type(teacher).__name__}")
    if student.ndim < 1:
        raise ValueError(f"{student_name} and {teacher_name} must include a batch dimension")
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(
            f"{student_name} and {teacher_name} must have exactly the same shape, "
            f"got {tuple(student.shape)} and {tuple(teacher.shape)}"
        )
    if student.shape[0] <= 0 or student[0].numel() <= 0:
        raise ValueError(f"{student_name} must contain at least one non-empty sample")
    if student.device != teacher.device:
        raise ValueError(
            f"{student_name} and {teacher_name} must be on the same device, "
            f"got {student.device} and {teacher.device}"
        )
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise TypeError(f"{student_name} and {teacher_name} must be floating-point tensors")


def _expand_like(value: Tensor, reference: Tensor, *, name: str) -> Tensor:
    """Broadcast ``value`` to ``reference`` with batch/token prefix support.

    PyTorch cannot directly broadcast a token mask ``[B, L]`` against a packed
    latent ``[B, L, C]`` because broadcasting is right aligned.  When the input
    matches a leading prefix of the reference, trailing singleton dimensions
    are appended before regular broadcasting is attempted.
    """

    if value.ndim > reference.ndim:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot broadcast to target shape {tuple(reference.shape)}"
        )

    candidate = value
    prefix_shape = tuple(reference.shape[: value.ndim])
    prefix_broadcastable = all(size == 1 or size == target for size, target in zip(value.shape, prefix_shape))
    if value.ndim > 0 and prefix_broadcastable and value.ndim < reference.ndim:
        candidate = value.reshape(*value.shape, *((1,) * (reference.ndim - value.ndim)))

    try:
        return torch.broadcast_to(candidate, reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(value.shape)} cannot broadcast to target shape {tuple(reference.shape)}"
        ) from exc


def _prepare_mask(mask: Optional[Tensor], reference: Tensor) -> Optional[Tensor]:
    if mask is None:
        return None
    if not isinstance(mask, Tensor):
        raise TypeError(f"token_mask must be a torch.Tensor or None, got {type(mask).__name__}")
    if mask.device != reference.device:
        raise ValueError(f"token_mask must be on {reference.device}, got {mask.device}")
    expanded = _expand_like(mask.detach(), reference, name="token_mask").to(dtype=torch.float32)
    if not torch.isfinite(expanded).all().item():
        raise ValueError("token_mask must contain only finite values")
    if (expanded < 0).any().item():
        raise ValueError("token_mask weights must be non-negative")
    return expanded


def _masked_per_sample_mean(elementwise: Tensor, mask: Optional[Tensor]) -> Tensor:
    reduction_dims = tuple(range(1, elementwise.ndim))
    if mask is None:
        if reduction_dims:
            return elementwise.mean(dim=reduction_dims)
        return elementwise

    weighted = elementwise * mask
    if reduction_dims:
        numerator = weighted.sum(dim=reduction_dims)
        denominator = mask.sum(dim=reduction_dims)
    else:
        numerator = weighted
        denominator = mask
    if (denominator <= 0).any().item():
        bad_samples = (denominator <= 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"token_mask must select positive weight for every sample; empty samples: {bad_samples}")
    return numerator / denominator


def _prepare_teacher_weight(teacher_weight: Optional[Tensor], per_sample: Tensor, *, validate_values=True) -> Optional[Tensor]:
    if teacher_weight is None:
        return None
    if not isinstance(teacher_weight, Tensor):
        raise TypeError(f"teacher_weight must be a torch.Tensor or None, got {type(teacher_weight).__name__}")
    expected = (per_sample.shape[0],)
    if tuple(teacher_weight.shape) != expected:
        raise ValueError(f"teacher_weight must have shape [B]={expected}, got {tuple(teacher_weight.shape)}")
    if teacher_weight.device != per_sample.device:
        raise ValueError(f"teacher_weight must be on {per_sample.device}, got {teacher_weight.device}")
    weight = teacher_weight.detach().to(dtype=torch.float32)
    if not validate_values:
        return weight
    if not torch.isfinite(weight).all().item():
        raise ValueError("teacher_weight must contain only finite values")
    if (weight < 0).any().item():
        raise ValueError("teacher_weight must be non-negative")
    if weight.sum().item() <= 0:
        raise ValueError("teacher_weight must have a positive sum")
    return weight


def _batch_mean(per_sample: Tensor, teacher_weight: Optional[Tensor], *, prevalidated=False) -> Tensor:
    weight = _prepare_teacher_weight(teacher_weight, per_sample, validate_values=not prevalidated)
    if weight is None:
        return per_sample.mean()
    # Absolute weights compose with DDP and gradient accumulation. Dividing by
    # local sum(weight) cancels the teacher's influence at batch_size=1.
    return (per_sample * weight).mean()


def _resolve_mask(token_mask: Optional[Tensor], mask: Optional[Tensor]) -> Optional[Tensor]:
    if token_mask is not None and mask is not None:
        raise ValueError("pass only one of token_mask or mask; mask is an alias for token_mask")
    return token_mask if token_mask is not None else mask


def _as_tensor(value: TensorOrScalar, reference: Tensor, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {value.device}")
        return value
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return torch.as_tensor(float(value), device=reference.device, dtype=torch.float32)
    raise TypeError(f"{name} must be a torch.Tensor or finite real scalar, got {type(value).__name__}")


class VelocityMatchingObjective(nn.Module):
    """Match student and teacher flow velocities with ``0.5 * MSE``.

    A token mask of shape ``[B, L]`` is expanded over latent channels.  Any
    other non-negative mask that can broadcast to the prediction is accepted as
    well.  ``teacher_weight`` is a confidence weight per batch item; it affects
    the final batch mean as an absolute multiplier, not the returned ``per_sample``
    values.
    """

    prediction_space = "velocity"
    supports_prevalidated_teacher_weight = True

    def __init__(self, coefficient: float = 0.5) -> None:
        super().__init__()
        if not isinstance(coefficient, Real) or isinstance(coefficient, bool):
            raise TypeError("coefficient must be a finite non-negative real number")
        if not math.isfinite(float(coefficient)) or coefficient < 0:
            raise ValueError("coefficient must be a finite non-negative real number")
        self.coefficient = float(coefficient)

    def forward(
        self,
        student_velocity: Tensor,
        teacher_velocity: Tensor,
        token_mask: Optional[Tensor] = None,
        teacher_weight: Optional[Tensor] = None,
        *,
        mask: Optional[Tensor] = None,
        teacher_weight_prevalidated: bool = False,
    ) -> DistillationResult:
        _validate_prediction_pair(
            student_velocity,
            teacher_velocity,
            student_name="student_velocity",
            teacher_name="teacher_velocity",
        )
        resolved_mask = _prepare_mask(_resolve_mask(token_mask, mask), student_velocity)
        squared_error = (student_velocity.float() - teacher_velocity.detach().float()).square()
        per_sample_mse = _masked_per_sample_mean(squared_error, resolved_mask)
        per_sample = self.coefficient * per_sample_mse
        mse = _batch_mean(per_sample_mse, teacher_weight, prevalidated=teacher_weight_prevalidated)
        loss = self.coefficient * mse
        return DistillationResult(
            loss=loss,
            per_sample=per_sample,
            metrics={"loss": loss.detach(), "mse": mse.detach()},
        )


class TransitionMeanMatchingObjective(nn.Module):
    """Match one-step transition means for ODE or SDE trajectories.

    In ``ode`` mode the objective is ``0.5 * MSE``.  In ``sde`` mode it is the
    reverse-KL mean term for equal-variance transitions,
    ``MSE / (2 * variance)``.  SDE variance can be a positive scalar or a tensor
    broadcastable to the transition mean (including leading shapes such as
    ``[B]`` and ``[B, L]``).
    """

    prediction_space = "transition_mean"
    supports_prevalidated_teacher_weight = True
    MODES = ("ode", "sde")

    def __init__(self, mode: str = "ode") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

    def forward(
        self,
        student_mean: Tensor,
        teacher_mean: Tensor,
        variance: Optional[TensorOrScalar] = None,
        token_mask: Optional[Tensor] = None,
        teacher_weight: Optional[Tensor] = None,
        *,
        mask: Optional[Tensor] = None,
        teacher_weight_prevalidated: bool = False,
    ) -> DistillationResult:
        _validate_prediction_pair(
            student_mean,
            teacher_mean,
            student_name="student_mean",
            teacher_name="teacher_mean",
        )
        resolved_mask = _prepare_mask(_resolve_mask(token_mask, mask), student_mean)
        squared_error = (student_mean.float() - teacher_mean.detach().float()).square()

        metrics = {}
        if self.mode == "ode":
            if variance is not None:
                raise ValueError("variance must be None in ode mode")
            elementwise = 0.5 * squared_error
        else:
            if variance is None:
                raise ValueError("variance is required in sde mode")
            raw_variance = _as_tensor(variance, student_mean, name="variance")
            expanded_variance = _expand_like(raw_variance.detach(), student_mean, name="variance").float()
            if not torch.isfinite(expanded_variance).all().item():
                raise ValueError("variance must contain only finite values")
            if (expanded_variance <= 0).any().item():
                raise ValueError("variance must be strictly positive everywhere")
            elementwise = squared_error / (2.0 * expanded_variance)
            metrics["variance"] = _masked_per_sample_mean(expanded_variance, resolved_mask).mean().detach()

        per_sample_mse = _masked_per_sample_mean(squared_error, resolved_mask)
        per_sample = _masked_per_sample_mean(elementwise, resolved_mask)
        loss = _batch_mean(per_sample, teacher_weight, prevalidated=teacher_weight_prevalidated)
        mse = _batch_mean(per_sample_mse, teacher_weight, prevalidated=teacher_weight_prevalidated)
        metrics.update({"loss": loss.detach(), "mse": mse.detach()})
        if self.mode == "sde":
            metrics["reverse_kl"] = loss.detach()
        return DistillationResult(loss=loss, per_sample=per_sample, metrics=metrics)


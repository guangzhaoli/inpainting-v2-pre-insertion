"""Shared, framework-light types for online policy distillation.

The types in this module deliberately depend only on PyTorch and the standard
library.  They can therefore be shared by different teacher backends and OPD
strategies without coupling those implementations to a particular trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowTrajectory:
    """A detached batch of states sampled along a flow trajectory.

    ``states[:, i]`` is the state at trajectory boundary ``i``.  Consequently,
    a trajectory with ``T`` transitions stores ``T + 1`` states and ``T``
    transition timesteps.  Timesteps may either be shared by the whole batch
    (shape ``[T]``) or sampled per item (shape ``[B, T]``).

    All tensors are detached in ``__post_init__``.  This makes the type safe to
    retain while the student loss is constructed: a teacher/sampler autograd
    graph can never accidentally be kept alive by the trajectory object.
    """

    states: Tensor
    timesteps: Tensor
    sigmas: Optional[Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.states, Tensor):
            raise TypeError(f"states must be a torch.Tensor, got {type(self.states).__name__}")
        if not isinstance(self.timesteps, Tensor):
            raise TypeError(f"timesteps must be a torch.Tensor, got {type(self.timesteps).__name__}")
        if self.sigmas is not None and not isinstance(self.sigmas, Tensor):
            raise TypeError(f"sigmas must be a torch.Tensor or None, got {type(self.sigmas).__name__}")

        if self.states.ndim < 2:
            raise ValueError(f"states must have shape [B, T+1, ...], got {tuple(self.states.shape)}")

        batch_size = self.states.shape[0]
        num_boundaries = self.states.shape[1]
        if batch_size <= 0:
            raise ValueError("states must contain at least one batch item")
        if num_boundaries < 2:
            raise ValueError(
                "states must contain at least two trajectory boundaries "
                f"([B, T+1, ...] with T >= 1), got shape {tuple(self.states.shape)}"
            )
        num_steps = num_boundaries - 1

        if self.timesteps.ndim == 1:
            expected = (num_steps,)
        elif self.timesteps.ndim == 2:
            expected = (batch_size, num_steps)
        else:
            raise ValueError(
                "timesteps must have shape [T] or [B, T], "
                f"got {tuple(self.timesteps.shape)} for B={batch_size}, T={num_steps}"
            )
        if tuple(self.timesteps.shape) != expected:
            raise ValueError(
                "timesteps must have shape [T] or [B, T] matching states; "
                f"expected {expected}, got {tuple(self.timesteps.shape)}"
            )

        if self.sigmas is not None:
            expected_sigmas = (num_boundaries,)
            if tuple(self.sigmas.shape) != expected_sigmas:
                raise ValueError(
                    "sigmas must have shape [T+1] matching states; "
                    f"expected {expected_sigmas}, got {tuple(self.sigmas.shape)}"
                )
            if not bool(torch.isfinite(self.sigmas).all().item()):
                raise ValueError("sigmas must contain only finite values")

        if not bool(torch.isfinite(self.timesteps).all().item()):
            raise ValueError("timesteps must contain only finite values")

        object.__setattr__(self, "states", self.states.detach())
        object.__setattr__(self, "timesteps", self.timesteps.detach())
        if self.sigmas is not None:
            object.__setattr__(self, "sigmas", self.sigmas.detach())

    @property
    def batch_size(self) -> int:
        """Number of trajectories in the batch."""

        return self.states.shape[0]

    @property
    def num_steps(self) -> int:
        """Number of transitions (``T``) in each trajectory."""

        return self.states.shape[1] - 1


@dataclass(frozen=True)
class DistillationResult:
    """Uniform output returned by all distillation objectives.

    ``per_sample`` intentionally remains attached to the student graph so a
    strategy can apply an additional curriculum/reduction if needed.  Logging
    values in ``metrics`` are detached by the objective implementations.
    """

    loss: Tensor
    per_sample: Tensor
    metrics: Mapping[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.loss, Tensor):
            raise TypeError(f"loss must be a torch.Tensor, got {type(self.loss).__name__}")
        if self.loss.ndim != 0:
            raise ValueError(f"loss must be scalar, got shape {tuple(self.loss.shape)}")
        if not isinstance(self.per_sample, Tensor):
            raise TypeError(f"per_sample must be a torch.Tensor, got {type(self.per_sample).__name__}")
        if self.per_sample.ndim != 1:
            raise ValueError(f"per_sample must have shape [B], got {tuple(self.per_sample.shape)}")
        if self.loss.device != self.per_sample.device:
            raise ValueError(
                "loss and per_sample must be on the same device, got "
                f"{self.loss.device} and {self.per_sample.device}"
            )
        if not isinstance(self.metrics, Mapping):
            raise TypeError(f"metrics must be a mapping, got {type(self.metrics).__name__}")
        invalid = [key for key in self.metrics if not isinstance(key, str)]
        if invalid:
            raise TypeError(f"metric keys must be strings, got invalid keys: {invalid!r}")

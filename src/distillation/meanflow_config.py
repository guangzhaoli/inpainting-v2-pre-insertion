"""Serializable model and training contracts for MeanFlow + conditional DMD."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FlowMapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = False
    source_time_weight: float = Field(default=0.25, gt=0, lt=1)
    inference_steps: int = Field(default=8, ge=1)


class MeanFlowDMDConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stage: Literal["warmup", "dmd"] = "warmup"
    teacher_steps: int = Field(default=40, ge=2)
    student_steps: int = Field(default=8, ge=1)
    rollout: Literal["full", "shortcut"] = "full"
    gradient_mode: Literal["full", "last"] = "last"
    finite_difference_epsilon: float = Field(default=0.01, gt=0, lt=0.1)
    diagonal_probability: float = Field(default=0.5, ge=0, le=1)
    adaptive_loss_power: float = Field(default=0.0, ge=0, le=1)
    forward_weight: float = Field(default=1.0, ge=0)
    dmd_weight: float = Field(default=1.0, gt=0)
    keep_weight: float = Field(default=0.0, ge=0)
    foreground_weight: float = Field(default=3.0, gt=0)
    background_weight: float = Field(default=1.0, gt=0)
    tau_min: float = Field(default=0.02, gt=0, lt=1)
    tau_max: float = Field(default=0.98, gt=0, lt=1)
    normalize_dmd: bool = True
    fake_updates: int = Field(default=1, ge=1)
    fake_lr: float = Field(default=1e-4, gt=0)
    fake_adapter: str = "meanflow_fake_score"
    teacher_score_probe_interval: int = Field(default=0, ge=0)
    anchoring: Literal["disabled", "deployment"] = "deployment"

    @model_validator(mode="after")
    def validate_ranges(self):
        if self.tau_min >= self.tau_max:
            raise ValueError("tau_min must be smaller than tau_max")
        if self.stage == "warmup" and self.forward_weight <= 0:
            raise ValueError("warmup requires positive forward_weight")
        if self.teacher_score_probe_interval and self.forward_weight <= 0:
            raise ValueError(
                "teacher score probes require online targets (forward_weight > 0)"
            )
        if self.stage == "warmup" and self.keep_weight:
            raise ValueError("keep_weight is a DMD-stage auxiliary loss")
        if self.rollout == "shortcut" and self.gradient_mode != "full":
            raise ValueError("AnyFlow shortcut requires gradient_mode=full")
        if not self.fake_adapter.strip():
            raise ValueError("fake_adapter must be non-empty")
        return self

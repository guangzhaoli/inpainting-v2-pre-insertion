import pytest
import torch

from src.distillation import (
    DistillationResult,
    FlowTrajectory,
    TransitionMeanMatchingObjective,
    VelocityMatchingObjective,
)


def test_flow_trajectory_accepts_shared_or_per_sample_timesteps_and_detaches():
    states = torch.randn(2, 4, 5, requires_grad=True) * 2
    shared = torch.arange(3.0, requires_grad=True)
    sigmas = torch.linspace(1.0, 0.0, 4, requires_grad=True)

    trajectory = FlowTrajectory(states=states, timesteps=shared, sigmas=sigmas)

    assert trajectory.batch_size == 2
    assert trajectory.num_steps == 3
    assert trajectory.states.shape == (2, 4, 5)
    assert trajectory.timesteps.shape == (3,)
    assert trajectory.sigmas is not None and trajectory.sigmas.shape == (4,)
    assert not trajectory.states.requires_grad and trajectory.states.grad_fn is None
    assert not trajectory.timesteps.requires_grad and trajectory.timesteps.grad_fn is None
    assert not trajectory.sigmas.requires_grad and trajectory.sigmas.grad_fn is None

    per_sample = FlowTrajectory(states=torch.randn(2, 4, 5), timesteps=torch.randn(2, 3))
    assert per_sample.timesteps.shape == (2, 3)


@pytest.mark.parametrize(
    ("states", "timesteps", "sigmas", "message"),
    [
        (torch.randn(2, 1, 4), torch.empty(0), None, "at least two"),
        (torch.randn(2, 4, 4), torch.randn(4), None, "timesteps"),
        (torch.randn(2, 4, 4), torch.randn(3, 1), None, "timesteps"),
        (torch.randn(2, 4, 4), torch.randn(3), torch.randn(3), "sigmas"),
        (torch.randn(2, 4, 4), torch.randn(2, 3, 1), None, "timesteps"),
    ],
)
def test_flow_trajectory_rejects_inconsistent_shapes(states, timesteps, sigmas, message):
    with pytest.raises(ValueError, match=message):
        FlowTrajectory(states=states, timesteps=timesteps, sigmas=sigmas)


def test_flow_trajectory_rejects_non_finite_schedule_values():
    with pytest.raises(ValueError, match="finite"):
        FlowTrajectory(
            states=torch.randn(1, 3, 2),
            timesteps=torch.tensor([float("nan"), 0.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        FlowTrajectory(
            states=torch.randn(1, 3, 2),
            timesteps=torch.tensor([1.0, 0.0]),
            sigmas=torch.tensor([1.0, float("inf"), 0.0]),
        )


def test_velocity_matching_defaults_to_half_mse_and_keeps_student_gradient():
    student = torch.tensor([[1.0, 3.0], [2.0, 6.0]], requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0], [0.0, 2.0]], requires_grad=True)

    result = VelocityMatchingObjective()(student, teacher)

    expected_per_sample = torch.tensor([1.25, 5.0])
    assert isinstance(result, DistillationResult)
    assert torch.allclose(result.per_sample, expected_per_sample)
    assert torch.allclose(result.loss, expected_per_sample.mean())
    assert torch.allclose(result.metrics["mse"], torch.tensor(6.25))
    result.loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert all(not value.requires_grad for value in result.metrics.values())


def test_velocity_matching_supports_token_mask_broadcast_and_teacher_weight():
    student = torch.tensor(
        [
            [[1.0, 3.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        requires_grad=True,
    )
    teacher = torch.zeros_like(student)
    token_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher_weight = torch.tensor([1.0, 3.0])

    result = VelocityMatchingObjective()(
        student,
        teacher,
        token_mask=token_mask,
        teacher_weight=teacher_weight,
    )

    # sample losses: 0.5 * mean([1, 9]) = 2.5 and
    # 0.5 * mean([36, 64]) = 25; weights are absolute, then averaged over samples.
    assert torch.allclose(result.per_sample, torch.tensor([2.5, 25.0]))
    assert torch.allclose(result.loss, torch.tensor((2.5 + 3 * 25.0) / 2.0))

    broadcast_mask = torch.ones(2, 1, 1)
    full = VelocityMatchingObjective()(student, teacher, mask=broadcast_mask)
    assert full.per_sample.shape == (2,)


@pytest.mark.parametrize(
    ("mask", "weight", "message"),
    [
        (torch.ones(3, 2), None, "cannot broadcast"),
        (torch.tensor([[0.0, 0.0], [1.0, 1.0]]), None, "empty samples"),
        (torch.tensor([[1.0, -1.0], [1.0, 1.0]]), None, "non-negative"),
        (None, torch.ones(2, 1), "shape"),
        (None, torch.tensor([0.0, 0.0]), "positive sum"),
    ],
)
def test_velocity_matching_rejects_invalid_masks_and_weights(mask, weight, message):
    with pytest.raises(ValueError, match=message):
        VelocityMatchingObjective()(
            torch.randn(2, 2, 3),
            torch.randn(2, 2, 3),
            token_mask=mask,
            teacher_weight=weight,
        )


def test_transition_mean_ode_is_half_mse():
    student = torch.tensor([[[1.0], [3.0]], [[2.0], [2.0]]])
    teacher = torch.zeros_like(student)

    result = TransitionMeanMatchingObjective(mode="ode")(student, teacher)

    assert torch.allclose(result.per_sample, torch.tensor([2.5, 2.0]))
    assert torch.allclose(result.loss, torch.tensor(2.25))
    assert torch.allclose(result.metrics["mse"], torch.tensor(4.5))


def test_transition_mean_sde_is_reverse_kl_and_broadcasts_per_sample_variance():
    student = torch.tensor([[[2.0], [4.0]], [[3.0], [6.0]]], requires_grad=True)
    teacher = torch.zeros_like(student)
    variance = torch.tensor([2.0, 3.0])

    result = TransitionMeanMatchingObjective(mode="sde")(student, teacher, variance=variance)

    # sample 0: mean([4, 16] / 4) = 2.5
    # sample 1: mean([9, 36] / 6) = 3.75
    assert torch.allclose(result.per_sample, torch.tensor([2.5, 3.75]))
    assert torch.allclose(result.loss, torch.tensor(3.125))
    assert torch.allclose(result.metrics["reverse_kl"], result.loss.detach())


@pytest.mark.parametrize(
    ("mode", "variance", "message"),
    [
        ("sde", None, "required"),
        ("sde", 0.0, "strictly positive"),
        ("sde", torch.tensor([1.0, -1.0]), "strictly positive"),
        ("sde", torch.ones(4), "cannot broadcast"),
        ("sde", float("nan"), "finite"),
        ("ode", 1.0, "must be None"),
    ],
)
def test_transition_mean_strictly_validates_variance(mode, variance, message):
    objective = TransitionMeanMatchingObjective(mode=mode)
    with pytest.raises(ValueError, match=message):
        objective(torch.randn(2, 2, 3), torch.randn(2, 2, 3), variance=variance)


def test_objectives_reject_prediction_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        VelocityMatchingObjective()(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(ValueError, match="same shape"):
        TransitionMeanMatchingObjective()(torch.randn(2, 3), torch.randn(2, 3, 1))

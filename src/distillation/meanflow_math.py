"""Flow-map and conditional DMD primitives (linear rectified-flow convention).

Times here are actual sigmas, not indices in a shifted inference schedule.
The score networks predict instantaneous velocity; the generator predicts an
interval-average velocity. All score targets are detached from generator loss.
"""

import torch
import torch.distributed as dist


def expand_time(time, like):
    time = torch.as_tensor(time, device=like.device, dtype=torch.float32)
    if time.ndim == 0:
        time = time.expand(like.shape[0])
    if time.shape != (like.shape[0],):
        raise ValueError("time must be scalar or one value per sample")
    return time.reshape(-1, *([1] * (like.ndim - 1)))


def flow_step(state, average_velocity, start, end):
    if state.shape != average_velocity.shape:
        raise ValueError("state and average velocity shapes differ")
    return (
        state.float()
        + (expand_time(end, state) - expand_time(start, state))
        * average_velocity.float()
    )


def add_noise(clean, noise, sigma):
    if clean.shape != noise.shape:
        raise ValueError("clean and noise shapes differ")
    s = expand_time(sigma, clean)
    return (1 - s) * clean.float() + s * noise.float()


def velocity_to_score(noisy, velocity, sigma):
    s = expand_time(sigma, noisy)
    if bool(((s <= 0) | (s >= 1)).any()):
        raise ValueError("score conversion requires 0 < sigma < 1")
    return -(noisy.float() + (1 - s) * velocity.float()) / s


def dmd_surrogate(
    clean, noisy, teacher_velocity, fake_velocity, sigma, *, normalize=True
):
    """DMD x0-difference gradient, with the optional official DMD normalization.

    x0_fake - x0_teacher = sigma * (v_teacher - v_fake).
    This is a positive rescaling of score_fake - score_teacher. The surrogate
    gradient is exactly this detached direction divided by batch/element count.
    """
    s = expand_time(sigma, clean)
    if bool(((s <= 0) | (s >= 1)).any()):
        raise ValueError("DMD requires interior re-noising times")
    real_x0 = noisy.float() - s * teacher_velocity.detach().float()
    fake_x0 = noisy.float() - s * fake_velocity.detach().float()
    direction = fake_x0 - real_x0
    if normalize:
        scale = (clean.detach().float() - real_x0).abs().flatten(1).mean(1)
        direction = direction / expand_time(scale.clamp_min(1e-5), clean)
    if not bool(torch.isfinite(direction).all()):
        raise FloatingPointError("Non-finite conditional DMD gradient")
    target = (clean.detach().float() - direction).detach()
    loss = 0.5 * (clean.float() - target).square().mean()
    return loss, direction.detach()


def meanflow_loss(
    predict, clean, noise, start, end, *, epsilon=1e-3, adaptive_power=0.0
):
    """AnyFlow-style finite-difference MeanFlow target, guidance already absorbed.

    The tangent is epsilon_noise - teacher_clean. Central differences remain
    inside [0,1] and keep end <= start-epsilon. Diagonal samples need no JVP.
    """
    x = add_noise(clean.detach(), noise, start)
    v = noise.float() - clean.detach().float()
    t = torch.as_tensor(start, device=x.device, dtype=torch.float32).reshape(-1)
    r = torch.as_tensor(end, device=x.device, dtype=torch.float32).reshape(-1)
    if t.numel() == 1:
        t = t.expand(x.shape[0])
    if r.numel() == 1:
        r = r.expand(x.shape[0])
    if bool(((r < 0) | (t > 1) | (r > t)).any()):
        raise ValueError("MeanFlow requires 0 <= end <= start <= 1")
    gap = t - r
    h = torch.minimum(torch.full_like(t, epsilon), torch.minimum(t, 1 - t) / 2)
    h = torch.minimum(h, gap / 2)
    active = h > 0
    safe_h = h.clamp_min(torch.finfo(torch.float32).eps)
    target = v
    if bool(active.any()):
        with torch.no_grad():
            plus = predict(x + expand_time(h, x) * v, t + h, r)
            minus = predict(x - expand_time(h, x) * v, t - h, r)
            derivative = (plus.float() - minus.float()) / (2 * expand_time(safe_h, x))
            derivative = derivative * expand_time(active.float(), x)
            target = v - expand_time(gap, x) * derivative
    predicted = predict(x, t, r)
    error = (predicted.float() - target.detach()).square().flatten(1).mean(1)
    weights = (error.detach() + 1e-3).pow(-adaptive_power)
    return (error * weights).mean()


def anchor_step(generated, source, noise, sigma, edit_weight):
    if not (generated.shape == source.shape == noise.shape == edit_weight.shape):
        raise ValueError("anchoring tensors must share packed latent geometry")
    return torch.lerp(
        add_noise(source, noise, sigma), generated.float(), edit_weight.float()
    )


def synchronize_gradients(parameters, bucket_bytes=16 * 1024 * 1024):
    """Average gradients explicitly for shared G/F adapters without DDP hooks.

    Every rank must provide the same parameter list. A missing local gradient
    contributes zero, including ranks without a particular valid loss region.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    parameters = list(parameters)
    if not parameters:
        return
    used = torch.tensor(
        [p.grad is not None for p in parameters],
        device=parameters[0].device,
        dtype=torch.int32,
    )
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    active = []
    for p, present in zip(parameters, used.tolist()):
        if present:
            active.append(p)
        else:
            # Preserve AdamW's skip semantics for globally unused parameters.
            p.grad = None
    parameters = active
    bucket, size = [], 0

    def flush(items):
        if not items:
            return
        flat = torch.cat(
            [
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in items
            ]
        )
        dist.all_reduce(flat)
        flat.div_(dist.get_world_size())
        offset = 0
        for p in items:
            value = flat[offset : offset + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
            offset += p.numel()

    for p in parameters:
        if bucket and (
            size + p.numel() * p.element_size() > bucket_bytes
            or p.dtype != bucket[0].dtype
        ):
            flush(bucket)
            bucket, size = [], 0
        bucket.append(p)
        size += p.numel() * p.element_size()
    flush(bucket)

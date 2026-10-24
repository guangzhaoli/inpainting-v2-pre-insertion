"""Fused elementwise kernels of the Qwen-Image block for the fast canvas sampler (Triton, CUDA).

Each kernel rounds to bf16 where the eager block does (after the LayerNorm, ``1 + scale``, the
product, the sum, ...), so it matches the eager ops up to fp32 reduction order. Every op has a
torch reference with the same semantics (``*_reference``), used on CPU and by the tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments use the references
    triton = None


def available(tensor: torch.Tensor) -> bool:
    return triton is not None and tensor.is_cuda and tensor.dtype == torch.bfloat16


if triton is not None:

    @triton.jit
    def _round_bf16(x):
        """fp32 -> nearest-even bf16 value, kept in fp32. Integer ops, because the compiler folds a
        plain ``x.to(bfloat16).to(float32)`` round trip away (finite inputs only)."""

        bits = x.to(tl.uint32, bitcast=True)
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return bits.to(tl.float32, bitcast=True)


# --------------------------------------------------------------------------------------------
# LayerNorm (no affine) + modulation: round(round(round(LN(x)) * round(1 + scale)) + shift)


def ln_modulate_reference(x, shift, scale, eps):
    normed = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return normed * (1 + scale) + shift


if triton is not None:

    @triton.jit
    def _ln_modulate_kernel(x_ptr, shift_ptr, scale_ptr, y_ptr, stride_x, stride_y, width, eps,
                            BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < width
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / width
        centered = tl.where(mask, x - mean, 0.0)
        rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / width + eps)
        normed = _round_bf16(centered * rstd)
        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        product = _round_bf16(normed * _round_bf16(1.0 + scale))
        tl.store(y_ptr + row * stride_y + cols, (product + shift).to(tl.bfloat16), mask=mask)


def ln_modulate(x, shift, scale, eps, out=None):
    """``x`` [M, D]; ``shift``/``scale`` [D] -> [M, D] (into ``out`` if given)."""

    if not available(x):
        result = ln_modulate_reference(x, shift, scale, eps)
        if out is None:
            return result
        out.copy_(result)
        return out
    rows, width = x.shape
    out = torch.empty_like(x) if out is None else out
    if x.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("ln_modulate needs a contiguous last dimension")
    block = triton.next_power_of_2(width)
    _ln_modulate_kernel[(rows,)](x, shift.contiguous(), scale.contiguous(), out, x.stride(0), out.stride(0), width,
                                 eps, BLOCK=block, num_warps=8)
    return out


# --------------------------------------------------------------------------------------------
# gate * values + residual, each product and sum rounded like the eager ops


def gate_residual_reference(residual, values, gate):
    return residual + gate * values


if triton is not None:

    @triton.jit
    def _gate_residual_kernel(residual_ptr, values_ptr, gate_ptr, y_ptr, stride_r, stride_v, stride_y, width,
                              BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        residual = tl.load(residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        values = tl.load(values_ptr + row * stride_v + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride_y + cols, (residual + _round_bf16(gate * values)).to(tl.bfloat16), mask=mask)


def gate_residual(residual, values, gate, out=None):
    """``residual``/``values`` [M, D], ``gate`` [D] -> ``residual + gate * values``."""

    if not available(residual):
        result = gate_residual_reference(residual, values, gate)
        if out is None:
            return result
        out.copy_(result)
        return out
    rows, width = residual.shape
    out = torch.empty_like(residual) if out is None else out
    block = 1024
    _gate_residual_kernel[(rows, triton.cdiv(width, block))](
        residual, values, gate.contiguous(), out, residual.stride(0), values.stride(0), out.stride(0), width,
        BLOCK=block, num_warps=4,
    )
    return out


# --------------------------------------------------------------------------------------------
# q/k RMSNorm + RoPE of a fused q|k|v projection, scattered into the packed query and key rows


def qk_norm_rope_reference(qkv, q_weight, k_weight, cos, sin, eps, heads):
    """``qkv`` [M, 3D] -> (q, k, v) [M, H, Dh]: diffusers RMSNorm (fp32 statistics, bf16 rounding,
    weight) and ``apply_rotary_emb_qwen(use_real=False)`` with ``freqs = cos + i sin``."""

    rows = qkv.shape[0]
    q, k, v = qkv.view(rows, 3 * heads, -1).chunk(3, dim=1)

    def norm(x, weight):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = (x * torch.rsqrt(variance + eps)).to(weight.dtype)
        return x * weight

    def rope(x):
        x0, x1 = x.float().unflatten(-1, (-1, 2)).unbind(-1)
        c, s = cos[:, None], sin[:, None]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2).to(x.dtype)

    return rope(norm(q, q_weight)), rope(norm(k, k_weight)), v


if triton is not None:

    @triton.jit
    def _qk_norm_rope_kernel(qkv_ptr, stride_qkv, q_weight_ptr, k_weight_ptr, cos_ptr, sin_ptr, stride_freq,
                             q_rows_ptr, k_rows_ptr, q_out_ptr, k_out_ptr, v_out_ptr, stride_q, stride_k, stride_v,
                             width, eps, HEADS: tl.constexpr, HEAD_BLOCK: tl.constexpr, HEAD_DIM: tl.constexpr):
        row = tl.program_id(0)
        heads = tl.arange(0, HEAD_BLOCK)[:, None]
        pairs = tl.arange(0, HEAD_DIM // 2)[None, :]
        mask = heads < HEADS
        offsets = heads * HEAD_DIM + 2 * pairs
        cos = tl.load(cos_ptr + row * stride_freq + pairs)
        sin = tl.load(sin_ptr + row * stride_freq + pairs)
        q_row = tl.load(q_rows_ptr + row).to(tl.int64)
        k_row = tl.load(k_rows_ptr + row).to(tl.int64)
        source = qkv_ptr + row * stride_qkv
        for part in tl.static_range(2):
            even = tl.load(source + part * width + offsets, mask=mask, other=0.0).to(tl.float32)
            odd = tl.load(source + part * width + offsets + 1, mask=mask, other=0.0).to(tl.float32)
            variance = (tl.sum(even * even, axis=1) + tl.sum(odd * odd, axis=1)) / HEAD_DIM
            rstd = tl.rsqrt(variance + eps)[:, None]
            weight_ptr = q_weight_ptr if part == 0 else k_weight_ptr
            w_even = tl.load(weight_ptr + 2 * pairs).to(tl.float32)
            w_odd = tl.load(weight_ptr + 2 * pairs + 1).to(tl.float32)
            even = _round_bf16(_round_bf16(even * rstd) * w_even)
            odd = _round_bf16(_round_bf16(odd * rstd) * w_odd)
            out_even = (even * cos - odd * sin).to(tl.bfloat16)
            out_odd = (even * sin + odd * cos).to(tl.bfloat16)
            if part == 0:
                target = q_out_ptr + q_row * stride_q
            else:
                target = k_out_ptr + k_row * stride_k
            tl.store(target + offsets, out_even, mask=mask)
            tl.store(target + offsets + 1, out_odd, mask=mask)
        dims = tl.arange(0, HEAD_DIM)[None, :]
        value = tl.load(source + 2 * width + heads * HEAD_DIM + dims, mask=mask)
        tl.store(v_out_ptr + k_row * stride_v + heads * HEAD_DIM + dims, value, mask=mask)


def qk_norm_rope_scatter(qkv, q_weight, k_weight, cos, sin, eps, heads, q_rows, k_rows, q_out, k_out, v_out):
    """Normalize and rotate q and k of ``qkv`` [M, 3D] (RoPE ``cos``/``sin`` [M, Dh/2]) and write row
    ``i`` to ``q_out[q_rows[i]]``, ``k_out[k_rows[i]]`` and its v to ``v_out[k_rows[i]]`` (all [N, H, Dh])."""

    if not available(qkv):
        q, k, v = qk_norm_rope_reference(qkv, q_weight, k_weight, cos, sin, eps, heads)
        q_out[q_rows.long()] = q
        k_out[k_rows.long()] = k
        v_out[k_rows.long()] = v
        return
    rows, width3 = qkv.shape
    width = width3 // 3
    head_dim = width // heads
    for tensor in (q_out, k_out, v_out):
        if tensor.stride(-1) != 1 or tensor.stride(-2) != head_dim:
            raise ValueError("qk_norm_rope_scatter needs [N, H, Dh] outputs with contiguous heads")
    _qk_norm_rope_kernel[(rows,)](
        qkv, qkv.stride(0), q_weight, k_weight, cos, sin, cos.stride(0), q_rows, k_rows, q_out, k_out, v_out,
        q_out.stride(0), k_out.stride(0), v_out.stride(0), width, eps, HEADS=heads,
        HEAD_BLOCK=triton.next_power_of_2(heads), HEAD_DIM=head_dim, num_warps=4,
    )

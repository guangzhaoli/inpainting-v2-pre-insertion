# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# follow from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _dispatch_attention_compat(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    backend=None,
):
    """Fall back to plain SDPA on torch builds that do not support newer kwargs."""
    try:
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            backend=backend,
        )
    except TypeError as exc:
        if "enable_gqa" not in str(exc):
            raise

        query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        out = F.scaled_dot_product_attention(
            query=query_t,
            key=key_t,
            value=value_t,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return out.permute(0, 2, 1, 3)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    if len(timesteps.shape) != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


@dataclass
class ConditionKVCache:
    """Per-block post-RoPE K/V of isolated conditions, reused at every sampling step.

    ``layers[block][group]`` is ``(key, value)`` shaped ``[B, N_group, heads, head_dim]``;
    ``segment_lengths`` are the attention groups (one per image, or one for all images),
    ``image_lengths`` the per-image token counts. ``text_layers[block]`` holds the text
    K/V when the text stream is a condition.
    """

    segment_lengths: Tuple[int, ...]
    layers: List[List[Tuple[torch.Tensor, torch.Tensor]]]
    image_lengths: Optional[Tuple[int, ...]] = None
    text_layers: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    def __post_init__(self):
        if self.image_lengths is None:
            self.image_lengths = tuple(self.segment_lengths)

    @property
    def per_image(self) -> bool:
        return tuple(self.segment_lengths) == tuple(self.image_lengths)

    def select(self, images: Sequence[int]) -> "ConditionKVCache":
        """Keep only some condition images, e.g. the source for the null-reference CFG branch."""

        if not self.per_image or self.text_layers is not None:
            raise ValueError("only a per-image condition cache without text can drop images")
        return ConditionKVCache(
            tuple(self.segment_lengths[image] for image in images),
            [[layer[image] for image in images] for layer in self.layers],
        )

    @property
    def nbytes(self) -> int:
        tensors = [tensor for layer in self.layers for kv in layer for tensor in kv]
        tensors += [tensor for kv in self.text_layers or () for tensor in kv]
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


try:
    from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward, _wrapped_flash_attn_forward
except ImportError:  # CPU-only environments use the reference attention below
    _wrapped_flash_attn_backward = _wrapped_flash_attn_forward = None


def _lse_attention_forward(query, key, value, scale):
    """Softmax attention on ``[B, S, H, D]`` tensors; also returns the fp32 log-sum-exp ``[B, H, S]``."""

    if _uses_flash(query):
        out, lse, _, _ = _wrapped_flash_attn_forward(query, key, value, 0.0, scale, False, -1, -1, 0.0, None, False)
        return out, lse
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", torch.exp(scores - lse[..., None]), value.float())
    return out.to(query.dtype), lse


def _lse_attention_backward(grad_out, query, key, value, out, lse, scale):
    """Gradients of one key block of a softmax whose output ``out`` and log-sum-exp ``lse`` span all blocks."""

    if _uses_flash(query):
        grads = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in (query, key, value)]
        _wrapped_flash_attn_backward(
            grad_out, query, key, value, out, lse, *grads, 0.0, scale, False, -1, -1, 0.0, None, False, None
        )
        return grads
    scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * scale
    probs = torch.exp(scores - lse[..., None])
    grad_out = grad_out.float()
    grad_value = torch.einsum("bhqk,bqhd->bkhd", probs, grad_out)
    delta = (grad_out * out.float()).sum(-1).transpose(1, 2)
    grad_scores = probs * (torch.einsum("bqhd,bkhd->bhqk", grad_out, value.float()) - delta[..., None])
    grad_query = torch.einsum("bhqk,bkhd->bqhd", grad_scores, key.float()) * scale
    grad_key = torch.einsum("bhqk,bqhd->bkhd", grad_scores, query.float()) * scale
    return grad_query.to(query.dtype), grad_key.to(key.dtype), grad_value.to(value.dtype)


def _uses_flash(query: torch.Tensor) -> bool:
    return _wrapped_flash_attn_forward is not None and query.is_cuda and query.dtype in (torch.float16, torch.bfloat16)


class _BiasedTwoBlockAttention(torch.autograd.Function):
    """Softmax attention over keys ``[a; b]`` with a logit bias of ``-bias`` on block ``b``.

    Two attention calls merged with their log-sum-exps. The backward runs each block's
    attention backward with the merged output and log-sum-exp (shifted by the bias for
    ``b``), which is the exact gradient of the joint biased softmax.
    """

    @staticmethod
    def forward(ctx, query, key_a, value_a, key_b, value_b, bias):
        query, key_a, value_a, key_b, value_b = (x.contiguous() for x in (query, key_a, value_a, key_b, value_b))
        scale = query.shape[-1] ** -0.5
        out_a, lse_a = _lse_attention_forward(query, key_a, value_a, scale)
        out_b, lse_b = _lse_attention_forward(query, key_b, value_b, scale)
        lse_b = lse_b - bias
        lse = torch.logaddexp(lse_a, lse_b)
        weight_a = torch.exp(lse_a - lse).transpose(1, 2).unsqueeze(-1)
        weight_b = torch.exp(lse_b - lse).transpose(1, 2).unsqueeze(-1)
        out = (weight_a * out_a.float() + weight_b * out_b.float()).to(query.dtype)
        ctx.save_for_backward(query, key_a, value_a, key_b, value_b, out, lse)
        ctx.bias, ctx.scale = bias, scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        query, key_a, value_a, key_b, value_b, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_query_a, grad_key_a, grad_value_a = _lse_attention_backward(grad_out, query, key_a, value_a, out, lse, ctx.scale)
        grad_query_b, grad_key_b, grad_value_b = _lse_attention_backward(
            grad_out, query, key_b, value_b, out, lse + ctx.bias, ctx.scale
        )
        return grad_query_a + grad_query_b, grad_key_a, grad_value_a, grad_key_b, grad_value_b, None


class QwenTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep, hidden_states):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb

        return conditioning


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = OrderedDict()

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        if dim % 2 != 0:
            raise ValueError("rope dimension must be even")
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    @torch.inference_mode(False)
    @torch.no_grad()
    def forward(self, video_fhw, txt_seq_lens, device, token_index=None):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        token_index: optional per-image 1-D indices into that image's row-major grid
            (complementary canvas); missing or ``None`` entries keep the full grid.
        """
        # These cached constants survive validation and are reused by training.
        # Always create ordinary tensors, even inside an inference-mode caller;
        # autograd must save the rotary constants to differentiate the input.
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)
            self.rope_cache.clear()

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = (idx, frame, height, width)

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                    if len(self.rope_cache) > 64:
                        self.rope_cache.popitem(last=False)
                self.rope_cache.move_to_end(rope_key)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            if token_index is not None and idx < len(token_index) and token_index[idx] is not None:
                video_freq = video_freq[token_index[idx].to(device)]
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    @staticmethod
    def _image_qkv(attn: Attention, hidden_states: torch.Tensor, img_freqs: Optional[torch.Tensor]):
        img_query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        img_key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        img_value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if img_freqs is not None:
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
        return img_query, img_key, img_value

    @staticmethod
    def _text_qkv(attn: Attention, encoder_hidden_states: torch.Tensor, txt_freqs: Optional[torch.Tensor]):
        txt_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        txt_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)
        if txt_freqs is not None:
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
        return txt_query, txt_key, txt_value

    def _attend(self, query, key, value, attention_mask=None):
        return _dispatch_attention_compat(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

    def _attend_isolated(self, query, keys, values, other_keys, other_values, bias):
        """Attend to ``keys``; with a ``bias`` also to ``other_keys`` with logit bias ``-bias``."""

        key, value = (keys[0], values[0]) if len(keys) == 1 else (torch.cat(keys, dim=1), torch.cat(values, dim=1))
        if bias is None:
            return self._attend(query, key, value)
        return _BiasedTwoBlockAttention.apply(
            query, key, value, torch.cat(other_keys, dim=1), torch.cat(other_values, dim=1), float(bias)
        )

    def condition_attention(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        img_freqs: torch.Tensor,
        segments: Tuple[int, ...],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        txt_freqs: Optional[torch.Tensor] = None,
    ):
        """Self-attention inside each isolated condition group, and for a text condition, text over text and
        conditions. Returns both outputs, each group's post-RoPE K and V, and the text K and V."""

        img_query, img_key, img_value = self._image_qkv(attn, hidden_states, img_freqs)
        outputs, condition_kv, start = [], [], 0
        txt_attn_output = text_kv = None
        if encoder_hidden_states is not None:
            txt_query, txt_key, txt_value = self._text_qkv(attn, encoder_hidden_states, txt_freqs)
            text_kv = (txt_key, txt_value)
            txt_attn_output = self._attend(
                txt_query, torch.cat([txt_key, img_key], dim=1), torch.cat([txt_value, img_value], dim=1)
            )
            txt_attn_output = attn.to_add_out(txt_attn_output.flatten(2, 3).to(txt_query.dtype))
        for length in segments:
            key, value = img_key[:, start : start + length], img_value[:, start : start + length]
            condition_kv.append((key, value))
            if length:
                outputs.append(self._attend(img_query[:, start : start + length], key, value))
            start += length
        img_attn_output = torch.cat(outputs, dim=1).flatten(2, 3).to(img_query.dtype)
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        return img_attn_output, txt_attn_output, condition_kv, text_kv

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        condition_segments: Optional[Tuple[int, ...]] = None,
        condition_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        isolation_bias: Optional[float] = None,
        text_as_condition: bool = False,
        text_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        """``condition_segments`` (lengths of the trailing condition groups: one per image, or one for
        all images) isolates each group to self-attention while target queries, and text queries
        unless ``text_as_condition`` (then text attends to text and conditions), see every key.
        ``condition_kv`` appends cached condition keys/values for a target-only stream; ``text_kv``
        replaces an absent text stream by its cached keys/values. ``isolation_bias`` softens the
        isolation into a logit bias of minus that value on the otherwise excluded keys."""
        img_freqs, txt_freqs = image_rotary_emb if image_rotary_emb is not None else (None, None)

        # Image stream (sample projections): QKV, QK normalization and RoPE
        img_query, img_key, img_value = self._image_qkv(attn, hidden_states, img_freqs)

        # Text stream (context projections), or its cached keys/values
        if encoder_hidden_states is not None:
            txt_query, txt_key, txt_value = self._text_qkv(attn, encoder_hidden_states, txt_freqs)
        elif text_kv is not None:
            txt_query, (txt_key, txt_value) = None, text_kv
        else:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")
        seq_txt = 0 if txt_query is None else txt_query.shape[1]

        if condition_segments is not None or condition_kv is not None:
            if attention_mask is not None:
                raise ValueError("isolated condition attention does not take an attention mask")
            # FA2 has no block mask: one call per query group over the keys it may see.
            target_count = img_query.shape[1] - sum(condition_segments or ())
            cached_keys = [key for key, _ in condition_kv or ()]
            cached_values = [value for _, value in condition_kv or ()]
            all_keys = torch.cat([txt_key, img_key, *cached_keys], dim=1)
            all_values = torch.cat([txt_value, img_value, *cached_values], dim=1)
            target_query = img_query[:, :target_count]
            if txt_query is not None and not text_as_condition:
                outputs = [self._attend(torch.cat([txt_query, target_query], dim=1), all_keys, all_values)]
            else:
                outputs = []
                if txt_query is not None:
                    outputs.append(
                        self._attend_isolated(
                            txt_query,
                            [txt_key, img_key[:, target_count:], *cached_keys],
                            [txt_value, img_value[:, target_count:], *cached_values],
                            [img_key[:, :target_count]],
                            [img_value[:, :target_count]],
                            isolation_bias,
                        )
                    )
                outputs.append(self._attend(target_query, all_keys, all_values))
            start = target_count
            for length in condition_segments or ():
                if length:
                    end = start + length
                    outputs.append(
                        self._attend_isolated(
                            img_query[:, start:end],
                            [img_key[:, start:end]],
                            [img_value[:, start:end]],
                            [txt_key, img_key[:, :start], img_key[:, end:]],
                            [txt_value, img_value[:, :start], img_value[:, end:]],
                            isolation_bias,
                        )
                    )
                start += length
            joint_hidden_states = torch.cat(outputs, dim=1).flatten(2, 3).to(img_query.dtype)
        else:
            if txt_query is None:
                raise ValueError("cached text keys/values require isolated conditions")
            # Joint attention, order: [text, image]
            joint_query = torch.cat([txt_query, img_query], dim=1)
            joint_hidden_states = self._attend(
                joint_query,
                torch.cat([txt_key, img_key], dim=1),
                torch.cat([txt_value, img_value], dim=1),
                attention_mask,
            )
            joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output) if txt_query is not None else None

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6,
        zero_cond_t: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.zero_cond_t = zero_cond_t
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params, target_token_count=None):
        """Apply modulation to input tensor.

        ``target_token_count`` (an int, or a tuple of prefix group lengths such as
        ``(core, ring)``) splits the tokens into consecutive groups with the
        conditions last; ``mod_params`` stacks one row block per group.
        """
        if target_token_count is not None:
            lengths = (target_token_count,) if isinstance(target_token_count, int) else tuple(target_token_count)
            lengths = (*lengths, x.shape[1] - sum(lengths))
            if mod_params.shape[0] != len(lengths) * x.shape[0]:
                raise ValueError(f"{len(lengths)} modulation groups need {len(lengths)} timestep rows per sample")
            pieces, gates, start = [], [], 0
            for length, params in zip(lengths, mod_params.chunk(len(lengths), dim=0)):
                piece, gate = self._modulate(x[:, start : start + length], params)
                pieces.append(piece)
                gates.append(gate)
                start += length
            # The groups are always shared prefixes in this trainer. Slicing
            # avoids materializing a per-token modulation index or [B,N,D]
            # gate/shift/scale tensors on every block and diffusion step.
            return torch.cat(pieces, dim=1), (tuple(gates), lengths)
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    @staticmethod
    def _gate(values, gate):
        if isinstance(gate, tuple):
            gates, lengths = gate
            pieces, start = [], 0
            for group_gate, length in zip(gates, lengths):
                pieces.append(values[:, start : start + length] * group_gate)
                start += length
            return torch.cat(pieces, dim=1)
        return gate * values

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        target_token_count: Optional[Union[int, Tuple[int, ...]]] = None,
        skip_text_output: bool = False,
        condition_segments: Optional[Tuple[int, ...]] = None,
        condition_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        isolation_bias: Optional[float] = None,
        text_as_condition: bool = False,
        text_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """``encoder_hidden_states`` is None when the text stream is cached (``text_kv``)."""
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [groups*B, 6*dim]
        if self.zero_cond_t:
            if target_token_count is None:
                raise ValueError("zero_cond_t requires the target image prefix length")
            # The text is modulated with the target's t, or with t=0 when it is a condition.
            batch = hidden_states.shape[0]
            temb = temb[-batch:] if text_as_condition else temb[:batch]
        text = encoder_hidden_states is not None

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [groups*B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1, target_token_count)

        # Process text stream - norm1 + modulation
        txt_modulated = None
        if text:
            txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)  # Each [B, 3*dim]
            txt_normed = self.txt_norm1(encoder_hidden_states)
            txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        isolation_kwargs = {
            name: value
            for name, value in (
                ("condition_segments", condition_segments),
                ("condition_kv", condition_kv),
                ("isolation_bias", isolation_bias),
                ("text_kv", text_kv),
            )
            if value is not None
        }
        if text_as_condition:
            isolation_kwargs["text_as_condition"] = True
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
            **isolation_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + self._gate(img_attn_output, img_gate1)

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2, target_token_count)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + self._gate(img_mlp_output, img_gate2)

        # The final block's text output has no consumer. Earlier blocks still
        # update the text stream because subsequent image attention needs it.
        if text and not skip_text_output:
            encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
            txt_normed2 = self.txt_norm2(encoder_hidden_states)
            txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
            txt_mlp_output = self.txt_mlp(txt_modulated2)
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if text and encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

    def forward_conditions(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        img_freqs: torch.Tensor,
        segments: Tuple[int, ...],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        txt_freqs: Optional[torch.Tensor] = None,
    ):
        """Advance isolated clean condition tokens (and a text condition) one block.

        ``temb`` is t=0 embedded in the joint forward's ``[target groups..., condition]``
        row layout; the condition rows are used, so the modulation matches ``forward``
        bit for bit (bf16 GEMM results depend on the row count). Returns the updated
        condition and text tokens and this block's K/V per condition group and of the text.
        """
        batch = hidden_states.shape[0]
        img_mod1, img_mod2 = self.img_mod(temb).chunk(temb.shape[0] // batch, dim=0)[-1].chunk(2, dim=-1)
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated = None
        if encoder_hidden_states is not None:
            txt_mod1, txt_mod2 = self.txt_mod(temb[-batch:]).chunk(2, dim=-1)
            txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)
        attn_output, txt_attn_output, condition_kv, text_kv = self.attn.processor.condition_attention(
            self.attn, img_modulated, img_freqs, segments, txt_modulated, txt_freqs
        )
        hidden_states = hidden_states + self._gate(attn_output, img_gate1)
        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + self._gate(self.img_mlp(img_modulated2), img_gate2)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output
            txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
            encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states, encoder_hidden_states, condition_kv, text_kv


class QwenImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    """
    The Transformer model introduced in Qwen.

    Args:
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `64`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `None`):
            The number of channels in the output. If not specified, it defaults to `in_channels`.
        num_layers (`int`, defaults to `60`):
            The number of layers of dual stream DiT blocks to use.
        attention_head_dim (`int`, defaults to `128`):
            The number of dimensions to use for each attention head.
        num_attention_heads (`int`, defaults to `24`):
            The number of attention heads to use.
        joint_attention_dim (`int`, defaults to `3584`):
            The number of dimensions to use for the joint attention (embedding/channel dimension of
            `encoder_hidden_states`).
        guidance_embeds (`bool`, defaults to `False`):
            Whether to use guidance embeddings for guidance-distilled variant of the model.
        axes_dims_rope (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions to use for the rotary positional embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.zero_cond_t = zero_cond_t
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    zero_cond_t=zero_cond_t,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    @staticmethod
    def _mask_geometry_target_grid(
        img_shapes: Optional[List[List[Tuple[int, int, int]]]],
        batch_size: int,
    ) -> tuple[int, int]:
        """Validate and return the shared target-token grid for a batch."""

        if not img_shapes or not img_shapes[0]:
            raise ValueError("mask geometry injection requires non-empty img_shapes")
        if len(img_shapes) != batch_size:
            raise ValueError(
                "mask geometry requires one img_shapes entry per batch item, "
                f"got {len(img_shapes)} for batch size {batch_size}"
            )

        target_shape = tuple(img_shapes[0][0])
        if len(target_shape) != 3:
            raise ValueError(f"target img_shape must be (frame, height, width), got {target_shape}")
        frame, target_height, target_width = (int(value) for value in target_shape)
        if frame != 1:
            raise ValueError(f"2D mask geometry requires target frame=1, got {frame}")

        for sample_shapes in img_shapes[1:]:
            if not sample_shapes or tuple(sample_shapes[0]) != target_shape:
                raise ValueError("mask geometry requires one shared target token grid per batch")

        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                "mask geometry target token grid must be positive, "
                f"got {(target_height, target_width)}"
            )
        return target_height, target_width

    def prepare_mask_geometry_tokens(
        self,
        mask_geometry: torch.Tensor,
        img_shapes: Optional[List[List[Tuple[int, int, int]]]],
    ) -> torch.Tensor:
        """Tokenize a static geometry condition once for inference reuse."""

        if not isinstance(mask_geometry, torch.Tensor) or mask_geometry.ndim != 4:
            shape = getattr(mask_geometry, "shape", None)
            raise ValueError(
                "mask_geometry must have shape [B,3,H,W], "
                f"got {tuple(shape) if shape is not None else type(mask_geometry).__name__}"
            )
        tokenizer = getattr(self, "mask_geometry_tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "mask_geometry was provided but the transformer has no mask_geometry_tokenizer"
            )
        target_height, target_width = self._mask_geometry_target_grid(
            img_shapes,
            int(mask_geometry.shape[0]),
        )
        return tokenizer(
            mask_geometry,
            target_size=(target_height, target_width),
        )

    def _inject_mask_geometry(
        self,
        hidden_states: torch.Tensor,
        mask_geometry: Optional[torch.Tensor],
        img_shapes: Optional[List[List[Tuple[int, int, int]]]],
        mask_geometry_tokens: Optional[torch.Tensor] = None,
        target_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add geometric mask features to the target prefix of the image sequence.

        ``target_index`` (complementary canvas) selects the grid tokens held by the prefix.
        """

        if mask_geometry is not None and mask_geometry_tokens is not None:
            raise ValueError("provide either mask_geometry or mask_geometry_tokens, not both")
        if mask_geometry is None and mask_geometry_tokens is None:
            return hidden_states

        target_height, target_width = self._mask_geometry_target_grid(
            img_shapes,
            int(hidden_states.shape[0]),
        )

        grid_token_count = target_height * target_width
        target_token_count = grid_token_count if target_index is None else int(target_index.numel())
        if hidden_states.shape[1] < target_token_count:
            raise ValueError(
                "target token geometry is incompatible with hidden_states: "
                f"grid={(target_height, target_width)}, hidden={tuple(hidden_states.shape)}"
            )

        if mask_geometry_tokens is None:
            mask_geometry_tokens = self.prepare_mask_geometry_tokens(mask_geometry, img_shapes)
        expected_shape = (
            hidden_states.shape[0],
            grid_token_count,
            hidden_states.shape[2],
        )
        if tuple(mask_geometry_tokens.shape) != expected_shape:
            raise ValueError(
                "mask geometry tokens must have shape "
                f"{expected_shape}, got {tuple(mask_geometry_tokens.shape)}"
            )
        mask_geometry_tokens = mask_geometry_tokens.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if target_index is not None:
            mask_geometry_tokens = mask_geometry_tokens.index_select(1, target_index.to(hidden_states.device))

        target_hidden = hidden_states[:, :target_token_count] + mask_geometry_tokens
        if hidden_states.shape[1] == target_token_count:
            return target_hidden
        return torch.cat([target_hidden, hidden_states[:, target_token_count:]], dim=1)

    @staticmethod
    def _image_token_counts(
        img_shapes: Optional[List[List[Tuple[int, int, int]]]],
        image_token_index: Optional[Sequence[Optional[torch.Tensor]]],
    ) -> List[int]:
        """Tokens per image in sequence order; a token index keeps part of that image's grid."""

        if not img_shapes or any(list(shapes) != list(img_shapes[0]) for shapes in img_shapes):
            raise ValueError("token selection requires one shared img_shapes entry per batch item")
        image_token_index = tuple(image_token_index or ())
        if len(image_token_index) > len(img_shapes[0]):
            raise ValueError("image_token_index has more entries than img_shapes")
        counts = []
        for position, shape in enumerate(img_shapes[0]):
            index = image_token_index[position] if position < len(image_token_index) else None
            if index is not None and index.ndim != 1:
                raise ValueError("image_token_index entries must be 1-D and shared by the batch")
            counts.append(math.prod(shape) if index is None else int(index.numel()))
        return counts

    @torch.no_grad()
    def build_condition_cache(
        self,
        condition_states: torch.Tensor,
        img_shapes: List[List[Tuple[int, int, int]]],
        image_token_index: Optional[Sequence[Optional[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        *,
        isolation: str = "image",
        target_groups: int = 1,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        txt_seq_lens: Optional[List[int]] = None,
    ) -> ConditionKVCache:
        """Run isolated condition images once through every block and keep their K/V.

        ``condition_states`` are the packed condition latents in sequence order (the
        image stream without its target prefix). With zero_cond_t and isolation they
        depend on neither x_t, t, the text nor the target, so ``forward`` with
        ``condition_cache`` reproduces the uncached isolated forward. ``isolation`` is
        "image" (each image attends to itself) or "group" (the images attend to each
        other); ``target_groups`` is the joint forward's number of target modulation
        groups (2 with a canvas ring). ``encoder_hidden_states`` (a text condition,
        ``text_as_condition``) is cached too.
        """
        if not self.zero_cond_t:
            raise ValueError("condition caching requires zero_cond_t: conditions must not depend on t")
        if isolation not in ("image", "group"):
            raise ValueError(f"condition isolation must be 'image' or 'group', got {isolation!r}")
        counts = self._image_token_counts(img_shapes, image_token_index)
        image_lengths = tuple(counts[1:])
        segments = (sum(image_lengths),) if isolation == "group" else image_lengths
        if condition_states.shape[1] != sum(image_lengths):
            raise ValueError(f"condition_states has {condition_states.shape[1]} tokens, img_shapes give {sum(image_lengths)}")
        lora_scale = (attention_kwargs or {}).get("scale", 1.0)
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        hidden_states = self.img_in(condition_states)
        # zero_cond_t embeds [t, (t_ring,) 0]; only the condition (t=0) rows are used below.
        timestep = torch.zeros(
            (target_groups + 1) * hidden_states.shape[0], device=hidden_states.device, dtype=hidden_states.dtype
        )
        temb = self.time_text_embed(timestep, hidden_states)
        text = None
        if encoder_hidden_states is not None:
            text = self.txt_in(self.txt_norm(encoder_hidden_states))
        img_freqs, txt_freqs = self.pos_embed(
            img_shapes, txt_seq_lens or [0], device=hidden_states.device, token_index=image_token_index
        )
        img_freqs = img_freqs[counts[0] :]
        layers, text_layers = [], []
        for block in self.transformer_blocks:
            hidden_states, text, condition_kv, text_kv = block.forward_conditions(
                hidden_states, temb, img_freqs, segments, text, txt_freqs if text is not None else None
            )
            layers.append(condition_kv)
            text_layers.append(text_kv)
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        return ConditionKVCache(
            segment_lengths=segments,
            layers=layers,
            image_lengths=image_lengths,
            text_layers=text_layers if encoder_hidden_states is not None else None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[List[Tuple[int, int, int]]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        mask_geometry: Optional[torch.Tensor] = None,
        mask_geometry_tokens: Optional[torch.Tensor] = None,
        output_token_count: Optional[int] = None,
        end_timestep: Optional[torch.Tensor] = None,
        return_alpha_logits: bool = False,
        image_token_index: Optional[Sequence[Optional[torch.Tensor]]] = None,
        isolate_conditions: bool = False,
        condition_cache: Optional[ConditionKVCache] = None,
        isolation: str = "image",
        isolation_bias: Optional[float] = None,
        ring_token_count: int = 0,
        ring_timestep: Optional[torch.Tensor] = None,
        text_as_condition: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            mask_geometry (`torch.Tensor`, *optional*):
                Three-channel geometric mask encoding. When provided, its
                zero-initialized tokenizer output is added only to the target
                image-token prefix before ``img_in``.
            mask_geometry_tokens (`torch.Tensor`, *optional*):
                Pre-tokenized geometry shaped ``[B,N_target,C]``. Inference may
                cache this static tensor and reuse it across denoising and CFG
                forwards. Mutually exclusive with ``mask_geometry``.
            output_token_count (`int`, *optional*):
                Read out only this target prefix; conditioning image tokens
                still participate in every transformer block.
            return_alpha_logits (`bool`, defaults to `False`):
                Also return the attached ``alpha_head`` logits ``[B, N_target]``
                computed from the final target hidden states (tuple output only).
            image_token_index (`Sequence[Optional[torch.Tensor]]`, *optional*):
                Complementary canvas: per image in ``img_shapes`` order, 1-D row-major
                indices of the grid tokens present in the sequence (shared by the
                batch). Each token keeps its own grid RoPE coordinate; ``None`` or a
                missing entry keeps the full grid.
            isolate_conditions (`bool`, defaults to `False`):
                Condition images never attend to the text or the target; text and
                target queries still attend to every key.
            condition_cache (`ConditionKVCache`, *optional*):
                Cached per-block K/V of isolated conditions from
                ``build_condition_cache``; ``hidden_states`` then holds only the
                target tokens. A cache with text K/V replaces the text stream.
            isolation (`str`, defaults to `"image"`):
                With ``isolate_conditions``: "image" lets each condition image attend
                only to itself, "group" lets the condition images attend to each other.
            isolation_bias (`float`, *optional*):
                Soft isolation (training warm-up): isolated queries also attend to
                their excluded keys, with this logit bias subtracted.
            ring_token_count (`int`, defaults to 0):
                Complementary-canvas ring: the last this many target tokens form a
                second target modulation group with ``ring_timestep``.
            text_as_condition (`bool`, defaults to `False`):
                With isolated conditions: the text stream is modulated at t=0 and
                attends only to text and condition tokens, so it can be cached.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        image_counts = None
        if image_token_index is not None or isolate_conditions or condition_cache is not None:
            image_counts = self._image_token_counts(img_shapes, image_token_index)
            if condition_cache is not None and tuple(condition_cache.image_lengths) != tuple(image_counts[1:]):
                raise ValueError(
                    f"condition cache holds images {condition_cache.image_lengths}, expected {tuple(image_counts[1:])}"
                )
            expected_length = image_counts[0] if condition_cache is not None else sum(image_counts)
            if hidden_states.shape[1] != expected_length:
                raise ValueError(
                    f"image sequence has {hidden_states.shape[1]} tokens, img_shapes/token index give {expected_length}"
                )

        hidden_states = self._inject_mask_geometry(
            hidden_states,
            mask_geometry,
            img_shapes,
            mask_geometry_tokens=mask_geometry_tokens,
            target_index=image_token_index[0] if image_token_index else None,
        )
        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(torch.float32 if end_timestep is not None else hidden_states.dtype)
        if end_timestep is not None:
            if not hasattr(self, "flow_map_time_embedding"):
                raise ValueError("end_timestep requires a trained flow-map time module")
            end_timestep = end_timestep.to(device=timestep.device, dtype=timestep.dtype)
            if end_timestep.shape != timestep.shape:
                raise ValueError("end_timestep must match timestep shape")
        target_token_count = None
        if self.zero_cond_t:
            if not img_shapes or len(img_shapes) != hidden_states.shape[0]:
                raise ValueError("zero_cond_t requires img_shapes for every batch item")
            if image_counts is not None:
                target_token_count = image_counts[0]
            else:
                counts = [math.prod(shapes[0]) for shapes in img_shapes]
                if len(set(counts)) != 1 or any(sum(math.prod(s) for s in shapes) != hidden_states.shape[1] for shapes in img_shapes):
                    raise ValueError("zero_cond_t requires a shared target prefix and img_shapes matching the image sequence")
                target_token_count = counts[0]
            groups = [timestep]
            if ring_token_count:
                if ring_timestep is None or end_timestep is not None or not 0 < ring_token_count < target_token_count:
                    raise ValueError("a canvas ring needs ring_timestep and a non-empty core, without end_timestep")
                groups.append(ring_timestep.to(device=timestep.device, dtype=timestep.dtype))
                target_token_count = (target_token_count - ring_token_count, ring_token_count)
            timestep = torch.cat((*groups, torch.zeros_like(timestep)))
            if end_timestep is not None:
                end_timestep = torch.cat((end_timestep, torch.zeros_like(end_timestep)))
        elif ring_token_count or text_as_condition:
            raise ValueError("a canvas ring and a text condition need zero_cond_t (per-group timesteps)")
        text_cached = condition_cache is not None and condition_cache.text_layers is not None
        if text_as_condition != text_cached and not (text_as_condition and isolate_conditions):
            raise ValueError("text_as_condition needs isolated conditions, and a cache with text needs it set")
        if text_cached:
            encoder_hidden_states = None
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        if end_timestep is not None:
            temb = self.flow_map_time_embedding(temb, end_timestep, hidden_states, guidance)

        rope_kwargs = {"token_index": image_token_index} if image_token_index is not None else {}
        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens or [0], device=hidden_states.device, **rope_kwargs
        )
        condition_segments = None
        if condition_cache is not None:
            # Cached condition keys are already rotated; only the target tokens remain in the stream.
            image_rotary_emb = (image_rotary_emb[0][: image_counts[0]], image_rotary_emb[1])
        elif isolate_conditions:
            if isolation not in ("image", "group"):
                raise ValueError(f"condition isolation must be 'image' or 'group', got {isolation!r}")
            condition_segments = tuple(image_counts[1:])
            if isolation == "group":
                condition_segments = (sum(condition_segments),)
        elif isolation_bias is not None:
            raise ValueError("isolation_bias softens isolate_conditions")

        isolated = condition_segments is not None or condition_cache is not None
        for index_block, block in enumerate(self.transformer_blocks):
            skip_text_output = index_block == len(self.transformer_blocks) - 1
            condition_kv = condition_cache.layers[index_block] if condition_cache is not None else None
            text_kv = condition_cache.text_layers[index_block] if text_cached else None
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # The dense call keeps its original positional signature.
                isolation_args = (
                    (condition_segments, condition_kv, isolation_bias, text_as_condition, text_kv) if isolated else ()
                )
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    attention_kwargs,
                    target_token_count,
                    skip_text_output,
                    *isolation_args,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                    target_token_count=target_token_count,
                    skip_text_output=skip_text_output,
                    condition_segments=condition_segments,
                    condition_kv=condition_kv,
                    isolation_bias=isolation_bias,
                    text_as_condition=text_as_condition,
                    text_kv=text_kv,
                )

        alpha_logits = None
        if return_alpha_logits:
            alpha_head = getattr(self, "alpha_head", None)
            if alpha_head is None:
                raise ValueError("return_alpha_logits requires an attached alpha_head")
            if return_dict:
                raise ValueError("return_alpha_logits requires return_dict=False")
            alpha_logits = alpha_head(hidden_states[:, : math.prod(img_shapes[0][0])])

        # Use only the image part (hidden_states) from the dual-stream blocks
        if output_token_count is not None:
            hidden_states = hidden_states[:, :output_token_count]
        if isinstance(target_token_count, tuple):
            # [core | ring]: each target group is read out with its own timestep embedding.
            batch, core = hidden_states.shape[0], target_token_count[0]
            hidden_states = torch.cat(
                [
                    self.norm_out(hidden_states[:, :core], temb[:batch]),
                    self.norm_out(hidden_states[:, core:], temb[batch : 2 * batch]),
                ],
                dim=1,
            )
        else:
            hidden_states = self.norm_out(hidden_states, temb[:hidden_states.shape[0]] if self.zero_cond_t else temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if alpha_logits is not None:
            return (output, alpha_logits)
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

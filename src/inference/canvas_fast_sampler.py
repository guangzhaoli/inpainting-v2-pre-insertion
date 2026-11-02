"""Fast sampling for the complementary canvas with isolated, cached conditions (B2 + cache).

Inference only. The math is that of the trainer's cached canvas loop
(``QwenImageEditTrainer.sampling_from_embeddings`` with ``model.complementary_canvas.{enabled,
isolate_conditions, cache_conditions}`` and reference CFG); only the execution differs. Each
technique is a switch of :class:`FastCanvasOptions`:

``merge_lora``
    LoRA folded into the base weights once (fp32 sum, one bf16 rounding, :func:`merge_lora_`), so
    a projection is one GEMM. PEFT then reports the layers as merged, so the model's own forward
    uses the merged weights as well.
``fuse_qkv``
    One ``[3D, D]`` QKV weight per stream and block (:func:`fuse_qkv_`); the q/k/v module weights
    become views of it, so the model's own forward is unchanged.
``batch_cfg``
    Both reference-CFG branches in one forward: their tokens are packed along the sequence (image
    GEMMs over both copies of the query tokens, text GEMMs over both prompts) and one FA2 varlen
    call attends each branch to its own keys.
``kv_buffer``
    Per-block key/value buffers in the layout of :class:`KeyLayout`, filled once per sample by the
    condition pass; a step writes only its own rows in place. Without it, every block
    concatenates the same layout from the per-image condition K/V (what the trainer's cached
    forward does).
``cuda_graph``
    The per-step forward of every branch plus the CFG combine is captured once per sample, after
    the condition pass, and replayed for all steps.
``attention``
    ``"flash"``: FA2 varlen, every packed branch in one call. ``"cudnn"``: PyTorch's cuDNN SDPA
    (Hopper fused attention), one call per branch on views of the same key rows.
``modulation_table``
    The modulation rows depend on t only: compute them for all steps of a sample at once (one
    GEMM per block and stream instead of one GEMV per step), then read the step's rows.
``fused_kernels``
    Triton kernels (:mod:`src.inference.fused_ops`) for LayerNorm+modulation, gate+residual and
    q/k RMSNorm+RoPE, the last writing q into the packed query and k/v into the key buffers.
``fused_lora``
    Unmerged LoRA in fewer kernels: one GEMM for the q/k/v (and add_q/k/v) LoRA-A projections of a
    stream, and every LoRA-B product accumulated into the base output (``addmm_``, one rounding)
    instead of PEFT's separate product, scaling and sum.

The per-step query group is generic: a branch's queries are its text tokens plus the image query
segments (:class:`QuerySegment`), each with its own grid positions (RoPE, mask-geometry tokens)
and timestep. Today there is one segment, the core edit region R; a ring band is one more
segment with its own positions and, if its noise level differs, its own timestep. A text
condition (``text_as_condition``) has no step queries: it runs once with its branch's condition
pass (t=0, attending to itself and that pass's images) and the steps attend to its cached K/V.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.inference import fused_ops
from src.models.complementary_canvas import gather_tokens, scatter_tokens
from src.models.transformer_qwenimage import apply_rotary_emb_qwen

CLEAN = "clean"  # timestep name of the t=0 (condition) modulation row


@dataclass(frozen=True)
class FastCanvasOptions:
    """Defaults: the fastest configuration whose per-step outputs stay at the batch-size noise floor
    of the trainer's cached path (LoRA unmerged; ``merge_lora=True`` is faster but about twice the
    floor; cuDNN re-plans every new shape, so FA2)."""

    merge_lora: bool = False
    fuse_qkv: bool = True
    batch_cfg: bool = True
    kv_buffer: bool = True
    cuda_graph: bool = True
    attention: str = "flash"
    modulation_table: bool = True
    fused_kernels: bool = True
    fused_lora: bool = True

    def __post_init__(self):
        if self.attention not in ("flash", "cudnn"):
            raise ValueError(f"attention must be 'flash' or 'cudnn', got {self.attention!r}")
        if self.fused_kernels and not self.kv_buffer:
            raise ValueError("fused_kernels write the step K/V straight into the key buffers (kv_buffer)")
        if self.fused_lora and not (self.fused_kernels and self.fuse_qkv):
            raise ValueError("fused_lora is part of the fused path (fused_kernels and fuse_qkv)")

    @property
    def label(self) -> str:
        names = ("merge_lora", "fuse_qkv", "batch_cfg", "kv_buffer", "cuda_graph", "modulation_table", "fused_kernels",
                 "fused_lora")
        on = [name for name in names if getattr(self, name)]
        if self.attention != "flash":
            on.append(self.attention)
        return "+".join(on) or "eager"


@dataclass(frozen=True)
class QuerySegment:
    """A run of target-grid image tokens recomputed at every step, shared by every CFG branch.

    ``token_index``: row-major positions on the target grid (RoPE coordinates and geometry
    tokens). ``timestep``: name of the per-step timestep that modulates them (``"t"`` is the one
    the sampler denoises with; :data:`CLEAN` is the t=0 row of the conditions).
    """

    name: str
    token_index: torch.Tensor
    timestep: str = "t"

    @property
    def length(self) -> int:
        return int(self.token_index.numel())


@dataclass(frozen=True)
class BranchInputs:
    """One CFG branch: its prompt, its ``img_shapes`` (they set the text RoPE offset) and the ids of
    the condition images it attends, in its sequence order."""

    prompt_embeds: torch.Tensor  # [1, T, joint_attention_dim]
    img_shapes: list
    conditions: Tuple[int, ...]


@dataclass(frozen=True)
class ConditionPass:
    """One isolated condition stream (``build_condition_cache`` inputs) and the ids of its images.

    ``isolation``: ``"image"`` (each image attends to itself) or ``"group"`` (the stream's images
    attend to each other). ``text``: the branch whose prompt is a text condition run with this
    stream (its queries attend to its own keys and the stream's images), or None.
    """

    states: torch.Tensor  # [1, sum(lengths), C] packed condition latents
    img_shapes: list
    image_token_index: tuple
    ids: Tuple[int, ...]
    isolation: str = "image"
    text: Optional[int] = None


# --------------------------------------------------------------------------------------------
# Weights


def _lora_layers(model):
    from peft.tuners.lora import Linear as LoraLinear

    return [module for module in model.modules() if isinstance(module, LoraLinear)]


def _base(module):
    return module.get_base_layer() if hasattr(module, "get_base_layer") else module


@torch.no_grad()
def merge_lora_(model) -> int:
    """Fold every active vanilla LoRA adapter into its base weight, in place: one bf16 rounding of
    ``W + scaling * B @ A`` computed in fp32 (PEFT's own merge rounds the delta and the sum).

    PEFT marks the layers merged and skips the adapter from then on. Returns the merged count.
    """

    count = 0
    for module in _lora_layers(model):
        if module.merged or module.disable_adapters:
            continue
        weight = module.get_base_layer().weight
        for adapter in module.active_adapters:
            if adapter not in module.lora_A:
                continue
            if adapter in module.lora_variant or module.lora_bias[adapter] or module.fan_in_fan_out:
                raise NotImplementedError(f"merge_lora_ supports vanilla LoRA only (adapter {adapter})")
            delta = module.lora_B[adapter].weight.float() @ module.lora_A[adapter].weight.float()
            weight.data.copy_(weight.float() + delta * module.scaling[adapter])
            module.merged_adapters.append(adapter)
            count += 1
    return count


_QKV = (("to_q", "to_k", "to_v"), ("add_q_proj", "add_k_proj", "add_v_proj"))
_FUSED = ("_fast_qkv", "_fast_added_qkv")


@torch.no_grad()
def fuse_qkv_(model) -> None:
    """Store each block's image (to_q/k/v) and text (add_q/k/v_proj) projection weights in one
    ``[3D, D]`` tensor per stream; the module weights become views of it (same values, the model's
    forward is unchanged). Idempotent."""

    for block in model.transformer_blocks:
        attn = block.attn
        for attribute, names in zip(_FUSED, _QKV):
            if getattr(attn, attribute, None) is not None:
                continue
            layers = [_base(getattr(attn, name)) for name in names]
            weight = torch.cat([layer.weight for layer in layers])
            bias = torch.cat([layer.bias for layer in layers])
            rows = layers[0].out_features
            for index, layer in enumerate(layers):
                layer.weight.data = weight[index * rows : (index + 1) * rows]
                layer.bias.data = bias[index * rows : (index + 1) * rows]
            setattr(attn, attribute, (weight, bias))


def _lora_terms(module):
    """PEFT ``lora.Linear`` adapters still applied at run time (not merged)."""

    if not hasattr(module, "lora_A") or module.merged or module.disable_adapters:
        return []
    return [(module.lora_A[name], module.lora_B[name], module.scaling[name])
            for name in module.active_adapters if name in module.lora_A]


def _qkv(x, attn, stream: int, fused: bool):
    """``[1, M, 3D]`` q|k|v of one stream (0 image, 1 text), as the attention processor computes them."""

    names = _QKV[stream]
    weights = getattr(attn, _FUSED[stream], None) if fused else None
    if weights is None:
        return torch.cat([getattr(attn, name)(x) for name in names], dim=-1)
    out = F.linear(x, *weights)
    rows = weights[0].shape[0] // 3
    for index, name in enumerate(names):
        for lora_a, lora_b, scaling in _lora_terms(getattr(attn, name)):
            # PEFT: result + lora_B(lora_A(x)) * scaling, in the adapter dtype.
            out[..., index * rows : (index + 1) * rows] += lora_b(lora_a(x.to(lora_a.weight.dtype))) * scaling
    return out


def _single_lora(module):
    """(lora_A, lora_B, scaling) of a PEFT layer with one unmerged adapter in the base dtype, else None."""

    terms = _lora_terms(module)
    if len(terms) != 1 or terms[0][0].weight.dtype != _base(module).weight.dtype:
        return None
    return terms[0]


def _linear_fused_lora(x, module):
    """``module(x)`` for a 2-D ``x``: base GEMM, LoRA-A GEMM, LoRA-B accumulated in place."""

    base = _base(module)
    out = F.linear(x, base.weight, base.bias)
    lora = _single_lora(module)
    if lora is not None:
        lora_a, lora_b, scaling = lora
        out.addmm_(F.linear(x, lora_a.weight), lora_b.weight.t(), alpha=scaling)
    elif _lora_terms(module):
        return module(x)
    return out


def _qkv_fused_lora(x, attn, stream: int):
    """``[M, 3D]`` fused q|k|v of a 2-D ``x`` with one GEMM for the three LoRA-A projections."""

    fused = getattr(attn, _FUSED[stream], None)
    if fused is None:  # fuse_qkv_ not applied to this model
        return _qkv(x[None], attn, stream, True)[0]
    weight, bias = fused
    out = F.linear(x, weight, bias)
    loras = [_single_lora(getattr(attn, name)) for name in _QKV[stream]]
    if all(lora is None for lora in loras) and not any(_lora_terms(getattr(attn, name)) for name in _QKV[stream]):
        return out
    if any(lora is None for lora in loras):
        return _qkv(x[None], attn, stream, True)[0]
    key = f"_fast_lora_a_{stream}"
    cached = getattr(attn, key, None)
    if cached is None or any(a.data_ptr() != b.weight.data_ptr() for a, (b, _, _) in zip(cached[1], loras)):
        stacked = torch.cat([lora_a.weight for lora_a, _, _ in loras])
        cached = (stacked, [lora_a.weight for lora_a, _, _ in loras])
        setattr(attn, key, cached)
    projected = F.linear(x, cached[0])  # [M, 3r]
    rows, rank = weight.shape[0] // 3, loras[0][0].weight.shape[0]
    for index, (_, lora_b, scaling) in enumerate(loras):
        out[:, index * rows : (index + 1) * rows].addmm_(projected[:, index * rank : (index + 1) * rank],
                                                         lora_b.weight.t(), alpha=scaling)
    return out


def _feed_forward(x, feed_forward, fused_lora: bool):
    """diffusers ``FeedForward`` (GELU projection, dropout, linear) with :func:`_linear_fused_lora`."""

    if not fused_lora:
        return feed_forward(x)
    for module in feed_forward.net:
        if hasattr(module, "proj") and hasattr(module, "approximate"):  # diffusers GELU
            x = F.gelu(_linear_fused_lora(x, module.proj), approximate=module.approximate)
        elif isinstance(module, torch.nn.Dropout):
            x = module(x)
        else:
            x = _linear_fused_lora(x, module)
    return x


def _modulate(x, params):
    """``QwenImageTransformerBlock._modulate`` with one modulation row ``params`` [1, 3D]."""

    shift, scale, gate = params.chunk(3, dim=-1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)


# --------------------------------------------------------------------------------------------
# Key layout and attention


@dataclass(frozen=True)
class KeyLayout:
    """Key rows shared by the branches: ``[text_0|q_0 | own_0 | shared | own_1 | q_1|text_1]``.

    Branch 0 attends ``[text_0, q_0, own_0, shared]``, branch 1 ``[shared, own_1, q_1, text_1]``:
    two contiguous ranges that overlap on the conditions both branches see (the masked source),
    so those are stored once. With one branch: ``[text_0|q_0 | conditions]``. A text condition is a
    part of its own between its branch's step rows and images, ``[q_0 | text_0 | own_0 | shared |
    own_1 | text_1 | q_1]``, contiguous with the images its condition pass attends.
    ``parts`` are ``(kind, key, start, length)`` with kind ``"step"`` or ``"text"`` (key = branch) or
    ``"image"``.
    """

    parts: Tuple[Tuple[str, int, int, int], ...]
    branch_keys: Tuple[Tuple[int, int], ...]  # per branch (start, length)

    @property
    def total(self) -> int:
        _, _, start, length = self.parts[-1]
        return start + length

    def part(self, kind: str, key: int) -> Tuple[int, int]:
        for kind_, key_, start, length in self.parts:
            if (kind_, key_) == (kind, key):
                return start, length
        raise KeyError((kind, key))

    @classmethod
    def build(cls, step_lengths: Sequence[int], conditions: Sequence[Sequence[int]], image_lengths: Mapping[int, int],
              text_lengths: Optional[Sequence[int]] = None):
        texts = [[("text", branch)] if text_lengths is not None else [] for branch in range(len(conditions))]
        if len(conditions) == 1:
            order = [("step", 0)] + texts[0] + [("image", image) for image in conditions[0]]
            spans = [(0, len(order))]
        elif len(conditions) == 2:
            shared = [image for image in conditions[0] if image in conditions[1]]
            if [image for image in conditions[1] if image in shared] != shared:
                raise ValueError("shared condition images must keep one order in both branches")
            own = [[image for image in images if image not in shared] for images in conditions]
            images = [("image", image) for image in own[0] + shared + own[1]]
            order = [("step", 0)] + texts[0] + images + texts[1] + [("step", 1)]
            head = 1 + len(texts[0]) + len(own[0])
            first_1 = head if (shared or own[1]) else len(order) - 1 - len(texts[1])
            spans = [(0, head + len(shared)), (first_1, len(order))]
        else:
            raise NotImplementedError("the shared key layout holds one or two branches")
        lengths = {"step": step_lengths, "text": text_lengths, "image": image_lengths}
        parts, row = [], 0
        for kind, key in order:
            length = lengths[kind][key]
            parts.append((kind, key, row, length))
            row += length
        branch_keys = tuple((parts[a][2], parts[b - 1][2] + parts[b - 1][3] - parts[a][2]) for a, b in spans)
        return cls(tuple(parts), branch_keys)


def _flash_varlen():
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    return _flash_attn_varlen_forward


class _Attention:
    """Attention of packed branches. ``sequences`` are ``(q_start, q_len, k_start, k_len)``; key ranges
    may overlap. ``backend``: ``"flash"`` (FA2 varlen with ``seqused_k``, one call), ``"cudnn"``
    (cuDNN SDPA per sequence) or ``"sdpa"`` (per-sequence SDPA, the CPU reference)."""

    def __init__(self, sequences: Sequence[Tuple[int, int, int, int]], device, scale: float, backend: str = "flash"):
        self.sequences = tuple(sequences)
        self.scale = scale
        self.backend = backend if device.type == "cuda" else "sdpa"
        self.flash = self.backend == "flash"
        if self.flash:
            ends = [q_start + q_len for q_start, q_len, _, _ in sequences]
            self.cu_q = torch.tensor([s[0] for s in sequences] + [ends[-1]], dtype=torch.int32, device=device)
            # seqused_k gives each length, so cu_k only needs the starts (they may overlap).
            self.cu_k = torch.tensor([s[2] for s in sequences] + [max(s[2] + s[3] for s in sequences)],
                                     dtype=torch.int32, device=device)
            self.used_k = torch.tensor([s[3] for s in sequences], dtype=torch.int32, device=device)
            self.max_q = max(s[1] for s in sequences)
            self.max_k = max(s[3] for s in sequences)
            self.varlen = _flash_varlen()

    def __call__(self, query, key, value):
        """``query`` [Sq, H, Dh], ``key``/``value`` [Sk, H, Dh] -> [Sq, H, Dh]."""

        if self.flash:
            return self.varlen(query, key, value, self.cu_q, self.cu_k, self.max_q, self.max_k, 0.0, self.scale,
                               False, seqused_k=self.used_k)[0]
        outputs = []
        for q_start, q_len, k_start, k_len in self.sequences:
            # [S, H, D] row ranges as [1, H, S, D] views
            q, k, v = (x.transpose(0, 1)[None] for x in (query[q_start : q_start + q_len], key[k_start : k_start + k_len],
                                                         value[k_start : k_start + k_len]))
            if self.backend == "cudnn":
                from torch.nn.attention import SDPBackend, sdpa_kernel

                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
                    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            else:
                out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            outputs.append(out[0].transpose(0, 1))
        return torch.cat(outputs)


@dataclass(frozen=True)
class _Group:
    """Branches evaluated in one forward. Packed query order: one branch ``[text_0 | q_0]`` or
    ``[q_1 | text_1]`` (the order of its step rows in the layout), two ``[text_0 | q_0 | q_1 | text_1]``;
    a text condition has no text queries (empty text rows)."""

    branches: Tuple[int, ...]
    text_rows: Tuple[Tuple[int, int], ...]  # per branch, in the packed query
    img_rows: Tuple[int, int]
    step_writes: Tuple[Tuple[int, int, int], ...]  # (query row, key row, length) of this step's K/V
    key_parts: Tuple[Tuple[str, int, int, int], ...]  # layout parts this group's keys span
    attention: _Attention


# --------------------------------------------------------------------------------------------
# Per-sample forward


class CanvasStepForward:
    """The isolated-condition DiT forward of a fixed set of query tokens and CFG branches.

    Construction runs the condition passes once (every condition image through every block at t=0,
    self-attention only, and a text condition over itself and its pass's images, as
    ``build_condition_cache``) and keeps their per-block K/V; :meth:`step` then runs only the
    per-step tokens of every branch. A step's ``latents`` are the image query segments packed in
    order, ``[1, sum(len), C]``; it returns one prediction per branch.
    """

    def __init__(
        self,
        dit,
        segments: Sequence[QuerySegment],
        branches: Sequence[BranchInputs],
        passes: Sequence[ConditionPass],
        mask_geometry_tokens: Optional[torch.Tensor],
        options: FastCanvasOptions,
        timesteps: Sequence[str] = ("t",),
    ):
        if not dit.zero_cond_t:
            raise ValueError("isolated cached conditions require zero_cond_t")
        self.dit, self.options = dit, options
        self.segments, self.branches = tuple(segments), tuple(branches)
        self.device, self.dtype = dit.img_in.weight.device, dit.img_in.weight.dtype
        self.heads = dit.transformer_blocks[0].attn.heads
        self.head_dim = dit.inner_dim // self.heads
        self.query_count = sum(segment.length for segment in self.segments)
        # Modulation rows: the named timesteps, then t=0 (the zero_cond_t layout [t, 0]).
        self.timestep_names = tuple(name for name in timesteps if name != CLEAN)
        rows = {name: index for index, name in enumerate(self.timestep_names)}
        rows[CLEAN] = len(self.timestep_names)
        runs, start = [], 0  # consecutive segments sharing a modulation row: (start, end, row)
        for segment in self.segments:
            if segment.timestep not in rows:
                raise ValueError(f"segment {segment.name} uses timestep {segment.timestep!r}, not in {timesteps}")
            if runs and runs[-1][2] == rows[segment.timestep]:
                runs[-1] = (runs[-1][0], start + segment.length, runs[-1][2])
            else:
                runs.append((start, start + segment.length, rows[segment.timestep]))
            start += segment.length
        self.runs = tuple(runs)

        with torch.inference_mode():
            device, dtype = self.device, self.dtype
            query_index = torch.cat([segment.token_index.to(device) for segment in self.segments])
            # Image 0 of every branch's img_shapes is the target grid.
            target_shapes = [[tuple(self.branches[0].img_shapes[0][0])]]
            self.img_freqs, _ = dit.pos_embed(target_shapes, [0], device=device, token_index=(query_index,))
            self.geometry = None
            if mask_geometry_tokens is not None:
                self.geometry = mask_geometry_tokens.to(device=device, dtype=dtype).index_select(1, query_index)
            self.text_lengths = [int(branch.prompt_embeds.shape[1]) for branch in self.branches]
            self.text_freqs, self.text_hidden = [], []
            for branch, length in zip(self.branches, self.text_lengths):
                self.text_freqs.append(dit.pos_embed(branch.img_shapes, [length], device=device)[1])
                prompt = branch.prompt_embeds.to(device=device, dtype=dtype)
                self.text_hidden.append(dit.txt_in(dit.txt_norm(prompt)))  # step-invariant

            image_lengths = {}
            for condition in passes:
                counts = dit._image_token_counts(condition.img_shapes, condition.image_token_index)
                if len(counts) - 1 != len(condition.ids):
                    raise ValueError(f"condition pass has {len(counts) - 1} images, ids {condition.ids}")
                image_lengths.update(zip(condition.ids, counts[1:]))
            texts = [condition for condition in passes if condition.text is not None]
            self.text_condition = bool(texts)
            if texts and (sorted(condition.text for condition in texts) != list(range(len(self.branches))) or any(
                    condition.ids != self.branches[condition.text].conditions for condition in texts)):
                raise ValueError("a text condition runs with one condition pass per branch, holding that branch's images")
            if self.text_condition:  # the text K/V are cached with the conditions
                steps = [self.query_count] * len(self.branches)
            else:
                steps = [length + self.query_count for length in self.text_lengths]
            self.layout = KeyLayout.build(steps, [branch.conditions for branch in self.branches], image_lengths,
                                          self.text_lengths if self.text_condition else None)
            start_time = time.perf_counter()
            self._condition_pass(passes)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self.condition_pass_s = time.perf_counter() - start_time
            self.groups = self._groups()
            # fused path: Triton on CUDA bf16, the torch references of fused_ops otherwise
            self.fused_state = [self._fused_group_state(group) for group in self.groups] if options.fused_kernels else None

    def _condition_pass(self, passes: Sequence[ConditionPass]):
        """Every condition stream once through every block (``build_condition_cache``: t=0
        modulation, attention only within each image or group, a text condition over itself and
        the stream's images), its post-RoPE K/V written into the key buffers (or kept per part)."""

        dit, heads, head_dim = self.dit, self.heads, self.head_dim
        layers = len(dit.transformer_blocks)
        if self.options.kv_buffer:
            shape = (layers, self.layout.total, heads, head_dim)
            self.key_buffer = torch.empty(shape, device=self.device, dtype=self.dtype)
            self.value_buffer = torch.empty_like(self.key_buffer)
        else:
            self.condition_kv = [{} for _ in range(layers)]
        for condition in passes:
            counts = dit._image_token_counts(condition.img_shapes, condition.image_token_index)
            lengths = counts[1:]
            offsets = [sum(lengths[:index]) for index in range(len(lengths))]
            slots = [self.layout.part("image", image)[0] if self.options.kv_buffer else offset
                     for image, offset in zip(condition.ids, offsets)]
            if condition.isolation == "image":
                sequences = [(offset, length, slot, length) for offset, length, slot in zip(offsets, lengths, slots)]
            elif condition.isolation == "group":
                first = min(slots)
                if max(slot + length for slot, length in zip(slots, lengths)) - first != sum(lengths):
                    raise ValueError("a condition group must occupy contiguous key rows")
                sequences = [(0, sum(lengths), first, sum(lengths))]
            else:
                raise ValueError(f"condition isolation must be 'image' or 'group', got {condition.isolation!r}")
            attention = _Attention(sequences, self.device, head_dim ** -0.5, self.options.attention)
            hidden = dit.img_in(condition.states.to(device=self.device, dtype=self.dtype))
            # zero_cond_t row layout [target timesteps..., 0]; the condition (t=0) row is used.
            rows = torch.zeros(len(self.timestep_names) + 1, device=self.device, dtype=hidden.dtype)
            temb = dit.time_text_embed(rows, hidden)
            freqs, _ = dit.pos_embed(condition.img_shapes, [0], device=self.device,
                                     token_index=condition.image_token_index)
            freqs = freqs[counts[0] :]
            text = None if condition.text is None else self._text_condition(condition.text, slots, lengths)
            if self.options.fused_kernels:
                self._condition_pass_fused(hidden, temb, freqs, lengths, slots, attention, text)
                continue
            txt = None if text is None else text["hidden"]
            last = len(dit.transformer_blocks) - 1
            for layer, block in enumerate(dit.transformer_blocks):
                attn = block.attn
                img_mod1, img_mod2 = block.img_mod(temb)[-1:].chunk(2, dim=-1)
                modulated, gate1 = _modulate(block.img_norm1(hidden), img_mod1)
                query, key, value = _qkv(modulated, attn, 0, self.options.fuse_qkv).unflatten(
                    -1, (3 * heads, head_dim)).chunk(3, 2)
                query = apply_rotary_emb_qwen(attn.norm_q(query), freqs, use_real=False)
                key = apply_rotary_emb_qwen(attn.norm_k(key), freqs, use_real=False)
                if self.options.kv_buffer:
                    keys, values = self.key_buffer[layer], self.value_buffer[layer]
                    for offset, length, slot in zip(offsets, lengths, slots):
                        keys[slot : slot + length] = key[0, offset : offset + length]
                        values[slot : slot + length] = value[0, offset : offset + length]
                else:
                    keys, values = key[0], value[0].contiguous()
                    for image, offset, length in zip(condition.ids, offsets, lengths):
                        self.condition_kv[layer][("image", image)] = (keys[offset : offset + length],
                                                                      values[offset : offset + length])
                if txt is not None:
                    # the text stream at t=0 (the condition row), as forward_conditions
                    txt_mod1, txt_mod2 = block.txt_mod(temb[-1:]).chunk(2, dim=-1)
                    txt_modulated, txt_gate1 = _modulate(block.txt_norm1(txt), txt_mod1)
                    txt_q, txt_k, txt_v = _qkv(txt_modulated, attn, 1, self.options.fuse_qkv).unflatten(
                        -1, (3 * heads, head_dim)).chunk(3, 2)
                    txt_q = apply_rotary_emb_qwen(attn.norm_added_q(txt_q), text["freqs"], use_real=False)
                    txt_k = apply_rotary_emb_qwen(attn.norm_added_k(txt_k), text["freqs"], use_real=False)
                    if self.options.kv_buffer:
                        start, length = text["rows"]
                        keys[start : start + length] = txt_k[0]
                        values[start : start + length] = txt_v[0]
                        txt_keys, txt_values = keys, values
                    else:
                        self.condition_kv[layer][("text", condition.text)] = (txt_k[0], txt_v[0].contiguous())
                        txt_keys, txt_values = torch.cat([txt_k[0], keys]), torch.cat([txt_v[0], values])
                    txt_attn = attn.to_add_out(text["attention"](txt_q[0], txt_keys, txt_values).flatten(1, 2)[None])
                out = attention(query[0], keys, values).flatten(1, 2)[None]
                out = attn.to_out[0](out)
                if len(attn.to_out) > 1:
                    out = attn.to_out[1](out)
                hidden = hidden + gate1 * out
                modulated2, gate2 = _modulate(block.img_norm2(hidden), img_mod2)
                hidden = hidden + gate2 * block.img_mlp(modulated2)
                if txt is not None and layer != last:  # the last block's text output has no consumer
                    txt = txt + txt_gate1 * txt_attn
                    txt_modulated2, txt_gate2 = _modulate(block.txt_norm2(txt), txt_mod2)
                    txt = txt + txt_gate2 * block.txt_mlp(txt_modulated2)

    def _text_condition(self, branch: int, slots: Sequence[int], lengths: Sequence[int]) -> dict:
        """A branch's text condition in its condition pass: its hidden states, RoPE, key rows and the
        attention of its queries over its own keys and the pass's images (without key buffers, over
        ``[text | images]`` concatenated per block)."""

        length, count = self.text_lengths[branch], sum(lengths)
        rows, keys = None, (0, length + count)
        if self.options.kv_buffer:
            start, first = self.layout.part("text", branch)[0], min(slots)
            if max(slot + size for slot, size in zip(slots, lengths)) - first != count or first not in (
                start + length, start - count
            ):
                raise ValueError("a text condition must be contiguous with its pass's images in the key rows")
            rows, keys = (start, length), (min(start, first), length + count)
        return {"hidden": self.text_hidden[branch], "freqs": self.text_freqs[branch], "rows": rows,
                "attention": _Attention([(0, length, *keys)], self.device, self.head_dim ** -0.5, self.options.attention)}

    def _condition_pass_fused(self, hidden, temb, freqs, lengths, slots, attention, text=None):
        heads, head_dim = self.heads, self.head_dim
        hidden = hidden[0]
        rows = hidden.shape[0]
        query = torch.empty(rows, heads, head_dim, device=self.device, dtype=self.dtype)
        q_rows = torch.arange(rows, device=self.device, dtype=torch.int32)
        k_rows = torch.cat([torch.arange(slot, slot + length, device=self.device, dtype=torch.int32)
                            for slot, length in zip(slots, lengths)])
        cos, sin = freqs.real.float().contiguous(), freqs.imag.float().contiguous()
        if text is not None:
            txt, (start, length) = text["hidden"][0], text["rows"]
            txt_query = torch.empty(length, heads, head_dim, device=self.device, dtype=self.dtype)
            txt_q_rows = torch.arange(length, device=self.device, dtype=torch.int32)
            txt_k_rows = torch.arange(start, start + length, device=self.device, dtype=torch.int32)
            txt_cos, txt_sin = text["freqs"].real.float().contiguous(), text["freqs"].imag.float().contiguous()
        last = len(self.dit.transformer_blocks) - 1
        for layer, block in enumerate(self.dit.transformer_blocks):
            attn = block.attn
            (shift1, scale1, gate1), (shift2, scale2, gate2) = (
                part.chunk(3, dim=-1) for part in block.img_mod(temb)[-1].chunk(2, dim=-1))
            modulated = fused_ops.ln_modulate(hidden, shift1, scale1, block.img_norm1.eps)
            qkv = self._fused_qkv(modulated, attn, 0)
            keys, values = self.key_buffer[layer], self.value_buffer[layer]
            fused_ops.qk_norm_rope_scatter(qkv, attn.norm_q.weight, attn.norm_k.weight, cos, sin, attn.norm_q.eps,
                                           heads, q_rows, k_rows, query, keys, values)
            if text is not None:
                (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = (
                    part.chunk(3, dim=-1) for part in block.txt_mod(temb[-1:])[0].chunk(2, dim=-1))
                txt_modulated = fused_ops.ln_modulate(txt, txt_shift1, txt_scale1, block.txt_norm1.eps)
                fused_ops.qk_norm_rope_scatter(self._fused_qkv(txt_modulated, attn, 1), attn.norm_added_q.weight,
                                               attn.norm_added_k.weight, txt_cos, txt_sin, attn.norm_added_q.eps, heads,
                                               txt_q_rows, txt_k_rows, txt_query, keys, values)
                txt_attn = self._project(text["attention"](txt_query, keys, values).flatten(1, 2), attn.to_add_out)
            out = self._project(attention(query, keys, values).flatten(1, 2), attn.to_out[0])
            if len(attn.to_out) > 1:
                out = attn.to_out[1](out)
            hidden = fused_ops.gate_residual(hidden, out, gate1)
            modulated2 = fused_ops.ln_modulate(hidden, shift2, scale2, block.img_norm2.eps)
            hidden = fused_ops.gate_residual(hidden, _feed_forward(modulated2, block.img_mlp, self.options.fused_lora), gate2)
            if text is not None and layer != last:  # the last block's text output has no consumer
                txt = fused_ops.gate_residual(txt, txt_attn, txt_gate1)
                txt_modulated2 = fused_ops.ln_modulate(txt, txt_shift2, txt_scale2, block.txt_norm2.eps)
                txt = fused_ops.gate_residual(
                    txt, _feed_forward(txt_modulated2, block.txt_mlp, self.options.fused_lora), txt_gate2)

    def _fused_qkv(self, x, attn, stream: int):
        if self.options.fused_lora:
            return _qkv_fused_lora(x, attn, stream)
        return _qkv(x[None], attn, stream, self.options.fuse_qkv)[0]

    def _project(self, x, module):
        return _linear_fused_lora(x, module) if self.options.fused_lora else module(x)

    @property
    def condition_bytes(self) -> int:
        if self.options.kv_buffer:
            return 2 * self.key_buffer.numel() * self.key_buffer.element_size()
        return sum(t.numel() * t.element_size() for layer in self.condition_kv for kv in layer.values() for t in kv)

    def _groups(self) -> List[_Group]:
        n, scale = self.query_count, self.head_dim ** -0.5
        layout = self.layout
        # A text condition is cached with the conditions: a branch's step rows are its image queries.
        text_lengths = [0] * len(self.branches) if self.text_condition else self.text_lengths
        if len(self.branches) == 2 and self.options.batch_cfg:
            t0, t1 = text_lengths
            members = [((0, 1),((0, t0), (t0 + 2 * n, t0 + 2 * n + t1)), (t0, t0 + 2 * n),
                        ((0, layout.part("step", 0)[0], t0 + n), (t0 + n, layout.part("step", 1)[0], n + t1)))]
            queries = [(0, t0 + n), (t0 + n, n + t1)]
        else:
            members, queries = [], []
            for branch, length in enumerate(text_lengths):
                text, img = ((0, length), (length, length + n)) if branch == 0 else ((n, n + length), (0, n))
                members.append(((branch,), (text,), img, ((0, layout.part("step", branch)[0], length + n),)))
                queries.append([(0, length + n)])
        groups = []
        for index, (branches, text_rows, img_rows, writes) in enumerate(members):
            branch_queries = queries if len(branches) == 2 else queries[index]
            first = min(layout.branch_keys[branch][0] for branch in branches)
            last = max(sum(layout.branch_keys[branch]) for branch in branches)
            key_parts = tuple(part for part in layout.parts if part[2] >= first and part[2] + part[3] <= last)
            base = 0 if self.options.kv_buffer else first  # concatenated keys start at the group's first part
            sequences = [(q_start, q_len, layout.branch_keys[branch][0] - base, layout.branch_keys[branch][1])
                         for (q_start, q_len), branch in zip(branch_queries, branches)]
            groups.append(_Group(branches, text_rows, img_rows, writes, key_parts,
                                 _Attention(sequences, self.device, scale, self.options.attention)))
        return groups

    def _fused_group_state(self, group: _Group) -> dict:
        """Static per-group tensors of the fused path: the packed query, the query/key row of every
        image and text token, and their RoPE as cos/sin."""

        def key_row(q_row):
            for q_start, k_start, length in group.step_writes:
                if q_start <= q_row < q_start + length:
                    return k_start + q_row - q_start
            raise ValueError(f"query row {q_row} has no key row")

        def rows(values):
            return torch.tensor(values, device=self.device, dtype=torch.int32)

        img_q = list(range(*group.img_rows))
        txt_q = [row for start, end in group.text_rows for row in range(start, end)]
        copies = len(group.branches)
        img_freqs = self.img_freqs if copies == 1 else self.img_freqs.repeat(copies, 1)
        query_rows = sum(length for _, length, _, _ in group.attention.sequences)
        state = {
            "query": torch.empty(query_rows, self.heads, self.head_dim, device=self.device, dtype=self.dtype),
            "img_q": rows(img_q), "img_k": rows([key_row(row) for row in img_q]),
            "img_cos": img_freqs.real.float().contiguous(), "img_sin": img_freqs.imag.float().contiguous(),
        }
        if txt_q:  # no text queries with a text condition
            txt_freqs = torch.cat([self.text_freqs[branch] for branch in group.branches])
            state.update(txt_q=rows(txt_q), txt_k=rows([key_row(row) for row in txt_q]),
                         txt_cos=txt_freqs.real.float().contiguous(), txt_sin=txt_freqs.imag.float().contiguous())
        return state

    def warm_attention(self):
        """One attention call per group at this sample's shapes, outside any graph capture (cuDNN
        builds its execution plans at the first call of a shape)."""

        shape = (self.heads, self.head_dim)
        for group in self.groups:
            rows = sum(length for _, length, _, _ in group.attention.sequences)
            query = torch.zeros(rows, *shape, device=self.device, dtype=self.dtype)
            if self.options.kv_buffer:
                keys = self.key_buffer[0]
            else:
                keys = torch.zeros(sum(part[3] for part in group.key_parts), *shape, device=self.device, dtype=self.dtype)
            group.attention(query, keys, keys)

    def _keys(self, layer: int, step_key, step_value, group: _Group):
        """This block's keys/values of ``group``: the buffer with this step's rows written in place,
        or the same layout concatenated from the per-image condition K/V."""

        if self.options.kv_buffer:
            keys, values = self.key_buffer[layer], self.value_buffer[layer]
            for q_row, k_row, length in group.step_writes:
                keys[k_row : k_row + length] = step_key[q_row : q_row + length]
                values[k_row : k_row + length] = step_value[q_row : q_row + length]
            return keys, values
        writes = {k_row: (q_row, length) for q_row, k_row, length in group.step_writes}
        parts_k, parts_v = [], []
        for kind, key, start, length in group.key_parts:
            if kind == "step":
                q_row, _ = writes[start]
                parts_k.append(step_key[q_row : q_row + length])
                parts_v.append(step_value[q_row : q_row + length])
            else:
                parts_k.append(self.condition_kv[layer][(kind, key)][0])
                parts_v.append(self.condition_kv[layer][(kind, key)][1])
        return torch.cat(parts_k), torch.cat(parts_v)

    # ---- one step ---------------------------------------------------------------------------

    def _timestep_rows(self, timesteps: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = [timesteps[name].to(self.dtype) for name in self.timestep_names]
        return torch.cat(rows + [torch.zeros_like(rows[0])])

    def modulation_table(self, steps: Sequence[Mapping[str, torch.Tensor]]):
        """The t-only part of the forward for many steps at once: ``temb`` [S, R, D] and every block's
        image / text modulation rows [S, blocks, R, 6D] / [S, blocks, 1, 6D] (one GEMM per block and
        stream; the per-step forward computes the same rows with a GEMV per step). No text rows (None)
        with a text condition: it is modulated at t=0 in the condition pass."""

        dit = self.dit
        rows = torch.stack([self._timestep_rows(timesteps) for timesteps in steps])  # [S, R]
        count, width = rows.shape
        temb = dit.time_text_embed(rows.flatten(), self.text_hidden[0]).view(count, width, -1)
        image = torch.stack([block.img_mod(temb.flatten(0, 1)).view(count, width, -1)
                             for block in dit.transformer_blocks], dim=1)
        text = None if self.text_condition else torch.stack(
            [block.txt_mod(temb[:, :1].flatten(0, 1)).view(count, 1, -1) for block in dit.transformer_blocks], dim=1)
        return temb, image, text

    def step(self, latents: torch.Tensor, timesteps: Mapping[str, torch.Tensor], modulation=None) -> List[torch.Tensor]:
        """One forward of every branch. ``latents`` [1, n, C] (segment order); ``timesteps`` maps every
        timestep name to a ``[1]`` tensor scaled like the trainer's ``timestep / 1000``;
        ``modulation`` optionally gives this step's rows of :meth:`modulation_table` instead.
        Returns ``[1, n, C_out]`` per branch."""

        dit, n = self.dit, self.query_count
        hidden = latents.to(self.dtype)
        if self.geometry is not None:
            hidden = hidden + self.geometry
        image = dit.img_in(hidden)
        if modulation is None:
            temb = dit.time_text_embed(self._timestep_rows(timesteps), image)
        else:
            temb = modulation[0]
        outputs: List[Optional[torch.Tensor]] = [None] * len(self.branches)
        for position, group in enumerate(self.groups):
            copies = len(group.branches)
            img = image if copies == 1 else image.repeat(1, copies, 1)
            txt = None if self.text_condition else torch.cat([self.text_hidden[branch] for branch in group.branches], dim=1)
            mods = None if modulation is None else modulation[1:]
            if self.fused_state is not None:
                img = self._blocks_fused(group, self.fused_state[position], img, txt, temb, mods)
            else:
                img = self._blocks(group, img, txt, temb, mods)
            out = dit.proj_out(self._norm_out(img, temb))
            for slot, branch in enumerate(group.branches):
                outputs[branch] = out[:, slot * n : (slot + 1) * n]
        return outputs

    def _norm_out(self, img, temb):
        """``norm_out`` with each segment run's own timestep row (the model's ``[core | ring]`` readout)."""

        if len(self.runs) == 1:
            row = self.runs[0][2]
            return self.dit.norm_out(img, temb[row : row + 1])
        n = self.query_count
        return torch.cat([self.dit.norm_out(img[:, copy * n + start : copy * n + end], temb[row : row + 1])
                          for copy in range(img.shape[1] // n) for start, end, row in self.runs], dim=1)

    def _modulate_image(self, x, params):
        """Image-stream modulation: each run of segments with its timestep row. Returns (x, gates)."""

        if len(self.runs) == 1:
            row = self.runs[0][2]
            out, gate = _modulate(x, params[row : row + 1])
            return out, ((0, x.shape[1], gate),)
        n, pieces, gates = self.query_count, [], []
        for copy in range(x.shape[1] // n):
            for start, end, row in self.runs:
                piece, gate = _modulate(x[:, copy * n + start : copy * n + end], params[row : row + 1])
                pieces.append(piece)
                gates.append((copy * n + start, copy * n + end, gate))
        return torch.cat(pieces, dim=1), tuple(gates)

    @staticmethod
    def _gated(residual, values, gates):
        if len(gates) == 1:
            return residual + gates[0][2] * values
        return residual + torch.cat([gate * values[:, start:end] for start, end, gate in gates], dim=1)

    def _pack(self, text, image, group: _Group):
        """``[1, M, H, Dh]`` text and image parts -> ``[S, H, Dh]`` in the packed query order (``text``
        None: a text condition, no text rows)."""

        if text is None:
            return image[0]
        text, image = text[0], image[0]
        if len(group.branches) == 2:
            split = self.text_lengths[group.branches[0]]
            return torch.cat([text[:split], image, text[split:]])
        return torch.cat([text, image] if group.text_rows[0][0] == 0 else [image, text])

    def _blocks(self, group: _Group, img, txt, temb, modulation=None):
        dit, heads, head_dim = self.dit, self.heads, self.head_dim
        fused = self.options.fuse_qkv
        copies = len(group.branches)
        img_freqs = self.img_freqs if copies == 1 else self.img_freqs.repeat(copies, 1)
        txt_freqs = None if txt is None else torch.cat([self.text_freqs[branch] for branch in group.branches])
        img_start, img_end = group.img_rows
        last = len(dit.transformer_blocks) - 1
        for index, block in enumerate(dit.transformer_blocks):
            attn = block.attn
            if modulation is None:
                img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
            else:
                img_mod1, img_mod2 = modulation[0][index].chunk(2, dim=-1)
            img_modulated, img_gate1 = self._modulate_image(block.img_norm1(img), img_mod1)
            img_q, img_k, img_v = _qkv(img_modulated, attn, 0, fused).unflatten(-1, (3 * heads, head_dim)).chunk(3, 2)
            img_q = apply_rotary_emb_qwen(attn.norm_q(img_q), img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(attn.norm_k(img_k), img_freqs, use_real=False)
            txt_q = txt_k = txt_v = None  # a text condition has no text stream: its K/V are cached
            if txt is not None:
                txt_mod1, txt_mod2 = (block.txt_mod(temb[:1]) if modulation is None else modulation[1][index]).chunk(
                    2, dim=-1)
                txt_modulated, txt_gate1 = _modulate(block.txt_norm1(txt), txt_mod1)
                txt_q, txt_k, txt_v = _qkv(txt_modulated, attn, 1, fused).unflatten(-1, (3 * heads, head_dim)).chunk(3, 2)
                txt_q = apply_rotary_emb_qwen(attn.norm_added_q(txt_q), txt_freqs, use_real=False)
                txt_k = apply_rotary_emb_qwen(attn.norm_added_k(txt_k), txt_freqs, use_real=False)

            keys, values = self._keys(index, self._pack(txt_k, img_k, group), self._pack(txt_v, img_v, group), group)
            joint = group.attention(self._pack(txt_q, img_q, group), keys, values).flatten(1, 2)[None]

            img_attn = attn.to_out[0](joint[:, img_start:img_end])
            if len(attn.to_out) > 1:
                img_attn = attn.to_out[1](img_attn)
            if txt is not None:
                text_parts = [joint[:, start:end] for start, end in group.text_rows]
                txt_attn = attn.to_add_out(text_parts[0] if len(text_parts) == 1 else torch.cat(text_parts, dim=1))

            img = self._gated(img, img_attn, img_gate1)
            img_modulated2, img_gate2 = self._modulate_image(block.img_norm2(img), img_mod2)
            img = self._gated(img, block.img_mlp(img_modulated2), img_gate2)
            if txt is not None and index != last:  # the last block's text output has no consumer
                txt = txt + txt_gate1 * txt_attn
                txt_modulated2, txt_gate2 = _modulate(block.txt_norm2(txt), txt_mod2)
                txt = txt + txt_gate2 * block.txt_mlp(txt_modulated2)
        return img


    def _image_runs(self, rows: int):
        """(start, end, modulation row) of every run of segments in an image stream of ``rows`` tokens."""

        n = self.query_count
        return [(copy * n + start, copy * n + end, row) for copy in range(rows // n) for start, end, row in self.runs]

    def _blocks_fused(self, group: _Group, state: dict, img, txt, temb, modulation=None):
        dit, heads = self.dit, self.heads
        img, txt = img[0], None if txt is None else txt[0]
        runs = self._image_runs(img.shape[0])
        query = state["query"]
        img_start, img_end = group.img_rows
        last = len(dit.transformer_blocks) - 1

        def image_modulate(x, params, eps):
            if len(runs) == 1:
                shift, scale, _ = params[runs[0][2]].chunk(3)
                return fused_ops.ln_modulate(x, shift, scale, eps)
            out = torch.empty_like(x)
            for start, end, row in runs:
                shift, scale, _ = params[row].chunk(3)
                fused_ops.ln_modulate(x[start:end], shift, scale, eps, out=out[start:end])
            return out

        def image_gate(residual, values, params):
            if len(runs) == 1:
                return fused_ops.gate_residual(residual, values, params[runs[0][2]].chunk(3)[2])
            out = torch.empty_like(residual)
            for start, end, row in runs:
                fused_ops.gate_residual(residual[start:end], values[start:end], params[row].chunk(3)[2],
                                        out=out[start:end])
            return out

        for index, block in enumerate(dit.transformer_blocks):
            attn = block.attn
            if modulation is None:
                img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
            else:
                img_mod1, img_mod2 = modulation[0][index].chunk(2, dim=-1)
            img_modulated = image_modulate(img, img_mod1, block.img_norm1.eps)
            streams = [(0, img_modulated, (attn.norm_q, attn.norm_k), "img")]
            if txt is not None:  # a text condition has no text stream: its K/V are cached
                txt_mod1, txt_mod2 = (block.txt_mod(temb[:1]) if modulation is None else modulation[1][index]).chunk(
                    2, dim=-1)
                (txt_shift1, txt_scale1, txt_gate1), (txt_shift2, txt_scale2, txt_gate2) = (
                    txt_mod1[0].chunk(3), txt_mod2[0].chunk(3))
                txt_modulated = fused_ops.ln_modulate(txt, txt_shift1, txt_scale1, block.txt_norm1.eps)
                streams.append((1, txt_modulated, (attn.norm_added_q, attn.norm_added_k), "txt"))

            keys, values = self.key_buffer[index], self.value_buffer[index]
            for stream, modulated, norms, prefix in streams:
                qkv = self._fused_qkv(modulated, attn, stream)
                fused_ops.qk_norm_rope_scatter(
                    qkv, norms[0].weight, norms[1].weight, state[f"{prefix}_cos"], state[f"{prefix}_sin"],
                    norms[0].eps, heads, state[f"{prefix}_q"], state[f"{prefix}_k"], query, keys, values,
                )
            joint = group.attention(query, keys, values).flatten(1, 2)

            img_attn = self._project(joint[img_start:img_end], attn.to_out[0])
            if len(attn.to_out) > 1:
                img_attn = attn.to_out[1](img_attn)
            if txt is not None:
                text_parts = [joint[start:end] for start, end in group.text_rows]
                txt_attn = self._project(text_parts[0] if len(text_parts) == 1 else torch.cat(text_parts), attn.to_add_out)

            fused_lora = self.options.fused_lora
            img = image_gate(img, img_attn, img_mod1)
            img = image_gate(img, _feed_forward(image_modulate(img, img_mod2, block.img_norm2.eps), block.img_mlp, fused_lora),
                             img_mod2)
            if txt is not None and index != last:  # the last block's text output has no consumer
                txt = fused_ops.gate_residual(txt, txt_attn, txt_gate1)
                txt_modulated2 = fused_ops.ln_modulate(txt, txt_shift2, txt_scale2, block.txt_norm2.eps)
                txt = fused_ops.gate_residual(txt, _feed_forward(txt_modulated2, block.txt_mlp, fused_lora), txt_gate2)
        return img[None]


# --------------------------------------------------------------------------------------------
# CUDA graph


class _GraphedStep:
    """``function(inputs)`` captured as one CUDA graph; ``inputs`` (a dict of tensors) are copied into
    static tensors at every call."""

    def __init__(self, function, inputs: Mapping[str, torch.Tensor], stream, warmup: bool):
        self.inputs = {name: value.clone() for name, value in inputs.items()}
        if warmup:  # lazy library state (cuBLAS workspaces) must exist before the first capture
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                function(self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
        # capture_begin/end directly: the torch.cuda.graph context also runs gc.collect() and
        # empty_cache() at every capture, which costs a per-sample recapture far more than the capture.
        # Each graph keeps its own private pool: a pool shared across samples is released when its last
        # graph is destroyed, and capturing into the stale handle fails.
        self.graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            try:
                self.outputs = function(self.inputs)
            finally:
                self.graph.capture_end()
        torch.cuda.current_stream().wait_stream(stream)

    def __call__(self, inputs: Mapping[str, torch.Tensor]):
        for name, value in inputs.items():
            self.inputs[name].copy_(value)
        self.graph.replay()
        return self.outputs


# --------------------------------------------------------------------------------------------
# Sampler


class FastCanvasSampler:
    """Drop-in for the trainer's cached canvas sampling (``sampling_from_embeddings``).

    Covers the canvas predict path with isolated conditions: zero_cond_t, reference CFG (or none),
    the text in the step or as a condition (``text_as_condition``), latent anchoring inside R (the
    trainer's post-step blend to the source trajectory), no text true CFG, no ring, batch size 1,
    the whole schedule from noise. Construction merges/fuses the DiT weights in place when asked
    (the model's own forward stays valid). ``install()`` routes the trainer's
    ``sampling_from_embeddings`` here.
    """

    def __init__(self, trainer, options: FastCanvasOptions = FastCanvasOptions()):
        self.trainer = trainer
        self.options = options
        self.dit = trainer._unwrap_dit()
        if options.cuda_graph and self.dit.img_in.weight.device.type != "cuda":
            raise ValueError("cuda_graph needs the DiT on a CUDA device")
        self.merged_layers = merge_lora_(self.dit) if options.merge_lora else 0
        if options.fuse_qkv:
            fuse_qkv_(self.dit)
        if options.cuda_graph:
            self._stream = torch.cuda.Stream()
        self._warm = False
        self.last: Dict[str, object] = {}

    def install(self):
        """Route ``trainer.sampling_from_embeddings`` to :meth:`sample`; returns the original."""

        original = self.trainer.sampling_from_embeddings
        self.trainer.sampling_from_embeddings = lambda embeddings, **kwargs: (
            original(embeddings, **kwargs) if kwargs else self.sample(embeddings)
        )
        return original

    @torch.inference_mode()
    def prepare(self, embeddings: dict) -> dict:
        """The trainer loop's set-up for the canvas + cache path: initial noise (same RNG draw), plan,
        condition streams, schedule, CFG inputs, latent anchoring inside R."""

        trainer, dit = self.trainer, self.dit
        canvas = trainer._complementary_canvas_cfg()
        if canvas is None or not canvas.isolate_conditions:
            raise ValueError("the fast canvas sampler needs model.complementary_canvas.isolate_conditions")
        # Later canvas variants: a ring is one more QuerySegment (own positions and timestep, its own
        # schedule in the loop).
        unsupported = [name for name in ("ring_tokens", "anneal_steps") if getattr(canvas, name, 0)]
        if unsupported:
            raise NotImplementedError(f"the fast canvas sampler does not cover {', '.join(unsupported)} yet")
        isolation = getattr(canvas, "isolation", "image")
        text_condition = bool(getattr(canvas, "text_as_condition", False))
        if getattr(getattr(trainer.config.model, "flow_map", None), "enabled", False):
            raise ValueError("flow-map sampling has its own loop")
        anchoring_settings = getattr(getattr(trainer.config, "predict", None), "latent_anchoring", None)
        if embeddings.get("_sigma_handoff") is not None or getattr(anchoring_settings, "source_noise_start", None) is not None:
            raise ValueError("the complementary canvas samples the whole schedule from noise (no σ hand-off or "
                             "source_noise_start)")
        anchor_source, anchor_weight = embeddings.get("anchor_source_latents"), embeddings.get("anchor_edit_weight")
        if (anchor_source is None) != (anchor_weight is None):
            raise ValueError("latent anchoring requires both anchor_source_latents and anchor_edit_weight")
        if embeddings["true_cfg_scale"] > 1 and embeddings.get("negative_prompt") is not None:
            raise NotImplementedError("text true CFG is not part of the fast canvas path")
        reference = embeddings.get("reference_cfg_embeddings")
        reference_scale = float(embeddings.get("reference_cfg_scale", 1.0))
        do_reference_cfg = reference is not None and reference_scale > 1
        batch_size = embeddings["control_latents"].shape[0]
        if batch_size != 1:
            raise ValueError("the complementary canvas samples one image at a time")
        device, dtype = trainer._dit_device(), trainer.weight_dtype
        height, width = embeddings["height"], embeddings["width"]
        img_shapes = trainer._get_image_shapes(embeddings, batch_size)
        branches = [(embeddings["prompt_embeds"], embeddings["prompt_embeds_mask"], img_shapes,
                     embeddings["control_latents"].to(device=device, dtype=dtype))]

        mask_geometry = embeddings.get("mask_geometry")
        if mask_geometry is not None:
            trainer._validate_mask_geometry_image_grid(mask_geometry, name="positive mask_geometry",
                                                       batch_size=batch_size, height_image=height, width_image=width)
            mask_geometry = mask_geometry.to(device=device, dtype=dtype)
        if do_reference_cfg:
            reference_geometry = reference.get("mask_geometry")
            if (mask_geometry is None) != (reference_geometry is None) or (
                mask_geometry is not None
                and not trainer._mask_geometries_equal(embeddings["mask_geometry"], reference_geometry)
            ):
                raise ValueError("reference_cfg positive and null-reference mask geometry must be identical")
            reference_shapes = trainer._get_image_shapes(reference, batch_size)
            if reference_shapes[0][0] != img_shapes[0][0]:
                raise ValueError("reference_cfg branches must share the target token grid")
            branches.append((reference["prompt_embeds"], reference["prompt_embeds_mask"], reference_shapes,
                             reference["control_latents"].to(device=device, dtype=dtype)))
        for prompt, mask, _, _ in branches:
            if int(mask.sum()) != prompt.shape[1]:
                raise ValueError("padded prompts are not supported (one sample has no padding)")
        mask_geometry_tokens = None
        if mask_geometry is not None:
            mask_geometry_tokens = dit.prepare_mask_geometry_tokens(mask_geometry, img_shapes)

        latents = trainer.prepare_initial_latents(
            batch_size=batch_size, num_channels_latents=dit.config.in_channels // 4, height=height, width=width,
            device=device, dtype=dtype,
        )
        image_seq_len = latents.shape[1]
        reference_mask = None
        if do_reference_cfg and embeddings.get("reference_cfg_mask_aware", False):
            reference_mask = trainer._prepare_reference_cfg_mask(
                reference_cfg_mask=embeddings["reference_cfg_mask"], batch_size=batch_size, height_image=height,
                width_image=width, image_seq_len=image_seq_len, device=device,
            )
        plan = trainer._canvas_plan(embeddings, img_shapes, device)
        if getattr(plan, "ring_count", 0):
            raise NotImplementedError("ring tokens: pass them as a QuerySegment with their positions and timestep")
        anchoring = None
        if anchor_source is not None:
            # R's latent pixels outside M_free (and its soft transition) follow the source trajectory, as in a dense
            # anchored run: the trainer's post-step blend on the R tokens, the full-grid draw as trajectory noise.
            full = {"anchor_source_latents": anchor_source, "edit_weight": anchor_weight, "anchor_noise": latents}
            if any(value.shape != latents.shape for value in full.values()):
                raise ValueError("latent anchoring tensors must match the packed target latents")
            anchoring = {name: gather_tokens(value.to(device=device, dtype=latents.dtype), plan.target_index)
                         for name, value in full.items()}
        latents = gather_tokens(latents, plan.target_index)
        if reference_mask is not None:
            reference_mask = gather_tokens(reference_mask, plan.target_index)
        timesteps, _ = trainer.prepare_predict_timesteps(embeddings["num_inference_steps"], image_seq_len)
        if anchoring is not None and abs(float(trainer.scheduler.sigmas[0]) - 1.0) > 1e-6:
            raise ValueError("latent anchoring requires full-noise initialization (first scheduler sigma = 1)")

        conditions = [trainer._complementary_canvas_conditions(branch[3], plan) for branch in branches]
        ids = tuple(range(len(img_shapes[0]) - 1))
        passes = [ConditionPass(conditions[0], img_shapes, plan.image_token_index, ids, isolation,
                                0 if text_condition else None)]
        branch_ids = [ids]
        if do_reference_cfg:
            source_count = int(plan.condition_source_index.numel())
            if not text_condition and isolation == "image" and conditions[1].shape[1] == source_count and torch.equal(
                conditions[1], conditions[0][:, :source_count]
            ):
                branch_ids.append(ids[:1])  # per-image isolated source tokens never see the reference: shared
            else:  # a condition group or a text condition needs its own null pass
                null_ids = tuple(range(len(ids), len(ids) + len(branches[1][2][0]) - 1))
                passes.append(ConditionPass(conditions[1], branches[1][2], plan.image_token_index, null_ids, isolation,
                                            1 if text_condition else None))
                branch_ids.append(null_ids)
        return {
            "plan": plan, "latents": latents, "timesteps": timesteps, "passes": passes,
            "branches": [BranchInputs(branch[0], branch[2], branch_id) for branch, branch_id in zip(branches, branch_ids)],
            "source_latents": embeddings["canvas_source_latents"].to(device=device, dtype=dtype),
            "mask_geometry_tokens": mask_geometry_tokens, "reference_mask": reference_mask,
            "reference_scale": reference_scale, "do_reference_cfg": do_reference_cfg,
            "delta_clip": (bool(embeddings.get("reference_cfg_delta_clip_enabled", False)),
                           float(embeddings.get("reference_cfg_delta_clip_max_ratio", 0.5))),
            "sigma_min": float(embeddings.get("reference_cfg_sigma_min", 0.0)),
            "apg_eta": float(embeddings.get("reference_cfg_apg_eta", 1.0)),
            "anchoring": anchoring, "text_condition": text_condition,
        }

    def build_forward(self, prepared: dict) -> CanvasStepForward:
        plan = prepared["plan"]
        return CanvasStepForward(self.dit, [QuerySegment("core", plan.target_index)], prepared["branches"],
                                 prepared["passes"], prepared["mask_geometry_tokens"], self.options)

    def combine(self, prepared: dict, predictions: Sequence[torch.Tensor]) -> torch.Tensor:
        if not prepared["do_reference_cfg"]:
            return predictions[0]
        clip, ratio = prepared["delta_clip"]
        return self.trainer._combine_reference_cfg_predictions(
            noise_pred=predictions[0], reference_cfg_noise_pred=predictions[1],
            reference_cfg_scale=prepared["reference_scale"], reference_cfg_mask=prepared["reference_mask"],
            delta_clip_enabled=clip, delta_clip_max_ratio=ratio,
        )

    @staticmethod
    def scaled_timestep(t: torch.Tensor, dtype) -> torch.Tensor:
        """The DiT timestep of the trainer loop: ``t.expand(1).to(dtype) / 1000``."""

        return t.expand(1).to(dtype) / 1000

    def step_function(self, prepared: dict, forward: CanvasStepForward):
        """``step(latents, index) -> [prediction per branch..., guided prediction]`` at schedule step
        ``index`` (graphed if asked; the outputs of a graphed step are its static tensors)."""

        dtype = prepared["latents"].dtype
        scaled = [self.scaled_timestep(t, dtype) for t in prepared["timesteps"]]
        table = forward.modulation_table([{"t": t} for t in scaled]) if self.options.modulation_table else None

        def inputs(latents, index):
            entry = {"latents": latents, "t": scaled[index]}
            if table is not None:
                entry.update(temb=table[0][index], img_mod=table[1][index])
                if table[2] is not None:  # no text rows with a text condition
                    entry["txt_mod"] = table[2][index]
            return entry

        def run(entry):
            modulation = None if table is None else (entry["temb"], entry["img_mod"], entry.get("txt_mod"))
            predictions = forward.step(entry["latents"], {"t": entry["t"]}, modulation)
            return predictions + [self.combine(prepared, predictions)]

        if not self.options.cuda_graph:
            return lambda latents, index: run(inputs(latents, index))
        forward.warm_attention()
        graphed = _GraphedStep(run, inputs(prepared["latents"], 0), self._stream, warmup=not self._warm)
        self._warm = True
        return lambda latents, index: graphed(inputs(latents, index))

    @torch.inference_mode()
    def sample(self, embeddings: dict) -> torch.Tensor:
        trainer = self.trainer
        cuda = self.dit.img_in.weight.device.type == "cuda"

        def clock():
            if cuda:
                torch.cuda.synchronize()
            return time.perf_counter()

        start = clock()
        prepared = self.prepare(embeddings)
        forward = self.build_forward(prepared)
        built = clock()
        step = self.step_function(prepared, forward)
        captured = clock()
        latents, timesteps, anchoring = prepared["latents"], prepared["timesteps"], prepared["anchoring"]
        trainer.scheduler.set_begin_index(0)
        trainer._num_timesteps = len(timesteps)
        for index, t in enumerate(timesteps):
            outputs = step(latents, index)
            sigma = float(trainer.scheduler.sigmas[index])
            if not prepared["do_reference_cfg"] or sigma < prepared["sigma_min"]:
                noise_pred = outputs[0]
            elif prepared["apg_eta"] < 1.0:  # outside the graph: APG needs sigma and the current latents
                clip, ratio = prepared["delta_clip"]
                noise_pred = trainer._combine_reference_cfg_predictions(
                    noise_pred=outputs[0], reference_cfg_noise_pred=outputs[1],
                    reference_cfg_scale=prepared["reference_scale"], reference_cfg_mask=prepared["reference_mask"],
                    delta_clip_enabled=clip, delta_clip_max_ratio=ratio, apg_eta=prepared["apg_eta"],
                    latents=latents, sigma=sigma,
                )
            else:
                noise_pred = outputs[-1]
            latents_dtype = latents.dtype
            latents = trainer.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            if anchoring is not None:
                latents = trainer._apply_post_step_latent_anchoring(
                    generated_next=latents, timesteps=timesteps, step_index=index, **anchoring)
        done = clock()
        plan, branches = prepared["plan"], prepared["branches"]
        self.last = {
            "setup_s": built - start, "condition_pass_s": forward.condition_pass_s, "capture_s": captured - built,
            "loop_s": done - captured, "steps": len(timesteps), "options": self.options.label,
        }
        trainer._last_complementary_canvas = {  # the trainer's keys, and this sampler's timing
            "edit_tokens": int(plan.core_count),
            "ring_tokens": int(plan.ring_count),
            "grid_tokens": int(plan.num_tokens),
            "source_condition_tokens": int(plan.condition_source_index.numel()),
            "text_tokens": int(branches[0].prompt_embeds.shape[1]),
            "text_tokens_null": int(branches[1].prompt_embeds.shape[1]) if len(branches) == 2 else None,
            "text_cached": prepared["text_condition"],
            "condition_cache_bytes": int(forward.condition_bytes),
            "fast_sampler": dict(self.last),
        }
        del step, forward
        return scatter_tokens(prepared["source_latents"], plan.target_index, latents)

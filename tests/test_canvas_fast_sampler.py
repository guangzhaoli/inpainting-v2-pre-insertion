"""Fast canvas sampler: LoRA merge, fused QKV, packed CFG branches, key buffers and CUDA graphs
reproduce the model's cached isolated forward and the trainer's cached sampling loop."""

import inspect
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from src.data.config import ComplementaryCanvasConfig
from src.inference.canvas_fast_sampler import (
    BranchInputs,
    CanvasStepForward,
    ConditionPass,
    FastCanvasOptions,
    FastCanvasSampler,
    KeyLayout,
    QuerySegment,
    _Attention,
    fuse_qkv_,
    merge_lora_,
)
from src.models.complementary_canvas import canvas_plan_from_region
from src.models.mask_geometry import MaskGeometryTokenizer
from src.models.transformer_qwenimage import QwenImageTransformer2DModel
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

# 4x4 target/source token grids (64 px, 16 px per token) and a 3x3 reference.
IMG_SHAPES = [[(1, 4, 4), (1, 4, 4), (1, 3, 3)]]
NULL_SHAPES = [IMG_SHAPES[0][:2]]
REGION = torch.tensor([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
                  "img_mlp.net.0.proj", "img_mlp.net.2"]
CUDA = torch.cuda.is_available()


def _tiny_dit(lora=True, seed=7):
    torch.manual_seed(seed)
    model = QwenImageTransformer2DModel(
        num_layers=3, attention_head_dim=16, num_attention_heads=2, in_channels=16, out_channels=4,
        joint_attention_dim=16, axes_dims_rope=(4, 6, 6), zero_cond_t=True,
    )
    model.mask_geometry_tokenizer = MaskGeometryTokenizer(4, 16)
    torch.nn.init.normal_(model.mask_geometry_tokenizer.conv_out.weight, std=0.2)
    if lora:
        from peft import LoraConfig

        model.add_adapter(LoraConfig(r=4, lora_alpha=8, target_modules=TARGET_MODULES))
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.1)
    return model.eval()


def _inputs(seed=2):
    generator = torch.Generator().manual_seed(seed)
    plan = canvas_plan_from_region(REGION)
    n = int(plan.target_index.numel())
    return NS(
        plan=plan,
        target=torch.randn(1, n, 16, generator=generator),
        conditions=torch.randn(1, 10 + 9, 16, generator=generator),
        null_conditions=torch.randn(1, 10, 16, generator=generator),
        geometry=torch.randn(1, 3, 64, 64, generator=generator),
        positive=torch.randn(1, 5, 16, generator=generator),
        null=torch.randn(1, 3, 16, generator=generator),
    )


def _reference(model, x, timestep=0.6, shared=True, isolation="image", text=False):
    """The model's own cached forward per branch (what the trainer's cached loop calls); ``text``: each
    branch's prompt is a text condition, cached with that branch's condition images."""

    plan, device = x.plan, model.img_in.weight.device
    common = dict(mask_geometry_tokens=model.prepare_mask_geometry_tokens(x.geometry, IMG_SHAPES),
                  image_token_index=plan.image_token_index, output_token_count=x.target.shape[1], return_dict=False,
                  timestep=torch.full((1,), timestep, device=device), **({"text_as_condition": True} if text else {}))

    def build_cache(conditions, shapes, prompt):
        kwargs = {"isolation": isolation} if isolation != "image" else {}
        if text:
            kwargs.update(encoder_hidden_states=prompt, txt_seq_lens=[prompt.shape[1]])
        return model.build_condition_cache(conditions, shapes, plan.image_token_index, **kwargs)

    with torch.no_grad():
        cache = build_cache(x.conditions, IMG_SHAPES, x.positive)
        null_cache = cache.select([0]) if shared else build_cache(x.null_conditions, NULL_SHAPES, x.null)
        outputs = []
        for prompt, shapes, branch_cache in ((x.positive, IMG_SHAPES, cache), (x.null, NULL_SHAPES, null_cache)):
            outputs.append(model(
                x.target, encoder_hidden_states=prompt,
                encoder_hidden_states_mask=torch.ones(1, prompt.shape[1], device=device),
                img_shapes=shapes, txt_seq_lens=[prompt.shape[1]], condition_cache=branch_cache, **common,
            )[0])
    return outputs


def _fast(model, x, options, shared=True, segments=None, isolation="image", text=False):
    plan = x.plan
    passes = [ConditionPass(x.conditions, IMG_SHAPES, plan.image_token_index, (0, 1), isolation, 0 if text else None)]
    null_ids = (0,)
    if not shared:
        passes.append(ConditionPass(x.null_conditions, NULL_SHAPES, plan.image_token_index, (2,), isolation,
                                    1 if text else None))
        null_ids = (2,)
    branches = [BranchInputs(x.positive, IMG_SHAPES, (0, 1)), BranchInputs(x.null, NULL_SHAPES, null_ids)]
    geometry = model.prepare_mask_geometry_tokens(x.geometry, IMG_SHAPES)
    return CanvasStepForward(model, segments or [QuerySegment("core", plan.target_index)], branches, passes,
                             geometry, options)


PLAIN = dict(merge_lora=False, modulation_table=False, fused_kernels=False, fused_lora=False, cuda_graph=False)
OPTION_GRID = [
    FastCanvasOptions(**{**PLAIN, "fuse_qkv": fuse, "batch_cfg": batch, "kv_buffer": buffer})
    for fuse in (False, True) for batch in (False, True) for buffer in (False, True)
] + [FastCanvasOptions(**{**PLAIN, "batch_cfg": batch, "fused_kernels": True, "fused_lora": lora})
      for batch in (False, True) for lora in (False, True)]


@pytest.mark.parametrize("options", OPTION_GRID, ids=lambda o: o.label)
@pytest.mark.parametrize("shared", [True, False], ids=["shared-source", "own-null-conditions"])
def test_step_matches_the_cached_isolated_forward(options, shared):
    model, x = _tiny_dit(), _inputs()
    expected = _reference(model, x, shared=shared)
    if options.fuse_qkv:
        fuse_qkv_(model)
    forward = _fast(model, x, options, shared=shared)
    with torch.inference_mode():
        actual = forward.step(x.target, {"t": torch.full((1,), 0.6)})
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # The source K/V is stored once when both branches attend it.
    if options.kv_buffer:
        n, keys = x.target.shape[1], forward.layout.total
        assert keys == (5 + n) + 9 + 10 + (0 if shared else 10) + (n + 3)


@pytest.mark.skipif("text_as_condition" not in inspect.signature(QwenImageTransformer2DModel.forward).parameters,
                    reason="the model has no text condition")
@pytest.mark.parametrize("options", OPTION_GRID, ids=lambda o: o.label)
@pytest.mark.parametrize("isolation", ["image", "group"])
def test_text_condition_step_matches_the_cached_forward(options, isolation):
    """text_as_condition: each prompt runs once with its branch's condition pass (t=0, over itself and the
    pass's images); a step has only image queries and attends to the cached text K/V."""

    model, x = _tiny_dit(), _inputs()
    expected = _reference(model, x, shared=False, isolation=isolation, text=True)
    if options.fuse_qkv:
        fuse_qkv_(model)
    forward = _fast(model, x, options, shared=False, isolation=isolation, text=True)
    with torch.inference_mode():
        actual = forward.step(x.target, {"t": torch.full((1,), 0.6)})
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    if options.kv_buffer:  # [q_0 | text_0 | images_0 | images_1 | text_1 | q_1]
        n, keys = x.target.shape[1], forward.layout.total
        assert keys == n + 5 + (10 + 9) + 10 + 3 + n


def test_merged_lora_and_fused_qkv_keep_the_model_forward():
    model, x = _tiny_dit(), _inputs()
    expected = _reference(model, x)
    fuse_qkv_(model)
    for got, want in zip(_reference(model, x), expected):
        assert torch.equal(got, want)  # fused weights are views with the same values
    assert merge_lora_(model) == 3 * len(TARGET_MODULES)
    assert merge_lora_(model) == 0
    merged = _reference(model, x)  # PEFT now skips the merged adapters
    for got, want in zip(merged, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    with torch.inference_mode():
        actual = _fast(model, x, FastCanvasOptions(**{**PLAIN, "merge_lora": True})).step(x.target, {"t": torch.full((1,), 0.6)})
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_extra_query_segment_equals_a_larger_target():
    """A ring-like second segment with the core's timestep is just more target tokens."""

    model, x = _tiny_dit(), _inputs()
    region = REGION.clone()
    ring = torch.tensor([3, 4, 8])
    region[ring] = True
    plan = canvas_plan_from_region(region)
    core, extra = plan.target_index[~torch.isin(plan.target_index, ring)], ring
    # Expected: the cached forward with target [core, ring] (the tokens in that order).
    target_index = torch.cat([core, extra])
    target = torch.randn(1, target_index.numel(), 16, generator=torch.Generator().manual_seed(3))
    conditions = torch.randn(1, plan.source_index.numel() + 9, 16, generator=torch.Generator().manual_seed(4))
    index = (target_index, plan.source_index)
    x2 = NS(plan=NS(target_index=target_index, image_token_index=index), target=target, conditions=conditions,
            geometry=x.geometry, positive=x.positive, null=x.null, null_conditions=None)
    expected = _reference(model, x2)
    forward = _fast(model, x2, FastCanvasOptions(**PLAIN), segments=[QuerySegment("core", core), QuerySegment("ring", extra)])
    with torch.inference_mode():
        actual = forward.step(target, {"t": torch.full((1,), 0.6)})
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_modulation_table_matches_the_per_step_modulation():
    model, x = _tiny_dit(), _inputs()
    forward = _fast(model, x, FastCanvasOptions(**{**PLAIN, "modulation_table": True}))
    with torch.inference_mode():
        steps = [{"t": torch.full((1,), value)} for value in (0.9, 0.6, 0.3)]
        temb, image, text = forward.modulation_table(steps)
        assert temb.shape[:2] == (3, 2) and image.shape[:3] == (3, 3, 2) and text.shape[:3] == (3, 3, 1)
        for index, timesteps in enumerate(steps):
            expected = forward.step(x.target, timesteps)
            actual = forward.step(x.target, timesteps, (temb[index], image[index], text[index]))
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif("isolation" not in inspect.signature(QwenImageTransformer2DModel.build_condition_cache).parameters,
                    reason="the model has no condition-group isolation")
def test_group_isolated_conditions_match_the_model():
    model, x = _tiny_dit(), _inputs()
    plan = x.plan
    common = dict(mask_geometry_tokens=model.prepare_mask_geometry_tokens(x.geometry, IMG_SHAPES),
                  image_token_index=plan.image_token_index, output_token_count=x.target.shape[1], return_dict=False,
                  timestep=torch.full((1,), 0.6))
    with torch.no_grad():
        cache = model.build_condition_cache(x.conditions, IMG_SHAPES, plan.image_token_index, isolation="group")
        null_cache = model.build_condition_cache(x.conditions[:, :10], NULL_SHAPES, plan.image_token_index,
                                                 isolation="group")
        expected = [model(x.target, encoder_hidden_states=prompt, encoder_hidden_states_mask=torch.ones(1, prompt.shape[1]),
                          img_shapes=shapes, txt_seq_lens=[prompt.shape[1]], condition_cache=branch_cache, **common)[0]
                    for prompt, shapes, branch_cache in ((x.positive, IMG_SHAPES, cache), (x.null, NULL_SHAPES, null_cache))]
    passes = [ConditionPass(x.conditions, IMG_SHAPES, plan.image_token_index, (0, 1), "group"),
              ConditionPass(x.conditions[:, :10], NULL_SHAPES, plan.image_token_index, (2,), "group")]
    branches = [BranchInputs(x.positive, IMG_SHAPES, (0, 1)), BranchInputs(x.null, NULL_SHAPES, (2,))]
    for options in (FastCanvasOptions(**PLAIN),
                    FastCanvasOptions(**{**PLAIN, "kv_buffer": False, "batch_cfg": False}),
                    FastCanvasOptions(**{**PLAIN, "fused_kernels": True})):
        forward = CanvasStepForward(model, [QuerySegment("core", plan.target_index)], branches, passes,
                                    model.prepare_mask_geometry_tokens(x.geometry, IMG_SHAPES), options)
        with torch.inference_mode():
            actual = forward.step(x.target, {"t": torch.full((1,), 0.6)})
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif("ring_token_count" not in inspect.signature(QwenImageTransformer2DModel.forward).parameters,
                    reason="the model has no canvas ring")
def test_ring_segment_with_its_own_timestep_matches_the_model_ring():
    """The ring plugs in as a second query segment with its own timestep row ([t, t_ring, 0])."""

    model, x = _tiny_dit(), _inputs()
    region = REGION.clone()
    ring = torch.tensor([3, 4, 8, 10])
    core = torch.nonzero(region).flatten()
    region[ring] = True
    source = torch.nonzero(~region).flatten()
    target_index = torch.cat([core, ring])
    index = (target_index, source)
    generator = torch.Generator().manual_seed(5)
    target = torch.randn(1, target_index.numel(), 16, generator=generator)
    conditions = torch.randn(1, source.numel() + 9, 16, generator=generator)
    t, t_ring = torch.full((1,), 0.8), torch.full((1,), 0.3)
    geometry = model.prepare_mask_geometry_tokens(x.geometry, IMG_SHAPES)
    with torch.no_grad():
        cache = model.build_condition_cache(conditions, IMG_SHAPES, index, target_groups=2)
        expected = [model(target, encoder_hidden_states=prompt, encoder_hidden_states_mask=torch.ones(1, prompt.shape[1]),
                          timestep=t, img_shapes=shapes, txt_seq_lens=[prompt.shape[1]], mask_geometry_tokens=geometry,
                          image_token_index=index, output_token_count=target.shape[1], condition_cache=branch_cache,
                          ring_token_count=ring.numel(), ring_timestep=t_ring, return_dict=False)[0]
                    for prompt, shapes, branch_cache in ((x.positive, IMG_SHAPES, cache), (x.null, NULL_SHAPES, cache.select([0])))]
    passes = [ConditionPass(conditions, IMG_SHAPES, index, (0, 1))]
    branches = [BranchInputs(x.positive, IMG_SHAPES, (0, 1)), BranchInputs(x.null, NULL_SHAPES, (0,))]
    segments = [QuerySegment("core", core), QuerySegment("ring", ring, timestep="t_ring")]
    for options in (FastCanvasOptions(**PLAIN), FastCanvasOptions(cuda_graph=False)):
        forward = CanvasStepForward(model, segments, branches, passes, geometry, options, timesteps=("t", "t_ring"))
        with torch.inference_mode():
            steps = {"t": t, "t_ring": t_ring}
            table = forward.modulation_table([steps])
            actual = forward.step(target, steps, tuple(part[0] for part in table) if options.modulation_table else None)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_key_layout_overlaps_on_shared_conditions():
    layout = KeyLayout.build([7, 9], [(0, 1), (0,)], {0: 10, 1: 4})
    assert [part[:2] for part in layout.parts] == [("step", 0), ("image", 1), ("image", 0), ("step", 1)]
    assert layout.branch_keys == ((0, 7 + 4 + 10), (11, 10 + 9)) and layout.total == 30
    own = KeyLayout.build([7, 9], [(0, 1), (2,)], {0: 10, 1: 4, 2: 5})
    assert own.branch_keys == ((0, 21), (21, 14))
    single = KeyLayout.build([7], [(0, 1)], {0: 10, 1: 4})
    assert single.branch_keys == ((0, 21),)
    # A text condition sits between its branch's step rows and images.
    text = KeyLayout.build([7, 9], [(0, 1), (2,)], {0: 10, 1: 4, 2: 5}, text_lengths=[3, 2])
    assert [part[:2] for part in text.parts] == [("step", 0), ("text", 0), ("image", 0), ("image", 1), ("image", 2),
                                                 ("text", 1), ("step", 1)]
    assert text.branch_keys == ((0, 7 + 3 + 14), (24, 5 + 2 + 9)) and text.total == 40


class _EulerScheduler:
    def __init__(self, sigmas):
        self.sigmas = sigmas
        self.index = 0

    def set_begin_index(self, index):
        self.index = index

    def step(self, velocity, _timestep, latents, return_dict=False):
        latents = latents + (self.sigmas[self.index + 1] - self.sigmas[self.index]) * velocity
        self.index += 1
        return (latents,)

    def scale_noise(self, sample, _timestep, noise):
        """diffusers after ``set_begin_index``: the σ of the current step index (the next step's after ``step``)."""

        sigma = self.sigmas[self.index]
        return sigma * noise + (1.0 - sigma) * sample


def _trainer(dit, canvas=None):
    trainer = object.__new__(QwenImageEditTrainer)
    canvas = canvas or ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, cache_conditions=True)
    trainer.config = NS(model=NS(complementary_canvas=canvas))
    trainer.accelerator = None
    trainer.dit = dit
    trainer.weight_dtype = torch.float32
    trainer.vae_scale_factor = 8
    trainer.scheduler = _EulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    trainer.prepare_initial_latents = Mock(return_value=torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(9)))
    trainer.prepare_predict_timesteps = Mock(return_value=(torch.tensor([1000.0, 700.0, 400.0]), 3))
    return trainer


def _embeddings():
    torch.manual_seed(4)
    hole = torch.zeros(1, 64, 64)
    hole[:, 16:48, 16:40] = 1.0
    branch = {"mask_geometry": torch.randn(1, 3, 64, 64), "height": 64, "width": 64, "height_control": 64,
              "width_control": 64}
    controls = torch.randn(1, 16 + 9, 16)
    null = {**branch, "prompt_embeds": torch.randn(1, 3, 16), "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
            "control_latents": controls[:, :16], "n_controls": 0}
    return {
        **branch, "prompt_embeds": torch.randn(1, 5, 16), "prompt_embeds_mask": torch.ones(1, 5, dtype=torch.long),
        "control_latents": controls, "n_controls": 1, "control_1": torch.zeros(1), "height_control_1": 48,
        "width_control_1": 48, "num_inference_steps": 3, "true_cfg_scale": 2.5, "guidance": 1.0,
        "reference_cfg_embeddings": null, "reference_cfg_scale": 2.5, "reference_cfg_mask_aware": True,
        "reference_cfg_mask": hole, "reference_cfg_delta_clip_enabled": True,
        "reference_cfg_delta_clip_max_ratio": 1.0,
        "canvas_region": REGION[None].clone(), "canvas_source_latents": torch.randn(1, 16, 16),
    }


@pytest.mark.parametrize("merge", [False, True], ids=["default-unmerged", "merged"])
@pytest.mark.parametrize("batch_cfg", [False, True])
def test_sampler_matches_the_trainer_cached_loop(batch_cfg, merge):
    embeddings = _embeddings()
    trainer = _trainer(_tiny_dit())
    with patch("src.trainer.qwen_image_edit_trainer.tqdm", side_effect=lambda iterable, **_: iterable):
        expected = trainer.sampling_from_embeddings(embeddings)
    expected_canvas = dict(trainer._last_complementary_canvas)
    trainer.scheduler = _EulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    sampler = FastCanvasSampler(trainer, FastCanvasOptions(batch_cfg=batch_cfg, cuda_graph=False, merge_lora=merge))
    assert sampler.merged_layers == (3 * len(TARGET_MODULES) if merge else 0)
    sampler.install()
    actual = trainer.sampling_from_embeddings(embeddings)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    plan = canvas_plan_from_region(REGION)
    assert torch.equal(actual[:, plan.source_index], embeddings["canvas_source_latents"][:, plan.source_index])
    assert trainer._last_complementary_canvas["edit_tokens"] == expected_canvas["edit_tokens"] == 6
    assert sampler.last["steps"] == 3


@pytest.mark.parametrize("keep_region_source", [False, True], ids=["source-outside-r", "source-whole-grid"])
@pytest.mark.parametrize("batch_cfg", [False, True])
def test_sampler_with_text_condition_and_anchoring_matches_the_trainer_cached_loop(batch_cfg, keep_region_source):
    """The deploy path: group isolation, the text as a condition and soft latent anchoring inside R."""

    generator = torch.Generator().manual_seed(11)
    anchoring = {"latent_anchoring_mode": "expanded_soft", "anchor_source_latents": torch.randn(1, 16, 16, generator=generator),
                 "anchor_edit_weight": torch.rand(1, 16, 16, generator=generator)}
    embeddings = {**_embeddings(), **anchoring}
    canvas = ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, isolation="group", cache_conditions=True,
                                       text_as_condition=True, keep_region_source=keep_region_source)
    trainer = _trainer(_tiny_dit(), canvas)
    with patch("src.trainer.qwen_image_edit_trainer.tqdm", side_effect=lambda iterable, **_: iterable):
        expected = trainer.sampling_from_embeddings(embeddings)
    expected_canvas = dict(trainer._last_complementary_canvas)
    trainer.scheduler = _EulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    sampler = FastCanvasSampler(trainer, FastCanvasOptions(batch_cfg=batch_cfg, cuda_graph=False))
    sampler.install()
    actual = trainer.sampling_from_embeddings(embeddings)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    record = dict(trainer._last_complementary_canvas)
    assert record.pop("fast_sampler")["steps"] == 3
    assert record.keys() == expected_canvas.keys() and record["text_cached"] and expected_canvas["text_cached"]
    assert record["source_condition_tokens"] == expected_canvas["source_condition_tokens"] == (16 if keep_region_source else 10)
    # The anchoring is live: without it R ends elsewhere.
    trainer.scheduler = _EulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    plain = {key: value for key, value in embeddings.items() if key not in anchoring}
    assert (trainer.sampling_from_embeddings(plain) - actual).abs().max() > 1e-2


def test_sampler_rejects_paths_it_does_not_cover():
    trainer = _trainer(_tiny_dit(lora=False))
    sampler = FastCanvasSampler(trainer, FastCanvasOptions(cuda_graph=False))
    assert sampler.merged_layers == 0  # the default keeps LoRA unmerged
    with pytest.raises(NotImplementedError, match="true CFG"):
        sampler.prepare({**_embeddings(), "negative_prompt": [""], "reference_cfg_embeddings": None})
    with pytest.raises(ValueError, match="whole schedule"):
        sampler.prepare({**_embeddings(), "_sigma_handoff": {}})
    trainer.config.model.complementary_canvas = ComplementaryCanvasConfig(enabled=True)
    with pytest.raises(ValueError, match="isolate_conditions"):
        sampler.prepare(_embeddings())
    with pytest.raises(ValueError, match="CUDA"):
        FastCanvasSampler(trainer, FastCanvasOptions(cuda_graph=True))


# ---- CUDA: FA2 varlen with overlapping key ranges, graph replay, bf16 batching -------------------


@pytest.mark.skipif(not CUDA, reason="needs CUDA (flash-attn)")
def test_flash_varlen_with_overlapping_key_ranges_matches_per_branch_attention():
    generator = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(40, 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(70, 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn_like(k)
    sequences = [(0, 25, 0, 50), (25, 15, 30, 40)]  # keys [0, 50) and [30, 70) overlap on [30, 50)
    flash = _Attention(sequences, torch.device("cuda"), 64 ** -0.5)
    reference = _Attention(sequences, torch.device("cpu"), 64 ** -0.5)
    expected = reference(q.float().cpu(), k.float().cpu(), v.float().cpu())
    torch.testing.assert_close(flash(q, k, v).float().cpu(), expected, rtol=0, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA (cuDNN SDPA)")
def test_cudnn_attention_matches_per_branch_attention():
    generator = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(40, 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(70, 2, 64, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn_like(k)
    sequences = [(0, 25, 0, 50), (25, 15, 30, 40)]
    expected = _Attention(sequences, torch.device("cpu"), 64 ** -0.5)(q.float().cpu(), k.float().cpu(), v.float().cpu())
    for backend in ("cudnn", "flash"):
        actual = _Attention(sequences, torch.device("cuda"), 64 ** -0.5, backend)(q, k, v)
        torch.testing.assert_close(actual.float().cpu(), expected, rtol=0, atol=2e-2)


@pytest.mark.skipif(not CUDA, reason="needs CUDA (Triton)")
def test_triton_kernels_match_their_references_in_bf16():
    from src.inference import fused_ops

    generator = torch.Generator(device="cuda").manual_seed(2)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator).to(torch.bfloat16)

    def within_one_ulp(actual, expected, fraction):
        """bf16 results that differ only where an fp32 value of a different reduction order rounds the
        other way (1 ulp of the O(1) operands; more ulps of a result after cancellation), on at most
        ``fraction`` of the elements."""

        torch.testing.assert_close(actual.float(), expected.float(), rtol=2 ** -7, atol=2 ** -6)
        assert float((actual != expected).float().mean()) <= fraction

    x, shift, scale = randn(37, 3072), randn(3072) * 0.1, randn(3072) * 0.1
    within_one_ulp(fused_ops.ln_modulate(x, shift, scale, 1e-6), fused_ops.ln_modulate_reference(x, shift, scale, 1e-6), 0.05)
    values, gate = randn(37, 3072), randn(3072)
    torch.testing.assert_close(fused_ops.gate_residual(x, values, gate), fused_ops.gate_residual_reference(x, values, gate),
                               rtol=0, atol=0)

    heads, rows = 4, 9
    qkv = randn(rows, 3 * heads * 128)
    q_weight, k_weight = 1 + 0.1 * randn(128), 1 + 0.1 * randn(128)
    angles = torch.rand(rows, 64, device="cuda", generator=generator) * 6
    q_rows = torch.randperm(rows, device="cuda").to(torch.int32)
    k_rows = (torch.randperm(rows, device="cuda") + 3).to(torch.int32)
    outputs = [torch.zeros(rows + 3, heads, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    fused_ops.qk_norm_rope_scatter(qkv, q_weight, k_weight, angles.cos(), angles.sin(), 1e-6, heads,
                                   q_rows, k_rows, *outputs)
    q, k, v = fused_ops.qk_norm_rope_reference(qkv, q_weight, k_weight, angles.cos(), angles.sin(), 1e-6, heads)
    within_one_ulp(outputs[0][q_rows.long()], q, 0.05)
    within_one_ulp(outputs[1][k_rows.long()], k, 0.05)
    assert torch.equal(outputs[2][k_rows.long()], v)


def _cuda_model_and_inputs():
    model = _tiny_dit().to("cuda", torch.bfloat16)
    x = _inputs()
    for name in ("target", "conditions", "null_conditions", "geometry", "positive", "null"):
        setattr(x, name, getattr(x, name).to("cuda", torch.bfloat16))
    x.plan = canvas_plan_from_region(REGION.cuda())
    return model, x


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_graph_replay_equals_the_eager_step():
    from src.inference.canvas_fast_sampler import _GraphedStep

    model, x = _cuda_model_and_inputs()
    fuse_qkv_(model)
    merge_lora_(model)
    forward = _fast(model, x, FastCanvasOptions())
    with torch.inference_mode():
        def run(inputs):
            return forward.step(inputs["latents"], {"t": inputs["t"]})

        example = {"latents": x.target, "t": torch.full((1,), 0.6, device="cuda", dtype=torch.bfloat16)}
        stream = torch.cuda.Stream()
        graphed = _GraphedStep(run, example, stream, True)
        del graphed  # a later capture after the first graph is gone (one sample after another)
        graphed = _GraphedStep(run, example, stream, False)
        for timestep in (0.9, 0.3):
            inputs = {"latents": torch.randn_like(x.target),
                      "t": torch.full((1,), timestep, device="cuda", dtype=torch.bfloat16)}
            replayed = [out.clone() for out in graphed(inputs)]
            for got, want in zip(replayed, run(inputs)):
                assert torch.equal(got, want)


@pytest.mark.skipif(not CUDA, reason="needs CUDA (Triton)")
def test_fused_path_and_fused_lora_match_the_model_in_bf16():
    model, x = _cuda_model_and_inputs()
    expected = _reference(model, x)
    fuse_qkv_(model)
    with torch.inference_mode():
        for fused_lora in (False, True):
            options = FastCanvasOptions(cuda_graph=False, fused_lora=fused_lora)
            forward = _fast(model, x, options)
            steps = {"t": torch.full((1,), 0.6, device="cuda")}
            table = forward.modulation_table([steps])
            actual = forward.step(x.target, steps, tuple(part[0] for part in table))
            for got, want in zip(actual, expected):
                relative = (got.float() - want.float()).norm() / want.float().norm()
                assert relative < 2e-2


@pytest.mark.skipif(not CUDA, reason="needs CUDA (Triton)")
def test_text_condition_fused_path_matches_the_model_in_bf16():
    model, x = _cuda_model_and_inputs()
    expected = _reference(model, x, shared=False, isolation="group", text=True)
    fuse_qkv_(model)
    with torch.inference_mode():
        forward = _fast(model, x, FastCanvasOptions(cuda_graph=False), shared=False, isolation="group", text=True)
        steps = {"t": torch.full((1,), 0.6, device="cuda")}
        temb, image, text = forward.modulation_table([steps])
        assert text is None
        actual = forward.step(x.target, steps, (temb[0], image[0], None))
        for got, want in zip(actual, expected):
            relative = (got.float() - want.float()).norm() / want.float().norm()
            assert relative < 2e-2


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_batched_cfg_matches_separate_branches_in_bf16():
    model, x = _cuda_model_and_inputs()
    expected = _reference(model, x)
    with torch.inference_mode():
        for batch_cfg in (False, True):
            options = FastCanvasOptions(**{**PLAIN, "fuse_qkv": False, "batch_cfg": batch_cfg})
            actual = _fast(model, x, options).step(x.target, {"t": torch.full((1,), 0.6, device="cuda")})
            for got, want in zip(actual, expected):
                relative = (got.float() - want.float()).norm() / want.float().norm()
                assert relative < 2e-2

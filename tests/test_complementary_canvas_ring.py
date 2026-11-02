"""Complementary canvas round 2: three-zone ring, condition-group isolation, annealed isolation,
text as a cacheable condition, local pixel-loss decode, MaskMix grow mode, VL source ablation."""

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import QwenImageEditPipeline

from src.data.config import ComplementaryCanvasConfig, Config
from src.loss.edit_mask_loss import MaskEditLoss
from src.models.complementary_canvas import (
    canvas_plan_from_region,
    compose_canvas_latents,
    gather_tokens,
    ring_feather_weights,
)
from src.models.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0, _BiasedTwoBlockAttention
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.inference_metadata import _checkpoint_contract, _flatten_contract_mismatches
from tests.test_complementary_canvas import IMG_SHAPES, REGION, GT_OBJECT, _EulerScheduler, _dit_kwargs, _tiny_dit

# REGION on the 4x4 grid; its 1-token ring is every other token except the far corner 15.
RING = [0, 3, 4, 8, 10, 11, 12, 13, 14]


def _ring_plan(ring_tokens=1):
    return canvas_plan_from_region(REGION, grid=(4, 4), ring_tokens=ring_tokens)


def test_ring_plan_orders_core_then_ring_and_keeps_the_rest_as_source():
    plan = _ring_plan()
    assert plan.core_index.tolist() == torch.nonzero(REGION).flatten().tolist()
    assert plan.ring_index.tolist() == RING and plan.ring_count == 9 and plan.core_count == 6
    assert plan.source_index.tolist() == [15]
    assert plan.image_token_index[0].tolist() == plan.core_index.tolist() + RING
    assert canvas_plan_from_region(REGION, grid=(4, 4), ring_tokens=2).source_index.numel() == 0
    with pytest.raises(ValueError, match="grid"):
        canvas_plan_from_region(REGION, ring_tokens=1)


def test_keep_region_source_feeds_the_whole_masked_source_grid_as_condition():
    plan = canvas_plan_from_region(REGION, keep_region_source=True)
    assert plan.source_index.tolist() == torch.nonzero(~REGION).flatten().tolist()
    assert plan.condition_source_index.tolist() == list(range(16))
    assert plan.image_token_index[1].tolist() == list(range(16))

    calls = []

    def fake_dit(**kwargs):
        calls.append(kwargs)
        return (torch.full((1, kwargs["hidden_states"].shape[1], 16), 0.5),)

    canvas = ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, isolation="group", keep_region_source=True)
    trainer = _loss_trainer(fake_dit, canvas)
    embeddings = _loss_embeddings(u=0.05)  # index 0: σ = 1.0
    trainer._compute_loss(embeddings)
    noise = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(3))
    expected_input = torch.cat([noise[:, plan.core_index], embeddings["control_latents"]], dim=1)
    torch.testing.assert_close(calls[0]["hidden_states"], expected_input)
    assert [index.tolist() for index in calls[0]["image_token_index"]] == [plan.core_index.tolist(), list(range(16))]


def test_ring_feather_weights_fall_linearly_with_latent_distance_in_the_packed_layout():
    plan = _ring_plan()
    weights = ring_feather_weights(plan, (4, 4), 1, channels=3)
    assert weights.shape == (9, 12)
    # Token 4 sits left of core token 5: its right latent column is 1 px from R, the left one 2 px.
    token4 = weights[RING.index(4)]
    torch.testing.assert_close(token4[:4], torch.tensor([1 / 3, 2 / 3, 1 / 3, 2 / 3]))
    assert torch.equal(token4[4:8], token4[:4])  # every channel shares its pixel's weight

    # Independent reference: brute-force Euclidean distance on the 8x8 latent grid, then QwenImage packing.
    core = REGION.view(4, 4).repeat_interleave(2, 0).repeat_interleave(2, 1)
    points = torch.nonzero(core).float()
    grid = torch.stack(torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij"), -1).view(-1, 1, 2)
    distance = (grid - points[None]).norm(dim=-1).min(dim=1).values.view(8, 8)
    weight_map = (1 - distance / 3).clamp(0, 1).expand(1, 3, 8, 8)
    packed = QwenImageEditPipeline._pack_latents(weight_map, 1, 3, 8, 8)[0]
    torch.testing.assert_close(weights, packed[RING])


def test_compose_canvas_latents_feathers_only_the_ring():
    plan = _ring_plan()
    torch.manual_seed(0)
    source, generated = torch.randn(1, 16, 8), torch.randn(1, 15, 8)
    weights = torch.rand(9, 8)
    merged = compose_canvas_latents(source, plan, generated, weights)
    assert torch.equal(merged[:, plan.core_index], generated[:, :6])
    torch.testing.assert_close(merged[:, plan.ring_index], source[:, RING] + weights * (generated[:, 6:] - source[:, RING]))
    assert torch.equal(merged[:, 15], source[:, 15])


def _ring_inputs(plan, seed=1):
    torch.manual_seed(seed)
    hidden = torch.randn(1, plan.target_index.numel() + plan.source_index.numel() + 9, 16)
    geometry = torch.randn(1, 3, 64, 64)
    return hidden, geometry


@pytest.mark.parametrize("isolate", [False, True])
def test_ring_group_with_the_core_timestep_equals_one_target_group(isolate):
    model = _tiny_dit(lora=True)
    plan = _ring_plan()
    hidden, geometry = _ring_inputs(plan)
    kwargs = dict(_dit_kwargs(timestep=0.6), mask_geometry=geometry, output_token_count=15,
                  image_token_index=plan.image_token_index, isolate_conditions=isolate)
    with torch.no_grad():
        single = model(hidden, **kwargs)[0]
        ring = model(hidden, **kwargs, ring_token_count=9, ring_timestep=torch.tensor([0.6]))[0]
        capped = model(hidden, **kwargs, ring_token_count=9, ring_timestep=torch.tensor([0.3]))[0]
    torch.testing.assert_close(ring, single, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(capped, single, atol=1e-4)
    with pytest.raises(ValueError, match="zero_cond_t"):
        _tiny_dit(zero_cond_t=False)(hidden, **kwargs, ring_token_count=9, ring_timestep=torch.tensor([0.3]))


def _attention_module(dim=32):
    torch.manual_seed(3)
    return Attention(
        query_dim=dim, cross_attention_dim=None, added_kv_proj_dim=dim, dim_head=16, heads=2, out_dim=dim,
        context_pre_only=False, bias=True, processor=QwenDoubleStreamAttnProcessor2_0(), qk_norm="rms_norm", eps=1e-6,
    )


def _processor_inputs(target=6, text=5, segments=(4, 3)):
    total = target + sum(segments)
    hidden, context = torch.randn(1, total, 32), torch.randn(1, text, 32)
    freqs = (torch.polar(torch.ones(total, 8), torch.randn(total, 8)), torch.polar(torch.ones(text, 8), torch.randn(text, 8)))
    return hidden, context, freqs


def _dense_with_mask(attn, hidden, context, freqs, allowed):
    return attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, attention_mask=allowed[None, None])


def test_condition_group_isolation_and_text_condition_equal_masked_dense_sdpa():
    attn = _attention_module()
    hidden, context, freqs = _processor_inputs()
    text, target, conditions = 5, 6, 7
    total = text + target + conditions
    # Group isolation: conditions see all conditions, nothing else; text and target see everything.
    allowed = torch.ones(total, total, dtype=torch.bool)
    allowed[text + target :, : text + target] = False
    grouped = attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, condition_segments=(7,))
    for actual, expected in zip(grouped, _dense_with_mask(attn, hidden, context, freqs, allowed)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    # Text as a condition: the text no longer sees the target either.
    allowed[:text, text : text + target] = False
    as_condition = attn.processor(
        attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, condition_segments=(7,), text_as_condition=True
    )
    for actual, expected in zip(as_condition, _dense_with_mask(attn, hidden, context, freqs, allowed)):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_biased_two_block_attention_matches_autograd_of_the_biased_softmax():
    torch.manual_seed(5)
    query = torch.randn(1, 7, 2, 16, requires_grad=True)
    key_a, value_a = torch.randn(1, 5, 2, 16, requires_grad=True), torch.randn(1, 5, 2, 16, requires_grad=True)
    key_b, value_b = torch.randn(1, 9, 2, 16, requires_grad=True), torch.randn(1, 9, 2, 16, requires_grad=True)
    inputs = (query, key_a, value_a, key_b, value_b)
    grad = torch.randn(1, 7, 2, 16)
    for bias in (0.0, 2.5, 30.0):
        merged = _BiasedTwoBlockAttention.apply(*inputs, bias)
        grads = torch.autograd.grad(merged, inputs, grad)
        scores = torch.einsum("bqhd,bkhd->bhqk", query, torch.cat([key_a, key_b], 1)) / 4.0
        scores = scores - torch.cat([torch.zeros(5), torch.full((9,), bias)])
        expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), torch.cat([value_a, value_b], 1))
        expected_grads = torch.autograd.grad(expected, inputs, grad)
        torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-5)
        for actual, reference in zip(grads, expected_grads):
            torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)


def test_annealed_isolation_spans_joint_attention_and_hard_isolation():
    attn = _attention_module()
    hidden, context, freqs = _processor_inputs()
    joint = attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs)
    hard = attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, condition_segments=(4, 3))
    for bias, reference in ((0.0, joint), (1e4, hard)):
        soft = attn.processor(
            attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, condition_segments=(4, 3), isolation_bias=bias
        )
        for actual, expected in zip(soft, reference):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("isolation,text_condition", [("group", False), ("image", True), ("group", True)])
def test_condition_cache_with_ring_groups_and_text_equals_the_uncached_forward(isolation, text_condition):
    model = _tiny_dit(lora=True)
    plan = _ring_plan()
    torch.manual_seed(2)
    target = torch.randn(1, 15, 16)
    conditions = torch.randn(1, 1 + 9, 16)
    geometry = model.prepare_mask_geometry_tokens(torch.randn(1, 3, 64, 64), IMG_SHAPES)
    common = dict(mask_geometry_tokens=geometry, image_token_index=plan.image_token_index, output_token_count=15,
                  ring_token_count=9, isolation=isolation, text_as_condition=text_condition)
    prompt = torch.randn(1, 3, 16)
    text = {"encoder_hidden_states": prompt, "txt_seq_lens": [3]} if text_condition else {}
    cache = model.build_condition_cache(
        conditions, IMG_SHAPES, plan.image_token_index, isolation=isolation, target_groups=2, **text
    )
    assert cache.image_lengths == (1, 9) and cache.segment_lengths == ((10,) if isolation == "group" else (1, 9))
    assert (cache.text_layers is not None) == text_condition
    with torch.no_grad():
        for timestep, ring_timestep in ((0.9, 0.3), (0.2, 0.2)):
            kwargs = {**_dit_kwargs(timestep=timestep), **common, "encoder_hidden_states": prompt,
                      "ring_timestep": torch.tensor([ring_timestep])}
            uncached = model(torch.cat([target, conditions], dim=1), **kwargs, isolate_conditions=True)[0]
            cached = model(target, **kwargs, condition_cache=cache)[0]
            torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-5)
    if isolation == "group" or text_condition:
        with pytest.raises(ValueError, match="per-image"):
            cache.select([0])


def _loss_trainer(dit, canvas, region_perceptual=False, **loss_fields):
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.accelerator = NS(device=torch.device("cpu"))
    trainer.weight_dtype = torch.float32
    trainer.vae_scale_factor = 8
    trainer.global_step = 0
    trainer.scheduler = NS(config=NS(num_train_timesteps=10))
    trainer._get_cached_scheduler_training_tensors = Mock(
        return_value=(torch.linspace(1000, 100, 10), torch.linspace(1.0, 0.1, 10))
    )
    loss = NS(region_perceptual=region_perceptual, region_sigma_power=2.0, region_max_sigma=0.5,
              latent_mse_weight=1.0, fg_lpips_weight=0.5, bg_l1_weight=0.5, ref_dino_weight=0.0,
              mask_threshold=0.5, lpips_bbox_pad_ratio=0.1, dino_bbox_pad_ratio=0.1, detail_lpips_weight=0.0,
              detail_patch_size=256, interaction_radius_px=0, **loss_fields)
    trainer.config = NS(model=NS(complementary_canvas=canvas), loss=loss)
    trainer.criterion = MaskEditLoss(3.0, 1.0, mode="region_balanced")
    trainer.dit = dit
    return trainer


def _loss_embeddings(u=0.4):
    torch.manual_seed(11)
    return {
        "image_latents": torch.randn(1, 16, 16),
        "control_latents": torch.randn(1, 16 + 9, 16),
        "prompt_embeds": torch.randn(1, 3, 16),
        "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
        "img_shapes": IMG_SHAPES,
        "mask": GT_OBJECT[None].clone(),
        "mask_geometry": torch.randn(1, 3, 64, 64),
        "canvas_region": REGION[None].clone(),
        "height": 64,
        "width": 64,
        "_analysis_noise_seed": 3,
        "_analysis_timestep_u": u,
    }


def test_training_ring_input_timestep_and_loss():
    calls = []

    def fake_dit(**kwargs):
        calls.append(kwargs)
        return (torch.full((1, kwargs["hidden_states"].shape[1], 16), 0.5),)

    canvas = ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, isolation="group", ring_tokens=1,
                                       ring_sigma_max=0.3, ring_loss_weight=0.25)
    trainer = _loss_trainer(fake_dit, canvas)
    embeddings = _loss_embeddings(u=0.05)  # index 0: σ = 1.0
    loss = trainer._compute_loss(embeddings)
    plan, call = _ring_plan(), calls[0]
    noise = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(3))
    x0 = embeddings["image_latents"]
    expected_input = torch.cat([
        noise[:, plan.core_index],  # σ = 1
        0.7 * x0[:, RING] + 0.3 * noise[:, RING],  # σ_ring = min(σ, 0.3)
        embeddings["control_latents"][:, [15]],
        embeddings["control_latents"][:, 16:],
    ], dim=1)
    torch.testing.assert_close(call["hidden_states"], expected_input)
    assert call["output_token_count"] == 15 and call["ring_token_count"] == 9 and call["isolation"] == "group"
    torch.testing.assert_close(call["ring_timestep"], torch.tensor([0.3]))
    assert [index.tolist() for index in call["image_token_index"]] == [plan.target_index.tolist(), [15]]
    assert "isolation_bias" not in call

    target = noise - x0
    core_loss = MaskEditLoss(3.0, 1.0, mode="region_balanced")(
        GT_OBJECT[None][:, plan.core_index], torch.full((1, 6, 16), 0.5), target[:, plan.core_index], torch.ones(1, 1, 1)
    )
    ring_mse = ((0.5 - target[:, RING]) ** 2).mean()
    torch.testing.assert_close(loss, core_loss + 0.25 * ring_mse)
    torch.testing.assert_close(trainer._step_loss_metrics["train/ring_mse"], ring_mse)

    # Annealing: a logit bias ramps up over anneal_steps, then isolation is hard again.
    trainer.config.model.complementary_canvas = ComplementaryCanvasConfig(
        enabled=True, isolate_conditions=True, ring_tokens=1, anneal_steps=100, anneal_max_bias=10.0
    )
    for step, bias in ((0, 0.0), (25, 2.5), (100, None)):
        trainer.global_step = step
        trainer._compute_loss(_loss_embeddings())
        assert calls[-1].get("isolation_bias") == bias


@pytest.mark.parametrize("canvas_fields", [
    dict(isolate_conditions=False, ring_tokens=1),
    dict(isolate_conditions=True, isolation="group", ring_tokens=1, anneal_steps=10),
    dict(isolate_conditions=True, isolation="group", ring_tokens=1, text_as_condition=True),
])
def test_tiny_end_to_end_ring_forward_backward(canvas_fields):
    dit = _tiny_dit(lora=False).train()
    trainer = _loss_trainer(dit, ComplementaryCanvasConfig(enabled=True, **canvas_fields))
    trainer.global_step = 3
    loss = trainer._compute_loss(_loss_embeddings())
    assert torch.isfinite(loss)
    loss.backward()
    assert dit.img_in.weight.grad.abs().sum() > 0
    if canvas_fields.get("text_as_condition"):
        assert dit.transformer_blocks[0].txt_mlp.net[2].weight.grad.abs().sum() > 0


def test_pixel_losses_decode_the_feathered_ring_on_a_local_crop():
    calls = []

    def fake_dit(**kwargs):
        calls.append(kwargs)
        return (torch.ones(1, kwargs["hidden_states"].shape[1], 16, requires_grad=True),)

    # A 8x8-token (128 px) grid with R in the top-left corner and a small GT object inside it.
    shapes = [[(1, 8, 8), (1, 8, 8), (1, 3, 3)]]
    region = torch.zeros(64, dtype=torch.bool)
    region[[0, 1, 8, 9]] = True
    pixel_mask = torch.zeros(1, 128, 128)
    pixel_mask[:, 8:20, 8:20] = 1.0
    canvas = ComplementaryCanvasConfig(enabled=True, ring_tokens=1, local_decode_margin_px=16)
    trainer = _loss_trainer(fake_dit, canvas, region_perceptual=True)
    trainer.decode_vae_latent_for_loss = Mock(side_effect=lambda latents, h, w: torch.zeros(1, 3, h, w, requires_grad=True))
    trainer._loaded_model_decode_context = Mock(return_value=nullcontext())
    trainer.region_perceptual_criterion = Mock(return_value={
        "total": torch.tensor(0.0, requires_grad=True), "fg_lpips": torch.tensor(0.0),
        "bg_l1": torch.tensor(0.0), "ref_dino": torch.tensor(0.0),
    })
    torch.manual_seed(11)
    embeddings = {
        "image_latents": torch.randn(1, 64, 16), "control_latents": torch.randn(1, 64 + 9, 16),
        "prompt_embeds": torch.randn(1, 3, 16), "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
        "img_shapes": shapes, "mask": torch.zeros(1, 64), "mask_geometry": torch.randn(1, 3, 128, 128),
        "canvas_region": region[None], "height": 128, "width": 128, "pixel_mask": pixel_mask,
        "target_image_for_loss": torch.rand(1, 3, 128, 128), "_analysis_noise_seed": 3, "_analysis_timestep_u": 0.95,
    }
    trainer._compute_loss(embeddings)

    plan = canvas_plan_from_region(region, grid=(8, 8), ring_tokens=1)
    assert plan.ring_index.tolist() == [2, 10, 16, 17, 18]
    # R + ring span tokens rows/cols 0..2 (48 px); +16 px margin -> 64 px; the GT box pads stay inside.
    latents, height, width = trainer.decode_vae_latent_for_loss.call_args.args
    assert (height, width) == (64, 64)
    crop_index = (torch.arange(4)[:, None] * 8 + torch.arange(4)[None, :]).flatten()
    sigma = torch.tensor(0.1)
    noise = torch.randn(1, 64, 16, generator=torch.Generator().manual_seed(3))
    x0 = embeddings["image_latents"]
    generated = torch.cat([
        ((1 - sigma) * x0 + sigma * noise)[:, plan.core_index],
        ((1 - sigma) * x0 + sigma * noise)[:, plan.ring_index],  # σ = 0.1 is below the 0.3 cap
    ], dim=1) - sigma
    weights = ring_feather_weights(plan, (8, 8), 1, 4)
    expected = compose_canvas_latents(x0, plan, generated, weights)
    torch.testing.assert_close(latents, gather_tokens(expected, crop_index))
    pred, target, mask = trainer.region_perceptual_criterion.call_args.args
    assert pred.shape[-2:] == target.shape[-2:] == mask.shape[-2:] == (64, 64)
    torch.testing.assert_close(target, embeddings["target_image_for_loss"][..., :64, :64])
    background = 1 - pixel_mask
    expected_weight = 0.5 * float(background[:, :64, :64].sum() / background.sum())
    assert trainer.region_perceptual_criterion.call_args.kwargs["bg_weight"] == pytest.approx(expected_weight)


class _AnchoringEulerScheduler(_EulerScheduler):
    def scale_noise(self, sample, timestep, noise):
        sigma = float(timestep[0]) / 1000.0
        return sigma * noise + (1.0 - sigma) * sample


def _ring_sample(canvas, dit, velocity=None, edit_weight=None):
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.config = NS(model=NS(complementary_canvas=canvas))
    trainer.accelerator = None
    trainer.dit = dit
    trainer.weight_dtype = torch.float32
    trainer.vae_scale_factor = 8
    trainer.scheduler = _AnchoringEulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    trainer.prepare_initial_latents = Mock(return_value=torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(9)))
    trainer.prepare_predict_timesteps = Mock(return_value=(torch.tensor([1000.0, 700.0, 400.0]), 3))
    torch.manual_seed(4)
    hole = torch.zeros(1, 64, 64)
    hole[:, 16:48, 16:40] = 1.0
    branch = {
        "prompt_embeds": torch.randn(1, 3, 16), "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
        "mask_geometry": torch.randn(1, 3, 64, 64), "height": 64, "width": 64, "height_control": 64, "width_control": 64,
    }
    controls = torch.randn(1, 16 + 9, 16)
    null = {**branch, "prompt_embeds": torch.randn(1, 2, 16), "prompt_embeds_mask": torch.ones(1, 2, dtype=torch.long),
            "control_latents": controls[:, :16], "n_controls": 0}
    embeddings = {
        **branch, "control_latents": controls, "n_controls": 1, "control_1": torch.zeros(1),
        "height_control_1": 48, "width_control_1": 48, "num_inference_steps": 3, "true_cfg_scale": 2.5,
        "guidance": 1.0, "reference_cfg_embeddings": null, "reference_cfg_scale": 2.5,
        "reference_cfg_mask_aware": True, "reference_cfg_mask": hole,
        "canvas_region": REGION[None].clone(), "canvas_source_latents": torch.randn(1, 16, 16),
    }
    if edit_weight is not None:
        embeddings.update(anchor_source_latents=embeddings["canvas_source_latents"], anchor_edit_weight=edit_weight,
                          latent_anchoring_mode="expanded_soft")
    calls = []
    dit.register_forward_pre_hook(lambda _module, _args, kwargs: calls.append(kwargs), with_kwargs=True)
    if velocity is not None:  # replace the prediction by a constant velocity field
        dit.register_forward_hook(lambda _module, _args, _kwargs, _output: (velocity.clone(),), with_kwargs=True)
    with patch("src.trainer.qwen_image_edit_trainer.tqdm", side_effect=lambda iterable, **_: iterable):
        latents = trainer.sampling_from_embeddings(embeddings)
    return latents, embeddings, calls, trainer


def test_sampling_ring_waits_below_the_cap_then_is_feathered_into_the_source():
    velocity = torch.randn(1, 15, 16, generator=torch.Generator().manual_seed(1))
    canvas = ComplementaryCanvasConfig(enabled=True, ring_tokens=1, ring_sigma_max=0.5)
    latents, embeddings, calls, trainer = _ring_sample(canvas, _tiny_dit(), velocity=velocity)
    plan = _ring_plan()
    noise = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(9))
    source = embeddings["canvas_source_latents"]
    ring_start = 0.5 * source[:, RING] + 0.5 * noise[:, RING]
    # σ_ring = 0.5, 0.5, 0.4, 0.0: no motion in the first step, then Euler on the capped schedule.
    ring_end = ring_start + (0.0 - 0.5) * velocity[:, 6:]
    core_end = noise[:, plan.core_index] + (0.0 - 1.0) * velocity[:, :6]
    weights = ring_feather_weights(plan, (4, 4), 1, 4)
    torch.testing.assert_close(latents, compose_canvas_latents(source, plan, torch.cat([core_end, ring_end], 1), weights))
    assert [call["ring_timestep"].item() for call in calls] == pytest.approx([0.5, 0.5, 0.5, 0.5, 0.4, 0.4])
    assert trainer._last_complementary_canvas["ring_tokens"] == 9


def test_sampling_anchors_r_latent_pixels_outside_m_free_to_the_source_trajectory():
    velocity = torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(1))
    weight = torch.rand(1, 16, 16, generator=torch.Generator().manual_seed(3))
    canvas = ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, isolation="group", cache_conditions=True)
    latents, embeddings, _, _ = _ring_sample(canvas, _tiny_dit(), velocity=velocity, edit_weight=weight)
    plan = _ring_plan(0)
    noise = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(9))
    source = embeddings["canvas_source_latents"]
    core, w = noise[:, plan.core_index], weight[:, plan.core_index]
    for sigma, sigma_next in ((1.0, 0.7), (0.7, 0.4), (0.4, 0.0)):
        core = core + (sigma_next - sigma) * velocity
        anchor = sigma_next * noise + (1 - sigma_next) * source
        core = torch.lerp(anchor[:, plan.core_index], core, w)
    torch.testing.assert_close(latents, compose_canvas_latents(source, plan, core))
    unanchored, _, _, _ = _ring_sample(canvas, _tiny_dit(), velocity=velocity)
    full, _, _, _ = _ring_sample(canvas, _tiny_dit(), velocity=velocity, edit_weight=torch.ones(1, 16, 16))
    torch.testing.assert_close(full, unanchored)
    with pytest.raises(ValueError, match="does not combine with a canvas ring"):
        _ring_sample(ComplementaryCanvasConfig(enabled=True, ring_tokens=1), _tiny_dit(), velocity=velocity,
                     edit_weight=weight)


@pytest.mark.parametrize("text_condition", [False, True])
def test_sampling_group_isolation_ring_cached_equals_uncached(text_condition):
    fields = dict(enabled=True, isolate_conditions=True, isolation="group", ring_tokens=1, text_as_condition=text_condition)
    uncached, embeddings, calls, _ = _ring_sample(ComplementaryCanvasConfig(**fields), _tiny_dit(lora=True))
    assert [call["hidden_states"].shape[1] for call in calls] == [15 + 1 + 9, 15 + 1] * 3
    cached, _, cached_calls, trainer = _ring_sample(
        ComplementaryCanvasConfig(**fields, cache_conditions=True), _tiny_dit(lora=True)
    )
    # The null branch builds its own source-only condition (and text) pass instead of reusing the positive one.
    caches = [call["condition_cache"] for call in cached_calls]
    assert [cache.image_lengths for cache in caches] == [(1, 9), (1,)] * 3
    assert all(cache.segment_lengths == (sum(cache.image_lengths),) for cache in caches)
    assert all((cache.text_layers is not None) == text_condition for cache in caches)
    assert trainer._last_complementary_canvas["text_cached"] is text_condition
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-5)
    plan = _ring_plan()
    assert torch.equal(cached[:, plan.source_index], embeddings["canvas_source_latents"][:, plan.source_index])


@pytest.mark.parametrize("text_condition", [False, True])
def test_sampling_with_the_region_source_cached_equals_uncached(text_condition):
    fields = dict(enabled=True, isolate_conditions=True, isolation="group", text_as_condition=text_condition,
                  keep_region_source=True)
    uncached, embeddings, calls, _ = _ring_sample(ComplementaryCanvasConfig(**fields), _tiny_dit(lora=True))
    assert [call["hidden_states"].shape[1] for call in calls] == [6 + 16 + 9, 6 + 16] * 3
    cached, _, cached_calls, trainer = _ring_sample(
        ComplementaryCanvasConfig(**fields, cache_conditions=True), _tiny_dit(lora=True)
    )
    assert [call["condition_cache"].image_lengths for call in cached_calls] == [(16, 9), (16,)] * 3
    assert trainer._last_complementary_canvas["source_condition_tokens"] == 16
    torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-5)
    # The masked source inside R reaches the generated tokens.
    outside_r, _, _, _ = _ring_sample(ComplementaryCanvasConfig(**{**fields, "keep_region_source": False}),
                                      _tiny_dit(lora=True))
    plan = _ring_plan(0)
    assert (outside_r[:, plan.core_index] - uncached[:, plan.core_index]).abs().max() > 1e-3
    assert torch.equal(cached[:, plan.source_index], embeddings["canvas_source_latents"][:, plan.source_index])


def test_round2_canvas_config_contracts():
    insert = {"trainer": "QwenImageEditInsertAnything", "predict": {"devices": {}}}

    def config(**fields):
        return Config.model_validate({**insert, "model": {"zero_cond_t": True, "complementary_canvas": {"enabled": True, **fields}}})

    with pytest.raises(ValueError, match="require enabled"):
        ComplementaryCanvasConfig(ring_tokens=1)
    for fields, message in (({"isolation": "group"}, "isolation=group requires"),
                            ({"text_as_condition": True}, "text_as_condition requires"),
                            ({"anneal_steps": 10}, "anneal_steps requires")):
        with pytest.raises(ValueError, match=message):
            ComplementaryCanvasConfig(enabled=True, **fields)
    with pytest.raises(ValueError, match="zero_cond_t"):
        Config.model_validate({**insert, "model": {"complementary_canvas": {"enabled": True, "ring_tokens": 1}}})

    def mismatches(checkpoint, inference):
        return {
            item["field"]
            for item in _flatten_contract_mismatches(
                _checkpoint_contract(checkpoint, "train_enabled"), _checkpoint_contract(inference, "predict_enabled")
            )
        }

    b2 = config(isolate_conditions=True)
    b2_group_ring = config(isolate_conditions=True, isolation="group", ring_tokens=2, anneal_steps=500,
                           local_decode_margin_px=64, ring_loss_weight=0.5)
    assert mismatches(b2, b2_group_ring) == {
        "model.complementary_canvas.isolation", "model.complementary_canvas.ring_tokens",
        "model.complementary_canvas.ring_sigma_max",
    }
    # Training-only knobs and the exact cache leave the contract alone.
    assert mismatches(b2_group_ring, config(isolate_conditions=True, isolation="group", ring_tokens=2, cache_conditions=True)) == set()
    text = config(isolate_conditions=True, text_as_condition=True)
    assert mismatches(b2, text) == {"model.complementary_canvas.text_as_condition"}
    assert mismatches(b2, config(isolate_conditions=True, keep_region_source=True)) == {
        "model.complementary_canvas.keep_region_source"
    }
    with pytest.raises(ValueError, match="require enabled"):
        ComplementaryCanvasConfig(keep_region_source=True)
    no_source = config()
    no_source.model.vl_masked_source = False
    assert mismatches(config(), no_source) == {"model.vl_masked_source"}


def test_interaction_context_grow_mode_dilates_the_sampled_shape_instead_of_tracing_the_gt():
    from src.utils.hybrid_masking import add_interaction_context

    gt = torch.zeros(1, 64, 64)
    gt[0, 20:40, 24:30] = 1.0  # a thin object poking out of the sampled box
    box = torch.zeros(1, 64, 64)
    box[0, 26:34, 18:36] = 1.0
    union = add_interaction_context(box, gt, 4)
    grown = add_interaction_context(box, gt, 4, mode="grow")
    import cv2

    band = cv2.dilate(gt[0].numpy().astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    assert bool((grown[0].numpy()[band > 0] > 0).all()) and bool((grown >= box).all())
    # The grown hole is a Euclidean dilation of the box: its distance to the box is bounded by one radius.
    distance = cv2.distanceTransform(1 - box[0].numpy().astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    reach = distance[band > 0].max()
    assert torch.equal(grown[0], torch.from_numpy((distance <= reach).astype(np.float32)))
    assert not torch.equal(union, grown)
    with pytest.raises(ValueError, match="interaction context mode"):
        add_interaction_context(box, gt, 4, mode="trace")


def test_vl_prompt_can_drop_the_masked_source_but_keeps_it_in_the_dit():
    from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

    trainer = object.__new__(QwenImageEditPlusTrainer)
    trainer.config = NS(loss=NS(mask_loss=False))
    trainer.num_channels_latents, trainer.weight_dtype, trainer.vae_scale_factor = 1, torch.float32, 8
    trainer.preprocess_image_for_text_encoder = lambda image, **_kwargs: image
    trainer.preprocess_image_for_vae_encoder = lambda image, **_kwargs: image.unsqueeze(2)
    trainer.encode_image_latents = Mock(side_effect=lambda image, *_args: torch.zeros(1, 4, 4))
    seen = {}

    def encode_prompt(prompt, image):
        seen["images"] = image
        return torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.long)

    trainer.encode_prompt = encode_prompt
    source, reference = torch.full((1, 3, 32, 32), 1.0), torch.full((1, 3, 32, 32), 2.0)
    batch = {"control": source, "control_1": reference, "n_controls": 1, "prompt": [""],
             "vae_only_control_keys": ["control"]}
    embeddings = trainer.prepare_embeddings(batch, stage="predict")
    assert len(seen["images"]) == 1 and torch.equal(seen["images"][0], reference[0])
    assert embeddings["control_latents"].shape[1] == 8  # both images still reach the DiT


def test_crop_guard_support_covers_the_ring():
    from src.utils import adaptive_crop_anchoring as module
    from PIL import Image

    def config(ring):
        return Config.model_validate({
            "trainer": "QwenImageEditInsertAnything", "predict": {"devices": {}},
            "model": {"zero_cond_t": True, "complementary_canvas": {"enabled": True, "ring_tokens": ring}},
            "data": {"init_args": {"adaptive_crop": {"predict_enabled": True}, "processor": {
                "class_path": "src.data.preprocess.ImageProcessor",
                "init_args": {"process_type": "resize", "target_size": [64, 64]}}}},
        })

    mask = Image.new("L", (256, 256))
    mask.paste(255, (100, 100, 130, 140))
    seen = []
    original = module.ensure_crop_plan_for_latent_anchoring

    def capture(plan, mask_image, *, anchoring_config, **kwargs):
        seen.append(int(anchoring_config.interaction_dilation_px))
        return original(plan, mask_image, anchoring_config=anchoring_config, **kwargs)

    with patch.object(module, "ensure_crop_plan_for_latent_anchoring", side_effect=capture):
        for ring in (0, 2):
            cfg = config(ring)
            module.prepare_adaptive_crop_for_inference(
                image_size=(256, 256), mask=mask, adaptive_crop_config=cfg.data.init_args.adaptive_crop, config=cfg,
            )
    assert seen == [32, 32 + 46]  # ceil(16 px * 2 tokens * sqrt(2))


def test_text_encoder_streaming_keeps_one_host_copy():
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.text_encoder = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
    weight = trainer.text_encoder[0].weight.detach().clone()
    for _ in range(2):
        trainer._stream_text_encoder(torch.device("cpu"))
        trainer._stream_text_encoder(None)
    assert torch.equal(trainer.text_encoder[0].weight, weight)
    assert len(trainer._text_encoder_host) == 4 + 3  # Linear and BatchNorm weight/bias, BatchNorm's 3 buffers


def test_every_rank_logs_the_ring_metric_even_without_ring_tokens():
    """DDP reduces the per-step loss metrics of all ranks together, so their keys must not depend on the sample."""

    def fake_dit(**kwargs):
        return (torch.full((1, kwargs["hidden_states"].shape[1], 16), 0.5),)

    trainer = _loss_trainer(fake_dit, ComplementaryCanvasConfig(enabled=True, ring_tokens=1))
    embeddings = _loss_embeddings()
    embeddings["canvas_region"] = torch.ones(1, 16, dtype=torch.bool)  # R covers the grid: no ring tokens
    embeddings["mask"] = torch.ones(1, 16)
    trainer._compute_loss(embeddings)
    assert trainer._step_loss_metrics["train/ring_mse"] == 0
    trainer._compute_loss(_loss_embeddings())
    assert trainer._step_loss_metrics["train/ring_mse"] > 0

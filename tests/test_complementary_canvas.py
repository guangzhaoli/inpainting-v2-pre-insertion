"""Complementary canvas: per-token RoPE, gather/scatter, isolation, condition cache, train/sample paths."""

from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
import yaml
from diffusers.models.attention_processor import Attention
from pydantic import ValidationError

from src.data.config import ComplementaryCanvasConfig, Config
from src.loss.edit_mask_loss import MaskEditLoss
from src.models.complementary_canvas import (
    canvas_plan_from_region,
    gather_tokens,
    scatter_tokens,
    token_region_from_mask,
)
from src.models.mask_geometry import MaskGeometryTokenizer
from src.models.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenEmbedRope,
    QwenImageTransformer2DModel,
)
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from src.utils.inference_metadata import (
    _checkpoint_contract,
    _flatten_contract_mismatches,
    validate_resume_mask_control_compatibility,
)

# 4x4 target/source token grids (64 px, 16 px per token) and a 3x3 reference.
IMG_SHAPES = [[(1, 4, 4), (1, 4, 4), (1, 3, 3)]]
REGION = torch.tensor([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool)
GT_OBJECT = torch.tensor([0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)


def _plan():
    return canvas_plan_from_region(REGION)


def _tiny_dit(zero_cond_t=True, lora=False):
    torch.manual_seed(7)
    model = QwenImageTransformer2DModel(
        num_layers=2, attention_head_dim=16, num_attention_heads=2, in_channels=16, out_channels=4,
        joint_attention_dim=16, axes_dims_rope=(4, 6, 6), zero_cond_t=zero_cond_t,
    )
    model.mask_geometry_tokenizer = MaskGeometryTokenizer(4, 16)
    torch.nn.init.normal_(model.mask_geometry_tokenizer.conv_out.weight, std=0.2)
    if lora:
        from peft import LoraConfig

        model.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=["to_q", "to_k", "to_v", "to_out.0", "img_mlp.net.2"]))
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, std=0.1)
    return model.eval()


def _dit_kwargs(batch=1, timestep=0.6):
    return dict(
        encoder_hidden_states=torch.randn(batch, 3, 16),
        encoder_hidden_states_mask=torch.ones(batch, 3),
        timestep=torch.full((batch,), timestep),
        img_shapes=IMG_SHAPES * batch,
        txt_seq_lens=[3] * batch,
        return_dict=False,
    )


def test_per_token_rope_with_all_indices_reproduces_existing_freqs():
    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    shapes = [[(1, 6, 8), (1, 6, 8), (1, 5, 7)]]
    full_img, full_txt = rope(shapes, [11], torch.device("cpu"))
    every = (torch.arange(48), torch.arange(48), torch.arange(35))
    img, txt = rope(shapes, [11], torch.device("cpu"), token_index=every)
    assert torch.equal(img, full_img) and torch.equal(txt, full_txt)

    region = torch.zeros(48, dtype=torch.bool)
    region[[3, 10, 11, 40]] = True
    plan = canvas_plan_from_region(region)
    img, txt = rope(shapes, [11], torch.device("cpu"), token_index=plan.image_token_index)
    expected = torch.cat([full_img[:48][plan.target_index], full_img[48:96][plan.source_index], full_img[96:]])
    assert torch.equal(img, expected) and torch.equal(txt, full_txt)


def test_all_index_canvas_forward_equals_dense_forward():
    model = _tiny_dit()
    torch.manual_seed(1)
    hidden = torch.randn(1, 16 + 16 + 9, 16)
    kwargs = dict(_dit_kwargs(), mask_geometry=torch.randn(1, 3, 64, 64), output_token_count=16)
    dense = model(hidden, **kwargs)[0]
    indexed = model(hidden, **kwargs, image_token_index=(torch.arange(16), torch.arange(16)))[0]
    torch.testing.assert_close(indexed, dense, rtol=0, atol=0)


def test_token_region_and_gather_scatter_round_trip():
    mask = torch.zeros(1, 32, 48)
    mask[0, 17, 3] = 1.0  # one pixel is enough to put its 16 px token in R
    region = token_region_from_mask(mask, 16)
    assert region.shape == (1, 6) and region[0].tolist() == [False, False, False, True, False, False]

    plan = _plan()
    assert torch.equal(torch.sort(torch.cat([plan.target_index, plan.source_index])).values, torch.arange(16))
    tokens, base = torch.randn(2, 16, 5), torch.randn(2, 16, 5)
    values = gather_tokens(tokens, plan.target_index).requires_grad_()
    merged = scatter_tokens(base, plan.target_index, values)
    torch.testing.assert_close(gather_tokens(merged, plan.target_index), values)
    assert torch.equal(gather_tokens(merged, plan.source_index), gather_tokens(base, plan.source_index))
    assert torch.equal(scatter_tokens(tokens, plan.target_index, gather_tokens(tokens, plan.target_index)), tokens)
    merged.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))
    with pytest.raises(ValueError, match="empty"):
        canvas_plan_from_region(torch.zeros(16, dtype=torch.bool))
    with pytest.raises(ValueError, match="shared edit region"):
        canvas_plan_from_region(torch.stack([REGION, ~REGION]))


@pytest.mark.parametrize("segments", [(4, 3), (0, 7)])
def test_split_attention_isolation_equals_masked_dense_sdpa(segments):
    torch.manual_seed(3)
    attn = Attention(
        query_dim=32, cross_attention_dim=None, added_kv_proj_dim=32, dim_head=16, heads=2, out_dim=32,
        context_pre_only=False, bias=True, processor=QwenDoubleStreamAttnProcessor2_0(), qk_norm="rms_norm", eps=1e-6,
    )
    target, text = 6, 5
    total = target + sum(segments)
    hidden, context = torch.randn(1, total, 32), torch.randn(1, text, 32)
    freqs = (torch.polar(torch.ones(total, 8), torch.randn(total, 8)), torch.polar(torch.ones(text, 8), torch.randn(text, 8)))

    isolated = attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, condition_segments=segments)

    allowed = torch.ones(text + total, text + total, dtype=torch.bool)
    start = text + target
    for length in segments:
        allowed[start : start + length] = False
        allowed[start : start + length, start : start + length] = True
        start += length
    dense = attn.processor(attn, hidden, encoder_hidden_states=context, image_rotary_emb=freqs, attention_mask=allowed[None, None])
    for actual, expected in zip(isolated, dense):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_condition_cache_equals_uncached_isolated_forward():
    model = _tiny_dit(lora=True)
    plan = _plan()
    torch.manual_seed(2)
    target = torch.randn(1, 6, 16)
    conditions = torch.randn(1, 10 + 9, 16)
    geometry = model.prepare_mask_geometry_tokens(torch.randn(1, 3, 64, 64), IMG_SHAPES)
    common = dict(mask_geometry_tokens=geometry, image_token_index=plan.image_token_index, output_token_count=6)
    cache = model.build_condition_cache(conditions, IMG_SHAPES, plan.image_token_index)
    assert cache.segment_lengths == (10, 9)
    assert cache.nbytes == 2 * 2 * 19 * 32 * 4  # blocks x (K, V) x tokens x (heads * head_dim) x fp32
    with torch.no_grad():
        for timestep in (0.9, 0.3):
            kwargs = {**_dit_kwargs(timestep=timestep), **common}
            torch.manual_seed(5)
            kwargs["encoder_hidden_states"] = torch.randn(1, 3, 16)
            uncached = model(torch.cat([target, conditions], dim=1), **kwargs, isolate_conditions=True)[0]
            cached = model(target, **kwargs, condition_cache=cache)[0]
            torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-5)

        # The null-reference branch reuses only the source image's cache.
        null_shapes = [IMG_SHAPES[0][:2]]
        null_kwargs = {**_dit_kwargs(), **common, "img_shapes": null_shapes}
        uncached = model(torch.cat([target, conditions[:, :10]], dim=1), **null_kwargs, isolate_conditions=True)[0]
        cached = model(target, **null_kwargs, condition_cache=cache.select([0]))[0]
        torch.testing.assert_close(cached, uncached, rtol=1e-5, atol=1e-5)
        # Isolation changes what the conditions (and so the target) see: B1 and B2 differ.
        kwargs = {**_dit_kwargs(), **common}
        joint = model(torch.cat([target, conditions], dim=1), **kwargs)[0]
        isolated = model(torch.cat([target, conditions], dim=1), **kwargs, isolate_conditions=True)[0]
        assert not torch.allclose(joint, isolated)
    with pytest.raises(ValueError, match="zero_cond_t"):
        _tiny_dit(zero_cond_t=False).build_condition_cache(conditions, IMG_SHAPES, plan.image_token_index)


def _loss_trainer(dit, isolate=False, region_perceptual=False):
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.accelerator = NS(device=torch.device("cpu"))
    trainer.weight_dtype = torch.float32
    trainer.scheduler = NS(config=NS(num_train_timesteps=10))
    trainer._get_cached_scheduler_training_tensors = Mock(
        return_value=(torch.linspace(1000, 100, 10), torch.linspace(1.0, 0.1, 10))
    )
    canvas = ComplementaryCanvasConfig(enabled=True, isolate_conditions=isolate)
    loss = NS(region_perceptual=region_perceptual, region_sigma_power=2.0, region_max_sigma=0.5,
              latent_mse_weight=1.0, fg_lpips_weight=0.5, bg_l1_weight=0.5, ref_dino_weight=0.0)
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


def test_training_input_loss_and_pixel_scatter_cover_only_the_edit_region():
    calls = []

    def fake_dit(**kwargs):
        calls.append(kwargs)
        return (torch.ones(1, kwargs["hidden_states"].shape[1], 16, requires_grad=True),)

    trainer = _loss_trainer(fake_dit, region_perceptual=True)
    trainer.decode_vae_latent_for_loss = Mock(return_value=torch.zeros(1, 3, 64, 64, requires_grad=True))
    trainer._loaded_model_decode_context = Mock(return_value=nullcontext())
    trainer.region_perceptual_criterion = Mock(return_value={
        "total": torch.tensor(0.0, requires_grad=True), "fg_lpips": torch.tensor(0.0),
        "bg_l1": torch.tensor(0.0), "ref_dino": torch.tensor(0.0),
    })
    embeddings = _loss_embeddings(u=0.95)
    embeddings["pixel_mask"] = torch.ones(1, 64, 64)
    embeddings["target_image_for_loss"] = torch.zeros(1, 3, 64, 64)
    trainer._compute_loss(embeddings)

    plan, call = _plan(), calls[0]
    sigma = torch.linspace(1.0, 0.1, 10)[9]
    noise = torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(3))
    noisy = (1 - sigma) * embeddings["image_latents"] + sigma * noise
    expected_input = torch.cat([
        noisy[:, plan.target_index],
        embeddings["control_latents"][:, plan.source_index],
        embeddings["control_latents"][:, 16:],
    ], dim=1)
    torch.testing.assert_close(call["hidden_states"], expected_input)
    assert call["output_token_count"] == 6 and call["isolate_conditions"] is False
    assert [index.tolist() for index in call["image_token_index"]] == [plan.target_index.tolist(), plan.source_index.tolist()]

    decoded = trainer.decode_vae_latent_for_loss.call_args.args[0]
    assert torch.equal(decoded[:, plan.source_index], embeddings["image_latents"][:, plan.source_index])
    torch.testing.assert_close(decoded[:, plan.target_index], noisy[:, plan.target_index] - sigma)


@pytest.mark.parametrize("isolate", [False, True])
def test_tiny_end_to_end_forward_backward(isolate):
    dit = _tiny_dit(lora=False).train()
    trainer = _loss_trainer(dit, isolate=isolate)
    embeddings = _loss_embeddings()
    loss = trainer._compute_loss(embeddings)
    assert torch.isfinite(loss)
    loss.backward()
    assert dit.img_in.weight.grad.abs().sum() > 0
    assert dit.mask_geometry_tokenizer.conv_out.weight.grad.abs().sum() > 0

    # The target outside R is never seen nor supervised; inside R it is.
    plan = _plan()
    with torch.no_grad():
        shifted = dict(embeddings, image_latents=embeddings["image_latents"].clone())
        shifted["image_latents"][:, plan.source_index] += 5.0
        assert torch.equal(trainer._compute_loss(shifted), loss.detach())
        shifted["image_latents"][:, plan.target_index[:1]] += 5.0
        assert not torch.equal(trainer._compute_loss(shifted), loss.detach())


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


def _sample(canvas, dit):
    trainer = object.__new__(QwenImageEditTrainer)
    trainer.config = NS(model=NS(complementary_canvas=canvas))
    trainer.accelerator = None
    trainer.dit = dit
    trainer.weight_dtype = torch.float32
    trainer.vae_scale_factor = 8
    trainer.scheduler = _EulerScheduler(torch.tensor([1.0, 0.7, 0.4, 0.0]))
    trainer.prepare_initial_latents = Mock(return_value=torch.randn(1, 16, 16, generator=torch.Generator().manual_seed(9)))
    trainer.prepare_predict_timesteps = Mock(return_value=(torch.tensor([1000.0, 700.0, 400.0]), 3))
    torch.manual_seed(4)
    geometry = torch.randn(1, 3, 64, 64)
    hole = torch.zeros(1, 64, 64)
    hole[:, 16:48, 16:40] = 1.0
    branch = {
        "prompt_embeds": torch.randn(1, 3, 16), "prompt_embeds_mask": torch.ones(1, 3, dtype=torch.long),
        "mask_geometry": geometry, "height": 64, "width": 64, "height_control": 64, "width_control": 64,
    }
    controls = torch.randn(1, 16 + 9, 16)
    null = {**branch, "control_latents": controls[:, :16], "n_controls": 0}
    embeddings = {
        **branch, "control_latents": controls, "n_controls": 1, "control_1": torch.zeros(1),
        "height_control_1": 48, "width_control_1": 48, "num_inference_steps": 3, "true_cfg_scale": 2.5,
        "guidance": 1.0, "reference_cfg_embeddings": null, "reference_cfg_scale": 2.5,
        "reference_cfg_mask_aware": True, "reference_cfg_mask": hole,
        "canvas_region": REGION[None].clone(), "canvas_source_latents": torch.randn(1, 16, 16),
    }
    calls = []
    dit.register_forward_pre_hook(
        lambda _module, _args, kwargs: calls.append((kwargs["hidden_states"].shape[1], kwargs.get("condition_cache"))),
        with_kwargs=True,
    )
    with patch("src.trainer.qwen_image_edit_trainer.tqdm", side_effect=lambda iterable, **_: iterable):
        latents = trainer.sampling_from_embeddings(embeddings)
    return latents, embeddings, calls, trainer


def test_sampling_keeps_source_exact_and_cached_equals_uncached():
    plan = _plan()
    isolated, embeddings, calls, _ = _sample(ComplementaryCanvasConfig(enabled=True, isolate_conditions=True), _tiny_dit(lora=True))
    # Positive [R, source outside R, reference] and null [R, source outside R] at every step.
    assert [length for length, _ in calls] == [6 + 10 + 9, 6 + 10] * 3
    assert torch.equal(isolated[:, plan.source_index], embeddings["canvas_source_latents"][:, plan.source_index])

    cached, _, cached_calls, trainer = _sample(
        ComplementaryCanvasConfig(enabled=True, isolate_conditions=True, cache_conditions=True), _tiny_dit(lora=True)
    )
    assert [(length, cache.segment_lengths) for length, cache in cached_calls] == [(6, (10, 9)), (6, (10,))] * 3
    assert trainer._last_complementary_canvas == {
        "edit_tokens": 6, "ring_tokens": 0, "grid_tokens": 16, "source_condition_tokens": 10, "text_tokens": 3,
        "text_tokens_null": 3,
        "text_cached": False, "condition_cache_bytes": 2 * 2 * 19 * 32 * 4,
    }
    torch.testing.assert_close(cached, isolated, rtol=1e-5, atol=1e-5)

    joint, _, _, _ = _sample(ComplementaryCanvasConfig(enabled=True), _tiny_dit(lora=True))
    assert torch.equal(joint[:, plan.source_index], embeddings["canvas_source_latents"][:, plan.source_index])
    assert not torch.allclose(joint, isolated)


def test_canvas_config_defaults_dependencies_and_contracts(tmp_path):
    insert = {"trainer": "QwenImageEditInsertAnything", "predict": {"devices": {}}}
    dense = Config.model_validate(insert)
    assert dense.model.complementary_canvas.model_dump() == {
        "enabled": False, "isolate_conditions": False, "isolation": "image", "cache_conditions": False,
        "ring_tokens": 0, "ring_sigma_max": 0.3, "ring_loss_weight": 0.1, "text_as_condition": False,
        "anneal_steps": 0, "anneal_max_bias": 12.0, "local_decode_margin_px": None, "keep_region_source": False,
    }
    with pytest.raises(ValidationError, match="require enabled"):
        ComplementaryCanvasConfig(isolate_conditions=True)
    with pytest.raises(ValidationError, match="requires isolate_conditions"):
        ComplementaryCanvasConfig(enabled=True, cache_conditions=True)

    def canvas(**fields):
        return {**insert, "model": {"zero_cond_t": True, "complementary_canvas": {"enabled": True, **fields}}}

    def anchored(anchoring, **fields):
        return {**canvas(**fields), "predict": {"devices": {}, "latent_anchoring": anchoring}}

    deploy = Config.model_validate(anchored({"mode": "expanded_soft", "transition_width_px": 16, "paste_back": "change_gated"}))
    assert deploy.predict.latent_anchoring.paste_back == "change_gated"
    Config.model_validate(anchored({"mode": "expanded"}))
    with pytest.raises(ValidationError, match="mode expanded or expanded_soft"):
        Config.model_validate(anchored({"mode": "core"}))
    with pytest.raises(ValidationError, match="ring_tokens"):
        Config.model_validate(anchored({"mode": "expanded_soft"}, ring_tokens=1))
    with pytest.raises(ValidationError, match="source_noise_start"):
        Config.model_validate(anchored({"mode": "expanded", "source_noise_start": 0.9}))
    with pytest.raises(ValidationError, match="strict M_core composite"):
        Config.model_validate({**insert, "predict": {"devices": {}, "latent_anchoring": {"paste_back": "change_gated"}}})
    Config.model_validate(anchored({"paste_back": "change_gated"}))  # legacy canvas: the restored output is primary
    with pytest.raises(ValidationError, match="batch_size=1"):
        Config.model_validate({**canvas(), "data": {"batch_size": 2}})
    with pytest.raises(ValidationError, match="requires trainer"):
        Config.model_validate({**canvas(), "trainer": "QwenImageEdit"})
    with pytest.raises(ValidationError, match="zero_cond_t"):
        Config.model_validate({**canvas(isolate_conditions=True, cache_conditions=True), "model": {
            "complementary_canvas": {"enabled": True, "isolate_conditions": True, "cache_conditions": True}}})

    b1 = Config.model_validate(canvas())
    b2 = Config.model_validate(canvas(isolate_conditions=True))
    b2_cached = Config.model_validate(canvas(isolate_conditions=True, cache_conditions=True))
    dense = Config.model_validate({**insert, "model": {"zero_cond_t": True}})

    def mismatches(checkpoint, inference):
        return {
            item["field"]
            for item in _flatten_contract_mismatches(
                _checkpoint_contract(checkpoint, "train_enabled"), _checkpoint_contract(inference, "predict_enabled")
            )
        }

    assert mismatches(dense, b1) == {"model.complementary_canvas.enabled", "model.complementary_canvas.isolate_conditions"}
    assert mismatches(b1, b2) == {"model.complementary_canvas.isolate_conditions"}
    assert mismatches(b2, b2_cached) == set()  # caching is an exact inference-only speedup

    # Old checkpoints carry no canvas field and stay dense; resume cannot switch modes.
    raw = dense.model_dump(mode="json")
    raw["model"].pop("complementary_canvas")
    (tmp_path / "train_config.yaml").write_text(yaml.safe_dump(raw))
    validate_resume_mask_control_compatibility(dense, tmp_path)
    with pytest.raises(ValueError, match="complementary_canvas mismatch"):
        validate_resume_mask_control_compatibility(b1, tmp_path)


def test_predict_edits_m_free_tokens_and_scatters_into_the_unmasked_source():
    from PIL import Image

    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
    from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
    from src.utils.latent_anchoring import build_latent_anchoring_masks
    from tests.test_latent_anchoring import DummyPredictPreprocessor

    trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
    trainer.config = Config()
    trainer.config.model.complementary_canvas = ComplementaryCanvasConfig(enabled=True)
    trainer.preprocessor = DummyPredictPreprocessor()
    data = trainer.prepare_predict_batch_data(
        image=[Image.new("RGB", (24, 24)), Image.new("RGB", (12, 12), color="blue")],
        src_mask=Image.new("L", (24, 24), color=255),
        ref_mask=Image.new("L", (12, 12), color=255),
        prompt="insert",
        interaction_dilation_px=3,
    )
    core = torch.zeros(1, 24, 24)
    core[:, 10:14, 10:14] = 1
    expected = build_latent_anchoring_masks(core, mode="expanded", interaction_dilation_px=3)
    assert not any(key.startswith("anchor_") for key in data)
    assert torch.equal(data["tar_mask"], expected.core) and torch.equal(data["source_hole_mask"], expected.free)

    trainer.num_channels_latents = 1
    trainer.vae_scale_factor = 2  # 4 px tokens on the 24 px dummy canvas
    trainer.weight_dtype = torch.float32
    trainer.preprocess_image_for_vae_encoder = lambda image, **_kwargs: image.unsqueeze(2)
    encoded = torch.full((1, 36, 4), 7.0)
    with (
        patch.object(QwenImageEditPlusTrainer, "prepare_embeddings", autospec=True,
                     side_effect=lambda _self, parent_batch, stage="fit": parent_batch),
        patch.object(trainer, "encode_image_latents", return_value=encoded) as encode,
    ):
        embeddings = trainer.prepare_embeddings(data, stage="predict")
    torch.testing.assert_close(encode.call_args.args[0], data["tar_image"].unsqueeze(2))
    assert embeddings["canvas_source_latents"] is encoded
    assert torch.equal(embeddings["canvas_region"], token_region_from_mask(expected.free, 4))
    assert torch.all(embeddings["control"][:, :, expected.free[0].bool()] == 0)


def test_inference_runtime_and_crop_guard_treat_canvas_as_m_free_support():
    from PIL import Image

    from src.data.config import LatentAnchoringConfig
    from src.utils.adaptive_crop_anchoring import prepare_adaptive_crop_for_inference
    from src.utils.latent_anchoring import validate_latent_anchoring_runtime

    def config(canvas):
        return Config.model_validate({
            "trainer": "QwenImageEditInsertAnything", "predict": {"devices": {}},
            "model": {"complementary_canvas": {"enabled": canvas}},
            "data": {"init_args": {"adaptive_crop": {"predict_enabled": True}, "processor": {
                "class_path": "src.data.preprocess.ImageProcessor",
                "init_args": {"process_type": "resize", "target_size": [64, 64]}}}},
        })

    mask = Image.new("L", (256, 256))
    mask.paste(255, (100, 100, 130, 140))
    for canvas, checked in ((False, False), (True, True)):
        cfg = config(canvas)
        _, compatibility = prepare_adaptive_crop_for_inference(
            image_size=(256, 256), mask=mask, adaptive_crop_config=cfg.data.init_args.adaptive_crop, config=cfg,
        )
        assert compatibility["anchoring_enabled"] is checked
        assert (compatibility["safety"] != "legacy_no_check") is checked

    cfg = config(True)
    validate_latent_anchoring_runtime(cfg)
    with pytest.raises(ValueError, match="explicit"):
        validate_latent_anchoring_runtime(cfg, height=64, width=64)
    # CLI overrides replace the anchoring settings after Config validation; the runtime check reruns the contract.
    cfg.predict.latent_anchoring = LatentAnchoringConfig(mode="expanded_soft", transition_width_px=4)
    validate_latent_anchoring_runtime(cfg)
    cfg.predict.latent_anchoring = LatentAnchoringConfig(mode="core")
    with pytest.raises(ValueError, match="mode expanded or expanded_soft"):
        validate_latent_anchoring_runtime(cfg)
    dense = config(False)
    dense.predict.latent_anchoring = LatentAnchoringConfig(paste_back="change_gated")
    with pytest.raises(ValueError, match="strict M_core composite"):
        validate_latent_anchoring_runtime(dense)

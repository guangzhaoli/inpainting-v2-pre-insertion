"""Numerical and execution checks for the normal teacher/OPD training path."""

import copy
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader

from src.distillation.backends import set_active_adapter
from src.distillation.opd import QwenImageEditOPDStrategy
from src.distillation.types import FlowTrajectory
from src.models.mask_geometry import MaskGeometryTokenizer
from src.models.transformer_qwenimage import QwenImageTransformer2DModel, QwenEmbedRope
from src.trainer.base_trainer import BaseTrainer
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer
from tests.smoke_opd_distributed import Tiny
from tests.test_opd_strategy import FakeTrainer, _guidance, _plugin, _teacher


@pytest.mark.parametrize("inference", [False, True])
def test_text_encoder_is_offloaded_before_vae_encoding(inference):
    trainer = QwenImageEditPlusTrainer.__new__(QwenImageEditPlusTrainer)
    trainer.config = NS(train=NS(offload_text_encoder=True), loss=NS(mask_loss=False))
    trainer.accelerator = NS(device=torch.device("cuda:0"))
    trainer.text_encoder = NS(device=torch.device("cpu"))

    def stream(device):
        # The pinned host copy is streamed outside inference mode; None returns to the host copy.
        assert not torch.is_inference_mode_enabled()
        trainer.text_encoder.device = torch.device("cpu" if device is None else device)

    trainer._stream_text_encoder = stream
    trainer.weight_dtype = torch.float32
    trainer.num_channels_latents = 1
    trainer.preprocess_image_for_text_encoder = lambda value, **kwargs: value * 255
    trainer.preprocess_image_for_vae_encoder = lambda value, **kwargs: value.unsqueeze(2)
    calls = []

    def text(*args):
        assert trainer.text_encoder.device.type == "cuda"
        calls.append("text")
        return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.long)

    def vae(*args):
        assert trainer.text_encoder.device.type == "cpu"
        calls.append("vae")
        return torch.ones(1, 4, 4)

    trainer._get_qwen_prompt_embeds = text
    trainer.encode_image_latents = vae
    batch = dict(control=torch.rand(1, 3, 32, 32), control_1=torch.rand(1, 3, 32, 32),
                 prompt=[""], n_controls=1, height=32, width=32)
    with torch.inference_mode(inference):
        trainer.prepare_embeddings(batch, stage="opd")
    assert calls == ["text", "vae", "vae"]


def test_standalone_prompt_encoding_keeps_explicit_prediction_device():
    trainer = QwenImageEditPlusTrainer.__new__(QwenImageEditPlusTrainer)
    trainer.config = NS(train=NS(offload_text_encoder=True))
    trainer.accelerator = None
    trainer.text_encoder = NS(device=torch.device("cpu"), to=Mock())
    trainer._get_qwen_prompt_embeds = lambda *args: (torch.ones(1, 2, 4), torch.ones(1, 2))
    trainer.encode_prompt("")
    trainer.text_encoder.to.assert_not_called()


def test_cpu_hybrid_metrics_are_sent_to_the_communication_device():
    trainer = QwenImageEditInsertAnythingTrainer.__new__(QwenImageEditInsertAnythingTrainer)
    records = torch.arange(32, dtype=torch.float64).reshape(2, 16)
    trainer._hybrid_masking_metric_buffer = [records[:1], records[1:]]

    def gather(value):
        # Meta exercises an actual device transfer on a CPU-only test host.
        assert value.device.type == "meta"
        return records

    trainer.accelerator = NS(device=torch.device("meta"), gather=gather)
    with patch.object(BaseTrainer, "build_training_logs", return_value={}):
        logs = trainer.build_training_logs(0, [None], None, 1.)
    assert logs["hybrid/gt_area_mean"] == 8.
    assert logs["hybrid/condition_area_p90"] == pytest.approx(16.4)
    assert logs["hybrid/mode_oversized_box_rate"] == 21.
    assert logs["hybrid/condition_expansion_max"] == 31.
    assert not trainer._hybrid_masking_metric_buffer


@pytest.mark.parametrize("microbatches,expected", [(3, .2), (5, .2), (8, .36)])
def test_actual_training_loop_averages_full_and_partial_windows(microbatches, expected):
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=5, mixed_precision="no")
    accelerator.step = 0
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    loader = DataLoader(torch.ones(microbatches, 1), batch_size=1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    trainer = NS(accelerator=accelerator, dit=model, optimizer=optimizer, lr_scheduler=Mock(),
                 config=NS(train=NS(max_grad_norm=100., gradient_accumulation_steps=5)),
                 start_epoch=0, resume_batch_in_epoch=0, global_step=0,
                 effective_max_train_steps=10, batch_size=1, running_loss=0.,
                 validation_sampler=None, fps_logger=Mock(), _trainable_params=list(model.parameters()),
                 save_checkpoint=Mock(), build_training_logs=Mock(return_value={}))
    trainer._set_dataloader_epoch = BaseTrainer._set_dataloader_epoch
    trainer._sync_training_interrupted = lambda: False
    trainer.clip_gradients = MethodType(BaseTrainer.clip_gradients, trainer)

    def backward(batch):
        loss = (model(batch) - 1).square().mean()
        accelerator.backward(loss)
        return loss.detach()

    trainer.backward_training_step = backward
    trainer.update_progressbar = lambda logs: setattr(trainer, "global_step", trainer.global_step + 1)
    BaseTrainer.train_epoch(trainer, 0, loader)
    assert model.weight.item() == pytest.approx(expected)


def test_batched_teacher_phase_matches_interleaved_gradients_with_real_peft():
    torch.manual_seed(79)
    lora = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    model = get_peft_model(Tiny(), lora, adapter_name="student")
    teachers = [_teacher(f"expert{i}", "cat" if i == 0 else f"category{i}", weight=2.) for i in range(5)]
    for teacher in teachers:
        model.add_adapter(teacher.name, lora)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.1)
    set_active_adapter(model, "student", trainable_adapter="student")
    trajectory = FlowTrajectory(states=torch.randn(1, 11, 2, 3),
                                timesteps=torch.linspace(1000, 100, 10), sigmas=torch.linspace(1, 0, 11))
    config = NS(objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
                teacher_backend=_plugin("src.distillation.backends.SharedLoRATeacherBackend"),
                teachers=teachers, fallback_teacher=None, student_guidance=_guidance(),
                rollout=NS(num_inference_steps=10, timestep_fraction=1., store_device="cpu", seed=42))
    runs = []
    for grouped in (False, True):
        trainer = FakeTrainer()
        trainer.dit = copy.deepcopy(model)
        trainer._unwrap_dit = lambda: trainer.dit
        trainer._dit_config = lambda: NS(in_channels=4, guidance_embeds=False)
        strategy = QwenImageEditOPDStrategy(config)
        strategy.backend._trainer = trainer
        strategy.backend._student_adapter = "student"
        strategy.backend._teacher_to_adapter = {t.name: t.name for t in teachers}
        strategy._is_setup = True
        batch = {"category": ["cat"], "prompt": [""]}

        def rollout(*args):
            strategy.backend.activate_student()
            return trajectory

        strategy._rollout = rollout
        with patch("src.distillation.backends.set_active_adapter", wraps=set_active_adapter) as switch:
            if grouped:
                loss = strategy.backward_step(trainer, batch)
                assert switch.call_count == 3
            else:
                batch = trainer.prepare_opd_batch(batch)
                conditions = strategy._prepare_conditions(trainer, batch, ["expert0"])
                condition = conditions[strategy._condition_key(config.student_guidance)]
                rollout()
                losses = []
                for step in range(10):
                    state, timestep = trajectory.states[:, step], trajectory.timesteps[step]
                    trainer.dit.eval()
                    with torch.no_grad():
                        target = strategy.backend.predict_velocity(
                            teachers[0], trainer=trainer, state=state, timestep=timestep,
                            condition=condition, batch=batch, guidance=teachers[0].guidance,
                            shared_predictor=strategy._predict_guided_velocity,
                        )
                    trainer.dit.train()
                    prediction = strategy._predict_guided_velocity(trainer, state, timestep, condition, config.student_guidance)
                    loss = strategy.objective(prediction, target, teacher_weight=torch.tensor([2.])).loss
                    (loss / 10).backward()
                    losses.append(loss.detach())
                loss = torch.stack(losses).mean()
        grads = {name: p.grad.clone() for name, p in trainer.dit.named_parameters() if p.grad is not None}
        assert all(".student." in name for name in grads)
        runs.append((loss, grads))
    torch.testing.assert_close(runs[0][0], runs[1][0])
    assert runs[0][1].keys() == runs[1][1].keys()
    for name in runs[0][1]:
        torch.testing.assert_close(runs[0][1][name], runs[1][1][name])


@pytest.mark.parametrize("checkpointed", [False, True])
def test_target_readout_and_final_text_pruning_preserve_outputs_and_gradients(checkpointed):
    torch.manual_seed(15)
    model = QwenImageTransformer2DModel(num_layers=2, attention_head_dim=16, num_attention_heads=2,
        in_channels=16, out_channels=4, joint_attention_dim=16, axes_dims_rope=(4, 6, 6), zero_cond_t=True)
    reference = copy.deepcopy(model)
    original_forward = reference.transformer_blocks[-1].forward

    def full_text(*args, **kwargs):
        if len(args) == 8:
            args = (*args[:7], False)
        else:
            kwargs["skip_text_output"] = False
        return original_forward(*args, **kwargs)

    reference.transformer_blocks[-1].forward = full_text
    if checkpointed:
        model.enable_gradient_checkpointing()
        reference.enable_gradient_checkpointing()
    kwargs = dict(encoder_hidden_states=torch.randn(1, 3, 16), encoder_hidden_states_mask=torch.ones(1, 3),
                  timestep=torch.tensor([.5]), img_shapes=[[(1, 2, 2), (1, 2, 2)]],
                  txt_seq_lens=[3], return_dict=False)
    source = torch.randn(1, 8, 16)
    outputs = []
    calls = []
    model.transformer_blocks[-1].txt_mlp.register_forward_hook(lambda *args: calls.append(True))
    for m, prefix in ((reference, None), (model, 4)):
        value = m(source, output_token_count=prefix, **kwargs)[0][:, :4]
        value.square().mean().backward()
        outputs.append(value)
    assert calls == []
    torch.testing.assert_close(*outputs)
    for p, q in zip(model.parameters(), reference.parameters()):
        if p.grad is not None or q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad)


def test_prepooled_geometry_keeps_bf16_cast_order_and_trainable_gradients():
    torch.manual_seed(31)
    tokenizer = MaskGeometryTokenizer(8, 16).float()
    nn.init.normal_(tokenizer.conv_out.weight)
    reference = copy.deepcopy(tokenizer)
    geometry = torch.randn(1, 3, 64, 96)
    model = NS(mask_geometry_tokenizer=tokenizer,
               _mask_geometry_target_grid=lambda shapes, batch: (4, 6))
    trainer = NS(dit=model, weight_dtype=torch.bfloat16, _dit_device=lambda: torch.device("cpu"),
                 _get_image_shapes=lambda condition, batch: [[(1, 4, 6)]])
    condition = {"mask_geometry": geometry, "reference_cfg_embeddings": {"mask_geometry": geometry}}
    prepared = QwenImageEditOPDStrategy._prepare_student_condition(trainer, condition)
    assert prepared["mask_geometry"] is prepared["reference_cfg_embeddings"]["mask_geometry"]
    assert prepared["mask_geometry"].shape == (1, 3, 4, 6)
    actual = tokenizer(prepared["mask_geometry"], (4, 6))
    expected = reference(geometry.bfloat16(), (4, 6))
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for p, q in zip(tokenizer.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    model.prepare_mask_geometry_tokens = lambda value, shapes: tokenizer(value, (4, 6))
    trainer.accelerator = NS(autocast=lambda: torch.autocast("cpu", dtype=torch.bfloat16))
    cached = QwenImageEditOPDStrategy._cache_student_geometry(trainer, prepared)
    with torch.no_grad(), trainer.accelerator.autocast():
        training_tokens = tokenizer(geometry.bfloat16(), (4, 6))
    torch.testing.assert_close(cached["mask_geometry_tokens"], training_tokens, rtol=0, atol=0)


def test_rope_cache_evicts_old_shapes_without_changing_values():
    rope = QwenEmbedRope(10000, [4, 6, 6], scale_rope=True)
    original = rope([[(1, 2, 2)]], [3], torch.device("cpu"))[0].clone()
    for height in range(3, 100):
        rope([[(1, height, 2)]], [3], torch.device("cpu"))
    assert len(rope.rope_cache) <= 64
    recreated = rope([[(1, 2, 2)]], [3], torch.device("cpu"))[0]
    torch.testing.assert_close(original, recreated, rtol=0, atol=0)

"""Exercise real checkpoint formats and BF16 reloads of FP32 trainables."""

import copy
from pathlib import Path

import pytest
import torch
from diffusers.utils import convert_state_dict_to_diffusers
from peft import get_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from scripts.build_meanflow_config import build_config
from src.data.config import Config
from src.trainer.base_trainer import BaseTrainer
from src.utils.seed import prepare_trainable_precision
from tests.test_meanflow_dmd import FitTrainer, TinyAccelerator, make_case, tiny_model


# Same targets as the production Qwen InsertFuse configs; q/k/v-only fixtures
# do not expose the mixed-format export bug.
QWEN_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def reset_accelerator():
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.fixture(autouse=True)
def isolate_training_state():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    reset_accelerator()
    yield
    reset_accelerator()
    torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "filename",
    ["model.safetensors", "pytorch_lora_weights.safetensors", "legacy.safetensors"],
)
@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_all_qwen_lora_targets_survive_checkpoint_loading(tmp_path, filename, stage):
    trainer, _, _ = make_case(tmp_path)
    trainer.dit = tiny_model()
    trainer.config.model.lora.target_modules = QWEN_TARGETS
    trainer.config.train.mixed_precision = "bf16"
    BaseTrainer.add_lora_adapter(trainer.dit, trainer.config, trainer.adapter_name)
    with torch.no_grad():
        for name, parameter in trainer.dit.named_parameters():
            if "lora_" in name:
                parameter.uniform_(-0.1, 0.1)
    expected = get_peft_model_state_dict(trainer.dit, adapter_name=trainer.adapter_name)
    folder = tmp_path / "saved"
    folder.mkdir()
    trainer.save_lora(str(folder))
    # Files previously produced by this repository are a normal input too.
    trainer.pipeline_class.save_lora_weights(
        str(folder),
        convert_state_dict_to_diffusers(expected),
        weight_name="legacy.safetensors",
        safe_serialization=True,
    )
    config = trainer.config.model_copy(deep=True)
    config.model.lora.pretrained_weight = str(folder / filename)
    restored = tiny_model().bfloat16()
    BaseTrainer.load_pretrain_lora_model(restored, config, "restored", stage=stage)
    actual = get_peft_model_state_dict(restored, adapter_name="restored")
    assert actual.keys() == expected.keys()
    dtype = torch.float32 if stage == "fit" else torch.bfloat16
    for key in expected:
        assert actual[key].dtype == dtype
        torch.testing.assert_close(actual[key], expected[key].to(dtype), rtol=0, atol=0)
    assert restored.img_in.weight.dtype == torch.bfloat16
    if filename == "pytorch_lora_weights.safetensors":
        exported = load_file(str(folder / filename))
        assert not any(".lora.down." in key or ".lora.up." in key for key in exported)
        official = tiny_model().bfloat16()
        official.load_lora_adapter(
            str(folder),
            weight_name=filename,
            adapter_name="official",
            local_files_only=True,
        )
        official_state = get_peft_model_state_dict(official, adapter_name="official")
        assert official_state.keys() == expected.keys()
        for key in expected:
            torch.testing.assert_close(
                official_state[key], expected[key].bfloat16(), rtol=0, atol=0
            )


def make_fit_config(prototype, output_dir, stage):
    config = prototype.config.model_copy(deep=True)
    config.train.mixed_precision = "bf16"
    config.train.gradient_accumulation_steps = 2
    config.train.max_train_steps = 2
    config.train.num_epochs = 1
    config.train.checkpointing_steps = 1
    config.data.num_workers = 0
    config.data.drop_last = False
    config.logging.output_dir = str(output_dir)
    config.logging.sampling.enable = False
    config.distillation.meanflow.stage = stage
    return config


def loader(batch):
    return DataLoader([batch] * 4, batch_size=1, collate_fn=lambda rows: rows[0])


def assert_optimizer_equal(left, right):
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for parameter_id in left["state"]:
        for key, value in left["state"][parameter_id].items():
            other = right["state"][parameter_id][key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                assert value == other


def test_bf16_resume_matches_uninterrupted_training(tmp_path):
    prototype, _, batch = make_case(tmp_path, "dmd")
    config = make_fit_config(prototype, tmp_path / "continuous", "dmd")
    original = FitTrainer(config)
    original.fit(loader(batch))
    folder = Path(original.config.logging.output_dir) / "checkpoint-0-1"
    assert folder.is_dir()
    expected = {
        name: parameter.detach().clone()
        for name, parameter in original.dit.named_parameters()
        if parameter.requires_grad
    }
    expected_g_optimizer = copy.deepcopy(original.optimizer.state_dict())
    expected_f_optimizer = copy.deepcopy(
        original.training_strategy.fake_optimizer.state_dict()
    )

    reset_accelerator()
    resume_config = make_fit_config(prototype, tmp_path / "resumed", "dmd")
    resume_config.resume = str(folder)
    resumed = FitTrainer(resume_config)
    resumed.fit(loader(batch))
    assert resumed.global_step == original.global_step == 2
    assert (
        resumed.training_strategy.fake_update_count
        == original.training_strategy.fake_update_count
        == 8
    )
    for name, parameter in resumed.dit.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    assert_optimizer_equal(resumed.optimizer.state_dict(), expected_g_optimizer)
    assert_optimizer_equal(
        resumed.training_strategy.fake_optimizer.state_dict(), expected_f_optimizer
    )


def test_bf16_warmup_to_dmd_preserves_trainable_weights(tmp_path):
    prototype, _, batch = make_case(tmp_path, "warmup")
    warmup = FitTrainer(make_fit_config(prototype, tmp_path / "warmup", "warmup"))
    warmup.fit(loader(batch))
    folder = next(Path(warmup.config.logging.output_dir).glob("checkpoint-last-*"))
    config = Config.model_validate(
        build_config(tmp_path / "teacher", stage="dmd", generator=folder)
    )
    target = FitTrainer(config)
    target.accelerator = TinyAccelerator()
    target.accelerator.mixed_precision = "bf16"
    target.load_model()
    target.load_pretrain_lora_model(target.dit, config, target.adapter_name)
    target.load_additional_modules(stage="fit")
    target.setup_training_strategy()
    target.setup_model_device_train_mode()
    target.training_strategy.configure_trainability(target)
    prepare_trainable_precision(target.dit)

    expected = get_peft_model_state_dict(warmup.dit, adapter_name=warmup.adapter_name)
    actual = get_peft_model_state_dict(target.dit, adapter_name=target.adapter_name)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for name in ("mask_geometry_tokenizer", "flow_map_time_embedding"):
        expected_module = getattr(warmup.dit, name).state_dict()
        for key, value in getattr(target.dit, name).state_dict().items():
            torch.testing.assert_close(value, expected_module[key], rtol=0, atol=0)
    expected_fake = load_file(str(tmp_path / "teacher" / "model.safetensors"))
    actual_fake = get_peft_model_state_dict(
        target.dit, adapter_name=target.training_strategy.fake_adapter
    )
    for key in expected_fake:
        torch.testing.assert_close(actual_fake[key], expected_fake[key], rtol=0, atol=0)

import os
import tempfile
import unittest

import torch
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from src.loss.edit_mask_loss import map_mask_to_latent
from src.trainer.base_trainer import BaseTrainer
from src.utils.images import build_preinserted_composite, resize_mask_bhw
from src.utils.lora_utils import classify_lora_weight, filter_lora_state_keys, is_lora_state_key


class MaskUtilsTest(unittest.TestCase):
    def test_map_mask_to_latent_uses_vae_scale_factor(self):
        mask = torch.zeros(1, 32, 32)
        mask[:, 16:, 16:] = 1.0

        packed = map_mask_to_latent(mask, vae_scale_factor=8, expected_seq_len=4)

        self.assertEqual(tuple(packed.shape), (1, 4))
        self.assertEqual(packed.tolist(), [[0.0, 0.0, 0.0, 1.0]])

    def test_map_mask_to_latent_rejects_seq_len_mismatch(self):
        mask = torch.zeros(1, 32, 32)

        with self.assertRaises(ValueError):
            map_mask_to_latent(mask, vae_scale_factor=8, expected_seq_len=5)

    def test_resize_mask_bhw_uses_nearest(self):
        mask = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

        resized = resize_mask_bhw(mask, 4, 4)

        self.assertEqual(sorted(set(resized.flatten().tolist())), [0.0, 1.0])

    def test_preinserted_composite_rejects_unknown_paste_mode(self):
        image = torch.zeros(1, 3, 4, 4)
        mask = torch.zeros(1, 4, 4)

        with self.assertRaises(ValueError):
            build_preinserted_composite(image, mask, image, mask, paste_mode="unknown")


class LoraUtilsTest(unittest.TestCase):
    def test_lora_key_filter_handles_prefix_and_nested_keys(self):
        keys = [
            "lora_A.default.weight",
            "transformer.block.attn.to_q.lora_B.default.weight",
            "transformer.block.weight",
        ]

        self.assertTrue(is_lora_state_key(keys[0]))
        self.assertTrue(is_lora_state_key(keys[1]))
        self.assertFalse(is_lora_state_key(keys[2]))
        self.assertEqual(filter_lora_state_keys(keys), keys[:2])

    def test_classify_lora_weight_handles_prefix_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            weight_path = os.path.join(tmpdir, "model.safetensors")
            save_file({"lora_A.default.weight": torch.zeros(1, 1)}, weight_path)

            self.assertEqual(classify_lora_weight(weight_path), "PEFT")

    def test_peft_loader_accepts_saved_stripped_adapter_keys(self):
        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(4, 3, bias=False)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                block = nn.Module()
                block.attn = TinyAttention()
                self.transformer_blocks = nn.ModuleList([block])

            def forward(self, x):
                return self.transformer_blocks[0].attn.to_q(x)

        source = inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]),
            TinyModel(),
            adapter_name="source_adapter",
        )
        saved = get_peft_model_state_dict(source, adapter_name="source_adapter")
        saved = {
            key: torch.full_like(value, fill_value=index + 1.0)
            for index, (key, value) in enumerate(saved.items())
        }

        target = inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]),
            TinyModel(),
            adapter_name="target_adapter",
        )

        missing = BaseTrainer._load_peft_lora_state_dict(
            target,
            saved,
            adapter_name="target_adapter",
            stage="fit",
        )

        self.assertEqual(filter_lora_state_keys(missing), [])
        target_state = target.state_dict()
        self.assertTrue(
            torch.equal(
                target_state["transformer_blocks.0.attn.to_q.lora_A.target_adapter.weight"],
                saved["transformer_blocks.0.attn.to_q.lora_A.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                target_state["transformer_blocks.0.attn.to_q.lora_B.target_adapter.weight"],
                saved["transformer_blocks.0.attn.to_q.lora_B.weight"],
            )
        )

    def test_peft_loader_rewrites_saved_adapter_name(self):
        class TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(4, 3, bias=False)

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                block = nn.Module()
                block.attn = TinyAttention()
                self.transformer_blocks = nn.ModuleList([block])

            def forward(self, x):
                return self.transformer_blocks[0].attn.to_q(x)

        source = inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]),
            TinyModel(),
            adapter_name="source_adapter",
        )
        saved = get_peft_model_state_dict(source, adapter_name="source_adapter")
        saved = {
            key.replace(".lora_A.weight", ".lora_A.source_adapter.weight")
            .replace(".lora_B.weight", ".lora_B.source_adapter.weight"): torch.full_like(
                value,
                fill_value=index + 1.0,
            )
            for index, (key, value) in enumerate(saved.items())
        }

        target = inject_adapter_in_model(
            LoraConfig(r=2, lora_alpha=2, target_modules=["to_q"]),
            TinyModel(),
            adapter_name="target_adapter",
        )

        missing = BaseTrainer._load_peft_lora_state_dict(
            target,
            saved,
            adapter_name="target_adapter",
            stage="fit",
        )

        self.assertEqual(filter_lora_state_keys(missing), [])
        target_state = target.state_dict()
        self.assertTrue(
            torch.equal(
                target_state["transformer_blocks.0.attn.to_q.lora_A.target_adapter.weight"],
                saved["transformer_blocks.0.attn.to_q.lora_A.source_adapter.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                target_state["transformer_blocks.0.attn.to_q.lora_B.target_adapter.weight"],
                saved["transformer_blocks.0.attn.to_q.lora_B.source_adapter.weight"],
            )
        )


if __name__ == "__main__":
    unittest.main()

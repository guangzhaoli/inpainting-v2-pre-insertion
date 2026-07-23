import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

from src.data.config import Config, MaskControlConfig, ModelConfig
from src.loss.edit_mask_loss import map_mask_to_latent
from src.trainer.base_trainer import BaseTrainer
from src.trainer.qwen_image_edit_insert_anything_trainer import (
    MASK_GEOMETRY_TOKENIZER_FILE_NAME,
    QwenImageEditInsertAnythingTrainer,
)
from src.utils.images import resize_mask_bhw
from src.utils.lora_utils import classify_lora_weight, filter_lora_state_keys, is_lora_state_key
from src.utils.tools import calculate_sha256_file


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

class LoraUtilsTest(unittest.TestCase):
    def test_checkpoint_manifest_records_lora_and_geometry_companion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.safetensors").write_bytes(b"lora")
            (root / MASK_GEOMETRY_TOKENIZER_FILE_NAME).write_bytes(b"geometry")

            manifest = BaseTrainer._build_checkpoint_artifact_manifest(tmpdir)

            by_role = {entry["role"]: entry for entry in manifest["files"]}
            self.assertEqual(manifest["version"], 1)
            self.assertEqual(
                set(by_role),
                {"lora_peft", "mask_geometry_tokenizer"},
            )
            for entry in manifest["files"]:
                path = root / entry["filename"]
                self.assertEqual(entry["size_bytes"], path.stat().st_size)
                self.assertEqual(
                    entry["sha256"],
                    calculate_sha256_file(str(path)),
                )

    def test_resume_rejects_tampered_geometry_companion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.safetensors").write_bytes(b"lora")
            geometry_path = root / MASK_GEOMETRY_TOKENIZER_FILE_NAME
            geometry_path.write_bytes(b"geometry")

            checkpoint_config = Config(
                trainer="QwenImageEditInsertAnything",
                model=ModelConfig(
                    mask_control=MaskControlConfig(
                        enabled=True,
                        encoding="geometry_tokenizer",
                        tokenizer_hidden_channels=4,
                    )
                ),
            )
            (root / "train_config.yaml").write_text(
                yaml.safe_dump(
                    checkpoint_config.model_dump(mode="json", exclude_none=True),
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "artifacts": BaseTrainer._build_checkpoint_artifact_manifest(
                            tmpdir
                        )
                    }
                ),
                encoding="utf-8",
            )
            geometry_path.write_bytes(b"tampered geometry")

            trainer = object.__new__(QwenImageEditInsertAnythingTrainer)
            trainer.config = checkpoint_config
            trainer.accelerator = None
            with self.assertRaisesRegex(ValueError, "artifact integrity"):
                trainer._validate_lora_resume_checkpoint(tmpdir)

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

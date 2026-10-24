import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

from scripts.analyze_category_gradient_conflict import (
    build_balanced_sample_plan,
    build_parameter_layout,
    compute_component_grams,
    cosine_from_gram,
    load_analysis_config,
    parse_group_specs,
    resolve_config_path,
    resolve_checkpoint_weight,
    stratified_timestep_plan,
    summarize_pairwise_cosines,
    write_analysis_outputs,
    write_gradient_memmap,
)
from src.data.config import Config
from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer


class CategoryGradientConflictTest(unittest.TestCase):
    @staticmethod
    def _loss_harness():
        class CaptureDit:
            def __init__(self):
                self.timesteps = []

            def __call__(self, hidden_states, timestep, **_kwargs):
                self.timesteps.append(timestep.detach().clone())
                return (torch.zeros_like(hidden_states),)

        trainer = object.__new__(QwenImageEditTrainer)
        trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
        trainer.weight_dtype = torch.float32
        trainer.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=10))
        trainer.config = SimpleNamespace(loss=SimpleNamespace(region_perceptual=False))
        trainer.dit = CaptureDit()
        trainer._get_cached_scheduler_training_tensors = lambda _device, dtype: (
            torch.arange(10, dtype=dtype),
            torch.linspace(0.0, 0.9, 10, dtype=dtype),
        )
        trainer._get_sigmas = lambda timesteps, n_dim, dtype: (
            timesteps.to(dtype=dtype).reshape(-1, *([1] * (n_dim - 1))) / 10.0
        )
        embeddings = {
            "image_latents": torch.zeros(1, 2, 1),
            "control_latents": torch.zeros(1, 2, 1),
            "prompt_embeds": torch.zeros(1, 1, 2),
            "prompt_embeds_mask": torch.ones(1, 1, dtype=torch.long),
            "img_shapes": [[]],
        }
        return trainer, embeddings

    def test_analysis_randomness_override_is_stateless_and_matches_timestep(self):
        trainer, embeddings = self._loss_harness()
        controlled = {
            **embeddings,
            "_analysis_noise_seed": 123,
            "_analysis_timestep_u": 0.35,
        }

        torch.manual_seed(1)
        first = QwenImageEditTrainer._compute_loss(trainer, controlled)
        torch.manual_seed(999)
        second = QwenImageEditTrainer._compute_loss(trainer, controlled)

        torch.testing.assert_close(first, second)
        self.assertEqual(len(trainer.dit.timesteps), 2)
        torch.testing.assert_close(trainer.dit.timesteps[0], torch.tensor([0.003]))
        torch.testing.assert_close(trainer.dit.timesteps[1], torch.tensor([0.003]))

    def test_ordinary_training_keeps_original_random_sampling_path(self):
        trainer, embeddings = self._loss_harness()
        with (
            patch(
                "src.trainer.qwen_image_edit_trainer.torch.randn_like",
                return_value=torch.ones_like(embeddings["image_latents"]),
            ) as randn_like,
            patch(
                "src.trainer.qwen_image_edit_trainer.compute_density_for_timestep_sampling",
                return_value=torch.tensor([0.2]),
            ) as sample_timestep,
        ):
            QwenImageEditTrainer._compute_loss(trainer, embeddings)

        randn_like.assert_called_once()
        sample_timestep.assert_called_once()

    def test_group_parser_rejects_overlapping_native_categories(self):
        with self.assertRaisesRegex(ValueError, "multiple groups"):
            parse_group_specs(["accessory=watch,jewel", "garment=shirt,watch"])

    def test_balanced_plan_is_unique_and_balanced_per_repeat(self):
        category_to_indices = {
            "a": list(range(0, 20)),
            "b": list(range(20, 40)),
            "c": list(range(40, 60)),
        }
        groups = OrderedDict([("first", ("a", "b")), ("second", ("c",))])
        plan = build_balanced_sample_plan(
            category_to_indices,
            groups,
            samples_per_group=6,
            repeats=3,
            seed=17,
        )

        first_flat = [index for repeat in plan["first"] for index in repeat]
        self.assertEqual(len(first_flat), len(set(first_flat)))
        for repeat in plan["first"]:
            a_count = sum(index < 20 for index in repeat)
            b_count = sum(20 <= index < 40 for index in repeat)
            self.assertEqual((a_count, b_count), (3, 3))

    def test_stratified_timestep_plan_covers_uniform_bin_midpoints(self):
        plan = stratified_timestep_plan(8, repeat_index=2, seed=5)
        expected = (np.arange(8, dtype=np.float64) + 0.5) / 8
        np.testing.assert_allclose(np.sort(plan), expected)
        self.assertTrue(np.all((plan >= 0) & (plan < 1)))

    def test_streamed_component_cosines_match_dense_reference(self):
        lora = torch.nn.Parameter(torch.zeros(2), requires_grad=True)
        geometry = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        named_parameters = [
            ("transformer.block.lora_A.weight", lora),
            ("mask_geometry_tokenizer.proj.weight", geometry),
        ]
        segments, total_numel = build_parameter_layout(named_parameters)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = OrderedDict()
            gradients = {
                "accessory": ([1.0, 0.0], [1.0]),
                "garment": ([0.0, 1.0], [-1.0]),
            }
            for group, (lora_gradient, geometry_gradient) in gradients.items():
                lora.grad = torch.tensor(lora_gradient)
                geometry.grad = torch.tensor(geometry_gradient)
                path = root / f"{group}.mmap"
                write_gradient_memmap(named_parameters, path, total_numel=total_numel)
                paths[group] = path

            group_names, grams = compute_component_grams(
                paths,
                segments,
                total_numel=total_numel,
                workspace_mb=1,
            )

        self.assertEqual(group_names, ["accessory", "garment"])
        all_cosine, all_norms = cosine_from_gram(grams["all_trainable"])
        lora_cosine, _ = cosine_from_gram(grams["lora"])
        geometry_cosine, _ = cosine_from_gram(grams["mask_geometry_tokenizer"])
        self.assertAlmostEqual(all_cosine[0, 1], -0.5, places=7)
        self.assertAlmostEqual(lora_cosine[0, 1], 0.0, places=7)
        self.assertAlmostEqual(geometry_cosine[0, 1], -1.0, places=7)
        np.testing.assert_allclose(all_norms, [math.sqrt(2), math.sqrt(2)])

    def test_pairwise_summary_marks_consistently_negative_pair(self):
        matrices = []
        for value in (-0.4, -0.5, -0.6, -0.45, -0.55):
            matrices.append(np.asarray([[1.0, value], [value, 1.0]]))
        records = summarize_pairwise_cosines(
            {"all_trainable": matrices},
            ["accessory", "garment"],
            low_similarity_threshold=0.1,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["interpretation"], "robust_negative")
        self.assertEqual(records[0]["negative_fraction"], 1.0)
        self.assertLess(records[0]["ci95_high"], 0.0)

    def test_output_writer_generates_primary_tables_and_figures(self):
        matrices = [
            np.asarray([[1.0, -0.2], [-0.2, 1.0]]),
            np.asarray([[1.0, -0.4], [-0.4, 1.0]]),
        ]
        norms = [np.asarray([2.0, 3.0]), np.asarray([2.5, 3.5])]
        records = summarize_pairwise_cosines(
            {"all_trainable": matrices},
            ["accessory", "garment"],
            low_similarity_threshold=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            write_analysis_outputs(
                output,
                group_names=["accessory", "garment"],
                repeat_cosines={"all_trainable": matrices},
                repeat_norms={"all_trainable": norms},
                pairwise_records=records,
                low_similarity_threshold=0.1,
            )
            self.assertTrue((output / "pairwise_cosine_summary.csv").is_file())
            self.assertTrue(
                (output / "category_gradient_cosine_heatmap_all_trainable.png").is_file()
            )
            self.assertTrue((output / "supplementary_summary.md").is_file())

    def test_checkpoint_directory_prefers_peft_model_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.safetensors").touch()
            (root / "pytorch_lora_weights.safetensors").touch()
            weight, checkpoint_dir = resolve_checkpoint_weight(root)
            self.assertEqual(weight.name, "model.safetensors")
            self.assertEqual(checkpoint_dir, root.resolve())

    def test_saved_training_config_with_computed_fields_is_reloadable(self):
        saved = Config().model_dump(mode="json", exclude_none=True)
        self.assertIn("trainer_type", saved)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_config.yaml"
            path.write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
            reloaded = load_analysis_config(path, clear_predict_devices=True)
        self.assertEqual(reloaded.trainer, Config().trainer)

    def test_legacy_run_level_training_config_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "v1"
            checkpoint_dir = run_dir / "checkpoint-last"
            checkpoint_dir.mkdir(parents=True)
            config_path = run_dir / "train_config.yaml"
            config_path.touch()
            self.assertEqual(resolve_config_path(None, checkpoint_dir), config_path)


if __name__ == "__main__":
    unittest.main()

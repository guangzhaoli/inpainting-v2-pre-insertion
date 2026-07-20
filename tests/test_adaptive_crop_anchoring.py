import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from infer import save_latent_anchoring_masks
from infer_testset import run_sample
from src.data.config import (
    AdaptiveCropConfig,
    Config,
    ImageProcessorConfig,
    ImageProcessorInitArgs,
    LatentAnchoringConfig,
    ModelConfig,
    PredictConfig,
)
from src.utils.adaptive_crop import build_adaptive_crop_plan, restore_pil_output
from src.utils.adaptive_crop_anchoring import (
    build_crop_edge_alpha,
    ensure_crop_plan_for_latent_anchoring,
    estimate_model_geometry,
    evaluate_crop_candidate,
    map_model_mask_to_original,
    prepare_adaptive_crop_for_inference,
    require_matching_mask_thresholds,
)
from src.utils.latent_anchoring import latent_anchoring_metadata


def _resize_processor(target_size=(64, 64)) -> ImageProcessorConfig:
    # target_size is [H, W] in ImageProcessorInitArgs.
    return ImageProcessorConfig(
        init_args=ImageProcessorInitArgs(
            process_type="resize",
            target_size=list(target_size),
        )
    )


def _fixed_pixels_processor() -> ImageProcessorConfig:
    return ImageProcessorConfig(
        init_args=ImageProcessorInitArgs(
            process_type="fixed_pixels",
            target_pixels=1024 * 1024,
        )
    )


class AdaptiveCropAnchoringHelperTest(unittest.TestCase):
    def test_require_matching_thresholds(self):
        crop = AdaptiveCropConfig(mask_threshold=0.5)
        anchor = LatentAnchoringConfig(mode="core", mask_threshold=0.5)
        self.assertEqual(require_matching_mask_thresholds(crop, anchor), 0.5)
        with self.assertRaisesRegex(ValueError, "mask_threshold"):
            require_matching_mask_thresholds(
                AdaptiveCropConfig(mask_threshold=0.4),
                LatentAnchoringConfig(mode="core", mask_threshold=0.5),
            )

    def test_estimate_model_geometry_applies_16_alignment_and_controls_fallback(self):
        # YAML 65x97 becomes processor 64x96 after make_divisible.
        processor = ImageProcessorConfig(
            init_args=ImageProcessorInitArgs(
                process_type="resize",
                target_size=[65, 97],
            )
        )
        size = estimate_model_geometry((200, 150), processor)
        self.assertEqual(size, (96, 64))

        fallback = ImageProcessorConfig(
            init_args=ImageProcessorInitArgs(
                process_type="resize",
                target_size=None,
                controls_size=[[65, 97]],
            )
        )
        size_fb = estimate_model_geometry((200, 150), fallback)
        self.assertEqual(size_fb, (96, 64))

    def test_map_model_mask_to_original_pastes_into_crop_box(self):
        plan = build_adaptive_crop_plan(
            (100, 80),
            self._center_mask(100, 80),
            AdaptiveCropConfig(
                predict_enabled=True,
                trigger_bbox_area_ratio=0.10,
                min_crop_area_ratio=0.25,
            ),
        )
        self.assertTrue(plan.applied)
        x1, y1, x2, y2 = plan.crop_box
        local = Image.new("L", (32, 32), color=255)
        mapped = map_model_mask_to_original(
            local,
            original_size=(100, 80),
            crop_plan=plan,
            resample=Image.Resampling.NEAREST,
        )
        array = np.asarray(mapped)
        self.assertEqual(array.shape, (80, 100))
        self.assertEqual(int(array[0, 0]), 0)
        self.assertTrue(np.all(array[y1:y2, x1:x2] == 255))
        outside = np.ones_like(array, dtype=bool)
        outside[y1:y2, x1:x2] = False
        self.assertTrue(np.all(array[outside] == 0))

    def test_map_model_mask_accepts_unit_interval_float_ndarray(self):
        mapped = map_model_mask_to_original(
            np.full((2, 2), 0.5, dtype=np.float32),
            original_size=(4, 4),
            crop_plan=None,
            resample=Image.Resampling.NEAREST,
        )
        array = np.asarray(mapped)
        self.assertTrue(np.all(array == 128) or np.all(np.isclose(array, 128, atol=1)))

    def test_crop_edge_alpha_feathers_only_interior_edges(self):
        plan = build_adaptive_crop_plan(
            (100, 80),
            self._corner_mask(100, 80),
            AdaptiveCropConfig(
                predict_enabled=True,
                trigger_bbox_area_ratio=0.10,
                min_crop_area_ratio=0.25,
            ),
        )
        self.assertTrue(plan.applied)
        self.assertEqual(plan.crop_box[0], 0)
        self.assertEqual(plan.crop_box[1], 0)
        alpha = build_crop_edge_alpha(plan, feather_width_px=4)
        x1, y1, x2, y2 = plan.crop_box
        self.assertAlmostEqual(float(alpha[y1, x1]), 1.0, places=5)
        if x2 < 100:
            self.assertLess(float(alpha[y1 + 1, x2 - 1]), 1.0)

    def test_realized_support_not_in_feather_after_planning(self):
        # Review-style geometry: horizontal mask near bottom + expanded mode.
        height, width = 1191, 1632
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[1122:1140, 487:662] = 255
        adaptive = AdaptiveCropConfig(
            predict_enabled=True,
            trigger_bbox_area_ratio=0.10,
            min_crop_area_ratio=0.25,
            context_margin_ratio=0.20,
            mask_threshold=0.5,
        )
        processor = _resize_processor((64, 64))
        anchoring = LatentAnchoringConfig(
            mode="expanded",
            mask_threshold=0.5,
            interaction_dilation_px=16,
            transition_width_px=32,
        )
        plan = build_adaptive_crop_plan((width, height), mask, adaptive)
        adjusted, meta = ensure_crop_plan_for_latent_anchoring(
            plan,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
            feather_width_px=8,
            decoder_guard_px=2,
        )
        self.assertEqual(meta["safety"], "packed_support_lanczos_proxy")
        self.assertEqual(meta["support_overlap_pixels"], 0)
        recheck = evaluate_crop_candidate(
            adjusted,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
            feather_width_px=8,
            decoder_guard_px=2,
        )
        self.assertTrue(recheck["ok"])
        self.assertEqual(recheck["support_overlap_pixels"], 0)
        self.assertGreaterEqual(recheck["processed_core_pixels"], 1)

    def test_thin_core_does_not_silently_empty_after_expansion(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0, 24] = 255
        adaptive = AdaptiveCropConfig(
            predict_enabled=True,
            trigger_bbox_area_ratio=0.10,
            min_crop_area_ratio=0.25,
            context_margin_ratio=0.20,
            min_crop_side=16,
            mask_threshold=0.5,
        )
        processor = _resize_processor((64, 64))
        anchoring = LatentAnchoringConfig(
            mode="expanded_soft",
            mask_threshold=0.5,
            interaction_dilation_px=16,
            transition_width_px=16,
        )
        plan = build_adaptive_crop_plan((100, 100), mask, adaptive)
        adjusted, meta = ensure_crop_plan_for_latent_anchoring(
            plan,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
        )
        self.assertGreaterEqual(meta["processed_core_pixels"], 1)
        self.assertEqual(meta["support_overlap_pixels"], 0)
        recheck = evaluate_crop_candidate(
            adjusted,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
        )
        self.assertTrue(recheck["core_survived"])
        self.assertTrue(recheck["ok"])

    def test_per_edge_need_does_not_expand_orthogonal_edges(self):
        # Single-pixel top mask: overlap only on left/right feather of a tight crop.
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[0, 24] = 255
        adaptive = AdaptiveCropConfig(
            predict_enabled=True,
            trigger_bbox_area_ratio=0.1,
            min_crop_area_ratio=0.25,
            context_margin_ratio=0.2,
            min_crop_side=16,
            mask_threshold=0.5,
        )
        processor = _resize_processor((64, 64))
        anchoring = LatentAnchoringConfig(
            mode="expanded",
            mask_threshold=0.5,
            interaction_dilation_px=16,
            transition_width_px=32,
        )
        plan = build_adaptive_crop_plan((80, 80), mask, adaptive)
        evaluation = evaluate_crop_candidate(
            plan,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
        )
        need_left, need_top, need_right, need_bottom = evaluation["need"]
        # Top edge coincides with image border → no top expand; bottom should stay 0.
        self.assertEqual(need_top, 0)
        self.assertEqual(need_bottom, 0)
        self.assertTrue(need_left > 0 or need_right > 0 or evaluation["ok"])

        adjusted, meta = ensure_crop_plan_for_latent_anchoring(
            plan,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
        )
        self.assertTrue(adjusted.applied)
        self.assertGreaterEqual(meta["processed_core_pixels"], 1)
        self.assertEqual(meta["support_overlap_pixels"], 0)
        # Must not explode to full frame for this solvable case.
        self.assertNotEqual(adjusted.crop_box, (0, 0, 80, 80))

    def test_stale_last_eval_does_not_reject_full_frame_fallback(self):
        # With a tiny search budget the solver may reach full-frame on the last
        # expand; that full-frame plan must still be evaluated (not stale-rejected).
        mask = np.zeros((717, 383), dtype=np.uint8)
        mask[166:185, 244:297] = 255
        adaptive = AdaptiveCropConfig(
            predict_enabled=True,
            trigger_bbox_area_ratio=0.5,
            min_crop_area_ratio=0.05,
            context_margin_ratio=0.0,
            min_crop_side=16,
            mask_threshold=0.5,
        )
        processor = _resize_processor((64, 64))
        anchoring = LatentAnchoringConfig(
            mode="expanded",
            mask_threshold=0.5,
            interaction_dilation_px=16,
            transition_width_px=8,
        )
        plan = build_adaptive_crop_plan((383, 717), mask, adaptive)
        adjusted, meta = ensure_crop_plan_for_latent_anchoring(
            plan,
            mask,
            adaptive_crop_config=adaptive,
            anchoring_config=anchoring,
            processor_config=processor,
            max_expand_iters=6,
        )
        self.assertEqual(meta["support_overlap_pixels"], 0)
        self.assertGreaterEqual(meta["processed_core_pixels"], 1)
        # last evaluation must match returned crop
        last = meta["evaluations"][-1]
        self.assertEqual(tuple(last["crop_box"]), tuple(adjusted.crop_box))
        self.assertTrue(last["ok"])

    def test_restore_with_feather_matches_source_outside_crop(self):
        source = Image.new("RGB", (100, 80), color=(255, 0, 0))
        mask = Image.fromarray(self._center_mask(100, 80), mode="L")
        plan = build_adaptive_crop_plan(
            source.size,
            mask,
            AdaptiveCropConfig(
                predict_enabled=True,
                trigger_bbox_area_ratio=0.10,
                min_crop_area_ratio=0.25,
            ),
        )
        local = Image.new("RGB", (32, 32), color=(0, 255, 0))
        restored, composite = restore_pil_output(
            source,
            mask,
            local,
            plan,
            blend_mode="strict_mask",
            edge_feather_px=8,
        )
        restored_arr = np.asarray(restored)
        source_arr = np.asarray(source)
        x1, y1, x2, y2 = plan.crop_box
        outside = np.ones((80, 100), dtype=bool)
        outside[y1:y2, x1:x2] = False
        self.assertTrue(np.all(restored_arr[outside] == source_arr[outside]))
        self.assertEqual(composite.size, source.size)

    def test_metadata_capability_field(self):
        metadata = latent_anchoring_metadata(LatentAnchoringConfig(mode="core"))
        self.assertEqual(metadata["adaptive_crop_capability"], "supported_via_crop_aware_wrapper")
        self.assertNotIn("adaptive_crop_compatible", metadata)

    @staticmethod
    def _center_mask(width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2 - 5 : height // 2 + 5, width // 2 - 5 : width // 2 + 5] = 255
        return mask

    @staticmethod
    def _corner_mask(width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[0:8, 0:8] = 255
        return mask


class FakePredictTrainer:
    def __init__(self, adaptive_crop: AdaptiveCropConfig, anchoring_mode: str = "legacy"):
        self.config = SimpleNamespace(
            model=ModelConfig(),
            predict=PredictConfig(latent_anchoring=LatentAnchoringConfig(mode=anchoring_mode)),
            data=SimpleNamespace(
                init_args=SimpleNamespace(
                    adaptive_crop=adaptive_crop,
                    processor=_resize_processor((64, 64)),
                ),
            ),
        )
        self.last_predict_kwargs = None
        self.anchor_masks = None

    def predict(self, **kwargs):
        self.last_predict_kwargs = kwargs
        mode = kwargs.get("latent_anchoring_mode", "legacy")
        out = Image.new("RGB", (64, 64), color=(0, 255, 0))
        if mode != "legacy":
            core = torch.zeros(64, 64)
            core[20:44, 20:44] = 1.0
            free = core.clone()
            edit = core.clone()
            if mode == "expanded":
                free[16:48, 16:48] = 1.0
                edit = free.clone()
            elif mode == "expanded_soft":
                free[16:48, 16:48] = 1.0
                edit = free.clone()
                edit[12:16, 16:48] = 0.5
                edit[48:52, 16:48] = 0.5
            self.anchor_masks = {
                "mode": mode,
                "core": core,
                "free": free,
                "edit_weight": edit,
            }
        return [out]

    def get_last_latent_anchoring_masks(self):
        return self.anchor_masks


class InferAdaptiveCropLatentAnchoringTest(unittest.TestCase):
    def _write_sample(self, root: Path, size=(100, 80), mask_box=(45, 35, 55, 45)):
        width, height = size
        source = Image.new("RGB", size, color=(255, 0, 0))
        mask_arr = np.zeros((height, width), dtype=np.uint8)
        x1, y1, x2, y2 = mask_box
        mask_arr[y1:y2, x1:x2] = 255
        source_mask = Image.fromarray(mask_arr, mode="L")
        reference = Image.new("RGB", (40, 40), color=(0, 0, 255))
        reference_mask = Image.new("L", (40, 40), color=255)
        paths = {
            "source": root / "source.png",
            "source_mask": root / "source_mask.png",
            "ref_image": root / "ref_image.png",
            "ref_mask": root / "ref_mask.png",
        }
        source.save(paths["source"])
        source_mask.save(paths["source_mask"])
        reference.save(paths["ref_image"])
        reference_mask.save(paths["ref_mask"])
        return paths, source, source_mask

    def _args(self, root: Path):
        return SimpleNamespace(
            height=None,
            width=None,
            prompt="insert",
            negative_prompt=None,
            num_inference_steps=2,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            panel_size=64,
            config=root / "config.yaml",
            lora=root / "checkpoint",
        )

    def _assert_has_green_pixel(self, image_arr: np.ndarray):
        green = np.array([0, 255, 0], dtype=np.uint8)
        self.assertTrue(np.any(np.all(image_arr == green, axis=-1)))

    def test_applied_crop_plus_core_keeps_primary_raw_and_maps_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, source, _ = self._write_sample(root)
            trainer = FakePredictTrainer(
                AdaptiveCropConfig(
                    predict_enabled=True,
                    trigger_bbox_area_ratio=0.10,
                    min_crop_area_ratio=0.25,
                    context_margin_ratio=0.20,
                    mask_threshold=0.5,
                ),
                anchoring_mode="core",
            )
            sample_dir = root / "result"
            metadata, _ = run_sample(
                trainer,
                {"stem": "sample", **paths},
                sample_dir,
                self._args(root),
                sample_seed=3,
            )

            predict_source = trainer.last_predict_kwargs["image"][0]
            self.assertLess(predict_source.width * predict_source.height, source.width * source.height)
            self.assertEqual(trainer.last_predict_kwargs["latent_anchoring_mode"], "core")

            with Image.open(sample_dir / "output.png") as output:
                output_arr = np.asarray(output.convert("RGB"))
            self.assertTrue(np.all(output_arr[0, 0] == np.array([255, 0, 0])))
            self._assert_has_green_pixel(output_arr)

            with Image.open(sample_dir / "mask_core.png") as mask_core:
                mask_arr = np.asarray(mask_core.convert("L"))
            self.assertEqual(mask_arr.shape, (source.height, source.width))
            self.assertEqual(int(mask_arr[0, 0]), 0)
            self.assertGreater(int(mask_arr.max()), 0)
            self.assertEqual(metadata["latent_anchoring"]["variant"], "B")
            self.assertTrue(metadata["adaptive_crop"]["plan"]["applied"])
            self.assertIn("compatibility", metadata["adaptive_crop"])
            self.assertEqual(
                metadata["adaptive_crop"]["compatibility"].get("safety"),
                "packed_support_lanczos_proxy",
            )

            # Row / overlay must track primary, not hard-paste raw.
            with Image.open(sample_dir / "row.png") as row:
                self.assertGreater(row.size[0], 0)

    def test_applied_crop_plus_expanded_modes(self):
        for mode, variant in (("expanded", "C"), ("expanded_soft", "D")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                paths, source, _ = self._write_sample(root)
                trainer = FakePredictTrainer(
                    AdaptiveCropConfig(
                        predict_enabled=True,
                        trigger_bbox_area_ratio=0.10,
                        min_crop_area_ratio=0.25,
                        context_margin_ratio=0.20,
                        mask_threshold=0.5,
                    ),
                    anchoring_mode=mode,
                )
                sample_dir = root / "result"
                metadata, _ = run_sample(
                    trainer,
                    {"stem": "sample", **paths},
                    sample_dir,
                    self._args(root),
                    sample_seed=3,
                )
                with Image.open(sample_dir / "output.png") as output:
                    output_arr = np.asarray(output.convert("RGB"))
                plan = metadata["adaptive_crop"]["plan"]
                x1, y1, x2, y2 = plan["crop_box"]
                outside = np.ones(output_arr.shape[:2], dtype=bool)
                outside[y1:y2, x1:x2] = False
                if outside.any():
                    self.assertTrue(np.all(output_arr[outside] == np.array([255, 0, 0])))
                self._assert_has_green_pixel(output_arr)
                self.assertEqual(metadata["latent_anchoring"]["variant"], variant)
                self.assertEqual(metadata["adaptive_crop"]["compatibility"]["support_overlap_pixels"], 0)
                for name in ("mask_core.png", "mask_free.png", "edit_weight.png"):
                    self.assertTrue((sample_dir / name).is_file())

    def test_noop_crop_plus_core_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, source, _ = self._write_sample(
                root,
                size=(100, 80),
                mask_box=(10, 10, 90, 70),
            )
            trainer = FakePredictTrainer(
                AdaptiveCropConfig(
                    predict_enabled=True,
                    trigger_bbox_area_ratio=0.10,
                    min_crop_area_ratio=0.25,
                    mask_threshold=0.5,
                ),
                anchoring_mode="core",
            )
            sample_dir = root / "result"
            metadata, _ = run_sample(
                trainer,
                {"stem": "sample", **paths},
                sample_dir,
                self._args(root),
                sample_seed=5,
            )
            self.assertFalse(metadata["adaptive_crop"]["plan"]["applied"])
            self.assertEqual(trainer.last_predict_kwargs["image"][0].size, source.size)
            self.assertEqual(metadata["latent_anchoring"]["mode"], "core")
            for name in ("mask_core.png", "mask_free.png", "edit_weight.png", "output.png"):
                self.assertTrue((sample_dir / name).is_file())

    def test_threshold_mismatch_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths, _, _ = self._write_sample(root)
            trainer = FakePredictTrainer(
                AdaptiveCropConfig(
                    predict_enabled=True,
                    mask_threshold=0.4,
                    trigger_bbox_area_ratio=0.10,
                    min_crop_area_ratio=0.25,
                ),
                anchoring_mode="core",
            )
            trainer.config.predict.latent_anchoring.mask_threshold = 0.5
            with self.assertRaisesRegex(ValueError, "mask_threshold"):
                run_sample(
                    trainer,
                    {"stem": "sample", **paths},
                    root / "result",
                    self._args(root),
                    sample_seed=1,
                )

    def test_save_latent_anchoring_masks_uses_crop_mapping(self):
        trainer = FakePredictTrainer(AdaptiveCropConfig(predict_enabled=True), anchoring_mode="core")
        trainer.anchor_masks = {
            "mode": "core",
            "core": torch.ones(16, 16),
            "free": torch.ones(16, 16),
            "edit_weight": torch.ones(16, 16),
        }
        plan = build_adaptive_crop_plan(
            (40, 30),
            Image.fromarray(
                np.pad(np.ones((6, 6), dtype=np.uint8) * 255, ((12, 12), (17, 17))),
                mode="L",
            ),
            AdaptiveCropConfig(
                predict_enabled=True,
                trigger_bbox_area_ratio=0.10,
                min_crop_area_ratio=0.25,
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = {
                "mask_core": root / "mask_core.png",
                "mask_free": root / "mask_free.png",
                "edit_weight": root / "edit_weight.png",
            }
            saved = save_latent_anchoring_masks(
                trainer,
                paths,
                (40, 30),
                crop_plan=plan,
            )
            with Image.open(saved["mask_core"]) as image:
                array = np.asarray(image.convert("L"))
            self.assertEqual(array.shape, (30, 40))
            self.assertEqual(int(array[0, 0]), 0)
            x1, y1, x2, y2 = plan.crop_box
            self.assertGreater(int(array[y1:y2, x1:x2].max()), 0)


if __name__ == "__main__":
    unittest.main()

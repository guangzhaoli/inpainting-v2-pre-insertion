import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError

from src.data.config import DataConfig, DatasetInitArgs
from src.data.dataset import collate_fn
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.utils.adaptive_crop import build_adaptive_crop_plan


class InsertAnythingDatasetPromptTest(unittest.TestCase):
    def _build_category(self, root: Path) -> Path:
        category_dir = root / "object"
        for name in ("ref_image", "ref_mask", "tar_image", "tar_mask"):
            (category_dir / name).mkdir(parents=True, exist_ok=True)

        image = Image.new("RGB", (8, 8), color=(32, 64, 96))
        mask = Image.new("L", (8, 8), color=255)
        image.save(category_dir / "ref_image" / "sample.png")
        mask.save(category_dir / "ref_mask" / "sample.png")
        image.save(category_dir / "tar_image" / "sample.png")
        mask.save(category_dir / "tar_mask" / "sample.png")
        return category_dir

    def _build_adaptive_category(self, root: Path) -> np.ndarray:
        category_dir = root / "object"
        for name in ("ref_image", "ref_mask", "tar_image", "tar_mask"):
            (category_dir / name).mkdir(parents=True, exist_ok=True)

        height, width = 80, 100
        ref_image = np.full((height, width, 3), (32, 64, 96), dtype=np.uint8)
        ref_mask = np.zeros((height, width), dtype=np.uint8)
        ref_mask[25:55, 35:65] = 255

        tar_image = np.zeros((height, width, 3), dtype=np.uint8)
        tar_image[..., 0] = np.arange(width, dtype=np.uint8)[None, :]
        tar_image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
        tar_image[..., 2] = 127
        tar_mask = np.zeros((height, width), dtype=np.uint8)
        tar_mask[35:45, 45:55] = 255

        Image.fromarray(ref_image).save(category_dir / "ref_image" / "sample.png")
        Image.fromarray(ref_mask).save(category_dir / "ref_mask" / "sample.png")
        Image.fromarray(tar_image).save(category_dir / "tar_image" / "sample.png")
        Image.fromarray(tar_mask).save(category_dir / "tar_mask" / "sample.png")
        return tar_mask

    @staticmethod
    def _dataset_config(root: Path, train_enabled: bool) -> DatasetInitArgs:
        return DatasetInitArgs(
            dataset_path=str(root),
            categories=["object"],
            require_prompt=False,
            processor={
                "class_path": "src.data.preprocess.ImageProcessor",
                "init_args": {
                    "process_type": "fixed_pixels",
                    "resize_mode": "bilinear",
                    "target_pixels": 256 * 256,
                    "controls_pixels": 256 * 256,
                },
            },
            adaptive_crop={
                "train_enabled": train_enabled,
                "predict_enabled": False,
            },
        )

    @staticmethod
    def _record_preprocessor_input(dataset: InsertAnythingDataset) -> dict:
        captured = {}
        real_preprocess = dataset.preprocessor.preprocess

        def recording_preprocess(data, *args, **kwargs):
            captured["image_shape"] = tuple(np.asarray(data["image"]).shape)
            captured["mask_shape"] = tuple(np.asarray(data["mask"]).shape)
            return real_preprocess(data, *args, **kwargs)

        dataset.preprocessor.preprocess = recording_preprocess
        return captured

    def test_missing_prompt_is_empty_when_prompt_not_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = self._build_category(Path(tmpdir))
            dataset = object.__new__(InsertAnythingDataset)
            dataset.samples = []
            dataset.require_prompt = False

            dataset._scan_category(category_dir, "object")

            self.assertEqual(len(dataset.samples), 1)
            self.assertEqual(dataset.samples[0]["prompt"], "")

    def test_missing_prompt_raises_when_prompt_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category_dir = self._build_category(Path(tmpdir))
            dataset = object.__new__(InsertAnythingDataset)
            dataset.samples = []
            dataset.require_prompt = True

            with self.assertRaises(FileNotFoundError):
                dataset._scan_category(category_dir, "object")

    def test_training_adaptive_crop_runs_before_fixed_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tar_mask = self._build_adaptive_category(root)
            config = self._dataset_config(root, train_enabled=True)
            expected_plan = build_adaptive_crop_plan(
                (tar_mask.shape[1], tar_mask.shape[0]),
                tar_mask,
                config.adaptive_crop,
            )

            dataset = InsertAnythingDataset(config)
            captured = self._record_preprocessor_input(dataset)
            sample = dataset[0]

            x1, y1, x2, y2 = expected_plan.crop_box
            self.assertTrue(expected_plan.applied)
            self.assertEqual(captured["image_shape"], (y2 - y1, x2 - x1, 3))
            self.assertEqual(captured["mask_shape"], (y2 - y1, x2 - x1))
            self.assertEqual(sample["tar_image"].shape[1:], sample["tar_mask"].shape)
            self.assertGreater(float(sample["tar_mask"].sum()), 0.0)

    def test_disabled_adaptive_crop_preserves_original_preprocessor_input(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tar_mask = self._build_adaptive_category(root)
            config = self._dataset_config(root, train_enabled=False)

            dataset = InsertAnythingDataset(config)
            captured = self._record_preprocessor_input(dataset)
            sample = dataset[0]

            self.assertEqual(captured["image_shape"], (80, 100, 3))
            self.assertEqual(captured["mask_shape"], (80, 100))
            self.assertEqual(sample["tar_image"].shape[1:], sample["tar_mask"].shape)
            self.assertGreater(float(sample["tar_mask"].sum()), 0.0)
            self.assertEqual(int(tar_mask.max()), 255)

    def test_sample_metadata_is_stable_and_survives_main_collate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_category(root)
            dataset = InsertAnythingDataset(self._dataset_config(root, train_enabled=False))

            sample = dataset[0]
            metadata = sample["sample_metadata"]
            self.assertEqual(metadata["dataset_index"], 0)
            self.assertEqual(metadata["sample_id"], "object/sample")
            self.assertEqual(metadata["category"], "object")
            self.assertEqual(metadata["stem"], "sample")
            for key in (
                "ref_image_path",
                "ref_mask_path",
                "tar_image_path",
                "tar_mask_path",
            ):
                self.assertTrue(Path(metadata[key]).is_absolute())
                self.assertTrue(Path(metadata[key]).is_file())

            batch = collate_fn([sample, sample])
            self.assertEqual(batch["sample_metadata"]["dataset_index"], [0, 0])
            self.assertEqual(
                batch["sample_metadata"]["sample_id"],
                ["object/sample", "object/sample"],
            )
            self.assertEqual(
                batch["sample_metadata"]["tar_mask_path"],
                [metadata["tar_mask_path"], metadata["tar_mask_path"]],
            )

    def test_loading_error_keeps_root_exception_and_adds_sample_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._build_category(root)
            dataset = InsertAnythingDataset(self._dataset_config(root, train_enabled=False))
            missing_path = root / "object" / "tar_image" / "missing.png"
            dataset.samples[0]["tar_image"] = str(missing_path)

            with self.assertRaisesRegex(ValueError, "Failed to read image") as caught:
                dataset[0]

            notes = "\n".join(getattr(caught.exception, "__notes__", []))
            self.assertIn("dataset_index=0", notes)
            self.assertIn("sample_id=object/sample", notes)
            self.assertIn(f"tar_image_path={missing_path}", notes)

    def test_adaptive_crop_rejects_noninvertible_processor_geometry(self):
        for enabled_field in ("train_enabled", "predict_enabled"):
            for process_type in ("center_crop", "center_padding", "right_padding"):
                with self.subTest(enabled_field=enabled_field, process_type=process_type):
                    adaptive_crop = {enabled_field: True}
                    with self.assertRaisesRegex(ValidationError, "cannot be inverted"):
                        DatasetInitArgs(
                            processor={
                                "init_args": {
                                    "process_type": process_type,
                                    "target_size": [64, 64],
                                }
                            },
                            adaptive_crop=adaptive_crop,
                        )

    def test_adaptive_crop_accepts_full_frame_resize_processors(self):
        for process_type, extra_args in (
            ("fixed_pixels", {"target_pixels": 256 * 256}),
            ("resize", {"target_size": [64, 64]}),
        ):
            with self.subTest(process_type=process_type):
                config = DatasetInitArgs(
                    processor={
                        "init_args": {
                            "process_type": process_type,
                            **extra_args,
                        }
                    },
                    adaptive_crop={"train_enabled": True, "predict_enabled": True},
                )
                self.assertEqual(config.processor.init_args.process_type, process_type)

    def test_adaptive_fixed_pixels_rejects_unsafe_multi_sample_batches(self):
        with self.assertRaisesRegex(ValidationError, "valid-region loss mask"):
            DataConfig(
                batch_size=2,
                init_args={
                    "processor": {
                        "init_args": {
                            "process_type": "fixed_pixels",
                            "target_pixels": 256 * 256,
                        }
                    },
                    "adaptive_crop": {"train_enabled": True},
                },
            )

        safe_resize = DataConfig(
            batch_size=2,
            init_args={
                "processor": {
                    "init_args": {
                        "process_type": "resize",
                        "target_size": [64, 64],
                    }
                },
                "adaptive_crop": {"train_enabled": True},
            },
        )
        self.assertEqual(safe_resize.batch_size, 2)

    def test_center_crop_remains_valid_when_adaptive_crop_is_disabled(self):
        config = DatasetInitArgs(
            processor={
                "init_args": {
                    "process_type": "center_crop",
                    "target_size": [64, 64],
                }
            }
        )
        self.assertFalse(config.adaptive_crop.train_enabled)
        self.assertFalse(config.adaptive_crop.predict_enabled)


if __name__ == "__main__":
    unittest.main()

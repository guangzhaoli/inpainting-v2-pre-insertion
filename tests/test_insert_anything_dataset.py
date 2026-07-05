import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data.insert_anything_dataset import InsertAnythingDataset


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


if __name__ == "__main__":
    unittest.main()

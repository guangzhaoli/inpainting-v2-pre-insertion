"""Tests for DreamBooth FG/BG -> canonical insertion testset materialization."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_dreambooth_eval_dataset import (
    build_pairs,
    discover_backgrounds,
    discover_foregrounds,
    materialize,
)


def _write_rgba(path: Path, rgb=(200, 40, 40), size=(32, 40), box=(8, 8, 24, 32)) -> None:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            img.putpixel((x, y), (*rgb, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_bg_pair(bg_dir: Path, stem: str, size=(48, 48)) -> None:
    bg_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (30, 80, 160)).save(bg_dir / f"{stem}_GT.png")
    mask = Image.new("L", size, 0)
    for x in range(10, 30):
        for y in range(10, 30):
            mask.putpixel((x, y), 255)
    mask.save(bg_dir / f"{stem}_mask.png")


class PrepareDreamBoothEvalDatasetTest(unittest.TestCase):
    def test_flat_fg_and_gt_mask_bg_cartesian_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fg_dir = root / "FG"
            bg_dir = root / "BG"
            out = root / "out"
            _write_rgba(fg_dir / "00.png")
            _write_rgba(fg_dir / "01.png", rgb=(40, 200, 40))
            _write_bg_pair(bg_dir, "sceneA")
            _write_bg_pair(bg_dir, "sceneB")

            fgs = discover_foregrounds(fg_dir)
            bgs = discover_backgrounds(bg_dir)
            self.assertEqual([f.fg_id for f in fgs], ["00", "01"])
            self.assertEqual([b.bg_id for b in bgs], ["sceneA", "sceneB"])

            pairs = build_pairs(fgs, bgs)
            self.assertEqual(len(pairs), 4)
            summary = materialize(pairs, out)
            self.assertEqual(summary["n_cases"], 4)

            man = out / "manifests" / "dreambooth_eval.jsonl"
            rows = [json.loads(line) for line in man.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 4)
            for row in rows:
                for key in ("src_image", "src_mask", "tar_image", "tar_mask"):
                    self.assertTrue((out / row[key]).is_file(), msg=row[key])
                with Image.open(out / row["src_image"]) as src, Image.open(
                    out / row["src_mask"]
                ) as sm:
                    self.assertEqual(src.mode, "RGB")
                    self.assertEqual(sm.mode, "L")
                    self.assertGreater(sum(1 for p in sm.getdata() if p > 128), 0)

    def test_subfolder_fg_layout_and_max_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fg_dir = root / "AnyDoor_DreamBooth"
            bg_dir = root / "v1_800"
            _write_rgba(fg_dir / "dog" / "view0.png")
            _write_rgba(fg_dir / "cat" / "a.png", rgb=(10, 10, 200))
            _write_bg_pair(bg_dir, "0001")
            _write_bg_pair(bg_dir, "0002")
            _write_bg_pair(bg_dir, "0003")

            fgs = discover_foregrounds(fg_dir)
            bgs = discover_backgrounds(bg_dir)
            pairs = build_pairs(fgs, bgs, max_pairs_per_fg=2)
            self.assertEqual(len(pairs), 4)  # 2 fg * 2 bg
            self.assertTrue(all(p.case_id.startswith(("dog__", "cat__")) for p in pairs))

    def test_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fg_dir = root / "FG"
            bg_dir = root / "BG"
            out = root / "out"
            _write_rgba(fg_dir / "00.png")
            _write_bg_pair(bg_dir, "sceneA")
            pairs = build_pairs(discover_foregrounds(fg_dir), discover_backgrounds(bg_dir))
            materialize(pairs, out)
            with self.assertRaises(FileExistsError):
                materialize(pairs, out, overwrite=False)
            materialize(pairs, out, overwrite=True)


if __name__ == "__main__":
    unittest.main()

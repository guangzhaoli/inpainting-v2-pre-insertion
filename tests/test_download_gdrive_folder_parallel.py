import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts.download_gdrive_folder_parallel import (
    canonical_digest,
    download_one,
    file_sha256,
    validate_download,
)


class DownloadGdriveFolderParallelTest(unittest.TestCase):
    def test_part_file_is_validated_using_final_image_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            part = Path(tmpdir) / "sample.png.part"
            Image.new("RGB", (16, 16), color=(30, 90, 150)).save(part, format="PNG")

            self.assertTrue(validate_download(part, expected_suffix=".png"))

    def test_html_error_page_cannot_pass_as_part_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            part = Path(tmpdir) / "sample.png.part"
            part.write_text("<html>quota exceeded</html>", encoding="utf-8")

            self.assertFalse(validate_download(part, expected_suffix=".png"))

    def test_tree_digest_is_order_independent(self):
        left = canonical_digest([("b.png", "id-b"), ("a.png", "id-a")])
        right = canonical_digest([("a.png", "id-a"), ("b.png", "id-b")])

        self.assertEqual(left, right)

    def test_existing_file_is_skipped_only_with_matching_id_and_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / "sample.png"
            Image.new("RGB", (16, 16), color=(30, 90, 150)).save(destination)
            item = SimpleNamespace(
                id="remote-id",
                path="object/ref_image/sample.png",
                local_path=str(destination),
            )
            previous = {
                "id": "remote-id",
                "remote_path": "object/ref_image/sample.png",
                "sha256": file_sha256(destination),
            }

            status, path = download_one(item, attempts=1, previous_row=previous)

            self.assertEqual(status, "skipped")
            self.assertEqual(path, str(destination))


if __name__ == "__main__":
    unittest.main()

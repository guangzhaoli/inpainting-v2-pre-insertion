import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts.check_paper_dataset_leakage import (
    BKTree,
    file_sha256,
    image_fingerprints,
    main,
    scan,
)


def finalize_release_fixture(dataset_root: Path) -> Path:
    """Create the minimal authenticated release envelope used by scanner tests."""

    manifest_dir = dataset_root / "manifests"
    primary_names = ("paperinsert_900.jsonl", "anyinsertion_official.jsonl")
    primary_paths = [manifest_dir / name for name in primary_names if (manifest_dir / name).is_file()]
    if len(primary_paths) != 1:
        raise AssertionError(f"fixture needs one primary manifest, found {primary_paths}")
    primary = primary_paths[0]
    records = [
        json.loads(line)
        for line in primary.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical_paths: set[str] = set()
    normalized_records = []
    for record in records:
        normalized = dict(record)
        case_id = str(normalized.get("case_id", "case"))
        major = str(normalized.get("major", "object"))
        for role in ("src_image", "tar_image"):
            relative = normalized.get(role, f"{major}/{role}/{case_id}.png")
            normalized[role] = relative
            canonical_paths.add(relative)
        for role in ("src_mask", "tar_mask"):
            relative = f"{major}/{role}/{case_id}.png"
            normalized[role] = relative
            mask = dataset_root / relative
            mask.parent.mkdir(parents=True, exist_ok=True)
            if not mask.exists():
                Image.new("L", (32, 32), color=255).save(mask)
            canonical_paths.add(relative)
        normalized_records.append(normalized)
    primary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in normalized_records),
        encoding="utf-8",
    )

    if primary.name == "paperinsert_900.jsonl":
        authoritative = {
            "manifests/paperinsert_900.jsonl": primary.read_text(encoding="utf-8"),
            "manifests/paperinsert_900.csv": "case_id,major\n",
            "manifests/object.jsonl": primary.read_text(encoding="utf-8"),
            "manifests/person.jsonl": "",
            "manifests/garment.jsonl": "",
            "reports/sources.json": json.dumps(
                {"schema_version": 3, "dataset_name": "PaperInsert-900"}
            )
            + "\n",
        }
    else:
        authoritative = {
            "manifests/anyinsertion_official.jsonl": primary.read_text(encoding="utf-8"),
            "manifests/anyinsertion_clean67_for_current_model.jsonl": primary.read_text(
                encoding="utf-8"
            ),
            "manifests/anyinsertion_clean67_case_ids.txt": "".join(
                f"{record['case_id']}\n" for record in normalized_records
            ),
            "manifests/anyinsertion_current_model_overlap_excluded33.json": json.dumps(
                {"schema_version": 3, "excluded_case_ids": []}
            )
            + "\n",
            "reports/conversion.json": json.dumps({"schema_version": 3}) + "\n",
        }
    for relative, content in authoritative.items():
        path = dataset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    checksum_paths = canonical_paths.union(authoritative)
    checksums = "".join(
        f"{file_sha256(dataset_root / relative)}  {relative}\n"
        for relative in sorted(checksum_paths)
    )
    (dataset_root / "CHECKSUMS.sha256").write_text(checksums, encoding="utf-8")
    return primary


def make_training_case(
    train_root: Path,
    ref_image: Image.Image,
    tar_image: Image.Image,
    *,
    category: str = "cat",
    stem: str = "sample",
    image_suffix: str = ".png",
) -> tuple[Path, Path]:
    category_root = train_root / category
    ref_path = category_root / "ref_image" / f"{stem}{image_suffix}"
    tar_path = category_root / "tar_image" / f"{stem}{image_suffix}"
    ref_mask = category_root / "ref_mask" / f"{stem}.png"
    tar_mask = category_root / "tar_mask" / f"{stem}.png"
    for path in (ref_path, tar_path, ref_mask, tar_mask):
        path.parent.mkdir(parents=True, exist_ok=True)
    ref_image.save(ref_path)
    tar_image.save(tar_path)
    Image.new("L", ref_image.size, color=255).save(ref_mask)
    Image.new("L", tar_image.size, color=255).save(tar_mask)
    return ref_path, tar_path


def make_benchmark_case(dataset_root: Path, *, protocol: str | None = "paired_reconstruction"):
    for role, color in (("src_image", (10, 20, 30)), ("tar_image", (30, 20, 10))):
        path = dataset_root / "object" / role / "case.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=color).save(path)
    record = {
        "case_id": "case",
        "major": "object",
        "minor": "animal",
        "source_dataset": "fixture",
        "src_image": "object/src_image/case.png",
        "tar_image": "object/tar_image/case.png",
    }
    if protocol is not None:
        record["protocol"] = protocol
    manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    finalize_release_fixture(dataset_root)
    return record, manifest


class PaperDatasetLeakageTest(unittest.TestCase):
    def test_bk_tree_returns_only_hashes_within_hamming_radius(self):
        tree = BKTree()
        tree.add(0b0000, "zero")
        tree.add(0b0011, "two_bits")
        tree.add(0b1111, "four_bits")

        matches = sorted(tree.query(0b0001, radius=1))

        self.assertEqual(matches, [(1, "two_bits"), (1, "zero")])

    def test_scan_reports_decoded_exact_training_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            train_root = root / "train"
            source = Image.new("RGB", (96, 96), color=(240, 240, 240))
            draw = ImageDraw.Draw(source)
            draw.rectangle((20, 20, 75, 75), fill=(20, 90, 180))
            target = Image.new("RGB", (96, 96), color=(40, 180, 80))

            paths = {}
            for role, image in (("src_image", source), ("tar_image", target)):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths[role] = str(path.relative_to(dataset_root))
            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "animal",
                        "source_dataset": "fixture",
                        "protocol": "paired_reconstruction",
                        "src_image_id": "src",
                        "tar_image_id": "tar",
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            training_image, _ = make_training_case(
                train_root,
                source,
                Image.new("RGB", (96, 96), color=(1, 2, 3)),
                stem="same",
            )
            finalize_release_fixture(dataset_root)

            report = scan(
                dataset_root,
                train_root,
                root / "cache.jsonl",
                root / "report.json",
                phash_radius=4,
                workers=2,
            )

            self.assertEqual(report["training_image_count"], 2)
            self.assertEqual(report["workers"], 2)
            self.assertEqual(report["candidate_image_count"], 2)
            self.assertEqual(report["exact_match_count"], 1)
            self.assertEqual(report["exact_case_ids"], ["case"])
            self.assertEqual(report["exact_case_count"], 1)
            self.assertFalse(report["leakage_free_exact"])
            self.assertTrue((root / "report.json").is_file())
            self.assertTrue(report["release_integrity_verified"])
            self.assertEqual(report["release_type"], "paperinsert_900")
            self.assertEqual(report["checksum_entry_count"], 10)

    def test_cross_scan_matches_full_donor_when_canonical_crop_does_not(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            train_root = root / "train"
            crop = Image.new("RGB", (48, 48), color=(220, 30, 30))
            target = Image.new("RGB", (64, 64), color=(30, 220, 30))
            full_donor = Image.new("RGB", (96, 80), color=(30, 30, 220))
            ImageDraw.Draw(full_donor).ellipse((15, 10, 75, 70), fill=(240, 180, 20))

            paths = {}
            for role, image in (("src_image", crop), ("tar_image", target)):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths[role] = str(path.relative_to(dataset_root))

            training_image, _ = make_training_case(
                train_root,
                full_donor,
                Image.new("RGB", (96, 80), color=(1, 2, 3)),
                stem="full-donor",
            )
            full_fingerprints = image_fingerprints(training_image)

            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "animal",
                        "protocol": "cross_instance_replace",
                        "source_dataset": "fixture",
                        "src_image_id": "full-donor",
                        "tar_image_id": "tar",
                        "src_full_pixel_sha256": full_fingerprints["pixel_sha256"],
                        "src_full_phash": full_fingerprints["phash"],
                        "src_full_dhash": full_fingerprints["dhash"],
                        "src_full_width": full_fingerprints["width"],
                        "src_full_height": full_fingerprints["height"],
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            finalize_release_fixture(dataset_root)

            report = scan(
                dataset_root,
                train_root,
                root / "cache.jsonl",
                root / "report.json",
                phash_radius=4,
                workers=1,
            )

            self.assertEqual(report["candidate_image_count"], 3)
            self.assertEqual(report["exact_match_count"], 1)
            self.assertEqual(report["exact_case_ids"], ["case"])
            candidate = report["exact_matches"][0]["candidate"]
            self.assertEqual(candidate["role"], "src_full_image")
            self.assertIsNone(candidate["path"])
            self.assertEqual(
                candidate["fingerprint_origin"], "manifest_full_source_image"
            )
            self.assertTrue(
                any(
                    row["candidate"]["role"] == "src_full_image"
                    for row in report["near_matches"]
                ),
                "manifest full-donor fingerprints must also enter the pHash+dHash index",
            )

    def test_paired_scan_matches_optional_full_reference_when_crop_does_not(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            train_root = root / "train"
            crop = Image.new("RGB", (48, 48), color=(210, 40, 40))
            target = Image.new("RGB", (64, 64), color=(40, 210, 40))
            full_reference = Image.new("RGB", (96, 80), color=(40, 40, 210))
            ImageDraw.Draw(full_reference).rectangle(
                (12, 8, 78, 72), fill=(235, 175, 25)
            )

            paths = {}
            for role, image in (("src_image", crop), ("tar_image", target)):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths[role] = str(path.relative_to(dataset_root))

            training_image, _ = make_training_case(
                train_root,
                full_reference,
                Image.new("RGB", (96, 80), color=(1, 2, 3)),
                stem="full-reference",
            )
            full_fingerprints = image_fingerprints(training_image)

            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "animal",
                        "protocol": "paired_reconstruction",
                        "source_dataset": "fixture",
                        "src_image_id": "full-reference",
                        "tar_image_id": "tar",
                        "src_full_pixel_sha256": full_fingerprints["pixel_sha256"],
                        "src_full_phash": full_fingerprints["phash"],
                        "src_full_dhash": full_fingerprints["dhash"],
                        "src_full_width": full_fingerprints["width"],
                        "src_full_height": full_fingerprints["height"],
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            finalize_release_fixture(dataset_root)

            report = scan(
                dataset_root,
                train_root,
                root / "cache.jsonl",
                root / "report.json",
                phash_radius=4,
                workers=1,
            )

            self.assertEqual(report["candidate_image_count"], 3)
            self.assertEqual(report["exact_match_count"], 1)
            self.assertEqual(report["exact_case_ids"], ["case"])
            candidate = report["exact_matches"][0]["candidate"]
            self.assertEqual(candidate["role"], "src_full_image")
            self.assertIsNone(candidate["path"])
            self.assertEqual(
                candidate["fingerprint_origin"], "manifest_full_source_image"
            )

    def test_cross_scan_fails_closed_without_full_donor_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            train_root = root / "train"
            paths = {}
            for role, color in (
                ("src_image", (220, 30, 30)),
                ("tar_image", (30, 220, 30)),
            ):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 48), color=color).save(path)
                paths[role] = str(path.relative_to(dataset_root))
            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "animal",
                        "protocol": "cross_instance_replace",
                        "source_dataset": "fixture",
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            finalize_release_fixture(dataset_root)

            with self.assertRaisesRegex(
                ValueError, "missing required full donor fingerprint fields"
            ):
                scan(
                    dataset_root,
                    train_root,
                    root / "cache.jsonl",
                    root / "report.json",
                    phash_radius=4,
                    workers=1,
                )

    def test_scan_supports_normalized_anyinsertion_manifest_without_image_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            train_root = root / "train"
            source = Image.new("RGB", (64, 64), color=(20, 80, 160))
            target = Image.new("RGB", (64, 64), color=(160, 80, 20))
            paths = {}
            for role, image in (("src_image", source), ("tar_image", target)):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths[role] = str(path.relative_to(dataset_root))
            manifest = dataset_root / "manifests" / "anyinsertion_official.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "object",
                        "source_dataset": "anyinsertion",
                        "protocol": "official_anyinsertion",
                        "original_group": "object",
                        "original_stem": "7",
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            finalize_release_fixture(dataset_root)

            report = scan(
                dataset_root,
                train_root,
                root / "cache.jsonl",
                root / "report.json",
                phash_radius=4,
                workers=1,
                manifest_path=Path("manifests/anyinsertion_official.jsonl"),
            )

            self.assertEqual(report["candidate_image_count"], 2)
            self.assertEqual(report["exact_match_count"], 0)
            self.assertEqual(report["no_exact_match_case_ids"], ["case"])
            self.assertTrue(report["manifest"].endswith("anyinsertion_official.jsonl"))
            self.assertTrue(report["release_integrity_verified"])
            self.assertEqual(report["release_type"], "anyinsertion_official")
            self.assertEqual(report["checksum_entry_count"], 9)

    def test_explicit_manifest_must_remain_inside_dataset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            outside = root / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")
            train_root = root / "train"
            train_root.mkdir()

            with self.assertRaisesRegex(ValueError, "outside dataset root"):
                scan(
                    dataset_root,
                    train_root,
                    root / "cache.jsonl",
                    root / "report.json",
                    phash_radius=4,
                    workers=1,
                    manifest_path=outside,
                )

    def test_duplicate_manifest_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            for role, color in (
                ("src_image", (10, 20, 30)),
                ("tar_image", (30, 20, 10)),
            ):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), color=color).save(path)
            record = {
                "case_id": "case",
                "major": "object",
                "minor": "animal",
                "source_dataset": "fixture",
                "protocol": "paired_reconstruction",
                "src_image": "object/src_image/case.png",
                "tar_image": "object/tar_image/case.png",
            }
            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            finalize_release_fixture(dataset_root)
            train_root = root / "train"
            train_root.mkdir()

            with self.assertRaisesRegex(ValueError, "Duplicate benchmark case_id"):
                scan(
                    dataset_root,
                    train_root,
                    root / "cache.jsonl",
                    root / "report.json",
                    phash_radius=4,
                    workers=1,
                )

    def test_cache_recomputes_same_size_preserved_mtime_replacement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            source = Image.new("RGB", (40, 40), color=(20, 40, 220))
            target = Image.new("RGB", (40, 40), color=(30, 200, 40))
            paths = {}
            for role, image in (("src_image", source), ("tar_image", target)):
                path = dataset_root / "object" / role / "case.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                paths[role] = str(path.relative_to(dataset_root))
            manifest = dataset_root / "manifests" / "paperinsert_900.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "case_id": "case",
                        "major": "object",
                        "minor": "animal",
                        "source_dataset": "fixture",
                        "protocol": "paired_reconstruction",
                        **paths,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            training, _ = make_training_case(
                root / "train",
                Image.new("RGB", (40, 40), color=(220, 40, 20)),
                Image.new("RGB", (40, 40), color=(1, 2, 3)),
                image_suffix=".bmp",
            )
            finalize_release_fixture(dataset_root)
            original_stat = training.stat()
            cache = root / "cache.jsonl"

            first = scan(
                dataset_root,
                root / "train",
                cache,
                root / "first.json",
                phash_radius=0,
                workers=1,
            )
            source.save(training, format="BMP")
            self.assertEqual(training.stat().st_size, original_stat.st_size)
            os.utime(
                training,
                ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
            )
            second = scan(
                dataset_root,
                root / "train",
                cache,
                root / "second.json",
                phash_radius=0,
                workers=1,
            )

            self.assertEqual(first["training_fingerprints_computed"], 2)
            self.assertEqual(second["training_cache_hits"], 1)
            self.assertEqual(second["training_fingerprints_computed"], 1)
            self.assertEqual(second["exact_match_count"], 1)
            self.assertEqual(second["cache_schema_version"], 2)

    def test_cli_fails_when_training_decode_errors_make_scan_incomplete(self):
        result = {
            "exact_match_count": 0,
            "near_match_count": 0,
            "decode_errors": [{"path": "broken.png", "error": "decode failed"}],
        }
        with patch("scripts.check_paper_dataset_leakage.scan", return_value=result):
            exit_code = main(
                [
                    "--dataset-root",
                    "/tmp/dataset",
                    "--train-root",
                    "/tmp/train",
                ]
            )

        self.assertEqual(exit_code, 2)

    def test_missing_or_unknown_protocol_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            _, manifest = make_benchmark_case(dataset_root, protocol=None)
            train_root = root / "train"
            train_root.mkdir()

            with self.assertRaisesRegex(ValueError, "missing protocol"):
                scan(dataset_root, train_root, root / "cache", root / "report", 4)

            record = json.loads(manifest.read_text(encoding="utf-8"))
            record["protocol"] = "cross_instance_typo"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            finalize_release_fixture(dataset_root)
            with self.assertRaisesRegex(ValueError, "unsupported benchmark protocol"):
                scan(dataset_root, train_root, root / "cache", root / "report", 4)

    def test_empty_training_root_fails_and_mask_prompt_wrapper_is_scanned(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            train_root.mkdir()

            with self.assertRaisesRegex(ValueError, "No complete mask_prompt"):
                scan(dataset_root, train_root, root / "cache", root / "report", 4)

            make_training_case(
                train_root / "mask_prompt",
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            report = scan(
                dataset_root,
                train_root,
                root / "cache",
                root / "report",
                4,
            )
            self.assertEqual(report["training_image_count"], 2)
            self.assertTrue(
                report["training_inventory"]["resolved_mask_prompt_root"].endswith(
                    "/mask_prompt"
                )
            )

    def test_phash_hits_are_not_suppressed_by_large_dhash_distance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )

            def fingerprints(path):
                path = Path(path)
                if "train" in path.parts and path.parent.name == "ref_image":
                    return {
                        "width": 32,
                        "height": 32,
                        "pixel_sha256": "1" * 64,
                        "phash": "000000000000000f",
                        "dhash": "000000000000007f",
                    }
                if path.parent.name == "src_image":
                    return {
                        "width": 32,
                        "height": 32,
                        "pixel_sha256": "2" * 64,
                        "phash": "0000000000000000",
                        "dhash": "0000000000000000",
                    }
                return {
                    "width": 32,
                    "height": 32,
                    "pixel_sha256": "3" * 64,
                    "phash": "ffffffffffffffff",
                    "dhash": "ffffffffffffffff",
                }

            with patch(
                "scripts.check_paper_dataset_leakage.image_fingerprints",
                side_effect=fingerprints,
            ):
                report = scan(
                    dataset_root,
                    train_root,
                    root / "cache",
                    root / "report",
                    4,
                    workers=1,
                )
            hit = next(
                row
                for row in report["near_matches"]
                if row["candidate"]["role"] == "src_image"
            )
            self.assertEqual(hit["phash_distance"], 4)
            self.assertEqual(hit["dhash_distance"], 7)

    def test_truncated_cache_tail_is_ignored_and_output_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            _, manifest = make_benchmark_case(dataset_root)
            train_root = root / "train"
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            cache = root / "cache.jsonl"
            scan(dataset_root, train_root, cache, root / "first.json", 4)
            with cache.open("a", encoding="utf-8") as handle:
                handle.write('{"truncated"')
            report = scan(dataset_root, train_root, cache, root / "second.json", 4)
            self.assertEqual(report["ignored_invalid_cache_rows"], 1)
            self.assertEqual(report["training_cache_hits"], 2)

            with self.assertRaisesRegex(ValueError, "must not alias"):
                scan(dataset_root, train_root, root / "other-cache", manifest, 4)

    def test_report_cannot_overwrite_checksum_manifest_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            checksum = dataset_root / "CHECKSUMS.sha256"
            checksum_before = checksum.read_bytes()
            cache = root / "not-created" / "cache.jsonl"

            with self.assertRaisesRegex(ValueError, "must not alias CHECKSUMS"):
                scan(dataset_root, train_root, cache, checksum, 4)

            self.assertEqual(checksum.read_bytes(), checksum_before)
            self.assertFalse(cache.exists())
            self.assertFalse(cache.parent.exists())

    def test_checksums_bind_an_explicit_selected_release_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            selected = dataset_root / "manifests" / "object.jsonl"
            report = scan(
                dataset_root,
                train_root,
                root / "shared-schema2-cache.jsonl",
                root / "selected-report.json",
                4,
                manifest_path=selected,
            )
            self.assertEqual(Path(report["manifest"]), selected.resolve())
            self.assertTrue(report["release_integrity_verified"])

            selected.write_text(
                selected.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                scan(
                    dataset_root,
                    train_root,
                    root / "shared-schema2-cache.jsonl",
                    root / "tampered-selected-report.json",
                    4,
                    manifest_path=selected,
                )
            self.assertFalse((root / "tampered-selected-report.json").exists())

    def test_tampered_release_asset_and_checksum_fail_before_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            canonical = dataset_root / "object" / "src_image" / "case.png"
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(canonical)
            cache = root / "cache.jsonl"
            report = root / "report.json"
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                scan(dataset_root, train_root, cache, report, 4)
            self.assertFalse(cache.exists())
            self.assertFalse(report.exists())

            finalize_release_fixture(dataset_root)
            checksum = dataset_root / "CHECKSUMS.sha256"
            lines = checksum.read_text(encoding="utf-8").splitlines()
            _, relative = lines[0].split("  ", 1)
            lines[0] = f"{'0' * 64}  {relative}"
            checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                scan(dataset_root, train_root, cache, report, 4)
            self.assertFalse(cache.exists())
            self.assertFalse(report.exists())

    def test_output_symlinks_training_aliases_and_path_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            make_benchmark_case(dataset_root)
            train_root = root / "train"
            training_image, _ = make_training_case(
                train_root,
                Image.new("RGB", (32, 32), color=(1, 2, 3)),
                Image.new("RGB", (32, 32), color=(4, 5, 6)),
            )
            report = root / "report.json"
            with self.assertRaisesRegex(ValueError, "training input"):
                scan(dataset_root, train_root, training_image, report, 4)
            self.assertFalse(report.exists())

            symlink_parent = root / "output-link"
            symlink_parent.symlink_to(root / "real-output", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "unsafe symlink component"):
                scan(
                    dataset_root,
                    train_root,
                    root / "cache.jsonl",
                    symlink_parent / "report.json",
                    4,
                )
            self.assertFalse((root / "cache.jsonl").exists())

            with self.assertRaisesRegex(ValueError, "distinct and non-overlapping"):
                scan(
                    dataset_root,
                    train_root,
                    root / "collision",
                    root / "collision" / "report.json",
                    4,
                )
            self.assertFalse((root / "collision").exists())


if __name__ == "__main__":
    unittest.main()

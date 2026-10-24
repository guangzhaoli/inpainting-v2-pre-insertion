import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from infer_testset import benchmark_output_fields, discover_samples, prepare_fresh_output_dir


def make_four_dirs(root: Path, names: tuple[str, str, str, str]) -> None:
    for name in names:
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        mode = "L" if "mask" in name else "RGB"
        color = 255 if mode == "L" else (40, 80, 120)
        Image.new(mode, (32, 24), color=color).save(directory / "case.png")


def make_canonical_case(category: Path, case_id: str) -> None:
    for name in ("src_image", "src_mask", "tar_image", "tar_mask"):
        directory = category / name
        directory.mkdir(parents=True, exist_ok=True)
        mode = "L" if "mask" in name else "RGB"
        color = 255 if mode == "L" else (40, 80, 120)
        Image.new(mode, (32, 24), color=color).save(directory / f"{case_id}.png")


def benchmark_row(case_id: str, major: str = "object") -> dict:
    return {
        "case_id": case_id,
        "major": major,
        "minor": "animal",
        "leaf_label": "dog",
        "protocol": "cross_instance_replace",
        "gt_available": False,
        "src_image": f"{major}/src_image/{case_id}.png",
        "src_mask": f"{major}/src_mask/{case_id}.png",
        "tar_image": f"{major}/tar_image/{case_id}.png",
        "tar_mask": f"{major}/tar_mask/{case_id}.png",
    }


TEST_RELEASE_AUTHORITATIVE_PATHS = {
    "paperinsert_900.jsonl": (
        "manifests/paperinsert_900.jsonl",
        "manifests/paperinsert_900.csv",
        "manifests/object.jsonl",
        "manifests/person.jsonl",
        "manifests/garment.jsonl",
        "reports/sources.json",
    ),
    "anyinsertion_official.jsonl": (
        "manifests/anyinsertion_official.jsonl",
        "manifests/anyinsertion_clean67_for_current_model.jsonl",
        "manifests/anyinsertion_clean67_case_ids.txt",
        "manifests/anyinsertion_current_model_overlap_excluded33.json",
        "reports/conversion.json",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_test_release(
    root: Path,
    primary_name: str,
    *,
    allow_missing_assets: bool = False,
) -> None:
    """Create the complete authoritative inventory used by release-gate tests."""

    authoritative = TEST_RELEASE_AUTHORITATIVE_PATHS[primary_name]
    primary_path = root / "manifests" / primary_name
    rows = [json.loads(line) for line in primary_path.read_text(encoding="utf-8").splitlines()]
    for relative in authoritative:
        path = root / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            path.write_text("case_id\n", encoding="utf-8")
        elif path.suffix == ".json":
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    expected_paths = set(authoritative)
    for row in rows:
        expected_paths.update(
            str(row[field]) for field in ("src_image", "src_mask", "tar_image", "tar_mask")
        )
    checksum_rows = []
    for relative in sorted(expected_paths):
        path = root / relative
        if path.exists():
            digest = file_sha256(path)
        elif allow_missing_assets:
            digest = "0" * 64
        else:
            raise FileNotFoundError(path)
        checksum_rows.append(f"{digest}  {relative}\n")
    (root / "CHECKSUMS.sha256").write_text("".join(checksum_rows), encoding="utf-8")


def make_valid_anyinsertion_release(root: Path) -> dict:
    make_canonical_case(root / "object", "case")
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    row = benchmark_row("case")
    (manifest_dir / "anyinsertion_official.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )
    finalize_test_release(root, "anyinsertion_official.jsonl")
    return row


class InferTestsetSchemaTest(unittest.TestCase):
    def test_canonical_src_tar_schema_maps_donor_and_target_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category = Path(tmpdir) / "object"
            make_four_dirs(category, ("src_image", "src_mask", "tar_image", "tar_mask"))

            sample = discover_samples(Path(tmpdir), "object", limit=None)[0]

            self.assertEqual(sample["schema"], "src_tar_v1")
            self.assertEqual(sample["ref_image"].parent.name, "src_image")
            self.assertEqual(sample["ref_mask"].parent.name, "src_mask")
            self.assertEqual(sample["source"].parent.name, "tar_image")
            self.assertEqual(sample["source_mask"].parent.name, "tar_mask")

    def test_legacy_ref_tar_schema_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category = Path(tmpdir) / "object"
            make_four_dirs(category, ("ref_image", "ref_mask", "tar_image", "tar_mask"))

            sample = discover_samples(Path(tmpdir), "object", limit=None)[0]

            self.assertEqual(sample["schema"], "legacy_ref_tar")
            self.assertEqual(sample["ref_image"].parent.name, "ref_image")
            self.assertEqual(sample["source"].parent.name, "tar_image")

    def test_canonical_schema_wins_when_both_are_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            category = Path(tmpdir) / "object"
            make_four_dirs(category, ("src_image", "src_mask", "tar_image", "tar_mask"))
            make_four_dirs(category, ("ref_image", "ref_mask", "tar_image", "tar_mask"))

            sample = discover_samples(Path(tmpdir), "object", limit=None)[0]

            self.assertEqual(sample["schema"], "src_tar_v1")
            self.assertEqual(sample["ref_image"].parent.name, "src_image")

    def test_canonical_manifest_metadata_is_propagated_for_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            category = root / "object"
            make_four_dirs(category, ("src_image", "src_mask", "tar_image", "tar_mask"))
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            row = {
                "case_id": "case",
                "major": "object",
                "minor": "animal",
                "leaf_label": "dog",
                "protocol": "cross_instance_replace",
                "gt_available": False,
                "src_image": "object/src_image/case.png",
                "src_mask": "object/src_mask/case.png",
                "tar_image": "object/tar_image/case.png",
                "tar_mask": "object/tar_mask/case.png",
            }
            (manifest_dir / "paperinsert_900.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
            finalize_test_release(root, "paperinsert_900.jsonl")

            sample = discover_samples(root, "object", limit=None)[0]
            output_fields = benchmark_output_fields(sample)

            self.assertEqual(sample["benchmark_metadata"], row)
            self.assertEqual(output_fields["case_id"], "case")
            self.assertEqual(output_fields["minor"], "animal")
            self.assertEqual(output_fields["protocol"], "cross_instance_replace")
            self.assertFalse(output_fields["gt_available"])
            self.assertEqual(output_fields["benchmark"], row)
            self.assertEqual(
                output_fields["benchmark_manifest"]["path"],
                str((manifest_dir / "paperinsert_900.jsonl").resolve()),
            )
            self.assertEqual(output_fields["benchmark_manifest"]["mode"], "primary")
            self.assertEqual(output_fields["benchmark_manifest"]["selected_rows"], 1)
            self.assertEqual(len(output_fields["benchmark_manifest"]["sha256"]), 64)
            self.assertEqual(len(output_fields["benchmark_manifest"]["selection_sha256"]), 64)

    def test_supported_release_checksum_gate_accepts_complete_valid_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            row = make_valid_anyinsertion_release(root)

            samples = discover_samples(root, "object", limit=None)

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["benchmark_metadata"], row)
            self.assertEqual(samples[0]["ref_image"], root / row["src_image"])

    def test_supported_primary_requires_a_release_checksum_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_valid_anyinsertion_release(root)
            (root / "CHECKSUMS.sha256").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "release checksum manifest"):
                discover_samples(root, "object", limit=None)

    def test_false_release_checksum_is_rejected_before_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_valid_anyinsertion_release(root)
            checksum_path = root / "CHECKSUMS.sha256"
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            digest, relative = lines[0].split("  ", 1)
            self.assertNotEqual(digest, "0" * 64)
            lines[0] = f"{'0' * 64}  {relative}"
            checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                discover_samples(root, "object", limit=None)

    def test_tampered_release_png_and_mask_are_rejected_before_discovery(self):
        for field, mode, color in (
            ("src_image", "RGB", (220, 10, 30)),
            ("src_mask", "L", 0),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                row = make_valid_anyinsertion_release(root)
                Image.new(mode, (32, 24), color=color).save(root / row[field])

                with self.assertRaisesRegex(ValueError, f"Checksum mismatch {row[field]}"):
                    discover_samples(root, "object", limit=None)

    def test_manifest_file_directory_and_explicit_alias_symlinks_are_rejected(self):
        for alias_kind in ("primary_file", "manifest_directory", "explicit_file"):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                make_valid_anyinsertion_release(root)
                manifest_root = root / "manifests"
                primary = manifest_root / "anyinsertion_official.jsonl"
                explicit = None
                if alias_kind == "primary_file":
                    real_primary = manifest_root / "primary.real.jsonl"
                    primary.rename(real_primary)
                    primary.symlink_to(real_primary.name)
                elif alias_kind == "manifest_directory":
                    real_manifest_root = root / "manifest_store"
                    manifest_root.rename(real_manifest_root)
                    manifest_root.symlink_to(real_manifest_root.name, target_is_directory=True)
                else:
                    explicit = manifest_root / "selection.jsonl"
                    explicit.symlink_to(primary.name)

                with self.assertRaisesRegex(ValueError, "symlink"):
                    discover_samples(root, "object", limit=None, manifest=explicit)

    def test_canonical_role_directory_and_file_symlinks_are_rejected(self):
        for alias_kind in ("role_directory", "canonical_file"):
            with self.subTest(alias_kind=alias_kind), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                row = make_valid_anyinsertion_release(root)
                canonical_file = root / row["src_image"]
                if alias_kind == "role_directory":
                    role_dir = canonical_file.parent
                    real_role_dir = role_dir.parent / "src_image_store"
                    role_dir.rename(real_role_dir)
                    role_dir.symlink_to(real_role_dir.name, target_is_directory=True)
                else:
                    real_file = canonical_file.with_name("case.real.png")
                    canonical_file.rename(real_file)
                    canonical_file.symlink_to(real_file.name)

                with self.assertRaisesRegex(ValueError, "symlink"):
                    discover_samples(root, "object", limit=None)

    def test_checksum_inventory_rejects_unsafe_duplicate_and_missing_provenance_paths(self):
        for mutation, expected_error in (
            ("unsafe", "normalized safe POSIX"),
            ("duplicate", "Duplicate checksum path"),
            ("missing_provenance", "Checksum path set mismatch"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                make_valid_anyinsertion_release(root)
                checksum_path = root / "CHECKSUMS.sha256"
                lines = checksum_path.read_text(encoding="utf-8").splitlines()
                if mutation == "unsafe":
                    digest, _ = lines[0].split("  ", 1)
                    lines[0] = f"{digest}  ../escape.png"
                elif mutation == "duplicate":
                    lines.append(lines[0])
                else:
                    lines = [line for line in lines if not line.endswith("reports/conversion.json")]
                checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

                with self.assertRaisesRegex(ValueError, expected_error):
                    discover_samples(root, "object", limit=None)

    def test_explicit_manifest_selects_clean_subset_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            category = root / "object"
            for case_id in ("keep-1", "drop-1", "keep-2"):
                make_canonical_case(category, case_id)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            subset_manifest = manifest_dir / "anyinsertion_clean67_for_current_model.jsonl"
            rows = [benchmark_row("keep-1"), benchmark_row("keep-2")]
            primary_rows = [benchmark_row("keep-1"), benchmark_row("drop-1"), benchmark_row("keep-2")]
            (manifest_dir / "anyinsertion_official.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in primary_rows), encoding="utf-8"
            )
            subset_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            finalize_test_release(root, "anyinsertion_official.jsonl")

            samples = discover_samples(
                root,
                "object",
                limit=None,
                manifest=Path("manifests/anyinsertion_clean67_for_current_model.jsonl"),
            )

            self.assertEqual([sample["stem"] for sample in samples], ["keep-1", "keep-2"])
            expected_sha256 = hashlib.sha256(subset_manifest.read_bytes()).hexdigest()
            output_fields = benchmark_output_fields(samples[0])
            provenance = output_fields["benchmark_manifest"]
            self.assertEqual(provenance["path"], str(subset_manifest.resolve()))
            self.assertEqual(provenance["sha256"], expected_sha256)
            self.assertEqual(provenance["mode"], "explicit_subset")
            self.assertEqual(provenance["selected_rows"], 2)
            self.assertEqual(len(provenance["selection_sha256"]), 64)
            self.assertEqual(
                provenance["primary_path"],
                str((manifest_dir / "anyinsertion_official.jsonl").resolve()),
            )
            self.assertEqual(len(provenance["primary_sha256"]), 64)
            self.assertEqual(
                provenance,
                benchmark_output_fields(samples[1])["benchmark_manifest"],
            )

    def test_explicit_subset_still_requires_every_selected_case_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            category = root / "object"
            make_canonical_case(category, "present")
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            subset_manifest = manifest_dir / "clean.jsonl"
            (manifest_dir / "anyinsertion_official.jsonl").write_text(
                json.dumps(benchmark_row("missing")) + "\n", encoding="utf-8"
            )
            subset_manifest.write_text(
                json.dumps(benchmark_row("missing")) + "\n", encoding="utf-8"
            )
            finalize_test_release(
                root,
                "anyinsertion_official.jsonl",
                allow_missing_assets=True,
            )

            with self.assertRaisesRegex(FileNotFoundError, "Checksummed release file"):
                discover_samples(root, "object", limit=None, manifest=subset_manifest)

    def test_explicit_manifest_path_must_remain_inside_testset_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            root = workspace / "dataset"
            make_canonical_case(root / "object", "case")
            outside = workspace / "outside.jsonl"
            outside.write_text(json.dumps(benchmark_row("case")) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "inside testset root"):
                discover_samples(root, "object", limit=None, manifest=outside)
            with self.assertRaisesRegex(ValueError, "must not contain '\\.\\.'"):
                discover_samples(
                    root,
                    "object",
                    limit=None,
                    manifest=Path("../outside.jsonl"),
                )

    def test_manifest_record_traversal_and_noncanonical_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_canonical_case(root / "object", "case")
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            row = benchmark_row("case")
            row["src_image"] = "object/src_image/../tar_image/case.png"
            manifest = manifest_dir / "clean.jsonl"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not be absolute or contain"):
                discover_samples(root, "object", limit=None, manifest=manifest)

            row["src_image"] = "object/src_image/wrong-case.png"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match canonical"):
                discover_samples(root, "object", limit=None, manifest=manifest)

    def test_duplicate_manifest_case_key_and_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_canonical_case(root / "object", "case")
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            manifest = manifest_dir / "clean.jsonl"
            row = benchmark_row("case")
            manifest.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "Duplicate benchmark manifest key"):
                discover_samples(root, "object", limit=None, manifest=manifest)

            second = benchmark_row("second")
            second["src_image"] = row["src_image"]
            manifest.write_text(
                json.dumps(row) + "\n" + json.dumps(second) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Duplicate benchmark path"):
                discover_samples(root, "object", limit=None, manifest=manifest)

    def test_primary_manifest_remains_bidirectionally_strict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for case_id in ("listed", "extra"):
                make_canonical_case(root / "object", case_id)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            (manifest_dir / "paperinsert_900.jsonl").write_text(
                json.dumps(benchmark_row("listed")) + "\n", encoding="utf-8"
            )
            finalize_test_release(root, "paperinsert_900.jsonl")

            with self.assertRaisesRegex(ValueError, "has no row in primary manifest"):
                discover_samples(root, "object", limit=None)

    def test_explicit_manifest_must_be_unchanged_ordered_primary_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for case_id in ("first", "second"):
                make_canonical_case(root / "object", case_id)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            rows = [benchmark_row("first"), benchmark_row("second")]
            (manifest_dir / "anyinsertion_official.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            finalize_test_release(root, "anyinsertion_official.jsonl")
            explicit = manifest_dir / "clean.jsonl"
            altered = dict(rows[0])
            altered["minor"] = "tampered"
            explicit.write_text(json.dumps(altered) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "altered primary metadata"):
                discover_samples(root, "object", limit=None, manifest=explicit)

            explicit.write_text(
                json.dumps(rows[1]) + "\n" + json.dumps(rows[0]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "preserve primary row order"):
                discover_samples(root, "object", limit=None, manifest=explicit)

    def test_primary_manifest_order_controls_limit_and_seed_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for case_id in ("a", "b"):
                make_canonical_case(root / "object", case_id)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            (manifest_dir / "paperinsert_900.jsonl").write_text(
                json.dumps(benchmark_row("b")) + "\n"
                + json.dumps(benchmark_row("a")) + "\n",
                encoding="utf-8",
            )
            finalize_test_release(root, "paperinsert_900.jsonl")

            samples = discover_samples(root, "object", limit=1)

            self.assertEqual([sample["stem"] for sample in samples], ["b"])

    def test_nonempty_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "results"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "existing outputs"):
                prepare_fresh_output_dir(output)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_manifest_case_missing_from_src_image_cannot_be_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            category = root / "object"
            make_four_dirs(category, ("src_image", "src_mask", "tar_image", "tar_mask"))
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            rows = []
            for case_id in ("case", "missing"):
                rows.append(
                    {
                        "case_id": case_id,
                        "major": "object",
                        "src_image": f"object/src_image/{case_id}.png",
                        "src_mask": f"object/src_mask/{case_id}.png",
                        "tar_image": f"object/tar_image/{case_id}.png",
                        "tar_mask": f"object/tar_mask/{case_id}.png",
                    }
                )
            (manifest_dir / "paperinsert_900.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            finalize_test_release(
                root,
                "paperinsert_900.jsonl",
                allow_missing_assets=True,
            )

            with self.assertRaisesRegex(FileNotFoundError, "Checksummed release file"):
                discover_samples(root, "object", limit=1)

    def test_legacy_programmatic_sample_gets_a_schema_default(self):
        fields = benchmark_output_fields({"stem": "legacy-case"})

        self.assertEqual(fields["input_schema"], "legacy_ref_tar")


if __name__ == "__main__":
    unittest.main()

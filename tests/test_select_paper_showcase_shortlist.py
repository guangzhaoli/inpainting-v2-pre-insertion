import csv
import errno
import json
import tempfile
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import select_paper_showcase_shortlist as shortlist
from scripts.prepare_identity_paired_insertion_dataset import (
    GARMENT_NATIVE_SUBQUOTAS,
    MAJOR_MINOR_QUOTAS,
    file_sha256,
)


def _record(major: str, minor: str, index: int, native_category_id: int | None = None) -> dict:
    stem = f"{major}-{minor}-{index:03d}"
    leaf_label = (
        shortlist.builder.GARMENT_CATEGORY_NAMES[native_category_id]
        if native_category_id is not None
        else minor
    )
    row = {
        "case_id": stem,
        "major": major,
        "minor": minor,
        "leaf_label": leaf_label,
        "protocol": "paired_reconstruction",
        "pair_kind": "same_track_cross_frame" if major != "garment" else "same_product_cross_domain",
        "case_type": "paired_gt",
        "same_identity": True,
        "gt_available": True,
        "reference_derived_from_target": False,
        "identity_scope": "physical_instance" if major != "garment" else "product_identity",
        "identity_id": f"identity/{stem}",
        "src_identity_id": f"identity/{stem}",
        "tar_identity_id": f"identity/{stem}",
        "source_dataset": "mevis_v2" if major != "garment" else "deepfashion2",
        "src_image_id": f"image/src/{stem}",
        "tar_image_id": f"image/tar/{stem}",
        "src_image": f"{major}/src_image/{stem}.png",
        "src_mask": f"{major}/src_mask/{stem}.png",
        "tar_image": f"{major}/tar_image/{stem}.png",
        "tar_mask": f"{major}/tar_mask/{stem}.png",
        "src_pixel_sha256": f"src-pixels-{stem}",
        "src_mask_pixel_sha256": f"src-mask-pixels-{stem}",
        "tar_pixel_sha256": f"tar-pixels-{stem}",
        "tar_mask_pixel_sha256": f"tar-mask-pixels-{stem}",
        "difficulty": "easy",
        "selection_score": float(100 - index),
        "mask_origin_src": "native_instance_annotation",
        "mask_origin_tar": "native_instance_annotation",
        "auto_quality_pass": True,
        "showcase_candidate": index < 10,
        "agent_paper_avoid": False,
        "manual_qc_status": "pending",
        "paper_ok": False,
    }
    if native_category_id is not None:
        row["native_category_id"] = native_category_id
    return row


def _records() -> list[dict]:
    records: list[dict] = []
    for major, minors in MAJOR_MINOR_QUOTAS.items():
        for minor, quota in minors.items():
            native_ids: list[int | None]
            if major == "garment":
                native_ids = [
                    category_id
                    for category_id, count in GARMENT_NATIVE_SUBQUOTAS[minor].items()
                    for _ in range(count)
                ]
            else:
                native_ids = [None] * quota
            records.extend(
                _record(major, minor, index, native_category_id)
                for index, native_category_id in enumerate(native_ids)
            )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )


def _make_release(root: Path, records: list[dict]) -> Path:
    manifest = root / "manifests" / "paperinsert_900.jsonl"
    _write_jsonl(manifest, records)
    checksums = root / "CHECKSUMS.sha256"
    checksums.write_text("fixture checksum bytes used through mocked verifier\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "valid": True,
        "errors": [],
        "warnings": [],
        "record_count": 900,
        "identity_count": 900,
        "different_source_image_count": 900,
        "target_derived_reference_count": 0,
        "raw_backed_audit": True,
        "manifest_sha256": file_sha256(manifest),
        "checksum_manifest_sha256": file_sha256(checksums),
    }
    audit_path = root / "reports" / "audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def _fake_audit(dataset_root: Path, *, verify_raw: bool, write_report: bool) -> dict:
    if verify_raw or write_report:
        raise AssertionError("showcase audit must remain read-only and must not rerun raw reconstruction")
    return {
        "valid": True,
        "errors": [],
        "manifest_sha256": file_sha256(dataset_root / "manifests" / "paperinsert_900.jsonl"),
        "checksum_manifest_sha256": file_sha256(dataset_root / "CHECKSUMS.sha256"),
    }


@contextmanager
def _mock_release_verification():
    with (
        patch.object(shortlist.builder, "audit_dataset", side_effect=_fake_audit) as audit,
        patch.object(shortlist.builder, "verify_checksums", return_value=[]) as checksums,
    ):
        yield audit, checksums


def _materialize_candidate_images(root: Path, records: list[dict]) -> None:
    selected = shortlist.select_seed_candidates(records)
    by_case = {row["case_id"]: row for row in records}
    for candidate in selected:
        record = by_case[candidate["case_id"]]
        for role in ("src_image", "tar_image"):
            path = root / record[role]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (24, 20), (70, 120, 180)).save(path)
        for role in ("src_mask", "tar_mask"):
            path = root / record[role]
            path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("L", (24, 20), 0)
            for x in range(6, 18):
                for y in range(4, 16):
                    image.putpixel((x, y), 255)
            image.save(path)


def _complete_decisions(payload: dict, accepted_per_minor: int = shortlist.ACCEPTED_PER_MINOR) -> dict:
    counts: Counter[tuple[str, str]] = Counter()
    for row in payload["decisions"]:
        key = (row["major"], row["minor"])
        row["decision"] = "accepted" if counts[key] < accepted_per_minor else "rejected"
        row["agent_note"] = "agent visual shortlist decision"
        counts[key] += 1
    return payload


class PaperShowcaseShortlistTest(unittest.TestCase):
    def test_portable_directory_publish_falls_back_and_commits_index_last(self):
        for error_number in (errno.EINVAL, errno.EXDEV):
            with self.subTest(errno=error_number), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "staged"
                destination = root / "published"
                (source / "sheets" / "object").mkdir(parents=True)
                (source / "sheets" / "object" / "sheet.jpg").write_bytes(b"jpeg bytes")
                (source / "index.json").write_text("{}\n", encoding="utf-8")
                identities: list[tuple[Path, bool, int, int]] = []
                copied: list[Path] = []
                original_copy = shortlist._copy_file_atomic_noreplace

                def record_copy(source_path, destination_path, destination_root, owned):
                    copied.append(destination_path.relative_to(destination_root))
                    return original_copy(source_path, destination_path, destination_root, owned)

                with (
                    patch.object(
                        shortlist,
                        "_rename_noreplace",
                        side_effect=OSError(error_number, "injected unsupported rename"),
                    ),
                    patch.object(
                        shortlist,
                        "_copy_file_atomic_noreplace",
                        side_effect=record_copy,
                    ),
                ):
                    method = shortlist._publish_directory_noreplace(
                        source,
                        destination,
                        identities,
                    )
                self.assertEqual(method, "portable_index_last_noreplace")
                self.assertEqual(copied[-1], Path("index.json"))
                self.assertEqual((destination / "index.json").read_text(encoding="utf-8"), "{}\n")
                self.assertEqual(
                    (destination / "sheets" / "object" / "sheet.jpg").read_bytes(),
                    b"jpeg bytes",
                )
                shortlist._remove_owned_tree(destination, identities)
                self.assertFalse(destination.exists())

    def test_selection_rejects_none_ids_and_round_robins_garment_native_categories(self):
        records = _records()
        companion = [
            row
            for row in records
            if row["major"] == "object" and row["minor"] == "companion_animals"
        ]
        companion[0]["identity_id"] = None
        companion[1]["src_image_id"] = None
        selected = shortlist.select_seed_candidates(records)

        selected_ids = {row["case_id"] for row in selected}
        self.assertNotIn(companion[0]["case_id"], selected_ids)
        self.assertNotIn(companion[1]["case_id"], selected_ids)
        counts = Counter((row["major"], row["minor"]) for row in selected)
        self.assertEqual(set(counts.values()), {shortlist.CANDIDATES_PER_MINOR})
        outerwear = [
            row["native_category_id"]
            for row in selected
            if row["major"] == "garment" and row["minor"] == "outerwear_vest_sling"
        ]
        self.assertEqual(outerwear, [4, 5, 6, 3, 4, 5])
        trousers = [
            row["native_category_id"]
            for row in selected
            if row["major"] == "garment" and row["minor"] == "shorts_trousers"
        ]
        self.assertEqual(trousers, [7, 8, 7, 8, 7, 8])

        invalid_native = _records()
        outerwear_rows = [
            row
            for row in invalid_native
            if row["major"] == "garment" and row["minor"] == "outerwear_vest_sling"
        ]
        outerwear_rows[0]["native_category_id"] = True
        outerwear_rows[1]["leaf_label"] = "wrong_native_label"
        invalid_selected = shortlist.select_seed_candidates(invalid_native)
        invalid_ids = {row["case_id"] for row in invalid_selected}
        self.assertNotIn(outerwear_rows[0]["case_id"], invalid_ids)
        self.assertNotIn(outerwear_rows[1]["case_id"], invalid_ids)

        missing_native = _records()
        for row in missing_native:
            if row.get("native_category_id") == 3:
                row["native_category_id"] = "3"
        with self.assertRaisesRegex(ValueError, r"missing=\[3\]"):
            shortlist.select_seed_candidates(missing_native)

    def test_strict_audit_binds_snapshotted_release_and_detects_end_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            manifest = _make_release(root, _records())
            with _mock_release_verification() as (audit, checksums):
                release = shortlist.strict_release_snapshot(root)
                self.assertEqual(release.manifest.sha256, file_sha256(manifest))
                self.assertEqual(release.raw_audit_report["manifest_sha256"], release.manifest.sha256)
                audit.assert_called_once_with(root.resolve(), verify_raw=False, write_report=False)
                self.assertEqual(checksums.call_count, 0)

                audit_path = root / "reports" / "audit.json"
                audit_path.write_text(audit_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "changed during showcase operation"):
                    shortlist._assert_release_unchanged(release)

    def test_strict_audit_rejects_stale_checksums_and_builder_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            _make_release(root, _records())
            checksums = root / "CHECKSUMS.sha256"
            checksums.write_text(checksums.read_text(encoding="utf-8") + "stale\n", encoding="utf-8")
            with patch.object(shortlist.builder, "audit_dataset") as audit:
                with self.assertRaisesRegex(ValueError, "checksum_manifest_sha256"):
                    shortlist.strict_release_snapshot(root)
                audit.assert_not_called()

            _make_release(root, _records())
            failed_audit = {
                "valid": False,
                "errors": ["bad identity evidence", "decoded hash mismatch for tar_image"],
            }
            with patch.object(shortlist.builder, "audit_dataset", return_value=failed_audit):
                with self.assertRaisesRegex(ValueError, "bad identity evidence"):
                    shortlist.strict_release_snapshot(root)

    def test_seed_renders_bound_four_view_sheets_and_finalize_emits_54_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            manifest = _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            manifest_before = manifest.read_bytes()

            with _mock_release_verification():
                seed = shortlist.seed_decision_file(root, decisions, review_dir)

            self.assertEqual(manifest.read_bytes(), manifest_before)
            self.assertEqual(seed["candidate_count"], len(shortlist.TAXONOMY) * 6)
            self.assertEqual(seed["manifest_sha256"], file_sha256(manifest))
            self.assertEqual(len(list((review_dir / "sheets").rglob("*.jpg"))), len(shortlist.TAXONOMY))
            review_index = json.loads((review_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(review_index["four_view_panels"], [
                "independent_reference_with_native_mask_outline",
                "reference_native_mask",
                "target_masked_input",
                "target_ground_truth_with_native_mask_outline",
            ])
            self.assertEqual(review_index["candidate_set_sha256"], seed["candidate_set_sha256"])
            self.assertEqual(review_index["renderer_sha256"], seed["renderer_sha256"])
            for row in seed["decisions"]:
                self.assertTrue(row["review_sheet"].endswith("sheet_00.jpg"))
                self.assertEqual(len(row["review_tile_xywh"]), 4)
                for role in ("src_image", "src_mask", "tar_image", "tar_mask"):
                    self.assertTrue(row[role].endswith(".png"))

            complete = _complete_decisions(seed)
            decisions.write_text(json.dumps(complete, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            output_json = base / "shortlist.json"
            output_csv = base / "shortlist.csv"
            with _mock_release_verification():
                result = shortlist.finalize_shortlist(
                    root,
                    decisions,
                    output_json,
                    output_csv,
                )

            self.assertEqual(result["status"], shortlist.FINAL_STATUS)
            self.assertEqual(result["case_count"], len(shortlist.TAXONOMY) * 3)
            self.assertEqual(
                sum(count > 0 for count in result["garment_native_category_counts"].values()),
                11,
            )
            self.assertEqual(result["decision_sha256"], file_sha256(decisions))
            self.assertEqual(result["review_index_sha256"], file_sha256(review_dir / "index.json"))
            self.assertNotIn("paper_ok", output_json.read_text(encoding="utf-8"))
            with output_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 54)
            self.assertEqual({row["shortlist_status"] for row in rows}, {shortlist.FINAL_STATUS})

    def test_seed_uses_portable_publish_when_mounted_fs_rejects_rename_flag(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            with (
                _mock_release_verification(),
                patch.object(
                    shortlist,
                    "_rename_noreplace",
                    side_effect=OSError(errno.EINVAL, "mounted fs rejects RENAME_NOREPLACE"),
                ),
            ):
                payload = shortlist.seed_decision_file(root, decisions, review_dir)

            self.assertTrue(decisions.is_file())
            self.assertTrue((review_dir / "index.json").is_file())
            self.assertEqual(file_sha256(review_dir / "index.json"), payload["review_index_sha256"])
            self.assertEqual(len(list((review_dir / "sheets").rglob("*.jpg"))), 18)

    def test_portable_publish_rolls_back_partial_tree_without_overwriting_racer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staged"
            destination = root / "published"
            (source / "sheets").mkdir(parents=True)
            (source / "sheets" / "a.jpg").write_bytes(b"a")
            (source / "sheets" / "b.jpg").write_bytes(b"b")
            (source / "index.json").write_text("{}\n", encoding="utf-8")
            identities: list[tuple[Path, bool, int, int]] = []
            original_copy = shortlist._copy_file_atomic_noreplace
            copy_count = 0

            def fail_second_copy(source_path, destination_path, destination_root, owned):
                nonlocal copy_count
                copy_count += 1
                if copy_count == 2:
                    racer = destination_root / "racer-owned.txt"
                    racer.write_text("do not delete\n", encoding="utf-8")
                    raise KeyboardInterrupt("injected portable publish interruption")
                return original_copy(source_path, destination_path, destination_root, owned)

            with (
                patch.object(
                    shortlist,
                    "_rename_noreplace",
                    side_effect=OSError(errno.EINVAL, "unsupported"),
                ),
                patch.object(
                    shortlist,
                    "_copy_file_atomic_noreplace",
                    side_effect=fail_second_copy,
                ),
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "portable publish interruption"):
                    try:
                        shortlist._publish_directory_noreplace(
                            source,
                            destination,
                            identities,
                        )
                    finally:
                        shortlist._remove_owned_tree(destination, identities)
            self.assertEqual(
                (destination / "racer-owned.txt").read_text(encoding="utf-8"),
                "do not delete\n",
            )
            self.assertFalse((destination / "sheets" / "a.jpg").exists())
            self.assertFalse((destination / "index.json").exists())

    def test_portable_publish_refuses_existing_root_and_recovers_post_link_interrupt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staged"
            destination = root / "published"
            source.mkdir()
            (source / "index.json").write_text("{}\n", encoding="utf-8")
            destination.mkdir()
            sentinel = destination / "racer.txt"
            sentinel.write_text("owned by racer\n", encoding="utf-8")
            identities: list[tuple[Path, bool, int, int]] = []
            with patch.object(
                shortlist,
                "_rename_noreplace",
                side_effect=OSError(errno.EINVAL, "unsupported"),
            ):
                with self.assertRaises(FileExistsError):
                    shortlist._publish_directory_noreplace(source, destination, identities)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "owned by racer\n")
            self.assertEqual(identities, [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staged"
            destination = root / "published"
            (source / "sheets").mkdir(parents=True)
            (source / "sheets" / "a.jpg").write_bytes(b"a")
            (source / "index.json").write_text("{}\n", encoding="utf-8")
            identities = []
            original_link = shortlist.os.link

            def link_then_interrupt(source_path, destination_path, *, follow_symlinks):
                original_link(source_path, destination_path, follow_symlinks=follow_symlinks)
                if Path(destination_path).suffix == ".jpg":
                    raise KeyboardInterrupt("injected after fallback sheet link")

            with (
                patch.object(
                    shortlist,
                    "_rename_noreplace",
                    side_effect=OSError(errno.EINVAL, "unsupported"),
                ),
                patch.object(shortlist.os, "link", side_effect=link_then_interrupt),
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "after fallback sheet link"):
                    try:
                        shortlist._publish_directory_noreplace(
                            source,
                            destination,
                            identities,
                        )
                    finally:
                        shortlist._remove_owned_tree(destination, identities)
            self.assertFalse(destination.exists())

    def test_finalize_rejects_pending_and_tampered_review_sheet(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            with _mock_release_verification():
                seed = shortlist.seed_decision_file(root, decisions, review_dir)
                with self.assertRaisesRegex(ValueError, "explicit.*pending"):
                    shortlist.build_shortlist_payload(root, decisions)

            _complete_decisions(seed)
            decisions.write_text(json.dumps(seed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            first_sheet = next((review_dir / "sheets").rglob("*.jpg"))
            first_sheet.write_bytes(first_sheet.read_bytes() + b"tamper")
            with _mock_release_verification():
                with self.assertRaisesRegex(ValueError, "Review sheet SHA mismatch"):
                    shortlist.build_shortlist_payload(root, decisions)

    def test_review_structure_and_final_native_diversity_cannot_be_forged(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            with _mock_release_verification():
                seed = shortlist.seed_decision_file(root, decisions, review_dir)

            index_path = review_dir / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["sheets"][0]["cases"][0]["canonical_paths"]["src_image"] = "forged.png"
            index_path.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            seed["review_index_sha256"] = file_sha256(index_path)
            decisions.write_text(json.dumps(seed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with _mock_release_verification():
                with self.assertRaisesRegex(ValueError, "structure mismatch"):
                    shortlist.build_shortlist_payload(root, decisions)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            with _mock_release_verification():
                seed = shortlist.seed_decision_file(root, decisions, review_dir)
            _complete_decisions(seed)
            outerwear = [
                row
                for row in seed["decisions"]
                if row["major"] == "garment" and row["minor"] == "outerwear_vest_sling"
            ]
            for row in outerwear:
                row["decision"] = "accepted" if row["candidate_rank"] in {1, 2, 5} else "rejected"
            decisions.write_text(json.dumps(seed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            with _mock_release_verification():
                with self.assertRaisesRegex(ValueError, "maximize native-category diversity"):
                    shortlist.build_shortlist_payload(root, decisions)

    def test_seed_and_finalize_publish_no_partial_outputs_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            manifest = _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            original_render = shortlist.render_review_sheets

            def render_then_change_manifest(*args, **kwargs):
                review = original_render(*args, **kwargs)
                manifest.write_bytes(manifest.read_bytes() + b"\n")
                return review

            with (
                _mock_release_verification(),
                patch.object(shortlist, "render_review_sheets", side_effect=render_then_change_manifest),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed during showcase operation"):
                    shortlist.seed_decision_file(root, decisions, review_dir)
            self.assertFalse(decisions.exists())
            self.assertFalse(review_dir.exists())

            _make_release(root, records)
            original_link = shortlist.os.link

            def link_then_interrupt(source, destination, *, follow_symlinks):
                original_link(source, destination, follow_symlinks=follow_symlinks)
                raise KeyboardInterrupt("injected immediately after decision link")

            with (
                _mock_release_verification(),
                patch.object(shortlist.os, "link", side_effect=link_then_interrupt),
            ):
                with self.assertRaisesRegex(KeyboardInterrupt, "after decision link"):
                    shortlist.seed_decision_file(root, decisions, review_dir)
            self.assertFalse(decisions.exists())
            self.assertFalse(review_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "release"
            records = _records()
            _make_release(root, records)
            _materialize_candidate_images(root, records)
            decisions = base / "decisions.json"
            review_dir = base / "review"
            with _mock_release_verification():
                seed = shortlist.seed_decision_file(root, decisions, review_dir)
            _complete_decisions(seed)
            decisions.write_text(json.dumps(seed, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            output_json = base / "shortlist.json"
            output_csv = base / "shortlist.csv"
            original_create = shortlist._atomic_create_text
            call_count = 0

            def fail_second_create(path, text, authorized_identities):
                nonlocal call_count
                call_count += 1
                if call_count == 2:
                    raise OSError("injected second-output failure")
                return original_create(path, text, authorized_identities)

            with (
                _mock_release_verification(),
                patch.object(shortlist, "_atomic_create_text", side_effect=fail_second_create),
            ):
                with self.assertRaisesRegex(OSError, "injected second-output failure"):
                    shortlist.finalize_shortlist(
                        root,
                        decisions,
                        output_json,
                        output_csv,
                    )
            self.assertFalse(output_json.exists())
            self.assertFalse(output_csv.exists())


if __name__ == "__main__":
    unittest.main()

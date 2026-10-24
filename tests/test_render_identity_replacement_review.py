import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import render_identity_replacement_review as renderer


def _record(
    case_id: str,
    identity_id: str,
    signature_salt: str,
    **overrides,
) -> dict:
    record = {
        "case_id": case_id,
        "identity_id": identity_id,
        "major": "object",
        "minor": "companion_animals",
        "src_pixel_sha256": f"{signature_salt}-src-image",
        "src_mask_pixel_sha256": f"{signature_salt}-src-mask",
        "tar_pixel_sha256": f"{signature_salt}-tar-image",
        "tar_mask_pixel_sha256": f"{signature_salt}-tar-mask",
        # These fields are deliberately not part of the approval signature.
        "source_dataset": "fixture",
        "leaf_label": "dog",
        "difficulty": "easy",
    }
    record.update(overrides)
    return record


def _write_manifest(root: Path, records: list[dict]) -> Path:
    manifest = root / "manifests" / "paperinsert_900.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


class IdentityReplacementReviewRendererTest(unittest.TestCase):
    def test_signature_uses_exact_declared_fields_and_requires_each_one(self):
        record = _record("case", "identity", "v1")
        expected = tuple(str(record[field]) for field in renderer.SIGNATURE_FIELDS)
        self.assertEqual(renderer.review_signature(record), expected)

        # Presentation-only changes must not invalidate an already reviewed case.
        presentation_change = dict(record, case_id="renumbered", difficulty="moderate")
        self.assertEqual(
            renderer.review_signature(presentation_change),
            renderer.review_signature(record),
        )

        for field in renderer.SIGNATURE_FIELDS:
            with self.subTest(field=field):
                changed = dict(record)
                changed[field] = f"changed-{field}"
                self.assertNotEqual(
                    renderer.review_signature(changed),
                    renderer.review_signature(record),
                )
                missing = dict(record)
                del missing[field]
                with self.assertRaises(KeyError):
                    renderer.review_signature(missing)

    def test_main_accounts_for_approved_rejected_changed_and_new_signatures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_root = root / "baseline"
            candidate_root = root / "candidate"
            output_dir = root / "review"

            approved = _record("baseline-approved", "identity-approved", "approved")
            # A duplicate baseline signature exercises set-based signature accounting.
            approved_duplicate = dict(approved, case_id="baseline-approved-duplicate")
            rejected = _record("baseline-rejected", "identity-rejected", "rejected")
            baseline_manifest = _write_manifest(
                baseline_root,
                [approved, approved_duplicate, rejected],
            )

            unchanged = dict(
                approved,
                case_id="object_companion_animals_099",
                difficulty="moderate",
            )
            changed = dict(approved, case_id="object_companion_animals_010")
            changed["tar_mask_pixel_sha256"] = "changed-target-mask"
            rejected_again = dict(rejected, case_id="object_companion_animals_020")
            new = _record("object_companion_animals_030", "identity-new", "new")
            candidate_manifest = _write_manifest(
                candidate_root,
                [new, unchanged, rejected_again, changed],
            )

            exclusions = root / "exclusions.json"
            exclusions.write_text(
                json.dumps(
                    {
                        "excluded_identities": [
                            {"identity_id": "identity-rejected"},
                            "identity-not-in-baseline",
                        ]
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            rendered_case_ids: list[str] = []

            def fake_tile(_dataset_root: Path, record: dict) -> Image.Image:
                rendered_case_ids.append(str(record["case_id"]))
                return Image.new("RGB", (16, 16), "navy")

            stdout = io.StringIO()
            with (
                patch.object(renderer.builder, "_case_review_tile", side_effect=fake_tile),
                contextlib.redirect_stdout(stdout),
            ):
                result = renderer.main(
                    [
                        "--baseline-root",
                        str(baseline_root),
                        "--candidate-root",
                        str(candidate_root),
                        "--exclusions",
                        str(exclusions),
                        "--output-dir",
                        str(output_dir),
                        "--cases-per-sheet",
                        "2",
                    ]
                )

            self.assertEqual(result, 0)
            expected_pending = [
                "object_companion_animals_010",
                "object_companion_animals_020",
                "object_companion_animals_030",
            ]
            self.assertEqual(rendered_case_ids, expected_pending)
            self.assertTrue((output_dir / "sheet_00.jpg").is_file())
            self.assertTrue((output_dir / "sheet_01.jpg").is_file())
            self.assertFalse(list(output_dir.rglob("*.tmp")))

            report = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(json.loads(stdout.getvalue()), report)
            self.assertEqual(report["signature_fields"], list(renderer.SIGNATURE_FIELDS))
            self.assertEqual(report["baseline_case_count"], 3)
            self.assertEqual(report["candidate_case_count"], 4)
            self.assertEqual(report["approved_baseline_signature_count"], 1)
            self.assertEqual(report["pending_case_count"], 3)
            self.assertEqual(report["sheet_count"], 2)
            self.assertEqual(
                [case_id for sheet in report["sheets"] for case_id in sheet["case_ids"]],
                expected_pending,
            )
            self.assertTrue(all(sheet["review_status"] == "pending" for sheet in report["sheets"]))

            self.assertEqual(report["baseline_root"], str(baseline_root.resolve()))
            self.assertEqual(report["candidate_root"], str(candidate_root.resolve()))
            self.assertEqual(report["exclusion_config"], str(exclusions.resolve()))
            self.assertEqual(
                report["baseline_manifest_sha256"],
                renderer.builder.file_sha256(baseline_manifest),
            )
            self.assertEqual(
                report["candidate_manifest_sha256"],
                renderer.builder.file_sha256(candidate_manifest),
            )
            self.assertEqual(
                report["exclusion_config_sha256"],
                renderer.builder.file_sha256(exclusions),
            )

            # The recorded digest makes later manifest tampering detectable.
            recorded_candidate_digest = report["candidate_manifest_sha256"]
            candidate_manifest.write_text("{}\n", encoding="utf-8")
            self.assertNotEqual(
                recorded_candidate_digest,
                renderer.builder.file_sha256(candidate_manifest),
            )

    def test_empty_pending_set_emits_a_complete_zero_sheet_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline_root = root / "baseline"
            candidate_root = root / "candidate"
            output_dir = root / "review"
            record = _record("baseline-id", "same-identity", "same")
            _write_manifest(baseline_root, [record])
            _write_manifest(candidate_root, [dict(record, case_id="candidate-id")])
            exclusions = root / "exclusions.json"
            exclusions.write_text('{"excluded_identities": []}\n', encoding="utf-8")

            with (
                patch.object(renderer.builder, "_case_review_tile") as render_tile,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    renderer.main(
                        [
                            "--baseline-root",
                            str(baseline_root),
                            "--candidate-root",
                            str(candidate_root),
                            "--exclusions",
                            str(exclusions),
                            "--output-dir",
                            str(output_dir),
                        ]
                    ),
                    0,
                )

            render_tile.assert_not_called()
            report = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(report["approved_baseline_signature_count"], 1)
            self.assertEqual(report["pending_case_count"], 0)
            self.assertEqual(report["sheet_count"], 0)
            self.assertEqual(report["sheets"], [])
            self.assertFalse(list(output_dir.glob("*.jpg")))

    def test_existing_output_is_never_overwritten_and_invalid_size_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "review"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("do not replace\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "output already exists"):
                renderer.main(
                    [
                        "--baseline-root",
                        str(root / "missing-baseline"),
                        "--candidate-root",
                        str(root / "missing-candidate"),
                        "--exclusions",
                        str(root / "missing-exclusions.json"),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "do not replace\n")
            self.assertEqual(sorted(path.name for path in output_dir.iterdir()), ["keep.txt"])

            never_created = root / "invalid-size-review"
            with self.assertRaisesRegex(ValueError, "must be positive"):
                renderer.main(
                    [
                        "--baseline-root",
                        str(root / "missing-baseline"),
                        "--candidate-root",
                        str(root / "missing-candidate"),
                        "--exclusions",
                        str(root / "missing-exclusions.json"),
                        "--output-dir",
                        str(never_created),
                        "--cases-per-sheet",
                        "0",
                    ]
                )
            self.assertFalse(never_created.exists())


if __name__ == "__main__":
    unittest.main()

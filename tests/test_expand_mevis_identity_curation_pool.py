import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import expand_mevis_identity_curation_pool as expander


def _json_rle(values: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(values.astype(np.uint8)))
    return {
        "size": list(encoded["size"]),
        "counts": encoded["counts"].decode("ascii"),
    }


def _rectangle_rle(
    *,
    height: int = 100,
    width: int = 100,
    top: int = 0,
    left: int = 0,
    bottom: int = 20,
    right: int = 9,
) -> dict:
    values = np.zeros((height, width), dtype=np.uint8)
    values[top:bottom, left:right] = 1
    return _json_rle(values)


def _frame_info(
    index: int,
    frame_id: str,
    rle: dict,
    *,
    width: int = 40,
    height: int = 40,
    area_ratio: float = 0.25,
    bbox_xywh: tuple[float, float, float, float] = (10.0, 10.0, 20.0, 20.0),
    solidity: float = 1.0,
    margin_ratio: float = 0.25,
    geometry_score: float = 4.0,
) -> expander.builder.MevisFrameInfo:
    return expander.builder.MevisFrameInfo(
        index=index,
        frame_id=frame_id,
        rle=rle,
        width=width,
        height=height,
        area_ratio=area_ratio,
        bbox_xywh=bbox_xywh,
        aspect_ratio=bbox_xywh[2] / bbox_xywh[3],
        solidity=solidity,
        margin_ratio=margin_ratio,
        geometry_score=geometry_score,
    )


def _base_payload(candidates: list[dict] | None = None) -> dict:
    return {
        "schema_version": "mevis_diverse_candidate_pool_v1",
        "source": {
            "dataset": "MeViS v2",
            "split": "train",
            "meta": "meta_expressions_v2.json",
            "masks": "mask_dict.json",
            "images": "JPEGImages",
        },
        "contract": {"identity_unit": "(video_id, anno_id)"},
        "eligibility": {"common_pixel_delta": {"crop_phash_min": 2}},
        "candidates": list(candidates or []),
    }


def _minimal_base_row() -> dict:
    return {
        "candidate_id": "mevis_v2_train_base_video_7__object_companion_animals",
        "major": "object",
        "minor": "companion_animals",
        "video_id": "base_video",
        "anno_id": 7,
        "identity_key": "mevis_v2/train/base_video/7",
        "pixel_delta_tier": "strong",
    }


@contextlib.contextmanager
def _patched_source_pins(base_pool: Path, mevis_root: Path):
    builder = expander.builder
    with ExitStack() as stack:
        stack.enter_context(patch.object(expander, "BASE_POOL_SHA256", builder.file_sha256(base_pool)))
        stack.enter_context(
            patch.object(
                builder,
                "MEVIS_ARCHIVE_SHA256",
                builder.file_sha256(mevis_root / "JPEGImages.tar"),
            )
        )
        stack.enter_context(
            patch.object(
                builder,
                "MEVIS_META_SHA256",
                builder.file_sha256(mevis_root / "meta_expressions_v2.json"),
            )
        )
        stack.enter_context(
            patch.object(
                builder,
                "MEVIS_MASK_DICT_SHA256",
                builder.file_sha256(mevis_root / "mask_dict.json"),
            )
        )
        yield


class ExpandMevisIdentityCurationPoolTest(unittest.TestCase):
    def setUp(self):
        self._old_tracks = expander._TRACKS
        self._old_image_root = expander._IMAGE_ROOT
        self._old_seed = expander._SEED

    def tearDown(self):
        expander._TRACKS = self._old_tracks
        expander._IMAGE_ROOT = self._old_image_root
        expander._SEED = self._old_seed

    def test_pixel_tier_uses_both_thresholds_at_exact_boundaries(self):
        cases = (
            (0.04, 6, "strong"),
            (0.04, 5, "medium"),
            (0.039999, 6, "medium"),
            (0.03, 4, "medium"),
            (0.03, 3, "base"),
            (0.029999, 64, "base"),
        )
        for crop_mae, crop_phash, expected in cases:
            with self.subTest(crop_mae=crop_mae, crop_phash=crop_phash):
                self.assertEqual(
                    expander._pixel_tier({"crop_mae": crop_mae, "crop_phash": crop_phash}),
                    expected,
                )

    def test_relaxed_frame_info_accepts_only_the_documented_geometry_range(self):
        accepted_minimum = expander._relaxed_frame_info(3, "minimum", _rectangle_rle(bottom=20, right=9))
        self.assertIsNotNone(accepted_minimum)
        assert accepted_minimum is not None
        self.assertEqual((accepted_minimum.index, accepted_minimum.frame_id), (3, "minimum"))
        self.assertAlmostEqual(accepted_minimum.area_ratio, 0.018)
        self.assertEqual(accepted_minimum.bbox_xywh, (0.0, 0.0, 9.0, 20.0))
        self.assertEqual(accepted_minimum.margin_ratio, 0.0)
        self.assertGreater(accepted_minimum.geometry_score, 0.0)

        accepted_maximum = expander._relaxed_frame_info(0, "maximum", _rectangle_rle(bottom=100, right=60))
        self.assertIsNotNone(accepted_maximum)
        assert accepted_maximum is not None
        self.assertAlmostEqual(accepted_maximum.area_ratio, 0.60)

        fragmented = np.zeros((100, 100), dtype=np.uint8)
        fragmented[:10, :10] = 1
        fragmented[90:, 90:] = 1
        rejected = {
            "area_below_minimum": _rectangle_rle(bottom=19, right=9),
            "area_above_maximum": _rectangle_rle(bottom=100, right=61),
            "short_side_below_minimum": _rectangle_rle(bottom=100, right=3),
            "solidity_below_minimum": _json_rle(fragmented),
            "empty": _json_rle(np.zeros((100, 100), dtype=np.uint8)),
        }
        for label, rle in rejected.items():
            with self.subTest(label=label):
                self.assertIsNone(expander._relaxed_frame_info(0, label, rle))

    def test_frame_geometry_and_white_canvas_preserve_only_masked_foreground(self):
        mask_values = np.zeros((40, 40), dtype=np.uint8)
        mask_values[10:30, 10:30] = 1
        rle = _json_rle(mask_values)
        info = _frame_info(0, "00000", rle)

        geometry = expander._frame_geometry(info)
        self.assertEqual(geometry["mask_area_pixels"], 400.0)
        self.assertEqual(geometry["bbox_xywh"], [10.0, 10.0, 20.0, 20.0])
        self.assertEqual(geometry["bbox_short_side_ratio"], 0.5)

        values = np.full((40, 40, 3), (0, 0, 255), dtype=np.uint8)
        values[mask_values > 0] = (255, 0, 0)
        image = Image.fromarray(values, mode="RGB")
        canvas = expander._white_foreground_canvas(image, rle, size=64)
        canvas_values = np.asarray(canvas, dtype=np.uint8)

        self.assertEqual(canvas.size, (64, 64))
        self.assertTrue(np.all(canvas_values[0, 0] == 255))
        self.assertTrue(np.any(np.all(canvas_values == (255, 0, 0), axis=2)))
        self.assertFalse(np.any(np.all(canvas_values == (0, 0, 255), axis=2)))

        pixel = expander._pixel_delta(image, image.copy(), info, info)
        self.assertEqual(
            pixel,
            {
                "crop_mae": 0.0,
                "crop_phash": 0,
                "crop_dhash": 0,
                "frame_mae": 0.0,
                "frame_phash": 0,
            },
        )

    def test_pair_selector_uses_fallback_and_restores_builder_helper(self):
        track = expander.builder.MevisTrack(
            video_id="video",
            anno_id="17",
            frame_ids=("00000", "00009"),
            rles=(None, None),
            expressions=("a person walking", "the person walking"),
        )
        rle = _rectangle_rle(height=40, width=40, top=10, left=10, bottom=30, right=30)
        pair = (
            10.0,
            _frame_info(0, "00000", rle),
            _frame_info(1, "00009", rle),
            {},
        )
        expander._IMAGE_ROOT = Path("/unused")
        original = expander.builder._mevis_frame_info
        observations = []

        def select_pair(*_args, **kwargs):
            observations.append((kwargs["relaxed_daily_object"], expander.builder._mevis_frame_info))
            return None if len(observations) == 1 else pair

        with patch.object(expander.builder, "select_frame_pair", side_effect=select_pair):
            selected, rule = expander._select_supplemental_pair(track, {("person", "walking_running"): 1.0})

        self.assertIs(selected, pair)
        self.assertEqual(rule, "supplemental_relaxed_geometry_v1")
        self.assertEqual(observations[0], (False, original))
        self.assertEqual(observations[1], (True, expander._relaxed_frame_info))
        self.assertIs(expander.builder._mevis_frame_info, original)

        with patch.object(
            expander.builder,
            "select_frame_pair",
            side_effect=(None, RuntimeError("fallback failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "fallback failed"):
                expander._select_supplemental_pair(track, {("person", "walking_running"): 1.0})
        self.assertIs(expander.builder._mevis_frame_info, original)

    def test_pair_selector_labels_daily_and_moderate_primary_gates(self):
        track = expander.builder.MevisTrack("video", "17", (), (), ())
        pair = (1.0, object(), object(), {})
        expander._IMAGE_ROOT = Path("/unused")
        cases = (
            ({("object", "daily_sports_objects"): 1.0}, True, "daily_sports_custom"),
            ({("object", "vehicles"): 1.0}, False, "moderate"),
        )
        for classifications, relaxed, expected_rule in cases:
            with self.subTest(expected_rule=expected_rule):
                with patch.object(expander.builder, "select_frame_pair", return_value=pair) as select:
                    selected, rule = expander._select_supplemental_pair(track, classifications)
                self.assertIs(selected, pair)
                self.assertEqual(rule, expected_rule)
                self.assertEqual(select.call_args.kwargs["relaxed_daily_object"], relaxed)

    def test_supplemental_rows_emit_reviewable_relaxed_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_root = Path(tmpdir)
            video_root = image_root / "video"
            video_root.mkdir()
            Image.new("RGB", (40, 40), "red").save(video_root / "00000.jpg")
            Image.new("RGB", (40, 40), "green").save(video_root / "00009.jpg")
            rle = _rectangle_rle(height=40, width=40, top=10, left=10, bottom=30, right=30)
            track = expander.builder.MevisTrack(
                video_id="video",
                anno_id="17",
                frame_ids=("00000", "00009"),
                rles=(rle, rle),
                expressions=("a person walking", "the person standing"),
            )
            src = _frame_info(0, "00000", rle)
            tar = _frame_info(1, "00009", rle)
            evidence = {
                "temporal_gap_frames": 9,
                "temporal_gap_ratio": 1.0,
                "area_ratio_change": 1.0,
                "aspect_ratio_change": 1.0,
            }
            expander._TRACKS = [track]
            expander._IMAGE_ROOT = image_root
            classifications = {
                ("person", "walking_running"): 7.0,
                ("person", "standing_gesturing"): 6.0,
            }
            pixel = {
                "crop_mae": 0.04,
                "crop_phash": 6,
                "crop_dhash": 8,
                "frame_mae": 0.10,
                "frame_phash": 12,
            }
            with (
                patch.object(
                    expander.builder,
                    "classify_mevis_track",
                    return_value=classifications,
                ),
                patch.object(
                    expander,
                    "_select_supplemental_pair",
                    return_value=((10.0, src, tar, evidence), "supplemental_relaxed_geometry_v1"),
                ),
                patch.object(expander, "_pixel_delta", return_value=pixel),
                patch.object(
                    expander.builder,
                    "mevis_subject_votes",
                    return_value=Counter({"person": 3, "vehicles": 1}),
                ),
            ):
                rows = expander._supplemental_rows(0)

        self.assertEqual(
            [row["minor"] for row in rows],
            ["standing_gesturing", "walking_running"],
        )
        for row in rows:
            self.assertEqual(row["candidate_source"], "generic_supplement_relaxed_v1")
            self.assertEqual(row["selection_priority_penalty"], 5.0)
            self.assertEqual(row["identity_key"], "mevis_v2/train/video/17")
            self.assertEqual((row["src_frame_name"], row["tar_frame_name"]), ("00000", "00009"))
            self.assertEqual(row["pixel_delta_tier"], "strong")
            self.assertEqual(row["geometry_rule"], "supplemental_relaxed_geometry_v1")
            self.assertEqual(row["geometry"]["score"], 0.5)
            self.assertTrue(row["requires_visual_qc"])

    def test_supplemental_rows_enforce_pixel_floor_with_equality_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_root = Path(tmpdir)
            video_root = image_root / "video"
            video_root.mkdir()
            for frame_id in ("00000", "00009"):
                Image.new("RGB", (40, 40), "red").save(video_root / f"{frame_id}.jpg")
            rle = _rectangle_rle(height=40, width=40, top=10, left=10, bottom=30, right=30)
            track = expander.builder.MevisTrack("video", "17", ("00000", "00009"), (rle, rle), ("a car", "the car"))
            src, tar = _frame_info(0, "00000", rle), _frame_info(1, "00009", rle)
            evidence = {
                "temporal_gap_frames": 9,
                "temporal_gap_ratio": 1.0,
                "area_ratio_change": 1.0,
                "aspect_ratio_change": 1.0,
            }
            expander._TRACKS = [track]
            expander._IMAGE_ROOT = image_root
            common_patches = (
                patch.object(
                    expander.builder,
                    "classify_mevis_track",
                    return_value={("object", "vehicles"): 1.0},
                ),
                patch.object(
                    expander,
                    "_select_supplemental_pair",
                    return_value=((10.0, src, tar, evidence), "moderate"),
                ),
                patch.object(
                    expander.builder,
                    "mevis_subject_votes",
                    return_value=Counter({"vehicles": 2}),
                ),
            )
            pixels = (
                ({"crop_mae": 0.019999, "crop_phash": 64}, 0),
                ({"crop_mae": 1.0, "crop_phash": 1}, 0),
                ({"crop_mae": 0.02, "crop_phash": 2}, 1),
            )
            for pixel_fields, expected_count in pixels:
                pixel = {
                    **pixel_fields,
                    "crop_dhash": 2,
                    "frame_mae": 0.1,
                    "frame_phash": 2,
                }
                with self.subTest(pixel_fields=pixel_fields):
                    with ExitStack() as stack:
                        for candidate_patch in common_patches:
                            stack.enter_context(candidate_patch)
                        stack.enter_context(patch.object(expander, "_pixel_delta", return_value=pixel))
                        rows = expander._supplemental_rows(0)
                    self.assertEqual(len(rows), expected_count)
                    if rows:
                        self.assertEqual(rows[0]["pixel_delta_tier"], "base")
                        self.assertEqual(rows[0]["candidate_source"], "generic_supplement_v1")
                        self.assertEqual(rows[0]["selection_priority_penalty"], 4.0)

    def test_main_is_deterministic_and_records_v5_source_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mevis_root = root / "mevis"
            mevis_root.mkdir()
            (mevis_root / "JPEGImages.tar").write_bytes(b"pinned archive fixture")
            (mevis_root / "meta_expressions_v2.json").write_text("{}", encoding="utf-8")
            (mevis_root / "mask_dict.json").write_text("{}", encoding="utf-8")
            (mevis_root / "JPEGImages").mkdir()
            base_pool = root / "base.json"
            original_base = _base_payload([_minimal_base_row()])
            base_pool.write_text(json.dumps(original_base), encoding="utf-8")
            first, second = root / "first.json", root / "second.json"

            with (
                _patched_source_pins(base_pool, mevis_root),
                patch.object(expander.builder, "load_mevis_tracks", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    expander.main(
                        [
                            "--base-pool",
                            str(base_pool),
                            "--mevis-root",
                            str(mevis_root),
                            "--output",
                            str(first),
                            "--seed",
                            "123",
                            "--workers",
                            "1",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    expander.main(
                        [
                            "--base-pool",
                            str(base_pool),
                            "--mevis-root",
                            str(mevis_root),
                            "--output",
                            str(second),
                            "--seed",
                            "123",
                            "--workers",
                            "1",
                        ]
                    ),
                    0,
                )

            self.assertEqual(expander.builder.file_sha256(first), expander.builder.file_sha256(second))
            payload = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "mevis_diverse_candidate_pool_v5")
            self.assertEqual(payload["contract"]["seed"], 123)
            self.assertEqual(
                payload["contract"]["source_pins"],
                {
                    "jpeg_archive_sha256": expander.builder.file_sha256(mevis_root / "JPEGImages.tar"),
                    "meta_sha256": expander.builder.file_sha256(mevis_root / "meta_expressions_v2.json"),
                    "mask_dict_sha256": expander.builder.file_sha256(mevis_root / "mask_dict.json"),
                },
            )
            self.assertEqual(
                payload["counts"],
                {
                    "membership_rows": 1,
                    "unique_tracks": 1,
                    "unique_videos": 1,
                    "base_membership_rows": 1,
                    "supplemental_membership_rows": 0,
                    "by_minor": {"object/companion_animals": 1},
                    "pixel_tiers_by_minor": {"object/companion_animals": {"strong": 1}},
                },
            )
            self.assertEqual(payload["candidates"][0]["candidate_source"], "curated_v1")
            self.assertEqual(payload["candidates"][0]["selection_priority_penalty"], 0.0)
            self.assertEqual(json.loads(base_pool.read_text(encoding="utf-8")), original_base)

    def test_main_fails_closed_before_reading_unpinned_or_unsafe_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_pool = root / "base.json"
            base_pool.write_text(json.dumps(_base_payload()), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "workers must be positive"):
                expander.main(
                    [
                        "--base-pool",
                        str(base_pool),
                        "--mevis-root",
                        str(root),
                        "--output",
                        str(root / "out.json"),
                        "--workers",
                        "0",
                    ]
                )
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                expander.main(
                    [
                        "--base-pool",
                        str(base_pool),
                        "--mevis-root",
                        str(root),
                        "--output",
                        str(base_pool),
                        "--workers",
                        "1",
                    ]
                )

            with patch.object(expander, "BASE_POOL_SHA256", expander.builder.file_sha256(base_pool)):
                with self.assertRaisesRegex(FileNotFoundError, "Missing regular pinned MeViS source"):
                    expander.main(
                        [
                            "--base-pool",
                            str(base_pool),
                            "--mevis-root",
                            str(root),
                            "--output",
                            str(root / "missing.json"),
                            "--workers",
                            "1",
                        ]
                    )

                archive_target = root / "archive-target"
                archive_target.write_bytes(b"archive")
                (root / "JPEGImages.tar").symlink_to(archive_target)
                with self.assertRaisesRegex(FileNotFoundError, "Missing regular pinned MeViS source"):
                    expander.main(
                        [
                            "--base-pool",
                            str(base_pool),
                            "--mevis-root",
                            str(root),
                            "--output",
                            str(root / "symlink.json"),
                            "--workers",
                            "1",
                        ]
                    )

    def test_main_rejects_source_hash_mismatch_before_loading_tracks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_pool = root / "base.json"
            base_pool.write_text(json.dumps(_base_payload()), encoding="utf-8")
            (root / "JPEGImages.tar").write_bytes(b"wrong archive")
            with (
                patch.object(expander, "BASE_POOL_SHA256", expander.builder.file_sha256(base_pool)),
                patch.object(expander.builder, "MEVIS_ARCHIVE_SHA256", "0" * 64),
                patch.object(expander.builder, "load_mevis_tracks") as load_tracks,
            ):
                with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
                    expander.main(
                        [
                            "--base-pool",
                            str(base_pool),
                            "--mevis-root",
                            str(root),
                            "--output",
                            str(root / "out.json"),
                            "--workers",
                            "1",
                        ]
                    )
            load_tracks.assert_not_called()

    def test_main_rejects_missing_or_symlinked_extracted_image_tree(self):
        for symlink_tree in (False, True):
            with self.subTest(symlink_tree=symlink_tree), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                mevis_root = root / "mevis"
                mevis_root.mkdir()
                (mevis_root / "JPEGImages.tar").write_bytes(b"archive")
                (mevis_root / "meta_expressions_v2.json").write_text("{}", encoding="utf-8")
                (mevis_root / "mask_dict.json").write_text("{}", encoding="utf-8")
                if symlink_tree:
                    real_tree = root / "real-images"
                    real_tree.mkdir()
                    (mevis_root / "JPEGImages").symlink_to(real_tree, target_is_directory=True)
                base_pool = root / "base.json"
                base_pool.write_text(json.dumps(_base_payload()), encoding="utf-8")
                with (
                    _patched_source_pins(base_pool, mevis_root),
                    patch.object(expander.builder, "load_mevis_tracks", return_value=[]),
                ):
                    with self.assertRaisesRegex(FileNotFoundError, "Missing regular extracted MeViS image tree"):
                        expander.main(
                            [
                                "--base-pool",
                                str(base_pool),
                                "--mevis-root",
                                str(mevis_root),
                                "--output",
                                str(root / "out.json"),
                                "--workers",
                                "1",
                            ]
                        )


if __name__ == "__main__":
    unittest.main()

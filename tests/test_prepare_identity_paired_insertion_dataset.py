import copy
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts.prepare_identity_paired_insertion_dataset import (
    GARMENT_CATEGORY_NAMES,
    GARMENT_CATEGORY_TO_MINOR,
    GARMENT_NATIVE_SUBQUOTAS,
    MAJOR_MINOR_QUOTAS,
    PROTOCOL,
    DeepFashionItem,
    PairPlan,
    assign_case_ids,
    decode_coco_rle,
    deepfashion_polygon_to_mask,
    deepfashion_positive_pair,
    exclusive_build_lock,
    file_sha256,
    load_plan_cache,
    load_mevis_tracks,
    materialize_case,
    select_mevis_plans,
    validate_identity_evidence,
    write_plan_cache,
)


def _json_rle(values: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(values.astype(np.uint8)))
    return {
        "size": list(encoded["size"]),
        "counts": encoded["counts"].decode("ascii"),
    }


def _fashion_item(
    root: Path,
    *,
    image_id: str,
    source: str,
    pair_id: int = 17,
    style: int = 3,
    category_id: int = 1,
) -> DeepFashionItem:
    return DeepFashionItem(
        image_id=image_id,
        item_key="item1",
        source=source,
        pair_id=pair_id,
        style=style,
        category_id=category_id,
        category_name=GARMENT_CATEGORY_NAMES[category_id],
        bbox_xyxy=(10, 10, 50, 50),
        segmentation=((10.0, 10.0, 50.0, 10.0, 50.0, 50.0, 10.0, 50.0),),
        scale=3,
        occlusion=1,
        zoom_in=1,
        viewpoint=2,
        image_path=root / f"{image_id}.png",
    )


def _plan_template(**overrides) -> PairPlan:
    values = {
        "major": "object",
        "minor": "companion_animals",
        "leaf_label": "dog",
        "source_dataset": "mevis_v2",
        "pair_kind": "same_track_cross_frame",
        "identity_scope": "physical_instance",
        "identity_id": "mevis_v2/train/video/anno",
        "src_image_path": Path("src.png"),
        "tar_image_path": Path("tar.png"),
        "src_mask_payload": {},
        "tar_mask_payload": {},
        "src_image_id": "train/video/frame_src",
        "tar_image_id": "train/video/frame_tar",
        "src_ann_id": "anno",
        "tar_ann_id": "anno",
        "src_upstream": {
            "video_id": "video",
            "anno_id": "anno",
            "frame_id": "frame_src",
            "frame_index": 0,
        },
        "tar_upstream": {
            "video_id": "video",
            "anno_id": "anno",
            "frame_id": "frame_tar",
            "frame_index": 1,
        },
        "identity_evidence": {
            "type": "official_tracked_instance_id",
            "key_fields": ["video_id", "anno_id"],
        },
        "classification_evidence": "a dog",
        "selection_score": 1.0,
        "difficulty": "easy",
        "license_note": "test fixture",
    }
    values.update(overrides)
    return PairPlan(**values)


class IdentityPairedInsertionDatasetTest(unittest.TestCase):
    def test_coco_rle_and_deepfashion_multi_polygon_are_binary(self):
        expected = np.zeros((12, 14), dtype=np.uint8)
        expected[2:7, 3:10] = 1
        decoded = decode_coco_rle(_json_rle(expected))

        np.testing.assert_array_equal(
            np.asarray(decoded, dtype=np.uint8), expected * 255
        )
        self.assertEqual(decoded.size, (14, 12))

        polygon_mask = deepfashion_polygon_to_mask(
            (
                (2, 2, 8, 2, 8, 8, 2, 8),
                (12, 12, 18, 12, 18, 18, 12, 18),
            ),
            width=20,
            height=20,
        )
        values = np.asarray(polygon_mask, dtype=np.uint8)
        self.assertEqual(set(np.unique(values).tolist()), {0, 255})
        self.assertEqual(int(values[4, 4]), 255)
        self.assertEqual(int(values[14, 14]), 255)
        self.assertEqual(int(values[10, 10]), 0)
        self.assertGreater(int(np.count_nonzero(values)), 60)

    def test_deepfashion_positive_pair_requires_official_direction_and_identity(self):
        root = Path("/unused")
        src = _fashion_item(root, image_id="000001", source="shop")
        tar = _fashion_item(root, image_id="000002", source="user")
        self.assertTrue(deepfashion_positive_pair(src, tar))

        invalid_pairs = (
            (replace(src, source="user"), tar),
            (src, replace(tar, source="shop")),
            (src, replace(tar, image_id=src.image_id)),
            (src, replace(tar, pair_id=src.pair_id + 1)),
            (src, replace(tar, style=src.style + 1)),
            (replace(src, style=0), replace(tar, style=0)),
            (src, replace(tar, category_id=2)),
        )
        for candidate_src, candidate_tar in invalid_pairs:
            with self.subTest(src=candidate_src, tar=candidate_tar):
                self.assertFalse(deepfashion_positive_pair(candidate_src, candidate_tar))

    def test_all_thirteen_deepfashion_classes_map_to_balanced_minor_quotas(self):
        expected_mapping = {
            1: "short_sleeve_top",
            2: "long_sleeve_top",
            3: "outerwear_vest_sling",
            4: "outerwear_vest_sling",
            5: "outerwear_vest_sling",
            6: "outerwear_vest_sling",
            7: "shorts_trousers",
            8: "shorts_trousers",
            9: "skirt",
            10: "dress_onepiece",
            11: "dress_onepiece",
            12: "dress_onepiece",
            13: "dress_onepiece",
        }
        self.assertEqual(GARMENT_CATEGORY_TO_MINOR, expected_mapping)
        self.assertEqual(set(GARMENT_CATEGORY_NAMES), set(range(1, 14)))

        native_categories = Counter()
        for minor, native_subquotas in GARMENT_NATIVE_SUBQUOTAS.items():
            self.assertEqual(
                sum(native_subquotas.values()), MAJOR_MINOR_QUOTAS["garment"][minor]
            )
            for category_id, quota in native_subquotas.items():
                self.assertEqual(GARMENT_CATEGORY_TO_MINOR[category_id], minor)
                native_categories[category_id] += quota
        self.assertEqual(set(native_categories), set(range(1, 14)))
        self.assertEqual(sum(native_categories.values()), 300)

    def test_mevis_loader_deduplicates_paraphrases_per_track(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mask = np.zeros((12, 14), dtype=np.uint8)
            mask[2:8, 3:11] = 1
            rle = _json_rle(mask)
            metadata = {
                "videos": {
                    "video_1": {
                        "frames": ["00000", "00001"],
                        "expressions": {
                            "0": {"exp": "a dog", "anno_id": ["101"]},
                            "1": {"exp": "a dog", "anno_id": ["101"]},
                            "2": {"exp": "  the   running dog  ", "anno_id": [101]},
                            "3": {
                                "exp": "two objects together",
                                "anno_id": ["101", "102"],
                            },
                        },
                    }
                }
            }
            (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            (root / "masks.json").write_text(
                json.dumps({"101": [rle, rle]}), encoding="utf-8"
            )

            tracks = load_mevis_tracks(root / "meta.json", root / "masks.json")

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertEqual((track.video_id, track.anno_id), ("video_1", "101"))
        self.assertEqual(track.frame_ids, ("00000", "00001"))
        self.assertEqual(track.expressions, ("a dog", "the running dog"))
        self.assertEqual(track.identity_id, "mevis_v2/train/video_1/101")

    def test_mevis_selection_reflows_until_all_reference_and_target_frames_are_unique(self):
        shared_video = "shared_video"
        common = {
            "src_upstream": {
                "video_id": shared_video,
                "anno_id": "unused",
                "frame_id": "unused",
                "frame_index": 0,
            },
        }
        candidates = [
            _plan_template(
                **common,
                major="object",
                minor="object_kind",
                identity_id="identity/object",
                src_image_id=f"train/{shared_video}/src_object",
                tar_image_id=f"train/{shared_video}/shared_target",
                selection_score=10.0,
            ),
            _plan_template(
                **common,
                major="person",
                minor="person_kind",
                identity_id="identity/person_conflict",
                src_image_id=f"train/{shared_video}/src_person_conflict",
                tar_image_id=f"train/{shared_video}/shared_target",
                selection_score=9.0,
            ),
            _plan_template(
                **common,
                major="person",
                minor="person_kind",
                identity_id="identity/person_alternative",
                src_image_id=f"train/{shared_video}/src_person_alternative",
                tar_image_id=f"train/{shared_video}/unique_target",
                selection_score=8.0,
            ),
        ]
        tiny_quotas = {
            "object": {"object_kind": 1},
            "person": {"person_kind": 1},
            "garment": {},
        }

        with patch(
            "scripts.prepare_identity_paired_insertion_dataset.MAJOR_MINOR_QUOTAS",
            tiny_quotas,
        ):
            selected = select_mevis_plans(candidates, seed=20260725)

        self.assertEqual(
            {plan.identity_id for plan in selected},
            {"identity/object", "identity/person_alternative"},
        )
        selected_frames = [
            image_id
            for plan in selected
            for image_id in (plan.src_image_id, plan.tar_image_id)
        ]
        self.assertEqual(len(selected_frames), len(set(selected_frames)))

    def test_candidate_cache_is_deterministic_and_bound_to_raw_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            image_root = raw_root / "mevis_v2" / "train" / "JPEGImages" / "video"
            image_root.mkdir(parents=True)
            src_path = image_root / "frame_src.jpg"
            tar_path = image_root / "frame_tar.jpg"
            Image.new("RGB", (8, 8), "red").save(src_path)
            Image.new("RGB", (8, 8), "blue").save(tar_path)
            plan = _plan_template(src_image_path=src_path, tar_image_path=tar_path)
            first = root / "first.json.gz"
            second = root / "second.json.gz"

            for cache_path in (first, second):
                write_plan_cache(
                    cache_path,
                    seed=20260725,
                    raw_root=raw_root,
                    mevis_curation_pool_path=None,
                    mevis_candidates=[plan],
                    fashion_candidates=[],
                )

            self.assertEqual(file_sha256(first), file_sha256(second))
            loaded_mevis, loaded_fashion = load_plan_cache(
                first,
                seed=20260725,
                raw_root=raw_root,
                mevis_curation_pool_path=None,
            )
            self.assertEqual(len(loaded_mevis), 1)
            self.assertEqual(loaded_fashion, [])
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                load_plan_cache(
                    first,
                    seed=20260725,
                    raw_root=root / "different_raw_root",
                    mevis_curation_pool_path=None,
                )

    def test_output_build_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "candidate"
            with exclusive_build_lock(output):
                with self.assertRaisesRegex(FileExistsError, "Another build owns"):
                    with exclusive_build_lock(output):
                        self.fail("second lock must never be acquired")
            with exclusive_build_lock(output):
                self.assertTrue(True)

    def test_materialize_uses_red_reference_and_never_blue_target_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_path = root / "independent_red_reference.png"
            tar_path = root / "independent_blue_target.png"
            Image.new("RGB", (80, 80), (255, 0, 0)).save(src_path)
            Image.new("RGB", (100, 80), (0, 0, 255)).save(tar_path)
            src_polygon = ((20, 20, 60, 20, 60, 60, 20, 60),)
            tar_polygon = ((30, 20, 70, 20, 70, 60, 30, 60),)
            identity_id = "deepfashion2/validation/17/3"
            plan = _plan_template(
                major="garment",
                minor="short_sleeve_top",
                leaf_label="short_sleeve_top",
                source_dataset="deepfashion2",
                pair_kind="same_product_cross_domain",
                identity_scope="product_identity",
                identity_id=identity_id,
                src_image_path=src_path,
                tar_image_path=tar_path,
                src_mask_payload=src_polygon,
                tar_mask_payload=tar_polygon,
                src_image_id="validation/000001",
                tar_image_id="validation/000002",
                src_ann_id="000001/item1",
                tar_ann_id="000002/item1",
                src_upstream={
                    "image_id": "000001",
                    "item_key": "item1",
                    "pair_id": 17,
                    "style": 3,
                    "category_id": 1,
                    "domain": "shop",
                },
                tar_upstream={
                    "image_id": "000002",
                    "item_key": "item1",
                    "pair_id": 17,
                    "style": 3,
                    "category_id": 1,
                    "domain": "user",
                },
                identity_evidence={
                    "type": "official_consumer_shop_product_pair",
                    "key_fields": ["pair_id", "style"],
                },
                extra={"native_category_id": 1},
            )
            output = root / "dataset"

            with self.assertRaisesRegex(ValueError, "reserved manifest fields"):
                materialize_case(
                    replace(plan, extra={"protocol": "malicious_override"}),
                    "gar_short_sleeve_top_bad",
                    root / "bad_dataset",
                )

            record = materialize_case(plan, "gar_short_sleeve_top_000", output)

            with Image.open(output / record["src_image"]) as handle:
                src = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            with Image.open(output / record["src_mask"]) as handle:
                src_mask = np.asarray(handle.convert("L"), dtype=np.uint8) > 0
            with Image.open(output / record["tar_image"]) as handle:
                tar = np.asarray(handle.convert("RGB"), dtype=np.uint8)

        self.assertTrue(np.all(src[src_mask] == np.array([255, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.all(src[~src_mask] == 255))
        self.assertFalse(np.any(np.all(src == np.array([0, 0, 255]), axis=2)))
        self.assertTrue(np.all(tar == np.array([0, 0, 255], dtype=np.uint8)))
        self.assertFalse(record["reference_derived_from_target"])
        self.assertNotEqual(record["src_pixel_sha256"], record["tar_pixel_sha256"])

    def test_identity_validation_recomputes_upstream_evidence_and_rejects_mismatch(self):
        identity_id = "mevis_v2/train/video_1/101"
        good = {
            "case_id": "obj_companion_animals_000",
            "manifest_schema_version": 4,
            "protocol": PROTOCOL,
            "pair_kind": "same_track_cross_frame",
            "case_type": "paired_gt",
            "same_identity": True,
            "gt_available": True,
            "reference_derived_from_target": False,
            "source_dataset": "mevis_v2",
            "src_source_dataset": "mevis_v2",
            "source_split": "train",
            "src_source_split": "train",
            "identity_scope": "physical_instance",
            "identity_evidence_status": "verified_upstream_annotation",
            "identity_evidence": {"type": "official_tracked_instance_id"},
            "identity_id": identity_id,
            "src_identity_id": identity_id,
            "tar_identity_id": identity_id,
            "src_image_id": "train/video_1/00000",
            "tar_image_id": "train/video_1/00009",
            "src_ann_id": "101",
            "tar_ann_id": "101",
            "src_upstream": {
                "video_id": "video_1",
                "anno_id": "101",
                "frame_id": "00000",
            },
            "tar_upstream": {
                "video_id": "video_1",
                "anno_id": "101",
                "frame_id": "00009",
            },
        }
        self.assertEqual(validate_identity_evidence(good), [])

        bad = copy.deepcopy(good)
        bad["tar_upstream"]["anno_id"] = "different_instance"
        # A claimed boolean is insufficient: the official track keys must agree.
        bad["same_identity"] = True
        errors = validate_identity_evidence(bad)
        self.assertTrue(any("identity mismatch" in error for error in errors), errors)

        same_frame = copy.deepcopy(good)
        same_frame["tar_upstream"]["frame_id"] = "00000"
        self.assertTrue(
            any("frames must differ" in error for error in validate_identity_evidence(same_frame))
        )

        target_derived = copy.deepcopy(good)
        target_derived["reference_derived_from_target"] = True
        self.assertTrue(
            any(
                "reference_derived_from_target must be false" in error
                for error in validate_identity_evidence(target_derived)
            )
        )

    def test_quota_matrix_assigns_exactly_nine_hundred_unique_cases(self):
        self.assertEqual(set(MAJOR_MINOR_QUOTAS), {"object", "person", "garment"})
        self.assertEqual(
            {major: sum(minors.values()) for major, minors in MAJOR_MINOR_QUOTAS.items()},
            {"object": 300, "person": 300, "garment": 300},
        )
        self.assertEqual(sum(map(len, MAJOR_MINOR_QUOTAS.values())), 18)
        self.assertTrue(
            all(quota == 50 for minors in MAJOR_MINOR_QUOTAS.values() for quota in minors.values())
        )

        plans = []
        serial = 0
        for major, minors in MAJOR_MINOR_QUOTAS.items():
            for minor, quota in minors.items():
                for index in range(quota):
                    serial += 1
                    plans.append(
                        _plan_template(
                            major=major,
                            minor=minor,
                            identity_id=f"identity/{serial}",
                            selection_score=float(quota - index),
                        )
                    )

        assigned = assign_case_ids(plans)

        self.assertEqual(len(assigned), 900)
        self.assertEqual(len({case_id for case_id, _ in assigned}), 900)
        self.assertEqual(
            Counter((plan.major, plan.minor) for _, plan in assigned),
            Counter(
                {
                    (major, minor): quota
                    for major, minors in MAJOR_MINOR_QUOTAS.items()
                    for minor, quota in minors.items()
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()

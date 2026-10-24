import io
import json
import tempfile
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.prepare_paper_insertion_dataset import (
    Candidate,
    anyinsertion_clean_split_errors,
    assign_cross_donors,
    assign_cross_donors_from_pools,
    audit_dataset,
    binary_mask,
    candidate_paper_avoid_reasons,
    coco_segmentation_to_mask,
    convert_anyinsertion,
    freeze_showcase_candidates,
    file_sha256,
    generate_qc_contact_sheets,
    load_exclusion_config,
    load_fashion_candidates,
    main,
    person_minor_eligibility,
    person_minor_scores,
    refresh_anyinsertion_metadata,
    refresh_paper_avoid_metadata,
    safe_extract_zip,
    select_targets,
    select_person_targets_globally,
    write_checksums_atomic,
    write_anyinsertion_clean_split,
)


def encoded_png(image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": None}


class PreparePaperInsertionDatasetTest(unittest.TestCase):
    def test_default_exclusion_config_is_required(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "scripts.prepare_paper_insertion_dataset.DEFAULT_EXCLUSION_CONFIG",
            Path(tmpdir) / "missing.json",
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "required release exclusion config",
            ):
                load_exclusion_config()

    def test_coco_polygon_is_rasterized_as_binary_exact_shape(self):
        mask = coco_segmentation_to_mask([[10, 10, 30, 10, 30, 30, 10, 30]], 40, 40)

        values = set(np.unique(np.asarray(mask)).tolist())
        self.assertEqual(values, {0, 255})
        self.assertGreater(int(np.count_nonzero(np.asarray(mask))), 300)
        self.assertLess(int(np.count_nonzero(np.asarray(mask))), 500)

    def test_person_caption_evidence_overrides_contradictory_geometry(self):
        keypoints = [0.0] * (17 * 3)
        for index, (x, y) in {
            5: (120, 100),
            6: (160, 100),
            11: (125, 190),
            12: (155, 190),
            13: (120, 260),
            14: (160, 260),
            15: (115, 330),
            16: (165, 330),
        }.items():
            keypoints[index * 3 : index * 3 + 3] = [x, y, 2]

        scores = person_minor_scores(
            "A person is sitting on a bench.",
            (80, 60, 120, 300),
            keypoints,
            nearby_object_count=1,
        )

        self.assertGreater(scores["sitting_kneeling"], scores["standing_fullbody"])

    def test_standing_accepts_cross_side_fullbody_keypoint_evidence(self):
        keypoints = [0.0] * (17 * 3)
        # Visible joints establish the whole body, but no one side has a
        # complete hip-knee-ankle chain (a common COCO annotation pattern).
        for index, (x, y) in {
            0: (140, 60),
            5: (120, 100),
            6: (160, 100),
            9: (110, 175),
            10: (170, 175),
            11: (125, 190),
            12: (155, 190),
            13: (120, 260),
            16: (165, 335),
        }.items():
            keypoints[index * 3 : index * 3 + 3] = [x, y, 2]

        eligibility = person_minor_eligibility(
            "A person standing beside a wall.",
            (80, 40, 120, 320),
            0.18,
            keypoints,
        )

        self.assertTrue(eligibility["standing_fullbody"])
        self.assertFalse(eligibility["_standing_fullbody_fallback"])

    def test_static_sport_pose_is_marked_as_standing_fallback(self):
        keypoints = [0.0] * (17 * 3)
        for index, (x, y) in {
            0: (140, 60),
            5: (120, 100),
            6: (160, 100),
            11: (125, 190),
            12: (155, 190),
            13: (120, 260),
            14: (160, 260),
            15: (115, 330),
            16: (165, 330),
        }.items():
            keypoints[index * 3 : index * 3 + 3] = [x, y, 2]

        eligibility = person_minor_eligibility(
            "A tennis player stands still and poses on the court.",
            (80, 40, 120, 320),
            0.18,
            keypoints,
        )

        self.assertTrue(eligibility["standing_fullbody"])
        self.assertTrue(eligibility["_standing_fullbody_fallback"])

    def test_explicit_walking_caption_precedes_pose_only_fallback(self):
        keypoints = [0.0] * (17 * 3)
        for index, (x, y) in {
            5: (120, 100),
            6: (160, 100),
            11: (125, 190),
            12: (155, 190),
            13: (105, 250),
            14: (165, 260),
            15: (90, 325),
            16: (180, 330),
        }.items():
            keypoints[index * 3 : index * 3 + 3] = [x, y, 2]

        explicit = person_minor_eligibility(
            "A person walks across the street.",
            (80, 40, 140, 320),
            0.18,
            keypoints,
        )
        pose_only = person_minor_eligibility(
            "A tennis player reaches for the ball.",
            (80, 40, 140, 320),
            0.18,
            keypoints,
        )

        self.assertTrue(explicit["walking_running"])
        self.assertFalse(explicit["_walking_running_fallback"])
        self.assertTrue(pose_only["walking_running"])
        self.assertTrue(pose_only["_walking_running_fallback"])

    def test_possible_minor_scene_is_paper_only_not_hard_excluded(self):
        candidate = Candidate(
            dataset="coco2017",
            split="val2017",
            major="person",
            minor="standing_fullbody",
            leaf_label="person",
            image_id="17",
            ann_id="19",
            width=256,
            height=256,
            bbox_xywh=(40.0, 20.0, 100.0, 180.0),
            area=12000.0,
            payload={},
            caption="A little girl standing in a park.",
            quality_score=1.0,
        )

        self.assertEqual(
            candidate_paper_avoid_reasons(candidate, set()),
            ["caption_indicates_possible_minor_in_scene"],
        )
        candidate.caption = "An adult musician standing beside a keyboard."
        self.assertEqual(candidate_paper_avoid_reasons(candidate, set()), [])

    def test_fashion_exact_mask_parquet_is_required_and_parsed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir)
            parquet_path = raw_root / "fashionpedia" / "test_with_masks.parquet"
            parquet_path.parent.mkdir(parents=True)
            image = Image.new("RGB", (400, 400), color=(220, 220, 220))
            mask_array = np.zeros((400, 400), dtype=np.uint8)
            mask_array[100:300, 120:280] = 255
            mask = Image.fromarray(mask_array, mode="L")
            table = pa.Table.from_pylist(
                [
                    {
                        "image_id": 7,
                        "image": encoded_png(image),
                        "image_width": 400,
                        "image_height": 400,
                        "object_masks": [encoded_png(mask)],
                        "object_bboxes": [[120.0, 100.0, 280.0, 300.0]],
                        "object_category_ids": [10],
                        "object_attribute_ids": [[1, 7, 12]],
                    }
                ]
            )
            pq.write_table(table, parquet_path)

            pools, loaded_table = load_fashion_candidates(raw_root, seed=9)

            self.assertEqual(loaded_table.num_rows, 1)
            self.assertEqual(len(pools["garment/dress_onepiece"]), 1)
            candidate = pools["garment/dress_onepiece"][0]
            self.assertEqual(candidate.leaf_label, "dress")
            self.assertEqual(candidate.payload["mask"].size, (400, 400))

    def test_safe_zip_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../escape.txt", "bad")

            with self.assertRaisesRegex(ValueError, "Unsafe zip member"):
                safe_extract_zip(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())

    def test_safe_zip_replaces_same_size_crc_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/payload.bin", b"pinned-bytes")

            destination = root / "out"
            safe_extract_zip(archive, destination)
            extracted = destination / "nested" / "payload.bin"
            extracted.write_bytes(b"broken-byte!")
            self.assertEqual(extracted.stat().st_size, len(b"pinned-bytes"))

            safe_extract_zip(archive, destination)

            self.assertEqual(extracted.read_bytes(), b"pinned-bytes")

    def test_showcase_pool_balances_leaf_diversity(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "case_id": f"cross_instance_replace_{index:02d}",
                    "major": "object",
                    "minor": "animal",
                    "leaf_label": ("cat", "dog", "horse")[index % 3],
                    "protocol": "cross_instance_replace",
                    "selection_score": 1.0 - index / 100.0,
                    "showcase_candidate": True,
                }
            )

        freeze_showcase_candidates(records, per_minor=10)

        selected = [row for row in records if row["showcase_candidate"]]
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            sum(row["protocol"] == "cross_instance_replace" for row in selected),
            10,
        )
        leaves = {row["leaf_label"] for row in selected}
        self.assertEqual(leaves, {"cat", "dog", "horse"})

    def test_showcase_pool_never_uses_agent_paper_avoid_cases(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "case_id": f"cross_instance_replace_{index}",
                    "major": "object",
                    "minor": "vehicle",
                    "leaf_label": "bus",
                    "protocol": "cross_instance_replace",
                    "selection_score": 1.0 - index / 10.0,
                    "agent_paper_avoid": index < 2,
                }
            )

        freeze_showcase_candidates(records, per_minor=10)

        selected = [row for row in records if row["showcase_candidate"]]
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(not row["agent_paper_avoid"] for row in selected))

    def test_refresh_paper_avoid_metadata_preserves_caption_flags(self):
        records = [
            {
                "case_id": "per_example_000",
                "src_source_dataset": "coco2017",
                "src_image_id": "old-configured",
                "source_dataset": "coco2017",
                "tar_image_id": "new-configured",
                "agent_paper_avoid": True,
                "agent_paper_avoid_roles": ["src"],
                "agent_paper_avoid_reasons": {
                    "src": [
                        "agent_visual_qc_image_flag",
                        "caption_indicates_possible_minor_in_scene",
                    ]
                },
            }
        ]

        refresh_paper_avoid_metadata(
            records,
            {("coco2017", "new-configured")},
        )

        self.assertTrue(records[0]["agent_paper_avoid"])
        self.assertEqual(records[0]["agent_paper_avoid_roles"], ["src", "tar"])
        self.assertEqual(
            records[0]["agent_paper_avoid_reasons"],
            {
                "src": ["caption_indicates_possible_minor_in_scene"],
                "tar": ["agent_visual_qc_image_flag"],
            },
        )

    def test_showcase_pool_reports_quality_by_selecting_fewer_not_avoided_cases(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "case_id": f"cross_instance_replace_{index}",
                    "major": "person",
                    "minor": "upperbody_portrait",
                    "leaf_label": "person",
                    "protocol": "cross_instance_replace",
                    "selection_score": 1.0 - index / 10.0,
                    "agent_paper_avoid": index >= 8,
                }
            )

        freeze_showcase_candidates(records, per_minor=10)

        selected = [row for row in records if row["showcase_candidate"]]
        self.assertEqual(len(selected), 8)
        self.assertTrue(all(not row["agent_paper_avoid"] for row in selected))

    def test_showcase_pool_skips_large_match_gaps_when_coherent_cases_exist(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "case_id": f"cross_instance_replace_{index}",
                    "major": "person",
                    "minor": "standing_fullbody",
                    "leaf_label": "person",
                    "protocol": "cross_instance_replace",
                    "selection_score": 1.0,
                    "showcase_score": 10.0 if index == 0 else 1.0 - index / 10.0,
                    "agent_paper_avoid": False,
                    "challenge_flags": ["large_cross_match_gap"] if index == 0 else [],
                }
            )

        freeze_showcase_candidates(records, per_minor=10)

        selected = [row for row in records if row["showcase_candidate"]]
        self.assertEqual(len(selected), 10)
        self.assertTrue(
            all("large_cross_match_gap" not in row["challenge_flags"] for row in selected)
        )

    def test_showcase_pool_selects_fewer_instead_of_promoting_large_match_gaps(self):
        records = []
        for index in range(12):
            records.append(
                {
                    "case_id": f"cross_instance_replace_{index}",
                    "major": "person",
                    "minor": "walking_running",
                    "leaf_label": "person",
                    "protocol": "cross_instance_replace",
                    "selection_score": 1.0 - index / 10.0,
                    "showcase_score": 1.0 - index / 10.0,
                    "agent_paper_avoid": False,
                    "challenge_flags": (
                        ["large_cross_match_gap"] if index >= 8 else []
                    ),
                }
            )

        freeze_showcase_candidates(records, per_minor=10)

        selected = [row for row in records if row["showcase_candidate"]]
        self.assertEqual(len(selected), 8)
        self.assertTrue(
            all("large_cross_match_gap" not in row["challenge_flags"] for row in selected)
        )

    def test_target_selection_uses_paper_avoid_only_as_fallback(self):
        def candidate(image_id: str) -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="object",
                minor="animal",
                leaf_label="cat",
                image_id=image_id,
                ann_id=f"{image_id}:0",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=1.0,
            )

        avoided = candidate("best-but-avoided")
        clean = candidate("clean")
        selected = select_targets(
            [avoided, clean],
            1,
            set(),
            set(),
            max_per_leaf=2,
            deprioritized_images={("coco2017", avoided.image_id)},
        )
        self.assertEqual([row.image_id for row in selected], ["clean"])

        fallback = select_targets(
            [avoided, clean],
            2,
            set(),
            set(),
            max_per_leaf=2,
            deprioritized_images={("coco2017", avoided.image_id)},
        )
        self.assertEqual([row.image_id for row in fallback], ["clean", "best-but-avoided"])

    def test_person_targets_are_allocated_globally_across_overlapping_strata(self):
        def candidate(image_id: str, minor: str, score: float = 1.0) -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="person",
                minor=minor,
                leaf_label="person",
                image_id=image_id,
                ann_id=f"{image_id}:{minor}",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=score,
            )

        # Greedily taking shared-a/shared-b for alpha would strand beta.  The
        # joint assignment must give alpha its private images instead.
        pools = {
            "person/alpha": [
                candidate("shared-a", "alpha", 3.0),
                candidate("shared-b", "alpha", 2.9),
                candidate("alpha-a", "alpha", 1.0),
                candidate("alpha-b", "alpha", 0.9),
            ],
            "person/beta": [
                candidate("shared-a", "beta", 2.0),
                candidate("shared-b", "beta", 1.9),
            ],
        }
        used: set[tuple[str, str]] = set()
        selected = select_person_targets_globally(
            pools,
            {"alpha": 2, "beta": 2},
            used,
            set(),
            seed=7,
        )

        self.assertEqual({row.image_id for row in selected["beta"]}, {"shared-a", "shared-b"})
        self.assertEqual({row.image_id for row in selected["alpha"]}, {"alpha-a", "alpha-b"})
        self.assertEqual(len(used), 4)

    def test_cross_donor_batch_matching_deranges_exhausted_leaf(self):
        def candidate(image_id: str) -> Candidate:
            return Candidate(
                dataset="fashionpedia",
                split="test",
                major="garment",
                minor="dress_onepiece",
                leaf_label="jumpsuit",
                image_id=image_id,
                ann_id=f"{image_id}:0",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        targets = [candidate(str(index)) for index in range(5)]
        counts = Counter()
        donors = assign_cross_donors(targets, targets, counts, seed=7)

        self.assertEqual(len({donor.image_id for donor in donors}), 5)
        self.assertTrue(all(donor.image_id != target.image_id for donor, target in zip(donors, targets)))
        self.assertEqual(set(counts.values()), {1})

    def test_global_cross_matching_reroutes_a_flexible_target(self):
        def candidate(image_id: str) -> Candidate:
            return Candidate(
                dataset="coco",
                split="val2017",
                major="object",
                minor="animal",
                leaf_label="cat",
                image_id=image_id,
                ann_id=f"{image_id}:0",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        flexible_target = candidate("target-flexible")
        constrained_target = candidate("target-constrained")
        donor_a = candidate("donor-a")
        donor_b = candidate("donor-b")
        counts = Counter()

        donors = assign_cross_donors_from_pools(
            [flexible_target, constrained_target],
            [[donor_a, donor_b], [donor_b]],
            counts,
            seed=11,
        )

        self.assertEqual(donors[1].image_id, "donor-b")
        self.assertEqual(donors[0].image_id, "donor-a")
        self.assertEqual(set(counts.values()), {1})

    def test_global_cross_matching_uses_minimum_cost_multiway_exchange(self):
        def candidate(image_id: str) -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="person",
                minor="upperbody_portrait",
                leaf_label="person",
                image_id=image_id,
                ann_id=image_id,
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        targets = [candidate(name) for name in ("target-a", "target-b", "target-c")]
        donors = [candidate(name) for name in ("donor-x", "donor-y", "donor-z")]
        costs = {
            ("target-a", "donor-x"): 10.0,
            ("target-a", "donor-y"): 0.0,
            ("target-a", "donor-z"): 100.0,
            ("target-b", "donor-x"): 100.0,
            ("target-b", "donor-y"): 10.0,
            ("target-b", "donor-z"): 0.0,
            ("target-c", "donor-x"): 0.0,
            ("target-c", "donor-y"): 100.0,
            ("target-c", "donor-z"): 10.0,
        }
        with patch(
            "scripts.prepare_paper_insertion_dataset.cross_donor_match_score",
            side_effect=lambda target, donor, seed: costs[(target.ann_id, donor.ann_id)],
        ):
            assigned = assign_cross_donors_from_pools(
                targets,
                [donors, donors, donors],
                Counter(),
                seed=5,
            )

        self.assertEqual([row.image_id for row in assigned], ["donor-y", "donor-z", "donor-x"])

    def test_cross_donor_dynamic_paper_avoid_is_used_only_as_fallback(self):
        def candidate(image_id: str, caption: str = "") -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="person",
                minor="upperbody_portrait",
                leaf_label="person",
                image_id=image_id,
                ann_id=image_id,
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
                caption=caption,
            )

        target = candidate("target")
        child_scene = candidate("child-scene", "A child standing near a tree")
        clean = candidate("clean")
        with patch(
            "scripts.prepare_paper_insertion_dataset.cross_donor_match_score",
            side_effect=lambda _, donor, __: 0.0 if donor.image_id == "child-scene" else 0.8,
        ):
            selected = assign_cross_donors_from_pools(
                [target],
                [[child_scene, clean]],
                Counter(),
                seed=5,
                paper_avoid_image_ids=set(),
            )
        self.assertEqual(selected[0].image_id, "clean")

        fallback = assign_cross_donors_from_pools(
            [target],
            [[child_scene]],
            Counter(),
            seed=5,
            paper_avoid_image_ids=set(),
        )
        self.assertEqual(fallback[0].image_id, "child-scene")

        # A paper-only donor is preferable to making the metric case a severe
        # geometry mismatch.  It remains excluded from showcase selection.
        with patch(
            "scripts.prepare_paper_insertion_dataset.cross_donor_match_score",
            side_effect=lambda _, donor, __: 0.8 if donor.image_id == "child-scene" else 2.0,
        ):
            bounded_difficulty = assign_cross_donors_from_pools(
                [target],
                [[child_scene, clean]],
                Counter(),
                seed=5,
                paper_avoid_image_ids=set(),
            )
        self.assertEqual(bounded_difficulty[0].image_id, "child-scene")

    def test_cross_donor_hall_failure_is_atomic(self):
        def candidate(image_id: str) -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="object",
                minor="animal",
                leaf_label="cat",
                image_id=image_id,
                ann_id=image_id,
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        targets = [candidate(f"target-{index}") for index in range(3)]
        donor_a, donor_b, donor_c = [candidate(name) for name in ("a", "b", "c")]
        counts = Counter({("coco2017", "already-used"): 1})
        original = counts.copy()
        with self.assertRaisesRegex(ValueError, "No complete unique-donor matching"):
            assign_cross_donors_from_pools(
                targets,
                [[donor_a], [donor_a], [donor_a, donor_b, donor_c]],
                counts,
                seed=5,
            )
        self.assertEqual(counts, original)

    def test_cross_donor_rejects_nonfinite_scores(self):
        target = Candidate(
            dataset="coco2017",
            split="val2017",
            major="object",
            minor="animal",
            leaf_label="cat",
            image_id="target",
            ann_id="target",
            width=256,
            height=256,
            bbox_xywh=(40.0, 20.0, 100.0, 180.0),
            area=12000.0,
            payload={},
        )
        donor = Candidate(**{**target.__dict__, "image_id": "donor", "ann_id": "donor"})
        with patch(
            "scripts.prepare_paper_insertion_dataset.cross_donor_match_score",
            return_value=float("nan"),
        ):
            with self.assertRaisesRegex(ValueError, "score must be finite"):
                assign_cross_donors_from_pools(
                    [target],
                    [[donor]],
                    Counter(),
                    seed=5,
                )

    def test_person_target_hall_failure_does_not_mutate_used_images(self):
        def candidate(image_id: str, minor: str) -> Candidate:
            return Candidate(
                dataset="coco2017",
                split="val2017",
                major="person",
                minor=minor,
                leaf_label="person",
                image_id=image_id,
                ann_id=f"{minor}:{image_id}",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        pools = {
            "person/alpha": [candidate("x", "alpha"), candidate("y", "alpha")],
            "person/beta": [candidate("x", "beta"), candidate("y", "beta")],
            "person/gamma": [
                candidate("z", "gamma"),
                candidate("w", "gamma"),
                candidate("v", "gamma"),
            ],
        }
        used = {("coco2017", "preexisting")}
        with self.assertRaisesRegex(ValueError, "No feasible globally unique person target"):
            select_person_targets_globally(
                pools,
                {"alpha": 2, "beta": 2, "gamma": 1},
                used,
                set(),
                seed=5,
            )
        self.assertEqual(used, {("coco2017", "preexisting")})

    def test_targets_never_exceed_cross_donor_image_capacity(self):
        def candidate(image_id: str, leaf: str) -> Candidate:
            return Candidate(
                dataset="coco",
                split="val2017",
                major="object",
                minor="sport_tool_toy",
                leaf_label=leaf,
                image_id=image_id,
                ann_id=f"{image_id}:0",
                width=256,
                height=256,
                bbox_xywh=(40.0, 20.0, 100.0, 180.0),
                area=12000.0,
                payload={},
                quality_score=0.8,
            )

        single_image_leaf = candidate("lonely-0", "scissors")
        three_image_leaf = [candidate(f"tri-{index}", "frisbee") for index in range(3)]
        common = [candidate(f"common-{index}", "sports ball") for index in range(12)]
        pool = [single_image_leaf, *three_image_leaf, *common]

        # A leaf with a single image can never supply a cross-image donor,
        # so it is skipped; other leaves are capped at their distinct image
        # count because every target needs a unique donor image.
        selected = select_targets(pool, 14, set(), set(), max_per_leaf=10)
        self.assertEqual(len(selected), 14)
        self.assertNotIn("scissors", {row.leaf_label for row in selected})
        self.assertEqual(sum(row.leaf_label == "frisbee" for row in selected), 3)

        # Asking for more targets than the total donor-image capacity fails
        # instead of silently breaking the cross-image donor invariant.
        with self.assertRaises(ValueError):
            select_targets(pool, 16, set(), set(), max_per_leaf=10)

    def test_strict_audit_rejects_cross_case_from_the_same_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_id = "obj_animal_000"
            paths = {}
            for directory in ("src_image", "src_mask", "tar_image", "tar_mask"):
                path = root / "object" / directory / f"{case_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                paths[directory] = str(path.relative_to(root))
                if "mask" in directory:
                    array = np.zeros((64, 64), dtype=np.uint8)
                    array[16:48, 16:48] = 255
                    Image.fromarray(array, mode="L").save(path)
                else:
                    Image.new("RGB", (64, 64), color=(128, 90, 60)).save(path)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            record = {
                "case_id": case_id,
                "major": "object",
                "minor": "animal",
                "leaf_label": "cat",
                "src_leaf_label": "cat",
                "source_dataset": "coco",
                "src_source_dataset": "coco",
                "protocol": "cross_instance_replace",
                "same_identity": False,
                "gt_available": False,
                "mask_origin_src": "native_instance",
                "mask_origin_tar": "native_instance",
                "src_image_id": "shared-image",
                "tar_image_id": "shared-image",
                "src_ann_id": "ann-a",
                "tar_ann_id": "ann-b",
                "src_area_ratio": 0.25,
                "tar_area_ratio": 0.25,
                "agent_paper_avoid": "yes",
                "agent_paper_avoid_roles": [["src"]],
                "agent_paper_avoid_reasons": {"src": ["unknown_reason"]},
                **paths,
            }
            (manifest_dir / "paperinsert_900.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            with patch(
                "scripts.prepare_paper_insertion_dataset.load_excluded_image_ids",
                return_value={("coco", "shared-image")},
            ), self.assertRaises(ValueError) as raised:
                audit_dataset(root, strict_900=True)
            self.assertIn("invalid agent_paper_avoid metadata", str(raised.exception))
            self.assertIn("invalid agent_paper_avoid reasons", str(raised.exception))
            self.assertIn("cross protocol must use a different source image", str(raised.exception))
            self.assertIn("hard-excluded src image remains in manifest", str(raised.exception))
            self.assertIn("hard-excluded tar image remains in manifest", str(raised.exception))

    def test_anyinsertion_audit_checks_four_file_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            case_id = "sample"
            paths = {}
            for directory in ("src_image", "src_mask", "tar_image", "tar_mask"):
                path = root / "object" / directory / f"{case_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                paths[directory] = str(path.relative_to(root))
                if "mask" in directory:
                    array = np.zeros((64, 64), dtype=np.uint8)
                    array[16:48, 16:48] = 255
                    binary_mask(Image.fromarray(array, mode="L")).save(path)
                else:
                    Image.new("RGB", (64, 64), color=(128, 90, 60)).save(path)
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            record = {
                "case_id": case_id,
                "major": "object",
                "minor": "official_object",
                "protocol": "official_anyinsertion",
                **paths,
            }
            (manifest_dir / "anyinsertion_official.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            report = audit_dataset(root, strict_900=False, finalize_checksums=True)
            qc_report = generate_qc_contact_sheets(root, [record])

            self.assertTrue(report["valid"])
            self.assertEqual(report["record_count"], 1)
            self.assertTrue((root / "CHECKSUMS.sha256").is_file())
            self.assertEqual(qc_report["record_count"], 1)
            self.assertEqual(qc_report["sheet_count"], 1)
            self.assertTrue((root / qc_report["sheets"][0]["path"]).is_file())

            # Unsigned nested content used to disappear from the flat orphan
            # inventory and leave a strict release looking valid.
            nested = root / "object" / "src_image" / "unsigned"
            nested.mkdir()
            (nested / "payload.bin").write_bytes(b"unsigned")
            with self.assertRaisesRegex(
                ValueError,
                "Unsigned non-regular or nested canonical content",
            ):
                audit_dataset(root, strict_900=False)
            (nested / "payload.bin").unlink()
            nested.rmdir()

            # A normal audit verifies the original evidence and must never
            # replace it with checksums computed from a tampered manifest.
            checksum_path = root / "CHECKSUMS.sha256"
            original_checksums = checksum_path.read_text(encoding="utf-8")
            manifest_path = manifest_dir / "anyinsertion_official.jsonl"
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8").replace("sample", "tamper"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                audit_dataset(root, strict_900=False)
            self.assertEqual(checksum_path.read_text(encoding="utf-8"), original_checksums)

    def test_anyinsertion_converter_reads_labels_and_excludes_bbox_only_group(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw = root / "raw" / "anyinsertion"
            exact_group = raw / "object"
            bbox_group = raw / "person" / "complex_scene"
            image = Image.new("RGB", (64, 64), color=(100, 120, 140))
            mask_array = np.zeros((64, 64), dtype=np.uint8)
            mask_array[12:52, 18:46] = 255
            mask = Image.fromarray(mask_array, mode="L")
            for stem in ("0", "5", "34"):
                for directory, value in (
                    ("ref_image", image),
                    ("ref_mask", mask),
                    ("tar_image", image),
                    ("tar_mask", mask),
                ):
                    path = exact_group / directory / f"{stem}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.save(path)
            (exact_group / "label.json").write_text(
                json.dumps({"0.png": "vase", "34.png": "shoe"}), encoding="utf-8"
            )
            # The official complex-person group has src_mask rather than a
            # contour tar_mask; it must not enter the exact-mask supplement.
            for directory, value in (
                ("ref_image", image),
                ("ref_mask", mask),
                ("src_image", image),
                ("src_mask", mask),
            ):
                path = bbox_group / directory / "0.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                value.save(path)

            records = convert_anyinsertion(
                root / "raw",
                root / "converted",
                verify_source_snapshot=False,
            )

            self.assertEqual(len(records), 3)
            by_stem = {row["original_stem"]: row for row in records}
            self.assertEqual(by_stem["0"]["leaf_label"], "vase")
            self.assertEqual(by_stem["0"]["case_type"], "paired_gt")
            self.assertTrue(by_stem["0"]["gt_available"])
            self.assertFalse(by_stem["0"]["official_clean_source_available"])
            self.assertEqual(by_stem["5"]["case_type"], "open_insertion")
            self.assertFalse(by_stem["5"]["gt_available"])
            self.assertFalse(by_stem["5"]["same_identity"])
            self.assertEqual(by_stem["34"]["original_leaf_label"], "shoe")
            self.assertEqual(by_stem["34"]["leaf_label"], "backpack")
            self.assertIsNotNone(by_stem["34"]["label_override_reason"])

            # Reviewed overrides can also repair metadata on an already
            # materialized conversion without rewriting image pixels.
            manifest = root / "converted" / "manifests" / "anyinsertion_official.jsonl"
            stale = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            for row in stale:
                if row["original_stem"] == "5":
                    row.update({"gt_available": True, "case_type": "paired_gt", "same_identity": True})
                if row["original_stem"] == "34":
                    row.update({"leaf_label": "shoe", "label_override_reason": None})
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in stale),
                encoding="utf-8",
            )
            # Refresh accepts only a fully authenticated schema-v3 tree: the
            # canonical assets, primary manifest, and conversion provenance
            # must all match the current checksum inventory.
            audit_dataset(
                root / "converted",
                strict_900=False,
                finalize_checksums=True,
            )
            canonical_path = root / "converted" / stale[0]["src_image"]
            original_bytes = canonical_path.read_bytes()
            canonical_path.write_bytes(original_bytes + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum verification"):
                refresh_anyinsertion_metadata(root / "converted")
            canonical_path.write_bytes(original_bytes)
            refreshed = refresh_anyinsertion_metadata(root / "converted")
            refreshed_by_stem = {row["original_stem"]: row for row in refreshed}
            self.assertFalse(refreshed_by_stem["5"]["gt_available"])
            self.assertEqual(refreshed_by_stem["34"]["leaf_label"], "backpack")

            report = json.loads(
                (root / "converted" / "reports" / "conversion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["counts_major"], {"object": 3})
            self.assertEqual(report["counts_case_type"], {"open_insertion": 1, "paired_gt": 2})
            self.assertEqual(report["label_override_count"], 1)
            self.assertEqual(report["excluded_groups"][0]["group"], "person/complex_scene")
            self.assertIn("tar_mask", report["excluded_groups"][0]["missing_directories"])

    def test_anyinsertion_clean67_is_pinned_ordered_primary_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifests" / "anyinsertion_official.jsonl"
            manifest.parent.mkdir(parents=True)
            records = []
            for major, group, count in (
                ("garment", "garment", 30),
                ("object", "object", 40),
                ("person", "person/simple_scene", 30),
            ):
                for stem in range(count):
                    records.append(
                        {
                            "case_id": f"ai_{major[:3]}_{stem:03d}",
                            "major": major,
                            "original_group": group,
                            "original_stem": str(stem),
                            "case_type": (
                                "open_insertion"
                                if (group, str(stem)) == ("object", "5")
                                else "paired_gt"
                            ),
                        }
                    )
            manifest.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                encoding="utf-8",
            )

            clean = write_anyinsertion_clean_split(root)

            self.assertEqual(len(clean), 67)
            self.assertEqual(
                Counter(row["major"] for row in clean),
                {"garment": 27, "object": 38, "person": 2},
            )
            self.assertEqual(anyinsertion_clean_split_errors(root, records), [])
            ids_path = root / "manifests" / "anyinsertion_clean67_case_ids.txt"
            ids_path.write_text(ids_path.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
            self.assertTrue(
                any(
                    "case-ID sidecar" in error
                    for error in anyinsertion_clean_split_errors(root, records)
                )
            )

    def test_strict_anyinsertion_release_covers_provenance_and_clean_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_root = root / "raw"
            any_root = raw_root / "anyinsertion"
            image = Image.new("RGB", (16, 16), color=(80, 110, 140))
            mask_array = np.zeros((16, 16), dtype=np.uint8)
            mask_array[3:13, 4:12] = 255
            mask = Image.fromarray(mask_array, mode="L")

            for group, count in (
                ("garment", 30),
                ("object", 40),
                ("person/simple_scene", 30),
            ):
                group_root = any_root / group
                for stem in range(count):
                    for directory, value in (
                        ("ref_image", image),
                        ("ref_mask", mask),
                        ("tar_image", image),
                        ("tar_mask", mask),
                    ):
                        path = group_root / directory / f"{stem}.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        value.save(path)
                clean_source_count = 22 if group == "garment" else 16 if group == "object" else 0
                for stem in range(clean_source_count):
                    path = group_root / "src_image" / f"{stem}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(path)

            complex_root = any_root / "person" / "complex_scene"
            for stem in range(30):
                for directory, value in (("ref_image", image), ("ref_mask", mask)):
                    path = complex_root / directory / f"{stem}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.save(path)

            output_root = root / "converted"
            records = convert_anyinsertion(
                raw_root,
                output_root,
                verify_source_snapshot=False,
            )
            self.assertEqual(len(records), 100)
            conversion_path = output_root / "reports" / "conversion.json"
            conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
            conversion["source_snapshot"]["verified"] = True
            conversion_path.write_text(
                json.dumps(conversion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_anyinsertion_clean_split(output_root)

            report = audit_dataset(
                output_root,
                strict_900=False,
                strict_anyinsertion=True,
                finalize_checksums=True,
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["checksum_entry_count"], 405)
            checksum_text = (output_root / "CHECKSUMS.sha256").read_text(encoding="utf-8")
            self.assertIn("reports/conversion.json", checksum_text)
            self.assertIn("manifests/anyinsertion_clean67_for_current_model.jsonl", checksum_text)
            self.assertEqual(
                (output_root / "CHECKSUMS.sha256").stat().st_mode & 0o777,
                0o644,
            )
            verified = audit_dataset(
                output_root,
                strict_900=False,
                strict_anyinsertion=True,
            )
            self.assertTrue(verified["checksum_verified"])

            conversion["identity_review_method"] = "tampered"
            conversion_path.write_text(
                json.dumps(conversion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "conversion provenance mismatch"):
                audit_dataset(
                    output_root,
                    strict_900=False,
                    strict_anyinsertion=True,
                    finalize_checksums=True,
                )

    def test_checksum_writer_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checksum_path = Path(tmpdir) / "CHECKSUMS.sha256"
            with self.assertRaisesRegex(ValueError, "Unsafe checksum path"):
                write_checksums_atomic(
                    checksum_path,
                    [("0" * 64, "../outside.png")],
                )
            self.assertFalse(checksum_path.exists())

    def test_qc_rejects_symlinked_report_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            external = root / "external"
            external.mkdir()
            (dataset_root / "reports").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinked output directory"):
                generate_qc_contact_sheets(dataset_root, [])
            self.assertEqual(list(external.iterdir()), [])

    def test_anyinsertion_refresh_rejects_symlinked_manifest_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            external = root / "external_manifests"
            external.mkdir()
            manifest = external / "anyinsertion_official.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            (dataset_root / "manifests").symlink_to(external, target_is_directory=True)
            before = manifest.read_bytes()

            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                refresh_anyinsertion_metadata(dataset_root)

            self.assertEqual(manifest.read_bytes(), before)

    def test_anyinsertion_refresh_cli_honors_output_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate"
            raw_root = root / "raw"
            with patch(
                "scripts.prepare_paper_insertion_dataset.refresh_anyinsertion_metadata",
                return_value=[],
            ) as refresh, patch(
                "scripts.prepare_paper_insertion_dataset.write_anyinsertion_clean_split",
                return_value=[],
            ), patch(
                "scripts.prepare_paper_insertion_dataset.audit_dataset",
                return_value={"case_types": {}},
            ):
                self.assertEqual(
                    main(
                        [
                            "refresh-anyinsertion-metadata",
                            "--output-root",
                            str(candidate),
                            "--raw-root",
                            str(raw_root),
                        ]
                    ),
                    0,
                )
            refresh.assert_called_once_with(candidate.resolve(), raw_root=raw_root.resolve())


if __name__ == "__main__":
    unittest.main()

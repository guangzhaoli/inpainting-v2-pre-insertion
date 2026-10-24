import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from scripts import prepare_identity_paired_insertion_dataset as builder
from scripts import render_daily_sports_pool_review as renderer


def _rle(mask: np.ndarray) -> dict[str, object]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": list(encoded["size"]), "counts": encoded["counts"]}


def _plan(root: Path, serial: int, *, minor: str = renderer.MINOR) -> builder.PairPlan:
    video_id = f"video_{serial}"
    src_path = root / f"src_{serial}.jpg"
    tar_path = root / f"tar_{serial}.jpg"
    Image.new("RGB", (32, 24), (40 + serial, 80, 120)).save(src_path)
    Image.new("RGB", (32, 24), (80, 40 + serial, 120)).save(tar_path)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:20, 8:25] = 1
    return builder.PairPlan(
        major=renderer.MAJOR,
        minor=minor,
        leaf_label="daily_objects",
        source_dataset="mevis_v2",
        pair_kind="same_track_cross_frame",
        identity_scope="physical_instance",
        identity_id=f"mevis_v2/train/{video_id}/{serial}",
        src_image_path=src_path,
        tar_image_path=tar_path,
        src_mask_payload=_rle(mask),
        tar_mask_payload=_rle(mask),
        src_image_id=f"train/{video_id}/00001",
        tar_image_id=f"train/{video_id}/00009",
        src_ann_id=str(serial),
        tar_ann_id=str(serial),
        src_upstream={
            "video_id": video_id,
            "anno_id": str(serial),
            "frame_id": "00001",
            "frame_index": 1,
        },
        tar_upstream={
            "video_id": video_id,
            "anno_id": str(serial),
            "frame_id": "00009",
            "frame_index": 9,
        },
        identity_evidence={"type": "official_tracked_instance_id"},
        classification_evidence="the ball",
        selection_score=10.0 + serial,
        difficulty="easy",
        license_note="test",
        extra={
            "curation_candidate_id": f"candidate_{serial}",
            "curation_candidate_source": "curated_v1",
            "selection_priority_penalty": 0.0,
        },
    )


class DailyPoolRendererTests(unittest.TestCase):
    def test_backup_filter_is_identity_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plans = [_plan(root, serial) for serial in range(3)]
            plans.append(_plan(root, 8, minor="vehicles"))
            backups, counts = renderer.select_daily_backup_plans(
                plans,
                excluded_identities={plans[0].identity_id},
                selected_identities={plans[1].identity_id},
            )
            self.assertEqual(
                [plan.identity_id for plan in backups], [plans[2].identity_id]
            )
            self.assertEqual(
                counts,
                {
                    "category_candidate_count": 3,
                    "excluded_category_candidate_count": 1,
                    "nonexcluded_category_candidate_count": 2,
                    "currently_selected_category_candidate_count": 1,
                    "backup_candidate_count": 1,
                },
            )

    def test_duplicate_daily_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = _plan(Path(temporary), 1)
            with self.assertRaisesRegex(ValueError, "not identity-unique"):
                renderer.select_daily_backup_plans(
                    [plan, plan],
                    excluded_identities=set(),
                    selected_identities=set(),
                )

    def test_row_and_tile_show_exact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = _plan(Path(temporary), 3)
            row = renderer._plan_review_row(
                plan,
                rank=1,
                selected_cases_by_video={"video_3": ["selected_case"]},
                selected_cases_by_frame={plan.src_image_id: ["selected_case"]},
            )
            self.assertEqual(row["candidate_id"], "candidate_3")
            self.assertEqual(row["identity_id"], "mevis_v2/train/video_3/3")
            self.assertEqual(row["candidate_source"], "curated_v1")
            self.assertEqual(row["src_frame"]["frame_id"], "00001")
            self.assertEqual(row["tar_frame"]["frame_id"], "00009")
            self.assertTrue(row["selected_frame_overlap"])
            tile = renderer._review_tile(plan, row)
            self.assertEqual(tile.size, renderer.TILE_SIZE)


if __name__ == "__main__":
    unittest.main()

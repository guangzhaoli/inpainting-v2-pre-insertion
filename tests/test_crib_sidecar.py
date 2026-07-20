"""Unit tests for standalone CRIB sidecar pipeline (no DINOv2 download required)."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.crib.cluster import cluster_same_id
from script.crib.config import CribBuildConfig
from script.crib.features import l2_normalize, prepare_masked_ref_crop
from script.crib.hard_neg import mine_hard_negatives
from script.crib.io_utils import load_manifest, make_sample_uid
from script.crib.neighbors import knn_search
from script.crib.pipeline import run_pipeline_from_arrays


class MaskedCropTests(unittest.TestCase):
    def test_masked_crop_keeps_foreground_and_fills_background(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        image[20:60, 30:80] = (10, 200, 30)  # green object
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[20:60, 30:80] = 255

        crop, area, bbox, fallback = prepare_masked_ref_crop(
            image,
            mask,
            image_size=64,
            bbox_pad_ratio=0.0,
        )
        self.assertEqual(crop.shape, (64, 64, 3))
        self.assertFalse(fallback)
        self.assertGreater(area, 0.05)
        # object bbox roughly preserved before resize
        self.assertEqual(bbox, (30, 20, 80, 60))
        # background fill is grayish, object remains green-ish in center
        center = crop[32, 32]
        self.assertGreater(int(center[1]), 100)

    def test_empty_mask_falls_back_to_full_frame(self):
        image = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)
        mask = np.zeros((40, 50), dtype=np.uint8)
        crop, area, bbox, fallback = prepare_masked_ref_crop(
            image,
            mask,
            image_size=32,
            min_mask_pixels=16,
        )
        self.assertTrue(fallback)
        self.assertEqual(area, 0.0)
        self.assertEqual(crop.shape, (32, 32, 3))


class NeighborAndClusterTests(unittest.TestCase):
    def _synthetic_embeddings(self):
        # 6 samples in 8D:
        # 0,1 nearly identical (same id)
        # 2 similar to 0 but below same-id threshold (hard neg)
        # 3,4 another near-dup pair
        # 5 far away
        rng = np.random.default_rng(0)
        base_a = l2_normalize(rng.normal(size=(8,)).astype(np.float32))
        base_b = l2_normalize(rng.normal(size=(8,)).astype(np.float32))
        noise = rng.normal(scale=0.01, size=(8,)).astype(np.float32)

        e0 = base_a
        e1 = l2_normalize(base_a + noise)
        e2 = l2_normalize(base_a + 0.35 * rng.normal(size=8).astype(np.float32))
        e3 = base_b
        e4 = l2_normalize(base_b + noise)
        e5 = l2_normalize(rng.normal(size=(8,)).astype(np.float32))
        emb = np.stack([e0, e1, e2, e3, e4, e5], axis=0)
        return emb

    def test_knn_excludes_self_and_orders_by_similarity(self):
        emb = self._synthetic_embeddings()
        idx, scores = knn_search(emb, topk=3, use_faiss=False)
        self.assertEqual(idx.shape, (6, 3))
        # self should not appear
        for i in range(6):
            self.assertNotIn(i, idx[i].tolist())
        # sample 0's nearest should be 1 (near-dup)
        self.assertEqual(int(idx[0, 0]), 1)
        self.assertGreater(float(scores[0, 0]), 0.9)

    def test_faiss_and_numpy_agree_on_top1(self):
        emb = self._synthetic_embeddings()
        idx_np, sc_np = knn_search(emb, topk=3, use_faiss=False)
        try:
            idx_f, sc_f = knn_search(emb, topk=3, use_faiss=True)
        except Exception:
            self.skipTest("faiss not available")
        np.testing.assert_array_equal(idx_np[:, 0], idx_f[:, 0])
        np.testing.assert_allclose(sc_np[:, 0], sc_f[:, 0], atol=1e-5)

    def test_same_id_clustering_merges_near_dups_only(self):
        emb = self._synthetic_embeddings()
        idx, scores = knn_search(emb, topk=5, use_faiss=False)
        uids = [f"acc/{i}" for i in range(6)]
        cats = ["accessory"] * 6
        result = cluster_same_id(
            idx,
            scores,
            uids,
            cats,
            sim_threshold=0.95,
            require_mutual=False,
        )
        # 0 and 1 should share instance; 3 and 4 should share
        self.assertEqual(result.instance_ids[0], result.instance_ids[1])
        self.assertEqual(result.instance_ids[3], result.instance_ids[4])
        self.assertNotEqual(result.instance_ids[0], result.instance_ids[3])
        # 2 should usually stay separate from 0 under high threshold
        self.assertNotEqual(result.instance_ids[0], result.instance_ids[2])

    def test_hard_neg_excludes_same_instance_and_respects_sim_band(self):
        emb = self._synthetic_embeddings()
        idx, scores = knn_search(emb, topk=5, use_faiss=False)
        uids = [f"acc/{i}" for i in range(6)]
        cats = ["accessory"] * 6
        cluster = cluster_same_id(idx, scores, uids, cats, sim_threshold=0.95)
        hard = mine_hard_negatives(
            idx,
            scores,
            uids,
            cats,
            cluster.instance_ids,
            topk=3,
            min_sim=0.50,
            max_sim=0.95,
            same_category_only=True,
        )
        # for sample 0, hard neg must not include same-instance member (acc/1)
        self.assertNotIn("acc/1", hard.hard_neg_uids[0])
        # if any hard neg present, scores in band
        for sc in hard.hard_neg_scores[0]:
            self.assertGreaterEqual(sc, 0.50)
            self.assertLess(sc, 0.95)


class PipelineIntegrationTests(unittest.TestCase):
    def test_run_pipeline_from_arrays_end_to_end(self):
        rng = np.random.default_rng(1)
        # two tight clusters + fillers
        c1 = l2_normalize(rng.normal(size=(4,)).astype(np.float32))
        c2 = l2_normalize(rng.normal(size=(4,)).astype(np.float32))
        emb = []
        for _ in range(3):
            emb.append(l2_normalize(c1 + 0.01 * rng.normal(size=4).astype(np.float32)))
        for _ in range(3):
            emb.append(l2_normalize(c2 + 0.01 * rng.normal(size=4).astype(np.float32)))
        for _ in range(4):
            emb.append(l2_normalize(rng.normal(size=4).astype(np.float32)))
        emb = np.stack(emb, axis=0)
        uids = [f"accessory/{i}" for i in range(len(emb))]
        cats = ["accessory"] * len(emb)

        cfg = CribBuildConfig(
            neighbor_topk=5,
            same_id_sim_threshold=0.95,
            hard_neg_topk=3,
            hard_neg_min_sim=0.2,
            hard_neg_max_sim=0.95,
            use_faiss=False,
        )
        result = run_pipeline_from_arrays(emb, uids, cats, cfg)
        self.assertEqual(len(result.sidecar_rows), len(emb))
        self.assertIn("same_id", result.stats)
        self.assertIn("hard_neg", result.stats)
        # cluster 0 should have size >=2
        self.assertGreaterEqual(result.cluster.cluster_sizes[result.instance_ids[0]], 2)
        # hard neg of sample 0 should not include other same-id members
        same_group = {
            uids[i]
            for i, iid in enumerate(result.instance_ids)
            if iid == result.instance_ids[0]
        }
        for neg in result.hard_neg_uids[0]:
            self.assertNotIn(neg, same_group)

    def test_config_rejects_inconsistent_thresholds(self):
        cfg = CribBuildConfig(hard_neg_max_sim=0.99, same_id_sim_threshold=0.95)
        with self.assertRaises(ValueError):
            cfg.validate()


class ManifestIOTests(unittest.TestCase):
    def test_load_manifest_and_uid(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "ref_image",
                        "ref_mask",
                        "tar_image",
                        "tar_mask",
                        "category",
                        "stem",
                        "prompt",
                    ],
                )
                w.writeheader()
                w.writerow(
                    {
                        "ref_image": "accessory/ref_image/0.png",
                        "ref_mask": "accessory/ref_mask/0.png",
                        "tar_image": "accessory/tar_image/0.png",
                        "tar_mask": "accessory/tar_mask/0.png",
                        "category": "accessory",
                        "stem": "0",
                        "prompt": "",
                    }
                )
                w.writerow(
                    {
                        "ref_image": "jewel_filter/ref_image/1.png",
                        "ref_mask": "jewel_filter/ref_mask/1.png",
                        "tar_image": "jewel_filter/tar_image/1.png",
                        "tar_mask": "jewel_filter/tar_mask/1.png",
                        "category": "jewel_filter",
                        "stem": "1",
                        "prompt": "",
                    }
                )
            rows = load_manifest(path, categories=["accessory"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].sample_uid, "accessory/0")
            self.assertEqual(make_sample_uid("a", "b"), "a/b")


class RealDataSmokeOptional(unittest.TestCase):
    """Optional tiny real-data checks; skipped if dataset missing."""

    DATA_ROOT = Path(
        "/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data"
    )
    MANIFEST = REPO_ROOT / "manifests/insert_anything_teacher_accessory_jewelry.csv"

    def test_manifest_paths_exist_for_first_rows(self):
        if not self.MANIFEST.is_file() or not self.DATA_ROOT.is_dir():
            self.skipTest("dataset/manifest not available")
        rows = load_manifest(self.MANIFEST, limit=5)
        for r in rows:
            for key in ("ref_image", "ref_mask"):
                path = r.resolve(self.DATA_ROOT, key)
                self.assertTrue(path.is_file(), msg=f"missing {path}")


if __name__ == "__main__":
    unittest.main()

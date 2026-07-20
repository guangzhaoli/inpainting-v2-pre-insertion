# CRIB Sidecar Builder (MVP)

Standalone offline pipeline to build:

- `instance_id` (conservative near-dup / same-id clusters)
- `hard_neg_uids` (same-category appearance decoys)
- review packs for human QC

**Does not touch training code.** Outputs a sidecar you can join later.

## What DINOv2 / FAISS do here

- **DINOv2**: encode *masked reference appearance* into a vector  
  (identity-ish similarity, **not** category classification)
- **FAISS / numpy KNN**: retrieve nearest neighbors by cosine similarity
- **Rules**:
  - sim ≥ `same_id_threshold` (default 0.95) → merge into same `instance_id`
  - `hard_neg_min_sim` ≤ sim < `hard_neg_max_sim` and different instance → hard decoy

## Install deps

```bash
conda activate qwen-inpainting-pre-insertion
pip install faiss-cpu   # or faiss-gpu if available
# DINOv2 is loaded via torch.hub: facebookresearch/dinov2
```

First run will download DINOv2 weights from torch.hub.

## Smoke test (64 samples, CPU)

```bash
python -m script.crib.build_crib_sidecar \
  --dataset-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data \
  --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \
  --output-dir outputs/crib/accessory_jewelry_smoke \
  --limit 64 \
  --device cpu \
  --batch-size 8
```

## Full accessory/jewelry

```bash
python -m script.crib.build_crib_sidecar \
  --dataset-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data \
  --manifest manifests/insert_anything_teacher_accessory_jewelry.csv \
  --output-dir outputs/crib/accessory_jewelry \
  --device cuda \
  --batch-size 32
```

## Outputs

```text
output_dir/
  embeddings.npy
  sample_uids.json
  sidecar.jsonl          # main training join file
  instance_index.json
  neighbors.npz
  stats.json
  config.json
  summary.json
  review/
    hard_neg_pairs.csv
    hard_neg_images/
    same_id_clusters.csv
    same_id_images/
    top1_neighbors.csv
```

### `sidecar.jsonl` fields

| field | meaning |
|-------|---------|
| `sample_uid` | `category/stem` |
| `instance_id` | conservative identity cluster |
| `same_id_group` | all uids in that cluster |
| `hard_neg_uids` | up to 8 decoy sample uids |
| `hard_neg_scores` | cosine sims of those decoys |
| `mask_area_ratio` | ref mask fill ratio |

## Recommended thresholds (2026-07-13 multi-scale sweep)

Validated on 256 / 512 / 1024 / **1961 strided full-manifest** samples.

| param | **recommended** | role |
|-------|-----------------|------|
| `same_id_sim_threshold` | **0.97** | near-dup / same-id merge |
| `require_mutual_same_id` | **false** | little gain at thr≥0.96 |
| `hard_neg_min_sim` | **0.70** (fallback 0.65) | ignore weak neighbors |
| `hard_neg_max_sim` | **0.97** (= same_id) | never take near-dups as neg |
| `hard_neg_topk` | **8** | decoys stored per sample |
| `neighbor_topk` | 50 | KNN list length |

### Why these numbers?

| thr | observed issue |
|----:|----------------|
| 0.95 | 1024-run: size-11 different sunglasses merged |
| 0.96 | 1961-run: no 11+, but same-brand green vs tortoise frames still merged (pairwise 0.95–0.98) |
| **0.97** | splits wrong colorway merges; keeps true near-dups; hard-neg cov unchanged (~84%) |

Different colorways should stay as **hard-negs**, not same-id — over-merging wastes the best decoys.

### Hard-neg band

| min_sim | cov @1961 | note |
|--------:|----------:|------|
| 0.65 | ~90% | higher coverage, easier negs |
| **0.70** | **~84%** | **recommended default** |
| 0.75 | ~lower | too strict |

### 8-GPU full run

```bash
# edit DATA_ROOT if needed
bash script/crib/run_8gpu_full.sh

# or override:
SAME_ID=0.97 HARD_MIN=0.70 HARD_MAX=0.97 bash script/crib/run_8gpu_full.sh
```

See also `script/crib/merge_embedding_shards.py` and `outputs/crib/param_search_final.md`.

## Unit tests

```bash
conda activate qwen-inpainting-pre-insertion
python -m pytest tests/test_crib_sidecar.py -q
```

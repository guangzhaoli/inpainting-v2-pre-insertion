#!/usr/bin/env bash
# Full accessory/jewelry CRIB sidecar on 8 GPUs.
# Stage-1: per-GPU embedding shards
# Stage-2: merge + cluster + hard-neg (single process)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data}"
MANIFEST="${MANIFEST:-manifests/insert_anything_teacher_accessory_jewelry.csv}"
OUT_ROOT="${OUT_ROOT:-outputs/crib/accessory_jewelry_full}"
NUM_SHARDS="${NUM_SHARDS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
# Recommended defaults from multi-scale sweeps + human QC (256/512/1024/1961):
#   same_id=0.97, hard_neg=[0.70, 0.97), mutual=0
#   (0.96 still merges same-brand different sunglasses colorways)
# Fallback if hard-neg coverage <75% after full run: HARD_MIN=0.65
SAME_ID="${SAME_ID:-0.97}"
HARD_MIN="${HARD_MIN:-0.70}"
HARD_MAX="${HARD_MAX:-0.97}"
REQUIRE_MUTUAL="${REQUIRE_MUTUAL:-0}"  # 0/1
MODEL="${MODEL:-dinov2_vits14}"

# Cap intra-op threads so NUM_SHARDS concurrent workers do not thrash the host CPU.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export CV_NUM_THREADS="${CV_NUM_THREADS:-1}"

mkdir -p "$OUT_ROOT/shards" "$OUT_ROOT/logs"

echo "[stage-1] launching $NUM_SHARDS embedding shards"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m script.crib.build_crib_sidecar \
    --dataset-root "$DATA_ROOT" \
    --manifest "$MANIFEST" \
    --output-dir "$OUT_ROOT/shards/shard_$i" \
    --shard-index "$i" \
    --num-shards "$NUM_SHARDS" \
    --embeddings-only \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --model-name "$MODEL" \
    >"$OUT_ROOT/logs/shard_${i}.log" 2>&1 &
  echo "  started shard $i on GPU $i"
done
wait
echo "[stage-1] all shards finished"

SHARD_DIRS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIRS+=("$OUT_ROOT/shards/shard_$i")
done

MUTUAL_FLAG="--no-require-mutual-same-id"
if [[ "$REQUIRE_MUTUAL" == "1" ]]; then
  MUTUAL_FLAG="--require-mutual-same-id"
fi

echo "[stage-2] merge + mine with same_id=$SAME_ID hard=[$HARD_MIN,$HARD_MAX) mutual=$REQUIRE_MUTUAL"
python -m script.crib.merge_embedding_shards \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --output-dir "$OUT_ROOT" \
  --dataset-root "$DATA_ROOT" \
  --manifest "$MANIFEST" \
  --same-id-threshold "$SAME_ID" \
  --hard-neg-min-sim "$HARD_MIN" \
  --hard-neg-max-sim "$HARD_MAX" \
  $MUTUAL_FLAG \
  --hard-neg-topk 8 \
  --neighbor-topk 50 \
  --review-pairs 160 \
  >"$OUT_ROOT/logs/merge.log" 2>&1

echo "done: $OUT_ROOT"
echo "sidecar: $OUT_ROOT/sidecar.jsonl"
echo "stats:   $OUT_ROOT/stats.json"

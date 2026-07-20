#!/usr/bin/env bash
# Launch accessory/jewelry full CRIB on 8 GPUs (for zmx).
set -euo pipefail

cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen-inpainting-pre-insertion

export DATA_ROOT=/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data
export MANIFEST=manifests/insert_anything_teacher_accessory_jewelry.csv
export OUT_ROOT=outputs/crib/accessory_jewelry_full
export NUM_SHARDS=8
export BATCH_SIZE=128
export SAME_ID=0.97
export HARD_MIN=0.70
export HARD_MAX=0.97
export REQUIRE_MUTUAL=0
export MODEL=dinov2_vits14

# Avoid 8-way OpenCV/BLAS/OpenMP thread oversubscription (was thrashing ~300 load).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export CV_NUM_THREADS=1

echo "=== CRIB accessory_jewelry_full start $(date -Is) ==="
echo "python: $(which python)"
echo "DATA_ROOT=$DATA_ROOT"
echo "MANIFEST=$MANIFEST"
echo "OUT_ROOT=$OUT_ROOT"
echo "BATCH_SIZE=$BATCH_SIZE MODEL=$MODEL"
nvidia-smi -L

bash script/crib/run_8gpu_full.sh

echo "=== CRIB accessory_jewelry_full done $(date -Is) ==="

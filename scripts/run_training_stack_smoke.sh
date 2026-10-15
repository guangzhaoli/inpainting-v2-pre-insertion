#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /inspire/hdd/global_user/250041225002/uv_config/env.sh
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${ROOT}/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
exec uv run --offline --no-sync torchrun --nnodes="${WORLD_SIZE:-1}" --node_rank="${RANK:-0}" \
    --nproc_per_node=8 --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
    scripts/smoke_training_stack.py --output "${SMOKE_OUTPUT:?SMOKE_OUTPUT required}"

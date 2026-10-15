#!/usr/bin/env bash
set -euo pipefail
MANIFEST="${1:?manifest required}"
DOMAIN="${2:?domain required}"
source /inspire/hdd/global_user/250041225002/uv_config/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$(dirname "${MANIFEST}")"
CODE_DIR="${RUN_DIR}/code"
cd "${CODE_DIR}"
export NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
export MACHINE_RANK="${MACHINE_RANK:-${RANK:-${NODE_RANK:-0}}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TOTAL_PROCESSES="${TOTAL_PROCESSES:-$((NUM_MACHINES * NPROC_PER_NODE))}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-19001}}"
export ACCELERATE_CONFIG="${CODE_DIR}/accelerate_config_8.yaml"
export CONFIG_FILE="${CODE_DIR}/configs/${DOMAIN}.yaml"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export UV_BIN=/inspire/hdd/global_user/250041225002/uv_config/bin/uv
export PYTHONPATH="${CODE_DIR}"
export LD_LIBRARY_PATH="${CODE_DIR}/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export INSERTFUSE_DOMAIN="${DOMAIN}" INSERTFUSE_RUN_DIR="${RUN_DIR}"
finish() {
    local code=$?
    trap - EXIT
    export INSERTFUSE_EXIT_CODE="${code}"
    uv run --offline --no-sync python -c 'import os,json,datetime,pathlib; d={"exit_code":int(os.environ["INSERTFUSE_EXIT_CODE"]),"ended_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}; p=pathlib.Path(os.environ["INSERTFUSE_RUN_DIR"])/(os.environ["INSERTFUSE_DOMAIN"]+".exit.json"); p.write_text(json.dumps(d))' || true
    exit "${code}"
}
trap finish EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv
set +e
bash "${CODE_DIR}/run.sh" "${CONFIG_FILE}" 2>&1 | tee -a "${RUN_DIR}/${DOMAIN}.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
exit "${TRAIN_EXIT}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

count_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local devices="${CUDA_VISIBLE_DEVICES// /}"
        if [[ "${devices}" != "all" && "${devices}" != "none" ]]; then
            awk -F',' 'NF {print NF; exit}' <<<"${devices}"
            return 0
        fi
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
        return 0
    fi

    echo "ERROR: NPROC_PER_NODE is not set and GPU count could not be detected." >&2
    return 1
}

MASTER_ADDR="${MASTER_ADDR:-${MAIN_PROCESS_IP:-}}"
MASTER_PORT="${MASTER_PORT:-${MAIN_PROCESS_PORT:-19001}}"
WORLD_SIZE="${WORLD_SIZE:-${NUM_MACHINES:-1}}"
RANK="${RANK:-${MACHINE_RANK:-${NODE_RANK:-0}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_gpus)}"
TOTAL_PROCESSES="${TOTAL_PROCESSES:-${NUM_PROCESSES:-$((WORLD_SIZE * NPROC_PER_NODE))}}"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/configs/insert_anything_${TOTAL_PROCESSES}.yaml"
USER_CONFIG_ARG="${1:-}"
if [[ -n "${USER_CONFIG_ARG}" && -z "${CONFIG_FILE:-}" ]]; then
    CONFIG_FILE="${USER_CONFIG_ARG}"
    shift
fi
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"

if [[ -z "${MASTER_ADDR}" ]]; then
    if [[ "${WORLD_SIZE}" -eq 1 ]]; then
        MASTER_ADDR="127.0.0.1"
    else
        echo "ERROR: MASTER_ADDR is empty. In DLC PyTorchJob, MASTER_ADDR should be auto-injected." >&2
        echo "Set MASTER_ADDR manually if your environment did not inject it." >&2
        exit 1
    fi
fi

export NUM_MACHINES="${WORLD_SIZE}"
export MACHINE_RANK="${RANK}"
export MAIN_PROCESS_IP="${MASTER_ADDR}"
export MAIN_PROCESS_PORT="${MASTER_PORT}"
export NPROC_PER_NODE
export TOTAL_PROCESSES
export CONFIG_FILE
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ -z "${ACCELERATE_CONFIG:-}" ]]; then
    export ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config_${TOTAL_PROCESSES}.yaml"
fi

echo "=================================================="
echo "Detected DLC-style distributed environment"
echo "DLC_PLATFORM: ${DLC_PLATFORM:-<unset>}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "TOTAL_PROCESSES: ${TOTAL_PROCESSES}"
echo "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=================================================="

exec "${SCRIPT_DIR}/run.sh" "$@"

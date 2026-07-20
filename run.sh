#!/usr/bin/env bash
set -euo pipefail

# Distributed worker aborts can otherwise write one multi-GiB core per rank.
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USER_CONFIG_ARG="${1:-}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-19001}"
TOTAL_PROCESSES="${TOTAL_PROCESSES:-$((NUM_MACHINES * NPROC_PER_NODE))}"
RESUME="${RESUME:-}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/configs/scale/${TOTAL_PROCESSES}.yaml"
CONFIG_FILE="${CONFIG_FILE:-${USER_CONFIG_ARG:-${DEFAULT_CONFIG_FILE}}}"

DEFAULT_ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config_${TOTAL_PROCESSES}.yaml"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

require_file() {
    local label="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${label} not found: ${path}" >&2
        echo "Set ${label} explicitly or create the file matching TOTAL_PROCESSES=${TOTAL_PROCESSES}." >&2
        exit 1
    fi
}

require_file "CONFIG_FILE" "${CONFIG_FILE}"
require_file "ACCELERATE_CONFIG" "${ACCELERATE_CONFIG}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "=================================================="
echo "Launching training from: ${SCRIPT_DIR}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}"
echo "MIXED_PRECISION: ${MIXED_PRECISION}"
echo "NUM_MACHINES: ${NUM_MACHINES}"
echo "MACHINE_RANK: ${MACHINE_RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "TOTAL_PROCESSES: ${TOTAL_PROCESSES}"
echo "MAIN_PROCESS_IP: ${MAIN_PROCESS_IP}"
echo "MAIN_PROCESS_PORT: ${MAIN_PROCESS_PORT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CORE_DUMP_LIMIT: $(ulimit -c)"
echo "=================================================="

LAUNCH_ARGS=(
    --config_file "${ACCELERATE_CONFIG}"
    --num_processes "${TOTAL_PROCESSES}"
    --mixed_precision "${MIXED_PRECISION}"
    --main_process_port "${MAIN_PROCESS_PORT}"
)

if [[ "${NUM_MACHINES}" -gt 1 ]]; then
    LAUNCH_ARGS+=(
        --num_machines "${NUM_MACHINES}"
        --machine_rank "${MACHINE_RANK}"
        --main_process_ip "${MAIN_PROCESS_IP}"
    )
fi

TRAIN_ARGS=(
    -m src.main
    --config "${CONFIG_FILE}"
)

if [[ -n "${RESUME}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME}")
fi

CMD=(accelerate launch "${LAUNCH_ARGS[@]}" "${TRAIN_ARGS[@]}")

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'Resolved command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"

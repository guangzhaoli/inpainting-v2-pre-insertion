#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MASTER_PORT="${MASTER_PORT:-19001}"
RANK="${RANK:-${NODE_RANK:-${1:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-${MAIN_PROCESS_IP:-127.0.0.1}}"
WORLD_SIZE="${WORLD_SIZE:-${NUM_MACHINES:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_PROCESSES="${TOTAL_PROCESSES:-$((WORLD_SIZE * NPROC_PER_NODE))}"
DEFAULT_CONFIG_FILE="${SCRIPT_DIR}/configs/insert_anything_${TOTAL_PROCESSES}.yaml"
DEFAULT_ACCELERATE_CONFIG="${SCRIPT_DIR}/accelerate_config_${TOTAL_PROCESSES}.yaml"

export NUM_MACHINES="${WORLD_SIZE}"
export MACHINE_RANK="${RANK}"
export MAIN_PROCESS_IP="${MASTER_ADDR}"
export MAIN_PROCESS_PORT="${MASTER_PORT}"
export NPROC_PER_NODE
export TOTAL_PROCESSES
export CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "=================================================="
echo "Parsing multi-node environment variables..."
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "RANK: ${RANK}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "TOTAL_PROCESSES: ${TOTAL_PROCESSES}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}"
echo "=================================================="

exec "${SCRIPT_DIR}/run.sh"

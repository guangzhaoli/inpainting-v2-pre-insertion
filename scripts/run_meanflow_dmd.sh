#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="${1:?Usage: run_meanflow_dmd.sh CONFIG.yaml}"
: "${NUM_MACHINES:?Set allocated NUM_MACHINES explicitly; evaluate 2/4/8 nodes}"
: "${MACHINE_RANK:?Set a distinct MACHINE_RANK on every allocated node}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TOTAL_PROCESSES="${TOTAL_PROCESSES:-$((NUM_MACHINES * NPROC_PER_NODE))}"
if (( NUM_MACHINES < 1 || NUM_MACHINES > 8 || NPROC_PER_NODE != 8 || TOTAL_PROCESSES != NUM_MACHINES * NPROC_PER_NODE )); then
    echo "Expected 1-8 allocated nodes, 8 GPUs per node, and TOTAL_PROCESSES=NUM_MACHINES*8" >&2
    exit 2
fi
if (( MACHINE_RANK < 0 || MACHINE_RANK >= NUM_MACHINES )); then
    echo "MACHINE_RANK must be in [0, NUM_MACHINES)" >&2
    exit 2
fi
if (( NUM_MACHINES > 1 )); then
    : "${MAIN_PROCESS_IP:?Set the same coordinator address on every node}"
fi
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_DIR}/accelerate_config_8.yaml}"
exec bash "${REPO_DIR}/run.sh" "${CONFIG}"

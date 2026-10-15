#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN="${1:?Usage: run_teacher_v2.sh accessory_jewelry|animal|fashion_tryon|general_object|human}"
case "${DOMAIN}" in
  accessory_jewelry|animal|fashion_tryon|general_object|human) ;;
  *) echo "Unknown teacher domain: ${DOMAIN}" >&2; exit 2 ;;
esac
export CONFIG_FILE="${ROOT}/configs/teacher_v2/${DOMAIN}.yaml"
# Inspire supplies node-level WORLD_SIZE/RANK/MASTER_ADDR. Translate these
# before run.sh applies its single-node defaults; explicit user values win.
export NUM_MACHINES="${NUM_MACHINES:-${WORLD_SIZE:-1}}"
export MACHINE_RANK="${MACHINE_RANK:-${RANK:-${NODE_RANK:-0}}}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TOTAL_PROCESSES="${TOTAL_PROCESSES:-$((NUM_MACHINES * NPROC_PER_NODE))}"
export MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${MASTER_ADDR:-127.0.0.1}}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-19001}}"
# A common DDP schema; run.sh supplies the exact node/process topology.
export ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${ROOT}/accelerate_config_8.yaml}"
if [[ "${NUM_MACHINES:-1}" -gt 1 && "${MAIN_PROCESS_IP:-127.0.0.1}" == "127.0.0.1" ]]; then
  echo "Set MAIN_PROCESS_IP and MACHINE_RANK on every node for multi-node training." >&2
  exit 2
fi
exec bash "${ROOT}/run.sh" "${CONFIG_FILE}"

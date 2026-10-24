#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_category_gradient_conflict.sh CHECKPOINT [extra analysis arguments...]

Required:
  CHECKPOINT               Joint-training checkpoint directory (recommended) or LoRA file.

Environment overrides:
  CONFIG                   Training YAML. By default train_config.yaml in the checkpoint or run directory is used.
  OUTPUT_DIR               Result directory (default: outputs/category_gradient_conflict/<checkpoint>_<UTC>).
  ACCELERATE_CONFIG        Accelerate YAML (default: accelerate_config_8.yaml).
  SESSION_NAME             zmx session name (default: grad-conflict-<UTC>).
  SAMPLES_PER_GROUP        Samples/group/repeat (default: 32; divisible by process count).
  REPEATS                  Independent repeats (default: 5).
  CUDA_VISIBLE_DEVICES     Optional GPU restriction, forwarded unchanged.

Example:
  CONFIG=configs/exp/final_architecture_all.yaml \
  bash scripts/run_category_gradient_conflict.sh \
    logs/final_architecture/final_architecture_all/v0/checkpoint-last-... \
    --preset-groups accessory garment human

The job is detached into zmx. Follow it with:
  zmx tail <session-name>
EOF
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit "$([[ $# -ge 1 ]] && echo 0 || echo 2)"
fi

CHECKPOINT="$1"
shift

if [[ ! -e "$CHECKPOINT" ]]; then
  echo "Checkpoint does not exist: $CHECKPOINT" >&2
  exit 2
fi

ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_config_8.yaml}"
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Accelerate config does not exist: $ACCELERATE_CONFIG" >&2
  exit 2
fi

UTC_STAMP="$(date -u +%Y%m%d-%H%M%S)"
CHECKPOINT_NAME="$(basename "${CHECKPOINT%/}")"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/category_gradient_conflict/${CHECKPOINT_NAME}_${UTC_STAMP}}"
SESSION_NAME="${SESSION_NAME:-grad-conflict-${UTC_STAMP}}"
SAMPLES_PER_GROUP="${SAMPLES_PER_GROUP:-32}"
REPEATS="${REPEATS:-5}"

COMMAND=(
  conda run --no-capture-output -n qwen-inpainting-pre-insertion
  accelerate launch --config_file "$ACCELERATE_CONFIG"
  scripts/analyze_category_gradient_conflict.py
  --checkpoint "$CHECKPOINT"
  --output-dir "$OUTPUT_DIR"
  --samples-per-group "$SAMPLES_PER_GROUP"
  --repeats "$REPEATS"
)

if [[ -n "${CONFIG:-}" ]]; then
  if [[ ! -f "$CONFIG" ]]; then
    echo "Training config does not exist: $CONFIG" >&2
    exit 2
  fi
  COMMAND+=(--config "$CONFIG")
fi

COMMAND+=("$@")

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  COMMAND=(env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" "${COMMAND[@]}")
fi

echo "Starting category-gradient analysis in zmx session: $SESSION_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Follow progress: zmx tail $SESSION_NAME"
zmx run "$SESSION_NAME" -d "${COMMAND[@]}"

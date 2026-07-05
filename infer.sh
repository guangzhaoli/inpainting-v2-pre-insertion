#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/configs/insert_anything_fixed_pixels_parquet.yaml}"
LORA_PATH="${LORA_PATH:-${LORA:-${SCRIPT_DIR}/logs/insert_anything_fixed_pixels_parquet/v0/checkpoint-2-43500}}"
SRC_IMAGE="${SRC_IMAGE:-/inspire/dataset/anyinsertion/v1/examples/tar_image.png}"
SRC_MASK="${SRC_MASK:-/inspire/dataset/anyinsertion/v1/examples/tar_mask.png}"
REF_IMAGE="${REF_IMAGE:-/inspire/dataset/anyinsertion/v1/examples/ref_image.png}"
REF_MASK="${REF_MASK:-/inspire/dataset/anyinsertion/v1/examples/ref_mask.png}"
PROMPT="${PROMPT:-Insert Ref into the image}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-4.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
OUTPUT="${OUTPUT:-${SCRIPT_DIR}/output/generated.png}"
COMPOSITE_OUTPUT="${COMPOSITE_OUTPUT:-${SCRIPT_DIR}/output/composite.png}"

CMD=(
    python "${SCRIPT_DIR}/infer.py"
    --config "${CONFIG_FILE}"
    --lora "${LORA_PATH}"
    --src-image "${SRC_IMAGE}"
    --src-mask "${SRC_MASK}"
    --ref-image "${REF_IMAGE}"
    --ref-mask "${REF_MASK}"
    --prompt "${PROMPT}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --true-cfg-scale "${TRUE_CFG_SCALE}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --output "${OUTPUT}"
    --composite-output "${COMPOSITE_OUTPUT}"
)

printf 'Resolved command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"

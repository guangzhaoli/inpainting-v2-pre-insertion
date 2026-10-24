#!/usr/bin/env bash
# Run OUR final-architecture LoRA checkpoint on evaluation testsets with 8 GPUs.
#
# Default checkpoint (the real training artifact, not the previous inference output dir):
#   logs/final_architecture/final_architecture_maskmix_geometry_region_perceptual/v0/checkpoint-0-1000
#
# Usage:
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh paperinsert   # only paperinsert_900
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh testset       # only testset_20260120
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh anyinsertion  # Insert Anything official
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh dreambooth
#   bash scripts/run_ours_ckpt_two_testsets_8gpu.sh anyinsertion_dreambooth
#   LIMIT=4 bash scripts/run_ours_ckpt_two_testsets_8gpu.sh paperinsert
#   CKPT=/path/to/checkpoint-0-1000 bash scripts/run_ours_ckpt_two_testsets_8gpu.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/250041225002/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-qwen-inpainting-pre-insertion}"

# Real LoRA checkpoint (PEFT model.safetensors + mask_geometry_tokenizer.safetensors)
CKPT="${CKPT:-${REPO_ROOT}/logs/final_architecture/final_architecture_maskmix_geometry_region_perceptual/v0/checkpoint-0-1000}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/exp/final_architecture.yaml}"
LORA_WEIGHT="${LORA_WEIGHT:-${CKPT}/model.safetensors}"

PAPERINSERT_ROOT="${PAPERINSERT_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900}"
TESTSET_ROOT="${TESTSET_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/testset_20260120}"
ANYINSERTION_ROOT="${ANYINSERTION_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/anyinsertion_official}"
DREAMBOOTH_ROOT="${DREAMBOOTH_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/dreambooth_eval}"
DREAMBOOTH_PREP="${REPO_ROOT}/scripts/prepare_dreambooth_eval_dataset.py"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/inference/ours_ckpt_two_testsets}"
NPROC="${NPROC:-8}"
SEED="${SEED:-42}"
STEPS="${STEPS:-40}"
TRUE_CFG="${TRUE_CFG:-4.0}"
GUIDANCE="${GUIDANCE:-1.0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

WHICH="${1:-both}"  # both | paperinsert | testset | anyinsertion | dreambooth | anyinsertion_dreambooth | all

# PaperInsert majors (canonical src_*/tar_*)
PAPERINSERT_CATEGORIES=(object person garment)
# Insert Anything / AnyInsertion official majors
ANYINSERTION_CATEGORIES=(object person garment)
# DreamBooth prepared major
DREAMBOOTH_CATEGORIES=(subject)
# testset_20260120 leaves (legacy ref_*/tar_*)
TESTSET_CATEGORIES=(
  accessory
  person
  hair
  clothes
  clothes/pure_color
  pets/cats
  pets/dogs
  corner_cases/light
  corner_cases/low_quality_mask
)

if [[ ! -s "${LORA_WEIGHT}" ]]; then
  echo "Missing LoRA weight: ${LORA_WEIGHT}" >&2
  echo "Note: the path under output/inference/.../checkpoint-0-1000 is a previous" >&2
  echo "      inference run, not the training checkpoint. Use logs/.../checkpoint-0-1000." >&2
  exit 1
fi
if [[ ! -s "${CKPT}/mask_geometry_tokenizer.safetensors" ]]; then
  echo "Missing geometry tokenizer: ${CKPT}/mask_geometry_tokenizer.safetensors" >&2
  exit 1
fi
if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 1
fi

# Avoid Docker system-torch pollution.
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true
sanitize_ld() {
  local cleaned="" part raw="${LD_LIBRARY_PATH:-}"
  if [[ -n "${raw}" ]]; then
    local old_ifs="${IFS}"
    IFS=':'
    # shellcheck disable=SC2206
    local parts=(${raw})
    IFS="${old_ifs}"
    for part in "${parts[@]}"; do
      [[ -z "${part}" ]] && continue
      case "${part}" in
        /usr/local/lib/python*|*/dist-packages/torch*|*/site-packages/torch*) continue ;;
      esac
      cleaned="${cleaned:+$cleaned:}${part}"
    done
  fi
  local env_prefix="${CONDA_PREFIX:-}"
  if [[ -n "${env_prefix}" && -d "${env_prefix}/lib" ]]; then
    cleaned="${env_prefix}/lib${cleaned:+:${cleaned}}"
  fi
  export LD_LIBRARY_PATH="${cleaned}"
}
sanitize_ld

mkdir -p "${OUT_ROOT}/logs"
cd "${REPO_ROOT}"

echo "============================================================"
echo "Ours ckpt two-testset inference"
echo "  ckpt:     ${CKPT}"
echo "  lora:     ${LORA_WEIGHT}"
echo "  config:   ${CONFIG}"
echo "  nproc:    ${NPROC}"
echo "  which:    ${WHICH}"
echo "  out_root: ${OUT_ROOT}"
echo "  run_id:   ${RUN_ID}"
echo "============================================================"

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA required"
PY

run_category() {
  local testset_root="$1"
  local category="$2"
  local out_dir="$3"
  local label="$4"

  mkdir -p "${out_dir}" "${OUT_ROOT}/logs"
  echo
  echo ">>> [${label}] category=${category} out=${out_dir}"

  local pids=()
  local rank
  for ((rank = 0; rank < NPROC; rank++)); do
    local rank_log="${OUT_ROOT}/logs/${label}__${category//\//_}__${RUN_ID}__rank$(printf '%02d' "${rank}").log"
    (
      export RANK="${rank}"
      export WORLD_SIZE="${NPROC}"
      export CUDA_VISIBLE_DEVICES="${rank}"
      export PYTHONNOUSERSITE=1
      exec python -u "${REPO_ROOT}/infer_testset.py" \
        --config "${CONFIG}" \
        --lora "${LORA_WEIGHT}" \
        --testset-root "${testset_root}" \
        --category "${category}" \
        --output-dir "${out_dir}" \
        --seed "${SEED}" \
        --num-inference-steps "${STEPS}" \
        --true-cfg-scale "${TRUE_CFG}" \
        --guidance-scale "${GUIDANCE}" \
        --rank "${rank}" \
        --world-size "${NPROC}" \
        --skip-existing \
        "${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}"
    ) >"${rank_log}" 2>&1 &
    pids+=("$!")
    echo "  rank ${rank} pid=${pids[-1]} log=${rank_log}"
  done

  local rc=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  if [[ "${rc}" -ne 0 ]]; then
    echo ">>> [${label}/${category}] FAILED" >&2
    tail -n 40 "${OUT_ROOT}/logs/${label}__${category//\//_}__${RUN_ID}__rank"*.log 2>/dev/null | tail -n 60 || true
    return 1
  fi
  echo ">>> [${label}/${category}] OK"
}

run_paperinsert() {
  local out_base="${OUT_ROOT}/paperinsert_900/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
  local cat
  for cat in "${PAPERINSERT_CATEGORIES[@]}"; do
    run_category "${PAPERINSERT_ROOT}" "${cat}" "${out_base}/${cat}" "paperinsert"
  done
}

run_testset() {
  local out_base="${OUT_ROOT}/testset_20260120/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
  local cat
  for cat in "${TESTSET_CATEGORIES[@]}"; do
    if [[ ! -d "${TESTSET_ROOT}/${cat}/ref_image" && ! -d "${TESTSET_ROOT}/${cat}/src_image" ]]; then
      echo "skip missing category dir: ${cat}" >&2
      continue
    fi
    run_category "${TESTSET_ROOT}" "${cat}" "${out_base}/${cat}" "testset"
  done
}

run_anyinsertion() {
  if [[ ! -d "${ANYINSERTION_ROOT}" ]]; then
    echo "AnyInsertion root missing: ${ANYINSERTION_ROOT}" >&2
    return 1
  fi
  local out_base="${OUT_ROOT}/anyinsertion_official/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
  local cat
  for cat in "${ANYINSERTION_CATEGORIES[@]}"; do
    if [[ ! -d "${ANYINSERTION_ROOT}/${cat}/src_image" && ! -d "${ANYINSERTION_ROOT}/${cat}/ref_image" ]]; then
      echo "skip missing anyinsertion category: ${cat}" >&2
      continue
    fi
    run_category "${ANYINSERTION_ROOT}" "${cat}" "${out_base}/${cat}" "anyinsertion"
  done
}

ensure_dreambooth() {
  if [[ -d "${DREAMBOOTH_ROOT}/subject/src_image" ]]; then
    return 0
  fi
  if [[ ! -f "${DREAMBOOTH_PREP}" ]]; then
    echo "DreamBooth set missing and no prepare script: ${DREAMBOOTH_PREP}" >&2
    return 1
  fi
  echo "Building DreamBooth eval set at ${DREAMBOOTH_ROOT} ..."
  python "${DREAMBOOTH_PREP}" --output-root "${DREAMBOOTH_ROOT}" --overwrite
}

run_dreambooth() {
  ensure_dreambooth
  local out_base="${OUT_ROOT}/dreambooth_eval/final_architecture_maskmix_geometry_region_perceptual/checkpoint-0-1000"
  local cat
  for cat in "${DREAMBOOTH_CATEGORIES[@]}"; do
    run_category "${DREAMBOOTH_ROOT}" "${cat}" "${out_base}/${cat}" "dreambooth"
  done
}

case "${WHICH}" in
  both)
    run_paperinsert
    run_testset
    ;;
  paperinsert|paperinsert_900)
    run_paperinsert
    ;;
  testset|testset_20260120)
    run_testset
    ;;
  anyinsertion|insert_anything|insert-anything|ia)
    run_anyinsertion
    ;;
  dreambooth|db)
    run_dreambooth
    ;;
  anyinsertion_dreambooth|ia_db|insert_anything_dreambooth)
    run_anyinsertion
    run_dreambooth
    ;;
  all)
    run_paperinsert
    run_testset
    run_anyinsertion
    run_dreambooth
    ;;
  *)
    echo "Unknown selector: ${WHICH}" >&2
    echo "Use: both|paperinsert|testset|anyinsertion|dreambooth|anyinsertion_dreambooth|all" >&2
    exit 2
    ;;
esac

echo
echo "Summarizing outputs under ${OUT_ROOT}"
python - <<PY
from pathlib import Path
root = Path("${OUT_ROOT}")
for suite in ["paperinsert_900", "testset_20260120", "anyinsertion_official", "dreambooth_eval"]:
    base = root / suite
    if not base.exists():
        print(f"{suite}: (not run)")
        continue
    n = len(list(base.rglob("output.png")))
    print(f"{suite}: output.png count = {n}")
print("done")
PY
echo "All requested suite(s) finished."

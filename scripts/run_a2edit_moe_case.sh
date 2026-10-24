#!/usr/bin/env bash
# Single-case A2Edit-MoE inference (defaults to testset_20260120 accessory/6,
# the keep-export case accessory__6 that was missing from batch MoE).
#
# Usage (from anywhere):
#   bash scripts/run_a2edit_moe_case.sh
#   GPU=0 SEED=42 bash scripts/run_a2edit_moe_case.sh
#
# Optional overrides:
#   STEM=6 CATEGORY=accessory
#   REF_IMAGE / REF_MASK / SRC_IMAGE / SRC_MASK  (SRC = target scene = tar_*)
#   OUT_PNG  final path of the result image
#   COPY_TO_KEEP=1  also write review_keep_export/.../results/a2edit_moe.png
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A2EDIT_ROOT="${A2EDIT_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/baselines/a2edit_router}"
TESTSET_ROOT="${TESTSET_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/testset_20260120}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/250041225002/miniconda3}"

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"

# Prefer FLUX / A2Edit stack envs used by batch MoE.
CONDA_ENV="${CONDA_ENV:-}"
if [[ -z "${CONDA_ENV}" ]]; then
  if [[ -d "${CONDA_ROOT}/envs/a2edit" ]]; then
    CONDA_ENV=a2edit
  elif [[ -d "${CONDA_ROOT}/envs/insertanything" ]]; then
    CONDA_ENV=insertanything
  else
    CONDA_ENV=qwen-inpainting-pre-insertion
  fi
fi
conda activate "${CONDA_ENV}"

CATEGORY="${CATEGORY:-accessory}"
STEM="${STEM:-6}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
OUTPUT_SIZE="${OUTPUT_SIZE:-768}"

CONFIG="${CONFIG:-${A2EDIT_ROOT}/experiments/config/moe_infer_local.yaml}"
ROUTER_PATH="${ROUTER_PATH:-${A2EDIT_ROOT}/runs/router_only/step_4000/moe_lora.pt}"
WEIGHTS_ROOT="${A2EDIT_WEIGHTS_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/baselines/a2edit_ckpt/weights}"
export A2EDIT_WEIGHTS_ROOT="${WEIGHTS_ROOT}"

# Resolve default paths for testset leaf layout (ref_* / tar_*).
leaf="${TESTSET_ROOT}/${CATEGORY}"
if [[ -z "${REF_IMAGE:-}" ]]; then
  for ext in png jpg jpeg webp; do
    if [[ -f "${leaf}/ref_image/${STEM}.${ext}" ]]; then REF_IMAGE="${leaf}/ref_image/${STEM}.${ext}"; break; fi
  done
fi
if [[ -z "${REF_MASK:-}" ]]; then
  for ext in png jpg jpeg webp; do
    if [[ -f "${leaf}/ref_mask/${STEM}.${ext}" ]]; then REF_MASK="${leaf}/ref_mask/${STEM}.${ext}"; break; fi
  done
fi
if [[ -z "${SRC_IMAGE:-}" ]]; then
  for ext in png jpg jpeg webp; do
    if [[ -f "${leaf}/tar_image/${STEM}.${ext}" ]]; then SRC_IMAGE="${leaf}/tar_image/${STEM}.${ext}"; break; fi
  done
fi
if [[ -z "${SRC_MASK:-}" ]]; then
  for ext in png jpg jpeg webp; do
    if [[ -f "${leaf}/tar_mask/${STEM}.${ext}" ]]; then SRC_MASK="${leaf}/tar_mask/${STEM}.${ext}"; break; fi
  done
fi

OUT_ROOT="${OUT_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/testset_20260120_baselines_infer/a2edit_moe}"
OUT_PNG="${OUT_PNG:-${OUT_ROOT}/${CATEGORY}/${STEM}.png}"
KEEP_DIR="${KEEP_DIR:-${REPO_ROOT}/output/review_keep_export/testset_20260120/accessory__${STEM}}"
COPY_TO_KEEP="${COPY_TO_KEEP:-1}"

for p in "${REF_IMAGE:-}" "${REF_MASK:-}" "${SRC_IMAGE:-}" "${SRC_MASK:-}" "${CONFIG}" "${ROUTER_PATH}"; do
  if [[ -z "${p}" || ! -s "${p}" ]]; then
    echo "Missing required path: ${p:-<empty>}" >&2
    echo "  REF_IMAGE=${REF_IMAGE:-}" >&2
    echo "  REF_MASK=${REF_MASK:-}" >&2
    echo "  SRC_IMAGE=${SRC_IMAGE:-}" >&2
    echo "  SRC_MASK=${SRC_MASK:-}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${OUT_PNG}")" "${OUT_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1
# Critical: a2edit_router owns package name ``src``; the inpainting repo also has
# a top-level ``src/``. Run from A2EDIT_ROOT and put it first so imports resolve
# to A2Edit (my_model / moe_lora), not the inpainting trainer package.
export PYTHONPATH="${A2EDIT_ROOT}"

# Drop docker/system torch pollution (same as run_moe_8gpu.sh).
sanitize_ld() {
  local raw="${LD_LIBRARY_PATH:-}" cleaned="" part
  local old_ifs="${IFS}"
  IFS=':'
  # shellcheck disable=SC2206
  local parts=(${raw})
  IFS="${old_ifs}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == *"/usr/local/lib"* || "${part}" == *"/usr/lib"* ]]; then
      continue
    fi
    cleaned="${cleaned:+${cleaned}:}${part}"
  done
  export LD_LIBRARY_PATH="${cleaned}"
}
sanitize_ld

echo "=== A2Edit-MoE single case ==="
echo "conda     : ${CONDA_ENV}"
echo "gpu       : ${GPU}"
echo "seed      : ${SEED}"
echo "output_sz : ${OUTPUT_SIZE}"
echo "ref_image : ${REF_IMAGE}"
echo "ref_mask  : ${REF_MASK}"
echo "src_image : ${SRC_IMAGE}  (target scene)"
echo "src_mask  : ${SRC_MASK}"
echo "config    : ${CONFIG}"
echo "router    : ${ROUTER_PATH}"
echo "out_png   : ${OUT_PNG}"
echo "workdir   : ${A2EDIT_ROOT}"
echo

RUNTIME_CFG="${OUT_ROOT}/logs/runtime_config_accessory${STEM}_single.yaml"
LOG="${OUT_ROOT}/logs/accessory_${STEM}_seed${SEED}.log"

# cwd must be a2edit_router so ``import src.models...`` hits the right package.
(
  cd "${A2EDIT_ROOT}"
  python -u - <<PY 2>&1 | tee "${LOG}"
import os
import sys
import time
from pathlib import Path

# Drop '' / other repo roots that would shadow a2edit_router.src
_root = Path(${A2EDIT_ROOT@Q}).resolve()
sys.path = [str(_root)] + [
    p for p in sys.path
    if p not in ("", ".") and Path(p).resolve() != Path(${REPO_ROOT@Q}).resolve()
]

import torch
from batch_inference_moe_datasets import write_runtime_config
from inference_moe import MoEInference

config_src = Path(${CONFIG@Q})
runtime_cfg = Path(${RUNTIME_CFG@Q})
write_runtime_config(config_src, runtime_cfg)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
print(f"src_pkg={__import__('src').__file__}", flush=True)

t0 = time.time()
pipe = MoEInference(config_path=str(runtime_cfg), device=device)
pipe.load_weights(router_path=${ROUTER_PATH@Q})
print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

t1 = time.time()
result = pipe.inference(
    src_image_path=${SRC_IMAGE@Q},
    src_mask_path=${SRC_MASK@Q},
    ref_image_path=${REF_IMAGE@Q},
    ref_mask_path=${REF_MASK@Q},
    seed=int(${SEED}),
    output_size=(int(${OUTPUT_SIZE}), int(${OUTPUT_SIZE})),
)
out = Path(${OUT_PNG@Q})
out.parent.mkdir(parents=True, exist_ok=True)
result.save(out)
print(f"saved {out}  size={result.size}  infer={time.time()-t1:.1f}s", flush=True)
PY
)

if [[ ! -s "${OUT_PNG}" ]]; then
  echo "FAILED: missing output ${OUT_PNG} (see ${LOG})" >&2
  exit 1
fi

if [[ "${COPY_TO_KEEP}" == "1" && -d "${KEEP_DIR}/results" ]]; then
  cp -f "${OUT_PNG}" "${KEEP_DIR}/results/a2edit_moe.png"
  echo "copied -> ${KEEP_DIR}/results/a2edit_moe.png"
fi

echo
echo "=== done ==="
echo "result : ${OUT_PNG}"
echo "log    : ${LOG}"
echo "To refresh the paper grid:"
echo "  cd ${REPO_ROOT} && python scripts/make_paper_qualitative_grid.py --also-compact --cell-w 420 --cell-h 560 --fit cover"

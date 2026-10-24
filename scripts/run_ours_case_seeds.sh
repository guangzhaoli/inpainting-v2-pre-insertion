#!/usr/bin/env bash
# Sweep ours (final-architecture LoRA) over multiple seeds for a single case.
#
# Default case: anyinsertion keep-export ai_gar_garment_006
#   (same ckpt / config / hyperparams as scripts/run_ours_ckpt_two_testsets_8gpu.sh)
#
# Usage:
#   bash scripts/run_ours_case_seeds.sh
#   SEEDS="42 48 0 7 123 2024" GPU=0 bash scripts/run_ours_case_seeds.sh
#   SEEDS="48" bash scripts/run_ours_case_seeds.sh   # reproduce official seed only
#
# Optional overrides:
#   CASE_ID=ai_gar_garment_006
#   CASE_DIR=output/review_keep_export/anyinsertion/ai_gar_garment_006
#   REF_IMAGE / REF_MASK / SRC_IMAGE / SRC_MASK  (absolute paths)
#   CKPT / CONFIG / STEPS / TRUE_CFG / GUIDANCE / GPU
#   OUT_ROOT   default: output/seed_sweeps/<CASE_ID>
#   COPY_TO_RESULTS=1  also write results/ours_seed<N>.png under CASE_DIR
#   SKIP_EXISTING=1    skip seeds that already have output_restored_raw.png
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/250041225002/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-qwen-inpainting-pre-insertion}"

CASE_ID="${CASE_ID:-ai_gar_garment_006}"
CASE_DIR="${CASE_DIR:-${REPO_ROOT}/output/review_keep_export/anyinsertion/${CASE_ID}}"
ANYINSERTION_ROOT="${ANYINSERTION_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/anyinsertion_official}"

# Prefer keep-export inputs; fall back to official anyinsertion garment layout.
if [[ -f "${CASE_DIR}/tar_image.png" ]]; then
  SRC_IMAGE="${SRC_IMAGE:-${CASE_DIR}/tar_image.png}"
  SRC_MASK="${SRC_MASK:-${CASE_DIR}/tar_mask.png}"
  REF_IMAGE="${REF_IMAGE:-${CASE_DIR}/ref_image.png}"
  REF_MASK="${REF_MASK:-${CASE_DIR}/ref_mask.png}"
else
  SRC_IMAGE="${SRC_IMAGE:-${ANYINSERTION_ROOT}/garment/tar_image/${CASE_ID}.png}"
  SRC_MASK="${SRC_MASK:-${ANYINSERTION_ROOT}/garment/tar_mask/${CASE_ID}.png}"
  REF_IMAGE="${REF_IMAGE:-${ANYINSERTION_ROOT}/garment/src_image/${CASE_ID}.png}"
  REF_MASK="${REF_MASK:-${ANYINSERTION_ROOT}/garment/src_mask/${CASE_ID}.png}"
fi

CKPT="${CKPT:-${REPO_ROOT}/logs/final_architecture/final_architecture_maskmix_geometry_region_perceptual/v0/checkpoint-0-1000}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/exp/final_architecture.yaml}"
LORA_WEIGHT="${LORA_WEIGHT:-${CKPT}/model.safetensors}"

# Match batch anyinsertion garment eval (sample index 6 -> seed 42+6=48 was the published ours).
# Default sweep: official seed first, then a small diverse set.
SEEDS="${SEEDS:-48 0 1 7 13 21 42 123 2024 3407}"
STEPS="${STEPS:-40}"
TRUE_CFG="${TRUE_CFG:-4.0}"
GUIDANCE="${GUIDANCE:-1.0}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/seed_sweeps/${CASE_ID}}"
COPY_TO_RESULTS="${COPY_TO_RESULTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

for p in "${SRC_IMAGE}" "${SRC_MASK}" "${REF_IMAGE}" "${REF_MASK}" "${CONFIG}" "${LORA_WEIGHT}"; do
  if [[ ! -s "${p}" ]]; then
    echo "Missing required path: ${p}" >&2
    exit 1
  fi
done
if [[ ! -s "${CKPT}/mask_geometry_tokenizer.safetensors" ]]; then
  echo "Missing geometry tokenizer: ${CKPT}/mask_geometry_tokenizer.safetensors" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1

echo "=== ours seed sweep ==="
echo "case_id   : ${CASE_ID}"
echo "src_image : ${SRC_IMAGE}"
echo "ref_image : ${REF_IMAGE}"
echo "ckpt      : ${CKPT}"
echo "config    : ${CONFIG}"
echo "seeds     : ${SEEDS}"
echo "out_root  : ${OUT_ROOT}"
echo "gpu       : ${GPU}"
echo

ok=0
fail=0
declare -a done_seeds=()

for seed in ${SEEDS}; do
  seed_dir="${OUT_ROOT}/seed_$(printf '%04d' "${seed}")"
  restored="${seed_dir}/output_restored_raw.png"
  # infer.py writes next to --output; prefer a stable primary name
  primary="${seed_dir}/output.png"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${restored}" ]]; then
    echo ">>> seed=${seed} SKIP (exists ${restored})"
    done_seeds+=("${seed}")
    ok=$((ok + 1))
    continue
  fi

  mkdir -p "${seed_dir}"
  log="${seed_dir}/infer.log"
  echo ">>> seed=${seed} -> ${seed_dir}"

  if python -u "${REPO_ROOT}/infer.py" \
      --config "${CONFIG}" \
      --lora "${LORA_WEIGHT}" \
      --src-image "${SRC_IMAGE}" \
      --src-mask "${SRC_MASK}" \
      --ref-image "${REF_IMAGE}" \
      --ref-mask "${REF_MASK}" \
      --seed "${seed}" \
      --num-inference-steps "${STEPS}" \
      --true-cfg-scale "${TRUE_CFG}" \
      --guidance-scale "${GUIDANCE}" \
      --output "${primary}" \
      >"${log}" 2>&1; then
    # Prefer restored-raw (same artifact as gallery/export); fall back to primary.
    art=""
    if [[ -s "${restored}" ]]; then
      art="${restored}"
    elif [[ -s "${primary}" ]]; then
      art="${primary}"
    fi
    if [[ -z "${art}" ]]; then
      echo "    FAILED: no output under ${seed_dir} (see ${log})" >&2
      fail=$((fail + 1))
      continue
    fi
    # Flat copies for quick visual compare
    cp -f "${art}" "${OUT_ROOT}/ours_seed${seed}.png"
    if [[ "${COPY_TO_RESULTS}" == "1" && -d "${CASE_DIR}/results" ]]; then
      cp -f "${art}" "${CASE_DIR}/results/ours_seed${seed}.png"
      echo "    wrote ${CASE_DIR}/results/ours_seed${seed}.png"
    fi
    echo "    ok: ${art}"
    done_seeds+=("${seed}")
    ok=$((ok + 1))
  else
    echo "    FAILED (see ${log})" >&2
    tail -n 30 "${log}" >&2 || true
    fail=$((fail + 1))
  fi
done

# Simple index for the sweep
index_json="${OUT_ROOT}/index.json"
python - <<PY
import json
from pathlib import Path
out = Path(${OUT_ROOT@Q})
seeds = [int(s) for s in """${SEEDS}""".split()]
rows = []
for s in seeds:
    p = out / f"ours_seed{s}.png"
    d = out / f"seed_{s:04d}"
    rows.append({
        "seed": s,
        "flat": str(p) if p.is_file() else None,
        "dir": str(d),
        "restored_raw": str(d / "output_restored_raw.png") if (d / "output_restored_raw.png").is_file() else None,
        "output": str(d / "output.png") if (d / "output.png").is_file() else None,
        "log": str(d / "infer.log") if (d / "infer.log").is_file() else None,
    })
meta = {
    "case_id": ${CASE_ID@Q},
    "src_image": ${SRC_IMAGE@Q},
    "src_mask": ${SRC_MASK@Q},
    "ref_image": ${REF_IMAGE@Q},
    "ref_mask": ${REF_MASK@Q},
    "ckpt": ${CKPT@Q},
    "config": ${CONFIG@Q},
    "steps": int(${STEPS}),
    "true_cfg_scale": float(${TRUE_CFG}),
    "guidance_scale": float(${GUIDANCE}),
    "official_seed_note": "anyinsertion garment sample index 6 used seed 42+6=48 in batch eval",
    "seeds": rows,
}
Path(${index_json@Q}).write_text(json.dumps(meta, indent=2) + "\n")
print(f"wrote {Path(${index_json@Q})}")
PY

echo
echo "=== done: ok=${ok} fail=${fail} ==="
echo "flat pngs : ${OUT_ROOT}/ours_seed*.png"
echo "per-seed  : ${OUT_ROOT}/seed_XXXX/"
echo "index     : ${index_json}"
if [[ "${COPY_TO_RESULTS}" == "1" ]]; then
  echo "results   : ${CASE_DIR}/results/ours_seed*.png"
fi
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

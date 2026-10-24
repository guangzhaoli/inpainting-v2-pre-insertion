#!/usr/bin/env bash
# 8×H200 offline metrics: DINO-I↑ CLIP-I↑ PSNR↑ SSIM↑ LPIPS↓ FID↓
# PaperInsert-900 + AnyInsertion × all methods (including ours).
#
# GPU nodes are offline — uses local weights under:
#   pretrained_weights/metrics_offline/
# Prepare once (on a node that already has HF/torch caches):
#   bash scripts/prepare_metrics_offline_weights.sh
#
# Usage:
#   bash scripts/run_eval_metrics_all.sh
#   NPROC=8 bash scripts/run_eval_metrics_all.sh
#   LIMIT=8 bash scripts/run_eval_metrics_all.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/250041225002/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-qwen-inpainting-pre-insertion}"

NPROC="${NPROC:-8}"
SUITES="${SUITES:-paperinsert_900,anyinsertion}"
METHODS="${METHODS:-ours,anydoor,insert_anything,qwen_edit_2511,qwen_image_2_0,flux_kontext,a2edit_moe}"
OUT="${OUT:-${ROOT}/output/metrics}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# ---- offline metric weights (no huggingface.co / pytorch.org) ----
METRICS_OFFLINE_ROOT="${METRICS_OFFLINE_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/pretrained_weights/metrics_offline}"
export METRICS_OFFLINE_ROOT
export CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${METRICS_OFFLINE_ROOT}/clip-vit-base-patch32}"
export DINO_MODEL_PATH="${DINO_MODEL_PATH:-${METRICS_OFFLINE_ROOT}/dinov2-small}"
export TORCH_HOME="${TORCH_HOME:-${METRICS_OFFLINE_ROOT}/torch_home}"
export LPIPS_ALEX_PATH="${LPIPS_ALEX_PATH:-${METRICS_OFFLINE_ROOT}/lpips/alex.pth}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

mkdir -p "${OUT}/logs" "${OUT}/shards"
cd "${ROOT}"

# Avoid Docker system-torch pollution.
export PYTHONNOUSERSITE=1
unset PYTHONHOME || true
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Validate offline weights before launching 8 ranks.
python - <<PY
from pathlib import Path
import os
checks = {
    "CLIP_MODEL_PATH": Path(os.environ["CLIP_MODEL_PATH"]),
    "DINO_MODEL_PATH": Path(os.environ["DINO_MODEL_PATH"]),
    "inception": Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/inception_v3_google-0cc3c7bd.pth",
    "alexnet": Path(os.environ["TORCH_HOME"]) / "hub/checkpoints/alexnet-owt-7be5be79.pth",
}
bad = []
for k, p in checks.items():
    ok = p.is_dir() if k.endswith("PATH") and "MODEL" in k else p.is_file()
    # CLIP/DINO are dirs
    if k in ("CLIP_MODEL_PATH", "DINO_MODEL_PATH"):
        ok = p.is_dir() and any(p.iterdir())
    print(f"[{'ok' if ok else 'MISSING'}] {k}={p}")
    if not ok:
        bad.append(k)
if bad:
    raise SystemExit(
        "Offline metric weights missing: "
        + ", ".join(bad)
        + "\nRun: bash scripts/prepare_metrics_offline_weights.sh"
    )
PY

NGPU=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${NGPU}" -lt 1 ]]; then
  echo "[error] no CUDA GPU visible" >&2
  exit 1
fi
if [[ "${NPROC}" -gt "${NGPU}" ]]; then
  echo "[warn] NPROC=${NPROC} > visible GPUs=${NGPU}; clamping"
  NPROC="${NGPU}"
fi

echo "============================================================"
echo "Insertion metrics · multi-GPU · OFFLINE"
echo "  suites:   ${SUITES}"
echo "  methods:  ${METHODS}"
echo "  nproc:    ${NPROC}  (visible GPUs=${NGPU})"
echo "  output:   ${OUT}"
echo "  run_id:   ${RUN_ID}"
echo "  CLIP:     ${CLIP_MODEL_PATH}"
echo "  DINO:     ${DINO_MODEL_PATH}"
echo "  TORCH_HOME: ${TORCH_HOME}"
echo "============================================================"

python - <<PY
from pathlib import Path
out = Path("${OUT}") / "shards"
out.mkdir(parents=True, exist_ok=True)
suites = [s.strip() for s in "${SUITES}".split(",") if s.strip()]
for p in list(out.glob("*")):
    name = p.name
    if any(name.startswith(s + "__") for s in suites):
        p.unlink()
print(f"[clean] shards for {suites}")
PY

pids=()
for ((rank = 0; rank < NPROC; rank++)); do
  log="${OUT}/logs/metrics_${RUN_ID}_rank$(printf '%02d' "${rank}").log"
  (
    export RANK="${rank}"
    export WORLD_SIZE="${NPROC}"
    export CUDA_VISIBLE_DEVICES="${rank}"
    export PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export CLIP_MODEL_PATH TORCH_HOME DINO_MODEL_PATH LPIPS_ALEX_PATH METRICS_OFFLINE_ROOT
    exec python -u scripts/eval_insertion_metrics.py \
      --suites "${SUITES}" \
      --methods "${METHODS}" \
      --device cuda:0 \
      --rank "${rank}" \
      --world-size "${NPROC}" \
      --output-dir "${OUT}" \
      --clip-path "${CLIP_MODEL_PATH}" \
      --dino-path "${DINO_MODEL_PATH}" \
      --torch-home "${TORCH_HOME}" \
      "${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  echo "  rank ${rank} pid=${pids[-1]} log=${log}  GPU=${rank}"
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

if [[ "${rc}" -ne 0 ]]; then
  echo "[error] one or more ranks failed; see ${OUT}/logs/" >&2
  tail -n 50 "${OUT}/logs/metrics_${RUN_ID}_rank"*.log 2>/dev/null | tail -n 100 || true
  exit 1
fi

echo
echo "Merging ${NPROC} shards …"
python -u scripts/eval_insertion_metrics.py \
  --merge-only \
  --suites "${SUITES}" \
  --methods "${METHODS}" \
  --world-size "${NPROC}" \
  --output-dir "${OUT}"

echo
echo "Done (8-GPU offline). Summary:"
echo "  ${OUT}/metrics_summary.md"
echo "  ${OUT}/metrics_summary.json"

#!/usr/bin/env bash
# Materialize offline metric backbones for air-gapped GPU nodes.
# Sources: local HF hub cache + torch hub cache (already downloaded once).
#
#   bash scripts/prepare_metrics_offline_weights.sh
#
set -euo pipefail

ROOT="${METRICS_OFFLINE_ROOT:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/pretrained_weights/metrics_offline}"
HF_HUB="${HF_HUB:-/inspire/hdd/global_user/250041225002/.cache/huggingface/hub}"
TORCH_HUB="${TORCH_HUB:-/inspire/hdd/global_user/250041225002/.cache/torch}"
PW="${PW:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/pretrained_weights}"

CLIP_SNAP="${HF_HUB}/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DINO_SNAP="${HF_HUB}/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056"

mkdir -p "${ROOT}/torch_home/hub/checkpoints" "${ROOT}/lpips"

python - <<PY
import shutil
from pathlib import Path

root = Path("${ROOT}")
pairs = [
    ("clip-vit-base-patch32", Path("${CLIP_SNAP}")),
    ("dinov2-small", Path("${DINO_SNAP}")),
]
for name, src in pairs:
    if not src.is_dir():
        raise SystemExit(f"missing source snapshot: {src}")
    dst = root / name
    if dst.exists():
        shutil.rmtree(dst)
    # materialize files (HF cache uses symlinks to blobs)
    shutil.copytree(src, dst, symlinks=False)
    n = sum(1 for p in dst.rglob("*") if p.is_file())
    b = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"[ok] {name}: {n} files, {b/1e6:.1f} MB -> {dst}")
PY

copy_ckpt() {
  local name="$1"
  local src=""
  for cand in \
    "${TORCH_HUB}/hub/checkpoints/${name}" \
    "${PW}/torch/hub/checkpoints/${name}" \
    ; do
    if [[ -f "${cand}" ]]; then src="${cand}"; break; fi
  done
  if [[ -z "${src}" ]]; then
    echo "[warn] missing ${name}" >&2
    return 1
  fi
  cp -a "${src}" "${ROOT}/torch_home/hub/checkpoints/${name}"
  echo "[ok] ${name} -> ${ROOT}/torch_home/hub/checkpoints/${name}"
}

copy_ckpt "inception_v3_google-0cc3c7bd.pth"
copy_ckpt "alexnet-owt-7be5be79.pth" || true
copy_ckpt "vgg16-397923af.pth" || true

if [[ -f "${PW}/lpips/linear/alex.pth" ]]; then
  cp -a "${PW}/lpips/linear/alex.pth" "${ROOT}/lpips/alex.pth"
  echo "[ok] lpips/alex.pth"
fi

echo
echo "Offline metrics root: ${ROOT}"
echo "Export before eval (run_eval_metrics_all.sh sets these automatically):"
echo "  export METRICS_OFFLINE_ROOT=${ROOT}"
echo "  export CLIP_MODEL_PATH=${ROOT}/clip-vit-base-patch32"
echo "  export DINO_MODEL_PATH=${ROOT}/dinov2-small"
echo "  export TORCH_HOME=${ROOT}/torch_home"
echo "  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"

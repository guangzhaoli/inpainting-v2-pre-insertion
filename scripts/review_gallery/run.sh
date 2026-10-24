#!/usr/bin/env bash
# Launch the multi-method insertion review gallery.
#
#   bash scripts/review_gallery/run.sh
#   bash scripts/review_gallery/run.sh --port 8765
#   HOST=127.0.0.1 bash scripts/review_gallery/run.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/250041225002/miniconda3}"
# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-qwen-inpainting-pre-insertion}"
cd "${ROOT}"
export PYTHONUNBUFFERED=1
exec python -u scripts/review_gallery/server.py --host "${HOST:-0.0.0.0}" --port "${PORT:-8765}" "$@"

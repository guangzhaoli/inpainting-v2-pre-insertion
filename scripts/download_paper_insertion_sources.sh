#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1}"
RAW_ROOT="${WORK_ROOT}/_raw"
CONDA_INIT="/inspire/hdd/global_user/250041225002/miniconda3/etc/profile.d/conda.sh"

source "${CONDA_INIT}"
conda activate qwen-inpainting-pre-insertion

mkdir -p "${RAW_ROOT}/coco" "${RAW_ROOT}/fashionpedia" "${RAW_ROOT}/anyinsertion"

verify_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "${path}" ]] || return 1
  local actual
  actual="$(sha256sum "${path}" | cut -d ' ' -f 1)"
  [[ "${actual}" == "${expected}" ]]
}

download_pinned() {
  local destination="$1"
  local expected="$2"
  local url="$3"
  if verify_sha256 "${destination}" "${expected}"; then
    echo "verified existing source: ${destination}"
    return 0
  fi
  if [[ -e "${destination}" ]]; then
    echo "refusing to overwrite source with unexpected SHA256: ${destination}" >&2
    return 1
  fi
  local partial="${destination}.part"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "${partial}" "${url}"
  if ! verify_sha256 "${partial}" "${expected}"; then
    echo "downloaded source failed pinned SHA256: ${partial}" >&2
    return 1
  fi
  mv "${partial}" "${destination}"
}

download_pinned \
  "${RAW_ROOT}/coco/val2017.zip" \
  "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05" \
  "http://images.cocodataset.org/zips/val2017.zip"
download_pinned \
  "${RAW_ROOT}/coco/annotations_trainval2017.zip" \
  "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268" \
  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# This pinned Fashionpedia mirror retains the full-resolution native instance
# masks.  The smaller detection-datasets/fashionpedia parquet is intentionally
# not used because it strips segmentation and contains only bounding boxes.
download_pinned \
  "${RAW_ROOT}/fashionpedia/test_with_masks.parquet" \
  "9d2376c8acbbbb7f23c9aee2b2a1cb4d760d0538732d2bb8a617946cbedaeb31" \
  "https://huggingface.co/datasets/tilak1114/fashion-pedia/resolve/81c7b35b9ff8951d08ab1818143392755ef0d91c/data/test-00000-of-00001.parquet?download=true"
download_pinned \
  "${RAW_ROOT}/fashionpedia/categories.json" \
  "b73375b257e3bd7b69f9a8142386997d5dee624b4754e52de72a4b93d728861f" \
  "https://huggingface.co/datasets/tilak1114/fashion-pedia/resolve/81c7b35b9ff8951d08ab1818143392755ef0d91c/categories.json"

# A stock `gdown --folder` aborts the whole traversal after one content-host
# timeout.  Resolve the official tree with gdown, then fetch independent files
# concurrently with atomic partial files and image/JSON validation.
python "$(dirname "$0")/download_gdrive_folder_parallel.py" \
  "https://drive.google.com/drive/folders/1uUD_iGm4dAEVYGmXp5KvjWl1e_j2pjiK?usp=sharing" \
  "${RAW_ROOT}/anyinsertion" \
  --workers 12 \
  --attempts 10 \
  --expected-count 562 \
  --expected-tree-sha256 c2303ddb287cf191907d8ef8bf6489a7dfa94632f9f33cdd339cec1b19c6f1ea \
  --expected-content-sha256 f6c56823fb13ab53e503773e2291b6e06d1b4edf09276d2465d872b0838906e5

python "$(dirname "$0")/prepare_paper_insertion_dataset.py" extract --work-root "${WORK_ROOT}"

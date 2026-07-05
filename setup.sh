#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-qwen-inpainting-pre-insertion}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required but was not found in PATH" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip --override-channels -c "${CONDA_CHANNEL}"
fi

PIP_CONFIG_FILE=/dev/null conda run -n "${ENV_NAME}" python -m pip install \
    --timeout 60 \
    --retries 2 \
    torch torchvision \
    --index-url "${TORCH_INDEX_URL}"

PIP_CONFIG_FILE=/dev/null conda run -n "${ENV_NAME}" python -m pip install \
    --timeout 60 \
    --retries 2 \
    --index-url "${PYPI_INDEX_URL}" \
    -r requirements.txt

conda run -n "${ENV_NAME}" python -m ipykernel install \
    --user \
    --name "${ENV_NAME}" \
    --display-name "Python (${ENV_NAME})"

echo "Environment ready: ${ENV_NAME}"
echo "Activate it with: conda activate ${ENV_NAME}"

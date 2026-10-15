#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f /inspire/hdd/global_user/250041225002/uv_config/env.sh ]]; then
    source /inspire/hdd/global_user/250041225002/uv_config/env.sh
fi
UV_BIN="${UV_BIN:-uv}"
# Downloads belong on a newly created Inspire CPU Notebook. Compute workers
# recreate the locked environment strictly from the shared cache.
ARGS=(sync --project "${ROOT}" --python 3.12 --locked)
if [[ "${1:-}" != "--online" ]]; then
    ARGS+=(--offline)
fi
exec "${UV_BIN}" "${ARGS[@]}"

#!/usr/bin/env bash
# Run only on a newly created Inspire CPU preparation Notebook.
set -euo pipefail
if [[ "${INSPIRE_CPU_DOWNLOAD:-0}" != 1 ]]; then
    echo 'Run on a new CPU资源空间 Notebook with INSPIRE_CPU_DOWNLOAD=1.' >&2
    exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /inspire/hdd/global_user/250041225002/uv_config/env.sh
cd "${ROOT}"
WHEEL_NAME='flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
WHEEL_REL="../pretrained_weights/wheels/${WHEEL_NAME}"
CANONICAL_REL='../pretrained_weights/wheels/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl'
mkdir -p ../pretrained_weights/wheels
uv run --offline --no-sync python - <<'PY'
import platform, sys, torch
assert sys.version_info[:2] == (3, 12) and platform.machine() == 'x86_64'
assert torch.__version__.startswith('2.9.') and torch.version.cuda.startswith('12.')
assert torch._C._GLIBCXX_USE_CXX11_ABI
print('Verified wheel ABI:', torch.__version__, torch.version.cuda, flush=True)
PY
curl --fail --location --retry 3 --max-time 120 \
    https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.8.3 \
    -o ../pretrained_weights/wheels/flash-attn-v2.8.3-release.json
aria2c --continue=true --max-connection-per-server=16 --split=16 --min-split-size=1M \
    --max-tries=5 --retry-wait=3 --timeout=60 --summary-interval=30 \
    --dir=../pretrained_weights/wheels --out="${WHEEL_NAME}" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL_NAME}"
uv run --offline --no-sync python - "${WHEEL_REL}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); release=json.loads((p.parent/'flash-attn-v2.8.3-release.json').read_text())
asset=next(a for a in release['assets'] if a['name']==p.name)
digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert p.stat().st_size==asset['size'], 'wheel size mismatch'
if asset.get('digest'):
    assert asset['digest']=='sha256:'+digest, 'wheel SHA256 mismatch'
canonical=p.with_name('flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl')
if not canonical.exists():
    os.link(p, canonical)
assert hashlib.file_digest(canonical.open('rb'),'sha256').hexdigest()==digest
(p.parent/'flash-attn-source.json').write_text(json.dumps(dict(version='2.8.3',url=asset['browser_download_url'],sha256=digest,size=p.stat().st_size,local_wheel=canonical.name),indent=2)+'\n')
print('Verified official wheel:',digest,flush=True)
PY
# A shared local wheel makes fresh offline environments reproducible without
# relying on an expiring cache or on GitHub access from a GPU worker.
# The official filename advertises a local ABI version but METADATA says 2.8.3.
# A byte-identical hard link with the canonical version avoids endless uv
# reinstall attempts caused by that mismatch. The original ABI name is retained.
uv add --offline --no-sync "${CANONICAL_REL}"
uv sync --offline --locked --link-mode symlink
uv run --offline --no-sync python -c 'import torch, flash_attn; print(torch.__version__, flash_attn.__version__)'

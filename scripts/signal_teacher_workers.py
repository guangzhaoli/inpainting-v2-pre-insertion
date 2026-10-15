"""Gracefully checkpoint only the workers running the specified config."""

import argparse
import os
from pathlib import Path
import signal

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
found = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = (proc/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if ('-m' in argv and argv[argv.index('-m')+1] == 'src.main'
                and 'python' in Path(argv[0]).name
                and (proc/'comm').read_text().strip().startswith('python')
                and '--config' in argv and argv[argv.index('--config')+1] == args.config):
            found.append(int(proc.name))
    except (OSError, ValueError, IndexError):
        continue
if len(found) != 8:
    raise RuntimeError(f'Expected exactly eight matching workers, found {found}')
for pid in found:
    os.kill(pid, signal.SIGINT)
print('Sent SIGINT to eight training workers:', found, flush=True)

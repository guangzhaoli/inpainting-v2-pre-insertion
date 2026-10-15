import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('nodes,rank', [(2, 1), (4, 3), (8, 7)])
def test_teacher_launcher_maps_platform_topology(nodes, rank):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for key in ('NUM_MACHINES', 'MACHINE_RANK', 'TOTAL_PROCESSES', 'NPROC_PER_NODE',
                'MAIN_PROCESS_IP', 'MAIN_PROCESS_PORT', 'CONFIG_FILE', 'ACCELERATE_CONFIG'):
        env.pop(key, None)
    env.update(DRY_RUN='1', WORLD_SIZE=str(nodes), RANK=str(rank), MASTER_ADDR='master.test', MASTER_PORT='29501')
    result = subprocess.run(['bash', str(root/'scripts/run_teacher_v2.sh'), 'human'],
                            env=env, capture_output=True, text=True, check=True)
    assert f'--num_processes {nodes*8}' in result.stdout
    assert f'--num_machines {nodes}' in result.stdout
    assert f'--machine_rank {rank}' in result.stdout
    assert '--main_process_ip master.test' in result.stdout

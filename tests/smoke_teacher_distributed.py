"""Run via torchrun --standalone --nproc_per_node=2; no Qwen weights or GPUs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from src.data.config import SamplingConfig
from src.trainer.insert_anything_validation_sampling import InsertAnythingValidationSampler
from src.utils.seed import prepare_trainable_precision, seed_training_rank
from tests.test_validation_sampling import FakeDataset, make_config


def main():
    torch.set_num_threads(2)
    accelerator = Accelerator(cpu=True)
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 3).bfloat16()
    prepare_trainable_precision(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer = accelerator.prepare(model, optimizer)
    seed_training_rank(42, accelerator.process_index)
    draw = torch.rand(1)
    values = accelerator.gather(draw)
    assert values.unique().numel() == accelerator.num_processes
    for _ in range(2):
        loss = model(torch.randn(2, 4)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())

    root = Path(__file__).resolve().parents[1] / 'output/teacher_v2_distributed_smoke'
    sampling = SamplingConfig(enable=True, validation_steps=1, num_samples=3,
                              seed=42, fixed_seed_across_steps=True, num_inference_steps=2)
    config = make_config(str(root), sampling)
    sampler = InsertAnythingValidationSampler(config, SimpleNamespace(dataset=FakeDataset()), accelerator)

    class Trainer:
        weight_dtype = torch.float32

        def __init__(self):
            self.accelerator = accelerator
            self.calls = 0

        def predict_loaded_model(self, **kwargs):
            self.calls += 1
            with torch.no_grad():
                color = model(torch.randn(1, 4)).sigmoid()[0].cpu().numpy()
            return [Image.fromarray(np.tile((color * 255).astype(np.uint8), (32, 32, 1)))]

    trainer = Trainer()
    rng = torch.get_rng_state().clone()
    sampler.run_validation_loop(1, trainer)
    assert torch.equal(rng, torch.get_rng_state())
    calls = accelerator.gather(torch.tensor([trainer.calls]))
    assert calls.unique().numel() == 1 and calls[0] == 2
    if accelerator.is_main_process:
        metadata = json.loads((root/'validation/step-00000001/metadata.json').read_text())
        assert len(metadata['samples']) == 3
        assert sorted(r['order'] for r in metadata['samples']) == [0, 1, 2]
        report = {'world_size': accelerator.num_processes, 'distinct_rng_draws': values.tolist(),
                  'validation_forward_calls_per_rank': calls.tolist(), 'unique_validation_outputs': 3,
                  'rng_restored': True, 'trainable_precision': 'float32'}
        (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()

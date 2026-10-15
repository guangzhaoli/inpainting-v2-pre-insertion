"""CPU/Gloo regression for PEFT switching, micro-backward and late SIGINT."""

import json, tempfile
from types import SimpleNamespace as NS
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
from src.distillation.opd import QwenImageEditOPDStrategy
from src.distillation.backends import set_active_adapter
from src.trainer.base_trainer import BaseTrainer
from tests.test_opd_strategy import FakeTrainer, _plugin, _guidance, _teacher


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, hidden_states, **kwargs):
        y = (
            checkpoint(self.proj, hidden_states, use_reentrant=False)
            if torch.is_grad_enabled()
            else self.proj(hidden_states)
        )
        return (y,)


class Accel:
    device = torch.device("cpu")
    num_processes = 2

    def __init__(self, rank):
        self.process_index = rank

    def no_sync(self, model):
        return model.no_sync()

    def backward(self, loss):
        (loss / 2).backward()

    def reduce(self, tensor, reduction="sum"):
        dist.all_reduce(tensor)
        return tensor


def worker(rank, init_file):
    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", init_method="file://" + init_file, rank=rank, world_size=2
    )
    torch.manual_seed(42)
    lora = LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])
    model = get_peft_model(Tiny(), lora, adapter_name="student")
    for name in ["teacher_animal", "teacher_object"]:
        model.add_adapter(name, lora)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".teacher_animal." in n:
                p.fill_(0.2)
            if ".teacher_object." in n:
                p.fill_(0.3)
    set_active_adapter(model, "student", trainable_adapter="student")
    trainer = FakeTrainer()
    trainer.accelerator = Accel(rank)
    trainer.prepare_initial_latents = lambda batch_size, **kw: torch.ones(
        batch_size, 2, 3
    )
    trainer.dit = DDP(model, find_unused_parameters=True)
    trainer._dit_config = lambda: NS(in_channels=4, guidance_embeds=False)
    trainer._unwrap_dit = lambda: trainer.dit.module
    trainer.config = NS(data=NS(batch_size=1))
    trainer.adapter_name = "student"
    teachers = [_teacher("teacher_animal", "cat"), _teacher("teacher_object", "object")]
    config = NS(
        objective=_plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=_plugin("src.distillation.backends.SharedLoRATeacherBackend"),
        teachers=teachers,
        fallback_teacher=None,
        student_guidance=_guidance(),
        rollout=NS(
            num_inference_steps=2, timestep_fraction=1.0, store_device="cpu", seed=7
        ),
    )
    strategy = QwenImageEditOPDStrategy(config)
    strategy.backend._trainer = trainer
    strategy.backend._student_adapter = "student"
    strategy.backend._teacher_to_adapter = {t.name: t.name for t in teachers}
    strategy._is_setup = True
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    results = []
    for iteration in range(2):
        for micro in range(2):
            with trainer.dit.no_sync() if micro == 0 else nullcontext():
                loss = strategy.backward_step(
                    trainer,
                    {"category": ["cat" if rank == 0 else "object"], "prompt": [""]},
                )
                assert torch.isfinite(loss) and not loss.requires_grad
        grads = torch.cat(
            [p.grad.flatten() for n, p in model.named_parameters() if ".student." in n]
        )
        gathered = [torch.empty_like(grads) for _ in range(2)]
        dist.all_gather(gathered, grads)
        assert torch.equal(*gathered)
        assert torch.isfinite(grads).all() and grads.abs().sum() > 0
        assert all(
            p.grad is None for n, p in model.named_parameters() if ".teacher_" in n
        )
        results.append(
            {
                "iteration": iteration,
                "grad_norm": grads.norm().item(),
                "student_grads_equal": True,
                "teachers_frozen": True,
            }
        )
        opt.step()
        opt.zero_grad(set_to_none=True)
        trainer.global_step += 1
    if rank == 0:
        print("real_ddp_opd:", json.dumps(results), flush=True)
    # Inject the same effect as SIGINT between flag capture and collective return.
    interrupted = NS(training_interrupted=False, accelerator=Accel(rank))
    original = dist.all_reduce

    def delayed_signal(flag, op):
        original(flag, op=op)
        if rank == 0:
            interrupted.training_interrupted = True

    dist.all_reduce = delayed_signal
    decision = BaseTrainer._sync_training_interrupted(interrupted)
    dist.all_reduce = original
    flags = [None, None]
    dist.all_gather_object(flags, decision)
    assert flags == [False, False]
    if rank == 0:
        print("interrupt_decisions_after_same_collective:", flags, flush=True)
    decision = BaseTrainer._sync_training_interrupted(interrupted)
    dist.all_gather_object(flags, decision)
    assert flags == [True, True]
    if rank == 0:
        print("interrupt_decisions_next_boundary:", flags, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="insertfuse_gloo_") as directory:
        torch.multiprocessing.start_processes(
            worker,
            args=(directory + "/rendezvous",),
            nprocs=2,
            start_method="fork",
            join=True,
        )

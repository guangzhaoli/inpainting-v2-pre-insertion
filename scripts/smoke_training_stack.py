"""Offline GPU integration check: FA2, full teacher backward, validation, OPD.

Run with torchrun on every allocated GPU. Uses real local model/loss weights and
real input images. The five teacher adapters are initialization fixtures, so
this checks execution/gradients/memory, not trained teacher image quality.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft.utils import get_peft_model_state_dict

from src.data.config import load_config_from_yaml
from src.data.dataset import collate_fn
from src.data.insert_anything_dataset import InsertAnythingDataset
from src.distillation.opd import QwenImageEditOPDStrategy
from src.loss.local_detail import interaction_masks
from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
from src.utils.opencv_runtime import configure_opencv_runtime
from src.utils.seed import prepare_trainable_precision, seed_everything, seed_training_rank


def kernel_checks(device):
    from flash_attn import flash_attn_func
    torch.manual_seed(123)
    inputs = [torch.randn(1, 256, 4, 64, device=device, dtype=torch.bfloat16) for _ in range(3)]
    a = [x.clone().requires_grad_() for x in inputs]
    b = [x.clone().requires_grad_() for x in inputs]
    actual = flash_attn_func(*a)
    expected = F.scaled_dot_product_attention(*(x.transpose(1, 2) for x in b)).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=.01, rtol=.02)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for left, right in zip(a, b):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=.03)
    mask = torch.zeros(1, 1, 1024, 1024, device=device)
    mask[:, :, 256:768, 256:768] = 1
    def original():
        d = F.max_pool2d(mask, 65, 1, 32)
        e = -F.max_pool2d(-mask, 65, 1, 32)
        return (d-e).clamp(0, 1), 1-d
    for left, right in zip(original(), interaction_masks(mask, 32)):
        assert torch.equal(left, right)
    timings = {}
    for name, function in (("original", original), ("separable", lambda: interaction_masks(mask, 32))):
        function()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            function()
        end.record(); end.synchronize()
        timings[name] = start.elapsed_time(end) / 5
    return {"fa2_max_error":float((actual-expected).abs().max()), "morphology_ms":timings}


def pil_image(tensor, mask=False):
    array = tensor.detach().cpu().numpy()
    if not mask:
        array = array.transpose(1, 2, 0)
    return Image.fromarray((array.clip(0, 1)*255).round().astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher_v2/human.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_opencv_runtime()
    torch.set_num_threads(2)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    report = kernel_checks(device)
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    def save_progress(stage):
        payload = dict(report, rank=rank, stage=stage)
        (args.output / f"progress-{rank:03d}.json").write_text(json.dumps(payload, indent=2)+'\n')
    save_progress("kernels_checked")
    seed_everything(42)
    cfg = load_config_from_yaml(args.config)
    cfg.logging.report_to = None
    cfg.logging.output_dir = str(args.output / "trainer")
    cfg.logging.sampling.enable = False
    cfg.model.lora.adapter_name = "smoke_student"
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.num_epochs = 1
    cfg.train.max_train_steps = 2
    cfg.lr_scheduler.warmup_steps = 0
    trainer = QwenImageEditInsertAnythingTrainer(cfg)
    trainer.setup_accelerator()
    trainer.load_model()
    report["zero_cond_t"] = bool(trainer.dit.config.zero_cond_t)
    trainer.load_pretrain_lora_model(trainer.dit, cfg, trainer.adapter_name)
    expert_names = [f"smoke_teacher_{i}" for i in range(5)]
    frozen_geometry = {}
    student_weights = get_peft_model_state_dict(trainer.dit, adapter_name=trainer.adapter_name)
    for name in expert_names:
        fixture = cfg.model_copy(deep=True)
        fixture.model.lora.adapter_name = name
        trainer.add_lora_adapter(trainer.dit, fixture, name)
        trainer._load_peft_lora_state_dict(trainer.dit, student_weights, name, "teacher")
        frozen_geometry[name] = copy.deepcopy(trainer._get_mask_geometry_tokenizer()).requires_grad_(False).eval()
    del student_weights
    trainer.setup_model_device_train_mode("fit")
    prepare_trainable_precision(trainer.dit)
    trainer.load_resume_progress()
    trainer.configure_optimizers()
    trainer.setup_criterion()
    trainer.validation_sampler = None
    dataset = InsertAnythingDataset(cfg.data.init_args)
    world = trainer.accelerator.num_processes
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [i % len(dataset) for i in range(world)]),
        batch_size=1, collate_fn=collate_fn,
    )
    loader = trainer.prepare_resume_dataloader(loader)
    loader = trainer.accelerator_prepare(loader)
    trainer.effective_max_train_steps = 2
    trainer.configure_lr_scheduler()
    trainer.setup_progressbar()
    trainer.fps_logger.start()
    seed_training_rank(42, trainer.accelerator.process_index)
    prepare = trainer.prepare_embeddings
    def low_sigma_batch(*a, **kw):
        result = prepare(*a, **kw)
        if kw.get("stage") == "fit":
            result["_analysis_timestep_u"] = .9
        return result
    trainer.prepare_embeddings = low_sigma_batch
    encode_vae = trainer._encode_vae_image
    def check_vae_placement(*args, **kwargs):
        if cfg.train.offload_text_encoder:
            assert trainer.text_encoder.device.type == "cpu", "VAE encoding overlaps the offloaded text encoder"
        return encode_vae(*args, **kwargs)
    trainer._encode_vae_image = check_vae_placement
    step_logs = []
    update = trainer.update_progressbar
    def record_update(logs):
        assert np.isfinite(logs["loss"]) and np.isfinite(logs["optim/grad_norm"])
        assert "hybrid/condition_area_mean" in logs
        step_logs.append(dict(logs))
        update(logs)
    trainer.update_progressbar = record_update
    start = time.perf_counter()
    trainer.train_epoch(0, loader)
    assert trainer.global_step == 1 and step_logs[-1]["train/region_w_t_mean"] > 0
    report["supervised"] = dict(logs=step_logs[-1], seconds=time.perf_counter()-start, full_train_epoch=True)
    save_progress("supervised_train_epoch_checked")
    batch = collate_fn([dataset[trainer.accelerator.process_index % len(dataset)]])
    images = trainer.predict_loaded_model(
        image=[pil_image(batch["tar_image"][0]), pil_image(batch["ref_image"][0])], prompt="",
        src_mask=pil_image(batch["tar_mask"][0], True), ref_mask=pil_image(batch["ref_mask"][0], True),
        num_inference_steps=2, true_cfg_scale=2.5, weight_dtype=trainer.weight_dtype,
    )
    assert len(images) == 1
    save_progress("validation_checked")
    plugin = lambda path: NS(class_path=path, init_args={})
    guidance = NS(mode="reference_cfg", scale=2.5, prompt_mode="empty", mask_aware=True,
                  delta_clip_enabled=True, delta_clip_max_ratio=1.)
    experts = [NS(name=name, route_categories=[f"smoke_category_{i}"], weight=1., guidance=guidance)
               for i, name in enumerate(expert_names)]
    strategy = QwenImageEditOPDStrategy(NS(
        objective=plugin("src.distillation.objectives.VelocityMatchingObjective"),
        teacher_backend=plugin("src.distillation.backends.SharedLoRATeacherBackend"), teachers=experts,
        fallback_teacher=None, student_guidance=NS(mode="none", scale=1., prompt_mode="empty"),
        rollout=NS(num_inference_steps=10, timestep_fraction=1., store_device="cuda", seed=42),
    ))
    strategy.backend._trainer = trainer
    strategy.backend._student_adapter = trainer.adapter_name
    strategy.backend._teacher_to_adapter = {name:name for name in expert_names}
    strategy.backend._teacher_geometry_tokenizers = frozen_geometry
    strategy._is_setup = True
    trainer.training_strategy = strategy
    cfg.train.gradient_accumulation_steps = 5
    trainer.accelerator.gradient_accumulation_steps = 5
    # Three microbatches exercise the short final window of the five-teacher
    # accumulation recipe. Each rank routes to one of the five mounted experts.
    def opd_collate(samples):
        prepared = collate_fn(samples)
        prepared["category"] = [f"smoke_category_{trainer.accelerator.process_index % 5}"]
        return prepared
    opd_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, [i % len(dataset) for i in range(world * 3)]),
        batch_size=1, collate_fn=opd_collate,
    )
    opd_loader = trainer.accelerator.prepare_data_loader(opd_loader, device_placement=False)
    original_backward = trainer.backward_training_step
    microbatches = 0
    def check_backward(value):
        nonlocal microbatches
        loss = original_backward(value)
        assert torch.isfinite(loss)
        assert all(p.grad is None for name,p in trainer._unwrap_dit().named_parameters() if ".smoke_teacher_" in name)
        microbatches += 1
        save_progress(f"opd_microbatch_{microbatches}_checked")
        return loss
    trainer.backward_training_step = check_backward
    start = time.perf_counter()
    trainer.train_epoch(1, opd_loader)
    assert trainer.global_step == 2 and microbatches == 3
    assert step_logs[-1]["optim/grad_norm"] > 0
    report["opd"] = dict(logs=step_logs[-1], seconds=time.perf_counter()-start,
                         teacher_count=5, teacher_frozen=True, steps_per_rollout=10,
                         accumulation=5, actual_tail_microbatches=3, full_train_epoch=True)
    report.update(rank=trainer.accelerator.process_index, world_size=world,
                  gpu=torch.cuda.get_device_name(), max_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    (args.output / f"rank-{report['rank']:03d}.json").write_text(json.dumps(report, indent=2)+'\n')
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        print(json.dumps(report), flush=True)
    trainer.accelerator.end_training()


if __name__ == "__main__":
    main()

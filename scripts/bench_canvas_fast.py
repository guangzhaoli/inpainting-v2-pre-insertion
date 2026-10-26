"""Speed and equivalence of the fast complementary-canvas sampler (B2 + cached conditions).

One GPU per process; the training run's validation cases (same seeds, crops, outputs) are
sharded over processes with ``--shard i --num-shards n``. Per case and variant it records the
DiT sampling time (condition pass, graph capture and the 40 steps), peak memory and the output;
every variant is compared with the trainer's cached path (``reference``): latent / image
difference, reference-DINO difference and, per step, the relative L2 of each branch's and the
guided prediction on the reference trajectory's inputs. Variants run in a fixed order because
merging LoRA is in place: the unmerged ones first.

  python scripts/bench_canvas_fast.py run --checkpoint <ckpt dir> --out <dir> [--shard 0 --num-shards 8]
      [--variants reference,full,...] [--manifest <validation json>] [--cases 16]
      [--scaling-manifest <json> --scaling-cases 24 --scaling-variants reference,equivalent,fast]
  python scripts/bench_canvas_fast.py summarize --out <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.inference.canvas_fast_sampler import FastCanvasOptions, FastCanvasSampler, merge_lora_  # noqa: E402

# name -> options; None is the trainer's own cached path. Merging LoRA is in place, so every unmerged
# variant runs first. "equivalent" stacks every technique that keeps the per-step outputs at the
# batch-size noise floor; "fast" adds the bf16 LoRA merge.
BASE = dict(merge_lora=True, fuse_qkv=True, batch_cfg=True, kv_buffer=True, cuda_graph=True, attention="flash",
            modulation_table=False, fused_kernels=False, fused_lora=False)  # explicit: library defaults may change


def O(**fields):  # noqa: E743 - short on purpose, the table below reads as a list of switches
    return FastCanvasOptions(**{**BASE, **fields})


EQUIVALENT = dict(merge_lora=False, modulation_table=True, fused_kernels=True)
VARIANTS = {
    "reference": None,
    "graph": O(merge_lora=False, fuse_qkv=False, batch_cfg=False, kv_buffer=False),
    "graph+batch": O(merge_lora=False, fuse_qkv=False, kv_buffer=False),
    "graph+batch+buffer": O(merge_lora=False, fuse_qkv=False),
    "graph+batch+buffer+fuse": O(merge_lora=False),
    "graph+batch+buffer+fuse+table": O(merge_lora=False, modulation_table=True),
    "equivalent": O(**EQUIVALENT),
    "equivalent-graph": O(**EQUIVALENT, cuda_graph=False),
    "equivalent-batch": O(**EQUIVALENT, batch_cfg=False),
    "equivalent-buffer": O(merge_lora=False, modulation_table=True, kv_buffer=False),  # fused kernels write the buffer
    "equivalent-fuse": O(**EQUIVALENT, fuse_qkv=False),
    "equivalent-table": O(merge_lora=False, fused_kernels=True),
    "equivalent+cudnn": O(**EQUIVALENT, attention="cudnn"),
    "equivalent+lora": O(**EQUIVALENT, fused_lora=True),
    "reference_merged": None,
    "fast": O(modulation_table=True, fused_kernels=True),
    "fast+cudnn": O(modulation_table=True, fused_kernels=True, attention="cudnn"),
}
UNMERGED = tuple(name for name, options in VARIANTS.items()
                 if name != "reference_merged" and (options is None or not options.merge_lora))
CHECK_STEPS = (0, 10, 20, 30, 39)  # steps of the batch-2 noise floor


class ShardProcess:
    """Accelerator stand-in: the validation loop runs orders ``shard, shard + n, ...`` on this GPU."""

    is_main_process = True
    device = torch.device("cuda")

    def __init__(self, shard: int, shards: int):
        self.process_index, self.num_processes = shard, shards

    def wait_for_everyone(self):
        torch.cuda.synchronize()


def sync_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def load_trainer(checkpoint: Path, out: Path):
    from infer import resolve_lora_path
    from src.data.config import DeviceConfig, load_config_from_yaml
    from src.loss.region_perceptual_loss import RegionPerceptualLoss
    from src.trainer.qwen_image_edit_insert_anything_trainer import QwenImageEditInsertAnythingTrainer
    from src.utils.inference_metadata import validate_checkpoint_compatibility

    lora = resolve_lora_path(checkpoint)
    config = load_config_from_yaml(str(checkpoint.parent / "train_config.yaml"), for_inference=True)
    config.predict.devices = DeviceConfig(vae="cuda", text_encoder="cuda", dit="cuda")
    config.model.lora.pretrained_weight = str(lora)
    config.model.complementary_canvas.cache_conditions = True
    config.logging.output_dir = str(out)  # never write into the checkpoint's run
    config.logging.sampling.output_dir = str(out / "validation")
    compatibility = validate_checkpoint_compatibility(config, lora, allow_mismatch=False)
    # Staged copies (e.g. /dev/shm): mmap loads from the shared filesystem fault page by page.
    if os.environ.get("CANVAS_FAST_MODEL_ROOT"):
        config.model.pretrained_model_name_or_path = os.environ["CANVAS_FAST_MODEL_ROOT"]
    if os.environ.get("CANVAS_FAST_LORA_DIR"):  # the LoRA and its mask-tokenizer companion
        config.model.lora.pretrained_weight = str(Path(os.environ["CANVAS_FAST_LORA_DIR"]) / Path(lora).name)
    trainer = QwenImageEditInsertAnythingTrainer(config)
    trainer.setup_predict()
    loss = config.loss
    trainer.region_perceptual_criterion = RegionPerceptualLoss(
        fg_metric=loss.fg_metric, lpips_net=loss.lpips_net, lpips_input_size=loss.lpips_input_size,
        lpips_bbox_pad_ratio=loss.lpips_bbox_pad_ratio, mask_threshold=loss.mask_threshold,
        lpips_linear_path=loss.lpips_linear_path, lpips_torch_home=loss.lpips_torch_home,
        lpips_trunk_path=loss.lpips_trunk_path, enable_ref_dino=True, dino_backend=loss.dino_backend,
        dino_model_name=loss.dino_model_name, dino_input_size=loss.dino_input_size,
        dino_bbox_pad_ratio=loss.dino_bbox_pad_ratio,
    ).to("cuda").eval()
    return trainer, config, compatibility


class Recorder:
    """Wraps the predict stages; one record per predicted case."""

    def __init__(self, trainer):
        self.trainer = trainer
        self.records, self.current = [], {}
        self.prepare, self.decode = trainer.prepare_embeddings, trainer.decode_vae_latent
        self.reference_sampling = trainer.sampling_from_embeddings
        self.sampling = self.reference_sampling  # set per variant
        self.after_sampling = None  # untimed per-case hook (per-step checks)
        trainer.prepare_embeddings = self._prepare
        trainer.sampling_from_embeddings = self._sample
        trainer.decode_vae_latent = self._decode

    def restore(self):
        for name in ("prepare_embeddings", "sampling_from_embeddings", "decode_vae_latent"):
            self.trainer.__dict__.pop(name, None)

    def _prepare(self, batch, stage="fit"):
        self.current = {}
        start = sync_time()
        embeddings = self.prepare(batch, stage=stage)
        self.current["prepare_s"] = sync_time() - start
        null = embeddings.get("reference_cfg_embeddings")
        self.current["tokens"] = {
            "text": int(embeddings["prompt_embeds"].shape[1]),
            "text_null": int(null["prompt_embeds"].shape[1]) if null is not None else None,
        }
        return embeddings

    def _sample(self, embeddings, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        start = sync_time()
        latents = self.sampling(embeddings, **kwargs)
        self.current["sampling_s"] = sync_time() - start
        self.current["sampling_peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        self.current["latents"] = latents.detach().float().cpu()
        self.current["canvas"] = dict(getattr(self.trainer, "_last_complementary_canvas", None) or {})
        if self.after_sampling is not None:
            start = sync_time()
            self.current["check"] = self.after_sampling(embeddings)
            self.current["check_s"] = sync_time() - start
        return latents

    def _decode(self, latents, height, width):
        start = sync_time()
        image = self.decode(latents, height, width)
        self.current["decode_s"] = sync_time() - start
        self.current["output"] = image.detach().float().cpu()
        self.records.append(dict(self.current))
        return image


class TrajectoryTap:
    """Records every DiT call of the reference loop: (hidden_states, timestep, output) per branch."""

    def __init__(self, dit):
        self.calls, self.enabled = [], False
        self.handle = dit.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, _module, _args, kwargs, output):
        if self.enabled:
            self.calls.append({
                "x": kwargs["hidden_states"].detach().cpu(), "t": kwargs["timestep"].detach().cpu(),
                "out": output[0].detach().cpu(),
            })

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def relative(value, base):
    value, base = value.float(), base.float()
    return float((value - base).norm() / base.norm())


def per_step_check(sampler: FastCanvasSampler, embeddings, reference_calls):
    """The fast step on the reference trajectory's inputs: relative L2 per step of the positive,
    null and guided predictions (reference: the trainer's cached forward)."""

    with torch.inference_mode():
        prepared = sampler.prepare(embeddings)
        forward = sampler.build_forward(prepared)
        step = sampler.step_function(prepared, forward)
        branches = len(prepared["branches"])
        rows = {"positive": [], "null": [], "guided": []}
        dtype = prepared["latents"].dtype
        for index in range(0, len(reference_calls), branches):
            calls = reference_calls[index : index + branches]
            scaled = sampler.scaled_timestep(prepared["timesteps"][index // branches], dtype).cpu()
            if not torch.equal(scaled, calls[0]["t"]):
                raise RuntimeError(f"step {index // branches}: schedule {scaled} != reference timestep {calls[0]['t']}")
            outputs = [value.clone() for value in step(calls[0]["x"].cuda(), index // branches)]
            expected = [call["out"].cuda() for call in calls]
            rows["positive"].append(relative(outputs[0], expected[0]))
            if branches == 2:
                rows["null"].append(relative(outputs[1], expected[1]))
            rows["guided"].append(relative(outputs[-1], sampler.combine(prepared, expected)))
        del step, forward
    return rows


def reference_check(trainer, embeddings, reference_calls, batch: int):
    """The trainer's cached forward, called as its loop calls it, on the reference trajectory's inputs at
    CHECK_STEPS with the batch duplicated ``batch`` times; relative L2 against the recorded outputs per
    branch and for the guided prediction. ``batch=2`` on the reference weights is round 1's noise floor
    (GEMM shapes only); ``batch=1`` on merged weights is the deviation of the LoRA merge alone."""

    helper = FastCanvasSampler(trainer, O(merge_lora=False, fuse_qkv=False, batch_cfg=False, kv_buffer=False,
                                          cuda_graph=False))
    dit = trainer._unwrap_dit()
    with torch.inference_mode():
        prepared = helper.prepare(embeddings)
        plan, branches, passes = prepared["plan"], prepared["branches"], prepared["passes"]
        device, dtype = trainer._dit_device(), trainer.weight_dtype

        def build_cache(condition):  # the trainer's build_cache: the pass's isolation and its text condition
            text = {}
            if condition.text is not None:
                prompt = branches[condition.text].prompt_embeds.to(device, dtype).repeat(batch, 1, 1)
                text = {"encoder_hidden_states": prompt, "txt_seq_lens": [prompt.shape[1]] * batch}
            return dit.build_condition_cache(condition.states.repeat(batch, 1, 1), condition.img_shapes * batch,
                                             plan.image_token_index, isolation=condition.isolation, **text)

        cache = build_cache(passes[0])
        caches = [cache]
        if len(branches) == 2:
            caches.append(cache.select([0]) if len(passes) == 1 else build_cache(passes[1]))
        text_condition = {"text_as_condition": True} if prepared["text_condition"] else {}
        geometry = prepared["mask_geometry_tokens"]
        geometry = None if geometry is None else geometry.repeat(batch, 1, 1)
        rows = {"positive": [], "null": [], "guided": []}
        count = len(branches)
        for step in CHECK_STEPS:
            calls = reference_calls[count * step : count * step + count]
            outputs = []
            for call, branch, branch_cache in zip(calls, branches, caches):
                prompt = branch.prompt_embeds.to(device, dtype).repeat(batch, 1, 1)
                outputs.append(dit(
                    hidden_states=call["x"].to(device).repeat(batch, 1, 1), timestep=call["t"].to(device).repeat(batch),
                    encoder_hidden_states=prompt,
                    encoder_hidden_states_mask=torch.ones(batch, prompt.shape[1], device=device, dtype=torch.int64),
                    img_shapes=branch.img_shapes * batch, txt_seq_lens=[prompt.shape[1]] * batch,
                    mask_geometry_tokens=geometry, image_token_index=plan.image_token_index,
                    condition_cache=branch_cache, output_token_count=call["x"].shape[1], return_dict=False,
                    **text_condition,
                )[0][:1])
            expected = [call["out"].to(device) for call in calls]
            rows["positive"].append(relative(outputs[0], expected[0]))
            if count == 2:
                rows["null"].append(relative(outputs[1], expected[1]))
            rows["guided"].append(relative(helper.combine(prepared, outputs), helper.combine(prepared, expected)))
        del caches, cache
    return {"steps": list(CHECK_STEPS), **rows}


@contextlib.contextmanager
def lean_image_saves(keep=("_output.png", "_row.png")):
    """The validation loop writes ~13 full-resolution PNGs per case (source, masks, overlays, crops,
    composites; seconds each on the shared disk). Keep only the output and the small row panel."""

    from PIL import Image

    original = Image.Image.save

    def save(image, fp, *args, **kwargs):
        name = str(fp) if isinstance(fp, (str, os.PathLike)) else ""
        if name and "/step-" in name and not name.endswith(keep):
            return None
        if name.endswith(".png"):
            kwargs.setdefault("compress_level", 1)
        return original(image, fp, *args, **kwargs)

    Image.Image.save = save
    try:
        yield
    finally:
        Image.Image.save = original


def run_loop(trainer, config, shard, shards, out: Path, name: str, step: int, num_samples=None, manifest=None):
    from src.trainer.insert_anything_validation_sampling import InsertAnythingValidationSampler

    sampling = config.logging.sampling
    config.logging.sampling.output_dir = str(out / name)
    saved = sampling.num_samples, sampling.validation_data
    if num_samples is not None:
        sampling.num_samples = num_samples
    if manifest is not None:
        sampling.validation_data = str(manifest)
    try:
        sampler = InsertAnythingValidationSampler(config, None, ShardProcess(shard, shards))
        start = sync_time()
        with lean_image_saves():
            sampler.run_validation_loop(global_step=step, trainer=trainer)
        elapsed = sync_time() - start
    finally:
        sampling.num_samples, sampling.validation_data = saved
    metadata = json.loads((out / name / f"step-{step:08d}" / "metadata.json").read_text())
    return metadata, elapsed


def case_entry(record, sample, reference=None):
    entry = {
        "case": f"{sample['category']}-{sample['stem']}",
        **{key: record.get(key) for key in ("prepare_s", "sampling_s", "decode_s", "sampling_peak_gib", "check_s")},
        "tokens": record["tokens"], "canvas": record.get("canvas"),
        "reference_dino_cosine": sample.get("metrics", {}).get("reference_dino_cosine"),
        "fast": record.get("fast"), "check": record.get("check"), "floor": record.get("floor"),
    }
    if reference is not None:
        mse = float(((record["output"] - reference["output"]) ** 2).mean())
        entry["vs_reference"] = {
            "latent_rel_l2": relative(record["latents"], reference["latents"]),
            "latent_max_abs": float((record["latents"] - reference["latents"]).abs().max()),
            "image_psnr_db": 10 * math.log10(1.0 / mse) if mse > 0 else float("inf"),
            "image_max_abs_255": float((record["output"] - reference["output"]).abs().max() * 255),
            "speedup": reference["sampling_s"] / record["sampling_s"],
        }
    return entry


def run(args):
    out = args.out / f"shard{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    loaded = time.perf_counter()
    trainer, config, compatibility = load_trainer(args.checkpoint, out)
    run_loaded(args, trainer, config, compatibility, time.perf_counter() - loaded)


def run_loaded(args, trainer, config, compatibility, load_s):
    """The benchmark on an already loaded trainer (unmerged LoRA; it is merged in place on the way)."""

    out = args.out / f"shard{args.shard:02d}"
    out.mkdir(parents=True, exist_ok=True)
    step = int(json.loads((args.checkpoint / "state.json").read_text())["global_step"])
    # phase -> (manifest, cases, variants, per-step checks, noise floor)
    phases = {"main": (args.manifest, args.cases, args.variants.split(",") if args.variants else list(VARIANTS),
                       args.check, args.floor)}
    if args.scaling_manifest is not None:
        phases["scaling"] = (args.scaling_manifest, args.scaling_cases, args.scaling_variants.split(","), False, False)
    requested = {name for spec in phases.values() for name in spec[2]}
    if requested - set(VARIANTS):
        raise ValueError(f"unknown variants {sorted(requested - set(VARIANTS))}")
    recorder = Recorder(trainer)
    tap = TrajectoryTap(trainer._unwrap_dit())
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "shard": args.shard,
              "num_shards": args.num_shards, "checkpoint": str(args.checkpoint), "step": step,
              "model_load_s": load_s, "checkpoint_compatibility": compatibility.get("status"),
              "phases": {phase: {"manifest": str(spec[0]) if spec[0] else None, "variants": {}}
                         for phase, spec in phases.items()}}
    def save():
        (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))

    try:
        _run_variants(args, trainer, config, out, step, phases, requested, recorder, tap, result, save)
    finally:
        recorder.restore()
        tap.handle.remove()
    save()


def _run_variants(args, trainer, config, out, step, phases, requested, recorder, tap, result, save):
    references = {phase: None for phase in phases}
    trajectories: list = []
    merged = False
    # One untimed case warms kernels / the allocator before anything is timed. Warm-ups run as the
    # only shard: a shard > 0 would get no real sample (the loop then has no row to stack).
    run_loop(trainer, config, 0, 1, out, "warmup", step, num_samples=1)
    for name in [name for name in VARIANTS if name in requested]:  # unmerged variants first
        options = VARIANTS[name]
        if name not in UNMERGED and not merged:
            result["merged_lora_layers"] = merge_lora_(trainer._unwrap_dit())
            merged = True
        sampler = None
        if options is None:
            recorder.sampling = recorder.reference_sampling
        else:
            sampler = FastCanvasSampler(trainer, options)
            recorder.sampling = lambda embeddings, sampler=sampler, **kwargs: sampler.sample(embeddings)
        # Warm-up case of this variant (first capture, kernel plans), untimed.
        recorder.after_sampling = None
        run_loop(trainer, config, 0, 1, out, f"warmup_{name}", step, num_samples=1)
        for phase, (manifest, cases, variants, check, floor) in phases.items():
            if name not in variants:
                continue
            recorder.records.clear()
            recorder.after_sampling = None
            if name == "reference" and phase == "main":
                def after_reference(embeddings, floor=floor):
                    tap.enabled = False
                    calls = tap.take()
                    trajectories.append(calls)
                    noise_floor = reference_check(trainer, embeddings, calls, batch=2) if floor else None
                    tap.enabled = True
                    return {"floor": noise_floor}

                tap.take()
                tap.enabled = True
                recorder.after_sampling = after_reference
            elif name == "reference_merged" and phase == "main" and check and trajectories:
                queue = iter(trajectories)

                def after_merged(embeddings, queue=queue):
                    calls = next(queue, None)
                    return {"per_step": reference_check(trainer, embeddings, calls, batch=1) if calls else None}

                recorder.after_sampling = after_merged
            elif sampler is not None:
                queue = iter(trajectories if check else ())

                def after_fast(embeddings, sampler=sampler, queue=queue):
                    stats = dict(sampler.last)  # before the check builds another forward
                    calls = next(queue, None)
                    return {"fast": stats, "per_step": per_step_check(sampler, embeddings, calls) if calls else None}

                recorder.after_sampling = after_fast
            label = name if phase == "main" else f"{phase}_{name}"
            metadata, loop_s = run_loop(trainer, config, args.shard, args.num_shards, out, label, step,
                                        num_samples=cases, manifest=manifest)
            tap.enabled = False
            samples = metadata["samples"]
            records = list(recorder.records)[: len(samples)]  # a padded last round predicts once more
            for record in records:
                extra = record.pop("check", None) or {}
                record["fast"], record["check"], record["floor"] = extra.get("fast"), extra.get("per_step"), extra.get("floor")
            if name == "reference":
                references[phase] = records
            reference = references[phase]
            result["phases"][phase]["variants"][name] = {
                "options": None if options is None else options.label,
                "loop_s": loop_s,
                "cases": [case_entry(record, sample, None if name == "reference" or not reference else reference[index])
                          for index, (record, sample) in enumerate(zip(records, samples))],
            }
            save()
            print(json.dumps({"phase": phase, "variant": name, "cases": len(records),
                              "median_sampling_s": float(np.median([r["sampling_s"] for r in records])) if records else None}),
                  flush=True)
        del sampler
        torch.cuda.empty_cache()


def summarize_phase(results, phase):
    summary = {"variants": {}}
    for name in VARIANTS:
        cases = [case for r in results for case in r["phases"].get(phase, {}).get("variants", {}).get(name, {}).get("cases", [])]
        if not cases:
            continue
        entry = {"cases": len(cases)}
        for key in ("sampling_s", "sampling_peak_gib"):
            values = [case[key] for case in cases]
            entry[f"{key}_median"], entry[f"{key}_mean"] = float(np.median(values)), float(np.mean(values))
        fast = [case["fast"] for case in cases if case.get("fast")]
        for key in ("setup_s", "condition_pass_s", "capture_s", "loop_s"):
            if fast:
                entry[f"{key}_median"] = float(np.median([f[key] for f in fast]))
        compared = [case["vs_reference"] for case in cases if case.get("vs_reference")]
        if compared:
            for key in ("latent_rel_l2", "image_psnr_db", "speedup"):
                entry[f"{key}_median"] = float(np.median([c[key] for c in compared]))
            entry["image_psnr_db_min"] = float(np.min([c["image_psnr_db"] for c in compared]))
        checks = [case["check"] for case in cases if case.get("check")]
        for branch in ("positive", "null", "guided"):
            values = [value for check in checks for value in check.get(branch, [])]
            if values:
                entry[f"step_{branch}_rel_l2_median"] = float(np.median(values))
                entry[f"step_{branch}_rel_l2_max"] = float(np.max(values))
        summary["variants"][name] = entry
    return summary


def summarize(args):
    shards = sorted(args.out.glob("shard*/result.json"))
    results = [json.loads(path.read_text()) for path in shards]
    phases = sorted({phase for r in results for phase in r["phases"]})
    summary = {"gpu": sorted({r["gpu"] for r in results}), "shards": len(results),
               "phases": {phase: summarize_phase(results, phase) for phase in phases}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--checkpoint", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--shard", type=int, default=0)
    run_parser.add_argument("--num-shards", type=int, default=1)
    run_parser.add_argument("--variants", default=None, help="comma list (default: all, in the fixed order)")
    run_parser.add_argument("--manifest", type=Path, default=None, help="validation manifest replacing the run's")
    run_parser.add_argument("--cases", type=int, default=None, help="logging.sampling.num_samples override")
    run_parser.add_argument("--scaling-manifest", type=Path, default=None, help="second phase: larger-mask cases")
    run_parser.add_argument("--scaling-cases", type=int, default=None)
    run_parser.add_argument("--scaling-variants", default="reference,equivalent,fast")
    run_parser.add_argument("--no-check", dest="check", action="store_false", help="skip per-step checks")
    run_parser.add_argument("--no-floor", dest="floor", action="store_false", help="skip the batch-2 noise floor")
    summary_parser = sub.add_parser("summarize")
    summary_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else summarize)(args)


if __name__ == "__main__":
    main()

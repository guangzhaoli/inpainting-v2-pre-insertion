# Teacher v2 and OPD runtime contracts

## Environment and attention

The uv environment pins Torch 2.9.1 (CUDA 12.8) and FlashAttention2 2.8.3.
`model.attention_backend: flash_attention_2` selects FA2 in both Qwen-VL and
the DiT's Diffusers attention dispatcher. `sdpa` explicitly selects the native
backend in both components. Merely passing Transformers' `attn_implementation`
to a Diffusers model does not configure its attention processor.

The official CPython 3.12 / Linux x86-64 / CUDA 12 / Torch 2.9 / CXX11 ABI wheel
is stored under `../pretrained_weights/wheels/` and hashed in `uv.lock`.
The adjacent `flash-attn-source.json` records the upstream URL, SHA256 and size.
uv references a byte-identical hard link named
`flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl`, matching the wheel's internal
METADATA version. This avoids repeated reinstalls caused by the upstream
filename's additional CUDA/Torch/ABI version suffix.
Prepare it only on a **new Inspire CPU资源空间 Notebook**:

```bash
INSPIRE_CPU_DOWNLOAD=1 bash scripts/prepare_flash_attention.sh
```

The script checks the Torch ABI, downloads from the official Dao-AILab release,
verifies its release digest, and uses uv to lock/install the local wheel. GPU
workers run `uv sync --offline --locked` or reuse `.venv` with `uv run
--offline --no-sync`. The frozen base and all loss-model weights remain local.
`requirements.txt` is historical; `pyproject.toml` and `uv.lock` are authoritative.

The 2511 base also declares `zero_cond_t=true`, a behavior missing from the
previous custom transformer and Diffusers 0.36. New Teacher v2 configs enable
`model.zero_cond_t: true`: target tokens use the sampled timestep while clean
source/reference tokens use timestep zero; text retains the target timestep.
Prefix slicing implements modulation without per-token index transfers or
materialized full-size gate/scale/shift tensors. Dense-oracle tests check values
and gradients with and without gradient checkpointing. The timestep contract
follows the [upstream QwenImage transformer](https://github.com/huggingface/diffusers/blob/v0.39.0/src/diffusers/models/transformers/transformer_qwenimage.py).

**Old checkpoints keep `zero_cond_t: false`**, the default when metadata omits
this field. Resume, inference and OPD reject mixing the two contracts. Use an
old checkpoint's saved config for its original behavior; regenerating a new v2
config is for new training, not an automatic migration of existing LoRA weights.
Saved configs containing old read-only computed fields can now be loaded directly.

## Build OPD from actual experts

Use the checkpoint builder for Teacher v2 rather than replacing paths in the
old rank-256 template:

```bash
uv run --offline --no-sync python -m scripts.build_opd_config \
  --teacher accessory=/absolute/path/to/accessory/checkpoint \
  --teacher animal=/absolute/path/to/animal/checkpoint \
  --teacher fashion=/absolute/path/to/fashion/checkpoint \
  --teacher object=/absolute/path/to/object/checkpoint \
  --teacher human=/absolute/path/to/human/checkpoint \
  --output configs/opd/teacher_v2_local.yaml
```

Each checkpoint needs `model.safetensors`, `train_config.yaml`, and its geometry
companion. Category routes, rank, alpha and target modules come from metadata.
The builder rejects overlapping routes and incompatible base models,
processors or mask-geometry contracts. The same checks run before the large
model is loaded in `fit`, and in the shared backend for direct callers.
`--manifest` selects a different joint manifest; `--student-checkpoint` supplies
a compatible student warm start. Without it, the student LoRA starts fresh.

Each teacher gets its own mask-aware/clipping policy, inherited from its
checkpoint when omitted. Guidance scale is an explicit distillation setting;
the builder defaults it to that teacher's recorded validation scale. Per-domain
adaptive crops and mask augmentations are retained in
`data.init_args.category_conditioning`, including the accessory 16-pixel and
other-domain 32-pixel interaction margins. Public student inference defaults
come from the first teacher and can be configured separately. Standalone
unguided student inference still requires `--prompt ""`.

The generated config retains 10 OPD epochs, 10 rollout steps and all rollout
timesteps in the loss. It uses batch size 1 and a teacher-count accumulation
cycle. `teacher.weight` is now an **absolute loss multiplier** before batch
averaging, so it works at batch size 1 and composes with DDP/accumulation. All-one
weights retain the previous objective; non-unit weights now have their intended
effect. They are no longer normalized independently on each rank.

## Cache and device boundaries

- One OPD batch shares frozen VAE and Qwen-VL encodings by actual image/prompt
  identity. The common unguided-student/reference-CFG-teacher case performs two
  Qwen-VL encodings (positive/null) and two VAE encodings (source/reference).
- Only teachers routed in the local batch have their conditions prepared.
  Each frozen teacher geometry companion is evaluated once for the shared
  positive/null mask. Student rollout geometry is cached under no-grad and the
  same autocast policy. Static geometry pooling is cached once with the
  training path's BF16/FP32 cast order; each student forward still runs the
  trainable tokenizer convolutions inside DiT with gradients. No autograd graph
  is shared across timestep backward calls.
- Shared-LoRA OPD computes the selected teacher targets together after rollout,
  then runs the student timestep backward passes. One routed teacher with ten
  timesteps now needs three adapter-setting calls per batch instead of 32.
  All ten rollout steps, CFG branches and objective terms are retained. Static
  teacher weights are validated once on CPU, and the reference CFG latent mask
  is prepared once per condition.
- Rollout states default to GPU storage. CPU storage remains an explicit memory
  tradeoff. Text lengths are prepared once instead of synchronizing every step.
- `train.offload_text_encoder` covers supervised training, OPD and validation,
  including exceptions. Its GPU residency now covers only the actual prompt
  encoder call, so it returns to CPU **before VAE encoding**, not just before
  denoising/backward. This avoids overlapping Qwen-VL weights with VAE workspace
  and accumulated Student gradients. Standalone prediction retains its explicit
  `predict.devices` placement. Keep offloading disabled when VRAM permits;
  offloading trades PCIe transfers for memory headroom.
- New Teacher v2 and generated OPD configs use `train.memory_policy: auto`.
  GPUs with at most 96 GiB (including H100 80GB) offload the text encoder and
  checkpoint the VAE decoder when pixel losses are active. Larger GPUs (H200)
  keep the encoder resident and avoid decoder recomputation for throughput.
  `memory_policy: configured` retains explicit manual settings. The 71.21 GiB
  H100 result below used precisely the offload/checkpoint combination selected
  by the automatic policy; it is not a VRAM estimate for disabling both options.
- Data stays on CPU through CPU image/mask preprocessing. Only encoder inputs
  and final loss/DiT tensors are transferred, avoiding raw-batch GPU round trips.
- Rectangle dilation/erosion in the pixel loss uses exactly equivalent
  separable pooling. Loss logging uses one packed reduction per update. Hybrid
  augmentation statistics move to the communication device before NCCL gather,
  then return to CPU once for all summary calculations.
- Training sigma values use the already sampled schedule indices directly.
  DiT skips the final block's unused text MLP and training/inference read out
  only target-image tokens; all control tokens still participate in attention.
  RoPE uses a single 64-entry LRU cache, avoiding permanent retention of every
  aspect-ratio variant in two independent caches.
- A partial final accumulation window is normalized by its actual microbatch
  count before gradient clipping. For the current joint manifest, the
  five-teacher sampler has 236720 entries; the 32/64-GPU layouts leave two/three
  microbatches respectively in the final five-microbatch window each epoch.

## Checkpoint and interruption

`scripts/run_teacher_v2.sh` and the snapshot job entry translate Inspire's
node-level `WORLD_SIZE`, `RANK`, `MASTER_ADDR` and `MASTER_PORT` into Accelerate's
topology. Explicit `NUM_MACHINES`/`MACHINE_RANK` settings take precedence. A
multi-node platform job therefore starts one distributed run, rather than an
independent eight-GPU run on each node. Dry-run coverage includes 2/4/8 nodes.

Checkpoints record a global sample cursor **before distributed sharding**, the
sampler length, batch size and optimizer-step budget. Resume slices the global
epoch stream before Accelerate shards it, so expanding/contracting the world
does not skip unseen samples. On a topology change, the remaining partial global batch is padded;
some tail examples may be repeated to keep rank forward counts equal.
Same-world resume retains the saved `drop_last` policy, including the policy
recorded after an earlier expansion, and preserves its cursor even if no new
optimizer step runs before another checkpoint is saved.

Existing ranks restore their RNG files. Added ranks receive deterministic
streams derived from the training seed, optimizer step and rank. A separate
dataloader generator prevents iterator construction from consuming the training
noise RNG. Elastic resume changes the per-update sample grouping, so it is not
bitwise equivalent to staying at the old world size.

Expansion retains the saved optimizer-step budget, extending the epoch loop
when necessary; it does not silently finish early due to shorter sharded epochs.
An explicit lower `max_train_steps` still takes precedence. Legacy checkpoints
map their local cursor using the saved world size; their step budget is inferred
from their saved config. Batch-size or sampler-length changes are rejected.

All ranks decide whether to stop using the **same all-reduced interrupt value**.
A SIGINT arriving during that collective stays latched and is propagated at the
next boundary, avoiding a checkpoint-barrier/training-collective mismatch.

## Verification

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --offline --no-sync pytest -q \
  tests/test_optimization_regressions.py tests/test_opd_*.py \
  tests/test_teacher_optimization.py tests/test_training_signal_handling.py
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 uv run --offline --no-sync \
  python -m tests.smoke_opd_distributed
```

`scripts/run_training_stack_smoke.sh` runs on every allocated node and GPU using
the platform's `WORLD_SIZE`, `RANK`, `MASTER_ADDR` and `MASTER_PORT`. Set a shared
`SMOKE_OUTPUT` directory. It checks FA2/SDPA values and input gradients, exact
GPU morphology, the actual supervised `train_epoch` including NCCL hybrid
metrics, validation, and an OPD `train_epoch` with five resident teacher adapters
and a three-microbatch tail in a five-microbatch accumulation window. Every OPD
rollout retains all ten steps. The five teacher adapters are initialization
fixtures; this validates execution and gradients, not trained-expert image
quality. Rank JSONs record optimizer/logging metrics, timings and peak VRAM.

Both legacy and native-2511 contracts passed separate two-node / sixteen-H100
checks with the earlier smoke script. Every rank completed the supervised pixel-loss backward, validation,
and all ten OPD timesteps, with finite synchronized student gradients and frozen
teacher gradients. Peak allocated memory was 71.21 GiB. Across sixteen ranks,
the 1MP/radius-32 morphology kernel median fell from 6.099 ms to 0.335 ms (18.2x);
this is an isolated kernel speedup, not an end-to-end training speedup.
FA2/SDPA forward maximum absolute error in the BF16 kernel probe was 3.052e-5,
and input-gradient comparisons passed. Full results and verification limits are
in `outputs/optimization-20261015/verification-summary.json`.
Those earlier checks bypassed `train_epoch` logging; the updated smoke above
exercises that path. Normal-path unit tests are in
`tests/test_normal_training_path.py`, including real PEFT gradient equivalence,
actual Accelerate accumulation updates, and target-readout equivalence with
gradient checkpointing enabled and disabled.

# Final Inpainting Architecture

The integrated experiment is defined by
`configs/exp/final_architecture.yaml` (accessory_jewelry) and domain siblings
`configs/exp/final_architecture_{animal,fashion_tryon,general_object,human,all}.yaml`.
Architecture knobs are shared; only the data split / adapter / tracker / validation
folder change per domain. It combines the useful parts of five idea branches with
the maintained `inpainting-v2` inference path.

## Data and conditioning contract

```text
source + GT mask
       │
       ├── adaptive crop ──────────────────────────────┐
       │                                               │
       ├── tar_mask (M_gt) ── region-balanced latent loss
       │                    └─ region-perceptual pixel loss
       │
       └── MaskMix-v2 ── tar_mask_cond/source_hole_mask
                            ├─ masked source RGB control
                            └─ [mask, signed distance, boundary ring]
                                  └─ zero-init geometry tokenizer
                                       └─ target-token residual

reference image ── positive reference-CFG branch
reference dropped ─ null-reference branch
                     (same source and same geometry tokens)
```

There are two masks with intentionally different jobs:

- `tar_mask` is the ground-truth edit mask. It drives latent loss,
  region-perceptual loss, evaluation and final strict compositing.
- `tar_mask_cond` / `source_hole_mask` is the training condition mask. It may
  be expanded by MaskMix-v2 and drives the source hole and geometric
  conditioning. It must cover `tar_mask` but never replaces it as supervision.

MaskMix-v2 samples exact mask 60%, random morphology 20%, scale-aware
augmented box 10%, and asymmetric box 10%. The context guard falls back to the
GT mask when a sampled condition consumes more than 80% of the frame.

The geometry encoder constructs
`[M, tanh(D / sigma), exp(-abs(D) / sigma)]`. A lightweight convolutional
tokenizer projects this to the target image-token grid. Its final projection is
zero-initialized, so the new branch starts as an exact no-op. Geometry replaces
the older RGB-VAE mask control; both are not attached at once.

## Loss

The selected objective is:

```text
L = latent_mse_weight * L_region_balanced_latent
  + w(t) * (fg_lpips_weight * L_fg_lpips
            + bg_l1_weight * L_bg_l1)
```

`w(t) = (1 - sigma)^region_sigma_power` emphasizes lower-noise steps for the
pixel-space terms. The final config sets `region_max_sigma: 0.5`; steps above
that threshold have exactly zero pixel-space weight. If every sample in the
local batch is gated, training skips the VAE decode, LPIPS, background L1, and
optional DINO path while retaining the weighted latent loss and all metrics.
Mixed batches retain the original per-sample weighting behavior. The optional
gate is an explicit objective/performance tradeoff rather than a numerically
equivalent optimization: under uniform sigma sampling and power 2 it skips
about half of pixel-space decodes while retaining 87.5% of the ungated
integrated timestep-weight mass. The optional
reference-DINO identity term is implemented but disabled in the final config:
it adds a frozen encoder and an extra offline checkpoint dependency before its
benefit has been established.

Region-perceptual training keeps the frozen VAE decoder on the training device
and backpropagates through its input. This raises peak memory substantially at
1 MP even though VAE parameters remain frozen. Start with a short smoke run and
monitor peak memory before launching the full job.

## Adaptive crop and reference CFG

Adaptive crop runs before model preprocessing and is restored to original
coordinates after generation. All masks and geometry are built on the final
processed target grid.

Reference CFG keeps the source/context and geometric condition identical in
the positive and null-reference branches. The null branch drops only reference
conditioning. Geometry tokens are computed once per denoising run and shared by
both branches. The positive prediction stays on GPU while the null-reference
branch runs, so this path does not add per-step CPU transfers or allocator-cache
flushes.

## Checkpoints

Geometry adds a trainable companion file:

```text
checkpoint-*/
  model.safetensors
  pytorch_lora_weights.safetensors
  mask_geometry_tokenizer.safetensors
  train_config.yaml
  state.json
```

`state.json` records size and SHA256 for LoRA and geometry artifacts. Resume
and inference verify the active mask-control contract and companion integrity.
Old checkpoints without an artifact manifest remain loadable but are marked as
artifact-unverified. Geometry inference cannot run from the base model alone.

## Launch

```bash
conda activate qwen-inpainting-pre-insertion

# Config/schema and unit smoke tests
python -m unittest -v tests.test_final_architecture

# Example 8-GPU training launch
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 NUM_MACHINES=1 MACHINE_RANK=0 \
  bash run.sh configs/exp/final_architecture.yaml
```

OpenCV is limited to one internal CPU thread in each training rank and
DataLoader worker. This avoids MaskMix morphology and geometry distance
transforms creating a host-sized thread pool in every process. Keep the default
for multi-GPU training; an intentional single-process override can be supplied
with `QWEN_INPAINTING_OPENCV_THREADS=N`. The value must be a positive integer.

The checked-in config uses the project's local model, dataset, manifest and
offline VGG-LPIPS weights. Update those paths before using it outside this
workspace.

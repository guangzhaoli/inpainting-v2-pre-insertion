# InsertFuse — Qwen Insert Anything Inpainting

MeanFlow initialization and conditional DMD post-training are available through
scripts/build_meanflow_config.py and scripts/run_meanflow_dmd.sh. See
[the runtime guide](docs/meanflow_dmd_runtime.md) for both stages, distributed
launch, checkpoint/resume and flow-map inference.

The integrated **Teacher v2** configs are in `configs/teacher_v2/`. See
[`docs/teacher_v2.md`](docs/teacher_v2.md) for the combined precision, RNG,
conditioning, identity/detail losses, validation and uv launch workflow.
Use these configs for new runs; existing checkpoints retain their original configs.

Teacher v2 and OPD now use pinned FlashAttention2 in both Qwen-VL and DiT.
See [`docs/training_optimizations.md`](docs/training_optimizations.md) for CPU-only
dependency preparation, checkpoint-derived OPD configs, batch-local encoding
reuse, per-teacher conditioning, and elastic checkpoint recovery. Build new
Teacher v2 OPD runs with `python -m scripts.build_opd_config`; the rank-256
templates below remain descriptions of the older experts.

This directory is the integrated InsertFuse worktree at `inpainting-v2-final-architecture`. It keeps one maintained training and inference path:

- `QwenImageEditInsertAnythingTrainer`
- `InsertAnythingDataset`
- LoRA training for Qwen Image Edit / Qwen Image Edit Plus style multi-image conditioning
- Adaptive Crop, random mask expansion (Hybrid Masking), and reference CFG

Pre-insertion/layout conditioning and Query-prompt are isolated in sibling worktrees `inpainting-v2-layout-condition` and `inpainting-v2-query-prompt`. This baseline accepts their disabled fields when validating legacy checkpoint metadata, but rejects checkpoints that require either retired conditioning path.

Current baseline configs enable `reference_cfg`. In this mode, `true_cfg_scale` controls reference guidance: the positive branch uses both Qwen-VL reference conditioning and reference VAE control, while the null-reference branch removes both reference paths and keeps only the source/context condition. The default configs also set `reference_cfg.force_empty_prompt: true`, so this path does not use text prompt guidance.

## Environment

The project environment is managed by:

```bash
uv sync --offline --locked
```

The pinned environment files are:

```bash
pyproject.toml
uv.lock
```

## Configs

Current configs live under `configs/`. See `configs/README.md` for the full catalog.

Layout:

```text
configs/
  default.yaml              # default training/inference
  scale/{4,8,16,32}.yaml    # run.sh auto-select by TOTAL_PROCESSES
  teacher/                  # domain baselines (all / accessory_jewelry / ...)
  exp/accessory/            # accessory idea ablations
  legacy/native.yaml
```

Important paths:

- `configs/default.yaml`: default reference-CFG training/inference config.
- `configs/legacy/native.yaml`: older native-style mask prompt baseline.
- `configs/scale/4.yaml` ... `configs/scale/32.yaml`: scale-out launch defaults for `run.sh`.
- `configs/teacher/accessory_jewelry.yaml`: accessory/jewelry teacher baseline.
- `configs/exp/accessory/adaptive_crop_r256_fg3.yaml`: Adaptive Crop experiment (r256, fg:bg=3:1).
- `configs/exp/accessory/hybrid_masking_r256_fg3.yaml`: Hybrid Masking experiment (r256, fg:bg=3:1).
- `configs/exp/final_architecture.yaml`: integrated MaskMix-v2 + geometric mask +
  region-balanced/region-perceptual + Adaptive Crop + reference-CFG proposal.

See `docs/final_architecture.md` for the mask contract, loss composition,
checkpoint companion and launch notes.

Online policy distillation (OPD) is an optional second training phase. Train
the domain-specific final-architecture experts first, then use
`configs/opd/final_architecture_five_teacher.yaml.example` to route their LoRA
checkpoints into one Student. The OPD workflow, geometry companion handling,
and smoke checklist are documented in `docs/opd.md`.

Each config exposes:

```yaml
model:
  mask_control:
    enabled: false
  reference_cfg:
    enabled: true
    train_dropout_rate: 0.1
    force_empty_prompt: true
    mask_aware: false
    delta_clip_enabled: false
    delta_clip_max_ratio: 0.5
```

Set `model.reference_cfg.enabled` to `false` to use the native text true-CFG path.

Set `model.reference_cfg.mask_aware` to `true` to apply reference CFG only inside the target mask. Mask tokens use reference CFG, while non-mask tokens keep the positive reference-conditioned prediction without extra CFG amplification.

Set `model.reference_cfg.delta_clip_enabled` to `true` to clip the per-token reference delta before applying reference CFG. The clipping threshold is `delta_clip_max_ratio * norm(eps_BR)` for each latent token, where `eps_BR` is the positive reference-conditioned prediction. When enabled, this replaces the older full-prediction norm rescale used after CFG combination.

By default reference CFG runs at `model.reference_cfg.scale: 3.0` (used when `--true-cfg-scale` is not given), only at schedule steps with `sigma >= sigma_min: 0.75` (the first 16 of 40 steps; `0` = every step), and with adaptive projected guidance `apg_eta: 0.0`: the part of the reference delta parallel to the conditional x0 prediction (inside the CFG mask) is removed, which drops most of the saturation/brightness shift and chroma blotches of plain CFG. `apg_eta: 1.0` restores plain CFG. These defaults won the 2026-10-29…11-01 sweeps (output/teacher_v2_runs/cfgsweep_20261029, cfgband_20261030, cfgapg_20261031).

Set `model.mask_control.enabled` to `true` to attach the placement mask as an extra VAE-only RGB control (binary mask repeated to 3 channels). It does **not** force `true_cfg_scale` to 1, so it can be combined with `reference_cfg`. When reference CFG is active, both the positive and null-reference branches receive the placement mask; only the reference image is dropped on the null branch. Default `false` preserves previous checkpoints and inference.

## Training

Single-process entry:

```bash
uv run --offline --no-sync python -m src.main --config configs/default.yaml
```

Accelerate entry:

```bash
uv run --offline --no-sync accelerate launch -m src.main --config configs/default.yaml
```

8-GPU example with an experiment config:

```bash
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 NUM_MACHINES=1 MACHINE_RANK=0 \
bash run.sh configs/exp/accessory/adaptive_crop_r256_fg3.yaml
```

Training checkpoints are LoRA-only by default. Resume expects a checkpoint directory with LoRA weights, optimizer and scheduler state, `state.json`, and per-rank `random_states_*.pkl` files; new checkpoints also contain a `train_config.yaml` snapshot, while older checkpoints can use the run-level `train_config.yaml` in their direct parent directory. The frozen base model is loaded again from `model.pretrained_model_name_or_path`.

## Inference

Mask-prompt inference with a trained checkpoint:

```bash
python infer.py \
  --config configs/default.yaml \
  --lora /path/to/checkpoint-dir \
  --src-image /path/to/source.png \
  --src-mask /path/to/source_mask.png \
  --ref-image /path/to/reference.png \
  --ref-mask /path/to/reference_mask.png \
  --output output/generated.png
```

The default `legacy` mode is experiment A and preserves the previous behavior:
`--output` is the final strict-mask composite in original source coordinates.
Single-image inference also writes output-scoped debug artifacts:

```text
output/generated.png
output/generated.output_local.png
output/generated.output_restored_raw.png
output/generated.composite.png
output/generated.ref_mask_overlay.png
output/generated.source_mask_overlay.png
output/generated.output_mask.png
output/generated.output_mask_overlay.png
output/generated.row.png
output/generated.metadata.json
```

When Adaptive Crop is enabled, it additionally writes
`generated.crop_source.png` and `generated.crop_mask.png`. The compatibility
`generated.composite.png` copy can be disabled with `--skip-composite`.

Inference-only latent anchoring exposes four mutually exclusive A/B/C/D modes:

```text
legacy        A: no anchoring; strict M_core composite is the primary output
core          B: hard M_core latent anchoring
expanded      C: hard dilated M_free latent anchoring
expanded_soft D: M_free plus a soft outer transition
```

Select them with:

```bash
--latent-anchoring-mode legacy|core|expanded|expanded_soft
--interaction-dilation-px 32
--transition-width-px 32
```

The same defaults can be stored in YAML under `predict`:

```yaml
predict:
  latent_anchoring:
    mode: expanded_soft
    mask_threshold: 0.5
    interaction_dilation_px: 32
    transition_width_px: 32
```

`A`, `B`, `C`, and `D` are accepted as mode aliases. In B/C/D, `--output` is the
latent-anchored primary result and `generated.composite.png` remains the strict
`M_core` compatibility comparison. These modes additionally save
`generated.mask_core.png`, `generated.mask_free.png`, and
`generated.edit_weight.png`.

B/C/D Latent Anchoring is compatible with Adaptive Crop prediction through the
crop-aware inference wrappers (`infer.py`, `infer_testset.py`). Crop planning
validates realized processor geometry and packed latent support against crop-edge
feathering; candidates expand or fall back when needed, and empty processed
`M_core` is rejected with a wrapper-level error. Explicit `--height/--width`
remains unsupported for B/C/D. Training-time periodic validation still forces
legacy A until its saver uses the same crop-aware primary-output path.

Running without LoRA is blocked by default. Use `--allow-base-model` only when intentionally testing the untrained base model. LoRA inference verifies the nearest checkpoint `train_config.yaml` against the selected config, including Adaptive Crop, processor, base model, LoRA structure, and conditioning modes. Missing or incompatible checkpoint metadata fails by default; use `--allow-checkpoint-mismatch` only for an intentional, recorded override.

Reference CFG does not use `--negative-prompt`. With `model.reference_cfg.enabled: true`, passing a negative prompt is rejected before model loading because text CFG and reference CFG are separate guidance modes. When `reference_cfg.force_empty_prompt: true`, `--prompt` is optional and ignored.

## Data Expectations

`InsertAnythingDataset` now validates sample completeness strictly. For local folder datasets:

```text
mask_prompt/<category>/
  ref_image/
  ref_mask/
  tar_image/
  tar_mask/
```

For each sample stem, `ref_image`, `ref_mask`, `tar_image`, and `tar_mask` are required. Prompt text files next to `tar_image` are controlled by `data.init_args.require_prompt`:

- `require_prompt: true`: each `tar_image/<stem>.txt` must exist.
- `require_prompt: false`: missing prompt files are accepted and the sample prompt is set to `""`.

The current reference-CFG configs set `require_prompt: false` because this path runs with `reference_cfg.force_empty_prompt: true` and does not use text prompt guidance.

Missing required image or mask files are treated as data errors instead of being skipped.

## Validation

Training-time validation sampling is available for `QwenImageEditInsertAnythingTrainer`.
It is disabled by default. Enable it under `logging.sampling`:

```yaml
logging:
  sampling:
    enable: true
    validation_steps: 1000
    num_samples: 4
    seed: 42
    num_inference_steps: 20
    true_cfg_scale: 4.0
    guidance_scale: 1.0
    output_dir: "validation"
    panel_size: 320
    save_composite: true
    validation_data: null
```

With `validation_data: null`, the sampler deterministically selects `num_samples`
examples from the training dataset and reuses the current mask-only contract:
`ref_image`, `ref_mask`, `tar_image`, `tar_mask`, `prompt`, and `category`.
Generated validation artifacts are written under:

```text
<logging.output_dir>/validation/step-00001000/
  all_rows.png
  grid.png
  metadata.json
  *_source.png
  *_source_mask.png
  *_source_mask_overlay.png
  *_reference.png
  *_reference_mask.png
  *_reference_mask_overlay.png
  *_output.png
  *_output_mask.png
  *_output_mask_overlay.png
  *_composite.png
  *_row.png
```

To use a separate validation set, set `validation_data` to a local dataset folder
with the same `InsertAnythingDataset` structure, or to a JSON/JSONL/CSV/TSV
manifest. Manifest rows must contain:

```json
{
  "ref_image": "ref_image/sample.png",
  "ref_mask": "ref_mask/sample.png",
  "tar_image": "tar_image/sample.png",
  "tar_mask": "tar_mask/sample.png",
  "prompt": "",
  "category": "object"
}
```

Relative manifest paths are resolved against the manifest file directory.
In distributed training, every rank participates in validation forward passes
to keep wrapped model execution synchronized; only the main process writes files
and TensorBoard image grids.

Lightweight checks:

```bash
python -m compileall -q src infer.py
pip check
python - <<'PY'
from pathlib import Path
import torch
from src.data.config import load_config_from_yaml
torch.cuda.is_available = lambda: True
for path in sorted(Path("configs").rglob("*.yaml")):
    cfg = load_config_from_yaml(str(path))
    print(
        path.as_posix(),
        cfg.trainer_type,
        cfg.model.reference_cfg.enabled,
        cfg.model.reference_cfg.force_empty_prompt,
        cfg.data.init_args.require_prompt,
    )
PY
```

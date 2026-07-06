# Qwen Insert Anything Inpainting

This repository is the active `inpainting-v2-pre-insertion` codebase for Qwen image inpainting experiments. It keeps one maintained training and inference path:

- `QwenImageEditInsertAnythingTrainer`
- `InsertAnythingDataset`
- LoRA training for Qwen Image Edit / Qwen Image Edit Plus style multi-image conditioning
- explicit experiment switches for `query_prompt`, `pre_insertion`, and `reference_cfg`

Current configs disable `query_prompt` and `pre_insertion` and enable `reference_cfg`. In this mode, `true_cfg_scale` controls reference guidance: the positive branch uses both Qwen-VL reference conditioning and reference VAE control, while the null-reference branch removes both reference paths and keeps only the source/context condition. The default configs also set `reference_cfg.force_empty_prompt: true`, so this path does not use text prompt guidance.

## Environment

The project environment is managed by:

```bash
./setup.sh
conda activate qwen-inpainting-pre-insertion
```

The pinned environment file is:

```bash
environment.yml
```

## Configs

Current configs live under `configs/` and all use:

```yaml
trainer: QwenImageEditInsertAnything
data:
  class_path: "src.data.insert_anything_dataset.InsertAnythingDataset"
  init_args:
    require_prompt: false
```

Important configs:

- `configs/insert_anything_fixed_pixels_parquet.yaml`: default reference-CFG training/inference config.
- `configs/insert_anything_native_qwen_image_inpainting.yaml`: native-style mask prompt path with reference CFG enabled.
- `configs/insert_anything_4.yaml`, `insert_anything_8.yaml`, `insert_anything_16.yaml`, `insert_anything_32.yaml`: scale-out variants with the same reference-CFG switches.

Each config exposes:

```yaml
model:
  query_prompt:
    enabled: false
  pre_insertion:
    enabled: false
  reference_cfg:
    enabled: true
    train_dropout_rate: 0.1
    force_empty_prompt: true
```

Set `model.reference_cfg.enabled` to `false` to use the native text true-CFG path. Keep `query_prompt` and `pre_insertion` disabled when using reference CFG.

## Training

Single-process entry:

```bash
python -m src.main --config configs/insert_anything_fixed_pixels_parquet.yaml
```

Accelerate entry:

```bash
accelerate launch -m src.main --config configs/insert_anything_fixed_pixels_parquet.yaml
```

Training checkpoints are LoRA-only by default. Resume expects a checkpoint directory with LoRA weights, optimizer and scheduler state, `state.json`, and per-rank `random_states_*.pkl` files; the frozen base model is loaded again from `model.pretrained_model_name_or_path`.

## Inference

Mask-prompt inference with a trained checkpoint:

```bash
python infer.py \
  --config configs/insert_anything_fixed_pixels_parquet.yaml \
  --lora /path/to/checkpoint-dir \
  --src-image /path/to/source.png \
  --src-mask /path/to/source_mask.png \
  --ref-image /path/to/reference.png \
  --ref-mask /path/to/reference_mask.png \
  --output output/generated.png
```

If `query_prompt.enabled: true`, the checkpoint directory must also contain:

```text
query_prompt.safetensors
```

Running without LoRA is blocked by default. Use `--allow-base-model` only when intentionally testing the untrained base model.

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

Lightweight checks:

```bash
python -m compileall -q src infer.py
pip check
python - <<'PY'
from pathlib import Path
import torch
from src.data.config import load_config_from_yaml
torch.cuda.is_available = lambda: True
for path in sorted(Path("configs").glob("*.yaml")):
    cfg = load_config_from_yaml(str(path))
    print(
        path.name,
        cfg.trainer_type,
        cfg.model.query_prompt.enabled,
        cfg.model.pre_insertion.enabled,
        cfg.model.reference_cfg.enabled,
        cfg.model.reference_cfg.force_empty_prompt,
        cfg.data.init_args.require_prompt,
    )
PY
```

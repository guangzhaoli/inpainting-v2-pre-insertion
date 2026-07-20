# Config Catalog

Configs are grouped by purpose. Filenames keep only the distinguishing idea;
shared baseline knobs live in the file body and the one-line header.

```text
configs/
  default.yaml                 # default train/infer
  scale/
    4.yaml | 8.yaml | 16.yaml | 32.yaml   # run.sh by TOTAL_PROCESSES
  teacher/
    all.yaml
    accessory_jewelry.yaml
    animal.yaml
    fashion_tryon.yaml
    general_object.yaml
    human.yaml
  exp/
    accessory/
      r256.yaml
      fg5.yaml
      adaptive_crop.yaml
      adaptive_crop_r256_fg3.yaml
      hybrid_masking.yaml
      hybrid_masking_r256_fg3.yaml
  legacy/
    native.yaml
  README.md
```

## Naming

| Prefix | Meaning |
|---|---|
| `default.yaml` | Main default config |
| `scale/N.yaml` | Launch default when `TOTAL_PROCESSES=N` |
| `teacher/<domain>.yaml` | Domain baseline (data split only / domain baseline) |
| `exp/accessory/<idea>.yaml` | Accessory/jewelry idea ablation |
| `legacy/` | Old path kept for reference |

## Current active accessory experiments (r256, fg:bg=3:1)

| File | Idea | LoRA | Loss | Notes |
|---|---|---|---|---|
| `exp/accessory/adaptive_crop_r256_fg3.yaml` | Adaptive Crop | 256 | 3:1 | hybrid off |
| `exp/accessory/hybrid_masking_r256_fg3.yaml` | Hybrid Masking | 256 | 3:1 | adaptive crop off |

Shared baseline for both:

- `reference_cfg: true`
- other optional controls off: `mask_control`
- data: accessory + jewel_filter
- `batch_size: 1`

## Teacher domain baselines

| File | Data | reference_cfg | LoRA | fg:bg |
|---|---|---|---|---|
| `teacher/all.yaml` | all categories | on | 128 | 2:1 |
| `teacher/accessory_jewelry.yaml` | accessory + jewel_filter | on | 128 | 2:1 |
| `teacher/animal.yaml` | animal | off | 128 | 2:1 |
| `teacher/fashion_tryon.yaml` | fashion_tryon | off | 128 | 2:1 |
| `teacher/general_object.yaml` | general_object | off | 128 | 2:1 |
| `teacher/human.yaml` | human | off | 128 | 2:1 |

## Other accessory ablations

| File | What it changes vs teacher/accessory_jewelry |
|---|---|
| `exp/accessory/r256.yaml` | LoRA r/alpha 256, fg:bg still 2:1 |
| `exp/accessory/fg5.yaml` | fg:bg = 5:1 |
| `exp/accessory/adaptive_crop.yaml` | Adaptive Crop, r128, 2:1 |
| `exp/accessory/hybrid_masking.yaml` | Hybrid Masking, r128, 2:1 |
| `exp/accessory/mask_control.yaml` | independent mask_control on (+ reference_cfg), r256, fg:bg=3:1 |

## Scale / default / legacy

| File | Role |
|---|---|
| `default.yaml` | Default train/infer entry |
| `scale/4.yaml` ... `scale/32.yaml` | Auto-selected by `run.sh` when no config arg is given |
| `legacy/native.yaml` | Older native-style baseline |

## Rename map (old → new)

| Old | New |
|---|---|
| `insert_anything_fixed_pixels_parquet.yaml` | `default.yaml` |
| `insert_anything_native_qwen_image_inpainting.yaml` | `legacy/native.yaml` |
| `insert_anything_4.yaml` | `scale/4.yaml` |
| `insert_anything_8.yaml` | `scale/8.yaml` |
| `insert_anything_16.yaml` | `scale/16.yaml` |
| `insert_anything_32.yaml` | `scale/32.yaml` |
| `insert_anything_teacher_all_data.yaml` | `teacher/all.yaml` |
| `insert_anything_teacher_accessory_jewelry.yaml` | `teacher/accessory_jewelry.yaml` |
| `insert_anything_teacher_animal.yaml` | `teacher/animal.yaml` |
| `insert_anything_teacher_fashion_tryon.yaml` | `teacher/fashion_tryon.yaml` |
| `insert_anything_teacher_general_object.yaml` | `teacher/general_object.yaml` |
| `insert_anything_teacher_human.yaml` | `teacher/human.yaml` |
| `insert_anything_teacher_accessory_jewelry_lora256_reference_cfg.yaml` | `exp/accessory/r256.yaml` |
| `insert_anything_teacher_accessory_jewelry_fg5_bg1.yaml` | `exp/accessory/fg5.yaml` |
| `insert_anything_teacher_accessory_jewelry_adaptive_crop.yaml` | `exp/accessory/adaptive_crop.yaml` |
| `insert_anything_teacher_accessory_jewelry_adaptive_crop_r256_fg3.yaml` | `exp/accessory/adaptive_crop_r256_fg3.yaml` |
| `insert_anything_teacher_accessory_jewelry_hybrid_masking.yaml` | `exp/accessory/hybrid_masking.yaml` |
| `insert_anything_teacher_accessory_jewelry_hybrid_masking_r256_fg3.yaml` | `exp/accessory/hybrid_masking_r256_fg3.yaml` |

## Launch examples

```bash
# 8-GPU Adaptive Crop experiment
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 NUM_MACHINES=1 MACHINE_RANK=0 \
bash run.sh configs/exp/accessory/adaptive_crop_r256_fg3.yaml

# 8-GPU Hybrid Masking experiment
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 NUM_MACHINES=1 MACHINE_RANK=0 \
bash run.sh configs/exp/accessory/hybrid_masking_r256_fg3.yaml

# Domain baseline
bash run.sh configs/teacher/accessory_jewelry.yaml

# Default scale config for 8 processes (no explicit config arg)
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 bash run.sh
```

Logs land under `logging.output_dir` / `tracker_project_name` / `v*`.
Tracker names now match the short config identity, e.g.
`exp_accessory_adaptive_crop_r256_fg3`.

## Note on old logs

Existing checkpoints under old tracker names such as
`teacher_accessory_jewelry_native_qwen_image_inpainting` are **not** moved.
Only config paths and future run names were cleaned. Resume still points at the
old log directory path if you continue an old run.

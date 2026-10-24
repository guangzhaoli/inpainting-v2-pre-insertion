# MeanFlow and conditional DMD configs

Derive runnable configs from the fused checkpoint's actual model, LoRA,
processor and geometry contracts:

    uv run --offline --no-sync python scripts/build_meanflow_config.py \
      --teacher /absolute/fused/checkpoint --stage warmup \
      --output configs/meanflow/warmup.local.yaml

    uv run --offline --no-sync python scripts/build_meanflow_config.py \
      --teacher /absolute/fused/checkpoint --generator /absolute/warmup/checkpoint \
      --stage dmd --output configs/meanflow/dmd.local.yaml

For AnyFlow-style three-segment backward simulation, supply
--rollout shortcut --gradient-mode full. The default uses full 8-step rollout
with last-step gradient truncation. Full rollout with full gradients is also
available as a control. These approximations are recorded in checkpoints.

The builder preserves the source data and training schedule. Select the intended
budget and output directory before launching. See the
[runtime guide](../../docs/meanflow_dmd_runtime.md) for multi-node launch,
checkpoint/resume, inference and validation.

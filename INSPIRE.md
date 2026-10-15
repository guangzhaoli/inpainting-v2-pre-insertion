# InsertFuse compute assets

Project: `具身世界模型`. Image: `guangzhao:v3` (user-specified defaults).

Repository and shared uv environment:

```
/inspire/hdd3/project/agent-driven-world-model/250041225002/qwen-inpainting/inpainting-v2-final-architecture
/inspire/hdd3/project/agent-driven-world-model/250041225002/qwen-inpainting/inpainting-v2-final-architecture/.venv
```

The project uses Python 3.12 and `pyproject.toml` / `uv.lock`. Prepare new
downloads in a newly created `CPU资源空间` Notebook; the shared environment can
then be used offline by GPU jobs. The current training entry is
`scripts/run_teacher_v2.sh <domain>`, using `configs/teacher_v2/`.

Stable data/model assets, relative to the repository's parent directory:

- `a2edit/data`: training images and masks; domain manifests live in this repo.
- `a2edit/eval`: existing external validation/test pools.
- `pretrained_weights/Qwen-Image-Edit-2511`: frozen base model.
- `pretrained_weights/metrics_offline/dinov2-small`: identity encoder.
- `pretrained_weights/lpips/linear/vgg.pth`: LPIPS linear weights.
- `pretrained_weights/torch/hub/checkpoints/vgg16-397923af.pth`: VGG trunk.
- `pretrained_weights/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`:
  official FA2 wheel for the shared Torch 2.9.1/CUDA 12.8/Python 3.12 environment;
  SHA256 `4e2f9e39313266b1544b68138b15b91ee6221eccf14f7902b7c6620351340810`.
  Source provenance is in `pretrained_weights/wheels/flash-attn-source.json`.
  uv uses its byte-identical canonical-version hard link
  `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` so the filename agrees with
  the package's internal METADATA version and offline sync remains idempotent.
  This asset is shared by CPU preparation and GPU workers in project `具身世界模型`;
  fresh GPU environments install it with `uv sync --offline --locked`.

Use live quotas and availability when submitting. Prefer the user-authorized
`通用世界模型专项空间` H200 resources, then schedulable H200/H100 in
`分布式训练空间`, following the user's per-job limit of eight nodes / 64 GPUs.
Use the highest priority accepted by the selected live quota/project. Set up
Feishu CLI success/failure notifications when launching actual training jobs.

See `docs/teacher_v2.md` for precision, mask/identity supervision, validation,
multi-node environment variables, and old-checkpoint compatibility.

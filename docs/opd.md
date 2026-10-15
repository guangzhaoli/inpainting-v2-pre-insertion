# 多 Teacher On-Policy Distillation（OPD）

Teacher v2 请使用 `uv run --offline --no-sync python -m scripts.build_opd_config`
从实际 checkpoint 生成配置，不要仅替换旧 rank-256 模板里的路径。详见
[`training_optimizations.md`](training_optimizations.md)。`teacher.weight` 现在是
样本 loss 的绝对倍率，在每卡 batch size 1 和梯度累积下仍然生效；全 1 权重不变。
默认轨迹驻留 GPU，条件编码和冻结 teacher 的几何 token 在 batch 内复用。

这套实现把多个 task-specific Teacher 蒸馏到一个 Student LoRA，同时把会频繁变化的部分拆成配置和插件边界。当前可直接运行的路径是“同一 Qwen-Image-Edit 基座 + 多个兼容 LoRA Teacher”；异构完整模型 Teacher 也有独立 policy 后端契约，但需要对应 idea 分支提供具体 policy 类。

## 训练语义

一次 OPD batch 的顺序是：

1. 当前 Student 从噪声执行一次无梯度、确定性的 Euler flow rollout。
2. rollout state 全部 detach；Teacher 看到的是当前 Student 实际访问的状态。
3. 每个样本按 `category` 路由到唯一 Teacher；混合 batch 会按 Teacher 分组。
4. 冻结 Teacher 在这些 state/timestep 上给出目标 velocity。
5. 切回 Student，在相同 state/timestep 上重新前向并反向传播。
6. 每个 timestep 的 loss 立即 backward，避免同时保留整条 Qwen 轨迹的计算图。

确定性更新为：

```text
x[k+1] = x[k] + (sigma[k+1] - sigma[k]) * v_student(x[k], t[k])
```

示例让 Teacher 使用 reference CFG、Student 不使用 CFG，使 guidance 被内化到最终 Student。final architecture 还会训练 geometry tokenizer，因此部署必须同时携带 Student LoRA 和 `mask_geometry_tokenizer.safetensors` companion。

## 模块边界

```text
src/distillation/
├── opd.py          # rollout、路由、Teacher target、Student backward、metrics
├── backends.py     # 共享 LoRA / 独立 Teacher policy 后端
├── objectives.py   # velocity 与 ODE/SDE transition-mean 目标
├── routing.py      # category 路由与 round-robin 任务均衡
└── types.py        # detach 后的 FlowTrajectory 与 loss 输出
```

三类主要组件都由 class path 配置：

```yaml
distillation:
  algorithm:
    class_path: src.distillation.opd.QwenImageEditOPDStrategy
    init_args: {}
  teacher_backend:
    class_path: src.distillation.backends.SharedLoRATeacherBackend
    init_args: {}
  objective:
    class_path: src.distillation.objectives.VelocityMatchingObjective
    init_args:
      coefficient: 0.5
```

- 只改变 loss 时新增 `objective`。
- 改变 Teacher 的加载/前向方式时新增 backend 或 independent policy。
- 改变 rollout、state、scheduler 或 prediction space 时新增 `algorithm`。
- Student idea 只改变 Qwen 前向时，可在 Trainer 实现 `opd_forward_velocity(state, timestep, condition)`；也可以继承 OPD strategy 并通过 `algorithm.class_path` 切换。

## 当前可直接使用：共享 LoRA Teacher

`SharedLoRATeacherBackend` 在 Student 的同一个 PEFT/Qwen 容器中挂载多个命名 adapter。它会在 optimizer 和 DDP 构造之前加载 Teacher，并保证（final geometry-tokenizer 路径不支持 FSDP checkpointing）：

- 只有 Student adapter 可训练；
- Teacher 前向始终在 `no_grad` 下；
- 每次 Teacher 前向后恢复 Student adapter；
- checkpoint 只保存 Student adapter，以及 final geometry path 所需的 Student `mask_geometry_tokenizer.safetensors` companion；
- Teacher 若声明异构 `policy`，配置会被拒绝而不是静默忽略。

Teacher 示例：

```yaml
teachers:
  - name: animal
    route_categories: [cat, dog]
    weight: 1.0
    guidance:
      mode: reference_cfg
      scale: 4.0
      prompt_mode: empty
    lora:
      adapter_name: opd_teacher_animal
      pretrained_weight: /absolute/path/to/model.safetensors
      r: 256
      lora_alpha: 256
      target_modules: [to_q, to_k, to_v]
```

共享后端要求 Teacher 与 Student 的基座、LoRA target modules 和 DiT 前向兼容。启用 final geometry tokenizer 时，Teacher 的 LoRA 文件旁边必须有 `mask_geometry_tokenizer.safetensors`；也可以在 Teacher 节点显式设置 `mask_geometry_pretrained_weight`。后端会为每个 Teacher 加载冻结 companion，并在正/Null reference CFG 分支使用对应 token；Student tokenizer 则保留梯度。仅仅 checkpoint 文件能被找到，不代表结构一定兼容；首次真实运行仍要验证加载结果。

## 异构 Teacher：独立 policy 后端

当某个 Teacher idea 使用独立模型、不同模块或自己的预处理时，切换为：

```yaml
teacher_backend:
  class_path: src.distillation.backends.IndependentTeacherBackend
  init_args: {}
teachers:
  - name: depth_teacher
    route_categories: [person]
    policy:
      class_path: my_idea.opd.DepthTeacherPolicy
      init_args:
        checkpoint: /absolute/path/to/checkpoint
```

Policy 契约是：

```python
class DepthTeacherPolicy:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def setup(self, *, trainer, teacher):
        self.model = load_frozen_model(self.checkpoint)

    def predict_velocity(self, *, state, timestep, condition, batch, guidance):
        # batch 是路由后的原始输入；condition 是标准 Qwen OPD condition。
        # 输出必须和 state 同 shape，并处于 Student velocity space。
        return self.model.predict_velocity(state, timestep, batch)
```

后端会自动冻结暴露为 policy 本身或 `policy.model` 的 `nn.Module`。若 policy 管理多个非标准模块，则需实现 `assert_frozen()`。加载设备、offload 和模型专属预处理由 policy 负责。若 Teacher 输出不在 Student velocity space，则应改 algorithm，而不是在 loss 前做含义不清楚的硬转换。

## Guidance 与条件

每个 Teacher 和 Student 都可独立配置：

- `mode: none`
- `mode: reference_cfg`
- `mode: text_cfg`
- `prompt_mode: batch | empty`
- `scale >= 1`（`mode: none` 时必须为 `1`）

`QwenImageEditInsertAnythingTrainer.prepare_embeddings(stage="opd")` 不会生成 ground-truth `image_latents` 或 region-perceptual 的 GT 图像字段。它只保留生成画布尺寸和模型真正需要的 source/reference/geometry 条件；reference CFG 时正分支与 null-reference 分支拥有相同目标几何。MaskMix-v2 会在 `prepare_opd_batch()` 中每 batch 只采样一次，所有 policy 共享同一条件。

## Objective

默认 velocity matching：

```yaml
objective:
  class_path: src.distillation.objectives.VelocityMatchingObjective
  init_args: {coefficient: 0.5}
```

也可用确定性 ODE transition-mean matching：

```yaml
objective:
  class_path: src.distillation.objectives.TransitionMeanMatchingObjective
  init_args: {mode: ode}
```

`TransitionMeanMatchingObjective` 本身实现了 SDE reverse-KL mean term，但当前 Qwen rollout 是确定性 ODE，strategy 会明确拒绝 `mode: sde`。启用 SDE 必须同时提供匹配的随机 rollout algorithm。

## 路由与任务均衡

配置会在训练前拒绝重复 Teacher name、重复 adapter name、重叠 category、无效 fallback、缺失 checkpoint/policy。

```yaml
fallback_teacher: general_object
balance_tasks: true
```

`fallback_teacher` 只是未显式匹配类别的唯一后备路由，不会融合多个 Teacher，也不等同于另一个 `all` Teacher。

`balance_tasks: true` 使用 `RoundRobinTeacherSampler`：按 Teacher 建样本池、epoch 内确定性打乱、把小池重复到最大池长度，再按 Teacher 轮转。示例是 3 个 Teacher、`batch_size: 1`、`gradient_accumulation_steps: 3`，用于在一个 accumulation 周期覆盖完整 Teacher round。改变 Teacher 数、batch size 或进程数后，应重新检查每次 optimizer update 的任务组成。

在 final architecture 的五专家方案中，使用互不重叠的路由：`accessory/jewel_filter`、`cat/dog`、六个 fashion 类别、`others`、`person/person_hair`。模板位于 `configs/opd/final_architecture_five_teacher.yaml.example`。先分别运行五份互斥的 `configs/exp/final_architecture*.yaml`（accessory、animal、fashion_tryon、general_object、human）训练专家，再把每个专家 checkpoint 的 `model.safetensors` 路径填入模板并改名为 `.yaml`。`final_architecture_all.yaml` 是重叠类别的 aggregate baseline，不要把它作为第六个 routed Teacher。模板把 `loss.mask_loss` 和 `loss.region_perceptual` 关闭，因为 OPD 当前目标是全 latent velocity matching，不会调用监督 GT loss；geometry tokenizer 仍保持启用并参与 Student 优化。

模板里的 `student_guidance.prompt_mode: empty` 只描述蒸馏阶段的 Student 条件，并不会改变独立推理 CLI 的参数校验。由于模板关闭了 `model.reference_cfg.enabled`，部署时仍应显式传入 `--prompt ""`（或在调用方提供空字符串）；不要把 `reference_cfg.force_empty_prompt` 当作关闭 CFG 后的 prompt-free 开关。

`infer.py` 的配置加载会自动丢弃 `distillation` Teacher 图，因此可以直接复用填好路径的 OPD 配置作为部署配置；只需指定 Student checkpoint，例如：

```bash
python infer.py --config configs/opd/final_architecture_five_teacher.yaml \
  --lora /path/to/student/checkpoint --prompt "" \
  --src-image source.png --src-mask source_mask.png \
  --ref-image reference.png --ref-mask reference_mask.png
```

共享 LoRA 后端在 DDP 下通过被包装的 `trainer.dit` 做 Teacher 前向，因此各 rank 必须执行相同次数的模型 forward。当前实现会强制分布式共享后端使用 `batch_size: 1`，并要求所有 Teacher 都同为 guided（每次两次前向）或同为 unguided（每次一次前向）；不满足时会在模型 setup 阶段直接报错，避免 collective 卡死。启用 final geometry tokenizer 时，项目的 FSDP 路径目前明确不支持该 companion 的 checkpointing，请使用项目的 MULTI_GPU/DDP accelerate 配置；单进程仍支持 mixed-Teacher batch。独立 policy 后端不经过 Student 的分布式包装，因此不受上述 collective forward-count 限制，但 geometry tokenizer + FSDP 仍会被 BaseTrainer 拒绝。

## 本地三 Teacher smoke 示例

配置位于 `configs/opd/three_teacher_local.yaml`，连接当前 workspace 已完成的三个 r256 Teacher：

- `accessory`：`accessory`、`jewel_filter`
- `animal`：`cat`、`dog`
- `general_object`：`others`

配置使用当前机器的绝对 checkpoint/model/data 路径，复制到其他机器必须修改。启动前可只做解析：

```bash
/inspire/hdd/global_user/250041225002/miniconda3/envs/qwen-inpainting-pre-insertion/bin/python - <<'PY'
from src.data.config import load_config_from_yaml

config = load_config_from_yaml("configs/opd/three_teacher_local.yaml")
print([teacher.name for teacher in config.distillation.teachers])
PY
```

单进程启动：

```bash
python -m src.main --config configs/opd/three_teacher_local.yaml
```

多卡启动沿用项目 launcher，例如 8 卡：

```bash
TOTAL_PROCESSES=8 NPROC_PER_NODE=8 bash run.sh configs/opd/three_teacher_local.yaml
```

示例中的 `predict.devices` 为 CPU，因为 `mode: fit` 不使用这些字段；真正训练设备由 Accelerate 决定。

`teacher.weight` 是混合 batch 内的 confidence weight；分布式 SharedLoRA 路径要求 `batch_size: 1` 时，任务比例应通过 sampler/数据池控制，而不是依赖这个字段。五专家模板默认等频 round-robin，`gradient_accumulation_steps: 5`。

## 接入不同 worktree idea 的建议

不要把多个含未提交改动的 worktree 整体互相合并。更稳妥的实验单位是：

1. 以当前已包含 OPD 的 final architecture 分支为固定蒸馏底座。
2. 每次只移植一个 Student idea 的最小 commit/补丁。
3. 若只改前向，实现 Trainer hook 或 strategy 子类。
4. 若是兼容 LoRA Teacher，只改 Teacher 配置。
5. 若是异构 Teacher，实现独立 policy，通过 class path 切换。
6. 用相同 seed、route、rollout steps 和评测集做单 idea 与组合消融。

这套边界支持动态替换实现，但不会自动搜索“最佳 idea 组合”；组合优劣仍需要独立实验和统一指标判断。

## 首次 GPU smoke test 检查项

- 所有 Teacher LoRA 与 geometry companion 都成功加载；
- optimizer 中只有 Student 参数；
- Teacher 参数始终无梯度；
- `opd/teacher/*/route_fraction` 符合配置；
- rollout state 已 detach，loss 为有限值；
- reference CFG 的正/null 条件几何一致；
- 保存目录中的 LoRA 是 Student adapter，且 geometry companion 是 Student 版本（不是任一 Teacher 版本）；
- 先用少量 rollout steps、少量 optimizer steps 验证显存，再放大训练。

OPD resume 会继续验证 Student LoRA/geometry artifact contract；Teacher 依赖图仍应保持与原 run 相同（名称、路由、adapter 和 companion）。如果需要替换 Teacher 集合或路由，建议新建 Student run，而不是在旧 checkpoint 上混合恢复。

## 算法参考

- [DiffusionOPD: One-step mean matching / on-policy distillation](https://arxiv.org/abs/2605.15055)
- [DiffusionOPD 官方实现](https://github.com/ali-vilab/DiffusionOPD)
- [Qwen-Image-2.0 Technical Report：OPD velocity matching](https://arxiv.org/pdf/2606.27608)

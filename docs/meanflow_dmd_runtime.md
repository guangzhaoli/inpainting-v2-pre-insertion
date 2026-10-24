# MeanFlow 与条件 DMD 实现指南

已接入 warmup 和 dmd 两阶段，沿用 Qwen InsertAnything 的参考、场景、几何和预处理。原监督训练与原 OPD 默认不启用新路径。实现经过真实微型 Qwen+PEFT、FP32/BF16 CPU 训练循环和两进程同步验证，尚未完成完整 Qwen 权重的 GPU 显存、吞吐及质量验证。

实现：[训练策略](../src/distillation/meanflow.py)、[数学算子](../src/distillation/meanflow_math.py)、[两时间模块](../src/models/flow_map.py)、[配置生成器](../scripts/build_meanflow_config.py)。

**实际读取的官方版本。** 仓库在新建 Inspire CPU Notebook 上获取，准备机已停止并删除；共享盘源码位于父目录 pretrained_weights/source_refs/meanflow_dmd_20261015。

| 仓库 | Commit | 采用的机制 |
| --- | --- | --- |
| [AnyFlow](https://github.com/NVlabs/AnyFlow/tree/bf9195aa04b708a8cd9a7e742bb03265ec3a6c29) | bf9195aa04b708a8cd9a7e742bb03265ec3a6c29 | 插值时间条件、中心有限差分 MeanFlow 目标、三段 shortcut、real/fake clean prediction 差分 |
| [DMD2](https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/sd_guidance.py) | 8d8fa55633d47cfb81bbc7a892e7248f9518763f | 归一化 DMD 方向与 0.5 MSE surrogate |

官方参考文件为 AnyFlow 的 far/trainers/trainer_wan_anyflow_pretrain.py、trainer_wan_anyflow_onpolicy.py，以及 DMD2 的 main/sd_guidance.py。这里独立适配 Qwen，没有运行外部仓库的环境安装脚本，也不声称逐项复现视频论文最优超参数。

**两阶段配置。** 在仓库根目录使用 uv：

    uv run --offline --no-sync python scripts/build_meanflow_config.py \
      --teacher /absolute/path/to/fused/checkpoint \
      --stage warmup --student-steps 8 --teacher-steps 40 \
      --output configs/meanflow/warmup.local.yaml

warmup 从融合 LoRA 与 geometry 初始化 Generator，目标时间 embedding 从冻结原时间模块复制，默认源时间权重为 0.25。教师每批在线生成目标 latent，再做有限差分 MeanFlow 训练；没有离线教师目标缓存。时间参数保持 FP32，默认有限差分 epsilon 为真实 sigma 单位的 0.01。

Flow-map 时间直接以 sigma 传入，避免乘除 1000 导致 t/r 舍入不一致。全 batch 为对角样本时只执行一次带梯度的 forward，省去两个目标恒为零的差分查询。时间由内部采样逻辑生成，DiT 每次前向不再重复检查端点值。

    uv run --offline --no-sync python scripts/build_meanflow_config.py \
      --teacher /absolute/path/to/fused/checkpoint \
      --generator /absolute/path/to/warmup/checkpoint \
      --stage dmd --student-steps 8 --teacher-steps 40 \
      --output configs/meanflow/dmd.local.yaml

配置生成器从 checkpoint metadata 校验基座、LoRA、processor 和 geometry。DMD 初始化必须提供已训练的 flow-map companion；score Teacher 使用原单时间融合模型。生成配置保留原数据和训练 schedule，启动前设置输出目录与所需训练预算。阶段切换是新 run；resume 仅继续相同阶段。

配置生成器清除旧教师验证配置的 negative_prompt，并统一生成训练/推理使用的提示词设置。

processor 按解析后的配置比较；可选字段省略或写成 null 不会被误判为不同预处理。warmup checkpoint 可直接用于生成 DMD 配置。

**共享基座上的三个角色。** Generator、冻结融合 Teacher 和 fake-score 使用独立命名 LoRA。只有 Generator 使用可训练的两时间模块。Teacher/fake-score 使用单时间 forward 和冻结 Teacher geometry tokens，三者保留相同 source/reference/geometry 条件语义。

fake-score 有独立 AdamW、学习率和更新计数。全部 fake optimizer step 完成后，才建立 Generator 活计算图；score 查询不更新权重，且在 backward 前恢复 Generator adapter，使 activation checkpointing 重计算正确。默认每个 batch 一个 fake update，fake_updates 可调。

LoRA 层及 G/F 参数列表在 setup 时缓存，切换相同角色不会再次扫描或设置全模型。CUDA fake AdamW 使用 foreach 更新。只构建一次共同条件；文本编码器自身管理 offload，不嵌套执行整模型迁移。

共享多角色路径使用显式分桶梯度 all-reduce 进行同步数据并行，避开动态 PEFT adapter 与 DDP unused-parameter hooks 的耦合。所有 rank 参与训练，Generator 支持梯度累积，fake-score 每批独立更新；全局未使用的参数保持 grad=None。当前支持 bf16 或 no 精度，不支持 fp16 scaler、FSDP 或 DeepSpeed。原路径的分布式包装不变。

这条路径不继承 routed OPD 的 batch_size=1 限制，也不创建未使用的 OPD objective/router；实际 batch size 仍需满足数据集和预处理的约束。已验证真实 Accelerator/BaseTrainer 的两进程、每进程 batch size 2、梯度累积 2，在 FP32 和 BF16 下 G/F 更新一致。训练中断沿用原 BaseTrainer 行为。

**采样和损失配置。**

| rollout | gradient_mode | 语义 |
| --- | --- | --- |
| full | last | 完整指定步数采样；前缀无梯度，末步反传；默认 |
| full | full | 全少步轨迹反传，显存开销较高 |
| shortcut | full | AnyFlow 式初始→随机区间起点→区间终点→0；整链反传 |

选择 shortcut 时，配置生成器使用 --rollout shortcut --gradient-mode full。零长度段自动省略。真实指定步数采样和 shortcut 是不同训练近似，均写入 checkpoint；部署使用实际指定步数的 flow-map sampler。

DMD 的 re-noising tau 独立于区间端点 t/r，默认在 0.02–0.98 均匀采样。其方向为 fake_x0-real_x0，按每样本 mean(abs(generated-real_x0)) 归一化，再通过停止目标梯度的 surrogate 反传。分母设下界，非有限方向报错。DMD 阶段默认保留 forward MeanFlow 目标；没有额外 GAN 分支。

keep_weight 默认 0。开启后，在同一重新加噪状态比较教师 clean estimate 与 Generator 到零时间的预测，用 GT foreground/background 分开归一化 latent 损失。本策略尚未加入像素 LPIPS、参考差分或独立 target-score 校准训练。

保真项复用 DMD 已查询的 teacher velocity。无梯度 rollout 的 geometry tokens 每条轨迹只计算一次，带梯度步骤重新计算以保留 tokenizer 梯度；FP32/BF16 均测试了输出和梯度一致性。sigma schedule 按步数、分辨率和设备缓存，验证产生的缓存也可供后续 autograd 使用。

Qwen-Image-Edit-2511 的 shift_terminal=0.02 在单步 schedule 下会除零；1 步 flow map 直接使用 [1,0] 端点，不经过该变换。已用真实 diffusers scheduler 验证 40/8/4/1 步时间表；支持运行 1 步不代表该预算的生成质量已达标。

默认 teacher_steps=40、student_steps=8、fake_updates=1、full rollout、单样本且不启用辅助项时，每个 microbatch 的 DiT 前向次数是：对角 MeanFlow 样本 60 次，非对角 62 次，默认 50% 对角概率下期望 61 次。该计数不包含梯度检查点重计算、反向、文本编码和 VAE；其中 40 次是在线 teacher 的既定成本。实现未通过减少训练数据、训练步数或 teacher 步数来宣称提速。

teacher_score_probe_interval 默认 0。开启且 forward_weight>0 时，在在线教师输出的重加噪状态上记录 meanflow/teacher_score_probe_mse。该诊断不等于证明 score 校准或图像质量改善。

**背景约束。** anchoring=deployment 沿用 predict.latent_anchoring：legacy 不投影，其他模式每步应用 source trajectory 和 packed edit weight。训练优先使用 batch.source_image；未提供时仅编码挖洞后的场景作为 anchor，避免直接将完整 GT 主体编码进锚点。source_hole_mask、几何与 free mask 同步，GT mask 仍用于损失。

当前沿用固定逐步软权重，没有实现步数不变的累计投影。不同步数和 shortcut 的边界差异需单独评价；若需要真实干净背景配对，数据集须提供独立 source_image。

无独立 source_image 且 source/target 网格一致时，anchor 直接复用第一段已编码的 masked source control latent，省去一次重复 VAE 编码；有独立 source_image 时保留完整 source 的独立编码。

**多节点启动。** 先分配资源，再在所有节点运行，分别设置 MACHINE_RANK。脚本不分配机器，最多八节点、每节点八卡，不将单节点作为默认。

    NUM_MACHINES=2 NPROC_PER_NODE=8 TOTAL_PROCESSES=16 \
    MACHINE_RANK=0 MAIN_PROCESS_IP=<coordinator-ip> \
    bash scripts/run_meanflow_dmd.sh configs/meanflow/warmup.local.yaml

第二台机器使用 MACHINE_RANK=1；4/8 节点依次设置。DRY_RUN=1 仅检查命令。正式 Inspire 提交沿用项目、镜像、资源优先级及飞书通知约定。

**保存恢复。** 原 Generator LoRA、geometry、optimizer.bin、scheduler.bin 和各 rank RNG 继续保存，新增：

| 文件 | 用途 |
| --- | --- |
| flow_map_time_embedding.safetensors | 训练恢复与推理 |
| meanflow_training_state.json | 阶段、采样/损失配置、Teacher SHA256、fake 更新计数 |
| fake_score_lora.safetensors | DMD 恢复 |
| fake_score_optimizer.pt | DMD 恢复 |

state.json 记录新增文件的 SHA256。恢复拒绝阶段、教师内容或训练契约变化。部署只需要 Generator LoRA、geometry、时间模块和 train_config，配置读取会移除 Teacher 依赖图；不加载 fake optimizer。

在 BF16 基座上训练时，G/F LoRA、时间模块和 geometry 的目标参数先转为 FP32，再复制 checkpoint 权重，避免先截断到 BF16 后再升回 FP32。Fake-score 按可训练角色加载；冻结基座和 Teacher 保持模型精度。

model.safetensors 保存 PEFT adapter 权重；pytorch_lora_weights.safetensors 使用同一套 PEFT 键并添加 Qwen pipeline 的 transformer 前缀。加载时逐键统一前缀及 LoRA 命名，也能完整读取本仓库先前产生的混合格式导出文件。推理按目录选择 checkpoint 时会优先使用 pipeline 导出文件，因此导出与训练权重必须包含相同的 adapter 集合。

    RESUME=/absolute/path/to/same-stage/checkpoint \
    NUM_MACHINES=2 NPROC_PER_NODE=8 TOTAL_PROCESSES=16 \
    MACHINE_RANK=0 MAIN_PROCESS_IP=<coordinator-ip> \
    bash scripts/run_meanflow_dmd.sh configs/meanflow/dmd.local.yaml

**推理与验证。** infer.py、infer_testset.py 与周期验证共用 flow-map sampler。未显式指定时，CLI 使用 model.flow_map.inference_steps（默认 8）、true_cfg_scale=1；显式步数覆盖保留。额外 CFG 与 negative prompt 被拒绝，原非 flow-map CLI 默认行为不变。

    uv run --offline --no-sync python infer.py \
      --config /absolute/path/to/dmd/checkpoint/train_config.yaml \
      --lora /absolute/path/to/dmd/checkpoint --device cuda:0 \
      --src-image scene.png --src-mask placement.png \
      --ref-image reference.png --ref-mask reference_mask.png \
      --num-inference-steps 8 --true-cfg-scale 1 \
      --output output/meanflow_result.png

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --no-sync \
      pytest -q tests/test_meanflow_dmd.py tests/test_meanflow_checkpoint_loading.py

测试包含数学梯度方向、有限差分、真实微型 Qwen+PEFT 两阶段更新、checkpoint 重计算时的角色恢复、三种 rollout、背景约束、精确 optimizer 恢复、companion 完整性、配置/CLI、两进程同步及真实 BaseTrainer 的 FP32/BF16 生命周期。完整模型的 GPU 训练、吞吐与生成质量需另行验证。

额外 review 回归见 tests/test_meanflow_review.py：覆盖对角时间精度、真实测试集初始化、无文本验证数据、共享条件编码、前向预算及两进程真实训练。训练和推理入口均保留空提示词语义，metadata 记录实际使用的提示词。

实现保留正常训练、采样和推理路径的确定性修复与吞吐优化。梯度、恢复、条件编码和分布式正常路径测试保留；不增加残缺导出目录、手动改写恢复配置或特殊中断的专门分支。gdown 和 pycocotools 通过 uv 的现有离线缓存加入临时测试环境，项目依赖和锁文件未改变。

tests/test_meanflow_checkpoint_loading.py 使用实际 Qwen LoRA target 列表检查 PEFT、pipeline 导出和既有混合格式文件，并验证 BF16 warmup→DMD 权重完整性。正常保存后恢复的训练最终权重及 G/F optimizer 状态与连续训练逐值一致。完整报告位于 output/meanflow_dmd_loading_review_tests.xml。

本轮全仓回归为 581 passed、4 skipped；新增的 8 个 checkpoint 加载测试全部通过。跳过项为可选真实数据和 CUDA 模块迁移测试。此前通过 checkpoint 目录默认加载混合格式导出文件得到的推理结果，应使用修复后的完整 adapter 加载结果重新对比。

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      uv run --offline --no-sync --with gdown --with pycocotools \
      python -m pytest -q tests --disable-warnings --tb=short \
      --junitxml=output/meanflow_dmd_loading_review_tests.xml

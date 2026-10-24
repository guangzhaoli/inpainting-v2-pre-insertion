# InsertFuse 融合模型的 MeanFlow 与 DMD 后训练方案

本方案在已选定的 OPD 融合模型之后，先建立 flow-map 少步生成器，再用条件 DMD 改善实际生成质量。基础训练路径已实现并通过微型 Qwen 的 CPU 验证，尚未进行完整权重的 GPU 训练与质量评估；使用方法见 [实现指南](./meanflow_dmd_runtime.md)。已有教师区间回归方案保留为质量、成本及复杂度对照。

推荐顺序是冻结融合教师、MeanFlow 式初始化、固定 8 步条件 DMD、验证身份与细节保留、4 步实验。不同档位是否可用由完整测试集上的质量与实际延迟决定。推理步数压缩不意味着减少原定训练数据或最终评估范围。

依据是 [AnyFlow](https://arxiv.org/html/2605.13724v1) 的 flow-map 初始化加 DMD 后训练流程、[DMD2](https://arxiv.org/html/2405.14867) 对 fake-score 估计与更新速度的分析，以及 [MaskFlow 官方实现](https://github.com/ReyChiaro/MaskFlow) 的局部编辑 DMD 8/16 步变体。这些证据支持技术路线值得验证，不构成 InsertFuse 上的质量结论。

**三个逻辑角色必须分开。**

| 角色 | 初始化与训练状态 | 输出与用途 |
| --- | --- | --- |
| 冻结教师 T | 最终选定的 OPD 融合 checkpoint，含 geometry companion | 教师样本，以及目标分布的 score 近似 |
| 少步生成器 G | 从同一融合模型复制，增加两时间条件 | 区间平均速度与实际少步输出 |
| Fake-score 模型 F | 可从融合模型的单时间 velocity 网络初始化，独立优化 | 拟合当前 G 生成结果的重加噪分布 |

Fake-score 是去噪/score 估计器，不是必须输出真假概率的 GAN 判别器。第一版使用条件 DMD；DMD2 的 GAN 分支另作可选增量，不能无依据混入首版。

三个角色必须使用相同的条件语义：场景、参考图、放置 mask、geometry、文本策略与 crop 坐标。F 不接收参考而 G 接收参考，会使它们对应不同的条件分布，不能仅靠额外身份 loss 掩盖这一差异。

三个逻辑角色不必永久驻留三份完整基座。在冻结基座、仅训练 LoRA 时，可以评估共享基座，但须隔离 G 的两时间模块、T/F 的单时间路径、各自 optimizer 与 geometry 状态。保留 G 计算图期间切换 adapter，尤其与 activation checkpointing 结合，必须保证反向重计算使用正确 adapter；不得更新该图依赖的参数。

**阶段 A 用融合教师的能力建立少步初始化。**

冻结教师的完整推理契约，包括 LoRA、geometry、processor、参考 guidance、scheduler 和背景 anchoring。若融合教师已吸收 Reference CFG，使用其验证过的单分支设置，避免重复放大指导。

按输入条件 c 生成教师样本 y_T，保留条件—输出配对与 seed。以教师样本为数据端，构造加噪状态和条件 velocity，进行 MeanFlow 式训练：

\[
F_\theta(z_t,t,r,c)=z_t-(t-r)u_\theta(z_t,r,t,c).
\]

这使初始化学习融合教师的输出分布。只在原 GT 上重新训练，可能将模型重新推向普通监督目标；是否混入 GT 应作为 replay 对照。教师失败案例应记录分析，不能只保留容易样本来掩盖质量问题。

时间条件、有限差分/JVP 与损失平衡参考 AnyFlow，但其视频最优参数不能直接当作 Qwen 最优参数。保留原图像条件与几何路径，先训练时间模块及 Student LoRA，geometry 是否继续训练单独比较。平均速度 u 与单时间 score 估计所需的瞬时 velocity v 不可混用。

先比较教师原高质量步数、教师直接 8 步、G 仅 flow-map 初始化，确认放置、参考身份和局部结构。阶段 B 继续保留 forward flow-map 训练，避免只优化 DMD 后丢失其他区间的稳定性。

**阶段 B 匹配条件生成分布。**

目标是让 q_G(z_0|c) 接近 p_T(z_0|c)。理想的完整条件分布匹配包含身份信息，不能笼统认为 DMD 天生忽略身份。实际风险来自有限 score 模型、区域不平衡、优化误差和条件实现不一致。

t、r 表示 Generator 的采样区间端点；tau 表示 DMD 对最终生成样本重新加噪的时间。这些随机变量作用不同，不能直接将 OPD 低噪声 query 的结论当作 DMD tau 采样规则。

G 产生局部 latent 输出 z_G，F 更新使用：

\[
\widetilde z_\tau=(1-\tau)\operatorname{sg}(z_G)+\tau\epsilon,
\]
\[
L_F=\mathbb E\left\|
v_\psi(\widetilde z_\tau,\tau,c)
-\left(\epsilon-\operatorname{sg}(z_G)\right)
\right\|^2.
\]

该步骤只更新 F。Generator 更新时重新产生带正确计算图的 z_G，查询冻结 T 与当前 F，停止 score 目标梯度，仅将分布匹配方向传回 G。

在线性 forward noising 下，若 velocity 对应正确的条件去噪场，则

\[
s(z_\tau,\tau,c)
=-\frac{z_\tau+(1-\tau)v(z_\tau,\tau,c)}{\tau}.
\]

实现必须处理 tau 端点、scheduler 参数化和归一化，不能把两时间平均速度直接代入。DMD 的 Generator 梯度结构为

\[
\nabla_\theta L_{\mathrm{DMD}}
\propto
\mathbb E\left[w(\tau)(s_F-s_T)
\frac{\partial z_G}{\partial\theta}\right],
\]

w 包括噪声路径与时间权重因子。可通过停止目标梯度的 surrogate 实现；这不是当前 OPD 中的普通教师—学生 velocity MSE。

T/F 在 Generator 更新时不做 optimizer step。Fake-score 与 Generator 分阶段更新，明确梯度边界。DMD2 在其 ImageNet 实验中发现 5 次 fake-score 更新对应 1 次 Generator 更新有助稳定，但不是插入任务的通用最优比例。应观察 F 的去噪误差、score-gap 尺度和验证输出，再确定更新比率。

**需要检查融合教师作为 real score 的近似。**

上述 velocity-to-score 关系依赖所选 forward-noising 分布。融合教师经过 OPD、guidance 吸收及可能的背景投影，其生成质量好，不自动证明在所有重加噪状态上的 score 都校准良好。这是方法假设，不是已证实的当前模型故障。

首版可用 T 的 velocity 作 score 近似，先在教师输出的重加噪状态上检查去噪一致性、预测尺度与随 tau 的异常。若明显失配，再考虑在教师样本上训练独立 target-score 校准 adapter，保持原融合策略冻结。该校准增加成本，不是每个实验默认需要的模块。

**训练采样过程先保持明确，再比较 shortcut。**

G 的训练输出、F 拟合的生成样本和部署 sampler 必须有清楚的对应关系。第一组基线固定 8 步，明确 Generator 回传路径；显存需要的截断梯度是独立近似，应记录验证。现有 FlowTrajectory 自动 detach 的结构适合局部 velocity OPD，不能把其脱离计算图的终点直接用于 Generator 的 DMD 更新。

AnyFlow 的三段 shortcut backward simulation 可作为训练加速候选，但其输出与真实 8 步输出不天然完全相同，尤其在包含背景投影时。应比较两种采样的状态与最终输出，避免把训练采样偏移误判为 DMD 优化效果。

固定 8 步稳定后再做 4 步。若同一 Generator 混合多个预算，应明确 F 拟合预算混合分布还是按预算条件化的分布。匹配混合分布不自动保证每个预算都保真，需要逐预算评测。

**插入保留项逐个加入。**

建议 Generator 目标为：

\[
L_G=\lambda_{\mathrm{FM}}L_{\mathrm{flowmap}}
+\lambda_{\mathrm{DMD}}L_{\mathrm{conditional\ DMD}}
+\lambda_{\mathrm{keep}}L_{\mathrm{keep}}.
\]

先验证不带 L_keep 的标准 flow-map 加条件 DMD，再加入一个保留项。首选区域归一化教师区间约束，或在配对关系可靠的输出上做局部感知监督。不要首次就堆叠所有损失。

相同 scene/reference/seed 不保证 Teacher/Student 姿态完全像素对齐。像素/LPIPS 配对应来自可靠来源，例如同一中间状态的教师 continuation、有效局部对应或确实配对的 GT。姿态不同的参考图使用身份/局部对应特征，不作为全图像素目标。

主干稳定后可以比较参考响应蒸馏：固定场景、mask，更换合法参考，要求 G 保留教师的区间或输出响应。它增加前向成本，需要与等算力双条件独立监督比较。若标准条件 DMD 已经保住身份，额外项不应仅为叙事而保留。

不要任意用 mask 重权重 DMD score-gradient 后仍宣称优化相同全图 KL。空间重权重可作为预条件或区域目标，但应明确目标变化并独立评价。首版保持标准条件 DMD，用可解释辅助项处理区域质量。

**背景操作与 DMD 输出保持一致。**

若部署使用 expanded_soft，明确训练 Generator 是否执行相同操作，并保持 noise anchor、mask、crop 与坐标一致。固定逐步软混合的累计作用随步数变化，跨度校准应与不校准及仅推理校准分别比较。

硬背景投影位置的生成梯度可能为零。完整输出指标可以评价产品行为，但投影后的零背景误差不能证明网络独立保持背景。DMD 目标分布应与实际生成/投影规则相容。

训练统一在共同的 crop/latent 空间，恢复原图使用共同 wrapper，不能将恢复后整图与局部 latent 混入同一个 score 接口。

**关键实验对照。**

| 实验 | 要回答的问题 |
| --- | --- |
| 教师高质量步数、直接 8/4 步 | 原始少步误差 |
| MeanFlow 式初始化 | 平均速度表示的收益 |
| 初始化加标准条件 DMD | DMD 是否改善实际质量 |
| DMD 加一个保留项 | 身份/细节是否进一步保持 |
| 实际 rollout 与 AnyFlow shortcut | 训练加速是否引入采样偏移 |
| 8 步与 4 步分别评价 | 可用的质量速度点 |
| 同预算 FlowMap-OPD/区间回归 | DMD 是否值得额外成本 |
| 按需 score 校准 | 瓶颈是否来自教师 score 近似 |

保留完整固定测试集，按领域、小目标面积和边界复杂度分层。分别报告参考身份、局部结构、遮挡、交互自然度、原始背景误差及 anchored/strict composite 输出。记录 NFE、条件编码、VAE 开销、实际延迟、总训练 GPU 时间与峰值显存。

实现已提供 DMD strategy、独立 Fake-score optimizer 与 checkpoint、Generator 活计算图、两时间模块和 score 转换，并接入配置生成、推理与分布式训练。共享基座的多角色梯度隔离和恢复细节见实现指南。

MeanFlow 加 DMD 是已有路线。InsertFuse 的研究贡献仍需由明确的能力退化和修复机制支持；区间回归已在 FlowMap-OPD 中出现，mask 编辑 DMD 已有 MaskFlow 等先例。该方案首先验证技术效果，再决定哪些插入特定机制值得保留。

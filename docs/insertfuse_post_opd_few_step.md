# InsertFuse 融合模型的少步蒸馏设计

另一条候选路线是先做 MeanFlow 式初始化，再做条件 DMD，详见 [MeanFlow 与 DMD 后训练方案](./insertfuse_meanflow_dmd.md)。本文件的区间监督方案保留为对照与可选保留项。

目标是将已经完成多领域 OPD 的统一 InsertFuse 模型进一步转成少步模型。第三阶段使用一个冻结的融合教师 T_mix，少步 Student 从同一个融合 checkpoint 初始化。专家训练和多教师 OPD 是前置阶段；第三阶段不需要重新调用五个领域专家。该方案为研究设计，尚未实施训练或确认质量收益。

建议先用 8 步验证能力保留，再以 4 步作为主要加速实验点，2 步作为进一步的质量速度探索。采样步数压缩不等于减少原定训练数据、训练预算或最终评估范围。最终是否接受一个步数档位，由完整测试集上的身份、细节、融合、背景及延迟共同决定。

**方法主线是冻结融合策略，再在少步 Student 自己访问的状态上学习区间转移。**

领域教师训练 → 多教师 OPD → 冻结融合模型 T_mix → 少步 Student S_8 / S_4。

| 阶段 | 教师 | 训练目标 | 推理产物 |
| --- | --- | --- | --- |
| 领域训练 | 监督数据 | 各领域的插入、身份与细节能力 | 多个专家 |
| 多教师 OPD | 按领域选出的专家 | 统一不同领域的能力 | 一个融合模型 T_mix |
| 本方案 | 同一个冻结 T_mix | 用更少更新保留融合模型的插入行为 | 一个少步模型 |

这更接近在现有模型上进行 on-policy 步数蒸馏。[FlowMap-OPD](https://arxiv.org/html/2609.37851) 提供了平均速度/瞬时速度接口及一致性的启发，但它的主要多专家实验从已有 MeanFlow Student 起步；[AnyFlow](https://arxiv.org/html/2605.13724v1) 展示了将预训练模型适配成 flow map 再做 on-policy 蒸馏的路线。多教师融合和本阶段的步数压缩应有独立评价。

进一步核对新颖性后，基本区间蒸馏应明确归为已有方法基线：FlowMap-OPD 第 4.3 节式 22 已给出 Teacher/Student flow-map 的平方误差，并明确允许教师目标由积分教师 velocity 得到。因此“多小步教师 → 一大步学生”在数学形式上与其框架直接重合；改为一个 OPD 融合教师、Qwen 基座或插入数据，都不足以单独构成新的少步算法。本文的研究候选仅限于后述参考响应与步数变化下的编辑约束，且需要实验与邻近文献进一步验证。

**先冻结一个可复现的教师推理契约。**

T_mix 指最终选定的融合 checkpoint，不在本文预设某个当前路径已经是合格融合模型。冻结内容包括基座、融合 LoRA、geometry companion、预处理、参考条件、mask 构造、scheduler/noise schedule 和是否启用背景 anchoring。若该融合模型已经吸收 Reference CFG，应使用它验证过的单分支推理设置，不能无依据再叠加一次参考 CFG。

Student 复制融合 LoRA 与 geometry 权重，保留相同 VAE、Qwen-VL、图像尺寸和输入格式。最初训练 Student LoRA 与新增的时间区间条件；geometry 是否继续训练作为独立控制。数据继续覆盖全部领域并按领域记录指标。虽然不再路由五个教师，领域样本平衡仍然必要。

在固定 8 步、固定 schedule 的低成本基线里，目标时间可由当前采样节点决定，可沿用单时间输入。支持 4/8 步混合训练或更一般时间跨度的主模型，则显式输入起点和终点，学习：

\[
F_\theta(x_a,a,b,c)
=x_a-(a-b)u_\theta(x_a,a,b,c),\qquad a>b,
\]

其中 a、b 使用真实噪声尺度，c 包括场景、参考和几何条件。不能将平均速度 u 直接等同于教师在起点的瞬时 velocity。新的时间条件必须与 checkpoint 兼容，且在 a=b 时保持原有瞬时接口。零门控残差与 AnyFlow 的时间嵌入插值可作初始化对照，不能预设零初始化必然更稳定。

**一次 on-policy 训练 batch 可以这样执行。**

1. 对场景、参考及 mask 做一次预处理并固定条件包；Teacher/Student 不得各自随机采样不同 MaskMix 或 crop。
2. 用当前 Student 和目标少步 schedule 进行无梯度采样，抽取一个区间起点 x_a；随后 detach。
3. 从同一个 x_a 出发，冻结 Teacher 沿较细 schedule 走到 b，得到目标 y_T。教师从 Student 状态开始，不能直接拿离线教师轨迹的同序号状态替代。
4. Student 在同一个 x_a、相同条件下做一次 a→b 更新，得到 y_S。
5. 对 y_S 和停止梯度的 y_T 做区域归一化损失；仅在选定的终点/低噪声训练批次增加图像空间监督。
6. 更新 Student 后刷新 rollout，避免长期使用过时轨迹。

可先将教师 40 步细网格的每五个区间合成一个 Student 区间，形成 8 步网格；4 步时每个区间覆盖十个细步。这是明确的初始对照，实际 a、b 必须取 scheduler 返回的 sigmas，且需考虑 Qwen 随 image sequence length 变化的 schedule。教师子步数应通过目标稳定性检查确认，不能把五步视为所有状态下都足够。

训练初期可用 Teacher 状态做短暂区间学习预热，之后切换为 Student 状态训练。预热不是 on-policy，应与主体阶段明确区分。若只抽一个区间且该 batch 不计算最终输出项，Student 只需采样到该区间起点；不必为局部训练保留完整 rollout 计算图。

**监督实际编辑更新，包括部署时使用的背景操作。**

不含 anchoring 时，教师目标为在区间内多次积分 velocity 的结果，Student 用一个区间映射拟合。多步到一步的思想与 [Progressive Distillation](https://arxiv.org/html/2202.00512v2)、任意时间区间映射与 [Consistency Trajectory Models](https://arxiv.org/html/2310.02279v3) 有直接先例；本阶段的任务特定增量需要另行证明。

若目标部署启用 expanded_soft，令 Phi_T 表示教师在细网格上包含投影的完整更新组合，Phi_S 表示 Student 大步更新加部署投影：

\[
y_T=\Phi_T^{a\to b}(x_a;c),\qquad
y_S=\Phi_S^{a\to b}(x_a;c).
\]

应匹配 y_S 与 y_T。不能把已经投影后的 y_T 当作未经投影的 Student 输出目标，再额外投影一次；那会改变训练目标。也不应在包含离散投影的更新上直接套用无约束 ODE 的 MeanFlow 一致性保证。

建议先采用显式教师区间目标建立基线，不要求第一版就实现 JVP、自一致性、DMD fake-score 或完整轨迹反传。训练区间目标确定后，平均速度/瞬时速度一致性再作为独立增量评估。

**基础损失按区域有效面积归一化。**

用 F 表示训练可用的真实主体区域，I 表示选定的交互带，B 表示远背景。它们用于损失划分；部署的放置区域及 anchoring 权重必须来自推理可获得的输入，不能用 GT 标签替代。

\[
L_{\mathrm{segment}}
=\sum_{q\in\{F,I,B\}}
\lambda_q
\frac{\sum_p M_q(p)\|e(p)\|^2}
     {C\sum_p M_q(p)+\epsilon},
\qquad
e=\frac{y_S-\operatorname{sg}(y_T)}{\max(a-b,h_{\min})}.
\]

C 为 latent 通道数。归一化区间长度是使不同跨度的损失量级更可比的一个设计选择；不除以区间长度的 endpoint MSE 需要保留为对照。空区域要显式跳过/使用预定义回退，不能产生除零。

若硬背景投影令某些位置 Teacher/Student 更新完全相同，这些位置的零误差不能证明模型本身学到了背景保持。应记录投影前误差、投影后误差与实际像素输出。

**插入任务增量一是保留参考变化引起的区间响应。**

固定场景、mask、起点状态与区间，使用同类别、不同身份且适合该场景的参考 R_a、R_b：

\[
L_{\mathrm{response}}
=\mathcal E_{F\cup I}\left(
[\Phi_S(x;R_a)-\Phi_S(x;R_b)]
-\operatorname{sg}[\Phi_T(x;R_a)-\Phi_T(x;R_b)]
\right).
\]

它检查快速 Student 是否保留融合 Teacher 区分参考身份和细节的能力。与先前 velocity 响应方案相比，这里监督少步区间造成的实际变化。必须保留绝对区间目标，并与同样前向预算的双条件独立 MSE 比较。两个绝对输出都拟合完美时，差分约束自然成立；其价值在于有限预算下的误差分配，不能宣称产生额外真值。

起点可由随机选中的参考条件产生，另一参考在同状态上的预测只是辅助条件干预，不是另一条独立 on-policy rollout。融合模型若未训练过空参考，不默认用 null reference 作强教师；优先使用合法参考对，并检查教师响应。不得把同一个 GT 图像同时作为两个不同实例参考的正确输出。

**插入任务增量二是让背景软约束与采样跨度相容。**

当前 _apply_post_step_latent_anchoring 对每一步执行相同空间权重的 lerp。对软过渡区位置，在忽略新生成增量、仅分析已有偏差传播时，单步偏差乘以 w，经过 m 步变为 w^m。因此压缩步数也改变了累计背景收缩强度。

例如一个 Student 大步代替教师的五个细步时，对背景约束自身的齐次偏差分量，候选粗步权重为 w_coarse=w_fine^5。一般可以匹配参考细网格下权重的乘积，或按累计噪声进度定义衰减强度。核心自由区仍保持 w=1，硬背景区保持 w=0。

这只是背景约束自身的累计作用，不代表包含非线性生成增量的完整 Teacher/Student 更新已经等价。实际生成差异仍由区间监督学习。应比较固定逐步 W、按跨度校准 W，以及无软投影的控制实验，重点检查边缘、阴影和细结构。该分析是从当前代码推导出的研究候选；动态源图约束、软 mask 等相关先例仍需作为对照。

源码依据：[背景 anchoring](../src/trainer/qwen_image_edit_trainer.py)、[latent 编辑权重](../src/utils/latent_anchoring.py)。仅证明 w^m 关系不足以形成完整新方法，需要证明它确实解释了少步插入中的可观测退化，并能稳定修复。

**图像空间监督应落在可解释的清晰输出上。**

最小增量是在 b=0 的区间末端，把 Student 和同状态教师 continuation 的最终输出解码，比较前景/细节 patch LPIPS、边界与远背景误差。参考身份用特征损失辅助；参考图与目标姿态不同时不做直接像素 L1。

中间含噪 latent 不能直接当清晰图像计算身份或 LPIPS。若要在中间状态使用图像损失，必须通过明确定义的 clean prediction 或 continuation 得到可比较的输出，并单独计算额外成本。完整图像感知图只在部分选定 batch 启用，不改变最终评价范围。

第一版目标只用 L_segment；随后分别验证 detail 与 response：

\[
L=L_{\mathrm{segment}}
+\lambda_d L_{\mathrm{detail}}
+\lambda_r L_{\mathrm{response}}.
\]

损失权重和训练预算需要通过验证确定，本文不预设最佳值。细节项不负责把噪声状态强行重建为某一张 GT，也不保证修复融合教师本来就存在的错误。

**当前代码已有基础，但需要新增区间目标与部署路径。**

| 位置 | 所需工作 |
| --- | --- |
| [src/models/transformer_qwenimage.py](../src/models/transformer_qwenimage.py) | 主模型增加可选的目标时间/区间条件；Teacher legacy forward 保持原契约 |
| [src/distillation/opd.py](../src/distillation/opd.py) | 新策略负责少步 Student 前缀采样、抽区间、Teacher 子步 continuation；现有 transition_mean 只比较单个 Euler 更新，不能替代这些子步 |
| [src/distillation/backends.py](../src/distillation/backends.py) | 一个冻结融合 Teacher 与一个 Student；共享基座时新时间模块仅在 Student 路径启用，并隔离 geometry companion |
| [src/distillation/objectives.py](../src/distillation/objectives.py) | 区域归一化、跨度归一化及可选区间响应目标 |
| [src/data/config.py](../src/data/config.py) | few-step 模式、Student 网格、Teacher 细网格/子步、时间条件、损失与 anchoring 策略 |
| [infer.py](../infer.py) 与验证采样 | 使用训练对应的少步区间 sampler，统一 crop、原图恢复和输出含义 |
| checkpoint 与 metadata | 除 LoRA/geometry 外保存新增时间模块及 sampler 契约，并校验 Teacher/Student 的不同角色 |

Teacher 与 Student 不共享可变 scheduler.step_index 状态。Teacher 子步应使用独立实例或预先验证的无状态更新；时间输入与 sigma 更新必须一致。需要的输入尺寸、source/ref 条件、几何 tokens 与 mask 随每个 batch 固定，不能让 Teacher 监督另一种任务。

**实验需要同时区分融合能力、少步能力和新增模块。**

| 对照 | 作用 |
| --- | --- |
| T_mix 原始高质量步数 | 冻结的质量参照 |
| 同一 T_mix 直接 8/4 步 | 量化仅减步数的退化 |
| 固定网格的传统区间蒸馏 | 已知多步到一步基线 |
| 同预算的 Student 状态区间蒸馏 | 验证 on-policy 状态是否有用 |
| 加区域归一化 | 排除纯面积权重可解释的收益 |
| 加 detail / response 的独立与组合实验 | 定位身份与细节增益来源 |
| 固定 W / 跨度校准 W / 无软投影 | 检验累计背景约束的解释 |
| 两时间模型与一致性增量 | 检验可变步数与 flow-map 能力的代价 |

全部领域保持同一固定测试集，并按目标面积、边界复杂度及领域分层。使用同输入与 seed 的配对比较，至少区分参考身份、局部结构、阴影/遮挡、原始背景误差、anchored 输出和 strict composite。

固定图像分辨率与硬件测量实际 wall time，报告 Student NFE、CFG 分支、条件编码及 VAE 开销。40→8、40→4 只给出 5 倍、10 倍的去噪更新次数比，不是端到端加速承诺。若教师已吸收 CFG，其参考基线也应是单分支，不重复计算已在上一阶段获得的加速。

新工作可聚焦于：融合后的参考插入能力在少步区间更新中怎样退化，怎样通过参考响应监督与跨度相容的编辑约束保留。平均速度、区间回归、on-policy rollout 本身已有先例；任务专属贡献需要以上对照支持。

**与 FlowMap-OPD 的具体区别及研究边界。**

FlowMap-OPD 主实验使用 MeanFlow；文生图基座为已有 MeanFlow checkpoint 的 FLUX.1-lite-8B，专属教师通过 Flow-Map GRPO 优化，Student 使用五步原生 flow-map rollout。框架另外覆盖 CM。其主要训练路线不是 AnyFlow 的 forward-training 加 DMD backward-simulation；更换成 AnyFlow 也只是选择另一已有技术，不能自动解决新颖性问题。

| 设计变化 | 与已有工作的关系 | 论文定位 |
| --- | --- | --- |
| 已融合的单教师代替多个专家 | 改变教师集合，属于通用 OPD 可容纳的设置 | 实用流程选择 |
| 用 Qwen/参考图/mask/插入数据 | 更换基座和条件 | 任务应用与系统实现 |
| 两时间 Student、原生少步 rollout | MeanFlow 与 FlowMap-OPD 已有 | 方法基线 |
| 教师多个子步得到终点，再回归 Student | FlowMap-OPD 式 22 直接覆盖 | 必须比较的基线 |
| 主体区域加权、LPIPS/DINO | 已有区域/感知监督可自然扩展 | 强效果对照，单独新颖性有限 |
| 参考干预下的区间响应保持 | 引入同状态、不同参考之间的关系监督 | 有待验证的任务专属候选 |
| 不同步数预算下的编辑约束一致性 | 对显式背景投影与区间生成的组合过程建模 | 有待验证的动力学候选 |

另一个直接邻近工作是 [MaskFlow 官方实现](https://github.com/ReyChiaro/MaskFlow)：该项目已列出 mask 编辑模型的 DMD 8/16 步版本。因此“mask 编辑加少步蒸馏”也不能单独作为新贡献。[CoDi](https://arxiv.org/abs/2310.01407) 已研究图像条件任务的少步蒸馏。同状态条件差分也有相关先例，例如 [CADT](https://arxiv.org/abs/2610.11238) 用于扩散数据归因；它不是这里的 Student 蒸馏目标，但限制了对“首次反事实响应”的宽泛表述。

更严谨的主张应是：在给定算力与极少步预算下，哪些参考插入能力未被通用 flow-map 回归保留，以及新的结构约束如何改善这一缺口。不得宣称通用 FlowMap-OPD 无法接受图像/mask 条件，也不能把该框架中没有专门研究插入任务等同于证明其在插入上必然失败。

最低限度的区别验证包括：同一 T_mix 下的 FlowMap-OPD 速度加一致性方案、式 22 的区间映射回归、相同区域加权和相同 anchoring；在这些强基线上，再分别加入参考响应、跨度校准及二者组合。参考响应要与相同计算量的两个条件独立匹配比较；跨度校准要与仅改推理权重、不改训练比较。若收益仅来自后处理或额外前向，应如实归因，不能将其包装成新的蒸馏机制。

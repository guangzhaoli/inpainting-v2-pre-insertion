# InsertFuse Teacher 与 OPD 改进调研

调研日期：2026 年 10 月 15 日。讨论对象是当前 Teacher v2 与共享 LoRA 多教师 OPD 实现。目标是提高插入自然度、参考保真度和跨类别一致性，同时形成可检验的任务专属改进。文献中的机制与以下实验提案分别表述；提案尚未经过本项目训练验证。

最值得优先验证的两项是：Teacher 端的可见部位细节对应监督，以及 Student 端的参考响应蒸馏。部署背景约束与 OPD rollout 的一致性应先成为明确的实验基线。干净背景与物体交互区域的数据建设更可能提高真实融合质量，但成本和数据要求较高。

第一轮调研使用 Exa，沿参考插入、可控细节与场景交互、OPD 系列、相关蒸馏及创新重合检查四条线进行 20 次检索，共返回 104 条结果，按完整 URL 去重为 95 个页面入口。正文采用论文、作者项目页、官方代码和本地实现；不同版本、会议页与项目页不计作不同方法。没有把不同论文在不同数据集上的指标直接横向排名。文末追加了针对 OPD 与少步生成的专项检索。

**当前方法已经具备哪些能力。** 新增贡献必须以实际代码为参照。

| 已有能力 | 当前实现与含义 | 对新 idea 的约束 |
| --- | --- | --- |
| 参考与场景双路条件 | Qwen-VL 语义条件及 VAE 图像 tokens | 再加一个泛化的参考编码器不一定解决细节对应 |
| 几何条件与 MaskMix | mask、距离场、边界带；监督 mask 与挖洞 mask 分离 | mask 增强或显式位置控制已是基线 |
| Teacher v2 局部损失 | 前景 LPIPS、四个局部 GT patch、边界与远背景 L1、参考 DINO | “增加局部损失”“增加 DINO”不能再作为新增模块 |
| 参考 DINO | 使用 pooler_output 或 CLS token，对主体裁剪做整体余弦相似度 | 尚未直接监督参考与生成图的局部部位对应 |
| 推理背景约束 | expanded_soft 在每一步更新后混合原场景 latent 轨迹 | 背景约束本身已有实现 |
| 当前 OPD | 类别硬路由、Student ODE rollout、同状态 Teacher velocity、逐时间步反传 | 多专家加 MSE、冻结教师、CFG 吸收均已存在 |
| factorized-reference 分支 | Qwen-VL 语义与 VAE 外观条件分开 dropout/引导 | 语义与外观分解是已有探索 |
| CRIB 分支 | 正负参考的前景能量 margin 与远背景响应约束 | 负参考训练和参考局部性约束也是已有探索 |
| layout/query/depth 分支及 cycle 文档 | 预摆放、关系 queries、深度条件、插入移除循环 | 不应把这些方向重新包装为全新 idea |

本地依据：[Teacher v2](./teacher_v2.md)、[参考整体 DINO](../src/loss/region_perceptual_loss.py)、[局部 GT 细节监督](../src/loss/local_detail.py)、[OPD](../src/distillation/opd.py)、[CRIB 损失](../../inpainting-v2-crib/src/loss/crib_loss.py)、[factorized-reference 配置](../../inpainting-v2-factorized-ref/configs/exp/factorized_reference_cfg.yaml)、[已有 cycle 方案](../ideas/idea_03_insertion_removal_cycle.md)。

检索到的同名 [InsertFuse 公开稿](https://arxiv.org/html/2608.06490) 已表述 IOPD、TAGC、Region-Balanced Flow Matching 和 Reference CFG，因此不将其作为独立的外部改进证据。该稿描述 rank 256、参考 dropout 0.2、目标与 context 两路几何增强；当前 v2 配置是 rank 128、dropout 0.1，当前 transformer 的几何残差注入目标 token 前缀。公开稿与本地版本存在差异，讨论新贡献与复现实验时应固定具体版本。

**Teacher 相关工作的共同问题是：主体身份、局部细节、目标姿态和场景交互需要不同的监督。**

| 工作 | 实际做法 | 对 InsertFuse 的启发与限制 |
| --- | --- | --- |
| [AnyDoor，CVPR 2024](https://arxiv.org/html/2307.09481) | DINOv2 身份 tokens 配合高频细节图，并利用视频中的同一对象不同帧 | 身份特征不能代替细节通路；高频分支本身已有充分先例 |
| [IMPRINT，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Song_IMPRINT_Generative_Object_Compositing_by_Learning_Identity-Preserving_Representation_CVPR_2024_paper.pdf) | 先学习视角相对不变的身份表示，再学习合成与背景协调 | 将姿态适应和身份保持的学习过程分开；极端视角与小字仍是其局限 |
| [MimicBrush，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/98b2b307aa4aa323df2ba3a83460f25e-Paper-Conference.pdf) | 从视频采样两帧，遮住一帧区域，通过另一帧恢复；参考网络提供特征 | 用真实跨视角对应训练部位迁移，比同图复制更能检验参考利用 |
| [StableVITON，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_StableVITON_Learning_Semantic_Correspondence_with_Latent_Diffusion_Model_for_Virtual_CVPR_2024_paper.html) | zero cross-attention 学服装与人体的语义对应，attention TV 约束改善细节 | 服装能力来自对应关系；直接把同类约束移植到所有类别需要验证 |
| [Insert Anything，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37866) | DiT 中的 in-context 参考编辑，分别组织 mask 与文本编辑输入 | 是直接任务基线；统一参考输入并不是新增贡献 |
| [DreamO，2025](https://arxiv.org/html/2504.16915v5) | 约束条件图与生成区域的特征路由，并使用 placeholder 关联条件 | 可借鉴参考特征的空间绑定；其整物体路由不等于细粒度部位对应 |
| [HiFi-Inpaint，CVPR 2026](https://arxiv.org/html/2603.02210v3) | 高频图、共享参数的增强 attention 分支、GT 高频像素损失 | 与首饰和商品细节非常相关；只增加高频 loss 或共享增强分支与其高度重合 |
| [ObjectDrop，ECCV 2024](https://arxiv.org/html/2403.18818) | 真实移除前后场景对训练移除模型，再扩展出插入训练对 | 学阴影与反射需要物体出现前后的场景变化；仅挖洞重建可能保留答案线索 |
| [PS-Diffusion，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_PS-Diffusion_Photorealistic_Subject-Driven_Image_Editing_with_Disentangled_Control_and_Attention_CVPR_2025_paper.pdf) | 区分主体上的光照效应与背景中的阴影/反射；预测背景 effect 区域，并用 effect attention 解耦 | 固定 dilation ring 不是完整的物理交互区域；独立 effect 建模也有明确先例 |
| [OSInsert，2026 预印本](https://arxiv.org/html/2602.19523) | ObjectStitch 先生成合理形状姿态，SAM 提取轮廓，再由 InsertAnything 补外观 | 验证了姿态与细节的张力；两阶段生成成本较高，且与你已有预摆放方向相邻 |
| [SAM-Flow，2026 预印本](https://arxiv.org/html/2606.06228) | 时间变化的 core/transition/background mask 与 source-anchored flow 更新 | 动态软 mask 和原图轨迹约束不能单独主张新颖性 |
| [MaskFlow，2026 预印本](https://arxiv.org/html/2608.06929) | mask 进入概率路径与 flow 目标，配合 Soft-Poisson De-seaming | 训练路径、推理背景约束与边界融合需要一致考虑；其方法不是多专家插入 OPD |

与结构迁移有关的补充先例是 [Attention Distillation，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhou_Attention_Distillation_A_Unified_Approach_to_Visual_Characteristics_Transfer_CVPR_2025_paper.pdf)。它使用 attention 特征目标优化合成 latent，服务外观和风格迁移；这与训练一个多专家统一 Student 的 OPD 目标不同，但意味着“蒸馏 attention”本身不是新方向。

**Teacher 提案 T1：只对参考中可见、可靠对应的局部细节施加强约束。**

动机是当前整体 DINO 可以识别“像这件衣服/这只猫”，却不能直接回答“胸前文字、袖口条纹、吊坠星形是否来自同一参考部位”。现有 GT 局部 LPIPS 监督的是恢复正确细节，也没有直接规定参考细节如何传到目标位置。

最小实现先复用冻结 DINO 的 patch 特征，在训练的 GT 主体与参考主体之间构造软对应 A。用互为近邻、一致性和 mask 可见性筛掉不可靠匹配，为每个目标 patch 保留置信度 c_i。对生成图对应位置施加：

\[
L_{\mathrm{corr}}
=\frac{\sum_i c_i\sum_j A_{ij}
  \bigl(1-\cos(F_{\mathrm{pred},i},F_{\mathrm{ref},j})\bigr)}
 {\sum_i c_i+\epsilon}.
\]

A、c 与参考特征停止梯度；梯度仅经过生成图特征回到模型。特征网络冻结不等于关闭生成图输入梯度。不能直接把姿态不同的参考与生成图做逐像素 L1，也不能把目标背面强行匹配到参考正面。没有可靠对应的部位继续依靠 GT 重建和生成先验。

第二步才考虑把对应关系用于少数 DiT 层的参考特征路由：让目标核心区域查询对应参考 patch，让场景交互区域保留更多 context 注意力。与 DreamO 的区域绑定相比，需要证明增益确实来自部位对应和可见性，而非单纯更强 mask attention。高频参考 tokens 可作为对照或后续增量；其来源已经在 AnyDoor、HiFi-Inpaint 中出现，不宜单独命名为主贡献。

必要对照是当前 v2、仅高频 GT loss、仅粗粒度区域路由、无置信度 patch 匹配、带置信度 patch 匹配。特别检查重复花纹、金属反光、毛发和大视角变化：冻结特征的近邻对应可能错误，强制对齐反而会损害结构。

首个实验优先用首饰与服装，因为局部图案、边缘和重复结构容易做定向判断。先离线检查匹配可用率，再决定是否值得增加训练图。此方向是参考细节保真方面的优先研究候选，尚不能仅凭组件组合声称首次。

**Teacher 提案 T2：用干净场景对教会模型生成物体带来的交互变化。**

当前场景条件来自 tar_image 乘以挖洞 mask 的补集。若真实阴影、反射延伸到洞外，它们会保留在输入里；这个实现事实提示一种数据捷径，但本轮未量化它在数据中出现的比例。增加 dilation 只能覆盖一部分情况。

可先引入经筛选的 clean source 与完整 target 配对：真实拍摄、可靠视频静态片段、或经人工抽检的移除模型结果。目标同时包括主体、物体产生的接触阴影和反射，以及远背景不变。需要新增独立 source_image 字段，不能再用目标图的遮挡版本冒充物体不存在的场景。

利用对齐的 source/target 差异构造软交互区域，排除主体和无关相机变化。训练一个从场景、参考与位置条件预测交互支持区域的轻量头，供生成门控及部署背景约束使用。推理不能使用 target-source 差异或真实 target mask；它们只作为训练标签。

应分开比较三件事：只换反事实配对数据、配对数据加固定交互 ring、配对数据加学习的交互区域。否则无法判断提升来自数据还是结构。ObjectDrop 已做反事实数据、PS-Diffusion 已做 effect 分支，新增价值必须体现在你的多类别场景、相应鲁棒性和后续单 Student 迁移上。

不建议一开始把整个 insertion-removal cycle 加入在线训练。你已有该方向文档；先把移除模型用作离线数据生产器和筛选器，更容易隔离因果。只验证 cycle 重建误差也不足以证明物理合理性。

**Teacher 提案 T3：让领域专属目标真正对应不同错误类型。**

| 领域 | 可选增量 | 不宜采用的替代指标 |
| --- | --- | --- |
| 首饰/细小配件 | 在对齐 GT 上监督高分辨率边缘、细链连通性；结合 T1 的可见部位匹配 | 整物体 DINO 高就推断链条/文字准确 |
| 服装 | 局部花纹对应；仅在文字可见且有可靠标签时加入 OCR 一致性 | 对所有花纹使用 OCR 或强制复制参考姿态 |
| 人物 | 在有效对齐人脸上加入专门的人脸身份特征，并持续报告检测覆盖率 | 将现有 face-region DINO 当作人脸识别模型 |
| 宠物 | 同一实例跨帧的局部花纹、面部和轮廓对应 | 把“同品种”当作“同一只” |
| 通用物体 | 区分刚性几何适应与材质细节，优先收集视角变化的同实例对 | 将未知视角变化强行作为像素对应 |

这类改进偏效果工程，适合作为强 Teacher 的必要组成，不宜把五套独立损失堆叠作为统一论文主线。身份数据分组必须可靠；当前 manifest 中有 category、stem 等字段，但没有显式 identity_id，不能直接按同类别推断同身份。

**OPD 系列的差别主要在教师来源、访问的状态、监督密度和学习目标。**

| 工作 | 主要机制 | 对本任务的直接含义 |
| --- | --- | --- |
| [DiffusionOPD，2026](https://arxiv.org/html/2605.15055v2) | 专属教师、Student 轨迹上的逐步监督；相同协方差 SDE 的 KL 化为均值匹配；ODE 使用确定性转移匹配 surrogate | 是当前多教师 OPD 的直接基线；ODE MSE 不应写成两个确定性分布之间有限的精确 KL |
| [Flow-OPD，2026](https://arxiv.org/html/2605.08063) | 专属教师、cold start、路由蒸馏及任务无关的 Manifold Anchor Regularization | 可比较学生初始化与保底能力，但 full-frame anchor 对局部编辑可能压制合理变化 |
| [Qwen-Image-2.0-RL，2026](https://arxiv.org/html/2606.27608v1) | 将 T2I/editing 的 RL 专属教师用 trajectory velocity matching 统一；教师使用 CFG，学生不使用 | CFG 吸收和多任务模型统一已是明确先例；当前任务不必先引入 RL 才能用 OPD |
| [DanceOPD，2026](https://arxiv.org/html/2606.27377v2) | 每样本硬路由，查询 Student 轨迹上一个低噪声状态，用 velocity MSE；分析稠密查询相关性 | 单低噪声查询是必须比较的效率基线，不能重新包装为新增方法 |
| [STEP-OPD，2026](https://arxiv.org/html/2608.04887) | 沿教师相对基座的改变量外推目标，同时对齐相邻网络块的表示变化方向和幅度 | 中间特征蒸馏已有先例；外推过强会破坏保真，不能假设系数越大越好 |
| [D-OPSD，2026](https://arxiv.org/html/2605.05204v2) | 利用图文上下文构造更强的自教师，在 Student 状态上进行持续微调，保留已有少步能力 | 可启发训练期特权上下文，但教师必须能有效利用上下文，不能把 GT 任意塞入就视为强教师 |
| [DiffusionOPSD，2026](https://arxiv.org/abs/2608.24646) | 将 reward 梯度变成有界的正负 clean-output 目标，有限拟合并用 EMA 更新行为策略 | 后期可探索目标图像质量修正；奖励目标改善不必然在有限参数更新后实现 |
| [Self-OPD，2026](https://arxiv.org/html/2608.26872) | 从同一状态分叉多个随机候选，完整 rollout 后评价，按优势 pull/push，不依赖外部专属教师 | 可以超出固定教师监督，但增加采样成本且依赖任务奖励，不是现阶段最低成本的改法 |
| [MT-OPSD，2026](https://arxiv.org/abs/2609.35611) | 在自身生成的源图条件上训练，由 clean-conditioned 教师指导多轮编辑 | 处理的是轮与轮之间的条件分布偏移；与你单次去噪内的 on-policy 状态不是同一层次 |
| [DMD2，NeurIPS 2024](https://arxiv.org/html/2405.14867) | 分布匹配、fake critic 的双时间尺度训练、GAN 目标及 backward simulation | 更偏少步生成压缩；增加 critic 的工程代价较高，不应将少步压缩与多专家能力合并混为同一问题 |

这些工作中“减轻多任务干扰”的实验结果不意味着 Student 的跨任务梯度冲突在数学上消失。对 InsertFuse 仍应按领域检查能力保留，并可沿用已有梯度冲突分析工具。现有 10-step rollout 是训练采样设置，不构成 10-step 推理保真度的保证。

**OPD 首先需要明确三个基线问题。**

第一，Teacher v2 与旧 OPD 模板的 rank、target_modules、processor、reference mask-aware/clip 设置需要一致。否则比较的是条件和实现差异，不能解释为蒸馏算法优劣。参考引导 scale 也应记录教师实际使用值。

第二，Teacher 的 region-balanced 目标不会自动传给 Student。当前策略调用 VelocityMatchingObjective 时没有传 token_mask，默认是全 latent 均值。主体仅占小区域时，共同背景误差可能掩盖主体误差；这需要测量，不能仅由面积推断实际梯度占比。先增加按区域分开归一化的蒸馏对照，它是必要基线，创新性有限。

第三，当前 OPD 的 rollout 是直接 Euler 更新，没有调用推理中的 latent anchoring。若目标部署继续使用 expanded_soft，训练访问的状态和部署状态就不是同一条更新过程；若最终 Student 部署选择不使用 anchoring，则必须另立无 anchoring 的评价路线。两种目标不应混在一组结果里。

本地位置：[rollout 与时间步选择](../src/distillation/opd.py)、[损失归约](../src/distillation/objectives.py)、[推理背景投影](../src/trainer/qwen_image_edit_trainer.py)、[旧五教师配置](../configs/opd/final_architecture_five_teacher.yaml.example)。

**OPD 提案 O1：蒸馏更换参考图引起的响应。**

普通 OPD 匹配给定参考 R 的完整场。新辅助目标关注同一场景、同一 mask、同一状态下，R_a 与 R_b 改变时模型输出的差值：

\[
\Delta v_S=v_S(x_t,t;B,M,R_a)-v_S(x_t,t;B,M,R_b),
\]
\[
\Delta v_T=v_T^{\mathrm{guided}}(x_t,t;B,M,R_a)
            -v_T^{\mathrm{guided}}(x_t,t;B,M,R_b),
\]
\[
L_{\mathrm{response}}
=\mathcal E_{\mathrm{subject}}\!
  \left(\Delta v_S-\operatorname{sg}(\Delta v_T)\right).
\]

其中 \(\mathcal E\) 是按区域有效面积归一化的平方误差。保留普通 OPD 主目标，因为差值匹配不能确定两条分支共有的场。目标是保留教师响应的方向和大小，不能简单最大化 Student 对参考的敏感度。

若两个参考条件下的绝对输出都已完全拟合，差值约束自然满足；它不创造额外真值。这里的研究假设是：在有限容量、有限预算下，显式耦合两个条件的误差能减少参考响应遗失。必须与相同计算量的双条件独立 MSE 对照，才能排除只是增加训练样本或前向次数的解释。

最便宜的版本把 R_b 设为空参考，复用教师 Reference CFG 已计算的正/空预测，只增加 Student 空参考预测。更有区分力的版本使用同类别、不同实例的近似参考，例如两条形状近似但挂饰不同的项链。教师的空参考分支可复用，但真实换参考仍增加条件编码和前向成本。

锚点可以从随机选中的 R_a 或 R_b 条件下的 Student rollout 产生，轮换覆盖两者分布；另一参考在同状态上的预测是辅助条件干预，不是它自己的独立 on-policy 轨迹。这样的区分应在论文和实现中明确。

它与已有 CRIB 的区别是：CRIB 要求正确参考更接近 GT、错误参考更远，并约束远背景变化；这里拟合教师在参考变化下的具体向量场响应，使该能力迁移到统一 Student。它也与 [Delta Denoising Score](https://delta-denoising-score.github.io/) 有共同的差分思想，但 DDS 是用参考 score 抵消 SDS 优化偏差，并非多教师 Student 的条件响应匹配。两者和 Reference CFG 都应列为相关先例。

验证的核心是固定场景、mask、seed，成对替换同类别参考：输出身份是否跟着改变，目标细节是否匹配正确参考，远背景是否保持稳定。还需要对比“直接把 CRIB 加到 OPD”“正/空分支各自独立 MSE”和“响应差值 MSE”，排除只是多一次前向或多一个监督目标的收益。

这一方向适合做任务专属的主贡献候选，但需要进一步检查有限差分/关系蒸馏的邻近文献与实验，不能宣称已证明首次。教师若本身忽略参考，响应蒸馏不会凭空创造参考能力；应先筛查教师响应。

**OPD 提案 O2：在实际编辑更新过程上做 on-policy 训练。**

令 W 为部署实际使用的 packed latent 编辑权重，z_src,next 为下一噪声水平的原场景轨迹。定义：

\[
P_{t+1}(z)=W\odot z+(1-W)\odot z_{\mathrm{src},t+1},
\qquad
U_\theta(x_t)=P_{t+1}(x_t+\Delta\sigma_t v_\theta(x_t)).
\]

Student rollout 使用 U_S；Teacher 和 Student 在相同 x_t、相同 source 轨迹、相同 W 下比较更新结果。可分别统计主体和交互区域的转移误差，对远背景保留独立的原图一致性诊断。Teacher 使用已确定的参考引导，Student 使用单分支。

这里有一个必须承认的退化关系：若 W 对 Teacher/Student 完全相同，且 P 是固定线性投影，那么

\[
U_S-U_T=W\odot\Delta\sigma_t(v_S-v_T).
\]

因此，仅把投影加进损失公式，会退化为带 W 平方及步长平方权重的 velocity MSE，不能作为独立的新理论贡献。实质变化在于 Student 访问的状态、可学习的交互支持以及与 O1 参考响应的联合保持。若为了消除步长权重而除以绝对步长，也应在目标定义中明确，不要混称两个目标等价。

W=0 的远背景经过同一投影后没有 Teacher/Student 转移差异；不能用这个零误差证明模型自己学会了背景保持。若希望去掉部署 anchoring，必须另加能训练背景的目标并进行无投影评测。类似地，不能把严格像素合成后的零背景误差当作生成模型改进。

无需一开始学习动态 W。先使用现有静态 expanded_soft，分别比较仅改损失、仅改 rollout、两者同时改变，并用 SAM-Flow、MaskFlow 作为编辑路径的相关工作。后续若 T2 得到可靠交互区域，才研究该区域如何帮助蒸馏。

**OPD 提案 O3：依据区域需要分配监督时间步，并检查参考对应的内部传递。**

DanceOPD 的单低噪声查询应先做成对照。对本任务可以进一步检验：姿态、摆放和遮挡是否更依赖中高噪声状态，而文字、毛发和首饰细节是否更依赖低噪声状态。这个假设不能用固定“早期一定管结构”作为事实，应该按 mask 面积、边界复杂度、参考纹理及任务类别分层测量。

最小实验比较全轨迹、单低噪声查询、单中噪声查询、固定预算下中/低噪声混合查询。先保证教师查询数或总 GPU 时间可比，再尝试根据任务特征决定查询分布。改变查询分布就是改变优化目标；若要声称估计原目标，需要重要性校正，否则应明确采用新的能力侧重。

另一个增量是只在几个选定 block 上，蒸馏可见主体位置到参考 patch 的对应关系，或这些位置的隐藏表示变化。STEP-OPD 已有表示变化对齐，DreamO 已有参考区域绑定；新增价值应是与 T1 细粒度、可见性约束的连接，不能只复现全 token hidden MSE。不要保存全层、全分辨率 attention 矩阵；选层、选主体 queries 或压缩统计后再测峰值显存。

**更远期的 OPD 候选。** 可以考虑强条件 Teacher 指导弱条件 Student：Teacher 使用可靠的训练期姿态/可见性辅助标签，Student 只使用部署可获得的参考、场景和粗 mask。这受 D-OPSD 的上下文自教师启发，但必须保证两者对应同一个期望输出，且 Teacher 真能用好额外条件。随意让 Teacher 看 GT 图像或精确 GT mask，可能导致监督场与部署任务不相容。

Teacher 质量门控也可减少错误传递，但单一 DINO 分数容易奖励不自然的复制。至少应联合参考匹配、位置/遮挡正确性、背景保留和自然度，并抽检。当前 _batch_mean 对 teacher_weight 归一化，在共享后端的 batch_size=1 下，单个正权重会被约掉；真正的置信度缩放应作用在单样本 loss 上或改变采样，不能只修改现有 teacher.weight。各 rank 仍须保持一致的前向次数，不能局部跳过整个教师分支。

直接套用 STEP-OPD 的大幅外推、或马上改为 Self-OPD/多奖励 RL，不列为首批。它们可能改善某些分数，但对细节精确保持与背景稳定的取舍需要另外建立可靠奖励。专属 Teacher 还没有达到所需质量时，应先区分教师能力不足和蒸馏遗失。

**实验顺序应让每个结论有独立对照。**

| 阶段 | 实验 | 需要回答的问题 |
| --- | --- | --- |
| D0 | 固定原 v2 Teacher 与普通 OPD，审计条件一致性 | 比较是否由版本或推理配置混用造成 |
| D1 | 固定 scene/mask/seed，正确参考、近似负参考、空参考 | Teacher/Student 是否真的使用参考，能力在哪些区域丢失 |
| D2 | 目标图经过冻结 VAE 的编码解码重建 | 微小文字和链条细节是否在生成训练之前已明显损失 |
| T0 | 原 Teacher v2 | 所有 Teacher 增量的共同基线 |
| T1a | 仅高频 GT loss 或粗粒度参考路由 | 简单已知方法能否解决大部分问题 |
| T1b | 可见性置信度局部对应监督 | 是否优于整体 DINO 和不筛选匹配 |
| T2a | 仅加入筛选过的干净背景配对 | 自然度提升是否主要来自训练信息变化 |
| T2b | 同一配对数据加学习的交互区域 | 学习区域是否优于固定 ring |
| O0 | 用同一组强 Teacher 跑普通 OPD | 将教师提升与蒸馏提升分开 |
| O1 | 按区域归一化、DanceOPD 单查询对照 | 排除简单权重和查询预算的解释 |
| O2 | 参考响应蒸馏 | 是否保住局部身份响应，尤其小物体 |
| O3 | 部署轨迹一致的 rollout 与更新监督 | 改善是否来自状态分布一致性 |
| O4 | O2/O3 的二乘二组合，再试对应表示蒸馏 | 两个组件是否互补，是否值得附加内部监督 |

先用首饰和服装做诊断与筛选，再在五个领域、固定完整测试集上确认结论。小诊断集不能替代最终全量比较。各 ablation 固定 Teacher checkpoint、数据、基座、LoRA 容量、分辨率、随机种子与推理设置；涉及新参数或新数据的实验必须单独给出对应控制。

至少同时报告两类训练预算：相同数据/更新次数下的质量，以及相同总 GPU 时间或教师查询预算下的质量。单查询省下的算力可以覆盖更多样本，但不能把这种预算变化隐藏成算法质量提升。最终推理先固定 40 步与单分支/CFG 设置；少步部署质量另列曲线。

评测需要覆盖以下四个方面：

1. 参考保真：masked DINO/CLIP 作为辅助；同类别近似参考集合上的身份检索、局部对应误差和人工身份选择更能区分参考利用。尽可能使用与训练损失不同的评价特征。
2. 细节：配对样本上的局部 LPIPS、高频/边缘误差；有真实可读标签时的文字一致性；首饰的断裂和连通错误。不能在不对齐的跨视角参考上解释像素 PSNR。
3. 融合：接触阴影、反射方向、遮挡关系和透视的分别评分。干净背景配对提供额外客观约束，盲评需将保真和自然度分开。
4. 背景与稳定性：对原始生成结果、anchored 输出、strict composite 分开记录；远背景 MAE/LPIPS、边界连续性、换 seed 与 mask 扰动敏感度。

按物体身份或原视频划分训练和测试，避免同一对象相邻帧泄漏。若使用多 seed，按身份/场景汇总后做配对置信区间；不能将同一对象的多个 crop 当作独立样本提高显著性。

**优先级与预期研究价值。**

| 项目 | 预期作用 | 增量成本 | 研究定位 |
| --- | --- | --- | --- |
| v2/OPD 条件对齐、区域归一化、单低噪声基线 | 保证比较可信，改善训练利用率 | 低至中 | 必须做的基线 |
| T1 可见局部对应 | 参考细节、身份与姿态兼容 | 中 | 优先 Teacher 候选 |
| O1 参考响应蒸馏 | 防止统一模型只学到类别平均外观 | 中，增加参考条件前向 | 优先 OPD 候选 |
| O2 部署编辑轨迹蒸馏 | 训练状态与实际插入更新一致 | 中 | 应与响应/交互支持结合验证 |
| T2 反事实交互数据与区域 | 阴影、接触、反射和背景合理性 | 高，数据成本为主 | 后续高价值路线 |
| 内部对应/表示蒸馏 | 保留教师细节传递过程 | 中至高 | 主线有效后再加 |
| 直接 RL、自探索、大幅教师外推 | 超出固定教师能力 | 高且依赖奖励 | 后续独立方向 |

推荐形成一条连贯的假设：领域 Teacher 学会参考中的可见细节如何对应到目标，同时区分应该改变的交互区域；统一 Student 在实际编辑轨迹上继承这些参考响应与局部变化。论文的核心证据应是这些能力如何在统一时丢失、怎样被恢复，以及各领域的代价，而不是模块数量。

上述优先级是基于文献机制、当前代码和实现成本的研究判断。能否成为新贡献取决于与最近邻方法的直接对照及实验结果，当前没有把预期增益写成已测得结论。

**OPD 与少步生成的补充调研。** 以下补充来自同日针对 few-step 的 7 次 Exa 检索。需要区分多专家合并、推理步数压缩、已有少步模型的持续训练，以及训练时少步 rollout 这四种设置。

| 工作 | 已核对的少步设置 | OPD 如何参与 | 对 InsertFuse 的意义 |
| --- | --- | --- | --- |
| [FlowMap-OPD](https://arxiv.org/html/2609.37851)，2026 年 9 月 29 日预印本 | ImageNet 为 4 步；文生图将 OCR、PickScore、GenEval 三个教师合到一个 5 步 Student | 原生 flow-map rollout 获取状态，局部速度或映射监督，并用 flow–velocity consistency 联系生成与监督 | 最接近多领域专家合并到少步 Student，但文生图从预训练 MeanFlow checkpoint 起步 |
| [AnyFlow](https://arxiv.org/html/2605.13724v1)，2026 | 视频生成报告 4 NFE，并研究更高步数预算；覆盖 1.3B 至 14B、双向与因果架构 | 先训练两时间 flow map，再联合 forward training 与 DMD 式 on-policy backward simulation | 提供从原多步基座建立少步/任意步能力的具体路线；视频结果不能直接当作插入结果 |
| [D-OPSD](https://arxiv.org/html/2605.05204v3)，2026 | Z-Image-Turbo 8 步、FLUX.2-klein 4 步 | 在少步 Student 自身轨迹上，用更丰富上下文的自教师指导继续训练 | 适合已有少步模型之后学习插入或继续适配；其目标是保留已有少步能力 |
| [DMD2](https://arxiv.org/html/2405.14867)，NeurIPS 2024 | 真正的单步/少步压缩，包含 SDXL 4 步设置 | backward simulation 使用当前 Student 生成的中间输入，配合分布匹配、fake score 与 GAN 目标 | 广义 on-policy 少步蒸馏的重要先例；与当前逐状态教师 velocity MSE 的损失和资源需求不同 |

FlowMap-OPD 的文生图基座为 FLUX.1-lite-8B 的预训练 MeanFlow 版本，三类 prompt 按 1:1:1 路由。关键是分开决定“如何取得 Student 状态”与“在该状态比较什么”：长区间 flow map 负责五步生成，瞬时 velocity 监督配合可调自一致性约束，支持专家能力转移。本次读取的[官方仓库](https://github.com/ZhiqiLi-CG/Flowmap_OPD_source)仍写着 Code coming soon，不能将仓库存在视为训练实现已发布。

AnyFlow 的 Stage 1 将原视频模型训练成 \(f_\theta(z_t,t,r)\approx z_r\) 的两时间映射；Stage 2 保留 forward flow-map 训练，并加入 DMD 式分布监督。其 Flow Map Backward Simulation 用 \(T\to t\)、\(t\to r\)、\(r\to0\) 三段 shortcut 近似所需采样过程，在最终输出上计算分布匹配梯度。这里需要真实/生成分布的 score 机制，不能仅替换本项目 rollout 后仍称为完整复现。[官方代码](https://github.com/NVLabs/AnyFlow)包含训练配置和模型使用说明。

D-OPSD 在已有 Z-Image-Turbo、FLUX.2-klein 上研究持续训练。Student 在弱上下文下采样，Teacher 在更强的图文上下文下于同状态预测；部署保留原少步路径。[官方仓库](https://github.com/vvvvvjdy/D-OPSD)还列有 FLUX.2-klein 图像编辑训练路径。此证据支持少步模型的继续适配，不是把普通 Qwen 多步模型直接转成四步的证明。

相邻方法 [TDM](https://arxiv.org/html/2503.06674) 通过轨迹分布匹配和步数感知目标研究单步/四步生成，值得作为真正步数压缩的对照。[Any-OPD](https://arxiv.org/abs/2608.03316) 处理跨模型家族与 VAE 的异构迁移；其摘要的主要结果是 FLUX.1-dev 到 SD3.5-Medium 的容量迁移，不应仅因架构或 schedule 不同就把它列为已验证的少步压缩方法。OPTD 则是扩散语言模型的转移压缩，不属于直接的图像插入证据。

普通 velocity OPD 学的是某状态的瞬时方向；减少 Euler 步数会扩大积分区间，局部方向相同不意味着大步长更新也准确。两时间 Student 可以学习区间平均速度：

\[
F_\theta(x_t,t,s)=x_t+(s-t)\,u_\theta(x_t,t,s),\qquad s<t,
\]

其中 \(u_\theta\) 需要近似整段轨迹的平均速度，而非直接把 \(v_T(x_t,t)\) 当作不变的整段速度。FlowMap-OPD 用映射与速度之间的一致性维护这一联系；AnyFlow 则提供 forward flow-map 训练与分布蒸馏的组合。

对 InsertFuse 的一个可验证方案是：保留现有领域教师，在同一 Qwen latent/条件体系内先建立 8 步 flow-map Student，再做领域路由的 on-policy 合并，最后评价 8/4/2 步的保真与融合曲线。这样可以把“少步初始化是否足够”和“多教师能力是否被保留”分开诊断；这是一项迁移设计建议，相关论文尚未替你验证多类别插入效果。

另一项更直接的实验是 on-policy 区间转移监督：从少步 Student 实际访问的 \(x_t\) 出发，由匹配的教师在 \([t,s]\) 内用多个小步产生 \(x_s^T\)，训练 Student 一次跨越该区间：

\[
L_{\mathrm{segment}}
=\mathcal E_{\mathrm{regions}}\!
 \left(F_\theta(x_t,t,s)-\operatorname{sg}(x_s^T)\right).
\]

这与现有同一步 velocity MSE 不同，也与 AnyFlow 的 DMD 目标不同。应将它当作需要对照的区间蒸馏方案，而非已确立的新贡献。Teacher 多子步增加训练成本；source/reference/mask、时间与噪声尺度、geometry、参考 CFG 和背景投影必须统一。背景投影也可能破坏未经修改的 flow-map 组合关系，不能直接引用无约束 ODE 的一致性保证。

本任务真正值得研究的问题是：在少步大跨度更新中，如何同时保持微小主体的参考响应、边界与交互细节。可将前述局部对应、参考响应蒸馏和区域归一化用于区间目标，但分别消融。评价必须同时列采样步数、实际网络前向次数 NFE、延迟和各质量指标；带两分支 Reference CFG 的一步，与吸收 guidance 后单分支的一步，计算量不同。

**MeanFlow、FlowMap-OPD 与 AnyFlow 的机制关系。** 三者共享 flow-map 表示，但各自处于不同层次：[MeanFlow](https://arxiv.org/html/2505.13447) 给出平均速度的建模与训练原则；[FlowMap-OPD](https://arxiv.org/html/2609.37851) 研究已有少步生成器的教师能力迁移；[AnyFlow](https://arxiv.org/html/2605.13724v1) 将 flow-map 训练与 DMD 式 on-policy 蒸馏结合，用于少步及可变步数视频生成。

统一取 t 为当前噪声时间、r 为更接近图像的目标时间，r<t。MeanFlow 学习

\[
u(x_t,r,t)=\frac{1}{t-r}\int_r^t v(x_\tau,\tau)\,d\tau,\qquad
\psi_{t\to r}(x_t)=x_t-(t-r)u(x_t,r,t).
\]

当 r=t 时，网络的对角输出提供瞬时速度 \(v_\theta(x,t)=u_\theta(x,t,t)\)。同一个网络因此有瞬时速度与区间平均速度两个接口。MeanFlow 恒等式为 \(u+(t-r)(\partial_tu+J_xu\,v)=v\)。原始 MeanFlow 可以用数据与噪声构造的 conditional velocity 训练，无需先有专属教师；实际训练通过 JVP 计算导数项，并停止回归目标梯度。

FlowMap-OPD 的主要实用方案按以下顺序执行：从预训练的 MeanFlow Student 开始；使用长区间映射生成原生少步轨迹；detach 访问到的状态；按当前任务选择冻结教师；在相同状态匹配 Teacher/Student 的瞬时速度；同时约束 Student 的区间平均速度与其瞬时速度保持一致；更新 Student 后刷新采样状态。

其瞬时速度方案的目标是

\[
L=\mathbb E\left[w_v\|v_\theta-v_T\|^2+
 \lambda_c\|c_\theta\|^2\right],
\quad
c_\theta=u_\theta+(t-r)(\partial_tu_\theta+J_xu_\theta v_\theta)-v_\theta.
\]

w_v 来自论文采用的局部比较 kernel；实现对 JVP 项停止梯度。速度匹配传递教师能力，一致性将其联系到部署使用的大跨度映射。论文也比较直接 flow-map 和由 map 导出的 velocity 监督；主实验选择的是瞬时速度加独立可调的一致性项。上文提出的“教师多个子步产生区间终点，再回归 Student 一步”是 InsertFuse 的可选实验，不是该论文主实验的实际目标。

Rollout–kernel separation 指采样更新规则与局部监督的分布比较规则分开，不是拆成两个独立神经网络。理论上，若查询时间分布、局部损失与权重固定，两个 rollout 在查询时刻的状态边缘分布相同，则相应期望局部损失和停止轨迹梯度的期望更新相同。有限步近似模型并不自动满足这一条件；论文也讨论边缘分布误差与一致性残差。

瞬时速度对齐不足以单独确定所有 r<t 的输出，因此需要已有 flow-map 初始化或一致性学习来连接两个接口。另一方面，一致性越强也不等于任务迁移越好：论文文生图实验采用较小的 \(\lambda_c=0.001\)，较大的测试权重损害任务分数。这是该实验的结论，不是可直接照搬给 Qwen 的最佳超参数。

| 比较点 | FlowMap-OPD 的主要实用方案 | AnyFlow |
| --- | --- | --- |
| 起点 | 已有少步 MeanFlow checkpoint 和专属教师 | 预训练多步视频模型，先进行 forward flow-map 适配 |
| on-policy 状态 | 原生少步 Student rollout 的状态 | Flow Map Backward Simulation 的 shortcut 链产生输出 |
| 主要蒸馏信号 | 同状态教师瞬时 velocity 加 Student 自一致性 | 生成结果的 DMD 分布匹配梯度，并保留 forward flow-map 训练 |
| score/critic 需求 | 该速度匹配方案不需要 DMD fake-score 模型 | 使用教师 score 与 Student 分布的 fake score |
| 梯度路径 | rollout 状态 detach，仅局部预测和一致性更新 | 论文通过三段 shortcut 链反传输出端的分布梯度 |
| 重点 | 多专家能力合并及跨容量转移，同时保持原生少步采样 | 提升少步视频生成并保留增加步数时的质量扩展性 |

因此二者是共享 MeanFlow/flow-map 基础的不同后训练路线，并非必须依次执行的三个阶段。OPD 表示状态来自当前策略这一训练性质，不能仅凭名字就推断所有方法使用同一种 loss。

对 InsertFuse，可探究复用当前领域教师的 velocity 接口，只为 Student 增加目标时间 r 和区间映射能力。选择瞬时速度监督时，概念上不要求每个教师都显式输出 flow map，但必须统一 latent、时间尺度与条件，并将该异于论文主实验的接法单独验证。实际工作包括少步 Student 初始化、两时间条件、JVP/一致性、原生 rollout，以及 mask/background anchoring 对一致性关系的影响；它超出修改 rollout.num_inference_steps 的范围。

若目标是先完成多教师 OPD，再将融合模型转成 few-step，则采用一个冻结融合教师的第三阶段方案，详见 [InsertFuse 融合模型的少步蒸馏设计](./insertfuse_post_opd_few_step.md)。其中将区间教师监督、参考响应保留与背景投影的累计作用分别设计为可消融组件。

新颖性补充核对：FlowMap-OPD 第 4.3 节式 22 已包含直接区间映射回归，且允许通过积分教师 velocity 构造目标；该路线应归为基线，不作为区别于 FlowMap-OPD 的新贡献。其主实验使用 MeanFlow，教师通过 Flow-Map GRPO 优化，没有采用 AnyFlow 的 DMD 训练路线。此外，[MaskFlow 官方仓库](https://github.com/ReyChiaro/MaskFlow)已有 mask 编辑的 DMD 8/16 步变体；“局部编辑加 few-step”也不足以单独建立新颖性。具体对照与任务专属候选的边界已在上述设计文档末尾更新。

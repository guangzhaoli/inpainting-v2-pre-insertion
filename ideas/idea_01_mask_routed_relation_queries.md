# Idea 1: Mask-Routed Relation Queries

## 核心动机

当前 active 路线更像条件重建：

- `image = tar_image`
- `control = tar_image * (1 - tar_mask)`
- `control_1 = ref_image`

模型知道：

- 要补的区域在哪里
- 参考物体长什么样
- 周围背景是什么样

但它没有被显式要求学习一件最关键的事：

`reference object` 应该如何被搬运到 `target hole` 中，并且与周围上下文协调融合。

因此，现有做法容易出现：

- 插入目标大体正确，但局部细节丢失
- 物体和背景边界融合差
- 尺度、朝向、贴合位置不稳定
- 模型更像“根据条件重建 target”，而不是“显式执行插入”

## 方法概述

提出一个显式的关系查询模块，把 conditioning 拆成三类 token：

1. `Q_ref`
   从 `ref_image` 的前景区域提取，负责表示“要插入什么”。
2. `Q_ctx`
   从 hole 周围的上下文区域提取，负责表示“应该如何融入当前场景”。
3. `Q_mask`
   从 `tar_mask` 或其 latent mask 提取，负责表示“应该插到哪里、形状和覆盖范围如何”。

然后不是简单把这些 token 拼到文本 prompt 前面，而是做 mask-routed attention：

- mask 内区域主要关注 `Q_ref + Q_mask`
- 边界 ring 区域主要关注 `Q_ctx + Q_mask`
- 背景区域只保留很弱的插入条件，避免污染未编辑区域

## 关键创新点

### 1. 关系显式化

不是把 `ref_image` 和 `masked target background` 当成两个普通 control image，而是把它们拆成不同语义角色：

- 物体内容
- 场景上下文
- 空间区域约束

### 2. 区域路由

不同区域使用不同条件组合，而不是让全图统一吸收所有条件。

### 3. 对论文叙事友好

这个方法天然支持清晰的故事：

- baseline 只做隐式条件重建
- 我们显式建模 reference-to-hole relation
- 我们通过 mask-routed queries 实现区域化的条件注入

## 可实现形式

### Query 构造

- `Q_ref = f_ref(ref_fg_tokens)`
- `Q_ctx = f_ctx(boundary_tokens)`
- `Q_mask = f_mask(mask_tokens or latent mask)`

其中：

- `ref_fg_tokens` 来自 reference 前景 token
- `boundary_tokens` 来自 hole 周围一圈 ring 区域
- `mask_tokens` 来自 `tar_mask` 的图像空间或 latent 空间编码

### 路由形式

可以对 latent token 定义区域 gate：

`g_ref(i), g_ctx(i), g_mask(i) in [0, 1]`

其中 `i` 表示某个 latent token 所在位置。则条件注意力可以写成：

`E_i = [g_ref(i) * Q_ref; g_ctx(i) * Q_ctx; g_mask(i) * Q_mask; E_text]`

更简单的实现也可以不改 transformer 内部 attention，而是在 query token 注入前对不同区域使用不同权重。

## 训练目标

建议从单一 MSE / flow loss 升级成区域分解目标：

`L = lambda_in * L_in + lambda_bd * L_boundary + lambda_out * L_outside + lambda_ref * L_ref`

其中：

- `L_in`：mask 内重建或去噪损失
- `L_boundary`：边界 ring 区域损失，强调融合平滑和结构连续性
- `L_outside`：未编辑区域保持损失
- `L_ref`：参考一致性损失，可用 CLIP / DINO / Qwen-VL feature similarity

## 为什么适合当前代码

你当前 repo 已经有一个雏形：

- 已有 `query_prompt` 模块
- 已有 `mask_prompt`
- 已有 `tar_mask` 的额外 control 分支

因此从工程上最自然的升级路径是：

1. 扩展 `QueryPromptAdapter`
2. 引入 `region_type_ids` 或 region-specific gates
3. 把 `tar_mask` 不再只当成额外 control，而是变成 routing 信号
4. 新增 boundary-aware 与 reference-consistency loss

## 风险与难点

- 需要清楚处理 query token 对应的区域来源
- 需要设计 hole boundary 的抽取方式
- 如果 routing 做得太重，训练可能不稳定

## 结论

这是最值得作为主方法推进的方向。它兼顾：

- 新颖性
- 可实现性
- 与当前分支的连续性
- 论文叙事清晰度

# Idea 2: Align-Then-Refine with Pre-insertion

## 核心动机

很多 insertion / inpainting 的失败并不是“模型完全不会插”，而是：

- 初始布局不准
- 参考物体的大体位置和尺度不稳定
- 边界不干净
- 局部纹理和 surrounding context 不能自然衔接

这说明一个事实：

让模型从纯 hole 状态直接“凭空补出”最终结果，难度太高。

因此，一个很自然的改法是分两步：

1. 先给模型一个粗糙但位置合理的初始插入结果
2. 再让模型做 refinement，去修复边界、光照、纹理和整体和谐度

## 你的新想法：Pre-insertion

你提的 `Pre-insertion` 非常合理，核心是：

先将要插入的内容粗贴到 `mask` 区域，再进行生成。

这个想法本质上是把生成任务从：

`hole -> final image`

改成：

`coarse pasted image -> harmonized final image`

这通常会显著降低生成难度，并提升生产效果。

## 为什么它有效

### 1. 给模型一个强初始化

如果 reference object 已经被粗放进目标区域，模型不再需要同时解决：

- 放哪
- 多大
- 大概长什么样
- 怎么和背景融合

模型只需要重点解决：

- 边缘修复
- 纹理适配
- 光照与色彩 harmonization
- 局部形变与遮挡合理化

### 2. 更符合生产场景

实际生产中，用户往往更在意：

- 插入位置稳定
- 目标主体不要变形太离谱
- 边界和周围要自然

Pre-insertion 直接对准这三个目标。

### 3. 更容易控效果

相较纯生成，Pre-insertion 对“主体保真”通常更强。

## 可以怎样做

### 最简单版本

1. 从 `ref_image + ref_mask` 抠出前景
2. 根据 `tar_mask` 的 bbox 或更细的 shape，把参考前景 resize / place 到目标区域
3. 生成一个 coarse pasted image
4. 用这个 coarse pasted image 替代或补充现有 `control`
5. 让模型输出最终 harmonized 结果

可以把它写成：

- `control_bg = tar_image * (1 - tar_mask)`
- `control_paste = paste(ref_fg, tar_mask)`
- `control = blend(control_bg, control_paste, tar_mask)`

然后模型学习：

`final = G(control, ref_image, tar_mask, optional text)`

### 更强版本

不仅做 resize + paste，还加一个轻量几何对齐模块：

- predict scale
- predict offset
- predict small affine transform

这样粗贴结果会更接近真实目标位置。

## 作为方法，应该怎么包装

如果只是“先贴一下再生成”，它更像工程 trick。

如果要变成论文里的 method，建议把它系统化为：

`Pre-inserted Coarse Layout Guidance`

核心贡献可以写成两点：

1. 通过 pre-inserted coarse layout，把 open-ended generation 转化为 layout-conditioned harmonization
2. 通过 refinement objective 强化边界与上下文一致性

## 建议的输入设计

你现在的 active 路线里，`mask_prompt` 模式已经有：

- masked background control
- reference image control
- tar_mask extra control

那么 Pre-insertion 最自然的接入方式是再加一个 control：

- `control`: masked background
- `control_1`: reference image
- `control_2`: tar_mask or latent mask
- `control_3`: pre-inserted coarse composite

也可以做更省改动的版本：

- 用 `pre-inserted coarse composite` 直接替换当前 `control`
- 保留 `control_1 = ref_image`
- 保留 `control_2 = tar_mask`

## 配套损失

建议加两个更贴近这个方法的目标：

### 1. Boundary refinement loss

只在 mask 边界 ring 上强调修复质量。

### 2. Reference preservation loss

在 mask 内区域对生成结果与 reference 的高层语义一致性做约束。

## 这条线的优点

- 很实用，生产效果大概率能涨
- 比纯生成更稳
- 容易解释为什么主体保真更好
- 非常适合做 demo，效果图容易直观打动人

## 这条线的局限

- 如果只做 paste，再生成，method novelty 不够强
- 容易被 reviewer 质疑为 heuristic
- 若没有更系统的 refinement / alignment 设计，论文贡献偏弱

## 如何让它更像论文方法

你可以把它升级成：

`Alignment-aware Pre-insertion Refinement`

完整叙事是：

1. 先构造 coarse insertion layout
2. 再基于 coarse layout 做 context-aware refinement
3. 利用 boundary loss 和 identity-preserving loss 约束结果

如果再和第一条的 relation queries 结合，会更强：

- Pre-insertion 负责给出稳定初始布局
- Relation queries 负责在 refinement 中保留主体语义并协调局部区域

## 结论

`Pre-insertion` 非常适合作为生产增强方案，也有机会升级成论文中的一个重要组件。

单独使用时：

- 更像高价值工程方案

系统化后：

- 可以成为一篇论文中非常有说服力的子模块

我对它的判断是：

- 对 production value：高
- 对单独成 main novelty：中
- 对作为第二条方向中的核心组件：高

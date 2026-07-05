# Idea 3: Insertion-Removal Cycle

## 核心动机

当前训练主要是单向任务：

- 输入背景和参考
- 输出插入后的目标图

但对象级编辑天然包含一对互逆操作：

1. insertion: 把目标物体放进去
2. removal: 把目标物体拿出来

如果模型只学 insertion，它可能只是记住 target reconstruction 的统计规律，而没有真正学到：

- 什么是对象本体
- 什么是背景内容
- 对象和背景如何解耦

## 方法概述

把 insertion 和 removal 作为同一框架中的双向任务统一建模：

- forward：`src + ref + mask -> target`
- backward：`target + mask -> src`

再通过 cycle consistency 强化：

- 插进去再拿掉，应该回到原图
- 拿掉再放回去，应该恢复目标图

## 方法价值

### 1. 强化对象与背景解耦

模型不再只是“看结果图像然后拟合”，而是被迫理解：

- 哪部分属于插入对象
- 哪部分属于原背景

### 2. 提升未编辑区域保持能力

因为 backward task 天然要求 background consistency。

### 3. 论文故事更完整

相比单一 inpainting，它更像一个完整的对象编辑框架。

## 可实现形式

### 数据构造

如果已有：

- `tar_image`
- `tar_mask`
- `ref_image`

则 removal 目标可以由：

- `src_image/add`
- 或者 `tar_image * (1 - tar_mask)` 的更完整版本

来近似或直接提供。

### 双向训练

定义两个分支：

- insertion branch
- removal branch

损失可以写成：

`L = L_insert + lambda_remove * L_remove + lambda_cycle * L_cycle`

其中：

- `L_insert`：插入结果监督
- `L_remove`：移除结果监督
- `L_cycle`：双向循环一致性

## 可加入的约束

### 1. Object identity consistency

插入结果中的对象应该和 reference 保持语义一致。

### 2. Background preservation consistency

remove 之后背景应与真实背景接近。

### 3. Bidirectional latent consistency

可以在 latent 层面对 insertion / removal 的中间表示加入一致性约束。

## 这条线的优点

- 论文叙事大
- 与对象级编辑任务高度匹配
- 比单纯做 inpainting 更有 framework 感

## 这条线的缺点

- 数据准备和任务定义更复杂
- 工程开销明显高于前两条
- 如果 removal supervision 质量不够，训练会不稳定

## 适合什么节奏

如果你想快速把当前系统升级成有新意的方法，不推荐先做这个。

如果你已经把第一条或第二条跑通，想进一步扩大论文野心，可以把它作为后续版本。

## 结论

这条路线有较强的论文框架感，但不是最省力的首选。

更合理的策略是：

1. 先做第一条，拿到一个强主方法
2. 或做第二条，快速提升生产效果
3. 在此基础上再考虑把 insertion-removal cycle 做成更大的版本

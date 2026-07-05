# Inpainting Method Ideas

这个目录整理当前较值得推进的三个方向，目标是从“只做数据域微调”升级到“有明确方法贡献”的 inpainting / insertion 方案。

文件列表：

- `idea_01_mask_routed_relation_queries.md`
- `idea_02_align_then_refine_and_pre_insertion.md`
- `idea_03_insertion_removal_cycle.md`

推荐优先级：

1. `idea_01_mask_routed_relation_queries.md`
2. `idea_02_align_then_refine_and_pre_insertion.md`
3. `idea_03_insertion_removal_cycle.md`

说明：

- 第一条最适合做主方法，和当前 `query_prompt + mask control` 代码演进最连续。
- 第二条更偏生产效果和落地收益，其中 `Pre-insertion` 是一个很自然、很实用的增强版本。
- 第三条论文故事更大，但训练和数据构造成本也最高。

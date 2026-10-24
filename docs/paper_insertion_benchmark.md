# PaperInsert-900：同身份配对重建数据集与评测协议

## 1. 任务定义

`PaperInsert-900` 是参考图驱动插入/补全的像素 GT 测试集。每个 case 必须同时满足：

- `src` 和 `tar` 来自不同的原始图像或不同视频帧；
- 两侧前景是同一身份，而不是仅有相同类别；
- reference 只从独立的 source 图像生成，禁止从 target 像素裁出或复制；
- 完整 `tar_image` 是该 case 的 ground truth，推理输入中的洞由 `tar_mask` 在线构造。

manifest 中对应的固定语义为：

```text
protocol                    = "paired_reconstruction"
case_type                   = "paired_gt"
same_identity               = true
gt_available                = true
reference_derived_from_target = false
```

这里的“同一身份”有两种经过上游标注验证的定义：MeViS v2 使用同一个被跟踪的物理实例；DeepFashion2 使用同一个官方 consumer/shop 商品身份。后者不是“同类衣服随机配对”，而是相同 `pair_id + style` 的正商品对。

主集固定为 900 条：`object`、`person`、`garment` 各 300 条，每个大类包含 6 个小类，每个小类 50 条。目标稳定目录为：

```text
/inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/
└── paper_insertion_benchmark_v1/
    └── paperinsert_900/
```

只有当该目录通过本文 §7 的严格审计、泄漏检查和发布证据核对后，才能称为当前同身份发布版。历史上使用 COCO/Fashionpedia 的 `cross_instance_replace` 构建不是本协议，也不能作为本协议的稳定版。

AnyInsertion 如需保留，只能作为独立补充集；它不进入上述 900 条，不共享主集统计，也不能替代主集的身份约束，见 §10。

## 2. Canonical 四文件契约

每个 case 有且只有以下四个 canonical PNG：

- `src_image`：从独立 reference 原图裁出的干净参考。裁剪框在原生 mask bbox 外保留 15% margin，mask 外像素置为纯白。
- `src_mask`：与 `src_image` 同尺寸的 reference 精确实例轮廓。
- `tar_image`：未经遮挡或涂抹的完整 target GT 图像。
- `tar_mask`：与 `tar_image` 同尺寸的 target 精确实例轮廓，也是推理时的待补全区域。

mask 使用来源数据集的原生逐实例标注：MeViS v2 的 COCO RLE，或 DeepFashion2 的实例 polygon。落盘后均为单通道、非空、非全图、像素值只含 0/255 的 PNG；构建阶段不做 dilation。推理需要的扩张或软化只允许在运行时生成，不能回写数据集。

目录结构为：

```text
paperinsert_900/
├── object/
│   ├── src_image/
│   ├── src_mask/
│   ├── tar_image/
│   └── tar_mask/
├── person/
│   └── ...
├── garment/
│   └── ...
├── manifests/
│   ├── paperinsert_900.jsonl
│   ├── paperinsert_900.csv
│   ├── object.jsonl
│   ├── person.jsonl
│   └── garment.jsonl
├── reports/
│   ├── sources.json
│   ├── audit.json
│   ├── training_leakage.json
│   ├── training_overlap_visual_review.json
│   ├── agent_visual_qc.json
│   ├── paper_showcase_shortlist.json
│   └── paper_showcase_shortlist.csv
├── qc/
│   ├── index.json
│   ├── *.jpg
│   ├── replacement_review_round12/
│   ├── training_near_review/
│   ├── showcase_decisions.json
│   └── showcase_review/
└── CHECKSUMS.sha256
```

`reports/training_overlap_visual_review.json` 和 `reports/agent_visual_qc.json` 只有在相应复核真正完成后才应存在。它们不能伪装成人类审批。

`CHECKSUMS.sha256` 的权威路径集合固定为 3,606 条：3,600 个 canonical PNG、5 份 manifest 和 `reports/sources.json`。`audit.json`、QC 图和后续泄漏报告是派生证据，不在这 3,606 条中。

## 3. 身份配对与上游证据

| 主类 | 数量 | 来源 | `pair_kind` | 身份键 | 配对方向 |
|---|---:|---|---|---|---|
| object | 300 | MeViS v2 train | `same_track_cross_frame` | `video_id + anno_id` | 同一 track 的两个不同 frame |
| person | 300 | MeViS v2 train | `same_track_cross_frame` | `video_id + anno_id` | 同一 track 的两个不同 frame |
| garment | 300 | DeepFashion2 validation | `same_product_cross_domain` | `pair_id + style`，且 `style > 0` | shop reference → user target |

### 3.1 MeViS v2

MeViS 官方元数据中的 `(video_id, anno_id)` 是物理实例的跟踪身份。source 和 target 必须共享这两个字段，但 `frame_id` 必须不同；两帧的 mask 都从同一个 `mask_dict.json` track 读取。每个最终 track 最多使用一次，每个视频最多使用两个 case。600 个 object/person case 至少覆盖 596 个不同视频，避免少数视频主导测试集。

MeViS 的类别由 motion expression 证据整理，manifest 保留 `mevis_expressions`、语义得分、候选池 ID、帧索引和几何/像素差异字段，使类别归属和帧选择可追溯。两个帧还必须通过前景像素差异下限，防止把几乎相同的相邻帧当作独立 reference/target。

### 3.2 DeepFashion2

DeepFashion2 官方说明中，来自 shop 和 user 图像的衣物只有在 `pair_id` 相同、`style` 相同且 `style > 0` 时才是正 commercial-consumer pair。构建器进一步要求两侧 `category_id` 相同，并以 validation 的 `val_gallery.json` 和 `val_query.json` 为成员资格权威：shop/gallery 作为 reference，user/query 作为 target。

因此 garment 测试的是同一商品身份跨拍摄域重建，而不是同类别跨实例替换。manifest 中同时保留两侧 image/item ID、`pair_id`、`style`、`category_id`、质量属性和原生 bbox。身份规则可参见 [DeepFashion2 官方仓库](https://github.com/switchablenorms/DeepFashion2)。

### 3.3 审计不信任布尔字段

严格审计会从 `src_upstream`、`tar_upstream` 和固定上游标注重新推导身份，不会仅相信 `same_identity=true`：

- MeViS 必须重新匹配同一 `(video_id, anno_id)`、不同 frame；
- DeepFashion2 必须重新匹配 shop/user、相同 `pair_id + style + category_id` 和官方 query/gallery 成员；
- 两侧原始路径、解码像素和 raw mask 必须能重建出 canonical 四文件；
- 任意 target-derived reference、同原图配对或身份键不一致都会令 audit 失败。

## 4. 类别与配额

| 大类 | 小类 | 数量 |
|---|---|---:|
| object | `companion_animals` | 50 |
| object | `wild_or_livestock` | 50 |
| object | `birds` | 50 |
| object | `aquatic_reptiles` | 50 |
| object | `vehicles` | 50 |
| object | `daily_sports_objects` | 50 |
| person | `sport_dance_exercise` | 50 |
| person | `sitting_kneeling_lying` | 50 |
| person | `walking_running` | 50 |
| person | `social_interaction_care` | 50 |
| person | `work_object_handling` | 50 |
| person | `standing_gesturing` | 50 |
| garment | `short_sleeve_top` | 50 |
| garment | `long_sleeve_top` | 50 |
| garment | `outerwear_vest_sling` | 50 |
| garment | `shorts_trousers` | 50 |
| garment | `skirt` | 50 |
| garment | `dress_onepiece` | 50 |
| **合计** | **18 个小类** | **900** |

garment 的 6 个小类还强制覆盖 DeepFashion2 全部 13 个原生类别：

| 聚合小类 | 原生 `category_id: 数量` |
|---|---|
| `short_sleeve_top` | `1: 50` |
| `long_sleeve_top` | `2: 50` |
| `outerwear_vest_sling` | `3: 4, 4: 18, 5: 18, 6: 10` |
| `shorts_trousers` | `7: 25, 8: 25` |
| `skirt` | `9: 50` |
| `dress_onepiece` | `10: 13, 11: 12, 12: 13, 13: 12` |

## 5. 固定来源、v5 候选池与可追溯性

### 5.1 固定原始快照

主集只读取以下固定快照：

| 文件 | SHA256 |
|---|---|
| MeViS v2 `train/JPEGImages.tar` | `bdbd8bd6deb17b10705e43ecba66772e9be6a702ca10306e52aff76f3d4178d2` |
| MeViS v2 `train/meta_expressions_v2.json` | `134b6e6ca013ea648b3bd71c7f65662d0b660ece932e4c0e9e9c2d1c93fc1d3d` |
| MeViS v2 `train/mask_dict.json` | `f9d9b01a2a6d7a004e2e7a664ce2b9b1576218e975030f8e4df4f66111fb253f` |
| DeepFashion2 `validation.zip` | `0da4c5eabd8693606a26b8a5d95542d4e467655cb3c7331c85ece1f45a75ca51` |
| DeepFashion2 `val_query.json` | `f822c94c77d3b1b2277203700ea16b6f75ac359eb9e66112de9798c69c66be9b` |
| DeepFashion2 `val_gallery.json` | `dd06bac3eedb8c956df2b12e0f9dd0d28354140fa3f55306b56f0a492c21f11b` |

MeViS 的官方数据结构包含逐帧图像、表达式元数据和 COCO RLE mask，详见 [MeViS 官方仓库](https://github.com/henghuiding/MeViS)。DeepFashion2 validation 解压后必须恰有 32,153 张图像和 32,153 份 annotation；不匹配时构建器直接失败。

构建还会把所有 900 条所使用的原图和 DeepFashion2 annotation 与上述固定 archive member 逐字节比较。`reports/sources.json` 记录 archive member 集合摘要、builder SHA、依赖版本、seed、排除配置、候选 plan cache 和 curation pool 的路径及 SHA。

### 5.2 MeViS curation pool v5

当前 builder 固定使用：

```text
_raw/mevis_v2/curation/mevis_diverse_candidate_pool_v5.json
SHA256 = 7414e68c16f71bd11c0bc37763465146bc4593728e145e1befe04d9eac13a993
schema = mevis_diverse_candidate_pool_v5
```

v5 不是新的像素或自动生成 mask 数据集，而是一个只保存上游 ID、两个帧索引、表达式证据和确定性质量统计的成员池。构建时仍从固定 MeViS archive 重新读取图像与原生 RLE。

v5 的扩展链条固定为：

- 基础 curated v1：1,604 行，SHA256 `04ade013a827a26923c44eae919e998a2b41956ed8ee600705d95309915a7392`；
- 确定性补充：825 行，只允许基础池中没有的 track；
- 合并结果：2,429 个 category-membership row、2,077 个唯一 track、1,097 个唯一视频；
- 补充项通过固定几何/像素门槛和有界 relaxed-geometry fallback，并带 selection penalty；
- 所有补充项明确标记 `supplemental_visual_qc_required=true`，自动入池不等于视觉通过。

扩展器为 `scripts/expand_mevis_identity_curation_pool.py`。候选 plan cache 还绑定 builder SHA、raw root、依赖版本、seed、所有 source pin 和 v5 pool SHA；builder 或输入变化后，旧 cache 会被拒绝，必须使用新的 cache 文件名重新规划。

### 5.3 排除与重建

`configs/identity_paired_insertion_exclusions.json` 保存 agent-assisted 视觉复核发现的 identity 级硬排除。排除单位是完整 track 或商品身份，避免同一坏身份换一个 case ID 后重新进入数据集。

当前发布冻结了 365 个唯一 identity 排除项，配置 SHA256 为 `3c903a98dc98d6d284d058b3e2a6684fc1c2e97215ab6187e5297a1b924422bc`。复核历史中存在临时加入后撤回的条目，因此不能把该文件描述成严格 append-only 日志；重建时只以冻结配置内容及其 SHA 为准。排除来源汇总和这一限制记录在最终发布的 `reports/agent_visual_qc.json`。

排除配置变化时必须构建新的空 candidate 目录。不能在已签名目录中删除 case、手工补文件或重写 manifest。新 candidate 通过审计、泄漏复核和视觉复核后，才可由发布流程提升；旧稳定树必须保留为明确标注的 superseded 备份。

## 6. 自动 QC、人工 QC 与论文展示

builder 按 `difficulty` 和 `selection_score` 为每个小类预选 10 条 `showcase_candidate`，总计 180 条。这个字段只表示“优先复核”，不表示可直接放论文。

初始审批字段必须保持：

```text
agent_visual_qc_status = "pending"
manual_qc_status       = "pending"
paper_ok               = false
```

agent-assisted 全量 contact-sheet 检查可以：

- 将 mask 严重污染、主体几乎不可见、明显错类、非真实前景或损坏图像对应的 identity 加入硬排除；
- 把复核范围、拒绝原因、替换轮次和最终 manifest SHA 写入独立的 `reports/agent_visual_qc.json`；
- 为下一轮只渲染新增替换项，避免把未变化的 case 错当作重新审批。

但 agent-assisted QC 不能把 `manual_qc_status` 改为 `passed`，也不能把 `paper_ok` 改为 `true`。论文展示前仍需人类逐张确认：mask 边界、遮挡、主体身份、构图、水印、敏感内容、人物隐私，以及原媒体的展示与署名权利。建议从预先冻结的 shortlist 同时选成功例和失败例，不要按模型输出事后挑 best-of-k。

当前发布使用独立 shortlist 流程，从每个 minor 的确定性排序中取 6 个候选，共 108 条；代理四视图复核后每个 minor 接受 3 条、拒绝 3 条，最终得到 18 × 3 = 54 条。对应文件为：

- `qc/showcase_review/`：18 张四视图复核表；
- `qc/showcase_decisions.json`：108 条代理接受/拒绝决策；
- `reports/paper_showcase_shortlist.json` 与 `.csv`：54 条冻结 shortlist。

另有 19 条技术上仍合格、但因边界/遮挡、分辨率或构图、动作语义、敏感场景、近重复等原因不建议用于论文展示的 case，记录在 `reports/agent_visual_qc.json`；它们与 54 条 shortlist 的交集为 0。shortlist 的状态仍为 `agent_shortlisted_pending_human_content_and_rights_review`，不是最终论文批准名单。

## 7. 构建、审计与泄漏检查

所有 Python 命令先进入项目环境；长任务使用 `zmx`，不要使用 `tmux`。

### 7.1 原始目录契约

官方数据需按各自访问条款取得，并整理为：

```text
paper_insertion_benchmark_v1/_raw/
├── mevis_v2/train/
│   ├── JPEGImages.tar
│   ├── JPEGImages/<video_id>/<frame>.jpg
│   ├── meta_expressions_v2.json
│   └── mask_dict.json
├── mevis_v2/curation/mevis_diverse_candidate_pool_v5.json
└── deepfashion2_validation/
    ├── validation.zip
    ├── validation/image/*.jpg
    ├── validation/annos/*.json
    └── json_for_validation/
        ├── val_query.json
        └── val_gallery.json
```

不要用旧的 COCO/Fashionpedia 下载或构建命令生成主集。

### 7.2 构建新的 candidate

下面的 `release_YYYYMMDD` 是版本标签；每次重建都换一个从未使用过的空 candidate 目录。plan cache 只有在 builder、依赖、raw root、source pin、seed 和 curation pool 全部仍匹配时才能复用；其中任一变化都应使用新的 cache 文件名重新规划。

```bash
zmx run paperinsert_identity_build_release_YYYYMMDD -d bash -lc '
source /inspire/hdd/global_user/250041225002/miniconda3/etc/profile.d/conda.sh &&
conda activate qwen-inpainting-pre-insertion &&
cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-final-architecture &&
python scripts/prepare_identity_paired_insertion_dataset.py build \
  --work-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1 \
  --raw-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/_raw \
  --output-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_candidate_identity_release_YYYYMMDD \
  --seed 20260725 \
  --workers 8 \
  --exclusions configs/identity_paired_insertion_exclusions.json \
  --mevis-curation-pool /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/_raw/mevis_v2/curation/mevis_diverse_candidate_pool_v5.json \
  --plan-cache /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/_cache/identity_paired_candidate_plans_seed20260725_v5.json.gz
'
zmx wait paperinsert_identity_build_release_YYYYMMDD
```

`build` 会先规划完整配额、校验固定 archive、生成 canonical 文件和 checksum，再做 raw-backed audit 和 900-case QC sheets。它拒绝覆盖非空输出目录。不要在正式构建中使用 `--skip-large-archive-hash` 或 `--no-raw-audit`。

### 7.3 严格重审

```bash
source /inspire/hdd/global_user/250041225002/miniconda3/etc/profile.d/conda.sh
conda activate qwen-inpainting-pre-insertion
cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-final-architecture

python scripts/prepare_identity_paired_insertion_dataset.py audit \
  --work-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1 \
  --raw-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/_raw \
  --output-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_candidate_identity_release_YYYYMMDD
```

合格结果至少应有：`valid=true`、`errors=[]`、900 个唯一 identity、`major=300`、18 个 minor 各 50、600 个 `same_track_cross_frame`、300 个 `same_product_cross_domain`、900 个不同 source/target image、0 个 target-derived reference，并通过 3,606 条 checksum 和 raw reconstruction。

### 7.4 替换项视觉复核

全量 sheet 位于 candidate 的 `qc/`。若这一轮只替换了上一轮被排除的 identity，可额外生成差分复核图：

```bash
python scripts/render_identity_replacement_review.py \
  --baseline-root /absolute/path/to/previous_reviewed_candidate \
  --candidate-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_candidate_identity_release_YYYYMMDD \
  --exclusions configs/identity_paired_insertion_exclusions.json \
  --output-dir /absolute/new/path/replacement_review_release_YYYYMMDD
```

该工具用 identity 与四路像素 hash 的组合识别真正新增的 case；输出仍标为 pending，不是人类认证。

### 7.5 训练泄漏扫描

主集扫描 2,700 个 image-role：900 个 canonical reference crop、manifest 中的 900 个完整 reference 原图指纹、900 个 target。先做解码像素 SHA256 精确匹配，再以 pHash 半径给出完整近邻复核队列；近邻不能静默当作无泄漏。

```bash
zmx run paperinsert_identity_leakage_release_YYYYMMDD -d bash -lc '
source /inspire/hdd/global_user/250041225002/miniconda3/etc/profile.d/conda.sh &&
conda activate qwen-inpainting-pre-insertion &&
cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-final-architecture &&
python scripts/check_paper_dataset_leakage.py \
  --dataset-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_candidate_identity_release_YYYYMMDD \
  --train-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/data \
  --cache /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/_cache/training_image_fingerprints_schema2.jsonl \
  --report /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_candidate_identity_release_YYYYMMDD/reports/training_leakage.json \
  --phash-radius 4 \
  --workers 8
'
zmx wait paperinsert_identity_leakage_release_YYYYMMDD
```

精确命中会返回非零；解码错误表示扫描不完整，也会失败。近邻复核图可用：

```bash
python scripts/render_leakage_near_review.py \
  --report /absolute/path/to/reports/training_leakage.json \
  --output-dir /absolute/new/path/leakage_near_review
```

最终视觉结论需绑定 candidate manifest SHA，写入 `training_overlap_visual_review.json`。训练池变化后必须重扫；某一模型的无泄漏结论不能外推给其他训练集。

### 7.6 回归测试

```bash
pytest -q \
  tests/test_prepare_identity_paired_insertion_dataset.py \
  tests/test_expand_mevis_identity_curation_pool.py \
  tests/test_paper_dataset_leakage.py \
  tests/test_infer_testset_schema.py \
  tests/test_render_identity_replacement_review.py \
  tests/test_render_daily_sports_pool_review.py \
  tests/test_select_paper_showcase_shortlist.py \
  tests/test_promote_identity_paired_dataset.py
```

当前最终代码状态下，上述精确命令结果为 `97 passed, 44 subtests passed, 11 warnings`；11 条均为 NumPy 2 的弃用警告。

## 8. 推理与评测

`infer_testset.py` 会优先发现 canonical `src_image/src_mask/tar_image/tar_mask` schema，并验证受支持发布版的 manifest/checksum。每个大类应使用独立 session 和全新输出目录。以 `object` 为例：

```bash
zmx run paperinsert_object_seed42 -d bash -lc '
source /inspire/hdd/global_user/250041225002/miniconda3/etc/profile.d/conda.sh &&
conda activate qwen-inpainting-pre-insertion &&
cd /inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2-final-architecture &&
python infer_testset.py \
  --config /absolute/path/to/matching_config.yaml \
  --lora /absolute/path/to/checkpoint_or_lora \
  --testset-root /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900 \
  --category object \
  --output-dir /absolute/new/path/paperinsert_900/current_model/object_seed42 \
  --seed 42
'
zmx wait paperinsert_object_seed42
```

对 `person`、`garment` 重复运行并保持相同推理配置。不要使用已有非空输出目录，也不要在未记录理由时使用 `--allow-checkpoint-mismatch`。

由于 900 条均为 `paired_gt`，原始 `tar_image` 可以作为 target mask 内的像素 GT；同时还应报告 reference-to-output 身份保持和感知质量。若启用 strict composite，最终 `output.png` 的 mask 外区域可能是输入复制，不能用它证明背景保持；背景指标应读取未复合的 `output_restored_raw.png`。

所有方法必须使用同一 manifest 顺序、同一 mask、同一 seed 规则和同一输出数量。不得针对单个 case 事后 best-of-k。论文中需明确 mask 内/外区域、是否 composite，以及失败或未生成 case 的计分规则。

## 9. 发布证据区块

下面第一个区块必须由最终 candidate 提升后产生的文件替换；保留 `TODO` 时，文档不得声称同身份 v5-curation 版本已经正式冻结。

<!-- FINAL_RELEASE_EVIDENCE_BEGIN -->

```yaml
status: released_agent_qc_complete_human_paper_review_pending
release_root: /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900
release_tag: identity_v19_20260727
manifest_sha256: 6e2afd9c2a63fed67501352301d49b3ea38d5c44cb34fe221b7db13ee8c5272c
checksum_manifest_sha256: 4c3e578f2b2cf3b5750b6114fbc910489e05a1a3eb40d849a23ba2a084ac343e
checksum_verification: 3606_of_3606_passed
source_report_sha256: fa6e348b0f11407ac8b194c6f274c18ab45fc593fba354f97a4c9fc34db124a1
audit_report_sha256: ae68364bcdabb7fb6c1876d485f2bbc4ad85998ce810dcdf012f83b773e48a52
audit_valid: true
audit_error_count: 0
audit_warning_count: 0
record_count: 900
major_counts: {object: 300, person: 300, garment: 300}
minor_quota_status: 18_of_18_at_50
identity_count: 900
pair_kind_counts: {same_track_cross_frame: 600, same_product_cross_domain: 300}
difficulty_counts: {easy: 739, moderate: 161}
different_source_image_count: 900
target_derived_reference_count: 0
mevis_unique_video_count: 598
mevis_repeated_video_count: 2
raw_backed_audit: true
training_leakage_report_sha256: 962f64bbbc2c2c482aa0b72a789815da83438b6eb67e4984907ed62ad80b6819
training_inventory_count: 247084
training_exact_match_count: 0
training_near_match_count: 12
training_near_match_candidate_count: 7
training_overlap_visual_review_sha256: 7a16a9a966d6b15083842309927808e3d544b49b5f0b8118fd791cbc08320596
visually_confirmed_overlap_count: 0
agent_visual_qc_report_sha256: 650bf277521b747285d8dcf6c7733a394ecaf43d4cf418a0a2d5054b6d87bb29
agent_visual_qc_reviewed_case_count: 900
agent_final_replacement_delta: {reviewed: 8, hard_reject: 0}
manual_qc_status_counts: {pending: 900}
paper_ok_true_count: 0
agent_paper_avoid_count: 19
showcase_shortlist: {total: 54, minor_count: 18, per_minor: 3}
showcase_shortlist_sha256: aab2d8bffd514da6be55550a87bf9493ede5a04d13c8dc791a1589d7879ba1fa
showcase_content_review_status: pending_human_review
showcase_rights_review_status: pending_human_review
core_test_result: 97_passed_44_subtests_passed_11_warnings
promotion_method: verified_same_filesystem_two_rename_fallback_under_flock
promotion_single_syscall_atomic_exchange: false
superseded_backup_root: /inspire/hdd/global_user/250041225002/project/qwen-inpainting/a2edit/eval/paper_insertion_benchmark_v1/paperinsert_900_cross_instance_superseded_20260725_ca2aa07d
```

<!-- FINAL_RELEASE_EVIDENCE_END -->

发布工具首先尝试单系统调用的 `renameat2(RENAME_EXCHANGE)`，但当前文件系统返回不支持，且在变更目录前安全退出。随后使用同文件系统、`flock` 保护、切换前后双重哈希校验、显式回滚分支和 `fsync` 的两次 rename fallback 完成提升；该 fallback 不是单系统调用的原子 exchange。旧的跨实例稳定树保留在上述 superseded 目录中。

以下仅是 2026-07-25 identity candidate v3 的历史诊断证据，用于发现协议实现问题；它早于 identity 排除替换和 v5 curation pool 扩展，**不是最终发布证据，不得引用为当前稳定版**：

- `audit.valid=true`，900 条，3 个大类各 300、18 个小类各 50；
- 900 个唯一 identity，600 个同 track case、300 个同商品 case；
- source/target 不同图像 900，target-derived reference 0；
- difficulty 为 easy 781、moderate 119；MeViS 覆盖 596 个视频，其中 4 个视频使用两次；
- manifest SHA256 `4d6988ff2bd0dc68aab8e8b603bd8d7a4d18f4939cbbe2038b64c5ef16028c1d`；
- `CHECKSUMS.sha256` SHA256 `bc136f786eed772a64081b8493036c2488f7221da818735ad3e135b5df5dbf5a`；
- 对当时 247,084 张训练图扫描得到 0 个精确命中、29 个 pHash 近邻 row，涉及 8 个 source role；代理高分辨率复核确认 0 个真实重叠。

## 10. AnyInsertion 仅作为独立补充

若继续维护 `anyinsertion_official/`，它必须拥有自己的 manifest、checksum、身份审查、泄漏报告和评测表。不得把它的 case 加入 PaperInsert-900 的 900 条，也不得把它的 `same_identity` 或 `gt_available` 结论推断到主集。

`scripts/prepare_paper_insertion_dataset.py` 是历史主集构建器；它不能再用于生成本协议的 PaperInsert-900。它若仍承担 AnyInsertion 转换，也只能处理单独的 supplement 目录，并以 supplement 自身审计结果为准。

## 11. 许可、隐私与发布边界

- [MeViS 官方仓库](https://github.com/henghuiding/MeViS) 将数据声明为 CC BY-NC-SA 4.0，并限定非商业研究用途；底层视频/图像权利仍需遵守。
- [DeepFashion2 官方仓库](https://github.com/switchablenorms/DeepFashion2) 通过申请表和密码提供数据。本项目不假设存在开放的图片再分发授权；原图和论文展示用途需逐项核权。
- 人物 case 还需检查隐私、敏感内容、未成年人和肖像展示风险。
- builder、manifest 和 checksum 的可公开性不等于原始媒体可公开再分发。

因此默认公开构建代码、固定 ID、来源链接、manifest 和校验信息，而不是重新托管全部图片。`paper_ok=true` 只能由团队完成人工内容与权利检查后设置，不能从 `audit.valid=true`、`showcase_candidate=true` 或 agent-assisted QC 自动推导。

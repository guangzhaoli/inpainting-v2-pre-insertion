# Cross-category gradient conflict analysis

This experiment measures whether categories in the jointly trained model ask
the shared trainable parameters to move in compatible directions. It does not
retrain or update the model: the code runs training-mode forward/backward
passes on a fixed checkpoint, records gradients, and never creates an
optimizer.

For semantic group `c`, repeat `r`, and `N` samples, the script computes

\[
g_c^{(r)} = \frac{1}{N}\sum_{n=1}^{N}\nabla_\theta
\mathcal{L}(x_{c,n};\theta),\qquad
s_{ij}^{(r)} = \frac{g_i^{(r)\top}g_j^{(r)}}
{\lVert g_i^{(r)}\rVert_2\lVert g_j^{(r)}\rVert_2}.
\]

Here, \(\theta\) is exactly the set of parameters trainable in the joint run
(the active LoRA plus any trainable geometry tokenizer), and
\(\mathcal{L}\) is the repository's actual supervised training objective.
Native categories are balanced within each semantic group. Samples are unique
across repeats, while diffusion timestep, noise seed, reference-dropout seed,
and MaskMix seed are matched by sample slot across groups. Timesteps use a
shuffled stratified-uniform grid so every group has identical noise-level
coverage in every repeat.

## One-command launch

Use the joint model checkpoint, not any of the category-specific teacher
checkpoints. A checkpoint directory is preferred because it provides the
context needed to discover `train_config.yaml` and any companion module
weights. Legacy runs that store `train_config.yaml` one directory above the
checkpoint are detected too.

```bash
bash scripts/run_category_gradient_conflict.sh \
  /path/to/joint/checkpoint-last-epoch-step-last
```

If a legacy config points to a manifest that has moved, override only the data
location while retaining the checkpoint's training graph:

```bash
bash scripts/run_category_gradient_conflict.sh CHECKPOINT \
  --manifest-path /current/path/to/insert_anything_teacher_all_data.csv
```

The launcher uses the project Conda environment, starts eight-process
`accelerate` from `accelerate_config_8.yaml`, and detaches the job into a `zmx`
session. It prints the exact session name and result directory. Follow it with:

```bash
zmx tail grad-conflict-YYYYMMDD-HHMMSS
```

### Ready-to-run joint checkpoint in this workspace

The completed pre-OPD joint model currently available on this machine is:

```text
/inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2/logs/inpainting_v2_teacher_logs/teacher_all_data_native_qwen_image_inpainting/v1/checkpoint-last-2-46326-last
```

Its saved manifest path belongs to a moved worktree. The current manifest is
byte-identical (SHA256
`3575af6be879351d6f7a5210459b0e5f37e84b1fc66470da2558f88949b6b82f`), so the
ready command is:

```bash
JOINT_CHECKPOINT=/inspire/hdd/global_user/250041225002/project/qwen-inpainting/inpainting-v2/logs/inpainting_v2_teacher_logs/teacher_all_data_native_qwen_image_inpainting/v1/checkpoint-last-2-46326-last

bash scripts/run_category_gradient_conflict.sh "$JOINT_CHECKPOINT" \
  --manifest-path manifests/insert_anything_teacher_all_data.csv
```

This checkpoint uses r=128 LoRA and the original joint supervised objective,
which makes it the appropriate model for diagnosing the interference that
motivates subsequent OPD.

To focus the main figure on the three groups in the paper motivation:

```bash
bash scripts/run_category_gradient_conflict.sh \
  /path/to/joint/checkpoint-last-epoch-step-last \
  --preset-groups accessory garment human
```

The built-in five-group mapping is:

| Paper group | Native dataset categories |
|---|---|
| accessory | `accessory`, `jewel_filter` |
| garment | `clothes`, `dress`, `TopRef_BodyMask`, `BottomRef_BodyMask`, `TopRef_body_man`, `pure_color` |
| human | `person`, `person_hair` |
| animal | `cat`, `dog` |
| general_object | `others` |

Custom group definitions can be supplied by repeating `--group`:

```bash
bash scripts/run_category_gradient_conflict.sh CHECKPOINT \
  --group accessory=accessory,jewel_filter \
  --group garment=clothes,dress \
  --group human=person,person_hair
```

Useful launcher overrides are:

```bash
ACCELERATE_CONFIG=accelerate_config_4.yaml \
SAMPLES_PER_GROUP=32 \
REPEATS=5 \
OUTPUT_DIR=outputs/category_gradient_conflict/paper_main \
bash scripts/run_category_gradient_conflict.sh CHECKPOINT
```

`SAMPLES_PER_GROUP` must be divisible by the number of accelerate processes.
For a quick plumbing check, use one repeat, one sample per GPU, and
`--latent-only`; do not use that smoke result in the paper.

```bash
SAMPLES_PER_GROUP=8 REPEATS=1 \
bash scripts/run_category_gradient_conflict.sh CHECKPOINT --latent-only
```

## Outputs

The output directory is self-contained:

- `category_gradient_cosine_heatmap_all_trainable.{png,pdf}`: primary mean
  cosine heatmap on all shared trainable parameters.
- `pairwise_cosine_intervals_all_trainable.{png,pdf}`: pairwise means and 95%
  Student-t confidence intervals over repeats.
- `category_gradient_negative_fraction_all_trainable.{png,pdf}`: fraction of
  repeats in which each pair has negative cosine.
- `category_gradient_norms_all_trainable.{png,pdf}`: category gradient
  magnitudes, useful for diagnosing domination by one category.
- `pairwise_cosine_summary.csv`: paper-ready pair statistics, including mean,
  standard deviation, 95% interval, negative fraction, and a deterministic
  interpretation label.
- `pairwise_cosine_per_repeat.csv`: all underlying repeat observations.
- `cosine_matrix_mean_*.csv` and `cosine_matrix_std_*.csv`: numeric matrices
  for every parameter component (`all_trainable`, `lora`, and the geometry
  tokenizer when present).
- `run_metadata.json`: checkpoint identity, exact loss/configuration, semantic
  mapping, parameter counts, and every sampled example/seed.
- `supplementary_summary.md`: compact table that can be copied into the
  supplementary-material drafting workflow.

The calculation is exact in the selected parameter space: category gradients
are cached as float32 and dot products are accumulated as float64. The
temporary cache is roughly
`number_of_groups × 4 × trainable_parameter_count` bytes. For the ready-to-run
r=128 checkpoint above, one float32 gradient is 3.16 GiB and the five-group
cache is 15.82 GiB. It is streamed rather than held in RAM and deleted after
every repeat unless `--keep-gradient-cache` is supplied.

## Interpretation for the supplement

A negative cosine is the clearest evidence of interference: locally reducing
one category's loss moves parameters in a direction that increases the other
category's first-order loss. A cosine near zero indicates weak transfer or
nearly orthogonal objectives, but is not by itself proof of destructive
interference. Report the raw values and uncertainty instead of only the
script's `low_alignment` label.

A defensible wording template after inserting the measured values is:

> On the jointly trained checkpoint, category-conditioned gradients showed
> weak or conflicting alignment. In particular, the gradients for **A** and
> **B** had mean cosine similarity **x.xx** (95% CI **[l, u]**), and were
> negative in **p%** of independent sample repeats. This directly exposes
> cross-category interference in the shared parameter updates, motivating
> expert-preserving OPD rather than relying only on downstream metric gaps.

Do not claim statistically robust negative conflict when the interval crosses
zero. In that case, describe the pair as weakly aligned and use its negative
repeat fraction as descriptive evidence.

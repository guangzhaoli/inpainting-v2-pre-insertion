#!/usr/bin/env python3
"""Measure cross-category gradient conflict on a trained joint model.

This is a diagnostic training-mode pass: it runs the same
``prepare_embeddings``/``_compute_loss`` path used by ``training_step`` and its
backward graph, but never constructs an optimizer and never updates model
weights.  Category-mean gradients are written to temporary float32 memory maps
so cosine similarities are exact for the selected trainable parameters without
retaining one multi-GB vector per category in RAM.

The script is distributed-aware.  Under ``accelerate launch``, every rank sees
the same semantic group while processing a disjoint, equally sized sample
subset.  DDP therefore produces the exact mean gradient over all ranks.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
import zlib
from collections import OrderedDict, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOGGER = logging.getLogger("category_gradient_conflict")


# These names follow the five expert/data splits used by final_architecture_*.yaml.
# ``garment`` is the paper-facing name for the fashion-try-on split.
DEFAULT_CATEGORY_GROUPS: "OrderedDict[str, tuple[str, ...]]" = OrderedDict(
    [
        ("accessory", ("accessory", "jewel_filter")),
        (
            "garment",
            (
                "clothes",
                "dress",
                "TopRef_BodyMask",
                "BottomRef_BodyMask",
                "TopRef_body_man",
                "pure_color",
            ),
        ),
        ("human", ("person", "person_hair")),
        ("animal", ("cat", "dog")),
        ("general_object", ("others",)),
    ]
)


@dataclass(frozen=True)
class ParameterSegment:
    name: str
    offset: int
    numel: int
    component: str


def parse_group_specs(specs: Sequence[str]) -> "OrderedDict[str, tuple[str, ...]]":
    """Parse repeated ``NAME=category_a,category_b`` CLI specifications."""

    groups: "OrderedDict[str, tuple[str, ...]]" = OrderedDict()
    seen_categories: dict[str, str] = {}
    for raw_spec in specs:
        if "=" not in raw_spec:
            raise ValueError(
                f"Invalid --group {raw_spec!r}; expected NAME=category_a,category_b"
            )
        raw_name, raw_categories = raw_spec.split("=", 1)
        name = raw_name.strip()
        categories = tuple(
            category.strip()
            for category in raw_categories.split(",")
            if category.strip()
        )
        if not name:
            raise ValueError(f"Invalid --group {raw_spec!r}: group name is empty")
        if not categories:
            raise ValueError(f"Invalid --group {raw_spec!r}: category list is empty")
        if name in groups:
            raise ValueError(f"Duplicate group name: {name!r}")
        duplicate_categories = [category for category in categories if category in seen_categories]
        if duplicate_categories:
            owners = {category: seen_categories[category] for category in duplicate_categories}
            raise ValueError(
                f"Categories cannot belong to multiple groups; duplicate ownership: {owners}"
            )
        if len(set(categories)) != len(categories):
            raise ValueError(f"Group {name!r} contains duplicate categories: {categories}")
        groups[name] = categories
        for category in categories:
            seen_categories[category] = name
    if len(groups) < 2:
        raise ValueError("At least two category groups are required for cosine analysis")
    return groups


def _dataset_category(dataset: Any, index: int) -> str:
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, Sequence) and index < len(samples):
        sample = samples[index]
        if isinstance(sample, Mapping) and sample.get("category") not in (None, ""):
            return str(sample["category"])
    get_sample_info = getattr(dataset, "_get_sample_info", None)
    if callable(get_sample_info):
        sample = get_sample_info(index)
        if isinstance(sample, Mapping) and sample.get("category") not in (None, ""):
            return str(sample["category"])
    raise ValueError(
        "The configured dataset does not expose category metadata through "
        "dataset.samples or _get_sample_info(); category-gradient analysis requires it"
    )


def build_category_index(dataset: Any) -> dict[str, list[int]]:
    category_to_indices: dict[str, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        category_to_indices[_dataset_category(dataset, index)].append(index)
    return dict(category_to_indices)


def resolve_groups(
    available_categories: Iterable[str],
    *,
    custom_specs: Sequence[str] | None = None,
    preset_names: Sequence[str] | None = None,
) -> "OrderedDict[str, tuple[str, ...]]":
    available = set(available_categories)
    if custom_specs:
        if preset_names:
            raise ValueError("--group and --preset-groups cannot be used together")
        groups = parse_group_specs(custom_specs)
        missing = {
            name: [category for category in categories if category not in available]
            for name, categories in groups.items()
        }
        missing = {name: categories for name, categories in missing.items() if categories}
        if missing:
            raise ValueError(f"Requested group categories are absent from the dataset: {missing}")
        return groups

    if preset_names:
        unknown = [name for name in preset_names if name not in DEFAULT_CATEGORY_GROUPS]
        if unknown:
            raise ValueError(
                f"Unknown preset groups {unknown}; choices are {list(DEFAULT_CATEGORY_GROUPS)}"
            )
        source = OrderedDict((name, DEFAULT_CATEGORY_GROUPS[name]) for name in preset_names)
    else:
        source = DEFAULT_CATEGORY_GROUPS

    groups: "OrderedDict[str, tuple[str, ...]]" = OrderedDict()
    for name, expected_categories in source.items():
        present = tuple(category for category in expected_categories if category in available)
        if present:
            groups[name] = present
    if len(groups) < 2:
        raise ValueError(
            "The dataset resolves to fewer than two semantic groups. Use the joint-training "
            "manifest/checkpoint or provide explicit --group specifications. "
            f"Available categories: {sorted(available)}"
        )
    return groups


def _stable_name_offset(name: str) -> int:
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


def stratified_timestep_plan(samples_per_group: int, repeat_index: int, seed: int) -> np.ndarray:
    """Return a shuffled stratified-uniform ``u`` grid shared by every group."""

    rng = np.random.default_rng(seed + 104729 * (repeat_index + 1))
    bins = rng.permutation(samples_per_group)
    return (bins.astype(np.float64) + 0.5) / samples_per_group


def build_balanced_sample_plan(
    category_to_indices: Mapping[str, Sequence[int]],
    groups: Mapping[str, Sequence[str]],
    *,
    samples_per_group: int,
    repeats: int,
    seed: int,
) -> dict[str, list[list[int]]]:
    """Sample without replacement and balance native categories within each group."""

    if samples_per_group <= 0:
        raise ValueError("samples_per_group must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    plans: dict[str, list[list[int]]] = {}
    for group_name, categories_value in groups.items():
        categories = list(categories_value)
        if not categories:
            raise ValueError(f"Group {group_name!r} has no categories")

        shuffled: dict[str, list[int]] = {}
        for category in categories:
            indices = list(category_to_indices.get(category, ()))
            if not indices:
                raise ValueError(f"Group {group_name!r} category {category!r} has no samples")
            rng = np.random.default_rng(seed + _stable_name_offset(f"{group_name}/{category}"))
            rng.shuffle(indices)
            shuffled[category] = indices

        cursors = {category: 0 for category in categories}
        group_repeats: list[list[int]] = []
        for repeat_index in range(repeats):
            # Rotate the remainder assignment so non-divisible budgets do not
            # systematically favor the first native category across repeats.
            rotation = repeat_index % len(categories)
            category_order = categories[rotation:] + categories[:rotation]
            repeat_indices: list[int] = []
            for slot in range(samples_per_group):
                category = category_order[slot % len(category_order)]
                cursor = cursors[category]
                if cursor >= len(shuffled[category]):
                    required_by_category = {
                        cat: cursors[cat] + int(cat == category) for cat in categories
                    }
                    available_by_category = {
                        cat: len(shuffled[cat]) for cat in categories
                    }
                    raise ValueError(
                        f"Not enough unique samples for group {group_name!r}; "
                        f"required so far={required_by_category}, available={available_by_category}. "
                        "Reduce --samples-per-group or --repeats."
                    )
                repeat_indices.append(shuffled[category][cursor])
                cursors[category] += 1
            group_repeats.append(repeat_indices)
        plans[group_name] = group_repeats
    return plans


def sample_identifier(dataset: Any, index: int) -> str:
    category = _dataset_category(dataset, index)
    samples = getattr(dataset, "samples", None)
    if isinstance(samples, Sequence) and index < len(samples):
        sample = samples[index]
        if isinstance(sample, Mapping):
            stem = sample.get("stem")
            if stem not in (None, ""):
                return f"{category}/{stem}"
    return f"{category}/dataset-index-{index}"


def classify_parameter_component(name: str) -> str:
    lowered = name.lower()
    if "mask_geometry_tokenizer" in lowered:
        return "mask_geometry_tokenizer"
    if "lora_" in lowered or ".lora" in lowered:
        return "lora"
    return "other_trainable"


def build_parameter_layout(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[list[ParameterSegment], int]:
    segments: list[ParameterSegment] = []
    offset = 0
    for name, parameter in named_parameters:
        numel = int(parameter.numel())
        segments.append(
            ParameterSegment(
                name=name,
                offset=offset,
                numel=numel,
                component=classify_parameter_component(name),
            )
        )
        offset += numel
    return segments, offset


def write_gradient_memmap(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    path: Path,
    *,
    total_numel: int,
) -> dict[str, Any]:
    """Stream current gradients to a contiguous float32 memory map."""

    gradient_map = np.memmap(path, dtype=np.float32, mode="w+", shape=(total_numel,))
    offset = 0
    missing_names: list[str] = []
    nonfinite_names: list[str] = []
    for name, parameter in named_parameters:
        numel = int(parameter.numel())
        gradient = parameter.grad
        if gradient is None:
            gradient_map[offset : offset + numel] = 0.0
            missing_names.append(name)
        else:
            detached = gradient.detach().float()
            if not bool(torch.isfinite(detached).all().item()):
                nonfinite_names.append(name)
            cpu_gradient = detached.reshape(-1).cpu().numpy()
            gradient_map[offset : offset + numel] = cpu_gradient
        offset += numel
    gradient_map.flush()
    del gradient_map
    if offset != total_numel:
        raise RuntimeError(f"Gradient layout mismatch: wrote {offset}, expected {total_numel}")
    if nonfinite_names:
        raise FloatingPointError(
            f"Non-finite gradients in {len(nonfinite_names)} parameters; first={nonfinite_names[0]}"
        )
    return {
        "missing_gradient_tensors": len(missing_names),
        "first_missing_gradient": missing_names[0] if missing_names else None,
    }


def cosine_from_gram(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"Gram matrix must be square, got {gram.shape}")
    diagonal = np.diag(gram)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        raise ValueError(f"Every category gradient must have positive finite norm; gram diagonal={diagonal}")
    norms = np.sqrt(diagonal)
    cosine = gram / np.outer(norms, norms)
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    return cosine, norms


def compute_component_grams(
    gradient_paths: Mapping[str, Path],
    segments: Sequence[ParameterSegment],
    *,
    total_numel: int,
    workspace_mb: int = 256,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Compute exact float64 Gram matrices by streaming memory-mapped gradients."""

    if workspace_mb <= 0:
        raise ValueError("workspace_mb must be positive")
    group_names = list(gradient_paths)
    if len(group_names) < 2:
        raise ValueError("At least two gradient files are required")

    maps = {
        name: np.memmap(path, dtype=np.float32, mode="r", shape=(total_numel,))
        for name, path in gradient_paths.items()
    }
    components = sorted({segment.component for segment in segments})
    component_grams = {
        component: np.zeros((len(group_names), len(group_names)), dtype=np.float64)
        for component in components
    }
    # Account for both the temporary float32 stack and its float64 conversion.
    bytes_per_element_across_groups = max(1, len(group_names) * (4 + 8))
    chunk_elements = max(
        1,
        int(workspace_mb * 1024 * 1024 // bytes_per_element_across_groups),
    )

    try:
        for segment in segments:
            segment_end = segment.offset + segment.numel
            for start in range(segment.offset, segment_end, chunk_elements):
                end = min(start + chunk_elements, segment_end)
                float32_block = np.stack(
                    [np.asarray(maps[name][start:end]) for name in group_names],
                    axis=0,
                )
                float64_block = float32_block.astype(np.float64)
                component_grams[segment.component] += float64_block @ float64_block.T
                del float32_block, float64_block
    finally:
        maps.clear()

    component_grams["all_trainable"] = sum(
        component_grams[component] for component in components
    )
    ordered = {"all_trainable": component_grams.pop("all_trainable")}
    ordered.update(component_grams)
    return group_names, ordered


def _student_t_critical_975(degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return math.nan
    try:
        from scipy.stats import t as student_t

        return float(student_t.ppf(0.975, degrees_of_freedom))
    except Exception:
        return 1.96


def summarize_pairwise_cosines(
    repeat_cosines: Mapping[str, Sequence[np.ndarray]],
    group_names: Sequence[str],
    *,
    low_similarity_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scope, matrices in repeat_cosines.items():
        if not matrices:
            continue
        for left_index, left_name in enumerate(group_names):
            for right_index in range(left_index + 1, len(group_names)):
                right_name = group_names[right_index]
                values = np.asarray(
                    [matrix[left_index, right_index] for matrix in matrices],
                    dtype=np.float64,
                )
                mean = float(values.mean())
                if len(values) > 1:
                    std = float(values.std(ddof=1))
                    sem = std / math.sqrt(len(values))
                    half_width = _student_t_critical_975(len(values) - 1) * sem
                    ci_low = mean - half_width
                    ci_high = mean + half_width
                else:
                    std = math.nan
                    sem = math.nan
                    ci_low = math.nan
                    ci_high = math.nan

                if len(values) > 1 and ci_high < 0:
                    interpretation = "robust_negative"
                elif mean < 0:
                    interpretation = "negative_mean"
                elif mean < low_similarity_threshold:
                    interpretation = "low_alignment"
                else:
                    interpretation = "aligned"
                records.append(
                    {
                        "scope": scope,
                        "category_a": left_name,
                        "category_b": right_name,
                        "n_repeats": int(len(values)),
                        "mean_cosine": mean,
                        "std_cosine": std,
                        "sem_cosine": sem,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "negative_fraction": float(np.mean(values < 0)),
                        "low_similarity_fraction": float(
                            np.mean(values < low_similarity_threshold)
                        ),
                        "min_cosine": float(values.min()),
                        "max_cosine": float(values.max()),
                        "interpretation": interpretation,
                    }
                )
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_matrix_csv(path: Path, group_names: Sequence[str], matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", *group_names])
        for name, row in zip(group_names, matrix):
            writer.writerow([name, *[f"{float(value):.10f}" for value in row]])


def _safe_scope_name(scope: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in scope)


def render_heatmap(
    matrix: np.ndarray,
    group_names: Sequence[str],
    output_base: Path,
    *,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    value_format: str = ".2f",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max(5.2, 0.85 * len(group_names) + 2.2)
    fig, axis = plt.subplots(figsize=(size, size * 0.9), constrained_layout=True)
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(group_names)), labels=group_names, rotation=35, ha="right")
    axis.set_yticks(range(len(group_names)), labels=group_names)
    axis.set_title(title)
    for row in range(len(group_names)):
        for column in range(len(group_names)):
            value = float(matrix[row, column])
            normalized = 0.0 if vmax == vmin else 2.0 * (value - vmin) / (vmax - vmin) - 1.0
            text_color = "white" if abs(normalized) > 0.64 else "black"
            axis.text(
                column,
                row,
                format(value, value_format),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label(colorbar_label)
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None)
    plt.close(fig)


def render_pairwise_intervals(
    records: Sequence[Mapping[str, Any]],
    output_base: Path,
    *,
    scope: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [record for record in records if record["scope"] == scope]
    selected.sort(key=lambda record: float(record["mean_cosine"]))
    if not selected:
        return
    labels = [f"{record['category_a']} vs {record['category_b']}" for record in selected]
    means = np.asarray([record["mean_cosine"] for record in selected], dtype=float)
    lows = np.asarray([record["ci95_low"] for record in selected], dtype=float)
    highs = np.asarray([record["ci95_high"] for record in selected], dtype=float)
    has_intervals = np.all(np.isfinite(lows)) and np.all(np.isfinite(highs))

    fig_height = max(4.0, 0.42 * len(selected) + 1.8)
    fig, axis = plt.subplots(figsize=(8.2, fig_height), constrained_layout=True)
    positions = np.arange(len(selected))
    colors = ["#b2182b" if mean < 0 else "#2166ac" for mean in means]
    if has_intervals:
        errors = np.vstack([means - lows, highs - means])
        for position, mean, error, color in zip(positions, means, errors.T, colors):
            axis.errorbar(
                mean,
                position,
                xerr=np.asarray(error).reshape(2, 1),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
            )
    else:
        axis.scatter(means, positions, c=colors, s=34)
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlim(-1.02, 1.02)
    axis.set_yticks(positions, labels=labels)
    axis.set_xlabel("Gradient cosine similarity (mean and 95% CI)")
    axis.set_title(f"Cross-category gradient alignment — {scope}")
    axis.grid(axis="x", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None)
    plt.close(fig)


def render_gradient_norms(
    norm_records: Sequence[Mapping[str, Any]],
    output_base: Path,
    *,
    scope: str,
    group_names: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        [
            float(record["gradient_norm"])
            for record in norm_records
            if record["scope"] == scope and record["category_group"] == group
        ]
        for group in group_names
    ]
    if not all(values):
        return
    fig, axis = plt.subplots(figsize=(max(6.0, len(group_names) * 1.2), 4.8), constrained_layout=True)
    axis.boxplot(values, tick_labels=group_names, showmeans=True)
    axis.set_yscale("log")
    axis.set_ylabel("L2 norm of category-mean gradient (log scale)")
    axis.set_title(f"Category gradient magnitudes — {scope}")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    for suffix in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{suffix}"), dpi=300 if suffix == "png" else None)
    plt.close(fig)


def write_analysis_outputs(
    output_dir: Path,
    *,
    group_names: Sequence[str],
    repeat_cosines: Mapping[str, Sequence[np.ndarray]],
    repeat_norms: Mapping[str, Sequence[np.ndarray]],
    pairwise_records: Sequence[Mapping[str, Any]],
    low_similarity_threshold: float,
) -> None:
    scopes = list(repeat_cosines)
    all_repeat_rows: list[dict[str, Any]] = []
    norm_rows: list[dict[str, Any]] = []
    for scope in scopes:
        matrices = repeat_cosines[scope]
        mean_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
        std_matrix = (
            np.std(np.stack(matrices, axis=0), axis=0, ddof=1)
            if len(matrices) > 1
            else np.zeros_like(mean_matrix)
        )
        negative_fraction_matrix = np.mean(np.stack(matrices, axis=0) < 0, axis=0)
        np.fill_diagonal(negative_fraction_matrix, 0.0)
        safe_scope = _safe_scope_name(scope)
        _write_matrix_csv(
            output_dir / f"cosine_matrix_mean_{safe_scope}.csv",
            group_names,
            mean_matrix,
        )
        _write_matrix_csv(
            output_dir / f"cosine_matrix_std_{safe_scope}.csv",
            group_names,
            std_matrix,
        )
        _write_matrix_csv(
            output_dir / f"negative_fraction_matrix_{safe_scope}.csv",
            group_names,
            negative_fraction_matrix,
        )
        render_heatmap(
            mean_matrix,
            group_names,
            output_dir / f"category_gradient_cosine_heatmap_{safe_scope}",
            title=f"Category gradient cosine similarity — {scope}",
            colorbar_label="Mean cosine similarity",
            vmin=-1.0,
            vmax=1.0,
        )
        render_heatmap(
            negative_fraction_matrix,
            group_names,
            output_dir / f"category_gradient_negative_fraction_{safe_scope}",
            title=f"Fraction of repeats with negative cosine — {scope}",
            colorbar_label="Negative fraction",
            vmin=0.0,
            vmax=1.0,
        )
        render_pairwise_intervals(
            pairwise_records,
            output_dir / f"pairwise_cosine_intervals_{safe_scope}",
            scope=scope,
        )
        render_gradient_norms(
            [
                {
                    "scope": scope_name,
                    "repeat": repeat_index,
                    "category_group": group_names[group_index],
                    "gradient_norm": float(norm_vector[group_index]),
                }
                for scope_name, norm_vectors in repeat_norms.items()
                for repeat_index, norm_vector in enumerate(norm_vectors)
                for group_index in range(len(group_names))
            ],
            output_dir / f"category_gradient_norms_{safe_scope}",
            scope=scope,
            group_names=group_names,
        )

        for repeat_index, matrix in enumerate(matrices):
            for left_index, left_name in enumerate(group_names):
                for right_index in range(left_index + 1, len(group_names)):
                    all_repeat_rows.append(
                        {
                            "scope": scope,
                            "repeat": repeat_index,
                            "category_a": left_name,
                            "category_b": group_names[right_index],
                            "cosine_similarity": float(matrix[left_index, right_index]),
                        }
                    )
        for repeat_index, norm_vector in enumerate(repeat_norms[scope]):
            for group_index, group_name in enumerate(group_names):
                norm_rows.append(
                    {
                        "scope": scope,
                        "repeat": repeat_index,
                        "category_group": group_name,
                        "gradient_norm": float(norm_vector[group_index]),
                    }
                )

    pair_fields = [
        "scope",
        "category_a",
        "category_b",
        "n_repeats",
        "mean_cosine",
        "std_cosine",
        "sem_cosine",
        "ci95_low",
        "ci95_high",
        "negative_fraction",
        "low_similarity_fraction",
        "min_cosine",
        "max_cosine",
        "interpretation",
    ]
    _write_csv(output_dir / "pairwise_cosine_summary.csv", pairwise_records, pair_fields)
    _write_csv(
        output_dir / "pairwise_cosine_per_repeat.csv",
        all_repeat_rows,
        ["scope", "repeat", "category_a", "category_b", "cosine_similarity"],
    )
    _write_csv(
        output_dir / "gradient_norms.csv",
        norm_rows,
        ["scope", "repeat", "category_group", "gradient_norm"],
    )

    primary_records = [
        record for record in pairwise_records if record["scope"] == "all_trainable"
    ]
    primary_records.sort(key=lambda record: float(record["mean_cosine"]))
    with (output_dir / "supplementary_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Cross-category gradient interference\n\n")
        handle.write(
            "Each entry is the cosine similarity between category-mean gradients of the "
            "jointly trained model. Negative values are directly conflicting update "
            "directions; near-zero values indicate weak alignment.\n\n"
        )
        handle.write(
            f"The predeclared low-alignment threshold is {low_similarity_threshold:.3f}. "
            "Confidence intervals are Student-t intervals over independent sample repeats.\n\n"
        )
        handle.write("| Category pair | Mean cosine | 95% CI | Negative repeats | Label |\n")
        handle.write("|---|---:|---:|---:|---|\n")
        for record in primary_records:
            if math.isfinite(float(record["ci95_low"])):
                interval = f"[{record['ci95_low']:.3f}, {record['ci95_high']:.3f}]"
            else:
                interval = "n/a (one repeat)"
            handle.write(
                f"| {record['category_a']} / {record['category_b']} "
                f"| {record['mean_cosine']:.3f} | {interval} "
                f"| {record['negative_fraction']:.1%} | {record['interpretation']} |\n"
            )


def resolve_checkpoint_weight(checkpoint: Path) -> tuple[Path, Path]:
    """Return ``(weight_file, checkpoint_directory)`` with deterministic priority."""

    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.is_file():
        if checkpoint.suffix != ".safetensors":
            raise ValueError(f"Checkpoint file must be .safetensors, got {checkpoint}")
        return checkpoint, checkpoint.parent
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    for filename in ("model.safetensors", "pytorch_lora_weights.safetensors"):
        candidate = checkpoint / filename
        if candidate.is_file():
            return candidate, checkpoint
    candidates = sorted(
        path
        for path in checkpoint.glob("*.safetensors")
        if path.name != "mask_geometry_tokenizer.safetensors"
    )
    if len(candidates) == 1:
        return candidates[0], checkpoint
    raise FileNotFoundError(
        f"Could not resolve LoRA weights in {checkpoint}; expected model.safetensors "
        "or pytorch_lora_weights.safetensors"
    )


def find_checkpoint_train_config(checkpoint_dir: Path) -> Path | None:
    """Support both new per-checkpoint configs and legacy run-level configs."""

    for candidate in (
        checkpoint_dir / "train_config.yaml",
        checkpoint_dir.parent / "train_config.yaml",
    ):
        if candidate.is_file():
            return candidate
    return None


def resolve_config_path(config_arg: str | None, checkpoint_dir: Path) -> Path:
    if config_arg:
        config_path = Path(config_arg).expanduser().resolve()
    else:
        discovered = find_checkpoint_train_config(checkpoint_dir)
        config_path = discovered if discovered is not None else checkpoint_dir / "train_config.yaml"
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            f"Training config not found: {config_path}. Pass --config explicitly."
        )
    return config_path


def load_analysis_config(
    config_path: Path,
    *,
    clear_predict_devices: bool = False,
) -> Any:
    """Load source YAML or the trainer's computed-field-bearing saved YAML."""

    from omegaconf import OmegaConf

    from src.data.config import Config

    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    # BaseTrainer.save_train_config uses model_dump(), which includes computed
    # fields. They are useful metadata but are forbidden as Config inputs.
    for computed_name in Config.model_computed_fields:
        raw.pop(computed_name, None)
    # The checkpoint under analysis is supplied explicitly. Saved warm-start
    # and resume paths are provenance, not dependencies, and commonly become
    # stale after a run directory is moved.
    raw["resume"] = None
    model_raw = raw.get("model")
    if isinstance(model_raw, dict) and isinstance(model_raw.get("lora"), dict):
        model_raw["lora"]["pretrained_weight"] = None
    if clear_predict_devices:
        raw["predict"] = {**(raw.get("predict") or {}), "devices": {}}
    if bool((raw.get("distillation") or {}).get("enabled", False)):
        raw["distillation"] = {"enabled": False}
    return Config.model_validate(raw)


def _trainer_name(config: Any) -> str:
    return str(getattr(config.trainer, "value", config.trainer))


def _architecture_signature(config: Any) -> dict[str, Any]:
    """Fields that must agree for checkpoint weights and the analysis graph."""

    lora = config.model.lora
    target_modules = lora.target_modules
    if isinstance(target_modules, str):
        normalized_targets: str | list[str] = target_modules
    else:
        normalized_targets = sorted(str(value) for value in target_modules)
    return {
        "trainer": _trainer_name(config),
        "lora_r": int(lora.r),
        "lora_alpha": int(lora.lora_alpha),
        "lora_target_modules": normalized_targets,
        "mask_control": config.model.mask_control.model_dump(mode="json"),
        "reference_cfg": config.model.reference_cfg.model_dump(mode="json"),
        "processor": config.data.init_args.processor.model_dump(mode="json"),
        "adaptive_crop": config.data.init_args.adaptive_crop.model_dump(mode="json"),
        "hybrid_masking": config.data.init_args.hybrid_masking.model_dump(mode="json"),
    }


def validate_joint_checkpoint_provenance(
    checkpoint_dir: Path,
    active_config: Any,
    groups: Mapping[str, Sequence[str]],
    *,
    compare_loss: bool,
) -> dict[str, Any]:
    """Reject category-specific/OPD checkpoints when provenance is available."""

    checkpoint_config_path = find_checkpoint_train_config(checkpoint_dir)
    if checkpoint_config_path is None:
        LOGGER.warning(
            "Checkpoint has no train_config.yaml; joint-training provenance cannot be verified"
        )
        return {
            "verified": False,
            "reason": "checkpoint train_config.yaml is absent",
        }

    import yaml

    raw = yaml.safe_load(checkpoint_config_path.read_text(encoding="utf-8")) or {}
    if bool((raw.get("distillation") or {}).get("enabled", False)):
        raise ValueError(
            f"{checkpoint_config_path} identifies an OPD/distilled checkpoint. This experiment "
            "must measure the ordinary jointly supervised model that motivates OPD."
        )

    # Prediction devices and Teacher dependency paths are irrelevant to this
    # provenance comparison and may be unavailable on a login/test node.
    checkpoint_config = load_analysis_config(
        checkpoint_config_path,
        clear_predict_devices=True,
    )
    active_signature = _architecture_signature(active_config)
    checkpoint_signature = _architecture_signature(checkpoint_config)
    mismatches = {
        key: {"analysis": active_signature[key], "checkpoint": checkpoint_signature[key]}
        for key in active_signature
        if active_signature[key] != checkpoint_signature[key]
    }
    if compare_loss:
        active_loss = active_config.loss.model_dump(mode="json")
        checkpoint_loss = checkpoint_config.loss.model_dump(mode="json")
        if active_loss != checkpoint_loss:
            mismatches["loss"] = {
                "analysis": active_loss,
                "checkpoint": checkpoint_loss,
            }
    if mismatches:
        raise ValueError(
            "Analysis config is incompatible with the checkpoint training graph. "
            f"Mismatched fields: {json.dumps(mismatches, ensure_ascii=False)}"
        )

    configured_categories = checkpoint_config.data.init_args.categories
    if isinstance(configured_categories, str):
        checkpoint_categories = {configured_categories}
    elif configured_categories:
        checkpoint_categories = {str(value) for value in configured_categories}
    else:
        checkpoint_categories = set()

    if checkpoint_categories:
        missing_by_group = {
            group_name: [
                category for category in categories if category not in checkpoint_categories
            ]
            for group_name, categories in groups.items()
        }
        missing_by_group = {
            group_name: missing
            for group_name, missing in missing_by_group.items()
            if missing
        }
        if missing_by_group:
            raise ValueError(
                "The checkpoint training config did not include every requested analysis "
                f"category; this is not the required joint checkpoint: {missing_by_group}"
            )

    return {
        "verified": True,
        "checkpoint_train_config": str(checkpoint_config_path),
        "distillation_enabled": False,
        "configured_categories": sorted(checkpoint_categories),
        "architecture_signature_match": True,
        "loss_signature_match": bool(compare_loss),
    }


def validate_checkpoint_companion(config: Any, checkpoint_dir: Path) -> None:
    mask_control = config.model.mask_control
    geometry_enabled = bool(mask_control.enabled) and str(mask_control.encoding) == "geometry_tokenizer"
    if geometry_enabled:
        companion = checkpoint_dir / "mask_geometry_tokenizer.safetensors"
        if not companion.is_file():
            raise FileNotFoundError(
                "The model uses geometry-tokenizer mask control, but its checkpoint companion "
                f"is missing: {companion}"
            )


def instantiate_dataset(config: Any) -> Any:
    module_name, class_name = config.data.class_path.rsplit(".", 1)
    dataset_class = getattr(importlib.import_module(module_name), class_name)
    return dataset_class(config.data.init_args)


def import_trainer_class(config: Any) -> type:
    trainer_name = str(getattr(config.trainer, "value", config.trainer))
    if trainer_name == "QwenImageEdit":
        from src.trainer.qwen_image_edit_trainer import QwenImageEditTrainer

        return QwenImageEditTrainer
    if trainer_name == "QwenImageEditPlus":
        from src.trainer.qwen_image_edit_plus_trainer import QwenImageEditPlusTrainer

        return QwenImageEditPlusTrainer
    if trainer_name == "QwenImageEditInsertAnything":
        from src.trainer.qwen_image_edit_insert_anything_trainer import (
            QwenImageEditInsertAnythingTrainer,
        )

        return QwenImageEditInsertAnythingTrainer
    raise ValueError(f"Unsupported trainer: {trainer_name}")


def configure_file_logging(output_dir: Path, rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(name)s | %(message)s",
    )
    if rank == 0:
        file_handler = logging.FileHandler(output_dir / "analysis.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def _checkpoint_sha256_from_manifest(checkpoint_dir: Path, weight_file: Path) -> str | None:
    state_path = checkpoint_dir / "state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        files = state.get("artifacts", {}).get("files", [])
        for artifact in files:
            if artifact.get("filename") == weight_file.name:
                return artifact.get("sha256")
    except Exception:
        return None
    return None


def _read_checkpoint_state(checkpoint_dir: Path) -> dict[str, Any] | None:
    state_path = checkpoint_dir / "state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not parse checkpoint state %s: %s", state_path, exc)
        return None
    return state if isinstance(state, dict) else None


def _sha256_file_if_present(path_value: str | Path | None) -> str | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if not path.is_file():
        return None
    from src.utils.tools import calculate_sha256_file

    return calculate_sha256_file(str(path))


def _broadcast_object(accelerator: Any, value: Any) -> Any:
    from accelerate.utils import broadcast_object_list

    payload = [value]
    broadcast_object_list(payload, from_process=0)
    return payload[0]


def prepare_output_directory(accelerator: Any, output_arg: str | None, overwrite: bool) -> Path:
    status: dict[str, str | None]
    if accelerator.is_main_process:
        if output_arg:
            output_dir = Path(output_arg).expanduser().resolve()
        else:
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            output_dir = (Path.cwd() / "outputs" / "category_gradient_conflict" / timestamp).resolve()
        error = None
        if (output_dir / "run_complete.json").exists() and not overwrite:
            error = (
                f"Completed output already exists at {output_dir}; choose another --output-dir "
                "or pass --overwrite"
            )
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        status = {"path": str(output_dir), "error": error}
    else:
        status = {"path": None, "error": None}
    status = _broadcast_object(accelerator, status)
    if status["error"]:
        raise FileExistsError(status["error"])
    output_dir = Path(str(status["path"]))
    accelerator.wait_for_everyone()
    return output_dir


def zero_trainable_gradients(named_parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    for _, parameter in named_parameters:
        parameter.grad = None


def run_analysis(args: argparse.Namespace) -> Path:
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from src.data.config import TrMode
    from src.data.dataset import collate_fn
    from src.utils.opencv_runtime import configure_opencv_runtime
    from src.utils.seed import seed_everything

    configure_opencv_runtime()
    checkpoint_weight, checkpoint_dir = resolve_checkpoint_weight(Path(args.checkpoint))
    config_path = resolve_config_path(args.config, checkpoint_dir)
    # Saved OPD dependencies are disabled while loading; the analysis is then
    # forced into fit mode and explicitly evaluates the supervised loss graph.
    config = load_analysis_config(config_path, clear_predict_devices=True)
    config.mode = TrMode.fit
    config.resume = None
    config.model.lora.pretrained_weight = str(checkpoint_weight)
    config.logging.report_to = "none"
    config.train.gradient_accumulation_steps = 1
    config.data.batch_size = 1
    if args.latent_only:
        config.loss.region_perceptual = False
    if args.disable_gradient_checkpointing:
        config.train.gradient_checkpointing = False
    if args.dataset_path:
        config.data.init_args.dataset_path = str(Path(args.dataset_path).expanduser().resolve())
    if args.manifest_path:
        manifest_override = Path(args.manifest_path).expanduser().resolve()
        if not manifest_override.is_file():
            raise FileNotFoundError(f"Manifest override does not exist: {manifest_override}")
        config.data.init_args.manifest_path = str(manifest_override)
    validate_checkpoint_companion(config, checkpoint_dir)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=config.train.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            "Category-gradient analysis requires a CUDA allocation; no CUDA device is visible. "
            "Run the launcher on a GPU node."
        )
    output_dir = prepare_output_directory(accelerator, args.output_dir, args.overwrite)
    configure_file_logging(output_dir, accelerator.process_index)

    if args.samples_per_group % accelerator.num_processes != 0:
        raise ValueError(
            f"--samples-per-group ({args.samples_per_group}) must be divisible by the "
            f"number of processes ({accelerator.num_processes}) so every DDP rank runs "
            "the same number of backward passes"
        )

    seed_everything(args.seed + accelerator.process_index)
    dataset = instantiate_dataset(config)
    category_to_indices = build_category_index(dataset)
    groups = resolve_groups(
        category_to_indices,
        custom_specs=args.group,
        preset_names=args.preset_groups,
    )
    sample_plan = build_balanced_sample_plan(
        category_to_indices,
        groups,
        samples_per_group=args.samples_per_group,
        repeats=args.repeats,
        seed=args.seed,
    )
    group_names = list(groups)
    checkpoint_provenance = validate_joint_checkpoint_provenance(
        checkpoint_dir,
        config,
        groups,
        compare_loss=not args.latent_only,
    )

    LOGGER.info("Resolved semantic groups: %s", dict(groups))
    LOGGER.info(
        "Analysis budget: repeats=%d, samples/group/repeat=%d, processes=%d",
        args.repeats,
        args.samples_per_group,
        accelerator.num_processes,
    )

    checkpoint_sha256 = _checkpoint_sha256_from_manifest(checkpoint_dir, checkpoint_weight)
    if accelerator.is_main_process and checkpoint_sha256 is None:
        LOGGER.info("Checkpoint manifest has no SHA256; hashing %s once", checkpoint_weight)
        checkpoint_sha256 = _sha256_file_if_present(checkpoint_weight)
    checkpoint_sha256 = _broadcast_object(
        accelerator,
        checkpoint_sha256 if accelerator.is_main_process else None,
    )
    manifest_sha256 = None
    if accelerator.is_main_process:
        manifest_sha256 = _sha256_file_if_present(config.data.init_args.manifest_path)
    manifest_sha256 = _broadcast_object(
        accelerator,
        manifest_sha256 if accelerator.is_main_process else None,
    )
    reproducibility_info = None
    if accelerator.is_main_process:
        from src.utils.tools import get_git_info

        checkpoint_state = _read_checkpoint_state(checkpoint_dir)
        git_info = get_git_info()
        reproducibility_info = {
            "checkpoint_state": checkpoint_state,
            "analysis_git": {
                "commit": git_info.get("commit"),
                "short_commit": git_info.get("short_commit"),
                "branch": git_info.get("branch"),
            },
            "analysis_config_sha256": _sha256_file_if_present(config_path),
            "analysis_script_sha256": _sha256_file_if_present(Path(__file__).resolve()),
            "trainer_source_sha256": _sha256_file_if_present(
                Path(__file__).resolve().parents[1] / "src" / "trainer" / "qwen_image_edit_trainer.py"
            ),
        }
        checkpoint_commit = (checkpoint_state or {}).get("commit")
        analysis_commit = git_info.get("commit")
        if checkpoint_commit and analysis_commit and checkpoint_commit != analysis_commit:
            LOGGER.warning(
                "Checkpoint commit %s differs from analysis commit %s; source hashes are recorded "
                "in run_metadata.json. Verify that intervening source changes do not alter the "
                "enabled supervised path before using the result in a paper.",
                checkpoint_commit,
                analysis_commit,
            )
    reproducibility_info = _broadcast_object(
        accelerator,
        reproducibility_info if accelerator.is_main_process else None,
    )

    Trainer = import_trainer_class(config)
    trainer = Trainer(config)
    trainer.accelerator = accelerator
    trainer.weight_dtype = trainer._resolve_weight_dtype(accelerator.mixed_precision)
    trainer.load_model()
    if config.model.quantize:
        trainer.dit = trainer.quantize_model(trainer.dit, accelerator.device)
    Trainer.load_pretrain_lora_model(
        trainer.dit,
        config,
        trainer.adapter_name,
        stage="fit",
    )
    trainer.load_additional_modules(stage="fit")
    trainer.setup_model_device_train_mode(stage="fit")
    trainer.setup_criterion()
    if config.train.gradient_checkpointing:
        trainer.dit.enable_gradient_checkpointing()
    trainer.dit = accelerator.prepare(trainer.dit)

    unwrapped_dit = accelerator.unwrap_model(trainer.dit)
    named_parameters = [
        (name, parameter)
        for name, parameter in unwrapped_dit.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("No trainable parameters were found after loading the joint checkpoint")
    segments, total_numel = build_parameter_layout(named_parameters)
    component_numel: dict[str, int] = defaultdict(int)
    for segment in segments:
        component_numel[segment.component] += segment.numel
    LOGGER.info(
        "Gradient vector: tensors=%d, elements=%d, components=%s",
        len(named_parameters),
        total_numel,
        dict(component_numel),
    )
    cache_repeats_on_disk = args.repeats if args.keep_gradient_cache else 1
    estimated_cache_bytes = (
        total_numel * np.dtype(np.float32).itemsize * len(group_names) * cache_repeats_on_disk
    )
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < estimated_cache_bytes:
        raise OSError(
            "Insufficient free space for exact gradient cache: "
            f"need about {estimated_cache_bytes / 2**30:.1f} GiB, "
            f"available {free_bytes / 2**30:.1f} GiB at {output_dir}"
        )
    LOGGER.info(
        "Exact gradient cache estimate: %.1f GiB (free: %.1f GiB)",
        estimated_cache_bytes / 2**30,
        free_bytes / 2**30,
    )

    repeat_cosines: dict[str, list[np.ndarray]] = defaultdict(list)
    repeat_norms: dict[str, list[np.ndarray]] = defaultdict(list)
    invalid_component_scopes: set[str] = set()
    loss_records: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    cache_dir: Path | None = None
    success = False
    if accelerator.is_main_process:
        cache_dir = Path(tempfile.mkdtemp(prefix=".gradient_cache_", dir=output_dir))
    cache_dir_value = _broadcast_object(accelerator, str(cache_dir) if cache_dir else None)
    cache_dir = Path(str(cache_dir_value))

    try:
        local_steps = args.samples_per_group // accelerator.num_processes
        for repeat_index in range(args.repeats):
            gradient_paths: "OrderedDict[str, Path]" = OrderedDict()
            timestep_u_by_slot = stratified_timestep_plan(
                args.samples_per_group,
                repeat_index,
                args.seed,
            )
            LOGGER.info("Starting repeat %d/%d", repeat_index + 1, args.repeats)
            for group_index, group_name in enumerate(group_names):
                zero_trainable_gradients(named_parameters)
                full_indices = sample_plan[group_name][repeat_index]
                local_indices = full_indices[
                    accelerator.process_index :: accelerator.num_processes
                ]
                if len(local_indices) != local_steps:
                    raise RuntimeError(
                        f"Rank partition mismatch for {group_name}: {len(local_indices)} vs {local_steps}"
                    )

                local_loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
                for local_step, dataset_index in enumerate(local_indices):
                    global_slot = accelerator.process_index + local_step * accelerator.num_processes
                    stochastic_seed = (
                        args.seed
                        + repeat_index * args.samples_per_group
                        + global_slot
                    )
                    # Matching this seed by slot across groups controls timestep,
                    # diffusion noise, reference dropout, and MaskMix randomness.
                    # Delay DDP synchronization until the last local sample.
                    # Each rank accumulates a local mean, then the final all-reduce
                    # averages those equal-size means into the global group mean.
                    sync_context = (
                        nullcontext()
                        if local_step == local_steps - 1
                        else accelerator.no_sync(trainer.dit)
                    )
                    with sync_context:
                        seed_everything(stochastic_seed)
                        sample = dataset[dataset_index]
                        batch = collate_fn([sample])
                        embeddings = trainer.prepare_embeddings(batch, stage="fit")
                        embeddings["_analysis_noise_seed"] = stochastic_seed + 1_000_003
                        embeddings["_analysis_timestep_u"] = float(
                            timestep_u_by_slot[global_slot]
                        )
                        loss = trainer._compute_loss(embeddings)
                        if loss.ndim != 0:
                            loss = loss.mean()
                        if not bool(torch.isfinite(loss.detach()).item()):
                            raise FloatingPointError(
                                f"Non-finite loss for group={group_name}, repeat={repeat_index}, "
                                f"dataset_index={dataset_index}"
                            )
                        local_loss_sum += loss.detach().double()
                        accelerator.backward(loss / local_steps)

                gathered_loss_sums = accelerator.gather(local_loss_sum.reshape(1))
                mean_loss = float(gathered_loss_sums.sum().item() / args.samples_per_group)
                loss_records.append(
                    {
                        "repeat": repeat_index,
                        "category_group": group_name,
                        "mean_loss": mean_loss,
                    }
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    gradient_path = cache_dir / f"repeat-{repeat_index:03d}-{group_index:03d}.f32.mmap"
                    gradient_info = write_gradient_memmap(
                        named_parameters,
                        gradient_path,
                        total_numel=total_numel,
                    )
                    gradient_paths[group_name] = gradient_path
                    gradient_records.append(
                        {
                            "repeat": repeat_index,
                            "category_group": group_name,
                            **gradient_info,
                        }
                    )
                    LOGGER.info(
                        "Collected group=%s repeat=%d mean_loss=%.6f",
                        group_name,
                        repeat_index,
                        mean_loss,
                    )
                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                ordered_names, grams = compute_component_grams(
                    gradient_paths,
                    segments,
                    total_numel=total_numel,
                    workspace_mb=args.dot_workspace_mb,
                )
                if ordered_names != group_names:
                    raise RuntimeError(f"Group order changed: {ordered_names} vs {group_names}")
                for scope, gram in grams.items():
                    if scope in invalid_component_scopes:
                        continue
                    try:
                        cosine, norms = cosine_from_gram(gram)
                    except ValueError:
                        if scope == "all_trainable":
                            raise
                        invalid_component_scopes.add(scope)
                        repeat_cosines.pop(scope, None)
                        repeat_norms.pop(scope, None)
                        LOGGER.warning(
                            "Skipping component scope %s because at least one category has "
                            "a zero/non-finite gradient norm",
                            scope,
                        )
                        continue
                    repeat_cosines[scope].append(cosine)
                    repeat_norms[scope].append(norms)
                if not args.keep_gradient_cache:
                    for gradient_path in gradient_paths.values():
                        gradient_path.unlink()
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            pairwise_records = summarize_pairwise_cosines(
                repeat_cosines,
                group_names,
                low_similarity_threshold=args.low_similarity_threshold,
            )
            write_analysis_outputs(
                output_dir,
                group_names=group_names,
                repeat_cosines=repeat_cosines,
                repeat_norms=repeat_norms,
                pairwise_records=pairwise_records,
                low_similarity_threshold=args.low_similarity_threshold,
            )
            _write_csv(
                output_dir / "category_losses.csv",
                loss_records,
                ["repeat", "category_group", "mean_loss"],
            )
            _write_csv(
                output_dir / "gradient_collection_diagnostics.csv",
                gradient_records,
                [
                    "repeat",
                    "category_group",
                    "missing_gradient_tensors",
                    "first_missing_gradient",
                ],
            )
            sample_plan_json = {
                group: [
                    [
                        {
                            "dataset_index": index,
                            "sample_id": sample_identifier(dataset, index),
                            "native_category": _dataset_category(dataset, index),
                            "stochastic_seed": args.seed
                            + repeat_index * args.samples_per_group
                            + slot,
                        }
                        for slot, index in enumerate(indices)
                    ]
                    for repeat_index, indices in enumerate(group_repeats)
                ]
                for group, group_repeats in sample_plan.items()
            }
            metadata = {
                "analysis": "cross_category_gradient_cosine",
                "no_optimizer_steps": True,
                "config": str(config_path),
                "analysis_dataset_path": config.data.init_args.dataset_path,
                "analysis_manifest_path": config.data.init_args.manifest_path,
                "analysis_manifest_sha256": manifest_sha256,
                "checkpoint": str(checkpoint_dir),
                "checkpoint_weight": str(checkpoint_weight),
                "checkpoint_weight_sha256": checkpoint_sha256,
                "checkpoint_provenance": checkpoint_provenance,
                "reproducibility": reproducibility_info,
                "groups": {name: list(categories) for name, categories in groups.items()},
                "samples_per_group_per_repeat": args.samples_per_group,
                "repeats": args.repeats,
                "seed": args.seed,
                "matched_preprocessing_stochastic_seed_by_slot": True,
                "matched_diffusion_noise_seed_by_slot": True,
                "matched_timestep_by_slot": True,
                "timestep_sampling": "shuffled_stratified_uniform_midpoints",
                "sample_without_replacement_across_repeats": True,
                "native_category_balancing": True,
                "world_size": accelerator.num_processes,
                "ddp_gradient_synchronizations_per_group": 1,
                "mixed_precision": accelerator.mixed_precision,
                "loss_scope": "latent_only" if args.latent_only else "full_training_objective",
                "gradient_parameter_scope": "all parameters trainable in the joint run",
                "gradient_cache_dtype": "float32",
                "estimated_peak_gradient_cache_bytes": estimated_cache_bytes,
                "dot_accumulation_dtype": "float64",
                "trainable_parameter_tensors": len(named_parameters),
                "trainable_parameter_elements": total_numel,
                "component_parameter_elements": dict(component_numel),
                "skipped_zero_norm_component_scopes": sorted(invalid_component_scopes),
                "low_similarity_threshold": args.low_similarity_threshold,
                "sample_plan": sample_plan_json,
                "loss_config": config.loss.model_dump(mode="json"),
                "model_lora_config": config.model.lora.model_dump(mode="json"),
                "mask_control_config": config.model.mask_control.model_dump(mode="json"),
            }
            (output_dir / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            completion = {
                "status": "complete",
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "primary_heatmap": "category_gradient_cosine_heatmap_all_trainable.png",
                "primary_table": "pairwise_cosine_summary.csv",
            }
            (output_dir / "run_complete.json").write_text(
                json.dumps(completion, indent=2) + "\n",
                encoding="utf-8",
            )
            success = True
            LOGGER.info("Analysis complete: %s", output_dir)
        success = bool(_broadcast_object(accelerator, success if accelerator.is_main_process else None))
        accelerator.wait_for_everyone()
    finally:
        zero_trainable_gradients(named_parameters)
        if (
            accelerator.is_main_process
            and success
            and not args.keep_gradient_cache
            and cache_dir.exists()
        ):
            shutil.rmtree(cache_dir)
        accelerator.wait_for_everyone()
        accelerator.end_training()
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact cross-category gradient cosine similarities on a trained joint "
            "checkpoint and automatically render publication-ready figures."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Joint checkpoint directory or its LoRA .safetensors file.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Training YAML. Defaults to train_config.yaml in the checkpoint "
            "directory or its parent run directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/category_gradient_conflict/<UTC timestamp>.",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional dataset root override for a moved legacy run.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional joint manifest override for a moved legacy run.",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        metavar="NAME=CATEGORY_A,CATEGORY_B",
        help="Custom semantic group; repeat at least twice. Cannot be combined with --preset-groups.",
    )
    parser.add_argument(
        "--preset-groups",
        nargs="+",
        choices=list(DEFAULT_CATEGORY_GROUPS),
        default=None,
        help="Subset/order of built-in groups. By default all groups present in the dataset are used.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=32,
        help="Unique samples per semantic group and repeat; must be divisible by world size.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent, non-overlapping sample repeats used for uncertainty estimates.",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--low-similarity-threshold",
        type=float,
        default=0.10,
        help="Predeclared threshold used only for summary labels; raw cosines are always reported.",
    )
    parser.add_argument(
        "--dot-workspace-mb",
        type=int,
        default=256,
        help="Approximate RAM workspace for float64 streaming dot products.",
    )
    parser.add_argument(
        "--latent-only",
        action="store_true",
        help="Debug/ablation: disable region-perceptual loss. The default uses the full training objective.",
    )
    parser.add_argument(
        "--disable-gradient-checkpointing",
        action="store_true",
        help="Disable DiT gradient checkpointing (faster but uses more GPU memory).",
    )
    parser.add_argument(
        "--keep-gradient-cache",
        action="store_true",
        help="Keep large temporary float32 gradient memory maps after analysis.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an output directory containing run_complete.json.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = run_analysis(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Category-gradient analysis written to: {output_dir}")


if __name__ == "__main__":
    main()

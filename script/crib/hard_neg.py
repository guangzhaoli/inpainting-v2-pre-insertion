"""Hard-negative mining from neighbor lists + instance ids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class HardNegResult:
    hard_neg_uids: List[List[str]]
    hard_neg_scores: List[List[float]]
    n_with_any: int
    n_full: int
    mean_count: float


def mine_hard_negatives(
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    sample_uids: Sequence[str],
    categories: Sequence[str],
    instance_ids: Sequence[str],
    *,
    topk: int = 8,
    min_sim: float = 0.70,
    max_sim: float = 0.95,
    same_category_only: bool = True,
) -> HardNegResult:
    """Select hard decoys: similar appearance, different instance.

    Candidates are taken from precomputed neighbors and filtered by:
    - not self
    - different instance_id
    - optional same category
    - similarity in [min_sim, max_sim)
    """
    n = len(sample_uids)
    uid_list = list(sample_uids)
    hard_uids: List[List[str]] = [[] for _ in range(n)]
    hard_scores: List[List[float]] = [[] for _ in range(n)]

    for i in range(n):
        chosen_uids: List[str] = []
        chosen_scores: List[float] = []
        for j, sc in zip(neighbor_indices[i], neighbor_scores[i]):
            j = int(j)
            if j < 0 or j == i:
                continue
            sc = float(sc)
            if sc < min_sim or sc >= max_sim:
                continue
            if instance_ids[i] == instance_ids[j]:
                continue
            if same_category_only and categories[i] != categories[j]:
                continue
            chosen_uids.append(uid_list[j])
            chosen_scores.append(sc)
            if len(chosen_uids) >= topk:
                break
        hard_uids[i] = chosen_uids
        hard_scores[i] = chosen_scores

    counts = [len(x) for x in hard_uids]
    n_with_any = sum(c > 0 for c in counts)
    n_full = sum(c >= topk for c in counts)
    mean_count = float(np.mean(counts)) if counts else 0.0
    return HardNegResult(
        hard_neg_uids=hard_uids,
        hard_neg_scores=hard_scores,
        n_with_any=n_with_any,
        n_full=n_full,
        mean_count=mean_count,
    )


def build_instance_index(
    sample_uids: Sequence[str],
    instance_ids: Sequence[str],
) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for uid, iid in zip(sample_uids, instance_ids):
        index.setdefault(iid, []).append(uid)
    return index

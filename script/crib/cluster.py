"""Conservative same-id / near-duplicate clustering via union-find."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


@dataclass
class ClusterResult:
    instance_ids: List[str]
    roots: np.ndarray
    cluster_sizes: Dict[str, int]
    edge_count: int
    n_multi: int
    n_singleton: int


def cluster_same_id(
    neighbor_indices: np.ndarray,
    neighbor_scores: np.ndarray,
    sample_uids: Sequence[str],
    categories: Sequence[str],
    *,
    sim_threshold: float = 0.95,
    require_mutual: bool = False,
    id_prefix: str = "inst",
) -> ClusterResult:
    """Merge samples connected by high-similarity neighbor edges.

    Conservative by design: only edges with score >= sim_threshold are merged,
    and only within the same category.
    """
    n = len(sample_uids)
    if neighbor_indices.shape[0] != n:
        raise ValueError("neighbor_indices rows must match sample count")

    uf = UnionFind(n)
    # map for mutual check: i -> set of neighbor indices above threshold
    above: List[set] = [set() for _ in range(n)]
    edge_count = 0

    for i in range(n):
        for j, sc in zip(neighbor_indices[i], neighbor_scores[i]):
            j = int(j)
            if j < 0:
                continue
            sc = float(sc)
            if sc < sim_threshold:
                continue
            if categories[i] != categories[j]:
                continue
            above[i].add(j)

    for i in range(n):
        for j in above[i]:
            if j <= i:
                continue
            if require_mutual and i not in above[j]:
                continue
            uf.union(i, j)
            edge_count += 1

    roots = np.asarray([uf.find(i) for i in range(n)], dtype=np.int64)
    # stable instance ids ordered by first occurrence of root
    root_to_id: Dict[int, str] = {}
    instance_ids: List[str] = []
    next_id = 0
    for i, root in enumerate(roots):
        if root not in root_to_id:
            root_to_id[root] = f"{id_prefix}_{next_id:06d}"
            next_id += 1
        instance_ids.append(root_to_id[root])

    sizes: Dict[str, int] = {}
    for iid in instance_ids:
        sizes[iid] = sizes.get(iid, 0) + 1
    n_singleton = sum(1 for s in sizes.values() if s == 1)
    n_multi = len(sizes) - n_singleton
    return ClusterResult(
        instance_ids=instance_ids,
        roots=roots,
        cluster_sizes=sizes,
        edge_count=edge_count,
        n_multi=n_multi,
        n_singleton=n_singleton,
    )

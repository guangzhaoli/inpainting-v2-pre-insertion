"""Nearest-neighbor search helpers (FAISS preferred, numpy fallback)."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def _numpy_knn(embeddings: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact cosine KNN via matmul. embeddings must be L2-normalized."""
    n = embeddings.shape[0]
    k = min(topk, max(n - 1, 1))
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -np.inf)
    # argpartition then sort the top-k slice
    idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part_sims = np.take_along_axis(sims, idx_part, axis=1)
    order = np.argsort(-part_sims, axis=1)
    indices = np.take_along_axis(idx_part, order, axis=1)
    scores = np.take_along_axis(part_sims, order, axis=1)
    return indices.astype(np.int64), scores.astype(np.float32)


def knn_search(
    embeddings: np.ndarray,
    topk: int,
    *,
    use_faiss: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices [N,K], scores [N,K]) excluding self.

    Scores are cosine similarities for L2-normalized embeddings.
    """
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    n, d = embeddings.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
    if n == 1:
        return np.zeros((1, 0), dtype=np.int64), np.zeros((1, 0), dtype=np.float32)

    k = min(topk, n - 1)

    if use_faiss:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            # query k+1 then drop self
            scores, indices = index.search(embeddings, k + 1)
            out_idx = np.full((n, k), -1, dtype=np.int64)
            out_scores = np.full((n, k), -np.inf, dtype=np.float32)
            for i in range(n):
                row_idx = []
                row_sc = []
                for j, sc in zip(indices[i], scores[i]):
                    if int(j) == i or int(j) < 0:
                        continue
                    row_idx.append(int(j))
                    row_sc.append(float(sc))
                    if len(row_idx) >= k:
                        break
                if row_idx:
                    out_idx[i, : len(row_idx)] = np.asarray(row_idx, dtype=np.int64)
                    out_scores[i, : len(row_sc)] = np.asarray(row_sc, dtype=np.float32)
            return out_idx, out_scores
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.warning("FAISS unavailable (%s); falling back to numpy KNN", exc)

    return _numpy_knn(embeddings, k)


def pairwise_cosine(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if b is None:
        return a @ a.T
    b = np.asarray(b, dtype=np.float32)
    return a @ b.T

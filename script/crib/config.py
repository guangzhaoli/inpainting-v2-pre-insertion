"""Default hyperparameters for the CRIB MVP sidecar builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CribBuildConfig:
    """Conservative MVP defaults for accessory/jewelry.

    Design intent
    -------------
    - DINOv2 embeddings measure *appearance similarity*, not category labels.
    - FAISS/numpy KNN retrieves nearest neighbors inside the same specialist pool.
    - Very high similarity => near-duplicate / same-id (must NOT be hard negatives).
    - High-but-not-identical similarity => hard decoy candidates.
    - Thresholds are intentionally conservative: prefer under-merging instance ids.
    """

    # Feature extraction
    model_name: str = "dinov2_vits14"
    image_size: int = 224
    mask_threshold: float = 0.5
    bbox_pad_ratio: float = 0.15
    min_mask_pixels: int = 16
    batch_size: int = 16
    num_workers: int = 0
    device: str = "cpu"  # "cuda" | "cpu" | "auto"

    # Neighbor search
    neighbor_topk: int = 50
    use_faiss: bool = True

    # Same-id / near-dup clustering (conservative)
    # Tuned on 256/512/1024/1961 (strided full-manifest) accessory+jewelry runs.
    # 0.95: size-11 wrong sunglasses merges (1024-run).
    # 0.96: no 11+, but can still merge same-brand different colorways (sim~0.95-0.97).
    # 0.97: preferred full-run default — keeps near-dups, leaves colorway variants as hard-negs.
    same_id_sim_threshold: float = 0.97
    # Mutual NN has little effect once thr>=0.96 on these pools; keep False for recall.
    require_mutual_nearest_for_same_id: bool = False

    # Hard-negative mining
    # On 1961 strided samples: min=0.70 => cov~84%, mean~5.7; min=0.65 => cov~90%.
    # Prefer 0.70 as default (harder decoys); use 0.65 if coverage is low after full run.
    hard_neg_topk: int = 8
    hard_neg_min_sim: float = 0.70
    hard_neg_max_sim: float = 0.97  # must be <= same_id threshold
    # False: mine within the whole loaded manifest (specialist pool).
    # True: require exact category match (stricter; blocks accessory↔jewel_filter).
    hard_neg_same_category_only: bool = False

    # Review export
    review_pairs: int = 120
    review_same_id_clusters: int = 40

    # Runtime
    seed: int = 0
    limit: Optional[int] = None  # optional subsample for smoke tests
    offset: int = 0
    shard_index: Optional[int] = None
    num_shards: Optional[int] = None
    embeddings_only: bool = False  # extract features only (for multi-GPU stage-1)
    save_neighbor_matrix: bool = True

    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not (0.0 < self.mask_threshold < 1.0):
            raise ValueError("mask_threshold must be in (0, 1)")
        if self.bbox_pad_ratio < 0:
            raise ValueError("bbox_pad_ratio must be >= 0")
        if self.neighbor_topk < 1:
            raise ValueError("neighbor_topk must be >= 1")
        if self.hard_neg_topk < 1:
            raise ValueError("hard_neg_topk must be >= 1")
        if not (0.0 <= self.hard_neg_min_sim <= self.hard_neg_max_sim <= 1.0):
            raise ValueError("require 0 <= hard_neg_min_sim <= hard_neg_max_sim <= 1")
        if not (0.0 < self.same_id_sim_threshold <= 1.0):
            raise ValueError("same_id_sim_threshold must be in (0, 1]")
        if self.hard_neg_max_sim > self.same_id_sim_threshold:
            raise ValueError(
                "hard_neg_max_sim should be <= same_id_sim_threshold so near-dups "
                "are never selected as hard negatives"
            )
        if self.image_size < 16:
            raise ValueError("image_size too small")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ACCESSORY_CONFIG = CribBuildConfig()

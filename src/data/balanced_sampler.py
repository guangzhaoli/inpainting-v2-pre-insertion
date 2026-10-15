"""Coverage-preserving category sampling for heterogeneous supervised teachers."""

import math
from collections import defaultdict

import torch
from torch.utils.data import Sampler


class CategoryBalancedSampler(Sampler):
    """Visit every example, repeat smaller pools, then shuffle globally.

    A random replacement sampler can silently omit training examples. This
    sampler retains every example each epoch and interleaves categories after
    expansion, so DDP ranks are not permanently assigned a single category.
    """

    def __init__(self, dataset, seed=42):
        self.seed = int(seed)
        self.epoch = 0
        self.pools = defaultdict(list)
        for index, sample in enumerate(dataset.samples):
            category = sample.get("category")
            if category is None:
                category = dataset._get_sample_info(index)["category"]
            self.pools[str(category)].append(index)
        if not self.pools:
            raise ValueError("category balancing requires a nonempty dataset")
        self.pool_size = max(map(len, self.pools.values()))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.pool_size * len(self.pools)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = []
        for category in sorted(self.pools):
            pool = self.pools[category]
            expanded = []
            for _ in range(math.ceil(self.pool_size / len(pool))):
                expanded.extend(pool[i] for i in torch.randperm(len(pool), generator=generator).tolist())
            indices.extend(expanded[:self.pool_size])
        yield from (indices[i] for i in torch.randperm(len(indices), generator=generator).tolist())

import os
import random
import numpy as np
import torch


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # required for strict CUDA determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_training_rank(seed: int, process_index: int) -> int:
    """Seed stochastic training AFTER identical model initialization/DDP setup.

    The data sampler has its own shared generator. Checkpoint restore must run
    after this function so resumed runs retain their saved per-rank streams.
    """
    rank_seed = (int(seed) + 1_000_003 * int(process_index)) % (2**32)
    seed_everything(rank_seed)
    return rank_seed


def prepare_trainable_precision(model, precision="float32"):
    """Keep small optimizer updates in FP32; leave frozen backbone weights alone."""
    if precision not in {"float32", "model"}:
        raise ValueError(f"unsupported trainable precision: {precision}")
    if precision == "float32":
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.float()

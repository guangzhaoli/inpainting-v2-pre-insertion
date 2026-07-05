import logging
from collections import Counter
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _human_int(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.2f}K"
    return str(value)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _device_name(device: torch.device) -> str:
    return str(device)


def summarize_model(model: nn.Module) -> Dict[str, Any]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())

    total_params = sum(param.numel() for param in parameters)
    trainable_params = sum(param.numel() for param in parameters if param.requires_grad)
    frozen_params = total_params - trainable_params
    parameter_bytes = sum(param.numel() * param.element_size() for param in parameters)
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in buffers)

    dtype_counts = Counter(_dtype_name(param.dtype) for param in parameters)
    device_counts = Counter(_device_name(param.device) for param in parameters)
    trainable_tensors = sum(1 for param in parameters if param.requires_grad)

    return {
        "class_name": model.__class__.__name__,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_tensors": trainable_tensors,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "device_counts": dict(sorted(device_counts.items())),
    }


def print_model_summary_table(model: nn.Module, title: str = "Model Summary") -> Dict[str, Any]:
    stats = summarize_model(model)
    logger.info(
        "%s | class=%s total=%s trainable=%s frozen=%s trainable_tensors=%s param_bytes=%.2fGB buffer_bytes=%.2fGB dtypes=%s devices=%s",
        title,
        stats["class_name"],
        _human_int(stats["total_params"]),
        _human_int(stats["trainable_params"]),
        _human_int(stats["frozen_params"]),
        stats["trainable_tensors"],
        stats["parameter_bytes"] / 1024**3,
        stats["buffer_bytes"] / 1024**3,
        stats["dtype_counts"],
        stats["device_counts"],
    )
    return stats

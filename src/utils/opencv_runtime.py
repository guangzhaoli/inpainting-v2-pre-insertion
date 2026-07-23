"""Process-local OpenCV runtime limits for training and data workers.

OpenCV otherwise sizes its internal thread pool from the host CPU count.  That
is counterproductive when every distributed rank (and every DataLoader worker)
runs mask morphology or distance transforms at the same time.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2


OPENCV_THREADS_ENV = "QWEN_INPAINTING_OPENCV_THREADS"
DEFAULT_OPENCV_THREADS = 1
_MAX_OPENCV_THREADS = 2**31 - 1

_configured_pid: Optional[int] = None
_configured_threads: Optional[int] = None


def _requested_thread_count() -> int:
    raw_value = os.environ.get(OPENCV_THREADS_ENV)
    if raw_value is None:
        return DEFAULT_OPENCV_THREADS

    try:
        thread_count = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{OPENCV_THREADS_ENV} must be a positive integer, got {raw_value!r}"
        ) from exc

    if not 1 <= thread_count <= _MAX_OPENCV_THREADS:
        raise ValueError(
            f"{OPENCV_THREADS_ENV} must be between 1 and {_MAX_OPENCV_THREADS}, "
            f"got {raw_value!r}"
        )
    return thread_count


def configure_opencv_runtime() -> int:
    """Limit OpenCV to the requested threads once in the current process.

    The PID check intentionally makes this fork-safe: DataLoader children
    inherit the parent's module state, but configure their own OpenCV runtime
    once when their worker initializer runs.
    """

    global _configured_pid, _configured_threads

    thread_count = _requested_thread_count()
    process_id = os.getpid()
    if _configured_pid == process_id:
        if _configured_threads != thread_count:
            raise RuntimeError(
                f"{OPENCV_THREADS_ENV} changed from {_configured_threads} to "
                f"{thread_count} after OpenCV was configured"
            )
        return thread_count

    cv2.setNumThreads(thread_count)
    # Training uses CUDA through PyTorch.  OpenCV OpenCL adds another implicit
    # device/runtime and does not benefit these small mask operations.
    cv2.ocl.setUseOpenCL(False)

    _configured_pid = process_id
    _configured_threads = thread_count
    return thread_count


def configure_opencv_dataloader_worker(_worker_id: int) -> None:
    """Apply the process-local limit without altering worker RNG state."""

    configure_opencv_runtime()

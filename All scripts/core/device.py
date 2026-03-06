"""Shared device/runtime helpers for training and visualization."""

from __future__ import annotations

import torch


def has_cuda_gpu() -> bool:
    """Return True when CUDA is available."""
    return torch.cuda.is_available()


def has_mps_backend() -> bool:
    """Return True when the MPS backend is built in this PyTorch build."""
    return torch.backends.mps.is_built()


def get_compute_device() -> str:
    """Select device string used across the project."""
    if has_mps_backend():
        return "mps"
    if has_cuda_gpu():
        return "cuda:0"
    return "cpu"


def clear_cuda_cache() -> None:
    """Best-effort CUDA cache cleanup."""
    if not has_cuda_gpu():
        return

    import gc

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

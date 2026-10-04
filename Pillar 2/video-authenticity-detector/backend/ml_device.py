"""
Shared inference device selection for Pillar 2 backend (CUDA when available).
Set FORCE_CPU=1 to disable GPU (debugging or machines without drivers).
"""
import os
from typing import Optional, Tuple

import torch

_FORCE_CPU = os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes")


def get_torch_device() -> torch.device:
    if not _FORCE_CPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hf_pipeline_device() -> int:
    """HuggingFace pipelines use device index (0) or -1 for CPU."""
    if not _FORCE_CPU and torch.cuda.is_available():
        return 0
    return -1


def describe_compute_device() -> str:
    dev = get_torch_device()
    if dev.type == "cuda":
        try:
            name = torch.cuda.get_device_name(dev)
            return f"CUDA ({name})"
        except Exception:
            return "CUDA"
    return "CPU (set FORCE_CPU=0 and install CUDA PyTorch to use GPU)"


def device_summary() -> Tuple[str, Optional[str]]:
    dev = get_torch_device()
    if dev.type == "cuda":
        try:
            return "cuda", torch.cuda.get_device_name(dev)
        except Exception:
            return "cuda", None
    return "cpu", None

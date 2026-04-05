"""PyTorch helper functions used by deep RL examples."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Pick the best available torch device for small examples."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensor(array: Any, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert array-like data to a torch tensor on the target device."""
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft-update target parameters: target <- (1 - tau) * target + tau * source."""
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")

    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.mul_(1.0 - tau).add_(source_param, alpha=tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    """Copy source parameters into target."""
    target.load_state_dict(source.state_dict())


"""Utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may not be installed yet
    torch = None  # type: ignore[assignment]


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_gym_env(env: object, seed: int) -> Optional[tuple[object, dict]]:
    """Seed a Gymnasium-style environment when the API is available."""
    reset = getattr(env, "reset", None)
    if reset is None:
        return None

    try:
        return reset(seed=seed)
    except TypeError:
        return None


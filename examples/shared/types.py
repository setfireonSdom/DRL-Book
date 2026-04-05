"""Common data structures for the new examples package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Transition:
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


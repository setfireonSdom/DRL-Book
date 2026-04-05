"""Small plotting helpers for reward curves and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def running_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """Compute a simple running average with a trailing window."""
    if window <= 0:
        raise ValueError("window must be positive")

    array = np.asarray(values, dtype=np.float64)
    averages = np.empty_like(array)

    for index in range(len(array)):
        start = max(0, index - window + 1)
        averages[index] = array[start : index + 1].mean()

    return averages


def plot_series(
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "Episode",
    running_window: int | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a 1D series and optionally overlay a running average."""
    array = np.asarray(values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(array, label="raw", alpha=0.6)

    if running_window is not None:
        ax.plot(
            running_average(array, window=running_window),
            label=f"running avg ({running_window})",
            linewidth=2,
        )
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=160, bbox_inches="tight")

    plt.show()


def plot_multiple_series(
    named_series: Iterable[tuple[str, Sequence[float]]],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "Episode",
) -> None:
    """Plot several comparable 1D series on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for label, values in named_series:
        ax.plot(np.asarray(values, dtype=np.float64), label=label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plt.show()


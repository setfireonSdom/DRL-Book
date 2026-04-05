"""Shared helpers for bandit experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BernoulliBandit:
    """A Bernoulli arm with fixed success probability."""

    p: float
    estimate: float = 0.0
    pulls: int = 0

    def pull(self, rng: np.random.Generator) -> float:
        return float(rng.random() < self.p)

    def update(self, reward: float) -> None:
        self.pulls += 1
        self.estimate += (reward - self.estimate) / self.pulls


def run_bandit_experiment(
    bandits: list[BernoulliBandit],
    *,
    steps: int,
    select_arm,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a generic bandit loop with a pluggable arm selector."""
    rewards = np.zeros(steps, dtype=np.float64)
    choices = np.zeros(steps, dtype=np.int64)

    for step in range(steps):
        arm_index = select_arm(step, bandits, rng)
        reward = bandits[arm_index].pull(rng)
        bandits[arm_index].update(reward)
        rewards[step] = reward
        choices[step] = arm_index

    return rewards, choices


def cumulative_average(values: np.ndarray) -> np.ndarray:
    """Compute cumulative average reward."""
    return np.cumsum(values) / (np.arange(len(values)) + 1)


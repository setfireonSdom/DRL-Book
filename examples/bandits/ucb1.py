"""Upper Confidence Bound (UCB1) exploration for Bernoulli bandits."""

from __future__ import annotations

import math

import numpy as np

from examples.bandits.bandit_utils import BernoulliBandit, cumulative_average, run_bandit_experiment
from examples.shared.plotting import plot_series
from examples.shared.seed import set_seed


def select_ucb1_arm(step: int, bandits: list[BernoulliBandit], rng: np.random.Generator) -> int:
    if step < len(bandits):
        return step

    ucb_scores = []
    total_pulls = step + 1
    for bandit in bandits:
        bonus = math.sqrt((2.0 * math.log(total_pulls)) / bandit.pulls)
        ucb_scores.append(bandit.estimate + bonus)
    return int(np.argmax(ucb_scores))


def run_ucb1_experiment(
    probabilities: tuple[float, ...] = (0.2, 0.5, 0.75),
    *,
    steps: int = 10_000,
    seed: int = 123,
) -> tuple[np.ndarray, list[BernoulliBandit]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    bandits = [BernoulliBandit(p) for p in probabilities]

    rewards, _ = run_bandit_experiment(
        bandits,
        steps=steps,
        rng=rng,
        select_arm=select_ucb1_arm,
    )
    return rewards, bandits


def main() -> None:
    rewards, bandits = run_ucb1_experiment()
    averages = cumulative_average(rewards)

    for index, bandit in enumerate(bandits):
        print(f"arm {index}: true p={bandit.p:.2f}, estimate={bandit.estimate:.3f}, pulls={bandit.pulls}")

    plot_series(
        averages,
        title="UCB1 Bandit Performance",
        ylabel="Cumulative Average Reward",
        xlabel="Step",
    )


if __name__ == "__main__":
    main()


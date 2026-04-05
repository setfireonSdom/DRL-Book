"""Optimistic initial values for Bernoulli bandits."""

from __future__ import annotations

import numpy as np

from examples.bandits.bandit_utils import BernoulliBandit, cumulative_average, run_bandit_experiment
from examples.shared.plotting import plot_series
from examples.shared.seed import set_seed


class OptimisticBandit(BernoulliBandit):
    def __init__(self, p: float, optimistic_value: float) -> None:
        super().__init__(p=p, estimate=optimistic_value, pulls=1)


def run_optimistic_initial_values(
    probabilities: tuple[float, ...] = (0.2, 0.5, 0.75),
    *,
    optimistic_value: float = 5.0,
    steps: int = 10_000,
    seed: int = 123,
) -> tuple[np.ndarray, list[OptimisticBandit]]:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    bandits = [OptimisticBandit(p, optimistic_value=optimistic_value) for p in probabilities]

    rewards, _ = run_bandit_experiment(
        bandits,
        steps=steps,
        rng=rng,
        select_arm=lambda step, bandits, rng: int(np.argmax([bandit.estimate for bandit in bandits])),
    )
    return rewards, bandits


def main() -> None:
    rewards, bandits = run_optimistic_initial_values()
    averages = cumulative_average(rewards)

    for index, bandit in enumerate(bandits):
        print(f"arm {index}: true p={bandit.p:.2f}, estimate={bandit.estimate:.3f}, pulls={bandit.pulls}")

    plot_series(
        averages,
        title="Optimistic Initial Values",
        ylabel="Cumulative Average Reward",
        xlabel="Step",
    )


if __name__ == "__main__":
    main()


"""Epsilon-greedy exploration on a Bernoulli bandit problem."""

from __future__ import annotations

import numpy as np

from examples.bandits.bandit_utils import BernoulliBandit, cumulative_average, run_bandit_experiment
from examples.shared.plotting import plot_multiple_series
from examples.shared.seed import set_seed


def select_epsilon_greedy_arm(step: int, bandits: list[BernoulliBandit], rng: np.random.Generator, *, epsilon: float) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(bandits)))
    estimates = [bandit.estimate for bandit in bandits]
    return int(np.argmax(estimates))


def run_epsilon_greedy_experiment(
    probabilities: tuple[float, ...] = (0.2, 0.5, 0.75),
    *,
    epsilon: float = 0.1,
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
        select_arm=lambda step, bandits, rng: select_epsilon_greedy_arm(
            step, bandits, rng, epsilon=epsilon
        ),
    )
    return rewards, bandits


def main() -> None:
    configs = [0.1, 0.05, 0.01]
    named_series = []

    for epsilon in configs:
        rewards, bandits = run_epsilon_greedy_experiment(epsilon=epsilon)
        averages = cumulative_average(rewards)
        named_series.append((f"epsilon={epsilon}", averages))

        print(f"\nEpsilon {epsilon}")
        for index, bandit in enumerate(bandits):
            print(
                f"arm {index}: true p={bandit.p:.2f}, estimate={bandit.estimate:.3f}, pulls={bandit.pulls}"
            )

    plot_multiple_series(
        named_series,
        title="Epsilon-Greedy Bandit Performance",
        ylabel="Cumulative Average Reward",
        xlabel="Step",
    )


if __name__ == "__main__":
    main()


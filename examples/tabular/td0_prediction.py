"""TD(0) prediction on the tabular GridWorld."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from examples.shared.seed import set_seed
from examples.shared.plotting import plot_series
from examples.tabular.gridworld import ALL_ACTIONS, Action, GridWorld, State, standard_grid
from examples.tabular.policy_evaluation import print_policy, print_values


def epsilon_greedy_action(
    policy: Mapping[State, Action],
    state: State,
    *,
    rng: np.random.Generator,
    epsilon: float,
) -> Action:
    if rng.random() < epsilon:
        return ALL_ACTIONS[int(rng.integers(len(ALL_ACTIONS)))]
    return policy[state]


def run_td0_prediction(
    policy: Mapping[State, Action],
    *,
    episodes: int = 10_000,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    seed: int = 123,
) -> tuple[dict[State, float], list[float]]:
    """Estimate state values with one-step TD learning."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    grid = standard_grid()

    values = {state: 0.0 for state in grid.all_states()}
    deltas: list[float] = []

    for _ in range(episodes):
        state = grid.reset()
        episode_delta = 0.0

        while not grid.game_over():
            action = epsilon_greedy_action(policy, state, rng=rng, epsilon=epsilon)
            reward = grid.move(action)
            next_state = grid.current_state()

            old_value = values[state]
            values[state] += alpha * (reward + gamma * values[next_state] - values[state])
            episode_delta = max(episode_delta, abs(values[state] - old_value))
            state = next_state

        deltas.append(episode_delta)

    return values, deltas


def default_policy() -> dict[State, Action]:
    return {
        (2, 0): "U",
        (1, 0): "U",
        (0, 0): "R",
        (0, 1): "R",
        (0, 2): "R",
        (1, 2): "R",
        (2, 1): "R",
        (2, 2): "R",
        (2, 3): "U",
    }


def main() -> None:
    grid = standard_grid()
    policy = default_policy()
    values, deltas = run_td0_prediction(policy)

    print("Rewards:")
    print_values(grid.rewards, grid)
    print("Estimated values:")
    print_values(values, grid)
    print("Policy:")
    print_policy(policy, grid)

    plot_series(
        deltas,
        title="TD(0) Max Value Change Per Episode",
        ylabel="Max delta",
        running_window=100,
    )


if __name__ == "__main__":
    main()


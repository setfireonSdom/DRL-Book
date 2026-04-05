"""On-policy SARSA control on the tabular GridWorld."""

from __future__ import annotations

import numpy as np

from examples.shared.plotting import plot_series
from examples.shared.seed import set_seed
from examples.tabular.control_utils import (
    epsilon_greedy_action,
    greedy_policy_and_values,
    initialize_q_table,
    normalize_counts,
)
from examples.tabular.gridworld import GridWorld, negative_grid
from examples.tabular.policy_evaluation import print_policy, print_values


def run_sarsa(
    *,
    episodes: int = 10_000,
    gamma: float = 0.9,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    step_cost: float = -0.1,
    seed: int = 123,
) -> tuple[dict[tuple[int, int], dict[str, float]], list[float], dict[tuple[int, int], int]]:
    """Train a SARSA agent on the negative GridWorld."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    grid: GridWorld = negative_grid(step_cost=step_cost)
    q_table = initialize_q_table(grid)
    update_counts: dict[tuple[int, int], int] = {}
    rewards_per_episode: list[float] = []

    for _ in range(episodes):
        state = grid.reset()
        action = epsilon_greedy_action(q_table, grid, state, rng=rng, epsilon=epsilon)
        episode_reward = 0.0

        while not grid.game_over():
            reward = grid.move(action)
            next_state = grid.current_state()
            episode_reward += reward

            update_counts[state] = update_counts.get(state, 0) + 1

            if grid.game_over():
                td_target = reward
            else:
                next_action = epsilon_greedy_action(q_table, grid, next_state, rng=rng, epsilon=epsilon)
                td_target = reward + gamma * q_table[next_state][next_action]

            q_table[state][action] += alpha * (td_target - q_table[state][action])

            if grid.game_over():
                break

            state = next_state
            action = next_action

        rewards_per_episode.append(episode_reward)

    return q_table, rewards_per_episode, update_counts


def main() -> None:
    grid = negative_grid(step_cost=-0.1)
    q_table, rewards_per_episode, update_counts = run_sarsa()
    policy, values = greedy_policy_and_values(grid, q_table)

    print("Rewards:")
    print_values(grid.rewards, grid)
    print("Update counts:")
    print_values(normalize_counts(update_counts), grid)
    print("Learned values:")
    print_values(values, grid)
    print("Learned policy:")
    print_policy(policy, grid)

    plot_series(
        rewards_per_episode,
        title="SARSA Reward Per Episode",
        ylabel="Reward",
        running_window=100,
    )


if __name__ == "__main__":
    main()


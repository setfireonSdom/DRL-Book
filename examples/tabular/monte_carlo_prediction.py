"""First-visit Monte Carlo prediction on the tabular GridWorld."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import numpy as np

from examples.shared.seed import set_seed
from examples.tabular.gridworld import Action, GridWorld, State, standard_grid
from examples.tabular.policy_evaluation import print_policy, print_values


def sample_episode(
    grid: GridWorld,
    policy: Mapping[State, Action],
    *,
    rng: np.random.Generator,
    max_steps: int = 20,
) -> tuple[list[State], list[float]]:
    """Generate one episode under a fixed deterministic policy."""
    start_states = list(grid.actions)
    grid.set_state(start_states[int(rng.integers(len(start_states)))])

    state = grid.current_state()
    states = [state]
    rewards = [0.0]

    steps = 0
    while not grid.game_over():
        action = policy[state]
        reward = grid.move(action)
        next_state = grid.current_state()

        states.append(next_state)
        rewards.append(reward)

        steps += 1
        if steps >= max_steps:
            break

        state = next_state

    return states, rewards


def run_first_visit_mc_prediction(
    policy: Mapping[State, Action],
    *,
    episodes: int = 100,
    gamma: float = 0.9,
    seed: int = 123,
) -> dict[State, float]:
    """Estimate state values with first-visit Monte Carlo prediction."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    grid = standard_grid()

    values: dict[State, float] = {state: 0.0 for state in grid.all_states()}
    returns: dict[State, list[float]] = defaultdict(list)

    for _ in range(episodes):
        states, rewards = sample_episode(grid, policy, rng=rng)
        return_so_far = 0.0

        for t in range(len(states) - 2, -1, -1):
            state = states[t]
            reward = rewards[t + 1]
            return_so_far = reward + gamma * return_so_far

            if state not in states[:t]:
                returns[state].append(return_so_far)
                values[state] = float(np.mean(returns[state]))

    return values


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
    values = run_first_visit_mc_prediction(policy)

    print("Rewards:")
    print_values(grid.rewards, grid)
    print("Estimated values:")
    print_values(values, grid)
    print("Policy:")
    print_policy(policy, grid)


if __name__ == "__main__":
    main()


"""Policy iteration on the tabular GridWorld."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from examples.shared.seed import set_seed
from examples.tabular.gridworld import ALL_ACTIONS, Action, GridWorld, standard_grid, transition_reward_tables
from examples.tabular.policy_evaluation import evaluate_deterministic_policy, print_policy, print_values


def random_policy(grid: GridWorld, rng: np.random.Generator) -> dict[tuple[int, int], Action]:
    """Create a random deterministic policy over non-terminal states."""
    return {state: rng.choice(ALL_ACTIONS) for state in grid.actions}


def improve_policy(
    grid: GridWorld,
    policy: Mapping[tuple[int, int], Action],
    values: Mapping[tuple[int, int], float],
    *,
    gamma: float = 0.9,
) -> tuple[dict[tuple[int, int], Action], bool]:
    """Greedily improve a deterministic policy from a value function."""
    transition_probs, rewards = transition_reward_tables(grid)
    new_policy: dict[tuple[int, int], Action] = {}
    policy_stable = True

    for state, available_actions in grid.actions.items():
        best_action = available_actions[0]
        best_value = float("-inf")

        for action in available_actions:
            candidate_value = 0.0
            for next_state in grid.all_states():
                reward = rewards.get((state, action, next_state), 0.0)
                transition_prob = transition_probs.get((state, action, next_state), 0.0)
                candidate_value += transition_prob * (reward + gamma * values[next_state])

            if candidate_value > best_value:
                best_value = candidate_value
                best_action = action

        if policy.get(state) != best_action:
            policy_stable = False

        new_policy[state] = best_action

    return new_policy, policy_stable


def run_policy_iteration(
    *,
    gamma: float = 0.9,
    tolerance: float = 1e-3,
    seed: int = 123,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], Action]]:
    """Run policy iteration until the policy stops changing."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    grid = standard_grid()
    policy = random_policy(grid, rng)
    values: dict[tuple[int, int], float] | None = None

    while True:
        values = evaluate_deterministic_policy(
            grid,
            policy,
            gamma=gamma,
            tolerance=tolerance,
            initial_values=values,
        )

        next_policy, policy_stable = improve_policy(grid, policy, values, gamma=gamma)
        if next_policy == policy:
            break

        policy = next_policy
        if policy_stable:
            break

    return values, policy


def main() -> None:
    values, policy = run_policy_iteration()
    grid = standard_grid()

    print("Rewards:")
    print_values(grid.rewards, grid)
    print("Optimal values:")
    print_values(values, grid)
    print("Optimal policy:")
    print_policy(policy, grid)


if __name__ == "__main__":
    main()

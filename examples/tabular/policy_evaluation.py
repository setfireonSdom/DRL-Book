"""Shared policy evaluation utilities for tabular examples."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from examples.tabular.gridworld import ALL_ACTIONS, Action, GridWorld, State, transition_reward_tables


def print_values(values: Mapping[State, float], grid: GridWorld) -> None:
    for row in range(grid.rows):
        print("-" * 27)
        for col in range(grid.cols):
            value = values.get((row, col), 0.0)
            print(f"{value:6.2f}|", end="")
        print()


def print_policy(policy: Mapping[State, Action], grid: GridWorld) -> None:
    for row in range(grid.rows):
        print("-" * 27)
        for col in range(grid.cols):
            action = policy.get((row, col), " ")
            print(f"  {action}   |", end="")
        print()


def evaluate_deterministic_policy(
    grid: GridWorld,
    policy: Mapping[State, Action],
    *,
    gamma: float = 0.9,
    tolerance: float = 1e-3,
    initial_values: Mapping[State, float] | None = None,
) -> dict[State, float]:
    """Iteratively evaluate a deterministic policy."""
    transition_probs, rewards = transition_reward_tables(grid)
    values = {state: 0.0 for state in grid.all_states()}

    if initial_values is not None:
        values.update(initial_values)

    while True:
        biggest_change = 0.0

        for state in grid.all_states():
            if grid.is_terminal(state):
                continue

            old_value = values[state]
            new_value = 0.0

            for action in ALL_ACTIONS:
                action_prob = 1.0 if policy.get(state) == action else 0.0
                if action_prob == 0.0:
                    continue

                for next_state in grid.all_states():
                    reward = rewards.get((state, action, next_state), 0.0)
                    transition_prob = transition_probs.get((state, action, next_state), 0.0)
                    new_value += action_prob * transition_prob * (reward + gamma * values[next_state])

            values[state] = new_value
            biggest_change = max(biggest_change, abs(old_value - new_value))

        if biggest_change < tolerance:
            break

    return values


def greedy_policy_from_value_function(
    grid: GridWorld,
    values: Mapping[State, float],
    *,
    gamma: float = 0.9,
) -> dict[State, Action]:
    """Construct a greedy policy from a state-value function."""
    transition_probs, rewards = transition_reward_tables(grid)
    policy: dict[State, Action] = {}

    for state in grid.actions:
        best_action = ALL_ACTIONS[0]
        best_value = -np.inf

        for action in ALL_ACTIONS:
            candidate_value = 0.0
            for next_state in grid.all_states():
                reward = rewards.get((state, action, next_state), 0.0)
                transition_prob = transition_probs.get((state, action, next_state), 0.0)
                candidate_value += transition_prob * (reward + gamma * values[next_state])

            if candidate_value > best_value:
                best_value = candidate_value
                best_action = action

        policy[state] = best_action

    return policy

"""Shared helpers for tabular control algorithms."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from examples.tabular.gridworld import Action, GridWorld, State

QTable = dict[State, dict[Action, float]]


def initialize_q_table(grid: GridWorld) -> QTable:
    """Initialize Q(s, a) = 0 for every action at every state."""
    q_table: QTable = {}
    for state in grid.all_states():
        q_table[state] = {action: 0.0 for action in grid.available_actions(state)}
    return q_table


def epsilon_greedy_action(
    q_table: QTable,
    grid: GridWorld,
    state: State,
    *,
    rng: np.random.Generator,
    epsilon: float,
) -> Action:
    """Choose an action epsilon-greedily from the current Q-table."""
    available_actions = grid.available_actions(state)
    if len(available_actions) == 0:
        raise ValueError(f"state {state} is terminal and has no available actions")

    if rng.random() < epsilon:
        return available_actions[int(rng.integers(len(available_actions)))]

    return greedy_action(q_table, state)


def greedy_action(q_table: QTable, state: State) -> Action:
    """Return the greedy action under the current Q-values."""
    action_values = q_table[state]
    return max(action_values, key=action_values.get)


def greedy_policy_and_values(grid: GridWorld, q_table: QTable) -> tuple[dict[State, Action], dict[State, float]]:
    """Extract a greedy policy and value function from Q(s, a)."""
    policy: dict[State, Action] = {}
    values: dict[State, float] = {}

    for state in grid.all_states():
        if grid.is_terminal(state):
            values[state] = 0.0
            continue

        best_action = greedy_action(q_table, state)
        policy[state] = best_action
        values[state] = q_table[state][best_action]

    return policy, values


def normalize_counts(counts: Mapping[State, int]) -> dict[State, float]:
    """Normalize state visit or update counts into proportions."""
    total = sum(counts.values())
    if total == 0:
        return {state: 0.0 for state in counts}
    return {state: count / total for state, count in counts.items()}


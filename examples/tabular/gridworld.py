"""GridWorld environments used by the tabular RL examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

State = tuple[int, int]
Action = str
ActionSpace = tuple[Action, ...]

ALL_ACTIONS: ActionSpace = ("U", "D", "L", "R")


def move_state(state: State, action: Action) -> State:
    row, col = state
    if action == "U":
        return row - 1, col
    if action == "D":
        return row + 1, col
    if action == "L":
        return row, col - 1
    if action == "R":
        return row, col + 1
    raise ValueError(f"unknown action: {action}")


@dataclass
class GridWorld:
    rows: int
    cols: int
    start_state: State
    rewards: dict[State, float]
    actions: dict[State, ActionSpace]
    state: State = field(init=False)

    def __post_init__(self) -> None:
        self.state = self.start_state

    def reset(self) -> State:
        self.state = self.start_state
        return self.state

    def set_state(self, state: State) -> None:
        self.state = state

    def current_state(self) -> State:
        return self.state

    def is_terminal(self, state: State) -> bool:
        return state not in self.actions

    def game_over(self) -> bool:
        return self.is_terminal(self.state)

    def all_states(self) -> set[State]:
        return set(self.actions) | set(self.rewards)

    def available_actions(self, state: State | None = None) -> ActionSpace:
        actual_state = self.state if state is None else state
        return self.actions.get(actual_state, ())

    def get_next_state(self, state: State, action: Action) -> State:
        if action not in self.available_actions(state):
            return state
        return move_state(state, action)

    def move(self, action: Action) -> float:
        self.state = self.get_next_state(self.state, action)
        return self.rewards.get(self.state, 0.0)


@dataclass
class WindyGridWorld(GridWorld):
    transition_probs: dict[tuple[State, Action], dict[State, float]]

    def move(self, action: Action) -> float:
        probs = self.transition_probs[(self.state, action)]
        next_states = list(probs)
        weights = list(probs.values())
        self.state = next_states[int(np.random.choice(len(next_states), p=weights))]
        return self.rewards.get(self.state, 0.0)


def standard_grid() -> GridWorld:
    rewards = {(0, 3): 1.0, (1, 3): -1.0}
    actions = {
        (0, 0): ("D", "R"),
        (0, 1): ("L", "R"),
        (0, 2): ("L", "D", "R"),
        (1, 0): ("U", "D"),
        (1, 2): ("U", "D", "R"),
        (2, 0): ("U", "R"),
        (2, 1): ("L", "R"),
        (2, 2): ("L", "R", "U"),
        (2, 3): ("L", "U"),
    }
    return GridWorld(rows=3, cols=4, start_state=(2, 0), rewards=rewards, actions=actions)


def negative_grid(step_cost: float = -0.1) -> GridWorld:
    grid = standard_grid()
    grid.rewards.update(
        {
            (0, 0): step_cost,
            (0, 1): step_cost,
            (0, 2): step_cost,
            (1, 0): step_cost,
            (1, 2): step_cost,
            (2, 0): step_cost,
            (2, 1): step_cost,
            (2, 2): step_cost,
            (2, 3): step_cost,
        }
    )
    return grid


def windy_grid() -> WindyGridWorld:
    rewards = {(0, 3): 1.0, (1, 3): -1.0}
    actions = {
        (0, 0): ("D", "R"),
        (0, 1): ("L", "R"),
        (0, 2): ("L", "D", "R"),
        (1, 0): ("U", "D"),
        (1, 2): ("U", "D", "R"),
        (2, 0): ("U", "R"),
        (2, 1): ("L", "R"),
        (2, 2): ("L", "R", "U"),
        (2, 3): ("L", "U"),
    }
    probs = {
        ((2, 0), "U"): {(1, 0): 1.0},
        ((2, 0), "D"): {(2, 0): 1.0},
        ((2, 0), "L"): {(2, 0): 1.0},
        ((2, 0), "R"): {(2, 1): 1.0},
        ((1, 0), "U"): {(0, 0): 1.0},
        ((1, 0), "D"): {(2, 0): 1.0},
        ((1, 0), "L"): {(1, 0): 1.0},
        ((1, 0), "R"): {(1, 0): 1.0},
        ((0, 0), "U"): {(0, 0): 1.0},
        ((0, 0), "D"): {(1, 0): 1.0},
        ((0, 0), "L"): {(0, 0): 1.0},
        ((0, 0), "R"): {(0, 1): 1.0},
        ((0, 1), "U"): {(0, 1): 1.0},
        ((0, 1), "D"): {(0, 1): 1.0},
        ((0, 1), "L"): {(0, 0): 1.0},
        ((0, 1), "R"): {(0, 2): 1.0},
        ((0, 2), "U"): {(0, 2): 1.0},
        ((0, 2), "D"): {(1, 2): 1.0},
        ((0, 2), "L"): {(0, 1): 1.0},
        ((0, 2), "R"): {(0, 3): 1.0},
        ((2, 1), "U"): {(2, 1): 1.0},
        ((2, 1), "D"): {(2, 1): 1.0},
        ((2, 1), "L"): {(2, 0): 1.0},
        ((2, 1), "R"): {(2, 2): 1.0},
        ((2, 2), "U"): {(1, 2): 1.0},
        ((2, 2), "D"): {(2, 2): 1.0},
        ((2, 2), "L"): {(2, 1): 1.0},
        ((2, 2), "R"): {(2, 3): 1.0},
        ((2, 3), "U"): {(1, 3): 1.0},
        ((2, 3), "D"): {(2, 3): 1.0},
        ((2, 3), "L"): {(2, 2): 1.0},
        ((2, 3), "R"): {(2, 3): 1.0},
        ((1, 2), "U"): {(0, 2): 0.5, (1, 3): 0.5},
        ((1, 2), "D"): {(2, 2): 1.0},
        ((1, 2), "L"): {(1, 2): 1.0},
        ((1, 2), "R"): {(1, 3): 1.0},
    }
    return WindyGridWorld(
        rows=3,
        cols=4,
        start_state=(2, 0),
        rewards=rewards,
        actions=actions,
        transition_probs=probs,
    )


def windy_grid_penalized(step_cost: float = -0.1) -> WindyGridWorld:
    grid = windy_grid()
    grid.rewards.update(
        {
            (0, 0): step_cost,
            (0, 1): step_cost,
            (0, 2): step_cost,
            (1, 0): step_cost,
            (1, 2): step_cost,
            (2, 0): step_cost,
            (2, 1): step_cost,
            (2, 2): step_cost,
            (2, 3): step_cost,
        }
    )
    return grid


def transition_reward_tables(
    grid: GridWorld | WindyGridWorld,
) -> tuple[dict[tuple[State, Action, State], float], dict[tuple[State, Action, State], float]]:
    """Return transition probabilities and rewards in explicit table form."""
    transition_probs: dict[tuple[State, Action, State], float] = {}
    rewards: dict[tuple[State, Action, State], float] = {}

    if isinstance(grid, WindyGridWorld):
        for (state, action), next_state_probs in grid.transition_probs.items():
            for next_state, prob in next_state_probs.items():
                transition_probs[(state, action, next_state)] = prob
                rewards[(state, action, next_state)] = grid.rewards.get(next_state, 0.0)
        return transition_probs, rewards

    for state, actions in grid.actions.items():
        for action in actions:
            next_state = grid.get_next_state(state, action)
            transition_probs[(state, action, next_state)] = 1.0
            rewards[(state, action, next_state)] = grid.rewards.get(next_state, 0.0)

    return transition_probs, rewards


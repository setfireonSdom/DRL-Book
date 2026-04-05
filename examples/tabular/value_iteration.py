"""Value iteration on deterministic and windy GridWorlds."""

from __future__ import annotations

from examples.tabular.gridworld import ALL_ACTIONS, Action, GridWorld, standard_grid, transition_reward_tables, windy_grid
from examples.tabular.policy_evaluation import print_policy, print_values


def run_value_iteration(
    grid: GridWorld,
    *,
    gamma: float = 0.9,
    tolerance: float = 1e-3,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], Action]]:
    """Compute optimal values and a greedy policy with value iteration."""
    transition_probs, rewards = transition_reward_tables(grid)
    values = {state: 0.0 for state in grid.all_states()}

    while True:
        biggest_change = 0.0

        for state in grid.all_states():
            if grid.is_terminal(state):
                continue

            old_value = values[state]
            new_value = float("-inf")

            for action in grid.available_actions(state):
                candidate_value = 0.0
                for next_state in grid.all_states():
                    reward = rewards.get((state, action, next_state), 0.0)
                    transition_prob = transition_probs.get((state, action, next_state), 0.0)
                    candidate_value += transition_prob * (reward + gamma * values[next_state])

                new_value = max(new_value, candidate_value)

            values[state] = new_value
            biggest_change = max(biggest_change, abs(old_value - new_value))

        if biggest_change < tolerance:
            break

    policy: dict[tuple[int, int], Action] = {}
    for state in grid.actions:
        best_action = grid.available_actions(state)[0]
        best_value = float("-inf")

        for action in grid.available_actions(state):
            candidate_value = 0.0
            for next_state in grid.all_states():
                reward = rewards.get((state, action, next_state), 0.0)
                transition_prob = transition_probs.get((state, action, next_state), 0.0)
                candidate_value += transition_prob * (reward + gamma * values[next_state])

            if candidate_value > best_value:
                best_value = candidate_value
                best_action = action

        policy[state] = best_action

    return values, policy


def main() -> None:
    for name, grid in (("Deterministic Grid", standard_grid()), ("Windy Grid", windy_grid())):
        print(f"\n{name}")
        print("Rewards:")
        print_values(grid.rewards, grid)
        values, policy = run_value_iteration(grid)
        print("Optimal values:")
        print_values(values, grid)
        print("Optimal policy:")
        print_policy(policy, grid)


if __name__ == "__main__":
    main()


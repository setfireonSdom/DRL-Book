"""A compact Rainbow-ish CartPole example.

This teaching version combines:

- Double DQN target construction
- dueling network architecture
- prioritized replay

It intentionally omits the more complex Rainbow components such as
distributional RL and noisy nets so the implementation stays readable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.deep_rl.dqn_cartpole import clone_state_dict, epsilon_by_step, evaluate_policy, select_action
from examples.deep_rl.dueling_dqn_cartpole import DuelingQNetwork
from examples.deep_rl.prioritized_dqn_cartpole import PrioritizedReplayBuffer, beta_by_step
from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, hard_update, to_tensor


@dataclass
class RainbowishConfig:
    episodes: int = 400
    gamma: float = 0.99
    batch_size: int = 64
    learning_rate: float = 5e-4
    replay_capacity: int = 20_000
    min_replay_size: int = 1_000
    target_update_interval: int = 200
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
    hidden_dim: int = 128
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_epsilon: float = 1e-5
    gradient_clip_norm: float = 10.0
    eval_interval: int = 25
    eval_episodes: int = 50
    seed: int = 123


def optimize_rainbowish(
    q_network: DuelingQNetwork,
    target_network: DuelingQNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: PrioritizedReplayBuffer,
    *,
    config: RainbowishConfig,
    device: torch.device,
    rng: np.random.Generator,
    global_step: int,
) -> float | None:
    if len(replay_buffer) < config.min_replay_size:
        return None

    beta = beta_by_step(global_step, config)
    states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(
        config.batch_size,
        beta=beta,
        rng=rng,
    )

    state_tensor = to_tensor(states, device=device)
    action_tensor = to_tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    reward_tensor = to_tensor(rewards, device=device)
    next_state_tensor = to_tensor(next_states, device=device)
    done_tensor = to_tensor(dones.astype(np.float32), device=device)
    weight_tensor = to_tensor(weights, device=device)

    current_q_values = q_network(state_tensor).gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_actions = q_network(next_state_tensor).argmax(dim=1, keepdim=True)
        next_q_values = target_network(next_state_tensor).gather(1, next_actions).squeeze(1)
        targets = reward_tensor + config.gamma * (1.0 - done_tensor) * next_q_values

    td_errors = targets - current_q_values
    loss = (weight_tensor * td_errors.pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.gradient_clip_norm)
    optimizer.step()

    new_priorities = td_errors.detach().abs().cpu().numpy() + config.priority_epsilon
    replay_buffer.update_priorities(indices, new_priorities)
    return float(loss.item())


def train_rainbowish_cartpole(
    config: RainbowishConfig | None = None,
) -> tuple[DuelingQNetwork, list[float], list[float], float, torch.device]:
    config = config or RainbowishConfig()
    set_seed(config.seed)
    device = get_device(prefer_mps=True)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    seed_gym_env(env, config.seed)

    rng = np.random.default_rng(config.seed)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = DuelingQNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    target_network = DuelingQNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    hard_update(target_network, q_network)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = PrioritizedReplayBuffer(config.replay_capacity, alpha=config.priority_alpha)

    episode_rewards: list[float] = []
    losses: list[float] = []
    global_step = 0
    best_eval_reward = float("-inf")
    best_state_dict = clone_state_dict(q_network)

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            epsilon = epsilon_by_step(global_step, config)
            action = select_action(
                q_network,
                state,
                action_dim,
                epsilon=epsilon,
                device=device,
                rng=rng,
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            terminal = bool(terminated)

            replay_buffer.add(
                np.asarray(state, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                terminal,
            )

            loss = optimize_rainbowish(
                q_network,
                target_network,
                optimizer,
                replay_buffer,
                config=config,
                device=device,
                rng=rng,
                global_step=global_step,
            )
            if loss is not None:
                losses.append(loss)

            if global_step % config.target_update_interval == 0:
                hard_update(target_network, q_network)

            state = next_state
            episode_reward += reward
            global_step += 1

        episode_rewards.append(episode_reward)

        if (episode + 1) % config.eval_interval == 0:
            running_reward = float(np.mean(episode_rewards[-25:]))
            eval_reward = evaluate_policy(
                eval_env,
                q_network,
                device=device,
                episodes=10,
                seed=config.seed + 50_000 + episode,
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_state_dict = clone_state_dict(q_network)
            print(
                f"episode={episode + 1} "
                f"device={device} "
                f"epsilon={epsilon_by_step(global_step, config):.3f} "
                f"beta={beta_by_step(global_step, config):.3f} "
                f"avg_reward_25={running_reward:.2f} "
                f"greedy_eval_10={eval_reward:.2f}"
            )

    q_network.load_state_dict(best_state_dict)
    average_test_reward = evaluate_policy(
        eval_env,
        q_network,
        device=device,
        episodes=config.eval_episodes,
        seed=config.seed + 10_000,
    )

    env.close()
    eval_env.close()
    return q_network, episode_rewards, losses, average_test_reward, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact Rainbow-ish CartPole example.")
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, episode_rewards, losses, average_test_reward, device = train_rainbowish_cartpole(
        RainbowishConfig(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")

    plot_series(
        episode_rewards,
        title="Rainbow-ish CartPole Reward",
        ylabel="Reward",
        running_window=20,
    )

    if losses:
        plot_series(
            losses,
            title="Rainbow-ish CartPole Loss",
            ylabel="Loss",
            xlabel="Gradient step",
            running_window=100,
        )


if __name__ == "__main__":
    main()

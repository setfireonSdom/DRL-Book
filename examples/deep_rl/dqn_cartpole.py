"""PyTorch DQN for CartPole with MPS-aware device selection."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, hard_update, to_tensor


@dataclass
class DQNConfig:
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
    gradient_clip_norm: float = 10.0
    eval_interval: int = 25
    eval_episodes: int = 50
    seed: int = 123


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def epsilon_by_step(step: int, config: DQNConfig) -> float:
    fraction = min(step / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def select_action(
    q_network: QNetwork,
    state: np.ndarray,
    action_dim: int,
    *,
    epsilon: float,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(action_dim))

    with torch.no_grad():
        state_tensor = to_tensor(state, device=device).unsqueeze(0)
        q_values = q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def optimize(
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    *,
    config: DQNConfig,
    device: torch.device,
) -> float | None:
    if len(replay_buffer) < config.min_replay_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)

    state_tensor = to_tensor(states, device=device)
    action_tensor = to_tensor(actions, device=device, dtype=torch.long).unsqueeze(1)
    reward_tensor = to_tensor(rewards, device=device)
    next_state_tensor = to_tensor(next_states, device=device)
    done_tensor = to_tensor(dones.astype(np.float32), device=device)

    current_q_values = q_network(state_tensor).gather(1, action_tensor).squeeze(1)

    with torch.no_grad():
        next_q_values = target_network(next_state_tensor).max(dim=1).values
        targets = reward_tensor + config.gamma * (1.0 - done_tensor) * next_q_values

    loss = nn.functional.mse_loss(current_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_network.parameters(), config.gradient_clip_norm)
    optimizer.step()
    return float(loss.item())


def evaluate_policy(
    env: gym.Env,
    q_network: QNetwork,
    *,
    device: torch.device,
    episodes: int,
    seed: int,
) -> float:
    """Return mean episode return over several greedy evaluation episodes."""
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            with torch.no_grad():
                state_tensor = to_tensor(state, device=device).unsqueeze(0)
                action = int(torch.argmax(q_network(state_tensor), dim=1).item())
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return float(np.mean(rewards))


def clone_state_dict(module: nn.Module) -> dict[str, Any]:
    """Clone a module state dict so we can keep the best checkpoint in memory."""
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def train_dqn_cartpole(
    config: DQNConfig | None = None,
) -> tuple[QNetwork, list[float], list[float], float, torch.device]:
    config = config or DQNConfig()
    set_seed(config.seed)
    device = get_device(prefer_mps=True)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    seed_gym_env(env, config.seed)

    rng = np.random.default_rng(config.seed)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    target_network = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    hard_update(target_network, q_network)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)

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
            # A time-limit truncation is not the same as a true terminal failure
            # for the bootstrap target, so only `terminated` zeros out the target.
            terminal = bool(terminated)

            replay_buffer.add(
                np.asarray(state, dtype=np.float32),
                action,
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                terminal,
            )

            loss = optimize(
                q_network,
                target_network,
                optimizer,
                replay_buffer,
                config=config,
                device=device,
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
    parser = argparse.ArgumentParser(description="PyTorch DQN on CartPole.")
    parser.add_argument("--episodes", type=int, default=400, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, episode_rewards, losses, average_test_reward, device = train_dqn_cartpole(
        DQNConfig(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")

    plot_series(
        episode_rewards,
        title="DQN CartPole Reward",
        ylabel="Reward",
        running_window=20,
    )

    if losses:
        plot_series(
            losses,
            title="DQN CartPole Loss",
            ylabel="Loss",
            xlabel="Gradient step",
            running_window=100,
        )


if __name__ == "__main__":
    main()

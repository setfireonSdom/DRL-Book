"""PyTorch DDPG for Pendulum with MPS-aware device selection."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, soft_update, to_tensor


@dataclass
class DDPGConfig:
    episodes: int = 200
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 256
    action_noise: float = 0.1
    seed: int = 123


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_limit: float) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


def evaluate_policy(
    env: gym.Env,
    actor: Actor,
    *,
    device: torch.device,
    episodes: int,
    seed: int,
) -> float:
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0
        while not (done or truncated):
            with torch.no_grad():
                action = actor(to_tensor(state, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return float(np.mean(rewards))


def train_ddpg_pendulum(
    config: DDPGConfig | None = None,
) -> tuple[Actor, list[float], list[float], float, torch.device]:
    config = config or DDPGConfig()
    set_seed(config.seed)
    device = get_device(prefer_mps=True)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    seed_gym_env(env, config.seed)

    rng = np.random.default_rng(config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, config.hidden_dim, action_limit).to(device)
    critic = Critic(state_dim, action_dim, config.hidden_dim).to(device)
    target_actor = Actor(state_dim, action_dim, config.hidden_dim, action_limit).to(device)
    target_critic = Critic(state_dim, action_dim, config.hidden_dim).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.critic_lr)
    replay_buffer = ReplayBuffer(config.replay_capacity)

    rewards_per_episode: list[float] = []
    critic_losses: list[float] = []

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            with torch.no_grad():
                action = actor(to_tensor(state, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            action += rng.normal(0.0, config.action_noise, size=action_dim)
            action = np.clip(action, -action_limit, action_limit)

            next_state, reward, done, truncated, _ = env.step(action)
            terminal = bool(done or truncated)
            replay_buffer.add(
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                terminal,
            )

            if len(replay_buffer) >= config.min_replay_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size)
                state_tensor = to_tensor(states, device=device)
                action_tensor = to_tensor(actions, device=device)
                reward_tensor = to_tensor(rewards, device=device)
                next_state_tensor = to_tensor(next_states, device=device)
                done_tensor = to_tensor(dones.astype(np.float32), device=device)

                with torch.no_grad():
                    next_actions = target_actor(next_state_tensor)
                    target_q = reward_tensor + config.gamma * (1.0 - done_tensor) * target_critic(
                        next_state_tensor, next_actions
                    )

                current_q = critic(state_tensor, action_tensor)
                critic_loss = nn.functional.mse_loss(current_q, target_q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = -critic(state_tensor, actor(state_tensor)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(target_actor, actor, config.tau)
                soft_update(target_critic, critic, config.tau)
                critic_losses.append(float(critic_loss.item()))

            state = next_state
            episode_reward += reward

        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 20 == 0:
            running = float(np.mean(rewards_per_episode[-20:]))
            print(f"episode={episode + 1} device={device} avg_reward_20={running:.2f}")

    average_test_reward = evaluate_policy(eval_env, actor, device=device, episodes=10, seed=config.seed + 10_000)
    env.close()
    eval_env.close()
    return actor, rewards_per_episode, critic_losses, average_test_reward, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch DDPG on Pendulum.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, critic_losses, average_test_reward, device = train_ddpg_pendulum(
        DDPGConfig(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")

    plot_series(rewards_per_episode, title="DDPG Pendulum Reward", ylabel="Reward", running_window=10)
    if critic_losses:
        plot_series(
            critic_losses,
            title="DDPG Pendulum Critic Loss",
            ylabel="Loss",
            xlabel="Update step",
            running_window=100,
        )


if __name__ == "__main__":
    main()

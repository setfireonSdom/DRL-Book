"""PyTorch SAC for Pendulum with MPS-aware device selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.deep_rl.ddpg_pendulum import ReplayBuffer
from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, soft_update, to_tensor


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class SACConfig:
    episodes: int = 200
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden_dim: int = 256
    alpha: float = 0.2
    seed: int = 123


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, action_limit: float) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_limit

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self(state)
        return torch.tanh(mean) * self.action_limit


class CriticTwin(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


def evaluate_policy(
    env: gym.Env,
    actor: GaussianPolicy,
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
                action = actor.deterministic(to_tensor(state, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return float(np.mean(rewards))


def train_sac_pendulum(
    config: SACConfig | None = None,
) -> tuple[GaussianPolicy, list[float], list[float], float, torch.device]:
    config = config or SACConfig()
    set_seed(config.seed)
    device = get_device(prefer_mps=True)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    seed_gym_env(env, config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(env.action_space.high[0])

    actor = GaussianPolicy(state_dim, action_dim, config.hidden_dim, action_limit).to(device)
    critics = CriticTwin(state_dim, action_dim, config.hidden_dim).to(device)
    target_critics = CriticTwin(state_dim, action_dim, config.hidden_dim).to(device)
    target_critics.load_state_dict(critics.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=config.critic_lr)
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
                action_tensor, _ = actor.sample(to_tensor(state, device=device).unsqueeze(0))
                action = action_tensor.squeeze(0).cpu().numpy()

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
                    next_actions, next_log_probs = actor.sample(next_state_tensor)
                    target_q1, target_q2 = target_critics(next_state_tensor, next_actions)
                    target_q = torch.min(target_q1, target_q2) - config.alpha * next_log_probs
                    target = reward_tensor + config.gamma * (1.0 - done_tensor) * target_q

                current_q1, current_q2 = critics(state_tensor, action_tensor)
                critic_loss = nn.functional.mse_loss(current_q1, target) + nn.functional.mse_loss(current_q2, target)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))

                sampled_actions, log_probs = actor.sample(state_tensor)
                q1_pi, q2_pi = critics(state_tensor, sampled_actions)
                actor_loss = (config.alpha * log_probs - torch.min(q1_pi, q2_pi)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                soft_update(target_critics, critics, config.tau)

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
    parser = argparse.ArgumentParser(description="PyTorch SAC on Pendulum.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, critic_losses, average_test_reward, device = train_sac_pendulum(
        SACConfig(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")

    plot_series(rewards_per_episode, title="SAC Pendulum Reward", ylabel="Reward", running_window=10)
    if critic_losses:
        plot_series(
            critic_losses,
            title="SAC Pendulum Critic Loss",
            ylabel="Loss",
            xlabel="Update step",
            running_window=100,
        )


if __name__ == "__main__":
    main()

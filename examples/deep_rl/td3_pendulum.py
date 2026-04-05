"""PyTorch TD3 for Pendulum with MPS-aware device selection."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.deep_rl.ddpg_pendulum import Actor, ReplayBuffer, evaluate_policy
from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, soft_update, to_tensor


@dataclass
class TD3Config:
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
    target_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    seed: int = 123


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

    def q1_only(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([state, action], dim=-1)).squeeze(-1)


def train_td3_pendulum(
    config: TD3Config | None = None,
) -> tuple[Actor, list[float], list[float], float, torch.device]:
    config = config or TD3Config()
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
    target_actor = Actor(state_dim, action_dim, config.hidden_dim, action_limit).to(device)
    target_actor.load_state_dict(actor.state_dict())

    critics = CriticTwin(state_dim, action_dim, config.hidden_dim).to(device)
    target_critics = CriticTwin(state_dim, action_dim, config.hidden_dim).to(device)
    target_critics.load_state_dict(critics.state_dict())

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
    critic_optimizer = torch.optim.Adam(critics.parameters(), lr=config.critic_lr)
    replay_buffer = ReplayBuffer(config.replay_capacity)

    rewards_per_episode: list[float] = []
    critic_losses: list[float] = []
    update_step = 0

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
                    noise = torch.randn_like(action_tensor) * config.target_noise
                    noise = noise.clamp(-config.noise_clip, config.noise_clip)
                    next_actions = (target_actor(next_state_tensor) + noise).clamp(-action_limit, action_limit)
                    target_q1, target_q2 = target_critics(next_state_tensor, next_actions)
                    target_q = reward_tensor + config.gamma * (1.0 - done_tensor) * torch.min(target_q1, target_q2)

                current_q1, current_q2 = critics(state_tensor, action_tensor)
                critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(
                    current_q2, target_q
                )
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))

                if update_step % config.policy_delay == 0:
                    actor_loss = -critics.q1_only(state_tensor, actor(state_tensor)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    soft_update(target_actor, actor, config.tau)
                    soft_update(target_critics, critics, config.tau)

                update_step += 1

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
    parser = argparse.ArgumentParser(description="PyTorch TD3 on Pendulum.")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, critic_losses, average_test_reward, device = train_td3_pendulum(
        TD3Config(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")

    plot_series(rewards_per_episode, title="TD3 Pendulum Reward", ylabel="Reward", running_window=10)
    if critic_losses:
        plot_series(
            critic_losses,
            title="TD3 Pendulum Critic Loss",
            ylabel="Loss",
            xlabel="Update step",
            running_window=100,
        )


if __name__ == "__main__":
    main()

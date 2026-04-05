"""REINFORCE baseline for CartPole."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.deep_rl.on_policy_utils import clone_state_dict, evaluate_greedy_policy
from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed
from examples.shared.torch_utils import get_device, to_tensor


@dataclass
class REINFORCEConfig:
    episodes: int = 500
    gamma: float = 0.99
    learning_rate: float = 5e-4
    hidden_dim: int = 128
    normalize_returns: bool = True
    entropy_weight: float = 1e-3
    gradient_clip_norm: float = 10.0
    eval_interval: int = 25
    eval_episodes: int = 50
    seed: int = 123


class PolicyNetwork(nn.Module):
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


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def train_reinforce_cartpole(
    config: REINFORCEConfig | None = None,
) -> tuple[PolicyNetwork, list[float], float, torch.device]:
    config = config or REINFORCEConfig()
    set_seed(config.seed)
    device = get_device(prefer_mps=True)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")
    seed_gym_env(env, config.seed)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    rewards_per_episode: list[float] = []
    best_eval_reward = float("-inf")
    best_state_dict = clone_state_dict(policy)

    for episode in range(config.episodes):
        state, _ = env.reset(seed=config.seed + episode)
        terminated = False
        truncated = False
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        rewards: list[float] = []

        while not (terminated or truncated):
            logits = policy(to_tensor(state, device=device).unsqueeze(0))
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            rewards.append(float(reward))
            state = next_state

        return_tensor = to_tensor(discounted_returns(rewards, config.gamma), device=device)
        if config.normalize_returns and len(return_tensor) > 1:
            return_tensor = (return_tensor - return_tensor.mean()) / (return_tensor.std() + 1e-8)

        policy_loss = torch.stack([-log_prob * ret for log_prob, ret in zip(log_probs, return_tensor)]).sum()
        entropy_bonus = torch.stack(entropies).mean()
        loss = policy_loss - config.entropy_weight * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip_norm)
        optimizer.step()

        rewards_per_episode.append(float(sum(rewards)))

        if (episode + 1) % config.eval_interval == 0:
            running_reward = float(np.mean(rewards_per_episode[-25:]))
            eval_reward = evaluate_greedy_policy(
                eval_env,
                policy,
                device=device,
                episodes=10,
                seed=config.seed + 50_000 + episode,
            )
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_state_dict = clone_state_dict(policy)
            print(
                f"episode={episode + 1} "
                f"device={device} "
                f"avg_reward_25={running_reward:.2f} "
                f"policy_eval_10={eval_reward:.2f}"
            )

    policy.load_state_dict(best_state_dict)
    average_test_reward = evaluate_greedy_policy(
        eval_env,
        policy,
        device=device,
        episodes=config.eval_episodes,
        seed=config.seed + 10_000,
    )
    env.close()
    eval_env.close()
    return policy, rewards_per_episode, average_test_reward, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REINFORCE on CartPole.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, average_test_reward, device = train_reinforce_cartpole(
        REINFORCEConfig(episodes=args.episodes)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device}")
    plot_series(rewards_per_episode, title="REINFORCE CartPole Reward", ylabel="Reward", running_window=20)


if __name__ == "__main__":
    main()

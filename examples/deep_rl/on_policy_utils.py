"""Shared utilities for discrete-action on-policy examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from examples.shared.torch_utils import to_tensor


class DiscretePolicyValueNet(nn.Module):
    """Small shared backbone with separate policy / value heads."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


@dataclass
class EpisodeTrajectory:
    states: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    log_probs: list[float]
    values: list[float]
    total_reward: float


def clone_state_dict(module: nn.Module) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def compute_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float32)
    running_return = 0.0
    for index in reversed(range(len(rewards))):
        running_return = rewards[index] + gamma * running_return
        returns[index] = running_return
    return returns


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 or dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        last_gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * last_gae
        advantages[t] = last_gae

    returns = advantages + np.asarray(values, dtype=np.float32)
    return returns, advantages


def evaluate_greedy_policy(
    env: gym.Env,
    model: DiscretePolicyValueNet,
    *,
    device: torch.device,
    episodes: int,
    seed: int,
) -> float:
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            with torch.no_grad():
                logits, _ = model(to_tensor(state, device=device).unsqueeze(0))
                action = int(torch.argmax(logits, dim=1).item())
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return float(np.mean(rewards))


def collect_episode(
    env: gym.Env,
    model: DiscretePolicyValueNet,
    *,
    device: torch.device,
    seed: int,
) -> EpisodeTrajectory:
    state, _ = env.reset(seed=seed)
    terminated = False
    truncated = False

    states: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[float] = []
    values: list[float] = []

    while not (terminated or truncated):
        state_tensor = to_tensor(state, device=device).unsqueeze(0)
        logits, value = model(state_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample()
        next_state, reward, terminated, truncated, _ = env.step(int(action.item()))

        states.append(np.asarray(state, dtype=np.float32))
        actions.append(int(action.item()))
        rewards.append(float(reward))
        dones.append(bool(terminated or truncated))
        log_probs.append(float(distribution.log_prob(action).item()))
        values.append(float(value.item()))
        state = next_state

    return EpisodeTrajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        log_probs=log_probs,
        values=values,
        total_reward=float(sum(rewards)),
    )

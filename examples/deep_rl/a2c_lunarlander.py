"""A2C on LunarLander using Stable-Baselines3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from stable_baselines3 import A2C

from examples.deep_rl.sb3_lunarlander_utils import SB3Algo, train_sb3_lunarlander
from examples.shared.plotting import plot_series
from examples.shared.seed import set_seed
from examples.shared.torch_utils import get_device


@dataclass
class A2CConfig:
    episodes: int = 1000
    total_timesteps: int = 750_000
    hidden_dim: int = 256
    seed: int = 123


def train_a2c_lunarlander(
    config: A2CConfig | None = None,
) -> tuple[SB3Algo, list[float], float, str]:
    config = config or A2CConfig()
    set_seed(config.seed)
    device = str(get_device(prefer_mps=False))
    result = train_sb3_lunarlander(
        algo_name="a2c_lunarlander",
        algo_class=A2C,
        total_timesteps=config.total_timesteps,
        eval_interval_episodes=50,
        eval_episodes=30,
        seed=config.seed,
        policy_kwargs=dict(net_arch=dict(pi=[config.hidden_dim, config.hidden_dim], vf=[config.hidden_dim, config.hidden_dim])),
        algo_kwargs=dict(
            learning_rate=7e-4,
            n_steps=32,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.25,
            max_grad_norm=0.5,
            normalize_advantage=True,
            n_envs=8,
            device=device,
        ),
    )
    return result.model, result.rewards_per_episode, result.average_test_reward, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2C on LunarLander.")
    parser.add_argument("--episodes", type=int, default=1000, help="Approximate episode budget for reporting.")
    parser.add_argument("--timesteps", type=int, default=750000, help="Total training timesteps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, average_test_reward, device = train_a2c_lunarlander(
        A2CConfig(episodes=args.episodes, total_timesteps=args.timesteps)
    )
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    print(f"Training device: {device} (SB3 defaulted to CPU for stability)")
    if rewards_per_episode:
        plot_series(
            rewards_per_episode,
            title="A2C LunarLander Reward",
            ylabel="Reward",
            running_window=20,
        )


if __name__ == "__main__":
    main()

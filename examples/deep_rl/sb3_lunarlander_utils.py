"""Shared Stable-Baselines3 utilities for LunarLander policy-gradient examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor


SB3Algo = PPO | A2C


@dataclass
class SB3TrainResult:
    model: SB3Algo
    rewards_per_episode: list[float]
    average_test_reward: float


class EpisodeLoggingCallback(BaseCallback):
    def __init__(self, label: str, log_interval_episodes: int) -> None:
        super().__init__()
        self.label = label
        self.log_interval_episodes = log_interval_episodes
        self.rewards_per_episode: list[float] = []
        self._next_log = log_interval_episodes

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                reward = float(info["episode"]["r"])
                self.rewards_per_episode.append(reward)
                if len(self.rewards_per_episode) >= self._next_log:
                    running = float(np.mean(self.rewards_per_episode[-25:]))
                    print(f"episode={self._next_log} {self.label} avg_reward_25={running:.2f}")
                    self._next_log += self.log_interval_episodes
        return True


def make_monitored_env(seed: int) -> gym.Env:
    env = gym.make("LunarLander-v3")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate_sb3_model(model: SB3Algo, *, episodes: int, seed: int) -> float:
    eval_env = Monitor(gym.make("LunarLander-v3"))
    eval_env.reset(seed=seed)
    average_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=episodes, deterministic=True)
    eval_env.close()
    return float(average_reward)


def train_sb3_lunarlander(
    *,
    algo_name: str,
    algo_class: type[SB3Algo],
    total_timesteps: int,
    eval_interval_episodes: int,
    eval_episodes: int,
    seed: int,
    policy_kwargs: dict[str, Any] | None = None,
    algo_kwargs: dict[str, Any] | None = None,
) -> SB3TrainResult:
    policy_kwargs = policy_kwargs or {}
    algo_kwargs = algo_kwargs or {}

    num_envs = int(algo_kwargs.pop("n_envs", 8))
    train_env = make_vec_env("LunarLander-v3", n_envs=num_envs, seed=seed)
    train_env = VecMonitor(train_env)

    eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=seed + 10_000)
    eval_env = VecMonitor(eval_env)

    log_callback = EpisodeLoggingCallback(label=f"algo={algo_name}", log_interval_episodes=eval_interval_episodes)
    best_model_dir = Path("examples/.sb3_checkpoints") / algo_name
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(best_model_dir),
        eval_freq=max(total_timesteps // max(eval_interval_episodes, 1), 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    model = algo_class(
        "MlpPolicy",
        train_env,
        seed=seed,
        policy_kwargs=policy_kwargs,
        verbose=0,
        **algo_kwargs,
    )
    model.learn(total_timesteps=total_timesteps, callback=[log_callback, eval_callback], progress_bar=False)

    best_model_path = best_model_dir / "best_model.zip"
    if best_model_path.exists():
        model = algo_class.load(best_model_path, env=train_env)

    average_test_reward = evaluate_sb3_model(model, episodes=eval_episodes, seed=seed + 20_000)
    train_env.close()
    eval_env.close()
    return SB3TrainResult(model=model, rewards_per_episode=log_callback.rewards_per_episode, average_test_reward=average_test_reward)

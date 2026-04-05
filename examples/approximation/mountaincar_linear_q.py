"""Linear function approximation for MountainCar with Gymnasium."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed


class SGDRegressor:
    """A minimal SGD regressor for readable value approximation examples."""

    def __init__(self, dimension: int, learning_rate: float = 0.05) -> None:
        # Zero-initialized Q-values are optimistic in MountainCar because most
        # observed rewards are negative.
        self.weights = np.zeros(dimension, dtype=np.float64)
        self.learning_rate = learning_rate

    def partial_fit(self, x: np.ndarray, y: float) -> None:
        prediction = float(x @ self.weights)
        self.weights += self.learning_rate * (y - prediction) * x

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.weights)


@dataclass
class FeatureTransformer:
    scaler: StandardScaler
    featurizer: FeatureUnion
    dimension: int

    @classmethod
    def from_env(cls, env: gym.Env, *, seed: int, n_samples: int = 8_000) -> "FeatureTransformer":
        rng = np.random.default_rng(seed)
        observation_examples = np.array([env.observation_space.sample() for _ in range(n_samples)])
        rng.shuffle(observation_examples)

        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion(
            [
                ("rbf_very_narrow", RBFSampler(gamma=5.0, n_components=300, random_state=seed)),
                ("rbf_narrow", RBFSampler(gamma=2.0, n_components=300, random_state=seed + 1)),
                ("rbf_medium", RBFSampler(gamma=1.0, n_components=300, random_state=seed + 2)),
                ("rbf_wide", RBFSampler(gamma=0.5, n_components=300, random_state=seed + 3)),
            ]
        )

        features = featurizer.fit_transform(scaler.transform(observation_examples))
        return cls(scaler=scaler, featurizer=featurizer, dimension=features.shape[1])

    def transform(self, observation: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.atleast_2d(observation))
        return self.featurizer.transform(scaled)[0]


class LinearQAgent:
    def __init__(
        self,
        env: gym.Env,
        feature_transformer: FeatureTransformer,
        *,
        learning_rate: float = 0.05,
    ) -> None:
        self.env = env
        self.feature_transformer = feature_transformer
        self.models = [
            SGDRegressor(feature_transformer.dimension, learning_rate=learning_rate)
            for _ in range(env.action_space.n)
        ]

    def predict(self, observation: np.ndarray) -> np.ndarray:
        features = self.feature_transformer.transform(observation)
        return np.array([model.predict(features) for model in self.models], dtype=np.float64)

    def update(self, observation: np.ndarray, action: int, target: float) -> None:
        features = self.feature_transformer.transform(observation)
        self.models[action].partial_fit(features, target)

    def sample_action(self, observation: np.ndarray, *, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.predict(observation)))


def evaluate_agent(
    env: gym.Env,
    agent: LinearQAgent,
    *,
    episodes: int = 50,
    seed: int = 123,
) -> float:
    rewards = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action = int(np.argmax(agent.predict(observation)))
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return float(np.mean(rewards))


def plot_cost_to_go(env: gym.Env, agent: LinearQAgent, *, num_tiles: int = 30) -> None:
    """Visualize the learned negative value landscape."""
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    grid_x, grid_y = np.meshgrid(x, y)

    stacked = np.dstack([grid_x, grid_y])
    cost_to_go = np.apply_along_axis(lambda obs: -np.max(agent.predict(obs)), 2, stacked)

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(grid_x, grid_y, cost_to_go, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-to-go = -max_a Q(s, a)")
    ax.set_title("MountainCar Cost-to-Go")
    fig.colorbar(surface, shrink=0.7, aspect=12)
    fig.tight_layout()
    plt.show()


def train_mountaincar_linear_q(
    *,
    episodes: int = 350,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    learning_rate: float = 0.05,
    seed: int = 123,
) -> tuple[LinearQAgent, list[float], float]:
    """Train a linear Q-learning agent on MountainCar-v0."""
    set_seed(seed)
    env = gym.make("MountainCar-v0")
    seed_gym_env(env, seed)
    rng = np.random.default_rng(seed)

    feature_transformer = FeatureTransformer.from_env(env, seed=seed)
    agent = LinearQAgent(env, feature_transformer, learning_rate=learning_rate)
    rewards_per_episode: list[float] = []

    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_start / (1.0 + 0.1 * episode))
        observation, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action = agent.sample_action(observation, epsilon=epsilon, rng=rng)
            next_observation, reward, done, truncated, _ = env.step(action)

            if done and not truncated:
                target = reward
            else:
                target = reward + gamma * float(np.max(agent.predict(next_observation)))

            agent.update(observation, action, target)
            observation = next_observation
            episode_reward += reward

        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 25 == 0:
            running = np.mean(rewards_per_episode[-25:])
            print(f"episode={episode + 1} epsilon={epsilon:.3f} avg_reward_25={running:.2f}")

    eval_env = gym.make("MountainCar-v0")
    average_test_reward = evaluate_agent(eval_env, agent, seed=seed + 10_000)
    env.close()
    eval_env.close()
    return agent, rewards_per_episode, average_test_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear Q-learning with RBF features on MountainCar.")
    parser.add_argument("--episodes", type=int, default=350, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent, rewards_per_episode, average_test_reward = train_mountaincar_linear_q(episodes=args.episodes)
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")

    plot_series(
        rewards_per_episode,
        title="MountainCar Linear Q-Learning",
        ylabel="Reward",
        running_window=20,
    )

    env = gym.make("MountainCar-v0")
    plot_cost_to_go(env, agent)
    env.close()


if __name__ == "__main__":
    main()

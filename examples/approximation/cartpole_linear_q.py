"""Linear function approximation for CartPole with Gymnasium."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from examples.shared.plotting import plot_series
from examples.shared.seed import seed_gym_env, set_seed


class SGDRegressor:
    """A tiny SGD regressor so the example stays dependency-light and readable."""

    def __init__(self, dimension: int, learning_rate: float = 0.01) -> None:
        # Zero initialization tends to be more stable than random weights for
        # this small linear-Q teaching example.
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

        # Sampling observation_space directly can overrepresent extreme values.
        # Random states in a bounded range are enough for a lightweight teaching example.
        observation_examples = rng.uniform(low=-1.0, high=1.0, size=(n_samples, env.observation_space.shape[0]))

        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion(
            [
                ("rbf_small", RBFSampler(gamma=0.05, n_components=250, random_state=seed)),
                ("rbf_medium", RBFSampler(gamma=0.5, n_components=250, random_state=seed + 1)),
                ("rbf_large", RBFSampler(gamma=1.0, n_components=250, random_state=seed + 2)),
                ("rbf_wide", RBFSampler(gamma=0.1, n_components=250, random_state=seed + 3)),
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
        learning_rate: float = 0.01,
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
    """Run a greedy evaluation loop and return average episode reward."""
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


def train_cartpole_linear_q(
    *,
    episodes: int = 500,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    learning_rate: float = 0.01,
    seed: int = 123,
) -> tuple[LinearQAgent, list[float], float]:
    """Train a linear Q-learning agent on CartPole-v1."""
    set_seed(seed)
    env = gym.make("CartPole-v1")
    seed_gym_env(env, seed)
    rng = np.random.default_rng(seed)

    feature_transformer = FeatureTransformer.from_env(env, seed=seed)
    agent = LinearQAgent(env, feature_transformer, learning_rate=learning_rate)
    rewards_per_episode: list[float] = []

    for episode in range(episodes):
        epsilon = max(epsilon_end, epsilon_start * (0.995**episode))
        observation, _ = env.reset(seed=seed + episode)
        done = False
        truncated = False
        episode_reward = 0.0

        while not (done or truncated):
            action = agent.sample_action(observation, epsilon=epsilon, rng=rng)
            next_observation, reward, done, truncated, _ = env.step(action)

            if done or truncated:
                target = reward
            else:
                target = reward + gamma * float(np.max(agent.predict(next_observation)))

            agent.update(observation, action, target)
            observation = next_observation
            episode_reward += reward

        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 50 == 0:
            running = np.mean(rewards_per_episode[-50:])
            print(f"episode={episode + 1} epsilon={epsilon:.3f} avg_reward_50={running:.2f}")

    eval_env = gym.make("CartPole-v1")
    average_test_reward = evaluate_agent(eval_env, agent, seed=seed + 10_000)
    env.close()
    eval_env.close()
    return agent, rewards_per_episode, average_test_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear Q-learning with RBF features on CartPole.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, rewards_per_episode, average_test_reward = train_cartpole_linear_q(episodes=args.episodes)
    print(f"Average greedy evaluation reward: {average_test_reward:.2f}")
    plot_series(
        rewards_per_episode,
        title="CartPole Linear Q-Learning",
        ylabel="Reward",
        running_window=20,
    )


if __name__ == "__main__":
    main()

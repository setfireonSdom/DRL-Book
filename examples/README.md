# Examples

This directory is the new home for runnable, maintained reinforcement learning examples used by this project.

The migration strategy is:

1. Rebuild the teaching path from simple tabular environments to modern deep RL.
2. Keep examples small, readable, and executable on a current Python stack.
3. Standardize the maintained stack on NumPy + Gymnasium + PyTorch.
4. Separate actively maintained code from archived legacy code in `machine_learning_examples/`.

Planned layout:

- `bandits/`: multi-armed bandit examples
- `tabular/`: gridworld, DP, Monte Carlo, TD, SARSA, Q-learning
- `approximation/`: linear / feature-based value approximation
- `deep_rl/`: PyTorch implementations of DQN, policy gradient, actor-critic, continuous control
- `shared/`: reusable environments, plotting, training utilities, seeding helpers

Until migration is complete, `machine_learning_examples/` remains the legacy reference.

## Setup

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

If you are using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## How To Run

Run examples from the project root with `python3 -m ...`.

For training-based examples, the default episode counts are kept as the maintained baseline. If you want a quicker smoke test, many approximation and deep RL scripts now also accept `--episodes` to temporarily override the default without editing code:

```bash
python3 -m examples.approximation.cartpole_linear_q --episodes 200
python3 -m examples.deep_rl.dqn_cartpole --episodes 100
python3 -m examples.deep_rl.ppo_lunarlander --timesteps 200000
```

Bandits:

```bash
python3 -m examples.bandits.epsilon_greedy
python3 -m examples.bandits.optimistic_initial_values
python3 -m examples.bandits.ucb1
```

Tabular:

```bash
python3 -m examples.tabular.policy_iteration
python3 -m examples.tabular.value_iteration
python3 -m examples.tabular.monte_carlo_prediction
python3 -m examples.tabular.td0_prediction
python3 -m examples.tabular.sarsa
python3 -m examples.tabular.q_learning
```

Approximation:

```bash
python3 -m examples.approximation.cartpole_linear_q
python3 -m examples.approximation.mountaincar_linear_q
```

Deep RL:

```bash
python3 -m examples.deep_rl.dqn_cartpole
python3 -m examples.deep_rl.double_dqn_cartpole
python3 -m examples.deep_rl.dueling_dqn_cartpole
python3 -m examples.deep_rl.prioritized_dqn_cartpole
python3 -m examples.deep_rl.rainbowish_cartpole

# Self-implemented baseline
python3 -m examples.deep_rl.reinforce_cartpole

# Stable-Baselines3-backed policy-gradient / actor-critic baselines
python3 -m examples.deep_rl.actor_critic_lunarlander
python3 -m examples.deep_rl.a2c_lunarlander
python3 -m examples.deep_rl.ppo_lunarlander

# Self-implemented continuous-control baselines
python3 -m examples.deep_rl.ddpg_pendulum
python3 -m examples.deep_rl.td3_pendulum
python3 -m examples.deep_rl.sac_pendulum
```

## Current Coverage

Implemented now:

- Bandits: epsilon-greedy, optimistic initial values, UCB1
- Tabular RL: policy evaluation, policy iteration, value iteration, Monte Carlo prediction, TD(0), SARSA, Q-learning
- Function approximation: linear / RBF-style Q-learning on CartPole and MountainCar
- Deep RL (self-implemented): DQN, Double DQN, Dueling DQN, prioritized-replay DQN, a Rainbow-ish value-based variant, REINFORCE on CartPole, plus DDPG, TD3, and SAC on Pendulum in PyTorch
- Deep RL (third-party implementation): actor-critic / A2C / PPO on LunarLander via Stable-Baselines3

Current policy-gradient status:

- `examples.deep_rl.reinforce_cartpole` is a self-implemented baseline kept as the simplest policy-gradient reference.
- `examples.deep_rl.ppo_lunarlander` is currently the main policy-gradient path that has been manually verified to learn with the Stable-Baselines3 implementation.
- `examples.deep_rl.actor_critic_lunarlander` and `examples.deep_rl.a2c_lunarlander` are also backed by Stable-Baselines3 rather than local from-scratch implementations.
- `examples.deep_rl.ddpg_pendulum`, `examples.deep_rl.td3_pendulum`, and `examples.deep_rl.sac_pendulum` are local implementations and have now shown clear learning behavior in manual Pendulum runs. Treat them as validated self-implemented continuous-control references, while keeping in mind that this validation is still lighter than a full multi-seed benchmark study.

Box2D note:

- The LunarLander examples need Box2D support from Gymnasium.
- The bundled [examples/requirements.txt](/Users/yummmy/Downloads/book/RL_Book/examples/requirements.txt) now installs `gymnasium[box2d]` and `swig`.
- `actor_critic_lunarlander`, `a2c_lunarlander`, and `ppo_lunarlander` also depend on `stable-baselines3`.

Practical note on the approximation examples:

- `examples.approximation.cartpole_linear_q` is currently stable enough to serve as the main bridge from tabular Q-learning to deep value learning.
- `examples.approximation.mountaincar_linear_q` appears logically correct, but it is more sensitive to feature choice and hyperparameters, so treat it as a useful but less stable teaching example.

Not implemented yet:

- A3C
- Atari-scale DQN

## Apple Silicon Notes

The PyTorch examples use `examples/shared/torch_utils.py` to choose a device.

Current priority is:

1. `cuda` if available
2. `mps` on Apple Silicon
3. `cpu` otherwise

So on your Mac M4, the deep RL examples should prefer `mps` automatically as long as your installed PyTorch build supports it.

## Philosophy

These examples are intentionally smaller and cleaner than the old legacy scripts.

The goal is:

- readable teaching code first
- current Python ecosystem compatibility
- a stable path from classical RL to deep RL

The goal is not:

- reproducing every old legacy implementation exactly
- maximizing benchmark performance
- packing every modern trick into the first pass

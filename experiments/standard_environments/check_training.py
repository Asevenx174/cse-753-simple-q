"""Check Tianshou collection and one DQN learning update."""

import gymnasium as gym
import numpy as np
import torch

from tianshou.algorithm.modelfree.dqn import (
    DQN,
    DiscreteQLearningPolicy,
)
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.algorithm.algorithm_base import policy_within_training_step

from src.agents.q_network import QNetwork


SEED = 42


def parameters(model):
    """Copy parameters so later changes can be detected."""
    return [parameter.detach().clone() for parameter in model.parameters()]


def changed(before, after):
    """Return True if at least one parameter changed."""
    return any(
        not torch.equal(old, new)
        for old, new in zip(before, after)
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    example_env = gym.make("CartPole-v1")
    network = QNetwork(4, 2)

    policy = DiscreteQLearningPolicy(
        model=network,
        action_space=example_env.action_space,
        observation_space=example_env.observation_space,
        eps_training=0.1,
    )

    algorithm = DQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=0.001),
        gamma=0.99,
        target_update_freq=100,
        is_double=False,
    )

    environments = DummyVectorEnv(
        [lambda: gym.make("CartPole-v1")]
    )
    replay = VectorReplayBuffer(
        total_size=500,
        buffer_num=1,
    )
    collector = Collector(algorithm, environments, replay)

    collector.collect(
        n_step=100,
        random=True,
        reset_before_collect=True,
        gym_reset_kwargs={"seed": SEED},
    )

    sample, _ = replay.sample(4)
    required = ("obs", "act", "rew", "obs_next", "terminated")

    assert all(hasattr(sample, field) for field in required)
    print("Replay fields:", required)
    print("Stored transitions:", len(replay))

    online_before = parameters(network)
    target_before = parameters(algorithm.model_old)

    with policy_within_training_step(algorithm.policy):
        statistics = algorithm.update(replay, sample_size=32)

    online_after = parameters(network)
    target_after = parameters(algorithm.model_old)

    online_changed = changed(online_before, online_after)
    target_changed = changed(target_before, target_after)

    assert np.isfinite(statistics.loss)
    # online network check
    assert online_changed
    # target network check
    assert not target_changed

    print("Loss:", statistics.loss)
    print("Online weights changed:", online_changed)
    print("Target weights changed:", target_changed)

    environments.close()
    example_env.close()


if __name__ == "__main__":
    main()
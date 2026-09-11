"""Create DQN or Double DQN through one shared path."""

from tianshou.algorithm.modelfree.dqn import (
    DiscreteQLearningPolicy,
)
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.agents.masked_dqn import MaskedDQN


def build_dqn(
    network,
    observation_space,
    action_space,
    algorithm_name: str,
):
    """Build DQN or Double DQN with shared settings."""

    if algorithm_name not in ("dqn", "double_dqn"):
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}"
        )

    policy = DiscreteQLearningPolicy(
        model=network,
        observation_space=observation_space,
        action_space=action_space,
        eps_training=0.1,
        eps_inference=0.0,
    )

    return MaskedDQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=0.001),
        gamma=0.99,
        n_step_return_horizon=1,
        target_update_freq=100,
        is_double=(algorithm_name == "double_dqn"),
    )
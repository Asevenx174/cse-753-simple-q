"""Create DQN and Double-DQN agents."""

from tianshou.algorithm.modelfree.dqn import (
    DiscreteQLearningPolicy,
)
from tianshou.algorithm.optim import AdamOptimizerFactory

from src.agents.masked_dqn import MaskedDQN


def build_dqn(
    network,
    observation_space,
    action_space,
    algorithm_name,
    epsilon=0.1,
    target_update_frequency=100,
):
    """Build DQN or Double DQN."""

    if algorithm_name not in ("dqn", "double_dqn"):
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    policy = DiscreteQLearningPolicy(
        model=network,
        observation_space=observation_space,
        action_space=action_space,
        eps_training=epsilon,
        eps_inference=0.0,
    )

    return MaskedDQN(
        policy=policy,
        optim=AdamOptimizerFactory(lr=0.001),
        gamma=0.99,
        n_step_return_horizon=1,
        target_update_freq=target_update_frequency,
        is_double=(algorithm_name == "double_dqn"),
    )
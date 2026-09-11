"""Verify the shared DQN/Double-DQN switch."""

import copy
import unittest

import gymnasium as gym
import torch

from src.agents.dqn_factory import build_dqn
from src.agents.q_network import QNetwork


class TestDQNSwitch(unittest.TestCase):

    def test_only_double_setting_changes(self) -> None:
        torch.manual_seed(42)
        environment = gym.make("CartPole-v1")

        base_network = QNetwork(4, 2)
        dqn_network = copy.deepcopy(base_network)
        double_network = copy.deepcopy(base_network)

        dqn = build_dqn(
            dqn_network,
            environment.observation_space,
            environment.action_space,
            "dqn",
        )
        double_dqn = build_dqn(
            double_network,
            environment.observation_space,
            environment.action_space,
            "double_dqn",
        )

        self.assertFalse(dqn.is_double)
        self.assertTrue(double_dqn.is_double)

        for dqn_parameter, double_parameter in zip(
            dqn_network.parameters(),
            double_network.parameters(),
        ):
            self.assertTrue(
                torch.equal(dqn_parameter, double_parameter)
            )

        environment.close()

    def test_invalid_algorithm_is_rejected(self) -> None:
        environment = gym.make("CartPole-v1")

        with self.assertRaises(ValueError):
            build_dqn(
                QNetwork(4, 2),
                environment.observation_space,
                environment.action_space,
                "unknown",
            )

        environment.close()


if __name__ == "__main__":
    unittest.main()
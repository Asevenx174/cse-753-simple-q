"""Integration tests for Tianshou action masks."""

import unittest

import numpy as np
import torch
from tianshou.algorithm.algorithm_base import (
    policy_within_training_step,
)
from tianshou.data import Batch

from src.agents.dqn_factory import build_dqn
from src.agents.q_network import QNetwork
from src.environments.noisy_max import NoisyMaxEnv


class TestActionMasks(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

        self.environment = NoisyMaxEnv(action_count=10)
        self.network = QNetwork(2, 10)

        self.algorithm = build_dqn(
            self.network,
            self.environment.observation_space,
            self.environment.action_space,
            "dqn",
        )

    def tearDown(self):
        self.environment.close()

    def make_initial_batch(self):
        observation, info = self.environment.reset(seed=42)

        return Batch(
            obs=Batch(
                obs=observation["obs"][None, :],
                mask=observation["mask"][None, :],
            ),
            info=[info],
        )

    def test_greedy_action_respects_mask(self):
        batch = self.make_initial_batch()

        with torch.no_grad():
            result = self.algorithm.policy(batch)

        self.assertIn(int(result.act[0]), (0, 1))

    def test_exploration_respects_mask(self):
        batch = self.make_initial_batch()
        original_epsilon = (
            self.algorithm.policy.eps_training
        )
        self.algorithm.policy.eps_training = 1.0

        try:
            with policy_within_training_step(
                self.algorithm.policy
            ):
                for _ in range(500):
                    action = np.array([0])
                    explored = (
                        self.algorithm.policy
                        .add_exploration_noise(
                            action,
                            batch,
                        )
                    )
                    self.assertIn(
                        int(explored[0]),
                        (0, 1),
                    )
        finally:
            self.algorithm.policy.eps_training = (
                original_epsilon
            )


if __name__ == "__main__":
    unittest.main()
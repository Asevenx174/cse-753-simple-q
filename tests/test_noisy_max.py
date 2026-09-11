"""Tests for the NoisyMax environment."""

import unittest

from src.environments.noisy_max import NoisyMaxEnv


class TestNoisyMax(unittest.TestCase):

    def test_initial_observation_and_mask(self):
        environment = NoisyMaxEnv(action_count=10)

        observation, info = environment.reset(seed=42)

        self.assertEqual(
            observation["obs"].tolist(),
            [1.0, 0.0],
        )
        self.assertEqual(
            observation["mask"].tolist(),
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        )
        self.assertEqual(
            observation["mask"].tolist(),
            info["action_mask"].tolist(),
        )

    def test_safe_action_terminates(self):
        environment = NoisyMaxEnv()
        environment.reset(seed=42)

        observation, reward, terminated, truncated, _ = (
            environment.step(0)
        )

        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(observation["mask"].any())

    def test_risky_action_reaches_noisy_state(self):
        environment = NoisyMaxEnv()
        environment.reset(seed=42)

        observation, reward, terminated, truncated, _ = (
            environment.step(1)
        )

        self.assertEqual(
            observation["obs"].tolist(),
            [0.0, 1.0],
        )
        self.assertTrue(observation["mask"].all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

        _, _, terminated, truncated, _ = (
            environment.step(5)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_same_seed_reproduces_reward(self):
        first = NoisyMaxEnv()
        second = NoisyMaxEnv()

        first.reset(seed=42)
        second.reset(seed=42)

        first.step(1)
        second.step(1)

        first_reward = first.step(0)[1]
        second_reward = second.step(0)[1]

        self.assertEqual(first_reward, second_reward)

    def test_invalid_initial_action_is_rejected(self):
        environment = NoisyMaxEnv(action_count=10)
        environment.reset(seed=42)

        with self.assertRaises(ValueError):
            environment.step(2)


if __name__ == "__main__":
    unittest.main()
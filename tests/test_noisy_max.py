"""Tests for the NoisyMax environment."""

import unittest

from src.environments.noisy_max import NoisyMaxEnv


class TestNoisyMax(unittest.TestCase):

    def test_safe_action_terminates(self):
        environment = NoisyMaxEnv()
        environment.reset(seed=42)

        _, reward, terminated, truncated, _ = (
            environment.step(0)
        )

        self.assertEqual(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_risky_action_reaches_noisy_state(self):
        environment = NoisyMaxEnv()
        environment.reset(seed=42)

        observation, reward, terminated, truncated, info = (
            environment.step(1)
        )

        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(
            observation.tolist(),
            [0.0, 1.0],
        )
        self.assertTrue(info["action_mask"].all())

        _, _, terminated, truncated, _ = environment.step(5)

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

    def test_other_initial_actions_follow_risky_path(self):
        environment = NoisyMaxEnv(action_count=10)
        environment.reset(seed=42)

        observation, reward, terminated, truncated, _ = (
            environment.step(5)
        )

        self.assertEqual(
            observation.tolist(),
            [0.0, 1.0],
        )
        self.assertEqual(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
"""Tests for the reference target calculations."""

import unittest

import torch

from src.agents.targets import double_dqn_target, dqn_target


class TestTargets(unittest.TestCase):
    """Test DQN and Double DQN targets."""

    def test_different_selected_actions(self) -> None:
        """The two methods should use different actions in this example."""
        online_q = torch.tensor([5.0, 4.0])
        target_q = torch.tensor([2.0, 6.0])

        dqn = dqn_target(1.0, 0.9, False, target_q)
        double_dqn = double_dqn_target(
            1.0,
            0.9,
            False,
            online_q,
            target_q,
        )

        self.assertAlmostEqual(dqn, 6.4)
        self.assertAlmostEqual(double_dqn, 2.8)

    def test_terminal_transition(self) -> None:
        """Terminal transitions should not use future values."""
        online_q = torch.tensor([5.0, 4.0])
        target_q = torch.tensor([2.0, 6.0])

        dqn = dqn_target(1.0, 0.9, True, target_q)
        double_dqn = double_dqn_target(
            1.0,
            0.9,
            True,
            online_q,
            target_q,
        )

        self.assertEqual(dqn, 1.0)
        self.assertEqual(double_dqn, 1.0)

    def test_identical_predictions(self) -> None:
        """Both methods agree when both networks give the same values."""
        online_q = torch.tensor([2.0, 6.0])
        target_q = torch.tensor([2.0, 6.0])

        dqn = dqn_target(1.0, 0.9, False, target_q)
        double_dqn = double_dqn_target(
            1.0,
            0.9,
            False,
            online_q,
            target_q,
        )

        self.assertAlmostEqual(dqn, 6.4)
        self.assertAlmostEqual(double_dqn, 6.4)


if __name__ == "__main__":
    unittest.main()
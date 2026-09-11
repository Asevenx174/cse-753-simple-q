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

    def test_dqn_mask_excludes_invalid_maximum(self):
        target_q = torch.tensor(
            [2.0, 3.0, 100.0, 200.0]
        )
        mask = torch.tensor(
            [True, True, False, False]
        )

        target = dqn_target(
            1.0,
            0.9,
            False,
            target_q,
            mask,
        )

        self.assertAlmostEqual(target, 3.7)

    def test_double_dqn_mask_excludes_invalid_action(self):
        online_q = torch.tensor(
            [2.0, 3.0, 100.0, 200.0]
        )
        target_q = torch.tensor(
            [5.0, 7.0, 500.0, 600.0]
        )
        mask = torch.tensor(
            [True, True, False, False]
        )

        target = double_dqn_target(
            1.0,
            0.9,
            False,
            online_q,
            target_q,
            mask,
        )

        self.assertAlmostEqual(target, 7.3)

    def test_terminal_target_ignores_empty_mask(self):
        empty_mask = torch.tensor(
            [False, False, False, False]
        )

        dqn = dqn_target(
            1.5,
            0.9,
            True,
            torch.tensor(
                [float("nan")] * 4
            ),
            empty_mask,
        )

        double_dqn = double_dqn_target(
            1.5,
            0.9,
            True,
            torch.tensor(
                [float("nan")] * 4
            ),
            torch.tensor(
                [float("nan")] * 4
            ),
            empty_mask,
        )

        self.assertEqual(dqn, 1.5)
        self.assertEqual(double_dqn, 1.5)
if __name__ == "__main__":
    unittest.main()
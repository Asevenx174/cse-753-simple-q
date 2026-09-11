"""Tests for NoisyMax diagnostics."""

import unittest

import torch
from torch import nn

from src.utils.noisy_max_diagnostics import (
    NoisyMaxTracker,
    calculate_noisy_max_diagnostics,
)


class FixedQNetwork(nn.Module):
    """Return fixed Q-values for hand calculations."""

    def __init__(self):
        super().__init__()

        self.register_buffer(
            "values",
            torch.tensor(
                [
                    [0.20, -0.05, 50.0, 60.0],
                    [-0.20, 0.10, -0.30, 0.00],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, observation, state=None, info=None):
        """Return values associated with each one-hot state."""

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.values.device,
        )

        state_indices = observation.argmax(dim=1)

        return self.values[state_indices], state


class TestNoisyMaxDiagnostics(unittest.TestCase):

    def test_tracker_records_training_behavior(self):
        tracker = NoisyMaxTracker(action_count=4)

        tracker.record_initial_action(0)
        tracker.record_initial_action(1)
        tracker.record_initial_action(1)

        tracker.record_noisy_action(0)
        tracker.record_noisy_action(3)
        tracker.record_noisy_action(3)

        self.assertEqual(tracker.initial_choices, 3)
        self.assertEqual(tracker.risky_choices, 2)
        self.assertEqual(tracker.noisy_state_visits, 2)
        self.assertAlmostEqual(
            tracker.training_risky_fraction,
            2 / 3,
        )
        self.assertEqual(
            tracker.action_sample_counts,
            [1, 0, 0, 2],
        )

    def test_diagnostics_match_hand_calculation(self):
        network = FixedQNetwork()
        tracker = NoisyMaxTracker(action_count=4)

        tracker.record_initial_action(0)
        tracker.record_initial_action(1)
        tracker.record_initial_action(1)

        tracker.record_noisy_action(0)
        tracker.record_noisy_action(3)
        tracker.record_noisy_action(3)

        result = calculate_noisy_max_diagnostics(
            network,
            tracker,
            gamma=0.99,
        )

        # True safe value = 0.
        self.assertAlmostEqual(
            result["safe_q"],
            0.20,
        )
        self.assertAlmostEqual(
            result["safe_signed_error"],
            0.20,
        )
        self.assertAlmostEqual(
            result["safe_absolute_error"],
            0.20,
        )

        # True risky value = 0.99 × -0.1 = -0.099.
        # Signed error = -0.05 - (-0.099) = 0.049.
        self.assertAlmostEqual(
            result["risky_q"],
            -0.05,
        )
        self.assertAlmostEqual(
            result["risky_signed_error"],
            0.049,
            places=6,
        )
        self.assertAlmostEqual(
            result["risky_absolute_error"],
            0.049,
            places=6,
        )

        # Maximum learned s1 value is 0.10.
        # Its true value is -0.10.
        # Signed error = 0.10 - (-0.10) = 0.20.
        self.assertAlmostEqual(
            result["maximum_noisy_q"],
            0.10,
        )
        self.assertAlmostEqual(
            result["noisy_signed_error"],
            0.20,
            places=6,
        )
        self.assertAlmostEqual(
            result["noisy_absolute_error"],
            0.20,
            places=6,
        )

        # Safe is greedily preferred: 0.20 > -0.05.
        self.assertEqual(
            result["greedy_risky_choice"],
            0,
        )

        self.assertAlmostEqual(
            result["training_risky_fraction"],
            2 / 3,
        )
        self.assertEqual(
            result["noisy_state_visits"],
            2,
        )
        self.assertEqual(result["action_0_samples"], 1)
        self.assertEqual(result["action_1_samples"], 0)
        self.assertEqual(result["action_2_samples"], 0)
        self.assertEqual(result["action_3_samples"], 2)

    def test_invalid_actions_are_rejected(self):
        tracker = NoisyMaxTracker(action_count=4)

        with self.assertRaises(ValueError):
            tracker.record_initial_action(2)

        with self.assertRaises(ValueError):
            tracker.record_noisy_action(4)


if __name__ == "__main__":
    unittest.main()
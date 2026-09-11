"""Tests for one synthetic maximization-bias trial."""

import unittest

import numpy as np

from experiments.bias_analysis.single_trial import run_single_trial


class TestSyntheticBiasTrial(unittest.TestCase):
    """Test the behavior of run_single_trial."""

    def test_same_seed_reproduces_trial(self) -> None:
        """The same seed should reproduce the same trial."""
        """zero noise is not defined thus considering the default value 1.0"""
        first_result = run_single_trial(
            action_count=5,
            rng=np.random.default_rng(42),
        )

        second_result = run_single_trial(
            action_count=5,
            rng=np.random.default_rng(42), # same seed, same array generation as first result
        )

        first_a, first_b, first_action, first_max, first_double = first_result

        (
            second_a,
            second_b,
            second_action,
            second_max,
            second_double,
        ) = second_result

        np.testing.assert_array_equal(first_a, second_a)
        np.testing.assert_array_equal(first_b, second_b)
        self.assertEqual(first_action, second_action)
        self.assertEqual(first_max, second_max)
        self.assertEqual(first_double, second_double)

    def test_zero_noise_produces_zero_errors(self) -> None:
        """Zero noise should produce zero estimation errors."""
        result = run_single_trial(
            action_count=5,
            rng=np.random.default_rng(42),
            noise_std=0.0,
        )

        estimator_a, estimator_b, _, maximum_error, double_error = result

        np.testing.assert_array_equal(
            estimator_a,
            np.zeros(5),
        )

        np.testing.assert_array_equal(
            estimator_b,
            np.zeros(5),
        )

        self.assertEqual(maximum_error, 0.0)
        self.assertEqual(double_error, 0.0)


if __name__ == "__main__":
    unittest.main()
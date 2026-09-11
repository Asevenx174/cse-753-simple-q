"""Shape test for the shared Q-network."""

import unittest

import torch

from src.agents.q_network import QNetwork

# expected transformation [4, 6] --> [4, 3]
# Check [batch. observation dimension] --> [batch, action count]
class TestQNetwork(unittest.TestCase):

    def test_output_shape(self) -> None:
        network = QNetwork(
            observation_dim=6,
            action_count=3,
        )

        observations = torch.zeros(4, 6)
        q_values, state = network(observations)

        self.assertEqual(q_values.shape, (4, 3))
        self.assertIsNone(state)


if __name__ == "__main__":
    unittest.main()
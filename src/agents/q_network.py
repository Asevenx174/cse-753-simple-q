"""Shared Q-network for DQN and Double DQN."""

import torch
from torch import nn


class QNetwork(nn.Module):
    """Predict one Q-value for every available action."""

    def __init__(
        self,
        observation_dim: int,
        action_count: int,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # converts the environment observation into 128 learned featuresss
            nn.Linear(observation_dim, 128), 
            # add non linearity
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observation, state=None, info=None):
        """Return action values and the unchanged network state."""

        device = next(self.parameters()).device
        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=device,
        )

        return self.model(observation), state
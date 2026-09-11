"""Two-state environment for studying maximization bias."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NoisyMaxEnv(gym.Env):
    """Safe action versus a noisy, slightly worse path."""

    metadata = {"render_modes": []}

    def __init__(self, action_count: int = 10):
        super().__init__()

        if action_count < 2:
            raise ValueError("action_count must be at least 2.")

        self.action_count = action_count
        self.action_space = spaces.Discrete(action_count)

        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "mask": spaces.MultiBinary(action_count),
            }
        )

        self.state = 0
        self.finished = False

    def state_observation(self):
        """Return the current state as a one-hot vector."""

        if self.finished:
            return np.zeros(2, dtype=np.float32)

        observation = np.zeros(2, dtype=np.float32)
        observation[self.state] = 1.0
        return observation

    def action_mask(self):
        """Return the valid-action mask for the current state."""

        if self.finished:
            return np.zeros(
                self.action_count,
                dtype=np.int8,
            )

        if self.state == 0:
            mask = np.zeros(
                self.action_count,
                dtype=np.int8,
            )
            mask[:2] = 1
            return mask

        return np.ones(
            self.action_count,
            dtype=np.int8,
        )

    def observation(self):
        """Return the state and its valid-action mask."""

        return {
            "obs": self.state_observation(),
            "mask": self.action_mask(),
        }

    def reset(self, *, seed=None, options=None):
        """Reset the environment to the initial state."""

        super().reset(seed=seed)

        self.state = 0
        self.finished = False

        return (
            self.observation(),
            {"action_mask": self.action_mask()},
        )

    def step(self, action):
        """Apply one action and return the transition."""

        if self.finished:
            raise RuntimeError(
                "Reset the environment after termination."
            )

        if not self.action_space.contains(action):
            raise ValueError(
                "Action is outside the action space."
            )

        if self.state == 0:
            if action == 0:
                reward = 0.0
                self.finished = True
            elif action == 1:
                reward = 0.0
                self.state = 1
            else:
                raise ValueError(
                    "Only actions 0 and 1 are valid at s0."
                )
        else:
            reward = float(
                self.np_random.normal(-0.1, 1.0)
            )
            self.finished = True

        return (
            self.observation(),
            reward,
            self.finished,
            False,
            {"action_mask": self.action_mask()},
        )
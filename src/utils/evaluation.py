"""Separate greedy-policy evaluation."""

import gymnasium as gym
import numpy as np
import torch


def evaluate_greedy(
    network,
    environment_id: str,
    episodes: int = 5,
    seed: int = 10_000,
) -> list[float]:
    """Evaluate without exploration or learning."""

    environment = gym.make(environment_id)
    returns = []

    network.eval()

    with torch.no_grad():
        for episode in range(episodes):
            observation, _ = environment.reset(
                seed=seed + episode
            )
            episode_return = 0.0
            finished = False

            while not finished:
                q_values, _ = network(observation[None, :])
                action = int(q_values.argmax(dim=1).item())

                observation, reward, terminated, truncated, _ = (
                    environment.step(action)
                )

                episode_return += reward
                finished = terminated or truncated

            returns.append(float(episode_return))

    environment.close()
    return returns
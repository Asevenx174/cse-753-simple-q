"""Record a reproducible CartPole random-policy baseline."""

import numpy as np
import gymnasium as gym

from src.utils.result_saving import (
    create_run_directory,
    save_csv,
    save_json,
)


ENVIRONMENT = "CartPole-v1"
EPISODES = 20
BASE_SEED = 42


def main():
    environment = gym.make(ENVIRONMENT)
    random_generator = np.random.default_rng(BASE_SEED)
    rows = []

    for episode in range(EPISODES):
        episode_seed = BASE_SEED + episode
        observation, _ = environment.reset(seed=episode_seed)

        episode_return = 0.0
        episode_length = 0
        finished = False

        while not finished:
            # random action selection
            action = int(
                random_generator.integers(
                    environment.action_space.n
                )
            )

            observation, reward, terminated, truncated, _ = (
                environment.step(action)
            )

            episode_return += reward
            episode_length += 1
            finished = terminated or truncated

        rows.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "episode_return": episode_return,
                "episode_length": episode_length,
            }
        )

    environment.close()

    returns = np.array(
        [row["episode_return"] for row in rows]
    )
    mean = returns.mean()
    standard_deviation = returns.std(ddof=1)
    standard_error = standard_deviation / np.sqrt(EPISODES)

    run_directory = create_run_directory(
        "results/baselines/cartpole"
    )

    save_json(
        run_directory / "config.json",
        {
            "environment": ENVIRONMENT,
            "policy": "uniform_random",
            "episodes": EPISODES,
            "base_seed": BASE_SEED,
        },
    )
    save_csv(run_directory / "episodes.csv", rows)

    print("Episode returns:", returns.tolist())
    print(f"Mean return: {mean:.2f}")
    print(f"Sample SD: {standard_deviation:.2f}")
    print(f"Standard error: {standard_error:.2f}")
    print("Saved:", run_directory)


if __name__ == "__main__":
    main()
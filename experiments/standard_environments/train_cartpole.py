"""Train DQN or Double DQN on CartPole."""

import argparse
import platform
import subprocess

import gymnasium as gym
import numpy as np
import tianshou
import torch

from tianshou.algorithm.algorithm_base import (
    policy_within_training_step,
)
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv

from src.agents.dqn_factory import build_dqn
from src.agents.q_network import QNetwork
from src.utils.evaluation import evaluate_greedy
from src.utils.result_saving import (
    create_run_directory,
    save_checkpoint,
    save_csv,
    save_json,
)


ENVIRONMENT = "CartPole-v1"
SEED = 42
TOTAL_STEPS = 50_000
WARMUP_STEPS = 1_000
COLLECTION_STEPS = 10
BATCH_SIZE = 64
EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 10

def record_evaluation(network, environment_step):
    """Evaluate and create one row per episode."""

    returns, lengths = evaluate_greedy(
        network,
        ENVIRONMENT,
        episodes=EVALUATION_EPISODES,
        seed=10_000,
    )

    rows = []

    for episode, (episode_return, episode_length) in enumerate(
        zip(returns, lengths)
    ):
        rows.append(
            {
                "environment_step": environment_step,
                "episode_id": episode,
                "evaluation_seed": 10_000 + episode,
                "episode_return": episode_return,
                "episode_length": episode_length,
            }
        )

    print(
        f"Step {environment_step}: "
        f"mean evaluation return = {np.mean(returns):.2f}"
    )

    return rows

def train(algorithm_name):
    """Run one complete CartPole validation experiment."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    example_environment = gym.make(ENVIRONMENT)
    network = QNetwork(4, 2)

    algorithm = build_dqn(
        network,
        example_environment.observation_space,
        example_environment.action_space,
        algorithm_name,
    )

    training_environments = DummyVectorEnv(
        [lambda: gym.make(ENVIRONMENT)]
    )
    replay = VectorReplayBuffer(
        total_size=20_000,
        buffer_num=1,
    )
    collector = Collector(
        algorithm,
        training_environments,
        replay,
    )

    environment_step = 0
    learning_update = 0
    training_episode = 0

    training_rows = []
    evaluation_rows = record_evaluation(network, 0)
    loss_rows = []

    while environment_step < TOTAL_STEPS:
        random_actions = environment_step < WARMUP_STEPS

        with policy_within_training_step(algorithm.policy):
            collection = collector.collect(
                n_step=COLLECTION_STEPS,
                random=random_actions,
                reset_before_collect=(environment_step == 0),
                gym_reset_kwargs=(
                    {"seed": SEED}
                    if environment_step == 0
                    else None
                ),
            )

        environment_step += collection.n_collected_steps

        for episode_return, episode_length in zip(
            collection.returns,
            collection.lens,
        ):
            training_rows.append(
                {
                    "episode": training_episode,
                    "ending_environment_step": environment_step,
                    "episode_return": float(episode_return),
                    "episode_length": int(episode_length),
                }
            )
            training_episode += 1

        if environment_step >= WARMUP_STEPS:
            with policy_within_training_step(algorithm.policy):
                statistics = algorithm.update(
                    replay,
                    sample_size=BATCH_SIZE,
                )

            learning_update += 1
            loss = float(statistics.loss)

            assert np.isfinite(loss)

            loss_rows.append(
                {
                    "environment_step": environment_step,
                    "learning_update": learning_update,
                    "td_loss": loss,
                }
            )

        if environment_step % EVALUATION_INTERVAL == 0:
            evaluation_rows.extend(
                record_evaluation(network, environment_step)
            )

    training_environments.close()
    example_environment.close()

    return network, training_rows, evaluation_rows, loss_rows

def current_git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def save_run(
    algorithm_name,
    network,
    training_rows,
    evaluation_rows,
    loss_rows,
):
    """Save one completed validation run."""

    run_directory = create_run_directory(
        f"results/runs/cartpole/{algorithm_name}"
    )

    config = {
        "environment": ENVIRONMENT,
        "algorithm": algorithm_name,
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "collection_steps": COLLECTION_STEPS,
        "batch_size": BATCH_SIZE,
        "evaluation_interval": EVALUATION_INTERVAL,
        "evaluation_episodes": EVALUATION_EPISODES,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "target_update_frequency": 100,
        "updates_per_collection": 1,
    }

    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "tianshou": tianshou.__version__,
        "device": "cpu",
        "git_commit": current_git_commit(),
        "run_status": "complete",
    }

    save_json(run_directory / "config.json", config)
    save_json(run_directory / "metadata.json", metadata)
    save_csv(run_directory / "training.csv", training_rows)
    save_csv(run_directory / "evaluation.csv", evaluation_rows)
    save_csv(run_directory / "loss.csv", loss_rows)

    save_checkpoint(
        run_directory / "final.pt",
        network,
        {
            "observation_dim": 4,
            "action_count": 2,
            "algorithm": algorithm_name,
        },
    )

    print("Completed run:", run_directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=("dqn", "double_dqn"),
    )
    arguments = parser.parse_args()

    outputs = train(arguments.algorithm)
    save_run(arguments.algorithm, *outputs)


if __name__ == "__main__":
    main()
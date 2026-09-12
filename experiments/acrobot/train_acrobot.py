"""Train DQN or Double DQN on Acrobot with paired seeds."""

import argparse
import os
import platform
import subprocess
import time

import gymnasium as gym
import numpy as np
import tianshou
import torch
from tianshou.algorithm.algorithm_base import policy_within_training_step
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

ENVIRONMENT = "Acrobot-v1"
SEED = int(os.getenv("ACROBOT_SEED", "100"))
TOTAL_STEPS = 200_000
WARMUP_STEPS = 1_000
COLLECTION_STEPS = 10
BATCH_SIZE = 64
EVALUATION_INTERVAL = 5_000
EVALUATION_EPISODES = 20
FINAL_EVALUATION_EPISODES = 100
FINAL_EVALUATION_SEED = 20_000


def record_evaluation(network, step):
    """Evaluate one training checkpoint on 20 fixed episodes."""
    returns, lengths = evaluate_greedy(
        network,
        ENVIRONMENT,
        episodes=EVALUATION_EPISODES,
        seed=10_000,
    )
    print(f"Step {step}: mean evaluation return = {np.mean(returns):.2f}")

    return [
        {
            "environment_step": step,
            "episode_id": i,
            "evaluation_seed": 10_000 + i,
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
        }
        for i, (episode_return, episode_length)
        in enumerate(zip(returns, lengths))
    ]


def train(algorithm_name):
    """Train one algorithm using the selected seed."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    example_env = gym.make(ENVIRONMENT)
    observation_dim = example_env.observation_space.shape[0]
    action_count = example_env.action_space.n
    network = QNetwork(observation_dim, action_count)

    algorithm = build_dqn(
        network,
        example_env.observation_space,
        example_env.action_space,
        algorithm_name,
    )
    environments = DummyVectorEnv([lambda: gym.make(ENVIRONMENT)])
    replay = VectorReplayBuffer(total_size=20_000, buffer_num=1)
    collector = Collector(algorithm, environments, replay)

    step = 0
    updates = 0
    episode_id = 0
    training_rows = []
    evaluation_rows = record_evaluation(network, 0)
    loss_rows = []
    start = time.perf_counter()

    while step < TOTAL_STEPS:
        with policy_within_training_step(algorithm.policy):
            collected = collector.collect(
                n_step=min(COLLECTION_STEPS, TOTAL_STEPS - step),
                random=(step < WARMUP_STEPS),
                reset_before_collect=(step == 0),
                gym_reset_kwargs={"seed": SEED} if step == 0 else None,
            )

        step += collected.n_collected_steps

        for episode_return, episode_length in zip(
            collected.returns, collected.lens
        ):
            training_rows.append({
                "episode": episode_id,
                "ending_environment_step": step,
                "episode_return": float(episode_return),
                "episode_length": int(episode_length),
            })
            episode_id += 1

        if step >= WARMUP_STEPS:
            with policy_within_training_step(algorithm.policy):
                statistics = algorithm.update(
                    replay, sample_size=BATCH_SIZE
                )

            updates += 1
            loss = float(statistics.loss)
            if not np.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")

            loss_rows.append({
                "environment_step": step,
                "learning_update": updates,
                "td_loss": loss,
            })

        if step % EVALUATION_INTERVAL == 0:
            evaluation_rows.extend(record_evaluation(network, step))

    runtime = time.perf_counter() - start
    environments.close()
    example_env.close()
    return network, training_rows, evaluation_rows, loss_rows, runtime, updates


def save_run(algorithm_name, network, training, evaluation, losses,
             runtime, updates):
    """Save training results and evaluate the final model."""
    directory = create_run_directory(
        f"results/runs/acrobot/{algorithm_name}"
    )

    save_json(directory / "config.json", {
        "environment": ENVIRONMENT,
        "algorithm": algorithm_name,
        "seed": SEED,
        "total_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "collection_steps": COLLECTION_STEPS,
        "batch_size": BATCH_SIZE,
        "replay_size": 20_000,
        "evaluation_interval": EVALUATION_INTERVAL,
        "evaluation_episodes": EVALUATION_EPISODES,
        "final_evaluation_episodes": FINAL_EVALUATION_EPISODES,
        "final_evaluation_seed": FINAL_EVALUATION_SEED,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "target_update_frequency": 100,
        "updates_per_collection": 1,
    })
    save_json(directory / "metadata.json", {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "tianshou": tianshou.__version__,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime_seconds": runtime,
        "learning_updates": updates,
        "run_status": "complete",
    })
    save_csv(directory / "training.csv", training)
    save_csv(directory / "evaluation.csv", evaluation)
    save_csv(directory / "loss.csv", losses)
    save_checkpoint(directory / "final.pt", network, {
        "observation_dim": 6,
        "action_count": 3,
        "algorithm": algorithm_name,
    })

    returns, lengths = evaluate_greedy(
        network,
        ENVIRONMENT,
        episodes=FINAL_EVALUATION_EPISODES,
        seed=FINAL_EVALUATION_SEED,
    )
    save_csv(directory / "final_evaluation.csv", [
        {
            "evaluation_seed": FINAL_EVALUATION_SEED + i,
            "episode_return": float(episode_return),
            "episode_length": int(episode_length),
        }
        for i, (episode_return, episode_length)
        in enumerate(zip(returns, lengths))
    ])

    print(f"Saved: {directory} (runtime: {runtime:.1f} seconds)")
    print(f"100-episode final mean: {np.mean(returns):.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=("dqn", "double_dqn"),
    )
    args = parser.parse_args()
    save_run(args.algorithm, *train(args.algorithm))


if __name__ == "__main__":
    main()
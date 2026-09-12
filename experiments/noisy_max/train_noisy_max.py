"""Run one NoisyMax pilot."""

import argparse
import json
import platform
import subprocess
import time
import os
from pathlib import Path

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
from src.environments.noisy_max import NoisyMaxEnv
from src.utils.noisy_max_diagnostics import (
    NoisyMaxTracker,
    calculate_noisy_max_diagnostics,
)
from src.utils.result_saving import (
    create_run_directory,
    save_checkpoint,
    save_csv,
)


SEED = int(os.getenv("NOISYMAX_SEED", "100"))
M = int(os.getenv("NOISYMAX_ACTION_COUNT", "10"))
C = int(os.getenv("NOISYMAX_TARGET_UPDATE", "320"))
if C <= 0:
    raise ValueError("Target-update interval must be positive.")
    
EPSILON = 0.10
TOTAL_STEPS = 50_000
DIAGNOSTIC_INTERVAL = 1_000
BATCH_SIZE = 64
COLLECTION_STEPS = 10


class TrackedEnv(NoisyMaxEnv):
    """Track training choices and coverage."""

    def __init__(self, tracker):
        super().__init__(action_count=M)
        self.tracker = tracker

    def step(self, action):
        previous_state = self.state
        result = super().step(action)

        if previous_state == 0:
            self.tracker.record_initial_action(int(action))
        else:
            self.tracker.record_noisy_action(int(action))

        return result


def save_json(path, data):
    """Save JSON data."""

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def train(algorithm_name):
    """Train one algorithm and save its results."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tracker = NoisyMaxTracker(M)
    example_env = NoisyMaxEnv(M)
    network = QNetwork(2, M)

    algorithm = build_dqn(
        network,
        example_env.observation_space,
        example_env.action_space,
        algorithm_name,
        epsilon=EPSILON,
        target_update_frequency=C,
    )

    environments = DummyVectorEnv(
        [lambda: TrackedEnv(tracker)]
    )
    replay = VectorReplayBuffer(20_000, 1)

    collector = Collector(
        algorithm,
        environments,
        replay,
        exploration_noise=True,
    )

    steps = 0
    updates = 0
    losses = []
    diagnostics = []
    start = time.perf_counter()

    while steps < TOTAL_STEPS:
        with policy_within_training_step(algorithm.policy):
            collection = collector.collect(
                n_step=min(
                    COLLECTION_STEPS,
                    TOTAL_STEPS - steps,
                ),
                random=False,
                reset_before_collect=(steps == 0),
                gym_reset_kwargs=(
                    {"seed": SEED} if steps == 0 else None
                ),
            )

        steps += collection.n_collected_steps

        if len(replay) >= BATCH_SIZE:
            with policy_within_training_step(algorithm.policy):
                statistics = algorithm.update(
                    replay,
                    sample_size=BATCH_SIZE,
                )

            updates += 1
            loss = float(statistics.loss)

            if not np.isfinite(loss):
                raise RuntimeError("Non-finite loss detected.")

            losses.append(
                {
                    "environment_step": steps,
                    "learning_update": updates,
                    "td_loss": loss,
                }
            )

        if steps % DIAGNOSTIC_INTERVAL == 0:
            row = calculate_noisy_max_diagnostics(
                network,
                tracker,
            )
            row = {
                "environment_step": steps,
                "learning_update": updates,
                **row,
            }
            diagnostics.append(row)

            print(
                algorithm_name,
                "step=", steps,
                "safe_q=", round(row["safe_q"], 3),
                "risky_q=", round(row["risky_q"], 3),
                "greedy_risky=",
                row["greedy_risky_choice"],
            )

    runtime = time.perf_counter() - start
    environments.close()
    example_env.close()

    directory = create_run_directory(
        f"results/runs/noisy_max/{algorithm_name}"
    )

    save_json(
        directory / "config.json",
        {
            "algorithm": algorithm_name,
            "seed": SEED,
            "M": M,
            "C": C,
            "epsilon": EPSILON,
            "total_steps": TOTAL_STEPS,
            "diagnostic_interval": DIAGNOSTIC_INTERVAL,
            "batch_size": BATCH_SIZE,
            "gamma": 0.99,
            "learning_rate": 0.001,
        },
    )

    save_json(
        directory / "metadata.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tianshou": tianshou.__version__,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "runtime_seconds": runtime,
            "learning_updates": updates,
            "run_status": "complete",
        },
    )

    coverage = [
        {"action": action, "sample_count": count}
        for action, count in enumerate(
            tracker.action_sample_counts
        )
    ]

    save_csv(directory / "diagnostics.csv", diagnostics)
    save_csv(directory / "loss.csv", losses)
    save_csv(directory / "action_coverage.csv", coverage)

    save_checkpoint(
        directory / "final.pt",
        network,
        {
            "observation_dim": 2,
            "action_count": M,
            "algorithm": algorithm_name,
        },
    )

    print("Saved:", directory)
    print("Runtime:", round(runtime, 2), "seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=("dqn", "double_dqn"),
    )
    args = parser.parse_args()
    train(args.algorithm)


if __name__ == "__main__":
    main()
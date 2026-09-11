"""Check configuration and metadata saving."""

import platform
import subprocess

import gymnasium
import tianshou
import torch

from src.utils.result_saving import (
    create_run_directory,
    save_json,
)


def git_commit() -> str:
    """Return the current Git commit."""

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    run_directory = create_run_directory()

    config = {
        "environment": "CartPole-v1",
        "algorithm": "DQN",
        "seed": 42,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "target_update_frequency": 100,
    }

    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "tianshou": tianshou.__version__,
        "device": "cpu",
        "git_commit": git_commit(),
        "run_status": "verification_complete",
    }

    save_json(run_directory / "config.json", config)
    save_json(run_directory / "metadata.json", metadata)

    print("Created:", run_directory)
    print("Saved: config.json")
    print("Saved: metadata.json")


if __name__ == "__main__":
    main()
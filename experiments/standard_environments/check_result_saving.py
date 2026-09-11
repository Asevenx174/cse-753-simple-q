"""Check metric and model saving."""

import platform
import subprocess

import gymnasium
import tianshou
import torch

from src.agents.q_network import QNetwork
from src.utils.result_saving import (
    create_run_directory,
    save_checkpoint,
    save_csv,
    save_json,
)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main():
    run_directory = create_run_directory()
    network = QNetwork(4, 2)

    config = {
        "purpose": "storage_verification",
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

    training_rows = [
        {
            "environment_step": 100,
            "episode_return": 12.0,
            "episode_length": 12,
        }
    ]
    evaluation_rows = [
        {
            "environment_step": 0,
            "episode_id": 0,
            "evaluation_seed": 10000,
            "episode_return": 17.0,
            "episode_length": 17,
        }
    ]
    loss_rows = [
        {
            "environment_step": 100,
            "learning_update": 1,
            "td_loss": 1.0172,
        }
    ]

    save_json(run_directory / "config.json", config)
    save_json(run_directory / "metadata.json", metadata)
    save_csv(run_directory / "training.csv", training_rows)
    save_csv(run_directory / "evaluation.csv", evaluation_rows)
    save_csv(run_directory / "loss.csv", loss_rows)

    model_information = {
        "observation_dim": 4,
        "action_count": 2,
    }
    save_checkpoint(
        run_directory / "final.pt",
        network,
        model_information,
    )

    test_input = torch.zeros(1, 4)
    original_output, _ = network(test_input)

    checkpoint = torch.load(
        run_directory / "final.pt",
        weights_only=True,
    )
    restored_network = QNetwork(4, 2)
    restored_network.load_state_dict(
        checkpoint["model_state_dict"]
    )
    restored_output, _ = restored_network(test_input)

    assert torch.equal(original_output, restored_output)

    print("Created:", run_directory)
    print("Saved files:")
    for path in sorted(run_directory.iterdir()):
        print(" ", path.name)
    print("Reloaded output matches:", True)


if __name__ == "__main__":
    main()
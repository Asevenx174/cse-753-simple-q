"""Compare our manual target example with Tianshou 2.0.1."""

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data import Batch


ONLINE_Q = torch.tensor([[5.0, 4.0]])
TARGET_Q = torch.tensor([[2.0, 6.0]])
REWARD, GAMMA = 1.0, 0.9
OUTPUT = Path("results/verification/tianshou_targets.csv")


class FakeBuffer:
    """Provide the next observation required by Tianshou."""

    def __getitem__(self, indices):
        return Batch(obs_next=np.zeros((len(indices), 1)))


def policy(batch, model=None):
    """Return online or target-network predictions."""

    logits = TARGET_Q if model == "target" else ONLINE_Q
    return Batch(logits=logits, act=logits.argmax(dim=1))


def tianshou_target(is_double):
    """Call Tianshou's actual target-selection method."""

    algorithm = SimpleNamespace(
        policy=policy,
        model_old="target",
        use_target_network=True,
        is_double=is_double,
    )

    # calls Tianshou's installed implementation
    next_q = DQN._target_q(
        algorithm,
        FakeBuffer(),
        np.array([0]),
    ).item()

    return next_q, REWARD + GAMMA * next_q


def main():
    standard_q, standard_target = tianshou_target(False)
    double_q, double_target = tianshou_target(True)

    assert np.isclose(standard_target, 6.4)
    assert np.isclose(double_target, 2.8)

    rows = [
        ["DQN", standard_q, standard_target],
        ["Double DQN", double_q, double_target],
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "selected_next_q", "target"])
        writer.writerows(rows)

    for row in rows:
        print(f"{row[0]}: next Q = {row[1]:.1f}, target = {row[2]:.1f}")

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
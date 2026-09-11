"""Check isolated greedy evaluation."""

import numpy as np
import torch

from src.agents.q_network import QNetwork
from src.utils.evaluation import evaluate_greedy


def main():
    torch.manual_seed(42)
    network = QNetwork(4, 2)

    before = [
        parameter.detach().clone()
        for parameter in network.parameters()
    ]

    returns = evaluate_greedy(
        network,
        "CartPole-v1",
        episodes=5,
    )

    after = list(network.parameters())

    weights_unchanged = all(
        torch.equal(old, new)
        for old, new in zip(before, after)
    )

    assert weights_unchanged
    assert len(returns) == 5
    assert np.isfinite(returns).all()

    print("Training step: 0")
    print("Evaluation returns:", returns)
    print("Mean return:", np.mean(returns))
    print("Weights unchanged:", weights_unchanged)


if __name__ == "__main__":
    main()
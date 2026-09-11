"""Diagnostics for NoisyMax experiments."""

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class NoisyMaxTracker:
    """Track behavior and action coverage during training."""

    action_count: int
    initial_choices: int = 0
    risky_choices: int = 0
    noisy_state_visits: int = 0
    action_sample_counts: list[int] = field(
        init=False
    )

    def __post_init__(self):
        if self.action_count < 2:
            raise ValueError(
                "action_count must be at least 2."
            )

        self.action_sample_counts = [
            0 for _ in range(self.action_count)
        ]

    def record_initial_action(self, action: int):
        """Record an actual training choice at s0."""

        if action not in (0, 1):
            raise ValueError(
                "Only actions 0 and 1 are valid at s0."
            )

        self.initial_choices += 1

        if action == 1:
            self.risky_choices += 1
            self.noisy_state_visits += 1

    def record_noisy_action(self, action: int):
        """Record an action sampled at s1."""

        if not 0 <= action < self.action_count:
            raise ValueError(
                "Noisy-state action is outside the action space."
            )

        self.action_sample_counts[action] += 1

    @property
    def training_risky_fraction(self) -> float:
        """Return the observed risky fraction during training."""

        if self.initial_choices == 0:
            return 0.0

        return self.risky_choices / self.initial_choices


def calculate_noisy_max_diagnostics(
    network,
    tracker: NoisyMaxTracker,
    gamma: float = 0.99,
) -> dict:
    """Calculate learned values and estimation errors."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")

    try:
        device = next(network.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    states = torch.tensor(
        [
            [1.0, 0.0],  # s0
            [0.0, 1.0],  # s1
        ],
        dtype=torch.float32,
        device=device,
    )

    was_training = network.training
    network.eval()

    with torch.no_grad():
        q_values, _ = network(states)

    if was_training:
        network.train()

    q_values = q_values.detach().cpu().numpy()

    safe_q = float(q_values[0, 0])
    risky_q = float(q_values[0, 1])
    maximum_noisy_q = float(np.max(q_values[1]))

    true_safe_q = 0.0
    true_risky_q = gamma * -0.1
    true_noisy_q = -0.1

    safe_signed_error = safe_q - true_safe_q
    risky_signed_error = risky_q - true_risky_q
    noisy_signed_error = (
        maximum_noisy_q - true_noisy_q
    )

    greedy_risky_choice = int(risky_q > safe_q)

    diagnostics = {
        "safe_q": safe_q,
        "risky_q": risky_q,
        "maximum_noisy_q": maximum_noisy_q,
        "safe_signed_error": safe_signed_error,
        "safe_absolute_error": abs(safe_signed_error),
        "risky_signed_error": risky_signed_error,
        "risky_absolute_error": abs(risky_signed_error),
        "noisy_signed_error": noisy_signed_error,
        "noisy_absolute_error": abs(noisy_signed_error),
        "greedy_risky_choice": greedy_risky_choice,
        "training_risky_fraction": (
            tracker.training_risky_fraction
        ),
        "noisy_state_visits": tracker.noisy_state_visits,
    }

    for action, count in enumerate(
        tracker.action_sample_counts
    ):
        diagnostics[f"action_{action}_samples"] = count

    return diagnostics
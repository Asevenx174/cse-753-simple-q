import numpy as np

"The purpose of this experiments is to resembles the selection/evaluation separation in Double DQN"

def run_single_trial(
    action_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Generate one standard and one independently evaluated estimate."""
    if action_count < 1:
        raise ValueError("action_count must be at least 1")
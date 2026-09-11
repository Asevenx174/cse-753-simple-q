import numpy as np

"The purpose of this experiments is to resembles the selection/evaluation separation in Double DQN"

def run_single_trial(
    action_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Generate one standard and one independently evaluated estimate."""
    if action_count < 1:
        raise ValueError("action_count must be at least 1")

    estimator_a = rng.normal(
        loc=0.0,
        scale=1.0,
        size=action_count,
    )

    estimator_b = rng.normal(
        loc=0.0,
        scale=1.0,
        size=action_count,
    )

    # selects an action index that contains the largest value
    selected_action = int(np.argmax(estimator_a))
    # evaluated by the same estimator that selects the actions
    maximum_error = float(estimator_a[selected_action])
    # independent evaluation by the other estimator
    double_error = float(estimator_b[selected_action])

     return (
        estimator_a,
        estimator_b,
        selected_action,
        maximum_error,
        double_error,
    )
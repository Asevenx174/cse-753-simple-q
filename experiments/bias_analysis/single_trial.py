import numpy as np

"The purpose of this experiments is to resembles the selection/evaluation separation in Double DQN"

def run_single_trial(
    action_count: int,
    rng: np.random.Generator,
    noise_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """Generate one standard and one independently evaluated estimate."""
    if noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative")

    if action_count < 1:
        raise ValueError("action_count must be at least 1")

    estimator_a = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=action_count,
    )

    estimator_b = rng.normal(
        loc=0.0,
        scale=noise_std,
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


def main() -> None:
    """Run and print one reproducible trial."""
    action_count = 5
    seed = 42

    rng = np.random.default_rng(seed)

    (
        estimator_a,
        estimator_b,
        selected_action,
        maximum_error,
        double_error,
    ) = run_single_trial(
        action_count=action_count,
        rng=rng,
    )

    np.set_printoptions(precision=4, suppress=True)

    print("Synthetic maximization-bias trial")
    print(f"Seed: {seed}")
    print(f"Action count: {action_count}")
    print("True value of every action: 0.0")
    print(f"Estimator A: {estimator_a}")
    print(f"Estimator B: {estimator_b}")
    print(f"Action selected using A: {selected_action}")
    print(f"Standard maximum error: {maximum_error:.4f}")
    print(f"Independent evaluation error: {double_error:.4f}")


if __name__ == "__main__":
    main()
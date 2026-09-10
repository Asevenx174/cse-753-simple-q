"""Verify the runtime required by the Double DQN project."""

from importlib.metadata import PackageNotFoundError, version
import platform
import sys

import gymnasium as gym
import numpy as np
import torch
import tianshou


REQUIRED_PACKAGES = (
    "gymnasium",
    "numpy",
    "torch",
    "tianshou",
    "matplotlib",
)

ENVIRONMENT_IDS = (
    "CartPole-v1",
    "Acrobot-v1",
)

CHECK_SEED = 42


def check_package_versions() -> None:
    """Print versions and fail if a required package is unavailable."""
    missing_packages = []

    print("Required package versions")

    for package_name in REQUIRED_PACKAGES:
        try:
            package_version = version(package_name)
            print(f"  {package_name}: {package_version}")
        except PackageNotFoundError:
            missing_packages.append(package_name)
            print(f"  {package_name}: NOT INSTALLED")

    if missing_packages:
        missing_text = ", ".join(missing_packages)
        raise RuntimeError(f"Missing required packages: {missing_text}")


def check_pytorch() -> None:
    """Verify a small deterministic PyTorch calculation."""
    values = torch.tensor([1.0, 2.0, 3.0])
    result = values.square()

    expected = torch.tensor([1.0, 4.0, 9.0])

    if not torch.equal(result, expected):
        raise RuntimeError("PyTorch calculation produced an unexpected result")

    print("\nPyTorch check")
    print(f"  Input: {values.tolist()}")
    print(f"  Squared: {result.tolist()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Device used for early checks: CPU")


def sample_transition(environment_id: str) -> tuple:
    """Generate one seeded transition."""
    env = gym.make(environment_id)

    try:
        observation, _ = env.reset(seed=CHECK_SEED)
        env.action_space.seed(CHECK_SEED)

        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, _ = env.step(action)

        return (
            observation,
            action,
            next_observation,
            float(reward),
            bool(terminated),
            bool(truncated),
        )
    finally:
        env.close()


def check_environment(environment_id: str) -> None:
    """Verify that a seeded environment transition is reproducible."""
    first_transition = sample_transition(environment_id)
    second_transition = sample_transition(environment_id)

    first_observation = first_transition[0]
    second_observation = second_transition[0]

    first_action = first_transition[1]
    second_action = second_transition[1]

    first_next_observation = first_transition[2]
    second_next_observation = second_transition[2]

    np.testing.assert_allclose(first_observation, second_observation)
    np.testing.assert_allclose(
        first_next_observation,
        second_next_observation,
    )

    if first_action != second_action:
        raise RuntimeError(f"{environment_id} action seeding failed")

    if not np.all(np.isfinite(first_observation)):
        raise RuntimeError(f"{environment_id} produced a nonfinite observation")

    if not np.all(np.isfinite(first_next_observation)):
        raise RuntimeError(
            f"{environment_id} produced a nonfinite next observation"
        )

    print(f"\n{environment_id}: OK")
    print(f"  Observation shape: {first_observation.shape}")
    print(f"  Seeded action: {first_action}")
    print(f"  Reward: {first_transition[3]}")
    print(f"  Terminated: {first_transition[4]}")
    print(f"  Truncated: {first_transition[5]}")
    print("  Reproducible transition: yes")


def main() -> None:
    """Run all project runtime checks."""
    print("System information")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Operating system: {platform.platform()}")
    print(f"  Tianshou import: {tianshou.__version__}")

    check_package_versions()
    check_pytorch()

    print("\nGymnasium environment checks")

    for environment_id in ENVIRONMENT_IDS:
        check_environment(environment_id)

    print("\nAll required runtime checks passed.")


if __name__ == "__main__":
    main()
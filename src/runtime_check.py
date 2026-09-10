"""Verify the Python environment and proposed Gymnasium environments."""

from importlib.metadata import PackageNotFoundError, version

import gymnasium as gym
import torch


REQUIRED_PACKAGES = [
    "gymnasium",
    "numpy",
    "torch",
    "tianshou",
    "matplotlib",
]

ENVIRONMENT_IDS = [
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
]


def print_package_versions() -> None:
    """Print the installed version of each required package."""
    print("Package versions")

    for package_name in REQUIRED_PACKAGES:
        try:
            installed_version = version(package_name)
            print(f"  {package_name}: {installed_version}")
        except PackageNotFoundError:
            print(f"  {package_name}: NOT INSTALLED")


def print_torch_device() -> None:
    """Print the computation device available to PyTorch."""
    print("\nPyTorch device")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Device: CPU")


def check_environment(environment_id: str) -> None:
    """Create and execute one random step in an environment."""
    env = gym.make(environment_id)

    try:
        observation, info = env.reset(seed=42)
        action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = env.step(action)

        print(f"\n{environment_id}: OK")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Initial observation shape: {observation.shape}")
        print(f"  Sample action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Next observation shape: {next_observation.shape}")
    finally:
        env.close()


def main() -> None:
    """Run all runtime checks."""
    print_package_versions()
    print_torch_device()

    print("\nEnvironment checks")

    failures = []

    for environment_id in ENVIRONMENT_IDS:
        try:
            check_environment(environment_id)
        except Exception as error:
            failures.append(environment_id)
            print(f"\n{environment_id}: FAILED")
            print(f"  Reason: {error}")

    if failures:
        failed_names = ", ".join(failures)
        raise RuntimeError(f"Environment checks failed: {failed_names}")

    print("\nAll runtime checks passed.")


if __name__ == "__main__":
    main()
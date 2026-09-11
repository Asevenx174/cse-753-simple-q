"""Repeat the synthetic maximization-bias experiment and save results."""

import csv
from pathlib import Path

import numpy as np

from experiments.bias_analysis.single_trial import run_single_trial


ACTION_COUNTS = (2, 8, 32, 128, 256)
TRIAL_COUNT = 10_000
SEED = 42

RESULT_DIRECTORY = Path("results/synthetic")
TRIALS_PATH = RESULT_DIRECTORY / "trials.csv"
SUMMARY_PATH = RESULT_DIRECTORY / "summary.csv"


def run_repeated_trials() -> tuple[list[dict], list[dict]]:
    """Run every trial and calculate summary statistics."""
    rng = np.random.default_rng(SEED)

    trial_rows = []
    errors = {
        action_count: {
            "standard_maximum": [],
            "independent_evaluation": [],
        }
        for action_count in ACTION_COUNTS
    }

    for action_count in ACTION_COUNTS:
        for trial_id in range(TRIAL_COUNT):
            (
                _,
                _,
                _,
                maximum_error,
                double_error,
            ) = run_single_trial(
                action_count=action_count,
                rng=rng,
            )

            trial_rows.append(
                {
                    "action_count": action_count,
                    "trial_id": trial_id,
                    "maximum_error": maximum_error,
                    "double_error": double_error,
                }
            )

            errors[action_count]["standard_maximum"].append(
                maximum_error
            )
            errors[action_count]["independent_evaluation"].append(
                double_error
            )

    summary_rows = []

    for action_count in ACTION_COUNTS:
        for estimator_name, estimator_errors in errors[action_count].items():
            values = np.asarray(estimator_errors, dtype=np.float64)

            mean_error = float(np.mean(values))
            sd_error = float(np.std(values, ddof=1))
            se_error = sd_error / np.sqrt(len(values))

            summary_rows.append(
                {
                    "action_count": action_count,
                    "estimator": estimator_name,
                    "n_trials": len(values),
                    "mean_error": mean_error,
                    "sd_error": sd_error,
                    "se_error": se_error,
                }
            )

    return trial_rows, summary_rows


def save_trials(trial_rows: list[dict]) -> None:
    """Save every trial to a CSV file."""
    with TRIALS_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "action_count",
                "trial_id",
                "maximum_error",
                "double_error",
            ],
        )

        writer.writeheader()
        writer.writerows(trial_rows)


def save_summary(summary_rows: list[dict]) -> None:
    """Save the aggregated statistics to a CSV file."""
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "action_count",
                "estimator",
                "n_trials",
                "mean_error",
                "sd_error",
                "se_error",
            ],
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def print_summary(summary_rows: list[dict]) -> None:
    """Print a readable summary to the terminal."""
    heading = (
        f"{'Actions':>8}  "
        f"{'Estimator':>23}  "
        f"{'Mean':>9}  "
        f"{'SD':>9}  "
        f"{'SE':>9}"
    )

    print(heading)
    print("-" * len(heading))

    for row in summary_rows:
        print(
            f"{row['action_count']:>8}  "
            f"{row['estimator']:>23}  "
            f"{row['mean_error']:>9.4f}  "
            f"{row['sd_error']:>9.4f}  "
            f"{row['se_error']:>9.4f}"
        )


def main() -> None:
    """Run, save, and display the complete experiment."""
    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    trial_rows, summary_rows = run_repeated_trials()

    save_trials(trial_rows)
    save_summary(summary_rows)
    print_summary(summary_rows)

    print(f"\nSaved raw trials to: {TRIALS_PATH}")
    print(f"Saved summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
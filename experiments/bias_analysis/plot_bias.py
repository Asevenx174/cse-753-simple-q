"""Plot the synthetic maximization-bias results."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_PATH = Path("results/synthetic/summary.csv")
FIGURE_DIRECTORY = Path("results/synthetic/figures")


def read_estimator(estimator_name: str) -> tuple[list[int], list[float], list[float]]:
    """Read one estimator's results from the summary file."""
    action_counts = []
    mean_errors = []
    standard_errors = []

    with SUMMARY_PATH.open(encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["estimator"] == estimator_name:
                action_counts.append(int(row["action_count"]))
                mean_errors.append(float(row["mean_error"]))
                standard_errors.append(float(row["se_error"]))

    return action_counts, mean_errors, standard_errors


def main() -> None:
    """Create and save the comparison figure."""
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    standard = read_estimator("standard_maximum")
    independent = read_estimator("independent_evaluation")

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        standard[0],
        standard[1],
        yerr=standard[2],
        marker="o",
        capsize=4,
        label="Standard maximum",
    )

    plt.errorbar(
        independent[0],
        independent[1],
        yerr=independent[2],
        marker="s",
        capsize=4,
        label="Independent evaluation",
    )

    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xscale("log", base=2)
    plt.xticks(standard[0], standard[0])

    plt.xlabel("Number of actions")
    plt.ylabel("Mean estimation error")
    plt.title("Synthetic Maximization Bias")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY / "maximization_bias.png",
        dpi=300,
    )
    plt.savefig(
        FIGURE_DIRECTORY / "maximization_bias.pdf",
    )

    plt.close()

    print(f"Saved figures to: {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
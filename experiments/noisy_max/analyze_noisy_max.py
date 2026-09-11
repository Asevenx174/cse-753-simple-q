"""Create the compact NoisyMax analysis."""

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT = Path("results/analysis/noisy_max")
LATE_START = 40_000
ALGORITHMS = ("dqn", "double_dqn")
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}
COLORS = {"dqn": "tab:blue", "double_dqn": "tab:orange"}


def read_csv(path):
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_csv(name, rows):
    path = OUTPUT / name

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    runs = []

    for path in glob.glob(
        "results/runs/noisy_max/*/*/config.json"
    ):
        directory = Path(path).parent

        with open(path, encoding="utf-8") as file:
            config = json.load(file)

        if (
            config.get("M") in (2, 10, 50)
            and config.get("seed") in (0, 1, 2)
        ):
            runs.append(
                {
                    "directory": directory,
                    "algorithm": config["algorithm"],
                    "M": config["M"],
                    "seed": config["seed"],
                    "diagnostics": read_csv(
                        directory / "diagnostics.csv"
                    ),
                    "losses": read_csv(
                        directory / "loss.csv"
                    ),
                    "coverage": read_csv(
                        directory / "action_coverage.csv"
                    ),
                }
            )

    if len(runs) != 18:
        raise RuntimeError(
            f"Expected 18 runs, found {len(runs)}."
        )

    return runs


def mean_se(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, se


def create_plot_data(runs):
    plot_rows = []

    for algorithm in ALGORITHMS:
        selected = [
            run for run in runs
            if run["algorithm"] == algorithm
            and run["M"] == 10
        ]

        for index in range(50):
            step = int(
                selected[0]["diagnostics"][index][
                    "environment_step"
                ]
            )

            errors = [
                float(run["diagnostics"][index][
                    "risky_signed_error"
                ])
                for run in selected
            ]
            risky = [
                float(run["diagnostics"][index][
                    "greedy_risky_choice"
                ])
                for run in selected
            ]

            error_mean, error_se = mean_se(errors)
            risky_mean, risky_se = mean_se(risky)

            plot_rows.append(
                {
                    "panel": "risky_error",
                    "algorithm": algorithm,
                    "x": step,
                    "mean": error_mean,
                    "se": error_se,
                }
            )
            plot_rows.append(
                {
                    "panel": "greedy_risky_fraction",
                    "algorithm": algorithm,
                    "x": step,
                    "mean": risky_mean,
                    "se": risky_se,
                }
            )

    for algorithm in ALGORITHMS:
        for action_count in (2, 10, 50):
            values = []

            for run in runs:
                if (
                    run["algorithm"] == algorithm
                    and run["M"] == action_count
                ):
                    late = [
                        float(row["risky_signed_error"])
                        for row in run["diagnostics"]
                        if int(row["environment_step"])
                        >= LATE_START
                    ]
                    values.append(float(np.mean(late)))

            mean, se = mean_se(values)

            plot_rows.append(
                {
                    "panel": "late_value_error",
                    "algorithm": algorithm,
                    "x": action_count,
                    "mean": mean,
                    "se": se,
                }
            )

    return plot_rows


def create_summaries(runs):
    error_rows = []
    loss_rows = []
    coverage_rows = []
    return_rows = []

    for run in runs:
        late_diagnostics = [
            row for row in run["diagnostics"]
            if int(row["environment_step"]) >= LATE_START
        ]
        late_losses = [
            float(row["td_loss"])
            for row in run["losses"]
            if int(row["environment_step"]) >= LATE_START
        ]
        counts = [
            int(row["sample_count"])
            for row in run["coverage"]
        ]

        base = {
            "algorithm": run["algorithm"],
            "M": run["M"],
            "seed": run["seed"],
        }

        error_rows.append(
            {
                **base,
                "safe_absolute_error": np.mean([
                    float(row["safe_absolute_error"])
                    for row in late_diagnostics
                ]),
                "risky_absolute_error": np.mean([
                    float(row["risky_absolute_error"])
                    for row in late_diagnostics
                ]),
                "noisy_absolute_error": np.mean([
                    float(row["noisy_absolute_error"])
                    for row in late_diagnostics
                ]),
            }
        )

        loss_rows.append(
            {
                **base,
                "late_mean_loss": np.mean(late_losses),
                "late_max_loss": np.max(late_losses),
                "all_finite": all(
                    math.isfinite(value)
                    for value in late_losses
                ),
            }
        )

        coverage_rows.append(
            {
                **base,
                "total_noisy_samples": sum(counts),
                "minimum_action_samples": min(counts),
                "maximum_action_samples": max(counts),
                "all_actions_sampled": min(counts) > 0,
            }
        )

        # Safe has expected return 0; risky has expected return -0.1.
        expected_returns = [
            -0.1 * float(row["greedy_risky_choice"])
            for row in late_diagnostics
        ]

        return_rows.append(
            {
                **base,
                "late_expected_greedy_return": np.mean(
                    expected_returns
                ),
                "late_greedy_risky_fraction": np.mean([
                    float(row["greedy_risky_choice"])
                    for row in late_diagnostics
                ]),
            }
        )

    return error_rows, loss_rows, coverage_rows, return_rows


def draw_figure(plot_rows):
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
    )

    panels = (
        (
            "risky_error",
            "Risky-value signed error, M=10",
            "Environment step",
            "Signed error",
        ),
        (
            "greedy_risky_fraction",
            "Greedy risky-choice fraction, M=10",
            "Environment step",
            "Fraction",
        ),
        (
            "late_value_error",
            "Late-training error vs action count",
            "Noisy actions (M)",
            "Signed error",
        ),
    )

    for axis, (panel, title, xlabel, ylabel) in zip(
        axes,
        panels,
    ):
        for algorithm in ALGORITHMS:
            rows = [
                row for row in plot_rows
                if row["panel"] == panel
                and row["algorithm"] == algorithm
            ]

            x = np.array([float(row["x"]) for row in rows])
            mean = np.array([
                float(row["mean"]) for row in rows
            ])
            se = np.array([float(row["se"]) for row in rows])

            axis.plot(
                x,
                mean,
                marker="o" if panel == "late_value_error" else None,
                label=LABELS[algorithm],
                color=COLORS[algorithm],
            )
            axis.fill_between(
                x,
                mean - se,
                mean + se,
                alpha=0.2,
                color=COLORS[algorithm],
            )

        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)

    axes[0].legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "noisy_max_analysis.png",
        dpi=200,
    )
    plt.close(figure)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    plot_rows = create_plot_data(runs)
    summaries = create_summaries(runs)

    save_csv("plot_data.csv", plot_rows)
    save_csv("absolute_error_summary.csv", summaries[0])
    save_csv("loss_summary.csv", summaries[1])
    save_csv("coverage_summary.csv", summaries[2])
    save_csv("return_summary.csv", summaries[3])

    draw_figure(plot_rows)

    print("Analyzed runs:", len(runs))
    print("Late training:", LATE_START, "to 50000")
    print("Saved analysis to:", OUTPUT)


if __name__ == "__main__":
    main()
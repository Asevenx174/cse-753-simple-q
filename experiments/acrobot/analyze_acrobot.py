"""Summarize the three paired Acrobot seeds."""

import csv
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/acrobot")
OUT = Path("results/analysis/acrobot")
SEEDS = (0, 1, 2)
ALGORITHMS = ("dqn", "double_dqn")
LATE_START = 160_000
THRESHOLD = -150


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    runs = {}
    for path in ROOT.glob("*/*/config.json"):
        config = json.loads(path.read_text())
        key = (config["algorithm"], config["seed"])
        if key[0] not in ALGORITHMS or key[1] not in SEEDS:
            continue
        if key in runs:
            raise ValueError(f"Duplicate run: {key}")
        assert config["total_steps"] == 200_000
        runs[key] = path.parent

    assert set(runs) == {
        (algorithm, seed)
        for algorithm in ALGORITHMS for seed in SEEDS
    }, "Expected exactly one run for each algorithm–seed pair"
    return runs


def checkpoints(rows):
    """Return step -> mean return and timeout count."""
    by_step = {}
    for row in rows:
        step = int(row["environment_step"])
        by_step.setdefault(step, []).append(float(row["episode_return"]))

    assert set(by_step) == set(range(0, 200_001, 5_000))
    assert all(len(values) == 20 for values in by_step.values())
    return {
        step: (stats.mean(values), sum(v <= -500 for v in values))
        for step, values in by_step.items()
    }


def speed(checkpoint_data):
    """First step ending three consecutive means >= -150."""
    steps = sorted(checkpoint_data)
    for i in range(2, len(steps)):
        if all(
            checkpoint_data[steps[j]][0] >= THRESHOLD
            for j in (i - 2, i - 1, i)
        ):
            return steps[i]
    return "not reached"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    per_seed, curve_rows, summary = [], [], []

    fig, ax = plt.subplots(figsize=(9, 5))

    for algorithm in ALGORITHMS:
        final_means = []
        curves = []

        for seed in SEEDS:
            directory = runs[algorithm, seed]
            final = read_csv(directory / "final_evaluation.csv")
            assert len(final) == 100
            assert [int(r["evaluation_seed"]) for r in final] == list(
                range(20_000, 20_100)
            )

            final_values = [float(r["episode_return"]) for r in final]
            data = checkpoints(read_csv(directory / "evaluation.csv"))
            late = [
                data[step][0] for step in sorted(data)
                if step >= LATE_START
            ]
            assert len(late) == 9

            row = {
                "algorithm": algorithm,
                "seed": seed,
                "final_100_episode_mean": stats.mean(final_values),
                "final_100_episode_timeouts": sum(
                    value <= -500 for value in final_values
                ),
                "late_checkpoint_mean": stats.mean(late),
                "late_checkpoint_sd": stats.stdev(late),
                "first_three_at_least_minus_150": speed(data),
            }
            per_seed.append(row)
            final_means.append(row["final_100_episode_mean"])
            curves.append(data)

            for step, (mean, timeouts) in sorted(data.items()):
                curve_rows.append({
                    "algorithm": algorithm,
                    "seed": seed,
                    "step": step,
                    "checkpoint_mean": mean,
                    "checkpoint_timeouts": timeouts,
                })

        steps = sorted(curves[0])
        ax.plot(
            steps,
            [stats.mean(c[step][0] for c in curves) for step in steps],
            label=algorithm,
        )
        summary.append({
            "algorithm": algorithm,
            "seeds": len(SEEDS),
            "final_mean_across_seeds": stats.mean(final_means),
            "final_sample_sd_across_seeds": stats.stdev(final_means),
        })

    ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Environment steps",
        ylabel="Mean return (higher is better)",
        title="Acrobot learning curves: mean across three seeds",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=180)
    plt.close(fig)

    write_csv(OUT / "per_seed.csv", per_seed)
    write_csv(OUT / "learning_curves.csv", curve_rows)
    write_csv(OUT / "summary.csv", summary)
    for row in summary:
        print(row)
    print("Saved analysis to:", OUT)


if __name__ == "__main__":
    main()
"""Generate report Acrobot figures from six selected corrected runs; no training."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SELECTED = ROOT / "results/analysis/selected_benchmarks_corrected.csv"
OUTPUT = ROOT / "report/figures"
COLORS = {"dqn": "tab:blue", "double_dqn": "tab:orange"}
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}
SEEDS = (0, 1, 2)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def selected_runs():
    runs = {}
    for row in read_csv(SELECTED):
        if row["environment"] != "acrobot":
            continue
        path = ROOT / row["source_directory"]
        config = json.loads((path / "config.json").read_text())
        key = (config["algorithm"], int(config["seed"]))
        assert key == (row["algorithm"], int(row["seed"]))
        assert config["total_steps"] == 200_000
        assert config["epsilon"] == 0.10
        assert config["collector_exploration_noise"] is True
        assert key not in runs, f"Duplicate run: {key}"
        runs[key] = path
    assert set(runs) == {(a, s) for a in COLORS for s in SEEDS}
    return runs


def monitoring(path):
    grouped = defaultdict(list)
    for row in read_csv(path / "evaluation.csv"):
        grouped[int(row["environment_step"])].append(float(row["episode_return"]))
    assert set(grouped) == set(range(0, 200_001, 5_000))
    assert all(len(values) == 20 for values in grouped.values())
    return {step: mean(values) for step, values in grouped.items()}


def binned(path, filename, step_column, value_column):
    grouped = defaultdict(list)
    for row in read_csv(path / filename):
        step = int(row[step_column])
        value = float(row[value_column])
        assert 0 < step <= 200_000 and math.isfinite(value)
        end_of_bin = math.ceil(step / 5_000) * 5_000
        grouped[end_of_bin].append(value)
    assert set(grouped) == set(range(5_000, 200_001, 5_000))
    return {step: mean(values) for step, values in grouped.items()}


def draw(ax, values_by_run, ylabel, threshold=False):
    for algorithm, color in COLORS.items():
        for seed in SEEDS:
            data = values_by_run[algorithm, seed]
            ax.plot(sorted(data), [data[s] for s in sorted(data)],
                    color=color, alpha=0.19, linewidth=0.9)
        steps = sorted(values_by_run[algorithm, SEEDS[0]])
        ax.plot(steps, [mean(values_by_run[algorithm, seed][s]
                             for seed in SEEDS) for s in steps],
                color=color, linewidth=2, label=LABELS[algorithm])
    if threshold:
        ax.axhline(-150, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="Environment steps", ylabel=ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    ax.legend()


def main():
    runs = selected_runs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10.5})

    figure, axis = plt.subplots(figsize=(9, 4.3))
    draw(axis, {key: monitoring(path) for key, path in runs.items()},
         "Greedy monitoring return (higher is better)", threshold=True)
    figure.tight_layout()
    figure.savefig(OUTPUT / "fig03_acrobot.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, spec in zip(axes, (
        ("training.csv", "ending_environment_step", "episode_return",
         "Exploratory training return"),
        ("loss.csv", "environment_step", "td_loss", "TD mean-square loss"),
    )):
        filename, step_column, value_column, ylabel = spec
        draw(axis, {key: binned(path, filename, step_column, value_column)
                    for key, path in runs.items()}, ylabel)
    figure.tight_layout()
    figure.savefig(OUTPUT / "fig05_training_loss.pdf")
    plt.close(figure)
    print("Generated six-run corrected Acrobot figures:")
    for filename in ("fig03_acrobot.pdf", "fig05_training_loss.pdf"):
        print(OUTPUT / filename)


if __name__ == "__main__":
    main()

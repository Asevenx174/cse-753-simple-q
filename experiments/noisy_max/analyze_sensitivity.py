"""Combine controlled NoisyMax sensitivity contrasts."""

import csv
import json
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/runs/noisy_max")
OUT = Path("results/analysis/noisy_max")
runs = {}


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


for path in ROOT.glob("*/*/config.json"):
    config = json.loads(path.read_text())
    key = (
        config.get("algorithm"),
        config.get("seed"),
        config.get("C"),
        config.get("epsilon"),
    )
    if (config.get("M") != 10
            or key[0] not in ("dqn", "double_dqn")
            or key[1] not in (0, 1, 2)
            or (key[2], key[3]) not in (
                (320, 0.10), (1000, 0.10), (320, 0.20)
            )):
        continue
    if key in runs:
        raise ValueError(f"Duplicate condition: {key}")
    runs[key] = path.parent

assert len(runs) == 18, f"Expected 18 unique runs, found {len(runs)}"

rows = []
for (algorithm, seed, c, epsilon), directory in sorted(runs.items()):
    diagnostics = read(directory / "diagnostics.csv")
    losses = read(directory / "loss.csv")
    late = [
        r for r in diagnostics
        if int(r["environment_step"]) >= 40_000
    ]
    late_losses = [
        float(r["td_loss"]) for r in losses
        if int(r["environment_step"]) >= 40_000
    ]
    assert len(late) == 11 and late_losses

    errors = [float(r["risky_signed_error"]) for r in late]
    rows.append({
        "algorithm": algorithm,
        "seed": seed,
        "C": c,
        "epsilon": epsilon,
        "late_signed_error": stats.mean(errors),
        "late_absolute_error": stats.mean(map(abs, errors)),
        "late_error_sd": stats.stdev(errors),
        "late_mean_loss": stats.mean(late_losses),
        "late_greedy_risky_fraction": stats.mean(
            int(r["greedy_risky_choice"]) for r in late
        ),
        "final_training_risky_fraction": float(
            diagnostics[-1]["training_risky_fraction"]
        ),
        "source_directory": str(directory),
    })

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "sensitivity_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
panels = [
    ("Target-copy interval", [(320, 0.10), (1000, 0.10)],
     ["C=320", "C=1000"]),
    ("Exploration rate", [(320, 0.10), (320, 0.20)],
     ["ε=0.10", "ε=0.20"]),
]

for axis, (title, conditions, labels) in zip(axes, panels):
    for algorithm, offset, color in (
        ("dqn", -0.13, "tab:blue"),
        ("double_dqn", 0.13, "tab:orange"),
    ):
        for x, (c, epsilon) in enumerate(conditions):
            values = [
                r["late_signed_error"] for r in rows
                if r["algorithm"] == algorithm
                and r["C"] == c and r["epsilon"] == epsilon
            ]
            assert len(values) == 3
            axis.scatter(
                [x + offset] * 3, values,
                color=color, alpha=0.65, s=25,
            )
            axis.scatter(
                x + offset, stats.mean(values),
                color=color, marker="_", s=250,
                label=algorithm if x == 0 else None,
            )
    axis.set_xticks(range(2), labels)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

axes[0].set_ylabel("Late risky-value signed error")
fig.tight_layout()
fig.savefig(OUT / "sensitivity_contrasts.png", dpi=180)
plt.close(fig)
print("Verified 18 unique runs; saved analysis to", OUT)
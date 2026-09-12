"""Summarize the paired NoisyMax target-update contrast."""

import csv
import json
import math
import statistics as stats
from pathlib import Path

ROOT = Path("results/runs/noisy_max")
OUT = Path("results/analysis/noisy_max/target_update.csv")
runs = {}

def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

for path in ROOT.glob("*/*/config.json"):
    config = json.loads(path.read_text())
    if (config.get("M") not in (10,)
            or config.get("seed") not in (0, 1, 2)
            or config.get("C") not in (320, 1000)):
        continue

    key = (config["algorithm"], config["seed"], config["C"])
    if key in runs:
        raise ValueError(f"Duplicate condition: {key}")
    assert config["epsilon"] == 0.10
    assert config["total_steps"] == 50_000
    runs[key] = path.parent

expected = {
    (algorithm, seed, c)
    for algorithm in ("dqn", "double_dqn")
    for seed in (0, 1, 2)
    for c in (320, 1000)
}
assert set(runs) == expected, f"Missing: {expected - set(runs)}"

rows = []
for (algorithm, seed, c), directory in sorted(runs.items()):
    diagnostics = read(directory / "diagnostics.csv")
    losses = read(directory / "loss.csv")
    coverage = read(directory / "action_coverage.csv")
    late = [
        row for row in diagnostics
        if int(row["environment_step"]) >= 40_000
    ]
    late_losses = [
        float(row["td_loss"]) for row in losses
        if int(row["environment_step"]) >= 40_000
    ]
    errors = [float(row["risky_signed_error"]) for row in late]

    assert len(late) == 11
    assert int(diagnostics[-1]["environment_step"]) == 50_000
    assert late_losses and all(math.isfinite(x) for x in late_losses)
    assert len(coverage) == 10
    assert all(int(row["sample_count"]) > 0 for row in coverage)

    rows.append({
        "algorithm": algorithm,
        "seed": seed,
        "C": c,
        "late_mean_risky_signed_error": stats.mean(errors),
        "late_mean_risky_absolute_error": stats.mean(map(abs, errors)),
        "late_greedy_risky_fraction": stats.mean(
            int(row["greedy_risky_choice"]) for row in late
        ),
        "late_error_sd": stats.stdev(errors),
        "late_mean_loss": stats.mean(late_losses),
        "final_greedy_risky": diagnostics[-1]["greedy_risky_choice"],
        "minimum_action_samples": min(
            int(row["sample_count"]) for row in coverage
        ),
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)

for row in rows:
    print(row)
print("Verified 12 matched runs; saved:", OUT)
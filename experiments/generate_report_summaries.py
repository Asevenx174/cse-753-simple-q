"""Build verified report summaries from explicitly selected runs."""

import csv
import json
import math
import shutil
import statistics as stats
from pathlib import Path

OUT = Path("results/analysis/report")

# Each list is ordered: M=2 seeds 0–2; M=10 seeds 0–2;
# M=50 seeds 0–2; M=10,C=1000 seeds 0–2;
# M=10,epsilon=0.20 seeds 0–2.
IDS = {
    "dqn": [
        "20260911_212330_157585", "20260911_212447_972853",
        "20260911_212605_112762", "20260911_212723_217864",
        "20260911_212840_770706", "20260911_212958_288294",
        "20260911_213115_848110", "20260911_213234_442382",
        "20260911_213351_325145", "20260912_135015_815213",
        "20260912_135134_115974", "20260912_135255_128799",
        "20260912_140129_519131", "20260912_140249_530851",
        "20260912_140410_417612",
    ],
    "double_dqn": [
        "20260911_212409_153370", "20260911_212526_662068",
        "20260911_212644_004314", "20260911_212802_172631",
        "20260911_212919_589769", "20260911_213037_247642",
        "20260911_213155_201318", "20260911_213313_000696",
        "20260911_213429_679598", "20260912_135055_166179",
        "20260912_135215_358866", "20260912_135334_739628",
        "20260912_140208_957212", "20260912_140330_221472",
        "20260912_140450_796705",
    ],
}

ACROBOT = {
    "dqn": [
        "20260912_175850_049315",  # seed 0
        "20260912_180225_870105",  # seed 1
        "20260912_180604_526662",  # seed 2
    ],
    "double_dqn": [
        "20260912_180038_836690",  # seed 0
        "20260912_180415_092895",  # seed 1
        "20260912_180752_214060",  # seed 2
    ],
}

FIGURES = {
    "synthetic_bias.png":
        "results/synthetic/figures/maximization_bias.png",
    "noisy_max.png":
        "results/analysis/noisy_max/noisy_max_analysis.png",
    "acrobot.png":
        "results/analysis/acrobot/learning_curves.png",
    "sensitivity.png":
        "results/analysis/noisy_max/sensitivity_contrasts.png",
}


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def checked_losses(directory):
    values = [
        float(row["td_loss"])
        for row in read(directory / "loss.csv")
    ]
    assert values and all(math.isfinite(v) for v in values)
    return len(values)


def noisy_condition(index):
    if index < 9:
        return ((2, 10, 50)[index // 3], 320, 0.10)
    if index < 12:
        return (10, 1000, 0.10)
    return (10, 320, 0.20)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    noisy_rows, acrobot_rows = [], []
    selected_paths = set()

    for algorithm, ids in IDS.items():
        assert len(ids) == 15
        for index, run_id in enumerate(ids):
            seed = index % 3
            m, c, epsilon = noisy_condition(index)
            directory = Path(
                f"results/runs/noisy_max/{algorithm}/{run_id}"
            )
            assert directory not in selected_paths
            selected_paths.add(directory)

            config = json.loads((directory / "config.json").read_text())
            assert (
                config["algorithm"], config["seed"],
                config["M"], config["C"], config["epsilon"],
                config["total_steps"],
            ) == (algorithm, seed, m, c, epsilon, 50_000)
            checked_losses(directory)

            diagnostics = read(directory / "diagnostics.csv")
            late = [
                row for row in diagnostics
                if int(row["environment_step"]) >= 40_000
            ]
            assert len(late) == 11
            assert int(diagnostics[-1]["environment_step"]) == 50_000
            errors = [float(r["risky_signed_error"]) for r in late]
            assert all(math.isfinite(v) for v in errors)

            coverage = read(directory / "action_coverage.csv")
            assert len(coverage) == m
            assert all(int(r["sample_count"]) > 0 for r in coverage)

            noisy_rows.append({
                "algorithm": algorithm, "seed": seed,
                "M": m, "C": c, "epsilon": epsilon,
                "late_signed_error": stats.mean(errors),
                "late_absolute_error": stats.mean(map(abs, errors)),
                "late_greedy_risky_fraction": stats.mean(
                    int(r["greedy_risky_choice"]) for r in late
                ),
                "source_directory": str(directory),
            })

    for algorithm, ids in ACROBOT.items():
        assert len(ids) == 3
        for seed, run_id in enumerate(ids):
            directory = Path(
                f"results/runs/acrobot/{algorithm}/{run_id}"
            )
            assert directory not in selected_paths
            selected_paths.add(directory)

            config = json.loads((directory / "config.json").read_text())
            assert (
                config["algorithm"], config["seed"],
                config["total_steps"], config["evaluation_episodes"],
            ) == (algorithm, seed, 200_000, 20)
            checked_losses(directory)

            evaluations = read(directory / "final_evaluation.csv")
            assert len(evaluations) == 100
            assert [int(r["evaluation_seed"]) for r in evaluations] == (
                list(range(20_000, 20_100))
            )
            returns = [float(r["episode_return"]) for r in evaluations]
            assert all(math.isfinite(v) for v in returns)

            acrobot_rows.append({
                "algorithm": algorithm, "seed": seed,
                "final_return": stats.mean(returns),
                "timeouts": sum(v <= -500 for v in returns),
                "source_directory": str(directory),
            })

    assert len(noisy_rows) == 30
    assert len(acrobot_rows) == 6
    assert len(selected_paths) == 36

    noisy_table = []
    for algorithm in IDS:
        for m, c, epsilon in (
            (2, 320, 0.10), (10, 320, 0.10),
            (50, 320, 0.10), (10, 1000, 0.10),
            (10, 320, 0.20),
        ):
            group = [
                r for r in noisy_rows
                if (r["algorithm"], r["M"], r["C"], r["epsilon"])
                == (algorithm, m, c, epsilon)
            ]
            assert {r["seed"] for r in group} == {0, 1, 2}
            values = [r["late_signed_error"] for r in group]
            noisy_table.append({
                "algorithm": algorithm, "M": m, "C": c,
                "epsilon": epsilon,
                "mean_late_signed_error": stats.mean(values),
                "sample_sd": stats.stdev(values),
            })

    acrobot_table = []
    for algorithm in ACROBOT:
        group = [r for r in acrobot_rows if r["algorithm"] == algorithm]
        assert {r["seed"] for r in group} == {0, 1, 2}
        values = [r["final_return"] for r in group]
        acrobot_table.append({
            "algorithm": algorithm,
            "mean_final_return": stats.mean(values),
            "sample_sd": stats.stdev(values),
        })

    write("noisy_max_seed_level.csv", noisy_rows)
    write("acrobot_seed_level.csv", acrobot_rows)
    write("table_noisy_max.csv", noisy_table)
    write("table_acrobot.csv", acrobot_table)

    figures_dir = OUT / "figures"
    figures_dir.mkdir(exist_ok=True)
    for name, source in FIGURES.items():
        path = Path(source)
        assert path.is_file() and path.stat().st_size > 0, source
        shutil.copy2(path, figures_dir / name)

    print("Verified 36 selected runs, two tables, four figures.")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
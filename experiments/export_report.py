"""Export selected report evidence without changing original runs."""

import csv
import shutil
from pathlib import Path

from PIL import Image

SOURCE = Path("results/analysis/report")
REPORT = Path("report")
TABLES = REPORT / "tables"
FIGURES = REPORT / "figures"

IMAGES = {
    "fig01_synthetic": "synthetic_bias.png",
    "fig02_noisymax": "noisy_max.png",
    "fig03_acrobot": "acrobot.png",
    "fig04_sensitivity": "sensitivity.png",
}


def read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    noisy = read(SOURCE / "noisy_max_seed_level.csv")
    acrobot = read(SOURCE / "acrobot_seed_level.csv")
    assert len(noisy) == 30 and len(acrobot) == 6

    selected = [
        {"environment": "NoisyMax", "source_directory": r["source_directory"]}
        for r in noisy
    ] + [
        {"environment": "Acrobot", "source_directory": r["source_directory"]}
        for r in acrobot
    ]
    assert len({r["source_directory"] for r in selected}) == 36
    assert all(Path(r["source_directory"]).is_dir() for r in selected)

    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    Path("results/analysis").mkdir(parents=True, exist_ok=True)

    write(Path("results/analysis/selected_runs.csv"), selected)
    write(
        Path("results/analysis/seed_summary.csv"),
        [{"environment": "NoisyMax", **r} for r in noisy]
        + [{"environment": "Acrobot", **r} for r in acrobot],
    )

    write(TABLES / "protocol.csv", [
        {"study": "NoisyMax matrix", "seeds": "0,1,2",
         "budget": 50000, "settings": "M=2,10,50; C=320; epsilon=0.10"},
        {"study": "Target-copy contrast", "seeds": "0,1,2",
         "budget": 50000, "settings": "M=10; C=320 or 1000; epsilon=0.10"},
        {"study": "Exploration contrast", "seeds": "0,1,2",
         "budget": 50000, "settings": "M=10; C=320; epsilon=0.10 or 0.20"},
        {
            "study": "Acrobot",
            "seeds": "0,1,2",
            "budget": 200000,
            "settings": (
                "epsilon=0.10; collector exploration enabled; "
                "100 fresh final-evaluation episodes"
            ),
        },
    ])

    results = [
        {
            "study": f"NoisyMax M={r['M']}, C={r['C']}, epsilon={r['epsilon']}",
            "algorithm": r["algorithm"],
            "measure": "late risky-value signed error",
            "mean": r["mean_late_signed_error"],
            "sample_sd": r["sample_sd"],
        }
        for r in read(SOURCE / "table_noisy_max.csv")
    ]
    results += [
        {
            "study": "Acrobot",
            "algorithm": r["algorithm"],
            "measure": "final 100-episode return",
            "mean": r["mean_final_return"],
            "sample_sd": r["sample_sd"],
        }
        for r in read(SOURCE / "table_acrobot.csv")
    ]
    write(TABLES / "main_results.csv", results)

    for name, filename in IMAGES.items():
        original = SOURCE / "figures" / filename
        assert original.is_file() and original.stat().st_size > 0
        shutil.copy2(original, FIGURES / f"{name}.png")
        with Image.open(original) as image:
            image.convert("RGB").save(
                FIGURES / f"{name}.pdf", "PDF", resolution=200
            )

    (REPORT / "captions.md").write_text(
        "Figure 1. Synthetic maximization bias.\n\n"
        "Figure 2. NoisyMax value error and behavior.\n\n"
        "Figure 3. Corrected Acrobot learning curves: mean across paired "
        "seeds 0, 1, and 2 with epsilon=0.10.\n\n"
        "Figure 4. Target-copy and exploration contrasts.\n",
        encoding="utf-8",
    )
    print("Exported 36 runs, two main tables, and four PNG/PDF pairs.")


if __name__ == "__main__":
    main()
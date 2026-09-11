"""Utilities for reproducible experiment saving."""

import csv

import torch

import json
from datetime import datetime
from pathlib import Path


def create_run_directory(
    root: str = "results/runs",
) -> Path:
    """Create a unique run directory without overwriting."""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = Path(root) / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    return run_directory


def save_json(
    path: Path,
    contents: dict,
) -> None:
    """Save a dictionary as readable JSON."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(contents, file, indent=2)

def save_csv(path: Path, rows: list[dict]) -> None:
    """Save records with column headings."""

    if not rows:
        raise ValueError("CSV rows cannot be empty.")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    network,
    model_information: dict,
) -> None:
    """Save model parameters and reconstruction information."""

    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "model_information": model_information,
        },
        path,
    )
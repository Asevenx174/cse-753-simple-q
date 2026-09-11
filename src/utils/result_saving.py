"""Utilities for reproducible experiment saving."""

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
"""Append-only experiment log. Never hand-edit the CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.paths import repo_root

LOG_COLUMNS = [
    "timestamp_utc",
    "experiment_id",
    "seed",
    "config_path",
    "config_hash",
    "git_commit",
    "mae",
    "rmse",
    "smape",
    "wape",
    "mase",
    "n_households",
    "train_seconds",
    "infer_seconds",
    "n_params",
    "peak_gpu_mb",
    "notes",
]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def config_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def append_result(row: dict, log_path: Path | None = None) -> None:
    log_path = log_path or (repo_root() / "experiments" / "results" / "experiment_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {k: row.get(k, "") for k in LOG_COLUMNS}
    record["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    record["git_commit"] = _git_commit()
    new_file = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)

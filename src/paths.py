"""Resolve repo paths and the REFIT raw directory (local or Kaggle)."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_kaggle() -> bool:
    return Path("/kaggle/input").exists() or bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE"))


def resolve_raw_dir(cfg: dict) -> Path:
    """Prefer an explicit env override, then Kaggle, then the local data/raw folder.

    The Kaggle mount name is not guaranteed. We accept the configured directory
    only if it actually contains House_*.csv or CLEAN_House*.csv; otherwise we
    search one level down so a nested zip extract still works.
    """
    env = os.environ.get("REFIT_RAW_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    paths = cfg["paths"]
    if is_kaggle():
        candidates.append(Path(paths["kaggle_raw_dir"]))
    candidates.append(repo_root() / paths["raw_dir"])

    for cand in candidates:
        hit = _dir_with_house_files(cand)
        if hit is not None:
            return hit
    return candidates[-1]


def _has_house_csvs(path: Path) -> bool:
    return any(path.glob("House_*.csv")) or any(path.glob("CLEAN_House*.csv"))


def _dir_with_house_files(path: Path) -> Path | None:
    if not path.exists():
        return None
    if _has_house_csvs(path):
        return path
    for child in path.iterdir():
        if child.is_dir() and _has_house_csvs(child):
            return child
    return None


def processed_dir(cfg: dict) -> Path:
    p = repo_root() / cfg["paths"]["processed_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def weather_dir(cfg: dict) -> Path:
    p = repo_root() / cfg["paths"]["weather_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(cfg: dict) -> Path:
    p = repo_root() / cfg["paths"]["reports_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def figures_dir(cfg: dict) -> Path:
    p = repo_root() / cfg["paths"]["figures_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p

"""YAML config loading. Experiments are files, not CLI flags."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.paths import repo_root


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = repo_root() / path
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_data_config() -> dict:
    return load_yaml("configs/data.yaml")


def load_weather_config() -> dict:
    return load_yaml("configs/weather.yaml")


def load_experiment(path: str | Path) -> dict:
    exp = load_yaml(path)
    if "model_config" in exp and exp["model_config"]:
        exp["model"] = {**load_yaml(exp["model_config"]), "kind": exp.get("model")}
    return exp

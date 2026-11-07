from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_project_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def get_model_path(config_path: Path | None = None) -> Path:
    override = os.getenv("MUSHROOM_MODEL_PATH")
    if override:
        return resolve_project_path(override)
    config = load_project_config(config_path)
    try:
        return resolve_project_path(config["paths"]["model"])
    except (KeyError, TypeError) as error:
        raise ValueError("Configuration must define paths.model.") from error


def get_monitoring_log_path(config_path: Path | None = None) -> Path:
    config = load_project_config(config_path)
    try:
        configured_path = config["api"]["monitoring_log"]
    except (KeyError, TypeError) as error:
        raise ValueError("Configuration must define api.monitoring_log.") from error
    override = os.getenv("MUSHROOM_MONITORING_LOG")
    return resolve_project_path(override or configured_path)

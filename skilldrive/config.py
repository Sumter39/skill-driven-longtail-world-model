"""Small path-configuration loader with local and environment overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PATH_KEYS = ("data_root", "cache_root", "output_root")
ENV_KEYS = {key: f"SKILLDRIVE_{key.upper()}" for key in PATH_KEYS}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    unknown = set(data) - set(PATH_KEYS)
    if unknown:
        raise ValueError(f"unknown path keys in {path}: {sorted(unknown)}")
    return data


def load_paths(
    example_path: str | Path = "configs/paths.example.yaml",
    local_path: str | Path = "configs/paths.local.yaml",
) -> dict[str, str]:
    """Load defaults, optional local values, then environment overrides."""
    values = _read_yaml(Path(example_path))
    local = Path(local_path)
    if local.exists():
        values.update(_read_yaml(local))
    for key, environment_name in ENV_KEYS.items():
        if environment_name in os.environ:
            values[key] = os.environ[environment_name]
    missing = [key for key in PATH_KEYS if not isinstance(values.get(key), str) or not values[key]]
    if missing:
        raise ValueError(f"missing non-empty path values: {missing}")
    return {key: values[key] for key in PATH_KEYS}

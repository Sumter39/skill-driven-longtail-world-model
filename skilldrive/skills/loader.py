"""Load executable skill specifications from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skilldrive.schemas import SkillSpec


REQUIRED_FIELDS = {
    "skill_id",
    "family",
    "implemented",
    "trigger",
    "actors",
    "parameters",
    "constraints",
    "risk_definition",
    "expected_behavior",
    "output_labels",
}


def validate_skill_dict(data: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(data)
    unknown = set(data) - REQUIRED_FIELDS
    if missing:
        raise ValueError(f"missing skill fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown skill fields: {sorted(unknown)}")
    if not isinstance(data["skill_id"], str) or not data["skill_id"]:
        raise ValueError("skill_id must be a non-empty string")
    if not isinstance(data["implemented"], bool):
        raise ValueError("implemented must be a boolean")
    for name in ("trigger", "actors", "parameters", "constraints", "risk_definition"):
        if not isinstance(data[name], dict) or not data[name]:
            raise ValueError(f"{name} must be a non-empty mapping")
    for name in ("expected_behavior", "output_labels"):
        if not isinstance(data[name], list) or not data[name]:
            raise ValueError(f"{name} must be a non-empty list")


def load_skill(path: str | Path) -> SkillSpec:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("skill YAML must contain one mapping")
    validate_skill_dict(data)
    return SkillSpec.from_dict(data)

"""Load executable skill specifications from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from skilldrive.schemas import SkillSpec


REQUIRED_FIELDS = {
    "skill_id",
    "name_zh",
    "family",
    "definition",
    "source",
    "data_support",
    "seed_requirements",
    "trigger",
    "actors",
    "parameters",
    "generation_operators",
    "constraints",
    "risk_definition",
    "expected_behavior",
    "validation_metrics",
    "known_limitations",
    "output_labels",
}

ALLOWED_SOURCES = {
    "course_example",
    "traffic_rule",
    "safety_metric",
    "literature",
    "train_pattern",
}

ALLOWED_FEASIBILITY = {"A", "B"}

ALLOWED_THRESHOLD_SOURCES = {"semantic", "train_statistics", "reference"}


def validate_skill_dict(data: dict[str, Any]) -> None:
    missing = REQUIRED_FIELDS - set(data)
    unknown = set(data) - REQUIRED_FIELDS
    if missing:
        raise ValueError(f"missing skill fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown skill fields: {sorted(unknown)}")
    for name in ("skill_id", "name_zh", "family", "definition"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name in (
        "data_support",
        "seed_requirements",
        "trigger",
        "actors",
        "parameters",
        "constraints",
        "risk_definition",
    ):
        if not isinstance(data[name], dict) or not data[name]:
            raise ValueError(f"{name} must be a non-empty mapping")
    for name in (
        "source",
        "generation_operators",
        "expected_behavior",
        "validation_metrics",
        "known_limitations",
        "output_labels",
    ):
        if not isinstance(data[name], list) or not data[name]:
            raise ValueError(f"{name} must be a non-empty list")
    unknown_sources = set(data["source"]) - ALLOWED_SOURCES
    if unknown_sources:
        raise ValueError(f"unknown skill sources: {sorted(unknown_sources)}")
    feasibility = data["data_support"].get("feasibility")
    if feasibility not in ALLOWED_FEASIBILITY:
        raise ValueError("data_support.feasibility must be A or B")
    if data["trigger"].get("threshold_source") not in ALLOWED_THRESHOLD_SOURCES:
        raise ValueError("trigger.threshold_source must identify the threshold origin")
    if data["seed_requirements"].get("threshold_source") not in ALLOWED_THRESHOLD_SOURCES:
        raise ValueError("seed_requirements.threshold_source must identify the threshold origin")
    if data["risk_definition"].get("source") not in ALLOWED_THRESHOLD_SOURCES:
        raise ValueError("risk_definition.source must identify the threshold origin")
    for parameter_name, parameter in data["parameters"].items():
        if not isinstance(parameter, dict):
            raise ValueError(f"parameter {parameter_name} must be a mapping")
        if parameter.get("source") not in ALLOWED_THRESHOLD_SOURCES:
            raise ValueError(f"parameter {parameter_name} must identify its source")
        if "range" in parameter:
            value_range = parameter["range"]
            if (
                not isinstance(value_range, list)
                or len(value_range) != 2
                or value_range[0] > value_range[1]
            ):
                raise ValueError(f"parameter {parameter_name} has an invalid range")
        elif "choices" in parameter:
            if not isinstance(parameter["choices"], list) or not parameter["choices"]:
                raise ValueError(f"parameter {parameter_name} choices must be non-empty")
        else:
            raise ValueError(f"parameter {parameter_name} needs range or choices")


def load_skill(path: str | Path) -> SkillSpec:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("skill YAML must contain one mapping")
    validate_skill_dict(data)
    return SkillSpec.from_dict(data)

"""Order-independent deterministic sampling from ``SkillSpec.parameters``."""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Mapping

from skilldrive.schemas import SkillSpec


def _parameter_rng(global_seed: int, skill_id: str, sample_key: str, name: str) -> random.Random:
    payload = json.dumps(
        [global_seed, skill_id, sample_key, name],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(payload).digest(), "big"))


def _number(value: Any, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return value


def _choice(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"{label} choices must be JSON scalar values")


def _parameter_kind(name: str, spec: Any) -> tuple[str, list[Any]]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"parameter {name} must be a mapping")
    has_range = "range" in spec
    has_choices = "choices" in spec
    required = {"source", "range" if has_range else "choices"}
    if has_range == has_choices or set(spec) != required:
        raise ValueError(f"parameter {name} must contain source and exactly one of range or choices")
    if has_range:
        bounds = spec["range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"parameter {name} range must contain two numbers")
        low = _number(bounds[0], f"parameter {name} lower bound")
        high = _number(bounds[1], f"parameter {name} upper bound")
        if low > high:
            raise ValueError(f"parameter {name} range must be ordered")
        return "range", [low, high]
    choices = spec["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"parameter {name} choices must be a non-empty list")
    return "choices", [_choice(value, f"parameter {name}") for value in choices]


def validate_sampled_parameters(skill: SkillSpec, values: Mapping[str, Any]) -> None:
    """Validate exact names and sampled values against one skill specification."""

    if not isinstance(values, Mapping):
        raise ValueError("sampled parameters must be a mapping")
    expected = set(skill.parameters)
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"sampled parameter names differ: missing={missing}, unknown={unknown}")
    for name, spec in skill.parameters.items():
        kind, allowed = _parameter_kind(name, spec)
        value = values[name]
        if kind == "range":
            number = _number(value, f"sampled parameter {name}")
            if not allowed[0] <= number <= allowed[1]:
                raise ValueError(f"sampled parameter {name} is outside its range")
        elif value not in allowed:
            raise ValueError(f"sampled parameter {name} is not an allowed choice")


def sample_skill_parameters(
    skill: SkillSpec,
    *,
    global_seed: int,
    sample_key: str,
) -> dict[str, Any]:
    """Sample parameters reproducibly, independently of processing order."""

    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise ValueError("global_seed must be an integer")
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("sample_key must be a non-empty string")
    if not skill.parameters:
        raise ValueError("skill parameters must be non-empty")

    sampled: dict[str, Any] = {}
    for name in sorted(skill.parameters):
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be non-empty strings")
        kind, values = _parameter_kind(name, skill.parameters[name])
        generator = _parameter_rng(global_seed, skill.skill_id, sample_key, name)
        if kind == "choices":
            sampled[name] = values[generator.randrange(len(values))]
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            sampled[name] = generator.randint(values[0], values[1])
        else:
            low, high = float(values[0]), float(values[1])
            sampled[name] = low if low == high else low + (high - low) * generator.random()

    validate_sampled_parameters(skill, sampled)
    return sampled

"""Lazy public exports for generated-trajectory quality checks."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "FilterCheck": ("skilldrive.filtering.contracts", "FilterCheck"),
    "FilterDecision": ("skilldrive.filtering.contracts", "FilterDecision"),
    "FilterRejection": ("skilldrive.filtering.contracts", "FilterRejection"),
    "FilterStage": ("skilldrive.filtering.contracts", "FilterStage"),
    "FutureKinematics": ("skilldrive.filtering.common", "FutureKinematics"),
    "KinematicLimits": ("skilldrive.filtering.common", "KinematicLimits"),
    "ProxyCollisionContact": (
        "skilldrive.filtering.collision",
        "ProxyCollisionContact",
    ),
    "ProxyCollisionReport": (
        "skilldrive.filtering.collision",
        "ProxyCollisionReport",
    ),
    "check_kinematics": ("skilldrive.filtering.common", "check_kinematics"),
    "check_proxy_collisions": (
        "skilldrive.filtering.collision",
        "check_proxy_collisions",
    ),
    "check_schema_and_finite": (
        "skilldrive.filtering.common",
        "check_schema_and_finite",
    ),
    "derive_future_kinematics": (
        "skilldrive.filtering.common",
        "derive_future_kinematics",
    ),
    "detect_synchronized_proxy_collisions": (
        "skilldrive.filtering.collision",
        "detect_synchronized_proxy_collisions",
    ),
    "oriented_boxes_overlap": (
        "skilldrive.filtering.collision",
        "oriented_boxes_overlap",
    ),
    "validate_observed_skill": (
        "skilldrive.filtering.observed",
        "validate_observed_skill",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

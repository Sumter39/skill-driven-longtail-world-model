"""Stable stages and rejection codes shared by counterfactual filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from skilldrive.generation.contracts import FilterDecision, FilterRejection


class FilterStage(str, Enum):
    SCHEMA_FINITE = "schema_finite"
    HISTORY_INVARIANTS = "history_invariants"
    KINEMATICS = "kinematics"
    MAP = "map"
    COLLISION = "collision"
    TARGET_RISK = "target_risk"
    SKILL_TRIGGER = "skill_trigger"
    PARAMETER_REALIZATION = "parameter_realization"
    DIVERSITY = "diversity"


@dataclass(frozen=True)
class FilterCheck:
    """One independently testable filter stage result."""

    stage: FilterStage
    rejection_reasons: tuple[FilterRejection, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stage, FilterStage):
            raise TypeError("stage must be a FilterStage")
        reasons = tuple(self.rejection_reasons)
        if any(not isinstance(reason, FilterRejection) for reason in reasons):
            raise TypeError("rejection_reasons must contain FilterRejection values")
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @property
    def passed(self) -> bool:
        return not self.rejection_reasons

    @property
    def rejection_values(self) -> tuple[str, ...]:
        return tuple(reason.value for reason in self.rejection_reasons)


__all__ = [
    "FilterCheck",
    "FilterDecision",
    "FilterRejection",
    "FilterStage",
]

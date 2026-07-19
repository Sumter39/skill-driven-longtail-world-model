"""Small, serialization-friendly schemas for scenes and skill metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    return array


def _vector(value: Any, name: str, length: int, dtype: Any = np.float64) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


@dataclass
class AgentTrack:
    track_id: str
    object_type: str
    positions: np.ndarray
    velocities: np.ndarray
    headings: np.ndarray
    observed_mask: np.ndarray
    is_focal: bool = False

    def __post_init__(self) -> None:
        self.positions = _points(self.positions, "positions")
        length = len(self.positions)
        self.velocities = _points(self.velocities, "velocities")
        if len(self.velocities) != length:
            raise ValueError("velocities and positions must have the same length")
        self.headings = _vector(self.headings, "headings", length)
        self.observed_mask = _vector(self.observed_mask, "observed_mask", length, bool)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "object_type": self.object_type,
            "positions": self.positions.tolist(),
            "velocities": self.velocities.tolist(),
            "headings": self.headings.tolist(),
            "observed_mask": self.observed_mask.tolist(),
            "is_focal": self.is_focal,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentTrack":
        return cls(**data)


@dataclass
class MapPolyline:
    polyline_id: str
    polyline_type: str
    points: np.ndarray
    direction: str = "unknown"
    is_intersection: bool = False

    def __post_init__(self) -> None:
        self.points = _points(self.points, "points")

    def to_dict(self) -> dict[str, Any]:
        return {
            "polyline_id": self.polyline_id,
            "polyline_type": self.polyline_type,
            "points": self.points.tolist(),
            "direction": self.direction,
            "is_intersection": self.is_intersection,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MapPolyline":
        return cls(**data)


@dataclass
class Scenario:
    scenario_id: str
    city_name: str
    timestamps: np.ndarray
    focal_track_id: str
    agents: list[AgentTrack]
    map_polylines: list[MapPolyline]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.int64)
        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")
        if len({agent.track_id for agent in self.agents}) != len(self.agents):
            raise ValueError("agent track IDs must be unique")
        if self.focal_track_id not in {agent.track_id for agent in self.agents}:
            raise ValueError("focal_track_id must reference an agent in the scenario")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "city_name": self.city_name,
            "timestamps": self.timestamps.tolist(),
            "focal_track_id": self.focal_track_id,
            "agents": [agent.to_dict() for agent in self.agents],
            "map_polylines": [polyline.to_dict() for polyline in self.map_polylines],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        values = dict(data)
        values["agents"] = [AgentTrack.from_dict(item) for item in data["agents"]]
        values["map_polylines"] = [MapPolyline.from_dict(item) for item in data["map_polylines"]]
        return cls(**values)


@dataclass
class SkillSpec:
    skill_id: str
    family: str
    implemented: bool
    trigger: dict[str, Any]
    actors: dict[str, Any]
    parameters: dict[str, Any]
    constraints: dict[str, Any]
    risk_definition: dict[str, Any]
    expected_behavior: list[str]
    output_labels: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "family": self.family,
            "implemented": self.implemented,
            "trigger": self.trigger,
            "actors": self.actors,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "risk_definition": self.risk_definition,
            "expected_behavior": self.expected_behavior,
            "output_labels": self.output_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpec":
        return cls(**data)


@dataclass
class FilterReport:
    passed: bool
    hard_failures: list[str]
    component_scores: dict[str, float]
    total_score: float
    risk_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "hard_failures": self.hard_failures,
            "component_scores": self.component_scores,
            "total_score": self.total_score,
            "risk_metrics": self.risk_metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterReport":
        return cls(**data)

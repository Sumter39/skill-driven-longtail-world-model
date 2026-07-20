"""Executable, deterministic seed detection for implemented skill rules."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import yaml

from skilldrive.schemas import AgentTrack, MapPolyline, Scenario, SkillSpec
from skilldrive.seeds import SeedRecord, sample_skill_parameters, sort_seed_records
from skilldrive.skills.geometry import (
    TrajectoryConflict,
    extract_valid_trajectory,
    find_trajectory_conflict,
    heading_difference,
    minimum_trajectory_distance,
    point_to_polyline_projection,
    time_headway,
    time_to_collision,
    trajectory_acceleration,
)
from skilldrive.skills.registry import get_skill_detection_rule


_VEHICLE_TYPES = {"vehicle", "bus"}
_VRU_TYPES = {"pedestrian", "cyclist", "motorcyclist"}
_STATIC_TYPES = {"static", "construction"}
_THRESHOLD_SOURCES = {"semantic", "train_statistics", "reference"}
_REQUIRED_THRESHOLDS = {
    "conflict_distance_m",
    "lane_heading_tolerance_deg",
    "lane_match_distance_m",
    "maximum_actor_distance_m",
    "risk_time_horizon_s",
    "same_lane_lateral_tolerance_m",
}


@dataclass(frozen=True)
class DetectionConfig:
    global_seed: int
    max_candidates_per_skill_per_scenario: int
    thresholds: dict[str, float]

    def threshold(self, name: str) -> float:
        try:
            return self.thresholds[name]
        except KeyError as exc:
            raise KeyError(f"missing detection threshold: {name}") from exc


@dataclass(frozen=True)
class ActorState:
    agent: AgentTrack
    reference_index: int
    position: np.ndarray
    velocity: np.ndarray
    speed_mps: float
    heading_rad: float

    @property
    def track_id(self) -> str:
        return self.agent.track_id

    @property
    def object_type(self) -> str:
        return self.agent.object_type.lower()


@dataclass(frozen=True)
class LaneMatch:
    lane: MapPolyline
    distance_m: float
    lateral_m: float
    heading_error_rad: float
    arc_length_m: float


@dataclass(frozen=True)
class RuleMatch:
    initiator: ActorState
    responder: ActorState
    additional_actors: tuple[ActorState, ...]
    trigger_score: float
    risk_metric: str
    risk_value: float
    evidence: dict[str, Any]


@dataclass
class DetectionRun:
    records: list[SeedRecord]
    rejection_counts: Counter[str]


def load_detection_config(path: str | Path) -> DetectionConfig:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("seed detection config must contain one mapping")
    if data.get("version") != 1:
        raise ValueError("seed detection config version must be 1")
    expected_fields = {
        "version",
        "global_seed",
        "max_candidates_per_skill_per_scenario",
        "thresholds",
    }
    if set(data) != expected_fields:
        raise ValueError(
            "seed detection config fields differ: "
            f"missing={sorted(expected_fields - set(data))}, "
            f"unknown={sorted(set(data) - expected_fields)}"
        )
    for name in (
        "global_seed",
        "max_candidates_per_skill_per_scenario",
        "thresholds",
    ):
        if name not in data:
            raise ValueError(f"missing seed detection config field: {name}")
    for name in (
        "global_seed",
        "max_candidates_per_skill_per_scenario",
    ):
        if isinstance(data[name], bool) or not isinstance(data[name], int) or data[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(data["thresholds"], dict) or not data["thresholds"]:
        raise ValueError("thresholds must be a non-empty mapping")
    threshold_names = set(data["thresholds"])
    if threshold_names != _REQUIRED_THRESHOLDS:
        raise ValueError(
            "seed detection thresholds differ: "
            f"missing={sorted(_REQUIRED_THRESHOLDS - threshold_names)}, "
            f"unknown={sorted(threshold_names - _REQUIRED_THRESHOLDS)}"
        )
    thresholds: dict[str, float] = {}
    for name, item in data["thresholds"].items():
        if not isinstance(item, dict) or set(item) != {"value", "source"}:
            raise ValueError(f"threshold {name} must contain value and source")
        value = item["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"threshold {name} value must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"threshold {name} value must be finite and nonnegative")
        if item["source"] not in _THRESHOLD_SOURCES:
            raise ValueError(f"threshold {name} has an unknown source")
        thresholds[name] = value
    return DetectionConfig(
        global_seed=data["global_seed"],
        max_candidates_per_skill_per_scenario=data[
            "max_candidates_per_skill_per_scenario"
        ],
        thresholds=thresholds,
    )


def _finite_state(agent: AgentTrack) -> ActorState | None:
    finite = (
        np.isfinite(agent.positions).all(axis=1)
        & np.isfinite(agent.velocities).all(axis=1)
        & np.isfinite(agent.headings)
    )
    observed = finite & agent.observed_mask
    indices = np.flatnonzero(observed)
    if not len(indices):
        return None
    index = int(indices[-1])
    velocity = agent.velocities[index].astype(np.float64, copy=True)
    return ActorState(
        agent=agent,
        reference_index=index,
        position=agent.positions[index].astype(np.float64, copy=True),
        velocity=velocity,
        speed_mps=float(np.linalg.norm(velocity)),
        heading_rad=float(agent.headings[index]),
    )


def _future_positions(state: ActorState) -> tuple[np.ndarray, np.ndarray]:
    positions = state.agent.positions[state.reference_index :]
    valid = np.isfinite(positions).all(axis=1)
    return positions, valid


def _future_heading_change(state: ActorState) -> float:
    headings = state.agent.headings[state.reference_index :]
    valid = np.flatnonzero(np.isfinite(headings))
    if len(valid) < 2:
        return 0.0
    return float(heading_difference(headings[valid[-1]], headings[valid[0]]))


def _future_lateral_displacement(state: ActorState) -> float:
    positions, valid = _future_positions(state)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        return 0.0
    delta = positions[indices[-1]] - positions[indices[0]]
    left = np.array([-math.sin(state.heading_rad), math.cos(state.heading_rad)])
    return float(np.dot(delta, left))


def _minimum_future_speed(state: ActorState) -> float:
    velocities = state.agent.velocities[state.reference_index :]
    valid = np.isfinite(velocities).all(axis=1)
    if not valid.any():
        return state.speed_mps
    return float(np.min(np.linalg.norm(velocities[valid], axis=1)))


def _minimum_future_acceleration(state: ActorState) -> float:
    positions = state.agent.positions[state.reference_index :]
    values = trajectory_acceleration(positions, sample_period_s=0.1)
    finite = values[np.isfinite(values)]
    return 0.0 if not len(finite) else float(np.min(finite))


def _relative_coordinates(other: ActorState, reference: ActorState) -> tuple[float, float]:
    delta = other.position - reference.position
    forward_axis = np.array([math.cos(reference.heading_rad), math.sin(reference.heading_rad)])
    left_axis = np.array([-forward_axis[1], forward_axis[0]])
    return float(np.dot(delta, forward_axis)), float(np.dot(delta, left_axis))


def _distance(first: ActorState, second: ActorState) -> float:
    return float(np.linalg.norm(first.position - second.position))


def _score_small(value: float, preferred_max: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(1.0 - value / max(preferred_max, 1e-6), 0.0, 1.0))


def _score_large(value: float, preferred_min: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value / max(preferred_min, 1e-6), 0.0, 1.0))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if len(polygon) < 4 or not np.isfinite(point).all():
        return False
    vertices = polygon[np.isfinite(polygon).all(axis=1)]
    if len(vertices) < 4:
        return False
    x, y = point
    inside = False
    previous = vertices[-1]
    for current in vertices:
        if (current[1] > y) != (previous[1] > y):
            crossing_x = (previous[0] - current[0]) * (y - current[1]) / (
                previous[1] - current[1]
            ) + current[0]
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


class ScenarioDetectionContext:
    def __init__(self, scenario: Scenario, config: DetectionConfig):
        self.scenario = scenario
        self.config = config
        states = [state for agent in scenario.agents if (state := _finite_state(agent))]
        focal = next((state for state in states if state.track_id == scenario.focal_track_id), None)
        if focal is None:
            raise ValueError("scenario focal agent has no finite state")
        states.sort(key=lambda state: (_distance(state, focal), state.track_id))
        dynamic = [state for state in states if state.object_type not in _STATIC_TYPES]
        static = [state for state in states if state.object_type in _STATIC_TYPES]
        self.states = dynamic + static
        self.state_by_id = {state.track_id: state for state in self.states}
        self.lanes = [
            item for item in scenario.map_polylines if item.polyline_type == "lane_centerline"
        ]
        self.crosswalks = [
            item for item in scenario.map_polylines if item.polyline_type == "pedestrian_crossing"
        ]
        self.drivable_areas = [
            item for item in scenario.map_polylines if item.polyline_type == "drivable_area"
        ]
        self._drivable_area_bounds: list[
            tuple[MapPolyline, float, float, float, float]
        ] = []
        for area in self.drivable_areas:
            finite_points = area.points[np.isfinite(area.points).all(axis=1)]
            if len(finite_points) < 4:
                continue
            self._drivable_area_bounds.append(
                (
                    area,
                    float(np.min(finite_points[:, 0])),
                    float(np.min(finite_points[:, 1])),
                    float(np.max(finite_points[:, 0])),
                    float(np.max(finite_points[:, 1])),
                )
            )
        self.intersection_lanes = [lane for lane in self.lanes if lane.is_intersection]
        self._lane_bounds = [
            (
                lane,
                float(np.nanmin(lane.points[:, 0])),
                float(np.nanmin(lane.points[:, 1])),
                float(np.nanmax(lane.points[:, 0])),
                float(np.nanmax(lane.points[:, 1])),
            )
            for lane in self.lanes
            if len(lane.points) and np.isfinite(lane.points).all()
        ]
        self.lane_by_id = {lane.lane_id: lane for lane in self.lanes if lane.lane_id}
        self.timestamps_s = _timestamps_seconds(scenario)
        self._future_conflict_cache: dict[tuple[str, str], TrajectoryConflict | None] = {}
        self._future_distance_cache: dict[tuple[str, str], float] = {}
        self._nearest_pairs_cache: dict[
            tuple[tuple[str, ...], tuple[str, ...]],
            tuple[tuple[ActorState, ActorState], ...],
        ] = {}
        self._intersection_match_cache: dict[
            tuple[str, float], tuple[MapPolyline, float] | None
        ] = {}
        self._crosswalk_distance_cache: dict[str, float] = {}
        self._drivable_inside_cache: dict[str, bool] = {}
        self._future_drivable_entry_cache: dict[str, bool] = {}
        self._drivable_boundary_cache: dict[str, float] = {}
        lane_matches = {
            state.track_id: self._match_lanes(state)
            for state in self.states
            if state.object_type not in _STATIC_TYPES | {"pedestrian"}
        }
        self._lane_matches = {
            track_id: matches[0] for track_id, matches in lane_matches.items()
        }
        self._geometry_lane_matches = {
            track_id: matches[1] for track_id, matches in lane_matches.items()
        }

    @property
    def vehicles(self) -> list[ActorState]:
        return [state for state in self.states if state.object_type in _VEHICLE_TYPES]

    @property
    def pedestrians(self) -> list[ActorState]:
        return [state for state in self.states if state.object_type == "pedestrian"]

    @property
    def cyclists(self) -> list[ActorState]:
        return [
            state
            for state in self.states
            if state.object_type in {"cyclist", "motorcyclist"}
        ]

    @property
    def static_actors(self) -> list[ActorState]:
        return [state for state in self.states if state.object_type in _STATIC_TYPES]

    def lane_match(
        self,
        state: ActorState,
        *,
        allow_opposite: bool = False,
    ) -> LaneMatch | None:
        matches = self._geometry_lane_matches if allow_opposite else self._lane_matches
        return matches.get(state.track_id)

    def actors_of_types(self, object_types: Iterable[str]) -> list[ActorState]:
        allowed = {str(value).lower() for value in object_types}
        return [state for state in self.states if state.object_type in allowed]

    def _match_lanes(self, state: ActorState) -> tuple[LaneMatch | None, LaneMatch | None]:
        allowed_directions = {"bike"} if state.object_type == "cyclist" else {"vehicle", "bus"}
        candidates: list[LaneMatch] = []
        maximum_distance = self.config.threshold("lane_match_distance_m")
        x, y = state.position
        for lane, minimum_x, minimum_y, maximum_x, maximum_y in self._lane_bounds:
            if lane.direction not in allowed_directions and lane.direction != "unknown":
                continue
            delta_x = max(minimum_x - x, 0.0, x - maximum_x)
            delta_y = max(minimum_y - y, 0.0, y - maximum_y)
            if math.hypot(delta_x, delta_y) > maximum_distance:
                continue
            projection = point_to_polyline_projection(state.position, lane.points)
            error = abs(float(heading_difference(state.heading_rad, projection.heading_rad)))
            candidates.append(
                LaneMatch(
                    lane=lane,
                    distance_m=projection.distance_m,
                    lateral_m=projection.signed_lateral_distance_m,
                    heading_error_rad=error,
                    arc_length_m=projection.arc_length_m,
                )
            )
        if not candidates:
            return None, None
        geometry = min(
            candidates,
            key=lambda match: (
                match.distance_m,
                match.heading_error_rad,
                match.lane.polyline_id,
            ),
        )
        if geometry.distance_m > maximum_distance:
            geometry = None
        heading_tolerance = math.radians(
            self.config.threshold("lane_heading_tolerance_deg")
        )
        aligned_candidates = [
            match
            for match in candidates
            if match.distance_m <= maximum_distance
            and match.heading_error_rad <= heading_tolerance
        ]
        aligned = (
            None
            if not aligned_candidates
            else min(
                aligned_candidates,
                key=lambda match: (
                    match.distance_m,
                    match.heading_error_rad,
                    match.lane.polyline_id,
                ),
            )
        )
        return aligned, geometry

    def lanes_same_or_successor(self, first: ActorState, second: ActorState) -> bool:
        first_match, second_match = self.lane_match(first), self.lane_match(second)
        if first_match is None or second_match is None:
            _, lateral = _relative_coordinates(first, second)
            return abs(lateral) <= self.config.threshold("same_lane_lateral_tolerance_m")
        first_id, second_id = first_match.lane.lane_id, second_match.lane.lane_id
        return bool(
            first_id == second_id
            or first_id in second_match.lane.successor_ids
            or second_id in first_match.lane.successor_ids
            or first_id in second_match.lane.predecessor_ids
            or second_id in first_match.lane.predecessor_ids
        )

    def lanes_adjacent(self, first: ActorState, second: ActorState) -> bool:
        first_match, second_match = self.lane_match(first), self.lane_match(second)
        if first_match is None or second_match is None:
            _, lateral = _relative_coordinates(first, second)
            return self.config.threshold("same_lane_lateral_tolerance_m") < abs(lateral) < 10.0
        first_lane, second_lane = first_match.lane, second_match.lane
        return bool(
            second_lane.lane_id
            in {first_lane.left_neighbor_id, first_lane.right_neighbor_id}
            or first_lane.lane_id
            in {second_lane.left_neighbor_id, second_lane.right_neighbor_id}
        )

    def lanes_converge(self, first: ActorState, second: ActorState) -> bool:
        first_match, second_match = self.lane_match(first), self.lane_match(second)
        if first_match is None or second_match is None:
            return False
        first_successors = set(first_match.lane.successor_ids)
        second_successors = set(second_match.lane.successor_ids)
        return bool(
            first_successors & second_successors
            or first_match.lane.lane_id in second_successors
            or second_match.lane.lane_id in first_successors
        )

    def lane_is_intersection(self, state: ActorState) -> bool:
        match = self.lane_match(state)
        return bool(match and match.lane.is_intersection)

    def distance_to_intersection(
        self,
        state: ActorState,
        *,
        maximum_distance_m: float | None = None,
    ) -> float:
        match = self.nearest_intersection_lane(
            state,
            maximum_distance_m=maximum_distance_m,
        )
        return float("inf") if match is None else match[1]

    def nearest_intersection_lane(
        self,
        state: ActorState,
        *,
        maximum_distance_m: float | None = None,
    ) -> tuple[MapPolyline, float] | None:
        global_radius = self.config.threshold("maximum_actor_distance_m")
        search_radius = (
            global_radius
            if maximum_distance_m is None
            else min(global_radius, maximum_distance_m)
        )
        cache_key = (state.track_id, search_radius)
        if cache_key not in self._intersection_match_cache:
            if not self.intersection_lanes:
                self._intersection_match_cache[cache_key] = None
            else:
                x, y = state.position
                candidates = [
                    lane
                    for lane, minimum_x, minimum_y, maximum_x, maximum_y in self._lane_bounds
                    if lane.is_intersection
                    and math.hypot(
                        max(minimum_x - x, 0.0, x - maximum_x),
                        max(minimum_y - y, 0.0, y - maximum_y),
                    )
                    <= search_radius
                ]
                if not candidates:
                    self._intersection_match_cache[cache_key] = None
                else:
                    lane, distance = min(
                        (
                            (
                                lane,
                                point_to_polyline_projection(
                                    state.position,
                                    lane.points,
                                ).distance_m,
                            )
                            for lane in candidates
                        ),
                        key=lambda item: (item[1], item[0].polyline_id),
                    )
                    self._intersection_match_cache[cache_key] = (lane, distance)
        return self._intersection_match_cache[cache_key]

    def point_near_intersection(self, point: np.ndarray, tolerance_m: float) -> bool:
        x, y = point
        for lane, minimum_x, minimum_y, maximum_x, maximum_y in self._lane_bounds:
            if not lane.is_intersection:
                continue
            delta_x = max(minimum_x - x, 0.0, x - maximum_x)
            delta_y = max(minimum_y - y, 0.0, y - maximum_y)
            if math.hypot(delta_x, delta_y) > tolerance_m:
                continue
            if point_to_polyline_projection(point, lane.points).distance_m <= tolerance_m:
                return True
        return False

    def lane_diverges(self, state: ActorState) -> bool:
        match = self.lane_match(state)
        if match is None or not match.lane.lane_id:
            return False
        valid_successors = {
            successor_id
            for successor_id in match.lane.successor_ids
            if (
                (successor := self.lane_by_id.get(successor_id)) is not None
                and match.lane.lane_id in successor.predecessor_ids
            )
        }
        return len(valid_successors) >= 2

    def distance_to_lane_end(self, state: ActorState) -> float:
        match = self.lane_match(state)
        if match is None or len(match.lane.points) < 2:
            return float("inf")
        total = float(np.linalg.norm(np.diff(match.lane.points, axis=0), axis=1).sum())
        return max(0.0, total - match.arc_length_m)

    def lane_has_neighbor(self, state: ActorState) -> bool:
        match = self.lane_match(state)
        return bool(
            match
            and (match.lane.left_neighbor_id is not None or match.lane.right_neighbor_id is not None)
        )

    def convergence_point(self, first: ActorState, second: ActorState) -> np.ndarray | None:
        first_match, second_match = self.lane_match(first), self.lane_match(second)
        if first_match is None or second_match is None:
            return None
        first_lane, second_lane = first_match.lane, second_match.lane
        common = sorted(set(first_lane.successor_ids) & set(second_lane.successor_ids))
        if common and common[0] in self.lane_by_id:
            return self.lane_by_id[common[0]].points[0].copy()
        if second_lane.lane_id in first_lane.successor_ids:
            return second_lane.points[0].copy()
        if first_lane.lane_id in second_lane.successor_ids:
            return first_lane.points[0].copy()
        return None

    def distance_to_crosswalk(self, state: ActorState) -> float:
        if state.track_id not in self._crosswalk_distance_cache:
            if not self.crosswalks:
                value = float("inf")
            elif any(
                _point_in_polygon(state.position, crossing.points)
                for crossing in self.crosswalks
            ):
                value = 0.0
            else:
                value = min(
                    point_to_polyline_projection(state.position, crossing.points).distance_m
                    for crossing in self.crosswalks
                )
            self._crosswalk_distance_cache[state.track_id] = value
        return self._crosswalk_distance_cache[state.track_id]

    def point_distance_to_crosswalk(self, point: np.ndarray) -> float:
        if not self.crosswalks:
            return float("inf")
        if any(_point_in_polygon(point, crossing.points) for crossing in self.crosswalks):
            return 0.0
        return min(
            point_to_polyline_projection(point, crossing.points).distance_m
            for crossing in self.crosswalks
        )

    def inside_drivable_area(self, state: ActorState) -> bool:
        if state.track_id not in self._drivable_inside_cache:
            self._drivable_inside_cache[state.track_id] = (
                self.point_inside_drivable_area(state.position)
            )
        return self._drivable_inside_cache[state.track_id]

    def point_inside_drivable_area(self, point: np.ndarray) -> bool:
        if not np.isfinite(point).all():
            return False
        x, y = point
        return any(
            minimum_x <= x <= maximum_x
            and minimum_y <= y <= maximum_y
            and _point_in_polygon(point, area.points)
            for area, minimum_x, minimum_y, maximum_x, maximum_y in self._drivable_area_bounds
        )

    def distance_to_drivable_boundary(self, state: ActorState) -> float:
        if state.track_id not in self._drivable_boundary_cache:
            self._drivable_boundary_cache[state.track_id] = (
                float("inf")
                if not self.drivable_areas
                else min(
                    point_to_polyline_projection(state.position, area.points).distance_m
                    for area in self.drivable_areas
                )
            )
        return self._drivable_boundary_cache[state.track_id]

    def has_map_requirement(self, requirement: str) -> bool:
        if requirement == "lane_centerline":
            return bool(self.lanes)
        if requirement == "lane_successor":
            return any(lane.successor_ids for lane in self.lanes)
        if requirement == "adjacent_lane":
            return any(
                lane.left_neighbor_id is not None or lane.right_neighbor_id is not None
                for lane in self.lanes
            )
        if requirement == "converging_lane":
            successor_counts: Counter[str] = Counter(
                successor for lane in self.lanes for successor in lane.successor_ids
            )
            return any(count >= 2 for count in successor_counts.values()) or any(
                len(lane.predecessor_ids) >= 2 for lane in self.lanes
            )
        if requirement == "diverging_lane":
            return any(len(lane.successor_ids) >= 2 for lane in self.lanes)
        if requirement == "intersection_lane":
            return any(lane.is_intersection for lane in self.lanes)
        if requirement == "pedestrian_crossing":
            return bool(self.crosswalks)
        if requirement == "drivable_area":
            return bool(self.drivable_areas)
        if requirement == "bike_lane":
            return any(lane.direction == "bike" for lane in self.lanes)
        if requirement == "lane_direction":
            return any(len(lane.points) >= 2 for lane in self.lanes)
        raise ValueError(f"unknown map requirement: {requirement}")

    def future_conflict(
        self,
        first: ActorState,
        second: ActorState,
    ) -> TrajectoryConflict | None:
        key = (first.track_id, second.track_id)
        if key not in self._future_conflict_cache:
            self._future_conflict_cache[key] = find_trajectory_conflict(
                first.agent.positions,
                second.agent.positions,
                self.timestamps_s,
                self.timestamps_s,
                first_valid_mask=_future_mask(first),
                second_valid_mask=_future_mask(second),
            )
        return self._future_conflict_cache[key]

    def future_minimum_distance(self, first: ActorState, second: ActorState) -> float:
        key = tuple(sorted((first.track_id, second.track_id)))
        if key not in self._future_distance_cache:
            result = minimum_trajectory_distance(
                first.agent.positions,
                second.agent.positions,
                first_valid_mask=_future_mask(first),
                second_valid_mask=_future_mask(second),
            )
            self._future_distance_cache[key] = float("inf") if result is None else result.distance_m
        return self._future_distance_cache[key]

    def nearest_pairs(
        self,
        first_group: Sequence[ActorState],
        second_group: Sequence[ActorState],
    ) -> list[tuple[ActorState, ActorState]]:
        key = (
            tuple(state.track_id for state in first_group),
            tuple(state.track_id for state in second_group),
        )
        if key not in self._nearest_pairs_cache:
            maximum = self.config.threshold("maximum_actor_distance_m")
            decorated: list[tuple[float, str, str, ActorState, ActorState]] = []
            for first in first_group:
                for second in second_group:
                    if first.track_id == second.track_id:
                        continue
                    distance = _distance(first, second)
                    if distance <= maximum:
                        decorated.append(
                            (
                                distance,
                                first.track_id,
                                second.track_id,
                                first,
                                second,
                            )
                        )
            decorated.sort(key=lambda item: (item[0], item[1], item[2]))
            self._nearest_pairs_cache[key] = tuple(
                (item[3], item[4]) for item in decorated
            )
        return list(self._nearest_pairs_cache[key])


def _history_steps(state: ActorState) -> int:
    valid = state.agent.observed_mask & np.isfinite(state.agent.positions).all(axis=1)
    return int(valid.sum())


def _shared_history_steps(states: Sequence[ActorState]) -> int:
    if not states:
        return 0
    shared = np.ones(len(states[0].agent.positions), dtype=bool)
    for state in states:
        if len(state.agent.positions) != len(shared):
            return 0
        shared &= state.agent.observed_mask & np.isfinite(state.agent.positions).all(axis=1)
    return int(shared.sum())


def _has_track_requirement(context: ScenarioDetectionContext, requirement: str) -> bool:
    return any(state.object_type == requirement for state in context.states)


def _future_mask(state: ActorState) -> np.ndarray:
    mask = np.zeros(len(state.agent.positions), dtype=bool)
    mask[state.reference_index :] = True
    return mask


def _timestamps_seconds(scenario: Scenario) -> np.ndarray:
    values = scenario.timestamps.astype(np.float64)
    if not len(values):
        return values
    return (values - values[0]) / 1_000_000_000.0


def _future_conflict(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
) -> TrajectoryConflict | None:
    return context.future_conflict(first, second)


def _future_minimum_distance(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
) -> float:
    return context.future_minimum_distance(first, second)


def _pair_ttc(
    first: ActorState,
    second: ActorState,
    collision_radius_m: float,
) -> float:
    return time_to_collision(
        second.position - first.position,
        second.velocity - first.velocity,
        collision_radius_m=collision_radius_m,
    )


def _minimum_future_ttc(
    first: ActorState,
    second: ActorState,
    *,
    collision_radius_m: float = 0.0,
) -> float:
    start = max(first.reference_index, second.reference_index)
    values: list[float] = []
    for index in range(start, len(first.agent.positions)):
        if not (
            np.isfinite(first.agent.positions[index]).all()
            and np.isfinite(second.agent.positions[index]).all()
            and np.isfinite(first.agent.velocities[index]).all()
            and np.isfinite(second.agent.velocities[index]).all()
        ):
            continue
        value = time_to_collision(
            second.agent.positions[index] - first.agent.positions[index],
            second.agent.velocities[index] - first.agent.velocities[index],
            collision_radius_m=collision_radius_m,
        )
        if math.isfinite(value):
            values.append(value)
    return min(values) if values else float("inf")


def _ttc_within_risk_horizon(
    context: ScenarioDetectionContext,
    value: float,
) -> bool:
    return (
        math.isfinite(value)
        and 0.0 <= value <= context.config.threshold("risk_time_horizon_s")
    )


def _future_endpoint(state: ActorState) -> tuple[np.ndarray, np.ndarray, float] | None:
    valid = (
        _future_mask(state)
        & np.isfinite(state.agent.positions).all(axis=1)
        & np.isfinite(state.agent.velocities).all(axis=1)
        & np.isfinite(state.agent.headings)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    index = int(indices[-1])
    return (
        state.agent.positions[index],
        state.agent.velocities[index],
        float(state.agent.headings[index]),
    )


def _longitudinal_gap(
    context: ScenarioDetectionContext,
    leader: ActorState,
    follower: ActorState,
) -> float:
    leader_match, follower_match = context.lane_match(leader), context.lane_match(follower)
    if (
        leader_match is not None
        and follower_match is not None
        and leader_match.lane.lane_id == follower_match.lane.lane_id
    ):
        return leader_match.arc_length_m - follower_match.arc_length_m
    forward, _ = _relative_coordinates(leader, follower)
    return forward


def _signed_future_heading_change(state: ActorState) -> float:
    headings = state.agent.headings[state.reference_index :]
    valid = np.flatnonzero(np.isfinite(headings))
    if len(valid) < 2:
        return 0.0
    return float((headings[valid[-1]] - headings[valid[0]] + math.pi) % (2 * math.pi) - math.pi)


def _sample_period_s(context: ScenarioDetectionContext) -> float:
    differences = np.diff(context.timestamps_s)
    finite = differences[np.isfinite(differences) & (differences > 0)]
    return 0.1 if not len(finite) else float(np.median(finite))


def _longest_run_duration(mask: np.ndarray, sample_period_s: float) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest * sample_period_s


def _speed_below_duration(
    context: ScenarioDetectionContext,
    state: ActorState,
    maximum_speed_mps: float,
    *,
    future_only: bool = False,
) -> float:
    valid = np.isfinite(state.agent.velocities).all(axis=1)
    if future_only:
        valid &= _future_mask(state)
    speeds = np.linalg.norm(np.where(valid[:, None], state.agent.velocities, 0.0), axis=1)
    return _longest_run_duration(
        valid & (speeds <= maximum_speed_mps),
        _sample_period_s(context),
    )


def _trailing_observed_speed_below_duration(
    context: ScenarioDetectionContext,
    state: ActorState,
    maximum_speed_mps: float,
) -> float:
    """Return the continuous observed low-speed duration ending at the reference frame."""

    valid = (
        state.agent.observed_mask
        & np.isfinite(state.agent.velocities).all(axis=1)
    )
    speeds = np.linalg.norm(
        np.where(valid[:, None], state.agent.velocities, 0.0),
        axis=1,
    )
    count = 0
    for index in range(state.reference_index, -1, -1):
        if not valid[index] or speeds[index] > maximum_speed_mps:
            break
        count += 1
    return count * _sample_period_s(context)


def _moving_to_stopped_metrics(
    context: ScenarioDetectionContext,
    state: ActorState,
    *,
    minimum_prior_speed_mps: float,
    stopped_speed_mps: float,
) -> tuple[float, float, float, float] | None:
    velocities = state.agent.velocities
    positions = state.agent.positions
    valid = np.isfinite(velocities).all(axis=1) & np.isfinite(positions).all(axis=1)
    speeds = np.linalg.norm(np.where(valid[:, None], velocities, 0.0), axis=1)
    future_indices = np.flatnonzero(
        valid & _future_mask(state) & (speeds <= stopped_speed_mps)
    )
    if not len(future_indices):
        return None
    stop_index = int(future_indices[0])
    prior_valid = valid[:stop_index]
    if not prior_valid.any():
        return None
    maximum_prior_speed = float(np.max(speeds[:stop_index][prior_valid]))
    if maximum_prior_speed < minimum_prior_speed_mps:
        return None
    stopped_duration = _longest_run_duration(
        valid[stop_index:] & (speeds[stop_index:] <= stopped_speed_mps),
        _sample_period_s(context),
    )
    stop_time_s = max(0.0, float(context.timestamps_s[stop_index] - context.timestamps_s[state.reference_index]))
    leader_travel_m = float(np.linalg.norm(positions[stop_index] - state.position))
    return stopped_duration, stop_time_s, leader_travel_m, maximum_prior_speed


def _sustained_headway_duration(
    context: ScenarioDetectionContext,
    leader: ActorState,
    follower: ActorState,
    maximum_headway_s: float,
    minimum_follower_speed_mps: float,
) -> float:
    valid = (
        leader.agent.observed_mask
        & follower.agent.observed_mask
        & np.isfinite(leader.agent.positions).all(axis=1)
        & np.isfinite(follower.agent.positions).all(axis=1)
        & np.isfinite(follower.agent.velocities).all(axis=1)
        & np.isfinite(follower.agent.headings)
    )
    matches = np.zeros(len(valid), dtype=bool)
    for index in np.flatnonzero(valid):
        heading = float(follower.agent.headings[index])
        axis = np.array([math.cos(heading), math.sin(heading)])
        gap = float(np.dot(leader.agent.positions[index] - follower.agent.positions[index], axis))
        speed = float(np.linalg.norm(follower.agent.velocities[index]))
        headway = time_headway(gap, speed)
        matches[index] = speed >= minimum_follower_speed_mps and 0 < headway <= maximum_headway_s
    return _longest_run_duration(matches, _sample_period_s(context))


def _risk_target(skill: SkillSpec) -> tuple[float, float]:
    values = skill.risk_definition.get("target_range")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or isinstance(values[0], bool)
        or isinstance(values[1], bool)
        or not isinstance(values[0], (int, float))
        or not isinstance(values[1], (int, float))
    ):
        raise ValueError(f"{skill.skill_id} risk_definition.target_range must contain two numbers")
    low, high = float(values[0]), float(values[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low <= high):
        raise ValueError(f"{skill.skill_id} has an invalid risk target range")
    return low, high


def _risk_score(value: float, skill: SkillSpec) -> float:
    if not math.isfinite(value):
        return 0.0
    low, high = _risk_target(skill)
    direction = skill.risk_definition.get("direction")
    if direction not in {"lower_is_riskier", "higher_is_riskier"}:
        raise ValueError(f"{skill.skill_id} has an invalid risk direction")
    if low <= value <= high:
        if high == low:
            return 1.0
        position = (value - low) / (high - low)
        risk_position = 1.0 - position if direction == "lower_is_riskier" else position
        return 0.75 + 0.25 * risk_position
    if value < low:
        distance_score = _score_large(value, low) if low > 0 else 0.0
    else:
        distance_score = _score_small(value - high, max(high - low, 1.0))
    return 0.75 * distance_score


def _skill_threshold(skill: SkillSpec, name: str) -> float:
    detection = skill.detection
    thresholds = detection.get("thresholds")
    if not isinstance(thresholds, dict) or name not in thresholds:
        raise KeyError(f"{skill.skill_id} missing detection threshold: {name}")
    item = thresholds[name]
    if not isinstance(item, dict) or set(item) != {"value", "source"}:
        raise ValueError(f"{skill.skill_id} detection threshold {name} is invalid")
    return float(item["value"])


def _threshold_evidence(
    skill: SkillSpec,
    evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    skill_id = skill.skill_id
    if skill_id == "lead_hard_brake":
        measured = {
            "minimum_deceleration_mps2": -evidence["leader_minimum_future_acceleration_mps2"],
            "minimum_closing_speed_mps": evidence["closing_speed_mps"],
            "maximum_pair_gap_m": evidence["longitudinal_gap_m"],
        }
    elif skill_id == "lead_sudden_stop":
        measured = {
            "stopped_speed_mps": evidence["leader_minimum_future_speed_mps"],
            "minimum_prior_speed_mps": evidence["leader_maximum_prior_stop_speed_mps"],
            "minimum_stopped_duration_s": evidence["stopped_duration_s"],
            "maximum_pair_gap_m": evidence["longitudinal_gap_m"],
            "minimum_follower_speed_mps": evidence["follower_current_speed_mps"],
            "minimum_closing_speed_mps": evidence["closing_speed_mps"],
            "minimum_pair_gap_m": evidence["longitudinal_gap_m"],
        }
    elif skill_id == "slow_lead_blockage":
        measured = {
            "maximum_leader_speed_mps": evidence["leader_current_speed_mps"],
            "minimum_low_speed_duration_s": evidence["low_speed_duration_s"],
            "maximum_pair_gap_m": evidence["longitudinal_gap_m"],
            "minimum_follower_speed_mps": evidence["follower_current_speed_mps"],
            "minimum_closing_speed_mps": evidence["closing_speed_mps"],
            "minimum_pair_gap_m": evidence["longitudinal_gap_m"],
        }
    elif skill_id == "short_headway_following":
        measured = {
            "minimum_follower_speed_mps": evidence["follower_current_speed_mps"],
            "maximum_time_headway_s": evidence["time_headway_s"],
            "minimum_duration_s": evidence["short_headway_duration_s"],
        }
    elif skill_id == "rear_vehicle_rapid_approach":
        measured = {
            "minimum_relative_speed_mps": evidence["closing_speed_mps"],
            "maximum_initial_gap_m": evidence["longitudinal_gap_m"],
            "maximum_time_to_collision_s": evidence["time_to_collision_s"],
        }
    elif skill_id == "chain_braking":
        measured = {
            "minimum_queue_gap_m": min(
                evidence["front_middle_gap_m"],
                evidence["middle_rear_gap_m"],
            ),
            "maximum_queue_gap_m": max(
                evidence["front_middle_gap_m"],
                evidence["middle_rear_gap_m"],
            ),
            "minimum_moving_speed_mps": evidence["minimum_vehicle_speed_mps"],
            "minimum_vehicle_center_distance_m": evidence[
                "minimum_vehicle_center_distance_m"
            ],
        }
    elif skill_id == "adjacent_vehicle_cut_in":
        measured = {
            "minimum_lateral_displacement_m": abs(evidence["future_lateral_displacement_m"]),
            "minimum_target_gap_m": evidence["post_cut_in_gap_m"],
            "maximum_target_gap_m": evidence["post_cut_in_gap_m"],
        }
    elif skill_id == "cut_out_reveals_slow_vehicle":
        measured = {
            "maximum_slow_vehicle_speed_mps": evidence["slow_vehicle_speed_mps"],
            "maximum_queue_gap_m": max(
                evidence["cut_out_to_target_gap_m"],
                evidence["slow_to_cut_out_gap_m"],
            ),
            "minimum_queue_gap_m": min(
                evidence["cut_out_to_target_gap_m"],
                evidence["slow_to_cut_out_gap_m"],
            ),
        }
    elif skill_id == "narrow_gap_lane_change":
        measured = {
            "minimum_lateral_displacement_m": abs(evidence["future_lateral_displacement_m"]),
            "maximum_front_gap_m": evidence["front_gap_m"],
            "maximum_rear_gap_m": evidence["rear_gap_m"],
        }
    elif skill_id == "simultaneous_lane_change_conflict":
        measured = {
            "maximum_longitudinal_gap_m": evidence["longitudinal_gap_m"],
            "minimum_shared_target_length_m": evidence["shared_target_lane_length_m"],
            "minimum_vehicle_center_distance_m": evidence[
                "current_vehicle_separation_m"
            ],
        }
    elif skill_id == "forced_lane_change_around_blockage":
        measured = {
            "maximum_blocker_speed_mps": evidence["obstacle_speed_mps"],
            "maximum_object_path_distance_m": evidence["minimum_object_path_clearance_m"],
            "minimum_blockage_distance_m": evidence["blockage_distance_ahead_m"],
            "maximum_blockage_distance_m": evidence["blockage_distance_ahead_m"],
            "minimum_moving_speed_mps": evidence["vehicle_speed_mps"],
            "minimum_vehicle_center_distance_m": evidence[
                "vehicle_center_distance_m"
            ],
        }
    elif skill_id == "late_lane_change_before_diverge":
        measured = {
            "maximum_distance_to_diverge_m": evidence["distance_to_diverge_m"],
            "maximum_target_gap_m": evidence["target_vehicle_longitudinal_gap_m"],
            "minimum_adjacent_lane_length_m": evidence["adjacent_lane_length_m"],
            "minimum_current_separation_m": evidence["current_separation_m"],
        }
    elif skill_id == "ramp_merge_small_gap":
        measured = {
            "maximum_convergence_distance_m": evidence["convergence_distance_m"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
        }
    elif skill_id == "lane_drop_merge_competition":
        measured = {
            "maximum_convergence_distance_m": evidence["convergence_distance_m"],
            "maximum_competing_vehicle_gap_m": evidence["current_pair_distance_m"],
        }
    elif skill_id == "merge_without_yield":
        measured = {
            "maximum_convergence_distance_m": evidence["convergence_distance_m"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
        }
    elif skill_id == "diverge_lane_crossing_conflict":
        measured = {
            "maximum_distance_to_diverge_m": evidence["distance_to_diverge_m"],
            "maximum_target_gap_m": evidence["target_vehicle_longitudinal_gap_m"],
            "minimum_lateral_displacement_m": abs(
                evidence["future_lateral_displacement_m"]
            ),
        }
    elif skill_id == "bike_lane_vehicle_merge_conflict":
        measured = {
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "zipper_merge_multi_vehicle":
        measured = {
            "maximum_convergence_distance_m": evidence["convergence_distance_m"],
            "maximum_competing_vehicle_gap_m": evidence[
                "main_flow_vehicle_gap_m"
            ],
            "maximum_arrival_time_gap_s": evidence["maximum_arrival_time_gap_s"],
            "minimum_current_separation_m": evidence[
                "minimum_current_separation_m"
            ],
        }
    elif skill_id == "unprotected_left_turn_conflict":
        measured = {
            "minimum_turn_heading_change_deg": evidence["initiator_signed_heading_change_deg"],
            "minimum_opposing_heading_difference_deg": evidence["actor_heading_difference_deg"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "right_turn_vehicle_conflict":
        measured = {
            "minimum_turn_heading_change_deg": abs(
                evidence["initiator_signed_heading_change_deg"]
            ),
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "crossing_path_conflict":
        measured = {
            "minimum_crossing_angle_deg": evidence["actor_heading_difference_deg"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "intersection_creep_conflict":
        measured = {
            "maximum_entry_distance_m": evidence["intersection_entry_distance_m"],
            "maximum_creep_speed_mps": evidence["creep_speed_mps"],
            "maximum_crossing_arrival_s": evidence["crossing_vehicle_arrival_s"],
            "minimum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "maximum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "minimum_crossing_vehicle_speed_mps": evidence[
                "crossing_vehicle_speed_mps"
            ],
            "minimum_current_separation_m": evidence["current_separation_m"],
        }
    elif skill_id == "intersection_blocking_vehicle":
        measured = {
            "maximum_blocker_speed_mps": evidence["blocking_vehicle_speed_mps"],
            "maximum_crossing_arrival_s": evidence["crossing_vehicle_arrival_s"],
            "minimum_conflict_area_length_m": evidence["conflict_area_length_m"],
            "minimum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "maximum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "minimum_crossing_vehicle_speed_mps": evidence[
                "crossing_vehicle_speed_mps"
            ],
            "minimum_current_separation_m": evidence["current_separation_m"],
        }
    elif skill_id == "mutual_yield_deadlock":
        measured = {
            "maximum_entry_distance_m": max(
                evidence["first_intersection_entry_distance_m"],
                evidence["second_intersection_entry_distance_m"],
            ),
            "maximum_creep_speed_mps": max(
                evidence["first_vehicle_speed_mps"],
                evidence["second_vehicle_speed_mps"],
            ),
            "minimum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "maximum_crossing_angle_deg": evidence["crossing_angle_deg"],
            "minimum_current_separation_m": evidence["current_separation_m"],
        }
    elif skill_id == "crosswalk_pedestrian_crossing":
        measured = {
            "maximum_crosswalk_distance_m": evidence["crosswalk_distance_m"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "jaywalking_pedestrian_crossing":
        measured = {
            "minimum_crosswalk_clearance_m": evidence["crosswalk_distance_m"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
            "minimum_crossing_angle_deg": evidence[
                "vru_vehicle_heading_difference_deg"
            ],
            "maximum_crossing_angle_deg": evidence[
                "vru_vehicle_heading_difference_deg"
            ],
        }
    elif skill_id == "roadside_pedestrian_emergence":
        measured = {
            "maximum_boundary_distance_m": evidence["drivable_boundary_distance_m"],
            "maximum_vehicle_arrival_s": evidence["vehicle_arrival_time_s"],
            "maximum_conflict_distance_m": evidence["vehicle_path_distance_m"],
        }
    elif skill_id == "cyclist_crossing":
        measured = {
            "minimum_crossing_angle_deg": evidence["vru_vehicle_heading_difference_deg"],
            "maximum_crossing_angle_deg": evidence["vru_vehicle_heading_difference_deg"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "turning_vehicle_crosswalk_conflict":
        measured = {
            "minimum_turn_heading_change_deg": abs(evidence["vehicle_heading_change_deg"]),
            "maximum_crosswalk_distance_m": evidence["crosswalk_distance_m"],
            "maximum_arrival_time_gap_s": evidence["arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence["minimum_trajectory_distance_m"],
        }
    elif skill_id == "group_pedestrian_crossing":
        measured = {
            "maximum_group_member_distance_m": evidence["group_member_distance_m"],
            "maximum_group_heading_difference_deg": evidence[
                "group_heading_difference_deg"
            ],
            "maximum_arrival_time_gap_s": evidence["maximum_arrival_time_gap_s"],
            "maximum_conflict_distance_m": evidence[
                "maximum_minimum_trajectory_distance_m"
            ],
        }
    elif skill_id == "cyclist_vehicle_merge":
        measured = {
            "minimum_lateral_displacement_m": abs(
                evidence["future_lateral_displacement_m"]
            ),
            "maximum_front_gap_m": evidence["front_gap_m"],
            "maximum_rear_gap_m": evidence["rear_gap_m"],
            "maximum_merge_heading_difference_deg": evidence[
                "merge_heading_difference_deg"
            ],
        }
    elif skill_id == "wrong_way_vehicle":
        measured = {
            "minimum_opposite_heading_difference_deg": evidence[
                "initiator_lane_heading_error_deg"
            ],
            "minimum_moving_speed_mps": evidence["initiator_speed_mps"],
            "minimum_opposite_heading_duration_s": evidence["opposite_heading_duration_s"],
            "maximum_oncoming_distance_m": evidence["oncoming_vehicle_distance_m"],
        }
    elif skill_id == "stopped_vehicle_reentry":
        measured = {
            "stopped_speed_mps": evidence["current_speed_mps"],
            "minimum_stopped_duration_s": evidence["stopped_duration_s"],
            "minimum_moving_speed_mps": min(
                evidence["front_main_flow_speed_mps"],
                evidence["rear_main_flow_speed_mps"],
            ),
            "minimum_vehicle_center_distance_m": min(
                evidence["front_minimum_trajectory_distance_m"],
                evidence["rear_minimum_trajectory_distance_m"],
            ),
            "maximum_lateral_reentry_distance_m": evidence[
                "lateral_reentry_distance_m"
            ],
            "maximum_front_gap_m": evidence["front_gap_m"],
            "maximum_rear_gap_m": evidence["rear_gap_m"],
        }
    elif skill_id in {"construction_object_lane_blockage", "static_object_avoidance"}:
        measured = {
            "maximum_object_speed_mps": evidence["obstacle_speed_mps"],
            "maximum_object_path_distance_m": evidence["minimum_object_path_clearance_m"],
            "maximum_vehicle_arrival_s": evidence["vehicle_arrival_time_s"],
        }
    elif skill_id == "cut_in_then_brake":
        measured = {
            "minimum_target_gap_m": evidence["relative_longitudinal_position_m"],
            "maximum_target_gap_m": evidence["relative_longitudinal_position_m"],
            "minimum_post_cut_in_braking_distance_m": evidence[
                "post_cut_in_braking_distance_m"
            ],
        }
    elif skill_id == "abrupt_u_turn_conflict":
        measured = {
            "maximum_entry_distance_m": evidence["intersection_entry_distance_m"],
            "minimum_moving_speed_mps": evidence["initiator_speed_mps"],
            "maximum_oncoming_distance_m": evidence["oncoming_vehicle_distance_m"],
            "minimum_opposing_heading_difference_deg": evidence[
                "actor_heading_difference_deg"
            ],
            "minimum_current_separation_m": evidence["current_separation_m"],
        }
    elif skill_id == "multi_vehicle_gap_squeeze":
        measured = {
            "maximum_front_gap_m": evidence["front_gap_m"],
            "maximum_rear_gap_m": evidence["rear_gap_m"],
            "minimum_combined_closing_speed_mps": evidence[
                "combined_closing_speed_mps"
            ],
            "minimum_current_separation_m": evidence[
                "minimum_vehicle_center_distance_m"
            ],
        }
    elif skill_id == "motorcyclist_filtering_conflict":
        measured = {
            "minimum_moving_speed_mps": evidence["motorcyclist_speed_mps"],
            "maximum_filtering_vehicle_gap_m": evidence["vehicle_gap_m"],
            "maximum_motorcyclist_vehicle_distance_m": evidence[
                "maximum_motorcyclist_vehicle_distance_m"
            ],
            "minimum_current_separation_m": evidence[
                "minimum_current_separation_m"
            ],
        }
    else:
        raise ValueError(f"no threshold evidence mapping for skill: {skill_id}")

    specifications = skill.detection["thresholds"]
    if set(measured) != set(specifications):
        raise ValueError(
            f"{skill_id} threshold evidence differs: "
            f"missing={sorted(set(specifications) - set(measured))}, "
            f"unknown={sorted(set(measured) - set(specifications))}"
        )
    result: dict[str, dict[str, Any]] = {}
    for name, specification in specifications.items():
        measured_value = float(measured[name])
        threshold_value = float(specification["value"])
        if not math.isfinite(measured_value):
            raise ValueError(f"{skill_id} threshold evidence {name} is not finite")
        minimum_comparison = name.startswith("minimum_")
        passed = (
            measured_value >= threshold_value
            if minimum_comparison
            else measured_value <= threshold_value
        )
        comparison = (
            "measured_value >= threshold_value"
            if minimum_comparison
            else "measured_value <= threshold_value"
        )
        if not passed:
            raise ValueError(
                f"{skill_id} emitted a match that failed threshold {name}: "
                f"{measured_value} vs {threshold_value}"
            )
        result[name] = {
            "threshold_value": threshold_value,
            "measured_value": measured_value,
            "comparison": comparison,
            "passed": True,
        }
    return result


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _lane_evidence(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
) -> dict[str, Any]:
    first_match, second_match = context.lane_match(first), context.lane_match(second)
    return {
        "initiator_lane_id": None if first_match is None else first_match.lane.lane_id,
        "responder_lane_id": None if second_match is None else second_match.lane.lane_id,
        "same_or_successor_lane": context.lanes_same_or_successor(first, second),
        "adjacent_lanes": context.lanes_adjacent(first, second),
    }


def _make_match(
    initiator: ActorState,
    responder: ActorState,
    *,
    score: float,
    risk_metric: str,
    risk_value: float,
    evidence: dict[str, Any],
    additional_actors: Sequence[ActorState] = (),
) -> RuleMatch:
    if not math.isfinite(risk_value):
        raise ValueError("detected seed risk_value must be finite")
    return RuleMatch(
        initiator=initiator,
        responder=responder,
        additional_actors=tuple(additional_actors),
        trigger_score=float(np.clip(score, 0.0, 1.0)),
        risk_metric=risk_metric,
        risk_value=float(risk_value),
        evidence=evidence,
    )


def _longitudinal_pair(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if skill.skill_id == "rear_vehicle_rapid_approach":
            rear, target_vehicle = initiator, responder
            gap = _longitudinal_gap(context, target_vehicle, rear)
            leader, follower = target_vehicle, rear
        else:
            leader, follower = initiator, responder
            gap = _longitudinal_gap(context, leader, follower)
        if not context.lanes_same_or_successor(leader, follower):
            continue
        if gap <= 0:
            continue

        closing = follower.speed_mps - leader.speed_mps
        ttc = _pair_ttc(follower, leader, collision_radius_m=0.0)
        future_acceleration = _minimum_future_acceleration(leader)
        minimum_future_speed = _minimum_future_speed(leader)
        headway = time_headway(gap, follower.speed_mps)
        risk_metric = str(skill.risk_definition["metric"])
        risk_value: float
        condition_score: float
        extra_evidence: dict[str, Any] = {}

        if skill.skill_id == "lead_hard_brake":
            maximum_gap = _skill_threshold(skill, "maximum_pair_gap_m")
            minimum_deceleration = _skill_threshold(skill, "minimum_deceleration_mps2")
            minimum_closing = _skill_threshold(skill, "minimum_closing_speed_mps")
            if (
                gap > maximum_gap
                or future_acceleration > -minimum_deceleration
                or closing < minimum_closing
                or not math.isfinite(ttc)
            ):
                continue
            risk_value = ttc
            condition_score = _score_large(-future_acceleration, minimum_deceleration)
        elif skill.skill_id == "lead_sudden_stop":
            maximum_gap = _skill_threshold(skill, "maximum_pair_gap_m")
            minimum_gap = _skill_threshold(skill, "minimum_pair_gap_m")
            stopped_speed = _skill_threshold(skill, "stopped_speed_mps")
            minimum_prior_speed = _skill_threshold(skill, "minimum_prior_speed_mps")
            minimum_duration = _skill_threshold(skill, "minimum_stopped_duration_s")
            minimum_follower_speed = _skill_threshold(
                skill,
                "minimum_follower_speed_mps",
            )
            minimum_closing = _skill_threshold(skill, "minimum_closing_speed_mps")
            stop_metrics = _moving_to_stopped_metrics(
                context,
                leader,
                minimum_prior_speed_mps=minimum_prior_speed,
                stopped_speed_mps=stopped_speed,
            )
            if (
                not minimum_gap <= gap <= maximum_gap
                or follower.speed_mps < minimum_follower_speed
                or closing < minimum_closing
                or stop_metrics is None
            ):
                continue
            (
                stopped_duration,
                stop_time,
                leader_travel,
                maximum_prior_speed,
            ) = stop_metrics
            if stopped_duration < minimum_duration:
                continue
            risk_value = gap + leader_travel - follower.speed_mps * stop_time
            if risk_value < 0:
                continue
            condition_score = (
                _score_large(stopped_duration, minimum_duration)
                + _score_small(minimum_future_speed, stopped_speed + 1e-6)
            ) / 2
            extra_evidence = {
                "stopped_duration_s": stopped_duration,
                "time_until_stop_s": stop_time,
                "leader_travel_until_stop_m": leader_travel,
                "leader_maximum_prior_stop_speed_mps": maximum_prior_speed,
            }
        elif skill.skill_id == "slow_lead_blockage":
            maximum_gap = _skill_threshold(skill, "maximum_pair_gap_m")
            minimum_gap = _skill_threshold(skill, "minimum_pair_gap_m")
            maximum_speed = _skill_threshold(skill, "maximum_leader_speed_mps")
            minimum_duration = _skill_threshold(skill, "minimum_low_speed_duration_s")
            minimum_follower_speed = _skill_threshold(
                skill,
                "minimum_follower_speed_mps",
            )
            minimum_closing = _skill_threshold(skill, "minimum_closing_speed_mps")
            low_speed_duration = _speed_below_duration(
                context,
                leader,
                maximum_speed,
                future_only=True,
            )
            if (
                not minimum_gap <= gap <= maximum_gap
                or leader.speed_mps > maximum_speed
                or follower.speed_mps < minimum_follower_speed
                or closing < minimum_closing
                or low_speed_duration < minimum_duration
            ):
                continue
            risk_value = gap
            condition_score = (
                _score_small(leader.speed_mps, maximum_speed)
                + _score_large(low_speed_duration, minimum_duration)
            ) / 2
            extra_evidence = {"low_speed_duration_s": low_speed_duration}
        elif skill.skill_id == "short_headway_following":
            minimum_speed = _skill_threshold(skill, "minimum_follower_speed_mps")
            maximum_headway = _skill_threshold(skill, "maximum_time_headway_s")
            minimum_duration = _skill_threshold(skill, "minimum_duration_s")
            sustained_duration = _sustained_headway_duration(
                context,
                leader,
                follower,
                maximum_headway,
                minimum_speed,
            )
            if (
                follower.speed_mps < minimum_speed
                or not math.isfinite(headway)
                or headway > maximum_headway
                or sustained_duration < minimum_duration
            ):
                continue
            risk_value = headway
            condition_score = _risk_score(headway, skill)
            extra_evidence = {"short_headway_duration_s": sustained_duration}
        elif skill.skill_id == "rear_vehicle_rapid_approach":
            minimum_relative_speed = _skill_threshold(skill, "minimum_relative_speed_mps")
            maximum_gap = _skill_threshold(skill, "maximum_initial_gap_m")
            maximum_ttc = _skill_threshold(skill, "maximum_time_to_collision_s")
            if (
                gap > maximum_gap
                or closing < minimum_relative_speed
                or not math.isfinite(ttc)
                or ttc > maximum_ttc
            ):
                continue
            risk_value = ttc
            condition_score = _score_large(closing, minimum_relative_speed)
        else:
            raise ValueError(f"unsupported longitudinal skill: {skill.skill_id}")

        results.append(
            _make_match(
                initiator,
                responder,
                score=(condition_score + _risk_score(risk_value, skill)) / 2,
                risk_metric=risk_metric,
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "longitudinal_gap_m": gap,
                    "closing_speed_mps": closing,
                    "leader_current_speed_mps": leader.speed_mps,
                    "follower_current_speed_mps": follower.speed_mps,
                    "time_to_collision_s": _finite_or_none(ttc),
                    "time_headway_s": _finite_or_none(headway),
                    "leader_minimum_future_speed_mps": minimum_future_speed,
                    "leader_minimum_future_acceleration_mps2": future_acceleration,
                    **extra_evidence,
                },
            )
        )
    return results


def _longitudinal_triple(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    """Find the nearest ordered front-middle-rear vehicle triple."""

    results: list[RuleMatch] = []
    actors = {
        state.track_id: state
        for state in (*initiators, *responders)
    }
    ordered_actors = [actors[track_id] for track_id in sorted(actors)]
    if skill.skill_id == "chain_braking":
        maximum_front_gap = maximum_rear_gap = _skill_threshold(
            skill,
            "maximum_queue_gap_m",
        )
        minimum_front_gap = minimum_rear_gap = _skill_threshold(
            skill,
            "minimum_queue_gap_m",
        )
        minimum_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
        minimum_center_distance = _skill_threshold(
            skill,
            "minimum_vehicle_center_distance_m",
        )
        minimum_combined_closing = None
    elif skill.skill_id == "multi_vehicle_gap_squeeze":
        maximum_front_gap = _skill_threshold(skill, "maximum_front_gap_m")
        maximum_rear_gap = _skill_threshold(skill, "maximum_rear_gap_m")
        minimum_front_gap = minimum_rear_gap = _skill_threshold(
            skill,
            "minimum_current_separation_m",
        )
        minimum_speed = 0.0
        minimum_center_distance = _skill_threshold(
            skill,
            "minimum_current_separation_m",
        )
        minimum_combined_closing = _skill_threshold(
            skill,
            "minimum_combined_closing_speed_mps",
        )
    else:
        raise ValueError(f"unsupported longitudinal triple skill: {skill.skill_id}")

    for middle in ordered_actors:
        front_candidates: list[tuple[float, ActorState]] = []
        rear_candidates: list[tuple[float, ActorState]] = []
        for other in ordered_actors:
            if other.track_id == middle.track_id:
                continue
            if not context.lanes_same_or_successor(other, middle):
                continue
            front_gap = _longitudinal_gap(context, other, middle)
            if minimum_front_gap <= front_gap <= maximum_front_gap:
                front_candidates.append((front_gap, other))
            rear_gap = _longitudinal_gap(context, middle, other)
            if minimum_rear_gap <= rear_gap <= maximum_rear_gap:
                rear_candidates.append((rear_gap, other))
        if not front_candidates or not rear_candidates:
            continue
        front_gap, front = min(
            front_candidates,
            key=lambda item: (item[0], item[1].track_id),
        )
        rear_gap, rear = min(
            rear_candidates,
            key=lambda item: (item[0], item[1].track_id),
        )
        if front.track_id == rear.track_id:
            continue
        front_distance = _distance(front, middle)
        rear_distance = _distance(middle, rear)
        minimum_distance = min(front_distance, rear_distance)
        minimum_vehicle_speed = min(
            front.speed_mps,
            middle.speed_mps,
            rear.speed_mps,
        )
        if (
            minimum_distance < minimum_center_distance
            or minimum_vehicle_speed < minimum_speed
        ):
            continue
        front_closing = middle.speed_mps - front.speed_mps
        rear_closing = rear.speed_mps - middle.speed_mps
        combined_closing = max(0.0, front_closing) + max(0.0, rear_closing)
        if (
            minimum_combined_closing is not None
            and combined_closing < minimum_combined_closing
        ):
            continue
        if skill.skill_id == "chain_braking" and max(front_closing, rear_closing) <= 0:
            continue

        front_ttc = _minimum_future_ttc(middle, front)
        rear_ttc = _minimum_future_ttc(rear, middle)
        risk_value = min(front_gap, rear_gap)
        common_evidence = {
            **_lane_evidence(context, front, middle),
            "front_middle_gap_m": front_gap,
            "middle_rear_gap_m": rear_gap,
            "front_gap_m": front_gap,
            "rear_gap_m": rear_gap,
            "front_closing_speed_mps": front_closing,
            "rear_closing_speed_mps": rear_closing,
            "combined_closing_speed_mps": combined_closing,
            "front_time_to_collision_s": _finite_or_none(front_ttc),
            "rear_time_to_collision_s": _finite_or_none(rear_ttc),
            "minimum_vehicle_speed_mps": minimum_vehicle_speed,
            "minimum_vehicle_center_distance_m": minimum_distance,
        }
        if skill.skill_id == "chain_braking":
            results.append(
                _make_match(
                    front,
                    middle,
                    additional_actors=(rear,),
                    score=(
                        _score_small(
                            front_gap + rear_gap,
                            maximum_front_gap + maximum_rear_gap,
                        )
                        + _score_large(max(front_closing, rear_closing), 0.5)
                    )
                    / 2,
                    risk_metric=str(skill.risk_definition["metric"]),
                    risk_value=risk_value,
                    evidence=common_evidence,
                )
            )
        else:
            results.append(
                _make_match(
                    middle,
                    front,
                    additional_actors=(rear,),
                    score=(
                        _score_small(
                            front_gap + rear_gap,
                            maximum_front_gap + maximum_rear_gap,
                        )
                        + _score_large(combined_closing, minimum_combined_closing or 0.5)
                    )
                    / 2,
                    risk_metric=str(skill.risk_definition["metric"]),
                    risk_value=risk_value,
                    evidence=common_evidence,
                )
            )
    return results


def _lane_change_pair(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    lateral_threshold = _skill_threshold(skill, "minimum_lateral_displacement_m")
    minimum_gap = _skill_threshold(skill, "minimum_target_gap_m")
    maximum_gap = _skill_threshold(skill, "maximum_target_gap_m")
    for initiator, responder in context.nearest_pairs(initiators, responders):
        lateral = _future_lateral_displacement(initiator)
        if abs(lateral) < lateral_threshold or not context.lanes_adjacent(initiator, responder):
            continue
        initiator_endpoint = _future_endpoint(initiator)
        responder_endpoint = _future_endpoint(responder)
        responder_lane = context.lane_match(responder)
        if initiator_endpoint is None or responder_endpoint is None or responder_lane is None:
            continue
        target_lane_distance = point_to_polyline_projection(
            initiator_endpoint[0],
            responder_lane.lane.points,
        ).distance_m
        if target_lane_distance > context.config.threshold("lane_match_distance_m"):
            continue
        responder_axis = np.array(
            [math.cos(responder_endpoint[2]), math.sin(responder_endpoint[2])]
        )
        final_gap = float(np.dot(initiator_endpoint[0] - responder_endpoint[0], responder_axis))
        if not minimum_gap <= final_gap <= maximum_gap:
            continue
        ttc = _minimum_future_ttc(responder, initiator)
        if not _ttc_within_risk_horizon(context, ttc):
            continue
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _score_large(abs(lateral), lateral_threshold)
                    + _risk_score(ttc, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=ttc,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "future_lateral_displacement_m": lateral,
                    "target_lane_distance_m": target_lane_distance,
                    "post_cut_in_gap_m": final_gap,
                    "post_cut_in_time_to_collision_s": ttc,
                    "minimum_trajectory_distance_m": minimum_distance,
                },
            )
        )
    return results


def _cyclist_lane_change_gap(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    lateral_threshold = _skill_threshold(skill, "minimum_lateral_displacement_m")
    maximum_front_gap = _skill_threshold(skill, "maximum_front_gap_m")
    maximum_rear_gap = _skill_threshold(skill, "maximum_rear_gap_m")
    maximum_heading_difference = math.radians(
        _skill_threshold(skill, "maximum_merge_heading_difference_deg")
    )
    for cyclist in initiators:
        cyclist_match = context.lane_match(cyclist)
        endpoint = _future_endpoint(cyclist)
        lateral = _future_lateral_displacement(cyclist)
        if cyclist_match is None or endpoint is None or abs(lateral) < lateral_threshold:
            continue
        by_target_lane: dict[str, list[tuple[LaneMatch, ActorState]]] = {}
        for vehicle in responders:
            if vehicle.track_id == cyclist.track_id or not context.lanes_adjacent(
                cyclist,
                vehicle,
            ):
                continue
            vehicle_match = context.lane_match(vehicle)
            if vehicle_match is None or not vehicle_match.lane.lane_id:
                continue
            by_target_lane.setdefault(vehicle_match.lane.lane_id, []).append(
                (vehicle_match, vehicle)
            )
        for target_lane_id, target_vehicles in sorted(by_target_lane.items()):
            target_lane = context.lane_by_id[target_lane_id]
            current_projection = point_to_polyline_projection(
                cyclist.position,
                target_lane.points,
            )
            endpoint_projection = point_to_polyline_projection(
                endpoint[0],
                target_lane.points,
            )
            merge_heading_difference = abs(
                heading_difference(endpoint[2], endpoint_projection.heading_rad)
            )
            if (
                endpoint_projection.distance_m
                > context.config.threshold("lane_match_distance_m")
                or endpoint_projection.distance_m >= current_projection.distance_m
                or merge_heading_difference > maximum_heading_difference
            ):
                continue
            front = sorted(
                (
                    (match.arc_length_m - current_projection.arc_length_m, vehicle)
                    for match, vehicle in target_vehicles
                    if 0
                    < match.arc_length_m - current_projection.arc_length_m
                    <= maximum_front_gap
                ),
                key=lambda item: (item[0], item[1].track_id),
            )
            rear = sorted(
                (
                    (current_projection.arc_length_m - match.arc_length_m, vehicle)
                    for match, vehicle in target_vehicles
                    if 0
                    < current_projection.arc_length_m - match.arc_length_m
                    <= maximum_rear_gap
                ),
                key=lambda item: (item[0], item[1].track_id),
            )
            if not front or not rear:
                continue
            front_gap, front_vehicle = front[0]
            rear_gap, rear_vehicle = rear[0]
            if front_vehicle.track_id == rear_vehicle.track_id:
                continue
            front_ttc = _minimum_future_ttc(cyclist, front_vehicle)
            rear_ttc = _minimum_future_ttc(rear_vehicle, cyclist)
            finite_ttc = [
                value
                for value in (front_ttc, rear_ttc)
                if _ttc_within_risk_horizon(context, value)
            ]
            front_minimum_distance = _future_minimum_distance(
                context,
                cyclist,
                front_vehicle,
            )
            rear_minimum_distance = _future_minimum_distance(
                context,
                cyclist,
                rear_vehicle,
            )
            observed_ttc = bool(finite_ttc)
            finite_minimum_distances = [
                value
                for value in (front_minimum_distance, rear_minimum_distance)
                if math.isfinite(value)
            ]
            if not observed_ttc and not finite_minimum_distances:
                continue
            risk_value = (
                min(finite_ttc)
                if observed_ttc
                else min(finite_minimum_distances)
            )
            results.append(
                _make_match(
                    cyclist,
                    front_vehicle,
                    additional_actors=(rear_vehicle,),
                    score=(
                        _score_large(abs(lateral), lateral_threshold)
                        + _score_small(
                            front_gap + rear_gap,
                            maximum_front_gap + maximum_rear_gap,
                        )
                        + (
                            _risk_score(risk_value, skill)
                            if observed_ttc
                            else _score_small(
                                risk_value,
                                context.config.threshold("maximum_actor_distance_m"),
                            )
                        )
                    )
                    / 3,
                    risk_metric=(
                        str(skill.risk_definition["metric"])
                        if observed_ttc
                        else "minimum_front_rear_trajectory_distance"
                    ),
                    risk_value=risk_value,
                    evidence={
                        **_lane_evidence(context, cyclist, front_vehicle),
                        "target_lane_id": target_lane_id,
                        "future_lateral_displacement_m": lateral,
                        "front_gap_m": front_gap,
                        "rear_gap_m": rear_gap,
                        "rear_vehicle_track_id": rear_vehicle.track_id,
                        "merge_heading_difference_deg": math.degrees(
                            merge_heading_difference
                        ),
                        "front_time_to_collision_s": _finite_or_none(front_ttc),
                        "rear_time_to_collision_s": _finite_or_none(rear_ttc),
                        "front_minimum_trajectory_distance_m": _finite_or_none(
                            front_minimum_distance
                        ),
                        "rear_minimum_trajectory_distance_m": _finite_or_none(
                            rear_minimum_distance
                        ),
                    },
                )
            )
    return results


def _motorcyclist_filtering_gap(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    minimum_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
    maximum_vehicle_gap = _skill_threshold(
        skill,
        "maximum_filtering_vehicle_gap_m",
    )
    maximum_motor_distance = _skill_threshold(
        skill,
        "maximum_motorcyclist_vehicle_distance_m",
    )
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    for motorcyclist in initiators:
        if motorcyclist.speed_mps < minimum_speed:
            continue
        nearby = sorted(
            (
                (_distance(motorcyclist, vehicle), vehicle)
                for vehicle in responders
                if vehicle.track_id != motorcyclist.track_id
                and minimum_separation
                <= _distance(motorcyclist, vehicle)
                <= maximum_motor_distance
            ),
            key=lambda item: (item[0], item[1].track_id),
        )[:12]
        for (_, first), (_, second) in combinations(nearby, 2):
            vehicle_gap = _distance(first, second)
            if not minimum_separation <= vehicle_gap <= maximum_vehicle_gap:
                continue
            segment = second.position - first.position
            squared_length = float(np.dot(segment, segment))
            if squared_length <= 1e-9:
                continue
            position = float(
                np.dot(motorcyclist.position - first.position, segment)
                / squared_length
            )
            if not 0.1 <= position <= 0.9:
                continue
            corridor_point = first.position + position * segment
            corridor_offset = float(
                np.linalg.norm(motorcyclist.position - corridor_point)
            )
            if corridor_offset > min(3.0, maximum_vehicle_gap / 2):
                continue
            first_distance = _distance(motorcyclist, first)
            second_distance = _distance(motorcyclist, second)
            minimum_clearance = min(first_distance, second_distance)
            maximum_distance = max(first_distance, second_distance)
            first_ttc = _minimum_future_ttc(motorcyclist, first)
            second_ttc = _minimum_future_ttc(motorcyclist, second)
            results.append(
                _make_match(
                    motorcyclist,
                    first,
                    additional_actors=(second,),
                    score=(
                        _score_small(vehicle_gap, maximum_vehicle_gap)
                        + _score_small(maximum_distance, maximum_motor_distance)
                    )
                    / 2,
                    risk_metric=str(skill.risk_definition["metric"]),
                    risk_value=minimum_clearance,
                    evidence={
                        **_lane_evidence(context, motorcyclist, first),
                        "motorcyclist_speed_mps": motorcyclist.speed_mps,
                        "vehicle_gap_m": vehicle_gap,
                        "first_vehicle_distance_m": first_distance,
                        "second_vehicle_distance_m": second_distance,
                        "maximum_motorcyclist_vehicle_distance_m": maximum_distance,
                        "minimum_current_separation_m": min(
                            minimum_clearance,
                            vehicle_gap,
                        ),
                        "corridor_projection_fraction": position,
                        "corridor_offset_m": corridor_offset,
                        "first_time_to_collision_s": _finite_or_none(first_ttc),
                        "second_time_to_collision_s": _finite_or_none(second_ttc),
                    },
                )
            )
    return results


def _lane_change_gap(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    if skill.skill_id == "cyclist_vehicle_merge":
        return _cyclist_lane_change_gap(context, skill, initiators, responders)
    if skill.skill_id == "motorcyclist_filtering_conflict":
        return _motorcyclist_filtering_gap(context, skill, initiators, responders)
    if skill.skill_id != "narrow_gap_lane_change":
        raise ValueError(f"unsupported lane-change gap skill: {skill.skill_id}")
    results: list[RuleMatch] = []
    lateral_threshold = _skill_threshold(skill, "minimum_lateral_displacement_m")
    maximum_front_gap = _skill_threshold(skill, "maximum_front_gap_m")
    maximum_rear_gap = _skill_threshold(skill, "maximum_rear_gap_m")
    search_gap = max(maximum_front_gap, maximum_rear_gap)
    for initiator in initiators:
        lateral = _future_lateral_displacement(initiator)
        if abs(lateral) < lateral_threshold:
            continue
        candidates = [
            state
            for state in responders
            if state.track_id != initiator.track_id and context.lanes_adjacent(initiator, state)
        ]
        front = sorted(
            (
                (_relative_coordinates(state, initiator)[0], state)
                for state in candidates
                if 0 < _relative_coordinates(state, initiator)[0] <= search_gap
            ),
            key=lambda item: (item[0], item[1].track_id),
        )
        rear = sorted(
            (
                (-_relative_coordinates(state, initiator)[0], state)
                for state in candidates
                if -search_gap <= _relative_coordinates(state, initiator)[0] < 0
            ),
            key=lambda item: (item[0], item[1].track_id),
        )
        if not front or not rear:
            continue
        front_gap, front_state = front[0]
        rear_gap, rear_state = rear[0]
        if front_gap > maximum_front_gap or rear_gap > maximum_rear_gap:
            continue
        front_ttc = _minimum_future_ttc(initiator, front_state)
        rear_ttc = _minimum_future_ttc(rear_state, initiator)
        finite_ttc = [
            value
            for value in (front_ttc, rear_ttc)
            if _ttc_within_risk_horizon(context, value)
        ]
        if not finite_ttc:
            continue
        risk_value = min(finite_ttc)
        results.append(
            _make_match(
                initiator,
                front_state,
                score=(
                    _score_large(abs(lateral), lateral_threshold)
                    + _score_small(front_gap + rear_gap, maximum_front_gap + maximum_rear_gap)
                    + _risk_score(risk_value, skill)
                )
                / 3,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                additional_actors=(rear_state,),
                evidence={
                    **_lane_evidence(context, initiator, front_state),
                    "future_lateral_displacement_m": lateral,
                    "front_gap_m": front_gap,
                    "rear_gap_m": rear_gap,
                    "rear_vehicle_track_id": rear_state.track_id,
                    "front_time_to_collision_s": _finite_or_none(front_ttc),
                    "rear_time_to_collision_s": _finite_or_none(rear_ttc),
                },
            )
        )
    return results


def _three_vehicle_reveal(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    minimum_gap = _skill_threshold(skill, "minimum_queue_gap_m")
    maximum_gap = _skill_threshold(skill, "maximum_queue_gap_m")
    maximum_slow_speed = _skill_threshold(skill, "maximum_slow_vehicle_speed_mps")
    for cut_out in initiators:
        lateral = _future_lateral_displacement(cut_out)
        if context.lane_match(cut_out) is None or not context.lane_has_neighbor(cut_out):
            continue
        followers = [
            state
            for state in responders
            if state.track_id != cut_out.track_id
            and context.lane_match(state) is not None
            and context.lanes_same_or_successor(cut_out, state)
            and minimum_gap
            <= _longitudinal_gap(context, cut_out, state)
            <= maximum_gap
        ]
        if not followers:
            continue
        follower = min(
            followers,
            key=lambda state: (_longitudinal_gap(context, cut_out, state), state.track_id),
        )
        leaders = [
            state
            for state in responders
            if state.track_id not in {cut_out.track_id, follower.track_id}
            and context.lane_match(state) is not None
            and context.lanes_same_or_successor(state, cut_out)
            and minimum_gap
            <= _longitudinal_gap(context, state, cut_out)
            <= maximum_gap
            and state.speed_mps <= maximum_slow_speed
        ]
        if not leaders:
            continue
        slow_vehicle = min(
            leaders,
            key=lambda state: (_longitudinal_gap(context, state, cut_out), state.track_id),
        )
        cut_out_to_target_gap = _longitudinal_gap(context, cut_out, follower)
        slow_to_cut_out_gap = _longitudinal_gap(context, slow_vehicle, cut_out)
        exposed_gap = _longitudinal_gap(context, slow_vehicle, follower)
        if exposed_gap <= 0:
            continue
        ttc = _minimum_future_ttc(follower, slow_vehicle)
        observed_ttc = _ttc_within_risk_horizon(context, ttc)
        risk_value = ttc if observed_ttc else exposed_gap
        risk_metric = (
            str(skill.risk_definition["metric"])
            if observed_ttc
            else "newly_exposed_longitudinal_gap"
        )
        results.append(
            _make_match(
                cut_out,
                follower,
                score=(
                    0.5
                    + 0.25 * float(abs(lateral) > 0.5)
                    + 0.25 * (
                        _risk_score(ttc, skill)
                        if observed_ttc
                        else _score_small(exposed_gap, 2 * maximum_gap)
                    )
                ),
                risk_metric=risk_metric,
                risk_value=risk_value,
                additional_actors=(slow_vehicle,),
                evidence={
                    **_lane_evidence(context, cut_out, follower),
                    "future_lateral_displacement_m": lateral,
                    "slow_vehicle_track_id": slow_vehicle.track_id,
                    "slow_vehicle_speed_mps": slow_vehicle.speed_mps,
                    "cut_out_to_target_gap_m": cut_out_to_target_gap,
                    "slow_to_cut_out_gap_m": slow_to_cut_out_gap,
                    "newly_exposed_gap_m": exposed_gap,
                    "newly_exposed_time_to_collision_s": (
                        ttc if observed_ttc else None
                    ),
                },
            )
        )
    return results


def _simultaneous_lane_change(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_gap = _skill_threshold(skill, "maximum_longitudinal_gap_m")
    minimum_target_length = _skill_threshold(skill, "minimum_shared_target_length_m")
    minimum_center_distance = _skill_threshold(
        skill,
        "minimum_vehicle_center_distance_m",
    )
    seen: set[tuple[str, str]] = set()
    for first, second in context.nearest_pairs(initiators, responders):
        key = tuple(sorted((first.track_id, second.track_id)))
        if key in seen:
            continue
        seen.add(key)
        first_match, second_match = context.lane_match(first), context.lane_match(second)
        if first_match is None or second_match is None:
            continue
        if (
            not first_match.lane.lane_id
            or not second_match.lane.lane_id
            or first_match.lane.lane_id == second_match.lane.lane_id
            or context.lanes_same_or_successor(first, second)
        ):
            continue
        first_neighbors = {
            first_match.lane.left_neighbor_id,
            first_match.lane.right_neighbor_id,
        }
        second_neighbors = {
            second_match.lane.left_neighbor_id,
            second_match.lane.right_neighbor_id,
        }
        shared_ids = sorted((first_neighbors & second_neighbors) - {None})
        if not shared_ids or shared_ids[0] not in context.lane_by_id:
            continue
        target_lane = context.lane_by_id[shared_ids[0]]
        target_id = target_lane.lane_id
        if (
            first_match.lane.right_neighbor_id == target_id
            and second_match.lane.left_neighbor_id == target_id
        ):
            left_vehicle, right_vehicle = first, second
        elif (
            second_match.lane.right_neighbor_id == target_id
            and first_match.lane.left_neighbor_id == target_id
        ):
            left_vehicle, right_vehicle = second, first
        else:
            continue
        left_match = first_match if left_vehicle is first else second_match
        right_match = second_match if right_vehicle is second else first_match
        if (
            target_lane.left_neighbor_id != left_match.lane.lane_id
            or target_lane.right_neighbor_id != right_match.lane.lane_id
        ):
            continue
        left_lateral = _future_lateral_displacement(left_vehicle)
        right_lateral = _future_lateral_displacement(right_vehicle)
        current_separation = _distance(left_vehicle, right_vehicle)
        if current_separation < minimum_center_distance:
            continue
        longitudinal_gap = abs(
            _relative_coordinates(left_vehicle, right_vehicle)[0]
        )
        if longitudinal_gap > maximum_gap:
            continue
        target_length = float(np.linalg.norm(np.diff(target_lane.points, axis=0), axis=1).sum())
        if target_length < minimum_target_length:
            continue
        left_target_lateral = point_to_polyline_projection(
            left_vehicle.position,
            target_lane.points,
        ).signed_lateral_distance_m
        right_target_lateral = point_to_polyline_projection(
            right_vehicle.position,
            target_lane.points,
        ).signed_lateral_distance_m
        if (
            not math.isfinite(left_target_lateral)
            or not math.isfinite(right_target_lateral)
            or left_target_lateral * right_target_lateral >= 0
        ):
            continue
        minimum_distance = _future_minimum_distance(
            context,
            left_vehicle,
            right_vehicle,
        )
        if not math.isfinite(minimum_distance):
            continue
        lateral_ttc = _minimum_future_ttc(left_vehicle, right_vehicle)
        observed_overlap = _ttc_within_risk_horizon(context, lateral_ttc)
        results.append(
            _make_match(
                left_vehicle,
                right_vehicle,
                score=0.5
                + 0.25 * _score_small(longitudinal_gap, maximum_gap)
                + 0.25 * (
                    _risk_score(lateral_ttc, skill)
                    if observed_overlap
                    else _score_small(
                        minimum_distance,
                        context.config.threshold("maximum_actor_distance_m"),
                    )
                ),
                risk_metric=(
                    str(skill.risk_definition["metric"])
                    if observed_overlap
                    else "minimum_trajectory_distance"
                ),
                risk_value=lateral_ttc if observed_overlap else minimum_distance,
                evidence={
                    **_lane_evidence(context, left_vehicle, right_vehicle),
                    "shared_target_lane_id": target_lane.lane_id,
                    "shared_target_lane_length_m": target_length,
                    "target_has_reciprocal_source_neighbors": True,
                    "left_vehicle_target_lane_lateral_offset_m": left_target_lateral,
                    "right_vehicle_target_lane_lateral_offset_m": right_target_lateral,
                    "current_vehicle_separation_m": current_separation,
                    "longitudinal_gap_m": longitudinal_gap,
                    "first_future_lateral_displacement_m": left_lateral,
                    "second_future_lateral_displacement_m": right_lateral,
                    "observed_lateral_time_to_collision_s": (
                        lateral_ttc if observed_overlap else None
                    ),
                    "minimum_trajectory_distance_m": minimum_distance,
                },
            )
        )
    return results


def _distance_to_future_path(obstacle: ActorState, vehicle: ActorState) -> float:
    points, _ = extract_valid_trajectory(
        vehicle.agent.positions,
        valid_mask=_future_mask(vehicle),
    )
    if not len(points):
        return float("inf")
    return point_to_polyline_projection(obstacle.position, points).distance_m


def _blockage_avoidance(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_blocker_speed = _skill_threshold(skill, "maximum_blocker_speed_mps")
    maximum_clearance = _skill_threshold(skill, "maximum_object_path_distance_m")
    minimum_vehicle_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
    minimum_center_distance = _skill_threshold(
        skill,
        "minimum_vehicle_center_distance_m",
    )
    minimum_blockage_distance = _skill_threshold(skill, "minimum_blockage_distance_m")
    maximum_blockage_distance = _skill_threshold(skill, "maximum_blockage_distance_m")
    for obstacle, vehicle in context.nearest_pairs(initiators, responders):
        center_distance = _distance(obstacle, vehicle)
        if (
            obstacle.speed_mps > maximum_blocker_speed
            or vehicle.speed_mps < minimum_vehicle_speed
            or center_distance < minimum_center_distance
        ):
            continue
        forward, _ = _relative_coordinates(obstacle, vehicle)
        if (
            not minimum_blockage_distance <= forward <= maximum_blockage_distance
            or not context.lane_has_neighbor(vehicle)
        ):
            continue
        if (
            obstacle.object_type in _VEHICLE_TYPES
            and not context.lanes_same_or_successor(obstacle, vehicle)
        ):
            continue
        clearance = _distance_to_future_path(obstacle, vehicle)
        if clearance > maximum_clearance:
            continue
        adjacent_traffic = [
            state
            for state in context.vehicles
            if state.track_id not in {obstacle.track_id, vehicle.track_id}
            and context.lanes_adjacent(vehicle, state)
        ]
        adjacent_clearance = min(
            (_distance(vehicle, state) for state in adjacent_traffic),
            default=clearance,
        )
        combined_clearance = min(clearance, adjacent_clearance)
        results.append(
            _make_match(
                obstacle,
                vehicle,
                score=(
                    _score_small(clearance, maximum_clearance)
                    + _risk_score(combined_clearance, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=combined_clearance,
                evidence={
                    **_lane_evidence(context, obstacle, vehicle),
                    "obstacle_type": obstacle.object_type,
                    "obstacle_speed_mps": obstacle.speed_mps,
                    "vehicle_speed_mps": vehicle.speed_mps,
                    "vehicle_center_distance_m": center_distance,
                    "blockage_distance_ahead_m": forward,
                    "minimum_object_path_clearance_m": clearance,
                    "minimum_adjacent_traffic_clearance_m": adjacent_clearance,
                    "minimum_combined_clearance_m": combined_clearance,
                    "adjacent_lane_available": context.lane_has_neighbor(vehicle),
                },
            )
        )
    return results


def _arrival_time(point: np.ndarray, state: ActorState) -> float:
    if state.speed_mps <= 1e-6:
        return float("inf")
    return float(np.linalg.norm(point - state.position) / state.speed_mps)


def _forward_distance_to_point(state: ActorState, point: np.ndarray) -> float:
    axis = np.array([math.cos(state.heading_rad), math.sin(state.heading_rad)])
    return float(np.dot(point - state.position, axis))


def _pair_specific_convergence(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
    *,
    require_two_source_lanes: bool,
) -> tuple[np.ndarray, str, str, float, float] | None:
    """Return a topology-backed future merge point for this exact lane pair."""

    first_match, second_match = context.lane_match(first), context.lane_match(second)
    if first_match is None or second_match is None:
        return None
    first_lane, second_lane = first_match.lane, second_match.lane
    first_id, second_id = first_lane.lane_id, second_lane.lane_id
    if not first_id or not second_id or first_id == second_id:
        return None

    candidates: list[tuple[MapPolyline, str]] = []
    for target_id in sorted(set(first_lane.successor_ids) & set(second_lane.successor_ids)):
        target = context.lane_by_id.get(target_id)
        if target is None:
            continue
        if {first_id, second_id} <= set(target.predecessor_ids):
            candidates.append((target, "two_source_lanes_share_successor"))

    if not require_two_source_lanes:
        if (
            second_id in first_lane.successor_ids
            and first_id in second_lane.predecessor_ids
            and len(set(second_lane.predecessor_ids)) >= 2
        ):
            candidates.append((second_lane, "first_source_feeds_multi_predecessor_target"))
        if (
            first_id in second_lane.successor_ids
            and second_id in first_lane.predecessor_ids
            and len(set(first_lane.predecessor_ids)) >= 2
        ):
            candidates.append((first_lane, "second_source_feeds_multi_predecessor_target"))

    valid: list[tuple[float, np.ndarray, str, str, float, float]] = []
    for target, relation in candidates:
        if not len(target.points):
            continue
        point = target.points[0].astype(np.float64, copy=True)
        first_forward = _forward_distance_to_point(first, point)
        second_forward = _forward_distance_to_point(second, point)
        if first_forward <= 0 or second_forward <= 0:
            continue
        valid.append(
            (
                max(first_forward, second_forward),
                point,
                relation,
                target.lane_id or "",
                first_forward,
                second_forward,
            )
        )
    if not valid:
        return None
    _, point, relation, target_id, first_forward, second_forward = min(
        valid,
        key=lambda item: (item[0], item[2], item[3]),
    )
    return point, relation, target_id, first_forward, second_forward


def _ordered_merge_roles(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
    *,
    target_lane_id: str,
    convergence_relation: str,
    allow_counterfactual_priority_assignment: bool,
) -> tuple[ActorState, ActorState, str] | None:
    if convergence_relation == "first_source_feeds_multi_predecessor_target":
        return first, second, "initiator_lane_feeds_responder_lane"
    if convergence_relation == "second_source_feeds_multi_predecessor_target":
        return second, first, "initiator_lane_feeds_responder_lane"

    first_match, second_match = context.lane_match(first), context.lane_match(second)
    target_lane = context.lane_by_id.get(target_lane_id)
    if first_match is None or second_match is None or target_lane is None:
        return None

    def endpoint_heading(points: np.ndarray, *, at_start: bool) -> float | None:
        pairs = zip(points[:-1], points[1:])
        if not at_start:
            pairs = reversed(list(pairs))
        for start, end in pairs:
            delta = end - start
            if np.isfinite(delta).all() and np.linalg.norm(delta) > 1e-6:
                return float(math.atan2(delta[1], delta[0]))
        return None

    target_heading = endpoint_heading(target_lane.points, at_start=True)
    first_heading = endpoint_heading(first_match.lane.points, at_start=False)
    second_heading = endpoint_heading(second_match.lane.points, at_start=False)
    if target_heading is None or first_heading is None or second_heading is None:
        return None
    first_change = heading_difference(first_heading, target_heading)
    second_change = heading_difference(second_heading, target_heading)
    if abs(first_change - second_change) <= math.radians(0.5):
        if not allow_counterfactual_priority_assignment:
            return None
        ordered = sorted(
            (
                (first_match.lane.lane_id or "", first.track_id, first),
                (second_match.lane.lane_id or "", second.track_id, second),
            ),
            key=lambda item: (item[0], item[1]),
        )
        return (
            ordered[0][2],
            ordered[1][2],
            "deterministic_counterfactual_priority_assignment",
        )
    if first_change > second_change:
        return first, second, "larger_heading_change_to_target"
    return second, first, "larger_heading_change_to_target"


def _merge_pair(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    horizon = context.config.threshold("risk_time_horizon_s")
    seen: set[tuple[str, str]] = set()
    for raw_initiator, raw_responder in context.nearest_pairs(initiators, responders):
        pair_key = tuple(sorted((raw_initiator.track_id, raw_responder.track_id)))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        if skill.skill_id == "bike_lane_vehicle_merge_conflict":
            initiator, responder = raw_initiator, raw_responder
            role_basis = "yaml_actor_types"
            conflict = _future_conflict(context, initiator, responder)
            convergence_point = context.convergence_point(initiator, responder)
            topology_match = context.lanes_converge(initiator, responder)
            topology_relation = "bike_vehicle_path_conflict"
            pair_specific = False
            target_lane_id = None
            initiator_forward = responder_forward = None
            if conflict is not None and math.isfinite(conflict.time_gap_s):
                risk_value = conflict.time_gap_s
                point = conflict.point
                initiator_arrival = conflict.first_time_s
                responder_arrival = conflict.second_time_s
            elif convergence_point is not None:
                point = convergence_point
                initiator_arrival = _arrival_time(point, initiator)
                responder_arrival = _arrival_time(point, responder)
                if max(initiator_arrival, responder_arrival) > horizon:
                    continue
                risk_value = abs(initiator_arrival - responder_arrival)
            else:
                continue
        else:
            convergence = _pair_specific_convergence(
                context,
                raw_initiator,
                raw_responder,
                require_two_source_lanes=(
                    skill.skill_id == "lane_drop_merge_competition"
                ),
            )
            if convergence is None:
                continue
            (
                point,
                topology_relation,
                target_lane_id,
                raw_first_forward,
                raw_second_forward,
            ) = convergence
            ordered_roles = _ordered_merge_roles(
                context,
                raw_initiator,
                raw_responder,
                target_lane_id=target_lane_id,
                convergence_relation=topology_relation,
                allow_counterfactual_priority_assignment=(
                    skill.skill_id == "merge_without_yield"
                    and skill.detection["mode"] == "compatible_seed"
                ),
            )
            if ordered_roles is None:
                continue
            initiator, responder, role_basis = ordered_roles
            topology_match = True
            pair_specific = True
            initiator_arrival = _arrival_time(point, initiator)
            responder_arrival = _arrival_time(point, responder)
            if max(initiator_arrival, responder_arrival) > horizon:
                continue
            risk_value = abs(initiator_arrival - responder_arrival)
            if initiator.track_id == raw_initiator.track_id:
                initiator_forward, responder_forward = (
                    raw_first_forward,
                    raw_second_forward,
                )
            else:
                initiator_forward, responder_forward = (
                    raw_second_forward,
                    raw_first_forward,
                )
        if not topology_match and skill.skill_id != "bike_lane_vehicle_merge_conflict":
            continue
        if not math.isfinite(risk_value):
            continue
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        convergence_distance = max(
            float(np.linalg.norm(point - initiator.position)),
            float(np.linalg.norm(point - responder.position)),
        )
        current_pair_distance = _distance(initiator, responder)

        if skill.skill_id == "ramp_merge_small_gap":
            if (
                convergence_distance
                > _skill_threshold(skill, "maximum_convergence_distance_m")
                or risk_value > _skill_threshold(skill, "maximum_arrival_time_gap_s")
                or minimum_distance > context.config.threshold("conflict_distance_m")
            ):
                continue
        elif skill.skill_id == "lane_drop_merge_competition":
            if (
                convergence_distance
                > _skill_threshold(skill, "maximum_convergence_distance_m")
                or current_pair_distance
                > _skill_threshold(skill, "maximum_competing_vehicle_gap_m")
            ):
                continue
        elif skill.skill_id == "merge_without_yield":
            if (
                convergence_distance
                > _skill_threshold(skill, "maximum_convergence_distance_m")
                or risk_value > _skill_threshold(skill, "maximum_arrival_time_gap_s")
            ):
                continue
        elif skill.skill_id == "bike_lane_vehicle_merge_conflict":
            if (
                risk_value > _skill_threshold(skill, "maximum_arrival_time_gap_s")
                or minimum_distance > _skill_threshold(skill, "maximum_conflict_distance_m")
            ):
                continue
        else:
            raise ValueError(f"unsupported merge skill: {skill.skill_id}")
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _risk_score(risk_value, skill)
                    + _score_small(
                        minimum_distance,
                        context.config.threshold("maximum_actor_distance_m"),
                    )
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "lanes_converge": topology_match,
                    "pair_specific_convergence": pair_specific,
                    "convergence_relation": topology_relation,
                    "convergence_target_lane_id": target_lane_id,
                    "initiator_forward_to_convergence_m": initiator_forward,
                    "responder_forward_to_convergence_m": responder_forward,
                    "role_assignment_basis": role_basis,
                    "conflict_point_xy": [float(point[0]), float(point[1])],
                    "convergence_distance_m": convergence_distance,
                    "current_pair_distance_m": current_pair_distance,
                    "initiator_arrival_time_s": initiator_arrival,
                    "responder_arrival_time_s": responder_arrival,
                    "arrival_time_gap_s": risk_value,
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                    "counterfactual_priority_required": skill.skill_id == "merge_without_yield",
                },
            )
        )
    return results


def _merge_triple(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    if skill.skill_id != "zipper_merge_multi_vehicle":
        raise ValueError(f"unsupported merge triple skill: {skill.skill_id}")
    results: list[RuleMatch] = []
    maximum_convergence = _skill_threshold(
        skill,
        "maximum_convergence_distance_m",
    )
    maximum_main_gap = _skill_threshold(
        skill,
        "maximum_competing_vehicle_gap_m",
    )
    maximum_arrival_gap = _skill_threshold(
        skill,
        "maximum_arrival_time_gap_s",
    )
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    actors = {
        state.track_id: state
        for state in (*initiators, *responders)
    }
    seen: set[tuple[str, str, str]] = set()
    for raw_first, raw_second in context.nearest_pairs(initiators, responders):
        convergence = _pair_specific_convergence(
            context,
            raw_first,
            raw_second,
            require_two_source_lanes=True,
        )
        if convergence is None:
            continue
        point, relation, target_lane_id, _, _ = convergence
        ordered_roles = _ordered_merge_roles(
            context,
            raw_first,
            raw_second,
            target_lane_id=target_lane_id,
            convergence_relation=relation,
            allow_counterfactual_priority_assignment=True,
        )
        if ordered_roles is None:
            continue
        merging, main_flow, role_basis = ordered_roles
        merging_distance = float(np.linalg.norm(point - merging.position))
        main_distance = float(np.linalg.norm(point - main_flow.position))
        convergence_distance = max(merging_distance, main_distance)
        if convergence_distance > maximum_convergence:
            continue
        third_candidates: list[tuple[float, ActorState]] = []
        for third in actors.values():
            if third.track_id in {merging.track_id, main_flow.track_id}:
                continue
            if not context.lanes_same_or_successor(third, main_flow):
                continue
            main_gap = abs(_longitudinal_gap(context, third, main_flow))
            if minimum_separation <= main_gap <= maximum_main_gap:
                third_candidates.append((main_gap, third))
        if not third_candidates:
            continue
        main_gap, third = min(
            third_candidates,
            key=lambda item: (item[0], item[1].track_id),
        )
        if _longitudinal_gap(context, third, main_flow) > 0:
            leading, trailing = third, main_flow
        else:
            leading, trailing = main_flow, third
        key = (merging.track_id, leading.track_id, trailing.track_id)
        if key in seen:
            continue
        seen.add(key)
        current_separations = (
            _distance(merging, leading),
            _distance(merging, trailing),
            _distance(leading, trailing),
        )
        minimum_current_separation = min(current_separations)
        if minimum_current_separation < minimum_separation:
            continue
        arrivals = {
            "merging": _arrival_time(point, merging),
            "leading": _arrival_time(point, leading),
            "trailing": _arrival_time(point, trailing),
        }
        if not all(math.isfinite(value) for value in arrivals.values()):
            continue
        maximum_observed_arrival_gap = max(
            abs(arrivals["merging"] - arrivals["leading"]),
            abs(arrivals["merging"] - arrivals["trailing"]),
        )
        if maximum_observed_arrival_gap > maximum_arrival_gap:
            continue
        sorted_arrivals = sorted(arrivals.values())
        adjacent_arrival_gaps = [
            second - first
            for first, second in zip(sorted_arrivals, sorted_arrivals[1:])
        ]
        risk_value = min(adjacent_arrival_gaps)
        results.append(
            _make_match(
                merging,
                leading,
                additional_actors=(trailing,),
                score=(
                    _score_small(convergence_distance, maximum_convergence)
                    + _score_small(
                        maximum_observed_arrival_gap,
                        maximum_arrival_gap,
                    )
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, merging, leading),
                    "convergence_relation": relation,
                    "convergence_target_lane_id": target_lane_id,
                    "role_assignment_basis": role_basis,
                    "conflict_point_xy": [float(point[0]), float(point[1])],
                    "convergence_distance_m": convergence_distance,
                    "main_flow_vehicle_gap_m": main_gap,
                    "minimum_current_separation_m": minimum_current_separation,
                    "merging_arrival_time_s": arrivals["merging"],
                    "leading_arrival_time_s": arrivals["leading"],
                    "trailing_arrival_time_s": arrivals["trailing"],
                    "maximum_arrival_time_gap_s": maximum_observed_arrival_gap,
                    "minimum_adjacent_arrival_gap_s": risk_value,
                },
            )
        )
    return results


def _late_lane_change_before_diverge(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_distance = _skill_threshold(skill, "maximum_distance_to_diverge_m")
    maximum_target_gap = _skill_threshold(skill, "maximum_target_gap_m")
    minimum_adjacent_length = _skill_threshold(
        skill,
        "minimum_adjacent_lane_length_m",
    )
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if not context.lane_diverges(initiator) or not context.lanes_adjacent(
            initiator,
            responder,
        ):
            continue
        distance_to_diverge = context.distance_to_lane_end(initiator)
        if not 0 < distance_to_diverge <= maximum_distance:
            continue
        initiator_match = context.lane_match(initiator)
        responder_match = context.lane_match(responder)
        if initiator_match is None or responder_match is None:
            continue
        adjacent_length = float(
            np.linalg.norm(
                np.diff(responder_match.lane.points, axis=0),
                axis=1,
            ).sum()
        )
        longitudinal_gap = abs(_relative_coordinates(responder, initiator)[0])
        separation = _distance(initiator, responder)
        if (
            adjacent_length < minimum_adjacent_length
            or longitudinal_gap > maximum_target_gap
            or separation < minimum_separation
        ):
            continue
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _score_small(distance_to_diverge, maximum_distance)
                    + _score_small(longitudinal_gap, maximum_target_gap)
                )
                / 2,
                risk_metric="distance_to_diverge",
                risk_value=distance_to_diverge,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "initiator_lane_diverges": True,
                    "distance_to_diverge_m": distance_to_diverge,
                    "target_vehicle_longitudinal_gap_m": longitudinal_gap,
                    "adjacent_lane_length_m": adjacent_length,
                    "current_separation_m": separation,
                    "minimum_trajectory_distance_m": _finite_or_none(
                        minimum_distance
                    ),
                },
            )
        )
    return results


def _diverge_crossing(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    if skill.skill_id == "late_lane_change_before_diverge":
        return _late_lane_change_before_diverge(
            context,
            skill,
            initiators,
            responders,
        )
    if skill.skill_id != "diverge_lane_crossing_conflict":
        raise ValueError(f"unsupported diverge skill: {skill.skill_id}")
    results: list[RuleMatch] = []
    maximum_distance_to_diverge = _skill_threshold(skill, "maximum_distance_to_diverge_m")
    maximum_target_gap = _skill_threshold(skill, "maximum_target_gap_m")
    minimum_lateral = _skill_threshold(skill, "minimum_lateral_displacement_m")
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if not context.lane_diverges(initiator):
            continue
        distance_to_diverge = context.distance_to_lane_end(initiator)
        if not 0 < distance_to_diverge <= maximum_distance_to_diverge:
            continue
        initiator_match = context.lane_match(initiator)
        responder_match = context.lane_match(responder)
        if (
            initiator_match is None
            or responder_match is None
            or not initiator_match.lane.lane_id
            or not responder_match.lane.lane_id
            or initiator_match.lane.lane_id == responder_match.lane.lane_id
            or not context.lanes_adjacent(initiator, responder)
        ):
            continue
        longitudinal_gap = abs(_relative_coordinates(responder, initiator)[0])
        if longitudinal_gap > maximum_target_gap:
            continue
        lateral = _future_lateral_displacement(initiator)
        _, responder_lateral = _relative_coordinates(responder, initiator)
        if (
            abs(lateral) < minimum_lateral
            or lateral * responder_lateral <= 0
        ):
            continue
        ttc = _minimum_future_ttc(initiator, responder)
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        observed_ttc = _ttc_within_risk_horizon(context, ttc)
        if observed_ttc:
            risk_value = ttc
            risk_metric = str(skill.risk_definition["metric"])
        elif math.isfinite(minimum_distance):
            risk_value = minimum_distance
            risk_metric = "minimum_trajectory_distance"
        else:
            continue
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    0.5
                    + 0.25 * _score_small(distance_to_diverge, maximum_distance_to_diverge)
                    + 0.25 * (
                        _risk_score(ttc, skill)
                        if observed_ttc
                        else _score_small(minimum_distance, maximum_target_gap)
                    )
                ),
                risk_metric=risk_metric,
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "initiator_lane_diverges": True,
                    "distance_to_diverge_m": distance_to_diverge,
                    "target_vehicle_longitudinal_gap_m": longitudinal_gap,
                    "target_vehicle_lateral_offset_m": responder_lateral,
                    "future_lateral_displacement_m": lateral,
                    "time_to_collision_s": _finite_or_none(ttc),
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


def _cross_flow_structure(
    context: ScenarioDetectionContext,
    first: ActorState,
    second: ActorState,
    *,
    minimum_angle_deg: float,
    maximum_angle_deg: float,
    minimum_second_speed_mps: float,
    minimum_separation_m: float,
) -> float | None:
    first_match, second_match = context.lane_match(first), context.lane_match(second)
    if (
        first_match is None
        or second_match is None
        or not first_match.lane.lane_id
        or not second_match.lane.lane_id
        or first_match.lane.lane_id == second_match.lane.lane_id
        or context.lanes_same_or_successor(first, second)
        or context.lanes_adjacent(first, second)
        or second.speed_mps < minimum_second_speed_mps
        or _distance(first, second) < minimum_separation_m
    ):
        return None
    angle_deg = math.degrees(
        heading_difference(first.heading_rad, second.heading_rad)
    )
    if not minimum_angle_deg <= angle_deg <= maximum_angle_deg:
        return None
    return angle_deg


def _mutual_yield_seed(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_entry = _skill_threshold(skill, "maximum_entry_distance_m")
    maximum_speed = _skill_threshold(skill, "maximum_creep_speed_mps")
    minimum_angle = _skill_threshold(skill, "minimum_crossing_angle_deg")
    maximum_angle = _skill_threshold(skill, "maximum_crossing_angle_deg")
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    seen: set[tuple[str, str]] = set()
    for first, second in context.nearest_pairs(initiators, responders):
        key = tuple(sorted((first.track_id, second.track_id)))
        if key in seen:
            continue
        seen.add(key)
        if first.speed_mps > maximum_speed or second.speed_mps > maximum_speed:
            continue
        first_entry = context.distance_to_intersection(
            first,
            maximum_distance_m=maximum_entry,
        )
        second_entry = context.distance_to_intersection(
            second,
            maximum_distance_m=maximum_entry,
        )
        if first_entry > maximum_entry or second_entry > maximum_entry:
            continue
        crossing_angle = _cross_flow_structure(
            context,
            first,
            second,
            minimum_angle_deg=minimum_angle,
            maximum_angle_deg=maximum_angle,
            minimum_second_speed_mps=0.0,
            minimum_separation_m=minimum_separation,
        )
        if crossing_angle is None:
            continue
        separation = _distance(first, second)
        risk_value = max(first_entry, second_entry)
        results.append(
            _make_match(
                first,
                second,
                score=(
                    _score_small(max(first.speed_mps, second.speed_mps), maximum_speed)
                    + _score_small(risk_value, maximum_entry)
                )
                / 2,
                risk_metric="maximum_distance_to_intersection_entry",
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, first, second),
                    "first_intersection_entry_distance_m": first_entry,
                    "second_intersection_entry_distance_m": second_entry,
                    "first_vehicle_speed_mps": first.speed_mps,
                    "second_vehicle_speed_mps": second.speed_mps,
                    "crossing_angle_deg": crossing_angle,
                    "current_separation_m": separation,
                },
            )
        )
    return results


def _u_turn_compatible_seed(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_entry = _skill_threshold(skill, "maximum_entry_distance_m")
    minimum_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
    maximum_oncoming = _skill_threshold(skill, "maximum_oncoming_distance_m")
    minimum_heading = _skill_threshold(
        skill,
        "minimum_opposing_heading_difference_deg",
    )
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if initiator.speed_mps < minimum_speed:
            continue
        entry_distance = context.distance_to_intersection(
            initiator,
            maximum_distance_m=maximum_entry,
        )
        separation = _distance(initiator, responder)
        heading_difference_deg = math.degrees(
            heading_difference(initiator.heading_rad, responder.heading_rad)
        )
        if (
            entry_distance > maximum_entry
            or not minimum_separation <= separation <= maximum_oncoming
            or heading_difference_deg < minimum_heading
        ):
            continue
        ttc = _pair_ttc(initiator, responder, collision_radius_m=0.0)
        observed_ttc = _ttc_within_risk_horizon(context, ttc)
        risk_value = ttc if observed_ttc else separation
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _score_small(entry_distance, maximum_entry)
                    + _score_large(heading_difference_deg, minimum_heading)
                )
                / 2,
                risk_metric=(
                    str(skill.risk_definition["metric"])
                    if observed_ttc
                    else "oncoming_vehicle_distance"
                ),
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "intersection_entry_distance_m": entry_distance,
                    "initiator_speed_mps": initiator.speed_mps,
                    "oncoming_vehicle_distance_m": separation,
                    "current_separation_m": separation,
                    "actor_heading_difference_deg": heading_difference_deg,
                    "head_on_time_to_collision_s": _finite_or_none(ttc),
                },
            )
        )
    return results


def _conflict_point_pair(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    if skill.skill_id == "mutual_yield_deadlock":
        return _mutual_yield_seed(context, skill, initiators, responders)
    if skill.skill_id == "abrupt_u_turn_conflict":
        return _u_turn_compatible_seed(context, skill, initiators, responders)
    results: list[RuleMatch] = []
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if skill.skill_id == "intersection_creep_conflict":
            maximum_entry = _skill_threshold(skill, "maximum_entry_distance_m")
            maximum_creep_speed = _skill_threshold(skill, "maximum_creep_speed_mps")
            maximum_arrival = _skill_threshold(skill, "maximum_crossing_arrival_s")
            minimum_crossing_angle = _skill_threshold(
                skill,
                "minimum_crossing_angle_deg",
            )
            maximum_crossing_angle = _skill_threshold(
                skill,
                "maximum_crossing_angle_deg",
            )
            minimum_crossing_speed = _skill_threshold(
                skill,
                "minimum_crossing_vehicle_speed_mps",
            )
            minimum_separation = _skill_threshold(
                skill,
                "minimum_current_separation_m",
            )
            if initiator.speed_mps > maximum_creep_speed:
                continue
            entry_distance = context.distance_to_intersection(
                initiator,
                maximum_distance_m=maximum_entry,
            )
            if entry_distance > maximum_entry:
                continue
            crossing_angle = _cross_flow_structure(
                context,
                initiator,
                responder,
                minimum_angle_deg=minimum_crossing_angle,
                maximum_angle_deg=maximum_crossing_angle,
                minimum_second_speed_mps=minimum_crossing_speed,
                minimum_separation_m=minimum_separation,
            )
            if crossing_angle is None:
                continue
            conflict = _future_conflict(context, initiator, responder)
            conflict_distance = context.config.threshold("conflict_distance_m")
            if (
                conflict is None
                or not math.isfinite(conflict.second_time_s)
                or not context.point_near_intersection(
                    conflict.point,
                    conflict_distance,
                )
            ):
                continue
            responder_arrival = max(
                0.0,
                conflict.second_time_s
                - context.timestamps_s[responder.reference_index],
            )
            point = [float(conflict.point[0]), float(conflict.point[1])]
            if responder_arrival > maximum_arrival:
                continue
            risk_value = max(entry_distance, 1e-6)
            minimum_distance = _future_minimum_distance(context, initiator, responder)
            results.append(
                _make_match(
                    initiator,
                    responder,
                    score=0.5
                    + 0.25 * _score_small(entry_distance, maximum_entry)
                    + 0.25 * _score_small(responder_arrival, maximum_arrival),
                    risk_metric=str(skill.risk_definition["metric"]),
                    risk_value=risk_value,
                    evidence={
                        **_lane_evidence(context, initiator, responder),
                        "intersection_entry_distance_m": entry_distance,
                        "creep_speed_mps": initiator.speed_mps,
                        "crossing_vehicle_speed_mps": responder.speed_mps,
                        "crossing_angle_deg": crossing_angle,
                        "current_separation_m": _distance(initiator, responder),
                        "crossing_vehicle_arrival_s": responder_arrival,
                        "conflict_point_xy": point,
                        "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                    },
                )
            )
            continue

        signed_turn = _signed_future_heading_change(initiator)
        responder_turn = _signed_future_heading_change(responder)
        angle = heading_difference(initiator.heading_rad, responder.heading_rad)

        if skill.skill_id == "unprotected_left_turn_conflict":
            turn_threshold = math.radians(
                _skill_threshold(skill, "minimum_turn_heading_change_deg")
            )
            opposing_threshold = math.radians(
                _skill_threshold(skill, "minimum_opposing_heading_difference_deg")
            )
            if signed_turn < turn_threshold or angle < opposing_threshold:
                continue
        elif skill.skill_id == "right_turn_vehicle_conflict":
            turn_threshold = math.radians(
                _skill_threshold(skill, "minimum_turn_heading_change_deg")
            )
            if signed_turn > -turn_threshold:
                continue
        elif skill.skill_id == "crossing_path_conflict":
            if initiator.track_id > responder.track_id:
                continue
            crossing_threshold = math.radians(
                _skill_threshold(skill, "minimum_crossing_angle_deg")
            )
            if angle < crossing_threshold:
                continue
        else:
            raise ValueError(f"unsupported conflict-point skill: {skill.skill_id}")

        maximum_arrival_gap = _skill_threshold(skill, "maximum_arrival_time_gap_s")
        maximum_conflict_distance = _skill_threshold(skill, "maximum_conflict_distance_m")
        conflict = _future_conflict(context, initiator, responder)
        if conflict is None or not math.isfinite(conflict.time_gap_s):
            continue
        if conflict.time_gap_s > maximum_arrival_gap:
            continue
        if not context.point_near_intersection(conflict.point, maximum_conflict_distance):
            continue
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        if minimum_distance > maximum_conflict_distance:
            continue
        risk_value = conflict.time_gap_s
        conflict_point = [float(conflict.point[0]), float(conflict.point[1])]
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _risk_score(risk_value, skill)
                    + _score_small(minimum_distance, maximum_conflict_distance)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "initiator_in_intersection_lane": context.lane_is_intersection(initiator),
                    "responder_in_intersection_lane": context.lane_is_intersection(responder),
                    "initiator_signed_heading_change_deg": math.degrees(signed_turn),
                    "responder_signed_heading_change_deg": math.degrees(responder_turn),
                    "actor_heading_difference_deg": math.degrees(angle),
                    "conflict_point_xy": conflict_point,
                    "arrival_time_gap_s": conflict.time_gap_s,
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


def _intersection_occupancy(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_speed = _skill_threshold(skill, "maximum_blocker_speed_mps")
    maximum_arrival = _skill_threshold(skill, "maximum_crossing_arrival_s")
    minimum_area_length = _skill_threshold(skill, "minimum_conflict_area_length_m")
    minimum_crossing_angle = _skill_threshold(skill, "minimum_crossing_angle_deg")
    maximum_crossing_angle = _skill_threshold(skill, "maximum_crossing_angle_deg")
    minimum_crossing_speed = _skill_threshold(
        skill,
        "minimum_crossing_vehicle_speed_mps",
    )
    minimum_separation = _skill_threshold(skill, "minimum_current_separation_m")
    conflict_distance = context.config.threshold("conflict_distance_m")
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if initiator.speed_mps > maximum_speed:
            continue
        blocker_match = context.lane_match(initiator)
        if blocker_match is None or not blocker_match.lane.is_intersection:
            continue
        intersection_lane = blocker_match.lane
        area_distance = blocker_match.distance_m
        area_length = float(
            np.linalg.norm(np.diff(intersection_lane.points, axis=0), axis=1).sum()
        )
        if area_distance > conflict_distance or area_length < minimum_area_length:
            continue
        crossing_angle = _cross_flow_structure(
            context,
            initiator,
            responder,
            minimum_angle_deg=minimum_crossing_angle,
            maximum_angle_deg=maximum_crossing_angle,
            minimum_second_speed_mps=minimum_crossing_speed,
            minimum_separation_m=minimum_separation,
        )
        if crossing_angle is None:
            continue
        conflict = _future_conflict(context, initiator, responder)
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        if (
            conflict is None
            or not math.isfinite(conflict.second_time_s)
            or point_to_polyline_projection(
                conflict.point,
                intersection_lane.points,
            ).distance_m
            > conflict_distance
        ):
            continue
        responder_arrival = max(
            0.0,
            conflict.second_time_s - context.timestamps_s[responder.reference_index],
        )
        point = [float(conflict.point[0]), float(conflict.point[1])]
        if responder_arrival > maximum_arrival:
            continue
        occupancy_duration = area_length / max(initiator.speed_mps, 0.1)
        risk_value = max(0.0, min(occupancy_duration, maximum_arrival) - responder_arrival)
        if risk_value <= 0:
            continue
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _score_small(initiator.speed_mps, maximum_speed)
                    + _risk_score(risk_value, skill)
                )
                / 2,
                risk_metric="potential_conflict_area_occupancy_overlap_proxy",
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "blocking_vehicle_speed_mps": initiator.speed_mps,
                    "crossing_vehicle_speed_mps": responder.speed_mps,
                    "crossing_angle_deg": crossing_angle,
                    "current_separation_m": _distance(initiator, responder),
                    "conflict_area_lane_id": intersection_lane.lane_id,
                    "conflict_area_length_m": area_length,
                    "conflict_area_distance_m": area_distance,
                    "potential_occupancy_duration_s": occupancy_duration,
                    "conflict_point_xy": point,
                    "crossing_vehicle_arrival_s": responder_arrival,
                    "potential_occupancy_overlap_s": risk_value,
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


def _future_enters_drivable_area(
    context: ScenarioDetectionContext,
    state: ActorState,
) -> bool:
    if state.track_id not in context._future_drivable_entry_cache:
        points, _ = extract_valid_trajectory(
            state.agent.positions,
            valid_mask=_future_mask(state),
        )
        context._future_drivable_entry_cache[state.track_id] = any(
            context.point_inside_drivable_area(point) for point in points
        )
    return context._future_drivable_entry_cache[state.track_id]


def _group_pedestrian_conflict(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_member_distance = _skill_threshold(
        skill,
        "maximum_group_member_distance_m",
    )
    maximum_heading_difference = _skill_threshold(
        skill,
        "maximum_group_heading_difference_deg",
    )
    maximum_arrival_gap = _skill_threshold(skill, "maximum_arrival_time_gap_s")
    maximum_conflict_distance = _skill_threshold(
        skill,
        "maximum_conflict_distance_m",
    )
    seen: set[tuple[str, str, str]] = set()
    for first, vehicle in context.nearest_pairs(initiators, responders):
        first_conflict = _future_conflict(context, first, vehicle)
        if first_conflict is None or not math.isfinite(first_conflict.time_gap_s):
            continue
        first_minimum_distance = _future_minimum_distance(context, first, vehicle)
        if (
            first_conflict.time_gap_s > maximum_arrival_gap
            or first_minimum_distance > maximum_conflict_distance
        ):
            continue
        second_candidates: list[
            tuple[float, float, float, str, ActorState, TrajectoryConflict]
        ] = []
        for second in initiators:
            if second.track_id in {first.track_id, vehicle.track_id}:
                continue
            member_distance = _distance(first, second)
            heading_difference_deg = math.degrees(
                heading_difference(first.heading_rad, second.heading_rad)
            )
            if (
                member_distance > maximum_member_distance
                or heading_difference_deg > maximum_heading_difference
            ):
                continue
            second_conflict = _future_conflict(context, second, vehicle)
            if second_conflict is None or not math.isfinite(second_conflict.time_gap_s):
                continue
            second_minimum_distance = _future_minimum_distance(
                context,
                second,
                vehicle,
            )
            conflict_point_distance = float(
                np.linalg.norm(first_conflict.point - second_conflict.point)
            )
            if (
                second_conflict.time_gap_s > maximum_arrival_gap
                or second_minimum_distance > maximum_conflict_distance
                or conflict_point_distance > maximum_conflict_distance
            ):
                continue
            second_candidates.append(
                (
                    member_distance,
                    heading_difference_deg,
                    second_minimum_distance,
                    second.track_id,
                    second,
                    second_conflict,
                )
            )
        if not second_candidates:
            continue
        (
            member_distance,
            group_heading_difference,
            second_minimum_distance,
            _,
            second,
            second_conflict,
        ) = min(second_candidates)
        group_key = (
            *sorted((first.track_id, second.track_id)),
            vehicle.track_id,
        )
        if group_key in seen:
            continue
        seen.add(group_key)
        maximum_observed_arrival_gap = max(
            first_conflict.time_gap_s,
            second_conflict.time_gap_s,
        )
        maximum_minimum_distance = max(
            first_minimum_distance,
            second_minimum_distance,
        )
        risk_value = min(first_conflict.time_gap_s, second_conflict.time_gap_s)
        results.append(
            _make_match(
                first,
                vehicle,
                additional_actors=(second,),
                score=(
                    _score_small(member_distance, maximum_member_distance)
                    + _risk_score(risk_value, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                evidence={
                    "group_member_track_id": second.track_id,
                    "group_member_distance_m": member_distance,
                    "group_heading_difference_deg": group_heading_difference,
                    "first_conflict_point_xy": [
                        float(first_conflict.point[0]),
                        float(first_conflict.point[1]),
                    ],
                    "second_conflict_point_xy": [
                        float(second_conflict.point[0]),
                        float(second_conflict.point[1]),
                    ],
                    "first_arrival_time_gap_s": first_conflict.time_gap_s,
                    "second_arrival_time_gap_s": second_conflict.time_gap_s,
                    "maximum_arrival_time_gap_s": maximum_observed_arrival_gap,
                    "first_minimum_trajectory_distance_m": first_minimum_distance,
                    "second_minimum_trajectory_distance_m": second_minimum_distance,
                    "maximum_minimum_trajectory_distance_m": maximum_minimum_distance,
                    "group_size": 2,
                },
            )
        )
    return results


def _vru_vehicle_conflict(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    if skill.skill_id == "group_pedestrian_crossing":
        return _group_pedestrian_conflict(
            context,
            skill,
            initiators,
            responders,
        )
    results: list[RuleMatch] = []
    for vru, vehicle in context.nearest_pairs(initiators, responders):
        crosswalk_distance = context.distance_to_crosswalk(vru)
        inside_drivable = context.inside_drivable_area(vru)
        boundary_distance = context.distance_to_drivable_boundary(vru)
        vehicle_turn = _signed_future_heading_change(vehicle)
        future_entry = _future_enters_drivable_area(context, vru)
        vru_vehicle_angle = math.degrees(
            heading_difference(vru.heading_rad, vehicle.heading_rad)
        )

        if skill.skill_id == "roadside_pedestrian_emergence":
            maximum_boundary = _skill_threshold(skill, "maximum_boundary_distance_m")
            maximum_arrival = _skill_threshold(skill, "maximum_vehicle_arrival_s")
            maximum_conflict = _skill_threshold(skill, "maximum_conflict_distance_m")
            path_distance = _distance_to_future_path(vru, vehicle)
            forward, _ = _relative_coordinates(vru, vehicle)
            vehicle_arrival = (
                float("inf")
                if vehicle.speed_mps <= 1e-6
                else _distance(vru, vehicle) / vehicle.speed_mps
            )
            if (
                boundary_distance > maximum_boundary
                or forward <= 0
                or vehicle_arrival > maximum_arrival
                or path_distance > maximum_conflict
            ):
                continue
            results.append(
                _make_match(
                    vru,
                    vehicle,
                    score=0.5
                    + 0.25 * _score_small(boundary_distance, maximum_boundary)
                    + 0.25 * _score_small(vehicle_arrival, maximum_arrival),
                    risk_metric="vehicle_arrival_time_to_boundary_proxy",
                    risk_value=vehicle_arrival,
                    evidence={
                        "vru_type": vru.object_type,
                        "drivable_boundary_distance_m": boundary_distance,
                        "inside_drivable_area": inside_drivable,
                        "future_enters_drivable_area": future_entry,
                        "vehicle_arrival_time_s": vehicle_arrival,
                        "vehicle_path_distance_m": path_distance,
                        "structural_seed_for_counterfactual": True,
                    },
                )
            )
            continue

        if skill.skill_id == "crosswalk_pedestrian_crossing":
            maximum_crosswalk = _skill_threshold(skill, "maximum_crosswalk_distance_m")
            if crosswalk_distance > maximum_crosswalk:
                continue
        elif skill.skill_id == "jaywalking_pedestrian_crossing":
            minimum_crossing_angle = _skill_threshold(
                skill,
                "minimum_crossing_angle_deg",
            )
            maximum_crossing_angle = _skill_threshold(
                skill,
                "maximum_crossing_angle_deg",
            )
            if (
                not minimum_crossing_angle
                <= vru_vehicle_angle
                <= maximum_crossing_angle
            ):
                continue
        elif skill.skill_id == "cyclist_crossing":
            if not (
                _skill_threshold(skill, "minimum_crossing_angle_deg")
                <= vru_vehicle_angle
                <= _skill_threshold(skill, "maximum_crossing_angle_deg")
            ):
                continue
        elif skill.skill_id == "turning_vehicle_crosswalk_conflict":
            maximum_crosswalk = _skill_threshold(skill, "maximum_crosswalk_distance_m")
            minimum_turn = math.radians(
                _skill_threshold(skill, "minimum_turn_heading_change_deg")
            )
            if crosswalk_distance > maximum_crosswalk or abs(vehicle_turn) < minimum_turn:
                continue
        else:
            raise ValueError(f"unsupported VRU skill: {skill.skill_id}")

        maximum_arrival_gap = _skill_threshold(skill, "maximum_arrival_time_gap_s")
        maximum_conflict = _skill_threshold(skill, "maximum_conflict_distance_m")
        conflict = _future_conflict(context, vru, vehicle)
        if conflict is None or not math.isfinite(conflict.time_gap_s):
            continue
        minimum_distance = _future_minimum_distance(context, vru, vehicle)
        if conflict.time_gap_s > maximum_arrival_gap or minimum_distance > maximum_conflict:
            continue
        risk_value = conflict.time_gap_s
        point = [float(conflict.point[0]), float(conflict.point[1])]
        if skill.skill_id == "jaywalking_pedestrian_crossing":
            minimum_clearance = _skill_threshold(
                skill,
                "minimum_crosswalk_clearance_m",
            )
            crosswalk_distance = context.point_distance_to_crosswalk(conflict.point)
            inside_drivable = context.point_inside_drivable_area(conflict.point)
            if not inside_drivable or crosswalk_distance < minimum_clearance:
                continue
        results.append(
            _make_match(
                vru,
                vehicle,
                score=(
                    _score_small(minimum_distance, maximum_conflict)
                    + _risk_score(risk_value, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=risk_value,
                evidence={
                    "vru_type": vru.object_type,
                    "crosswalk_distance_m": _finite_or_none(crosswalk_distance),
                    "drivable_boundary_distance_m": _finite_or_none(boundary_distance),
                    "inside_drivable_area": inside_drivable,
                    "future_enters_drivable_area": future_entry,
                    "vehicle_heading_change_deg": math.degrees(vehicle_turn),
                    "vru_vehicle_heading_difference_deg": vru_vehicle_angle,
                    "conflict_point_xy": point,
                    "arrival_time_gap_s": conflict.time_gap_s,
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


def _wrong_way_pair(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    wrong_way_threshold = math.radians(
        _skill_threshold(skill, "minimum_opposite_heading_difference_deg")
    )
    minimum_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
    minimum_duration = _skill_threshold(skill, "minimum_opposite_heading_duration_s")
    maximum_distance = _skill_threshold(skill, "maximum_oncoming_distance_m")
    aligned_threshold = math.radians(context.config.threshold("lane_heading_tolerance_deg"))
    for initiator, responder in context.nearest_pairs(initiators, responders):
        initiator_match = context.lane_match(initiator, allow_opposite=True)
        responder_match = context.lane_match(responder)
        if initiator_match is None or responder_match is None:
            continue
        if initiator.speed_mps < minimum_speed or context.lane_is_intersection(initiator):
            continue
        if (
            initiator_match.heading_error_rad < wrong_way_threshold
            or responder_match.heading_error_rad > aligned_threshold
        ):
            continue
        same_lane_path = bool(
            initiator_match.lane.lane_id == responder_match.lane.lane_id
            or initiator_match.lane.lane_id in responder_match.lane.predecessor_ids
            or responder_match.lane.lane_id in initiator_match.lane.successor_ids
        )
        if not same_lane_path or _distance(initiator, responder) > maximum_distance:
            continue
        valid_headings = initiator.agent.observed_mask & np.isfinite(
            initiator.agent.headings
        )
        opposite = np.zeros(len(valid_headings), dtype=bool)
        lane_heading = point_to_polyline_projection(
            initiator.position,
            initiator_match.lane.points,
        ).heading_rad
        opposite[valid_headings] = (
            heading_difference(
                initiator.agent.headings[valid_headings],
                lane_heading,
            )
            >= wrong_way_threshold
        )
        opposite_duration = _longest_run_duration(opposite, _sample_period_s(context))
        if opposite_duration < minimum_duration:
            continue
        ttc = _pair_ttc(initiator, responder, collision_radius_m=0.0)
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        if not math.isfinite(ttc):
            continue
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    _score_large(initiator_match.heading_error_rad, wrong_way_threshold)
                    + _risk_score(ttc, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=ttc,
                evidence={
                    "initiator_lane_id": initiator_match.lane.lane_id,
                    "responder_lane_id": responder_match.lane.lane_id,
                    "same_or_successor_lane": same_lane_path,
                    "adjacent_lanes": False,
                    "initiator_lane_heading_error_deg": math.degrees(
                        initiator_match.heading_error_rad
                    ),
                    "responder_lane_heading_error_deg": math.degrees(
                        responder_match.heading_error_rad
                    ),
                    "initiator_speed_mps": initiator.speed_mps,
                    "oncoming_vehicle_distance_m": _distance(initiator, responder),
                    "actor_heading_difference_deg": math.degrees(
                        heading_difference(initiator.heading_rad, responder.heading_rad)
                    ),
                    "opposite_heading_duration_s": opposite_duration,
                    "time_to_collision_s": ttc,
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


def _stopped_reentry(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    stopped_speed = _skill_threshold(skill, "stopped_speed_mps")
    minimum_stopped_duration = _skill_threshold(skill, "minimum_stopped_duration_s")
    minimum_moving_speed = _skill_threshold(skill, "minimum_moving_speed_mps")
    minimum_center_distance = _skill_threshold(
        skill,
        "minimum_vehicle_center_distance_m",
    )
    maximum_lateral_distance = _skill_threshold(
        skill,
        "maximum_lateral_reentry_distance_m",
    )
    maximum_front_gap = _skill_threshold(skill, "maximum_front_gap_m")
    maximum_rear_gap = _skill_threshold(skill, "maximum_rear_gap_m")

    for initiator in initiators:
        if initiator.speed_mps > stopped_speed:
            continue
        stopped_duration = _trailing_observed_speed_below_duration(
            context,
            initiator,
            stopped_speed,
        )
        if stopped_duration < minimum_stopped_duration:
            continue

        by_lane: dict[
            str,
            tuple[MapPolyline, float, float, list[tuple[ActorState, float, float]]],
        ] = {}
        for responder in responders:
            if (
                responder.track_id == initiator.track_id
                or responder.speed_mps < minimum_moving_speed
                or _distance(initiator, responder) < minimum_center_distance
            ):
                continue
            if not (
                context.lanes_same_or_successor(initiator, responder)
                or context.lanes_adjacent(initiator, responder)
            ):
                continue
            match = context.lane_match(responder)
            if match is None or not match.lane.lane_id:
                continue
            projection = point_to_polyline_projection(
                initiator.position,
                match.lane.points,
            )
            if projection.distance_m > maximum_lateral_distance:
                continue
            minimum_distance = _future_minimum_distance(
                context,
                initiator,
                responder,
            )
            if (
                not math.isfinite(minimum_distance)
                or minimum_distance < minimum_center_distance
            ):
                continue
            lane_id = match.lane.lane_id
            if lane_id not in by_lane:
                by_lane[lane_id] = (
                    match.lane,
                    projection.arc_length_m,
                    projection.distance_m,
                    [],
                )
            by_lane[lane_id][3].append(
                (responder, match.arc_length_m, minimum_distance)
            )

        for lane_id, (_, reentry_arc, lateral_distance, lane_actors) in sorted(
            by_lane.items()
        ):
            front_candidates = [
                (state, arc - reentry_arc, minimum_distance)
                for state, arc, minimum_distance in lane_actors
                if minimum_center_distance
                <= arc - reentry_arc
                <= maximum_front_gap
            ]
            rear_candidates = [
                (state, reentry_arc - arc, minimum_distance)
                for state, arc, minimum_distance in lane_actors
                if minimum_center_distance
                <= reentry_arc - arc
                <= maximum_rear_gap
            ]
            if not front_candidates or not rear_candidates:
                continue
            front, front_gap, front_minimum_distance = min(
                front_candidates,
                key=lambda item: (item[1], item[0].track_id),
            )
            rear, rear_gap, rear_minimum_distance = min(
                rear_candidates,
                key=lambda item: (item[1], item[0].track_id),
            )
            if front.track_id == rear.track_id:
                continue

            front_ttc = _minimum_future_ttc(initiator, front)
            rear_ttc = _minimum_future_ttc(rear, initiator)
            finite_ttc = [
                value
                for value in (front_ttc, rear_ttc)
                if _ttc_within_risk_horizon(context, value)
            ]
            if finite_ttc:
                risk_value = min(finite_ttc)
                risk_metric = str(skill.risk_definition["metric"])
                risk_score = _risk_score(risk_value, skill)
            else:
                risk_value = min(front_minimum_distance, rear_minimum_distance)
                risk_metric = "minimum_front_rear_trajectory_distance_proxy"
                risk_score = _score_small(
                    risk_value,
                    max(maximum_front_gap, maximum_rear_gap),
                )

            results.append(
                _make_match(
                    initiator,
                    front,
                    score=(
                        _score_small(initiator.speed_mps, stopped_speed + 1e-6)
                        + _score_large(stopped_duration, minimum_stopped_duration)
                        + risk_score
                    )
                    / 3,
                    risk_metric=risk_metric,
                    risk_value=risk_value,
                    additional_actors=(rear,),
                    evidence={
                        **_lane_evidence(context, initiator, front),
                        "main_flow_lane_id": lane_id,
                        "current_speed_mps": initiator.speed_mps,
                        "stopped_duration_s": stopped_duration,
                        "front_main_flow_track_id": front.track_id,
                        "rear_main_flow_track_id": rear.track_id,
                        "front_main_flow_speed_mps": front.speed_mps,
                        "rear_main_flow_speed_mps": rear.speed_mps,
                        "front_gap_m": front_gap,
                        "rear_gap_m": rear_gap,
                        "lateral_reentry_distance_m": lateral_distance,
                        "front_time_to_collision_s": _finite_or_none(front_ttc),
                        "rear_time_to_collision_s": _finite_or_none(rear_ttc),
                        "front_minimum_trajectory_distance_m": front_minimum_distance,
                        "rear_minimum_trajectory_distance_m": rear_minimum_distance,
                    },
                )
            )
    return results


def _static_blockage(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    maximum_object_speed = _skill_threshold(skill, "maximum_object_speed_mps")
    maximum_clearance = _skill_threshold(skill, "maximum_object_path_distance_m")
    maximum_arrival = _skill_threshold(skill, "maximum_vehicle_arrival_s")
    for obstacle, vehicle in context.nearest_pairs(initiators, responders):
        if obstacle.speed_mps > maximum_object_speed:
            continue
        forward, _ = _relative_coordinates(obstacle, vehicle)
        vehicle_arrival = (
            float("inf")
            if vehicle.speed_mps <= 1e-6
            else _distance(obstacle, vehicle) / vehicle.speed_mps
        )
        if forward <= 0 or vehicle_arrival > maximum_arrival:
            continue
        clearance = _distance_to_future_path(obstacle, vehicle)
        if clearance > maximum_clearance:
            continue
        results.append(
            _make_match(
                obstacle,
                vehicle,
                score=(
                    _score_small(clearance, maximum_clearance)
                    + _risk_score(clearance, skill)
                )
                / 2,
                risk_metric=str(skill.risk_definition["metric"]),
                risk_value=clearance,
                evidence={
                    **_lane_evidence(context, obstacle, vehicle),
                    "obstacle_type": obstacle.object_type,
                    "obstacle_speed_mps": obstacle.speed_mps,
                    "vehicle_speed_mps": vehicle.speed_mps,
                    "vehicle_arrival_time_s": vehicle_arrival,
                    "obstacle_distance_ahead_m": forward,
                    "minimum_object_path_clearance_m": clearance,
                    "obstacle_inside_drivable_area": context.inside_drivable_area(obstacle),
                },
            )
        )
    return results


def _cut_in_then_brake(
    context: ScenarioDetectionContext,
    skill: SkillSpec,
    initiators: Sequence[ActorState],
    responders: Sequence[ActorState],
) -> list[RuleMatch]:
    results: list[RuleMatch] = []
    minimum_gap = _skill_threshold(skill, "minimum_target_gap_m")
    maximum_gap = _skill_threshold(skill, "maximum_target_gap_m")
    minimum_braking_distance = _skill_threshold(
        skill,
        "minimum_post_cut_in_braking_distance_m",
    )
    for initiator, responder in context.nearest_pairs(initiators, responders):
        if not context.lanes_adjacent(initiator, responder):
            continue
        forward, _ = _relative_coordinates(initiator, responder)
        if not minimum_gap <= forward <= maximum_gap:
            continue
        responder_match = context.lane_match(responder)
        if responder_match is None:
            continue
        target_projection = point_to_polyline_projection(
            initiator.position,
            responder_match.lane.points,
        )
        target_lane_length = float(
            np.linalg.norm(np.diff(responder_match.lane.points, axis=0), axis=1).sum()
        )
        braking_distance = max(0.0, target_lane_length - target_projection.arc_length_m)
        if (
            target_projection.distance_m > context.config.threshold("lane_match_distance_m")
            or braking_distance < minimum_braking_distance
        ):
            continue
        lateral = _future_lateral_displacement(initiator)
        minimum_acceleration = _minimum_future_acceleration(initiator)
        ttc = _minimum_future_ttc(responder, initiator)
        minimum_distance = _future_minimum_distance(context, initiator, responder)
        observed_ttc = _ttc_within_risk_horizon(context, ttc)
        if observed_ttc:
            risk_value = ttc
            risk_metric = str(skill.risk_definition["metric"])
        elif math.isfinite(minimum_distance):
            risk_value = minimum_distance
            risk_metric = "minimum_trajectory_distance"
        else:
            continue
        results.append(
            _make_match(
                initiator,
                responder,
                score=(
                    0.5
                    + 0.25 * _score_large(braking_distance, minimum_braking_distance)
                    + 0.25 * (
                        _risk_score(ttc, skill)
                        if observed_ttc
                        else _score_small(minimum_distance, maximum_gap)
                    )
                ),
                risk_metric=risk_metric,
                risk_value=risk_value,
                evidence={
                    **_lane_evidence(context, initiator, responder),
                    "relative_longitudinal_position_m": forward,
                    "target_lane_id": responder_match.lane.lane_id,
                    "target_lane_distance_m": target_projection.distance_m,
                    "post_cut_in_braking_distance_m": braking_distance,
                    "future_lateral_displacement_m": lateral,
                    "minimum_future_acceleration_mps2": minimum_acceleration,
                    "structural_seed_for_counterfactual": True,
                    "time_to_collision_s": _finite_or_none(ttc),
                    "minimum_trajectory_distance_m": _finite_or_none(minimum_distance),
                },
            )
        )
    return results


StrategyHandler = Callable[
    [
        ScenarioDetectionContext,
        SkillSpec,
        Sequence[ActorState],
        Sequence[ActorState],
    ],
    list[RuleMatch],
]


_STRATEGY_HANDLERS: dict[str, StrategyHandler] = {
    "blockage_avoidance": _blockage_avoidance,
    "conflict_point_pair": _conflict_point_pair,
    "cut_in_then_brake": _cut_in_then_brake,
    "diverge_crossing": _diverge_crossing,
    "intersection_occupancy": _intersection_occupancy,
    "lane_change_gap": _lane_change_gap,
    "lane_change_pair": _lane_change_pair,
    "longitudinal_pair": _longitudinal_pair,
    "longitudinal_triple": _longitudinal_triple,
    "merge_pair": _merge_pair,
    "merge_triple": _merge_triple,
    "simultaneous_lane_change": _simultaneous_lane_change,
    "static_blockage": _static_blockage,
    "stopped_reentry": _stopped_reentry,
    "three_vehicle_reveal": _three_vehicle_reveal,
    "vru_vehicle_conflict": _vru_vehicle_conflict,
    "wrong_way_pair": _wrong_way_pair,
}


def detect_scenario(
    scenario: Scenario,
    skills: Iterable[SkillSpec],
    config: DetectionConfig,
) -> DetectionRun:
    """Detect deterministic candidate seeds for all supplied skill specifications."""

    context = ScenarioDetectionContext(scenario, config)
    skill_list = sorted(skills, key=lambda item: item.skill_id)
    skill_ids = [skill.skill_id for skill in skill_list]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("skills must have unique skill_id values")

    records: list[SeedRecord] = []
    rejections: Counter[str] = Counter()
    source_path = str(
        scenario.metadata.get("source_path") or f"scenario://{scenario.scenario_id}"
    )
    for skill in skill_list:
        rule = get_skill_detection_rule(skill.skill_id)
        try:
            handler = _STRATEGY_HANDLERS[rule.strategy]
        except KeyError as exc:
            raise ValueError(f"no detector implemented for strategy: {rule.strategy}") from exc

        minimum_history = skill.seed_requirements.get("minimum_history_steps")
        if (
            isinstance(minimum_history, bool)
            or not isinstance(minimum_history, int)
            or minimum_history <= 0
        ):
            raise ValueError(
                f"{skill.skill_id} seed_requirements.minimum_history_steps must be positive"
            )
        required_tracks = skill.data_support.get("required_tracks")
        required_map = skill.data_support.get("required_map")
        if not isinstance(required_tracks, list) or not required_tracks:
            raise ValueError(f"{skill.skill_id} data_support.required_tracks must be non-empty")
        if not isinstance(required_map, list) or not required_map:
            raise ValueError(f"{skill.skill_id} data_support.required_map must be non-empty")
        missing_tracks = [
            requirement
            for requirement in required_tracks
            if not _has_track_requirement(context, requirement)
        ]
        if missing_tracks:
            for requirement in missing_tracks:
                rejections[f"{skill.skill_id}:missing_track:{requirement}"] += 1
            continue
        missing_map = [
            requirement
            for requirement in required_map
            if not context.has_map_requirement(requirement)
        ]
        if missing_map:
            for requirement in missing_map:
                rejections[f"{skill.skill_id}:missing_map:{requirement}"] += 1
            continue
        initiator_types = skill.actors.get("initiator_types")
        responder_types = skill.actors.get("responder_types")
        if not isinstance(initiator_types, list) or not initiator_types:
            raise ValueError(f"{skill.skill_id} actors.initiator_types must be non-empty")
        if not isinstance(responder_types, list) or not responder_types:
            raise ValueError(f"{skill.skill_id} actors.responder_types must be non-empty")
        initiators = [
            state
            for state in context.actors_of_types(initiator_types)
            if _history_steps(state) >= minimum_history
        ]
        responders = [
            state
            for state in context.actors_of_types(responder_types)
            if _history_steps(state) >= minimum_history
        ]
        if not initiators:
            rejections[f"{skill.skill_id}:missing_history_qualified_initiator"] += 1
            continue
        if not responders:
            rejections[f"{skill.skill_id}:missing_history_qualified_responder"] += 1
            continue

        handler_matches = handler(context, skill, initiators, responders)
        raw_matches: list[RuleMatch] = []
        for match in handler_matches:
            actor_ids = (
                match.initiator.track_id,
                match.responder.track_id,
                *(actor.track_id for actor in match.additional_actors),
            )
            if len(set(actor_ids)) != len(actor_ids):
                rejections[f"{skill.skill_id}:duplicate_role_actor"] += 1
                continue
            raw_matches.append(match)
        matches = [
            match
            for match in raw_matches
            if _shared_history_steps(
                (match.initiator, match.responder, *match.additional_actors)
            )
            >= minimum_history
        ]
        if raw_matches and not matches:
            rejections[f"{skill.skill_id}:insufficient_shared_history"] += len(raw_matches)
        if not matches:
            rejections[f"{skill.skill_id}:no_rule_match"] += 1
            continue
        ordered = sorted(
            matches,
            key=lambda match: (
                -match.trigger_score,
                match.initiator.track_id,
                match.responder.track_id,
                tuple(actor.track_id for actor in match.additional_actors),
            ),
        )
        unique_matches: list[RuleMatch] = []
        seen_pairs: set[tuple[str, ...]] = set()
        for match in ordered:
            pair = (
                match.initiator.track_id,
                match.responder.track_id,
                *(actor.track_id for actor in match.additional_actors),
            )
            if pair not in seen_pairs:
                unique_matches.append(match)
                seen_pairs.add(pair)
        limit = config.max_candidates_per_skill_per_scenario
        if len(unique_matches) > limit:
            rejections[f"{skill.skill_id}:candidate_limit"] += len(unique_matches) - limit
        for match in unique_matches[:limit]:
            generated_roles = skill.actors.get("generated_roles")
            if not isinstance(generated_roles, list) or len(generated_roles) != 2 + len(
                match.additional_actors
            ):
                raise ValueError(
                    f"{skill.skill_id} generated_roles do not match detected actor count"
                )
            actor_ids = [
                match.initiator.track_id,
                match.responder.track_id,
                *(actor.track_id for actor in match.additional_actors),
            ]
            role_track_ids = dict(zip(generated_roles, actor_ids))
            sample_key = "|".join(
                (
                    scenario.scenario_id,
                    skill.skill_id,
                    *(f"{role}={track_id}" for role, track_id in role_track_ids.items()),
                )
            )
            parameters = sample_skill_parameters(
                skill,
                global_seed=config.global_seed,
                sample_key=sample_key,
            )
            threshold_evidence = _threshold_evidence(skill, match.evidence)
            records.append(
                SeedRecord(
                    scenario_id=scenario.scenario_id,
                    skill_id=skill.skill_id,
                    initiator_track_id=match.initiator.track_id,
                    responder_track_id=match.responder.track_id,
                    role_track_ids=role_track_ids,
                    trigger_score=match.trigger_score,
                    seed_risk_metric=match.risk_metric,
                    seed_risk_value=match.risk_value,
                    target_risk_definition=dict(skill.risk_definition),
                    source_path=source_path,
                    evidence={
                        "strategy": rule.strategy,
                        "feasibility": skill.data_support["feasibility"],
                        "detection_mode": skill.detection["mode"],
                        "matched_conditions": list(skill.detection["conditions"]),
                        "missing_generation_conditions": (
                            []
                            if skill.detection["mode"] == "observed_trigger"
                            else list(skill.trigger["conditions"])
                        ),
                        "trigger_conditions": list(skill.trigger["conditions"]),
                        "detection_thresholds": {
                            name: dict(item)
                            for name, item in skill.detection["thresholds"].items()
                        },
                        "threshold_evidence": threshold_evidence,
                        **match.evidence,
                    },
                    sampled_parameters=parameters,
                )
            )

    return DetectionRun(records=sort_seed_records(records), rejection_counts=rejections)


__all__ = [
    "ActorState",
    "DetectionConfig",
    "DetectionRun",
    "LaneMatch",
    "RuleMatch",
    "ScenarioDetectionContext",
    "detect_scenario",
    "load_detection_config",
]

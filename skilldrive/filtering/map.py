"""Map-compliance evidence for generated single-target futures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.filtering.prepared_map import (
    PreparedLaneGeometry,
    PreparedMapGeometry,
    PreparedMapVerificationSession,
    points_in_drivable_area,
    prepare_map_geometry,
    project_points_to_lanes_within_distance,
    require_compatible_map,
)
from skilldrive.generation.config import MapFilterPolicy
from skilldrive.schemas import Scenario
from skilldrive.skills.geometry import heading_difference, point_to_polyline_projection


@dataclass(frozen=True)
class MapComplianceResult:
    passed: bool
    inside_fraction: float | None
    outside_indices: tuple[int, ...]
    geometry_source: str
    reason: str | None = None


@dataclass(frozen=True)
class LaneComplianceResult:
    passed: bool
    assignment_fraction: float | None
    type_compatibility_fraction: float | None
    direction_fraction: float | None
    assigned_lane_ids: tuple[str | None, ...]
    assigned_lane_types: tuple[str | None, ...]
    incompatible_lane_indices: tuple[int, ...]
    incompatible_lane_types: tuple[str, ...]
    invalid_transitions: tuple[tuple[str, str], ...]
    direction_exempt: bool
    reason: str | None = None


_INCOMPATIBLE_LANE_TYPES_BY_ACTOR = {
    "vehicle": frozenset({"bike"}),
    "bus": frozenset({"bike"}),
}


def _lane_type(lane) -> str:
    return str(lane.direction).strip().lower()


def _point_on_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    delta = second - first
    squared = float(np.dot(delta, delta))
    if squared <= 1e-18:
        return bool(np.linalg.norm(point - first) <= 1e-8)
    fraction = float(np.dot(point - first, delta) / squared)
    if not -1e-10 <= fraction <= 1.0 + 1e-10:
        return False
    projection = first + np.clip(fraction, 0.0, 1.0) * delta
    return bool(np.linalg.norm(point - projection) <= 1e-8)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Return true for interior or boundary points of one simple polygon."""

    value = np.asarray(point, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    if value.shape != (2,) or vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("point and polygon must have shapes (2,) and (N, 2)")
    finite = vertices[np.isfinite(vertices).all(axis=1)]
    if len(finite) < 3 or not np.isfinite(value).all():
        return False
    if np.allclose(finite[0], finite[-1]):
        finite = finite[:-1]
    if len(finite) < 3:
        return False

    inside = False
    previous = finite[-1]
    for current in finite:
        if _point_on_segment(value, previous, current):
            return True
        y_crosses = (current[1] > value[1]) != (previous[1] > value[1])
        if y_crosses:
            x_crossing = (previous[0] - current[0]) * (
                value[1] - current[1]
            ) / (previous[1] - current[1]) + current[0]
            if value[0] < x_crossing:
                inside = not inside
        previous = current
    return inside


def evaluate_drivable_area(
    scenario: Scenario,
    future_positions_global: np.ndarray,
    *,
    required: bool,
    minimum_inside_fraction: float = 1.0,
    prepared_map: PreparedMapGeometry | None = None,
) -> MapComplianceResult:
    """Check generated positions against AV2 official drivable-area polygons."""

    positions = np.asarray(future_positions_global, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
        raise ValueError("future_positions_global must have shape (N, 2)")
    if not np.isfinite(positions).all():
        raise ValueError("future_positions_global must contain finite values")
    if not 0.0 <= minimum_inside_fraction <= 1.0:
        raise ValueError("minimum_inside_fraction must be in [0, 1]")
    if prepared_map is not None:
        require_compatible_map((scenario,), prepared_map)
        return _drivable_result_from_membership(
            points_in_drivable_area(prepared_map, positions),
            required=required,
            minimum_inside_fraction=minimum_inside_fraction,
            geometry_available=prepared_map.drivable_area_count > 0,
        )
    polygons = [
        polyline.points
        for polyline in scenario.map_polylines
        if polyline.polyline_type == "drivable_area"
    ]
    if not polygons:
        return MapComplianceResult(
            passed=not required,
            inside_fraction=None,
            outside_indices=tuple(range(len(positions))) if required else (),
            geometry_source="av2_drivable_area_polygon",
            reason="missing_drivable_area_geometry" if required else None,
        )
    inside = np.asarray(
        [any(point_in_polygon(point, polygon) for polygon in polygons) for point in positions],
        dtype=bool,
    )
    return _drivable_result_from_membership(
        inside,
        required=required,
        minimum_inside_fraction=minimum_inside_fraction,
        geometry_available=True,
    )


def _drivable_result_from_membership(
    inside: np.ndarray,
    *,
    required: bool,
    minimum_inside_fraction: float,
    geometry_available: bool,
) -> MapComplianceResult:
    if not geometry_available:
        return MapComplianceResult(
            passed=not required,
            inside_fraction=None,
            outside_indices=tuple(range(len(inside))) if required else (),
            geometry_source="av2_drivable_area_polygon",
            reason="missing_drivable_area_geometry" if required else None,
        )
    fraction = float(inside.mean())
    passed = (not required) or fraction >= minimum_inside_fraction
    return MapComplianceResult(
        passed=passed,
        inside_fraction=fraction,
        outside_indices=tuple(int(index) for index in np.flatnonzero(~inside)),
        geometry_source="av2_drivable_area_polygon",
        reason=None if passed else "outside_drivable_area",
    )


def _lane_transition_allowed(first, second) -> bool:
    if first.lane_id == second.lane_id:
        return True
    first_neighbors = {first.left_neighbor_id, first.right_neighbor_id}
    second_neighbors = {second.left_neighbor_id, second.right_neighbor_id}
    if second.lane_id in first_neighbors or first.lane_id in second_neighbors:
        return True
    return bool(
        second.lane_id in first.successor_ids
        or first.lane_id in second.predecessor_ids
        or first.lane_id in second.successor_ids
        or second.lane_id in first.predecessor_ids
    )


def evaluate_lane_compliance(
    scenario: Scenario,
    target_track_id: str,
    *,
    required: bool,
    minimum_assignment_fraction: float,
    maximum_lane_distance_m: float,
    maximum_heading_error_deg: float,
    direction_exempt: bool,
    prepared_map: PreparedMapGeometry | None = None,
) -> LaneComplianceResult:
    target = next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )
    if target is None or len(target.positions) < 110 or len(target.headings) < 110:
        raise ValueError("target track must contain 110 frames for lane compliance")
    if prepared_map is not None:
        require_compatible_map((scenario,), prepared_map)
        lane_indices, distances, projection_headings = (
            project_points_to_lanes_within_distance(
                prepared_map,
                target.positions[50:110],
                maximum_lane_distance_m,
            )
        )
        return _prepared_lane_result(
            target,
            prepared_map,
            lane_indices,
            distances,
            projection_headings,
            required=required,
            minimum_assignment_fraction=minimum_assignment_fraction,
            maximum_lane_distance_m=maximum_lane_distance_m,
            maximum_heading_error_deg=maximum_heading_error_deg,
            direction_exempt=direction_exempt,
        )
    lanes = [
        lane
        for lane in scenario.map_polylines
        if lane.polyline_type == "lane_centerline"
        and lane.lane_id
        and len(lane.points) >= 2
        and np.isfinite(lane.points).all()
    ]
    if not lanes:
        return LaneComplianceResult(
            passed=not required,
            assignment_fraction=None,
            type_compatibility_fraction=None,
            direction_fraction=None,
            assigned_lane_ids=(None,) * 60,
            assigned_lane_types=(None,) * 60,
            incompatible_lane_indices=(),
            incompatible_lane_types=(),
            invalid_transitions=(),
            direction_exempt=direction_exempt,
            reason="missing_lane_geometry" if required else None,
        )

    assigned = []
    direction_valid: list[bool] = []
    incompatible_types = _INCOMPATIBLE_LANE_TYPES_BY_ACTOR.get(
        target.object_type.strip().lower()
    )
    type_compatible: list[bool] = []
    incompatible_lane_indices: list[int] = []
    incompatible_lane_types: list[str] = []
    for future_index, (position, heading) in enumerate(
        zip(target.positions[50:110], target.headings[50:110])
    ):
        if not np.isfinite(position).all():
            assigned.append(None)
            continue
        candidates = []
        for lane in lanes:
            projection = point_to_polyline_projection(position, lane.points)
            candidates.append((projection.distance_m, lane.lane_id or "", lane, projection))
        distance, _, lane, projection = min(candidates, key=lambda item: (item[0], item[1]))
        if distance > maximum_lane_distance_m:
            assigned.append(None)
            continue
        assigned.append(lane)
        if incompatible_types is not None:
            lane_type = _lane_type(lane)
            compatible = lane_type not in incompatible_types
            type_compatible.append(compatible)
            if not compatible:
                incompatible_lane_indices.append(future_index)
                incompatible_lane_types.append(lane_type)
        if not direction_exempt and np.isfinite(heading) and np.isfinite(projection.heading_rad):
            direction_valid.append(
                np.degrees(heading_difference(float(heading), projection.heading_rad))
                <= maximum_heading_error_deg
            )

    assignment_fraction = float(sum(item is not None for item in assigned) / len(assigned))
    compact = []
    for lane in assigned:
        if lane is not None and (not compact or compact[-1].lane_id != lane.lane_id):
            compact.append(lane)
    invalid_transitions = tuple(
        (first.lane_id or "", second.lane_id or "")
        for first, second in zip(compact, compact[1:])
        if not _lane_transition_allowed(first, second)
    )
    direction_fraction = (
        None
        if direction_exempt
        else (float(np.mean(direction_valid)) if direction_valid else 0.0)
    )
    type_compatibility_fraction = (
        None
        if incompatible_types is None or not type_compatible
        else float(np.mean(type_compatible))
    )
    passed = bool(
        (not required)
        or (
            assignment_fraction >= minimum_assignment_fraction
            and (
                incompatible_types is None
                or type_compatibility_fraction is not None
                and type_compatibility_fraction >= minimum_assignment_fraction
            )
            and not invalid_transitions
            and (
                direction_exempt
                or direction_fraction is not None
                and direction_fraction >= minimum_assignment_fraction
            )
        )
    )
    if passed:
        reason = None
    elif assignment_fraction < minimum_assignment_fraction:
        reason = "insufficient_lane_assignment"
    elif (
        incompatible_types is not None
        and type_compatibility_fraction is not None
        and type_compatibility_fraction < minimum_assignment_fraction
    ):
        reason = "incompatible_lane_type"
    elif invalid_transitions:
        reason = "invalid_lane_transition"
    else:
        reason = "invalid_lane_direction"
    return LaneComplianceResult(
        passed=passed,
        assignment_fraction=assignment_fraction,
        type_compatibility_fraction=type_compatibility_fraction,
        direction_fraction=direction_fraction,
        assigned_lane_ids=tuple(
            None if lane is None else lane.lane_id for lane in assigned
        ),
        assigned_lane_types=tuple(
            None if lane is None else _lane_type(lane) for lane in assigned
        ),
        incompatible_lane_indices=tuple(incompatible_lane_indices),
        incompatible_lane_types=tuple(incompatible_lane_types),
        invalid_transitions=invalid_transitions,
        direction_exempt=direction_exempt,
        reason=reason,
    )


def _prepared_lane_result(
    target,
    prepared_map: PreparedMapGeometry,
    lane_indices: np.ndarray,
    distances: np.ndarray,
    projection_headings: np.ndarray,
    *,
    required: bool,
    minimum_assignment_fraction: float,
    maximum_lane_distance_m: float,
    maximum_heading_error_deg: float,
    direction_exempt: bool,
) -> LaneComplianceResult:
    if not prepared_map.lanes:
        return LaneComplianceResult(
            passed=not required,
            assignment_fraction=None,
            type_compatibility_fraction=None,
            direction_fraction=None,
            assigned_lane_ids=(None,) * 60,
            assigned_lane_types=(None,) * 60,
            incompatible_lane_indices=(),
            incompatible_lane_types=(),
            invalid_transitions=(),
            direction_exempt=direction_exempt,
            reason="missing_lane_geometry" if required else None,
        )
    if not (
        lane_indices.shape == distances.shape == projection_headings.shape == (60,)
    ):
        raise ValueError("prepared lane query must contain exactly 60 future frames")

    assigned: list[PreparedLaneGeometry | None] = []
    direction_valid: list[bool] = []
    incompatible_types = _INCOMPATIBLE_LANE_TYPES_BY_ACTOR.get(
        target.object_type.strip().lower()
    )
    type_compatible: list[bool] = []
    incompatible_lane_indices: list[int] = []
    incompatible_lane_types: list[str] = []
    for future_index, (lane_index, distance, projection_heading, heading) in enumerate(
        zip(
            lane_indices,
            distances,
            projection_headings,
            target.headings[50:110],
        )
    ):
        if lane_index < 0 or distance > maximum_lane_distance_m:
            assigned.append(None)
            continue
        lane = prepared_map.lanes[int(lane_index)]
        assigned.append(lane)
        if incompatible_types is not None:
            compatible = lane.lane_type not in incompatible_types
            type_compatible.append(compatible)
            if not compatible:
                incompatible_lane_indices.append(future_index)
                incompatible_lane_types.append(lane.lane_type)
        if (
            not direction_exempt
            and np.isfinite(heading)
            and np.isfinite(projection_heading)
        ):
            direction_valid.append(
                np.degrees(
                    heading_difference(float(heading), float(projection_heading))
                )
                <= maximum_heading_error_deg
            )

    assignment_fraction = float(sum(item is not None for item in assigned) / len(assigned))
    compact: list[PreparedLaneGeometry] = []
    for lane in assigned:
        if lane is not None and (not compact or compact[-1].lane_id != lane.lane_id):
            compact.append(lane)
    invalid_transitions = tuple(
        (first.lane_id, second.lane_id)
        for first, second in zip(compact, compact[1:])
        if not _lane_transition_allowed(first, second)
    )
    direction_fraction = (
        None
        if direction_exempt
        else (float(np.mean(direction_valid)) if direction_valid else 0.0)
    )
    type_compatibility_fraction = (
        None
        if incompatible_types is None or not type_compatible
        else float(np.mean(type_compatible))
    )
    passed = bool(
        (not required)
        or (
            assignment_fraction >= minimum_assignment_fraction
            and (
                incompatible_types is None
                or type_compatibility_fraction is not None
                and type_compatibility_fraction >= minimum_assignment_fraction
            )
            and not invalid_transitions
            and (
                direction_exempt
                or direction_fraction is not None
                and direction_fraction >= minimum_assignment_fraction
            )
        )
    )
    if passed:
        reason = None
    elif assignment_fraction < minimum_assignment_fraction:
        reason = "insufficient_lane_assignment"
    elif (
        incompatible_types is not None
        and type_compatibility_fraction is not None
        and type_compatibility_fraction < minimum_assignment_fraction
    ):
        reason = "incompatible_lane_type"
    elif invalid_transitions:
        reason = "invalid_lane_transition"
    else:
        reason = "invalid_lane_direction"
    return LaneComplianceResult(
        passed=passed,
        assignment_fraction=assignment_fraction,
        type_compatibility_fraction=type_compatibility_fraction,
        direction_fraction=direction_fraction,
        assigned_lane_ids=tuple(
            None if lane is None else lane.lane_id for lane in assigned
        ),
        assigned_lane_types=tuple(
            None if lane is None else lane.lane_type for lane in assigned
        ),
        incompatible_lane_indices=tuple(incompatible_lane_indices),
        incompatible_lane_types=tuple(incompatible_lane_types),
        invalid_transitions=invalid_transitions,
        direction_exempt=direction_exempt,
        reason=reason,
    )


def check_map_compliance(
    scenario: Scenario,
    target_track_id: str,
    skill_id: str,
    policy: MapFilterPolicy,
    *,
    prepared_map: PreparedMapGeometry | None = None,
    verification_session: PreparedMapVerificationSession | None = None,
) -> FilterCheck:
    target = next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )
    if target is None or len(target.positions) < 110:
        raise ValueError("target track must contain 110 frames for map compliance")
    actor_type = target.object_type.lower()
    drivable_required = actor_type in policy.required_drivable_actor_types
    lane_required = actor_type in policy.lane_required_actor_types
    if prepared_map is None:
        if verification_session is not None:
            raise ValueError("verification_session requires prepared_map")
        drivable = evaluate_drivable_area(
            scenario,
            target.positions[50:110],
            required=drivable_required,
            minimum_inside_fraction=policy.minimum_inside_fraction,
        )
        lane = evaluate_lane_compliance(
            scenario,
            target_track_id,
            required=lane_required,
            minimum_assignment_fraction=policy.minimum_lane_assignment_fraction,
            maximum_lane_distance_m=policy.maximum_lane_distance_m,
            maximum_heading_error_deg=policy.maximum_heading_error_deg,
            direction_exempt=skill_id in policy.direction_exempt_skills,
        )
    else:
        if verification_session is None:
            require_compatible_map((scenario,), prepared_map)
        else:
            verification_session.verify_query(scenario, prepared_map)
        positions = np.asarray(target.positions[50:110], dtype=np.float64)
        if positions.shape != (60, 2):
            raise ValueError("future_positions_global must have shape (N, 2)")
        if not np.isfinite(positions).all():
            raise ValueError("future_positions_global must contain finite values")
        inside = points_in_drivable_area(prepared_map, positions)
        lane_indices, distances, projection_headings = (
            project_points_to_lanes_within_distance(
                prepared_map,
                positions,
                policy.maximum_lane_distance_m,
            )
        )
        drivable, lane = _prepared_compliance_results(
            target,
            skill_id,
            policy,
            prepared_map,
            inside,
            lane_indices,
            distances,
            projection_headings,
        )
    return _map_filter_check(
        actor_type,
        drivable_required,
        lane_required,
        drivable,
        lane,
        policy,
    )


def _prepared_compliance_results(
    target,
    skill_id: str,
    policy: MapFilterPolicy,
    prepared_map: PreparedMapGeometry,
    inside: np.ndarray,
    lane_indices: np.ndarray,
    distances: np.ndarray,
    projection_headings: np.ndarray,
) -> tuple[MapComplianceResult, LaneComplianceResult]:
    actor_type = target.object_type.lower()
    drivable_required = actor_type in policy.required_drivable_actor_types
    lane_required = actor_type in policy.lane_required_actor_types
    return (
        _drivable_result_from_membership(
            inside,
            required=drivable_required,
            minimum_inside_fraction=policy.minimum_inside_fraction,
            geometry_available=prepared_map.drivable_area_count > 0,
        ),
        _prepared_lane_result(
            target,
            prepared_map,
            lane_indices,
            distances,
            projection_headings,
            required=lane_required,
            minimum_assignment_fraction=policy.minimum_lane_assignment_fraction,
            maximum_lane_distance_m=policy.maximum_lane_distance_m,
            maximum_heading_error_deg=policy.maximum_heading_error_deg,
            direction_exempt=skill_id in policy.direction_exempt_skills,
        ),
    )


def _map_filter_check(
    actor_type: str,
    drivable_required: bool,
    lane_required: bool,
    drivable: MapComplianceResult,
    lane: LaneComplianceResult,
    policy: MapFilterPolicy,
) -> FilterCheck:
    reasons: list[FilterRejection] = []
    if drivable_required and drivable.inside_fraction is None:
        reasons.append(FilterRejection.DRIVABLE_AREA_UNAVAILABLE)
    elif not drivable.passed:
        reasons.append(FilterRejection.OUTSIDE_DRIVABLE_AREA)
    if lane_required and lane.assignment_fraction is None:
        reasons.append(FilterRejection.LANE_GEOMETRY_UNAVAILABLE)
    elif lane_required and lane.assignment_fraction < policy.minimum_lane_assignment_fraction:
        reasons.append(FilterRejection.LANE_ASSIGNMENT_INSUFFICIENT)
    elif (
        lane_required
        and lane.type_compatibility_fraction is not None
        and lane.type_compatibility_fraction < policy.minimum_lane_assignment_fraction
    ):
        reasons.append(FilterRejection.LANE_TYPE_INCOMPATIBLE)
    elif lane.invalid_transitions:
        reasons.append(FilterRejection.LANE_CONNECTIVITY_VIOLATION)
    elif (
        lane_required
        and not lane.direction_exempt
        and lane.direction_fraction is not None
        and lane.direction_fraction < policy.minimum_lane_assignment_fraction
    ):
        reasons.append(FilterRejection.LANE_DIRECTION_VIOLATION)
    return FilterCheck(
        stage=FilterStage.MAP,
        rejection_reasons=tuple(reasons),
        metrics={
            "policy_source": policy.source,
            "actor_type": actor_type,
            "drivable_area_geometry_source": drivable.geometry_source,
            "drivable_area_required": drivable_required,
            "inside_drivable_area_fraction": drivable.inside_fraction,
            "outside_drivable_area_indices": list(drivable.outside_indices),
            "lane_required": lane_required,
            "lane_assignment_fraction": lane.assignment_fraction,
            "lane_type_compatibility_fraction": lane.type_compatibility_fraction,
            "lane_direction_fraction": lane.direction_fraction,
            "direction_exempt": lane.direction_exempt,
            "invalid_lane_transitions": [list(item) for item in lane.invalid_transitions],
            "assigned_lane_ids": list(lane.assigned_lane_ids),
            "assigned_lane_types": list(lane.assigned_lane_types),
            "incompatible_lane_indices": list(lane.incompatible_lane_indices),
            "incompatible_lane_types": list(lane.incompatible_lane_types),
        },
    )


def check_map_compliance_batch(
    scenarios: Sequence[Scenario],
    target_track_ids: Sequence[str],
    skill_ids: Sequence[str],
    policy: MapFilterPolicy,
    *,
    prepared_map: PreparedMapGeometry,
    verification_session: PreparedMapVerificationSession | None = None,
) -> tuple[FilterCheck, ...]:
    """Check generated copies sharing one source map in one vectorized query."""

    scenario_values = tuple(scenarios)
    target_values = tuple(target_track_ids)
    skill_values = tuple(skill_ids)
    if not (
        len(scenario_values) == len(target_values) == len(skill_values)
    ):
        raise ValueError(
            "scenarios, target_track_ids and skill_ids must have equal lengths"
        )
    if not scenario_values:
        return ()
    if verification_session is None:
        require_compatible_map(scenario_values, prepared_map)
    else:
        for scenario in scenario_values:
            verification_session.verify_query(scenario, prepared_map)

    targets = []
    futures = []
    for scenario, target_track_id in zip(scenario_values, target_values):
        target = next(
            (agent for agent in scenario.agents if agent.track_id == target_track_id),
            None,
        )
        if target is None or len(target.positions) < 110 or len(target.headings) < 110:
            raise ValueError("target track must contain 110 frames for map compliance")
        positions = np.asarray(target.positions[50:110], dtype=np.float64)
        if positions.shape != (60, 2):
            raise ValueError("future_positions_global must have shape (N, 2)")
        if not np.isfinite(positions).all():
            raise ValueError("future_positions_global must contain finite values")
        targets.append(target)
        futures.append(positions)

    all_positions = np.concatenate(futures, axis=0)
    all_inside = points_in_drivable_area(prepared_map, all_positions)
    all_lane_indices, all_distances, all_projection_headings = (
        project_points_to_lanes_within_distance(
            prepared_map,
            all_positions,
            policy.maximum_lane_distance_m,
        )
    )
    checks = []
    for index, (target, skill_id) in enumerate(zip(targets, skill_values)):
        start = index * 60
        stop = start + 60
        drivable, lane = _prepared_compliance_results(
            target,
            skill_id,
            policy,
            prepared_map,
            all_inside[start:stop],
            all_lane_indices[start:stop],
            all_distances[start:stop],
            all_projection_headings[start:stop],
        )
        actor_type = target.object_type.lower()
        checks.append(
            _map_filter_check(
                actor_type,
                actor_type in policy.required_drivable_actor_types,
                actor_type in policy.lane_required_actor_types,
                drivable,
                lane,
                policy,
            )
        )
    return tuple(checks)


__all__ = [
    "LaneComplianceResult",
    "MapComplianceResult",
    "PreparedMapGeometry",
    "check_map_compliance",
    "check_map_compliance_batch",
    "evaluate_drivable_area",
    "evaluate_lane_compliance",
    "point_in_polygon",
    "prepare_map_geometry",
]

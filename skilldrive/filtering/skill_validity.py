"""Exact-role post-generation validators for all 34 formal skills.

Observed-trigger skills reuse the frozen detector on only the requested role
tracks.  Compatible-seed skills never rerun their seed detector as proof of a
generated event: every ``missing_generation_conditions`` item is evaluated by
an explicit trajectory rule, while static topology evidence may be reused from
the frozen seed record and is labelled as such in the audit output.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.filtering.map import point_in_polygon
from skilldrive.filtering.observed import validate_observed_skill
from skilldrive.filtering.risk import RISK_CONTEXT_METADATA_KEY
from skilldrive.filtering.roles import validate_role_contract
from skilldrive.schemas import AgentTrack, Scenario, SkillSpec
from skilldrive.skills.detection import DetectionConfig
from skilldrive.skills.geometry import (
    find_trajectory_conflict,
    heading_difference,
    minimum_trajectory_distance,
    point_to_polyline_projection,
    trajectory_acceleration,
    trajectory_speed,
)


_HISTORY_END_FRAME = 49
_FUTURE_START_FRAME = 50
_TOTAL_STEPS = 110
_CONFLICT_HALF_EXTENT_M = 2.0
_CONFLICT_REACH_RADIUS_M = 6.0
_LATERAL_EVENT_M = 1.0
_MIN_PROGRESS_M = 0.5
_EPS = 1e-9

DetectionMode = Literal["observed_trigger", "compatible_seed"]
ConditionEvidence = dict[str, Any]


@dataclass(frozen=True)
class _ValidationContext:
    source_scenario: Scenario
    generated_scenario: Scenario
    skill: SkillSpec
    role_track_ids: Mapping[str, str]
    seed_evidence: Mapping[str, Any]
    agents_by_role: Mapping[str, AgentTrack]
    source_agents_by_role: Mapping[str, AgentTrack]


ConditionValidator = Callable[[_ValidationContext], ConditionEvidence]


@dataclass(frozen=True)
class SkillValidatorSpec:
    """Frozen dispatch contract for one formal skill."""

    skill_id: str
    detection_mode: DetectionMode
    required_roles: tuple[str, ...]
    condition_validators: Mapping[str, ConditionValidator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("skill_id must be non-empty")
        if self.detection_mode not in {"observed_trigger", "compatible_seed"}:
            raise ValueError(f"unsupported detection mode: {self.detection_mode}")
        if not self.required_roles or len(set(self.required_roles)) != len(
            self.required_roles
        ):
            raise ValueError(f"{self.skill_id} roles must be unique and non-empty")
        validators = dict(self.condition_validators)
        if self.detection_mode == "observed_trigger" and validators:
            raise ValueError("observed validators must use exact-role re-detection")
        if self.detection_mode == "compatible_seed" and not validators:
            raise ValueError("compatible validators require explicit condition rules")
        if any(
            not name or not callable(validator)
            for name, validator in validators.items()
        ):
            raise ValueError(f"{self.skill_id} contains an invalid condition validator")
        object.__setattr__(self, "condition_validators", MappingProxyType(validators))


def _result(passed: bool, source: str, **details: Any) -> ConditionEvidence:
    return {"passed": bool(passed), "source": source, **details}


def _finite_scalar(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _finite_point(value: Any) -> np.ndarray | None:
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if point.shape != (2,) or not np.isfinite(point).all():
        return None
    return point


def _seed_evidence_sha256(seed_evidence: Mapping[str, Any]) -> str | None:
    try:
        payload = json.dumps(
            seed_evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _times_s(scenario: Scenario) -> np.ndarray:
    timestamps = np.asarray(scenario.timestamps[:_TOTAL_STEPS], dtype=np.float64)
    if len(timestamps) != _TOTAL_STEPS or not np.isfinite(timestamps).all():
        return np.arange(_TOTAL_STEPS, dtype=np.float64) * 0.1
    elapsed = (timestamps - timestamps[_HISTORY_END_FRAME]) / 1_000_000_000.0
    if np.any(np.diff(elapsed) <= 0.0):
        return np.arange(_TOTAL_STEPS, dtype=np.float64) * 0.1
    return elapsed


def _speeds(scenario: Scenario, agent: AgentTrack) -> np.ndarray:
    return trajectory_speed(agent.positions[:_TOTAL_STEPS], _times_s(scenario))


def _accelerations(scenario: Scenario, agent: AgentTrack) -> np.ndarray:
    return trajectory_acceleration(agent.positions[:_TOTAL_STEPS], _times_s(scenario))


def _axis(agent: AgentTrack, frame: int = _HISTORY_END_FRAME) -> np.ndarray | None:
    if len(agent.headings) > frame and math.isfinite(float(agent.headings[frame])):
        heading = float(agent.headings[frame])
        return np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    if len(agent.velocities) > frame and np.isfinite(agent.velocities[frame]).all():
        speed = float(np.linalg.norm(agent.velocities[frame]))
        if speed > _EPS:
            return np.asarray(agent.velocities[frame], dtype=np.float64) / speed
    if frame > 0 and np.isfinite(agent.positions[frame - 1 : frame + 1]).all():
        delta = agent.positions[frame] - agent.positions[frame - 1]
        length = float(np.linalg.norm(delta))
        if length > _EPS:
            return delta / length
    return None


def _threshold(skill: SkillSpec, name: str) -> float | None:
    value = skill.detection.get("thresholds", {}).get(name)
    if isinstance(value, Mapping):
        value = value.get("value")
    return _finite_scalar(value)


def _parameter_bound(
    skill: SkillSpec,
    name: str,
    index: int,
) -> float | None:
    value = skill.parameters.get(name)
    if not isinstance(value, Mapping):
        return None
    bounds = value.get("range")
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
        return None
    if len(bounds) != 2:
        return None
    return _finite_scalar(bounds[index])


def _risk_upper(skill: SkillSpec) -> float | None:
    target = skill.risk_definition.get("target_range")
    if not isinstance(target, Sequence) or isinstance(target, (str, bytes)):
        return None
    if len(target) != 2:
        return None
    return _finite_scalar(target[1])


def _structural_seed_condition(
    *aliases: str,
    fields: tuple[str, ...] = (),
) -> ConditionValidator:
    """Accept only auditable, frozen structural evidence from seed detection."""

    def validate(context: _ValidationContext) -> ConditionEvidence:
        raw_matched = context.seed_evidence.get("matched_conditions", ())
        if isinstance(raw_matched, Sequence) and not isinstance(
            raw_matched, (str, bytes)
        ):
            matched = {str(value) for value in raw_matched}
        else:
            matched = set()
        matched_aliases = sorted(set(aliases) & matched)
        field_evidence = {
            name: context.seed_evidence[name]
            for name in fields
            if name in context.seed_evidence
            and context.seed_evidence[name] is not None
            and context.seed_evidence[name] is not False
        }
        passed = bool(matched_aliases or field_evidence)
        return _result(
            passed,
            "frozen_seed_structural_evidence",
            compatible_seed_detector_reused=False,
            matched_condition_aliases=matched_aliases,
            structural_fields=field_evidence,
        )

    return validate


def _minimum_distance(
    context: _ValidationContext,
    first_role: str,
    second_role: str,
) -> tuple[float | None, float | None]:
    first = context.agents_by_role[first_role]
    second = context.agents_by_role[second_role]
    result = minimum_trajectory_distance(
        first.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
        second.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
        _times_s(context.generated_scenario)[_HISTORY_END_FRAME:_TOTAL_STEPS],
    )
    if result is None:
        return None, None
    return float(result.distance_m), float(result.frame_index + _HISTORY_END_FRAME)


def _distance_series(
    context: _ValidationContext,
    first_role: str,
    second_role: str,
) -> np.ndarray:
    first = context.agents_by_role[first_role].positions[:_TOTAL_STEPS]
    second = context.agents_by_role[second_role].positions[:_TOTAL_STEPS]
    valid = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    result = np.full(_TOTAL_STEPS, np.nan, dtype=np.float64)
    result[valid] = np.linalg.norm(first[valid] - second[valid], axis=1)
    return result


def _seed_conflict_point(context: _ValidationContext) -> np.ndarray | None:
    return _finite_point(context.seed_evidence.get("conflict_point_xy"))


def _map_lane_conflict_point(context: _ValidationContext) -> np.ndarray | None:
    first_lane_id = context.seed_evidence.get("initiator_lane_id")
    second_lane_id = context.seed_evidence.get("responder_lane_id")
    if first_lane_id is None or second_lane_id is None:
        return None
    lanes = {
        str(polyline.lane_id): polyline
        for polyline in context.generated_scenario.map_polylines
        if polyline.polyline_type == "lane_centerline"
        and polyline.lane_id is not None
        and len(polyline.points) >= 2
    }
    first = lanes.get(str(first_lane_id))
    second = lanes.get(str(second_lane_id))
    if first is None or second is None:
        return None
    conflict = find_trajectory_conflict(first.points, second.points)
    return None if conflict is None else np.asarray(conflict.point, dtype=np.float64)


def _trajectory_conflict_point(
    context: _ValidationContext,
    first_role: str,
    second_role: str,
    *,
    use_source: bool = False,
) -> np.ndarray | None:
    agents = context.source_agents_by_role if use_source else context.agents_by_role
    first = agents[first_role]
    second = agents[second_role]
    conflict = find_trajectory_conflict(
        first.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
        second.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
    )
    return None if conflict is None else np.asarray(conflict.point, dtype=np.float64)


def _conflict_point(
    context: _ValidationContext,
    first_role: str,
    second_role: str,
) -> tuple[np.ndarray | None, str]:
    point = _seed_conflict_point(context)
    if point is not None:
        return point, "frozen_seed_conflict_point"
    point = _map_lane_conflict_point(context)
    if point is not None:
        return point, "frozen_seed_lane_map_intersection"
    point = _trajectory_conflict_point(context, first_role, second_role)
    if point is not None:
        return point, "generated_path_intersection"
    point = _trajectory_conflict_point(
        context,
        first_role,
        second_role,
        use_source=True,
    )
    if point is not None:
        return point, "source_path_intersection"
    first = context.agents_by_role[first_role]
    second = context.agents_by_role[second_role]
    closest = minimum_trajectory_distance(
        first.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
        second.positions[_HISTORY_END_FRAME:_TOTAL_STEPS],
    )
    if closest is None or closest.distance_m > _CONFLICT_REACH_RADIUS_M:
        return None, "unavailable"
    return (
        (np.asarray(closest.first_point) + np.asarray(closest.second_point)) / 2.0,
        "generated_closest_approach_proxy",
    )


def _arrival_to_point(
    context: _ValidationContext,
    role: str,
    point: np.ndarray,
) -> tuple[float | None, float | None, int | None]:
    positions = context.agents_by_role[role].positions[_FUTURE_START_FRAME:_TOTAL_STEPS]
    valid = np.isfinite(positions).all(axis=1)
    if not valid.any():
        return None, None, None
    indices = np.flatnonzero(valid)
    distances = np.linalg.norm(positions[indices] - point, axis=1)
    local = int(indices[int(np.argmin(distances))])
    frame = local + _FUTURE_START_FRAME
    return (
        float(_times_s(context.generated_scenario)[frame]),
        float(np.min(distances)),
        frame,
    )


def _lateral_displacement(
    agent: AgentTrack,
    reference: AgentTrack | None = None,
) -> np.ndarray:
    basis = reference or agent
    axis = _axis(basis)
    result = np.full(_TOTAL_STEPS, np.nan, dtype=np.float64)
    if axis is None:
        return result
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    if reference is None:
        origin = agent.positions[_HISTORY_END_FRAME]
        relative = agent.positions[:_TOTAL_STEPS] - origin
    else:
        relative = agent.positions[:_TOTAL_STEPS] - reference.positions[:_TOTAL_STEPS]
        relative -= relative[_HISTORY_END_FRAME]
    valid = np.isfinite(relative).all(axis=1)
    result[valid] = relative[valid] @ normal
    return result


def _lateral_event_frame(
    actor: AgentTrack,
    reference: AgentTrack | None = None,
    *,
    threshold_m: float = _LATERAL_EVENT_M,
) -> int | None:
    lateral = np.abs(_lateral_displacement(actor, reference))
    indices = np.flatnonzero(
        np.isfinite(lateral[_FUTURE_START_FRAME:])
        & (lateral[_FUTURE_START_FRAME:] >= threshold_m)
    )
    return None if not len(indices) else int(indices[0] + _FUTURE_START_FRAME)


def _cut_out_exposure_frame(context: _ValidationContext) -> int | None:
    return _lateral_event_frame(
        context.agents_by_role["cut_out_vehicle"],
        context.agents_by_role["target_vehicle"],
    )


def _cut_in_frame(context: _ValidationContext) -> int | None:
    actor = context.agents_by_role["cut_in_braking_vehicle"]
    responder = context.agents_by_role["responding_vehicle"]
    axis = _axis(responder)
    if axis is None:
        return None
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    relative = actor.positions[:_TOTAL_STEPS] - responder.positions[:_TOTAL_STEPS]
    valid = np.isfinite(relative).all(axis=1)
    lateral = np.full(_TOTAL_STEPS, np.nan, dtype=np.float64)
    longitudinal = np.full(_TOTAL_STEPS, np.nan, dtype=np.float64)
    lateral[valid] = relative[valid] @ normal
    longitudinal[valid] = relative[valid] @ axis
    start = abs(float(lateral[_HISTORY_END_FRAME]))
    if not math.isfinite(start) or start < _LATERAL_EVENT_M:
        return None
    threshold = max(0.75, 0.35 * start)
    candidates = np.flatnonzero(
        np.isfinite(lateral[_FUTURE_START_FRAME:])
        & np.isfinite(longitudinal[_FUTURE_START_FRAME:])
        & (np.abs(lateral[_FUTURE_START_FRAME:]) <= threshold)
        & (longitudinal[_FUTURE_START_FRAME:] > 0.0)
    )
    return None if not len(candidates) else int(candidates[0] + _FUTURE_START_FRAME)


def _brake_onset_frame(
    context: _ValidationContext,
    role: str,
    *,
    start_frame: int = _FUTURE_START_FRAME,
    minimum_deceleration_mps2: float,
) -> int | None:
    acceleration = _accelerations(
        context.generated_scenario,
        context.agents_by_role[role],
    )
    candidates = np.flatnonzero(
        np.isfinite(acceleration[start_frame:])
        & (acceleration[start_frame:] <= -minimum_deceleration_mps2)
    )
    return None if not len(candidates) else int(candidates[0] + start_frame)


def _cut_in_stage_start_frame(context: _ValidationContext) -> int | None:
    cut_in = _cut_in_frame(context)
    if cut_in is None:
        return None
    threshold = _parameter_bound(context.skill, "peak_deceleration_mps2", 0) or 2.0
    return _brake_onset_frame(
        context,
        "cut_in_braking_vehicle",
        start_frame=cut_in,
        minimum_deceleration_mps2=threshold,
    )


def _square_polygon(center: np.ndarray) -> list[list[float]]:
    x, y = float(center[0]), float(center[1])
    half = _CONFLICT_HALF_EXTENT_M
    return [
        [x - half, y - half],
        [x + half, y - half],
        [x + half, y + half],
        [x - half, y + half],
    ]


_CONTEXT_POLYGON_ROLES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "intersection_creep_conflict": ("creeping_vehicle", "crossing_vehicle"),
        "intersection_blocking_vehicle": ("blocking_vehicle", "crossing_vehicle"),
        "roadside_pedestrian_emergence": (
            "emerging_pedestrian",
            "responding_vehicle",
        ),
    }
)


def prepare_risk_context(
    source_scenario: Scenario,
    generated_scenario: Scenario,
    skill: SkillSpec,
    role_track_ids: Mapping[str, str],
    seed_evidence: Mapping[str, Any],
) -> Scenario:
    """Return a non-mutating scenario copy with explicit risk event context.

    Context is derived from the generated overlay whenever it describes an
    event frame.  A frozen seed conflict point may define static geometry, but
    seed-time risk scalars are never copied into the risk calculator input.
    """

    if skill.skill_id not in SKILL_VALIDATORS:
        raise KeyError(f"unknown formal skill_id: {skill.skill_id}")
    spec = SKILL_VALIDATORS[skill.skill_id]
    normalized = {str(role): str(track_id) for role, track_id in role_track_ids.items()}
    generated_agents = {agent.track_id: agent for agent in generated_scenario.agents}
    source_agents = {agent.track_id: agent for agent in source_scenario.agents}
    if set(normalized) != set(spec.required_roles):
        return generated_scenario
    if any(track_id not in generated_agents for track_id in normalized.values()):
        return generated_scenario
    if any(track_id not in source_agents for track_id in normalized.values()):
        return generated_scenario
    context = _ValidationContext(
        source_scenario=source_scenario,
        generated_scenario=generated_scenario,
        skill=skill,
        role_track_ids=MappingProxyType(normalized),
        seed_evidence=seed_evidence,
        agents_by_role=MappingProxyType(
            {role: generated_agents[track_id] for role, track_id in normalized.items()}
        ),
        source_agents_by_role=MappingProxyType(
            {role: source_agents[track_id] for role, track_id in normalized.items()}
        ),
    )

    item: dict[str, Any] = {
        "context_version": "skill_validity.risk_context.v1",
        "source_scenario_id": source_scenario.scenario_id,
        "generated_scenario_id": generated_scenario.scenario_id,
        "seed_evidence_sha256": _seed_evidence_sha256(seed_evidence),
    }
    if skill.skill_id == "cut_out_reveals_slow_vehicle":
        frame = _cut_out_exposure_frame(context)
        item.update(
            {
                "preparation_status": (
                    "computed" if frame is not None else "unavailable"
                ),
                "event_source": "generated_exact_role_lateral_departure",
            }
        )
        if frame is not None:
            item["exposure_frame_index"] = frame
    elif skill.skill_id == "cut_in_then_brake":
        frame = _cut_in_stage_start_frame(context)
        item.update(
            {
                "preparation_status": (
                    "computed" if frame is not None else "unavailable"
                ),
                "event_source": "generated_exact_role_post_cut_in_brake_onset",
            }
        )
        if frame is not None:
            item["stage_start_frame_index"] = frame
    elif skill.skill_id in _CONTEXT_POLYGON_ROLES:
        roles = _CONTEXT_POLYGON_ROLES[skill.skill_id]
        point, source = _conflict_point(context, *roles)
        item.update(
            {
                "preparation_status": (
                    "computed" if point is not None else "unavailable"
                ),
                "event_source": source,
                "geometry_contract": "axis_aligned_4m_conflict_proxy.v1",
            }
        )
        if point is not None:
            item["conflict_point_xy"] = point.tolist()
            item["conflict_area_polygon_xy"] = _square_polygon(point)
    else:
        return generated_scenario

    metadata = dict(generated_scenario.metadata)
    existing = metadata.get(RISK_CONTEXT_METADATA_KEY, {})
    root = dict(existing) if isinstance(existing, Mapping) else {}
    root[skill.skill_id] = item
    metadata[RISK_CONTEXT_METADATA_KEY] = root
    return Scenario(
        scenario_id=generated_scenario.scenario_id,
        city_name=generated_scenario.city_name,
        timestamps=generated_scenario.timestamps.copy(),
        focal_track_id=generated_scenario.focal_track_id,
        agents=list(generated_scenario.agents),
        map_polylines=list(generated_scenario.map_polylines),
        metadata=metadata,
    )


def _leader_decelerating(context: _ValidationContext) -> ConditionEvidence:
    threshold = _parameter_bound(context.skill, "peak_deceleration_mps2", 0) or 2.0
    acceleration = _accelerations(
        context.generated_scenario,
        context.agents_by_role["front_vehicle"],
    )[_FUTURE_START_FRAME:]
    finite = acceleration[np.isfinite(acceleration)]
    minimum = None if not len(finite) else float(np.min(finite))
    return _result(
        minimum is not None and minimum <= -threshold,
        "generated_exact_role_kinematics",
        minimum_acceleration_mps2=minimum,
        required_deceleration_mps2=threshold,
    )


def _delayed_braking_response(context: _ValidationContext) -> ConditionEvidence:
    front_threshold = (
        _parameter_bound(context.skill, "peak_deceleration_mps2", 0) or 2.0
    )
    response_threshold = min(1.0, front_threshold / 2.0)
    frames = {
        "front_vehicle": _brake_onset_frame(
            context,
            "front_vehicle",
            minimum_deceleration_mps2=front_threshold,
        ),
        "middle_vehicle": _brake_onset_frame(
            context,
            "middle_vehicle",
            minimum_deceleration_mps2=response_threshold,
        ),
        "rear_vehicle": _brake_onset_frame(
            context,
            "rear_vehicle",
            minimum_deceleration_mps2=response_threshold,
        ),
    }
    minimum_delay = _parameter_bound(context.skill, "propagation_delay_s", 0) or 0.2
    times = _times_s(context.generated_scenario)
    if any(frame is None for frame in frames.values()):
        delays: list[float] = []
        passed = False
    else:
        front = int(frames["front_vehicle"])
        middle = int(frames["middle_vehicle"])
        rear = int(frames["rear_vehicle"])
        delays = [
            float(times[middle] - times[front]),
            float(times[rear] - times[middle]),
        ]
        passed = front < middle < rear and all(
            delay >= minimum_delay for delay in delays
        )
    return _result(
        passed,
        "generated_exact_role_kinematics",
        brake_onset_frames=frames,
        propagation_delays_s=delays,
        minimum_delay_s=minimum_delay,
    )


def _lead_vehicle_cuts_out(context: _ValidationContext) -> ConditionEvidence:
    frame = _cut_out_exposure_frame(context)
    return _result(
        frame is not None,
        "generated_exact_role_lateral_motion",
        exposure_frame_index=frame,
        minimum_lateral_departure_m=_LATERAL_EVENT_M,
    )


def _newly_exposed_slow_vehicle(context: _ValidationContext) -> ConditionEvidence:
    frame = _cut_out_exposure_frame(context)
    if frame is None:
        return _result(
            False, "generated_exact_role_geometry", reason="no_cut_out_event"
        )
    slow = context.agents_by_role["slow_vehicle"]
    target = context.agents_by_role["target_vehicle"]
    cut_out = context.agents_by_role["cut_out_vehicle"]
    axis = _axis(target, frame)
    if axis is None:
        return _result(
            False, "generated_exact_role_geometry", reason="heading_unavailable"
        )
    slow_gap = float(np.dot(slow.positions[frame] - target.positions[frame], axis))
    cut_out_lateral = abs(
        float(
            np.dot(
                cut_out.positions[frame] - target.positions[frame],
                np.array([-axis[1], axis[0]]),
            )
        )
    )
    slow_speed = float(np.linalg.norm(slow.velocities[frame]))
    maximum_slow_speed = (
        _threshold(context.skill, "maximum_slow_vehicle_speed_mps") or 2.0
    )
    passed = (
        slow_gap > 0.0
        and cut_out_lateral >= _LATERAL_EVENT_M
        and slow_speed <= maximum_slow_speed
    )
    return _result(
        passed,
        "generated_exact_role_geometry",
        exposure_frame_index=frame,
        slow_vehicle_longitudinal_gap_m=slow_gap,
        cut_out_lateral_separation_m=cut_out_lateral,
        slow_vehicle_speed_mps=slow_speed,
        maximum_slow_vehicle_speed_mps=maximum_slow_speed,
    )


def _lane_change_window(agent: AgentTrack) -> tuple[int, int] | None:
    lateral = np.abs(_lateral_displacement(agent))
    indices = np.flatnonzero(
        np.isfinite(lateral[_FUTURE_START_FRAME:])
        & (lateral[_FUTURE_START_FRAME:] >= _LATERAL_EVENT_M)
    )
    if not len(indices):
        return None
    frame = int(indices[0] + _FUTURE_START_FRAME)
    return frame, _TOTAL_STEPS - 1


def _overlapping_lane_change_window(context: _ValidationContext) -> ConditionEvidence:
    first = _lane_change_window(context.agents_by_role["left_lane_changer"])
    second = _lane_change_window(context.agents_by_role["right_lane_changer"])
    passed = (
        first is not None
        and second is not None
        and max(first[0], second[0]) <= min(first[1], second[1])
    )
    return _result(
        passed,
        "generated_exact_role_lateral_motion",
        left_window=None if first is None else list(first),
        right_window=None if second is None else list(second),
    )


def _shared_target_lane(context: _ValidationContext) -> ConditionEvidence:
    target_lane_id = context.seed_evidence.get("shared_target_lane_id")
    if target_lane_id is None:
        return _result(
            False,
            "frozen_seed_structure_plus_generated_lane_occupancy",
            reason="shared_target_lane_id_unavailable",
        )
    lane = next(
        (
            polyline
            for polyline in context.generated_scenario.map_polylines
            if polyline.polyline_type == "lane_centerline"
            and str(polyline.lane_id) == str(target_lane_id)
            and len(polyline.points) >= 2
        ),
        None,
    )
    if lane is None:
        return _result(
            False,
            "frozen_seed_structure_plus_generated_lane_occupancy",
            shared_target_lane_id=str(target_lane_id),
            reason="shared_target_lane_geometry_unavailable",
        )
    maximum_distance = 2.5
    distances: dict[str, float | None] = {}
    for role in ("left_lane_changer", "right_lane_changer"):
        positions = context.agents_by_role[role].positions[_FUTURE_START_FRAME:]
        values = [
            point_to_polyline_projection(point, lane.points).distance_m
            for point in positions
            if np.isfinite(point).all()
        ]
        distances[role] = None if not values else float(min(values))
    passed = all(
        value is not None and value <= maximum_distance for value in distances.values()
    )
    return _result(
        passed,
        "frozen_seed_structure_plus_generated_lane_occupancy",
        compatible_seed_detector_reused=False,
        shared_target_lane_id=str(target_lane_id),
        minimum_lane_distances_m=distances,
        maximum_lane_distance_m=maximum_distance,
    )


def _close_longitudinal_position(context: _ValidationContext) -> ConditionEvidence:
    first = context.agents_by_role["left_lane_changer"]
    second = context.agents_by_role["right_lane_changer"]
    axis = _axis(first)
    if axis is None:
        return _result(
            False, "generated_exact_role_geometry", reason="heading_unavailable"
        )
    relative = (
        second.positions[_FUTURE_START_FRAME:] - first.positions[_FUTURE_START_FRAME:]
    )
    valid = np.isfinite(relative).all(axis=1)
    gaps = np.abs(relative[valid] @ axis)
    minimum = None if not len(gaps) else float(np.min(gaps))
    maximum = _threshold(context.skill, "maximum_longitudinal_gap_m") or 12.0
    return _result(
        minimum is not None and minimum <= maximum,
        "generated_exact_role_geometry",
        minimum_longitudinal_gap_m=minimum,
        maximum_longitudinal_gap_m=maximum,
    )


def _blockage_ahead(context: _ValidationContext) -> ConditionEvidence:
    structural = _structural_seed_condition("blockage_ahead")(context)
    blocker = context.agents_by_role["blocking_actor"]
    vehicle = context.agents_by_role["avoiding_vehicle"]
    axis = _axis(vehicle)
    if axis is None:
        return _result(False, "generated_and_frozen_structure", structural=structural)
    gap = float(
        np.dot(
            blocker.positions[_HISTORY_END_FRAME]
            - vehicle.positions[_HISTORY_END_FRAME],
            axis,
        )
    )
    speed = float(np.linalg.norm(blocker.velocities[_HISTORY_END_FRAME]))
    maximum_distance = _threshold(context.skill, "maximum_blockage_distance_m") or 35.0
    maximum_speed = _threshold(context.skill, "maximum_blocker_speed_mps") or 0.5
    passed = (
        structural["passed"]
        and 0.0 < gap <= maximum_distance
        and speed <= maximum_speed
    )
    return _result(
        passed,
        "generated_roles_plus_frozen_seed_structure",
        frozen_structure=structural,
        blocker_longitudinal_gap_m=gap,
        blocker_speed_mps=speed,
        maximum_blockage_distance_m=maximum_distance,
        maximum_blocker_speed_mps=maximum_speed,
    )


def _adjacent_lane_change_realized(context: _ValidationContext) -> ConditionEvidence:
    structural = _structural_seed_condition(
        "adjacent_lane_available",
        fields=("adjacent_lanes",),
    )(context)
    original_lane_id = context.seed_evidence.get("responder_lane_id")
    if original_lane_id is None:
        return _result(
            False,
            "frozen_seed_lane_plus_generated_target_lane_occupancy",
            frozen_structure=structural,
            reason="frozen_avoiding_vehicle_lane_unavailable",
        )
    original_lane_id = str(original_lane_id)
    lanes = {
        str(lane.lane_id): lane
        for lane in context.generated_scenario.map_polylines
        if lane.polyline_type == "lane_centerline"
        and lane.lane_id is not None
        and len(lane.points) >= 2
        and np.isfinite(lane.points).all()
    }
    original_lane = lanes.get(original_lane_id)
    if original_lane is None:
        return _result(
            False,
            "frozen_seed_lane_plus_generated_target_lane_occupancy",
            frozen_structure=structural,
            frozen_original_lane_id=original_lane_id,
            reason="frozen_avoiding_vehicle_lane_geometry_unavailable",
        )
    adjacent_lane_ids = sorted(
        {
            str(lane_id)
            for lane_id in (
                original_lane.left_neighbor_id,
                original_lane.right_neighbor_id,
            )
            if lane_id is not None and str(lane_id) in lanes
        }
    )
    if not adjacent_lane_ids:
        return _result(
            False,
            "frozen_seed_lane_plus_generated_target_lane_occupancy",
            frozen_structure=structural,
            frozen_original_lane_id=original_lane_id,
            reason="direct_adjacent_lane_geometry_unavailable",
        )

    def nearest_lane(point: np.ndarray) -> tuple[str, float]:
        distance, lane_id = min(
            (
                point_to_polyline_projection(point, lane.points).distance_m,
                lane_id,
            )
            for lane_id, lane in lanes.items()
        )
        return lane_id, float(distance)

    vehicle = context.agents_by_role["avoiding_vehicle"]
    blocker = context.agents_by_role["blocking_actor"]
    anchor_lane_id, anchor_lane_distance = nearest_lane(
        vehicle.positions[_HISTORY_END_FRAME]
    )
    compact_lane_sequence = [anchor_lane_id]
    transition_frame = None
    transition_from_lane_id = None
    transition_lane_id = None
    previous_lane_id = anchor_lane_id
    for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS):
        lane_id, _ = nearest_lane(vehicle.positions[frame])
        if lane_id == previous_lane_id:
            continue
        compact_lane_sequence.append(lane_id)
        previous_lane = lanes[previous_lane_id]
        previous_adjacent_lane_ids = {
            str(neighbor_id)
            for neighbor_id in (
                previous_lane.left_neighbor_id,
                previous_lane.right_neighbor_id,
            )
            if neighbor_id is not None
        }
        if (
            transition_frame is None
            and lane_id in previous_adjacent_lane_ids
        ):
            transition_frame = frame
            transition_from_lane_id = previous_lane_id
            transition_lane_id = lane_id
        previous_lane_id = lane_id

    axis = _axis(vehicle)
    blocker_longitudinal_gap_at_anchor = None
    first_blocker_pass_frame = None
    if axis is not None:
        anchor_relative = (
            blocker.positions[_HISTORY_END_FRAME]
            - vehicle.positions[_HISTORY_END_FRAME]
        )
        if np.isfinite(anchor_relative).all():
            blocker_longitudinal_gap_at_anchor = float(
                np.dot(anchor_relative, axis)
            )
        for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS):
            relative = blocker.positions[frame] - vehicle.positions[frame]
            if not np.isfinite(relative).all():
                continue
            if float(np.dot(relative, axis)) <= 0.0:
                first_blocker_pass_frame = frame
                break
    temporal_order_passed = bool(
        transition_frame is not None
        and (
            first_blocker_pass_frame is None
            or transition_frame <= first_blocker_pass_frame
        )
    )
    passed = bool(
        structural["passed"]
        and anchor_lane_id == original_lane_id
        and transition_frame is not None
        and axis is not None
        and temporal_order_passed
    )
    if not structural["passed"]:
        reason = "frozen_adjacent_lane_structure_unavailable"
    elif anchor_lane_id != original_lane_id:
        reason = "generated_anchor_lane_differs_from_frozen_role_lane"
    elif transition_frame is None:
        reason = "generated_future_has_no_adjacent_lane_transition"
    elif axis is None:
        reason = "generated_anchor_heading_unavailable"
    elif not temporal_order_passed:
        reason = "generated_lane_change_occurs_after_blocker_pass"
    else:
        reason = None
    return _result(
        passed,
        "frozen_seed_lane_plus_generated_target_lane_occupancy",
        compatible_seed_detector_reused=False,
        frozen_structure=structural,
        frozen_original_lane_id=original_lane_id,
        generated_anchor_lane_id=anchor_lane_id,
        generated_anchor_lane_distance_m=anchor_lane_distance,
        direct_adjacent_lane_ids=adjacent_lane_ids,
        generated_compact_lane_sequence=compact_lane_sequence,
        blocker_longitudinal_gap_at_anchor_m=blocker_longitudinal_gap_at_anchor,
        first_blocker_pass_frame_index=first_blocker_pass_frame,
        first_blocker_pass_future_index=(
            None
            if first_blocker_pass_frame is None
            else first_blocker_pass_frame - _FUTURE_START_FRAME
        ),
        lane_change_frame_index=transition_frame,
        lane_change_future_index=(
            None
            if transition_frame is None
            else transition_frame - _FUTURE_START_FRAME
        ),
        lane_change_from_lane_id=transition_from_lane_id,
        entered_adjacent_lane_id=transition_lane_id,
        temporal_order_passed=temporal_order_passed,
        reason=reason,
    )


def _approach_to_buffer(
    first_role: str,
    second_role: str,
) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        distances = _distance_series(context, first_role, second_role)
        start = _finite_scalar(distances[_HISTORY_END_FRAME])
        future = distances[_FUTURE_START_FRAME:]
        finite = future[np.isfinite(future)]
        minimum = None if not len(finite) else float(np.min(finite))
        maximum = _risk_upper(context.skill)
        passed = (
            start is not None
            and minimum is not None
            and maximum is not None
            and start - minimum >= _MIN_PROGRESS_M
            and minimum <= maximum
        )
        return _result(
            passed,
            "generated_exact_role_distance",
            initial_distance_m=start,
            minimum_future_distance_m=minimum,
            required_distance_reduction_m=_MIN_PROGRESS_M,
            target_range_upper_m=maximum,
        )

    return validate


def _late_lateral_crossing(role: str) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        threshold = (
            _threshold(context.skill, "minimum_lateral_displacement_m")
            or _LATERAL_EVENT_M
        )
        lateral = np.abs(_lateral_displacement(context.agents_by_role[role]))
        future = lateral[_FUTURE_START_FRAME:]
        finite = future[np.isfinite(future)]
        maximum = None if not len(finite) else float(np.max(finite))
        frame = _lateral_event_frame(
            context.agents_by_role[role],
            threshold_m=threshold,
        )
        return _result(
            frame is not None,
            "generated_exact_role_lateral_motion",
            crossing_frame_index=frame,
            maximum_lateral_displacement_m=maximum,
            minimum_lateral_displacement_m=threshold,
        )

    return validate


def _conflicting_vehicle_present(
    first_role: str,
    second_role: str,
) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        distance, frame = _minimum_distance(context, first_role, second_role)
        maximum = (
            _threshold(context.skill, "maximum_target_gap_m")
            or _threshold(context.skill, "maximum_competing_vehicle_gap_m")
            or 20.0
        )
        return _result(
            distance is not None and distance <= maximum,
            "generated_exact_role_geometry",
            minimum_distance_m=distance,
            closest_frame_index=frame,
            maximum_interaction_distance_m=maximum,
        )

    return validate


def _competing_arrival(
    first_role: str,
    second_role: str,
) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        point, point_source = _conflict_point(context, first_role, second_role)
        if point is None:
            return _result(
                False,
                "generated_exact_role_arrival",
                reason="conflict_point_unavailable",
            )
        first_time, first_distance, first_frame = _arrival_to_point(
            context, first_role, point
        )
        second_time, second_distance, second_frame = _arrival_to_point(
            context, second_role, point
        )
        maximum_gap = _risk_upper(context.skill) or 3.0
        gap = (
            None
            if first_time is None or second_time is None
            else abs(first_time - second_time)
        )
        passed = (
            gap is not None
            and first_distance is not None
            and second_distance is not None
            and first_distance <= _CONFLICT_REACH_RADIUS_M
            and second_distance <= _CONFLICT_REACH_RADIUS_M
            and gap <= maximum_gap
        )
        return _result(
            passed,
            "generated_exact_role_arrival",
            conflict_point_xy=point.tolist(),
            conflict_point_source=point_source,
            first_arrival_time_s=first_time,
            second_arrival_time_s=second_time,
            first_arrival_frame=first_frame,
            second_arrival_frame=second_frame,
            first_minimum_point_distance_m=first_distance,
            second_minimum_point_distance_m=second_distance,
            arrival_time_gap_s=gap,
            maximum_arrival_time_gap_s=maximum_gap,
        )

    return validate


def _yielding_actor_continues(context: _ValidationContext) -> ConditionEvidence:
    point, point_source = _conflict_point(
        context,
        "non_yielding_vehicle",
        "priority_vehicle",
    )
    if point is None:
        return _result(
            False, "generated_exact_role_motion", reason="conflict_point_unavailable"
        )
    actor = context.agents_by_role["non_yielding_vehicle"]
    start = float(np.linalg.norm(actor.positions[_HISTORY_END_FRAME] - point))
    future = np.linalg.norm(actor.positions[_FUTURE_START_FRAME:] - point, axis=1)
    finite = future[np.isfinite(future)]
    minimum = None if not len(finite) else float(np.min(finite))
    speeds = _speeds(context.generated_scenario, actor)[_FUTURE_START_FRAME:]
    finite_speeds = speeds[np.isfinite(speeds)]
    median_speed = None if not len(finite_speeds) else float(np.median(finite_speeds))
    minimum_speed = 0.5
    passed = (
        minimum is not None
        and median_speed is not None
        and start - minimum >= _MIN_PROGRESS_M
        and median_speed >= minimum_speed
    )
    return _result(
        passed,
        "generated_exact_role_motion",
        conflict_point_xy=point.tolist(),
        conflict_point_source=point_source,
        initial_conflict_distance_m=start,
        minimum_conflict_distance_m=minimum,
        median_future_speed_mps=median_speed,
        minimum_continuing_speed_mps=minimum_speed,
    )


def _delayed_merge_entry(context: _ValidationContext) -> ConditionEvidence:
    roles = (
        "merging_vehicle",
        "leading_main_flow_vehicle",
        "trailing_main_flow_vehicle",
    )
    point, point_source = _conflict_point(context, roles[0], roles[1])
    if point is None:
        return _result(
            False, "generated_exact_role_arrival", reason="conflict_point_unavailable"
        )
    arrivals = {role: _arrival_to_point(context, role, point) for role in roles}
    merge_time, merge_distance, _ = arrivals["merging_vehicle"]
    lead_time = arrivals["leading_main_flow_vehicle"][0]
    trail_time = arrivals["trailing_main_flow_vehicle"][0]
    minimum_delay = 0.5
    maximum_gap = _threshold(context.skill, "maximum_arrival_time_gap_s") or 5.0
    adjacent = (
        []
        if merge_time is None
        else [
            abs(merge_time - value)
            for value in (lead_time, trail_time)
            if value is not None
        ]
    )
    passed = (
        merge_time is not None
        and merge_distance is not None
        and merge_distance <= _CONFLICT_REACH_RADIUS_M
        and merge_time >= minimum_delay
        and bool(adjacent)
        and min(adjacent) <= maximum_gap
    )
    return _result(
        passed,
        "generated_exact_role_arrival",
        conflict_point_xy=point.tolist(),
        conflict_point_source=point_source,
        arrival_times_s={role: values[0] for role, values in arrivals.items()},
        arrival_distances_m={role: values[1] for role, values in arrivals.items()},
        minimum_entry_delay_s=minimum_delay,
        maximum_adjacent_arrival_gap_s=maximum_gap,
    )


def _sustained_speed(
    context: _ValidationContext,
    role: str,
    *,
    minimum_speed_mps: float,
    maximum_speed_mps: float,
    minimum_duration_s: float,
    inside_polygon: list[list[float]] | None = None,
) -> tuple[bool, float, float | None, float | None]:
    agent = context.agents_by_role[role]
    speed = _speeds(context.generated_scenario, agent)
    mask = (
        np.isfinite(speed) & (speed >= minimum_speed_mps) & (speed <= maximum_speed_mps)
    )
    if inside_polygon is not None:
        polygon = np.asarray(inside_polygon, dtype=np.float64)
        inside = np.asarray(
            [
                point_in_polygon(point, polygon)
                for point in agent.positions[:_TOTAL_STEPS]
            ],
            dtype=bool,
        )
        mask &= inside
    mask[:_FUTURE_START_FRAME] = False
    times = _times_s(context.generated_scenario)
    longest = 0.0
    start: int | None = None
    end: int | None = None
    run_start: int | None = None
    for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS):
        if mask[frame] and run_start is None:
            run_start = frame
        if run_start is not None and (not mask[frame] or frame == _TOTAL_STEPS - 1):
            run_end = frame if mask[frame] else frame - 1
            duration = max(0.0, float(times[run_end] - times[run_start]))
            if duration > longest:
                longest, start, end = duration, run_start, run_end
            run_start = None
    return longest >= minimum_duration_s, longest, start, end


def _sustained_creep_speed(context: _ValidationContext) -> ConditionEvidence:
    minimum = _parameter_bound(context.skill, "creep_speed_mps", 0) or 0.3
    maximum = _threshold(context.skill, "maximum_creep_speed_mps") or 2.0
    duration = 0.5
    passed, longest, start, end = _sustained_speed(
        context,
        "creeping_vehicle",
        minimum_speed_mps=minimum,
        maximum_speed_mps=maximum,
        minimum_duration_s=duration,
    )
    return _result(
        passed,
        "generated_exact_role_kinematics",
        longest_creep_duration_s=longest,
        creep_start_frame=start,
        creep_end_frame=end,
        speed_range_mps=[minimum, maximum],
        minimum_duration_s=duration,
    )


def _context_item(context: _ValidationContext) -> Mapping[str, Any] | None:
    root = context.generated_scenario.metadata.get(RISK_CONTEXT_METADATA_KEY)
    if not isinstance(root, Mapping):
        return None
    item = root.get(context.skill.skill_id)
    return item if isinstance(item, Mapping) else None


def _context_polygon(context: _ValidationContext) -> list[list[float]] | None:
    item = _context_item(context)
    if item is None:
        return None
    value = item.get("conflict_area_polygon_xy")
    try:
        polygon = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    return polygon.tolist()


def _inside_conflict_area(context: _ValidationContext) -> ConditionEvidence:
    polygon = _context_polygon(context)
    if polygon is None:
        return _result(False, "prepared_risk_context", reason="polygon_unavailable")
    agent = context.agents_by_role["blocking_vehicle"]
    frames = [
        frame
        for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS)
        if point_in_polygon(agent.positions[frame], np.asarray(polygon))
    ]
    return _result(
        bool(frames),
        "generated_exact_role_conflict_polygon",
        first_inside_frame=None if not frames else frames[0],
        inside_frame_count=len(frames),
        polygon_source=_context_item(context).get("event_source"),
    )


def _sustained_blocking(context: _ValidationContext) -> ConditionEvidence:
    polygon = _context_polygon(context)
    if polygon is None:
        return _result(False, "prepared_risk_context", reason="polygon_unavailable")
    maximum = _threshold(context.skill, "maximum_blocker_speed_mps") or 0.5
    minimum_duration = _parameter_bound(context.skill, "hold_duration_s", 0) or 1.0
    passed, longest, start, end = _sustained_speed(
        context,
        "blocking_vehicle",
        minimum_speed_mps=0.0,
        maximum_speed_mps=maximum,
        minimum_duration_s=minimum_duration,
        inside_polygon=polygon,
    )
    return _result(
        passed,
        "generated_exact_role_kinematics_and_polygon",
        longest_low_speed_occupancy_s=longest,
        occupancy_start_frame=start,
        occupancy_end_frame=end,
        maximum_blocker_speed_mps=maximum,
        minimum_duration_s=minimum_duration,
    )


def _crossing_flow_approaches(context: _ValidationContext) -> ConditionEvidence:
    item = _context_item(context)
    point = None if item is None else _finite_point(item.get("conflict_point_xy"))
    if point is None:
        return _result(
            False, "prepared_risk_context", reason="conflict_point_unavailable"
        )
    role = "crossing_vehicle"
    actor = context.agents_by_role[role]
    distances = np.linalg.norm(actor.positions[:_TOTAL_STEPS] - point, axis=1)
    start = _finite_scalar(distances[_HISTORY_END_FRAME])
    finite = distances[_FUTURE_START_FRAME:][
        np.isfinite(distances[_FUTURE_START_FRAME:])
    ]
    minimum = None if not len(finite) else float(np.min(finite))
    passed = (
        start is not None and minimum is not None and start - minimum >= _MIN_PROGRESS_M
    )
    return _result(
        passed,
        "generated_exact_role_conflict_approach",
        initial_conflict_distance_m=start,
        minimum_conflict_distance_m=minimum,
        required_distance_reduction_m=_MIN_PROGRESS_M,
    )


def _sustained_mutual_yield(context: _ValidationContext) -> ConditionEvidence:
    maximum = _threshold(context.skill, "maximum_creep_speed_mps") or 3.0
    minimum_duration = min(
        _parameter_bound(context.skill, "first_wait_duration_s", 0) or 1.0,
        _parameter_bound(context.skill, "second_wait_duration_s", 0) or 1.0,
    )
    results = {
        role: _sustained_speed(
            context,
            role,
            minimum_speed_mps=0.0,
            maximum_speed_mps=maximum,
            minimum_duration_s=minimum_duration,
        )
        for role in ("first_yielding_vehicle", "second_yielding_vehicle")
    }
    return _result(
        all(values[0] for values in results.values()),
        "generated_exact_role_kinematics",
        longest_low_speed_duration_s={
            role: values[1] for role, values in results.items()
        },
        maximum_speed_mps=maximum,
        minimum_duration_s=minimum_duration,
    )


def _delayed_conflict_entry(context: _ValidationContext) -> ConditionEvidence:
    point, point_source = _conflict_point(
        context,
        "first_yielding_vehicle",
        "second_yielding_vehicle",
    )
    if point is None:
        return _result(
            False, "generated_exact_role_motion", reason="conflict_point_unavailable"
        )
    minimum_wait = min(
        _parameter_bound(context.skill, "first_wait_duration_s", 0) or 1.0,
        _parameter_bound(context.skill, "second_wait_duration_s", 0) or 1.0,
    )
    times = _times_s(context.generated_scenario)
    threshold_time = float(times[_HISTORY_END_FRAME] + minimum_wait)
    entry_times: dict[str, float | None] = {}
    for role in ("first_yielding_vehicle", "second_yielding_vehicle"):
        actor = context.agents_by_role[role]
        distances = np.linalg.norm(actor.positions[:_TOTAL_STEPS] - point, axis=1)
        candidates = np.flatnonzero(
            np.isfinite(distances[_FUTURE_START_FRAME:])
            & (distances[_FUTURE_START_FRAME:] <= _CONFLICT_HALF_EXTENT_M)
        )
        entry_times[role] = (
            None
            if not len(candidates)
            else float(times[int(candidates[0] + _FUTURE_START_FRAME)])
        )
    passed = all(
        value is None or value >= threshold_time for value in entry_times.values()
    )
    return _result(
        passed,
        "generated_exact_role_conflict_entry",
        conflict_point_xy=point.tolist(),
        conflict_point_source=point_source,
        entry_times_s=entry_times,
        minimum_wait_s=minimum_wait,
    )


def _delayed_pedestrian_entry(context: _ValidationContext) -> ConditionEvidence:
    polygon = _context_polygon(context)
    if polygon is None:
        return _result(False, "prepared_risk_context", reason="polygon_unavailable")
    actor = context.agents_by_role["emerging_pedestrian"]
    polygon_array = np.asarray(polygon)
    inside = np.asarray(
        [
            point_in_polygon(point, polygon_array)
            for point in actor.positions[:_TOTAL_STEPS]
        ],
        dtype=bool,
    )
    transitions = np.flatnonzero(~inside[:-1] & inside[1:]) + 1
    transitions = transitions[transitions >= _FUTURE_START_FRAME]
    frame = None if not len(transitions) else int(transitions[0])
    delay = (
        None
        if frame is None
        else float(
            _times_s(context.generated_scenario)[frame]
            - _times_s(context.generated_scenario)[_HISTORY_END_FRAME]
        )
    )
    minimum = _parameter_bound(context.skill, "emergence_start_s", 0) or 1.0
    return _result(
        delay is not None and delay >= minimum,
        "generated_exact_role_polygon_transition",
        first_entry_frame=frame,
        entry_delay_s=delay,
        minimum_entry_delay_s=minimum,
    )


def _vehicle_approaches_prepared_conflict(
    context: _ValidationContext,
) -> ConditionEvidence:
    item = _context_item(context)
    point = None if item is None else _finite_point(item.get("conflict_point_xy"))
    if point is None:
        return _result(
            False, "prepared_risk_context", reason="conflict_point_unavailable"
        )
    vehicle = context.agents_by_role["responding_vehicle"]
    distances = np.linalg.norm(vehicle.positions[:_TOTAL_STEPS] - point, axis=1)
    start = _finite_scalar(distances[_HISTORY_END_FRAME])
    finite = distances[_FUTURE_START_FRAME:][
        np.isfinite(distances[_FUTURE_START_FRAME:])
    ]
    minimum = None if not len(finite) else float(np.min(finite))
    maximum = (
        _threshold(context.skill, "maximum_conflict_distance_m")
        or _CONFLICT_REACH_RADIUS_M
    )
    passed = (
        start is not None
        and minimum is not None
        and start - minimum >= _MIN_PROGRESS_M
        and minimum <= maximum
    )
    return _result(
        passed,
        "generated_exact_role_conflict_approach",
        initial_conflict_distance_m=start,
        minimum_conflict_distance_m=minimum,
        maximum_conflict_distance_m=maximum,
    )


def _previously_stopped(context: _ValidationContext) -> ConditionEvidence:
    agent = context.source_agents_by_role["reentering_vehicle"]
    speed = _speeds(context.source_scenario, agent)
    maximum = _threshold(context.skill, "stopped_speed_mps") or 0.5
    minimum_duration = _threshold(context.skill, "minimum_stopped_duration_s") or 1.0
    times = _times_s(context.source_scenario)
    start_time = times[_HISTORY_END_FRAME] - minimum_duration
    indices = np.flatnonzero(
        (times <= times[_HISTORY_END_FRAME] + _EPS)
        & (times >= start_time - _EPS)
        & np.isfinite(speed)
    )
    maximum_observed = None if not len(indices) else float(np.max(speed[indices]))
    return _result(
        maximum_observed is not None and maximum_observed <= maximum,
        "source_history_exact_role_kinematics",
        history_window_s=minimum_duration,
        maximum_history_speed_mps=maximum_observed,
        stopped_speed_threshold_mps=maximum,
    )


def _begins_moving(context: _ValidationContext) -> ConditionEvidence:
    speed = _speeds(
        context.generated_scenario,
        context.agents_by_role["reentering_vehicle"],
    )[_FUTURE_START_FRAME:]
    finite = speed[np.isfinite(speed)]
    maximum = None if not len(finite) else float(np.max(finite))
    minimum = _threshold(context.skill, "minimum_moving_speed_mps") or 1.0
    return _result(
        maximum is not None and maximum >= minimum,
        "generated_exact_role_kinematics",
        maximum_future_speed_mps=maximum,
        minimum_moving_speed_mps=minimum,
    )


def _enters_main_flow(context: _ValidationContext) -> ConditionEvidence:
    reentering = context.agents_by_role["reentering_vehicle"]
    front = context.agents_by_role["front_main_flow_vehicle"]
    rear = context.agents_by_role["rear_main_flow_vehicle"]
    maximum_lateral = (
        _threshold(context.skill, "maximum_lateral_reentry_distance_m") or 5.0
    )
    selected: dict[str, Any] | None = None
    for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS):
        points = np.vstack(
            (reentering.positions[frame], front.positions[frame], rear.positions[frame])
        )
        if not np.isfinite(points).all():
            continue
        axis = front.positions[frame] - rear.positions[frame]
        length = float(np.linalg.norm(axis))
        if length <= _EPS:
            continue
        unit = axis / length
        relative = reentering.positions[frame] - rear.positions[frame]
        longitudinal = float(np.dot(relative, unit))
        lateral = abs(float(unit[0] * relative[1] - unit[1] * relative[0]))
        if 0.0 <= longitudinal <= length and lateral <= maximum_lateral:
            selected = {
                "entry_frame": frame,
                "front_rear_gap_m": length,
                "longitudinal_position_m": longitudinal,
                "lateral_distance_m": lateral,
            }
            break
    return _result(
        selected is not None,
        "generated_exact_role_main_flow_geometry",
        maximum_lateral_reentry_distance_m=maximum_lateral,
        entry_evidence=selected,
    )


def _vehicle_approaching(
    object_role: str,
    vehicle_role: str,
) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        distances = _distance_series(context, object_role, vehicle_role)
        start = _finite_scalar(distances[_HISTORY_END_FRAME])
        future = distances[_FUTURE_START_FRAME:]
        finite = future[np.isfinite(future)]
        minimum = None if not len(finite) else float(np.min(finite))
        return _result(
            start is not None
            and minimum is not None
            and start - minimum >= _MIN_PROGRESS_M,
            "generated_exact_role_distance",
            initial_distance_m=start,
            minimum_future_distance_m=minimum,
            required_distance_reduction_m=_MIN_PROGRESS_M,
        )

    return validate


def _buffer_intrusion(
    object_role: str,
    vehicle_role: str,
) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        distance, frame = _minimum_distance(context, object_role, vehicle_role)
        maximum = _risk_upper(context.skill)
        return _result(
            distance is not None and maximum is not None and distance <= maximum,
            "generated_exact_role_distance",
            minimum_clearance_m=distance,
            closest_frame_index=frame,
            required_buffer_upper_m=maximum,
        )

    return validate


def _adjacent_vehicle_cuts_in(context: _ValidationContext) -> ConditionEvidence:
    frame = _cut_in_frame(context)
    return _result(
        frame is not None,
        "generated_exact_role_lateral_geometry",
        cut_in_completion_frame=frame,
    )


def _initiator_brakes(context: _ValidationContext) -> ConditionEvidence:
    cut_in = _cut_in_frame(context)
    if cut_in is None:
        return _result(
            False, "generated_exact_role_kinematics", reason="cut_in_not_observed"
        )
    threshold = _parameter_bound(context.skill, "peak_deceleration_mps2", 0) or 2.0
    frame = _brake_onset_frame(
        context,
        "cut_in_braking_vehicle",
        start_frame=cut_in,
        minimum_deceleration_mps2=threshold,
    )
    return _result(
        frame is not None,
        "generated_exact_role_kinematics",
        cut_in_completion_frame=cut_in,
        brake_onset_frame=frame,
        minimum_deceleration_mps2=threshold,
    )


def _short_post_cut_in_delay(context: _ValidationContext) -> ConditionEvidence:
    cut_in = _cut_in_frame(context)
    stage = _cut_in_stage_start_frame(context)
    minimum = _parameter_bound(context.skill, "brake_delay_after_cut_in_s", 0) or 0.2
    maximum = _parameter_bound(context.skill, "brake_delay_after_cut_in_s", 1) or 1.5
    delay = (
        None
        if cut_in is None or stage is None
        else float(
            _times_s(context.generated_scenario)[stage]
            - _times_s(context.generated_scenario)[cut_in]
        )
    )
    return _result(
        delay is not None and minimum <= delay <= maximum,
        "generated_exact_role_event_timing",
        cut_in_completion_frame=cut_in,
        brake_onset_frame=stage,
        brake_delay_s=delay,
        allowed_delay_s=[minimum, maximum],
    )


def _turning_vehicle(context: _ValidationContext) -> ConditionEvidence:
    actor = context.agents_by_role["u_turning_vehicle"]
    initial = float(actor.headings[_HISTORY_END_FRAME])
    future = actor.headings[_FUTURE_START_FRAME:_TOTAL_STEPS]
    changes = [
        math.degrees(heading_difference(initial, float(value)))
        for value in future
        if math.isfinite(float(value))
    ]
    maximum = None if not changes else float(max(changes))
    minimum = (
        _threshold(context.skill, "minimum_opposing_heading_difference_deg") or 120.0
    )
    return _result(
        maximum is not None and maximum >= minimum,
        "generated_exact_role_heading",
        maximum_heading_change_deg=maximum,
        minimum_heading_change_deg=minimum,
    )


def _vehicle_path_conflict(context: _ValidationContext) -> ConditionEvidence:
    first = "u_turning_vehicle"
    second = "conflicting_vehicle"
    point = _trajectory_conflict_point(context, first, second)
    distance, frame = _minimum_distance(context, first, second)
    passed = (
        point is not None
        or distance is not None
        and distance <= _CONFLICT_REACH_RADIUS_M
    )
    return _result(
        passed,
        "generated_exact_role_paths",
        path_intersection_xy=None if point is None else point.tolist(),
        minimum_synchronized_distance_m=distance,
        closest_frame_index=frame,
        conflict_proxy_radius_m=_CONFLICT_REACH_RADIUS_M,
    )


def _longitudinal_gap(
    context: _ValidationContext,
    other_role: str,
) -> np.ndarray:
    squeezed = context.agents_by_role["squeezed_vehicle"]
    other = context.agents_by_role[other_role]
    axis = _axis(squeezed)
    if axis is None:
        return np.full(_TOTAL_STEPS, np.nan)
    relative = other.positions[:_TOTAL_STEPS] - squeezed.positions[:_TOTAL_STEPS]
    valid = np.isfinite(relative).all(axis=1)
    gap = np.full(_TOTAL_STEPS, np.nan)
    gap[valid] = relative[valid] @ axis
    return gap


def _gap_small(role: str, threshold_name: str, *, front: bool) -> ConditionValidator:
    def validate(context: _ValidationContext) -> ConditionEvidence:
        signed = _longitudinal_gap(context, role)
        values = signed if front else -signed
        future = values[_FUTURE_START_FRAME:]
        valid = future[np.isfinite(future) & (future > 0.0)]
        minimum = None if not len(valid) else float(np.min(valid))
        maximum = _threshold(context.skill, threshold_name) or 25.0
        return _result(
            minimum is not None and minimum <= maximum,
            "generated_exact_role_longitudinal_geometry",
            minimum_positive_gap_m=minimum,
            maximum_gap_m=maximum,
        )

    return validate


def _multi_direction_gap_closing(context: _ValidationContext) -> ConditionEvidence:
    front = _longitudinal_gap(context, "front_pressure_vehicle")
    rear = -_longitudinal_gap(context, "rear_pressure_vehicle")
    initial_front = _finite_scalar(front[_HISTORY_END_FRAME])
    initial_rear = _finite_scalar(rear[_HISTORY_END_FRAME])
    future_front = front[_FUTURE_START_FRAME:]
    future_rear = rear[_FUTURE_START_FRAME:]
    finite_front = future_front[np.isfinite(future_front)]
    finite_rear = future_rear[np.isfinite(future_rear)]
    minimum_front = None if not len(finite_front) else float(np.min(finite_front))
    minimum_rear = None if not len(finite_rear) else float(np.min(finite_rear))
    combined = (
        None
        if None in (initial_front, initial_rear, minimum_front, minimum_rear)
        else float(initial_front - minimum_front + initial_rear - minimum_rear)
    )
    minimum_rate = (
        _threshold(context.skill, "minimum_combined_closing_speed_mps") or 0.5
    )
    horizon = float(
        _times_s(context.generated_scenario)[_TOTAL_STEPS - 1]
        - _times_s(context.generated_scenario)[_HISTORY_END_FRAME]
    )
    rate = None if combined is None or horizon <= 0.0 else combined / horizon
    passed = (
        combined is not None
        and initial_front is not None
        and initial_rear is not None
        and minimum_front is not None
        and minimum_rear is not None
        and minimum_front < initial_front
        and minimum_rear < initial_rear
        and rate is not None
        and rate >= minimum_rate
    )
    return _result(
        passed,
        "generated_exact_role_longitudinal_geometry",
        initial_front_gap_m=initial_front,
        minimum_front_gap_m=minimum_front,
        initial_rear_gap_m=initial_rear,
        minimum_rear_gap_m=minimum_rear,
        combined_closing_rate_mps=rate,
        minimum_combined_closing_speed_mps=minimum_rate,
    )


def _motorcyclist_between_vehicles(context: _ValidationContext) -> ConditionEvidence:
    motor = context.agents_by_role["filtering_motorcyclist"]
    first = context.agents_by_role["first_vehicle"]
    second = context.agents_by_role["second_vehicle"]
    maximum = (
        _threshold(context.skill, "maximum_motorcyclist_vehicle_distance_m") or 7.0
    )
    selected: dict[str, Any] | None = None
    for frame in range(_FUTURE_START_FRAME, _TOTAL_STEPS):
        points = np.vstack(
            (motor.positions[frame], first.positions[frame], second.positions[frame])
        )
        if not np.isfinite(points).all():
            continue
        segment = second.positions[frame] - first.positions[frame]
        squared = float(np.dot(segment, segment))
        if squared <= _EPS:
            continue
        fraction = float(
            np.dot(motor.positions[frame] - first.positions[frame], segment) / squared
        )
        projection = first.positions[frame] + np.clip(fraction, 0.0, 1.0) * segment
        distance = float(np.linalg.norm(motor.positions[frame] - projection))
        if 0.0 <= fraction <= 1.0 and distance <= maximum:
            selected = {
                "frame_index": frame,
                "between_fraction": fraction,
                "distance_to_vehicle_segment_m": distance,
            }
            break
    return _result(
        selected is not None,
        "generated_exact_role_geometry",
        between_evidence=selected,
        maximum_segment_distance_m=maximum,
    )


def _positive_relative_speed(context: _ValidationContext) -> ConditionEvidence:
    speeds = {
        role: _speeds(context.generated_scenario, context.agents_by_role[role])[
            _FUTURE_START_FRAME:
        ]
        for role in ("filtering_motorcyclist", "first_vehicle", "second_vehicle")
    }
    means = {}
    for role, values in speeds.items():
        finite = values[np.isfinite(values)]
        means[role] = None if not len(finite) else float(np.mean(finite))
    motor = means["filtering_motorcyclist"]
    vehicles = [means["first_vehicle"], means["second_vehicle"]]
    vehicle_mean = (
        None if any(value is None for value in vehicles) else float(np.mean(vehicles))
    )
    relative = None if motor is None or vehicle_mean is None else motor - vehicle_mean
    return _result(
        relative is not None and relative > 0.0,
        "generated_exact_role_kinematics",
        mean_speeds_mps=means,
        motorcyclist_relative_speed_mps=relative,
    )


def _motorcyclist_buffer_intrusion(context: _ValidationContext) -> ConditionEvidence:
    first, first_frame = _minimum_distance(
        context,
        "filtering_motorcyclist",
        "first_vehicle",
    )
    second, second_frame = _minimum_distance(
        context,
        "filtering_motorcyclist",
        "second_vehicle",
    )
    candidates = [
        (distance, frame, role)
        for distance, frame, role in (
            (first, first_frame, "first_vehicle"),
            (second, second_frame, "second_vehicle"),
        )
        if distance is not None
    ]
    selected = None if not candidates else min(candidates, key=lambda item: item[0])
    maximum = _risk_upper(context.skill)
    return _result(
        selected is not None and maximum is not None and selected[0] <= maximum,
        "generated_exact_role_distance",
        minimum_clearance_m=None if selected is None else selected[0],
        closest_frame_index=None if selected is None else selected[1],
        closest_vehicle_role=None if selected is None else selected[2],
        required_buffer_upper_m=maximum,
    )


def _insufficient_gap(context: _ValidationContext) -> ConditionEvidence:
    return _competing_arrival(
        "closing_lane_vehicle",
        "continuing_lane_vehicle",
    )(context)


def _observed(skill_id: str, roles: tuple[str, ...]) -> SkillValidatorSpec:
    return SkillValidatorSpec(skill_id, "observed_trigger", roles)


def _compatible(
    skill_id: str,
    roles: tuple[str, ...],
    conditions: Mapping[str, ConditionValidator],
) -> SkillValidatorSpec:
    return SkillValidatorSpec(skill_id, "compatible_seed", roles, conditions)


SKILL_VALIDATORS: Mapping[str, SkillValidatorSpec] = MappingProxyType(
    {
        "lead_sudden_stop": _observed(
            "lead_sudden_stop", ("stopping_leader", "follower")
        ),
        "slow_lead_blockage": _observed(
            "slow_lead_blockage", ("slow_leader", "follower")
        ),
        "short_headway_following": _observed(
            "short_headway_following", ("leader", "close_follower")
        ),
        "chain_braking": _compatible(
            "chain_braking",
            ("front_vehicle", "middle_vehicle", "rear_vehicle"),
            {
                "three_vehicle_queue": _structural_seed_condition(
                    "three_vehicle_queue"
                ),
                "leader_decelerating": _leader_decelerating,
                "delayed_braking_response": _delayed_braking_response,
            },
        ),
        "cut_out_reveals_slow_vehicle": _compatible(
            "cut_out_reveals_slow_vehicle",
            ("cut_out_vehicle", "target_vehicle", "slow_vehicle"),
            {
                "three_vehicle_queue": _structural_seed_condition(
                    "three_vehicle_queue"
                ),
                "lead_vehicle_cuts_out": _lead_vehicle_cuts_out,
                "newly_exposed_slow_vehicle": _newly_exposed_slow_vehicle,
            },
        ),
        "simultaneous_lane_change_conflict": _compatible(
            "simultaneous_lane_change_conflict",
            ("left_lane_changer", "right_lane_changer"),
            {
                "shared_target_lane": _shared_target_lane,
                "overlapping_lane_change_window": _overlapping_lane_change_window,
                "close_longitudinal_position": _close_longitudinal_position,
            },
        ),
        "forced_lane_change_around_blockage": _compatible(
            "forced_lane_change_around_blockage",
            ("blocking_actor", "avoiding_vehicle"),
            {
                "blockage_ahead": _blockage_ahead,
                "adjacent_lane_available": _adjacent_lane_change_realized,
                "approach_to_safety_buffer": _approach_to_buffer(
                    "blocking_actor", "avoiding_vehicle"
                ),
            },
        ),
        "late_lane_change_before_diverge": _compatible(
            "late_lane_change_before_diverge",
            ("late_lane_changer", "adjacent_lane_vehicle"),
            {
                "diverging_topology": _structural_seed_condition(
                    "diverging_topology",
                    fields=("initiator_lane_diverges", "distance_to_diverge_m"),
                ),
                "late_lateral_crossing": _late_lateral_crossing("late_lane_changer"),
                "conflicting_vehicle_present": _conflicting_vehicle_present(
                    "late_lane_changer", "adjacent_lane_vehicle"
                ),
            },
        ),
        "ramp_merge_small_gap": _observed(
            "ramp_merge_small_gap", ("merging_vehicle", "mainline_vehicle")
        ),
        "lane_drop_merge_competition": _compatible(
            "lane_drop_merge_competition",
            ("closing_lane_vehicle", "continuing_lane_vehicle"),
            {
                "lane_successors_converge": _structural_seed_condition(
                    "lane_successors_converge",
                    fields=("lanes_converge", "convergence_target_lane_id"),
                ),
                "competing_arrival": _competing_arrival(
                    "closing_lane_vehicle", "continuing_lane_vehicle"
                ),
                "insufficient_gap": _insufficient_gap,
            },
        ),
        "merge_without_yield": _compatible(
            "merge_without_yield",
            ("non_yielding_vehicle", "priority_vehicle"),
            {
                "converging_lanes": _structural_seed_condition(
                    "converging_lanes",
                    fields=("lanes_converge", "convergence_target_lane_id"),
                ),
                "explicit_priority_role": _structural_seed_condition(
                    "priority_roles_assignable",
                    fields=(
                        "role_assignment_basis",
                        "counterfactual_priority_required",
                    ),
                ),
                "yielding_actor_continues": _yielding_actor_continues,
            },
        ),
        "diverge_lane_crossing_conflict": _compatible(
            "diverge_lane_crossing_conflict",
            ("crossing_vehicle", "through_vehicle"),
            {
                "diverging_topology": _structural_seed_condition(
                    "diverging_topology",
                    fields=("initiator_lane_diverges", "distance_to_diverge_m"),
                ),
                "late_lateral_crossing": _late_lateral_crossing("crossing_vehicle"),
                "conflicting_vehicle_present": _conflicting_vehicle_present(
                    "crossing_vehicle", "through_vehicle"
                ),
            },
        ),
        "bike_lane_vehicle_merge_conflict": _observed(
            "bike_lane_vehicle_merge_conflict", ("cyclist", "motor_vehicle")
        ),
        "zipper_merge_multi_vehicle": _compatible(
            "zipper_merge_multi_vehicle",
            (
                "merging_vehicle",
                "leading_main_flow_vehicle",
                "trailing_main_flow_vehicle",
            ),
            {
                "converging_lanes": _structural_seed_condition(
                    "converging_lanes",
                    fields=("convergence_target_lane_id", "conflict_point_xy"),
                ),
                "competing_vehicles_present": _structural_seed_condition(
                    "competing_vehicles_present",
                    fields=("leading_arrival_time_s", "trailing_arrival_time_s"),
                ),
                "delayed_entry": _delayed_merge_entry,
            },
        ),
        "unprotected_left_turn_conflict": _observed(
            "unprotected_left_turn_conflict",
            ("left_turn_vehicle", "opposing_through_vehicle"),
        ),
        "right_turn_vehicle_conflict": _observed(
            "right_turn_vehicle_conflict",
            ("right_turn_vehicle", "conflicting_vehicle"),
        ),
        "crossing_path_conflict": _observed(
            "crossing_path_conflict", ("first_vehicle", "second_vehicle")
        ),
        "intersection_creep_conflict": _compatible(
            "intersection_creep_conflict",
            ("creeping_vehicle", "crossing_vehicle"),
            {
                "near_intersection_entry": _structural_seed_condition(
                    "near_intersection_entry",
                    fields=("intersection_entry_distance_m", "conflict_point_xy"),
                ),
                "sustained_creep_speed": _sustained_creep_speed,
                "crossing_vehicle_present": _structural_seed_condition(
                    "crossing_vehicle_present",
                    "crossing_flow_present",
                    fields=("crossing_vehicle_arrival_s",),
                ),
            },
        ),
        "intersection_blocking_vehicle": _compatible(
            "intersection_blocking_vehicle",
            ("blocking_vehicle", "crossing_vehicle"),
            {
                "inside_conflict_area": _inside_conflict_area,
                "sustained_low_speed_or_stop": _sustained_blocking,
                "crossing_flow_approaches": _crossing_flow_approaches,
            },
        ),
        "mutual_yield_deadlock": _compatible(
            "mutual_yield_deadlock",
            ("first_yielding_vehicle", "second_yielding_vehicle"),
            {
                "shared_conflict_point": _structural_seed_condition(
                    "crossing_flow_present",
                    fields=("conflict_point_xy", "crossing_angle_deg"),
                ),
                "sustained_low_speed_or_stop": _sustained_mutual_yield,
                "delayed_entry": _delayed_conflict_entry,
            },
        ),
        "crosswalk_pedestrian_crossing": _observed(
            "crosswalk_pedestrian_crossing", ("pedestrian", "yielding_vehicle")
        ),
        "jaywalking_pedestrian_crossing": _observed(
            "jaywalking_pedestrian_crossing", ("pedestrian", "responding_vehicle")
        ),
        "roadside_pedestrian_emergence": _compatible(
            "roadside_pedestrian_emergence",
            ("emerging_pedestrian", "responding_vehicle"),
            {
                "pedestrian_near_drivable_boundary": _structural_seed_condition(
                    "pedestrian_near_drivable_boundary",
                    fields=("drivable_boundary_distance_m",),
                ),
                "delayed_entry": _delayed_pedestrian_entry,
                "vehicle_approaching": _vehicle_approaches_prepared_conflict,
            },
        ),
        "cyclist_crossing": _observed(
            "cyclist_crossing", ("crossing_cyclist", "responding_vehicle")
        ),
        "turning_vehicle_crosswalk_conflict": _observed(
            "turning_vehicle_crosswalk_conflict", ("pedestrian", "turning_vehicle")
        ),
        "group_pedestrian_crossing": _observed(
            "group_pedestrian_crossing",
            (
                "first_crossing_pedestrian",
                "responding_vehicle",
                "second_crossing_pedestrian",
            ),
        ),
        "cyclist_vehicle_merge": _observed(
            "cyclist_vehicle_merge",
            ("merging_cyclist", "front_motor_vehicle", "rear_motor_vehicle"),
        ),
        "stopped_vehicle_reentry": _compatible(
            "stopped_vehicle_reentry",
            (
                "reentering_vehicle",
                "front_main_flow_vehicle",
                "rear_main_flow_vehicle",
            ),
            {
                "previously_stopped": _previously_stopped,
                "begins_moving": _begins_moving,
                "enters_main_flow": _enters_main_flow,
            },
        ),
        "construction_object_lane_blockage": _compatible(
            "construction_object_lane_blockage",
            ("construction_object", "responding_vehicle"),
            {
                "construction_actor_near_path": _structural_seed_condition(
                    "construction_actor_near_path",
                    fields=("minimum_object_path_clearance_m",),
                ),
                "vehicle_approaching": _vehicle_approaching(
                    "construction_object", "responding_vehicle"
                ),
                "safety_buffer_violation_predicted": _buffer_intrusion(
                    "construction_object", "responding_vehicle"
                ),
            },
        ),
        "static_object_avoidance": _compatible(
            "static_object_avoidance",
            ("static_object", "avoiding_vehicle"),
            {
                "static_actor_near_path": _structural_seed_condition(
                    "static_actor_near_path",
                    fields=("minimum_object_path_clearance_m",),
                ),
                "predicted_buffer_intrusion": _buffer_intrusion(
                    "static_object", "avoiding_vehicle"
                ),
                "response_space_available": _structural_seed_condition(
                    "response_space_available"
                ),
            },
        ),
        "cut_in_then_brake": _compatible(
            "cut_in_then_brake",
            ("cut_in_braking_vehicle", "responding_vehicle"),
            {
                "adjacent_vehicle_cuts_in": _adjacent_vehicle_cuts_in,
                "short_post_cut_in_delay": _short_post_cut_in_delay,
                "initiator_brakes": _initiator_brakes,
            },
        ),
        "abrupt_u_turn_conflict": _compatible(
            "abrupt_u_turn_conflict",
            ("u_turning_vehicle", "conflicting_vehicle"),
            {
                "turning_vehicle": _turning_vehicle,
                "oncoming_vehicle_present": _structural_seed_condition(
                    "oncoming_vehicle_present",
                    fields=("actor_heading_difference_deg",),
                ),
                "vehicle_path_conflict": _vehicle_path_conflict,
            },
        ),
        "multi_vehicle_gap_squeeze": _compatible(
            "multi_vehicle_gap_squeeze",
            ("squeezed_vehicle", "front_pressure_vehicle", "rear_pressure_vehicle"),
            {
                "front_gap_small": _gap_small(
                    "front_pressure_vehicle", "maximum_front_gap_m", front=True
                ),
                "rear_gap_small": _gap_small(
                    "rear_pressure_vehicle", "maximum_rear_gap_m", front=False
                ),
                "multi_direction_gap_closing": _multi_direction_gap_closing,
            },
        ),
        "motorcyclist_filtering_conflict": _compatible(
            "motorcyclist_filtering_conflict",
            ("filtering_motorcyclist", "first_vehicle", "second_vehicle"),
            {
                "motorcyclist_between_vehicles": _motorcyclist_between_vehicles,
                "positive_relative_speed": _positive_relative_speed,
                "safety_buffer_violation_predicted": _motorcyclist_buffer_intrusion,
            },
        ),
    }
)


def _role_type_mismatches(
    context: _ValidationContext,
    required_roles: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    initiator_types = {
        str(value).lower() for value in context.skill.actors["initiator_types"]
    }
    responder_types = {
        str(value).lower() for value in context.skill.actors["responder_types"]
    }
    mismatches: dict[str, dict[str, Any]] = {}
    for index, role in enumerate(required_roles):
        allowed = initiator_types if index == 0 else responder_types
        actual = context.agents_by_role[role].object_type.lower()
        if actual not in allowed:
            mismatches[role] = {"actual": actual, "allowed": sorted(allowed)}
    return mismatches


def _build_context(
    source_scenario: Scenario,
    generated_scenario: Scenario,
    skill: SkillSpec,
    role_track_ids: Mapping[str, str],
    seed_evidence: Mapping[str, Any],
) -> _ValidationContext | None:
    normalized = {str(role): str(track_id) for role, track_id in role_track_ids.items()}
    generated_agents = {agent.track_id: agent for agent in generated_scenario.agents}
    source_agents = {agent.track_id: agent for agent in source_scenario.agents}
    if any(track_id not in generated_agents for track_id in normalized.values()):
        return None
    if any(track_id not in source_agents for track_id in normalized.values()):
        return None
    return _ValidationContext(
        source_scenario=source_scenario,
        generated_scenario=generated_scenario,
        skill=skill,
        role_track_ids=MappingProxyType(normalized),
        seed_evidence=seed_evidence,
        agents_by_role=MappingProxyType(
            {role: generated_agents[track_id] for role, track_id in normalized.items()}
        ),
        source_agents_by_role=MappingProxyType(
            {role: source_agents[track_id] for role, track_id in normalized.items()}
        ),
    )


def validate_skill_trigger(
    source_scenario: Scenario,
    generated_scenario: Scenario,
    skill: SkillSpec,
    role_track_ids: Mapping[str, str],
    seed_evidence: Mapping[str, Any],
    detection_config: DetectionConfig | None = None,
) -> FilterCheck:
    """Validate one generated skill event using its exact requested roles."""

    try:
        spec = SKILL_VALIDATORS[skill.skill_id]
    except KeyError as exc:
        raise KeyError(f"unknown formal skill_id: {skill.skill_id}") from exc
    configured_mode = str(skill.detection.get("mode"))
    if configured_mode != spec.detection_mode:
        raise ValueError(
            f"validator mode drift for {skill.skill_id}: "
            f"registry={spec.detection_mode}, skill={configured_mode}"
        )

    if spec.detection_mode == "observed_trigger":
        if detection_config is None:
            raise ValueError(
                "detection_config is required for observed_trigger validation"
            )
        result = validate_observed_skill(
            generated_scenario,
            skill,
            role_track_ids,
            detection_config,
        )
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=result.rejection_reasons,
            metrics={
                **dict(result.metrics),
                "validator_mode": "observed_exact_role_redetection",
                "compatible_seed_detector_reused": False,
                "source_scenario_id": source_scenario.scenario_id,
                "seed_evidence_used": False,
            },
        )

    role_contract = validate_role_contract(generated_scenario, skill, role_track_ids)
    if not role_contract.passed:
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_ROLE_CONTRACT_MISMATCH,),
            metrics={
                **dict(role_contract.metrics),
                "validator_mode": "compatible_missing_condition_validation",
                "compatible_seed_detector_reused": False,
            },
        )
    seed_mode = seed_evidence.get("detection_mode")
    if seed_mode not in (None, "compatible_seed"):
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_TRIGGER_NOT_REALIZED,),
            metrics={
                "validator_mode": "compatible_missing_condition_validation",
                "reason": "seed_evidence_detection_mode_mismatch",
                "seed_detection_mode": seed_mode,
                "compatible_seed_detector_reused": False,
                "seed_evidence_sha256": _seed_evidence_sha256(seed_evidence),
            },
        )
    contextual_scenario = prepare_risk_context(
        source_scenario,
        generated_scenario,
        skill,
        role_track_ids,
        seed_evidence,
    )
    context = _build_context(
        source_scenario,
        contextual_scenario,
        skill,
        role_track_ids,
        seed_evidence,
    )
    if context is None:
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_ROLE_CONTRACT_MISMATCH,),
            metrics={
                "validator_mode": "compatible_missing_condition_validation",
                "reason": "source_or_generated_role_track_missing",
                "compatible_seed_detector_reused": False,
            },
        )
    type_mismatches = _role_type_mismatches(context, spec.required_roles)
    if type_mismatches:
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_ROLE_CONTRACT_MISMATCH,),
            metrics={
                "validator_mode": "compatible_missing_condition_validation",
                "requested_role_track_ids": dict(context.role_track_ids),
                "role_type_mismatches": type_mismatches,
                "compatible_seed_detector_reused": False,
            },
        )

    configured_conditions = tuple(str(value) for value in skill.trigger["conditions"])
    if set(configured_conditions) != set(spec.condition_validators):
        raise ValueError(
            f"validator condition drift for {skill.skill_id}: "
            f"registry={sorted(spec.condition_validators)}, "
            f"skill={sorted(configured_conditions)}"
        )
    raw_missing = seed_evidence.get("missing_generation_conditions")
    if not isinstance(raw_missing, Sequence) or isinstance(raw_missing, (str, bytes)):
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_TRIGGER_NOT_REALIZED,),
            metrics={
                "validator_mode": "compatible_missing_condition_validation",
                "reason": "missing_generation_conditions_unavailable",
                "compatible_seed_detector_reused": False,
                "seed_evidence_sha256": _seed_evidence_sha256(seed_evidence),
            },
        )
    missing_conditions = tuple(str(value) for value in raw_missing)
    if not missing_conditions or len(set(missing_conditions)) != len(
        missing_conditions
    ):
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.SKILL_TRIGGER_NOT_REALIZED,),
            metrics={
                "validator_mode": "compatible_missing_condition_validation",
                "reason": "missing_generation_conditions_empty_or_duplicated",
                "required_missing_conditions": list(missing_conditions),
                "compatible_seed_detector_reused": False,
                "seed_evidence_sha256": _seed_evidence_sha256(seed_evidence),
            },
        )

    condition_evidence: dict[str, ConditionEvidence] = {}
    for condition in missing_conditions:
        validator = spec.condition_validators.get(condition)
        if validator is None:
            condition_evidence[condition] = _result(
                False,
                "validator_registry",
                reason="missing_explicit_condition_validator",
            )
        else:
            condition_evidence[condition] = validator(context)
    failed = [
        condition
        for condition in missing_conditions
        if not condition_evidence[condition]["passed"]
    ]
    return FilterCheck(
        stage=FilterStage.SKILL_TRIGGER,
        rejection_reasons=(
            () if not failed else (FilterRejection.SKILL_TRIGGER_NOT_REALIZED,)
        ),
        metrics={
            "validator_mode": "compatible_missing_condition_validation",
            "compatible_seed_detector_reused": False,
            "requested_role_track_ids": dict(context.role_track_ids),
            "required_missing_conditions": list(missing_conditions),
            "failed_conditions": failed,
            "condition_evidence": condition_evidence,
            "seed_evidence_sha256": _seed_evidence_sha256(seed_evidence),
        },
    )


__all__ = [
    "SKILL_VALIDATORS",
    "SkillValidatorSpec",
    "prepare_risk_context",
    "validate_skill_trigger",
]

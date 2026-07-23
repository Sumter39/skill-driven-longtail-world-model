"""Auditable post-overlay risk calculations for the 34 formal skills.

Every calculator reads the materialized 110-frame :class:`Scenario` and the
exact role-to-track binding supplied for the generated task.  Seed-time risk
values are deliberately not part of this module's API.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.assembly import HISTORY_STEPS, TOTAL_STEPS
from skilldrive.schemas import AgentTrack, Scenario, SkillSpec
from skilldrive.skills.geometry import (
    find_trajectory_conflict,
    minimum_trajectory_distance,
    time_to_collision,
)


EVALUATION_START_FRAME = HISTORY_STEPS - 1
FUTURE_START_FRAME = HISTORY_STEPS
EVALUATION_END_FRAME = TOTAL_STEPS - 1
RISK_CONTEXT_METADATA_KEY = "risk_context_v1"
_EPS = 1e-9
_STOPPING_DECELERATION_MPS2 = 6.0


class RiskStatus(str, Enum):
    """A finite result, a valid no-event result, or unavailable evidence."""

    COMPUTED = "computed"
    NO_EVENT = "no_event"
    UNAVAILABLE = "unavailable"


class RiskReason(str, Enum):
    """Stable reasons used when no finite scalar risk value is emitted."""

    ROLE_CONTRACT_MISMATCH = "role_contract_mismatch"
    TRACK_NOT_FOUND = "track_not_found"
    INVALID_SCENARIO_WINDOW = "invalid_scenario_window"
    INSUFFICIENT_VALID_SAMPLES = "insufficient_valid_samples"
    NO_PREDICTED_COLLISION = "no_predicted_collision"
    NO_UNIQUE_PATH_CONFLICT = "no_unique_path_conflict"
    NON_UNIQUE_PATH_CONFLICT = "non_unique_path_conflict"
    REQUIRED_CONTEXT_MISSING = "required_context_missing"
    INVALID_CONTEXT = "invalid_context"
    EVENT_NOT_OBSERVED = "event_not_observed"


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _freeze_audit_value(value: Any, path: str) -> Any:
    """Deep-freeze JSON-like audit evidence while rejecting NaN and infinity."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, np.ndarray):
        return tuple(
            _freeze_audit_value(item, f"{path}[]") for item in value.tolist()
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_audit_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_audit_value(item, f"{path}[]") for item in value)
    raise TypeError(f"{path} contains unsupported audit value {type(value).__name__}")


def _thaw_audit_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_audit_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_audit_value(item) for item in value]
    return value


@dataclass(frozen=True)
class RiskEvaluation:
    """One finite, serialization-safe post-overlay risk evaluation."""

    skill_id: str
    metric: str
    unit: str
    formula_version: str
    status: RiskStatus
    value: float | None
    role_track_ids: Mapping[str, str]
    reason: RiskReason | None = None
    window_start_frame: int = EVALUATION_START_FRAME
    window_end_frame: int = EVALUATION_END_FRAME
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("skill_id", "metric", "unit", "formula_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.status, RiskStatus):
            raise TypeError("status must be a RiskStatus")
        if self.status is RiskStatus.COMPUTED:
            if self.value is None:
                raise ValueError("computed risk must have a finite value")
            value = _finite_number(self.value, "value")
            if self.reason is not None:
                raise ValueError("computed risk cannot have a reason")
            object.__setattr__(self, "value", value)
        else:
            if self.value is not None:
                raise ValueError("non-computed risk cannot have a value")
            if not isinstance(self.reason, RiskReason):
                raise ValueError("non-computed risk must have a RiskReason")
        if (
            isinstance(self.window_start_frame, bool)
            or isinstance(self.window_end_frame, bool)
            or not isinstance(self.window_start_frame, int)
            or not isinstance(self.window_end_frame, int)
            or self.window_start_frame < 0
            or self.window_end_frame < self.window_start_frame
        ):
            raise ValueError("evaluation window must be an ordered pair of frame indices")

        roles: dict[str, str] = {}
        for role, track_id in self.role_track_ids.items():
            if not isinstance(role, str) or not role:
                raise ValueError("role names must be non-empty strings")
            if not isinstance(track_id, str) or not track_id:
                raise ValueError("track IDs must be non-empty strings")
            roles[role] = track_id
        object.__setattr__(self, "role_track_ids", MappingProxyType(roles))
        frozen_details = _freeze_audit_value(self.details, "details")
        if not isinstance(frozen_details, Mapping):
            raise TypeError("details must be a mapping")
        object.__setattr__(self, "details", frozen_details)

    @property
    def computed(self) -> bool:
        return self.status is RiskStatus.COMPUTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "metric": self.metric,
            "unit": self.unit,
            "formula_version": self.formula_version,
            "status": self.status.value,
            "value": self.value,
            "role_track_ids": dict(self.role_track_ids),
            "reason": None if self.reason is None else self.reason.value,
            "window_start_frame": self.window_start_frame,
            "window_end_frame": self.window_end_frame,
            "details": _thaw_audit_value(self.details),
        }


RiskCalculator = Callable[["_RiskContext"], RiskEvaluation]


@dataclass(frozen=True)
class RiskCalculatorSpec:
    """Frozen dispatch and formula metadata for one formal skill."""

    skill_id: str
    metric: str
    unit: str
    formula_version: str
    required_roles: tuple[str, ...]
    interaction_pairs: tuple[tuple[str, str], ...]
    calculator: RiskCalculator = field(repr=False, compare=False)
    context_frame_key: str | None = None

    def __post_init__(self) -> None:
        if not self.skill_id or not self.metric or not self.unit or not self.formula_version:
            raise ValueError("risk calculator text fields must be non-empty")
        if not self.required_roles or len(set(self.required_roles)) != len(
            self.required_roles
        ):
            raise ValueError(f"{self.skill_id} required_roles must be unique and non-empty")
        known = set(self.required_roles)
        if any(
            first not in known or second not in known or first == second
            for first, second in self.interaction_pairs
        ):
            raise ValueError(f"{self.skill_id} has an invalid interaction pair")
        if not callable(self.calculator):
            raise TypeError(f"{self.skill_id} calculator must be callable")


@dataclass(frozen=True)
class _RiskContext:
    scenario: Scenario
    spec: RiskCalculatorSpec
    role_track_ids: Mapping[str, str]
    agents_by_role: Mapping[str, AgentTrack]
    timestamps_s: np.ndarray


def _evaluation(
    context: _RiskContext,
    status: RiskStatus,
    *,
    value: float | None = None,
    reason: RiskReason | None = None,
    details: Mapping[str, Any] | None = None,
) -> RiskEvaluation:
    return RiskEvaluation(
        skill_id=context.spec.skill_id,
        metric=context.spec.metric,
        unit=context.spec.unit,
        formula_version=context.spec.formula_version,
        status=status,
        value=value,
        role_track_ids=context.role_track_ids,
        reason=reason,
        details={} if details is None else details,
    )


def _computed(
    context: _RiskContext,
    value: float,
    details: Mapping[str, Any],
) -> RiskEvaluation:
    return _evaluation(
        context,
        RiskStatus.COMPUTED,
        value=value,
        details=details,
    )


def _no_event(
    context: _RiskContext,
    reason: RiskReason,
    details: Mapping[str, Any] | None = None,
) -> RiskEvaluation:
    return _evaluation(
        context,
        RiskStatus.NO_EVENT,
        reason=reason,
        details=details,
    )


def _unavailable(
    context: _RiskContext,
    reason: RiskReason,
    details: Mapping[str, Any] | None = None,
) -> RiskEvaluation:
    return _evaluation(
        context,
        RiskStatus.UNAVAILABLE,
        reason=reason,
        details=details,
    )


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}->{pair[1]}"


def _future_time(context: _RiskContext, frame_index: int) -> float:
    return float(context.timestamps_s[frame_index])


def _remaining_time(context: _RiskContext, frame_index: int) -> float:
    return float(context.timestamps_s[EVALUATION_END_FRAME] - context.timestamps_s[frame_index])


def _heading_axis(agent: AgentTrack, frame_index: int) -> np.ndarray | None:
    heading = float(agent.headings[frame_index])
    if math.isfinite(heading):
        return np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    velocity = agent.velocities[frame_index]
    if np.isfinite(velocity).all():
        speed = float(np.linalg.norm(velocity))
        if speed > _EPS:
            return velocity.astype(np.float64) / speed
    return None


def _longitudinal_samples(
    context: _RiskContext,
    pair: tuple[str, str],
    *,
    start_frame: int = FUTURE_START_FRAME,
) -> list[dict[str, float | int]]:
    leader = context.agents_by_role[pair[0]]
    follower = context.agents_by_role[pair[1]]
    samples: list[dict[str, float | int]] = []
    for frame in range(start_frame, TOTAL_STEPS):
        if not (
            np.isfinite(leader.positions[frame]).all()
            and np.isfinite(follower.positions[frame]).all()
        ):
            continue
        axis = _heading_axis(follower, frame)
        if axis is None:
            continue
        gap = float(np.dot(leader.positions[frame] - follower.positions[frame], axis))
        sample: dict[str, float | int] = {
            "frame_index": frame,
            "time_s": _future_time(context, frame),
            "gap_m": gap,
        }
        if np.isfinite(leader.velocities[frame]).all() and np.isfinite(
            follower.velocities[frame]
        ).all():
            sample["leader_speed_mps"] = float(
                np.dot(leader.velocities[frame], axis)
            )
            sample["follower_speed_mps"] = float(
                np.dot(follower.velocities[frame], axis)
            )
        samples.append(sample)
    return samples


def _calculate_minimum_longitudinal_gap(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    for pair in context.spec.interaction_pairs:
        for sample in _longitudinal_samples(context, pair):
            candidates.append((float(sample["gap_m"]), _pair_label(pair), sample))
    if not candidates:
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": "minimum signed center gap projected on the follower heading",
            "selected_pair": pair_label,
            **sample,
        },
    )


def _calculate_time_headway(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    had_geometry = False
    for pair in context.spec.interaction_pairs:
        for sample in _longitudinal_samples(context, pair):
            had_geometry = True
            if "follower_speed_mps" not in sample:
                continue
            gap = float(sample["gap_m"])
            speed = float(sample["follower_speed_mps"])
            if gap <= 0.0:
                headway = 0.0
            elif speed <= _EPS:
                continue
            else:
                headway = gap / speed
            candidates.append((headway, _pair_label(pair), sample))
    if not candidates:
        reason = (
            RiskReason.NO_PREDICTED_COLLISION
            if had_geometry
            else RiskReason.INSUFFICIENT_VALID_SAMPLES
        )
        return _no_event(context, reason) if had_geometry else _unavailable(context, reason)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": "signed longitudinal center gap divided by follower forward speed",
            "selected_pair": pair_label,
            **sample,
        },
    )


def _calculate_stopping_distance_margin(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    for pair in context.spec.interaction_pairs:
        for sample in _longitudinal_samples(context, pair):
            if "leader_speed_mps" not in sample or "follower_speed_mps" not in sample:
                continue
            leader_speed = max(0.0, float(sample["leader_speed_mps"]))
            follower_speed = max(0.0, float(sample["follower_speed_mps"]))
            differential_stopping_distance = max(
                0.0,
                (follower_speed**2 - leader_speed**2)
                / (2.0 * _STOPPING_DECELERATION_MPS2),
            )
            margin = float(sample["gap_m"]) - differential_stopping_distance
            evidence = dict(sample)
            evidence["differential_stopping_distance_m"] = differential_stopping_distance
            candidates.append((margin, _pair_label(pair), evidence))
    if not candidates:
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": "gap - max(0, (follower_speed^2 - leader_speed^2) / (2a))",
            "assumed_equal_deceleration_mps2": _STOPPING_DECELERATION_MPS2,
            "reaction_time_s": 0.0,
            "selected_pair": pair_label,
            **sample,
        },
    )


def _longitudinal_ttc_candidates(
    context: _RiskContext,
    *,
    start_frame: int = FUTURE_START_FRAME,
) -> tuple[list[tuple[float, str, dict[str, float | int]]], bool]:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    had_samples = False
    for pair in context.spec.interaction_pairs:
        for sample in _longitudinal_samples(context, pair, start_frame=start_frame):
            if "leader_speed_mps" not in sample or "follower_speed_mps" not in sample:
                continue
            had_samples = True
            gap = float(sample["gap_m"])
            closing_speed = float(sample["follower_speed_mps"]) - float(
                sample["leader_speed_mps"]
            )
            if gap <= 0.0:
                ttc = 0.0
            elif closing_speed <= _EPS:
                continue
            else:
                ttc = gap / closing_speed
            frame = int(sample["frame_index"])
            if ttc > _remaining_time(context, frame) + _EPS:
                continue
            evidence = dict(sample)
            evidence["closing_speed_mps"] = closing_speed
            evidence["predicted_contact_time_s"] = float(sample["time_s"]) + ttc
            candidates.append((ttc, _pair_label(pair), evidence))
    return candidates, had_samples


def _calculate_minimum_front_rear_ttc(context: _RiskContext) -> RiskEvaluation:
    candidates, had_samples = _longitudinal_ttc_candidates(context)
    if not candidates:
        if had_samples:
            return _no_event(context, RiskReason.NO_PREDICTED_COLLISION)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "minimum longitudinal constant-velocity TTC over exact front "
                "and rear role pairs"
            ),
            "selected_pair": pair_label,
            **sample,
        },
    )


def _risk_context(context: _RiskContext) -> tuple[Mapping[str, Any] | None, RiskReason | None]:
    root = context.scenario.metadata.get(RISK_CONTEXT_METADATA_KEY)
    if root is None:
        return None, RiskReason.REQUIRED_CONTEXT_MISSING
    if not isinstance(root, Mapping):
        return None, RiskReason.INVALID_CONTEXT
    item = root.get(context.spec.skill_id)
    if item is None:
        return None, RiskReason.REQUIRED_CONTEXT_MISSING
    if not isinstance(item, Mapping):
        return None, RiskReason.INVALID_CONTEXT
    return item, None


def _context_frame(
    context: _RiskContext,
) -> tuple[int | None, RiskReason | None]:
    item, reason = _risk_context(context)
    if item is None:
        return None, reason
    key = context.spec.context_frame_key
    if key is None or key not in item:
        return None, RiskReason.REQUIRED_CONTEXT_MISSING
    value = item[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not FUTURE_START_FRAME <= int(value) <= EVALUATION_END_FRAME
    ):
        return None, RiskReason.INVALID_CONTEXT
    return int(value), None


def _calculate_contextual_longitudinal_ttc(context: _RiskContext) -> RiskEvaluation:
    frame, reason = _context_frame(context)
    if frame is None:
        return _unavailable(context, reason or RiskReason.INVALID_CONTEXT)
    candidates, had_samples = _longitudinal_ttc_candidates(context, start_frame=frame)
    if not candidates:
        details = {"context_frame_key": context.spec.context_frame_key, "event_frame": frame}
        if had_samples:
            return _no_event(context, RiskReason.NO_PREDICTED_COLLISION, details)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES, details)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "minimum longitudinal constant-velocity TTC at or after an "
                "explicit post-overlay event frame"
            ),
            "context_frame_key": context.spec.context_frame_key,
            "event_frame": frame,
            "selected_pair": pair_label,
            **sample,
        },
    )


def _point_ttc_candidates(
    context: _RiskContext,
) -> tuple[list[tuple[float, str, dict[str, float | int]]], bool]:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    had_samples = False
    for pair in context.spec.interaction_pairs:
        first = context.agents_by_role[pair[0]]
        second = context.agents_by_role[pair[1]]
        for frame in range(FUTURE_START_FRAME, TOTAL_STEPS):
            if not (
                np.isfinite(first.positions[frame]).all()
                and np.isfinite(second.positions[frame]).all()
                and np.isfinite(first.velocities[frame]).all()
                and np.isfinite(second.velocities[frame]).all()
            ):
                continue
            had_samples = True
            value = time_to_collision(
                second.positions[frame] - first.positions[frame],
                second.velocities[frame] - first.velocities[frame],
            )
            if not math.isfinite(value) or value > _remaining_time(context, frame) + _EPS:
                continue
            candidates.append(
                (
                    value,
                    _pair_label(pair),
                    {
                        "frame_index": frame,
                        "time_s": _future_time(context, frame),
                        "predicted_contact_time_s": _future_time(context, frame) + value,
                    },
                )
            )
    return candidates, had_samples


def _calculate_point_mass_ttc(context: _RiskContext) -> RiskEvaluation:
    candidates, had_samples = _point_ttc_candidates(context)
    if not candidates:
        if had_samples:
            return _no_event(context, RiskReason.NO_PREDICTED_COLLISION)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "point-mass constant-relative-velocity TTC within the remaining "
                "overlay horizon"
            ),
            "collision_radius_m": 0.0,
            "selected_pair": pair_label,
            **sample,
        },
    )


def _calculate_radial_head_on_ttc(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    had_samples = False
    for pair in context.spec.interaction_pairs:
        first = context.agents_by_role[pair[0]]
        second = context.agents_by_role[pair[1]]
        for frame in range(FUTURE_START_FRAME, TOTAL_STEPS):
            if not (
                np.isfinite(first.positions[frame]).all()
                and np.isfinite(second.positions[frame]).all()
                and np.isfinite(first.velocities[frame]).all()
                and np.isfinite(second.velocities[frame]).all()
            ):
                continue
            had_samples = True
            relative_position = second.positions[frame] - first.positions[frame]
            distance = float(np.linalg.norm(relative_position))
            if distance <= _EPS:
                value = 0.0
                closing_speed = 0.0
            else:
                line_of_sight = relative_position / distance
                relative_velocity = second.velocities[frame] - first.velocities[frame]
                closing_speed = -float(np.dot(relative_velocity, line_of_sight))
                if closing_speed <= _EPS:
                    continue
                value = distance / closing_speed
            if value > _remaining_time(context, frame) + _EPS:
                continue
            candidates.append(
                (
                    value,
                    _pair_label(pair),
                    {
                        "frame_index": frame,
                        "time_s": _future_time(context, frame),
                        "center_distance_m": distance,
                        "radial_closing_speed_mps": closing_speed,
                        "predicted_contact_time_s": _future_time(context, frame) + value,
                    },
                )
            )
    if not candidates:
        if had_samples:
            return _no_event(context, RiskReason.NO_PREDICTED_COLLISION)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": "center distance divided by line-of-sight closing speed",
            "selected_pair": pair_label,
            **sample,
        },
    )


def _calculate_lateral_ttc(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, float | int]]] = []
    had_samples = False
    for pair in context.spec.interaction_pairs:
        first = context.agents_by_role[pair[0]]
        second = context.agents_by_role[pair[1]]
        for frame in range(FUTURE_START_FRAME, TOTAL_STEPS):
            if not (
                np.isfinite(first.positions[frame]).all()
                and np.isfinite(second.positions[frame]).all()
                and np.isfinite(first.velocities[frame]).all()
                and np.isfinite(second.velocities[frame]).all()
            ):
                continue
            first_axis = _heading_axis(first, frame)
            second_axis = _heading_axis(second, frame)
            if first_axis is None or second_axis is None:
                continue
            common_axis = first_axis + second_axis
            norm = float(np.linalg.norm(common_axis))
            if norm <= _EPS:
                common_axis = first_axis
            else:
                common_axis /= norm
            lateral_axis = np.array([-common_axis[1], common_axis[0]])
            separation = float(
                np.dot(second.positions[frame] - first.positions[frame], lateral_axis)
            )
            relative_lateral_velocity = float(
                np.dot(second.velocities[frame] - first.velocities[frame], lateral_axis)
            )
            had_samples = True
            if abs(separation) <= _EPS:
                value = 0.0
                closing_speed = 0.0
            else:
                value = (
                    -separation / relative_lateral_velocity
                    if abs(relative_lateral_velocity) > _EPS
                    else -1.0
                )
                closing_speed = -math.copysign(relative_lateral_velocity, separation)
                if value < 0.0:
                    continue
            if value > _remaining_time(context, frame) + _EPS:
                continue
            candidates.append(
                (
                    value,
                    _pair_label(pair),
                    {
                        "frame_index": frame,
                        "time_s": _future_time(context, frame),
                        "absolute_lateral_separation_m": abs(separation),
                        "lateral_closing_speed_mps": closing_speed,
                        "predicted_lateral_coincidence_time_s": _future_time(context, frame)
                        + value,
                    },
                )
            )
    if not candidates:
        if had_samples:
            return _no_event(context, RiskReason.NO_PREDICTED_COLLISION)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, sample = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "lateral center separation divided by closing speed on the "
                "mean-heading normal"
            ),
            "selected_pair": pair_label,
            **sample,
        },
    )


def _has_collinear_path_overlap(
    first: np.ndarray,
    second: np.ndarray,
    first_valid: np.ndarray,
    second_valid: np.ndarray,
) -> bool:
    for first_index in range(len(first) - 1):
        if not (first_valid[first_index] and first_valid[first_index + 1]):
            continue
        first_start = first[first_index]
        first_delta = first[first_index + 1] - first_start
        first_length = float(np.linalg.norm(first_delta))
        if first_length <= _EPS:
            continue
        first_axis = first_delta / first_length
        for second_index in range(len(second) - 1):
            if not (second_valid[second_index] and second_valid[second_index + 1]):
                continue
            second_start = second[second_index]
            second_delta = second[second_index + 1] - second_start
            second_length = float(np.linalg.norm(second_delta))
            if second_length <= _EPS:
                continue
            second_axis = second_delta / second_length
            direction_cross = first_axis[0] * second_axis[1] - first_axis[1] * second_axis[0]
            if abs(float(direction_cross)) > 1e-7:
                continue
            offset = second_start - first_start
            offset_cross = first_axis[0] * offset[1] - first_axis[1] * offset[0]
            if abs(float(offset_cross)) > 1e-7:
                continue
            first_interval = (0.0, first_length)
            second_projection = float(np.dot(second_start - first_start, first_axis))
            second_end_projection = second_projection + float(
                np.dot(second_delta, first_axis)
            )
            second_interval = (
                min(second_projection, second_end_projection),
                max(second_projection, second_end_projection),
            )
            if min(first_interval[1], second_interval[1]) - max(
                first_interval[0], second_interval[0]
            ) > _EPS:
                return True
    return False


def _calculate_path_arrival_gap(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    had_paths = False
    ambiguous_pairs: list[str] = []
    start = EVALUATION_START_FRAME
    end = TOTAL_STEPS
    times = context.timestamps_s[start:end]
    for pair in context.spec.interaction_pairs:
        first = context.agents_by_role[pair[0]].positions[start:end]
        second = context.agents_by_role[pair[1]].positions[start:end]
        first_valid = np.isfinite(first).all(axis=1)
        second_valid = np.isfinite(second).all(axis=1)
        if first_valid.sum() < 2 or second_valid.sum() < 2:
            continue
        had_paths = True
        conflict = find_trajectory_conflict(
            first,
            second,
            times,
            times,
            first_valid_mask=first_valid,
            second_valid_mask=second_valid,
        )
        if conflict is None:
            if _has_collinear_path_overlap(first, second, first_valid, second_valid):
                ambiguous_pairs.append(_pair_label(pair))
            continue
        values = (
            conflict.time_gap_s,
            conflict.first_time_s,
            conflict.second_time_s,
            conflict.point[0],
            conflict.point[1],
        )
        if not all(math.isfinite(float(value)) for value in values):
            continue
        candidates.append(
            (
                float(conflict.time_gap_s),
                _pair_label(pair),
                {
                    "conflict_point_xy": conflict.point,
                    "first_arrival_time_s": conflict.first_time_s,
                    "second_arrival_time_s": conflict.second_time_s,
                    "first_segment_start_frame": start + conflict.first_segment_index,
                    "second_segment_start_frame": start + conflict.second_segment_index,
                },
            )
        )
    if not candidates:
        if ambiguous_pairs:
            return _unavailable(
                context,
                RiskReason.NON_UNIQUE_PATH_CONFLICT,
                {"ambiguous_pairs": ambiguous_pairs},
            )
        if had_paths:
            return _no_event(context, RiskReason.NO_UNIQUE_PATH_CONFLICT)
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, evidence = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "absolute arrival-time gap at a unique piecewise-linear "
                "center-path intersection"
            ),
            "collinear_overlap_policy": "unavailable_non_unique_conflict",
            "selected_pair": pair_label,
            **evidence,
        },
    )


def _calculate_minimum_center_clearance(context: _RiskContext) -> RiskEvaluation:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    start = FUTURE_START_FRAME
    times = context.timestamps_s[start:TOTAL_STEPS]
    for pair in context.spec.interaction_pairs:
        first = context.agents_by_role[pair[0]].positions[start:TOTAL_STEPS]
        second = context.agents_by_role[pair[1]].positions[start:TOTAL_STEPS]
        first_valid = np.isfinite(first).all(axis=1)
        second_valid = np.isfinite(second).all(axis=1)
        result = minimum_trajectory_distance(
            first,
            second,
            times,
            first_valid_mask=first_valid,
            second_valid_mask=second_valid,
        )
        if result is None or not math.isfinite(result.distance_m):
            continue
        evidence: dict[str, Any] = {
            "frame_index": start + result.frame_index,
            "time_s": result.time_s,
            "first_center_xy": result.first_point,
            "second_center_xy": result.second_point,
        }
        candidates.append((result.distance_m, _pair_label(pair), evidence))
    if not candidates:
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, pair_label, evidence = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "minimum synchronized piecewise-linear center-to-center "
                "distance over exact role pairs"
            ),
            "physical_footprint_clearance": False,
            "selected_pair": pair_label,
            **evidence,
        },
    )


def _context_polygon(
    context: _RiskContext,
) -> tuple[np.ndarray | None, RiskReason | None]:
    item, reason = _risk_context(context)
    if item is None:
        return None, reason
    value = item.get("conflict_area_polygon_xy")
    if value is None:
        return None, RiskReason.REQUIRED_CONTEXT_MISSING
    try:
        polygon = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None, RiskReason.INVALID_CONTEXT
    if (
        polygon.ndim != 2
        or polygon.shape[1] != 2
        or len(polygon) < 3
        or not np.isfinite(polygon).all()
    ):
        return None, RiskReason.INVALID_CONTEXT
    if np.linalg.norm(polygon[0] - polygon[-1]) <= _EPS:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        return None, RiskReason.INVALID_CONTEXT
    area = 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )
    if area <= _EPS:
        return None, RiskReason.INVALID_CONTEXT
    return polygon, None


def _point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
    delta = end - start
    squared_length = float(np.dot(delta, delta))
    if squared_length <= _EPS:
        return float(np.linalg.norm(point - start)) <= _EPS
    fraction = float(np.dot(point - start, delta) / squared_length)
    if not -_EPS <= fraction <= 1.0 + _EPS:
        return False
    projection = start + float(np.clip(fraction, 0.0, 1.0)) * delta
    return float(np.linalg.norm(point - projection)) <= _EPS


def _point_inside_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, start, end):
            return True
        y1, y2 = float(start[1]), float(end[1])
        if (y1 > y) == (y2 > y):
            continue
        x_intersection = float(start[0]) + (y - y1) * float(end[0] - start[0]) / (
            y2 - y1
        )
        if x < x_intersection:
            inside = not inside
    return inside


def _point_polygon_distance(point: np.ndarray, polygon: np.ndarray) -> float:
    if _point_inside_polygon(point, polygon):
        return 0.0
    distances: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        delta = end - start
        squared_length = float(np.dot(delta, delta))
        if squared_length <= _EPS:
            projection = start
        else:
            fraction = float(
                np.clip(np.dot(point - start, delta) / squared_length, 0.0, 1.0)
            )
            projection = start + fraction * delta
        distances.append(float(np.linalg.norm(point - projection)))
    return min(distances)


def _calculate_conflict_area_intrusion_margin(context: _RiskContext) -> RiskEvaluation:
    polygon, reason = _context_polygon(context)
    if polygon is None:
        return _unavailable(context, reason or RiskReason.INVALID_CONTEXT)
    actor = context.agents_by_role[context.spec.required_roles[0]]
    candidates: list[tuple[float, int, np.ndarray]] = []
    for frame in range(FUTURE_START_FRAME, TOTAL_STEPS):
        point = actor.positions[frame]
        if not np.isfinite(point).all():
            continue
        candidates.append((_point_polygon_distance(point, polygon), frame, point))
    if not candidates:
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    value, frame, point = min(candidates, key=lambda item: (item[0], item[1]))
    return _computed(
        context,
        value,
        {
            "definition": (
                "minimum actor-center distance to an explicit conflict-area "
                "polygon; inside or boundary equals zero"
            ),
            "context_key": (
                f"{RISK_CONTEXT_METADATA_KEY}.{context.spec.skill_id}."
                "conflict_area_polygon_xy"
            ),
            "polygon_vertex_count": len(polygon),
            "frame_index": frame,
            "time_s": _future_time(context, frame),
            "actor_center_xy": point,
        },
    )


def _calculate_conflict_area_occupancy_overlap(context: _RiskContext) -> RiskEvaluation:
    polygon, reason = _context_polygon(context)
    if polygon is None:
        return _unavailable(context, reason or RiskReason.INVALID_CONTEXT)
    pair = context.spec.interaction_pairs[0]
    first = context.agents_by_role[pair[0]]
    second = context.agents_by_role[pair[1]]
    duration = 0.0
    valid_intervals = 0
    overlap_intervals = 0
    for frame in range(FUTURE_START_FRAME, EVALUATION_END_FRAME):
        if not (
            np.isfinite(first.positions[frame]).all()
            and np.isfinite(second.positions[frame]).all()
        ):
            continue
        interval = float(context.timestamps_s[frame + 1] - context.timestamps_s[frame])
        valid_intervals += 1
        if _point_inside_polygon(first.positions[frame], polygon) and _point_inside_polygon(
            second.positions[frame], polygon
        ):
            duration += interval
            overlap_intervals += 1
    if not valid_intervals:
        return _unavailable(context, RiskReason.INSUFFICIENT_VALID_SAMPLES)
    return _computed(
        context,
        duration,
        {
            "definition": (
                "sum of timestamp intervals whose left-end samples place both "
                "actor centers inside the explicit conflict polygon"
            ),
            "sampling_rule": "left_endpoint_zero_order_hold",
            "context_key": (
                f"{RISK_CONTEXT_METADATA_KEY}.{context.spec.skill_id}."
                "conflict_area_polygon_xy"
            ),
            "polygon_vertex_count": len(polygon),
            "valid_interval_count": valid_intervals,
            "overlap_interval_count": overlap_intervals,
            "selected_pair": _pair_label(pair),
        },
    )


def _calculate_first_intrusion_ttc(context: _RiskContext) -> RiskEvaluation:
    polygon, reason = _context_polygon(context)
    if polygon is None:
        return _unavailable(context, reason or RiskReason.INVALID_CONTEXT)
    emerging_role, vehicle_role = context.spec.interaction_pairs[0]
    emerging = context.agents_by_role[emerging_role]
    vehicle = context.agents_by_role[vehicle_role]
    intrusion_frame: int | None = None
    for frame in range(FUTURE_START_FRAME, TOTAL_STEPS):
        previous = frame - 1
        if not (
            np.isfinite(emerging.positions[previous]).all()
            and np.isfinite(emerging.positions[frame]).all()
        ):
            continue
        if not _point_inside_polygon(
            emerging.positions[previous], polygon
        ) and _point_inside_polygon(emerging.positions[frame], polygon):
            intrusion_frame = frame
            break
    if intrusion_frame is None:
        return _no_event(context, RiskReason.EVENT_NOT_OBSERVED)
    frame = intrusion_frame
    if not (
        np.isfinite(emerging.velocities[frame]).all()
        and np.isfinite(vehicle.positions[frame]).all()
        and np.isfinite(vehicle.velocities[frame]).all()
    ):
        return _unavailable(
            context,
            RiskReason.INSUFFICIENT_VALID_SAMPLES,
            {"intrusion_frame": frame},
        )
    value = time_to_collision(
        emerging.positions[frame] - vehicle.positions[frame],
        emerging.velocities[frame] - vehicle.velocities[frame],
    )
    details = {
        "intrusion_frame": frame,
        "intrusion_time_s": _future_time(context, frame),
        "intrusion_point_xy": emerging.positions[frame],
        "collision_radius_m": 0.0,
        "context_key": (
            f"{RISK_CONTEXT_METADATA_KEY}.{context.spec.skill_id}."
            "conflict_area_polygon_xy"
        ),
    }
    if not math.isfinite(value) or value > _remaining_time(context, frame) + _EPS:
        return _no_event(context, RiskReason.NO_PREDICTED_COLLISION, details)
    details["predicted_contact_time_s"] = _future_time(context, frame) + value
    return _computed(context, value, details)


def _spec(
    skill_id: str,
    metric: str,
    unit: str,
    formula_version: str,
    required_roles: tuple[str, ...],
    interaction_pairs: tuple[tuple[str, str], ...],
    calculator: RiskCalculator,
    *,
    context_frame_key: str | None = None,
) -> RiskCalculatorSpec:
    return RiskCalculatorSpec(
        skill_id=skill_id,
        metric=metric,
        unit=unit,
        formula_version=formula_version,
        required_roles=required_roles,
        interaction_pairs=interaction_pairs,
        calculator=calculator,
        context_frame_key=context_frame_key,
    )


_PATH_GAP = _calculate_path_arrival_gap
_CENTER_CLEARANCE = _calculate_minimum_center_clearance
_POINT_TTC = _calculate_point_mass_ttc


SKILL_RISK_CALCULATORS: Mapping[str, RiskCalculatorSpec] = MappingProxyType(
    {
        "lead_sudden_stop": _spec(
            "lead_sudden_stop", "stopping_distance_margin", "m",
            "stopping_distance_margin.same_deceleration_6mps2.v1",
            ("stopping_leader", "follower"), (("stopping_leader", "follower"),),
            _calculate_stopping_distance_margin,
        ),
        "slow_lead_blockage": _spec(
            "slow_lead_blockage", "minimum_longitudinal_gap", "m",
            "minimum_longitudinal_gap.follower_heading_projection.v1",
            ("slow_leader", "follower"), (("slow_leader", "follower"),),
            _calculate_minimum_longitudinal_gap,
        ),
        "short_headway_following": _spec(
            "short_headway_following", "time_headway", "s",
            "time_headway.follower_heading_projection.v1",
            ("leader", "close_follower"), (("leader", "close_follower"),),
            _calculate_time_headway,
        ),
        "chain_braking": _spec(
            "chain_braking", "minimum_longitudinal_gap", "m",
            "minimum_longitudinal_gap.ordered_triple_projection.v1",
            ("front_vehicle", "middle_vehicle", "rear_vehicle"),
            (("front_vehicle", "middle_vehicle"), ("middle_vehicle", "rear_vehicle")),
            _calculate_minimum_longitudinal_gap,
        ),
        "cut_out_reveals_slow_vehicle": _spec(
            "cut_out_reveals_slow_vehicle", "newly_exposed_time_to_collision", "s",
            "newly_exposed_ttc.explicit_exposure_frame_longitudinal.v1",
            ("cut_out_vehicle", "target_vehicle", "slow_vehicle"),
            (("slow_vehicle", "target_vehicle"),), _calculate_contextual_longitudinal_ttc,
            context_frame_key="exposure_frame_index",
        ),
        "simultaneous_lane_change_conflict": _spec(
            "simultaneous_lane_change_conflict", "lateral_time_to_collision", "s",
            "lateral_ttc.mean_heading_normal.v1",
            ("left_lane_changer", "right_lane_changer"),
            (("left_lane_changer", "right_lane_changer"),), _calculate_lateral_ttc,
        ),
        "forced_lane_change_around_blockage": _spec(
            "forced_lane_change_around_blockage", "minimum_combined_clearance", "m",
            "minimum_combined_clearance.exact_role_center_distance.v1",
            ("blocking_actor", "avoiding_vehicle"), (("blocking_actor", "avoiding_vehicle"),),
            _CENTER_CLEARANCE,
        ),
        "late_lane_change_before_diverge": _spec(
            "late_lane_change_before_diverge", "minimum_crossing_time_to_collision", "s",
            "minimum_crossing_ttc.point_mass_constant_velocity.v1",
            ("late_lane_changer", "adjacent_lane_vehicle"),
            (("late_lane_changer", "adjacent_lane_vehicle"),), _POINT_TTC,
        ),
        "ramp_merge_small_gap": _spec(
            "ramp_merge_small_gap", "conflict_point_time_gap", "s",
            "conflict_point_time_gap.unique_path_intersection.v1",
            ("merging_vehicle", "mainline_vehicle"), (("merging_vehicle", "mainline_vehicle"),),
            _PATH_GAP,
        ),
        "lane_drop_merge_competition": _spec(
            "lane_drop_merge_competition", "convergence_time_gap", "s",
            "convergence_time_gap.unique_path_intersection.v1",
            ("closing_lane_vehicle", "continuing_lane_vehicle"),
            (("closing_lane_vehicle", "continuing_lane_vehicle"),), _PATH_GAP,
        ),
        "merge_without_yield": _spec(
            "merge_without_yield", "conflict_point_time_gap", "s",
            "conflict_point_time_gap.unique_path_intersection.v1",
            ("non_yielding_vehicle", "priority_vehicle"),
            (("non_yielding_vehicle", "priority_vehicle"),), _PATH_GAP,
        ),
        "diverge_lane_crossing_conflict": _spec(
            "diverge_lane_crossing_conflict", "minimum_crossing_time_to_collision", "s",
            "minimum_crossing_ttc.point_mass_constant_velocity.v1",
            ("crossing_vehicle", "through_vehicle"), (("crossing_vehicle", "through_vehicle"),),
            _POINT_TTC,
        ),
        "bike_lane_vehicle_merge_conflict": _spec(
            "bike_lane_vehicle_merge_conflict", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("cyclist", "motor_vehicle"), (("cyclist", "motor_vehicle"),), _PATH_GAP,
        ),
        "zipper_merge_multi_vehicle": _spec(
            "zipper_merge_multi_vehicle", "convergence_time_gap", "s",
            "convergence_time_gap.unique_path_intersection.v1",
            ("merging_vehicle", "leading_main_flow_vehicle", "trailing_main_flow_vehicle"),
            (("merging_vehicle", "leading_main_flow_vehicle"),
             ("merging_vehicle", "trailing_main_flow_vehicle")), _PATH_GAP,
        ),
        "unprotected_left_turn_conflict": _spec(
            "unprotected_left_turn_conflict", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("left_turn_vehicle", "opposing_through_vehicle"),
            (("left_turn_vehicle", "opposing_through_vehicle"),), _PATH_GAP,
        ),
        "right_turn_vehicle_conflict": _spec(
            "right_turn_vehicle_conflict", "conflict_point_time_gap", "s",
            "conflict_point_time_gap.unique_path_intersection.v1",
            ("right_turn_vehicle", "conflicting_vehicle"),
            (("right_turn_vehicle", "conflicting_vehicle"),), _PATH_GAP,
        ),
        "crossing_path_conflict": _spec(
            "crossing_path_conflict", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("first_vehicle", "second_vehicle"), (("first_vehicle", "second_vehicle"),),
            _PATH_GAP,
        ),
        "intersection_creep_conflict": _spec(
            "intersection_creep_conflict", "conflict_area_intrusion_margin", "m",
            "conflict_area_intrusion_margin.explicit_polygon_center_distance.v1",
            ("creeping_vehicle", "crossing_vehicle"), (("creeping_vehicle", "crossing_vehicle"),),
            _calculate_conflict_area_intrusion_margin,
        ),
        "intersection_blocking_vehicle": _spec(
            "intersection_blocking_vehicle", "conflict_area_occupancy_overlap", "s",
            "conflict_area_occupancy_overlap.explicit_polygon_left_sample.v1",
            ("blocking_vehicle", "crossing_vehicle"), (("blocking_vehicle", "crossing_vehicle"),),
            _calculate_conflict_area_occupancy_overlap,
        ),
        "mutual_yield_deadlock": _spec(
            "mutual_yield_deadlock", "conflict_point_time_gap", "s",
            "conflict_point_time_gap.unique_path_intersection.v1",
            ("first_yielding_vehicle", "second_yielding_vehicle"),
            (("first_yielding_vehicle", "second_yielding_vehicle"),), _PATH_GAP,
        ),
        "crosswalk_pedestrian_crossing": _spec(
            "crosswalk_pedestrian_crossing", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("pedestrian", "yielding_vehicle"), (("pedestrian", "yielding_vehicle"),),
            _PATH_GAP,
        ),
        "jaywalking_pedestrian_crossing": _spec(
            "jaywalking_pedestrian_crossing", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("pedestrian", "responding_vehicle"), (("pedestrian", "responding_vehicle"),),
            _PATH_GAP,
        ),
        "roadside_pedestrian_emergence": _spec(
            "roadside_pedestrian_emergence", "first_intrusion_time_to_collision", "s",
            "first_intrusion_ttc.explicit_polygon_point_mass.v1",
            ("emerging_pedestrian", "responding_vehicle"),
            (("emerging_pedestrian", "responding_vehicle"),), _calculate_first_intrusion_ttc,
        ),
        "cyclist_crossing": _spec(
            "cyclist_crossing", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("crossing_cyclist", "responding_vehicle"),
            (("crossing_cyclist", "responding_vehicle"),), _PATH_GAP,
        ),
        "turning_vehicle_crosswalk_conflict": _spec(
            "turning_vehicle_crosswalk_conflict", "post_encroachment_time", "s",
            "post_encroachment_time.unique_path_arrival_gap.v1",
            ("pedestrian", "turning_vehicle"), (("pedestrian", "turning_vehicle"),),
            _PATH_GAP,
        ),
        "group_pedestrian_crossing": _spec(
            "group_pedestrian_crossing", "post_encroachment_time", "s",
            "post_encroachment_time.minimum_group_path_arrival_gap.v1",
            ("first_crossing_pedestrian", "responding_vehicle", "second_crossing_pedestrian"),
            (("first_crossing_pedestrian", "responding_vehicle"),
             ("second_crossing_pedestrian", "responding_vehicle")), _PATH_GAP,
        ),
        "cyclist_vehicle_merge": _spec(
            "cyclist_vehicle_merge", "minimum_front_rear_time_to_collision", "s",
            "minimum_front_rear_ttc.longitudinal_constant_velocity.v1",
            ("merging_cyclist", "front_motor_vehicle", "rear_motor_vehicle"),
            (("front_motor_vehicle", "merging_cyclist"),
             ("merging_cyclist", "rear_motor_vehicle")), _calculate_minimum_front_rear_ttc,
        ),
        "stopped_vehicle_reentry": _spec(
            "stopped_vehicle_reentry", "minimum_front_rear_time_to_collision", "s",
            "minimum_front_rear_ttc.longitudinal_constant_velocity.v1",
            ("reentering_vehicle", "front_main_flow_vehicle", "rear_main_flow_vehicle"),
            (("front_main_flow_vehicle", "reentering_vehicle"),
             ("reentering_vehicle", "rear_main_flow_vehicle")), _calculate_minimum_front_rear_ttc,
        ),
        "construction_object_lane_blockage": _spec(
            "construction_object_lane_blockage", "minimum_object_clearance", "m",
            "minimum_object_clearance.exact_role_center_distance.v1",
            ("construction_object", "responding_vehicle"),
            (("construction_object", "responding_vehicle"),), _CENTER_CLEARANCE,
        ),
        "static_object_avoidance": _spec(
            "static_object_avoidance", "minimum_object_clearance", "m",
            "minimum_object_clearance.exact_role_center_distance.v1",
            ("static_object", "avoiding_vehicle"), (("static_object", "avoiding_vehicle"),),
            _CENTER_CLEARANCE,
        ),
        "cut_in_then_brake": _spec(
            "cut_in_then_brake", "minimum_stage_time_to_collision", "s",
            "minimum_stage_ttc.explicit_stage_frame_longitudinal.v1",
            ("cut_in_braking_vehicle", "responding_vehicle"),
            (("cut_in_braking_vehicle", "responding_vehicle"),),
            _calculate_contextual_longitudinal_ttc, context_frame_key="stage_start_frame_index",
        ),
        "abrupt_u_turn_conflict": _spec(
            "abrupt_u_turn_conflict", "head_on_time_to_collision", "s",
            "head_on_ttc.radial_closing_speed.v1",
            ("u_turning_vehicle", "conflicting_vehicle"),
            (("u_turning_vehicle", "conflicting_vehicle"),), _calculate_radial_head_on_ttc,
        ),
        "multi_vehicle_gap_squeeze": _spec(
            "multi_vehicle_gap_squeeze", "minimum_combined_clearance", "m",
            "minimum_combined_clearance.exact_role_center_distance.v1",
            ("squeezed_vehicle", "front_pressure_vehicle", "rear_pressure_vehicle"),
            (("squeezed_vehicle", "front_pressure_vehicle"),
             ("squeezed_vehicle", "rear_pressure_vehicle")), _CENTER_CLEARANCE,
        ),
        "motorcyclist_filtering_conflict": _spec(
            "motorcyclist_filtering_conflict", "minimum_combined_clearance", "m",
            "minimum_combined_clearance.exact_role_center_distance.v1",
            ("filtering_motorcyclist", "first_vehicle", "second_vehicle"),
            (("filtering_motorcyclist", "first_vehicle"),
             ("filtering_motorcyclist", "second_vehicle")), _CENTER_CLEARANCE,
        ),
    }
)


def evaluate_skill_risk(
    scenario: Scenario,
    skill_id: str,
    role_track_ids: Mapping[str, str],
) -> RiskEvaluation:
    """Recompute one formal skill's target risk from an overlaid full scenario.

    The supplied role mapping must exactly match the registry contract.  No
    seed record or seed risk scalar is accepted, preventing accidental reuse.
    """

    if skill_id not in SKILL_RISK_CALCULATORS:
        raise KeyError(f"unknown formal skill_id: {skill_id}")
    spec = SKILL_RISK_CALCULATORS[skill_id]
    normalized_roles = {
        str(role): str(track_id) for role, track_id in role_track_ids.items()
    }
    placeholder = _RiskContext(
        scenario=scenario,
        spec=spec,
        role_track_ids=normalized_roles,
        agents_by_role={},
        timestamps_s=np.empty(0, dtype=np.float64),
    )
    if (
        set(normalized_roles) != set(spec.required_roles)
        or len(set(normalized_roles.values())) != len(normalized_roles)
    ):
        return _unavailable(
            placeholder,
            RiskReason.ROLE_CONTRACT_MISMATCH,
            {
                "required_roles": spec.required_roles,
                "provided_roles": tuple(sorted(normalized_roles)),
            },
        )

    agents_by_id = {agent.track_id: agent for agent in scenario.agents}
    missing = tuple(
        role for role in spec.required_roles if normalized_roles[role] not in agents_by_id
    )
    if missing:
        return _unavailable(
            placeholder,
            RiskReason.TRACK_NOT_FOUND,
            {"missing_roles": missing},
        )
    agents_by_role = {
        role: agents_by_id[normalized_roles[role]] for role in spec.required_roles
    }
    invalid_tracks = tuple(
        role
        for role, agent in agents_by_role.items()
        if len(agent.positions) < TOTAL_STEPS
        or len(agent.velocities) < TOTAL_STEPS
        or len(agent.headings) < TOTAL_STEPS
    )
    timestamps = np.asarray(scenario.timestamps, dtype=np.int64)
    if (
        len(timestamps) < TOTAL_STEPS
        or np.any(np.diff(timestamps[:TOTAL_STEPS].astype(np.float64)) <= 0.0)
        or invalid_tracks
    ):
        return _unavailable(
            placeholder,
            RiskReason.INVALID_SCENARIO_WINDOW,
            {
                "scenario_steps": len(timestamps),
                "invalid_track_roles": invalid_tracks,
            },
        )
    timestamps_s = (
        timestamps[:TOTAL_STEPS].astype(np.float64)
        - float(timestamps[EVALUATION_START_FRAME])
    ) / 1_000_000_000.0
    context = _RiskContext(
        scenario=scenario,
        spec=spec,
        role_track_ids=normalized_roles,
        agents_by_role=MappingProxyType(agents_by_role),
        timestamps_s=timestamps_s,
    )
    return spec.calculator(context)


def check_target_risk(skill: SkillSpec, evaluation: RiskEvaluation) -> FilterCheck:
    """Require the recomputed metric to match the frozen finite target interval."""

    definition = skill.risk_definition
    expected_metric = str(definition["metric"])
    target_range = tuple(float(value) for value in definition["target_range"])
    reasons: list[FilterRejection] = []
    if evaluation.metric != expected_metric:
        reasons.append(FilterRejection.RISK_METRIC_MISMATCH)
    if evaluation.status is not RiskStatus.COMPUTED or evaluation.value is None:
        reasons.append(FilterRejection.RISK_METRIC_UNAVAILABLE)
    elif not math.isfinite(evaluation.value):
        reasons.append(FilterRejection.RISK_NON_FINITE)
    elif not target_range[0] <= evaluation.value <= target_range[1]:
        reasons.append(FilterRejection.RISK_OUT_OF_TARGET_RANGE)
    return FilterCheck(
        stage=FilterStage.TARGET_RISK,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        metrics={
            "expected_metric": expected_metric,
            "target_range": list(target_range),
            "direction": str(definition["direction"]),
            "target_source": str(definition["source"]),
            "evaluation": evaluation.to_dict(),
        },
    )


__all__ = [
    "EVALUATION_END_FRAME",
    "EVALUATION_START_FRAME",
    "FUTURE_START_FRAME",
    "RISK_CONTEXT_METADATA_KEY",
    "SKILL_RISK_CALCULATORS",
    "RiskCalculatorSpec",
    "RiskEvaluation",
    "RiskReason",
    "RiskStatus",
    "check_target_risk",
    "evaluate_skill_risk",
]

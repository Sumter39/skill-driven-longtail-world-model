"""Schema, finite-value, seam, and common kinematic checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.assembly import HISTORY_STEPS, FUTURE_STEPS, TOTAL_STEPS
from skilldrive.schemas import AgentTrack, Scenario


@dataclass(frozen=True)
class KinematicLimits:
    """Explicit hard limits supplied by the caller's frozen filter policy."""

    maximum_seam_speed_mps: float | None
    maximum_speed_mps: float | None
    maximum_acceleration_mps2: float | None
    maximum_jerk_mps3: float | None
    maximum_deceleration_mps2: float | None = None
    maximum_curvature_per_m: float | None = None
    maximum_heading_rate_rad_s: float | None = None
    minimum_heading_speed_mps: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "maximum_seam_speed_mps",
            "maximum_speed_mps",
            "maximum_acceleration_mps2",
            "maximum_jerk_mps3",
            "maximum_deceleration_mps2",
            "maximum_curvature_per_m",
            "maximum_heading_rate_rad_s",
            "minimum_heading_speed_mps",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be None or a positive finite number")
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be None or a positive finite number")
            object.__setattr__(self, name, number)


@dataclass(frozen=True)
class FutureKinematics:
    """Frame-aligned motion derived from frame 49 and a generated 60-step future."""

    velocity_xy: np.ndarray
    speed_mps: np.ndarray
    acceleration_xy: np.ndarray
    acceleration_mps2: np.ndarray
    tangential_acceleration_mps2: np.ndarray
    deceleration_mps2: np.ndarray
    jerk_xy: np.ndarray
    jerk_mps3: np.ndarray
    heading_rad: np.ndarray
    heading_rate_rad_s: np.ndarray
    curvature_per_m: np.ndarray
    low_speed_heading_suppressed: np.ndarray


def _target(scenario: Scenario, target_track_id: str) -> AgentTrack | None:
    return next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )


def _future_array(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array


def check_schema_and_finite(
    scenario: Scenario,
    target_track_id: str,
    future_xy_global: Any,
) -> FilterCheck:
    """Validate the fixed 50-history/60-future numeric contract without thresholds."""

    reasons: list[FilterRejection] = []
    try:
        raw_future = np.asarray(future_xy_global)
    except (TypeError, ValueError):
        raw_future = None
    future = _future_array(future_xy_global)
    if future is None or future.shape != (FUTURE_STEPS, 2):
        reasons.append(FilterRejection.INVALID_FUTURE_SHAPE)
    elif raw_future is None or raw_future.dtype != np.float32:
        reasons.append(FilterRejection.INVALID_FUTURE_DTYPE)

    sample_period_s: float | None = None
    if len(scenario.timestamps) != TOTAL_STEPS:
        reasons.append(FilterRejection.INVALID_SCENARIO_LENGTH)
    else:
        timestamp_deltas = np.diff(scenario.timestamps.astype(np.float64))
        if np.any(timestamp_deltas <= 0.0):
            reasons.append(FilterRejection.NON_MONOTONIC_TIMESTAMPS)
        else:
            sample_period_s = float(np.median(timestamp_deltas) / 1_000_000_000.0)
            if not np.allclose(
                timestamp_deltas / 1_000_000_000.0,
                0.1,
                rtol=0.0,
                atol=1e-9,
            ):
                reasons.append(FilterRejection.INVALID_SAMPLE_PERIOD)

    target = _target(scenario, target_track_id)
    if target is None:
        reasons.append(FilterRejection.TARGET_TRACK_MISSING)
    elif len(target.positions) != TOTAL_STEPS:
        reasons.append(FilterRejection.INVALID_TARGET_LENGTH)
    else:
        anchor_index = HISTORY_STEPS - 1
        if not bool(target.observed_mask[anchor_index]):
            reasons.append(FilterRejection.TARGET_ANCHOR_NOT_OBSERVED)
        if not (
            np.isfinite(target.positions[anchor_index]).all()
            and np.isfinite(target.velocities[anchor_index]).all()
        ):
            reasons.append(FilterRejection.NON_FINITE_TARGET_ANCHOR)

    if future is not None and future.shape == (FUTURE_STEPS, 2):
        if not np.isfinite(future).all():
            reasons.append(FilterRejection.NON_FINITE_GENERATED_POSITIONS)

    return FilterCheck(
        stage=FilterStage.SCHEMA_FINITE,
        rejection_reasons=tuple(reasons),
        metrics={
            "scenario_steps": len(scenario.timestamps),
            "future_shape": None if future is None else list(future.shape),
            "future_dtype": None if raw_future is None else str(raw_future.dtype),
            "sample_period_s": sample_period_s,
        },
    )


def _initial_heading(target: AgentTrack) -> float:
    heading = float(target.headings[HISTORY_STEPS - 1])
    if math.isfinite(heading):
        return heading
    velocity = target.velocities[HISTORY_STEPS - 1]
    if np.isfinite(velocity).all() and float(np.linalg.norm(velocity)) > 0.0:
        return float(np.arctan2(velocity[1], velocity[0]))
    return float("nan")


def derive_future_kinematics(
    scenario: Scenario,
    target_track_id: str,
    future_xy_global: Any,
    *,
    minimum_heading_speed_mps: float = 0.5,
) -> FutureKinematics:
    """Derive exact-delta velocity, acceleration and jerk including the 49→50 seam."""

    schema = check_schema_and_finite(scenario, target_track_id, future_xy_global)
    if not schema.passed:
        raise ValueError(
            "candidate does not satisfy schema/finite contract: "
            + ", ".join(schema.rejection_values)
        )
    if (
        isinstance(minimum_heading_speed_mps, bool)
        or not isinstance(minimum_heading_speed_mps, (int, float))
        or not math.isfinite(float(minimum_heading_speed_mps))
        or float(minimum_heading_speed_mps) <= 0.0
    ):
        raise ValueError("minimum_heading_speed_mps must be a positive finite number")
    heading_speed_threshold = float(minimum_heading_speed_mps)
    target = _target(scenario, target_track_id)
    assert target is not None
    future = np.asarray(future_xy_global, dtype=np.float64)

    anchor_index = HISTORY_STEPS - 1
    points = np.vstack((target.positions[anchor_index], future))
    timestamps_s = (
        scenario.timestamps[anchor_index:TOTAL_STEPS].astype(np.float64)
        - float(scenario.timestamps[anchor_index])
    ) / 1_000_000_000.0
    elapsed = np.diff(timestamps_s)
    velocity = np.diff(points, axis=0) / elapsed[:, None]
    previous_velocity = target.velocities[anchor_index].astype(np.float64)
    velocity_sequence = np.vstack((previous_velocity, velocity))
    acceleration = np.diff(velocity_sequence, axis=0) / elapsed[:, None]

    previous_acceleration = np.full(2, np.nan, dtype=np.float64)
    if anchor_index > 0:
        history_elapsed = (
            float(scenario.timestamps[anchor_index] - scenario.timestamps[anchor_index - 1])
            / 1_000_000_000.0
        )
        previous_history_velocity = target.velocities[anchor_index - 1]
        if (
            history_elapsed > 0.0
            and np.isfinite(previous_history_velocity).all()
            and np.isfinite(previous_velocity).all()
        ):
            previous_acceleration = (
                previous_velocity - previous_history_velocity
            ) / history_elapsed
    acceleration_sequence = np.vstack((previous_acceleration, acceleration))
    jerk = np.diff(acceleration_sequence, axis=0) / elapsed[:, None]

    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    previous_speed = float(np.linalg.norm(previous_velocity))
    speed_sequence = np.concatenate(([previous_speed], speed))
    tangential_acceleration = np.diff(speed_sequence) / elapsed
    deceleration = np.maximum(-tangential_acceleration, 0.0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    headings = np.empty(FUTURE_STEPS, dtype=np.float64)
    heading_rates = np.zeros(FUTURE_STEPS, dtype=np.float64)
    curvature = np.zeros(FUTURE_STEPS, dtype=np.float64)
    suppressed = np.zeros(FUTURE_STEPS, dtype=bool)
    previous_heading = _initial_heading(target)
    for index, item in enumerate(velocity):
        current_speed = float(speed[index])
        if current_speed >= heading_speed_threshold:
            current_heading = float(np.arctan2(item[1], item[0]))
            if math.isfinite(previous_heading) and previous_speed >= heading_speed_threshold:
                delta = float(
                    (current_heading - previous_heading + math.pi) % (2.0 * math.pi)
                    - math.pi
                )
                heading_rates[index] = abs(delta) / float(elapsed[index])
                mean_speed = (previous_speed + current_speed) / 2.0
                curvature[index] = heading_rates[index] / mean_speed
            previous_heading = current_heading
        else:
            suppressed[index] = True
        headings[index] = previous_heading
        previous_speed = current_speed

    return FutureKinematics(
        velocity_xy=velocity,
        speed_mps=speed,
        acceleration_xy=acceleration,
        acceleration_mps2=acceleration_magnitude,
        tangential_acceleration_mps2=tangential_acceleration,
        deceleration_mps2=deceleration,
        jerk_xy=jerk,
        jerk_mps3=jerk_magnitude,
        heading_rad=headings,
        heading_rate_rad_s=heading_rates,
        curvature_per_m=curvature,
        low_speed_heading_suppressed=suppressed,
    )


def _maximum(values: np.ndarray) -> tuple[float | None, int | None]:
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return None, None
    local = int(np.argmax(values[finite]))
    index = int(finite[local])
    return float(values[index]), index


def _exceeds(value: float | None, limit: float | None) -> bool:
    if value is None or limit is None:
        return False
    tolerance = max(1e-9, abs(limit) * 1e-9)
    return value > limit + tolerance


def check_kinematics(
    scenario: Scenario,
    target_track_id: str,
    future_xy_global: Any,
    limits: KinematicLimits,
) -> FilterCheck:
    """Apply only the explicit limits supplied by ``limits``."""

    values = derive_future_kinematics(
        scenario,
        target_track_id,
        future_xy_global,
        minimum_heading_speed_mps=limits.minimum_heading_speed_mps,
    )
    reasons: list[FilterRejection] = []
    arrays = (
        values.velocity_xy,
        values.speed_mps,
        values.acceleration_xy,
        values.acceleration_mps2,
        values.tangential_acceleration_mps2,
        values.deceleration_mps2,
        values.jerk_xy,
        values.jerk_mps3,
        values.heading_rad,
        values.heading_rate_rad_s,
        values.curvature_per_m,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        reasons.append(FilterRejection.NON_FINITE_DERIVED_KINEMATICS)

    maximum_speed, maximum_speed_index = _maximum(values.speed_mps)
    maximum_acceleration, maximum_acceleration_index = _maximum(
        np.maximum(values.tangential_acceleration_mps2, 0.0)
    )
    maximum_deceleration, maximum_deceleration_index = _maximum(
        values.deceleration_mps2
    )
    maximum_jerk, maximum_jerk_index = _maximum(values.jerk_mps3)
    maximum_curvature, maximum_curvature_index = _maximum(values.curvature_per_m)
    maximum_heading_rate, maximum_heading_rate_index = _maximum(
        values.heading_rate_rad_s
    )
    seam_speed = float(values.speed_mps[0]) if np.isfinite(values.speed_mps[0]) else None

    if _exceeds(seam_speed, limits.maximum_seam_speed_mps):
        reasons.append(FilterRejection.SEAM_SPEED_LIMIT_EXCEEDED)
    if _exceeds(maximum_speed, limits.maximum_speed_mps):
        reasons.append(FilterRejection.SPEED_LIMIT_EXCEEDED)
    if _exceeds(maximum_acceleration, limits.maximum_acceleration_mps2):
        reasons.append(FilterRejection.ACCELERATION_LIMIT_EXCEEDED)
    if _exceeds(maximum_jerk, limits.maximum_jerk_mps3):
        reasons.append(FilterRejection.JERK_LIMIT_EXCEEDED)
    if _exceeds(maximum_deceleration, limits.maximum_deceleration_mps2):
        reasons.append(FilterRejection.DECELERATION_LIMIT_EXCEEDED)
    if _exceeds(maximum_curvature, limits.maximum_curvature_per_m):
        reasons.append(FilterRejection.CURVATURE_LIMIT_EXCEEDED)
    if _exceeds(maximum_heading_rate, limits.maximum_heading_rate_rad_s):
        reasons.append(FilterRejection.HEADING_RATE_LIMIT_EXCEEDED)

    return FilterCheck(
        stage=FilterStage.KINEMATICS,
        rejection_reasons=tuple(reasons),
        metrics={
            "seam_speed_mps": seam_speed,
            "maximum_speed_mps": maximum_speed,
            "maximum_speed_future_index": maximum_speed_index,
            "maximum_acceleration_mps2": maximum_acceleration,
            "maximum_acceleration_future_index": maximum_acceleration_index,
            "maximum_deceleration_mps2": maximum_deceleration,
            "maximum_deceleration_future_index": maximum_deceleration_index,
            "maximum_jerk_mps3": maximum_jerk,
            "maximum_jerk_future_index": maximum_jerk_index,
            "maximum_curvature_per_m": maximum_curvature,
            "maximum_curvature_future_index": maximum_curvature_index,
            "maximum_heading_rate_rad_s": maximum_heading_rate,
            "maximum_heading_rate_future_index": maximum_heading_rate_index,
            "minimum_heading_speed_mps": limits.minimum_heading_speed_mps,
            "low_speed_heading_suppressed_steps": int(
                values.low_speed_heading_suppressed.sum()
            ),
        },
    )


__all__ = [
    "FutureKinematics",
    "KinematicLimits",
    "check_kinematics",
    "check_schema_and_finite",
    "derive_future_kinematics",
]

"""Vectorized kinematic scoring for decoded CVAE latent candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from skilldrive.filtering.common import KinematicLimits
from skilldrive.filtering.contracts import FilterRejection
from skilldrive.generation.assembly import FUTURE_STEPS, HISTORY_STEPS, TOTAL_STEPS
from skilldrive.schemas import AgentTrack, Scenario


@dataclass(frozen=True)
class KinematicCandidateScores:
    """Per-candidate hard-gate evidence and one deterministic Top-K selection.

    Passing candidates rank before rejected candidates. Remaining ties are
    ordered by normalized violation score, latent seed, then candidate index.
    """

    candidate_indices: np.ndarray
    latent_seeds: np.ndarray
    passed: np.ndarray
    finite_kinematics: np.ndarray
    seam_speed_mps: np.ndarray
    maximum_speed_mps: np.ndarray
    maximum_speed_future_index: np.ndarray
    maximum_acceleration_mps2: np.ndarray
    maximum_acceleration_future_index: np.ndarray
    maximum_deceleration_mps2: np.ndarray
    maximum_deceleration_future_index: np.ndarray
    maximum_jerk_mps3: np.ndarray
    maximum_jerk_future_index: np.ndarray
    maximum_curvature_per_m: np.ndarray
    maximum_curvature_future_index: np.ndarray
    maximum_heading_rate_rad_s: np.ndarray
    maximum_heading_rate_future_index: np.ndarray
    low_speed_heading_suppressed_steps: np.ndarray
    normalized_violation_score: np.ndarray
    rejection_reasons: tuple[tuple[FilterRejection, ...], ...]
    top_k_indices: np.ndarray


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _integer_vector(value: Any, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array with shape ({length},)")
    result = raw.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return result


def _target(scenario: Scenario, target_track_id: str) -> AgentTrack:
    target = next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )
    if target is None:
        raise ValueError(f"target track is not present in scenario: {target_track_id}")
    return target


def _validate_source_context(scenario: Scenario, target: AgentTrack) -> np.ndarray:
    timestamp_count = len(scenario.timestamps)
    if timestamp_count not in (HISTORY_STEPS, TOTAL_STEPS):
        raise ValueError(
            f"scenario must contain {HISTORY_STEPS} history timestamps or "
            f"the complete {TOTAL_STEPS}-step time axis"
        )
    timestamp_deltas = np.diff(scenario.timestamps.astype(np.float64))
    if np.any(timestamp_deltas <= 0.0):
        raise ValueError("scenario timestamps must be strictly increasing")
    if not np.allclose(
        timestamp_deltas / 1_000_000_000.0,
        0.1,
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("scenario sample period must be 0.1 seconds")
    if len(target.positions) != timestamp_count:
        raise ValueError("target state count must match the scenario time axis")

    anchor_index = HISTORY_STEPS - 1
    if not bool(target.observed_mask[anchor_index]):
        raise ValueError("target anchor at frame 49 must be observed")
    if not (
        np.isfinite(target.positions[anchor_index]).all()
        and np.isfinite(target.velocities[anchor_index]).all()
    ):
        raise ValueError("target anchor position and velocity must be finite")
    if timestamp_count == TOTAL_STEPS:
        timestamps_s = (
            scenario.timestamps[anchor_index:TOTAL_STEPS].astype(np.float64)
            - float(scenario.timestamps[anchor_index])
        ) / 1_000_000_000.0
        return np.diff(timestamps_s)
    return np.full(FUTURE_STEPS, 0.1, dtype=np.float64)


def _initial_heading(target: AgentTrack) -> float:
    anchor_index = HISTORY_STEPS - 1
    heading = float(target.headings[anchor_index])
    if math.isfinite(heading):
        return heading
    velocity = target.velocities[anchor_index]
    if np.isfinite(velocity).all() and float(np.linalg.norm(velocity)) > 0.0:
        return float(np.arctan2(velocity[1], velocity[0]))
    return float("nan")


def _finite_maximum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    indices = np.argmax(safe, axis=1).astype(np.int64)
    maxima = safe[np.arange(len(safe)), indices]
    missing = ~finite.any(axis=1)
    maxima = np.where(missing, np.nan, maxima)
    indices = np.where(missing, -1, indices)
    return maxima, indices


def _limit_exceeded(values: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None:
        return np.zeros(len(values), dtype=bool)
    tolerance = max(1e-9, abs(limit) * 1e-9)
    return values > limit + tolerance


def _normalized_excess(values: np.ndarray, limit: float | None) -> np.ndarray:
    if limit is None:
        return np.zeros(len(values), dtype=np.float64)
    tolerance = max(1e-9, abs(limit) * 1e-9)
    return np.maximum(values - limit - tolerance, 0.0) / limit


def _stable_order(
    passed: np.ndarray,
    normalized_violation_score: np.ndarray,
    latent_seeds: np.ndarray,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    return np.lexsort(
        (
            candidate_indices,
            latent_seeds,
            normalized_violation_score,
            ~passed,
        )
    )


def score_kinematic_candidates(
    source_scenario: Scenario,
    target_track_id: str,
    future_xy_global: np.ndarray,
    limits: KinematicLimits,
    latent_seeds: np.ndarray,
    *,
    top_k: int,
    candidate_indices: np.ndarray | None = None,
) -> KinematicCandidateScores:
    """Score decoded futures without reading or modifying the source future.

    The calculations intentionally match ``filtering.common.check_kinematics``.
    Only frame 49, the frame 48/49 stored velocities, frame-49 heading, the
    observed anchor flag, and the scenario time axis are read from the source.
    Each enabled hard limit contributes ``max(value-limit-tolerance, 0)/limit``
    to the normalized violation score; non-finite derived motion scores infinity.
    """

    selected_count = _positive_integer(top_k, "top_k")
    raw_futures = np.asarray(future_xy_global)
    if raw_futures.ndim != 3 or raw_futures.shape[1:] != (FUTURE_STEPS, 2):
        raise ValueError("future_xy_global must have shape [N, 60, 2]")
    if raw_futures.dtype != np.float32:
        raise ValueError("future_xy_global must have float32 dtype")
    if not np.isfinite(raw_futures).all():
        raise ValueError("future_xy_global must contain only finite values")
    candidate_count = len(raw_futures)
    if candidate_count == 0:
        raise ValueError("future_xy_global must contain at least one candidate")

    seeds = _integer_vector(latent_seeds, "latent_seeds", candidate_count)
    indices = (
        np.arange(candidate_count, dtype=np.int64)
        if candidate_indices is None
        else _integer_vector(candidate_indices, "candidate_indices", candidate_count)
    )
    if len(np.unique(indices)) != candidate_count:
        raise ValueError("candidate_indices must be unique")

    target = _target(source_scenario, target_track_id)
    elapsed = _validate_source_context(source_scenario, target)
    anchor_index = HISTORY_STEPS - 1
    future = raw_futures.astype(np.float64)
    points = np.concatenate(
        (
            np.broadcast_to(target.positions[anchor_index], (candidate_count, 1, 2)),
            future,
        ),
        axis=1,
    )
    velocity = np.diff(points, axis=1) / elapsed[None, :, None]
    previous_velocity = target.velocities[anchor_index].astype(np.float64)
    velocity_sequence = np.concatenate(
        (
            np.broadcast_to(previous_velocity, (candidate_count, 1, 2)),
            velocity,
        ),
        axis=1,
    )
    acceleration = np.diff(velocity_sequence, axis=1) / elapsed[None, :, None]

    previous_acceleration = np.full(2, np.nan, dtype=np.float64)
    history_elapsed = (
        float(
            source_scenario.timestamps[anchor_index]
            - source_scenario.timestamps[anchor_index - 1]
        )
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
    acceleration_sequence = np.concatenate(
        (
            np.broadcast_to(previous_acceleration, (candidate_count, 1, 2)),
            acceleration,
        ),
        axis=1,
    )
    jerk = np.diff(acceleration_sequence, axis=1) / elapsed[None, :, None]

    speed = np.linalg.norm(velocity, axis=2)
    previous_speed_scalar = float(np.linalg.norm(previous_velocity))
    speed_sequence = np.concatenate(
        (np.full((candidate_count, 1), previous_speed_scalar), speed),
        axis=1,
    )
    tangential_acceleration = np.diff(speed_sequence, axis=1) / elapsed[None, :]
    deceleration = np.maximum(-tangential_acceleration, 0.0)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=2)
    jerk_magnitude = np.linalg.norm(jerk, axis=2)

    headings = np.empty((candidate_count, FUTURE_STEPS), dtype=np.float64)
    heading_rates = np.zeros_like(headings)
    curvature = np.zeros_like(headings)
    suppressed = np.zeros((candidate_count, FUTURE_STEPS), dtype=bool)
    previous_heading = np.full(candidate_count, _initial_heading(target))
    previous_speed = np.full(candidate_count, previous_speed_scalar)
    heading_threshold = limits.minimum_heading_speed_mps
    for future_index in range(FUTURE_STEPS):
        current_speed = speed[:, future_index]
        moving = current_speed >= heading_threshold
        current_heading = np.arctan2(
            velocity[:, future_index, 1],
            velocity[:, future_index, 0],
        )
        comparable = moving & np.isfinite(previous_heading) & (
            previous_speed >= heading_threshold
        )
        delta = (current_heading - previous_heading + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        heading_rates[comparable, future_index] = (
            np.abs(delta[comparable]) / elapsed[future_index]
        )
        mean_speed = (previous_speed + current_speed) / 2.0
        curvature[comparable, future_index] = (
            heading_rates[comparable, future_index] / mean_speed[comparable]
        )
        previous_heading = np.where(moving, current_heading, previous_heading)
        headings[:, future_index] = previous_heading
        suppressed[:, future_index] = ~moving
        previous_speed = current_speed

    finite_kinematics = np.ones(candidate_count, dtype=bool)
    for values in (
        velocity,
        speed,
        acceleration,
        acceleration_magnitude,
        tangential_acceleration,
        deceleration,
        jerk,
        jerk_magnitude,
        headings,
        heading_rates,
        curvature,
    ):
        finite_kinematics &= np.isfinite(values).reshape(candidate_count, -1).all(axis=1)

    maximum_speed, maximum_speed_index = _finite_maximum(speed)
    maximum_acceleration, maximum_acceleration_index = _finite_maximum(
        np.maximum(tangential_acceleration, 0.0)
    )
    maximum_deceleration, maximum_deceleration_index = _finite_maximum(deceleration)
    maximum_jerk, maximum_jerk_index = _finite_maximum(jerk_magnitude)
    maximum_curvature, maximum_curvature_index = _finite_maximum(curvature)
    maximum_heading_rate, maximum_heading_rate_index = _finite_maximum(heading_rates)
    seam_speed = speed[:, 0]

    exceeded = {
        FilterRejection.SEAM_SPEED_LIMIT_EXCEEDED: _limit_exceeded(
            seam_speed, limits.maximum_seam_speed_mps
        ),
        FilterRejection.SPEED_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_speed, limits.maximum_speed_mps
        ),
        FilterRejection.ACCELERATION_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_acceleration, limits.maximum_acceleration_mps2
        ),
        FilterRejection.JERK_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_jerk, limits.maximum_jerk_mps3
        ),
        FilterRejection.DECELERATION_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_deceleration, limits.maximum_deceleration_mps2
        ),
        FilterRejection.CURVATURE_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_curvature, limits.maximum_curvature_per_m
        ),
        FilterRejection.HEADING_RATE_LIMIT_EXCEEDED: _limit_exceeded(
            maximum_heading_rate, limits.maximum_heading_rate_rad_s
        ),
    }
    rejection_reasons: list[tuple[FilterRejection, ...]] = []
    for candidate_index in range(candidate_count):
        reasons: list[FilterRejection] = []
        if not finite_kinematics[candidate_index]:
            reasons.append(FilterRejection.NON_FINITE_DERIVED_KINEMATICS)
        reasons.extend(
            reason
            for reason, mask in exceeded.items()
            if bool(mask[candidate_index])
        )
        rejection_reasons.append(tuple(reasons))
    passed = np.asarray([not reasons for reasons in rejection_reasons], dtype=bool)

    normalized_violation_score = np.zeros(candidate_count, dtype=np.float64)
    for values, limit in (
        (seam_speed, limits.maximum_seam_speed_mps),
        (maximum_speed, limits.maximum_speed_mps),
        (maximum_acceleration, limits.maximum_acceleration_mps2),
        (maximum_jerk, limits.maximum_jerk_mps3),
        (maximum_deceleration, limits.maximum_deceleration_mps2),
        (maximum_curvature, limits.maximum_curvature_per_m),
        (maximum_heading_rate, limits.maximum_heading_rate_rad_s),
    ):
        normalized_violation_score += _normalized_excess(values, limit)
    normalized_violation_score = np.where(
        finite_kinematics,
        normalized_violation_score,
        np.inf,
    )
    order = _stable_order(passed, normalized_violation_score, seeds, indices)
    top_k_indices = indices[order[: min(selected_count, candidate_count)]]

    return KinematicCandidateScores(
        candidate_indices=indices,
        latent_seeds=seeds,
        passed=passed,
        finite_kinematics=finite_kinematics,
        seam_speed_mps=seam_speed,
        maximum_speed_mps=maximum_speed,
        maximum_speed_future_index=maximum_speed_index,
        maximum_acceleration_mps2=maximum_acceleration,
        maximum_acceleration_future_index=maximum_acceleration_index,
        maximum_deceleration_mps2=maximum_deceleration,
        maximum_deceleration_future_index=maximum_deceleration_index,
        maximum_jerk_mps3=maximum_jerk,
        maximum_jerk_future_index=maximum_jerk_index,
        maximum_curvature_per_m=maximum_curvature,
        maximum_curvature_future_index=maximum_curvature_index,
        maximum_heading_rate_rad_s=maximum_heading_rate,
        maximum_heading_rate_future_index=maximum_heading_rate_index,
        low_speed_heading_suppressed_steps=suppressed.sum(axis=1),
        normalized_violation_score=normalized_violation_score,
        rejection_reasons=tuple(rejection_reasons),
        top_k_indices=top_k_indices,
    )


@dataclass
class KinematicTopKAccumulator:
    """Keep the same deterministic Top-K while candidate chunks are scored."""

    top_k: int
    _candidate_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), init=False
    )
    _latent_seeds: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), init=False
    )
    _passed: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=bool), init=False
    )
    _scores: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), init=False
    )
    _seen_indices: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.top_k = _positive_integer(self.top_k, "top_k")

    def update(self, scores: KinematicCandidateScores) -> None:
        incoming = {int(value) for value in scores.candidate_indices}
        overlap = incoming & self._seen_indices
        if overlap:
            raise ValueError(
                "candidate chunks contain duplicate indices: "
                f"{sorted(overlap)[:5]}"
            )
        self._seen_indices.update(incoming)
        candidate_indices = np.concatenate(
            (self._candidate_indices, scores.candidate_indices)
        )
        latent_seeds = np.concatenate((self._latent_seeds, scores.latent_seeds))
        passed = np.concatenate((self._passed, scores.passed))
        violation_scores = np.concatenate(
            (self._scores, scores.normalized_violation_score)
        )
        order = _stable_order(
            passed,
            violation_scores,
            latent_seeds,
            candidate_indices,
        )[: self.top_k]
        self._candidate_indices = candidate_indices[order]
        self._latent_seeds = latent_seeds[order]
        self._passed = passed[order]
        self._scores = violation_scores[order]

    @property
    def top_k_indices(self) -> np.ndarray:
        return self._candidate_indices.copy()


__all__ = [
    "KinematicCandidateScores",
    "KinematicTopKAccumulator",
    "score_kinematic_candidates",
]

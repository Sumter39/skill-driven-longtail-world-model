"""History/source invariants and coordinate round-trip checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from skilldrive.data.coordinates import global_to_local, local_to_global
from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.assembly import HISTORY_STEPS
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


ROUND_TRIP_TOLERANCE_M = 1e-4


def _array_equal(first: np.ndarray, second: np.ndarray) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    return left.shape == right.shape and bool(
        np.array_equal(left, right, equal_nan=True)
    )


def _value_equal(first: Any, second: Any) -> bool:
    """Compare JSON-like metadata while treating colocated NaNs as unchanged."""

    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return _array_equal(np.asarray(first), np.asarray(second))
        except (TypeError, ValueError):
            return False
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            return False
        if set(first) != set(second):
            return False
        return all(_value_equal(first[key], second[key]) for key in first)
    if (
        isinstance(first, Sequence)
        and not isinstance(first, (str, bytes, bytearray))
    ) or (
        isinstance(second, Sequence)
        and not isinstance(second, (str, bytes, bytearray))
    ):
        if not (
            isinstance(first, Sequence)
            and not isinstance(first, (str, bytes, bytearray))
            and isinstance(second, Sequence)
            and not isinstance(second, (str, bytes, bytearray))
        ):
            return False
        return len(first) == len(second) and all(
            _value_equal(left, right) for left, right in zip(first, second)
        )
    if isinstance(first, (float, np.floating)) and isinstance(
        second, (float, np.floating)
    ):
        if math.isnan(float(first)) and math.isnan(float(second)):
            return True
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _map_polyline_equal(first: MapPolyline, second: MapPolyline) -> bool:
    return bool(
        first.polyline_id == second.polyline_id
        and first.polyline_type == second.polyline_type
        and _array_equal(first.points, second.points)
        and first.direction == second.direction
        and first.is_intersection == second.is_intersection
        and first.lane_id == second.lane_id
        and first.mark_type == second.mark_type
        and first.left_mark_type == second.left_mark_type
        and first.right_mark_type == second.right_mark_type
        and first.predecessor_ids == second.predecessor_ids
        and first.successor_ids == second.successor_ids
        and first.left_neighbor_id == second.left_neighbor_id
        and first.right_neighbor_id == second.right_neighbor_id
    )


def _target_history_equal(first: AgentTrack, second: AgentTrack) -> bool:
    if min(len(first.positions), len(second.positions)) < HISTORY_STEPS:
        return False
    return bool(
        first.track_id == second.track_id
        and first.object_type == second.object_type
        and first.is_focal == second.is_focal
        and _array_equal(
            first.positions[:HISTORY_STEPS], second.positions[:HISTORY_STEPS]
        )
        and _array_equal(
            first.velocities[:HISTORY_STEPS], second.velocities[:HISTORY_STEPS]
        )
        and _array_equal(
            first.headings[:HISTORY_STEPS], second.headings[:HISTORY_STEPS]
        )
        and _array_equal(first.observed_mask, second.observed_mask)
    )


def _background_track_equal(first: AgentTrack, second: AgentTrack) -> bool:
    return bool(
        first.track_id == second.track_id
        and first.object_type == second.object_type
        and first.is_focal == second.is_focal
        and _array_equal(first.positions, second.positions)
        and _array_equal(first.velocities, second.velocities)
        and _array_equal(first.headings, second.headings)
        and _array_equal(first.observed_mask, second.observed_mask)
    )


def check_history_invariants(
    source: Scenario,
    materialized: Scenario,
    target_track_id: str,
) -> FilterCheck:
    """Verify that a single-target overlay did not rewrite its source context."""

    source_agents = {agent.track_id: agent for agent in source.agents}
    materialized_agents = {agent.track_id: agent for agent in materialized.agents}
    source_target = source_agents.get(target_track_id)
    materialized_target = materialized_agents.get(target_track_id)

    source_identity_unchanged = bool(
        source.scenario_id == materialized.scenario_id
        and source.city_name == materialized.city_name
        and source.focal_track_id == materialized.focal_track_id
    )
    timestamps_unchanged = _array_equal(source.timestamps, materialized.timestamps)
    agent_ids_unchanged = set(source_agents) == set(materialized_agents)
    target_history_unchanged = bool(
        source_target is not None
        and materialized_target is not None
        and _target_history_equal(source_target, materialized_target)
    )

    source_background_ids = set(source_agents) - {target_track_id}
    materialized_background_ids = set(materialized_agents) - {target_track_id}
    changed_background_track_ids = sorted(
        track_id
        for track_id in source_background_ids & materialized_background_ids
        if not _background_track_equal(
            source_agents[track_id], materialized_agents[track_id]
        )
    )
    missing_background_track_ids = sorted(
        source_background_ids - materialized_background_ids
    )
    unexpected_background_track_ids = sorted(
        materialized_background_ids - source_background_ids
    )
    background_tracks_unchanged = not (
        changed_background_track_ids
        or missing_background_track_ids
        or unexpected_background_track_ids
    )

    map_unchanged = bool(
        len(source.map_polylines) == len(materialized.map_polylines)
        and all(
            _map_polyline_equal(first, second)
            for first, second in zip(
                source.map_polylines, materialized.map_polylines
            )
        )
    )
    metadata_unchanged = _value_equal(source.metadata, materialized.metadata)

    reasons: list[FilterRejection] = []
    if not timestamps_unchanged:
        reasons.append(FilterRejection.HISTORY_TIMESTAMPS_CHANGED)
    if not (source_identity_unchanged and agent_ids_unchanged and target_history_unchanged):
        reasons.append(FilterRejection.HISTORY_TARGET_CHANGED)
    if not background_tracks_unchanged:
        reasons.append(FilterRejection.BACKGROUND_TRACK_CHANGED)
    if not map_unchanged:
        reasons.append(FilterRejection.MAP_CHANGED)
    if not metadata_unchanged:
        reasons.append(FilterRejection.METADATA_CHANGED)

    return FilterCheck(
        stage=FilterStage.HISTORY_INVARIANTS,
        rejection_reasons=tuple(reasons),
        metrics={
            "source_identity_unchanged": source_identity_unchanged,
            "timestamps_unchanged": timestamps_unchanged,
            "agent_ids_unchanged": agent_ids_unchanged,
            "target_history_unchanged": target_history_unchanged,
            "target_observed_mask_unchanged": bool(
                source_target is not None
                and materialized_target is not None
                and _array_equal(
                    source_target.observed_mask,
                    materialized_target.observed_mask,
                )
            ),
            "background_tracks_unchanged": background_tracks_unchanged,
            "changed_background_track_ids": changed_background_track_ids,
            "missing_background_track_ids": missing_background_track_ids,
            "unexpected_background_track_ids": unexpected_background_track_ids,
            "map_unchanged": map_unchanged,
            "metadata_unchanged": metadata_unchanged,
        },
    )


def check_coordinate_round_trip(
    future_xy_local: Any,
    future_xy_global: Any,
    origin_global_xy: Any,
    heading_global_rad: float,
    *,
    tolerance_m: float = ROUND_TRIP_TOLERANCE_M,
) -> FilterCheck:
    """Check both local→global and global→local representations of one future."""

    local = np.asarray(future_xy_local, dtype=np.float64)
    global_positions = np.asarray(future_xy_global, dtype=np.float64)
    origin = np.asarray(origin_global_xy, dtype=np.float64)
    if local.ndim != 2 or local.shape[1:] != (2,):
        raise ValueError("future_xy_local must have shape (N, 2)")
    if global_positions.shape != local.shape:
        raise ValueError("future_xy_global must have the same shape as future_xy_local")
    if not len(local):
        raise ValueError("future coordinate arrays must not be empty")
    if not np.isfinite(local).all() or not np.isfinite(global_positions).all():
        raise ValueError("future coordinate arrays must contain only finite values")
    if origin.shape != (2,) or not np.isfinite(origin).all():
        raise ValueError("origin_global_xy must be a finite (2,) array")
    if not math.isfinite(float(heading_global_rad)):
        raise ValueError("heading_global_rad must be finite")
    if not math.isfinite(float(tolerance_m)) or tolerance_m <= 0.0:
        raise ValueError("tolerance_m must be finite and positive")

    reconstructed_global = local_to_global(local, origin, float(heading_global_rad))
    reconstructed_local = global_to_local(
        global_positions, origin, float(heading_global_rad)
    )
    local_to_global_errors = np.linalg.norm(
        reconstructed_global - global_positions, axis=1
    )
    global_to_local_errors = np.linalg.norm(reconstructed_local - local, axis=1)
    local_to_global_index = int(np.argmax(local_to_global_errors))
    global_to_local_index = int(np.argmax(global_to_local_errors))
    maximum_local_to_global_error = float(
        local_to_global_errors[local_to_global_index]
    )
    maximum_global_to_local_error = float(
        global_to_local_errors[global_to_local_index]
    )
    maximum_error = max(
        maximum_local_to_global_error, maximum_global_to_local_error
    )
    reasons = (
        ()
        if maximum_error < tolerance_m
        else (FilterRejection.COORDINATE_ROUND_TRIP_EXCEEDED,)
    )
    return FilterCheck(
        stage=FilterStage.HISTORY_INVARIANTS,
        rejection_reasons=reasons,
        metrics={
            "coordinate_round_trip_tolerance_m": float(tolerance_m),
            "maximum_coordinate_round_trip_error_m": maximum_error,
            "maximum_local_to_global_error_m": maximum_local_to_global_error,
            "maximum_local_to_global_error_future_index": local_to_global_index,
            "maximum_global_to_local_error_m": maximum_global_to_local_error,
            "maximum_global_to_local_error_future_index": global_to_local_index,
        },
    )


def check_history_and_coordinates(
    source: Scenario,
    materialized: Scenario,
    target_track_id: str,
    future_xy_local: Any,
    future_xy_global: Any,
    origin_global_xy: Any,
    heading_global_rad: float,
    *,
    tolerance_m: float = ROUND_TRIP_TOLERANCE_M,
) -> FilterCheck:
    """Return one pipeline-ready history stage check with all required evidence."""

    invariants = check_history_invariants(source, materialized, target_track_id)
    coordinates = check_coordinate_round_trip(
        future_xy_local,
        future_xy_global,
        origin_global_xy,
        heading_global_rad,
        tolerance_m=tolerance_m,
    )
    return FilterCheck(
        stage=FilterStage.HISTORY_INVARIANTS,
        rejection_reasons=tuple(
            dict.fromkeys(
                (*invariants.rejection_reasons, *coordinates.rejection_reasons)
            )
        ),
        metrics={
            **dict(invariants.metrics),
            **dict(coordinates.metrics),
        },
    )


__all__ = [
    "ROUND_TRIP_TOLERANCE_M",
    "check_coordinate_round_trip",
    "check_history_and_coordinates",
    "check_history_invariants",
]

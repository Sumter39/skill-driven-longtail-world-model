"""Shared trajectory geometry, kinematics, and point-mass risk metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class PolylineProjection:
    """Nearest point and local coordinates on a polyline."""

    point: np.ndarray
    distance_m: float
    signed_lateral_distance_m: float
    arc_length_m: float
    segment_index: int
    segment_fraction: float
    heading_rad: float


@dataclass(frozen=True)
class TrajectoryConflict:
    """One geometric intersection between two piecewise-linear trajectories."""

    point: np.ndarray
    first_segment_index: int
    second_segment_index: int
    first_segment_fraction: float
    second_segment_fraction: float
    first_time_s: float
    second_time_s: float
    time_gap_s: float


@dataclass(frozen=True)
class TrajectoryMinimumDistance:
    """Closest synchronized approach of two aligned trajectories."""

    distance_m: float
    first_point: np.ndarray
    second_point: np.ndarray
    frame_index: float
    time_s: float


def _points(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    return array


def _point(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    return array


def _valid_mask(value: Any | None, length: int, name: str = "valid_mask") -> np.ndarray:
    if value is None:
        return np.ones(length, dtype=bool)
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 1 or len(mask) != length:
        raise ValueError(f"{name} must have shape ({length},), got {mask.shape}")
    return mask


def _timestamps(
    timestamps_s: Any | None,
    length: int,
    sample_period_s: float,
) -> np.ndarray:
    if timestamps_s is None:
        if not np.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError("sample_period_s must be finite and positive")
        return np.arange(length, dtype=np.float64) * sample_period_s

    values = np.asarray(timestamps_s, dtype=np.float64)
    if values.ndim != 1 or len(values) != length:
        raise ValueError(f"timestamps_s must have shape ({length},), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("timestamps_s must contain only finite values")
    if length > 1 and np.any(np.diff(values) <= 0):
        raise ValueError("timestamps_s must be strictly increasing")
    return values


def extract_valid_trajectory(
    positions: Any,
    valid_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite points and their original frame indices.

    The optional mask can select an observed or future portion before finite-value
    filtering. Internal gaps are retained in the returned indices so callers can
    avoid treating disconnected samples as adjacent frames.
    """

    points = _points(positions, "positions")
    mask = _valid_mask(valid_mask, len(points))
    valid = mask & np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(valid)
    return points[indices].copy(), indices


def trajectory_speed(
    positions: Any,
    timestamps_s: Any | None = None,
    *,
    sample_period_s: float = 0.1,
    valid_mask: Any | None = None,
) -> np.ndarray:
    """Compute frame-aligned scalar speed without interpolating across gaps.

    Speed at frame ``i`` describes the interval ``i - 1 -> i``. The first frame,
    invalid frames, and the first frame after a gap are returned as ``NaN``.
    """

    points = _points(positions, "positions")
    mask = _valid_mask(valid_mask, len(points)) & np.isfinite(points).all(axis=1)
    times = _timestamps(timestamps_s, len(points), sample_period_s)
    result = np.full(len(points), np.nan, dtype=np.float64)
    if len(points) < 2:
        return result

    adjacent = mask[:-1] & mask[1:]
    indices = np.flatnonzero(adjacent) + 1
    if len(indices):
        displacement = points[indices] - points[indices - 1]
        elapsed = times[indices] - times[indices - 1]
        result[indices] = np.linalg.norm(displacement, axis=1) / elapsed
    return result


def trajectory_acceleration(
    positions: Any,
    timestamps_s: Any | None = None,
    *,
    sample_period_s: float = 0.1,
    valid_mask: Any | None = None,
) -> np.ndarray:
    """Compute frame-aligned tangential acceleration from scalar speed."""

    points = _points(positions, "positions")
    times = _timestamps(timestamps_s, len(points), sample_period_s)
    speeds = trajectory_speed(
        points,
        times,
        sample_period_s=sample_period_s,
        valid_mask=valid_mask,
    )
    result = np.full(len(points), np.nan, dtype=np.float64)
    if len(points) < 2:
        return result

    adjacent = np.isfinite(speeds[:-1]) & np.isfinite(speeds[1:])
    indices = np.flatnonzero(adjacent) + 1
    if len(indices):
        result[indices] = (speeds[indices] - speeds[indices - 1]) / (
            times[indices] - times[indices - 1]
        )
    return result


def point_to_polyline_projection(point: Any, polyline: Any) -> PolylineProjection:
    """Project a finite 2-D point onto the nearest valid polyline segment.

    Positive signed lateral distance is to the left of the selected segment.
    Invalid polyline vertices break the line instead of being bridged.
    """

    query = _point(point, "point")
    if not np.isfinite(query).all():
        raise ValueError("point must contain only finite values")
    line = _points(polyline, "polyline")
    if len(line) == 0 or not np.isfinite(line).all(axis=1).any():
        raise ValueError("polyline must contain at least one finite point")

    finite_vertices = np.isfinite(line).all(axis=1)
    valid_segments = finite_vertices[:-1] & finite_vertices[1:]
    segment_indices = np.flatnonzero(valid_segments)
    if len(segment_indices) == 0:
        finite_indices = np.flatnonzero(np.isfinite(line).all(axis=1))
        distances = np.linalg.norm(line[finite_indices] - query, axis=1)
        nearest = int(finite_indices[np.argmin(distances)])
        projected = line[nearest].copy()
        return PolylineProjection(
            point=projected,
            distance_m=float(np.linalg.norm(query - projected)),
            signed_lateral_distance_m=float("nan"),
            arc_length_m=0.0,
            segment_index=-1,
            segment_fraction=float("nan"),
            heading_rad=float("nan"),
        )

    starts = line[segment_indices]
    deltas = line[segment_indices + 1] - starts
    lengths = np.linalg.norm(deltas, axis=1)
    nondegenerate = lengths > _EPS
    fractions = np.zeros(len(segment_indices), dtype=np.float64)
    numerators = np.sum((query - starts) * deltas, axis=1)
    fractions[nondegenerate] = np.clip(
        numerators[nondegenerate] / (lengths[nondegenerate] ** 2),
        0.0,
        1.0,
    )
    projected_points = starts + fractions[:, None] * deltas
    distances = np.linalg.norm(query - projected_points, axis=1)
    headings = np.full(len(segment_indices), np.nan, dtype=np.float64)
    headings[nondegenerate] = np.arctan2(
        deltas[nondegenerate, 1],
        deltas[nondegenerate, 0],
    )
    arc_starts = np.zeros(len(segment_indices), dtype=np.float64)
    arc_starts[1:] = np.cumsum(lengths[:-1])
    along_distances = arc_starts + fractions * lengths

    if np.isnan(distances[0]):
        selected = 0
    else:
        comparable = np.flatnonzero(~np.isnan(distances))
        order = np.lexsort(
            (
                fractions[comparable],
                segment_indices[comparable],
                distances[comparable],
            )
        )
        selected = int(comparable[order[0]])

    distance = float(distances[selected])
    index = int(segment_indices[selected])
    fraction = float(fractions[selected])
    projected = projected_points[selected]
    along = float(along_distances[selected])
    heading = float(headings[selected])
    if np.isfinite(heading):
        tangent = np.array([np.cos(heading), np.sin(heading)])
        offset = query - projected
        signed_lateral = float(tangent[0] * offset[1] - tangent[1] * offset[0])
    else:
        signed_lateral = float("nan")
    return PolylineProjection(
        point=projected.copy(),
        distance_m=distance,
        signed_lateral_distance_m=signed_lateral,
        arc_length_m=along,
        segment_index=index,
        segment_fraction=fraction,
        heading_rad=heading,
    )


def heading_difference(first_heading_rad: Any, second_heading_rad: Any) -> float | np.ndarray:
    """Return the smallest absolute angular difference in ``[0, pi]``."""

    first, second = np.broadcast_arrays(
        np.asarray(first_heading_rad, dtype=np.float64),
        np.asarray(second_heading_rad, dtype=np.float64),
    )
    difference = np.abs((first - second + np.pi) % (2 * np.pi) - np.pi)
    if difference.ndim == 0:
        return float(difference)
    return difference


def _cross(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _point_on_segment_fraction(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float | None:
    delta = end - start
    length_squared = float(np.dot(delta, delta))
    if length_squared <= _EPS:
        return 0.0 if np.linalg.norm(point - start) <= _EPS else None
    fraction = float(np.dot(point - start, delta) / length_squared)
    projected = start + np.clip(fraction, 0.0, 1.0) * delta
    if -_EPS <= fraction <= 1.0 + _EPS and np.linalg.norm(point - projected) <= _EPS:
        return float(np.clip(fraction, 0.0, 1.0))
    return None


def _segment_intersection(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> tuple[np.ndarray, float, float] | None:
    first_delta = first_end - first_start
    second_delta = second_end - second_start
    first_length_squared = float(np.dot(first_delta, first_delta))
    second_length_squared = float(np.dot(second_delta, second_delta))

    if first_length_squared <= _EPS and second_length_squared <= _EPS:
        if np.linalg.norm(first_start - second_start) <= _EPS:
            return first_start.copy(), 0.0, 0.0
        return None
    if first_length_squared <= _EPS:
        second_fraction = _point_on_segment_fraction(first_start, second_start, second_end)
        if second_fraction is None:
            return None
        return first_start.copy(), 0.0, second_fraction
    if second_length_squared <= _EPS:
        first_fraction = _point_on_segment_fraction(second_start, first_start, first_end)
        if first_fraction is None:
            return None
        return second_start.copy(), first_fraction, 0.0

    denominator = _cross(first_delta, second_delta)
    if abs(denominator) <= _EPS:
        # Collinear overlap has no unique conflict point. Higher-level lane/topology
        # rules should choose a merge or diverge point explicitly.
        return None
    offset = second_start - first_start
    first_fraction = _cross(offset, second_delta) / denominator
    second_fraction = _cross(offset, first_delta) / denominator
    if not (-_EPS <= first_fraction <= 1.0 + _EPS):
        return None
    if not (-_EPS <= second_fraction <= 1.0 + _EPS):
        return None
    first_fraction = float(np.clip(first_fraction, 0.0, 1.0))
    second_fraction = float(np.clip(second_fraction, 0.0, 1.0))
    point = first_start + first_fraction * first_delta
    return point, first_fraction, second_fraction


def _trajectory_primitives(
    points: np.ndarray,
    valid: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    primitives: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    connected = np.zeros(len(points), dtype=bool)
    for index in range(len(points) - 1):
        if valid[index] and valid[index + 1]:
            primitives.append((points[index], points[index + 1], index, index + 1))
            connected[index : index + 2] = True
    for index in np.flatnonzero(valid & ~connected):
        primitives.append((points[index], points[index], int(index), int(index)))
    return primitives


def _interpolate_time(
    timestamps_s: np.ndarray | None,
    start_index: int,
    end_index: int,
    fraction: float,
) -> float:
    if timestamps_s is None:
        return float("nan")
    if start_index == end_index:
        return float(timestamps_s[start_index])
    return float(
        timestamps_s[start_index]
        + fraction * (timestamps_s[end_index] - timestamps_s[start_index])
    )


def find_trajectory_conflict(
    first_positions: Any,
    second_positions: Any,
    first_timestamps_s: Any | None = None,
    second_timestamps_s: Any | None = None,
    *,
    first_valid_mask: Any | None = None,
    second_valid_mask: Any | None = None,
) -> TrajectoryConflict | None:
    """Find the most temporally relevant exact trajectory intersection.

    Only consecutive valid frames form segments, so internal missing frames are
    never bridged. Collinear overlap is intentionally ignored because it has no
    unique conflict point.
    """

    first = _points(first_positions, "first_positions")
    second = _points(second_positions, "second_positions")
    first_valid = _valid_mask(first_valid_mask, len(first), "first_valid_mask")
    second_valid = _valid_mask(second_valid_mask, len(second), "second_valid_mask")
    first_valid &= np.isfinite(first).all(axis=1)
    second_valid &= np.isfinite(second).all(axis=1)
    first_times = (
        None
        if first_timestamps_s is None
        else _timestamps(first_timestamps_s, len(first), sample_period_s=0.1)
    )
    second_times = (
        None
        if second_timestamps_s is None
        else _timestamps(second_timestamps_s, len(second), sample_period_s=0.1)
    )

    candidates: list[tuple[tuple[float, ...], TrajectoryConflict]] = []
    first_primitives = _trajectory_primitives(first, first_valid)
    second_primitives = _trajectory_primitives(second, second_valid)
    if first_primitives and second_primitives:
        first_segments = np.asarray(
            [(start, end) for start, end, _, _ in first_primitives],
            dtype=np.float64,
        )
        second_segments = np.asarray(
            [(start, end) for start, end, _, _ in second_primitives],
            dtype=np.float64,
        )
        first_min = np.min(first_segments, axis=1)
        first_max = np.max(first_segments, axis=1)
        second_min = np.min(second_segments, axis=1)
        second_max = np.max(second_segments, axis=1)
        overlap = np.all(
            (first_min[:, None, :] <= second_max[None, :, :] + _EPS)
            & (second_min[None, :, :] <= first_max[:, None, :] + _EPS),
            axis=2,
        )

        for first_primitive_index, second_primitive_index in np.argwhere(overlap):
            (
                first_start,
                first_end,
                first_start_index,
                first_end_index,
            ) = first_primitives[first_primitive_index]
            (
                second_start,
                second_end,
                second_start_index,
                second_end_index,
            ) = second_primitives[second_primitive_index]
            intersection = _segment_intersection(
                first_start,
                first_end,
                second_start,
                second_end,
            )
            if intersection is None:
                continue
            point, first_fraction, second_fraction = intersection
            first_time = _interpolate_time(
                first_times,
                first_start_index,
                first_end_index,
                first_fraction,
            )
            second_time = _interpolate_time(
                second_times,
                second_start_index,
                second_end_index,
                second_fraction,
            )
            if np.isfinite(first_time) and np.isfinite(second_time):
                time_gap = abs(first_time - second_time)
                score = (
                    time_gap,
                    max(first_time, second_time),
                    first_start_index,
                    second_start_index,
                )
            else:
                time_gap = float("nan")
                score = (
                    float(first_start_index) + first_fraction,
                    float(second_start_index) + second_fraction,
                )
            conflict = TrajectoryConflict(
                point=point.copy(),
                first_segment_index=first_start_index,
                second_segment_index=second_start_index,
                first_segment_fraction=first_fraction,
                second_segment_fraction=second_fraction,
                first_time_s=first_time,
                second_time_s=second_time,
                time_gap_s=time_gap,
            )
            candidates.append((score, conflict))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def post_encroachment_time(
    first_positions: Any,
    second_positions: Any,
    first_timestamps_s: Any,
    second_timestamps_s: Any,
    *,
    first_valid_mask: Any | None = None,
    second_valid_mask: Any | None = None,
) -> float:
    """Return point-mass conflict arrival-time gap, or ``inf`` if paths do not cross."""

    conflict = find_trajectory_conflict(
        first_positions,
        second_positions,
        first_timestamps_s,
        second_timestamps_s,
        first_valid_mask=first_valid_mask,
        second_valid_mask=second_valid_mask,
    )
    if conflict is None:
        return float("inf")
    return conflict.time_gap_s


def minimum_trajectory_distance(
    first_positions: Any,
    second_positions: Any,
    timestamps_s: Any | None = None,
    *,
    first_valid_mask: Any | None = None,
    second_valid_mask: Any | None = None,
) -> TrajectoryMinimumDistance | None:
    """Return closest synchronized distance for two frame-aligned trajectories.

    Linear relative motion is minimized inside each pair of consecutive valid
    frames. Missing frames break interpolation. ``None`` means no shared valid
    frame exists.
    """

    first = _points(first_positions, "first_positions")
    second = _points(second_positions, "second_positions")
    if len(first) != len(second):
        raise ValueError("first_positions and second_positions must have the same length")
    first_valid = _valid_mask(first_valid_mask, len(first), "first_valid_mask")
    second_valid = _valid_mask(second_valid_mask, len(second), "second_valid_mask")
    valid = (
        first_valid
        & second_valid
        & np.isfinite(first).all(axis=1)
        & np.isfinite(second).all(axis=1)
    )
    times = None if timestamps_s is None else _timestamps(timestamps_s, len(first), 0.1)

    candidates: list[TrajectoryMinimumDistance] = []
    for index in np.flatnonzero(valid):
        distance = float(np.linalg.norm(first[index] - second[index]))
        candidates.append(
            TrajectoryMinimumDistance(
                distance_m=distance,
                first_point=first[index].copy(),
                second_point=second[index].copy(),
                frame_index=float(index),
                time_s=float("nan") if times is None else float(times[index]),
            )
        )

    for index in np.flatnonzero(valid[:-1] & valid[1:]):
        first_delta = first[index + 1] - first[index]
        second_delta = second[index + 1] - second[index]
        relative_start = first[index] - second[index]
        relative_delta = first_delta - second_delta
        denominator = float(np.dot(relative_delta, relative_delta))
        if denominator <= _EPS:
            fraction = 0.0
        else:
            fraction = float(
                np.clip(-np.dot(relative_start, relative_delta) / denominator, 0.0, 1.0)
            )
        first_point = first[index] + fraction * first_delta
        second_point = second[index] + fraction * second_delta
        time = (
            float("nan")
            if times is None
            else float(times[index] + fraction * (times[index + 1] - times[index]))
        )
        candidates.append(
            TrajectoryMinimumDistance(
                distance_m=float(np.linalg.norm(first_point - second_point)),
                first_point=first_point,
                second_point=second_point,
                frame_index=float(index) + fraction,
                time_s=time,
            )
        )

    if not candidates:
        return None
    return min(candidates, key=lambda result: (result.distance_m, result.frame_index))


def time_to_collision(
    relative_position: Any,
    relative_velocity: Any,
    *,
    collision_radius_m: float = 0.0,
) -> float:
    """Constant-velocity TTC for two point/disc actors.

    Inputs use ``other - reference`` convention. ``inf`` means valid motion with
    no future collision; ``NaN`` means a non-finite position or velocity input.
    """

    position = _point(relative_position, "relative_position")
    velocity = _point(relative_velocity, "relative_velocity")
    if not np.isfinite(collision_radius_m) or collision_radius_m < 0:
        raise ValueError("collision_radius_m must be finite and nonnegative")
    if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
        return float("nan")

    radius = float(collision_radius_m)
    squared_distance_margin = float(np.dot(position, position) - radius * radius)
    if squared_distance_margin <= 0:
        return 0.0
    speed_squared = float(np.dot(velocity, velocity))
    if speed_squared <= _EPS:
        return float("inf")

    linear = 2.0 * float(np.dot(position, velocity))
    discriminant = linear * linear - 4.0 * speed_squared * squared_distance_margin
    if discriminant < -_EPS:
        return float("inf")
    square_root = float(np.sqrt(max(discriminant, 0.0)))
    roots = (
        (-linear - square_root) / (2.0 * speed_squared),
        (-linear + square_root) / (2.0 * speed_squared),
    )
    future_roots = [root for root in roots if root >= -_EPS]
    if not future_roots:
        return float("inf")
    return max(0.0, min(future_roots))


def time_headway(longitudinal_gap_m: float, follower_speed_mps: float) -> float:
    """Return longitudinal time headway with explicit stationary semantics."""

    gap = float(longitudinal_gap_m)
    speed = float(follower_speed_mps)
    if not (np.isfinite(gap) and np.isfinite(speed)):
        return float("nan")
    if gap <= 0:
        return 0.0
    if speed <= _EPS:
        return float("inf")
    return gap / speed


__all__ = [
    "PolylineProjection",
    "TrajectoryConflict",
    "TrajectoryMinimumDistance",
    "extract_valid_trajectory",
    "find_trajectory_conflict",
    "heading_difference",
    "minimum_trajectory_distance",
    "point_to_polyline_projection",
    "post_encroachment_time",
    "time_headway",
    "time_to_collision",
    "trajectory_acceleration",
    "trajectory_speed",
]

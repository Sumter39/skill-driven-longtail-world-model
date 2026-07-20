from __future__ import annotations

import numpy as np
import pytest

import skilldrive.skills.geometry as geometry
from skilldrive.skills.geometry import (
    extract_valid_trajectory,
    find_trajectory_conflict,
    heading_difference,
    minimum_trajectory_distance,
    point_to_polyline_projection,
    post_encroachment_time,
    time_headway,
    time_to_collision,
    trajectory_acceleration,
    trajectory_speed,
)


def _point_to_polyline_projection_naive(
    point: np.ndarray,
    polyline: np.ndarray,
) -> geometry.PolylineProjection:
    query = geometry._point(point, "point")
    if not np.isfinite(query).all():
        raise ValueError("point must contain only finite values")
    line = geometry._points(polyline, "polyline")
    if len(line) == 0 or not np.isfinite(line).all(axis=1).any():
        raise ValueError("polyline must contain at least one finite point")

    candidates: list[tuple[float, int, float, np.ndarray, float, float]] = []
    arc_length = 0.0
    for index in range(len(line) - 1):
        start, end = line[index], line[index + 1]
        if not (np.isfinite(start).all() and np.isfinite(end).all()):
            continue
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= geometry._EPS:
            fraction = 0.0
            projected = start.copy()
            heading = float("nan")
        else:
            fraction = float(
                np.clip(np.dot(query - start, delta) / (length * length), 0.0, 1.0)
            )
            projected = start + fraction * delta
            heading = float(np.arctan2(delta[1], delta[0]))
        distance = float(np.linalg.norm(query - projected))
        candidates.append(
            (distance, index, fraction, projected, arc_length + fraction * length, heading)
        )
        arc_length += length

    if not candidates:
        finite_indices = np.flatnonzero(np.isfinite(line).all(axis=1))
        distances = np.linalg.norm(line[finite_indices] - query, axis=1)
        nearest = int(finite_indices[np.argmin(distances)])
        projected = line[nearest].copy()
        return geometry.PolylineProjection(
            point=projected,
            distance_m=float(np.linalg.norm(query - projected)),
            signed_lateral_distance_m=float("nan"),
            arc_length_m=0.0,
            segment_index=-1,
            segment_fraction=float("nan"),
            heading_rad=float("nan"),
        )

    distance, index, fraction, projected, along, heading = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    if np.isfinite(heading):
        tangent = np.array([np.cos(heading), np.sin(heading)])
        offset = query - projected
        signed_lateral = float(tangent[0] * offset[1] - tangent[1] * offset[0])
    else:
        signed_lateral = float("nan")
    return geometry.PolylineProjection(
        point=projected.copy(),
        distance_m=distance,
        signed_lateral_distance_m=signed_lateral,
        arc_length_m=along,
        segment_index=index,
        segment_fraction=fraction,
        heading_rad=heading,
    )


def _assert_projections_equal(
    actual: geometry.PolylineProjection,
    expected: geometry.PolylineProjection,
) -> None:
    np.testing.assert_allclose(actual.point, expected.point, equal_nan=True)
    assert actual.distance_m == pytest.approx(expected.distance_m, nan_ok=True)
    assert actual.signed_lateral_distance_m == pytest.approx(
        expected.signed_lateral_distance_m,
        nan_ok=True,
    )
    assert actual.arc_length_m == pytest.approx(expected.arc_length_m, nan_ok=True)
    assert actual.segment_index == expected.segment_index
    assert actual.segment_fraction == pytest.approx(
        expected.segment_fraction,
        nan_ok=True,
    )
    assert actual.heading_rad == pytest.approx(expected.heading_rad, nan_ok=True)


def _find_trajectory_conflict_naive(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
    first_timestamps_s: np.ndarray | None = None,
    second_timestamps_s: np.ndarray | None = None,
    *,
    first_valid_mask: np.ndarray | None = None,
    second_valid_mask: np.ndarray | None = None,
) -> geometry.TrajectoryConflict | None:
    first = geometry._points(first_positions, "first_positions")
    second = geometry._points(second_positions, "second_positions")
    first_valid = geometry._valid_mask(
        first_valid_mask,
        len(first),
        "first_valid_mask",
    )
    second_valid = geometry._valid_mask(
        second_valid_mask,
        len(second),
        "second_valid_mask",
    )
    first_valid &= np.isfinite(first).all(axis=1)
    second_valid &= np.isfinite(second).all(axis=1)
    first_times = (
        None
        if first_timestamps_s is None
        else geometry._timestamps(first_timestamps_s, len(first), sample_period_s=0.1)
    )
    second_times = (
        None
        if second_timestamps_s is None
        else geometry._timestamps(second_timestamps_s, len(second), sample_period_s=0.1)
    )

    candidates: list[tuple[tuple[float, ...], geometry.TrajectoryConflict]] = []
    first_primitives = geometry._trajectory_primitives(first, first_valid)
    second_primitives = geometry._trajectory_primitives(second, second_valid)
    for first_start, first_end, first_start_index, first_end_index in first_primitives:
        for (
            second_start,
            second_end,
            second_start_index,
            second_end_index,
        ) in second_primitives:
            intersection = geometry._segment_intersection(
                first_start,
                first_end,
                second_start,
                second_end,
            )
            if intersection is None:
                continue
            point, first_fraction, second_fraction = intersection
            first_time = geometry._interpolate_time(
                first_times,
                first_start_index,
                first_end_index,
                first_fraction,
            )
            second_time = geometry._interpolate_time(
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
            conflict = geometry.TrajectoryConflict(
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


def _assert_conflicts_equal(
    actual: geometry.TrajectoryConflict | None,
    expected: geometry.TrajectoryConflict | None,
) -> None:
    if expected is None:
        assert actual is None
        return

    assert actual is not None
    np.testing.assert_allclose(actual.point, expected.point)
    assert actual.first_segment_index == expected.first_segment_index
    assert actual.second_segment_index == expected.second_segment_index
    assert actual.first_segment_fraction == pytest.approx(expected.first_segment_fraction)
    assert actual.second_segment_fraction == pytest.approx(expected.second_segment_fraction)
    assert actual.first_time_s == pytest.approx(expected.first_time_s, nan_ok=True)
    assert actual.second_time_s == pytest.approx(expected.second_time_s, nan_ok=True)
    assert actual.time_gap_s == pytest.approx(expected.time_gap_s, nan_ok=True)


def test_extract_valid_trajectory_applies_mask_and_ignores_nan() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 0.0], [3.0, 0.0]])
    points, indices = extract_valid_trajectory(
        positions,
        valid_mask=np.array([True, False, True, True]),
    )

    np.testing.assert_allclose(points, [[0.0, 0.0], [3.0, 0.0]])
    np.testing.assert_array_equal(indices, [0, 3])


def test_extract_valid_trajectory_rejects_invalid_mask_shape() -> None:
    with pytest.raises(ValueError, match="valid_mask"):
        extract_valid_trajectory(np.zeros((3, 2)), valid_mask=np.ones(2, dtype=bool))


def test_speed_and_acceleration_use_elapsed_time() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])

    speed = trajectory_speed(positions, timestamps_s=timestamps)
    acceleration = trajectory_acceleration(positions, timestamps_s=timestamps)

    np.testing.assert_allclose(speed[1:], [1.0, 2.0, 3.0])
    assert np.isnan(speed[0])
    np.testing.assert_allclose(acceleration[2:], [1.0, 1.0])
    assert np.isnan(acceleration[:2]).all()


def test_kinematics_do_not_bridge_missing_frames_and_handle_stationary_track() -> None:
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, np.nan], [3.0, 0.0]])
    speed = trajectory_speed(positions, sample_period_s=0.1)
    acceleration = trajectory_acceleration(positions, sample_period_s=0.1)

    assert speed[1] == pytest.approx(10.0)
    assert np.isnan(speed[[0, 2, 3]]).all()
    assert np.isnan(acceleration).all()

    stationary = np.zeros((4, 2))
    np.testing.assert_allclose(trajectory_speed(stationary)[1:], 0.0)
    np.testing.assert_allclose(trajectory_acceleration(stationary)[2:], 0.0)


def test_kinematics_reject_non_increasing_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        trajectory_speed(
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            timestamps_s=np.array([0.0, 1.0, 1.0]),
        )


def test_point_to_polyline_projection_reports_geometry() -> None:
    result = point_to_polyline_projection(
        np.array([1.0, 2.0]),
        np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]),
    )

    np.testing.assert_allclose(result.point, [1.0, 0.0])
    assert result.distance_m == pytest.approx(2.0)
    assert result.signed_lateral_distance_m == pytest.approx(2.0)
    assert result.arc_length_m == pytest.approx(1.0)
    assert result.segment_index == 0
    assert result.segment_fraction == pytest.approx(1.0 / 3.0)
    assert result.heading_rad == pytest.approx(0.0)


def test_point_to_polyline_projection_handles_degenerate_polyline() -> None:
    result = point_to_polyline_projection(
        np.array([4.0, 6.0]),
        np.array([[1.0, 2.0], [1.0, 2.0]]),
    )

    np.testing.assert_allclose(result.point, [1.0, 2.0])
    assert result.distance_m == pytest.approx(5.0)
    assert np.isnan(result.heading_rad)
    assert np.isnan(result.signed_lateral_distance_m)


def test_point_to_polyline_projection_does_not_bridge_nan_gap() -> None:
    result = point_to_polyline_projection(
        np.array([1.0, 0.0]),
        np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]]),
    )

    assert result.segment_index == -1
    assert result.distance_m == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("point", "polyline"),
    [
        (
            np.array([0.0, 0.0]),
            np.array(
                [[-1.0, -1.0], [-1.0, 1.0], [np.nan, np.nan], [1.0, -1.0], [1.0, 1.0]]
            ),
        ),
        (
            np.array([3.0, 3.0]),
            np.array(
                [[0.0, 0.0], [2.0, 0.0], [np.nan, np.nan], [2.0, 2.0], [2.0, 2.0], [2.0, 4.0]]
            ),
        ),
        (
            np.array([1.0, 1.0]),
            np.array([[0.0, 0.0], [np.nan, np.nan], [2.0, 0.0]]),
        ),
        (
            np.array([0.5, 1.0]),
            np.array([[0.0, 0.0], [0.5e-12, 0.0], [1.0, 0.0]]),
        ),
        (
            np.array([0.0, 0.0]),
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        ),
    ],
)
def test_projection_matches_naive_reference_for_edge_cases(
    point: np.ndarray,
    polyline: np.ndarray,
) -> None:
    expected = _point_to_polyline_projection_naive(point, polyline)
    actual = point_to_polyline_projection(point, polyline)

    _assert_projections_equal(actual, expected)


def test_projection_matches_naive_reference_for_random_polylines() -> None:
    random = np.random.default_rng(2026)

    for _ in range(200):
        length = int(random.integers(1, 25))
        polyline = random.normal(size=(length, 2))
        if length > 1:
            repeated = random.random(length - 1) < 0.15
            polyline[1:][repeated] = polyline[:-1][repeated]
            missing = random.random(length) < 0.2
            missing[int(random.integers(length))] = False
            polyline[missing] = np.nan
        point = random.normal(size=2)

        expected = _point_to_polyline_projection_naive(point, polyline)
        actual = point_to_polyline_projection(point, polyline)

        _assert_projections_equal(actual, expected)


def test_heading_difference_wraps_at_pi_and_preserves_nan() -> None:
    values = heading_difference(
        np.deg2rad(np.array([179.0, 0.0, np.nan])),
        np.deg2rad(np.array([-179.0, 180.0, 0.0])),
    )

    np.testing.assert_allclose(values[:2], np.deg2rad([2.0, 180.0]), atol=1e-12)
    assert np.isnan(values[2])


def test_conflict_point_and_pet_interpolate_arrival_times() -> None:
    first = np.array([[-1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, -1.0], [0.0, 1.0]])
    first_times = np.array([0.0, 2.0])
    second_times = np.array([0.0, 3.0])

    conflict = find_trajectory_conflict(first, second, first_times, second_times)

    assert conflict is not None
    np.testing.assert_allclose(conflict.point, [0.0, 0.0])
    assert conflict.first_time_s == pytest.approx(1.0)
    assert conflict.second_time_s == pytest.approx(1.5)
    assert conflict.time_gap_s == pytest.approx(0.5)
    assert post_encroachment_time(first, second, first_times, second_times) == pytest.approx(0.5)


def test_conflict_detection_handles_stationary_actor_and_rejects_gap_bridge() -> None:
    stationary = np.array([[0.0, 0.0], [0.0, 0.0]])
    crossing = np.array([[0.0, -1.0], [0.0, 1.0]])
    assert find_trajectory_conflict(stationary, crossing) is not None

    gapped = np.array([[-1.0, 0.0], [np.nan, np.nan], [1.0, 0.0]])
    vertical = np.array([[0.0, -1.0], [0.0, 0.0], [0.0, 1.0]])
    assert find_trajectory_conflict(gapped, vertical) is None


def test_conflict_detection_skips_exact_checks_for_disjoint_aabbs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def count_intersection_calls(
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(geometry, "_segment_intersection", count_intersection_calls)

    result = find_trajectory_conflict(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[10.0, 10.0], [11.0, 10.0], [12.0, 10.0]]),
    )

    assert result is None
    assert calls == 0


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [1.0, 1.0]]),
        ),
        (
            np.array([[1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        ),
    ],
)
def test_conflict_detection_keeps_boundary_and_degenerate_point_contacts(
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    conflict = find_trajectory_conflict(first, second)

    assert conflict is not None
    np.testing.assert_allclose(conflict.point, [1.0, 0.0])


@pytest.mark.parametrize(
    ("first", "second", "kwargs"),
    [
        (
            np.array([[-1.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, -1.0], [0.0, 1.0]]),
            {
                "first_timestamps_s": np.array([0.0, 2.0]),
                "second_timestamps_s": np.array([0.0, 3.0]),
            },
        ),
        (
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            np.array([[0.0, 1.0], [2.0, 1.0]]),
            {},
        ),
        (
            np.array([[-1.0, 0.0], [np.nan, np.nan], [0.0, 0.0]]),
            np.array([[0.0, -1.0], [0.0, 1.0]]),
            {},
        ),
        (
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[-1.0, 0.0], [1.0, 0.0]]),
            {},
        ),
        (
            np.array([[-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
            np.array([[0.0, -1.0], [0.0, 1.0]]),
            {
                "first_timestamps_s": np.array([0.0, 1.0, 2.0]),
                "second_timestamps_s": np.array([0.0, 2.0]),
            },
        ),
        (
            np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            np.array([[0.0, -1.0], [0.0, 1.0], [2.0, 1.0]]),
            {
                "first_valid_mask": np.array([True, True, False]),
                "second_valid_mask": np.array([True, True, False]),
            },
        ),
    ],
)
def test_conflict_detection_matches_naive_reference(
    first: np.ndarray,
    second: np.ndarray,
    kwargs: dict[str, np.ndarray],
) -> None:
    expected = _find_trajectory_conflict_naive(first, second, **kwargs)
    actual = find_trajectory_conflict(first, second, **kwargs)

    _assert_conflicts_equal(actual, expected)


def test_parallel_trajectories_have_no_conflict_or_pet() -> None:
    first = np.array([[0.0, 0.0], [2.0, 0.0]])
    second = np.array([[0.0, 1.0], [2.0, 1.0]])
    times = np.array([0.0, 1.0])

    assert find_trajectory_conflict(first, second, times, times) is None
    assert np.isinf(post_encroachment_time(first, second, times, times))


def test_minimum_trajectory_distance_interpolates_between_frames() -> None:
    first = np.array([[-1.0, 0.0], [1.0, 0.0]])
    second = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = minimum_trajectory_distance(
        first,
        second,
        timestamps_s=np.array([4.0, 6.0]),
    )

    assert result is not None
    assert result.distance_m == pytest.approx(0.0)
    assert result.frame_index == pytest.approx(0.5)
    assert result.time_s == pytest.approx(5.0)
    np.testing.assert_allclose(result.first_point, [0.0, 0.0])
    np.testing.assert_allclose(result.second_point, [0.0, 0.0])


def test_minimum_trajectory_distance_requires_common_valid_frames() -> None:
    first = np.array([[0.0, 0.0], [np.nan, np.nan]])
    second = np.array([[np.nan, np.nan], [0.0, 0.0]])

    assert minimum_trajectory_distance(first, second) is None


def test_time_to_collision_positive_negative_and_boundary_cases() -> None:
    assert time_to_collision([10.0, 0.0], [-2.0, 0.0]) == pytest.approx(5.0)
    assert time_to_collision([10.0, 0.0], [-2.0, 0.0], collision_radius_m=2.0) == pytest.approx(4.0)
    assert np.isinf(time_to_collision([10.0, 0.0], [2.0, 0.0]))
    assert np.isinf(time_to_collision([10.0, 0.0], [0.0, 0.0]))
    assert time_to_collision([1.0, 0.0], [0.0, 0.0], collision_radius_m=1.0) == 0.0
    assert np.isnan(time_to_collision([np.nan, 0.0], [-1.0, 0.0]))


def test_time_headway_handles_stationary_overlap_and_invalid_values() -> None:
    assert time_headway(20.0, 10.0) == pytest.approx(2.0)
    assert np.isinf(time_headway(20.0, 0.0))
    assert time_headway(-0.5, 5.0) == 0.0
    assert np.isnan(time_headway(np.nan, 5.0))

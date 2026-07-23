from __future__ import annotations

import numpy as np
import pytest

from skilldrive.filtering.common import KinematicLimits, check_kinematics
from skilldrive.generation.search import (
    KinematicTopKAccumulator,
    score_kinematic_candidates,
)
from skilldrive.schemas import AgentTrack, Scenario


def _scenario() -> Scenario:
    timestamps = np.arange(110, dtype=np.int64) * 100_000_000
    positions = np.column_stack(
        (np.arange(110, dtype=np.float64) * 0.1, np.zeros(110))
    )
    target = AgentTrack(
        track_id="target",
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([1.0, 0.0], (110, 1)),
        headings=np.zeros(110),
        observed_mask=np.array([True] * 50 + [False] * 60),
        is_focal=True,
    )
    return Scenario(
        scenario_id="scene",
        city_name="city",
        timestamps=timestamps,
        focal_track_id="target",
        agents=[target],
        map_polylines=[],
    )


def _limits(**overrides: float | None) -> KinematicLimits:
    values = {
        "maximum_seam_speed_mps": 2.0,
        "maximum_speed_mps": 3.0,
        "maximum_acceleration_mps2": 2.0,
        "maximum_deceleration_mps2": 2.0,
        "maximum_jerk_mps3": 20.0,
        "maximum_curvature_per_m": 0.5,
        "maximum_heading_rate_rad_s": 0.5,
        "minimum_heading_speed_mps": 0.5,
    }
    values.update(overrides)
    return KinematicLimits(**values)


def _history_only(scenario: Scenario) -> Scenario:
    target = scenario.agents[0]
    return Scenario(
        scenario_id=scenario.scenario_id,
        city_name=scenario.city_name,
        timestamps=scenario.timestamps[:50].copy(),
        focal_track_id=scenario.focal_track_id,
        agents=[
            AgentTrack(
                track_id=target.track_id,
                object_type=target.object_type,
                positions=target.positions[:50].copy(),
                velocities=target.velocities[:50].copy(),
                headings=target.headings[:50].copy(),
                observed_mask=target.observed_mask[:50].copy(),
                is_focal=target.is_focal,
            )
        ],
        map_polylines=[],
    )


def test_vectorized_scores_match_scalar_filter_and_metrics() -> None:
    scenario = _scenario()
    anchor = scenario.agents[0].positions[49]
    rng = np.random.default_rng(2026)
    futures = []
    for _ in range(24):
        increments = np.column_stack(
            (
                0.1 + rng.normal(0.0, 0.015, 60),
                rng.normal(0.0, 0.008, 60),
            )
        )
        futures.append((anchor + np.cumsum(increments, axis=0)).astype(np.float32))
    candidates = np.stack(futures)
    limits = _limits()

    result = score_kinematic_candidates(
        scenario,
        "target",
        candidates,
        limits,
        np.arange(len(candidates), dtype=np.int64) + 100,
        top_k=7,
    )

    metric_fields = {
        "seam_speed_mps": result.seam_speed_mps,
        "maximum_speed_mps": result.maximum_speed_mps,
        "maximum_speed_future_index": result.maximum_speed_future_index,
        "maximum_acceleration_mps2": result.maximum_acceleration_mps2,
        "maximum_acceleration_future_index": result.maximum_acceleration_future_index,
        "maximum_deceleration_mps2": result.maximum_deceleration_mps2,
        "maximum_deceleration_future_index": result.maximum_deceleration_future_index,
        "maximum_jerk_mps3": result.maximum_jerk_mps3,
        "maximum_jerk_future_index": result.maximum_jerk_future_index,
        "maximum_curvature_per_m": result.maximum_curvature_per_m,
        "maximum_curvature_future_index": result.maximum_curvature_future_index,
        "maximum_heading_rate_rad_s": result.maximum_heading_rate_rad_s,
        "maximum_heading_rate_future_index": result.maximum_heading_rate_future_index,
        "low_speed_heading_suppressed_steps": result.low_speed_heading_suppressed_steps,
    }
    for index, future in enumerate(candidates):
        scalar = check_kinematics(scenario, "target", future, limits)
        assert result.passed[index] == scalar.passed
        assert result.rejection_reasons[index] == scalar.rejection_reasons
        for name, values in metric_fields.items():
            if name.endswith("_index") or name.endswith("_steps"):
                assert int(values[index]) == scalar.metrics[name]
            else:
                assert float(values[index]) == pytest.approx(
                    scalar.metrics[name], rel=0.0, abs=1e-12
                )


def test_limit_boundary_and_normalized_violation_match_scalar_tolerance() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    scalar = check_kinematics(scenario, "target", future, _limits())
    boundary = float(scalar.metrics["maximum_jerk_mps3"])
    limits = _limits(maximum_jerk_mps3=max(boundary, 1e-6))
    result = score_kinematic_candidates(
        scenario,
        "target",
        future[None],
        limits,
        np.array([7], dtype=np.int64),
        top_k=1,
    )

    assert result.passed[0]
    assert result.normalized_violation_score[0] == 0.0

    fast = np.column_stack(
        (
            scenario.agents[0].positions[49, 0]
            + 0.3 * np.arange(1, 61, dtype=np.float64),
            np.zeros(60),
        )
    ).astype(np.float32)
    rejected = score_kinematic_candidates(
        scenario,
        "target",
        fast[None],
        _limits(maximum_acceleration_mps2=10.0, maximum_jerk_mps3=1000.0),
        np.array([8], dtype=np.int64),
        top_k=1,
    )
    assert not rejected.passed[0]
    assert rejected.normalized_violation_score[0] > 0.0


def test_scoring_does_not_read_or_modify_source_future() -> None:
    first = _scenario()
    second = _scenario()
    second.agents[0].positions[50:] = np.nan
    second.agents[0].velocities[50:] = np.inf
    second.agents[0].headings[50:] = np.nan
    future = first.agents[0].positions[50:].astype(np.float32, copy=True)
    original = future.copy()
    seeds = np.array([1], dtype=np.int64)

    left = score_kinematic_candidates(
        first, "target", future[None], _limits(), seeds, top_k=1
    )
    right = score_kinematic_candidates(
        second, "target", future[None], _limits(), seeds, top_k=1
    )

    np.testing.assert_array_equal(future, original)
    np.testing.assert_array_equal(left.passed, right.passed)
    np.testing.assert_array_equal(
        left.normalized_violation_score, right.normalized_violation_score
    )
    np.testing.assert_array_equal(left.maximum_jerk_mps3, right.maximum_jerk_mps3)


def test_history_only_source_matches_complete_source() -> None:
    complete = _scenario()
    history = _history_only(complete)
    future = complete.agents[0].positions[50:].astype(np.float32, copy=True)
    seeds = np.array([31], dtype=np.int64)

    full_result = score_kinematic_candidates(
        complete, "target", future[None], _limits(), seeds, top_k=1
    )
    history_result = score_kinematic_candidates(
        history, "target", future[None], _limits(), seeds, top_k=1
    )

    np.testing.assert_array_equal(full_result.passed, history_result.passed)
    np.testing.assert_allclose(
        full_result.maximum_jerk_mps3,
        history_result.maximum_jerk_mps3,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        full_result.normalized_violation_score,
        history_result.normalized_violation_score,
        rtol=0.0,
        atol=1e-12,
    )


def test_nonfinite_previous_history_velocity_matches_scalar_rejection() -> None:
    scenario = _scenario()
    scenario.agents[0].velocities[48] = np.nan
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    limits = _limits()

    scalar = check_kinematics(scenario, "target", future, limits)
    result = score_kinematic_candidates(
        scenario,
        "target",
        future[None],
        limits,
        np.array([4], dtype=np.int64),
        top_k=1,
    )

    assert not result.finite_kinematics[0]
    assert result.rejection_reasons[0] == scalar.rejection_reasons
    assert np.isinf(result.normalized_violation_score[0])


def test_top_k_is_stable_by_score_then_latent_seed_then_candidate_index() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    candidates = np.repeat(future[None], 5, axis=0)
    result = score_kinematic_candidates(
        scenario,
        "target",
        candidates,
        _limits(),
        np.array([30, 10, 20, 10, 40], dtype=np.int64),
        candidate_indices=np.array([100, 101, 102, 103, 104], dtype=np.int64),
        top_k=4,
    )

    np.testing.assert_array_equal(result.top_k_indices, [101, 103, 102, 100])


def test_chunked_accumulator_matches_one_shot_top_k() -> None:
    scenario = _scenario()
    anchor = scenario.agents[0].positions[49]
    rng = np.random.default_rng(39)
    increments = np.zeros((17, 60, 2), dtype=np.float64)
    increments[..., 0] = 0.1 + rng.normal(0.0, 0.025, (17, 60))
    increments[..., 1] = rng.normal(0.0, 0.012, (17, 60))
    candidates = (anchor + np.cumsum(increments, axis=1)).astype(np.float32)
    seeds = rng.integers(0, 1_000_000, size=17, dtype=np.int64)
    indices = np.arange(500, 517, dtype=np.int64)
    limits = _limits()

    together = score_kinematic_candidates(
        scenario,
        "target",
        candidates,
        limits,
        seeds,
        candidate_indices=indices,
        top_k=6,
    )
    accumulator = KinematicTopKAccumulator(6)
    for start, stop in ((0, 3), (3, 11), (11, 17)):
        accumulator.update(
            score_kinematic_candidates(
                scenario,
                "target",
                candidates[start:stop],
                limits,
                seeds[start:stop],
                candidate_indices=indices[start:stop],
                top_k=6,
            )
        )

    np.testing.assert_array_equal(accumulator.top_k_indices, together.top_k_indices)


def test_chunked_accumulator_rejects_duplicate_candidate_indices() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    scores = score_kinematic_candidates(
        scenario,
        "target",
        future[None],
        _limits(),
        np.array([9], dtype=np.int64),
        candidate_indices=np.array([12], dtype=np.int64),
        top_k=1,
    )
    accumulator = KinematicTopKAccumulator(1)
    accumulator.update(scores)
    with pytest.raises(ValueError, match="duplicate indices"):
        accumulator.update(scores)


@pytest.mark.parametrize(
    ("futures", "seeds", "message"),
    [
        (np.zeros((2, 59, 2), dtype=np.float32), np.array([1, 2]), "shape"),
        (np.zeros((2, 60, 2), dtype=np.float64), np.array([1, 2]), "float32"),
        (np.zeros((2, 60, 2), dtype=np.float32), np.array([1.0, 2.0]), "integer"),
        (np.zeros((2, 60, 2), dtype=np.float32), np.array([1, -2]), "nonnegative"),
    ],
)
def test_scoring_rejects_invalid_candidate_contract(
    futures: np.ndarray, seeds: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_kinematic_candidates(
            _scenario(), "target", futures, _limits(), seeds, top_k=1
        )

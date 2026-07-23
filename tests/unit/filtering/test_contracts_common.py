from __future__ import annotations

import numpy as np
import pytest

from skilldrive.filtering.common import (
    KinematicLimits,
    check_kinematics,
    check_schema_and_finite,
    derive_future_kinematics,
)
from skilldrive.filtering.contracts import (
    FilterCheck,
    FilterDecision,
    FilterRejection,
    FilterStage,
)
from skilldrive.generation.config import FILTER_STAGES
from skilldrive.generation.contracts import FilterDecision as GenerationFilterDecision
from skilldrive.schemas import AgentTrack, Scenario


def _scenario() -> Scenario:
    timestamps = np.arange(110, dtype=np.int64) * 100_000_000
    x = np.arange(110, dtype=np.float64) * 0.1
    positions = np.column_stack((x, np.zeros(110)))
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    target = AgentTrack(
        track_id="target",
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([1.0, 0.0], (110, 1)),
        headings=np.zeros(110),
        observed_mask=observed,
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


def _limits(**overrides) -> KinematicLimits:
    values = {
        "maximum_seam_speed_mps": 2.0,
        "maximum_speed_mps": 2.0,
        "maximum_acceleration_mps2": 1.0,
        "maximum_jerk_mps3": 1.0,
    }
    values.update(overrides)
    return KinematicLimits(**values)


def test_filter_contract_reuses_generation_decision_and_freezes_stage_order() -> None:
    assert FilterDecision is GenerationFilterDecision
    assert tuple(stage.value for stage in FilterStage) == FILTER_STAGES
    check = FilterCheck(
        stage=FilterStage.SCHEMA_FINITE,
        rejection_reasons=(FilterRejection.INVALID_FUTURE_SHAPE,),
    )
    assert check.passed is False
    assert check.rejection_values == ("schema.invalid_future_shape",)

    with pytest.raises(ValueError, match="unknown filter rejection reason"):
        FilterDecision.create(
            candidate_id="a" * 64,
            filter_config_sha256="b" * 64,
            filter_contract_version=1,
            accepted=False,
            rejection_reasons=("ad_hoc_reason",),
        )


def test_schema_and_finite_checks_shape_anchor_timestamps_and_values() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    assert check_schema_and_finite(scenario, "target", future).passed

    invalid_dtype = future.astype(np.float64)
    assert check_schema_and_finite(
        scenario, "target", invalid_dtype
    ).rejection_reasons == (FilterRejection.INVALID_FUTURE_DTYPE,)

    invalid = future.copy()
    invalid[3, 0] = np.nan
    result = check_schema_and_finite(scenario, "target", invalid)
    assert result.rejection_reasons == (
        FilterRejection.NON_FINITE_GENERATED_POSITIONS,
    )
    assert check_schema_and_finite(scenario, "target", future[:-1]).rejection_reasons == (
        FilterRejection.INVALID_FUTURE_SHAPE,
    )
    assert check_schema_and_finite(scenario, "missing", future).rejection_reasons == (
        FilterRejection.TARGET_TRACK_MISSING,
    )

    scenario.timestamps[20] = scenario.timestamps[19]
    assert FilterRejection.NON_MONOTONIC_TIMESTAMPS in check_schema_and_finite(
        scenario, "target", future
    ).rejection_reasons

    scenario = _scenario()
    scenario.timestamps[1:] *= 2
    assert FilterRejection.INVALID_SAMPLE_PERIOD in check_schema_and_finite(
        scenario, "target", future
    ).rejection_reasons


def test_kinematics_uses_exact_seam_and_explicit_limits() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].astype(np.float32, copy=True)
    values = derive_future_kinematics(scenario, "target", future)

    np.testing.assert_allclose(values.speed_mps, 1.0, atol=6e-6)
    np.testing.assert_allclose(values.acceleration_mps2, 0.0, atol=1e-4)
    np.testing.assert_allclose(values.jerk_mps3, 0.0, atol=2e-3)
    np.testing.assert_allclose(values.heading_rad, 0.0)
    assert check_kinematics(scenario, "target", future, _limits()).passed

    anchor_x = scenario.agents[0].positions[49, 0]
    fast_future = np.column_stack(
        (anchor_x + np.arange(1, 61, dtype=np.float64), np.zeros(60))
    ).astype(np.float32)
    seam = check_kinematics(
        scenario,
        "target",
        fast_future,
        _limits(
            maximum_seam_speed_mps=5.0,
            maximum_speed_mps=20.0,
            maximum_acceleration_mps2=200.0,
            maximum_jerk_mps3=2000.0,
        ),
    )
    assert seam.rejection_reasons == (FilterRejection.SEAM_SPEED_LIMIT_EXCEEDED,)
    assert seam.metrics["seam_speed_mps"] == pytest.approx(10.0)


def test_acceleration_and_jerk_limits_are_not_implicit_defaults() -> None:
    scenario = _scenario()
    anchor_x = scenario.agents[0].positions[49, 0]
    future = np.column_stack(
        (anchor_x + 0.3 * np.arange(1, 61, dtype=np.float64), np.zeros(60))
    ).astype(np.float32)
    result = check_kinematics(
        scenario,
        "target",
        future,
        _limits(
            maximum_seam_speed_mps=None,
            maximum_speed_mps=None,
            maximum_acceleration_mps2=10.0,
            maximum_jerk_mps3=1000.0,
        ),
    )
    assert result.rejection_reasons == (
        FilterRejection.ACCELERATION_LIMIT_EXCEEDED,
    )
    assert result.metrics["maximum_acceleration_mps2"] == pytest.approx(20.0)

    with pytest.raises(ValueError, match="positive finite"):
        _limits(maximum_speed_mps=0.0)


def test_deceleration_has_an_independent_inclusive_boundary() -> None:
    scenario = _scenario()
    anchor = scenario.agents[0].positions[49].copy()
    speeds = np.maximum(0.0, 1.0 - 0.1 * np.arange(1, 61, dtype=np.float64))
    future_x = anchor[0] + np.cumsum(speeds * 0.1)
    future = np.column_stack((future_x, np.zeros(60))).astype(np.float32)
    boundary = float(
        np.max(derive_future_kinematics(scenario, "target", future).deceleration_mps2)
    )
    limits = _limits(
        maximum_seam_speed_mps=2.0,
        maximum_speed_mps=2.0,
        maximum_acceleration_mps2=2.0,
        maximum_deceleration_mps2=boundary,
        maximum_jerk_mps3=20.0,
    )

    assert check_kinematics(scenario, "target", future, limits).passed
    rejected = check_kinematics(
        scenario,
        "target",
        future,
        _limits(
            maximum_seam_speed_mps=2.0,
            maximum_speed_mps=2.0,
            maximum_acceleration_mps2=2.0,
            maximum_deceleration_mps2=boundary - 1e-5,
            maximum_jerk_mps3=20.0,
        ),
    )
    assert FilterRejection.DECELERATION_LIMIT_EXCEEDED in rejected.rejection_reasons


def test_low_speed_heading_noise_is_suppressed_before_rate_and_curvature() -> None:
    scenario = _scenario()
    anchor = scenario.agents[0].positions[49].copy()
    increments = np.column_stack(
        (
            np.full(60, 0.001),
            np.where(np.arange(60) % 2 == 0, 0.001, -0.001),
        )
    )
    future = (anchor + np.cumsum(increments, axis=0)).astype(np.float32)
    values = derive_future_kinematics(
        scenario,
        "target",
        future,
        minimum_heading_speed_mps=0.5,
    )

    assert values.low_speed_heading_suppressed.all()
    np.testing.assert_array_equal(values.heading_rate_rad_s, 0.0)
    np.testing.assert_array_equal(values.curvature_per_m, 0.0)


def test_turning_motion_is_checked_by_heading_rate_and_curvature() -> None:
    scenario = _scenario()
    anchor = scenario.agents[0].positions[49].copy()
    headings = np.arange(1, 61, dtype=np.float64) * 0.08
    increments = 0.1 * np.column_stack((np.cos(headings), np.sin(headings)))
    future = (anchor + np.cumsum(increments, axis=0)).astype(np.float32)
    result = check_kinematics(
        scenario,
        "target",
        future,
        _limits(
            maximum_seam_speed_mps=2.0,
            maximum_speed_mps=2.0,
            maximum_acceleration_mps2=2.0,
            maximum_deceleration_mps2=2.0,
            maximum_jerk_mps3=20.0,
            maximum_curvature_per_m=0.5,
            maximum_heading_rate_rad_s=0.5,
        ),
    )

    assert FilterRejection.CURVATURE_LIMIT_EXCEEDED in result.rejection_reasons
    assert FilterRejection.HEADING_RATE_LIMIT_EXCEEDED in result.rejection_reasons

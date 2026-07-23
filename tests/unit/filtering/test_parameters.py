from __future__ import annotations

from dataclasses import replace

import numpy as np

from skilldrive.filtering.common import derive_future_kinematics
from skilldrive.filtering.contracts import FilterRejection
from skilldrive.filtering.parameters import check_parameter_realization
from skilldrive.generation.assembly import materialize_overlay_scenario
from skilldrive.generation.config import load_filter_config
from skilldrive.schemas import AgentTrack, Scenario


def _source() -> Scenario:
    time = np.arange(110, dtype=np.float64) * 0.1
    positions = np.column_stack((time, np.zeros(110)))
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
        scenario_id="parameter",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="target",
        agents=[target],
        map_polylines=[],
    )


def _generated(source: Scenario, speed_mps: float) -> tuple[Scenario, np.ndarray]:
    anchor = source.agents[0].positions[49]
    future = anchor + np.column_stack(
        (
            np.arange(1, 61, dtype=np.float64) * 0.1 * speed_mps,
            np.zeros(60),
        )
    )
    future = future.astype(np.float32)
    return materialize_overlay_scenario(source, "target", future), future


def test_parameter_check_reports_computed_and_unavailable_fields() -> None:
    source = _source()
    generated, future = _generated(source, 2.0)
    values = derive_future_kinematics(source, "target", future)
    policy = load_filter_config().parameter_policy

    result = check_parameter_realization(
        requested_parameters={"leader_speed_scale": 2.0, "response_start_s": 1.0},
        source_scenario=source,
        generated_scenario=generated,
        target_track_id="target",
        kinematics=values,
        risk_metric=None,
        risk_value=None,
        policy=policy,
    )

    assert result.passed
    assert result.metrics["parameters"]["leader_speed_scale"]["within_tolerance"]
    assert result.metrics["parameters"]["response_start_s"]["status"] == "unavailable"
    assert result.metrics["parameters"]["leader_speed_scale"]["conditioning_claim"] == (
        "rule_search_realization_not_direct_parameter_control"
    )


def test_computed_parameter_outside_field_tolerance_is_rejected() -> None:
    source = _source()
    generated, future = _generated(source, 2.0)
    values = derive_future_kinematics(source, "target", future)
    policy = replace(
        load_filter_config().parameter_policy,
        absolute_tolerances={
            **load_filter_config().parameter_policy.absolute_tolerances,
            "scale": 0.01,
        },
    )

    result = check_parameter_realization(
        requested_parameters={"leader_speed_scale": 1.0},
        source_scenario=source,
        generated_scenario=generated,
        target_track_id="target",
        kinematics=values,
        risk_metric=None,
        risk_value=None,
        policy=policy,
    )

    assert result.rejection_reasons == (
        FilterRejection.PARAMETER_OUT_OF_TOLERANCE,
    )

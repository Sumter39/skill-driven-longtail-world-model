from __future__ import annotations

import copy

import numpy as np
import pytest

from skilldrive.data.coordinates import local_to_global
from skilldrive.filtering.contracts import FilterRejection, FilterStage
from skilldrive.filtering.history import (
    check_coordinate_round_trip,
    check_history_and_coordinates,
    check_history_invariants,
)
from skilldrive.generation.assembly import materialize_overlay_scenario
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _agent(track_id: str, offset: float) -> AgentTrack:
    x = np.arange(110, dtype=np.float64) * 0.1 + offset
    return AgentTrack(
        track_id=track_id,
        object_type="vehicle",
        positions=np.column_stack((x, np.zeros(110))),
        velocities=np.tile([1.0, 0.0], (110, 1)),
        headings=np.zeros(110),
        observed_mask=np.arange(110) < 50,
        is_focal=track_id == "target",
    )


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="scene",
        city_name="PIT",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="target",
        agents=[_agent("target", 0.0), _agent("background", 5.0)],
        map_polylines=[
            MapPolyline(
                polyline_id="lane:1:center",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 0.0], [20.0, 0.0]]),
                direction="vehicle",
                lane_id="1",
                successor_ids=["2"],
            )
        ],
        metadata={"source_path": "train/scene.parquet", "tags": ["formal"]},
    )


def _overlay(source: Scenario) -> tuple[Scenario, np.ndarray, np.ndarray]:
    local = np.column_stack((np.arange(1, 61) * 0.1, np.zeros(60)))
    origin = source.agents[0].positions[49]
    global_positions = local_to_global(local, origin, 0.0)
    return (
        materialize_overlay_scenario(source, "target", global_positions),
        local,
        global_positions,
    )


def test_history_and_coordinate_check_accepts_only_target_future_replacement() -> None:
    source = _scenario()
    materialized, local, global_positions = _overlay(source)

    result = check_history_and_coordinates(
        source,
        materialized,
        "target",
        local,
        global_positions,
        source.agents[0].positions[49],
        0.0,
    )

    assert result.stage is FilterStage.HISTORY_INVARIANTS
    assert result.passed
    assert result.metrics["target_history_unchanged"] is True
    assert result.metrics["background_tracks_unchanged"] is True
    assert result.metrics["maximum_coordinate_round_trip_error_m"] < 1e-4


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda scenario: scenario.timestamps.__setitem__(10, -1),
            FilterRejection.HISTORY_TIMESTAMPS_CHANGED,
        ),
        (
            lambda scenario: scenario.agents[0].positions.__setitem__((10, 0), 99.0),
            FilterRejection.HISTORY_TARGET_CHANGED,
        ),
        (
            lambda scenario: scenario.agents[0].observed_mask.__setitem__(80, True),
            FilterRejection.HISTORY_TARGET_CHANGED,
        ),
        (
            lambda scenario: scenario.agents[1].velocities.__setitem__((80, 0), 99.0),
            FilterRejection.BACKGROUND_TRACK_CHANGED,
        ),
        (
            lambda scenario: scenario.agents[1].observed_mask.__setitem__(80, True),
            FilterRejection.BACKGROUND_TRACK_CHANGED,
        ),
        (
            lambda scenario: scenario.map_polylines[0].points.__setitem__((0, 0), 1.0),
            FilterRejection.MAP_CHANGED,
        ),
        (
            lambda scenario: scenario.metadata.__setitem__("source_path", "other"),
            FilterRejection.METADATA_CHANGED,
        ),
    ],
)
def test_history_check_rejects_each_rewritten_source_component(
    mutation,
    reason: FilterRejection,
) -> None:
    source = _scenario()
    materialized, _, _ = _overlay(source)
    materialized = copy.deepcopy(materialized)
    mutation(materialized)

    result = check_history_invariants(source, materialized, "target")

    assert reason in result.rejection_reasons


def test_history_check_allows_target_future_motion_fields_to_change() -> None:
    source = _scenario()
    materialized, _, _ = _overlay(source)
    materialized.agents[0].positions[50:] += 2.0
    materialized.agents[0].velocities[50:] += 3.0
    materialized.agents[0].headings[50:] += 0.5

    assert check_history_invariants(source, materialized, "target").passed


def test_history_check_reports_missing_background_track_id() -> None:
    source = _scenario()
    materialized, _, _ = _overlay(source)
    materialized.agents.pop()

    result = check_history_invariants(source, materialized, "target")

    assert FilterRejection.HISTORY_TARGET_CHANGED in result.rejection_reasons
    assert FilterRejection.BACKGROUND_TRACK_CHANGED in result.rejection_reasons
    assert result.metrics["missing_background_track_ids"] == ["background"]


def test_coordinate_round_trip_uses_strict_one_e_minus_four_boundary() -> None:
    local = np.zeros((60, 2), dtype=np.float64)
    global_positions = local.copy()
    global_positions[4, 0] = 1e-4

    result = check_coordinate_round_trip(
        local,
        global_positions,
        np.zeros(2),
        0.0,
    )

    assert result.rejection_reasons == (
        FilterRejection.COORDINATE_ROUND_TRIP_EXCEEDED,
    )
    assert result.metrics["maximum_coordinate_round_trip_error_m"] == pytest.approx(
        1e-4
    )


def test_coordinate_round_trip_rejects_non_finite_input() -> None:
    local = np.zeros((60, 2), dtype=np.float64)
    local[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        check_coordinate_round_trip(local, np.zeros((60, 2)), np.zeros(2), 0.0)

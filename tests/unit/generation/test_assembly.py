from __future__ import annotations

import numpy as np
import pytest

from skilldrive.generation.assembly import (
    local_futures_to_global,
    materialize_overlay_scenario,
)
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _agent(track_id: str, offset: float = 0.0) -> AgentTrack:
    positions = np.column_stack((np.arange(110, dtype=np.float64) + offset, np.zeros(110)))
    velocities = np.tile(np.array([10.0, 0.0]), (110, 1))
    return AgentTrack(
        track_id=track_id,
        object_type="vehicle",
        positions=positions,
        velocities=velocities,
        headings=np.zeros(110),
        observed_mask=np.array([True] * 50 + [False] * 60),
        is_focal=track_id == "target",
    )


def _scenario() -> Scenario:
    return Scenario(
        scenario_id="scenario",
        city_name="PIT",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="target",
        agents=[_agent("target"), _agent("context", 5.0)],
        map_polylines=[
            MapPolyline(
                polyline_id="lane",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 0.0], [100.0, 0.0]]),
            )
        ],
        metadata={"source_path": "train/scenario/scenario_scenario.parquet"},
    )


def test_local_futures_to_global_supports_multiple_candidates() -> None:
    local = np.zeros((2, 60, 2), dtype=np.float32)
    local[0, :, 0] = np.arange(60)
    local[1, :, 1] = np.arange(60)

    global_positions = local_futures_to_global(local, np.array([10.0, 20.0]), np.pi / 2)

    assert global_positions.shape == (2, 60, 2)
    np.testing.assert_allclose(global_positions[0, 1], [10.0, 21.0], atol=1e-8)
    np.testing.assert_allclose(global_positions[1, 1], [9.0, 20.0], atol=1e-8)


def test_materialize_overlay_replaces_only_target_future() -> None:
    scenario = _scenario()
    original_target_positions = scenario.agents[0].positions.copy()
    original_target_velocities = scenario.agents[0].velocities.copy()
    original_target_headings = scenario.agents[0].headings.copy()
    original_context_positions = scenario.agents[1].positions.copy()
    original_context_velocities = scenario.agents[1].velocities.copy()
    original_context_headings = scenario.agents[1].headings.copy()
    future = np.column_stack((49.0 + 0.5 * np.arange(1, 61), np.ones(60)))

    overlaid = materialize_overlay_scenario(scenario, "target", future)

    np.testing.assert_array_equal(scenario.agents[0].positions, original_target_positions)
    np.testing.assert_array_equal(scenario.agents[0].velocities, original_target_velocities)
    np.testing.assert_array_equal(scenario.agents[0].headings, original_target_headings)
    np.testing.assert_array_equal(scenario.agents[1].positions, original_context_positions)
    np.testing.assert_array_equal(overlaid.agents[0].positions[:50], original_target_positions[:50])
    np.testing.assert_array_equal(overlaid.agents[0].positions[50:], future)
    np.testing.assert_array_equal(overlaid.agents[1].positions, original_context_positions)
    np.testing.assert_array_equal(overlaid.agents[1].velocities, original_context_velocities)
    np.testing.assert_array_equal(overlaid.agents[1].headings, original_context_headings)
    np.testing.assert_array_equal(
        overlaid.agents[0].velocities[:50], original_target_velocities[:50]
    )
    np.testing.assert_array_equal(
        overlaid.agents[0].headings[:50], original_target_headings[:50]
    )
    np.testing.assert_array_equal(overlaid.agents[0].observed_mask, scenario.agents[0].observed_mask)
    np.testing.assert_allclose(overlaid.agents[0].velocities[50], [5.0, 10.0])
    assert overlaid.agents[0].headings[50] == pytest.approx(np.arctan2(10.0, 5.0))


def test_materialize_overlay_resolves_anchor_heading_from_history_positions() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[48] = [0.0, 0.0]
    target.positions[49] = [0.0, 1.0]
    target.velocities[49] = [0.0, 0.0]
    target.headings[49] = np.nan
    future = np.repeat(target.positions[49][None, :], 60, axis=0)

    overlaid = materialize_overlay_scenario(scenario, "target", future)

    np.testing.assert_allclose(overlaid.agents[0].velocities[50:], 0.0)
    np.testing.assert_allclose(overlaid.agents[0].headings[50:], np.pi / 2)


def test_materialize_overlay_rejects_unresolved_anchor_heading() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[48] = target.positions[49]
    target.velocities[49] = [0.0, 0.0]
    target.headings[49] = np.nan
    future = np.repeat(target.positions[49][None, :], 60, axis=0)

    with pytest.raises(ValueError, match="heading cannot be resolved"):
        materialize_overlay_scenario(scenario, "target", future)


@pytest.mark.parametrize(
    "future",
    [np.zeros((59, 2)), np.full((60, 2), np.nan)],
)
def test_materialize_overlay_rejects_invalid_future(future: np.ndarray) -> None:
    with pytest.raises(ValueError):
        materialize_overlay_scenario(_scenario(), "target", future)

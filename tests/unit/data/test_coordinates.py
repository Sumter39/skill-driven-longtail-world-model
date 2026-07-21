import numpy as np
import pytest

from skilldrive.data.coordinates import (
    global_to_local,
    local_to_global,
    to_agent_frame,
    to_focal_frame,
    wrap_angle,
)
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _agent_frame_scenario() -> Scenario:
    steps = 110
    timestamps = np.arange(steps, dtype=np.int64)
    focal_positions = np.column_stack((timestamps.astype(float), np.zeros(steps)))
    anchor_positions = np.column_stack(
        (np.full(steps, 10.0), timestamps.astype(float))
    )
    observed = timestamps < 50
    focal = AgentTrack(
        track_id="focal",
        object_type="vehicle",
        positions=focal_positions,
        velocities=np.tile([1.0, 0.0], (steps, 1)),
        headings=np.zeros(steps),
        observed_mask=observed,
        is_focal=True,
    )
    anchor = AgentTrack(
        track_id="anchor",
        object_type="vehicle",
        positions=anchor_positions,
        velocities=np.tile([0.0, 1.0], (steps, 1)),
        headings=np.full(steps, np.pi / 2),
        observed_mask=observed,
    )
    lane = MapPolyline(
        polyline_id="lane",
        polyline_type="lane_centerline",
        points=np.array([[10.0, 40.0], [10.0, 60.0]]),
    )
    return Scenario(
        scenario_id="agent-frame",
        city_name="test-city",
        timestamps=timestamps,
        focal_track_id=focal.track_id,
        agents=[focal, anchor],
        map_polylines=[lane],
        metadata={"source": "synthetic"},
    )


def test_coordinate_round_trip() -> None:
    points = np.array([[10.0, -3.0], [25.5, 4.0], [-1.0, 8.0]])
    origin = np.array([7.0, -2.0])
    heading = 0.73
    recovered = local_to_global(global_to_local(points, origin, heading), origin, heading)
    np.testing.assert_allclose(recovered, points, atol=1e-10)


def test_heading_frame_orientation() -> None:
    point = np.array([[0.0, 2.0]])
    local = global_to_local(point, np.zeros(2), np.pi / 2)
    np.testing.assert_allclose(local, [[2.0, 0.0]], atol=1e-10)


def test_wrap_angle_range() -> None:
    wrapped = wrap_angle(np.array([-4 * np.pi, -np.pi, np.pi, 3 * np.pi]))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_agent_frame_uses_arbitrary_anchor_at_frame_49() -> None:
    scenario = _agent_frame_scenario()

    local = to_agent_frame(scenario, "anchor")

    anchor = next(agent for agent in local.agents if agent.track_id == "anchor")
    focal = next(agent for agent in local.agents if agent.track_id == "focal")
    np.testing.assert_allclose(anchor.positions[49], [0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(focal.positions[49], [-49.0, -39.0], atol=1e-10)
    np.testing.assert_allclose(local.map_polylines[0].points[0], [-9.0, 0.0], atol=1e-10)
    assert local.metadata == {
        "source": "synthetic",
        "coordinate_frame": "agent_at_frame",
        "anchor_track_id": "anchor",
        "frame_index": 49,
        "frame_origin_global_xy": [10.0, 49.0],
        "frame_heading_global_rad": np.pi / 2,
    }


def test_agent_frame_falls_back_to_velocity_heading() -> None:
    scenario = _agent_frame_scenario()
    anchor = next(agent for agent in scenario.agents if agent.track_id == "anchor")
    anchor.headings[49] = np.nan

    local = to_agent_frame(scenario, "anchor")

    assert local.metadata["frame_heading_global_rad"] == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(
        next(agent for agent in local.agents if agent.track_id == "anchor").velocities[49],
        [1.0, 0.0],
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("anchor_track_id", "frame_index", "mutation", "message"),
    [
        ("missing", 49, None, "does not reference an agent"),
        ("anchor", 110, None, "between 0 and 109"),
        ("anchor", -1, None, "between 0 and 109"),
        ("anchor", 49.5, None, "must be an integer"),
        (
            "anchor",
            49,
            "missing_state",
            "no finite observed state at frame 49",
        ),
        (
            "anchor",
            49,
            "unobserved_state",
            "no finite observed state at frame 49",
        ),
        (
            "anchor",
            49,
            "missing_heading",
            "heading and velocity are unavailable",
        ),
    ],
)
def test_agent_frame_rejects_invalid_anchor_or_frame(
    anchor_track_id: str,
    frame_index: int | float,
    mutation: str | None,
    message: str,
) -> None:
    scenario = _agent_frame_scenario()
    anchor = next(agent for agent in scenario.agents if agent.track_id == "anchor")
    if mutation == "missing_state":
        anchor.positions[49] = np.nan
    elif mutation == "unobserved_state":
        anchor.observed_mask[49] = False
    elif mutation == "missing_heading":
        anchor.headings[49] = np.nan
        anchor.velocities[49] = 0.0

    with pytest.raises(ValueError, match=message):
        to_agent_frame(scenario, anchor_track_id, frame_index)  # type: ignore[arg-type]


def test_scenario_focal_frame_places_focal_at_origin(synthetic_scenario: Scenario) -> None:
    lane = synthetic_scenario.map_polylines[0]
    lane.lane_id = "lane-1"
    lane.mark_type = "dashed_white"
    lane.predecessor_ids = ["lane-0"]
    lane.successor_ids = ["lane-2"]
    lane.left_neighbor_id = "lane-left"
    local = to_focal_frame(synthetic_scenario)
    focal = local.agents[0]
    last_observed = np.flatnonzero(focal.observed_mask)[-1]
    np.testing.assert_allclose(focal.positions[last_observed], [0.0, 0.0], atol=1e-10)
    assert local.metadata["coordinate_frame"] == "focal_agent_at_last_observation"
    assert local.metadata["anchor_track_id"] == synthetic_scenario.focal_track_id
    assert local.metadata["frame_index"] == 4
    local_lane = local.map_polylines[0]
    assert local_lane.lane_id == "lane-1"
    assert local_lane.mark_type == "dashed_white"
    assert local_lane.predecessor_ids == ["lane-0"]
    assert local_lane.successor_ids == ["lane-2"]
    assert local_lane.left_neighbor_id == "lane-left"

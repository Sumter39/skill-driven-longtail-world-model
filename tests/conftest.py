from __future__ import annotations

import numpy as np
import pytest

from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


@pytest.fixture
def synthetic_scenario() -> Scenario:
    steps = 12
    positions = np.column_stack((np.arange(steps, dtype=float), np.zeros(steps)))
    velocities = np.column_stack((np.ones(steps), np.zeros(steps)))
    focal = AgentTrack(
        track_id="focal",
        object_type="vehicle",
        positions=positions,
        velocities=velocities,
        headings=np.zeros(steps),
        observed_mask=np.arange(steps) < 5,
        is_focal=True,
    )
    lane = MapPolyline(
        polyline_id="lane-center",
        polyline_type="lane_centerline",
        points=np.column_stack((np.linspace(-10, 20, 20), np.zeros(20))),
        direction="east",
    )
    return Scenario(
        scenario_id="test-scene",
        city_name="test-city",
        timestamps=np.arange(steps, dtype=np.int64),
        focal_track_id="focal",
        agents=[focal],
        map_polylines=[lane],
    )

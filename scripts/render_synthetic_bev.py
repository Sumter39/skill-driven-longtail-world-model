"""Render one small synthetic scene to verify the preparation-only BEV pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.visualization import render_bev


def _agent(track_id: str, object_type: str, x: np.ndarray, y: np.ndarray, focal: bool = False) -> AgentTrack:
    positions = np.column_stack((x, y))
    velocities = np.gradient(positions, axis=0) * 10.0
    headings = np.arctan2(velocities[:, 1], velocities[:, 0])
    observed = np.arange(len(x)) < 50
    return AgentTrack(track_id, object_type, positions, velocities, headings, observed, focal)


def build_synthetic_scenario() -> Scenario:
    steps = 110
    time = np.arange(steps) / 10.0
    focal = _agent("focal", "vehicle", -30 + 5.5 * time, np.zeros(steps), True)
    neighbor = _agent("neighbor", "vehicle", -20 + 5.0 * time, 3.5 - 0.18 * np.maximum(time - 5, 0) ** 2)
    pedestrian = _agent("pedestrian", "pedestrian", 8 + np.zeros(steps), -8 + 1.2 * time)
    map_polylines = []
    for index, y in enumerate((-3.5, 0.0, 3.5, 7.0)):
        map_polylines.append(
            MapPolyline(
                polyline_id=f"lane-{index}",
                polyline_type="lane_boundary" if index in (0, 3) else "lane_centerline",
                points=np.column_stack((np.linspace(-60, 60, 80), np.full(80, y))),
                direction="east",
            )
        )
    return Scenario(
        scenario_id="synthetic-preparation-check",
        city_name="synthetic",
        timestamps=np.arange(steps, dtype=np.int64),
        focal_track_id="focal",
        agents=[focal, neighbor, pedestrian],
        map_polylines=map_polylines,
        metadata={"purpose": "preparation-only BEV smoke test"},
    )


if __name__ == "__main__":
    output = render_bev(build_synthetic_scenario(), Path("outputs") / "synthetic_bev.png")
    print(output.resolve())

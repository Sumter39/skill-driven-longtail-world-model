"""Optional adapter from one Argoverse 2 motion-forecasting scene to SkillDrive schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _resample_polyline(points: np.ndarray, count: int = 40) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points[:, :2]
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] == 0:
        return np.repeat(points[:1, :2], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def discover_map_path(scenario_path: str | Path) -> Path:
    """Find the AV2 static-map JSON stored beside a scenario parquet file."""
    directory = Path(scenario_path).parent
    matches = sorted(directory.glob("log_map_archive_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one log_map_archive_*.json beside {scenario_path}, found {len(matches)}"
        )
    return matches[0]


def load_av2_scenario(
    scenario_path: str | Path, map_path: str | Path | None = None
) -> Scenario:
    """Load one AV2 scenario without importing AV2 until the function is called."""
    try:
        from av2.datasets.motion_forecasting import scenario_serialization
        from av2.map.map_api import ArgoverseStaticMap
    except ImportError as error:
        raise RuntimeError(
            "The optional AV2 dependency is not installed. Install the project with '[av2]'."
        ) from error

    source = Path(scenario_path)
    av2_scenario = scenario_serialization.load_argoverse_scenario_parquet(source)
    static_map = ArgoverseStaticMap.from_json(map_path or discover_map_path(source))

    timestamps = np.asarray(av2_scenario.timestamps_ns, dtype=np.int64)
    agents: list[AgentTrack] = []
    for track in av2_scenario.tracks:
        positions = np.full((len(timestamps), 2), np.nan, dtype=np.float64)
        velocities = np.full((len(timestamps), 2), np.nan, dtype=np.float64)
        headings = np.full(len(timestamps), np.nan, dtype=np.float64)
        observed_mask = np.zeros(len(timestamps), dtype=bool)
        for state in track.object_states:
            timestep = int(state.timestep)
            if 0 <= timestep < len(timestamps):
                positions[timestep] = np.asarray(state.position, dtype=np.float64)[:2]
                velocities[timestep] = np.asarray(state.velocity, dtype=np.float64)[:2]
                headings[timestep] = float(state.heading)
                observed_mask[timestep] = bool(state.observed)
        agents.append(
            AgentTrack(
                track_id=str(track.track_id),
                object_type=_enum_value(track.object_type),
                positions=positions,
                velocities=velocities,
                headings=headings,
                observed_mask=observed_mask,
                is_focal=str(track.track_id) == str(av2_scenario.focal_track_id),
            )
        )

    map_polylines: list[MapPolyline] = []
    for lane in static_map.get_scenario_lane_segments():
        lane_id = str(lane.id)
        left = np.asarray(lane.left_lane_boundary.xyz, dtype=np.float64)[:, :2]
        right = np.asarray(lane.right_lane_boundary.xyz, dtype=np.float64)[:, :2]
        is_intersection = bool(lane.is_intersection)
        map_polylines.extend(
            [
                MapPolyline(
                    polyline_id=f"{lane_id}:left",
                    polyline_type="lane_boundary_left",
                    points=left,
                    is_intersection=is_intersection,
                ),
                MapPolyline(
                    polyline_id=f"{lane_id}:right",
                    polyline_type="lane_boundary_right",
                    points=right,
                    is_intersection=is_intersection,
                ),
                MapPolyline(
                    polyline_id=f"{lane_id}:center",
                    polyline_type="lane_centerline",
                    points=(_resample_polyline(left) + _resample_polyline(right)) / 2,
                    direction=_enum_value(lane.lane_type),
                    is_intersection=is_intersection,
                ),
            ]
        )

    return Scenario(
        scenario_id=str(av2_scenario.scenario_id),
        city_name=str(av2_scenario.city_name),
        timestamps=timestamps,
        focal_track_id=str(av2_scenario.focal_track_id),
        agents=agents,
        map_polylines=map_polylines,
        metadata={"source_path": str(source), "map_path": str(map_path or discover_map_path(source))},
    )

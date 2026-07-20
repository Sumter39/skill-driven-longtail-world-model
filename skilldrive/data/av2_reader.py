"""Optional adapter from one Argoverse 2 motion-forecasting scene to SkillDrive schemas."""

from __future__ import annotations

from importlib import import_module
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


def _close_outline(points: np.ndarray) -> np.ndarray:
    """Return a polygon outline with one explicit closing point."""
    points = np.asarray(points, dtype=np.float64)
    if not len(points) or np.array_equal(points[0, :2], points[-1, :2]):
        return points[:, :2]
    return np.vstack((points[:, :2], points[0, :2]))


def _optional_id(value: Any) -> str | None:
    return None if value is None else str(value)


def _map_polylines(static_map: Any) -> list[MapPolyline]:
    """Convert AV2 static-map objects without depending on their concrete classes."""
    map_polylines: list[MapPolyline] = []
    for lane in static_map.get_scenario_lane_segments():
        lane_id = str(lane.id)
        left = np.asarray(lane.left_lane_boundary.xyz, dtype=np.float64)[:, :2]
        right = np.asarray(lane.right_lane_boundary.xyz, dtype=np.float64)[:, :2]
        is_intersection = bool(lane.is_intersection)
        left_mark_type = _enum_value(lane.left_mark_type)
        right_mark_type = _enum_value(lane.right_mark_type)
        map_polylines.extend(
            [
                MapPolyline(
                    polyline_id=f"{lane_id}:left",
                    polyline_type="lane_boundary_left",
                    points=left,
                    is_intersection=is_intersection,
                    lane_id=lane_id,
                    mark_type=left_mark_type,
                ),
                MapPolyline(
                    polyline_id=f"{lane_id}:right",
                    polyline_type="lane_boundary_right",
                    points=right,
                    is_intersection=is_intersection,
                    lane_id=lane_id,
                    mark_type=right_mark_type,
                ),
                MapPolyline(
                    polyline_id=f"{lane_id}:center",
                    polyline_type="lane_centerline",
                    points=(_resample_polyline(left) + _resample_polyline(right)) / 2,
                    direction=_enum_value(lane.lane_type),
                    is_intersection=is_intersection,
                    lane_id=lane_id,
                    left_mark_type=left_mark_type,
                    right_mark_type=right_mark_type,
                    predecessor_ids=[str(value) for value in lane.predecessors],
                    successor_ids=[str(value) for value in lane.successors],
                    left_neighbor_id=_optional_id(lane.left_neighbor_id),
                    right_neighbor_id=_optional_id(lane.right_neighbor_id),
                ),
            ]
        )

    for crossing in static_map.get_scenario_ped_crossings():
        edge1 = np.asarray(crossing.edge1.xyz, dtype=np.float64)[:, :2]
        edge2 = np.asarray(crossing.edge2.xyz, dtype=np.float64)[:, :2]
        outline = _close_outline(np.concatenate((edge1, edge2[::-1]), axis=0))
        map_polylines.append(
            MapPolyline(
                polyline_id=f"crosswalk:{crossing.id}",
                polyline_type="pedestrian_crossing",
                points=outline,
            )
        )

    for area in static_map.get_scenario_vector_drivable_areas():
        boundary = np.asarray(
            [(point.x, point.y) for point in area.area_boundary], dtype=np.float64
        )
        map_polylines.append(
            MapPolyline(
                polyline_id=f"drivable_area:{area.id}",
                polyline_type="drivable_area",
                points=_close_outline(boundary),
            )
        )

    return map_polylines


def discover_map_path(scenario_path: str | Path) -> Path:
    """Find the AV2 static-map JSON stored beside a scenario parquet file."""
    source = Path(scenario_path)
    directory = source.parent
    if source.stem.startswith("scenario_"):
        scenario_id = source.stem.removeprefix("scenario_")
        direct_match = directory / f"log_map_archive_{scenario_id}.json"
        if direct_match.is_file():
            return direct_match
    matches = sorted(directory.glob("log_map_archive_*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one log_map_archive_*.json beside {scenario_path}, found {len(matches)}"
        )
    return matches[0]


def preload_av2_worker_dependencies() -> None:
    """Initialize pandas before forked AV2 scan workers are created."""

    try:
        pandas = import_module("pandas")
        getattr(pandas, "DataFrame")
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "The optional AV2 dependency is not installed. Install the project with '[av2]'."
        ) from error


def preload_av2_dependencies() -> tuple[Any, Any]:
    """Load the complete optional AV2 stack used by one scenario reader."""

    preload_av2_worker_dependencies()
    try:
        scenario_serialization = import_module(
            "av2.datasets.motion_forecasting.scenario_serialization"
        )
        map_api = import_module("av2.map.map_api")
        argoverse_static_map = map_api.ArgoverseStaticMap
    except (AttributeError, ImportError) as error:
        raise RuntimeError(
            "The optional AV2 dependency is not installed. Install the project with '[av2]'."
        ) from error
    return scenario_serialization, argoverse_static_map


def load_av2_scenario(
    scenario_path: str | Path, map_path: str | Path | None = None
) -> Scenario:
    """Load one AV2 scenario without importing AV2 until the function is called."""
    scenario_serialization, ArgoverseStaticMap = preload_av2_dependencies()

    source = Path(scenario_path)
    resolved_map_path = (
        discover_map_path(source) if map_path is None else Path(map_path)
    )
    av2_scenario = scenario_serialization.load_argoverse_scenario_parquet(source)
    static_map = ArgoverseStaticMap.from_json(resolved_map_path)

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

    map_polylines = _map_polylines(static_map)

    return Scenario(
        scenario_id=str(av2_scenario.scenario_id),
        city_name=str(av2_scenario.city_name),
        timestamps=timestamps,
        focal_track_id=str(av2_scenario.focal_track_id),
        agents=agents,
        map_polylines=map_polylines,
        metadata={"source_path": str(source), "map_path": str(resolved_map_path)},
    )

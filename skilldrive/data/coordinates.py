"""Two-dimensional coordinate transforms for agent-centric BEV scenes."""

from __future__ import annotations

import numpy as np

from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _check_points(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.shape[-1] != 2:
        raise ValueError(f"points must end with a coordinate dimension of 2, got {array.shape}")
    return array


def global_to_local(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """Translate and rotate global XY points into an agent-centric frame."""
    points_array = _check_points(points)
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,):
        raise ValueError("origin must have shape (2,)")
    delta = points_array - origin_array
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return delta @ rotation.T


def local_to_global(points: np.ndarray, origin: np.ndarray, heading: float) -> np.ndarray:
    """Transform agent-centric XY points back into the global frame."""
    points_array = _check_points(points)
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,):
        raise ValueError("origin must have shape (2,)")
    cosine, sine = np.cos(heading), np.sin(heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return points_array @ rotation.T + origin_array


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to the half-open interval [-pi, pi)."""
    array = np.asarray(angle, dtype=np.float64)
    return (array + np.pi) % (2 * np.pi) - np.pi


def to_focal_frame(scenario: Scenario) -> Scenario:
    """Return a copy of a scenario in the focal agent's last observed frame."""
    focal = next(agent for agent in scenario.agents if agent.track_id == scenario.focal_track_id)
    valid_observed = focal.observed_mask & np.isfinite(focal.positions).all(axis=1)
    indices = np.flatnonzero(valid_observed)
    if not len(indices):
        raise ValueError("focal agent has no finite observed state")
    index = int(indices[-1])
    origin = focal.positions[index].copy()
    heading = float(focal.headings[index])
    if not np.isfinite(heading):
        velocity = focal.velocities[index]
        if not np.isfinite(velocity).all() or np.linalg.norm(velocity) == 0:
            raise ValueError("focal heading and velocity are unavailable at the frame origin")
        heading = float(np.arctan2(velocity[1], velocity[0]))

    cosine, sine = np.cos(heading), np.sin(heading)
    velocity_rotation = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    agents = [
        AgentTrack(
            track_id=agent.track_id,
            object_type=agent.object_type,
            positions=global_to_local(agent.positions, origin, heading),
            velocities=agent.velocities @ velocity_rotation.T,
            headings=wrap_angle(agent.headings - heading),
            observed_mask=agent.observed_mask.copy(),
            is_focal=agent.is_focal,
        )
        for agent in scenario.agents
    ]
    map_polylines = [
        MapPolyline(
            polyline_id=polyline.polyline_id,
            polyline_type=polyline.polyline_type,
            points=global_to_local(polyline.points, origin, heading),
            direction=polyline.direction,
            is_intersection=polyline.is_intersection,
        )
        for polyline in scenario.map_polylines
    ]
    metadata = dict(scenario.metadata)
    metadata["coordinate_frame"] = "focal_agent_at_last_observation"
    metadata["frame_origin_global_xy"] = origin.tolist()
    metadata["frame_heading_global_rad"] = heading
    return Scenario(
        scenario_id=scenario.scenario_id,
        city_name=scenario.city_name,
        timestamps=scenario.timestamps.copy(),
        focal_track_id=scenario.focal_track_id,
        agents=agents,
        map_polylines=map_polylines,
        metadata=metadata,
    )

"""Assemble single-target trajectory overlays without mutating source scenes."""

from __future__ import annotations

import math

import numpy as np

from skilldrive.data.coordinates import local_to_global
from skilldrive.schemas import AgentTrack, Scenario


HISTORY_STEPS = 50
FUTURE_STEPS = 60
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS


def local_futures_to_global(
    future_position_local: np.ndarray,
    origin: np.ndarray,
    heading: float,
) -> np.ndarray:
    """Convert one or more ``[..., 60, 2]`` local futures to global positions."""

    local = np.asarray(future_position_local, dtype=np.float64)
    if local.ndim < 2 or local.shape[-2:] != (FUTURE_STEPS, 2):
        raise ValueError("future_position_local must end with shape (60, 2)")
    if not np.isfinite(local).all():
        raise ValueError("future_position_local must contain only finite values")
    flat = local.reshape(-1, 2)
    converted = local_to_global(flat, origin, heading)
    return converted.reshape(local.shape)


def _copy_agent(agent: AgentTrack) -> AgentTrack:
    return AgentTrack(
        track_id=agent.track_id,
        object_type=agent.object_type,
        positions=agent.positions.copy(),
        velocities=agent.velocities.copy(),
        headings=agent.headings.copy(),
        observed_mask=agent.observed_mask.copy(),
        is_focal=agent.is_focal,
    )


def _sample_period_seconds(scenario: Scenario) -> float:
    if len(scenario.timestamps) < 2:
        raise ValueError("scenario must contain at least two timestamps")
    deltas = np.diff(scenario.timestamps.astype(np.float64)) / 1_000_000_000.0
    finite = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if not len(finite):
        raise ValueError("scenario timestamps do not define a positive sample period")
    period = float(np.median(finite))
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("scenario sample period must be finite and positive")
    return period


def materialize_overlay_scenario(
    scenario: Scenario,
    target_track_id: str,
    future_positions_global: np.ndarray,
) -> Scenario:
    """Return a copied scene with one target future replaced and motion recomputed."""

    future = np.asarray(future_positions_global, dtype=np.float64)
    if future.shape != (FUTURE_STEPS, 2):
        raise ValueError("future_positions_global must have shape (60, 2)")
    if not np.isfinite(future).all():
        raise ValueError("future_positions_global must contain only finite values")

    agents_by_id = {agent.track_id: agent for agent in scenario.agents}
    if target_track_id not in agents_by_id:
        raise ValueError(f"target track is not present in scenario: {target_track_id}")
    target = agents_by_id[target_track_id]
    if len(target.positions) < TOTAL_STEPS:
        raise ValueError("target track must contain at least 110 frames")
    anchor = target.positions[HISTORY_STEPS - 1]
    if not np.isfinite(anchor).all():
        raise ValueError("target anchor position at frame 49 must be finite")

    period = _sample_period_seconds(scenario)
    copied_agents = [_copy_agent(agent) for agent in scenario.agents]
    copied_target = next(agent for agent in copied_agents if agent.track_id == target_track_id)
    copied_target.positions[HISTORY_STEPS:TOTAL_STEPS] = future

    previous_positions = np.vstack((anchor, future[:-1]))
    velocities = (future - previous_positions) / period
    copied_target.velocities[HISTORY_STEPS:TOTAL_STEPS] = velocities

    previous_heading = float(copied_target.headings[HISTORY_STEPS - 1])
    if not math.isfinite(previous_heading):
        anchor_velocity = copied_target.velocities[HISTORY_STEPS - 1]
        if np.isfinite(anchor_velocity).all() and np.linalg.norm(anchor_velocity) > 1e-6:
            previous_heading = float(np.arctan2(anchor_velocity[1], anchor_velocity[0]))
        else:
            anchor_delta = (
                copied_target.positions[HISTORY_STEPS - 1]
                - copied_target.positions[HISTORY_STEPS - 2]
            )
            if np.isfinite(anchor_delta).all() and np.linalg.norm(anchor_delta) > 1e-6:
                previous_heading = float(np.arctan2(anchor_delta[1], anchor_delta[0]))
            else:
                raise ValueError("target heading cannot be resolved at frame 49")
    headings = np.empty(FUTURE_STEPS, dtype=np.float64)
    for index, velocity in enumerate(velocities):
        if np.linalg.norm(velocity) > 1e-6:
            previous_heading = float(np.arctan2(velocity[1], velocity[0]))
        headings[index] = previous_heading
    copied_target.headings[HISTORY_STEPS:TOTAL_STEPS] = headings

    return Scenario(
        scenario_id=scenario.scenario_id,
        city_name=scenario.city_name,
        timestamps=scenario.timestamps.copy(),
        focal_track_id=scenario.focal_track_id,
        agents=copied_agents,
        map_polylines=list(scenario.map_polylines),
        metadata=dict(scenario.metadata),
    )


__all__ = ["local_futures_to_global", "materialize_overlay_scenario"]

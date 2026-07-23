"""Synchronized oriented-box collision checks using explicit class proxies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.assembly import HISTORY_STEPS, TOTAL_STEPS
from skilldrive.generation.config import CounterfactualFilterConfig, FootprintProxy
from skilldrive.schemas import Scenario


@dataclass(frozen=True)
class ProxyCollisionContact:
    other_track_id: str
    other_object_type: str
    frame_index: int


@dataclass(frozen=True)
class ProxyCollisionReport:
    proxy_source: str
    ground_truth: bool
    target_proxy_available: bool
    contacts: tuple[ProxyCollisionContact, ...]
    unsupported_tracks: tuple[tuple[str, str], ...]
    unevaluable_track_ids: tuple[str, ...]
    minimum_center_distance_m: float | None
    minimum_center_distance_track_id: str | None
    minimum_center_distance_frame_index: int | None

    @property
    def has_overlap(self) -> bool:
        return bool(self.contacts)


def _point(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite point with shape (2,)")
    return result


def _axes(heading_rad: float, name: str) -> tuple[np.ndarray, np.ndarray]:
    heading = float(heading_rad)
    if not math.isfinite(heading):
        raise ValueError(f"{name} must be finite")
    forward = np.array([math.cos(heading), math.sin(heading)], dtype=np.float64)
    lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
    return forward, lateral


def oriented_boxes_overlap(
    first_center: np.ndarray,
    first_heading_rad: float,
    first_proxy: FootprintProxy,
    second_center: np.ndarray,
    second_heading_rad: float,
    second_proxy: FootprintProxy,
) -> bool:
    """Return SAT overlap for two class-proxy rectangles; boundary contact overlaps."""

    first = _point(first_center, "first_center")
    second = _point(second_center, "second_center")
    first_forward, first_lateral = _axes(first_heading_rad, "first_heading_rad")
    second_forward, second_lateral = _axes(second_heading_rad, "second_heading_rad")
    delta = second - first
    first_half = (first_proxy.length_m / 2.0, first_proxy.width_m / 2.0)
    second_half = (second_proxy.length_m / 2.0, second_proxy.width_m / 2.0)

    for axis in (first_forward, first_lateral, second_forward, second_lateral):
        first_radius = first_half[0] * abs(float(np.dot(first_forward, axis))) + first_half[
            1
        ] * abs(float(np.dot(first_lateral, axis)))
        second_radius = second_half[0] * abs(
            float(np.dot(second_forward, axis))
        ) + second_half[1] * abs(float(np.dot(second_lateral, axis)))
        if abs(float(np.dot(delta, axis))) > first_radius + second_radius:
            return False
    return True


def detect_synchronized_proxy_collisions(
    scenario: Scenario,
    target_track_id: str,
    config: CounterfactualFilterConfig,
) -> ProxyCollisionReport:
    """Find the first synchronized proxy-box overlap with each background track."""

    if config.footprint_ground_truth:
        raise ValueError("class footprint proxies must not be marked as ground truth")
    agents = {agent.track_id: agent for agent in scenario.agents}
    if target_track_id not in agents:
        raise ValueError(f"target track is not present in scenario: {target_track_id}")
    target = agents[target_track_id]
    if len(target.positions) < TOTAL_STEPS:
        raise ValueError("target track must contain at least 110 frames")

    proxies = config.footprints_by_type
    target_proxy = proxies.get(target.object_type.lower())
    if target_proxy is None:
        return ProxyCollisionReport(
            proxy_source=config.footprint_source,
            ground_truth=False,
            target_proxy_available=False,
            contacts=(),
            unsupported_tracks=((target.track_id, target.object_type.lower()),),
            unevaluable_track_ids=(),
            minimum_center_distance_m=None,
            minimum_center_distance_track_id=None,
            minimum_center_distance_frame_index=None,
        )

    target_positions = target.positions[HISTORY_STEPS:TOTAL_STEPS]
    target_headings = target.headings[HISTORY_STEPS:TOTAL_STEPS]
    if not (
        np.isfinite(target_positions).all() and np.isfinite(target_headings).all()
    ):
        raise ValueError("target future positions and headings must be finite")

    contacts: list[ProxyCollisionContact] = []
    unsupported: list[tuple[str, str]] = []
    unevaluable: list[str] = []
    minimum_center: tuple[float, str, int] | None = None
    for other in sorted(scenario.agents, key=lambda item: item.track_id):
        if other.track_id == target_track_id:
            continue
        end = min(TOTAL_STEPS, len(other.positions))
        if end <= HISTORY_STEPS:
            unevaluable.append(other.track_id)
            continue
        valid = (
            np.isfinite(other.positions[HISTORY_STEPS:end]).all(axis=1)
            & np.isfinite(other.headings[HISTORY_STEPS:end])
        )
        if not valid.any():
            unevaluable.append(other.track_id)
            continue
        other_type = other.object_type.lower()
        other_proxy = proxies.get(other_type)
        if other_proxy is None:
            unsupported.append((other.track_id, other_type))
            continue
        for local_index in np.flatnonzero(valid):
            frame_index = HISTORY_STEPS + int(local_index)
            center_distance = float(
                np.linalg.norm(
                    target.positions[frame_index] - other.positions[frame_index]
                )
            )
            candidate_center = (center_distance, other.track_id, frame_index)
            if minimum_center is None or candidate_center < minimum_center:
                minimum_center = candidate_center
            if oriented_boxes_overlap(
                target.positions[frame_index],
                float(target.headings[frame_index]),
                target_proxy,
                other.positions[frame_index],
                float(other.headings[frame_index]),
                other_proxy,
            ):
                contacts.append(
                    ProxyCollisionContact(
                        other_track_id=other.track_id,
                        other_object_type=other_type,
                        frame_index=frame_index,
                    )
                )
                break

    return ProxyCollisionReport(
        proxy_source=config.footprint_source,
        ground_truth=False,
        target_proxy_available=True,
        contacts=tuple(contacts),
        unsupported_tracks=tuple(unsupported),
        unevaluable_track_ids=tuple(unevaluable),
        minimum_center_distance_m=(None if minimum_center is None else minimum_center[0]),
        minimum_center_distance_track_id=(
            None if minimum_center is None else minimum_center[1]
        ),
        minimum_center_distance_frame_index=(
            None if minimum_center is None else minimum_center[2]
        ),
    )


def check_proxy_collisions(
    scenario: Scenario,
    target_track_id: str,
    config: CounterfactualFilterConfig,
) -> FilterCheck:
    report = detect_synchronized_proxy_collisions(scenario, target_track_id, config)
    reasons: list[FilterRejection] = []
    if not report.target_proxy_available or report.unsupported_tracks:
        reasons.append(FilterRejection.COLLISION_PROXY_UNAVAILABLE)
    if report.has_overlap:
        reasons.append(FilterRejection.COLLISION_PROXY_OVERLAP)
    return FilterCheck(
        stage=FilterStage.COLLISION,
        rejection_reasons=tuple(reasons),
        metrics={
            "geometry": "class_proxy_oriented_box_sat",
            "proxy_source": report.proxy_source,
            "ground_truth": report.ground_truth,
            "contacts": [
                {
                    "other_track_id": item.other_track_id,
                    "other_object_type": item.other_object_type,
                    "frame_index": item.frame_index,
                }
                for item in report.contacts
            ],
            "unsupported_tracks": [list(item) for item in report.unsupported_tracks],
            "unevaluable_track_ids": list(report.unevaluable_track_ids),
            "minimum_center_distance_m": report.minimum_center_distance_m,
            "minimum_center_distance_track_id": report.minimum_center_distance_track_id,
            "minimum_center_distance_frame_index": report.minimum_center_distance_frame_index,
        },
    )


__all__ = [
    "ProxyCollisionContact",
    "ProxyCollisionReport",
    "check_proxy_collisions",
    "detect_synchronized_proxy_collisions",
    "oriented_boxes_overlap",
]

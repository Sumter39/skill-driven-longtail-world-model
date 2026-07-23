from __future__ import annotations

import math

import numpy as np

from skilldrive.filtering.collision import (
    check_proxy_collisions,
    detect_synchronized_proxy_collisions,
    oriented_boxes_overlap,
)
from skilldrive.filtering.contracts import FilterRejection
from skilldrive.generation.config import load_filter_config
from skilldrive.schemas import AgentTrack, Scenario


def _agent(
    track_id: str,
    object_type: str,
    future_x: float,
    *,
    heading: float = 0.0,
) -> AgentTrack:
    positions = np.zeros((110, 2), dtype=np.float64)
    positions[:, 0] = future_x
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=positions,
        velocities=np.zeros((110, 2)),
        headings=np.full(110, heading),
        observed_mask=observed,
        is_focal=track_id == "target",
    )


def _scenario(other: AgentTrack) -> Scenario:
    return Scenario(
        scenario_id="collision",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="target",
        agents=[_agent("target", "vehicle", 0.0), other],
        map_polylines=[],
    )


def test_oriented_box_sat_treats_boundary_contact_as_proxy_overlap() -> None:
    config = load_filter_config()
    vehicle = config.footprints_by_type["vehicle"]

    assert oriented_boxes_overlap([0.0, 0.0], 0.0, vehicle, [4.8, 0.0], 0.0, vehicle)
    assert not oriented_boxes_overlap(
        [0.0, 0.0], 0.0, vehicle, [4.8001, 0.0], 0.0, vehicle
    )
    assert oriented_boxes_overlap(
        [0.0, 0.0], 0.0, vehicle, [3.3, 0.0], math.pi / 2, vehicle
    )
    assert not oriented_boxes_overlap(
        [0.0, 0.0], 0.0, vehicle, [3.5, 0.0], math.pi / 2, vehicle
    )


def test_synchronized_collision_report_is_explicitly_a_proxy() -> None:
    config = load_filter_config()
    scenario = _scenario(_agent("other", "vehicle", 1.0))
    report = detect_synchronized_proxy_collisions(scenario, "target", config)

    assert report.ground_truth is False
    assert report.proxy_source == config.footprint_source
    assert report.contacts[0].other_track_id == "other"
    assert report.contacts[0].frame_index == 50
    assert report.minimum_center_distance_m == 1.0
    assert report.minimum_center_distance_track_id == "other"
    result = check_proxy_collisions(scenario, "target", config)
    assert result.rejection_reasons == (FilterRejection.COLLISION_PROXY_OVERLAP,)
    assert result.metrics["geometry"] == "class_proxy_oriented_box_sat"

    far = check_proxy_collisions(
        _scenario(_agent("other", "vehicle", 10.0)), "target", config
    )
    assert far.passed


def test_missing_class_proxy_is_not_silently_treated_as_collision_free() -> None:
    config = load_filter_config()
    result = check_proxy_collisions(
        _scenario(_agent("unsupported", "alien", 1.0)), "target", config
    )
    assert result.rejection_reasons == (
        FilterRejection.COLLISION_PROXY_UNAVAILABLE,
    )
    assert result.metrics["unsupported_tracks"] == [["unsupported", "alien"]]


def test_all_official_av2_object_types_have_explicit_collision_proxies() -> None:
    config = load_filter_config()

    assert set(config.footprints_by_type) == {
        "background",
        "bus",
        "construction",
        "cyclist",
        "motorcyclist",
        "pedestrian",
        "riderless_bicycle",
        "static",
        "unknown",
        "vehicle",
    }
    assert config.footprints_by_type["background"].length_m == 4.8
    assert config.footprints_by_type["riderless_bicycle"].width_m == 0.8

from __future__ import annotations

import numpy as np

from skilldrive.filtering.contracts import FilterRejection
from skilldrive.filtering.observed import validate_observed_skill
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.skills.detection import detect_scenario, load_detection_config
from skilldrive.skills.loader import load_skill


def _moving_agent(
    track_id: str,
    initial_x: float,
    speed_mps: float,
    *,
    focal: bool = False,
) -> AgentTrack:
    time = np.arange(110, dtype=np.float64) * 0.1
    positions = np.column_stack(
        (initial_x + speed_mps * time, np.zeros(110, dtype=np.float64))
    )
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    return AgentTrack(
        track_id=track_id,
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([speed_mps, 0.0], (110, 1)),
        headings=np.zeros(110),
        observed_mask=observed,
        is_focal=focal,
    )


def _map() -> list[MapPolyline]:
    return [
        MapPolyline(
            polyline_id="main:center",
            polyline_type="lane_centerline",
            points=np.array([[-100.0, 0.0], [300.0, 0.0]]),
            direction="vehicle",
            lane_id="main",
            right_neighbor_id="adjacent",
        ),
        MapPolyline(
            polyline_id="adjacent:center",
            polyline_type="lane_centerline",
            points=np.array([[-100.0, 4.0], [300.0, 4.0]]),
            direction="vehicle",
            lane_id="adjacent",
            left_neighbor_id="main",
        ),
    ]


def _scenario(agents: list[AgentTrack]) -> Scenario:
    agents[0].is_focal = True
    return Scenario(
        scenario_id="observed",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id=agents[0].track_id,
        agents=agents,
        map_polylines=_map(),
    )


def test_observed_validator_accepts_only_the_exact_requested_roles() -> None:
    skill = load_skill("configs/skills/slow_lead_blockage.yaml")
    config = load_detection_config("configs/seed_detection.yaml")
    scenario = _scenario(
        [
            _moving_agent("slow", 20.0, 1.0),
            _moving_agent("follower", 0.0, 2.0),
        ]
    )
    result = validate_observed_skill(
        scenario,
        skill,
        {"slow_leader": "slow", "follower": "follower"},
        config,
    )

    assert result.passed
    assert result.metrics["risk_metric"] == "minimum_longitudinal_gap"
    assert result.metrics["requested_role_track_ids"] == {
        "slow_leader": "slow",
        "follower": "follower",
    }


def test_other_scene_pair_cannot_masquerade_as_requested_roles() -> None:
    skill = load_skill("configs/skills/slow_lead_blockage.yaml")
    config = load_detection_config("configs/seed_detection.yaml")
    scenario = _scenario(
        [
            _moving_agent("bad-leader", 20.0, 4.0),
            _moving_agent("bad-follower", 0.0, 5.0),
            _moving_agent("real-slow", 70.0, 1.0),
            _moving_agent("real-follower", 50.0, 2.0),
        ]
    )
    assert detect_scenario(scenario, [skill], config).records

    result = validate_observed_skill(
        scenario,
        skill,
        {"slow_leader": "bad-leader", "follower": "bad-follower"},
        config,
    )
    assert result.rejection_reasons == (
        FilterRejection.OBSERVED_SKILL_NOT_REDETECTED,
    )


def test_observed_validator_rejects_role_contract_drift_and_accepts_untrained_observed_mode() -> None:
    config = load_detection_config("configs/seed_detection.yaml")
    scenario = _scenario(
        [
            _moving_agent("slow", 20.0, 1.0),
            _moving_agent("follower", 0.0, 2.0),
        ]
    )
    observed = load_skill("configs/skills/slow_lead_blockage.yaml")
    result = validate_observed_skill(
        scenario,
        observed,
        {"slow_leader": "slow"},
        config,
    )
    assert result.rejection_reasons == (
        FilterRejection.OBSERVED_ROLE_CONTRACT_MISMATCH,
    )

    untrained = load_skill("configs/skills/group_pedestrian_crossing.yaml")
    result = validate_observed_skill(scenario, untrained, {}, config)
    assert result.rejection_reasons == (
        FilterRejection.OBSERVED_ROLE_CONTRACT_MISMATCH,
    )

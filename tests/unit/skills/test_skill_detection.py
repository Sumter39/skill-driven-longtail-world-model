from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import skilldrive.skills.detection as detection
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.seeds import sample_skill_parameters
from skilldrive.skills import load_skill
from skilldrive.skills.detection import (
    DetectionConfig,
    RuleMatch,
    ScenarioDetectionContext,
    detect_scenario,
    load_detection_config,
)
from skilldrive.skills.registry import DETECTION_STRATEGIES, get_skill_detection_rule


SKILL_DIR = Path("configs/skills")
CONFIG_PATH = Path("configs/seed_detection.yaml")
SAMPLE_PERIOD_S = 0.1

STRATEGY_SKILLS = (
    ("blockage_avoidance", "forced_lane_change_around_blockage"),
    ("conflict_point_pair", "crossing_path_conflict"),
    ("cut_in_then_brake", "cut_in_then_brake"),
    ("diverge_crossing", "diverge_lane_crossing_conflict"),
    ("intersection_occupancy", "intersection_blocking_vehicle"),
    ("lane_change_gap", "narrow_gap_lane_change"),
    ("lane_change_pair", "adjacent_vehicle_cut_in"),
    ("longitudinal_pair", "lead_hard_brake"),
    ("merge_pair", "ramp_merge_small_gap"),
    ("simultaneous_lane_change", "simultaneous_lane_change_conflict"),
    ("static_blockage", "static_object_avoidance"),
    ("stopped_reentry", "stopped_vehicle_reentry"),
    ("three_vehicle_reveal", "cut_out_reveals_slow_vehicle"),
    ("vru_vehicle_conflict", "crosswalk_pedestrian_crossing"),
    ("wrong_way_pair", "wrong_way_vehicle"),
)


@pytest.fixture(scope="module")
def config() -> DetectionConfig:
    return load_detection_config(CONFIG_PATH)


def _skill(skill_id: str):
    return load_skill(SKILL_DIR / f"{skill_id}.yaml")


def _constant_agent(
    track_id: str,
    object_type: str,
    *,
    initial_x: float = 0.0,
    y: float = 0.0,
    speed_mps: float = 5.0,
    heading_rad: float = 0.0,
    steps: int = 40,
    observed_steps: int | None = None,
) -> AgentTrack:
    times = np.arange(steps, dtype=np.float64) * SAMPLE_PERIOD_S
    positions = np.column_stack(
        (
            initial_x + speed_mps * times * math.cos(heading_rad),
            y + speed_mps * times * math.sin(heading_rad),
        )
    )
    velocities = np.tile(
        [speed_mps * math.cos(heading_rad), speed_mps * math.sin(heading_rad)],
        (steps, 1),
    )
    observed_mask = np.arange(steps) < (steps if observed_steps is None else observed_steps)
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=positions,
        velocities=velocities,
        headings=np.full(steps, heading_rad),
        observed_mask=observed_mask,
    )


def _agent_from_speeds(
    track_id: str,
    speeds: np.ndarray,
    *,
    initial_x: float,
    observed_steps: int,
) -> AgentTrack:
    speeds = np.asarray(speeds, dtype=np.float64)
    positions = np.zeros((len(speeds), 2), dtype=np.float64)
    positions[0, 0] = initial_x
    for index in range(1, len(speeds)):
        positions[index, 0] = (
            positions[index - 1, 0] + speeds[index - 1] * SAMPLE_PERIOD_S
        )
    return AgentTrack(
        track_id=track_id,
        object_type="vehicle",
        positions=positions,
        velocities=np.column_stack((speeds, np.zeros(len(speeds)))),
        headings=np.zeros(len(speeds)),
        observed_mask=np.arange(len(speeds)) < observed_steps,
    )


def _agent_from_positions(
    track_id: str,
    object_type: str,
    positions: np.ndarray,
    *,
    observed_steps: int,
    headings: float | np.ndarray | None = None,
) -> AgentTrack:
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.gradient(positions, SAMPLE_PERIOD_S, axis=0)
    if headings is None:
        heading_values = np.arctan2(velocities[:, 1], velocities[:, 0])
    elif np.isscalar(headings):
        heading_values = np.full(len(positions), float(headings))
    else:
        heading_values = np.asarray(headings, dtype=np.float64)
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=positions,
        velocities=velocities,
        headings=heading_values,
        observed_mask=np.arange(len(positions)) < observed_steps,
    )


def _constant_agent_at_reference(
    track_id: str,
    object_type: str,
    *,
    reference_x: float,
    reference_y: float = 0.0,
    speed_mps: float = 5.0,
    heading_rad: float = 0.0,
    steps: int = 60,
    observed_steps: int = 30,
) -> AgentTrack:
    elapsed = (observed_steps - 1) * SAMPLE_PERIOD_S
    return _constant_agent(
        track_id,
        object_type,
        initial_x=reference_x - speed_mps * elapsed * math.cos(heading_rad),
        y=reference_y - speed_mps * elapsed * math.sin(heading_rad),
        speed_mps=speed_mps,
        heading_rad=heading_rad,
        steps=steps,
        observed_steps=observed_steps,
    )


def _full_map() -> list[MapPolyline]:
    main_points = np.column_stack((np.linspace(-100.0, 100.0, 81), np.zeros(81)))
    adjacent_points = np.column_stack(
        (np.linspace(-100.0, 100.0, 81), np.full(81, 4.0))
    )
    merge_points = np.column_stack((np.linspace(100.0, 160.0, 31), np.zeros(31)))
    out_left = np.column_stack(
        (np.linspace(160.0, 220.0, 31), np.linspace(0.0, 10.0, 31))
    )
    out_right = np.column_stack(
        (np.linspace(160.0, 220.0, 31), np.linspace(0.0, -10.0, 31))
    )
    bike_points = np.column_stack((np.linspace(-100.0, 100.0, 81), np.full(81, 8.0)))
    return [
        MapPolyline(
            "main:center",
            "lane_centerline",
            main_points,
            direction="vehicle",
            is_intersection=True,
            lane_id="main",
            successor_ids=["merge"],
            left_neighbor_id="adjacent",
        ),
        MapPolyline(
            "adjacent:center",
            "lane_centerline",
            adjacent_points,
            direction="vehicle",
            lane_id="adjacent",
            successor_ids=["merge"],
            right_neighbor_id="main",
        ),
        MapPolyline(
            "merge:center",
            "lane_centerline",
            merge_points,
            direction="vehicle",
            lane_id="merge",
            predecessor_ids=["main", "adjacent", "bike"],
            successor_ids=["out-left", "out-right"],
        ),
        MapPolyline(
            "out-left:center",
            "lane_centerline",
            out_left,
            direction="vehicle",
            lane_id="out-left",
            predecessor_ids=["merge"],
        ),
        MapPolyline(
            "out-right:center",
            "lane_centerline",
            out_right,
            direction="vehicle",
            lane_id="out-right",
            predecessor_ids=["merge"],
        ),
        MapPolyline(
            "bike:center",
            "lane_centerline",
            bike_points,
            direction="bike",
            lane_id="bike",
            successor_ids=["merge"],
        ),
        MapPolyline(
            "crosswalk:1",
            "pedestrian_crossing",
            np.array([[20.0, -5.0], [25.0, -5.0], [25.0, 10.0], [20.0, 10.0], [20.0, -5.0]]),
        ),
        MapPolyline(
            "drivable:1",
            "drivable_area",
            np.array(
                [
                    [-120.0, -20.0],
                    [230.0, -20.0],
                    [230.0, 20.0],
                    [-120.0, 20.0],
                    [-120.0, -20.0],
                ]
            ),
        ),
    ]


def _single_successor_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "main:center",
            "lane_centerline",
            np.column_stack((np.linspace(-50.0, 50.0, 41), np.zeros(41))),
            direction="vehicle",
            lane_id="main",
            successor_ids=["next"],
        ),
        MapPolyline(
            "next:center",
            "lane_centerline",
            np.column_stack((np.linspace(50.0, 100.0, 21), np.zeros(21))),
            direction="vehicle",
            lane_id="next",
            predecessor_ids=["main"],
        ),
    ]


def _parallel_lane_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "main:center",
            "lane_centerline",
            np.array([[-100.0, 0.0], [100.0, 0.0]]),
            direction="vehicle",
            lane_id="main",
            successor_ids=["main-next"],
            left_neighbor_id="adjacent",
        ),
        MapPolyline(
            "main-next:center",
            "lane_centerline",
            np.array([[100.0, 0.0], [160.0, 0.0]]),
            direction="vehicle",
            lane_id="main-next",
            predecessor_ids=["main"],
        ),
        MapPolyline(
            "adjacent:center",
            "lane_centerline",
            np.array([[-100.0, 4.0], [100.0, 4.0]]),
            direction="vehicle",
            lane_id="adjacent",
            right_neighbor_id="main",
        ),
        MapPolyline(
            "drivable:parallel",
            "drivable_area",
            np.array(
                [
                    [-120.0, -8.0],
                    [180.0, -8.0],
                    [180.0, 12.0],
                    [-120.0, 12.0],
                    [-120.0, -8.0],
                ]
            ),
        ),
    ]


def _pair_merge_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "main-source:center",
            "lane_centerline",
            np.array([[-20.0, 0.0], [30.0, 0.0]]),
            direction="vehicle",
            lane_id="main-source",
            successor_ids=["merge-target"],
        ),
        MapPolyline(
            "merge-source:center",
            "lane_centerline",
            np.array([[-20.0, 6.0], [30.0, 0.0]]),
            direction="vehicle",
            lane_id="merge-source",
            successor_ids=["merge-target"],
        ),
        MapPolyline(
            "merge-target:center",
            "lane_centerline",
            np.array([[30.0, 0.0], [90.0, 0.0]]),
            direction="vehicle",
            lane_id="merge-target",
            predecessor_ids=["main-source", "merge-source"],
        ),
    ]


def _symmetric_pair_merge_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "a-source:center",
            "lane_centerline",
            np.array([[-20.0, 4.0], [30.0, 0.0]]),
            direction="vehicle",
            lane_id="a-source",
            successor_ids=["merge-target"],
        ),
        MapPolyline(
            "b-source:center",
            "lane_centerline",
            np.array([[-20.0, -4.0], [30.0, 0.0]]),
            direction="vehicle",
            lane_id="b-source",
            successor_ids=["merge-target"],
        ),
        MapPolyline(
            "merge-target:center",
            "lane_centerline",
            np.array([[30.0, 0.0], [90.0, 0.0]]),
            direction="vehicle",
            lane_id="merge-target",
            predecessor_ids=["a-source", "b-source"],
        ),
    ]


def _serial_pair_with_unrelated_merge_map() -> list[MapPolyline]:
    return [
        *_pair_merge_map(),
        MapPolyline(
            "serial-a:center",
            "lane_centerline",
            np.array([[-20.0, 20.0], [20.0, 20.0]]),
            direction="vehicle",
            lane_id="serial-a",
            successor_ids=["serial-b"],
        ),
        MapPolyline(
            "serial-b:center",
            "lane_centerline",
            np.array([[20.0, 20.0], [70.0, 20.0]]),
            direction="vehicle",
            lane_id="serial-b",
            predecessor_ids=["serial-a"],
        ),
    ]


def _diverge_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "diverge:center",
            "lane_centerline",
            np.array([[-30.0, 0.0], [30.0, 0.0]]),
            direction="vehicle",
            lane_id="diverge",
            successor_ids=["branch-left", "branch-right"],
            left_neighbor_id="through",
        ),
        MapPolyline(
            "through:center",
            "lane_centerline",
            np.array([[-30.0, 4.0], [70.0, 4.0]]),
            direction="vehicle",
            lane_id="through",
            right_neighbor_id="diverge",
        ),
        MapPolyline(
            "branch-left:center",
            "lane_centerline",
            np.array([[30.0, 0.0], [70.0, 8.0]]),
            direction="vehicle",
            lane_id="branch-left",
            predecessor_ids=["diverge"],
        ),
        MapPolyline(
            "branch-right:center",
            "lane_centerline",
            np.array([[30.0, 0.0], [70.0, -8.0]]),
            direction="vehicle",
            lane_id="branch-right",
            predecessor_ids=["diverge"],
        ),
    ]


def _intersection_map() -> list[MapPolyline]:
    return [
        MapPolyline(
            "east-west:center",
            "lane_centerline",
            np.array([[-30.0, 0.0], [30.0, 0.0]]),
            direction="vehicle",
            is_intersection=True,
            lane_id="east-west",
        ),
        MapPolyline(
            "south-north:center",
            "lane_centerline",
            np.array([[0.0, -30.0], [0.0, 30.0]]),
            direction="vehicle",
            is_intersection=True,
            lane_id="south-north",
        ),
        MapPolyline(
            "drivable:intersection",
            "drivable_area",
            np.array(
                [
                    [-35.0, -35.0],
                    [35.0, -35.0],
                    [35.0, 35.0],
                    [-35.0, 35.0],
                    [-35.0, -35.0],
                ]
            ),
        ),
    ]


def _shared_target_lane_map(
    *,
    same_side: bool = False,
    source_offset_m: float = 4.0,
) -> list[MapPolyline]:
    second_y = 2 * source_offset_m if same_side else -source_offset_m
    return [
        MapPolyline(
            "first-source:center",
            "lane_centerline",
            np.array([[-50.0, source_offset_m], [50.0, source_offset_m]]),
            direction="vehicle",
            lane_id="first-source",
            right_neighbor_id="shared-target",
        ),
        MapPolyline(
            "second-source:center",
            "lane_centerline",
            np.array([[-50.0, second_y], [50.0, second_y]]),
            direction="vehicle",
            lane_id="second-source",
            right_neighbor_id="shared-target" if same_side else None,
            left_neighbor_id=None if same_side else "shared-target",
        ),
        MapPolyline(
            "shared-target:center",
            "lane_centerline",
            np.array([[-50.0, 0.0], [50.0, 0.0]]),
            direction="vehicle",
            lane_id="shared-target",
            left_neighbor_id="first-source",
            right_neighbor_id="second-source",
        ),
    ]


def _jaywalking_map(*, crosswalk_near_conflict: bool = False) -> list[MapPolyline]:
    crossing_x = 0.0 if crosswalk_near_conflict else 30.0
    return [
        MapPolyline(
            "crosswalk:test",
            "pedestrian_crossing",
            np.array(
                [
                    [crossing_x - 2.0, -2.0],
                    [crossing_x + 2.0, -2.0],
                    [crossing_x + 2.0, 2.0],
                    [crossing_x - 2.0, 2.0],
                    [crossing_x - 2.0, -2.0],
                ]
            ),
        ),
        MapPolyline(
            "drivable:jaywalking",
            "drivable_area",
            np.array(
                [
                    [-20.0, -20.0],
                    [20.0, -20.0],
                    [20.0, 20.0],
                    [-20.0, 20.0],
                    [-20.0, -20.0],
                ]
            ),
        ),
    ]


def _scenario(
    agents: list[AgentTrack],
    *,
    map_polylines: list[MapPolyline] | None = None,
    scenario_id: str = "synthetic-detection",
) -> Scenario:
    agents[0].is_focal = True
    steps = len(agents[0].positions)
    assert all(len(agent.positions) == steps for agent in agents)
    return Scenario(
        scenario_id=scenario_id,
        city_name="synthetic-city",
        timestamps=np.arange(steps, dtype=np.int64) * 100_000_000,
        focal_track_id=agents[0].track_id,
        agents=agents,
        map_polylines=[] if map_polylines is None else map_polylines,
        metadata={"source_path": f"train/{scenario_id}/scenario_{scenario_id}.parquet"},
    )


def _write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "seed_detection.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _assert_evidence_contract(record, skill) -> None:
    evidence = record.evidence
    required_fields = {"detection_mode", "matched_conditions", "threshold_evidence"}
    assert required_fields <= set(evidence), (
        f"missing evidence fields: {sorted(required_fields - set(evidence))}"
    )
    assert evidence["detection_mode"] == skill.detection["mode"]
    assert evidence["matched_conditions"] == skill.detection["conditions"]
    threshold_evidence = evidence["threshold_evidence"]
    assert set(threshold_evidence) == set(skill.detection["thresholds"])
    for name, specification in skill.detection["thresholds"].items():
        check = threshold_evidence[name]
        assert check["threshold_value"] == pytest.approx(specification["value"])
        assert math.isfinite(float(check["measured_value"]))
        assert isinstance(check["comparison"], str) and check["comparison"]
        assert check["passed"] is True


def test_load_detection_config_uses_only_engine_thresholds(config) -> None:
    assert config.global_seed == 2026
    assert config.max_candidates_per_skill_per_scenario == 1
    assert set(config.thresholds) == {
        "maximum_actor_distance_m",
        "lane_match_distance_m",
        "lane_heading_tolerance_deg",
        "same_lane_lateral_tolerance_m",
        "conflict_distance_m",
        "risk_time_horizon_s",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(version=2), "version must be 1"),
        (
            lambda data: data["thresholds"].update(
                moving_speed_mps={"value": 2.0, "source": "semantic"}
            ),
            "thresholds differ",
        ),
        (
            lambda data: data["thresholds"]["conflict_distance_m"].update(
                source="guessed"
            ),
            "unknown source",
        ),
    ],
)
def test_load_detection_config_rejects_contract_drift(
    tmp_path, mutation, message
) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mutation(data)
    path = _write_config(tmp_path, data)

    with pytest.raises(ValueError, match=message):
        load_detection_config(path)


def test_missing_required_track_is_rejected_before_rule_execution(config) -> None:
    skill = _skill("crosswalk_pedestrian_crossing")
    scenario = _scenario(
        [
            _constant_agent("ego", "vehicle"),
            _constant_agent("other", "vehicle", initial_x=15.0),
        ],
        map_polylines=_full_map(),
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts == {
        "crosswalk_pedestrian_crossing:missing_track:pedestrian": 1
    }


@pytest.mark.parametrize(
    ("skill_id", "substitute_type", "missing_type"),
    [
        ("bike_lane_vehicle_merge_conflict", "motorcyclist", "cyclist"),
        ("construction_object_lane_blockage", "static", "construction"),
        ("static_object_avoidance", "construction", "static"),
    ],
)
def test_required_track_types_are_exact_not_broadened(
    config, skill_id, substitute_type, missing_type
) -> None:
    skill = _skill(skill_id)
    scenario = _scenario(
        [
            _constant_agent("ego", "vehicle"),
            _constant_agent("substitute", substitute_type, initial_x=10.0),
        ],
        map_polylines=_full_map(),
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts == {f"{skill_id}:missing_track:{missing_type}": 1}


def test_missing_direct_map_layers_are_reported_individually(config) -> None:
    skill = _skill("lead_hard_brake")
    scenario = _scenario(
        [
            _constant_agent("leader", "vehicle", initial_x=20.0),
            _constant_agent("follower", "vehicle"),
        ]
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts == {
        "lead_hard_brake:missing_map:lane_centerline": 1,
        "lead_hard_brake:missing_map:lane_successor": 1,
    }


def test_missing_derived_converging_topology_is_rejected(config) -> None:
    skill = _skill("ramp_merge_small_gap")
    scenario = _scenario(
        [
            _constant_agent("first", "vehicle"),
            _constant_agent("second", "vehicle", initial_x=10.0),
        ],
        map_polylines=_single_successor_map(),
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts == {
        "ramp_merge_small_gap:missing_map:converging_lane": 1
    }


@pytest.mark.parametrize(
    ("pedestrian_history", "vehicle_history", "expected_reason"),
    [
        (19, 25, "missing_history_qualified_initiator"),
        (25, 19, "missing_history_qualified_responder"),
    ],
)
def test_history_requirement_rejects_each_actor_role(
    config, pedestrian_history, vehicle_history, expected_reason
) -> None:
    skill = _skill("crosswalk_pedestrian_crossing")
    scenario = _scenario(
        [
            _constant_agent(
                "pedestrian",
                "pedestrian",
                initial_x=20.0,
                observed_steps=pedestrian_history,
            ),
            _constant_agent(
                "vehicle",
                "vehicle",
                observed_steps=vehicle_history,
            ),
        ],
        map_polylines=_full_map(),
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts == {
        f"crosswalk_pedestrian_crossing:{expected_reason}": 1,
    }


def test_strategy_matrix_covers_every_registered_handler() -> None:
    assert {strategy for strategy, _ in STRATEGY_SKILLS} == DETECTION_STRATEGIES
    assert set(detection._STRATEGY_HANDLERS) == DETECTION_STRATEGIES


@pytest.mark.parametrize(
    ("strategy", "skill_id"),
    STRATEGY_SKILLS,
    ids=[strategy for strategy, _ in STRATEGY_SKILLS],
)
def test_strategy_has_executable_no_actor_negative(
    config, strategy, skill_id
) -> None:
    context = ScenarioDetectionContext(
        _scenario([_constant_agent("ego", "vehicle")], map_polylines=_full_map()),
        config,
    )
    skill = _skill(skill_id)
    rule = get_skill_detection_rule(skill_id)
    assert rule.strategy == strategy
    assert detection._STRATEGY_HANDLERS[strategy](context, skill, (), ()) == []


def _drivable_area_context(
    config,
    *,
    positions: np.ndarray,
    polygons: list[np.ndarray],
    scenario_id: str,
) -> tuple[ScenarioDetectionContext, detection.ActorState]:
    positions = np.asarray(positions, dtype=np.float64)
    pedestrian = AgentTrack(
        track_id="pedestrian",
        object_type="pedestrian",
        positions=positions,
        velocities=np.zeros_like(positions),
        headings=np.zeros(len(positions), dtype=np.float64),
        observed_mask=np.arange(len(positions)) < 2,
    )
    context = ScenarioDetectionContext(
        _scenario(
            [pedestrian],
            map_polylines=[
                MapPolyline(
                    f"drivable:test-{index}",
                    "drivable_area",
                    polygon,
                )
                for index, polygon in enumerate(polygons)
            ],
            scenario_id=scenario_id,
        ),
        config,
    )
    return context, context.state_by_id[pedestrian.track_id]


def test_future_drivable_entry_is_computed_once_per_track(
    config,
    monkeypatch,
) -> None:
    context, pedestrian = _drivable_area_context(
        config,
        positions=np.array(
            [
                [-2.0, 0.5],
                [-1.0, 0.5],
                [0.5, 0.5],
                [2.0, 0.5],
            ]
        ),
        polygons=[
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )
        ],
        scenario_id="future-drivable-cache",
    )
    original = context.point_inside_drivable_area
    queried_points: list[tuple[float, float]] = []

    def counted(point: np.ndarray) -> bool:
        queried_points.append((float(point[0]), float(point[1])))
        return original(point)

    monkeypatch.setattr(context, "point_inside_drivable_area", counted)

    assert detection._future_enters_drivable_area(context, pedestrian) is True
    first_query_count = len(queried_points)
    assert first_query_count == 2
    assert detection._future_enters_drivable_area(context, pedestrian) is True
    assert len(queried_points) == first_query_count


def test_future_drivable_entry_cache_is_isolated_by_context(config) -> None:
    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    entering_context, entering_pedestrian = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], [-1.0, 0.5], [0.5, 0.5]]),
        polygons=[polygon],
        scenario_id="future-drivable-entering",
    )
    outside_context, outside_pedestrian = _drivable_area_context(
        config,
        positions=np.array([[-3.0, 2.0], [-2.0, 2.0], [-1.0, 2.0]]),
        polygons=[polygon],
        scenario_id="future-drivable-outside",
    )

    assert (
        detection._future_enters_drivable_area(
            entering_context,
            entering_pedestrian,
        )
        is True
    )
    assert (
        detection._future_enters_drivable_area(
            outside_context,
            outside_pedestrian,
        )
        is False
    )


@pytest.mark.parametrize(
    ("future_points", "expected"),
    [
        ([[-1.0, 0.5], [math.nan, math.nan], [0.5, 0.5]], True),
        ([[-1.0, 0.5], [math.nan, math.nan], [2.0, 0.5]], False),
    ],
)
def test_future_drivable_entry_ignores_nan_future_points(
    config,
    future_points,
    expected,
) -> None:
    context, pedestrian = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], *future_points]),
        polygons=[
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )
        ],
        scenario_id=f"future-drivable-nan-{expected}",
    )

    assert detection._future_enters_drivable_area(context, pedestrian) is expected


def test_drivable_area_aabb_rejects_far_point_without_polygon_check(
    config,
    monkeypatch,
) -> None:
    context, _ = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], [-1.0, 0.5]]),
        polygons=[
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )
        ],
        scenario_id="drivable-aabb-outside",
    )
    calls = 0
    original = detection._point_in_polygon

    def counted(point: np.ndarray, polygon: np.ndarray) -> bool:
        nonlocal calls
        calls += 1
        return original(point, polygon)

    monkeypatch.setattr(detection, "_point_in_polygon", counted)

    assert context.point_inside_drivable_area(np.array([100.0, 100.0])) is False
    assert calls == 0


def test_drivable_area_aabb_preserves_concave_polygon_semantics(config) -> None:
    concave = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 1.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [0.0, 3.0],
            [0.0, 0.0],
        ]
    )
    context, _ = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], [-1.0, 0.5]]),
        polygons=[concave],
        scenario_id="drivable-concave",
    )

    assert context.point_inside_drivable_area(np.array([0.5, 2.0])) is True
    assert context.point_inside_drivable_area(np.array([2.0, 2.0])) is False


@pytest.mark.parametrize(
    "point",
    [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.5]),
        np.array([0.5, 1.0]),
    ],
)
def test_drivable_area_aabb_preserves_polygon_boundary_semantics(
    config,
    point,
) -> None:
    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    context, _ = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], [-1.0, 0.5]]),
        polygons=[polygon],
        scenario_id=f"drivable-boundary-{point.tolist()}",
    )

    assert context.point_inside_drivable_area(point) is detection._point_in_polygon(
        point,
        polygon,
    )


def test_drivable_area_aabb_preserves_polygon_with_nan_vertex(config) -> None:
    polygon = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [math.nan, math.nan],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ]
    )
    context, _ = _drivable_area_context(
        config,
        positions=np.array([[-2.0, 0.5], [-1.0, 0.5]]),
        polygons=[polygon],
        scenario_id="drivable-nan-polygon",
    )

    assert context.point_inside_drivable_area(np.array([1.0, 1.0])) is True


def _nearest_pair_context(
    config,
    positions: dict[str, tuple[float, float]],
    *,
    scenario_id: str,
) -> ScenarioDetectionContext:
    return ScenarioDetectionContext(
        _scenario(
            [
                _constant_agent_at_reference(
                    track_id,
                    "vehicle",
                    reference_x=x,
                    reference_y=y,
                    speed_mps=0.0,
                )
                for track_id, (x, y) in positions.items()
            ],
            scenario_id=scenario_id,
        ),
        config,
    )


def _pair_track_ids(
    pairs: list[tuple[detection.ActorState, detection.ActorState]],
) -> list[tuple[str, str]]:
    return [(first.track_id, second.track_id) for first, second in pairs]


def test_nearest_pairs_reuses_cached_distances_for_ordered_groups(
    config,
    monkeypatch,
) -> None:
    context = _nearest_pair_context(
        config,
        {
            "a": (0.0, 0.0),
            "b": (3.0, 0.0),
            "c": (8.0, 0.0),
            "d": (100.0, 0.0),
        },
        scenario_id="nearest-pairs-cache",
    )
    first_group = [context.state_by_id[track_id] for track_id in ("a", "b")]
    second_group = [
        context.state_by_id[track_id] for track_id in ("b", "c", "d")
    ]
    original = detection._distance
    distance_calls: list[tuple[str, str]] = []

    def counted(first: detection.ActorState, second: detection.ActorState) -> float:
        distance_calls.append((first.track_id, second.track_id))
        return original(first, second)

    monkeypatch.setattr(detection, "_distance", counted)

    first_result = context.nearest_pairs(first_group, second_group)
    assert _pair_track_ids(first_result) == [("a", "b"), ("b", "c"), ("a", "c")]
    assert len(distance_calls) == 5

    second_result = context.nearest_pairs(first_group, second_group)
    assert _pair_track_ids(second_result) == _pair_track_ids(first_result)
    assert len(distance_calls) == 5


def test_nearest_pairs_treats_reverse_group_direction_as_distinct(config) -> None:
    context = _nearest_pair_context(
        config,
        {
            "a": (0.0, 0.0),
            "b": (3.0, 0.0),
            "c": (8.0, 0.0),
        },
        scenario_id="nearest-pairs-direction",
    )
    first_group = [context.state_by_id[track_id] for track_id in ("a", "b")]
    second_group = [context.state_by_id["c"]]

    assert _pair_track_ids(context.nearest_pairs(first_group, second_group)) == [
        ("b", "c"),
        ("a", "c"),
    ]
    assert _pair_track_ids(context.nearest_pairs(second_group, first_group)) == [
        ("c", "b"),
        ("c", "a"),
    ]


def test_nearest_pairs_cache_is_isolated_by_context(config) -> None:
    near_context = _nearest_pair_context(
        config,
        {"a": (0.0, 0.0), "b": (1.0, 0.0)},
        scenario_id="nearest-pairs-near",
    )
    far_context = _nearest_pair_context(
        config,
        {"a": (0.0, 0.0), "b": (100.0, 0.0)},
        scenario_id="nearest-pairs-far",
    )

    assert _pair_track_ids(
        near_context.nearest_pairs(
            [near_context.state_by_id["a"]],
            [near_context.state_by_id["b"]],
        )
    ) == [("a", "b")]
    assert (
        far_context.nearest_pairs(
            [far_context.state_by_id["a"]],
            [far_context.state_by_id["b"]],
        )
        == []
    )


def test_nearest_pairs_matches_naive_distance_and_track_id_order(config) -> None:
    context = _nearest_pair_context(
        config,
        {
            "first-b": (0.0, 0.0),
            "first-a": (0.0, 0.0),
            "second-b": (3.0, 4.0),
            "second-a": (-3.0, 4.0),
            "far": (80.0, 0.0),
        },
        scenario_id="nearest-pairs-naive-reference",
    )
    first_group = [
        context.state_by_id[track_id]
        for track_id in ("first-b", "first-a", "second-a")
    ]
    second_group = [
        context.state_by_id[track_id]
        for track_id in ("second-b", "second-a", "first-a", "far")
    ]
    maximum = config.threshold("maximum_actor_distance_m")
    expected = sorted(
        [
            (first, second)
            for first in first_group
            for second in second_group
            if first.track_id != second.track_id
            and detection._distance(first, second) <= maximum
        ],
        key=lambda pair: (
            detection._distance(*pair),
            pair[0].track_id,
            pair[1].track_id,
        ),
    )

    assert _pair_track_ids(context.nearest_pairs(first_group, second_group)) == (
        _pair_track_ids(expected)
    )


def test_observed_trigger_record_has_complete_measurement_evidence(config) -> None:
    skill = _skill("lead_hard_brake")
    leader_speeds = np.concatenate(
        (np.full(30, 10.0), np.maximum(10.0 - 0.5 * np.arange(1, 31), 0.0))
    )
    follower_speeds = np.full(60, 12.0)
    scenario = _scenario(
        [
            _agent_from_speeds(
                "leader", leader_speeds, initial_x=20.0, observed_steps=30
            ),
            _agent_from_speeds(
                "follower", follower_speeds, initial_x=0.0, observed_steps=30
            ),
        ],
        map_polylines=_full_map(),
        scenario_id="observed-hard-brake",
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.rejection_counts == {}
    assert len(run.records) == 1
    record = run.records[0]
    assert record.seed_risk_metric == skill.risk_definition["metric"]
    assert math.isfinite(record.seed_risk_value)
    assert record.target_risk_definition == skill.risk_definition
    assert record.seed_risk_is_proxy is False
    _assert_evidence_contract(record, skill)


def test_compatible_seed_uses_structural_conditions_and_three_roles(config) -> None:
    skill = _skill("cut_out_reveals_slow_vehicle")
    scenario = _scenario(
        [
            _constant_agent(
                "cut-out", "vehicle", initial_x=20.0, speed_mps=8.0
            ),
            _constant_agent("target", "vehicle", speed_mps=10.0),
            _constant_agent(
                "slow", "vehicle", initial_x=60.0, speed_mps=1.0
            ),
        ],
        map_polylines=_full_map(),
        scenario_id="compatible-cut-out",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.role_track_ids == {
        "cut_out_vehicle": "cut-out",
        "target_vehicle": "target",
        "slow_vehicle": "slow",
    }
    assert record.target_risk_definition == skill.risk_definition
    assert record.seed_risk_is_proxy is False
    assert record.evidence["newly_exposed_gap_m"] > 0
    assert set(record.evidence["trigger_conditions"]) - set(
        skill.detection["conditions"]
    ) == {"lead_vehicle_cuts_out", "newly_exposed_slow_vehicle"}
    _assert_evidence_contract(record, skill)


def test_three_vehicle_reveal_rejects_nonpositive_exposed_gap(config) -> None:
    skill = _skill("cut_out_reveals_slow_vehicle")
    map_polylines = [
        MapPolyline(
            "target:center",
            "lane_centerline",
            np.array([[-20.0, 0.0], [20.0, 0.0]]),
            direction="vehicle",
            lane_id="target",
            predecessor_ids=["cut-out"],
        ),
        MapPolyline(
            "cut-out:center",
            "lane_centerline",
            np.array([[10.0, -20.0], [10.0, 20.0]]),
            direction="vehicle",
            lane_id="cut-out",
            successor_ids=["target", "slow"],
            left_neighbor_id="adjacent",
        ),
        MapPolyline(
            "slow:center",
            "lane_centerline",
            np.array([[-20.0, 10.0], [20.0, 10.0]]),
            direction="vehicle",
            lane_id="slow",
            predecessor_ids=["cut-out"],
        ),
    ]
    scenario = _scenario(
        [
            _constant_agent(
                "cut-out",
                "vehicle",
                initial_x=10.0,
                speed_mps=0.0,
                heading_rad=math.pi / 2,
            ),
            _constant_agent("target", "vehicle", speed_mps=0.0),
            _constant_agent(
                "slow",
                "vehicle",
                initial_x=-5.0,
                y=10.0,
                speed_mps=0.0,
            ),
        ],
        map_polylines=map_polylines,
        scenario_id="wrong-three-vehicle-order",
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []


@pytest.mark.parametrize(
    ("skill_id", "expected_roles"),
    [
        (
            "ramp_merge_small_gap",
            {"merging_vehicle": "merge-vehicle", "mainline_vehicle": "mainline"},
        ),
        (
            "lane_drop_merge_competition",
            {
                "closing_lane_vehicle": "merge-vehicle",
                "continuing_lane_vehicle": "mainline",
            },
        ),
        (
            "merge_without_yield",
            {
                "non_yielding_vehicle": "merge-vehicle",
                "priority_vehicle": "mainline",
            },
        ),
    ],
)
def test_pair_specific_merge_accepts_true_two_source_convergence(
    config,
    skill_id,
    expected_roles,
) -> None:
    skill = _skill(skill_id)
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "merge-vehicle",
                "vehicle",
                reference_x=8.0,
                reference_y=4.0,
                speed_mps=5.0,
            ),
            _constant_agent_at_reference(
                "mainline",
                "vehicle",
                reference_x=10.0,
                speed_mps=5.0,
            ),
        ],
        map_polylines=_pair_merge_map(),
        scenario_id=f"pair-specific-{skill_id}",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.role_track_ids == expected_roles
    assert record.evidence["pair_specific_convergence"] is True
    assert record.evidence["convergence_relation"] == "two_source_lanes_share_successor"
    assert record.evidence["initiator_forward_to_convergence_m"] > 0
    assert record.evidence["responder_forward_to_convergence_m"] > 0
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    "skill_id",
    ["ramp_merge_small_gap", "lane_drop_merge_competition"],
)
def test_symmetric_pair_merge_rejects_observed_or_lane_role_semantics(
    config,
    skill_id,
) -> None:
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "z-vehicle",
                "vehicle",
                reference_x=8.0,
                reference_y=1.76,
                speed_mps=5.0,
            ),
            _constant_agent_at_reference(
                "a-vehicle",
                "vehicle",
                reference_x=8.0,
                reference_y=-1.76,
                speed_mps=5.0,
            ),
        ],
        map_polylines=_symmetric_pair_merge_map(),
        scenario_id=f"symmetric-pair-{skill_id}",
    )

    assert detect_scenario(scenario, [_skill(skill_id)], config).records == []


def test_symmetric_pair_merge_assigns_counterfactual_priority_deterministically(
    config,
) -> None:
    skill = _skill("merge_without_yield")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "z-vehicle",
                "vehicle",
                reference_x=8.0,
                reference_y=1.76,
                speed_mps=5.0,
            ),
            _constant_agent_at_reference(
                "a-vehicle",
                "vehicle",
                reference_x=8.0,
                reference_y=-1.76,
                speed_mps=5.0,
            ),
        ],
        map_polylines=_symmetric_pair_merge_map(),
        scenario_id="symmetric-pair-merge-without-yield",
    )

    first_run = detect_scenario(scenario, [skill], config)
    second_run = detect_scenario(scenario, [skill], config)

    assert len(first_run.records) == 1
    record = first_run.records[0]
    assert record.role_track_ids == {
        "non_yielding_vehicle": "z-vehicle",
        "priority_vehicle": "a-vehicle",
    }
    assert (
        record.evidence["role_assignment_basis"]
        == "deterministic_counterfactual_priority_assignment"
    )
    assert second_run.records == first_run.records
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    "skill_id",
    [
        "ramp_merge_small_gap",
        "lane_drop_merge_competition",
        "merge_without_yield",
    ],
)
@pytest.mark.parametrize("topology", ["same_lane", "serial_successor"])
def test_pair_specific_merge_rejects_nonconverging_pair(
    config,
    skill_id,
    topology,
) -> None:
    skill = _skill(skill_id)
    if topology == "same_lane":
        agents = [
            _constant_agent_at_reference(
                "first", "vehicle", reference_x=0.0, reference_y=20.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "second", "vehicle", reference_x=5.0, reference_y=20.0, speed_mps=3.0
            ),
        ]
    else:
        agents = [
            _constant_agent_at_reference(
                "first", "vehicle", reference_x=0.0, reference_y=20.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "second", "vehicle", reference_x=30.0, reference_y=20.0, speed_mps=3.0
            ),
        ]
    scenario = _scenario(
        agents,
        map_polylines=_serial_pair_with_unrelated_merge_map(),
        scenario_id=f"reject-{topology}-{skill_id}",
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts[f"{skill_id}:no_rule_match"] == 1


def _diverge_agents(*, same_flow: bool = False, lateral_displacement_m: float = 1.0):
    steps = 60
    reference_index = 29
    x = (np.arange(steps, dtype=np.float64) - reference_index) * 0.4
    y = np.zeros(steps, dtype=np.float64)
    y[reference_index:] = np.linspace(0.0, lateral_displacement_m, steps - reference_index)
    initiator = _agent_from_positions(
        "crossing",
        "vehicle",
        np.column_stack((x, y)),
        observed_steps=30,
        headings=0.0,
    )
    responder = _constant_agent_at_reference(
        "through",
        "vehicle",
        reference_x=2.0,
        reference_y=0.0 if same_flow else 4.0,
        speed_mps=4.0,
    )
    return [initiator, responder]


def test_diverge_requires_adjacent_flow_and_minimum_lateral_motion(config) -> None:
    skill = _skill("diverge_lane_crossing_conflict")
    scenario = _scenario(
        _diverge_agents(),
        map_polylines=_diverge_map(),
        scenario_id="valid-diverge-crossing",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["adjacent_lanes"] is True
    assert abs(record.evidence["future_lateral_displacement_m"]) >= 0.25
    assert (
        record.evidence["future_lateral_displacement_m"]
        * record.evidence["target_vehicle_lateral_offset_m"]
        > 0
    )
    assert record.seed_risk_metric == "minimum_trajectory_distance"
    assert record.target_risk_definition == skill.risk_definition
    assert record.seed_risk_is_proxy is True
    _assert_evidence_contract(record, skill)


def test_diverge_rejects_dangling_successor_ids(config) -> None:
    skill = _skill("diverge_lane_crossing_conflict")
    scenario = _scenario(
        _diverge_agents(),
        map_polylines=_diverge_map()[:2],
        scenario_id="dangling-successor-diverge",
    )

    assert detect_scenario(scenario, [skill], config).records == []


@pytest.mark.parametrize(
    ("same_flow", "lateral_displacement_m"),
    [(True, 1.0), (False, 0.0), (False, -1.0)],
)
def test_diverge_rejects_same_flow_or_invalid_lateral_motion(
    config,
    same_flow,
    lateral_displacement_m,
) -> None:
    skill = _skill("diverge_lane_crossing_conflict")
    scenario = _scenario(
        _diverge_agents(
            same_flow=same_flow,
            lateral_displacement_m=lateral_displacement_m,
        ),
        map_polylines=_diverge_map(),
        scenario_id=f"invalid-diverge-{same_flow}-{lateral_displacement_m}",
    )

    run = detect_scenario(scenario, [skill], config)

    assert run.records == []


def test_intersection_creep_accepts_real_right_angle_conflict(config) -> None:
    skill = _skill("intersection_creep_conflict")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "creeping",
                "vehicle",
                reference_x=-2.0,
                speed_mps=1.0,
            ),
            _constant_agent_at_reference(
                "crossing",
                "vehicle",
                reference_x=0.0,
                reference_y=-4.0,
                speed_mps=2.0,
                heading_rad=math.pi / 2,
            ),
        ],
        map_polylines=_intersection_map(),
        scenario_id="valid-intersection-creep",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["crossing_angle_deg"] == pytest.approx(90.0)
    assert record.evidence["conflict_point_xy"] is not None
    assert record.evidence["same_or_successor_lane"] is False
    _assert_evidence_contract(record, skill)


def test_intersection_creep_rejects_same_flow(config) -> None:
    skill = _skill("intersection_creep_conflict")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "creeping", "vehicle", reference_x=-2.0, speed_mps=1.0
            ),
            _constant_agent_at_reference(
                "same-flow", "vehicle", reference_x=-6.0, speed_mps=2.0
            ),
        ],
        map_polylines=_intersection_map(),
        scenario_id="same-flow-intersection-creep",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_limited_intersection_search_keeps_radius_caches_isolated(config) -> None:
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "vehicle",
                "vehicle",
                reference_x=0.0,
                reference_y=0.0,
                speed_mps=0.0,
            )
        ],
        map_polylines=[
            MapPolyline(
                "intersection:center",
                "lane_centerline",
                np.array([[9.0, -5.0], [9.0, 5.0]]),
                direction="vehicle",
                is_intersection=True,
                lane_id="intersection",
            )
        ],
        scenario_id="limited-intersection-cache",
    )
    context = ScenarioDetectionContext(scenario, config)
    state = context.state_by_id["vehicle"]

    assert context.distance_to_intersection(state, maximum_distance_m=8.0) == float(
        "inf"
    )
    assert context.distance_to_intersection(state) == pytest.approx(9.0)


def test_limited_intersection_search_preserves_inclusive_boundary(config) -> None:
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "vehicle",
                "vehicle",
                reference_x=0.0,
                reference_y=0.0,
                speed_mps=0.0,
            )
        ],
        map_polylines=[
            MapPolyline(
                "intersection:center",
                "lane_centerline",
                np.array([[8.0, -5.0], [8.0, 5.0]]),
                direction="vehicle",
                is_intersection=True,
                lane_id="intersection",
            )
        ],
        scenario_id="limited-intersection-boundary",
    )
    context = ScenarioDetectionContext(scenario, config)
    state = context.state_by_id["vehicle"]

    assert context.distance_to_intersection(
        state,
        maximum_distance_m=8.0,
    ) == pytest.approx(8.0)


def test_intersection_creep_rejects_fast_vehicle_before_map_search(
    config,
    monkeypatch,
) -> None:
    skill = _skill("intersection_creep_conflict")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "fast",
                "vehicle",
                reference_x=-2.0,
                speed_mps=3.0,
            ),
            _constant_agent_at_reference(
                "crossing",
                "vehicle",
                reference_x=0.0,
                reference_y=-4.0,
                speed_mps=3.0,
                heading_rad=math.pi / 2,
            ),
        ],
        map_polylines=_intersection_map(),
        scenario_id="fast-intersection-creep",
    )

    def reject_search(*args, **kwargs):
        raise AssertionError("fast initiator should be rejected before map search")

    monkeypatch.setattr(
        ScenarioDetectionContext,
        "distance_to_intersection",
        reject_search,
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_intersection_blocking_accepts_real_right_angle_conflict(config) -> None:
    skill = _skill("intersection_blocking_vehicle")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "blocking", "vehicle", reference_x=0.0, speed_mps=0.0
            ),
            _constant_agent_at_reference(
                "crossing",
                "vehicle",
                reference_x=0.0,
                reference_y=-4.0,
                speed_mps=2.0,
                heading_rad=math.pi / 2,
            ),
        ],
        map_polylines=_intersection_map(),
        scenario_id="valid-intersection-blocking",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["crossing_angle_deg"] == pytest.approx(90.0)
    assert record.evidence["conflict_point_xy"] is not None
    assert record.evidence["potential_occupancy_overlap_s"] > 0
    assert (
        record.seed_risk_metric
        == "potential_conflict_area_occupancy_overlap_proxy"
    )
    assert record.seed_risk_is_proxy is True
    _assert_evidence_contract(record, skill)


def test_intersection_blocking_rejects_same_flow(config) -> None:
    skill = _skill("intersection_blocking_vehicle")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "blocking", "vehicle", reference_x=0.0, speed_mps=0.0
            ),
            _constant_agent_at_reference(
                "same-flow", "vehicle", reference_x=-4.0, speed_mps=2.0
            ),
        ],
        map_polylines=_intersection_map(),
        scenario_id="same-flow-intersection-blocking",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_intersection_blocking_rejects_blocker_only_near_intersection_lane(
    config,
) -> None:
    skill = _skill("intersection_blocking_vehicle")
    map_polylines = [
        MapPolyline(
            "ordinary-east-west:center",
            "lane_centerline",
            np.array([[-30.0, 5.0], [30.0, 5.0]]),
            direction="vehicle",
            lane_id="ordinary-east-west",
        ),
        MapPolyline(
            "intersection-east-west:center",
            "lane_centerline",
            np.array([[-30.0, 0.0], [30.0, 0.0]]),
            direction="vehicle",
            is_intersection=True,
            lane_id="intersection-east-west",
        ),
        MapPolyline(
            "ordinary-south-north:center",
            "lane_centerline",
            np.array([[0.0, -30.0], [0.0, 30.0]]),
            direction="vehicle",
            lane_id="ordinary-south-north",
        ),
        MapPolyline(
            "drivable:near-intersection",
            "drivable_area",
            np.array(
                [
                    [-35.0, -35.0],
                    [35.0, -35.0],
                    [35.0, 35.0],
                    [-35.0, 35.0],
                    [-35.0, -35.0],
                ]
            ),
        ),
    ]
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "blocking",
                "vehicle",
                reference_x=0.0,
                reference_y=5.0,
                speed_mps=0.0,
            ),
            _constant_agent_at_reference(
                "crossing",
                "vehicle",
                reference_x=0.0,
                reference_y=0.0,
                speed_mps=2.0,
                heading_rad=math.pi / 2,
            ),
        ],
        map_polylines=map_polylines,
        scenario_id="blocker-near-intersection-lane",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _forced_blockage_scenario(
    *,
    vehicle_speed_mps: float = 2.0,
    obstacle_x: float = 10.0,
    obstacle_y: float = 0.0,
    scenario_id: str,
) -> Scenario:
    return _scenario(
        [
            _constant_agent_at_reference(
                "blocker",
                "vehicle",
                reference_x=obstacle_x,
                reference_y=obstacle_y,
                speed_mps=0.0,
            ),
            _constant_agent_at_reference(
                "avoiding",
                "vehicle",
                reference_x=0.0,
                speed_mps=vehicle_speed_mps,
            ),
        ],
        map_polylines=_parallel_lane_map(),
        scenario_id=scenario_id,
    )


def test_forced_blockage_accepts_moving_vehicle_and_same_path_blocker(config) -> None:
    skill = _skill("forced_lane_change_around_blockage")
    scenario = _forced_blockage_scenario(scenario_id="valid-forced-blockage")

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["same_or_successor_lane"] is True
    assert record.evidence["vehicle_speed_mps"] >= 1.0
    assert record.evidence["vehicle_center_distance_m"] >= 2.0
    assert record.evidence["blockage_distance_ahead_m"] >= 2.0
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("vehicle_speed_mps", "obstacle_x", "obstacle_y"),
    [
        (0.9, 10.0, 0.0),
        (2.0, 10.0, 4.0),
        (2.0, 1.9, 0.0),
        (2.0, 1.5, 1.5),
    ],
)
def test_forced_blockage_rejects_weak_motion_wrong_path_or_insufficient_distance(
    config,
    vehicle_speed_mps,
    obstacle_x,
    obstacle_y,
) -> None:
    skill = _skill("forced_lane_change_around_blockage")
    scenario = _forced_blockage_scenario(
        vehicle_speed_mps=vehicle_speed_mps,
        obstacle_x=obstacle_x,
        obstacle_y=obstacle_y,
        scenario_id=f"invalid-forced-{vehicle_speed_mps}-{obstacle_x}-{obstacle_y}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_simultaneous_lane_change_accepts_opposite_sides_of_shared_target(config) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "left", "vehicle", reference_x=0.0, reference_y=4.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "right", "vehicle", reference_x=1.0, reference_y=-4.0, speed_mps=3.0
            ),
        ],
        map_polylines=_shared_target_lane_map(),
        scenario_id="valid-simultaneous-lane-change",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.role_track_ids == {
        "left_lane_changer": "left",
        "right_lane_changer": "right",
    }
    assert record.evidence["shared_target_lane_id"] == "shared-target"
    assert record.evidence["target_has_reciprocal_source_neighbors"] is True
    assert record.evidence["left_vehicle_target_lane_lateral_offset_m"] > 0
    assert record.evidence["right_vehicle_target_lane_lateral_offset_m"] < 0
    assert record.evidence["current_vehicle_separation_m"] >= 2.0
    assert record.seed_risk_is_proxy is True
    _assert_evidence_contract(record, skill)


def test_simultaneous_lane_change_rejects_target_without_reciprocal_neighbors(
    config,
) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    map_polylines = _shared_target_lane_map()
    target_lane = map_polylines[-1]
    target_lane.left_neighbor_id = None
    target_lane.right_neighbor_id = None
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "left", "vehicle", reference_x=0.0, reference_y=4.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "right", "vehicle", reference_x=1.0, reference_y=-4.0, speed_mps=3.0
            ),
        ],
        map_polylines=map_polylines,
        scenario_id="nonreciprocal-simultaneous-lane-change",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_simultaneous_lane_change_rejects_target_not_between_source_lanes(
    config,
) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    map_polylines = _shared_target_lane_map()
    map_polylines[-1].points[:, 1] = 100.0
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "left", "vehicle", reference_x=0.0, reference_y=4.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "right", "vehicle", reference_x=1.0, reference_y=-4.0, speed_mps=3.0
            ),
        ],
        map_polylines=map_polylines,
        scenario_id="geometrically-invalid-simultaneous-lane-change",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def test_simultaneous_lane_change_tiny_closing_rate_uses_distance_proxy(config) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "left",
                "vehicle",
                reference_x=0.0,
                reference_y=4.0,
                speed_mps=3.0,
                heading_rad=-0.001,
            ),
            _constant_agent_at_reference(
                "right",
                "vehicle",
                reference_x=0.0,
                reference_y=-4.0,
                speed_mps=3.0,
                heading_rad=0.001,
            ),
        ],
        map_polylines=_shared_target_lane_map(),
        scenario_id="tiny-closing-simultaneous-lane-change",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.seed_risk_metric == "minimum_trajectory_distance"
    assert record.evidence["observed_lateral_time_to_collision_s"] is None
    assert record.seed_risk_is_proxy is True


def test_simultaneous_lane_change_rejects_overlapping_vehicle_centers(config) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    source_offset_m = 0.9
    scenario = _scenario(
        [
            _constant_agent_at_reference(
                "left",
                "vehicle",
                reference_x=0.0,
                reference_y=source_offset_m,
                speed_mps=3.0,
            ),
            _constant_agent_at_reference(
                "right",
                "vehicle",
                reference_x=0.0,
                reference_y=-source_offset_m,
                speed_mps=3.0,
            ),
        ],
        map_polylines=_shared_target_lane_map(source_offset_m=source_offset_m),
        scenario_id="overlapping-simultaneous-lane-change",
    )

    assert detect_scenario(scenario, [skill], config).records == []


@pytest.mark.parametrize("invalid_topology", ["same_source", "same_side"])
def test_simultaneous_lane_change_rejects_invalid_source_topology(
    config,
    invalid_topology,
) -> None:
    skill = _skill("simultaneous_lane_change_conflict")
    if invalid_topology == "same_source":
        agents = [
            _constant_agent_at_reference(
                "first", "vehicle", reference_x=0.0, reference_y=4.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "second", "vehicle", reference_x=1.0, reference_y=4.0, speed_mps=3.0
            ),
        ]
        map_polylines = _shared_target_lane_map()
    else:
        agents = [
            _constant_agent_at_reference(
                "first", "vehicle", reference_x=0.0, reference_y=4.0, speed_mps=3.0
            ),
            _constant_agent_at_reference(
                "second", "vehicle", reference_x=1.0, reference_y=8.0, speed_mps=3.0
            ),
        ]
        map_polylines = _shared_target_lane_map(same_side=True)
    scenario = _scenario(
        agents,
        map_polylines=map_polylines,
        scenario_id=f"invalid-simultaneous-{invalid_topology}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _slow_blockage_scenario(
    *,
    leader_speed_mps: float,
    follower_speed_mps: float,
    gap_m: float,
    scenario_id: str,
) -> Scenario:
    return _scenario(
        [
            _constant_agent_at_reference(
                "slow-leader",
                "vehicle",
                reference_x=gap_m,
                speed_mps=leader_speed_mps,
            ),
            _constant_agent_at_reference(
                "follower",
                "vehicle",
                reference_x=0.0,
                speed_mps=follower_speed_mps,
            ),
        ],
        map_polylines=_parallel_lane_map(),
        scenario_id=scenario_id,
    )


def test_slow_lead_blockage_accepts_effective_moving_following_pair(config) -> None:
    skill = _skill("slow_lead_blockage")
    scenario = _slow_blockage_scenario(
        leader_speed_mps=1.0,
        follower_speed_mps=2.0,
        gap_m=10.0,
        scenario_id="valid-slow-lead",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["follower_current_speed_mps"] >= 1.0
    assert record.evidence["closing_speed_mps"] >= 0.5
    assert record.evidence["longitudinal_gap_m"] >= 2.0
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("leader_speed_mps", "follower_speed_mps", "gap_m"),
    [(0.0, 0.9, 10.0), (1.0, 1.49, 10.0), (1.0, 2.0, 1.9)],
)
def test_slow_lead_blockage_rejects_speed_closing_or_gap_below_threshold(
    config,
    leader_speed_mps,
    follower_speed_mps,
    gap_m,
) -> None:
    skill = _skill("slow_lead_blockage")
    scenario = _slow_blockage_scenario(
        leader_speed_mps=leader_speed_mps,
        follower_speed_mps=follower_speed_mps,
        gap_m=gap_m,
        scenario_id=f"invalid-slow-{leader_speed_mps}-{follower_speed_mps}-{gap_m}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _lead_stop_scenario(
    *,
    follower_speed_mps: float,
    gap_m: float,
    leader_already_stopped: bool = False,
    scenario_id: str,
) -> Scenario:
    if leader_already_stopped:
        leader_speeds = np.concatenate((np.full(20, 4.0), np.zeros(40)))
    else:
        leader_speeds = np.concatenate(
            (np.full(30, 4.0), np.array([3.0, 2.0, 1.0, 0.0]), np.zeros(26))
        )
    follower_speeds = np.full(60, follower_speed_mps)
    reference_index = 29
    leader_displacement = float(leader_speeds[:reference_index].sum() * SAMPLE_PERIOD_S)
    follower_displacement = follower_speed_mps * reference_index * SAMPLE_PERIOD_S
    leader_initial_x = gap_m + follower_displacement - leader_displacement
    return _scenario(
        [
            _agent_from_speeds(
                "stopping-leader",
                leader_speeds,
                initial_x=leader_initial_x,
                observed_steps=30,
            ),
            _agent_from_speeds(
                "follower",
                follower_speeds,
                initial_x=0.0,
                observed_steps=30,
            ),
        ],
        map_polylines=_parallel_lane_map(),
        scenario_id=scenario_id,
    )


def test_lead_sudden_stop_accepts_moving_follower_with_effective_closing(config) -> None:
    skill = _skill("lead_sudden_stop")
    scenario = _lead_stop_scenario(
        follower_speed_mps=5.0,
        gap_m=10.0,
        scenario_id="valid-lead-stop",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["follower_current_speed_mps"] >= 1.0
    assert record.evidence["closing_speed_mps"] >= 0.5
    assert record.evidence["longitudinal_gap_m"] >= 2.0
    assert record.evidence["stopped_duration_s"] >= 1.0
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("future_prior_speed_mps", "expected_records"),
    [(2.1, 1), (1.9, 0)],
)
def test_lead_sudden_stop_uses_maximum_speed_before_future_stop(
    config,
    future_prior_speed_mps,
    expected_records,
) -> None:
    skill = _skill("lead_sudden_stop")
    leader_speeds = np.concatenate(
        (
            np.full(30, 1.9),
            np.array([future_prior_speed_mps, 1.0, 0.0]),
            np.zeros(27),
        )
    )
    follower_speed_mps = 3.0
    reference_index = 29
    gap_m = 10.0
    leader_displacement = float(
        leader_speeds[:reference_index].sum() * SAMPLE_PERIOD_S
    )
    follower_displacement = follower_speed_mps * reference_index * SAMPLE_PERIOD_S
    scenario = _scenario(
        [
            _agent_from_speeds(
                "stopping-leader",
                leader_speeds,
                initial_x=gap_m + follower_displacement - leader_displacement,
                observed_steps=30,
            ),
            _agent_from_speeds(
                "follower",
                np.full(60, follower_speed_mps),
                initial_x=0.0,
                observed_steps=30,
            ),
        ],
        map_polylines=_parallel_lane_map(),
        scenario_id=f"future-prior-speed-{future_prior_speed_mps}",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == expected_records
    if run.records:
        record = run.records[0]
        assert record.evidence["leader_current_speed_mps"] == pytest.approx(1.9)
        assert record.evidence["leader_maximum_prior_stop_speed_mps"] == pytest.approx(
            future_prior_speed_mps
        )
        _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("follower_speed_mps", "gap_m", "leader_already_stopped"),
    [(0.9, 10.0, True), (4.49, 10.0, False), (5.0, 1.9, False)],
)
def test_lead_sudden_stop_rejects_speed_closing_or_gap_below_threshold(
    config,
    follower_speed_mps,
    gap_m,
    leader_already_stopped,
) -> None:
    skill = _skill("lead_sudden_stop")
    scenario = _lead_stop_scenario(
        follower_speed_mps=follower_speed_mps,
        gap_m=gap_m,
        leader_already_stopped=leader_already_stopped,
        scenario_id=f"invalid-lead-stop-{follower_speed_mps}-{gap_m}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _stationary_cut_out_scenario(
    *,
    target_x: float,
    cut_out_x: float,
    slow_x: float,
    scenario_id: str,
) -> Scenario:
    return _scenario(
        [
            _constant_agent_at_reference(
                "cut-out", "vehicle", reference_x=cut_out_x, speed_mps=0.0
            ),
            _constant_agent_at_reference(
                "target", "vehicle", reference_x=target_x, speed_mps=0.0
            ),
            _constant_agent_at_reference(
                "slow", "vehicle", reference_x=slow_x, speed_mps=0.0
            ),
        ],
        map_polylines=_parallel_lane_map(),
        scenario_id=scenario_id,
    )


def test_cut_out_requires_both_queue_segments_above_minimum_gap(config) -> None:
    skill = _skill("cut_out_reveals_slow_vehicle")
    scenario = _stationary_cut_out_scenario(
        target_x=0.0,
        cut_out_x=10.0,
        slow_x=20.0,
        scenario_id="valid-cut-out-minimum-gaps",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["cut_out_to_target_gap_m"] >= 2.0
    assert record.evidence["slow_to_cut_out_gap_m"] >= 2.0
    assert record.seed_risk_metric == "newly_exposed_longitudinal_gap"
    assert record.target_risk_definition == skill.risk_definition
    assert record.seed_risk_is_proxy is True
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("target_x", "cut_out_x", "slow_x"),
    [(8.1, 10.0, 20.0), (0.0, 10.0, 11.9)],
)
def test_cut_out_rejects_either_queue_segment_below_minimum_gap(
    config,
    target_x,
    cut_out_x,
    slow_x,
) -> None:
    skill = _skill("cut_out_reveals_slow_vehicle")
    scenario = _stationary_cut_out_scenario(
        target_x=target_x,
        cut_out_x=cut_out_x,
        slow_x=slow_x,
        scenario_id=f"invalid-cut-out-gap-{target_x}-{cut_out_x}-{slow_x}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _jaywalking_agents(*, pedestrian_heading_rad: float = math.pi / 2):
    steps = 60
    reference_index = 29
    tau = (np.arange(steps, dtype=np.float64) - reference_index) * SAMPLE_PERIOD_S
    pedestrian_positions = np.column_stack(
        (np.zeros(steps), -2.0 + tau)
    )
    pedestrian = _agent_from_positions(
        "pedestrian",
        "pedestrian",
        pedestrian_positions,
        observed_steps=30,
        headings=pedestrian_heading_rad,
    )
    vehicle = _constant_agent_at_reference(
        "vehicle",
        "vehicle",
        reference_x=-4.0,
        speed_mps=2.0,
    )
    return [pedestrian, vehicle]


def test_jaywalking_requires_crossing_angle_and_valid_conflict_location(config) -> None:
    skill = _skill("jaywalking_pedestrian_crossing")
    scenario = _scenario(
        _jaywalking_agents(),
        map_polylines=_jaywalking_map(),
        scenario_id="valid-jaywalking",
    )

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.evidence["vru_vehicle_heading_difference_deg"] == pytest.approx(90.0)
    assert record.evidence["inside_drivable_area"] is True
    assert record.evidence["crosswalk_distance_m"] >= 10.0
    assert record.evidence["conflict_point_xy"] == pytest.approx([0.0, 0.0])
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("pedestrian_heading_rad", "crosswalk_near_conflict"),
    [(0.0, False), (math.pi / 2, True)],
)
def test_jaywalking_rejects_parallel_heading_or_crosswalk_conflict_location(
    config,
    pedestrian_heading_rad,
    crosswalk_near_conflict,
) -> None:
    skill = _skill("jaywalking_pedestrian_crossing")
    scenario = _scenario(
        _jaywalking_agents(pedestrian_heading_rad=pedestrian_heading_rad),
        map_polylines=_jaywalking_map(
            crosswalk_near_conflict=crosswalk_near_conflict
        ),
        scenario_id=f"invalid-jaywalking-{pedestrian_heading_rad}-{crosswalk_near_conflict}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _stopped_reentry_scenario(
    *,
    include_front: bool = True,
    include_rear: bool = True,
    reentry_speed_mps: float = 0.0,
    scenario_id: str,
) -> Scenario:
    agents = [
        _constant_agent_at_reference(
            "reentering",
            "vehicle",
            reference_x=0.0,
            reference_y=3.0,
            speed_mps=reentry_speed_mps,
        )
    ]
    if include_front:
        agents.append(
            _constant_agent_at_reference(
                "front", "vehicle", reference_x=10.0, speed_mps=2.0
            )
        )
    if include_rear:
        agents.append(
            _constant_agent_at_reference(
                "rear", "vehicle", reference_x=-10.0, speed_mps=2.0
            )
        )
    return _scenario(
        agents,
        map_polylines=_parallel_lane_map(),
        scenario_id=scenario_id,
    )


def test_stopped_reentry_emits_three_roles_and_proxy_target_contract(config) -> None:
    skill = _skill("stopped_vehicle_reentry")
    scenario = _stopped_reentry_scenario(scenario_id="valid-stopped-reentry")

    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    assert record.role_track_ids == {
        "reentering_vehicle": "reentering",
        "front_main_flow_vehicle": "front",
        "rear_main_flow_vehicle": "rear",
    }
    assert record.evidence["front_gap_m"] == pytest.approx(10.0)
    assert record.evidence["rear_gap_m"] == pytest.approx(10.0)
    assert record.seed_risk_metric == "minimum_front_rear_trajectory_distance_proxy"
    assert record.target_risk_definition == skill.risk_definition
    assert record.seed_risk_is_proxy is True
    _assert_evidence_contract(record, skill)


@pytest.mark.parametrize(
    ("include_front", "include_rear", "reentry_speed_mps"),
    [(True, False, 0.0), (False, True, 0.0), (True, True, 0.6)],
)
def test_stopped_reentry_rejects_missing_side_or_currently_moving_vehicle(
    config,
    include_front,
    include_rear,
    reentry_speed_mps,
) -> None:
    skill = _skill("stopped_vehicle_reentry")
    scenario = _stopped_reentry_scenario(
        include_front=include_front,
        include_rear=include_rear,
        reentry_speed_mps=reentry_speed_mps,
        scenario_id=f"invalid-reentry-{include_front}-{include_rear}-{reentry_speed_mps}",
    )

    assert detect_scenario(scenario, [skill], config).records == []


def _left_turn_overlap_agents() -> list[AgentTrack]:
    left_observed = np.column_stack(
        (np.zeros(30), np.linspace(-9.0, -3.0, 30))
    )
    left_to_conflict = np.column_stack(
        (np.zeros(15), np.linspace(-2.8, 0.0, 15))
    )
    left_after_turn = np.column_stack(
        (np.linspace(-0.2, -3.0, 15), np.zeros(15))
    )
    left = _agent_from_positions(
        "a-left-turn",
        "vehicle",
        np.vstack((left_observed, left_to_conflict, left_after_turn)),
        observed_steps=30,
    )
    through = _agent_from_positions(
        "b-through",
        "vehicle",
        np.column_stack(
            (
                np.zeros(60),
                np.concatenate(
                    (np.linspace(9.0, 3.0, 30), np.linspace(2.8, -3.0, 30))
                ),
            )
        ),
        observed_steps=30,
    )
    return [left, through]


def test_same_scene_preserves_generic_and_specific_intersection_labels(config) -> None:
    skills = [
        _skill("crossing_path_conflict"),
        _skill("unprotected_left_turn_conflict"),
    ]
    scenario = _scenario(
        _left_turn_overlap_agents(),
        map_polylines=_intersection_map(),
        scenario_id="multi-label-left-turn-conflict",
    )

    run = detect_scenario(scenario, skills, config)

    assert len(run.records) == 2
    assert {record.skill_id for record in run.records} == {
        "crossing_path_conflict",
        "unprotected_left_turn_conflict",
    }
    assert {record.scenario_id for record in run.records} == {scenario.scenario_id}
    assert all(record.seed_risk_is_proxy is False for record in run.records)
    assert all(
        record.target_risk_definition == _skill(record.skill_id).risk_definition
        for record in run.records
    )


def test_three_role_sampling_key_contains_target_rear_vehicle(
    config, monkeypatch
) -> None:
    skill = _skill("narrow_gap_lane_change")
    scenario = _scenario(
        [
            _constant_agent("lane-changer", "vehicle"),
            _constant_agent("front", "vehicle", initial_x=10.0),
            _constant_agent("rear", "vehicle", initial_x=-10.0),
        ],
        map_polylines=_full_map(),
        scenario_id="three-role-sampling",
    )

    def fake_handler(context, loaded_skill, initiators, responders):
        assert loaded_skill.skill_id == skill.skill_id
        return [
            RuleMatch(
                initiator=context.state_by_id["lane-changer"],
                responder=context.state_by_id["front"],
                additional_actors=(context.state_by_id["rear"],),
                trigger_score=0.8,
                risk_metric=skill.risk_definition["metric"],
                risk_value=2.0,
                evidence={
                    "synthetic_match": True,
                    "future_lateral_displacement_m": 3.0,
                    "front_gap_m": 10.0,
                    "rear_gap_m": 10.0,
                },
            )
        ]

    monkeypatch.setitem(detection._STRATEGY_HANDLERS, "lane_change_gap", fake_handler)
    run = detect_scenario(scenario, [skill], config)

    assert len(run.records) == 1
    record = run.records[0]
    expected_roles = {
        "lane_changer": "lane-changer",
        "target_front_vehicle": "front",
        "target_rear_vehicle": "rear",
    }
    assert record.role_track_ids == expected_roles
    sample_key = "|".join(
        (
            scenario.scenario_id,
            skill.skill_id,
            *(f"{role}={track_id}" for role, track_id in expected_roles.items()),
        )
    )
    assert record.sampled_parameters == sample_skill_parameters(
        skill,
        global_seed=config.global_seed,
        sample_key=sample_key,
    )
    assert "target_rear_vehicle" in record.unique_key[-1]
    assert "rear" in record.unique_key[-1]


def test_duplicate_track_across_generated_roles_is_rejected(config, monkeypatch) -> None:
    skill = _skill("narrow_gap_lane_change")
    scenario = _scenario(
        [
            _constant_agent("lane-changer", "vehicle"),
            _constant_agent("front", "vehicle", initial_x=10.0),
        ],
        map_polylines=_full_map(),
        scenario_id="duplicate-role-track",
    )

    def fake_handler(context, loaded_skill, initiators, responders):
        front = context.state_by_id["front"]
        return [
            RuleMatch(
                initiator=context.state_by_id["lane-changer"],
                responder=front,
                additional_actors=(front,),
                trigger_score=0.8,
                risk_metric=loaded_skill.risk_definition["metric"],
                risk_value=2.0,
                evidence={"synthetic_match": True},
            )
        ]

    monkeypatch.setitem(detection._STRATEGY_HANDLERS, "lane_change_gap", fake_handler)
    run = detect_scenario(scenario, [skill], config)

    assert run.records == []
    assert run.rejection_counts["narrow_gap_lane_change:duplicate_role_actor"] == 1

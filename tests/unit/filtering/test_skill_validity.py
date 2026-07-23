from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

import skilldrive.filtering.skill_validity as skill_validity
from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.filtering.risk import (
    RISK_CONTEXT_METADATA_KEY,
    SKILL_RISK_CALCULATORS,
    RiskReason,
    evaluate_skill_risk,
)
from skilldrive.filtering.skill_validity import (
    SKILL_VALIDATORS,
    prepare_risk_context,
    validate_skill_trigger,
)
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.skills.detection import load_detection_config
from skilldrive.skills.loader import load_skill


ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = ROOT / "configs" / "skills"


def _agent(
    track_id: str,
    positions: np.ndarray,
    *,
    object_type: str = "vehicle",
) -> AgentTrack:
    points = np.asarray(positions, dtype=np.float64)
    velocities = np.zeros_like(points)
    velocities[1:] = np.diff(points, axis=0) / 0.1
    velocities[0] = velocities[1]
    headings = np.zeros(len(points), dtype=np.float64)
    moving = np.linalg.norm(velocities, axis=1) > 1e-9
    headings[moving] = np.arctan2(velocities[moving, 1], velocities[moving, 0])
    observed = np.zeros(len(points), dtype=bool)
    observed[:50] = True
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=points,
        velocities=velocities,
        headings=headings,
        observed_mask=observed,
    )


def _constant_agent(
    track_id: str,
    x: float,
    y: float = 0.0,
    *,
    object_type: str = "vehicle",
) -> AgentTrack:
    return _agent(
        track_id,
        np.tile([x, y], (110, 1)),
        object_type=object_type,
    )


def _scenario(
    agents: list[AgentTrack],
    *,
    scenario_id: str = "skill-validity",
    map_polylines: list[MapPolyline] | None = None,
) -> Scenario:
    agents[0].is_focal = True
    return Scenario(
        scenario_id=scenario_id,
        city_name="test-city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id=agents[0].track_id,
        agents=agents,
        map_polylines=[] if map_polylines is None else map_polylines,
        metadata={},
    )


def _formal_catalog_ids() -> set[str]:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    return {
        item["skill_id"] for entries in catalog["families"].values() for item in entries
    }


def test_registry_explicitly_maps_all_34_skills_modes_roles_and_conditions() -> None:
    assert len(SKILL_VALIDATORS) == 34
    assert set(SKILL_VALIDATORS) == _formal_catalog_ids()
    assert set(SKILL_VALIDATORS) == set(SKILL_RISK_CALCULATORS)
    assert (
        sum(
            spec.detection_mode == "observed_trigger"
            for spec in SKILL_VALIDATORS.values()
        )
        == 14
    )
    assert (
        sum(
            spec.detection_mode == "compatible_seed"
            for spec in SKILL_VALIDATORS.values()
        )
        == 20
    )

    for skill_id, spec in SKILL_VALIDATORS.items():
        skill = load_skill(SKILL_DIR / f"{skill_id}.yaml")
        assert spec.skill_id == skill_id
        assert spec.detection_mode == skill.detection["mode"]
        assert spec.required_roles == tuple(skill.actors["generated_roles"])
        assert spec.required_roles == SKILL_RISK_CALCULATORS[skill_id].required_roles
        if spec.detection_mode == "compatible_seed":
            assert set(spec.condition_validators) == set(skill.trigger["conditions"])
        else:
            assert not spec.condition_validators


def test_group_pedestrian_uses_the_same_exact_role_observed_redetection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = load_skill(SKILL_DIR / "group_pedestrian_crossing.yaml")
    scenario = _scenario(
        [
            _constant_agent("first-ped", 0.0, object_type="pedestrian"),
            _constant_agent("vehicle", 5.0),
            _constant_agent("second-ped", 1.0, object_type="pedestrian"),
        ]
    )
    roles = {
        "first_crossing_pedestrian": "first-ped",
        "responding_vehicle": "vehicle",
        "second_crossing_pedestrian": "second-ped",
    }
    captured: dict[str, object] = {}

    def fake_observed(
        received_scenario: Scenario,
        received_skill,
        received_roles,
        received_config,
    ) -> FilterCheck:
        captured.update(
            scenario=received_scenario,
            skill=received_skill,
            roles=dict(received_roles),
            config=received_config,
        )
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            metrics={"requested_role_track_ids": dict(received_roles)},
        )

    monkeypatch.setattr(skill_validity, "validate_observed_skill", fake_observed)
    config = load_detection_config(ROOT / "configs" / "seed_detection.yaml")
    result = validate_skill_trigger(
        scenario,
        scenario,
        skill,
        roles,
        {"detection_mode": "observed_trigger"},
        config,
    )

    assert result.passed
    assert result.metrics["validator_mode"] == "observed_exact_role_redetection"
    assert result.metrics["seed_evidence_used"] is False
    assert captured["scenario"] is scenario
    assert captured["skill"] is skill
    assert captured["roles"] == roles


def _construction_scenarios(*, approaching: bool) -> tuple[Scenario, Scenario]:
    construction = _constant_agent(
        "construction",
        10.0,
        object_type="construction",
    )
    history_x = np.linspace(-5.0, 0.0, 50)
    future_x = np.linspace(0.2, 8.0, 60) if approaching else np.linspace(-0.2, -6.0, 60)
    generated_vehicle = _agent(
        "vehicle",
        np.column_stack((np.concatenate((history_x, future_x)), np.zeros(110))),
    )
    source_vehicle = _agent(
        "vehicle",
        np.column_stack(
            (
                np.concatenate((history_x, np.linspace(0.2, 3.0, 60))),
                np.zeros(110),
            )
        ),
    )
    source = _scenario(
        [
            _constant_agent("construction", 10.0, object_type="construction"),
            source_vehicle,
        ],
        scenario_id="source",
    )
    generated = _scenario(
        [construction, generated_vehicle],
        scenario_id="generated",
    )
    return source, generated


def _construction_seed() -> dict:
    return {
        "detection_mode": "compatible_seed",
        "matched_conditions": [
            "construction_actor_near_path",
            "vehicle_approaching",
            "response_space_available",
        ],
        "missing_generation_conditions": [
            "construction_actor_near_path",
            "vehicle_approaching",
            "safety_buffer_violation_predicted",
        ],
        "minimum_object_path_clearance_m": 3.0,
    }


def test_compatible_validator_accepts_only_when_all_missing_conditions_are_realized() -> (
    None
):
    skill = load_skill(SKILL_DIR / "construction_object_lane_blockage.yaml")
    roles = {
        "construction_object": "construction",
        "responding_vehicle": "vehicle",
    }
    source, generated = _construction_scenarios(approaching=True)

    result = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _construction_seed(),
    )

    assert result.passed
    assert result.metrics["compatible_seed_detector_reused"] is False
    assert result.metrics["failed_conditions"] == []
    assert result.metrics["condition_evidence"]["construction_actor_near_path"] == {
        "passed": True,
        "source": "frozen_seed_structural_evidence",
        "compatible_seed_detector_reused": False,
        "matched_condition_aliases": ["construction_actor_near_path"],
        "structural_fields": {"minimum_object_path_clearance_m": 3.0},
    }


def test_compatible_validator_rejects_dynamic_failure_and_exact_role_type_swap() -> (
    None
):
    skill = load_skill(SKILL_DIR / "construction_object_lane_blockage.yaml")
    roles = {
        "construction_object": "construction",
        "responding_vehicle": "vehicle",
    }
    source, generated = _construction_scenarios(approaching=False)
    failed = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _construction_seed(),
    )

    assert failed.rejection_reasons == (FilterRejection.SKILL_TRIGGER_NOT_REALIZED,)
    assert set(failed.metrics["failed_conditions"]) == {
        "vehicle_approaching",
        "safety_buffer_violation_predicted",
    }

    swapped = validate_skill_trigger(
        source,
        generated,
        skill,
        {
            "construction_object": "vehicle",
            "responding_vehicle": "construction",
        },
        _construction_seed(),
    )
    assert swapped.rejection_reasons == (FilterRejection.SKILL_ROLE_CONTRACT_MISMATCH,)
    assert set(swapped.metrics["role_type_mismatches"]) == {
        "construction_object",
        "responding_vehicle",
    }


def _forced_lane_change_scenarios(
    *,
    lane_change_realized: bool,
    after_successor: bool = False,
    blocker_x: float = 5.0,
) -> tuple[Scenario, Scenario]:
    current_lane = MapPolyline(
        polyline_id="lane-current:center",
        polyline_type="lane_centerline",
        points=np.asarray([[-10.0, 0.0], [10.0, 0.0]]),
        lane_id="lane-current",
        successor_ids=["lane-successor"],
        left_neighbor_id="lane-left",
    )
    successor_lane = MapPolyline(
        polyline_id="lane-successor:center",
        polyline_type="lane_centerline",
        points=np.asarray([[10.0, 0.0], [40.0, 0.0]]),
        lane_id="lane-successor",
        predecessor_ids=["lane-current"],
        left_neighbor_id="lane-left-successor",
    )
    adjacent_lane = MapPolyline(
        polyline_id="lane-left:center",
        polyline_type="lane_centerline",
        points=np.asarray([[-10.0, 4.0], [10.0, 4.0]]),
        lane_id="lane-left",
        successor_ids=["lane-left-successor"],
        right_neighbor_id="lane-current",
    )
    successor_adjacent_lane = MapPolyline(
        polyline_id="lane-left-successor:center",
        polyline_type="lane_centerline",
        points=np.asarray([[10.0, 4.0], [40.0, 4.0]]),
        lane_id="lane-left-successor",
        predecessor_ids=["lane-left"],
        right_neighbor_id="lane-successor",
    )
    lanes = [
        current_lane,
        successor_lane,
        adjacent_lane,
        successor_adjacent_lane,
    ]
    history_x = np.linspace(-5.0, 0.0, 50)
    future_x = np.linspace(0.25, 35.0, 60)
    source_y = np.concatenate((np.linspace(0.0, 4.0, 9), np.full(51, 4.0)))
    late_lane_change_y = np.concatenate(
        (np.zeros(22), np.linspace(0.0, 4.0, 9), np.full(29, 4.0))
    )
    if not lane_change_realized:
        generated_y = np.zeros(60)
    elif after_successor:
        generated_y = late_lane_change_y
    else:
        generated_y = source_y
    source_vehicle = _agent(
        "avoider",
        np.column_stack(
            (
                np.concatenate((history_x, future_x)),
                np.concatenate((np.zeros(50), source_y)),
            )
        ),
    )
    generated_vehicle = _agent(
        "avoider",
        np.column_stack(
            (
                np.concatenate((history_x, future_x)),
                np.concatenate((np.zeros(50), generated_y)),
            )
        ),
    )
    source = _scenario(
        [_constant_agent("blocker", blocker_x), source_vehicle],
        scenario_id="forced-source",
        map_polylines=lanes,
    )
    generated = _scenario(
        [_constant_agent("blocker", blocker_x), generated_vehicle],
        scenario_id="forced-generated",
        map_polylines=lanes,
    )
    return source, generated


def _forced_lane_change_seed() -> dict:
    return {
        "detection_mode": "compatible_seed",
        "matched_conditions": [
            "blockage_ahead",
            "adjacent_lane_available",
            "response_space_available",
        ],
        "missing_generation_conditions": [
            "blockage_ahead",
            "adjacent_lane_available",
            "approach_to_safety_buffer",
        ],
        "initiator_lane_id": "lane-current",
        "responder_lane_id": "lane-current",
    }


def _real_forced_lane_change_seed() -> dict:
    seed = _forced_lane_change_seed()
    seed["initiator_lane_id"] = "353623830"
    seed["responder_lane_id"] = "353623830"
    return seed


@pytest.fixture(scope="module")
def real_forced_lane_change_source() -> Scenario:
    scenario_id = "901f8f00-5352-4f9e-b5c8-004a8c724954"
    scenario_path = (
        ROOT
        / "data"
        / "av2"
        / "motion-forecasting"
        / "train"
        / scenario_id
        / f"scenario_{scenario_id}.parquet"
    )
    if not scenario_path.is_file():
        pytest.skip("local forced-lane-change scenario is unavailable")
    pytest.importorskip("av2")
    from skilldrive.data.av2_reader import load_av2_scenario

    return load_av2_scenario(scenario_path)


def test_forced_lane_change_requires_generated_entry_into_direct_adjacent_lane() -> (
    None
):
    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    roles = {"blocking_actor": "blocker", "avoiding_vehicle": "avoider"}
    source, generated = _forced_lane_change_scenarios(lane_change_realized=True)

    result = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _forced_lane_change_seed(),
    )

    assert result.passed
    evidence = result.metrics["condition_evidence"]["adjacent_lane_available"]
    assert evidence["generated_compact_lane_sequence"] == [
        "lane-current",
        "lane-left",
        "lane-left-successor",
    ]
    assert evidence["entered_adjacent_lane_id"] == "lane-left"
    assert evidence["lane_change_frame_index"] < evidence[
        "first_blocker_pass_frame_index"
    ]
    assert evidence["temporal_order_passed"] is True


def test_forced_lane_change_accepts_successor_then_direct_neighbor_transition() -> (
    None
):
    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    roles = {"blocking_actor": "blocker", "avoiding_vehicle": "avoider"}
    source, generated = _forced_lane_change_scenarios(
        lane_change_realized=True,
        after_successor=True,
        blocker_x=25.0,
    )

    result = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _forced_lane_change_seed(),
    )

    assert result.passed
    evidence = result.metrics["condition_evidence"]["adjacent_lane_available"]
    assert evidence["generated_compact_lane_sequence"] == [
        "lane-current",
        "lane-successor",
        "lane-left-successor",
    ]
    assert evidence["lane_change_from_lane_id"] == "lane-successor"
    assert evidence["entered_adjacent_lane_id"] == "lane-left-successor"
    assert evidence["lane_change_frame_index"] < evidence[
        "first_blocker_pass_frame_index"
    ]
    assert evidence["temporal_order_passed"] is True


def test_forced_lane_change_rejects_lane_change_after_passing_blocker() -> None:
    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    roles = {"blocking_actor": "blocker", "avoiding_vehicle": "avoider"}
    source, generated = _forced_lane_change_scenarios(
        lane_change_realized=True,
        after_successor=True,
    )

    result = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _forced_lane_change_seed(),
    )

    assert result.rejection_reasons == (
        FilterRejection.SKILL_TRIGGER_NOT_REALIZED,
    )
    assert result.metrics["failed_conditions"] == ["adjacent_lane_available"]
    evidence = result.metrics["condition_evidence"]["adjacent_lane_available"]
    assert evidence["first_blocker_pass_future_index"] == 9
    assert evidence["lane_change_future_index"] > 9
    assert evidence["temporal_order_passed"] is False
    assert evidence["reason"] == "generated_lane_change_occurs_after_blocker_pass"


def test_forced_lane_change_rejects_successor_only_generated_future_even_when_source_changes_lane() -> (
    None
):
    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    roles = {"blocking_actor": "blocker", "avoiding_vehicle": "avoider"}
    source, generated = _forced_lane_change_scenarios(lane_change_realized=False)

    result = validate_skill_trigger(
        source,
        generated,
        skill,
        roles,
        _forced_lane_change_seed(),
    )

    assert result.rejection_reasons == (
        FilterRejection.SKILL_TRIGGER_NOT_REALIZED,
    )
    assert result.metrics["failed_conditions"] == ["adjacent_lane_available"]
    evidence = result.metrics["condition_evidence"]["adjacent_lane_available"]
    assert evidence["generated_compact_lane_sequence"] == [
        "lane-current",
        "lane-successor",
    ]
    assert evidence["direct_adjacent_lane_ids"] == ["lane-left"]
    assert evidence["reason"] == "generated_future_has_no_adjacent_lane_transition"


def test_forced_lane_change_real_raw_successor_only_regression(
    real_forced_lane_change_source: Scenario,
) -> None:
    raw_path = (
        ROOT
        / "outputs"
        / "generation"
        / "counterfactual_v1"
        / "pilot"
        / "skill-pilot-v1"
        / "5c8bf958e106a4bf84dfbf2cc53a00d195dd2754e6b53623378122fac6413a61"
        / "raw"
        / "shard-00378.npz"
    )
    if not raw_path.is_file():
        pytest.skip("local accepted #4 raw regression artifacts are unavailable")

    from skilldrive.generation.assembly import materialize_overlay_scenario

    with np.load(raw_path, allow_pickle=False) as payload:
        future = payload["future_xy_global"][7]
    generated = materialize_overlay_scenario(
        real_forced_lane_change_source,
        "59889",
        future,
    )
    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    result = validate_skill_trigger(
        real_forced_lane_change_source,
        generated,
        skill,
        {"blocking_actor": "59792", "avoiding_vehicle": "59889"},
        _real_forced_lane_change_seed(),
    )

    assert result.rejection_reasons == (
        FilterRejection.SKILL_TRIGGER_NOT_REALIZED,
    )
    evidence = result.metrics["condition_evidence"]["adjacent_lane_available"]
    assert evidence["generated_compact_lane_sequence"] == [
        "353623830",
        "353623686",
    ]
    assert evidence["direct_adjacent_lane_ids"] == ["353623626"]
    assert evidence["entered_adjacent_lane_id"] is None


def test_forced_lane_change_real_latent_search_rejects_all_three_late_changes(
    real_forced_lane_change_source: Scenario,
) -> None:
    raw_path = (
        ROOT
        / "outputs"
        / "generation"
        / "counterfactual_v1"
        / "pilot"
        / "latent-search-v1"
        / "959b1e3388f0a4da6259e5fc1654435c1cf429f6b34f5c94c10efaacbcf318d1"
        / "raw"
        / "shard-00000.npz"
    )
    if not raw_path.is_file():
        pytest.skip("local latent-search forced raw artifacts are unavailable")

    from skilldrive.generation.assembly import materialize_overlay_scenario

    skill = load_skill(SKILL_DIR / "forced_lane_change_around_blockage.yaml")
    roles = {"blocking_actor": "59792", "avoiding_vehicle": "59889"}
    expected = (
        (1, 6, 41),
        (42, 6, 45),
        (54, 7, 32),
    )
    with np.load(raw_path, allow_pickle=False) as payload:
        futures = payload["future_xy_global"]
        for raw_offset, pass_future_index, lane_change_future_index in expected:
            generated = materialize_overlay_scenario(
                real_forced_lane_change_source,
                "59889",
                futures[raw_offset],
            )
            result = validate_skill_trigger(
                real_forced_lane_change_source,
                generated,
                skill,
                roles,
                _real_forced_lane_change_seed(),
            )

            assert result.rejection_reasons == (
                FilterRejection.SKILL_TRIGGER_NOT_REALIZED,
            )
            assert result.metrics["failed_conditions"] == [
                "adjacent_lane_available"
            ]
            evidence = result.metrics["condition_evidence"][
                "adjacent_lane_available"
            ]
            assert evidence["first_blocker_pass_future_index"] == pass_future_index
            assert evidence["lane_change_future_index"] == lane_change_future_index
            assert evidence["first_blocker_pass_frame_index"] == (
                50 + pass_future_index
            )
            assert evidence["lane_change_frame_index"] == (
                50 + lane_change_future_index
            )
            assert evidence["temporal_order_passed"] is False
            assert (
                evidence["reason"]
                == "generated_lane_change_occurs_after_blocker_pass"
            )


def test_prepare_risk_context_extracts_exposure_stage_and_polygon_without_mutation() -> (
    None
):
    time = np.arange(110, dtype=np.float64) * 0.1

    target_xy = np.column_stack((time, np.zeros(110)))
    slow_xy = np.column_stack((time + 10.0, np.zeros(110)))
    cut_out_y = np.zeros(110)
    cut_out_y[50:] = np.linspace(0.0, 2.5, 60)
    cut_out_xy = np.column_stack((time + 5.0, cut_out_y))
    cut_out_source = _scenario(
        [
            _agent("cut-out", np.column_stack((time + 5.0, np.zeros(110)))),
            _agent("target", target_xy),
            _agent("slow", slow_xy),
        ],
        scenario_id="cut-out-source",
    )
    cut_out_generated = _scenario(
        [
            _agent("cut-out", cut_out_xy),
            _agent("target", target_xy),
            _agent("slow", slow_xy),
        ],
        scenario_id="cut-out-generated",
    )
    cut_out_skill = load_skill(SKILL_DIR / "cut_out_reveals_slow_vehicle.yaml")
    cut_out_context = prepare_risk_context(
        cut_out_source,
        cut_out_generated,
        cut_out_skill,
        {
            "cut_out_vehicle": "cut-out",
            "target_vehicle": "target",
            "slow_vehicle": "slow",
        },
        {"missing_generation_conditions": list(cut_out_skill.trigger["conditions"])},
    )
    exposure = cut_out_context.metadata[RISK_CONTEXT_METADATA_KEY][
        "cut_out_reveals_slow_vehicle"
    ]
    assert 50 <= exposure["exposure_frame_index"] < 110
    assert exposure["event_source"] == "generated_exact_role_lateral_departure"
    assert RISK_CONTEXT_METADATA_KEY not in cut_out_generated.metadata
    cut_out_risk = evaluate_skill_risk(
        cut_out_context,
        cut_out_skill.skill_id,
        {
            "cut_out_vehicle": "cut-out",
            "target_vehicle": "target",
            "slow_vehicle": "slow",
        },
    )
    assert cut_out_risk.reason is not RiskReason.REQUIRED_CONTEXT_MISSING

    responder_xy = np.column_stack((2.0 * time, np.zeros(110)))
    cut_in_y = np.full(110, 3.0)
    cut_in_y[50:61] = np.linspace(3.0, 0.0, 11)
    cut_in_y[61:] = 0.0
    cut_in_x = np.empty(110, dtype=np.float64)
    cut_in_x[0] = 5.0
    for frame in range(1, 110):
        speed = 2.0 if frame < 66 else 0.5
        cut_in_x[frame] = cut_in_x[frame - 1] + speed * 0.1
    cut_in_source = _scenario(
        [
            _agent("cut-in", np.column_stack((2.0 * time + 5.0, np.full(110, 3.0)))),
            _agent("responder", responder_xy),
        ],
        scenario_id="cut-in-source",
    )
    cut_in_generated = _scenario(
        [
            _agent("cut-in", np.column_stack((cut_in_x, cut_in_y))),
            _agent("responder", responder_xy),
        ],
        scenario_id="cut-in-generated",
    )
    cut_in_skill = load_skill(SKILL_DIR / "cut_in_then_brake.yaml")
    cut_in_context = prepare_risk_context(
        cut_in_source,
        cut_in_generated,
        cut_in_skill,
        {
            "cut_in_braking_vehicle": "cut-in",
            "responding_vehicle": "responder",
        },
        {"missing_generation_conditions": list(cut_in_skill.trigger["conditions"])},
    )
    stage = cut_in_context.metadata[RISK_CONTEXT_METADATA_KEY]["cut_in_then_brake"]
    assert 50 <= stage["stage_start_frame_index"] < 110
    assert stage["event_source"] == "generated_exact_role_post_cut_in_brake_onset"
    cut_in_risk = evaluate_skill_risk(
        cut_in_context,
        cut_in_skill.skill_id,
        {
            "cut_in_braking_vehicle": "cut-in",
            "responding_vehicle": "responder",
        },
    )
    assert cut_in_risk.reason is not RiskReason.REQUIRED_CONTEXT_MISSING

    intersection_skill = load_skill(SKILL_DIR / "intersection_blocking_vehicle.yaml")
    intersection = _scenario(
        [_constant_agent("blocker", 0.0), _constant_agent("crossing", 5.0)],
        scenario_id="intersection",
    )
    polygon_context = prepare_risk_context(
        intersection,
        intersection,
        intersection_skill,
        {"blocking_vehicle": "blocker", "crossing_vehicle": "crossing"},
        {"conflict_point_xy": [1.0, 2.0]},
    )
    polygon_item = polygon_context.metadata[RISK_CONTEXT_METADATA_KEY][
        "intersection_blocking_vehicle"
    ]
    assert polygon_item["conflict_point_xy"] == [1.0, 2.0]
    assert len(polygon_item["conflict_area_polygon_xy"]) == 4
    assert polygon_item["event_source"] == "frozen_seed_conflict_point"
    polygon_risk = evaluate_skill_risk(
        polygon_context,
        intersection_skill.skill_id,
        {"blocking_vehicle": "blocker", "crossing_vehicle": "crossing"},
    )
    assert polygon_risk.reason is not RiskReason.REQUIRED_CONTEXT_MISSING

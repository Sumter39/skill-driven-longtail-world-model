from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from skilldrive.filtering.risk import (
    RISK_CONTEXT_METADATA_KEY,
    SKILL_RISK_CALCULATORS,
    RiskEvaluation,
    RiskReason,
    RiskStatus,
    check_target_risk,
    evaluate_skill_risk,
)
from skilldrive.filtering.contracts import FilterRejection
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.skills.loader import load_skill


ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = ROOT / "configs" / "skills"


def _agent(
    track_id: str,
    positions: np.ndarray,
    *,
    velocities: np.ndarray | None = None,
    headings: np.ndarray | None = None,
    object_type: str = "vehicle",
) -> AgentTrack:
    points = np.asarray(positions, dtype=np.float64)
    steps = len(points)
    if velocities is None:
        velocities = np.zeros((steps, 2), dtype=np.float64)
    if headings is None:
        headings = np.zeros(steps, dtype=np.float64)
    observed = np.zeros(steps, dtype=bool)
    observed[: min(50, steps)] = True
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=points,
        velocities=velocities,
        headings=headings,
        observed_mask=observed,
        is_focal=False,
    )


def _constant_agent(
    track_id: str,
    x: float,
    y: float = 0.0,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    object_type: str = "vehicle",
) -> AgentTrack:
    return _agent(
        track_id,
        np.tile([x, y], (110, 1)),
        velocities=np.tile([vx, vy], (110, 1)),
        object_type=object_type,
    )


def _scenario(
    agents: list[AgentTrack],
    *,
    metadata: dict | None = None,
) -> Scenario:
    agents[0].is_focal = True
    return Scenario(
        scenario_id="risk-test",
        city_name="test-city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id=agents[0].track_id,
        agents=agents,
        map_polylines=[],
        metadata={} if metadata is None else metadata,
    )


def _formal_catalog_ids() -> set[str]:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    return {
        item["skill_id"]
        for entries in catalog["families"].values()
        for item in entries
    }


def test_registry_explicitly_covers_all_34_formal_skills_and_17_metrics() -> None:
    assert len(SKILL_RISK_CALCULATORS) == 34
    assert set(SKILL_RISK_CALCULATORS) == _formal_catalog_ids()
    assert len({spec.metric for spec in SKILL_RISK_CALCULATORS.values()}) == 17

    for skill_id, spec in SKILL_RISK_CALCULATORS.items():
        skill = yaml.safe_load((SKILL_DIR / f"{skill_id}.yaml").read_text(encoding="utf-8"))
        assert spec.skill_id == skill_id
        assert spec.metric == skill["risk_definition"]["metric"]
        assert set(spec.required_roles) == set(skill["actors"]["generated_roles"])
        assert spec.formula_version.endswith(".v1")
        assert spec.interaction_pairs


def test_risk_evaluation_rejects_non_finite_values_and_audit_evidence() -> None:
    common = {
        "skill_id": "skill",
        "metric": "metric",
        "unit": "s",
        "formula_version": "metric.v1",
        "role_track_ids": {"actor": "track"},
    }
    with pytest.raises(ValueError, match="finite"):
        RiskEvaluation(
            **common,
            status=RiskStatus.COMPUTED,
            value=math.inf,
        )
    with pytest.raises(ValueError, match="non-finite"):
        RiskEvaluation(
            **common,
            status=RiskStatus.NO_EVENT,
            value=None,
            reason=RiskReason.NO_PREDICTED_COLLISION,
            details={"nested": [1.0, float("nan")]},
        )

    result = RiskEvaluation(
        **common,
        status=RiskStatus.COMPUTED,
        value=1.0,
        details={"point": [1.0, 2.0]},
    )
    with pytest.raises(TypeError):
        result.details["new"] = 1
    json.dumps(result.to_dict(), allow_nan=False)


def test_pet_is_recomputed_from_overlay_paths_and_ignores_seed_metadata() -> None:
    relative_time = (np.arange(110, dtype=np.float64) - 49.0) * 0.1
    first_positions = np.column_stack((relative_time - 1.0, np.zeros(110)))
    second_positions = np.column_stack((np.zeros(110), relative_time - 2.0))
    scenario = _scenario(
        [
            _agent("first", first_positions, velocities=np.tile([1.0, 0.0], (110, 1))),
            _agent("second", second_positions, velocities=np.tile([0.0, 1.0], (110, 1))),
        ],
        metadata={"seed_risk_value": 99.0, "seed_risk_metric": "not_used"},
    )

    result = evaluate_skill_risk(
        scenario,
        "crossing_path_conflict",
        {"first_vehicle": "first", "second_vehicle": "second"},
    )

    assert result.status is RiskStatus.COMPUTED
    assert result.metric == "post_encroachment_time"
    assert result.value == pytest.approx(1.0)
    assert result.details["first_arrival_time_s"] == pytest.approx(1.0)
    assert result.details["second_arrival_time_s"] == pytest.approx(2.0)
    assert "seed_risk_value" not in result.to_dict()["details"]


def test_collinear_conflict_is_unavailable_instead_of_fabricated() -> None:
    relative_time = (np.arange(110, dtype=np.float64) - 49.0) * 0.1
    first = np.column_stack((relative_time, np.zeros(110)))
    second = np.column_stack((relative_time + 1.0, np.zeros(110)))
    scenario = _scenario([_agent("merge", first), _agent("main", second)])

    result = evaluate_skill_risk(
        scenario,
        "ramp_merge_small_gap",
        {"merging_vehicle": "merge", "mainline_vehicle": "main"},
    )

    assert result.status is RiskStatus.UNAVAILABLE
    assert result.reason is RiskReason.NON_UNIQUE_PATH_CONFLICT
    assert result.value is None


def test_longitudinal_gap_headway_and_stopping_margin_have_explicit_boundaries() -> None:
    leader = _constant_agent("leader", 10.0, vx=0.0)
    follower = _constant_agent("follower", 0.0, vx=10.0)
    scenario = _scenario([leader, follower])

    gap = evaluate_skill_risk(
        scenario,
        "slow_lead_blockage",
        {"slow_leader": "leader", "follower": "follower"},
    )
    headway = evaluate_skill_risk(
        scenario,
        "short_headway_following",
        {"leader": "leader", "close_follower": "follower"},
    )
    stopping = evaluate_skill_risk(
        scenario,
        "lead_sudden_stop",
        {"stopping_leader": "leader", "follower": "follower"},
    )

    assert gap.value == pytest.approx(10.0)
    assert headway.value == pytest.approx(1.0)
    assert stopping.value == pytest.approx(10.0 - 100.0 / 12.0)
    assert stopping.details["assumed_equal_deceleration_mps2"] == 6.0


def test_combined_clearance_uses_only_declared_exact_role_pairs() -> None:
    scenario = _scenario(
        [
            _constant_agent("squeezed", 0.0),
            _constant_agent("front", 3.0),
            _constant_agent("rear", -5.0),
        ]
    )
    result = evaluate_skill_risk(
        scenario,
        "multi_vehicle_gap_squeeze",
        {
            "squeezed_vehicle": "squeezed",
            "front_pressure_vehicle": "front",
            "rear_pressure_vehicle": "rear",
        },
    )

    assert result.status is RiskStatus.COMPUTED
    assert result.value == pytest.approx(3.0)
    assert result.details["selected_pair"] == "squeezed_vehicle->front_pressure_vehicle"
    assert result.details["physical_footprint_clearance"] is False


def test_no_collision_and_exact_contact_never_emit_infinity() -> None:
    no_event = evaluate_skill_risk(
        _scenario(
            [
                _constant_agent("turning", 0.0, vx=-1.0),
                _constant_agent("conflict", 10.0, vx=1.0),
            ]
        ),
        "abrupt_u_turn_conflict",
        {"u_turning_vehicle": "turning", "conflicting_vehicle": "conflict"},
    )
    contact = evaluate_skill_risk(
        _scenario([_constant_agent("turning", 0.0), _constant_agent("conflict", 0.0)]),
        "abrupt_u_turn_conflict",
        {"u_turning_vehicle": "turning", "conflicting_vehicle": "conflict"},
    )

    assert no_event.status is RiskStatus.NO_EVENT
    assert no_event.reason is RiskReason.NO_PREDICTED_COLLISION
    assert no_event.value is None
    json.dumps(no_event.to_dict(), allow_nan=False)
    assert contact.status is RiskStatus.COMPUTED
    assert contact.value == 0.0


def test_complex_conflict_area_metrics_require_explicit_polygon_context() -> None:
    agents = [_constant_agent("creep", 2.0), _constant_agent("crossing", 0.0)]
    missing = evaluate_skill_risk(
        _scenario(agents),
        "intersection_creep_conflict",
        {"creeping_vehicle": "creep", "crossing_vehicle": "crossing"},
    )
    assert missing.status is RiskStatus.UNAVAILABLE
    assert missing.reason is RiskReason.REQUIRED_CONTEXT_MISSING

    polygon = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    metadata = {
        RISK_CONTEXT_METADATA_KEY: {
            "intersection_creep_conflict": {
                "conflict_area_polygon_xy": polygon,
            },
            "intersection_blocking_vehicle": {
                "conflict_area_polygon_xy": polygon,
            },
        }
    }
    margin = evaluate_skill_risk(
        _scenario(
            [_constant_agent("creep", 2.0), _constant_agent("crossing", 0.0)],
            metadata=metadata,
        ),
        "intersection_creep_conflict",
        {"creeping_vehicle": "creep", "crossing_vehicle": "crossing"},
    )
    overlap = evaluate_skill_risk(
        _scenario(
            [_constant_agent("blocker", 0.0), _constant_agent("crossing", 0.0)],
            metadata=metadata,
        ),
        "intersection_blocking_vehicle",
        {"blocking_vehicle": "blocker", "crossing_vehicle": "crossing"},
    )

    assert margin.value == pytest.approx(1.0)
    assert margin.details["polygon_vertex_count"] == 4
    assert overlap.value == pytest.approx(5.9)
    assert overlap.details["sampling_rule"] == "left_endpoint_zero_order_hold"


def test_first_intrusion_uses_polygon_transition_and_finite_point_ttc() -> None:
    pedestrian_positions = np.tile([0.0, 0.0], (110, 1))
    pedestrian_positions[:50] = [-2.0, 0.0]
    pedestrian = _agent("pedestrian", pedestrian_positions, object_type="pedestrian")
    vehicle = _constant_agent("vehicle", 0.0)
    polygon = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    scenario = _scenario(
        [pedestrian, vehicle],
        metadata={
            RISK_CONTEXT_METADATA_KEY: {
                "roadside_pedestrian_emergence": {
                    "conflict_area_polygon_xy": polygon,
                }
            }
        },
    )

    result = evaluate_skill_risk(
        scenario,
        "roadside_pedestrian_emergence",
        {"emerging_pedestrian": "pedestrian", "responding_vehicle": "vehicle"},
    )

    assert result.status is RiskStatus.COMPUTED
    assert result.value == 0.0
    assert result.details["intrusion_frame"] == 50


def test_contextual_stage_metrics_do_not_fall_back_to_seed_proxies() -> None:
    scenario = _scenario(
        [
            _constant_agent("cut-out", 5.0),
            _constant_agent("target", 0.0, vx=2.0),
            _constant_agent("slow", 10.0),
        ],
        metadata={"seed_risk_value": 1.0},
    )
    result = evaluate_skill_risk(
        scenario,
        "cut_out_reveals_slow_vehicle",
        {
            "cut_out_vehicle": "cut-out",
            "target_vehicle": "target",
            "slow_vehicle": "slow",
        },
    )

    assert result.status is RiskStatus.UNAVAILABLE
    assert result.reason is RiskReason.REQUIRED_CONTEXT_MISSING
    assert result.value is None


def test_role_contract_and_non_finite_tracks_fail_auditably() -> None:
    scenario = _scenario([_constant_agent("object", 0.0), _constant_agent("vehicle", 2.0)])
    mismatch = evaluate_skill_risk(
        scenario,
        "static_object_avoidance",
        {"static_object": "object"},
    )
    assert mismatch.reason is RiskReason.ROLE_CONTRACT_MISMATCH

    invalid_positions = np.full((110, 2), np.nan)
    invalid_positions[:50] = [0.0, 0.0]
    non_finite = evaluate_skill_risk(
        _scenario(
            [
                _agent("object", invalid_positions),
                _agent("vehicle", invalid_positions.copy()),
            ]
        ),
        "static_object_avoidance",
        {"static_object": "object", "avoiding_vehicle": "vehicle"},
    )
    assert non_finite.status is RiskStatus.UNAVAILABLE
    assert non_finite.reason is RiskReason.INSUFFICIENT_VALID_SAMPLES
    json.dumps(non_finite.to_dict(), allow_nan=False)

    with pytest.raises(KeyError, match="unknown formal skill_id"):
        evaluate_skill_risk(scenario, "missing_skill", {})


def test_target_risk_check_requires_finite_value_inside_frozen_range() -> None:
    skill = load_skill(SKILL_DIR / "construction_object_lane_blockage.yaml")
    computed = RiskEvaluation(
        skill_id=skill.skill_id,
        metric="minimum_object_clearance",
        unit="m",
        formula_version="test.v1",
        status=RiskStatus.COMPUTED,
        value=2.0,
        role_track_ids={
            "construction_object": "object",
            "responding_vehicle": "vehicle",
        },
    )
    assert check_target_risk(skill, computed).passed

    outside = RiskEvaluation(
        skill_id=skill.skill_id,
        metric="minimum_object_clearance",
        unit="m",
        formula_version="test.v1",
        status=RiskStatus.COMPUTED,
        value=5.0,
        role_track_ids=computed.role_track_ids,
    )
    assert check_target_risk(skill, outside).rejection_reasons == (
        FilterRejection.RISK_OUT_OF_TARGET_RANGE,
    )

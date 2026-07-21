from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from skilldrive.data.cvae_samples import (
    BASE_TARGET_ROLE,
    CONTEXT_ROLE,
    FUTURE_STEPS,
    HISTORY_STEPS,
    MAX_ACTORS,
    MAX_MAP_POINTS,
    MAX_MAP_POLYLINES,
    NONE_SKILL_ID,
    SampleSpec,
    build_cvae_schema,
    make_base_sample_spec,
    observed_sample_specs,
    tensorize_scenario,
)
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.seeds.records import SeedRecord


def _agent(
    track_id: str,
    *,
    anchor_xy: tuple[float, float],
    heading: float = 0.0,
    speed_per_step: float = 1.0,
    object_type: str = "vehicle",
    is_focal: bool = False,
) -> AgentTrack:
    steps = HISTORY_STEPS + FUTURE_STEPS
    offsets = np.arange(steps, dtype=np.float64) - (HISTORY_STEPS - 1)
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    positions = np.asarray(anchor_xy, dtype=np.float64) + offsets[:, None] * (
        speed_per_step * direction
    )
    velocities = np.repeat((speed_per_step * direction)[None, :], steps, axis=0)
    return AgentTrack(
        track_id=track_id,
        object_type=object_type,
        positions=positions,
        velocities=velocities,
        headings=np.full(steps, heading, dtype=np.float64),
        observed_mask=np.arange(steps) < HISTORY_STEPS,
        is_focal=is_focal,
    )


def _maps(origin: tuple[float, float]) -> list[MapPolyline]:
    x, y = origin
    return [
        MapPolyline(
            polyline_id="map-b",
            polyline_type="lane_centerline",
            points=np.array([[x - 10.0, y + 2.0], [x + 10.0, y + 2.0]]),
        ),
        MapPolyline(
            polyline_id="map-a",
            polyline_type="lane_centerline",
            points=np.array([[x - 10.0, y - 2.0], [x + 10.0, y - 2.0]]),
        ),
        MapPolyline(
            polyline_id="crosswalk",
            polyline_type="pedestrian_crossing",
            points=np.array(
                [[x + 5.0, y], [np.nan, np.nan], [x + 5.0, y + 2.0]]
            ),
        ),
        MapPolyline(
            polyline_id="drivable",
            polyline_type="drivable_area",
            points=np.array(
                [
                    [x + 10.0, y - 3.0],
                    [x + 14.0, y - 3.0],
                    [x + 14.0, y + 3.0],
                    [x + 10.0, y - 3.0],
                ]
            ),
        ),
        MapPolyline(
            polyline_id="ignored-boundary",
            polyline_type="lane_boundary_left",
            points=np.array([[x, y], [x + 1.0, y]]),
        ),
        MapPolyline(
            polyline_id="far-away",
            polyline_type="lane_centerline",
            points=np.array([[x + 200.0, y], [x + 210.0, y]]),
        ),
    ]


def _scenario(*, reverse: bool = False) -> Scenario:
    target = _agent(
        "target",
        anchor_xy=(10.0, 20.0),
        heading=np.pi / 2,
        is_focal=False,
    )
    responder = _agent(
        "responder",
        anchor_xy=(10.0, 28.0),
        heading=np.pi / 2,
    )
    neighbor_a = _agent(
        "neighbor-a",
        anchor_xy=(0.0, 20.0),
        object_type="pedestrian",
    )
    neighbor_b = _agent(
        "neighbor-b",
        anchor_xy=(20.0, 20.0),
        object_type="cyclist",
    )
    focal = _agent(
        "focal",
        anchor_xy=(60.0, 20.0),
        is_focal=True,
    )
    far_neighbor = _agent(
        "far-neighbor",
        anchor_xy=(200.0, 20.0),
    )
    agents = [far_neighbor, focal, neighbor_b, target, neighbor_a, responder]
    maps = _maps((10.0, 20.0))
    if reverse:
        agents.reverse()
        maps.reverse()
    return Scenario(
        scenario_id="scene-a",
        city_name="test-city",
        timestamps=np.arange(HISTORY_STEPS + FUTURE_STEPS, dtype=np.int64),
        focal_track_id="focal",
        agents=agents,
        map_polylines=maps,
    )


def _short_headway_scenario(
    *,
    future_gap_m: float = 3.0,
    future_lateral_offset_m: float = 0.0,
) -> Scenario:
    scenario = _scenario()
    leader = _agent(
        "target",
        anchor_xy=(13.0, 20.0),
        heading=0.0,
        speed_per_step=3.0,
    )
    follower = _agent(
        "responder",
        anchor_xy=(10.0, 20.0),
        heading=0.0,
        speed_per_step=3.0,
    )
    positions = follower.positions.copy()
    positions[HISTORY_STEPS:] = leader.positions[HISTORY_STEPS:] - np.array(
        [future_gap_m, future_lateral_offset_m]
    )
    follower = replace(follower, positions=positions)
    replacements = {"target": leader, "responder": follower}
    return replace(
        scenario,
        agents=[replacements.get(agent.track_id, agent) for agent in scenario.agents],
    )


def _short_headway_spec() -> SampleSpec:
    return SampleSpec(
        scenario_id="scene-a",
        target_track_id="responder",
        skill_id="short_headway_following",
        skill_supervision_mask=True,
        responder_track_id="responder",
        role_track_ids=(
            ("leader", "target"),
            ("close_follower", "responder"),
        ),
    )


def _record(
    *,
    skill_id: str,
    mode: str,
    trigger_score: float = 0.8,
    scenario_id: str = "scene-a",
    initiator: str = "target",
    responder: str = "responder",
) -> SeedRecord:
    if skill_id in {"slow_lead_blockage", "lead_hard_brake"}:
        roles = {"slow_leader": initiator, "follower": responder}
    elif skill_id == "short_headway_following":
        roles = {"leader": initiator, "close_follower": responder}
    else:
        roles = {"initiator": initiator, "responder": responder}
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=initiator,
        responder_track_id=responder,
        role_track_ids=roles,
        trigger_score=trigger_score,
        seed_risk_metric="minimum_distance",
        seed_risk_value=3.0,
        target_risk_definition={
            "metric": "minimum_distance",
            "target_range": [1.0, 4.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={
            "detection_mode": mode,
            "missing_generation_conditions": [] if mode == "observed_trigger" else ["x"],
        },
        sampled_parameters={"unused": 1.0},
    )


@pytest.fixture(scope="module")
def cvae_schema():
    return build_cvae_schema()


def test_schema_contains_only_34_formal_skills_and_none(cvae_schema) -> None:
    assert len(cvae_schema.formal_skill_ids) == 34
    assert len(cvae_schema.candidate_skill_ids) == 5
    assert len(cvae_schema.skill_vocabulary.tokens) == 35
    assert cvae_schema.skill_vocabulary.tokens[0] == NONE_SKILL_ID
    assert set(cvae_schema.formal_skill_ids).isdisjoint(cvae_schema.candidate_skill_ids)
    assert "lead_hard_brake" in cvae_schema.candidate_skill_ids
    assert "lead_hard_brake" not in cvae_schema.skill_vocabulary.tokens
    assert cvae_schema.parameter_schema.dimension == 106


def test_parameter_schema_is_skill_local_and_encodes_masks(cvae_schema) -> None:
    values, mask = cvae_schema.parameter_schema.encode(
        "slow_lead_blockage",
        {
            "leader_speed_scale": 0.25,
            "blockage_duration_s": 4.0,
            "initial_gap_m": 20.0,
        },
    )
    assert values.shape == (106,)
    assert mask.sum() == 3
    assert np.isfinite(values).all()

    categorical_values, categorical_mask = cvae_schema.parameter_schema.encode(
        "merge_without_yield",
        {
            "merge_onset_s": 2.0,
            "accepted_gap_s": 1.5,
            "priority_role": "responder",
        },
    )
    definitions = {
        item.name: item
        for item in cvae_schema.parameter_schema.definitions
        if item.skill_id == "merge_without_yield"
    }
    category = definitions["priority_role"]
    np.testing.assert_array_equal(
        categorical_values[list(category.indices)],
        [0.0, 1.0],
    )
    assert categorical_mask[list(category.indices)].all()
    assert categorical_mask.sum() == 4

    with pytest.raises(ValueError, match="formal skill"):
        cvae_schema.parameter_schema.encode("lead_hard_brake", {})


def test_observed_records_become_samples_and_compatible_records_do_not(cvae_schema) -> None:
    lower_score = _record(
        skill_id="slow_lead_blockage",
        mode="observed_trigger",
        trigger_score=0.6,
    )
    higher_score = replace(lower_score, trigger_score=0.9)
    compatible = _record(skill_id="cut_in_then_brake", mode="compatible_seed")

    specs = observed_sample_specs([compatible, lower_score, higher_score], cvae_schema)

    assert len(specs) == 1
    assert specs[0].skill_id == "slow_lead_blockage"
    assert specs[0].target_track_id == "target"
    assert specs[0].skill_supervision_mask is True
    assert specs[0].trigger_score == 0.9
    assert dict(specs[0].role_track_ids) == {
        "follower": "responder",
        "slow_leader": "target",
    }

    with pytest.raises(ValueError, match="candidate skill"):
        observed_sample_specs(
            [_record(skill_id="lead_hard_brake", mode="observed_trigger")],
            cvae_schema,
        )


def test_short_headway_observed_sample_targets_close_follower_and_requires_roles(
    cvae_schema,
) -> None:
    record = _record(
        skill_id="short_headway_following",
        mode="observed_trigger",
    )

    spec = observed_sample_specs([record], cvae_schema)[0]

    assert spec.target_track_id == "responder"
    assert spec.target_track_id == dict(spec.role_track_ids)["close_follower"]
    with pytest.raises(
        ValueError,
        match="requires leader and close_follower roles",
    ):
        observed_sample_specs(
            [
                replace(
                    record,
                    role_track_ids={
                        "initiator": "target",
                        "responder": "responder",
                    },
                )
            ],
            cvae_schema,
        )


def test_short_headway_supervision_accepts_sustained_same_lane_future(
    cvae_schema,
) -> None:
    sample = tensorize_scenario(
        _short_headway_scenario(),
        _short_headway_spec(),
        cvae_schema,
    )

    assert sample.skill_supervision_mask


def test_short_headway_supervision_rejects_larger_future_gap(cvae_schema) -> None:
    with pytest.raises(ValueError, match="prediction frames 50-109"):
        tensorize_scenario(
            _short_headway_scenario(future_gap_m=10.0),
            _short_headway_spec(),
            cvae_schema,
        )


def test_short_headway_supervision_rejects_large_future_lateral_offset(
    cvae_schema,
) -> None:
    with pytest.raises(ValueError, match="prediction frames 50-109"):
        tensorize_scenario(
            _short_headway_scenario(future_lateral_offset_m=100.0),
            _short_headway_spec(),
            cvae_schema,
        )


def test_arbitrary_target_anchor_roles_and_fixed_shapes(cvae_schema) -> None:
    scenario = _scenario()
    spec = observed_sample_specs(
        [_record(skill_id="slow_lead_blockage", mode="observed_trigger")],
        cvae_schema,
    )[0]

    sample = tensorize_scenario(scenario, spec, cvae_schema)

    assert sample.actor_history.shape == (MAX_ACTORS, HISTORY_STEPS, 6)
    assert sample.target_future.shape == (FUTURE_STEPS, 2)
    assert sample.map_polylines.shape == (
        MAX_MAP_POLYLINES,
        MAX_MAP_POINTS,
        4,
    )
    assert sample.actor_track_ids[:4] == (
        "target",
        "responder",
        "neighbor-a",
        "neighbor-b",
    )
    assert "far-neighbor" not in sample.actor_track_ids
    assert sample.target_actor_index == 0
    np.testing.assert_allclose(sample.actor_history[0, -1, :2], [0.0, 0.0])
    np.testing.assert_allclose(sample.target_future[0], [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(sample.anchor_origin_global, [10.0, 20.0])
    assert sample.skill_supervision_mask
    assert sample.skill_id == cvae_schema.skill_vocabulary.encode("slow_lead_blockage")
    assert sample.actor_role_id[0] == cvae_schema.role_vocabulary.encode("slow_leader")
    assert sample.actor_role_id[1] == cvae_schema.role_vocabulary.encode("follower")
    assert sample.actor_type_id[2] == cvae_schema.actor_type_vocabulary.encode(
        "pedestrian"
    )
    assert not sample.parameter_mask.any()
    assert not sample.skill_parameters.any()


def test_masks_are_created_before_nan_replacement_and_map_scope_is_explicit(
    cvae_schema,
) -> None:
    scenario = _scenario()
    neighbor = next(agent for agent in scenario.agents if agent.track_id == "neighbor-a")
    neighbor.positions[10] = np.nan
    base = make_base_sample_spec(scenario)

    sample = tensorize_scenario(scenario, base, cvae_schema)

    neighbor_slot = sample.actor_track_ids.index("neighbor-a")
    assert not sample.actor_time_mask[neighbor_slot, 10]
    np.testing.assert_array_equal(sample.actor_history[neighbor_slot, 10], 0.0)
    assert np.isfinite(sample.actor_history).all()
    assert sample.actor_role_id[0] == cvae_schema.role_vocabulary.encode(BASE_TARGET_ROLE)
    context_slot = sample.actor_track_ids.index("target")
    assert sample.actor_role_id[context_slot] == cvae_schema.role_vocabulary.encode(
        CONTEXT_ROLE
    )

    selected_map_ids = {value for value in sample.map_polyline_ids if value}
    assert "ignored-boundary" not in selected_map_ids
    assert "far-away" not in selected_map_ids
    assert {"map-a", "map-b", "crosswalk", "drivable"} == selected_map_ids
    crosswalk_slot = sample.map_polyline_ids.index("crosswalk")
    assert sample.map_point_mask[crosswalk_slot].sum() == 2
    assert np.isfinite(sample.map_polylines).all()
    assert sample.map_type_id[crosswalk_slot] == cvae_schema.map_type_vocabulary.encode(
        "pedestrian_crossing"
    )
    assert sample.map_clip_statistics.eligible_polylines == 4
    assert sample.map_clip_statistics.retained_polylines == 4
    assert sample.map_clip_statistics.dropped_polylines_due_to_limit == 0
    assert sample.map_clip_statistics.original_in_radius_points == 10
    assert sample.map_clip_statistics.retained_in_radius_points == 10
    assert sample.map_clip_statistics.resampled_polylines_due_to_point_limit == 0
    assert sample.map_clip_statistics.excess_input_points_over_point_limit == 0


def test_map_clip_statistics_report_polyline_and_point_limits(cvae_schema) -> None:
    scenario = _scenario()
    dense_points = np.column_stack(
        (np.linspace(0.0, 20.0, MAX_MAP_POINTS + 5), np.full(MAX_MAP_POINTS + 5, 20.0))
    )
    scenario.map_polylines = [
        MapPolyline(
            polyline_id=f"dense-{index:03d}",
            polyline_type="lane_centerline",
            points=dense_points.copy(),
        )
        for index in range(MAX_MAP_POLYLINES + 2)
    ]

    sample = tensorize_scenario(scenario, make_base_sample_spec(scenario), cvae_schema)

    statistics = sample.map_clip_statistics
    assert statistics.eligible_polylines == MAX_MAP_POLYLINES + 2
    assert statistics.retained_polylines == MAX_MAP_POLYLINES
    assert statistics.dropped_polylines_due_to_limit == 2
    assert statistics.original_in_radius_points == (MAX_MAP_POLYLINES + 2) * 25
    assert statistics.retained_in_radius_points == MAX_MAP_POLYLINES * 25
    assert statistics.resampled_polylines_due_to_point_limit == MAX_MAP_POLYLINES
    assert statistics.excess_input_points_over_point_limit == MAX_MAP_POLYLINES * 5
    assert int(sample.map_polyline_mask.sum()) == MAX_MAP_POLYLINES
    assert int(sample.map_point_mask.sum()) == MAX_MAP_POLYLINES * MAX_MAP_POINTS


def test_actor_and_map_order_are_independent_of_input_order(cvae_schema) -> None:
    spec = SampleSpec(
        scenario_id="scene-a",
        target_track_id="target",
        skill_id="slow_lead_blockage",
        skill_supervision_mask=True,
        responder_track_id="responder",
        role_track_ids=(
            ("slow_leader", "target"),
            ("follower", "responder"),
        ),
    )
    forward = tensorize_scenario(_scenario(reverse=False), spec, cvae_schema)
    reversed_input = tensorize_scenario(_scenario(reverse=True), spec, cvae_schema)

    assert forward.actor_track_ids == reversed_input.actor_track_ids
    assert forward.map_polyline_ids == reversed_input.map_polyline_ids
    np.testing.assert_array_equal(forward.actor_history, reversed_input.actor_history)
    np.testing.assert_array_equal(forward.map_polylines, reversed_input.map_polylines)


def test_incomplete_target_future_is_rejected(cvae_schema) -> None:
    scenario = _scenario()
    target = next(agent for agent in scenario.agents if agent.track_id == "target")
    target.positions[HISTORY_STEPS + 5] = np.nan
    spec = SampleSpec(scenario_id="scene-a", target_track_id="target")

    with pytest.raises(ValueError, match="60 finite positions"):
        tensorize_scenario(scenario, spec, cvae_schema)

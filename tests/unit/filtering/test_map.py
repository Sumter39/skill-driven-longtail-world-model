from __future__ import annotations

import numpy as np
import pytest

import skilldrive.filtering.prepared_map as prepared_map_module
from skilldrive.filtering.contracts import FilterRejection
from skilldrive.filtering.map import (
    PreparedMapGeometry,
    check_map_compliance,
    check_map_compliance_batch,
    evaluate_drivable_area,
    point_in_polygon,
    prepare_map_geometry,
)
from skilldrive.filtering.prepared_map import (
    PreparedMapVerificationSession,
    points_in_drivable_area,
    project_points_to_lanes,
    project_points_to_lanes_within_distance,
)
from skilldrive.generation.config import load_filter_config
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario


def _scenario() -> Scenario:
    agent = AgentTrack(
        track_id="target",
        object_type="vehicle",
        positions=np.zeros((110, 2)),
        velocities=np.zeros((110, 2)),
        headings=np.zeros(110),
        observed_mask=np.ones(110, dtype=bool),
        is_focal=True,
    )
    return Scenario(
        scenario_id="scene",
        city_name="PIT",
        timestamps=np.arange(110, dtype=np.int64),
        focal_track_id="target",
        agents=[agent],
        map_polylines=[
            MapPolyline(
                polyline_id="area",
                polyline_type="drivable_area",
                points=np.array(
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
                ),
            )
        ],
    )


def test_point_in_polygon_includes_boundary() -> None:
    polygon = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    assert point_in_polygon(np.array([1.0, 1.0]), polygon)
    assert point_in_polygon(np.array([0.0, 1.0]), polygon)
    assert not point_in_polygon(np.array([3.0, 1.0]), polygon)


def test_required_drivable_area_rejects_outside_points() -> None:
    result = evaluate_drivable_area(
        _scenario(),
        np.array([[1.0, 1.0], [5.0, 5.0], [11.0, 5.0]]),
        required=True,
        minimum_inside_fraction=1.0,
    )
    assert result.passed is False
    assert result.inside_fraction == 2 / 3
    assert result.outside_indices == (2,)


def _road_scenario(*, object_type: str = "vehicle", heading: float = 0.0) -> Scenario:
    scenario = _scenario()
    target = scenario.agents[0]
    target.object_type = object_type
    target.positions[:, 0] = np.linspace(1.0, 9.0, 110)
    target.positions[:, 1] = 2.0
    target.headings[:] = heading
    scenario.map_polylines.extend(
        [
            MapPolyline(
                polyline_id="lane-a",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 2.0], [10.0, 2.0]]),
                direction="vehicle",
                lane_id="a",
            ),
            MapPolyline(
                polyline_id="lane-b",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 6.0], [10.0, 6.0]]),
                direction="vehicle",
                lane_id="b",
            ),
        ]
    )
    return scenario


def _assert_filter_checks_equal(actual, expected) -> None:
    assert actual.stage == expected.stage
    assert actual.rejection_reasons == expected.rejection_reasons
    assert dict(actual.metrics) == dict(expected.metrics)


def test_prepared_map_matches_legacy_check_field_by_field() -> None:
    scenario = _road_scenario(heading=np.pi)
    scenario.agents[0].positions[80:, 1] = 6.0
    policy = load_filter_config().map_policy

    legacy = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
    )
    prepared = prepare_map_geometry(scenario)
    optimized = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
        prepared_map=prepared,
    )

    assert isinstance(prepared, PreparedMapGeometry)
    _assert_filter_checks_equal(optimized, legacy)


def test_prepared_map_keeps_polygon_edges_and_vertices_inside() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[50:, 0] = np.linspace(0.0, 10.0, 60)
    target.positions[50:, 1] = 0.0
    scenario.map_polylines.append(
        MapPolyline(
            polyline_id="boundary-lane",
            polyline_type="lane_centerline",
            points=np.array([[0.0, 0.0], [10.0, 0.0]]),
            direction="vehicle",
            lane_id="boundary",
        )
    )
    policy = load_filter_config().map_policy

    legacy = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
    )
    optimized = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
        prepared_map=prepare_map_geometry(scenario),
    )

    _assert_filter_checks_equal(optimized, legacy)
    assert optimized.passed
    assert optimized.metrics["inside_drivable_area_fraction"] == 1.0
    assert optimized.metrics["outside_drivable_area_indices"] == []


def test_prepared_map_preserves_degenerate_lane_behavior() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[50:] = np.array([2.0, 2.0])
    scenario.map_polylines.append(
        MapPolyline(
            polyline_id="degenerate-lane",
            polyline_type="lane_centerline",
            points=np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]),
            direction="vehicle",
            lane_id="degenerate",
        )
    )
    policy = load_filter_config().map_policy

    legacy = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
    )
    optimized = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
        prepared_map=prepare_map_geometry(scenario),
    )

    _assert_filter_checks_equal(optimized, legacy)
    assert optimized.rejection_reasons == (
        FilterRejection.LANE_DIRECTION_VIOLATION,
    )
    assert optimized.metrics["lane_assignment_fraction"] == 1.0
    assert optimized.metrics["assigned_lane_ids"] == ["degenerate"] * 60
    assert optimized.metrics["lane_direction_fraction"] == 0.0


def test_prepared_map_uses_lane_id_to_break_equal_distance_ties() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[:, 0] = np.linspace(1.0, 9.0, 110)
    target.positions[:, 1] = 2.0
    scenario.map_polylines.extend(
        [
            MapPolyline(
                polyline_id="lane-z",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 1.0], [10.0, 1.0]]),
                direction="vehicle",
                lane_id="z",
            ),
            MapPolyline(
                polyline_id="lane-a",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 3.0], [10.0, 3.0]]),
                direction="vehicle",
                lane_id="a",
            ),
        ]
    )
    policy = load_filter_config().map_policy

    legacy = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
    )
    optimized = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
        prepared_map=prepare_map_geometry(scenario),
    )

    _assert_filter_checks_equal(optimized, legacy)
    assert optimized.metrics["assigned_lane_ids"] == ["a"] * 60


def test_prepared_map_preserves_declared_but_invalid_polygon_semantics() -> None:
    scenario = _scenario()
    scenario.map_polylines[0].points = np.array(
        [[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]]
    )
    positions = np.zeros((60, 2), dtype=np.float64)

    legacy = evaluate_drivable_area(scenario, positions, required=True)
    optimized = evaluate_drivable_area(
        scenario,
        positions,
        required=True,
        prepared_map=prepare_map_geometry(scenario),
    )

    assert optimized == legacy
    assert optimized.inside_fraction == 0.0
    assert optimized.reason == "outside_drivable_area"


def test_prepared_map_assigns_lane_at_exact_maximum_distance() -> None:
    scenario = _scenario()
    target = scenario.agents[0]
    target.positions[:, 0] = np.linspace(1.0, 9.0, 110)
    target.positions[:, 1] = 8.0
    scenario.map_polylines.append(
        MapPolyline(
            polyline_id="lane",
            polyline_type="lane_centerline",
            points=np.array([[0.0, 2.0], [10.0, 2.0]]),
            direction="vehicle",
            lane_id="lane",
        )
    )
    policy = load_filter_config().map_policy

    legacy = check_map_compliance(scenario, "target", "slow_lead_blockage", policy)
    optimized = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        policy,
        prepared_map=prepare_map_geometry(scenario),
    )

    _assert_filter_checks_equal(optimized, legacy)
    assert optimized.metrics["lane_assignment_fraction"] == 1.0
    assert optimized.metrics["assigned_lane_ids"] == ["lane"] * 60


def test_batch_map_checks_match_legacy_for_fixed_random_trajectories() -> None:
    source = _road_scenario()
    source.map_polylines.append(
        MapPolyline(
            polyline_id="lane-c",
            polyline_type="lane_centerline",
            points=np.array([[0.0, 9.0], [10.0, 9.0]]),
            direction="bike",
            lane_id="c",
        )
    )
    rng = np.random.default_rng(2026)
    scenarios: list[Scenario] = []
    skill_ids: list[str] = []
    for index in range(8):
        scenario = Scenario.from_dict(source.to_dict())
        scenario.agents[0].positions[50:] = np.column_stack(
            (
                np.linspace(0.0, 10.0, 60),
                rng.uniform(-1.0, 11.0, size=60),
            )
        )
        scenario.agents[0].headings[50:] = rng.uniform(-np.pi, np.pi, size=60)
        scenarios.append(scenario)
        skill_ids.append(
            "abrupt_u_turn_conflict" if index % 3 == 0 else "slow_lead_blockage"
        )

    policy = load_filter_config().map_policy
    expected = tuple(
        check_map_compliance(scenario, "target", skill_id, policy)
        for scenario, skill_id in zip(scenarios, skill_ids)
    )
    actual = check_map_compliance_batch(
        scenarios,
        ["target"] * len(scenarios),
        skill_ids,
        policy,
        prepared_map=prepare_map_geometry(source),
    )

    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for optimized, legacy in zip(actual, expected):
        _assert_filter_checks_equal(optimized, legacy)


def test_fixed_point_chunks_preserve_random_geometry_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    angles = np.linspace(0.0, 2.0 * np.pi, 513)
    scenario.map_polylines[0].points = np.column_stack(
        (20.0 * np.cos(angles), 20.0 * np.sin(angles))
    )
    rng = np.random.default_rng(20260722)
    for lane_index in range(12):
        steps = rng.normal(size=(257, 2))
        points = np.cumsum(steps, axis=0) + np.array([lane_index * 4.0, 0.0])
        scenario.map_polylines.append(
            MapPolyline(
                polyline_id=f"lane-{lane_index}",
                polyline_type="lane_centerline",
                points=points,
                direction="vehicle",
                lane_id=f"lane-{lane_index:02d}",
            )
        )
    prepared = prepare_map_geometry(scenario)
    positions = rng.uniform(-30.0, 50.0, size=(1025, 2))

    monkeypatch.setattr(prepared_map_module, "_POINT_QUERY_CHUNK_SIZE", 2048)
    expected_inside = points_in_drivable_area(prepared, positions)
    expected_projection = project_points_to_lanes(prepared, positions)
    monkeypatch.setattr(prepared_map_module, "_POINT_QUERY_CHUNK_SIZE", 17)
    actual_inside = points_in_drivable_area(prepared, positions)
    actual_projection = project_points_to_lanes(prepared, positions)

    np.testing.assert_array_equal(actual_inside, expected_inside)
    for actual, expected in zip(actual_projection, expected_projection):
        np.testing.assert_array_equal(actual, expected)


def test_large_geometry_never_builds_a_matrix_for_more_than_one_point_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _scenario()
    angles = np.linspace(0.0, 2.0 * np.pi, 513)
    scenario.map_polylines[0].points = np.column_stack(
        (50.0 * np.cos(angles), 50.0 * np.sin(angles))
    )
    scenario.map_polylines.append(
        MapPolyline(
            polyline_id="long-lane",
            polyline_type="lane_centerline",
            points=np.column_stack(
                (np.linspace(-50.0, 50.0, 513), np.sin(np.linspace(-4.0, 4.0, 513)))
            ),
            direction="vehicle",
            lane_id="long-lane",
        )
    )
    prepared = prepare_map_geometry(scenario)
    positions = np.random.default_rng(7).uniform(-40.0, 40.0, size=(1024, 2))
    polygon_rows: list[int] = []
    lane_rows: list[int] = []
    real_polygon_query = prepared_map_module._points_in_polygon
    real_lane_query = prepared_map_module._project_points_to_lane

    def tracked_polygon_query(polygon, query):
        polygon_rows.append(len(query))
        return real_polygon_query(polygon, query)

    def tracked_lane_query(query, lane):
        lane_rows.append(len(query))
        return real_lane_query(query, lane)

    monkeypatch.setattr(prepared_map_module, "_points_in_polygon", tracked_polygon_query)
    monkeypatch.setattr(
        prepared_map_module,
        "_project_points_to_lane",
        tracked_lane_query,
    )

    points_in_drivable_area(prepared, positions)
    project_points_to_lanes(prepared, positions)

    assert polygon_rows and lane_rows
    assert max(polygon_rows) <= prepared_map_module._POINT_QUERY_CHUNK_SIZE
    assert max(lane_rows) <= prepared_map_module._POINT_QUERY_CHUNK_SIZE


def test_filter_projection_prunes_only_lanes_beyond_the_inclusive_threshold() -> None:
    scenario = _scenario()
    scenario.map_polylines.append(
        MapPolyline(
            polyline_id="lane",
            polyline_type="lane_centerline",
            points=np.array([[0.0, 0.0], [10.0, 0.0]]),
            direction="vehicle",
            lane_id="lane",
        )
    )
    prepared = prepare_map_geometry(scenario)
    positions = np.array(
        [
            [16.0, 0.0],
            [5.0, 6.0],
            [16.0 + 5e-9, 0.0],
            [16.0 + 2e-8, 0.0],
            [-100.0, 0.0],
        ]
    )

    global_indices, _, _ = project_points_to_lanes(prepared, positions)
    filtered_indices, distances, _ = project_points_to_lanes_within_distance(
        prepared,
        positions,
        6.0,
    )

    np.testing.assert_array_equal(global_indices, np.zeros(5, dtype=np.int64))
    np.testing.assert_array_equal(filtered_indices, np.array([0, 0, -1, -1, -1]))
    np.testing.assert_array_equal(distances[:2], np.array([6.0, 6.0]))
    assert np.isinf(distances[2:]).all()


def test_filter_projection_keeps_lane_id_tie_order() -> None:
    scenario = _scenario()
    scenario.map_polylines.extend(
        [
            MapPolyline(
                polyline_id="lane-z",
                polyline_type="lane_centerline",
                points=np.array([[0.0, -1.0], [10.0, -1.0]]),
                direction="vehicle",
                lane_id="z",
            ),
            MapPolyline(
                polyline_id="lane-a",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 1.0], [10.0, 1.0]]),
                direction="vehicle",
                lane_id="a",
            ),
        ]
    )
    prepared = prepare_map_geometry(scenario)

    lane_indices, distances, _ = project_points_to_lanes_within_distance(
        prepared,
        np.array([[5.0, 0.0]]),
        1.0,
    )

    assert prepared.lanes[int(lane_indices[0])].lane_id == "a"
    assert distances[0] == 1.0


def test_batch_map_check_rejects_mismatched_sequence_lengths() -> None:
    scenario = _road_scenario()
    with pytest.raises(ValueError, match="equal lengths"):
        check_map_compliance_batch(
            [scenario],
            [],
            ["slow_lead_blockage"],
            load_filter_config().map_policy,
            prepared_map=prepare_map_geometry(scenario),
        )


def test_batch_map_check_rejects_incompatible_source_map() -> None:
    source = _road_scenario()
    incompatible = Scenario.from_dict(source.to_dict())
    incompatible.scenario_id = "different-scene"
    with pytest.raises(ValueError, match="scenario_id"):
        check_map_compliance_batch(
            [incompatible],
            ["target"],
            ["slow_lead_blockage"],
            load_filter_config().map_policy,
            prepared_map=prepare_map_geometry(source),
        )


def test_verification_session_skips_shared_map_hash_until_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _road_scenario()
    prepared = prepare_map_geometry(source)
    generated = Scenario.from_dict(source.to_dict())
    generated.map_polylines = list(source.map_polylines)
    session = PreparedMapVerificationSession(source, prepared)
    real_hash = prepared_map_module.map_geometry_sha256
    calls = []

    def tracking_hash(scenario: Scenario) -> str:
        calls.append(scenario)
        return real_hash(scenario)

    monkeypatch.setattr(prepared_map_module, "map_geometry_sha256", tracking_hash)
    policy = load_filter_config().map_policy
    for _ in range(3):
        check_map_compliance(
            generated,
            "target",
            "slow_lead_blockage",
            policy,
            prepared_map=prepared,
            verification_session=session,
        )

    assert calls == []
    session.finalize()
    assert calls == [source]
    assert session.closed


def test_batch_verification_session_keeps_shared_queries_hash_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _road_scenario()
    prepared = prepare_map_geometry(source)
    generated = []
    for offset in (0.0, 0.5, 1.0):
        scenario = Scenario.from_dict(source.to_dict())
        scenario.map_polylines = list(source.map_polylines)
        scenario.agents[0].positions[50:, 1] += offset
        generated.append(scenario)
    session = PreparedMapVerificationSession(source, prepared)
    real_hash = prepared_map_module.map_geometry_sha256
    calls = []

    def tracking_hash(scenario: Scenario) -> str:
        calls.append(scenario)
        return real_hash(scenario)

    monkeypatch.setattr(prepared_map_module, "map_geometry_sha256", tracking_hash)
    checks = check_map_compliance_batch(
        generated,
        ["target"] * len(generated),
        ["slow_lead_blockage"] * len(generated),
        load_filter_config().map_policy,
        prepared_map=prepared,
        verification_session=session,
    )

    assert len(checks) == len(generated)
    assert calls == []
    session.finalize()
    assert calls == [source]


def test_verification_session_deep_copy_falls_back_to_full_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _road_scenario(heading=np.pi)
    prepared = prepare_map_geometry(source)
    copied = Scenario.from_dict(source.to_dict())
    expected = check_map_compliance(
        copied,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
        prepared_map=prepared,
    )
    session = PreparedMapVerificationSession(source, prepared)
    real_hash = prepared_map_module.map_geometry_sha256
    calls = []

    def tracking_hash(scenario: Scenario) -> str:
        calls.append(scenario)
        return real_hash(scenario)

    monkeypatch.setattr(prepared_map_module, "map_geometry_sha256", tracking_hash)
    actual = check_map_compliance(
        copied,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
        prepared_map=prepared,
        verification_session=session,
    )

    _assert_filter_checks_equal(actual, expected)
    assert calls == [copied]
    session.finalize()
    assert calls == [copied, source]


def test_verification_session_rejects_shared_map_mutation_before_results_escape() -> None:
    source = _road_scenario()
    prepared = prepare_map_geometry(source)
    generated = Scenario.from_dict(source.to_dict())
    generated.map_polylines = list(source.map_polylines)
    session = PreparedMapVerificationSession(source, prepared)

    source.map_polylines[0].points[0, 0] += 1.0
    check_map_compliance(
        generated,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
        prepared_map=prepared,
        verification_session=session,
    )
    with pytest.raises(ValueError, match="geometry differs"):
        session.finalize()
    assert session.closed
    with pytest.raises(RuntimeError, match="closed"):
        session.verify_query(generated, prepared)


def test_verification_session_does_not_skip_reordered_map_objects() -> None:
    source = _road_scenario()
    prepared = prepare_map_geometry(source)
    generated = Scenario.from_dict(source.to_dict())
    generated.map_polylines = list(reversed(source.map_polylines))
    session = PreparedMapVerificationSession(source, prepared)

    with pytest.raises(ValueError, match="geometry differs"):
        check_map_compliance(
            generated,
            "target",
            "slow_lead_blockage",
            load_filter_config().map_policy,
            prepared_map=prepared,
            verification_session=session,
        )
    session.finalize()


def test_vehicle_map_policy_requires_drivable_area_lane_and_direction() -> None:
    scenario = _road_scenario(heading=np.pi)
    policy = load_filter_config().map_policy

    ordinary = check_map_compliance(scenario, "target", "slow_lead_blockage", policy)
    assert ordinary.rejection_reasons == (
        FilterRejection.LANE_DIRECTION_VIOLATION,
    )

    u_turn = check_map_compliance(
        scenario,
        "target",
        "abrupt_u_turn_conflict",
        policy,
    )
    assert u_turn.passed
    assert u_turn.metrics["direction_exempt"] is True


def test_pedestrian_is_not_forced_to_follow_vehicle_lane_direction() -> None:
    scenario = _road_scenario(object_type="pedestrian", heading=np.pi / 2)
    scenario.agents[0].positions[50:, 1] = 12.0
    result = check_map_compliance(
        scenario,
        "target",
        "crosswalk_pedestrian_crossing",
        load_filter_config().map_policy,
    )

    assert result.passed
    assert result.metrics["drivable_area_required"] is False
    assert result.metrics["lane_required"] is False
    assert result.metrics["lane_type_compatibility_fraction"] is None
    assert result.metrics["incompatible_lane_indices"] == []


def test_real_candidate_9a0d_vehicle_on_nearest_bike_lane_is_rejected() -> None:
    """Regression for Pilot task e1ce7e..., candidate 9a0d7f...."""

    scenario = _road_scenario()
    scenario.scenario_id = "0bafea95-dd45-43ea-8e76-19e4de4ce4c3"
    scenario.map_polylines[1].direction = "bike"
    result = check_map_compliance(
        scenario,
        "target",
        "forced_lane_change_around_blockage",
        load_filter_config().map_policy,
    )

    assert result.rejection_reasons == (
        FilterRejection.LANE_TYPE_INCOMPATIBLE,
    )
    assert result.metrics["lane_assignment_fraction"] == 1.0
    assert result.metrics["lane_type_compatibility_fraction"] == 0.0
    assert result.metrics["assigned_lane_ids"] == ["a"] * 60
    assert result.metrics["assigned_lane_types"] == ["bike"] * 60
    assert result.metrics["incompatible_lane_indices"] == list(range(60))
    assert result.metrics["incompatible_lane_types"] == ["bike"] * 60


def test_bus_on_nearest_bike_lane_is_rejected_without_farther_lane_reassignment() -> None:
    scenario = _road_scenario(object_type="bus")
    scenario.map_polylines[1].direction = "BIKE"
    result = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
    )

    assert result.rejection_reasons == (
        FilterRejection.LANE_TYPE_INCOMPATIBLE,
    )
    assert result.metrics["assigned_lane_ids"] == ["a"] * 60
    assert result.metrics["incompatible_lane_types"] == ["bike"] * 60


def test_vehicle_uses_nearest_vehicle_lane_when_bike_lane_is_farther() -> None:
    scenario = _road_scenario()
    scenario.map_polylines[2].direction = "bike"
    result = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
    )

    assert result.passed
    assert result.metrics["assigned_lane_ids"] == ["a"] * 60
    assert result.metrics["lane_type_compatibility_fraction"] == 1.0
    assert result.metrics["incompatible_lane_indices"] == []


def test_cyclist_bike_lane_assignment_keeps_existing_policy() -> None:
    scenario = _road_scenario(object_type="cyclist")
    scenario.map_polylines[1].direction = "bike"
    result = check_map_compliance(
        scenario,
        "target",
        "cyclist_vehicle_merge",
        load_filter_config().map_policy,
    )

    assert result.passed
    assert result.metrics["assigned_lane_ids"] == ["a"] * 60
    assert result.metrics["lane_type_compatibility_fraction"] is None
    assert result.metrics["incompatible_lane_indices"] == []


def test_unconnected_lane_jump_is_rejected_as_topology_violation() -> None:
    scenario = _road_scenario()
    scenario.agents[0].positions[80:, 1] = 6.0
    result = check_map_compliance(
        scenario,
        "target",
        "slow_lead_blockage",
        load_filter_config().map_policy,
    )

    assert result.rejection_reasons == (
        FilterRejection.LANE_CONNECTIVITY_VIOLATION,
    )

import json

from skilldrive.schemas import MapPolyline, Scenario


def test_scenario_json_round_trip(synthetic_scenario: Scenario) -> None:
    payload = json.loads(json.dumps(synthetic_scenario.to_dict()))
    restored = Scenario.from_dict(payload)
    assert restored.scenario_id == synthetic_scenario.scenario_id
    assert restored.focal_track_id == synthetic_scenario.focal_track_id
    assert restored.agents[0].observed_mask.tolist() == synthetic_scenario.agents[0].observed_mask.tolist()


def test_missing_point_is_not_replaced_with_zero(synthetic_scenario: Scenario) -> None:
    synthetic_scenario.agents[0].positions[7] = float("nan")
    restored = Scenario.from_dict(synthetic_scenario.to_dict())
    assert restored.agents[0].positions[7].tolist() != [0.0, 0.0]


def test_map_polyline_topology_round_trip_and_legacy_defaults() -> None:
    polyline = MapPolyline(
        polyline_id="10:center",
        polyline_type="lane_centerline",
        points=[[0.0, 0.0], [1.0, 0.0]],
        lane_id=10,
        left_mark_type="dashed_white",
        right_mark_type="solid_white",
        predecessor_ids=[8, 9],
        successor_ids=[11],
        left_neighbor_id=12,
    )
    restored = MapPolyline.from_dict(polyline.to_dict())
    assert restored.lane_id == "10"
    assert restored.predecessor_ids == ["8", "9"]
    assert restored.successor_ids == ["11"]
    assert restored.left_neighbor_id == "12"
    assert restored.right_neighbor_id is None

    legacy = MapPolyline.from_dict(
        {
            "polyline_id": "legacy",
            "polyline_type": "lane_centerline",
            "points": [[0.0, 0.0], [1.0, 0.0]],
        }
    )
    assert legacy.lane_id is None
    assert legacy.predecessor_ids == []
    assert legacy.successor_ids == []

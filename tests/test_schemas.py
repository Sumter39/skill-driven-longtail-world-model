import json

from skilldrive.schemas import Scenario


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

from pathlib import Path

import pytest
import yaml

from skilldrive.skills import load_skill, validate_skill_dict


SKILL_DIR = Path("configs/skills")


def _skill_data(skill_id: str = "lead_hard_brake") -> dict[str, object]:
    return yaml.safe_load((SKILL_DIR / f"{skill_id}.yaml").read_text(encoding="utf-8"))


def test_catalog_has_30_unique_confirmed_skills() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    ids = [entry["skill_id"] for entry in entries]
    assert len(entries) == 30
    assert len(set(ids)) == 30
    assert {entry["feasibility"] for entry in entries} <= {"A", "B"}
    assert sum(entry["feasibility"] == "A" for entry in entries) == 17
    assert sum(entry["feasibility"] == "B" for entry in entries) == 13
    assert all((SKILL_DIR / f"{skill_id}.yaml").is_file() for skill_id in ids)
    yaml_ids = {path.stem for path in SKILL_DIR.glob("*.yaml")} - {"catalog"}
    assert yaml_ids == set(ids)


def test_all_catalog_skills_are_complete_and_valid() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    for entry in entries:
        skill = load_skill(SKILL_DIR / f"{entry['skill_id']}.yaml")
        assert skill.skill_id == entry["skill_id"]
        assert skill.family == entry["family"]
        assert skill.data_support["feasibility"] == entry["feasibility"]
        assert skill.generation_operators
        assert skill.validation_metrics
        assert skill.known_limitations
        assert skill.detection["mode"] == (
            "observed_trigger" if entry["feasibility"] == "A" else "compatible_seed"
        )
        assert skill.detection["conditions"]
        assert skill.detection["thresholds"]
        assert skill.risk_definition["direction"] in {
            "lower_is_riskier",
            "higher_is_riskier",
        }
        assert type(skill).from_dict(skill.to_dict()) == skill


def test_catalog_families_are_balanced() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    assert len(catalog["families"]) == 6
    assert {family: len(entries) for family, entries in catalog["families"].items()} == {
        family: 5 for family in catalog["families"]
    }


def test_missing_skill_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_skill_dict({"skill_id": "incomplete"})


def test_c_feasibility_is_rejected() -> None:
    data = yaml.safe_load((SKILL_DIR / "lead_hard_brake.yaml").read_text(encoding="utf-8"))
    data["data_support"]["feasibility"] = "C"
    with pytest.raises(ValueError, match="A or B"):
        validate_skill_dict(data)


def test_parameter_source_and_range_are_validated() -> None:
    data = yaml.safe_load((SKILL_DIR / "lead_hard_brake.yaml").read_text(encoding="utf-8"))
    data["parameters"]["brake_onset_s"] = {"range": [3.0, 1.0], "source": "semantic"}
    with pytest.raises(ValueError, match="invalid range"):
        validate_skill_dict(data)


def test_detection_modes_and_risk_directions_follow_confirmed_contract() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    modes = {"observed_trigger": 0, "compatible_seed": 0}
    for entry in entries:
        data = _skill_data(entry["skill_id"])
        expected_mode = "observed_trigger" if entry["feasibility"] == "A" else "compatible_seed"
        assert data["detection"]["mode"] == expected_mode
        modes[expected_mode] += 1
        expected_direction = (
            "higher_is_riskier"
            if entry["skill_id"] == "intersection_blocking_vehicle"
            else "lower_is_riskier"
        )
        assert data["risk_definition"]["direction"] == expected_direction
    assert modes == {"observed_trigger": 17, "compatible_seed": 13}


@pytest.mark.parametrize(
    ("skill_id", "expected_thresholds"),
    [
        ("diverge_lane_crossing_conflict", {"minimum_lateral_displacement_m": 0.25}),
        (
            "intersection_creep_conflict",
            {
                "minimum_crossing_angle_deg": 30.0,
                "maximum_crossing_angle_deg": 150.0,
                "minimum_crossing_vehicle_speed_mps": 1.0,
                "minimum_current_separation_m": 2.0,
            },
        ),
        (
            "intersection_blocking_vehicle",
            {
                "minimum_crossing_angle_deg": 30.0,
                "maximum_crossing_angle_deg": 150.0,
                "minimum_crossing_vehicle_speed_mps": 1.0,
                "minimum_current_separation_m": 2.0,
            },
        ),
        (
            "forced_lane_change_around_blockage",
            {
                "minimum_moving_speed_mps": 1.0,
                "minimum_vehicle_center_distance_m": 2.0,
            },
        ),
        (
            "slow_lead_blockage",
            {
                "minimum_follower_speed_mps": 1.0,
                "minimum_closing_speed_mps": 0.5,
                "minimum_pair_gap_m": 2.0,
            },
        ),
        (
            "lead_sudden_stop",
            {
                "minimum_follower_speed_mps": 1.0,
                "minimum_closing_speed_mps": 0.5,
                "minimum_pair_gap_m": 2.0,
            },
        ),
        ("cut_out_reveals_slow_vehicle", {"minimum_queue_gap_m": 2.0}),
        (
            "jaywalking_pedestrian_crossing",
            {
                "minimum_crossing_angle_deg": 30.0,
                "maximum_crossing_angle_deg": 150.0,
            },
        ),
        (
            "stopped_vehicle_reentry",
            {
                "minimum_vehicle_center_distance_m": 2.0,
                "maximum_lateral_reentry_distance_m": 5.0,
            },
        ),
    ],
)
def test_confirmed_structural_detection_thresholds(
    skill_id: str,
    expected_thresholds: dict[str, float],
) -> None:
    data = _skill_data(skill_id)
    actual = data["detection"]["thresholds"]
    assert {
        name: specification["value"]
        for name, specification in actual.items()
        if name in expected_thresholds
    } == expected_thresholds


def test_stopped_reentry_uses_three_role_detection_vocabulary() -> None:
    data = _skill_data("stopped_vehicle_reentry")
    assert data["detection"]["conditions"] == [
        "currently_stopped",
        "front_and_rear_main_flow_vehicles_present",
        "reentry_space_available",
    ]


def test_required_track_actor_and_map_vocabularies_are_explicit() -> None:
    required_tracks: set[str] = set()
    actor_types: set[str] = set()
    required_map: set[str] = set()
    for path in SKILL_DIR.glob("*.yaml"):
        if path.name == "catalog.yaml":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        required_tracks.update(data["data_support"]["required_tracks"])
        required_map.update(data["data_support"]["required_map"])
        actor_types.update(data["actors"]["initiator_types"])
        actor_types.update(data["actors"]["responder_types"])

    assert required_tracks == {
        "vehicle",
        "pedestrian",
        "cyclist",
        "static",
        "construction",
    }
    assert actor_types == {"vehicle", "bus", "pedestrian", "cyclist", "static", "construction"}
    assert required_map == {
        "lane_centerline",
        "lane_successor",
        "adjacent_lane",
        "converging_lane",
        "diverging_lane",
        "intersection_lane",
        "pedestrian_crossing",
        "drivable_area",
        "bike_lane",
        "lane_direction",
    }


def test_seed_detection_config_contains_only_engine_and_geometry_controls() -> None:
    config = yaml.safe_load(Path("configs/seed_detection.yaml").read_text(encoding="utf-8"))
    assert set(config) == {
        "version",
        "global_seed",
        "max_candidates_per_skill_per_scenario",
        "thresholds",
    }
    assert set(config["thresholds"]) == {
        "maximum_actor_distance_m",
        "lane_match_distance_m",
        "lane_heading_tolerance_deg",
        "same_lane_lateral_tolerance_m",
        "conflict_distance_m",
        "risk_time_horizon_s",
    }
    skill_thresholds: set[str] = set()
    for path in SKILL_DIR.glob("*.yaml"):
        if path.name != "catalog.yaml":
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            skill_thresholds.update(data["detection"]["thresholds"])
    assert set(config["thresholds"]).isdisjoint(skill_thresholds)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("seed_requirements", "minimum_history_steps", 0, "positive integer"),
        ("seed_requirements", "minimum_history_steps", True, "positive integer"),
        ("data_support", "required_tracks", ["aircraft"], "unknown values"),
        ("data_support", "required_map", ["traffic_light_phase"], "unknown values"),
        ("actors", "initiator_types", ["motorcyclist"], "unknown values"),
        ("trigger", "conditions", ["misspelled_condition"], "unknown values"),
        ("detection", "conditions", ["misspelled_condition"], "unknown values"),
        ("risk_definition", "direction", "unknown", "direction"),
        ("risk_definition", "target_range", [4.0, 1.0], "invalid range"),
    ],
)
def test_nested_skill_contract_rejects_invalid_values(
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    data = _skill_data()
    data[section][field] = value
    with pytest.raises(ValueError, match=message):
        validate_skill_dict(data)


def test_detection_mode_must_match_feasibility() -> None:
    data = _skill_data()
    data["detection"]["mode"] = "compatible_seed"
    with pytest.raises(ValueError, match="does not match feasibility"):
        validate_skill_dict(data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("unknown_threshold", {"value": 1.0, "source": "semantic"}), "unknown name"),
        (("minimum_deceleration_mps2", {"value": float("inf"), "source": "semantic"}), "finite"),
        (("minimum_deceleration_mps2", {"value": 2.0, "source": "guess"}), "unknown threshold source"),
        (
            (
                "minimum_deceleration_mps2",
                {"value": 2.0, "source": "semantic", "extra": True},
            ),
            "unknown fields",
        ),
    ],
)
def test_detection_threshold_contract_is_strict(
    mutation: tuple[str, dict[str, object]],
    message: str,
) -> None:
    data = _skill_data()
    name, threshold = mutation
    data["detection"]["thresholds"][name] = threshold
    with pytest.raises(ValueError, match=message):
        validate_skill_dict(data)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("data_support", "unknown"),
        ("seed_requirements", "unknown"),
        ("trigger", "unknown"),
        ("detection", "unknown"),
        ("actors", "unknown"),
        ("risk_definition", "unknown"),
    ],
)
def test_nested_skill_contract_rejects_unknown_fields(section: str, field: str) -> None:
    data = _skill_data()
    data[section][field] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_skill_dict(data)


def test_constraints_and_parameter_shapes_are_strict() -> None:
    data = _skill_data()
    data["constraints"]["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_skill_dict(data)

    data = _skill_data()
    data["constraints"]["remain_in_drivable_area"] = "yes"
    with pytest.raises(ValueError, match="must be boolean"):
        validate_skill_dict(data)

    data = _skill_data()
    data["parameters"]["brake_onset_s"] = {
        "range": [0.5, 2.5],
        "choices": [1.0],
        "source": "semantic",
    }
    with pytest.raises(ValueError, match="exactly one"):
        validate_skill_dict(data)

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

from skilldrive.schemas import SkillSpec
from skilldrive.seeds import (
    SEED_CSV_FIELDS,
    SeedRecord,
    read_seed_records,
    sample_skill_parameters,
    validate_sampled_parameters,
    write_seed_records,
)
from skilldrive.skills import load_skill


SKILL_DIR = Path("configs/skills")
TARGET_RISK_DEFINITION = {
    "metric": "time_to_collision",
    "target_range": [1.0, 4.0],
    "source": "reference",
    "direction": "lower_is_riskier",
}


def _record(scenario_id: str = "scene-b", skill_id: str = "lead_hard_brake") -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id="leader",
        responder_track_id="follower",
        role_track_ids={"leader": "leader", "follower": "follower"},
        trigger_score=0.875,
        seed_risk_metric="time_to_collision",
        seed_risk_value=2.4,
        target_risk_definition=TARGET_RISK_DEFINITION,
        source_path=f"train/{scenario_id}",
        evidence={"closing": True, "metrics": {"gap_m": 8.5, "frames": [10, 11]}},
        sampled_parameters={"brake_onset_s": 1.25, "peak_deceleration_mps2": 4.0},
    )


def _skill_with_parameters(parameters: dict[str, object]) -> SkillSpec:
    return replace(load_skill(SKILL_DIR / "lead_hard_brake.yaml"), parameters=parameters)


def test_seed_csv_round_trip_is_sorted_and_byte_deterministic(tmp_path: Path) -> None:
    records = [_record("scene-b"), _record("scene-a")]
    first = write_seed_records(tmp_path / "first.csv", records)
    second = write_seed_records(tmp_path / "second.csv", reversed(records))

    assert first.read_bytes() == second.read_bytes()
    assert read_seed_records(first) == list(reversed(records))
    with first.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == SEED_CSV_FIELDS
    assert rows[0]["scenario_id"] == "scene-a"
    assert rows[0]["target_risk_definition_json"] == (
        '{"direction":"lower_is_riskier","metric":"time_to_collision",'
        '"source":"reference","target_range":[1.0,4.0]}'
    )
    assert rows[0]["evidence_json"] == '{"closing":true,"metrics":{"frames":[10,11],"gap_m":8.5}}'


def test_duplicate_seed_unique_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate seed record key"):
        write_seed_records(tmp_path / "duplicates.csv", [_record(), _record()])


def test_third_role_changes_candidate_identity() -> None:
    first = replace(
        _record(skill_id="narrow_gap_lane_change"),
        role_track_ids={
            "lane_changer": "leader",
            "target_front_vehicle": "follower",
            "target_rear_vehicle": "rear-a",
        },
    )
    second = replace(
        first,
        role_track_ids={**first.role_track_ids, "target_rear_vehicle": "rear-b"},
    )

    assert first.unique_key != second.unique_key


def test_seed_risk_proxy_relation_is_derived_from_target_metric() -> None:
    direct = _record()
    proxy = replace(
        direct,
        seed_risk_metric="minimum_trajectory_distance",
        seed_risk_value=8.0,
    )

    assert direct.seed_risk_is_proxy is False
    assert proxy.seed_risk_is_proxy is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"trigger_score": 1.1}, "between 0 and 1"),
        ({"seed_risk_value": float("nan")}, "finite number"),
        ({"responder_track_id": "leader"}, "must differ"),
        (
            {"role_track_ids": {"leader": "leader", "follower": "leader"}},
            "distinct tracks",
        ),
        ({"role_track_ids": {"leader": "leader", "other": "other"}}, "responder"),
        ({"evidence": {}}, "non-empty JSON object"),
        ({"sampled_parameters": {"bad": object()}}, "JSON-compatible"),
    ],
)
def test_invalid_seed_record_fields_are_rejected(changes: dict[str, object], message: str) -> None:
    values = _record().__dict__ | changes
    with pytest.raises(ValueError, match=message):
        SeedRecord(**values)


@pytest.mark.parametrize(
    ("target_risk_definition", "message"),
    [
        ({"metric": "time_to_collision"}, "contain exactly"),
        (
            {**TARGET_RISK_DEFINITION, "target_range": [4.0, 1.0]},
            "ordered and nonnegative",
        ),
        (
            {**TARGET_RISK_DEFINITION, "target_range": [1.0, float("inf")]},
            "finite number",
        ),
        ({**TARGET_RISK_DEFINITION, "source": "guess"}, "source is unknown"),
        ({**TARGET_RISK_DEFINITION, "direction": "unknown"}, "direction is unknown"),
    ],
)
def test_target_risk_definition_is_strictly_validated(
    target_risk_definition: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SeedRecord(
            **{
                **_record().__dict__,
                "target_risk_definition": target_risk_definition,
            }
        )


def test_seed_csv_rejects_unknown_header_and_invalid_json(tmp_path: Path) -> None:
    bad_header = tmp_path / "bad-header.csv"
    bad_header.write_text("scenario_id,unknown\nscene,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_seed_records(bad_header)

    legacy_header = tmp_path / "legacy-header.csv"
    legacy_header.write_text(
        "scenario_id,skill_id,initiator_track_id,responder_track_id,"
        "role_track_ids_json,trigger_score,risk_metric,risk_value,source_path,"
        "evidence_json,sampled_parameters_json\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="header"):
        read_seed_records(legacy_header)

    invalid_json = tmp_path / "invalid-json.csv"
    row = _record().to_csv_row()
    row["evidence_json"] = "[]"
    with invalid_json.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="non-empty JSON object"):
        read_seed_records(invalid_json)


def test_parameter_sampling_repeats_and_respects_range_and_choices() -> None:
    skill = _skill_with_parameters(
        {
            "continuous": {"range": [0.5, 1.5], "source": "semantic"},
            "integer": {"range": [2, 5], "source": "semantic"},
            "priority": {"choices": ["initiator", "responder"], "source": "semantic"},
        }
    )
    first = sample_skill_parameters(skill, global_seed=2026, sample_key="scene/leader/follower")
    second = sample_skill_parameters(skill, global_seed=2026, sample_key="scene/leader/follower")

    assert first == second
    assert 0.5 <= first["continuous"] <= 1.5
    assert isinstance(first["integer"], int) and 2 <= first["integer"] <= 5
    assert first["priority"] in {"initiator", "responder"}
    validate_sampled_parameters(skill, first)


def test_parameter_sampling_is_independent_of_parameter_order() -> None:
    parameters = {
        "alpha": {"range": [0.0, 1.0], "source": "semantic"},
        "beta": {"range": [1.0, 2.0], "source": "semantic"},
    }
    forward = _skill_with_parameters(parameters)
    reverse = _skill_with_parameters(dict(reversed(list(parameters.items()))))

    assert sample_skill_parameters(forward, global_seed=7, sample_key="candidate") == (
        sample_skill_parameters(reverse, global_seed=7, sample_key="candidate")
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"bad": {"range": [2.0, 1.0], "source": "semantic"}},
        {"bad": {"range": [0.0, float("inf")], "source": "semantic"}},
        {"bad": {"choices": [], "source": "semantic"}},
        {"bad": {"range": [0.0, 1.0], "choices": [0.5], "source": "semantic"}},
        {"bad": {"range": [0.0, 1.0], "source": "semantic", "unknown": True}},
    ],
)
def test_parameter_sampling_rejects_invalid_specs(parameters: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        sample_skill_parameters(
            _skill_with_parameters(parameters),
            global_seed=2026,
            sample_key="candidate",
        )


def test_sampled_parameter_validation_rejects_unknown_and_out_of_range_values() -> None:
    skill = _skill_with_parameters(
        {"amount": {"range": [1.0, 2.0], "source": "semantic"}}
    )
    with pytest.raises(ValueError, match="unknown"):
        validate_sampled_parameters(skill, {"amount": 1.5, "extra": 1})
    with pytest.raises(ValueError, match="outside"):
        validate_sampled_parameters(skill, {"amount": 3.0})

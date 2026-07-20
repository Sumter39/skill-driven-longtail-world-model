from __future__ import annotations

from dataclasses import replace

import pytest

from skilldrive.seeds.records import SeedRecord
from skilldrive.seeds.selection import _scenario_strata, select_seed_records


TARGET_RISK_DEFINITION = {
    "metric": "time_to_collision",
    "target_range": [1.0, 4.0],
    "source": "reference",
    "direction": "lower_is_riskier",
}


def _record(
    scenario_id: str,
    skill_id: str,
    risk_value: float,
    *,
    risk_metric: str = "time_to_collision",
    track_suffix: str = "",
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"initiator-{skill_id}{track_suffix}",
        responder_track_id=f"responder-{skill_id}{track_suffix}",
        role_track_ids={
            "initiator": f"initiator-{skill_id}{track_suffix}",
            "responder": f"responder-{skill_id}{track_suffix}",
        },
        trigger_score=0.8,
        seed_risk_metric=risk_metric,
        seed_risk_value=risk_value,
        target_risk_definition=TARGET_RISK_DEFINITION,
        source_path=f"train/{scenario_id}",
        evidence={"risk_value": risk_value},
        sampled_parameters={"amount": 1.0},
    )


def _scenario_ids(records: list[SeedRecord]) -> set[str]:
    return {record.scenario_id for record in records}


def test_selection_prioritizes_rare_stratum_when_target_cuts_first_round() -> None:
    records = [
        _record(f"common-{index}", "common", float(index))
        for index in range(8)
    ]
    records.append(_record("rare-only", "rare", 10.0, risk_metric="distance"))

    selected = select_seed_records(records, 1, seed=17)

    assert _scenario_ids(selected) == {"rare-only"}


def test_selection_round_robins_risk_quartiles() -> None:
    records = [
        _record(f"common-{index}", "common", float(index))
        for index in range(8)
    ]

    selected = select_seed_records(records, 4, seed=17)

    assert len(selected) == 4
    assert {int(record.seed_risk_value) // 2 for record in selected} == {
        0,
        1,
        2,
        3,
    }


def test_selection_retains_all_multi_skill_records_for_selected_scenario() -> None:
    shared_rare = _record("shared", "rare", 1.0, risk_metric="distance")
    shared_common = _record("shared", "common", 0.0)
    common = [
        _record(f"common-{index}", "common", float(index + 1))
        for index in range(7)
    ]

    selected = select_seed_records([shared_rare, shared_common, *common], 1, seed=3)

    assert selected == sorted(
        [shared_rare, shared_common],
        key=lambda record: record.unique_key,
    )


def test_selection_is_independent_of_input_order() -> None:
    records = [
        _record(f"scene-{index}", f"skill-{index % 3}", float(index % 5))
        for index in range(18)
    ]

    forward = select_seed_records(records, 7, seed=2026)
    reverse = select_seed_records(reversed(records), 7, seed=2026)

    assert forward == reverse
    assert len(_scenario_ids(forward)) == 7


def test_equal_risk_values_remain_in_the_same_quartile() -> None:
    records = [
        _record(f"same-risk-{index}", "same-skill", 2.0)
        for index in range(12)
    ]

    forward = _scenario_strata(records)
    reverse = _scenario_strata(list(reversed(records)))

    assert forward == reverse
    assert set(forward) == {("same-skill", "time_to_collision", 0)}
    assert forward[("same-skill", "time_to_collision", 0)] == {
        f"same-risk-{index}" for index in range(12)
    }


def test_selection_returns_all_records_when_target_exceeds_available_scenarios() -> None:
    first = _record("scene-a", "skill-a", 1.0)
    second = _record("scene-b", "skill-b", 2.0)
    second_skill = _record("scene-a", "skill-c", 3.0)
    records = [second, second_skill, first]

    selected = select_seed_records(records, 10)

    assert selected == sorted(records, key=lambda record: record.unique_key)
    assert len(_scenario_ids(selected)) == 2


def test_selection_accepts_empty_input() -> None:
    assert select_seed_records([], 5) == []


@pytest.mark.parametrize("target", [0, -1, 1.5, True])
def test_selection_rejects_invalid_target_count(target: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        select_seed_records([], target)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_selection_rejects_invalid_seed(seed: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        select_seed_records([], 1, seed=seed)  # type: ignore[arg-type]


def test_selection_rejects_non_records_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="only SeedRecord"):
        select_seed_records([object()], 1)  # type: ignore[list-item]

    duplicate = _record("scene", "skill", 1.0)
    with pytest.raises(ValueError, match="duplicate seed record key"):
        select_seed_records([duplicate, replace(duplicate)], 1)


def test_selection_rejects_multiple_records_for_same_scenario_skill_metric() -> None:
    first = _record("scene", "skill", 1.0)
    second = _record("scene", "skill", 2.0, track_suffix="-other")

    with pytest.raises(
        ValueError,
        match="multiple seed records for the same scenario, skill, and risk metric",
    ):
        select_seed_records([first, second], 2)

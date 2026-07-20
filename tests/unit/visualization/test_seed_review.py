from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds import SeedRecord
from skilldrive.visualization import (
    render_seed_review,
    seed_review_filename,
    select_stratified_review_records,
)
from skilldrive.visualization.seed_review import _risk_detail_lines


TARGET_RISK_DEFINITION = {
    "metric": "time_to_collision",
    "target_range": [1.0, 4.0],
    "source": "reference",
    "direction": "lower_is_riskier",
}


def _record(
    scenario_id: str = "test-scene",
    skill_id: str = "lead_hard_brake",
    initiator: str = "focal",
    responder: str = "responder",
    score: float = 0.8,
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=initiator,
        responder_track_id=responder,
        role_track_ids={"initiator": initiator, "responder": responder},
        trigger_score=score,
        seed_risk_metric="time_to_collision",
        seed_risk_value=2.5,
        target_risk_definition=TARGET_RISK_DEFINITION,
        source_path=f"train/{scenario_id}",
        evidence={"gap_m": 8.0, "closing": True},
        sampled_parameters={"peak_deceleration_mps2": 4.0},
    )


def _two_actor_scenario(scenario: Scenario) -> Scenario:
    steps = len(scenario.timestamps)
    responder = AgentTrack(
        track_id="responder",
        object_type="vehicle",
        positions=np.column_stack((np.arange(steps, dtype=float) + 7.0, np.ones(steps) * 2.0)),
        velocities=np.column_stack((np.ones(steps), np.zeros(steps))),
        headings=np.zeros(steps),
        observed_mask=np.arange(steps) < 5,
    )
    return replace(scenario, agents=[*scenario.agents, responder])


def test_seed_review_filename_is_deterministic_safe_and_key_specific() -> None:
    record = _record(scenario_id="scene with/slashes")
    first = seed_review_filename(record)

    assert first == seed_review_filename(record)
    assert first.endswith(".png")
    assert "/" not in first and " " not in first
    assert first != seed_review_filename(
        replace(
            record,
            responder_track_id="other",
            role_track_ids={"initiator": record.initiator_track_id, "responder": "other"},
        )
    )
    third_a = replace(
        record,
        role_track_ids={
            "initiator": record.initiator_track_id,
            "responder": record.responder_track_id,
            "third": "third-a",
        },
    )
    third_b = replace(
        third_a,
        role_track_ids={**third_a.role_track_ids, "third": "third-b"},
    )
    assert seed_review_filename(third_a) != seed_review_filename(third_b)


def test_seed_review_renderer_writes_highlighted_png(
    tmp_path: Path,
    synthetic_scenario: Scenario,
) -> None:
    scenario = _two_actor_scenario(synthetic_scenario)
    record = _record()

    output = render_seed_review(scenario, record, tmp_path, radius_m=20)

    assert output == tmp_path / seed_review_filename(record)
    assert output.exists()
    assert output.stat().st_size > 5_000


def test_seed_review_risk_details_distinguish_proxy_from_target() -> None:
    direct_lines = _risk_detail_lines(_record())
    proxy_lines = _risk_detail_lines(
        replace(
            _record(),
            seed_risk_metric="minimum_trajectory_distance",
            seed_risk_value=8.0,
        )
    )

    assert direct_lines == [
        "seed_risk=time_to_collision:2.5",
        (
            "target_risk=time_to_collision range=[1.0,4.0] "
            "direction=lower_is_riskier source=reference"
        ),
        "risk_relation=target_metric_observation",
    ]
    assert proxy_lines[0] == "seed_risk=minimum_trajectory_distance:8"
    assert proxy_lines[-1] == "risk_relation=proxy"


def test_seed_review_renderer_rejects_mismatched_scenario_and_missing_tracks(
    tmp_path: Path,
    synthetic_scenario: Scenario,
) -> None:
    scenario = _two_actor_scenario(synthetic_scenario)
    with pytest.raises(ValueError, match="scenario_id"):
        render_seed_review(scenario, _record(scenario_id="other-scene"), tmp_path)
    with pytest.raises(ValueError, match="missing tracks"):
        render_seed_review(scenario, _record(responder="unknown"), tmp_path)


def test_stratified_selection_covers_skills_then_round_robins_by_score() -> None:
    records = [
        _record("a-low", "skill_a", score=0.2),
        _record("a-high", "skill_a", score=0.9),
        _record("a-mid", "skill_a", score=0.5),
        _record("b-low", "skill_b", score=0.1),
        _record("b-high", "skill_b", score=0.8),
        _record("c-only", "skill_c", score=0.3),
    ]

    selected = select_stratified_review_records(reversed(records), target_count=5)

    assert [(item.skill_id, item.scenario_id) for item in selected] == [
        ("skill_a", "a-high"),
        ("skill_b", "b-high"),
        ("skill_c", "c-only"),
        ("skill_a", "a-mid"),
        ("skill_b", "b-low"),
    ]
    assert selected == select_stratified_review_records(records, target_count=5)


def test_stratified_selection_returns_all_when_target_exceeds_candidates() -> None:
    records = [_record("one", "skill_a"), _record("two", "skill_b")]
    assert len(select_stratified_review_records(records)) == 2
    with pytest.raises(ValueError, match="positive integer"):
        select_stratified_review_records(records, target_count=0)

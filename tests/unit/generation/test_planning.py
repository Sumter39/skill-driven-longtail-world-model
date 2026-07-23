from __future__ import annotations

from dataclasses import replace

import numpy as np

import skilldrive.generation as generation_api
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.planning import (
    build_generation_task,
    latent_group_id,
    latent_seeds_for_task,
    paired_latent_seeds_for_task,
    pilot_evaluation_arm,
    prior_context_spec_for_task,
    seed_record_id,
    select_eligible_pilot_records,
    semantic_generation_config_sha256,
)
from skilldrive.seeds.records import SeedRecord


def _record(skill_id: str, roles: dict[str, str], mode: str) -> SeedRecord:
    values = list(roles.values())
    return SeedRecord(
        scenario_id="scene",
        skill_id=skill_id,
        initiator_track_id=values[0],
        responder_track_id=values[1],
        role_track_ids=roles,
        trigger_score=0.5,
        seed_risk_metric="metric",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "metric",
            "direction": "lower_is_riskier",
            "source": "semantic",
            "target_range": [0.0, 2.0],
        },
        source_path="train/scene/scenario_scene.parquet",
        evidence={"detection_mode": mode},
        sampled_parameters={"value": 1.0},
    )


def test_rule_search_keeps_required_context_without_role_conditioning() -> None:
    config = load_counterfactual_config()
    record = _record(
        "construction_object_lane_blockage",
        {"construction_object": "object", "responding_vehicle": "vehicle"},
        "compatible_seed",
    )
    task = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=4,
    )
    spec = prior_context_spec_for_task(task, record)

    assert task.target_track_id == "vehicle"
    assert task.condition_skill_id == "<none>"
    assert spec.required_context_track_ids == ("object",)
    assert spec.role_track_ids == ()


def test_paired_pilot_contract_is_publicly_exported() -> None:
    expected = {
        "PilotEvaluationArm",
        "PilotEligibilityDecision",
        "PilotEligibilitySelection",
        "PairedPilotRecovery",
        "build_paired_pilot_task_plan",
        "latent_group_id",
        "paired_latent_seed",
        "paired_latent_seeds_for_task",
        "pilot_evaluation_arm",
        "prior_context_fingerprint",
        "recover_paired_pilot_tasks",
        "select_eligible_pilot_records",
    }

    assert expected <= set(generation_api.__all__)
    assert all(hasattr(generation_api, name) for name in expected)


def test_eligibility_selection_replaces_invalid_records_and_caches_contexts() -> None:
    skill_id = "slow_lead_blockage"

    def record_for(scenario_id: str, marker: float) -> SeedRecord:
        return replace(
            _record(
                skill_id,
                {"slow_leader": "leader", "following_vehicle": "follower"},
                "observed_trigger",
            ),
            scenario_id=scenario_id,
            source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
            sampled_parameters={"value": marker},
        )

    records = [
        record_for("invalid-shared", 1.0),
        record_for("invalid-shared", 2.0),
        *(record_for(f"valid-{index}", float(index)) for index in range(4)),
    ]
    validated_contexts: list[str] = []

    def validate(record: SeedRecord) -> None:
        validated_contexts.append(record.scenario_id)
        if record.scenario_id == "invalid-shared":
            raise ValueError("target has fewer than 30 valid history steps")

    selection = select_eligible_pilot_records(
        reversed(records),
        formal_skill_ids=(skill_id,),
        per_skill=4,
        base_seed=2026,
        context_fingerprint=lambda record: record.scenario_id,
        validate_record=validate,
    )

    def validate_repeated(record: SeedRecord) -> None:
        if record.scenario_id == "invalid-shared":
            raise ValueError("target has fewer than 30 valid history steps")

    repeated = select_eligible_pilot_records(
        records,
        formal_skill_ids=(skill_id,),
        per_skill=4,
        base_seed=2026,
        context_fingerprint=lambda record: record.scenario_id,
        validate_record=validate_repeated,
    )

    assert [seed_record_id(record) for record in selection.records] == [
        seed_record_id(record) for record in repeated.records
    ]
    assert len(selection.records) == 4
    assert {record.scenario_id for record in selection.records} == {
        "valid-0",
        "valid-1",
        "valid-2",
        "valid-3",
    }
    invalid = [
        decision
        for decision in selection.decisions
        if decision.context_fingerprint == "invalid-shared"
    ]
    assert len(invalid) == 2
    assert {decision.cache_hit for decision in invalid} == {False, True}
    assert all(not decision.eligible for decision in invalid)
    assert all(
        decision.failure_message == "target has fewer than 30 valid history steps"
        for decision in invalid
    )
    assert validated_contexts.count("invalid-shared") == 1


def test_eligibility_selection_preserves_zero_selected_skill_audit() -> None:
    record = _record(
        "group_pedestrian_crossing",
        {
            "first_crossing_pedestrian": "pedestrian-1",
            "responding_vehicle": "vehicle",
            "second_crossing_pedestrian": "pedestrian-2",
        },
        "observed_trigger",
    )

    def reject(_: SeedRecord) -> None:
        raise ValueError("target has fewer than 30 valid history steps")

    selection = select_eligible_pilot_records(
        [record],
        formal_skill_ids=(record.skill_id,),
        per_skill=16,
        base_seed=2026,
        context_fingerprint=lambda _: "shared-context",
        validate_record=reject,
    )

    assert selection.records == ()
    assert len(selection.decisions) == 1
    assert selection.decisions[0].eligible is False
    assert selection.decisions[0].failure_message == (
        "target has fewer than 30 valid history steps"
    )


def test_latent_seeds_are_stable_when_candidate_budget_extends() -> None:
    config = load_counterfactual_config()
    record = _record(
        "slow_lead_blockage",
        {"slow_leader": "leader", "following_vehicle": "follower"},
        "observed_trigger",
    )
    first = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=4,
    )
    extended = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=8,
    )

    assert first.task_id == extended.task_id
    np.testing.assert_array_equal(
        latent_seeds_for_task(first, base_seed=2026),
        latent_seeds_for_task(extended, base_seed=2026)[:4],
    )


def test_semantic_config_hash_excludes_all_candidate_budgets() -> None:
    config = load_counterfactual_config()
    changed = replace(
        config,
        sampling=replace(
            config.sampling,
            pilot_seed_records_per_skill=config.sampling.pilot_seed_records_per_skill + 1,
            pilot_candidates_per_task=config.sampling.pilot_candidates_per_task + 2,
            formal_candidates_per_task=config.sampling.formal_candidates_per_task + 3,
        ),
    )

    assert semantic_generation_config_sha256(config) == semantic_generation_config_sha256(
        changed
    )


def test_paired_latent_contract_ignores_only_the_learned_condition_arm() -> None:
    config = load_counterfactual_config()
    record = _record(
        "slow_lead_blockage",
        {"slow_leader": "leader", "following_vehicle": "follower"},
        "observed_trigger",
    )
    conditioned = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=4,
    )
    control = type(conditioned).create(
        task_index=1,
        seed_record_id=conditioned.seed_record_id,
        scenario_id=conditioned.scenario_id,
        skill_id=conditioned.skill_id,
        target_track_id=conditioned.target_track_id,
        proposal_mode=conditioned.proposal_mode,
        condition_skill_id=config.none_skill_id,
        candidate_budget=conditioned.candidate_budget,
        checkpoint_sha256=conditioned.checkpoint_sha256,
        semantic_config_sha256=conditioned.semantic_config_sha256,
    )

    assert conditioned.task_id != control.task_id
    assert pilot_evaluation_arm(conditioned) == "learned_conditioned"
    assert pilot_evaluation_arm(control) == "learned_none_control"
    assert latent_group_id(conditioned) == latent_group_id(control)
    np.testing.assert_array_equal(
        paired_latent_seeds_for_task(conditioned, base_seed=2026),
        paired_latent_seeds_for_task(control, base_seed=2026),
    )

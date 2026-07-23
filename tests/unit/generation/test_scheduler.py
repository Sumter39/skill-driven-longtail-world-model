from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.generation.contracts import (
    GenerationTask,
    GeneratedCandidate,
    GeneratedOverlay,
    candidate_id,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    latent_seed,
    paired_latent_seed,
    pilot_evaluation_arm,
)
from skilldrive.generation.scheduler import (
    ProcessingProgress,
    TaskPlan,
    build_paired_pilot_task_plan,
    build_pilot_task_plan,
    load_task_plan,
    recover_durable_tasks,
    recover_paired_pilot_tasks,
    recovery_progress,
    write_task_plan,
)
from skilldrive.generation.storage import write_raw_shard
from skilldrive.seeds import read_seed_records
from skilldrive.seeds.records import SeedRecord


EXECUTION_A = {"batch_size": 16, "workers": 2, "shard_size": 8}
EXECUTION_B = {"batch_size": 32, "workers": 4, "shard_size": 16}


def _config() -> CounterfactualGenerationConfig:
    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1",
        formal_catalog=Path("configs/skills/catalog.yaml"),
        candidate_catalog=Path("configs/skills/candidate_catalog.yaml"),
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=Path("best.pt"),
            sha256="a" * 64,
            run_manifest=Path("run_manifest.json"),
            run_manifest_sha256="b" * 64,
            schema_sha256="c" * 64,
        ),
        inputs=GenerationInputConfig(
            data_root=Path("data"),
            seed_manifest=Path("seeds.csv"),
            seed_manifest_sha256="d" * 64,
            training_cache_manifest=Path("cache.json"),
            training_cache_manifest_sha256="e" * 64,
            leakage_audit=Path("audit.json"),
            leakage_audit_sha256="f" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=2,
            pilot_candidates_per_task=2,
            formal_candidates_per_task=4,
        ),
        formal_skill_ids=("learned", "search"),
        candidate_skill_ids=(),
        skills=(
            SkillGenerationConfig(
                skill_id="learned",
                primary_generated_role="actor",
                proposal_mode="learned_conditioned_prior",
                condition_skill_strategy="requested_skill_id",
                joint_generation_limited=False,
            ),
            SkillGenerationConfig(
                skill_id="search",
                primary_generated_role="actor",
                proposal_mode="rule_guided_prior_search",
                condition_skill_strategy="none_skill_id",
                joint_generation_limited=False,
            ),
        ),
    )


def _record(skill_id: str, scenario_id: str) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"{scenario_id}-actor",
        responder_track_id=f"{scenario_id}-other",
        role_track_ids={
            "actor": f"{scenario_id}-actor",
            "other": f"{scenario_id}-other",
        },
        trigger_score=0.5,
        seed_risk_metric="metric",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "metric",
            "target_range": [0.0, 2.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"value": 1.0},
    )


def _records() -> list[SeedRecord]:
    return [
        _record("learned", "scene-c"),
        _record("learned", "scene-a"),
        _record("learned", "scene-b"),
        _record("search", "scene-b"),
        _record("search", "scene-d"),
        _record("search", "scene-a"),
    ]


def _candidate(plan, task_index: int, candidate_index: int) -> GeneratedCandidate:
    task = plan.tasks[task_index]
    seed = latent_seed(plan.base_seed, task.task_id, candidate_index)
    return GeneratedCandidate(
        task_id=task.task_id,
        candidate_index=candidate_index,
        latent_seed=seed,
        scenario_id=task.scenario_id,
        skill_id=task.skill_id,
        proposal_mode=task.proposal_mode,
        checkpoint_sha256=task.checkpoint_sha256,
        semantic_config_sha256=task.semantic_config_sha256,
        overlay=GeneratedOverlay(
            target_track_id=task.target_track_id,
            future_xy_global=np.full((60, 2), candidate_index, dtype=np.float32),
        ),
    )


def _paired_candidate(plan, task_index: int, candidate_index: int) -> GeneratedCandidate:
    task = plan.tasks[task_index]
    seed = paired_latent_seed(plan.base_seed, task, candidate_index)
    return GeneratedCandidate(
        task_id=task.task_id,
        candidate_index=candidate_index,
        latent_seed=seed,
        scenario_id=task.scenario_id,
        skill_id=task.skill_id,
        proposal_mode=task.proposal_mode,
        checkpoint_sha256=task.checkpoint_sha256,
        semantic_config_sha256=task.semantic_config_sha256,
        overlay=GeneratedOverlay(
            target_track_id=task.target_track_id,
            future_xy_global=np.full((60, 2), candidate_index, dtype=np.float32),
        ),
    )


def test_pilot_plan_is_per_skill_deterministic_and_scenario_grouped() -> None:
    config = _config()
    first = build_pilot_task_plan(
        reversed(_records()),
        config,
        execution_config=EXECUTION_A,
        per_skill=2,
        candidate_budget=2,
    )
    repeated = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=2,
        candidate_budget=2,
    )
    changed_execution = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_B,
        per_skill=2,
        candidate_budget=2,
    )

    assert [task.task_id for task in first.tasks] == [
        task.task_id for task in repeated.tasks
    ]
    assert Counter(task.skill_id for task in first.tasks) == {
        "learned": 2,
        "search": 2,
    }
    assert [group.scenario_id for group in first.scenario_groups] == sorted(
        group.scenario_id for group in first.scenario_groups
    )
    assert [task.task_id for task in first.tasks] == [
        task.task_id for task in changed_execution.tasks
    ]
    assert first.execution_config_sha256 != changed_execution.execution_config_sha256


def test_paired_pilot_plan_adds_only_learned_none_controls() -> None:
    config = _config()
    plan = build_paired_pilot_task_plan(
        reversed(_records()),
        config,
        execution_config=EXECUTION_A,
        per_skill=2,
        candidate_budget=2,
    )
    repeated = build_paired_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=2,
        candidate_budget=2,
    )

    assert [task.task_id for task in plan.tasks] == [
        task.task_id for task in repeated.tasks
    ]
    assert Counter(pilot_evaluation_arm(task) for task in plan.tasks) == {
        "learned_conditioned": 2,
        "learned_none_control": 2,
        "rule_guided_none": 2,
    }
    learned_by_record = {}
    for task in plan.tasks:
        if task.skill_id == "learned":
            learned_by_record.setdefault(task.seed_record_id, []).append(task)
        else:
            assert task.condition_skill_id == "<none>"
            assert pilot_evaluation_arm(task) == "rule_guided_none"
    assert all(len(pair) == 2 for pair in learned_by_record.values())
    for pair in learned_by_record.values():
        first, second = pair
        assert first.task_id != second.task_id
        for candidate_index in range(plan.candidate_budget):
            first_seed = paired_latent_seed(
                plan.base_seed,
                first,
                candidate_index,
            )
            second_seed = paired_latent_seed(
                plan.base_seed,
                second,
                candidate_index,
            )
            assert first_seed == second_seed
            assert candidate_id(
                task_id=first.task_id,
                candidate_index=candidate_index,
                latent_seed=first_seed,
                checkpoint_sha256=first.checkpoint_sha256,
                semantic_config_sha256=first.semantic_config_sha256,
            ) != candidate_id(
                task_id=second.task_id,
                candidate_index=candidate_index,
                latent_seed=second_seed,
                checkpoint_sha256=second.checkpoint_sha256,
                semantic_config_sha256=second.semantic_config_sha256,
            )


def test_frozen_paired_pilot_plan_has_expected_current_scale() -> None:
    from skilldrive.generation.config import load_counterfactual_config

    config = load_counterfactual_config()
    records = read_seed_records(config.inputs.seed_manifest)
    plan = build_paired_pilot_task_plan(
        records,
        config,
        execution_config=EXECUTION_A,
    )

    assert Counter(pilot_evaluation_arm(task) for task in plan.tasks) == {
        "learned_conditioned": 183,
        "learned_none_control": 183,
        "rule_guided_none": 323,
    }
    assert len(plan.tasks) == 689
    assert plan.total_candidates == 11_024
    learned_pairs = {}
    for task in plan.tasks:
        if task.proposal_mode == "learned_conditioned_prior":
            learned_pairs.setdefault(task.seed_record_id, []).append(task)
    assert len(learned_pairs) == 183
    for pair in learned_pairs.values():
        assert len(pair) == 2
        for candidate_index in range(plan.candidate_budget):
            assert paired_latent_seed(
                plan.base_seed,
                pair[0],
                candidate_index,
            ) == paired_latent_seed(
                plan.base_seed,
                pair[1],
                candidate_index,
            )


def test_candidate_budget_extension_preserves_existing_task_and_candidate_ids() -> None:
    config = _config()
    first = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    extended = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=4,
    )
    assert first.task_plan_id == extended.task_plan_id
    assert [task.task_id for task in first.tasks] == [task.task_id for task in extended.tasks]
    for first_task, extended_task in zip(first.tasks, extended.tasks):
        first_ids = [
            candidate_id(
                task_id=first_task.task_id,
                candidate_index=index,
                latent_seed=latent_seed(first.base_seed, first_task.task_id, index),
                checkpoint_sha256=first_task.checkpoint_sha256,
                semantic_config_sha256=first_task.semantic_config_sha256,
            )
            for index in range(2)
        ]
        extended_ids = [
            candidate_id(
                task_id=extended_task.task_id,
                candidate_index=index,
                latent_seed=latent_seed(extended.base_seed, extended_task.task_id, index),
                checkpoint_sha256=extended_task.checkpoint_sha256,
                semantic_config_sha256=extended_task.semantic_config_sha256,
            )
            for index in range(4)
        ]
        assert first_ids == extended_ids[:2]


def test_task_plan_round_trip_allows_execution_change_and_rejects_semantic_change(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    artifacts = write_task_plan(tmp_path, plan)
    loaded = load_task_plan(
        tmp_path,
        expected_semantic_config_sha256=plan.semantic_config_sha256,
        current_execution_config_sha256=canonical_sha256(EXECUTION_B),
    )
    assert loaded.plan == plan
    assert loaded.execution_config_changed is True
    assert artifacts.task_plan_path.name == "task_plan.jsonl"
    assert artifacts.summary_path.name == "task_plan.summary.json"

    changed_checkpoint = replace(
        config,
        active_checkpoint=replace(config.active_checkpoint, sha256="9" * 64),
    )
    changed_plan = build_pilot_task_plan(
        _records(),
        changed_checkpoint,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    with pytest.raises(ValueError, match="semantic configuration differs"):
        load_task_plan(
            tmp_path,
            expected_semantic_config_sha256=changed_plan.semantic_config_sha256,
            current_execution_config_sha256=canonical_sha256(EXECUTION_A),
        )

    artifacts.task_plan_path.write_bytes(artifacts.task_plan_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="size differs"):
        load_task_plan(
            tmp_path,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
            current_execution_config_sha256=canonical_sha256(EXECUTION_A),
        )


def test_raw_recovery_distinguishes_durable_partial_and_budget_extension(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        0,
        [_candidate(plan, 0, 0), _candidate(plan, 0, 1), _candidate(plan, 1, 0)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )
    recovery = recover_durable_tasks(plan, tmp_path / "raw")
    first_id, second_id = (task.task_id for task in plan.tasks)
    assert recovery.durable_task_ids == {first_id}
    assert recovery.partial_task_ids == {second_id}
    assert recovery.pending_candidate_indices[second_id] == (1,)

    extended = build_pilot_task_plan(
        _records(),
        config,
        execution_config=EXECUTION_B,
        per_skill=1,
        candidate_budget=4,
    )
    extended_recovery = recover_durable_tasks(extended, tmp_path / "raw")
    assert extended_recovery.durable_task_ids == set()
    assert extended_recovery.partial_task_ids == {first_id, second_id}
    assert extended_recovery.pending_candidate_indices[first_id] == (2, 3)
    assert extended_recovery.pending_candidate_indices[second_id] == (1, 2, 3)

    changed_config = replace(
        config,
        active_checkpoint=replace(config.active_checkpoint, sha256="8" * 64),
    )
    changed_plan = build_pilot_task_plan(
        _records(),
        changed_config,
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    with pytest.raises(ValueError, match="different semantic configuration"):
        recover_durable_tasks(changed_plan, tmp_path / "raw")


def test_paired_pilot_recovery_rebuilds_whole_partial_and_missing_tasks(
    tmp_path: Path,
) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        0,
        [_paired_candidate(plan, 0, 0), _paired_candidate(plan, 0, 1)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )
    write_raw_shard(
        tmp_path / "raw",
        1,
        [_paired_candidate(plan, 1, 0)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )

    recovery = recover_paired_pilot_tasks(plan, tmp_path / "raw")

    assert recovery.durable_task_ids == {plan.tasks[0].task_id}
    assert recovery.partial_task_ids == {plan.tasks[1].task_id}
    assert recovery.missing_task_ids == {plan.tasks[2].task_id}
    assert recovery.rebuild_task_ids == {
        plan.tasks[1].task_id,
        plan.tasks[2].task_id,
    }
    assert recovery.durable_candidate_count == 2


def test_paired_pilot_recovery_supports_frozen_source_task_latent_seeds(
    tmp_path: Path,
) -> None:
    source_task = GenerationTask.create(
        task_index=0,
        seed_record_id="1" * 64,
        scenario_id="scene-source-seed",
        skill_id="learned",
        target_track_id="actor",
        proposal_mode="learned_conditioned_prior",
        condition_skill_id="learned",
        candidate_budget=2,
        checkpoint_sha256="a" * 64,
        semantic_config_sha256="b" * 64,
    )
    rebound_task = GenerationTask.create(
        task_index=0,
        seed_record_id=source_task.seed_record_id,
        scenario_id=source_task.scenario_id,
        skill_id=source_task.skill_id,
        target_track_id=source_task.target_track_id,
        proposal_mode=source_task.proposal_mode,
        condition_skill_id=source_task.condition_skill_id,
        candidate_budget=source_task.candidate_budget,
        checkpoint_sha256="c" * 64,
        semantic_config_sha256="d" * 64,
    )
    rebound_plan = TaskPlan(
        semantic_config_sha256=rebound_task.semantic_config_sha256,
        execution_config_sha256="e" * 64,
        base_seed=2026,
        per_skill=1,
        candidate_budget=rebound_task.candidate_budget,
        tasks=(rebound_task,),
    )

    def candidates(seed_task: GenerationTask) -> list[GeneratedCandidate]:
        return [
            GeneratedCandidate(
                task_id=rebound_task.task_id,
                candidate_index=candidate_index,
                latent_seed=paired_latent_seed(
                    rebound_plan.base_seed,
                    seed_task,
                    candidate_index,
                ),
                scenario_id=rebound_task.scenario_id,
                skill_id=rebound_task.skill_id,
                proposal_mode=rebound_task.proposal_mode,
                checkpoint_sha256=rebound_task.checkpoint_sha256,
                semantic_config_sha256=rebound_task.semantic_config_sha256,
                overlay=GeneratedOverlay(
                    target_track_id=rebound_task.target_track_id,
                    future_xy_global=np.full(
                        (60, 2),
                        candidate_index,
                        dtype=np.float32,
                    ),
                ),
            )
            for candidate_index in range(rebound_task.candidate_budget)
        ]

    source_seed_raw = tmp_path / "source-seed-raw"
    write_raw_shard(
        source_seed_raw,
        0,
        candidates(source_task),
        semantic_config_sha256=rebound_plan.semantic_config_sha256,
        execution_config_sha256=rebound_plan.execution_config_sha256,
    )
    latent_sources = {rebound_task.task_id: source_task}
    mapped = recover_paired_pilot_tasks(
        rebound_plan,
        source_seed_raw,
        latent_seed_source_tasks=latent_sources,
    )
    assert mapped.durable_task_ids == {rebound_task.task_id}
    assert mapped.rebuild_task_ids == set()
    with pytest.raises(ValueError, match="different paired latent seed"):
        recover_paired_pilot_tasks(rebound_plan, source_seed_raw)

    rebound_seed_raw = tmp_path / "rebound-seed-raw"
    write_raw_shard(
        rebound_seed_raw,
        0,
        candidates(rebound_task),
        semantic_config_sha256=rebound_plan.semantic_config_sha256,
        execution_config_sha256=rebound_plan.execution_config_sha256,
    )
    default = recover_paired_pilot_tasks(rebound_plan, rebound_seed_raw)
    assert default.durable_task_ids == {rebound_task.task_id}
    assert default.rebuild_task_ids == set()
    with pytest.raises(ValueError, match="different paired latent seed"):
        recover_paired_pilot_tasks(
            rebound_plan,
            rebound_seed_raw,
            latent_seed_source_tasks=latent_sources,
        )


def test_paired_pilot_recovery_rejects_execution_configuration_mismatch(
    tmp_path: Path,
) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        0,
        [_paired_candidate(plan, 0, 0), _paired_candidate(plan, 0, 1)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=canonical_sha256(EXECUTION_B),
    )

    with pytest.raises(ValueError, match="different execution configuration"):
        recover_paired_pilot_tasks(plan, tmp_path / "raw")


def test_paired_pilot_recovery_rejects_extra_candidate_indices(tmp_path: Path) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        0,
        [
            _paired_candidate(plan, 0, 0),
            _paired_candidate(plan, 0, 1),
            _paired_candidate(plan, 0, 2),
        ],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )

    with pytest.raises(ValueError, match="outside the frozen budget"):
        recover_paired_pilot_tasks(plan, tmp_path / "raw")


def test_paired_pilot_recovery_rejects_unknown_shard_indices(tmp_path: Path) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        len(plan.tasks),
        [_paired_candidate(plan, 0, 0), _paired_candidate(plan, 0, 1)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )

    with pytest.raises(ValueError, match="outside the current task plan"):
        recover_paired_pilot_tasks(plan, tmp_path / "raw")


def test_paired_pilot_recovery_rejects_reordered_complete_candidates(
    tmp_path: Path,
) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    write_raw_shard(
        tmp_path / "raw",
        0,
        [_paired_candidate(plan, 0, 1), _paired_candidate(plan, 0, 0)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )

    with pytest.raises(ValueError, match="candidate order differs"):
        recover_paired_pilot_tasks(plan, tmp_path / "raw")


def test_paired_pilot_recovery_rebuilds_corrupt_task_shard(tmp_path: Path) -> None:
    plan = build_paired_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    commit = write_raw_shard(
        tmp_path / "raw",
        0,
        [_paired_candidate(plan, 0, 0), _paired_candidate(plan, 0, 1)],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )
    commit.arrays_path.write_bytes(commit.arrays_path.read_bytes() + b"corrupt")

    recovery = recover_paired_pilot_tasks(plan, tmp_path / "raw")

    assert recovery.durable_task_ids == set()
    assert recovery.invalid_task_ids == {plan.tasks[0].task_id}
    assert recovery.rebuild_task_ids == {task.task_id for task in plan.tasks}


def test_recovery_and_processing_progress_are_separate_models(tmp_path: Path) -> None:
    plan = build_pilot_task_plan(
        _records(),
        _config(),
        execution_config=EXECUTION_A,
        per_skill=1,
        candidate_budget=2,
    )
    recovery = recover_durable_tasks(plan, tmp_path / "empty-raw")
    scan_progress = recovery_progress(recovery)
    assert scan_progress.fraction == 1.0
    assert scan_progress.durable_tasks == 0

    processing = ProcessingProgress(
        total_tasks=len(plan.tasks),
        durable_tasks_at_start=0,
        newly_completed_tasks=1,
        in_flight_tasks=1,
        durable_candidates_at_start=0,
        newly_generated_candidates=2,
        elapsed_seconds=1.5,
    )
    assert processing.completed_tasks == 1
    assert processing.remaining_tasks == len(plan.tasks) - 1
    assert processing.fraction == 1 / len(plan.tasks)

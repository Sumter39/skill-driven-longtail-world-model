from __future__ import annotations

import json
from pathlib import Path

import pytest

from skilldrive.generation.contracts import GenerationTask, canonical_sha256
from skilldrive.generation.scheduler import TaskPlan
from skilldrive.performance.workload import (
    WORKLOAD_KIND,
    WORKLOAD_SCHEMA_VERSION,
    file_sha256,
    generation_task_to_row,
    load_fixed_workload,
    select_fixed_tasks,
    validate_active_pilot_summary,
    validate_active_task_plan,
    validate_shared_latent_pairs,
)


def _task(
    index: int,
    *,
    seed: str,
    skill: str,
    proposal_mode: str,
    condition: str,
) -> GenerationTask:
    return GenerationTask.create(
        task_index=index,
        seed_record_id=seed * 64,
        scenario_id=f"scenario-{seed}",
        skill_id=skill,
        target_track_id=f"track-{seed}",
        proposal_mode=proposal_mode,
        condition_skill_id=condition,
        candidate_budget=16,
        checkpoint_sha256="a" * 64,
        semantic_config_sha256="b" * 64,
    )


def _tasks() -> tuple[GenerationTask, ...]:
    return (
        _task(
            0,
            seed="1",
            skill="learned-a",
            proposal_mode="learned_conditioned_prior",
            condition="<none>",
        ),
        _task(
            1,
            seed="1",
            skill="learned-a",
            proposal_mode="learned_conditioned_prior",
            condition="learned-a",
        ),
        _task(
            2,
            seed="2",
            skill="search-b",
            proposal_mode="rule_guided_prior_search",
            condition="<none>",
        ),
        _task(
            3,
            seed="3",
            skill="learned-c",
            proposal_mode="learned_conditioned_prior",
            condition="<none>",
        ),
        _task(
            4,
            seed="3",
            skill="learned-c",
            proposal_mode="learned_conditioned_prior",
            condition="learned-c",
        ),
    )


def test_fixed_selection_is_order_independent_and_never_splits_pairs() -> None:
    tasks = _tasks()

    first = select_fixed_tasks(tasks, max_tasks=4, base_seed=2026)
    repeated = select_fixed_tasks(tuple(reversed(tasks)), max_tasks=4, base_seed=2026)

    assert [task.task_id for task in first] == [task.task_id for task in repeated]
    assert len(first) <= 4
    by_seed: dict[str, list[GenerationTask]] = {}
    for task in first:
        by_seed.setdefault(task.seed_record_id, []).append(task)
    for values in by_seed.values():
        if values[0].proposal_mode == "learned_conditioned_prior":
            assert len(values) == 2
            assert {task.condition_skill_id for task in values} == {
                "<none>",
                values[0].skill_id,
            }


def test_fixed_selection_rejects_incomplete_learned_pair() -> None:
    with pytest.raises(ValueError, match="atomic pair"):
        select_fixed_tasks((_tasks()[0],), max_tasks=1, base_seed=2026)


def test_active_pilot_summary_binds_current_semantics_and_execution() -> None:
    summary = {
        "status": "completed",
        "stage": "pilot",
        "checkpoint_sha256": "a" * 64,
        "generation_semantic_sha256": "b" * 64,
        "generation_execution_sha256": "c" * 64,
        "filter_semantic_sha256": "d" * 64,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }

    assert validate_active_pilot_summary(
        summary,
        active_checkpoint_sha256="a" * 64,
        generation_semantic_sha256="b" * 64,
    ) == ("c" * 64, "d" * 64)

    summary["generation_semantic_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="active frozen semantics"):
        validate_active_pilot_summary(
            summary,
            active_checkpoint_sha256="a" * 64,
            generation_semantic_sha256="b" * 64,
        )


def test_active_task_plan_rejects_execution_and_mixed_checkpoint_drift() -> None:
    tasks = _tasks()
    plan = TaskPlan(
        semantic_config_sha256="b" * 64,
        execution_config_sha256="c" * 64,
        base_seed=2026,
        per_skill=1,
        candidate_budget=16,
        tasks=tasks,
    )
    validate_active_task_plan(
        plan,
        generation_semantic_sha256="b" * 64,
        generation_execution_sha256="c" * 64,
        active_checkpoint_sha256="a" * 64,
    )

    with pytest.raises(ValueError, match="execution identity"):
        validate_active_task_plan(
            plan,
            generation_semantic_sha256="b" * 64,
            generation_execution_sha256="d" * 64,
            active_checkpoint_sha256="a" * 64,
        )

    stale = GenerationTask.create(
        task_index=tasks[-1].task_index,
        seed_record_id=tasks[-1].seed_record_id,
        scenario_id=tasks[-1].scenario_id,
        skill_id=tasks[-1].skill_id,
        target_track_id=tasks[-1].target_track_id,
        proposal_mode=tasks[-1].proposal_mode,
        condition_skill_id=tasks[-1].condition_skill_id,
        candidate_budget=16,
        checkpoint_sha256="e" * 64,
        semantic_config_sha256="b" * 64,
    )
    mixed = TaskPlan(
        semantic_config_sha256="b" * 64,
        execution_config_sha256="c" * 64,
        base_seed=2026,
        per_skill=1,
        candidate_budget=16,
        tasks=(*tasks[:-1], stale),
    )
    with pytest.raises(ValueError, match="non-active checkpoint"):
        validate_active_task_plan(
            mixed,
            generation_semantic_sha256="b" * 64,
            generation_execution_sha256="c" * 64,
            active_checkpoint_sha256="a" * 64,
        )


def test_shared_latent_validation_rejects_condition_control_drift() -> None:
    tasks = _tasks()
    latents = {task.task_id: (1, 2, 3) for task in tasks}
    validate_shared_latent_pairs(tasks, latents)

    latents[tasks[1].task_id] = (1, 2, 4)
    with pytest.raises(ValueError, match="do not share latent seeds"):
        validate_shared_latent_pairs(tasks, latents)


def test_workload_loader_binds_inputs_and_validation_boundary(tmp_path: Path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("stable", encoding="utf-8")
    task = _tasks()[2]
    value = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "kind": WORKLOAD_KIND,
        "selection_contract": "deterministic_atomic_condition_pairs_v1",
        "pilot": {},
        "counts": {
            "tasks": 1,
            "candidates": 16,
            "scenarios": 1,
            "by_proposal_mode": {"rule_guided_prior_search": 1},
        },
        "maximum_tasks": 1,
        "conditioned_control_shared_latents_verified": True,
        "filter_semantic_sha256": "c" * 64,
        "filter_dependency_sha256": {},
        "input_sha256": {"bound.txt": file_sha256(bound)},
        "tasks": [
            {
                "task": generation_task_to_row(task),
                "source_path": "train/scenario.parquet",
                "raw_commit": "raw/shard-00002.commit.json",
            }
        ],
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    value["workload_id"] = canonical_sha256(value)
    path = tmp_path / "workload.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    assert load_fixed_workload(path, repository_root=tmp_path) == value

    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        load_fixed_workload(path, repository_root=tmp_path)


def test_workload_loader_rejects_path_outside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    path = tmp_path / "workload.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes repository root"):
        load_fixed_workload(path, repository_root=repository_root)

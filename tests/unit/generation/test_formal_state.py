from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.generation.contracts import (
    FilterDecision,
    FilterRejection,
    GeneratedCandidate,
    GeneratedOverlay,
    canonical_sha256,
    filter_evaluation_id,
)
from skilldrive.generation.formal_storage import write_formal_filter_indexes
from skilldrive.generation.formal import (
    FormalPlanBindings,
    FormalTaskPlan,
    build_formal_task_plan,
    write_formal_task_plan,
)
from skilldrive.generation.formal_state import (
    FORMAL_PROGRESS_FILE_NAME,
    FORMAL_STATE_SUMMARY_NAME,
    FormalFilterReference,
    FormalProgressRuntime,
    FormalRawReference,
    FormalTaskState,
    build_formal_filter_references,
    build_formal_progress,
    build_formal_resume_plan,
    commit_formal_state_shard,
    commit_formal_state_shards,
    initialize_formal_state,
    load_formal_state,
    open_formal_state,
    recover_generated_from_raw,
    write_formal_candidate_invalid,
    write_formal_failure,
    write_formal_progress,
)
from skilldrive.generation.planning import latent_seed
from skilldrive.generation.storage import (
    RawShardCommit,
    verify_raw_shard,
    write_raw_shard,
)
from skilldrive.seeds.records import SeedRecord


FILTER_SEMANTIC_SHA = "6" * 64
EXECUTION_SHA = "7" * 64
OTHER_EXECUTION_SHA = "8" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(*, candidate_budget: int = 2) -> CounterfactualGenerationConfig:
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
            pilot_seed_records_per_skill=1,
            pilot_candidates_per_task=2,
            formal_candidates_per_task=candidate_budget,
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


def _plan(
    *,
    tasks_per_shard: int = 2,
    candidate_budget: int = 2,
) -> tuple[CounterfactualGenerationConfig, FormalTaskPlan]:
    config = _config(candidate_budget=candidate_budget)
    records = [
        _record("learned", "scene-a"),
        _record("search", "scene-b"),
        _record("learned", "scene-c"),
        _record("search", "scene-d"),
    ]
    bindings = FormalPlanBindings.for_fixture(
        config,
        config_sha256={
            "generation_config": "1" * 64,
            "filter_config": "2" * 64,
            "performance_config": "3" * 64,
        },
        source_sha256={
            "formal_seed_manifest": "4" * 64,
            "generation_source": "5" * 64,
            "filter_source": "9" * 64,
        },
        tasks_per_shard=tasks_per_shard,
        expected_task_count=len(records),
        expected_scenario_count=len(records),
        expected_skill_ids=config.formal_skill_ids,
        filter_semantic_sha256=FILTER_SEMANTIC_SHA,
        generation_execution_sha256=EXECUTION_SHA,
    )
    return config, build_formal_task_plan(records, config, bindings=bindings)


def _context(tmp_path: Path, config, plan):
    artifacts = write_formal_task_plan(tmp_path / "plan", plan, config=config)
    bindings = initialize_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    return artifacts, bindings


def _candidate(
    plan: FormalTaskPlan,
    task_index: int,
    candidate_index: int,
) -> GeneratedCandidate:
    task = plan.tasks[task_index]
    seed = latent_seed(plan.bindings.base_seed, task.task_id, candidate_index)
    future = np.column_stack(
        (
            np.linspace(0.0, 10.0 + candidate_index, 60),
            np.full(60, float(task_index)),
        )
    )
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
            future_xy_global=future,
        ),
    )


def _write_raw(
    root: Path,
    plan: FormalTaskPlan,
    shard_index: int,
    *,
    execution_sha256: str = EXECUTION_SHA,
    candidate_indices: dict[str, tuple[int, ...]] | None = None,
    directory_name: str = "raw",
) -> RawShardCommit:
    tasks = [task for task in plan.tasks if task.shard_index == shard_index]
    candidates = [
        _candidate(plan, task.task_index, index)
        for task in tasks
        for index in (
            range(task.candidate_budget)
            if candidate_indices is None
            else candidate_indices.get(task.task_id, ())
        )
    ]
    return write_raw_shard(
        root / directory_name,
        shard_index,
        candidates,
        semantic_config_sha256=plan.bindings.semantic_config_sha256,
        execution_config_sha256=execution_sha256,
    )


def _recover(root: Path, plan: FormalTaskPlan, bindings):
    return recover_generated_from_raw(
        plan,
        root / "raw",
        artifact_root=root,
        bindings=bindings,
    )


def _raw_states(root: Path, plan: FormalTaskPlan, bindings):
    recovery = _recover(root, plan, bindings)
    return {state.task_id: state for state in recovery.generated_task_states}


def _write_filter_commit(
    root: Path,
    raw: RawShardCommit | tuple[RawShardCommit, ...],
    bindings,
    *,
    accepted: set[tuple[str, int]],
    directory_name: str = "filter",
) -> Path:
    raw_commits = (raw,) if isinstance(raw, RawShardCommit) else raw
    directory = root / directory_name
    directory.mkdir(parents=True, exist_ok=True)
    accepted_rows: list[dict] = []
    rejected_rows: list[dict] = []
    task_ids: set[str] = set()
    for raw_commit in raw_commits:
        for reference in raw_commit.references:
            task_ids.add(reference.task_id)
            is_accepted = (reference.task_id, reference.candidate_index) in accepted
            stage = None if is_accepted else "map"
            row = {
                "candidate_id": reference.candidate_id,
                "filter_evaluation_id": filter_evaluation_id(
                    candidate_id=reference.candidate_id,
                    filter_config_sha256=bindings.filter_config_sha256,
                    filter_contract_version=bindings.filter_contract_version,
                ),
                "task_id": reference.task_id,
                "candidate_index": reference.candidate_index,
                "latent_seed": reference.latent_seed,
                "raw": {
                    "commit": raw_commit.commit_path.relative_to(root).as_posix(),
                    "arrays": raw_commit.arrays_path.relative_to(root).as_posix(),
                    "metadata": raw_commit.metadata_path.relative_to(root).as_posix(),
                    "offset": reference.raw_offset,
                },
                "metrics": {"first_failed_stage": stage},
            }
            if is_accepted:
                accepted_rows.append(row)
            else:
                rejected_rows.append(
                    {
                        **row,
                        "rejection_reasons": ["map.outside_drivable_area"],
                        "primary_rejection_reason": "map.outside_drivable_area",
                        "first_failed_stage": stage,
                    }
                )

    accepted_path = directory / "accepted.jsonl"
    rejected_path = directory / "rejected.jsonl"
    accepted_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted_rows),
        encoding="utf-8",
    )
    rejected_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected_rows),
        encoding="utf-8",
    )
    commit_path = directory / "filter-index.commit.json"
    commit = {
        "schema_version": 1,
        "kind": "formal_filter_commit",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "filter_config_sha256": bindings.filter_config_sha256,
        "filter_contract_version": bindings.filter_contract_version,
        "raw_commits": [
            {
                "path": raw_commit.commit_path.relative_to(root).as_posix(),
                "sha256": _sha256(raw_commit.commit_path),
            }
            for raw_commit in raw_commits
        ],
        "decision_sha256": canonical_sha256(
            {"accepted": accepted_rows, "rejected": rejected_rows}
        ),
        "counts": {
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "tasks": len(task_ids),
        },
        "task_statuses": {task_id: "complete" for task_id in sorted(task_ids)},
        "files": {
            "accepted": {
                "path": accepted_path.name,
                "size_bytes": accepted_path.stat().st_size,
                "sha256": _sha256(accepted_path),
            },
            "rejected": {
                "path": rejected_path.name,
                "size_bytes": rejected_path.stat().st_size,
                "sha256": _sha256(rejected_path),
            },
        },
    }
    commit_path.write_text(json.dumps(commit, sort_keys=True), encoding="utf-8")
    return commit_path


def _filter_reference(path: Path, root: Path, bindings, task, raw, *, plan):
    return FormalFilterReference.from_commit(
        path,
        artifact_root=root,
        plan=plan,
        bindings=bindings,
        task=task,
        raw=raw,
    )


def _refresh_filter_commit(commit_path: Path) -> None:
    value = json.loads(commit_path.read_text(encoding="utf-8"))
    accepted = [
        json.loads(line)
        for line in (commit_path.parent / "accepted.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rejected = [
        json.loads(line)
        for line in (commit_path.parent / "rejected.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for name in ("accepted", "rejected"):
        path = commit_path.parent / f"{name}.jsonl"
        value["files"][name].update(
            {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    value["decision_sha256"] = canonical_sha256(
        {"accepted": accepted, "rejected": rejected}
    )
    commit_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _rewrite_raw_metadata(commit: RawShardCommit, **changes) -> None:
    with gzip.open(commit.metadata_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    rows[0].update(changes)
    with gzip.open(commit.metadata_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    value = json.loads(commit.commit_path.read_text(encoding="utf-8"))
    value["files"]["metadata"].update(
        {
            "size_bytes": commit.metadata_path.stat().st_size,
            "sha256": _sha256(commit.metadata_path),
        }
    )
    commit.commit_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_open_auto_merges_raw_committed_before_state_checkpoint(tmp_path: Path) -> None:
    config, plan = _plan()
    artifacts = write_formal_task_plan(tmp_path / "plan", plan, config=config)
    _write_raw(tmp_path / "run", plan, 0)

    state = open_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )

    assert [item.status for item in state.task_states] == [
        "generated",
        "generated",
        "pending",
        "pending",
    ]
    assert (tmp_path / "run" / "state" / "shard-00000.commit.json").is_file()
    summary = json.loads(
        (tmp_path / "run" / FORMAL_STATE_SUMMARY_NAME).read_text(encoding="utf-8")
    )
    assert summary["bindings"]["filter_config_sha256"] == FILTER_SEMANTIC_SHA
    assert (
        summary["bindings"]["generation_execution_config_sha256"] == EXECUTION_SHA
    )
    assert summary["bindings"]["filter_contract_version"] == FILTER_CONTRACT_VERSION


def test_wrong_raw_execution_binding_rebuilds_only_its_shard(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    _write_raw(
        tmp_path / "run",
        plan,
        0,
        execution_sha256=OTHER_EXECUTION_SHA,
    )

    recovery = _recover(tmp_path / "run", plan, bindings)

    assert recovery.generated_task_states == ()
    assert recovery.rebuild_task_ids == frozenset(task.task_id for task in plan.tasks[:2])
    assert recovery.pending_task_ids == frozenset(task.task_id for task in plan.tasks[2:])


def test_raw_metadata_must_match_formal_task_semantics(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    commit = _write_raw(tmp_path / "run", plan, 0)
    _rewrite_raw_metadata(commit, scenario_id="wrong-scene")

    recovery = _recover(tmp_path / "run", plan, bindings)

    assert recovery.generated_task_states == ()
    assert recovery.rebuild_task_ids == frozenset(task.task_id for task in plan.tasks[:2])
    assert any(
        "scenario_id differs from the formal task" in issue.reason
        for issue in recovery.invalid_raw_shards
    )


def test_candidate_invalid_sidecar_closes_partial_raw_budget(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    raw = _write_raw(
        tmp_path / "run",
        plan,
        0,
        candidate_indices={task.task_id: (0,)},
    )
    invalid = write_formal_candidate_invalid(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        candidate_index=1,
        reason_code="schema.invalid_future_shape",
        message="decoder returned an invalid future shape",
    )

    recovery = _recover(tmp_path / "run", plan, bindings)

    assert len(recovery.generated_task_states) == 1
    state = recovery.generated_task_states[0]
    assert state.raw is not None and state.raw.candidate_indices == (0,)
    assert state.invalid_candidates == (invalid,)
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=(state,),
    )
    loaded = load_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=bindings.task_plan_sha256,
    )
    assert loaded.task_states[0] == state
    assert raw.candidate_count == 1


def test_open_preserves_partial_invalid_candidate_and_resumes_missing_index(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=1)
    artifacts, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    invalid = write_formal_candidate_invalid(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        candidate_index=0,
        reason_code="schema.invalid_future_shape",
        message="candidate zero is durably invalid",
    )

    partial = open_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    assert partial.task_states[0] == FormalTaskState.pending(
        task,
        invalid_candidates=(invalid,),
    )
    assert task.task_id in build_formal_resume_plan(partial).generate_task_ids

    _write_raw(
        tmp_path / "run",
        plan,
        0,
        candidate_indices={task.task_id: (1,)},
    )
    complete = open_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    state = complete.task_states[0]
    assert state.status == "generated"
    assert state.raw is not None and state.raw.candidate_indices == (1,)
    assert state.invalid_candidates == (invalid,)


def test_sidecar_writer_rejects_task_outside_frozen_plan(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    foreign = replace(
        plan.tasks[0],
        task_index=len(plan.tasks),
        phase_index=len(plan.tasks),
        shard_index=len(plan.tasks),
    )

    with pytest.raises(ValueError, match="outside the frozen task plan"):
        write_formal_candidate_invalid(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            task=foreign,
            candidate_index=0,
            reason_code="schema.invalid_future_shape",
            message="must not be written",
        )


def test_pending_state_cannot_hide_a_complete_invalid_budget(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    invalid = tuple(
        write_formal_candidate_invalid(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            task=task,
            candidate_index=index,
            reason_code="schema.invalid_future_shape",
            message="invalid",
        )
        for index in range(task.candidate_budget)
    )

    with pytest.raises(ValueError, match="pending task state already covers"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(FormalTaskState.pending(task, invalid_candidates=invalid),),
        )


def test_generation_state_rejects_budget_gap_and_overlap(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    commit = _write_raw(
        tmp_path / "run",
        plan,
        0,
        candidate_indices={task.task_id: (0,)},
    )
    raw = FormalRawReference.from_commit(
        commit,
        task_id=task.task_id,
        artifact_root=tmp_path / "run",
    )
    with pytest.raises(ValueError, match="cover candidate budget"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(FormalTaskState.generated(task, raw),),
        )

    overlapping = write_formal_candidate_invalid(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        candidate_index=0,
        reason_code="schema.invalid_future_shape",
        message="overlap fixture",
    )
    with pytest.raises(ValueError, match="overlap"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(
                FormalTaskState.generated(
                    task,
                    raw,
                    invalid_candidates=(overlapping,),
                ),
            ),
        )


def test_state_commit_does_not_publish_missing_artifact_reference(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    missing = FormalRawReference(
        commit_path="raw/missing.commit.json",
        commit_sha256="0" * 64,
        shard_index=0,
        execution_config_sha256=EXECUTION_SHA,
        arrays_path="raw/missing.npz",
        arrays_sha256="1" * 64,
        metadata_path="raw/missing.meta.jsonl.gz",
        metadata_sha256="2" * 64,
        candidate_indices=tuple(range(task.candidate_budget)),
        candidate_ids_sha256="3" * 64,
    )

    with pytest.raises(ValueError, match="raw artifact is missing"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(FormalTaskState.generated(task, missing),),
        )
    assert not (tmp_path / "run" / "state" / "shard-00000.commit.json").exists()


def test_invalid_candidate_sidecar_damage_is_rejected(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=1)
    artifacts, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    invalid = tuple(
        write_formal_candidate_invalid(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            task=task,
            candidate_index=index,
            reason_code="schema.invalid_future_shape",
            message=f"invalid {index}",
        )
        for index in range(task.candidate_budget)
    )
    state = FormalTaskState.generated(task, invalid_candidates=invalid)
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=(state,),
    )
    sidecar = tmp_path / "run" / invalid[0].sidecar_path
    sidecar.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid-generation sidecar SHA-256"):
        load_formal_state(
            tmp_path / "run",
            plan=plan,
            task_plan_sha256=artifacts.task_plan_sha256,
        )


def test_strict_filter_round_trip_binds_raw_and_decisions(tmp_path: Path) -> None:
    config, plan = _plan()
    artifacts, bindings = _context(tmp_path, config, plan)
    raw_commit = _write_raw(tmp_path / "run", plan, 0)
    raw_states = _raw_states(tmp_path / "run", plan, bindings)
    first, second = plan.tasks[:2]
    filter_commit = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(first.task_id, 0)},
    )
    filter_references = build_formal_filter_references(
        filter_commit,
        artifact_root=tmp_path / "run",
        plan=plan,
        bindings=bindings,
        raw_by_task={
            first.task_id: raw_states[first.task_id].raw,
            second.task_id: raw_states[second.task_id].raw,
        },
    )
    first_filter = filter_references[first.task_id]
    second_filter = filter_references[second.task_id]
    states = (
        FormalTaskState.accepted(first, raw_states[first.task_id].raw, first_filter),
        FormalTaskState.rejected(second, raw_states[second.task_id].raw, second_filter),
    )
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=states,
    )

    loaded = load_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )

    assert loaded.task_states[:2] == states
    assert first_filter.formal_plan_id == plan.formal_plan_id
    assert first_filter.task_plan_sha256 == artifacts.task_plan_sha256
    assert first_filter.raw_commit_sha256 == raw_states[first.task_id].raw.commit_sha256
    assert first_filter.decision_sha256 == second_filter.decision_sha256


def test_filter_commit_cannot_omit_an_entire_raw_task(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    raw_commit = _write_raw(root, plan, 0)
    raw_states = _raw_states(root, plan, bindings)
    first, omitted = plan.tasks[:2]
    commit_path = _write_filter_commit(
        root,
        raw_commit,
        bindings,
        accepted={(first.task_id, 0)},
    )
    for name in ("accepted", "rejected"):
        path = commit_path.parent / f"{name}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows = [row for row in rows if row["task_id"] != omitted.task_id]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    value = json.loads(commit_path.read_text(encoding="utf-8"))
    value["counts"] = {"accepted": 1, "rejected": 1, "tasks": 1}
    value["task_statuses"].pop(omitted.task_id)
    commit_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    _refresh_filter_commit(commit_path)

    with pytest.raises(ValueError, match="cover every raw candidate"):
        _filter_reference(
            commit_path,
            root,
            bindings,
            first,
            raw_states[first.task_id].raw,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("candidate_index", 99, "candidate_index"),
        ("latent_seed", 99, "latent_seed"),
        ("filter_evaluation_id", "0" * 64, "filter_evaluation_id"),
        ("raw.commit", "raw/wrong.commit.json", "unbound raw commit"),
        ("raw.offset", 99, "raw offset"),
        ("raw.arrays", "raw/wrong.npz", "raw arrays path"),
        ("raw.metadata", "raw/wrong.jsonl", "raw metadata path"),
    ],
)
def test_filter_row_field_drift_is_rejected(
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    raw_commit = _write_raw(tmp_path / "run", plan, 0)
    raw_states = _raw_states(tmp_path / "run", plan, bindings)
    task = plan.tasks[0]
    commit_path = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(task.task_id, 0)},
    )
    accepted_path = commit_path.parent / "accepted.jsonl"
    row = json.loads(accepted_path.read_text(encoding="utf-8"))
    if field.startswith("raw."):
        row["raw"][field.split(".", 1)[1]] = replacement
    else:
        row[field] = replacement
    accepted_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    _refresh_filter_commit(commit_path)

    with pytest.raises(ValueError, match=message):
        _filter_reference(
            commit_path,
            tmp_path / "run",
            bindings,
            task,
            raw_states[task.task_id].raw,
            plan=plan,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("formal_plan_id", "0" * 64, "semantic drift"),
        ("task_plan_sha256", "0" * 64, "semantic drift"),
        ("filter_config_sha256", "0" * 64, "semantic drift"),
        ("filter_contract_version", "wrong", "semantic drift"),
        ("decision_sha256", "0" * 64, "decision digest"),
        ("task_statuses", {}, "task status"),
    ],
)
def test_filter_commit_binding_drift_is_rejected(
    tmp_path: Path,
    field: str,
    replacement,
    message: str,
) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    raw_commit = _write_raw(tmp_path / "run", plan, 0)
    raw_states = _raw_states(tmp_path / "run", plan, bindings)
    task = plan.tasks[0]
    commit_path = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(task.task_id, 0)},
    )
    value = json.loads(commit_path.read_text(encoding="utf-8"))
    value[field] = replacement
    commit_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _filter_reference(
            commit_path,
            tmp_path / "run",
            bindings,
            task,
            raw_states[task.task_id].raw,
            plan=plan,
        )


def test_filter_raw_commit_sha_drift_is_rejected(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    raw_commit = _write_raw(tmp_path / "run", plan, 0)
    raw_states = _raw_states(tmp_path / "run", plan, bindings)
    task = plan.tasks[0]
    commit_path = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(task.task_id, 0)},
    )
    value = json.loads(commit_path.read_text(encoding="utf-8"))
    value["raw_commits"][0]["sha256"] = "0" * 64
    commit_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="raw commit SHA-256"):
        _filter_reference(
            commit_path,
            tmp_path / "run",
            bindings,
            task,
            raw_states[task.task_id].raw,
            plan=plan,
        )


def test_filter_cannot_substitute_byte_identical_raw_at_another_path(
    tmp_path: Path,
) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    raw_commit = _write_raw(root, plan, 0)
    raw_states = _raw_states(root, plan, bindings)
    copied_directory = root / "raw-copy"
    shutil.copytree(root / "raw", copied_directory)
    copied = verify_raw_shard(
        copied_directory / raw_commit.commit_path.name,
        expected_semantic_config_sha256=plan.bindings.semantic_config_sha256,
    )
    assert _sha256(copied.commit_path) == _sha256(raw_commit.commit_path)
    task = plan.tasks[0]
    commit_path = _write_filter_commit(
        root,
        copied,
        bindings,
        accepted={(task.task_id, 0)},
        directory_name="filter-copied-raw",
    )

    with pytest.raises(ValueError, match="raw commit differs from state"):
        _filter_reference(
            commit_path,
            root,
            bindings,
            task,
            raw_states[task.task_id].raw,
            plan=plan,
        )


def test_filter_commit_rejects_foreign_raw_task(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    raw_commit = _write_raw(root, plan, 0)
    raw_states = _raw_states(root, plan, bindings)
    foreign = GeneratedCandidate(
        task_id="0" * 64,
        candidate_index=0,
        latent_seed=1,
        scenario_id="foreign-scene",
        skill_id="learned",
        proposal_mode="learned_conditioned_prior",
        checkpoint_sha256=plan.bindings.checkpoint_sha256,
        semantic_config_sha256=plan.bindings.semantic_config_sha256,
        overlay=GeneratedOverlay(
            target_track_id="foreign-actor",
            future_xy_global=np.zeros((60, 2), dtype=np.float64),
        ),
    )
    foreign_raw = write_raw_shard(
        root / "foreign-raw",
        1,
        (foreign,),
        semantic_config_sha256=plan.bindings.semantic_config_sha256,
        execution_config_sha256=EXECUTION_SHA,
    )
    task = plan.tasks[0]
    commit_path = _write_filter_commit(
        root,
        (raw_commit, foreign_raw),
        bindings,
        accepted={(task.task_id, 0)},
        directory_name="filter-with-foreign-task",
    )

    with pytest.raises(ValueError, match="outside the task plan"):
        _filter_reference(
            commit_path,
            root,
            bindings,
            task,
            raw_states[task.task_id].raw,
            plan=plan,
        )


def test_generated_state_cannot_replace_durable_raw_reference(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    _write_raw(root, plan, 0)
    original = _raw_states(root, plan, bindings)
    commit_formal_state_shard(
        root,
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=tuple(original[task.task_id] for task in plan.tasks[:2]),
    )
    _write_raw(root, plan, 0, directory_name="raw-replacement")
    replacement = {
        state.task_id: state
        for state in recover_generated_from_raw(
            plan,
            root / "raw-replacement",
            artifact_root=root,
            bindings=bindings,
        ).generated_task_states
    }

    with pytest.raises(ValueError, match="durable raw reference"):
        commit_formal_state_shard(
            root,
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=tuple(replacement[task.task_id] for task in plan.tasks[:2]),
        )


def test_batch_state_commit_binds_one_filter_commit_across_shards(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=1)
    artifacts, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    first_raw = _write_raw(root, plan, 0)
    second_raw = _write_raw(root, plan, 1)
    raw_states = _raw_states(root, plan, bindings)
    first, second = plan.tasks[:2]
    filter_commit = _write_filter_commit(
        root,
        (first_raw, second_raw),
        bindings,
        accepted={(first.task_id, 0)},
    )
    references = build_formal_filter_references(
        filter_commit,
        artifact_root=root,
        plan=plan,
        bindings=bindings,
        raw_by_task={
            first.task_id: raw_states[first.task_id].raw,
            second.task_id: raw_states[second.task_id].raw,
        },
    )

    commits = commit_formal_state_shards(
        root,
        plan=plan,
        bindings=bindings,
        shard_states={
            0: (
                FormalTaskState.accepted(
                    first,
                    raw_states[first.task_id].raw,
                    references[first.task_id],
                ),
            ),
            1: (
                FormalTaskState.rejected(
                    second,
                    raw_states[second.task_id].raw,
                    references[second.task_id],
                ),
            ),
        },
    )

    assert [item.shard_index for item in commits] == [0, 1]
    state = load_formal_state(
        root,
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    assert state.task_states[0].status == "accepted"
    assert state.task_states[1].status == "rejected"


def test_formal_filter_writer_round_trips_through_state_verifier(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=1)
    _, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    raw = _write_raw(root, plan, 0)
    decisions = []
    for reference in raw.references:
        accepted = reference.candidate_index == 0
        decisions.append(
            FilterDecision.create(
                candidate_id=reference.candidate_id,
                filter_config_sha256=bindings.filter_config_sha256,
                filter_contract_version=FILTER_CONTRACT_VERSION,
                accepted=accepted,
                rejection_reasons=(
                    ()
                    if accepted
                    else (FilterRejection.OUTSIDE_DRIVABLE_AREA,)
                ),
                metrics={"first_failed_stage": None if accepted else "map"},
            )
        )
    commit = write_formal_filter_indexes(
        root / "filter-writer",
        (raw,),
        decisions,
        artifact_root=root,
        bindings=bindings,
    )
    raw_state = _raw_states(root, plan, bindings)[plan.tasks[0].task_id]

    reference = FormalFilterReference.from_commit(
        commit,
        artifact_root=root,
        plan=plan,
        bindings=bindings,
        task=plan.tasks[0],
        raw=raw_state.raw,
    )

    assert reference.accepted_count == 1
    assert reference.rejected_count == 1


def test_filtered_state_cannot_replace_filter_or_raw_reference(tmp_path: Path) -> None:
    config, plan = _plan()
    _, bindings = _context(tmp_path, config, plan)
    first_task, second_task = plan.tasks[:2]
    raw_commit = _write_raw(tmp_path / "run", plan, 0)
    raw_states = _raw_states(tmp_path / "run", plan, bindings)

    def filtered_states(commit_path: Path, raw_map):
        return tuple(
            FormalTaskState.filtered(
                task,
                raw_map[task.task_id].raw,
                _filter_reference(
                    commit_path,
                    tmp_path / "run",
                    bindings,
                    task,
                    raw_map[task.task_id].raw,
                    plan=plan,
                ),
            )
            for task in (first_task, second_task)
        )

    first_filter = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(first_task.task_id, 0)},
        directory_name="filter-first",
    )
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=filtered_states(first_filter, raw_states),
    )
    second_filter = _write_filter_commit(
        tmp_path / "run",
        raw_commit,
        bindings,
        accepted={(first_task.task_id, 0)},
        directory_name="filter-second",
    )
    with pytest.raises(ValueError, match="filter reference"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=filtered_states(second_filter, raw_states),
        )

    replacement_raw = _write_raw(
        tmp_path / "run",
        plan,
        0,
        directory_name="raw-replacement",
    )
    replacement_map = {
        state.task_id: state
        for state in recover_generated_from_raw(
            plan,
            tmp_path / "run" / "raw-replacement",
            artifact_root=tmp_path / "run",
            bindings=bindings,
        ).generated_task_states
    }
    replacement_filter = _write_filter_commit(
        tmp_path / "run",
        replacement_raw,
        bindings,
        accepted={(first_task.task_id, 0)},
        directory_name="filter-replacement-raw",
    )
    with pytest.raises(ValueError, match="raw reference"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=filtered_states(replacement_filter, replacement_map),
        )


def test_failure_resume_and_progress_include_invalid_candidates(tmp_path: Path) -> None:
    config, plan = _plan(tasks_per_shard=2)
    artifacts, bindings = _context(tmp_path, config, plan)
    retryable = write_formal_failure(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=plan.tasks[0],
        stage="generation",
        retryable=True,
        reason_code="worker_interrupted",
        message="retry",
        attempt=1,
    )
    terminal = write_formal_failure(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=plan.tasks[1],
        stage="generation",
        retryable=False,
        reason_code="unsupported_input",
        message="stop",
        attempt=1,
    )
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=(
            FormalTaskState.failed(plan.tasks[0], retryable),
            FormalTaskState.failed(plan.tasks[1], terminal),
        ),
    )
    invalid = tuple(
        write_formal_candidate_invalid(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            task=plan.tasks[2],
            candidate_index=index,
            reason_code="schema.invalid_future_shape",
            message="invalid",
        )
        for index in range(plan.tasks[2].candidate_budget)
    )
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=1,
        task_states=(
            FormalTaskState.rejected(plan.tasks[2], invalid_candidates=invalid),
            FormalTaskState.pending(plan.tasks[3]),
        ),
    )
    state = load_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    resume = build_formal_resume_plan(state)
    assert resume.generate_task_ids == (plan.tasks[0].task_id, plan.tasks[3].task_id)
    assert resume.terminal_failed_task_ids == (plan.tasks[1].task_id,)
    assert resume.completed_task_ids == (plan.tasks[2].task_id,)

    progress = build_formal_progress(
        state,
        plan=plan,
        runtime=FormalProgressRuntime(
            updated_at_utc="2026-07-22T00:00:00Z",
            elapsed_seconds=10,
            candidates_per_second=0.2,
            accepted_per_second=0,
            eta_seconds=20,
        ),
    )
    path = write_formal_progress(tmp_path / "run", progress)
    value = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == FORMAL_PROGRESS_FILE_NAME
    assert value["candidates"]["generated"] == 2
    assert value["candidates"]["raw_stored"] == 0
    assert value["candidates"]["invalid_generation"] == 2
    assert value["candidates"]["rejected"] == 2
    assert value["candidates"]["stage_rejection_counts"] == {
        "generation_invalid": 2
    }


def test_retryable_failure_preserves_durable_candidates_across_resume(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=1)
    artifacts, bindings = _context(tmp_path, config, plan)
    task = plan.tasks[0]
    first_invalid = write_formal_candidate_invalid(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        candidate_index=0,
        reason_code="schema.invalid_future_shape",
        message="first invalid candidate",
    )
    failure = write_formal_failure(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        stage="generation",
        retryable=True,
        reason_code="worker_interrupted",
        message="retry",
        attempt=1,
    )
    failed_state = FormalTaskState.failed(
        task,
        failure,
        invalid_candidates=(first_invalid,),
    )
    commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=(failed_state,),
    )

    reopened = open_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    assert reopened.task_states[0] == failed_state
    assert task.task_id in build_formal_resume_plan(reopened).generate_task_ids

    next_failure = write_formal_failure(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        stage="generation",
        retryable=True,
        reason_code="worker_interrupted",
        message="retry again",
        attempt=2,
    )
    with pytest.raises(ValueError, match="discard or replace an invalid candidate"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(FormalTaskState.failed(task, next_failure),),
        )

    second_invalid = write_formal_candidate_invalid(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        task=task,
        candidate_index=1,
        reason_code="schema.invalid_future_shape",
        message="second invalid candidate",
    )
    completed = open_formal_state(
        tmp_path / "run",
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    assert completed.task_states[0] == FormalTaskState.generated(
        task,
        invalid_candidates=(first_invalid, second_invalid),
    )
    assert task.task_id in build_formal_resume_plan(completed).finalize_task_ids


def test_open_preserves_terminal_raw_and_retryable_filtered_failures(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=2)
    artifacts, bindings = _context(tmp_path, config, plan)
    root = tmp_path / "run"
    first, second = plan.tasks[:2]
    raw_commit = _write_raw(root, plan, 0)
    raw_states = _raw_states(root, plan, bindings)
    filter_commit = _write_filter_commit(
        root,
        raw_commit,
        bindings,
        accepted={(second.task_id, 0)},
    )
    second_filter = _filter_reference(
        filter_commit,
        root,
        bindings,
        second,
        raw_states[second.task_id].raw,
        plan=plan,
    )
    terminal_failure = write_formal_failure(
        root,
        plan=plan,
        bindings=bindings,
        task=first,
        stage="filtering",
        retryable=False,
        reason_code="unsupported_input",
        message="terminal",
        attempt=1,
    )
    retryable_failure = write_formal_failure(
        root,
        plan=plan,
        bindings=bindings,
        task=second,
        stage="finalize",
        retryable=True,
        reason_code="worker_interrupted",
        message="retry finalize",
        attempt=1,
    )
    states = (
        FormalTaskState.failed(
            first,
            terminal_failure,
            raw=raw_states[first.task_id].raw,
        ),
        FormalTaskState.failed(
            second,
            retryable_failure,
            raw=raw_states[second.task_id].raw,
            filter=second_filter,
        ),
    )
    commit_formal_state_shard(
        root,
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=states,
    )

    reopened = open_formal_state(
        root,
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    assert reopened.task_states[:2] == states
    resume = build_formal_resume_plan(reopened)
    assert resume.terminal_failed_task_ids == (first.task_id,)
    assert resume.finalize_task_ids == (second.task_id,)


def test_state_commit_corruption_and_terminal_regression_are_rejected(
    tmp_path: Path,
) -> None:
    config, plan = _plan(tasks_per_shard=2)
    artifacts, bindings = _context(tmp_path, config, plan)
    invalid = tuple(
        write_formal_candidate_invalid(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            task=plan.tasks[0],
            candidate_index=index,
            reason_code="schema.invalid_future_shape",
            message="invalid",
        )
        for index in range(plan.tasks[0].candidate_budget)
    )
    commit = commit_formal_state_shard(
        tmp_path / "run",
        plan=plan,
        bindings=bindings,
        shard_index=0,
        task_states=(
            FormalTaskState.rejected(plan.tasks[0], invalid_candidates=invalid),
            FormalTaskState.pending(plan.tasks[1]),
        ),
    )
    with pytest.raises(ValueError, match="completed formal task state is immutable"):
        commit_formal_state_shard(
            tmp_path / "run",
            plan=plan,
            bindings=bindings,
            shard_index=0,
            task_states=(
                FormalTaskState.pending(plan.tasks[0]),
                FormalTaskState.pending(plan.tasks[1]),
            ),
        )
    value = json.loads(commit.path.read_text(encoding="utf-8"))
    value["tasks"][1]["status"] = "generated"
    commit.path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="state shard SHA-256"):
        load_formal_state(
            tmp_path / "run",
            plan=plan,
            task_plan_sha256=artifacts.task_plan_sha256,
        )

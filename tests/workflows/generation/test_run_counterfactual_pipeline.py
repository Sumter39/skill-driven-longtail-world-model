from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.generation.run_counterfactual_pipeline as pipeline_module
import skilldrive.data.av2_reader as av2_reader
import skilldrive.filtering.fingerprint as filter_fingerprint
import skilldrive.filtering.pipeline as filter_pipeline
import skilldrive.generation as generation
import skilldrive.skills.detection as skill_detection
import skilldrive.skills.loader as skill_loader
from scripts.generation.run_counterfactual_pipeline import (
    _validate_frozen_leakage_audit,
    _validate_seed_source_path,
)
from skilldrive.generation import (
    FilterDecision,
    FilterRejection,
    GeneratedCandidate,
    GeneratedOverlay,
    GenerationTask,
    TaskPlan,
    latent_seed,
    seed_record_id,
    write_raw_shard,
    write_task_plan,
)
from skilldrive.seeds.records import SeedRecord


SEMANTIC_SHA = "a" * 64
EXECUTION_SHA = "b" * 64
CHECKPOINT_SHA = "c" * 64


def _record(index: int) -> SeedRecord:
    scenario_id = f"scene-{index:02d}"
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id="test_skill",
        initiator_track_id=f"{scenario_id}-target",
        responder_track_id=f"{scenario_id}-other",
        role_track_ids={
            "actor": f"{scenario_id}-target",
            "other": f"{scenario_id}-other",
        },
        trigger_score=0.5,
        seed_risk_metric="minimum_distance_m",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "minimum_distance_m",
            "target_range": [0.0, 2.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"target_gap_m": 1.0},
    )


def _task_plan(records: tuple[SeedRecord, ...], candidate_budget: int) -> TaskPlan:
    tasks = tuple(
        GenerationTask.create(
            task_index=index,
            seed_record_id=seed_record_id(record),
            scenario_id=record.scenario_id,
            skill_id=record.skill_id,
            target_track_id=record.role_track_ids["actor"],
            proposal_mode="rule_guided_prior_search",
            condition_skill_id="<none>",
            candidate_budget=candidate_budget,
            checkpoint_sha256=CHECKPOINT_SHA,
            semantic_config_sha256=SEMANTIC_SHA,
        )
        for index, record in enumerate(records)
    )
    return TaskPlan(
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
        base_seed=2026,
        per_skill=1,
        candidate_budget=candidate_budget,
        tasks=tasks,
    )


def _candidate(
    plan: TaskPlan,
    records: tuple[SeedRecord, ...],
    task_index: int,
    candidate_index: int,
) -> GeneratedCandidate:
    task = plan.tasks[task_index]
    record = records[task_index]
    future = np.column_stack(
        (
            np.linspace(0.0, 10.0 + candidate_index, 60, dtype=np.float32),
            np.full(60, float(task_index), dtype=np.float32),
        )
    )
    return GeneratedCandidate(
        task_id=task.task_id,
        candidate_index=candidate_index,
        latent_seed=latent_seed(plan.base_seed, task.task_id, candidate_index),
        scenario_id=task.scenario_id,
        skill_id=task.skill_id,
        proposal_mode=task.proposal_mode,
        checkpoint_sha256=task.checkpoint_sha256,
        semantic_config_sha256=task.semantic_config_sha256,
        overlay=GeneratedOverlay(
            target_track_id=task.target_track_id,
            future_xy_global=future,
        ),
        metadata={
            "condition_skill_id": task.condition_skill_id,
            "primary_generated_role": "actor",
            "requested_parameters": record.sampled_parameters,
            "detection_mode": record.evidence["detection_mode"],
        },
    )


def _install_filter_smoke_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    records: tuple[SeedRecord, ...],
) -> tuple[dict[str, str], dict[str, Path]]:
    config = SimpleNamespace(
        inputs=SimpleNamespace(
            seed_manifest=tmp_path / "seed-manifest.csv",
            data_root=tmp_path / "data",
        ),
        sampling=SimpleNamespace(base_seed=2026),
        formal_catalog=tmp_path / "configs" / "skills" / "catalog.yaml",
        skills_by_id={
            "test_skill": SimpleNamespace(primary_generated_role="actor")
        },
    )
    filter_state = {"sha256": "1" * 64}
    paths = {
        "config": tmp_path / "counterfactual.yaml",
        "filter_config": tmp_path / "filters.yaml",
        "detection_config": tmp_path / "detection.yaml",
    }

    monkeypatch.setattr(
        pipeline_module,
        "load_counterfactual_config",
        lambda path: config,
    )
    monkeypatch.setattr(pipeline_module, "run_audit", lambda **kwargs: {"status": "passed"})
    monkeypatch.setattr(pipeline_module, "load_filter_config", lambda path: object())
    monkeypatch.setattr(pipeline_module, "read_seed_records", lambda path: records)
    monkeypatch.setattr(
        pipeline_module,
        "_select_smoke_records",
        lambda loaded_config, loaded_records: list(records),
    )
    monkeypatch.setattr(
        pipeline_module,
        "_validate_seed_source_path",
        lambda data_root, source_path: tmp_path / source_path,
    )
    monkeypatch.setattr(
        generation,
        "semantic_generation_config_sha256",
        lambda loaded_config: SEMANTIC_SHA,
    )
    monkeypatch.setattr(
        filter_fingerprint,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(
            semantic_sha256=filter_state["sha256"],
            file_sha256={"filters.yaml": filter_state["sha256"]},
        ),
    )
    monkeypatch.setattr(av2_reader, "load_av2_scenario", lambda path: object())
    monkeypatch.setattr(
        skill_detection,
        "load_detection_config",
        lambda path: object(),
    )
    monkeypatch.setattr(
        skill_loader,
        "load_skill",
        lambda path: SimpleNamespace(skill_id=path.stem),
    )

    def validate_candidates(inputs, *, filter_semantic_sha256, **kwargs):
        decisions = []
        validations = []
        for index, item in enumerate(inputs):
            accepted = index % 2 == 0
            decisions.append(
                FilterDecision.create(
                    candidate_id=item.bound.raw.candidate_id,
                    filter_config_sha256=filter_semantic_sha256,
                    filter_contract_version=filter_pipeline.FILTER_CONTRACT_VERSION,
                    accepted=accepted,
                    rejection_reasons=(
                        ()
                        if accepted
                        else (FilterRejection.INVALID_FUTURE_DTYPE,)
                    ),
                    metrics={
                        "first_failed_stage": None if accepted else "schema_finite",
                        "candidate_index": item.bound.raw.candidate_index,
                    },
                )
            )
            validations.append(SimpleNamespace(quality_passed=accepted))
        return SimpleNamespace(
            decisions=tuple(decisions),
            validations=tuple(validations),
            stage_execution_counts={"schema_finite": len(decisions)},
            stage_elapsed_seconds={"schema_finite": 0.0},
        )

    monkeypatch.setattr(filter_pipeline, "validate_candidates", validate_candidates)
    return filter_state, paths


def _prepare_filter_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_count: int,
    candidate_budget: int,
    candidate_indices: tuple[range, ...],
):
    records = tuple(_record(index) for index in range(task_count))
    plan = _task_plan(records, candidate_budget)
    output_root = tmp_path / "outputs"
    smoke_root = output_root / "pilot" / "smoke"
    write_task_plan(smoke_root, plan)
    candidates = [
        _candidate(plan, records, task_index, candidate_index)
        for task_index, indices in enumerate(candidate_indices)
        for candidate_index in indices
    ]
    commit = write_raw_shard(
        smoke_root / "raw",
        0,
        candidates,
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )
    filter_state, paths = _install_filter_smoke_dependencies(
        monkeypatch,
        tmp_path,
        records,
    )
    arguments = {
        "config_path": paths["config"],
        "filter_config_path": paths["filter_config"],
        "detection_config_path": paths["detection_config"],
        "output_root": output_root,
    }
    return commit, filter_state, arguments


def test_validation_guard_rejects_non_train_path_before_open(tmp_path) -> None:
    with pytest.raises(ValueError, match="Formal Train"):
        _validate_seed_source_path(tmp_path, "val/example/scenario.parquet")
    with pytest.raises(ValueError, match="Formal Train"):
        _validate_seed_source_path(tmp_path, "train/../val/scenario.parquet")


def test_frozen_leakage_audit_requires_zero_overlap() -> None:
    valid = {
        "leakage_check": {
            "candidate_pool_final_validation_overlap": 0,
            "candidate_pool_internal_validation_overlap": 0,
            "candidate_pool_outside_formal_train": 0,
            "selected_final_validation_overlap": 0,
            "selected_internal_validation_overlap": 0,
            "status": "passed",
        }
    }
    _validate_frozen_leakage_audit(valid)
    valid["leakage_check"]["selected_final_validation_overlap"] = 1
    with pytest.raises(ValueError, match="leakage audit failed"):
        _validate_frozen_leakage_audit(valid)


@pytest.mark.parametrize(
    ("task_count", "candidate_budget", "candidate_indices", "expected_fragments"),
    (
        (2, 9, (range(8), range(8)), ("'partial_tasks': 2",)),
        (
            2,
            8,
            (range(7), range(9)),
            ("'partial_tasks': 1", "'extra_candidate_tasks': 1"),
        ),
        (3, 8, (range(8), range(8), range(0)), ("'pending_tasks': 1",)),
    ),
    ids=("partial-budget", "extra-candidate", "missing-task-budget"),
)
def test_filter_smoke_rejects_incomplete_or_extra_durable_candidate_budgets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_count: int,
    candidate_budget: int,
    candidate_indices: tuple[range, ...],
    expected_fragments: tuple[str, ...],
) -> None:
    _, _, arguments = _prepare_filter_smoke(
        tmp_path,
        monkeypatch,
        task_count=task_count,
        candidate_budget=candidate_budget,
        candidate_indices=candidate_indices,
    )

    with pytest.raises(ValueError, match="complete raw candidate budget") as error:
        pipeline_module.run_filter_smoke(**arguments)

    for fragment in expected_fragments:
        assert fragment in str(error.value)


@pytest.mark.parametrize("mutation", ("sha256", "size", "mtime"))
def test_filter_smoke_detects_each_raw_snapshot_identity_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _, _, arguments = _prepare_filter_smoke(
        tmp_path,
        monkeypatch,
        task_count=2,
        candidate_budget=8,
        candidate_indices=(range(8), range(8)),
    )
    original_write = generation.write_filter_indexes

    def write_then_mutate(directory, raw_shards, decisions, **kwargs):
        shards = tuple(raw_shards)
        result = original_write(directory, shards, decisions, **kwargs)
        path = shards[0].arrays_path
        stat = path.stat()
        if mutation == "sha256":
            payload = bytearray(path.read_bytes())
            payload[len(payload) // 2] ^= 1
            path.write_bytes(payload)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        elif mutation == "size":
            path.write_bytes(path.read_bytes() + b"x")
        else:
            os.utime(
                path,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000),
            )
        return result

    monkeypatch.setattr(generation, "write_filter_indexes", write_then_mutate)

    with pytest.raises(RuntimeError, match="modified committed raw files"):
        pipeline_module.run_filter_smoke(**arguments)


def test_filter_smoke_rerun_is_deterministic_and_semantic_change_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit, filter_state, arguments = _prepare_filter_smoke(
        tmp_path,
        monkeypatch,
        task_count=2,
        candidate_budget=8,
        candidate_indices=(range(8), range(8)),
    )
    raw_paths = (commit.arrays_path, commit.metadata_path, commit.commit_path)
    raw_identity = {
        path: (pipeline_module._file_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in raw_paths
    }

    first = pipeline_module.run_filter_smoke(**arguments)
    first_paths = {name: Path(path) for name, path in first["outputs"].items()}
    first_payloads = {name: path.read_bytes() for name, path in first_paths.items()}
    assert first["raw_immutable_verified"] is True
    assert len(first["raw_snapshot"]) == 3
    assert all(
        set(identity) == {"sha256", "size_bytes", "mtime_ns"}
        for identity in first["raw_snapshot"].values()
    )

    repeated = pipeline_module.run_filter_smoke(**arguments)
    repeated_paths = {name: Path(path) for name, path in repeated["outputs"].items()}
    assert repeated["filter_semantic_sha256"] == first["filter_semantic_sha256"]
    assert repeated_paths == first_paths
    assert {
        name: path.read_bytes() for name, path in repeated_paths.items()
    } == first_payloads

    filter_state["sha256"] = "2" * 64
    changed = pipeline_module.run_filter_smoke(**arguments)
    changed_paths = {name: Path(path) for name, path in changed["outputs"].items()}
    assert changed["filter_semantic_sha256"] != first["filter_semantic_sha256"]
    assert changed_paths["commit"].parent != first_paths["commit"].parent
    assert changed_paths["commit"].parent.name == filter_state["sha256"]
    assert all(path.is_file() for path in (*first_paths.values(), *changed_paths.values()))
    assert {
        path: (pipeline_module._file_sha256(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in raw_paths
    } == raw_identity

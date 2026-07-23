from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import skilldrive.performance.benchmark as benchmark_module
from skilldrive.generation.contracts import GenerationTask, canonical_sha256
from skilldrive.performance.benchmark import aggregate_repeat_results
from skilldrive.performance.config import load_performance_config
from skilldrive.performance.workload import (
    WORKLOAD_KIND,
    WORKLOAD_SCHEMA_VERSION,
    file_sha256,
    generation_task_to_row,
)


def _task() -> GenerationTask:
    return GenerationTask.create(
        task_index=2,
        seed_record_id="2" * 64,
        scenario_id="scenario-2",
        skill_id="search-b",
        target_track_id="track-2",
        proposal_mode="rule_guided_prior_search",
        condition_skill_id="<none>",
        candidate_budget=16,
        checkpoint_sha256="a" * 64,
        semantic_config_sha256="b" * 64,
    )


def _repeat(index: int, elapsed: float, digest: str = "d" * 64) -> dict:
    return {
        "repeat_index": index,
        "task_count": 1,
        "candidate_count": 16,
        "accepted_count": 4,
        "rejected_count": 12,
        "quality_passed_before_diversity": 5,
        "elapsed_seconds": elapsed,
        "candidates_per_second": 16 / elapsed,
        "accepted_per_second": 4 / elapsed,
        "decision_sha256": digest,
        "stage_execution_counts": {
            "schema": 16,
            "kinematics": 16,
            "diversity": 5,
        },
        "stage_rejection_counts": {
            "schema": 0,
            "kinematics": 11,
            "diversity": 1,
        },
        "stage_elapsed_seconds": {
            "schema": elapsed * 0.1,
            "kinematics": elapsed * 0.6,
            "diversity": elapsed * 0.1,
        },
    }


def test_aggregate_reports_p50_p95_range_and_formal_projection() -> None:
    value = aggregate_repeat_results(
        [_repeat(0, 2.0), _repeat(1, 4.0), _repeat(2, 6.0)],
        formal_candidate_count=160,
    )

    assert value["elapsed_seconds"]["p50"] == 4.0
    assert value["elapsed_seconds"]["p95"] == pytest.approx(5.8)
    assert value["elapsed_seconds"]["range"] == 4.0
    assert value["tasks_per_second"]["p50"] == 0.25
    assert value["stage_elapsed_seconds"]["kinematics"]["p50"] == 2.4
    assert value["stage_rejection_rates"]["kinematics"] == 11 / 16
    assert value["stage_rejection_rates"]["diversity"] == 1 / 5
    assert value["stage_rejection_rates"]["schema"] == 0.0
    assert value["formal_projection"]["seconds"]["p50"] == 40.0
    assert value["formal_projection"]["seconds"]["p95"] == pytest.approx(58.0)


def test_aggregate_rejects_decision_drift() -> None:
    with pytest.raises(ValueError, match="changed workload counts or filter decisions"):
        aggregate_repeat_results(
            [_repeat(0, 2.0), _repeat(1, 2.1), _repeat(2, 2.2, "e" * 64)],
            formal_candidate_count=542_624,
        )


def test_aggregate_rejects_invalid_accept_reject_partition() -> None:
    values = [_repeat(0, 2.0), _repeat(1, 2.1), _repeat(2, 2.2)]
    for value in values:
        value["rejected_count"] = 11

    with pytest.raises(ValueError, match=r"accepted_count \+ rejected_count"):
        aggregate_repeat_results(values, formal_candidate_count=542_624)


def test_cpu_filter_runner_writes_three_repeat_summary_without_real_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("stable", encoding="utf-8")
    task = _task()
    workload = {
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
        "filter_semantic_sha256": "f" * 64,
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
    workload["workload_id"] = canonical_sha256(workload)
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(json.dumps(workload, sort_keys=True), encoding="utf-8")
    config_path = tmp_path / "performance.yaml"
    config_path.write_text("test", encoding="utf-8")
    config = replace(load_performance_config(), output_root=Path("out"))

    monkeypatch.setattr(
        benchmark_module,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(semantic_sha256="f" * 64),
    )
    monkeypatch.setattr(
        benchmark_module,
        "run_legacy_cpu_filter_once",
        lambda workload, *, config, repository_root, repeat_index: _repeat(
            repeat_index,
            2.0 + repeat_index,
        ),
    )

    path, summary = benchmark_module.run_cpu_filter_legacy_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=tmp_path,
    )

    assert path.is_file()
    assert summary["aggregate"]["repeat_count"] == 3
    assert len(list(path.parent.glob("repeat-*.json"))) == 3
    assert summary["validation_manifests_opened"] is False
    assert summary["final_validation_accessed"] is False

    def incomplete_repeat(
        workload,
        *,
        config,
        repository_root,
        repeat_index,
    ):
        value = _repeat(repeat_index, 2.0 + repeat_index)
        value["candidate_count"] = 15
        value["rejected_count"] = 11
        return value

    monkeypatch.setattr(
        benchmark_module,
        "run_legacy_cpu_filter_once",
        incomplete_repeat,
    )
    with pytest.raises(ValueError, match="fixed workload counts"):
        benchmark_module.run_cpu_filter_legacy_benchmark(
            config,
            config_path=config_path,
            workload_path=workload_path,
            repository_root=tmp_path,
        )

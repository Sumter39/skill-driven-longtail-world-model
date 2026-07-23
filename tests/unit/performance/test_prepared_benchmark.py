from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import skilldrive.data.av2_reader as av2_reader_module
import skilldrive.performance.prepared_benchmark as prepared_module
from skilldrive.generation.contracts import canonical_sha256
from skilldrive.performance.config import load_performance_config


def _repeat(index: int, digest: str, map_batch_size: int = 16) -> dict:
    elapsed = 2.0 + index
    return {
        "repeat_index": index,
        "map_batch_size": map_batch_size,
        "task_count": 1,
        "candidate_count": 2,
        "accepted_count": 1,
        "rejected_count": 1,
        "quality_passed_before_diversity": 1,
        "elapsed_seconds": elapsed,
        "candidates_per_second": 2 / elapsed,
        "accepted_per_second": 1 / elapsed,
        "decision_sha256": digest,
        "stage_execution_counts": {
            "schema_finite": 2,
            "map": 1,
            "diversity": 1,
        },
        "stage_rejection_counts": {
            "schema_finite": 1,
            "map": 0,
            "diversity": 0,
        },
        "stage_elapsed_seconds": {
            "schema_finite": elapsed * 0.2,
            "map": elapsed * 0.15,
            "diversity": elapsed * 0.1,
        },
        "prepared_map_seconds": elapsed * 0.05,
        "map_integrity_finalize_seconds": elapsed * 0.01,
        "map_subsystem_seconds": elapsed * 0.21,
    }


def test_prepared_runner_freezes_first_digest_and_writes_three_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "performance.yaml"
    config_path.write_text("test", encoding="utf-8")
    workload = {
        "workload_id": "a" * 64,
        "filter_semantic_sha256": "f" * 64,
        "counts": {"tasks": 1, "candidates": 2},
    }
    config = replace(load_performance_config(), output_root=Path("out"))
    digest = "d" * 64
    expected_values = []

    monkeypatch.setattr(
        prepared_module,
        "load_fixed_workload",
        lambda *args, **kwargs: workload,
    )
    monkeypatch.setattr(
        prepared_module,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(semantic_sha256="f" * 64),
    )
    monkeypatch.setattr(
        prepared_module,
        "_runner_source_sha256",
        lambda root: {"runner.py": "b" * 64},
    )

    def fake_once(
        workload,
        *,
        config,
        repository_root,
        repeat_index,
        expected_decision_sha256,
        map_batch_size,
    ):
        expected_values.append((expected_decision_sha256, map_batch_size))
        return _repeat(repeat_index, digest, map_batch_size)

    monkeypatch.setattr(
        prepared_module,
        "run_prepared_map_cpu_filter_once",
        fake_once,
    )

    path, summary = prepared_module.run_cpu_filter_prepared_map_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=tmp_path,
        map_batch_size=32,
    )

    assert expected_values == [(None, 32), (digest, 32), (digest, 32)]
    assert path.is_file()
    assert summary["aggregate"]["decision_sha256"] == digest
    assert summary["benchmark_contract"]["decision_reference_source"] == "first_repeat"
    assert summary["benchmark_contract"]["map_batch_size"] == 32
    repeat_paths = sorted(path.parent.glob("repeat-*.json"))
    assert len(repeat_paths) == 3
    repeat = json.loads(repeat_paths[0].read_text(encoding="utf-8"))
    assert repeat["decision_reference_sha256"] == digest


def test_prepared_runner_rejects_explicit_decision_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload_path = tmp_path / "workload.json"
    workload_path.write_text("{}", encoding="utf-8")
    config_path = tmp_path / "performance.yaml"
    config_path.write_text("test", encoding="utf-8")
    workload = {
        "workload_id": "a" * 64,
        "filter_semantic_sha256": "f" * 64,
        "counts": {"tasks": 1, "candidates": 2},
    }
    config = replace(load_performance_config(), output_root=Path("out"))
    monkeypatch.setattr(
        prepared_module,
        "load_fixed_workload",
        lambda *args, **kwargs: workload,
    )
    monkeypatch.setattr(
        prepared_module,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(semantic_sha256="f" * 64),
    )
    monkeypatch.setattr(
        prepared_module,
        "_runner_source_sha256",
        lambda root: {"runner.py": "b" * 64},
    )
    monkeypatch.setattr(
        prepared_module,
        "run_prepared_map_cpu_filter_once",
        lambda *args, repeat_index, **kwargs: _repeat(repeat_index, "e" * 64),
    )

    with pytest.raises(ValueError, match="decision_sha256 changed"):
        prepared_module.run_cpu_filter_prepared_map_benchmark(
            config,
            config_path=config_path,
            workload_path=workload_path,
            repository_root=tmp_path,
            expected_decision_sha256="d" * 64,
        )


def test_prepared_once_reuses_one_map_per_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_rows = []
    records = []
    for index, (scenario_id, source_path) in enumerate(
        (
            ("scene-a", "a/scenario.parquet"),
            ("scene-b", "b/scenario.parquet"),
            ("scene-a", "a/scenario.parquet"),
        )
    ):
        record_id = f"record-{index}"
        task = SimpleNamespace(
            task_index=index,
            seed_record_id=record_id,
            scenario_id=scenario_id,
            skill_id="skill-a",
            semantic_config_sha256="c" * 64,
        )
        task_rows.append(
            {
                "task": {"skill_id": "skill-a", "task_object": task},
                "source_path": source_path,
                "raw_commit": f"raw/{index}.json",
            }
        )
        records.append(
            SimpleNamespace(record_id=record_id, source_path=source_path)
        )
    workload = {
        "counts": {"tasks": 3, "candidates": 3},
        "tasks": task_rows,
        "filter_semantic_sha256": "f" * 64,
    }
    generation = SimpleNamespace(
        inputs=SimpleNamespace(
            seed_manifest=Path("seeds.csv"),
            data_root=Path("data"),
        ),
        formal_catalog=Path("configs/skills/catalog.yaml"),
        none_skill_id="<none>",
        skills_by_id={
            "skill-a": SimpleNamespace(primary_generated_role="initiator")
        },
    )
    prepared_calls = []
    sessions = []
    captured_inputs = []
    captured_batch_sizes = []

    monkeypatch.setattr(
        prepared_module,
        "load_counterfactual_config",
        lambda *args, **kwargs: generation,
    )
    monkeypatch.setattr(prepared_module, "load_filter_config", lambda *args: object())
    monkeypatch.setattr(
        prepared_module,
        "load_detection_config",
        lambda *args: object(),
    )
    monkeypatch.setattr(prepared_module, "read_seed_records", lambda *args: records)
    monkeypatch.setattr(
        prepared_module,
        "seed_record_id",
        lambda record: record.record_id,
    )
    monkeypatch.setattr(prepared_module, "load_skill", lambda *args: object())
    monkeypatch.setattr(
        prepared_module,
        "generation_task_from_row",
        lambda row: row["task_object"],
    )
    monkeypatch.setattr(
        prepared_module,
        "load_raw_shard_candidates",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        prepared_module,
        "bind_raw_candidates",
        lambda raw, tasks, records: [f"bound-{tasks[0].task_index}"],
    )
    monkeypatch.setattr(
        prepared_module,
        "pilot_evaluation_arm",
        lambda *args, **kwargs: "formal",
    )

    def load_scenario(path):
        return SimpleNamespace(load_token=object(), source_path=Path(path))

    monkeypatch.setattr(av2_reader_module, "load_av2_scenario", load_scenario)

    def prepare(source):
        value = SimpleNamespace(source=source, token=object())
        prepared_calls.append(value)
        return value

    monkeypatch.setattr(prepared_module, "prepare_map_geometry", prepare)

    class VerificationSession:
        def __init__(self, source, prepared):
            self.source = source
            self.prepared_map = prepared
            self.finalized = False
            sessions.append(self)

        def finalize(self):
            self.finalized = True

    monkeypatch.setattr(
        prepared_module,
        "PreparedMapVerificationSession",
        VerificationSession,
    )

    class Validation:
        def compact(self, *, cohort):
            return SimpleNamespace(quality_passed=True, cohort=cohort)

    def validate(candidates, *, filter_config, detection_config, map_batch_size):
        captured_inputs.extend(candidates)
        captured_batch_sizes.append(map_batch_size)
        return tuple(Validation() for _ in candidates)

    monkeypatch.setattr(
        prepared_module,
        "validate_candidate_individual_batch",
        validate,
    )

    def finalize(validations, *, filter_config, filter_semantic_sha256):
        decisions = tuple(
            SimpleNamespace(
                candidate_id=f"candidate-{index}",
                filter_evaluation_id=f"evaluation-{index}",
                accepted=True,
                rejection_reasons=(),
                metrics={"first_failed_stage": None},
            )
            for index in range(len(validations))
        )
        return SimpleNamespace(
            decisions=decisions,
            validations=tuple(validations),
            stage_execution_counts={"schema_finite": 3, "map": 3, "diversity": 3},
            stage_elapsed_seconds={
                "schema_finite": 0.1,
                "map": 0.2,
                "diversity": 0.1,
            },
        )

    monkeypatch.setattr(
        prepared_module,
        "finalize_candidate_validations",
        finalize,
    )
    config = replace(load_performance_config(), output_root=Path("out"))

    result = prepared_module.run_prepared_map_cpu_filter_once(
        workload,
        config=config,
        repository_root=tmp_path,
        repeat_index=0,
        map_batch_size=8,
    )

    assert len(prepared_calls) == 2
    assert len(sessions) == 3
    assert captured_batch_sizes == [8, 8, 8]
    assert all(session.finalized for session in sessions)
    assert result["scenario_load_count"] == 3
    assert result["prepared_map_count"] == 2
    assert result["map_batch_size"] == 8
    assert captured_inputs[0].prepared_map is captured_inputs[2].prepared_map
    assert captured_inputs[0].prepared_map is not captured_inputs[1].prepared_map
    assert captured_inputs[0].map_verification_session is sessions[0]
    assert captured_inputs[1].map_verification_session is sessions[1]
    assert captured_inputs[2].map_verification_session is sessions[2]
    assert result["prepared_map_seconds"] >= 0.0
    assert result["map_integrity_finalize_seconds"] >= 0.0
    assert result["map_subsystem_seconds"] == pytest.approx(
        result["prepared_map_seconds"]
        + result["stage_elapsed_seconds"]["map"]
        + result["map_integrity_finalize_seconds"]
    )
    assert set(result["timing_breakdown_seconds"]) == {
        "scenario_read",
        "raw_read",
        "candidate_bind",
        "individual_filter",
        "global_finalize",
        "task_bookkeeping",
    }
    assert all(value >= 0.0 for value in result["timing_breakdown_seconds"].values())
    assert sum(result["timing_breakdown_seconds"].values()) == pytest.approx(
        result["elapsed_seconds"]
    )
    assert result["timing_breakdown_seconds"]["individual_filter"] >= (
        result["prepared_map_seconds"] + result["map_integrity_finalize_seconds"]
    )
    assert result["decision_sha256"] == canonical_sha256(
        [
            {
                "candidate_id": f"candidate-{index}",
                "filter_evaluation_id": f"evaluation-{index}",
                "accepted": True,
                "rejection_reasons": [],
                "metrics": {"first_failed_stage": None},
            }
            for index in range(3)
        ]
    )

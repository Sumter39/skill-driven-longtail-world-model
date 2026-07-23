from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import skilldrive.data as data_module
import skilldrive.performance.parallel_filter as parallel_filter_module
import skilldrive.performance.runtime_benchmark as runtime_module
from skilldrive.performance.config import load_performance_config


def _fixed_inputs() -> runtime_module.FixedInputs:
    generation = SimpleNamespace(
        active_checkpoint=SimpleNamespace(sha256="c" * 64),
        formal_catalog=Path("configs/skills/formal_catalog.yaml"),
    )
    return runtime_module.FixedInputs(
        workload={
            "workload_id": "mock",
            "input_sha256": {"input": "i" * 64},
        },
        generation=generation,
        tasks=(),
        records=(),
        source_paths=(),
        source_values=(),
        latent_seeds=(),
        task_order_sha256="t" * 64,
        latent_seed_sha256="l" * 64,
    )


def _gpu_repeat(index: int, digest: str = "o" * 64) -> dict:
    seconds = 2.0 + index
    return {
        "repeat_index": index,
        "task_count": 512,
        "candidate_count": 8192,
        "gpu_seconds": seconds,
        "preparation_seconds": 0.2,
        "transfer_seconds": 0.1,
        "measured_wall_seconds": seconds + 0.3,
        "tasks_per_gpu_second": 512 / seconds,
        "candidates_per_gpu_second": 8192 / seconds,
        "peak_gpu_memory_allocated_bytes": 1024 + index,
        "peak_gpu_memory_reserved_bytes": 2048 + index,
        "output_sha256": digest,
    }


def _e2e_repeat(
    index: int,
    decision: str = "d" * 64,
    semantic_decision: str = "a" * 64,
) -> dict:
    seconds = 10.0 + index
    return {
        "repeat_index": index,
        "task_count": 512,
        "candidate_count": 8192,
        "accepted_count": 37,
        "rejected_count": 8155,
        "quality_passed_before_diversity": 55,
        "end_to_end_seconds": seconds,
        "candidates_per_second": 8192 / seconds,
        "accepted_per_second": 37 / seconds,
        "generation_gpu_seconds": 2.0 + index * 0.1,
        "filter_wall_seconds": 7.0 + index * 0.1,
        "stage_execution_counts": {"schema_finite": 8192},
        "stage_rejection_counts": {"schema_finite": 10},
        "output_sha256": "o" * 64,
        "decision_sha256": decision,
        "semantic_decision_sha256": semantic_decision,
        "bev_rendering_included": False,
    }


def test_gpu_aggregate_requires_three_stable_outputs() -> None:
    repeats = [_gpu_repeat(index) for index in range(3)]

    aggregate = runtime_module._aggregate_gpu(repeats, 542_624)

    assert aggregate["output_sha256"] == "o" * 64
    assert aggregate["gpu_seconds"]["p50"] == 3.0
    assert aggregate["formal_projection_hours"]["p50"] > 0.0

    repeats[2] = _gpu_repeat(2, "x" * 64)
    with pytest.raises(ValueError, match="changed output"):
        runtime_module._aggregate_gpu(repeats, 542_624)


def test_e2e_aggregate_enforces_semantic_reference_and_stable_decisions() -> None:
    repeats = [_e2e_repeat(index) for index in range(3)]

    aggregate = runtime_module._aggregate_e2e(repeats, 542_624, "a" * 64)

    assert aggregate["accepted_count"] == 37
    assert aggregate["decision_sha256"] == "d" * 64
    assert aggregate["semantic_decision_sha256"] == "a" * 64

    with pytest.raises(ValueError, match="semantic decision SHA"):
        runtime_module._aggregate_e2e(repeats, 542_624, "e" * 64)


def test_filter_worker_count_is_forwarded_without_duplicate_filter_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_filter(workload, **kwargs):
        captured.update(kwargs)
        return "filtered"

    monkeypatch.setattr(
        parallel_filter_module,
        "run_parallel_filter_workload",
        fake_filter,
    )

    result = runtime_module._filter_generated(
        {"tasks": []},
        root=Path("."),
        generation="generation",
        filter_config="filter",
        detection_config="detection",
        workers=8,
        map_batch_size=32,
    )

    assert result == "filtered"
    assert captured["worker_count"] == 8
    assert captured["map_batch_size"] == 32


def test_gpu_runner_separates_initialization_warmup_and_three_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_performance_config(), output_root=Path("out"))
    config_path = tmp_path / "performance.yaml"
    workload_path = tmp_path / "workload.json"
    config_path.write_text("config", encoding="utf-8")
    workload_path.write_text("workload", encoding="utf-8")
    inputs = _fixed_inputs()

    monkeypatch.setattr(runtime_module, "_load_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(data_module, "build_cvae_schema", lambda path: "schema")
    monkeypatch.setattr(
        runtime_module,
        "_load_runtime",
        lambda *args, **kwargs: ("runtime", 1.25),
    )
    monkeypatch.setattr(
        runtime_module,
        "_prepare_contexts",
        lambda *args, **kwargs: ((None,) * 512, {"elapsed_seconds": 2.5}),
    )
    monkeypatch.setattr(runtime_module, "_environment", lambda runtime: {"gpu": "mock"})
    monkeypatch.setattr(
        runtime_module,
        "_sources",
        lambda *args, **kwargs: {"runner.py": "s" * 64},
    )
    monkeypatch.setattr(
        runtime_module,
        "_warmup",
        lambda *args, **kwargs: {
            "iterations": 2,
            "wall_seconds": 0.5,
            "included_in_repeats": False,
        },
    )
    monkeypatch.setattr(
        runtime_module,
        "_gpu_pass",
        lambda *args, repeat_index, **kwargs: _gpu_repeat(repeat_index),
    )

    path, summary = runtime_module.run_gpu_generation_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=tmp_path,
    )

    assert path.is_file()
    assert summary["initialization"]["model_load_seconds"] == 1.25
    assert summary["warmup"]["included_in_repeats"] is False
    assert len(list(path.parent.glob("repeat-*.json"))) == 3
    contract = summary["benchmark_contract"]
    assert "cuda_events" in contract["measurement_scope"]
    assert contract["task_count"] == 512


def test_e2e_runner_excludes_bev_and_records_filter_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(load_performance_config(), output_root=Path("out"))
    config_path = tmp_path / "performance.yaml"
    workload_path = tmp_path / "workload.json"
    config_path.write_text("config", encoding="utf-8")
    workload_path.write_text("workload", encoding="utf-8")
    inputs = _fixed_inputs()

    monkeypatch.setattr(runtime_module, "_load_inputs", lambda *args, **kwargs: inputs)
    monkeypatch.setattr(runtime_module, "load_filter_config", lambda path: "filter")
    monkeypatch.setattr(runtime_module, "load_detection_config", lambda path: "detection")
    monkeypatch.setattr(data_module, "build_cvae_schema", lambda path: "schema")
    monkeypatch.setattr(
        runtime_module,
        "_load_runtime",
        lambda *args, **kwargs: ("runtime", 1.0),
    )
    monkeypatch.setattr(
        runtime_module,
        "_prepare_contexts",
        lambda *args, **kwargs: ((None,) * 512, {"elapsed_seconds": 2.0}),
    )
    monkeypatch.setattr(runtime_module, "_environment", lambda runtime: {"gpu": "mock"})
    monkeypatch.setattr(
        runtime_module,
        "_sources",
        lambda *args, **kwargs: {"runner.py": "s" * 64},
    )
    monkeypatch.setattr(
        runtime_module,
        "_warmup",
        lambda *args, **kwargs: {"included_in_repeats": False},
    )
    monkeypatch.setattr(
        runtime_module,
        "_e2e_repeat",
        lambda *args, repeat_index, **kwargs: _e2e_repeat(repeat_index),
    )

    path, summary = runtime_module.run_end_to_end_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=tmp_path,
        filter_workers=4,
        map_batch_size=32,
        expected_semantic_decision_sha256="a" * 64,
    )

    assert path.is_file()
    assert summary["bev_rendering_included"] is False
    assert summary["correctness"]["expected_semantic_decision_matched"] is True
    contract = summary["benchmark_contract"]
    assert contract["filter_workers"] == 4
    assert contract["map_batch_size"] == 32
    assert contract["bev_rendering_included"] is False
    repeat = json.loads((path.parent / "repeat-01.json").read_text(encoding="utf-8"))
    assert repeat["bev_rendering_included"] is False

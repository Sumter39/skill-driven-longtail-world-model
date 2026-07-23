from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.generation.run_parallel_filter_benchmark as runner


def _result(index: int, digest: str, *, workers: int = 4):
    decisions = (
        SimpleNamespace(
            candidate_id="a" * 64,
            filter_evaluation_id="b" * 64,
            accepted=True,
            rejection_reasons=(),
            metrics={"first_failed_stage": None},
        ),
        SimpleNamespace(
            candidate_id="c" * 64,
            filter_evaluation_id="d" * 64,
            accepted=False,
            rejection_reasons=("schema.invalid",),
            metrics={"first_failed_stage": "schema_finite"},
        ),
    )
    validations = (
        SimpleNamespace(quality_passed=True),
        SimpleNamespace(quality_passed=False),
    )
    batch = SimpleNamespace(
        decisions=decisions,
        validations=validations,
        stage_elapsed_seconds={"schema_finite": 0.2, "diversity": 0.1},
    )
    return SimpleNamespace(
        batch=batch,
        timings={
            "worker_startup_seconds": 0.5 + index,
            "worker_execution_seconds": 2.0 + index,
            "stable_total_seconds": 2.5 + index,
            "global_finalize_seconds": 0.25,
            "total_seconds": 3.0 + index,
        },
        decision_sha256=digest,
        semantic_decision_sha256="s" * 64,
        stage_execution_counts={"schema_finite": 2, "diversity": 1},
        stage_rejection_counts={"schema_finite": 1, "diversity": 0},
        requested_worker_count=workers,
        effective_worker_count=workers,
        worker_pids=tuple(range(101, 101 + workers)),
        scenario_load_count=2,
        prepared_map_count=2,
    )


def _patch_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_parallel,
):
    config = SimpleNamespace(
        inputs=SimpleNamespace(
            generation_config=Path("generation.yaml"),
            filter_config=Path("filter.yaml"),
            detection_config=Path("detection.yaml"),
        ),
        benchmark=SimpleNamespace(formal_candidate_count=20),
        output_root=Path("out"),
    )
    workload = {
        "workload_id": "e" * 64,
        "filter_semantic_sha256": "f" * 64,
        "counts": {"tasks": 2, "candidates": 2},
    }
    monkeypatch.setattr(runner, "load_performance_config", lambda path: config)
    monkeypatch.setattr(
        runner,
        "load_fixed_workload",
        lambda path, repository_root: workload,
    )
    monkeypatch.setattr(
        runner,
        "load_counterfactual_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runner, "load_filter_config", lambda path: object())
    monkeypatch.setattr(runner, "load_detection_config", lambda path: object())
    monkeypatch.setattr(runner, "run_parallel_filter_workload", fake_parallel)
    monkeypatch.setattr(runner, "file_sha256", lambda path: "b" * 64)
    config_path = tmp_path / "performance.yaml"
    workload_path = tmp_path / "workload.json"
    config_path.write_text("config", encoding="utf-8")
    workload_path.write_text("workload", encoding="utf-8")
    return config_path, workload_path


def test_parallel_runner_freezes_three_repeats_and_separates_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "f" * 64
    calls = []

    def fake_parallel(*args, worker_count, map_batch_size, **kwargs):
        calls.append((worker_count, map_batch_size))
        return _result(len(calls) - 1, digest)

    config_path, workload_path = _patch_inputs(
        monkeypatch,
        tmp_path,
        fake_parallel,
    )
    path, summary = runner.run_benchmark(
        repository_root=tmp_path,
        config_path=config_path,
        workload_path=workload_path,
        workers=4,
        expected_decision_sha256=digest,
        map_batch_size=32,
    )

    assert calls == [(4, 32), (4, 32), (4, 32)]
    assert path.is_file()
    assert len(list(path.parent.glob("repeat-*.json"))) == 3
    assert summary["aggregate"]["decision_sha256"] == digest
    assert summary["semantic_decision_sha256"] == "s" * 64
    assert summary["worker_startup_seconds"] == [0.5, 1.5, 2.5]
    assert summary["stable_worker_seconds"] == [2.0, 3.0, 4.0]
    assert summary["stable_total_seconds"] == [2.5, 3.5, 4.5]
    assert summary["wall_total_seconds"] == [3.0, 4.0, 5.0]
    assert summary["effective_worker_count"] == [4, 4, 4]
    assert summary["benchmark_contract"]["map_batch_size"] == 32


def test_parallel_runner_mismatch_writes_no_benchmark_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, workload_path = _patch_inputs(
        monkeypatch,
        tmp_path,
        lambda *args, **kwargs: _result(0, "a" * 64, workers=2),
    )

    with pytest.raises(ValueError, match="differs from expected"):
        runner.run_benchmark(
            repository_root=tmp_path,
            config_path=config_path,
            workload_path=workload_path,
            workers=2,
            expected_decision_sha256="f" * 64,
        )

    assert not (tmp_path / "out").exists()


def test_parallel_runner_rejects_missing_requested_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, workload_path = _patch_inputs(
        monkeypatch,
        tmp_path,
        lambda *args, **kwargs: _result(0, "f" * 64, workers=3),
    )

    with pytest.raises(ValueError, match="did not use every requested worker"):
        runner.run_benchmark(
            repository_root=tmp_path,
            config_path=config_path,
            workload_path=workload_path,
            workers=4,
            expected_decision_sha256="f" * 64,
        )

    assert not (tmp_path / "out").exists()


def test_parallel_runner_rejects_unsupported_worker_count(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="one of 1, 2, 4, 8, 12, 16, or 20"):
        runner.run_benchmark(
            repository_root=tmp_path,
            config_path=tmp_path / "missing.yaml",
            workload_path=tmp_path / "missing.json",
            workers=3,
            expected_decision_sha256="f" * 64,
        )


def test_parallel_runner_rejects_unsupported_map_batch_size(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="map_batch_size"):
        runner.run_benchmark(
            repository_root=tmp_path,
            config_path=tmp_path / "missing.yaml",
            workload_path=tmp_path / "missing.json",
            workers=4,
            expected_decision_sha256="f" * 64,
            map_batch_size=12,
        )

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import skilldrive.generation.formal_runner as formal_runner


def test_filter_progress_uses_stage_candidate_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(formal_runner.time, "perf_counter", lambda: 10.0)

    formal_runner._write_progress(
        tmp_path,
        stage="filter:example",
        completed=8,
        total=10,
        candidates=40,
        candidate_total=100,
        started=0.0,
    )

    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["total_candidates"] == 100
    assert progress["candidates_per_second"] == 4.0
    assert progress["eta_seconds"] == 15.0


def test_generation_skips_raw_scan_when_every_task_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = (
        SimpleNamespace(task_id="task-1"),
        SimpleNamespace(task_id="task-2"),
    )
    plan = SimpleNamespace(tasks=tasks)
    states = (
        SimpleNamespace(task_id="task-1", status="accepted"),
        SimpleNamespace(task_id="task-2", status="rejected"),
    )

    def fail_scan(*args, **kwargs):
        raise AssertionError("completed recovery must not rescan raw shards")

    monkeypatch.setattr(formal_runner, "scan_raw_shards", fail_scan)

    formal_runner._generate(
        root=tmp_path,
        run_root=tmp_path,
        plan=plan,
        config=SimpleNamespace(),
        records_by_id={},
        execution={},
        state_tasks=states,
        bindings=SimpleNamespace(),
    )

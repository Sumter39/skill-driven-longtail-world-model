from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from skilldrive.training.system_monitor import CpuTimes, SystemMonitor, read_proc_stat


class _ControlledStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)
        self.release = threading.Event()
        self.exhausted = threading.Event()
        self.closed = threading.Event()

    def __iter__(self) -> "_ControlledStream":
        return self

    def __next__(self) -> str:
        self.release.wait(timeout=1.0)
        try:
            return next(self._lines)
        except StopIteration:
            self.exhausted.set()
            self.closed.wait(timeout=1.0)
            raise

    def close(self) -> None:
        self.closed.set()


class _FakeProcess:
    def __init__(self, stream: _ControlledStream) -> None:
        self.stdout = stream
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.stdout.close()

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()

    def wait(self, timeout: float | None = None) -> int:
        assert timeout is None or timeout > 0.0
        return self.returncode or 0


def test_read_proc_stat_parses_aggregate_cpu_line(tmp_path: Path) -> None:
    proc_stat = tmp_path / "stat"
    proc_stat.write_text(
        "cpu  100 20 30 400 50 6 7 8 0 0\ncpu0 1 2 3 4 5 6 7 8\n",
        encoding="utf-8",
    )

    assert read_proc_stat(proc_stat) == CpuTimes(
        total=621,
        busy=171,
        iowait=50,
    )


def test_cpu_metrics_use_proc_stat_deltas_and_gpu_unavailable_is_explicit() -> None:
    snapshots = iter(
        [
            CpuTimes(total=1000, busy=600, iowait=50),
            CpuTimes(total=1100, busy=660, iowait=60),
        ]
    )
    monitor = SystemMonitor(gpu_index=None, cpu_stat_reader=lambda: next(snapshots))

    monitor.start()
    monitor.begin_window()
    metrics = monitor.end_window()
    monitor.stop()

    assert metrics.cpu_metrics_available
    assert metrics.cpu_busy_percent == pytest.approx(60.0)
    assert metrics.cpu_iowait_percent == pytest.approx(10.0)
    assert not metrics.gpu_utilization_available
    assert metrics.gpu_utilization_mean_percent is None
    assert metrics.gpu_utilization_p50_percent is None
    assert metrics.gpu_utilization_p95_percent is None
    assert metrics.gpu_utilization_sample_count == 0


def test_gpu_monitor_uses_one_long_lived_process_and_summarizes_window() -> None:
    stream = _ControlledStream(["10\n", "invalid\n", "50\n", "101\n", "90\n"])
    process = _FakeProcess(stream)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen_factory(command: list[str], **kwargs: Any) -> _FakeProcess:
        calls.append((command, kwargs))
        return process

    monitor = SystemMonitor(
        gpu_index=2,
        cpu_stat_reader=lambda: None,
        popen_factory=popen_factory,
    )
    monitor.start()
    monitor.begin_window()
    stream.release.set()
    assert stream.exhausted.wait(timeout=1.0)

    metrics = monitor.end_window()
    monitor.stop()

    assert len(calls) == 1
    assert calls[0][0] == [
        "nvidia-smi",
        "--id=2",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
        "--loop-ms",
        "100",
    ]
    assert metrics.gpu_utilization_available
    assert metrics.gpu_utilization_mean_percent == pytest.approx(50.0)
    assert metrics.gpu_utilization_p50_percent == pytest.approx(50.0)
    assert metrics.gpu_utilization_p95_percent == pytest.approx(86.0)
    assert metrics.gpu_utilization_sample_count == 3


def test_missing_nvidia_smi_reports_unavailable_without_fabricated_zero() -> None:
    def unavailable(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        raise FileNotFoundError("nvidia-smi")

    monitor = SystemMonitor(
        gpu_index=0,
        cpu_stat_reader=lambda: None,
        popen_factory=unavailable,
    )

    monitor.start()
    monitor.begin_window()
    metrics = monitor.end_window()
    monitor.stop()

    assert not metrics.cpu_metrics_available
    assert metrics.cpu_busy_percent is None
    assert metrics.cpu_iowait_percent is None
    assert not metrics.gpu_utilization_available
    assert metrics.gpu_utilization_mean_percent is None
    assert metrics.gpu_utilization_sample_count == 0

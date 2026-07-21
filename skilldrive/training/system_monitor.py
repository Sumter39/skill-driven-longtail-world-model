"""Low-overhead system monitoring for stable training benchmark windows."""

from __future__ import annotations

import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, IO, Sequence


@dataclass(frozen=True)
class CpuTimes:
    total: int
    busy: int
    iowait: int


@dataclass(frozen=True)
class SystemMetrics:
    cpu_metrics_available: bool
    cpu_busy_percent: float | None
    cpu_iowait_percent: float | None
    gpu_utilization_available: bool
    gpu_utilization_mean_percent: float | None
    gpu_utilization_p50_percent: float | None
    gpu_utilization_p95_percent: float | None
    gpu_utilization_sample_count: int


def read_proc_stat(path: str | Path = "/proc/stat") -> CpuTimes | None:
    """Read aggregate CPU counters without inventing values off Linux."""

    try:
        first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError, UnicodeError):
        return None
    fields = first_line.split()
    if not fields or fields[0] != "cpu" or len(fields) < 6:
        return None
    try:
        values = [int(value) for value in fields[1:9]]
    except ValueError:
        return None
    values.extend([0] * (8 - len(values)))
    user, nice, system, idle, iowait, irq, softirq, steal = values[:8]
    total = user + nice + system + idle + iowait + irq + softirq + steal
    busy = user + nice + system + irq + softirq + steal
    return CpuTimes(total=total, busy=busy, iowait=iowait)


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cpu_metrics(
    start: CpuTimes | None,
    end: CpuTimes | None,
) -> tuple[bool, float | None, float | None]:
    if start is None or end is None:
        return False, None, None
    total_delta = end.total - start.total
    busy_delta = end.busy - start.busy
    iowait_delta = end.iowait - start.iowait
    if total_delta <= 0 or busy_delta < 0 or iowait_delta < 0:
        return False, None, None
    return (
        True,
        100.0 * busy_delta / total_delta,
        100.0 * iowait_delta / total_delta,
    )


def _gpu_metrics(
    samples: Sequence[float],
) -> tuple[bool, float | None, float | None, float | None, int]:
    if not samples:
        return False, None, None, None, 0
    return (
        True,
        sum(samples) / len(samples),
        _percentile(samples, 0.50),
        _percentile(samples, 0.95),
        len(samples),
    )


class SystemMonitor:
    """Sample one GPU process and `/proc/stat` over an explicit window."""

    def __init__(
        self,
        *,
        gpu_index: int | None,
        sample_interval_ms: int = 100,
        cpu_stat_reader: Callable[[], CpuTimes | None] = read_proc_stat,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        if sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be positive")
        self._gpu_index = gpu_index
        self._sample_interval_ms = sample_interval_ms
        self._cpu_stat_reader = cpu_stat_reader
        self._popen_factory = popen_factory
        self._process: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._samples: list[float] = []
        self._samples_lock = threading.Lock()
        self._window_sample_start = 0
        self._cpu_start: CpuTimes | None = None

    def start(self) -> None:
        if self._gpu_index is None:
            return
        command = [
            "nvidia-smi",
            f"--id={self._gpu_index}",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
            "--loop-ms",
            str(self._sample_interval_ms),
        ]
        try:
            process = self._popen_factory(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError):
            return
        if process.stdout is None:
            process.terminate()
            return
        self._process = process
        self._reader_thread = threading.Thread(
            target=self._read_gpu_samples,
            args=(process.stdout,),
            name="skilldrive-nvidia-smi",
            daemon=True,
        )
        self._reader_thread.start()

    def _read_gpu_samples(self, stream: IO[str]) -> None:
        for raw_line in stream:
            try:
                value = float(raw_line.strip())
            except ValueError:
                continue
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                continue
            with self._samples_lock:
                self._samples.append(value)

    def begin_window(self) -> None:
        self._cpu_start = self._cpu_stat_reader()
        with self._samples_lock:
            self._window_sample_start = len(self._samples)

    def end_window(self) -> SystemMetrics:
        cpu_end = self._cpu_stat_reader()
        with self._samples_lock:
            gpu_samples = tuple(self._samples[self._window_sample_start :])
        cpu_available, cpu_busy, cpu_iowait = _cpu_metrics(self._cpu_start, cpu_end)
        gpu_available, gpu_mean, gpu_p50, gpu_p95, gpu_count = _gpu_metrics(
            gpu_samples
        )
        return SystemMetrics(
            cpu_metrics_available=cpu_available,
            cpu_busy_percent=cpu_busy,
            cpu_iowait_percent=cpu_iowait,
            gpu_utilization_available=gpu_available,
            gpu_utilization_mean_percent=gpu_mean,
            gpu_utilization_p50_percent=gpu_p50,
            gpu_utilization_p95_percent=gpu_p95,
            gpu_utilization_sample_count=gpu_count,
        )

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        self._process = None
        self._reader_thread = None


__all__ = ["CpuTimes", "SystemMetrics", "SystemMonitor", "read_proc_stat"]

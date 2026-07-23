"""Strict configuration for fixed counterfactual-generation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_PERFORMANCE_CONFIG = Path("configs/generation/performance_v1.yaml")
@dataclass(frozen=True)
class PerformanceInputsConfig:
    generation_config: Path
    filter_config: Path
    detection_config: Path
    pilot_summary: Path


@dataclass(frozen=True)
class FixedWorkloadConfig:
    max_tasks: int
    selection_contract: str


@dataclass(frozen=True)
class CpuFilterBenchmarkConfig:
    runner: str
    repeats: int
    formal_candidate_count: int


@dataclass(frozen=True)
class PerformanceBenchmarkConfig:
    version: int
    contract_name: str
    inputs: PerformanceInputsConfig
    workload: FixedWorkloadConfig
    benchmark: CpuFilterBenchmarkConfig
    output_root: Path


def _mapping(value: Any, name: str, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    expected = set(keys)
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{name} has missing or unknown keys: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _relative_path(value: Any, name: str) -> Path:
    path = Path(_text(value, name))
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a repository-relative path without '..'")
    return path


def _integer(value: Any, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer <= {maximum}")
    return value


def load_performance_config(
    path: str | Path = DEFAULT_PERFORMANCE_CONFIG,
) -> PerformanceBenchmarkConfig:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"failed to read performance config: {source}: {error}") from error
    root = _mapping(
        value,
        "performance config",
        ("version", "contract_name", "inputs", "workload", "benchmark", "output_root"),
    )
    if root["version"] != 1:
        raise ValueError("performance config version must be 1")
    if root["contract_name"] != "counterfactual_performance_v1":
        raise ValueError(
            "performance config contract_name must be counterfactual_performance_v1"
        )

    inputs = _mapping(
        root["inputs"],
        "inputs",
        (
            "generation_config",
            "filter_config",
            "detection_config",
            "pilot_summary",
        ),
    )

    workload = _mapping(
        root["workload"],
        "workload",
        ("max_tasks", "selection_contract"),
    )
    selection_contract = _text(
        workload["selection_contract"],
        "workload.selection_contract",
    )
    if selection_contract != "deterministic_atomic_condition_pairs_v1":
        raise ValueError(
            "workload.selection_contract must be "
            "deterministic_atomic_condition_pairs_v1"
        )

    benchmark = _mapping(
        root["benchmark"],
        "benchmark",
        ("runner", "repeats", "formal_candidate_count"),
    )
    if benchmark["runner"] != "cpu_filter_legacy":
        raise ValueError("benchmark.runner must be cpu_filter_legacy")
    repeats = _integer(benchmark["repeats"], "benchmark.repeats", minimum=1)
    if repeats != 3:
        raise ValueError("benchmark.repeats must be exactly 3")
    formal_candidate_count = _integer(
        benchmark["formal_candidate_count"],
        "benchmark.formal_candidate_count",
        minimum=1,
    )
    if formal_candidate_count != 542_624:
        raise ValueError("benchmark.formal_candidate_count must be 542624")

    return PerformanceBenchmarkConfig(
        version=1,
        contract_name="counterfactual_performance_v1",
        inputs=PerformanceInputsConfig(
            generation_config=_relative_path(
                inputs["generation_config"], "inputs.generation_config"
            ),
            filter_config=_relative_path(
                inputs["filter_config"], "inputs.filter_config"
            ),
            detection_config=_relative_path(
                inputs["detection_config"], "inputs.detection_config"
            ),
            pilot_summary=_relative_path(
                inputs["pilot_summary"], "inputs.pilot_summary"
            ),
        ),
        workload=FixedWorkloadConfig(
            max_tasks=_integer(
                workload["max_tasks"],
                "workload.max_tasks",
                minimum=1,
                maximum=512,
            ),
            selection_contract=selection_contract,
        ),
        benchmark=CpuFilterBenchmarkConfig(
            runner="cpu_filter_legacy",
            repeats=repeats,
            formal_candidate_count=formal_candidate_count,
        ),
        output_root=_relative_path(root["output_root"], "output_root"),
    )


__all__ = [
    "DEFAULT_PERFORMANCE_CONFIG",
    "CpuFilterBenchmarkConfig",
    "FixedWorkloadConfig",
    "PerformanceBenchmarkConfig",
    "PerformanceInputsConfig",
    "load_performance_config",
]

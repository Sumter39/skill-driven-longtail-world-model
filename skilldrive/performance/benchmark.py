"""Legacy CPU filtering benchmark and deterministic repeat aggregation."""

from __future__ import annotations

import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from skilldrive.filtering.context import bind_raw_candidates
from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.filtering.pipeline import (
    CandidateFilterInput,
    finalize_candidate_validations,
    validate_candidate,
)
from skilldrive.generation.config import (
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import canonical_json_bytes, canonical_sha256
from skilldrive.generation.planning import pilot_evaluation_arm, seed_record_id
from skilldrive.generation.storage import load_raw_shard_candidates
from skilldrive.performance.config import PerformanceBenchmarkConfig
from skilldrive.performance.workload import (
    file_sha256,
    generation_task_from_row,
    load_fixed_workload,
)
from skilldrive.seeds import read_seed_records
from skilldrive.skills.detection import load_detection_config
from skilldrive.skills.loader import load_skill


BENCHMARK_SCHEMA_VERSION = 1
LEGACY_RUNNER = "cpu_filter_legacy"


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = [float(value) for value in values]
    return {
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "range": max(ordered) - min(ordered),
        "minimum": min(ordered),
        "maximum": max(ordered),
    }


def _workload_counts(workload: Mapping[str, Any]) -> tuple[int, int]:
    counts = workload.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("fixed workload counts are missing")
    task_count = counts.get("tasks")
    candidate_count = counts.get("candidates")
    if (
        isinstance(task_count, bool)
        or not isinstance(task_count, int)
        or task_count <= 0
        or isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise ValueError("fixed workload counts are invalid")
    return task_count, candidate_count


def aggregate_repeat_results(
    repeats: Sequence[Mapping[str, Any]],
    *,
    formal_candidate_count: int,
) -> dict[str, Any]:
    """Aggregate exactly three equivalent repeats and project formal wall time."""

    if len(repeats) != 3:
        raise ValueError("performance aggregation requires exactly three repeats")
    if (
        isinstance(formal_candidate_count, bool)
        or not isinstance(formal_candidate_count, int)
        or formal_candidate_count <= 0
    ):
        raise ValueError("formal_candidate_count must be a positive integer")
    def stable_nonnegative_count(name: str) -> int:
        values = []
        for item in repeats:
            value = item.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"benchmark repeat {name} must be a nonnegative integer")
            values.append(value)
        if len(set(values)) != 1:
            raise ValueError("benchmark repeats changed workload counts or filter decisions")
        return values[0]

    task_count = stable_nonnegative_count("task_count")
    candidate_count = stable_nonnegative_count("candidate_count")
    accepted_count = stable_nonnegative_count("accepted_count")
    rejected_count = stable_nonnegative_count("rejected_count")
    quality_passed = stable_nonnegative_count("quality_passed_before_diversity")
    decision_digests = {item.get("decision_sha256") for item in repeats}
    if len(decision_digests) != 1:
        raise ValueError("benchmark repeats changed workload counts or filter decisions")
    if task_count <= 0 or candidate_count <= 0:
        raise ValueError("benchmark repeat candidate_count must be positive")
    if accepted_count + rejected_count != candidate_count:
        raise ValueError("accepted_count + rejected_count must equal candidate_count")
    if quality_passed > candidate_count:
        raise ValueError("quality_passed_before_diversity cannot exceed candidate_count")
    if accepted_count > quality_passed:
        raise ValueError("accepted_count cannot exceed quality_passed_before_diversity")
    decision_sha256 = next(iter(decision_digests))
    if not isinstance(decision_sha256, str) or len(decision_sha256) != 64:
        raise ValueError("benchmark repeat decision_sha256 is invalid")

    stage_execution_values = [item.get("stage_execution_counts") for item in repeats]
    stage_rejection_values = [item.get("stage_rejection_counts") for item in repeats]
    if any(not isinstance(value, Mapping) for value in stage_execution_values):
        raise ValueError("benchmark repeat stage_execution_counts must be mappings")
    if any(not isinstance(value, Mapping) for value in stage_rejection_values):
        raise ValueError("benchmark repeat stage_rejection_counts must be mappings")
    if any(dict(value) != dict(stage_execution_values[0]) for value in stage_execution_values[1:]):
        raise ValueError("benchmark repeats changed stage execution counts")
    if any(dict(value) != dict(stage_rejection_values[0]) for value in stage_rejection_values[1:]):
        raise ValueError("benchmark repeats changed stage rejection counts")
    stage_execution_counts = dict(stage_execution_values[0])
    stage_rejection_counts = dict(stage_rejection_values[0])
    for name, values in (
        ("stage_execution_counts", stage_execution_counts),
        ("stage_rejection_counts", stage_rejection_counts),
    ):
        if any(
            not isinstance(stage, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for stage, value in values.items()
        ):
            raise ValueError(f"benchmark repeat {name} is invalid")
    if set(stage_rejection_counts) != set(stage_execution_counts):
        raise ValueError("stage rejection counts must cover every executed stage")
    if any(
        stage_rejection_counts[stage] > execution_count
        for stage, execution_count in stage_execution_counts.items()
    ):
        raise ValueError("stage rejection count cannot exceed stage execution count")
    if sum(stage_rejection_counts.values()) != rejected_count:
        raise ValueError("stage rejection counts must sum to rejected_count")

    stage_elapsed_values = [item.get("stage_elapsed_seconds") for item in repeats]
    if any(not isinstance(value, Mapping) for value in stage_elapsed_values):
        raise ValueError("benchmark repeat stage_elapsed_seconds must be mappings")
    stage_names = set(stage_elapsed_values[0])
    if any(set(value) != stage_names for value in stage_elapsed_values[1:]):
        raise ValueError("benchmark repeats changed measured filter stages")
    if stage_names != set(stage_execution_counts):
        raise ValueError("measured filter stages differ from execution counts")
    stage_elapsed = {}
    for stage in sorted(stage_names):
        values = [float(item[stage]) for item in stage_elapsed_values]
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("benchmark stage elapsed seconds must be finite and nonnegative")
        stage_elapsed[stage] = _distribution(values)

    elapsed = [float(item["elapsed_seconds"]) for item in repeats]
    if any(not math.isfinite(value) or value <= 0.0 for value in elapsed):
        raise ValueError("benchmark repeat elapsed_seconds must be positive and finite")
    candidate_rates = [candidate_count / value for value in elapsed]
    task_rates = [task_count / value for value in elapsed]
    accepted_rates = [accepted_count / value for value in elapsed]
    scale = formal_candidate_count / candidate_count
    formal_seconds = [value * scale for value in elapsed]
    return {
        "repeat_count": 3,
        "workload_task_count": task_count,
        "workload_candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "quality_passed_before_diversity": quality_passed,
        "quality_pass_rate_before_diversity": quality_passed / candidate_count,
        "decision_sha256": decision_sha256,
        "elapsed_seconds": _distribution(elapsed),
        "tasks_per_second": _distribution(task_rates),
        "candidates_per_second": _distribution(candidate_rates),
        "accepted_per_second": _distribution(accepted_rates),
        "stage_elapsed_seconds": stage_elapsed,
        "stage_execution_counts": stage_execution_counts,
        "stage_rejection_counts": stage_rejection_counts,
        "stage_rejection_rates": {
            stage: (
                stage_rejection_counts[stage] / execution_count
                if execution_count
                else 0.0
            )
            for stage, execution_count in sorted(stage_execution_counts.items())
        },
        "formal_projection": {
            "candidate_count": formal_candidate_count,
            "seconds": _distribution(formal_seconds),
            "hours_p50": _percentile(formal_seconds, 0.50) / 3600.0,
            "hours_p95": _percentile(formal_seconds, 0.95) / 3600.0,
            "projection_contract": "linear_candidate_count_from_fixed_cpu_filter_v1",
        },
    }


def run_legacy_cpu_filter_once(
    workload: Mapping[str, Any],
    *,
    config: PerformanceBenchmarkConfig,
    repository_root: str | Path = ".",
    repeat_index: int,
) -> dict[str, Any]:
    """Run the current single-process Pilot filtering path on one fixed workload."""

    from skilldrive.data.av2_reader import load_av2_scenario

    root = Path(repository_root).resolve()
    generation = load_counterfactual_config(
        root / config.inputs.generation_config,
        repository_root=root,
    )
    filter_config = load_filter_config(root / config.inputs.filter_config)
    detection_config = load_detection_config(root / config.inputs.detection_config)
    records = read_seed_records(root / generation.inputs.seed_manifest)
    records_by_id = {seed_record_id(record): record for record in records}
    expected_task_count, expected_candidate_count = _workload_counts(workload)
    tasks = workload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("legacy CPU benchmark workload tasks are missing")
    if len(tasks) != expected_task_count:
        raise ValueError("legacy CPU benchmark tasks differ from fixed workload counts")
    skill_ids = {
        str(entry["task"]["skill_id"])
        for entry in tasks
        if isinstance(entry, Mapping) and isinstance(entry.get("task"), Mapping)
    }
    skills = {
        skill_id: load_skill(
            root / generation.formal_catalog.parent / f"{skill_id}.yaml"
        )
        for skill_id in skill_ids
    }

    validations = []
    current_scenario_id: str | None = None
    current_source = None
    scenario_load_count = 0
    measured_started = time.perf_counter()
    for entry in tasks:
        if not isinstance(entry, Mapping):
            raise ValueError("legacy CPU benchmark task entry is invalid")
        task = generation_task_from_row(entry["task"])
        record = records_by_id.get(task.seed_record_id)
        if record is None or record.source_path != entry.get("source_path"):
            raise ValueError("legacy CPU benchmark task differs from its seed record")
        source_path = (root / generation.inputs.data_root / record.source_path).resolve()
        if task.scenario_id != current_scenario_id:
            current_source = load_av2_scenario(source_path)
            current_scenario_id = task.scenario_id
            scenario_load_count += 1
        if current_source is None:
            raise RuntimeError("legacy CPU benchmark source scenario was not loaded")
        raw = load_raw_shard_candidates(
            root / str(entry["raw_commit"]),
            expected_semantic_config_sha256=task.semantic_config_sha256,
        )
        bound = bind_raw_candidates(raw, [task], [record])
        arm = pilot_evaluation_arm(task, none_skill_id=generation.none_skill_id)
        cohort = "learned_none_control" if arm == "learned_none_control" else "formal"
        primary_role = generation.skills_by_id[task.skill_id].primary_generated_role
        for candidate in bound:
            validations.append(
                validate_candidate(
                    CandidateFilterInput(
                        bound=candidate,
                        skill=skills[task.skill_id],
                        source_scenario=current_source,
                        primary_generated_role=primary_role,
                    ),
                    filter_config=filter_config,
                    detection_config=detection_config,
                ).compact(cohort=cohort)
            )
    batch = finalize_candidate_validations(
        validations,
        filter_config=filter_config,
        filter_semantic_sha256=str(workload["filter_semantic_sha256"]),
    )
    elapsed = time.perf_counter() - measured_started
    decisions = [
        {
            "candidate_id": item.candidate_id,
            "filter_evaluation_id": item.filter_evaluation_id,
            "accepted": item.accepted,
            "rejection_reasons": list(item.rejection_reasons),
            "metrics": dict(item.metrics),
        }
        for item in sorted(batch.decisions, key=lambda value: value.candidate_id)
    ]
    if len(decisions) != expected_candidate_count:
        raise ValueError(
            "legacy CPU benchmark candidates differ from fixed workload counts"
        )
    accepted = sum(item["accepted"] for item in decisions)
    stage_rejections: Counter[str] = Counter()
    for item in decisions:
        if item["accepted"]:
            continue
        failed_stage = item["metrics"].get("first_failed_stage")
        if not isinstance(failed_stage, str) or not failed_stage:
            raise ValueError("rejected benchmark decision lacks first_failed_stage")
        stage_rejections[failed_stage] += 1
    stage_execution_counts = dict(sorted(batch.stage_execution_counts.items()))
    unexpected_rejection_stages = set(stage_rejections).difference(
        stage_execution_counts
    )
    if unexpected_rejection_stages:
        raise ValueError("rejected benchmark decision used an unexecuted stage")
    stage_rejection_counts = {
        stage: stage_rejections.get(stage, 0) for stage in stage_execution_counts
    }
    if any(
        stage_rejection_counts[stage] > execution_count
        for stage, execution_count in stage_execution_counts.items()
    ):
        raise ValueError("stage rejection count cannot exceed stage execution count")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "counterfactual_performance_repeat",
        "runner": LEGACY_RUNNER,
        "repeat_index": repeat_index,
        "task_count": len(tasks),
        "candidate_count": len(decisions),
        "accepted_count": accepted,
        "rejected_count": len(decisions) - accepted,
        "quality_passed_before_diversity": sum(
            item.quality_passed for item in batch.validations
        ),
        "elapsed_seconds": elapsed,
        "candidates_per_second": len(decisions) / elapsed,
        "accepted_per_second": accepted / elapsed,
        "decision_sha256": canonical_sha256(decisions),
        "stage_execution_counts": stage_execution_counts,
        "stage_rejection_counts": stage_rejection_counts,
        "stage_rejection_rates": {
            stage: (
                stage_rejection_counts[stage] / execution_count
                if execution_count
                else 0.0
            )
            for stage, execution_count in stage_execution_counts.items()
        },
        "stage_elapsed_seconds": dict(batch.stage_elapsed_seconds),
        "scenario_load_count": scenario_load_count,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }


def run_cpu_filter_legacy_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    config_path: str | Path,
    workload_path: str | Path,
    repository_root: str | Path = ".",
) -> tuple[Path, dict[str, Any]]:
    root = Path(repository_root).resolve()
    preflight_started = time.perf_counter()
    workload = load_fixed_workload(workload_path, repository_root=root)
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=root,
        generation_config_path=config.inputs.generation_config,
        filter_config_path=config.inputs.filter_config,
        detection_config_path=config.inputs.detection_config,
    )
    if fingerprint.semantic_sha256 != workload.get("filter_semantic_sha256"):
        raise ValueError("legacy CPU benchmark filter semantic fingerprint changed")
    preflight_elapsed = time.perf_counter() - preflight_started
    expected_task_count, expected_candidate_count = _workload_counts(workload)

    repeats = []
    for repeat_index in range(config.benchmark.repeats):
        value = run_legacy_cpu_filter_once(
            workload,
            config=config,
            repository_root=root,
            repeat_index=repeat_index,
        )
        if (
            value.get("task_count") != expected_task_count
            or value.get("candidate_count") != expected_candidate_count
        ):
            raise ValueError("benchmark repeat differs from fixed workload counts")
        repeats.append(value)
        print(
            f"cpu-filter legacy repeat {repeat_index + 1}/3: "
            f"{value['candidate_count']} candidates, "
            f"{value['elapsed_seconds']:.3f}s, "
            f"{value['candidates_per_second']:.2f} candidates/s",
            flush=True,
        )
    aggregate = aggregate_repeat_results(
        repeats,
        formal_candidate_count=config.benchmark.formal_candidate_count,
    )
    benchmark_contract = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "runner": LEGACY_RUNNER,
        "workload_id": workload["workload_id"],
        "workload_sha256": file_sha256(workload_path),
        "performance_config_sha256": file_sha256(config_path),
        "filter_semantic_sha256": fingerprint.semantic_sha256,
        "repeats": config.benchmark.repeats,
        "formal_candidate_count": config.benchmark.formal_candidate_count,
        "measurement_scope": (
            "verified_raw_load_plus_source_load_plus_bind_plus_individual_filters_"
            "plus_global_diversity_v1"
        ),
    }
    benchmark_id = canonical_sha256(benchmark_contract)
    result_root = (
        root
        / config.output_root
        / "results"
        / str(workload["workload_id"])
        / LEGACY_RUNNER
        / benchmark_id
    )
    for repeat in repeats:
        _atomic_write(
            result_root / f"repeat-{int(repeat['repeat_index']) + 1:02d}.json",
            repeat,
        )
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "counterfactual_performance_benchmark_summary",
        "status": "completed",
        "benchmark_id": benchmark_id,
        "benchmark_contract": benchmark_contract,
        "preflight_seconds": preflight_elapsed,
        "aggregate": aggregate,
        "repeat_elapsed_seconds": [item["elapsed_seconds"] for item in repeats],
        "repeat_decision_sha256": [item["decision_sha256"] for item in repeats],
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    summary_path = result_root / "summary.json"
    _atomic_write(summary_path, summary)
    return summary_path, summary


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "LEGACY_RUNNER",
    "aggregate_repeat_results",
    "run_cpu_filter_legacy_benchmark",
    "run_legacy_cpu_filter_once",
]

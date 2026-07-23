"""Prepared-map single-process filtering benchmark on one fixed workload."""

from __future__ import annotations

import os
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from skilldrive.filtering.context import bind_raw_candidates
from skilldrive.filtering.contracts import FilterStage
from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.filtering.pipeline import (
    DEFAULT_MAP_BATCH_SIZE,
    MAP_BATCH_SIZES,
    CandidateFilterInput,
    finalize_candidate_validations,
    validate_candidate_individual_batch,
)
from skilldrive.filtering.prepared_map import (
    PreparedMapGeometry,
    PreparedMapVerificationSession,
    prepare_map_geometry,
)
from skilldrive.generation.config import (
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import canonical_json_bytes, canonical_sha256
from skilldrive.generation.planning import pilot_evaluation_arm, seed_record_id
from skilldrive.generation.storage import load_raw_shard_candidates
from skilldrive.performance.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    aggregate_repeat_results,
)
from skilldrive.performance.config import PerformanceBenchmarkConfig
from skilldrive.performance.workload import (
    file_sha256,
    generation_task_from_row,
    load_fixed_workload,
)
from skilldrive.seeds import read_seed_records
from skilldrive.skills.detection import load_detection_config
from skilldrive.skills.loader import load_skill


PREPARED_MAP_RUNNER = "cpu_filter_prepared_map"


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


def _normalized_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return normalized


def _runner_source_sha256(root: Path) -> dict[str, str]:
    paths = (
        "skilldrive/performance/prepared_benchmark.py",
        "scripts/generation/run_prepared_map_benchmark.py",
    )
    return {relative: file_sha256(root / relative) for relative in paths}


def run_prepared_map_cpu_filter_once(
    workload: Mapping[str, Any],
    *,
    config: PerformanceBenchmarkConfig,
    repository_root: str | Path = ".",
    repeat_index: int,
    expected_decision_sha256: str | None = None,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
) -> dict[str, Any]:
    """Run one fixed-workload repeat with one prepared map per scenario."""

    from skilldrive.data.av2_reader import load_av2_scenario

    expected_digest = _normalized_sha256(
        expected_decision_sha256,
        "expected_decision_sha256",
    )
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
        raise ValueError("prepared-map CPU benchmark workload tasks are missing")
    if len(tasks) != expected_task_count:
        raise ValueError(
            "prepared-map CPU benchmark tasks differ from fixed workload counts"
        )
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

    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")
    validations = []
    current_candidates: list[tuple[CandidateFilterInput, str]] = []
    current_scenario_id: str | None = None
    current_source = None
    current_prepared: PreparedMapGeometry | None = None
    current_session: PreparedMapVerificationSession | None = None
    prepared_by_scenario: dict[str, PreparedMapGeometry] = {}
    source_paths_by_scenario: dict[str, Path] = {}
    scenario_load_count = 0
    scenario_read_seconds = 0.0
    raw_read_seconds = 0.0
    candidate_bind_seconds = 0.0
    candidate_validation_seconds = 0.0
    prepared_map_seconds = 0.0
    map_integrity_finalize_seconds = 0.0

    def finalize_current_session() -> None:
        nonlocal current_session, map_integrity_finalize_seconds
        if current_session is None:
            return
        started = time.perf_counter()
        current_session.finalize()
        map_integrity_finalize_seconds += time.perf_counter() - started
        current_session = None

    def flush_current_candidates() -> None:
        nonlocal candidate_validation_seconds
        if not current_candidates:
            return
        candidate_inputs = [item for item, _ in current_candidates]
        started = time.perf_counter()
        candidate_results = validate_candidate_individual_batch(
            candidate_inputs,
            filter_config=filter_config,
            detection_config=detection_config,
            map_batch_size=map_batch_size,
        )
        candidate_validation_seconds += time.perf_counter() - started
        if len(candidate_results) != len(current_candidates):
            raise ValueError("prepared-map batching omitted a candidate")
        validations.extend(
            result.compact(cohort=cohort)
            for result, (_, cohort) in zip(
                candidate_results,
                current_candidates,
                strict=True,
            )
        )
        current_candidates.clear()

    measured_started = time.perf_counter()
    for entry in tasks:
        if not isinstance(entry, Mapping):
            raise ValueError("prepared-map CPU benchmark task entry is invalid")
        task = generation_task_from_row(entry["task"])
        record = records_by_id.get(task.seed_record_id)
        if record is None or record.source_path != entry.get("source_path"):
            raise ValueError(
                "prepared-map CPU benchmark task differs from its seed record"
            )
        source_path = (
            root / generation.inputs.data_root / record.source_path
        ).resolve()
        previous_path = source_paths_by_scenario.setdefault(
            task.scenario_id,
            source_path,
        )
        if previous_path != source_path:
            raise ValueError("one scenario_id references multiple source paths")
        if task.scenario_id != current_scenario_id:
            flush_current_candidates()
            finalize_current_session()
            started = time.perf_counter()
            current_source = load_av2_scenario(source_path)
            scenario_read_seconds += time.perf_counter() - started
            current_scenario_id = task.scenario_id
            scenario_load_count += 1
            current_prepared = prepared_by_scenario.get(task.scenario_id)
            if current_prepared is None:
                started = time.perf_counter()
                current_prepared = prepare_map_geometry(current_source)
                prepared_map_seconds += time.perf_counter() - started
                prepared_by_scenario[task.scenario_id] = current_prepared
            current_session = PreparedMapVerificationSession(
                current_source,
                current_prepared,
            )
        if current_source is None or current_prepared is None:
            raise RuntimeError("prepared-map CPU benchmark source was not loaded")
        started = time.perf_counter()
        raw = load_raw_shard_candidates(
            root / str(entry["raw_commit"]),
            expected_semantic_config_sha256=task.semantic_config_sha256,
        )
        raw_read_seconds += time.perf_counter() - started
        started = time.perf_counter()
        bound = bind_raw_candidates(raw, [task], [record])
        candidate_bind_seconds += time.perf_counter() - started
        arm = pilot_evaluation_arm(task, none_skill_id=generation.none_skill_id)
        cohort = "learned_none_control" if arm == "learned_none_control" else "formal"
        primary_role = generation.skills_by_id[
            task.skill_id
        ].primary_generated_role
        for candidate in bound:
            current_candidates.append(
                (
                    CandidateFilterInput(
                        bound=candidate,
                        skill=skills[task.skill_id],
                        source_scenario=current_source,
                        primary_generated_role=primary_role,
                        prepared_map=current_prepared,
                        map_verification_session=current_session,
                    ),
                    cohort,
                )
            )
    flush_current_candidates()
    finalize_current_session()
    global_finalize_started = time.perf_counter()
    batch = finalize_candidate_validations(
        validations,
        filter_config=filter_config,
        filter_semantic_sha256=str(workload["filter_semantic_sha256"]),
    )
    global_finalize_seconds = time.perf_counter() - global_finalize_started
    elapsed = time.perf_counter() - measured_started
    individual_filter_seconds = (
        candidate_validation_seconds
        + prepared_map_seconds
        + map_integrity_finalize_seconds
    )
    accounted_seconds = (
        scenario_read_seconds
        + raw_read_seconds
        + candidate_bind_seconds
        + individual_filter_seconds
        + global_finalize_seconds
    )
    task_bookkeeping_seconds = elapsed - accounted_seconds
    if task_bookkeeping_seconds < -1e-9:
        raise RuntimeError("prepared-map timing breakdown exceeds measured elapsed time")
    task_bookkeeping_seconds = max(0.0, task_bookkeeping_seconds)
    timing_breakdown_seconds = {
        "scenario_read": scenario_read_seconds,
        "raw_read": raw_read_seconds,
        "candidate_bind": candidate_bind_seconds,
        "individual_filter": individual_filter_seconds,
        "global_finalize": global_finalize_seconds,
        "task_bookkeeping": task_bookkeeping_seconds,
    }
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
            "prepared-map CPU benchmark candidates differ from fixed workload counts"
        )
    decision_sha256 = canonical_sha256(decisions)
    if expected_digest is not None and decision_sha256 != expected_digest:
        raise ValueError(
            "prepared-map CPU benchmark decision_sha256 differs from expected: "
            f"expected={expected_digest}, actual={decision_sha256}"
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
        "runner": PREPARED_MAP_RUNNER,
        "repeat_index": repeat_index,
        "map_batch_size": map_batch_size,
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
        "decision_sha256": decision_sha256,
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
        "timing_breakdown_seconds": timing_breakdown_seconds,
        "scenario_load_count": scenario_load_count,
        "prepared_map_count": len(prepared_by_scenario),
        "prepared_map_seconds": prepared_map_seconds,
        "map_integrity_finalize_seconds": map_integrity_finalize_seconds,
        "map_subsystem_seconds": (
            prepared_map_seconds
            + float(batch.stage_elapsed_seconds[FilterStage.MAP.value])
            + map_integrity_finalize_seconds
        ),
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }


def run_cpu_filter_prepared_map_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    config_path: str | Path,
    workload_path: str | Path,
    repository_root: str | Path = ".",
    expected_decision_sha256: str | None = None,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
) -> tuple[Path, dict[str, Any]]:
    """Execute three prepared-map repeats and persist legacy-compatible evidence."""

    root = Path(repository_root).resolve()
    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")
    explicit_expected = _normalized_sha256(
        expected_decision_sha256,
        "expected_decision_sha256",
    )
    preflight_started = time.perf_counter()
    workload = load_fixed_workload(workload_path, repository_root=root)
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=root,
        generation_config_path=config.inputs.generation_config,
        filter_config_path=config.inputs.filter_config,
        detection_config_path=config.inputs.detection_config,
    )
    if fingerprint.semantic_sha256 != workload.get("filter_semantic_sha256"):
        raise ValueError("prepared-map CPU benchmark filter semantic fingerprint changed")
    runner_source_sha256 = _runner_source_sha256(root)
    preflight_elapsed = time.perf_counter() - preflight_started
    expected_task_count, expected_candidate_count = _workload_counts(workload)

    repeats = []
    decision_reference = explicit_expected
    for repeat_index in range(config.benchmark.repeats):
        value = run_prepared_map_cpu_filter_once(
            workload,
            config=config,
            repository_root=root,
            repeat_index=repeat_index,
            expected_decision_sha256=decision_reference,
            map_batch_size=map_batch_size,
        )
        if (
            value.get("task_count") != expected_task_count
            or value.get("candidate_count") != expected_candidate_count
        ):
            raise ValueError("benchmark repeat differs from fixed workload counts")
        current_digest = _normalized_sha256(
            value.get("decision_sha256"),
            "repeat decision_sha256",
        )
        if decision_reference is None:
            decision_reference = current_digest
        elif current_digest != decision_reference:
            raise ValueError(
                "prepared-map benchmark repeat decision_sha256 changed: "
                f"expected={decision_reference}, actual={current_digest}"
            )
        repeats.append(value)
        print(
            f"cpu-filter prepared-map repeat {repeat_index + 1}/3: "
            f"{value['candidate_count']} candidates, "
            f"{value['elapsed_seconds']:.3f}s, "
            f"{value['candidates_per_second']:.2f} candidates/s",
            flush=True,
        )
    if decision_reference is None:
        raise RuntimeError("prepared-map benchmark produced no decision digest")
    aggregate = aggregate_repeat_results(
        repeats,
        formal_candidate_count=config.benchmark.formal_candidate_count,
    )
    if aggregate["decision_sha256"] != decision_reference:
        raise ValueError("prepared-map aggregate decision_sha256 changed")
    decision_reference_source = "cli" if explicit_expected is not None else "first_repeat"
    for repeat in repeats:
        repeat["decision_reference_sha256"] = decision_reference
        repeat["decision_reference_source"] = decision_reference_source
    benchmark_contract = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "runner": PREPARED_MAP_RUNNER,
        "workload_id": workload["workload_id"],
        "workload_sha256": file_sha256(workload_path),
        "performance_config_sha256": file_sha256(config_path),
        "filter_semantic_sha256": fingerprint.semantic_sha256,
        "runner_source_sha256": runner_source_sha256,
        "repeats": config.benchmark.repeats,
        "formal_candidate_count": config.benchmark.formal_candidate_count,
        "map_batch_size": map_batch_size,
        "decision_reference_sha256": decision_reference,
        "decision_reference_source": decision_reference_source,
        "measurement_scope": (
            "verified_raw_load_plus_source_load_plus_prepare_map_once_per_scenario_"
            "plus_bind_plus_chunked_batch_map_filter_plus_final_source_map_"
            "integrity_plus_global_diversity_v3"
        ),
    }
    benchmark_id = canonical_sha256(benchmark_contract)
    result_root = (
        root
        / config.output_root
        / "results"
        / str(workload["workload_id"])
        / PREPARED_MAP_RUNNER
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
    "PREPARED_MAP_RUNNER",
    "run_cpu_filter_prepared_map_benchmark",
    "run_prepared_map_cpu_filter_once",
]

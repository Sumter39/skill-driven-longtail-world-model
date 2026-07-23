"""Scenario-parallel individual filtering for one verified fixed workload."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.filtering.context import bind_raw_candidates
from skilldrive.filtering.contracts import FilterCheck, FilterStage
from skilldrive.filtering.diversity import DiversityCandidate
from skilldrive.filtering.pipeline import (
    DEFAULT_MAP_BATCH_SIZE,
    MAP_BATCH_SIZES,
    BatchValidationResult,
    CandidateFilterIdentity,
    CandidateFilterInput,
    CompactCandidateValidationResult,
    TimedFilterCheck,
    finalize_candidate_validations,
    validate_candidate_individual_batch,
)
from skilldrive.filtering.prepared_map import (
    PreparedMapVerificationSession,
    prepare_map_geometry,
)
from skilldrive.generation.config import (
    CounterfactualFilterConfig,
    CounterfactualGenerationConfig,
)
from skilldrive.generation.contracts import (
    FilterRejection,
    GenerationTask,
    candidate_id as make_candidate_id,
    canonical_json,
    canonical_sha256,
)
from skilldrive.generation.planning import pilot_evaluation_arm, seed_record_id
from skilldrive.generation.storage import StoredRawCandidate, load_raw_shard_candidates
from skilldrive.performance.workload import generation_task_from_row
from skilldrive.schemas import Scenario, SkillSpec
from skilldrive.seeds.records import SeedRecord, read_seed_records
from skilldrive.skills.detection import DetectionConfig
from skilldrive.skills.loader import load_skill


ScenarioLoader = Callable[[str | Path], Scenario]
RawLoader = Callable[..., Sequence[StoredRawCandidate]]
MapPreparer = Callable[[Scenario], Any]
CandidateValidator = Callable[..., Any]
_CheckWire = tuple[str, tuple[str, ...], str, float]
_WORKER_STATE: tuple[
    CounterfactualFilterConfig,
    DetectionConfig,
    ScenarioLoader,
    RawLoader,
    MapPreparer,
    CandidateValidator | None,
    int,
] | None = None


class ParallelFilterWorkerError(RuntimeError):
    """A worker failed before the parent could globally finalize."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class CompactValidationWire:
    """Pickle-safe compact result; only MappingProxy metrics are encoded."""

    task_index: int
    identity: CandidateFilterIdentity
    cohort: str
    checks: tuple[_CheckWire, ...]
    quality_score: float | None
    diversity_candidate: DiversityCandidate | None

    @property
    def order_key(self) -> tuple[int, int, str]:
        return self.task_index, self.identity.candidate_index, self.identity.candidate_id


def _to_wire(
    value: CompactCandidateValidationResult,
    task_index: int,
) -> CompactValidationWire:
    checks = tuple(
        (
            item.check.stage.value,
            item.check.rejection_values,
            canonical_json(_plain(item.check.metrics)),
            item.elapsed_seconds,
        )
        for item in value.checks
    )
    return CompactValidationWire(
        task_index, value.identity, value.cohort, checks,
        value.quality_score, value.diversity_candidate,
    )


def _from_wire(value: CompactValidationWire) -> CompactCandidateValidationResult:
    checks = tuple(
        TimedFilterCheck(
            FilterCheck(
                stage=FilterStage(stage),
                rejection_reasons=tuple(FilterRejection(reason) for reason in reasons),
                metrics=json.loads(metrics_json),
            ),
            elapsed_seconds=elapsed,
        )
        for stage, reasons, metrics_json, elapsed in value.checks
    )
    return CompactCandidateValidationResult(
        identity=value.identity,
        cohort=value.cohort,
        checks=checks,
        quality_score=value.quality_score,
        diversity_candidate=value.diversity_candidate,
    )


@dataclass(frozen=True)
class ParallelFilterResult:
    batch: BatchValidationResult
    timings: Mapping[str, float]
    requested_worker_count: int
    effective_worker_count: int
    worker_pids: tuple[int, ...]
    scenario_load_count: int
    prepared_map_count: int
    validation_order: tuple[tuple[int, int, str], ...]
    decision_sha256: str
    semantic_decision_sha256: str
    stage_rejection_counts: Mapping[str, int]
    map_batch_size: int

    @property
    def stage_execution_counts(self) -> Mapping[str, int]:
        return self.batch.stage_execution_counts


@dataclass(frozen=True)
class _TaskJob:
    task: GenerationTask
    raw_commit: Path
    record: SeedRecord
    skill: SkillSpec
    primary_role: str
    cohort: str


@dataclass(frozen=True)
class _ScenarioJob:
    scenario_id: str
    source_path: Path
    tasks: tuple[_TaskJob, ...]


@dataclass(frozen=True)
class _ScenarioResult:
    scenario_id: str
    task_ids: tuple[str, ...]
    wires: tuple[CompactValidationWire, ...]
    scenario_load_seconds: float
    prepared_map_seconds: float
    map_integrity_finalize_seconds: float
    raw_load_bind_seconds: float
    individual_filter_seconds: float
    worker_pid: int


def _cohort(task: GenerationTask, none_skill_id: str) -> str:
    arm = pilot_evaluation_arm(task, none_skill_id=none_skill_id)
    return "learned_none_control" if arm == "learned_none_control" else "formal"


def _spawn_safe(value: CounterfactualFilterConfig) -> CounterfactualFilterConfig:
    policy = replace(
        value.parameter_policy,
        absolute_tolerances=dict(value.parameter_policy.absolute_tolerances),
    )
    return replace(value, parameter_policy=policy)


def _build_jobs(
    workload: Mapping[str, Any],
    root: Path,
    generation: CounterfactualGenerationConfig,
) -> tuple[_ScenarioJob, ...]:
    rows, counts = workload.get("tasks"), workload.get("counts")
    if not isinstance(rows, list) or not rows or not isinstance(counts, Mapping):
        raise ValueError("fixed workload tasks or counts are missing")
    records = read_seed_records(root / generation.inputs.seed_manifest)
    records_by_id = {seed_record_id(record): record for record in records}
    parsed = []
    for entry in rows:
        if not isinstance(entry, Mapping):
            raise ValueError("fixed workload task entry is invalid")
        task = generation_task_from_row(entry.get("task"))
        record = records_by_id.get(task.seed_record_id)
        if record is None or record.source_path != entry.get("source_path"):
            raise ValueError("fixed workload task differs from its seed record")
        parsed.append((task, entry, record))
    if len({task.task_id for task, _, _ in parsed}) != len(parsed):
        raise ValueError("fixed workload task IDs must be unique")
    if counts.get("tasks") != len(parsed) or counts.get("candidates") != sum(
        task.candidate_budget for task, _, _ in parsed
    ):
        raise ValueError("fixed workload counts differ from task rows")

    skills_by_id = generation.skills_by_id
    skill_root = (root / generation.formal_catalog).parent
    skills = {
        skill_id: load_skill(skill_root / f"{skill_id}.yaml")
        for skill_id in {task.skill_id for task, _, _ in parsed}
    }
    data_root = root / generation.inputs.data_root
    grouped: dict[str, list[_TaskJob]] = {}
    source_paths: dict[str, Path] = {}
    for task, entry, record in parsed:
        skill_config = skills_by_id.get(task.skill_id)
        if skill_config is None:
            raise ValueError("fixed workload task references a non-formal skill")
        source = (data_root / record.source_path).resolve()
        if source_paths.setdefault(task.scenario_id, source) != source:
            raise ValueError("one scenario maps to multiple source paths")
        grouped.setdefault(task.scenario_id, []).append(
            _TaskJob(
                task=task,
                raw_commit=(root / str(entry["raw_commit"])).resolve(),
                record=record,
                skill=skills[task.skill_id],
                primary_role=skill_config.primary_generated_role,
                cohort=_cohort(task, generation.none_skill_id),
            )
        )
    if counts.get("scenarios") not in (None, len(grouped)):
        raise ValueError("fixed workload scenario count differs from task rows")
    jobs = [
        _ScenarioJob(
            scenario_id,
            source_paths[scenario_id],
            tuple(sorted(tasks, key=lambda item: (item.task.task_index, item.task.task_id))),
        )
        for scenario_id, tasks in grouped.items()
    ]
    return tuple(sorted(jobs, key=lambda item: (item.tasks[0].task.task_index, item.scenario_id)))


def _init_worker(filter_config, detection_config, dependencies, map_batch_size) -> None:
    global _WORKER_STATE
    _WORKER_STATE = (filter_config, detection_config, *dependencies, map_batch_size)


def _run_scenario(job: _ScenarioJob) -> _ScenarioResult:
    if _WORKER_STATE is None:
        raise RuntimeError("parallel filter worker was not initialized")
    (
        filter_config,
        detection_config,
        scenario_loader,
        raw_loader,
        map_preparer,
        candidate_validator,
        map_batch_size,
    ) = _WORKER_STATE
    started = time.perf_counter()
    source = scenario_loader(job.source_path)
    load_seconds = time.perf_counter() - started
    if not isinstance(source, Scenario) or source.scenario_id != job.scenario_id:
        raise ValueError("scenario loader returned the wrong scenario")
    started = time.perf_counter()
    prepared_map = map_preparer(source)
    map_seconds = time.perf_counter() - started
    map_session = PreparedMapVerificationSession(source, prepared_map)
    wires, raw_seconds = [], 0.0
    queued: list[tuple[CandidateFilterInput, _TaskJob]] = []
    for item in job.tasks:
        started = time.perf_counter()
        raw = raw_loader(
            item.raw_commit,
            expected_semantic_config_sha256=item.task.semantic_config_sha256,
        )
        bound = sorted(
            bind_raw_candidates(raw, [item.task], [item.record]),
            key=lambda candidate: (candidate.raw.candidate_index, candidate.raw.candidate_id),
        )
        raw_seconds += time.perf_counter() - started
        if [candidate.raw.candidate_index for candidate in bound] != list(
            range(item.task.candidate_budget)
        ):
            raise ValueError("raw candidates differ from the task budget")
        for candidate in bound:
            queued.append(
                (
                    CandidateFilterInput(
                        bound=candidate,
                        skill=item.skill,
                        source_scenario=source,
                        primary_generated_role=item.primary_role,
                        prepared_map=prepared_map,
                        map_verification_session=map_session,
                    ),
                    item,
                )
            )
    started = time.perf_counter()
    candidate_inputs = [candidate for candidate, _ in queued]
    if candidate_validator is None:
        candidate_results = validate_candidate_individual_batch(
            candidate_inputs,
            filter_config=filter_config,
            detection_config=detection_config,
            map_batch_size=map_batch_size,
        )
    else:
        candidate_results = tuple(
            candidate_validator(
                candidate,
                filter_config=filter_config,
                detection_config=detection_config,
            )
            for candidate in candidate_inputs
        )
    if len(candidate_results) != len(queued):
        raise ValueError("scenario filtering omitted a candidate")
    for validation, (_, item) in zip(candidate_results, queued, strict=True):
        compact = validation.compact(cohort=item.cohort)
        if not isinstance(compact, CompactCandidateValidationResult):
            raise TypeError("candidate validator returned an invalid compact result")
        wires.append(_to_wire(compact, item.task.task_index))
    filter_seconds = time.perf_counter() - started
    started = time.perf_counter()
    map_session.finalize()
    map_integrity_seconds = time.perf_counter() - started
    return _ScenarioResult(
        job.scenario_id,
        tuple(item.task.task_id for item in job.tasks),
        tuple(sorted(wires, key=lambda wire: wire.order_key)),
        load_seconds,
        map_seconds,
        map_integrity_seconds,
        raw_seconds,
        filter_seconds,
        os.getpid(),
    )


def _worker_pid() -> int:
    return os.getpid()


def _execute(
    jobs: Sequence[_ScenarioJob],
    worker_count: int,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
    dependencies: tuple[
        ScenarioLoader,
        RawLoader,
        MapPreparer,
        CandidateValidator | None,
    ],
    map_batch_size: int,
) -> tuple[tuple[_ScenarioResult, ...], float, float]:
    effective = min(worker_count, len(jobs))
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=effective,
        mp_context=context,
        initializer=_init_worker,
        initargs=(filter_config, detection_config, dependencies, map_batch_size),
    ) as executor:
        for future in [executor.submit(_worker_pid) for _ in range(effective)]:
            future.result()
        startup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        futures = {
            executor.submit(_run_scenario, job): job for job in jobs
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                for pending in futures:
                    pending.cancel()
                raise ParallelFilterWorkerError(
                    f"scenario filter worker failed for {futures[future].scenario_id}"
                ) from error
        execution_seconds = time.perf_counter() - started
    return tuple(results), startup_seconds, execution_seconds


def _restore(
    jobs: Sequence[_ScenarioJob],
    results: Sequence[_ScenarioResult],
) -> tuple[tuple[CompactCandidateValidationResult, ...], tuple[tuple[int, int, str], ...]]:
    task_jobs = {item.task.task_id: item for job in jobs for item in job.tasks}
    expected_scenarios = {
        job.scenario_id: tuple(item.task.task_id for item in job.tasks) for job in jobs
    }
    actual_scenarios = {result.scenario_id: result.task_ids for result in results}
    if len(actual_scenarios) != len(results) or actual_scenarios != expected_scenarios:
        raise ValueError("worker results do not exactly cover scenario jobs")
    wires = [wire for result in results for wire in result.wires]
    slots, candidate_ids = set(), set()
    for wire in wires:
        identity = wire.identity
        task_job = task_jobs.get(identity.task_id)
        if task_job is None:
            raise ValueError("wire references an unknown task")
        task, slot = task_job.task, (identity.task_id, identity.candidate_index)
        expected_id = make_candidate_id(
            task_id=task.task_id,
            candidate_index=identity.candidate_index,
            latent_seed=identity.latent_seed,
            checkpoint_sha256=identity.checkpoint_sha256,
            semantic_config_sha256=identity.semantic_config_sha256,
        )
        if (
            wire.task_index != task.task_index
            or wire.cohort != task_job.cohort
            or not 0 <= identity.candidate_index < task.candidate_budget
            or identity.candidate_id != expected_id
            or slot in slots
            or identity.candidate_id in candidate_ids
        ):
            raise ValueError("wire contains an invalid or duplicate candidate")
        slots.add(slot)
        candidate_ids.add(identity.candidate_id)
    expected_slots = {
        (task_id, index)
        for task_id, task_job in task_jobs.items()
        for index in range(task_job.task.candidate_budget)
    }
    if slots != expected_slots:
        raise ValueError("wire results do not cover the fixed workload")
    ordered = tuple(sorted(wires, key=lambda wire: wire.order_key))
    return tuple(_from_wire(wire) for wire in ordered), tuple(
        wire.order_key for wire in ordered
    )


def _decision_rows(batch: BatchValidationResult) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate_id,
            "filter_evaluation_id": item.filter_evaluation_id,
            "accepted": item.accepted,
            "rejection_reasons": list(item.rejection_reasons),
            "metrics": dict(item.metrics),
        }
        for item in sorted(batch.decisions, key=lambda decision: decision.candidate_id)
    ]


def _semantic_decision_rows(batch: BatchValidationResult) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": item.candidate_id,
            "accepted": item.accepted,
            "first_failed_stage": item.metrics.get("first_failed_stage"),
            "primary_rejection_reason": item.metrics.get(
                "primary_rejection_reason"
            ),
            "rejection_reasons": list(item.rejection_reasons),
            "evaluated_stages": list(item.metrics.get("evaluated_stages", ())),
            "skipped_stages": list(item.metrics.get("skipped_stages", ())),
        }
        for item in sorted(batch.decisions, key=lambda decision: decision.candidate_id)
    ]


def run_parallel_filter_workload(
    workload: Mapping[str, Any],
    *,
    repository_root: str | Path,
    generation_config: CounterfactualGenerationConfig,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
    worker_count: int,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
    scenario_loader: ScenarioLoader = load_av2_scenario,
    raw_loader: RawLoader = load_raw_shard_candidates,
    map_preparer: MapPreparer = prepare_map_geometry,
    candidate_validator: CandidateValidator | None = None,
) -> ParallelFilterResult:
    """Run individual gates by scenario and one parent diversity finalize."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")
    total_started = time.perf_counter()
    started = time.perf_counter()
    jobs = _build_jobs(workload, Path(repository_root).resolve(), generation_config)
    job_build_seconds = time.perf_counter() - started
    dependencies = (scenario_loader, raw_loader, map_preparer, candidate_validator)
    results, startup, execution = _execute(
        jobs,
        worker_count,
        _spawn_safe(filter_config),
        detection_config,
        dependencies,
        map_batch_size,
    )
    compact, validation_order = _restore(jobs, results)
    started = time.perf_counter()
    batch = finalize_candidate_validations(
        compact,
        filter_config=filter_config,
        filter_semantic_sha256=str(workload["filter_semantic_sha256"]),
    )
    finalize_seconds = time.perf_counter() - started
    if len(batch.decisions) != len(compact):
        raise ValueError("global finalization omitted a candidate")
    decision_sha = canonical_sha256(_decision_rows(batch))
    semantic_decision_sha = canonical_sha256(_semantic_decision_rows(batch))
    worker_pids = tuple(sorted({item.worker_pid for item in results}))
    rejections = Counter(
        item.metrics["first_failed_stage"] for item in batch.decisions if not item.accepted
    )
    total_seconds = time.perf_counter() - total_started
    timings = {
        "job_build_seconds": job_build_seconds,
        "worker_startup_seconds": startup,
        "worker_execution_seconds": execution,
        "stable_total_seconds": total_seconds - job_build_seconds - startup,
        "scenario_load_seconds": sum(item.scenario_load_seconds for item in results),
        "prepared_map_seconds": sum(item.prepared_map_seconds for item in results),
        "map_integrity_finalize_seconds": sum(
            item.map_integrity_finalize_seconds for item in results
        ),
        "raw_load_bind_seconds": sum(item.raw_load_bind_seconds for item in results),
        "individual_filter_seconds": sum(item.individual_filter_seconds for item in results),
        "global_diversity_seconds": float(
            batch.stage_elapsed_seconds[FilterStage.DIVERSITY.value]
        ),
        "global_finalize_seconds": finalize_seconds,
        "total_seconds": total_seconds,
    }
    timings["map_subsystem_seconds"] = (
        timings["prepared_map_seconds"]
        + float(batch.stage_elapsed_seconds[FilterStage.MAP.value])
        + timings["map_integrity_finalize_seconds"]
    )
    return ParallelFilterResult(
        batch=batch,
        timings=timings,
        requested_worker_count=worker_count,
        effective_worker_count=len(worker_pids),
        worker_pids=worker_pids,
        scenario_load_count=len(results),
        prepared_map_count=len(results),
        validation_order=validation_order,
        decision_sha256=decision_sha,
        semantic_decision_sha256=semantic_decision_sha,
        stage_rejection_counts={
            stage: rejections.get(stage, 0) for stage in batch.stage_execution_counts
        },
        map_batch_size=map_batch_size,
    )


__all__ = [
    "CompactValidationWire",
    "ParallelFilterResult",
    "ParallelFilterWorkerError",
    "run_parallel_filter_workload",
]

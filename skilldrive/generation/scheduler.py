"""Deterministic Pilot task plans and resumable scheduling state."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.generation.contracts import (
    GenerationTask,
    candidate_id,
    canonical_json_bytes,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    build_generation_task,
    latent_seed,
    paired_latent_seed,
    pilot_evaluation_arm,
    seed_record_id,
    select_pilot_records,
    semantic_generation_config_sha256,
)
from skilldrive.generation.storage import RawRecoveryScan, scan_raw_shards
from skilldrive.seeds.records import SeedRecord


TASK_PLAN_SCHEMA_VERSION = 1
TASK_PLAN_FILE_NAME = "task_plan.jsonl"
TASK_PLAN_SUMMARY_NAME = "task_plan.summary.json"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ARTIFACT_PATTERN = re.compile(
    r"^shard-(?P<index>\d+)\.(?:commit\.json|npz|meta\.jsonl\.gz)$"
)
_TASK_FIELDS = {
    "task_id",
    "task_index",
    "seed_record_id",
    "scenario_id",
    "skill_id",
    "target_track_id",
    "proposal_mode",
    "condition_skill_id",
    "candidate_budget",
    "checkpoint_sha256",
    "semantic_config_sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_sort_key(task: GenerationTask) -> tuple[str, str, str, str, str]:
    return (
        task.scenario_id,
        task.skill_id,
        task.seed_record_id,
        task.condition_skill_id,
        task.task_id,
    )


def _task_row(task: GenerationTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "seed_record_id": task.seed_record_id,
        "scenario_id": task.scenario_id,
        "skill_id": task.skill_id,
        "target_track_id": task.target_track_id,
        "proposal_mode": task.proposal_mode,
        "condition_skill_id": task.condition_skill_id,
        "candidate_budget": task.candidate_budget,
        "checkpoint_sha256": task.checkpoint_sha256,
        "semantic_config_sha256": task.semantic_config_sha256,
    }


def _task_from_row(value: Any) -> GenerationTask:
    if not isinstance(value, dict) or set(value) != _TASK_FIELDS:
        raise ValueError("task plan row has missing or unknown fields")
    return GenerationTask(status="pending", **value)


@dataclass(frozen=True)
class ScenarioTaskGroup:
    scenario_id: str
    tasks: tuple[GenerationTask, ...]


@dataclass(frozen=True)
class TaskPlan:
    """One immutable, scenario-grouped Pilot workload."""

    semantic_config_sha256: str
    execution_config_sha256: str
    base_seed: int
    per_skill: int
    candidate_budget: int
    tasks: tuple[GenerationTask, ...]

    def __post_init__(self) -> None:
        _sha256_text(self.semantic_config_sha256, "semantic_config_sha256")
        _sha256_text(self.execution_config_sha256, "execution_config_sha256")
        _nonnegative_integer(self.base_seed, "base_seed")
        _positive_integer(self.per_skill, "per_skill")
        _positive_integer(self.candidate_budget, "candidate_budget")
        if not self.tasks:
            raise ValueError("task plan must contain at least one task")
        if any(not isinstance(task, GenerationTask) for task in self.tasks):
            raise ValueError("task plan must contain only GenerationTask values")
        if [task.task_index for task in self.tasks] != list(range(len(self.tasks))):
            raise ValueError("task plan indices must be a contiguous zero-based sequence")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("task plan task IDs must be unique")
        if tuple(sorted(self.tasks, key=_task_sort_key)) != self.tasks:
            raise ValueError("task plan must be grouped and ordered by scenario")
        for task in self.tasks:
            if task.status != "pending":
                raise ValueError("task plan entries must start in pending state")
            if task.semantic_config_sha256 != self.semantic_config_sha256:
                raise ValueError("task semantic configuration differs from the plan")
            if task.candidate_budget != self.candidate_budget:
                raise ValueError("task candidate budget differs from the plan")

    @property
    def task_plan_id(self) -> str:
        """Semantic task identity; candidate-budget expansion leaves it unchanged."""

        return canonical_sha256(
            {
                "version": TASK_PLAN_SCHEMA_VERSION,
                "semantic_config_sha256": self.semantic_config_sha256,
                "task_ids": [task.task_id for task in self.tasks],
            }
        )

    @property
    def total_candidates(self) -> int:
        return len(self.tasks) * self.candidate_budget

    @property
    def scenario_groups(self) -> tuple[ScenarioTaskGroup, ...]:
        groups: list[ScenarioTaskGroup] = []
        start = 0
        while start < len(self.tasks):
            scenario_id = self.tasks[start].scenario_id
            end = start + 1
            while end < len(self.tasks) and self.tasks[end].scenario_id == scenario_id:
                end += 1
            groups.append(ScenarioTaskGroup(scenario_id, self.tasks[start:end]))
            start = end
        return tuple(groups)


@dataclass(frozen=True)
class TaskPlanArtifacts:
    task_plan_path: Path
    summary_path: Path
    task_plan_sha256: str
    summary_sha256: str


@dataclass(frozen=True)
class LoadedTaskPlan:
    plan: TaskPlan
    stored_execution_config_sha256: str
    current_execution_config_sha256: str

    @property
    def execution_config_changed(self) -> bool:
        return self.stored_execution_config_sha256 != self.current_execution_config_sha256


@dataclass(frozen=True)
class DurableTaskRecovery:
    raw_scan: RawRecoveryScan
    durable_task_ids: frozenset[str]
    partial_task_ids: frozenset[str]
    pending_task_ids: frozenset[str]
    durable_candidate_indices: Mapping[str, tuple[int, ...]]
    pending_candidate_indices: Mapping[str, tuple[int, ...]]
    extra_candidate_indices: Mapping[str, tuple[int, ...]]

    @property
    def durable_candidate_count(self) -> int:
        return sum(
            len(values) - len(self.extra_candidate_indices.get(task_id, ()))
            for task_id, values in self.durable_candidate_indices.items()
        )


@dataclass(frozen=True)
class PairedPilotRecovery:
    """Strict task-shard recovery for the paired 34-skill Pilot workload."""

    raw_scan: RawRecoveryScan
    durable_task_ids: frozenset[str]
    rebuild_task_ids: frozenset[str]
    missing_task_ids: frozenset[str]
    partial_task_ids: frozenset[str]
    invalid_task_ids: frozenset[str]
    orphaned_task_ids: frozenset[str]

    def __post_init__(self) -> None:
        classified = self.durable_task_ids | self.rebuild_task_ids
        if self.durable_task_ids & self.rebuild_task_ids:
            raise ValueError("Pilot recovery task classes must not overlap")
        if self.rebuild_task_ids != (
            self.missing_task_ids
            | self.partial_task_ids
            | self.invalid_task_ids
            | self.orphaned_task_ids
        ):
            raise ValueError("Pilot rebuild tasks differ from their recovery reasons")
        if not classified:
            raise ValueError("Pilot recovery must classify at least one task")

    @property
    def durable_candidate_count(self) -> int:
        return sum(
            shard.candidate_count
            for shard in self.raw_scan.valid_shards
            if shard.references[0].task_id in self.durable_task_ids
        )


@dataclass(frozen=True)
class RecoveryScanProgress:
    total_shards: int
    scanned_shards: int
    valid_shards: int
    invalid_shards: int
    orphaned_files: int
    durable_tasks: int
    durable_candidates: int

    def __post_init__(self) -> None:
        for name in (
            "total_shards",
            "scanned_shards",
            "valid_shards",
            "invalid_shards",
            "orphaned_files",
            "durable_tasks",
            "durable_candidates",
        ):
            _nonnegative_integer(getattr(self, name), name)
        if self.scanned_shards > self.total_shards:
            raise ValueError("scanned_shards cannot exceed total_shards")
        if self.valid_shards + self.invalid_shards > self.scanned_shards:
            raise ValueError("classified shards cannot exceed scanned_shards")

    @property
    def fraction(self) -> float:
        return 1.0 if self.total_shards == 0 else self.scanned_shards / self.total_shards


@dataclass(frozen=True)
class ProcessingProgress:
    total_tasks: int
    durable_tasks_at_start: int
    newly_completed_tasks: int
    in_flight_tasks: int
    durable_candidates_at_start: int
    newly_generated_candidates: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        for name in (
            "total_tasks",
            "durable_tasks_at_start",
            "newly_completed_tasks",
            "in_flight_tasks",
            "durable_candidates_at_start",
            "newly_generated_candidates",
        ):
            _nonnegative_integer(getattr(self, name), name)
        if self.durable_tasks_at_start + self.newly_completed_tasks > self.total_tasks:
            raise ValueError("completed tasks cannot exceed total_tasks")
        if self.in_flight_tasks > (
            self.total_tasks
            - self.durable_tasks_at_start
            - self.newly_completed_tasks
        ):
            raise ValueError("in_flight_tasks cannot exceed remaining tasks")
        if not isinstance(self.elapsed_seconds, (int, float)) or isinstance(
            self.elapsed_seconds,
            bool,
        ):
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if not math.isfinite(float(self.elapsed_seconds)) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")

    @property
    def completed_tasks(self) -> int:
        return self.durable_tasks_at_start + self.newly_completed_tasks

    @property
    def remaining_tasks(self) -> int:
        return self.total_tasks - self.completed_tasks

    @property
    def fraction(self) -> float:
        return 1.0 if self.total_tasks == 0 else self.completed_tasks / self.total_tasks


def build_pilot_task_plan(
    records: Iterable[SeedRecord],
    config: CounterfactualGenerationConfig,
    *,
    execution_config: Mapping[str, Any],
    per_skill: int | None = None,
    candidate_budget: int | None = None,
) -> TaskPlan:
    """Select per-skill records and group deterministic tasks by scenario."""

    selected_per_skill = (
        config.sampling.pilot_seed_records_per_skill
        if per_skill is None
        else _positive_integer(per_skill, "per_skill")
    )
    budget = (
        config.sampling.pilot_candidates_per_task
        if candidate_budget is None
        else _positive_integer(candidate_budget, "candidate_budget")
    )
    selected = select_pilot_records(
        records,
        formal_skill_ids=config.formal_skill_ids,
        per_skill=selected_per_skill,
        base_seed=config.sampling.base_seed,
    )
    ordered = sorted(
        selected,
        key=lambda record: (
            record.scenario_id,
            record.skill_id,
            seed_record_id(record),
        ),
    )
    tasks = tuple(
        build_generation_task(
            task_index=index,
            record=record,
            config=config,
            candidate_budget=budget,
        )
        for index, record in enumerate(ordered)
    )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("selected Pilot records produce duplicate task IDs")
    return TaskPlan(
        semantic_config_sha256=semantic_generation_config_sha256(config),
        execution_config_sha256=canonical_sha256(execution_config),
        base_seed=config.sampling.base_seed,
        per_skill=selected_per_skill,
        candidate_budget=budget,
        tasks=tasks,
    )


def build_paired_pilot_task_plan(
    records: Iterable[SeedRecord],
    config: CounterfactualGenerationConfig,
    *,
    execution_config: Mapping[str, Any],
    per_skill: int | None = None,
    candidate_budget: int | None = None,
    allow_missing_skills: bool = False,
) -> TaskPlan:
    """Build formal Pilot tasks plus paired ``<none>`` learned controls."""

    selected_per_skill = (
        config.sampling.pilot_seed_records_per_skill
        if per_skill is None
        else _positive_integer(per_skill, "per_skill")
    )
    budget = (
        config.sampling.pilot_candidates_per_task
        if candidate_budget is None
        else _positive_integer(candidate_budget, "candidate_budget")
    )
    record_values = tuple(records)
    selected_skill_ids = {record.skill_id for record in record_values}
    formal_skill_ids = (
        tuple(
            skill_id
            for skill_id in config.formal_skill_ids
            if skill_id in selected_skill_ids
        )
        if allow_missing_skills
        else config.formal_skill_ids
    )
    selected = select_pilot_records(
        record_values,
        formal_skill_ids=formal_skill_ids,
        per_skill=selected_per_skill,
        base_seed=config.sampling.base_seed,
    )
    provisional: list[GenerationTask] = []
    for record in selected:
        formal_task = build_generation_task(
            task_index=0,
            record=record,
            config=config,
            candidate_budget=budget,
        )
        arm = pilot_evaluation_arm(
            formal_task,
            none_skill_id=config.none_skill_id,
        )
        provisional.append(formal_task)
        if arm == "learned_conditioned":
            provisional.append(
                GenerationTask.create(
                    task_index=0,
                    seed_record_id=formal_task.seed_record_id,
                    scenario_id=formal_task.scenario_id,
                    skill_id=formal_task.skill_id,
                    target_track_id=formal_task.target_track_id,
                    proposal_mode=formal_task.proposal_mode,
                    condition_skill_id=config.none_skill_id,
                    candidate_budget=formal_task.candidate_budget,
                    checkpoint_sha256=formal_task.checkpoint_sha256,
                    semantic_config_sha256=formal_task.semantic_config_sha256,
                )
            )
        elif arm != "rule_guided_none":
            raise ValueError(f"unexpected formal Pilot task arm: {arm}")

    tasks = tuple(
        replace(task, task_index=index)
        for index, task in enumerate(sorted(provisional, key=_task_sort_key))
    )
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("selected paired Pilot records produce duplicate task IDs")
    return TaskPlan(
        semantic_config_sha256=semantic_generation_config_sha256(config),
        execution_config_sha256=canonical_sha256(execution_config),
        base_seed=config.sampling.base_seed,
        per_skill=selected_per_skill,
        candidate_budget=budget,
        tasks=tasks,
    )


def _summary(plan: TaskPlan, task_plan_path: Path) -> dict[str, Any]:
    by_skill = Counter(task.skill_id for task in plan.tasks)
    by_mode = Counter(task.proposal_mode for task in plan.tasks)
    return {
        "schema_version": TASK_PLAN_SCHEMA_VERSION,
        "kind": "pilot_task_plan_summary",
        "task_plan_id": plan.task_plan_id,
        "semantic_config_sha256": plan.semantic_config_sha256,
        "execution_config_sha256": plan.execution_config_sha256,
        "base_seed": plan.base_seed,
        "per_skill": plan.per_skill,
        "candidate_budget": plan.candidate_budget,
        "counts": {
            "tasks": len(plan.tasks),
            "scenarios": len(plan.scenario_groups),
            "candidates": plan.total_candidates,
            "by_skill": dict(sorted(by_skill.items())),
            "by_proposal_mode": dict(sorted(by_mode.items())),
        },
        "task_plan": {
            "path": task_plan_path.name,
            "size_bytes": task_plan_path.stat().st_size,
            "sha256": _sha256(task_plan_path),
        },
    }


def write_task_plan(directory: str | Path, plan: TaskPlan) -> TaskPlanArtifacts:
    """Atomically publish task_plan.jsonl followed by its summary contract."""

    root = Path(directory)
    task_plan_path = root / TASK_PLAN_FILE_NAME
    summary_path = root / TASK_PLAN_SUMMARY_NAME
    payload = b"".join(canonical_json_bytes(_task_row(task)) + b"\n" for task in plan.tasks)
    _atomic_write(task_plan_path, payload)
    _atomic_write(summary_path, canonical_json_bytes(_summary(plan, task_plan_path), indent=2))
    loaded = load_task_plan(
        root,
        expected_semantic_config_sha256=plan.semantic_config_sha256,
        current_execution_config_sha256=plan.execution_config_sha256,
    )
    if loaded.plan != plan:
        raise ValueError("written task plan differs after verification")
    return TaskPlanArtifacts(
        task_plan_path=task_plan_path,
        summary_path=summary_path,
        task_plan_sha256=_sha256(task_plan_path),
        summary_sha256=_sha256(summary_path),
    )


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read task plan summary {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("task plan summary must be a JSON object")
    return value


def load_task_plan(
    directory: str | Path,
    *,
    expected_semantic_config_sha256: str,
    current_execution_config_sha256: str,
) -> LoadedTaskPlan:
    """Load a plan, rejecting semantic drift but allowing execution changes."""

    root = Path(directory)
    summary_path = root / TASK_PLAN_SUMMARY_NAME
    summary = _read_summary(summary_path)
    if summary.get("schema_version") != TASK_PLAN_SCHEMA_VERSION:
        raise ValueError("task plan summary schema version is incompatible")
    if summary.get("kind") != "pilot_task_plan_summary":
        raise ValueError("task plan summary kind is invalid")
    expected_semantic = _sha256_text(
        expected_semantic_config_sha256,
        "expected_semantic_config_sha256",
    )
    stored_semantic = _sha256_text(
        summary.get("semantic_config_sha256"),
        "summary.semantic_config_sha256",
    )
    if stored_semantic != expected_semantic:
        raise ValueError(
            "task plan semantic configuration differs; create a new contract version"
        )
    stored_execution = _sha256_text(
        summary.get("execution_config_sha256"),
        "summary.execution_config_sha256",
    )
    current_execution = _sha256_text(
        current_execution_config_sha256,
        "current_execution_config_sha256",
    )
    descriptor = summary.get("task_plan")
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("task plan file descriptor is invalid")
    if descriptor["path"] != TASK_PLAN_FILE_NAME:
        raise ValueError("task plan summary references an unexpected file")
    task_plan_path = root / TASK_PLAN_FILE_NAME
    if task_plan_path.stat().st_size != descriptor["size_bytes"]:
        raise ValueError("task plan file size differs from its summary")
    if _sha256(task_plan_path) != descriptor["sha256"]:
        raise ValueError("task plan file SHA-256 differs from its summary")

    tasks: list[GenerationTask] = []
    for line_number, line in enumerate(
        task_plan_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"task plan contains a blank line at {line_number}")
        try:
            tasks.append(_task_from_row(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid task plan line {line_number}: {error}") from error
    counts = summary.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("task plan summary counts are invalid")
    plan = TaskPlan(
        semantic_config_sha256=stored_semantic,
        execution_config_sha256=stored_execution,
        base_seed=_nonnegative_integer(summary.get("base_seed"), "summary.base_seed"),
        per_skill=_positive_integer(summary.get("per_skill"), "summary.per_skill"),
        candidate_budget=_positive_integer(
            summary.get("candidate_budget"),
            "summary.candidate_budget",
        ),
        tasks=tuple(tasks),
    )
    expected_counts = _summary(plan, task_plan_path)["counts"]
    if counts != expected_counts:
        raise ValueError("task plan summary counts differ from the task rows")
    if summary.get("task_plan_id") != plan.task_plan_id:
        raise ValueError("task plan ID differs from the task rows")
    return LoadedTaskPlan(
        plan=plan,
        stored_execution_config_sha256=stored_execution,
        current_execution_config_sha256=current_execution,
    )


def recover_durable_tasks(
    plan: TaskPlan,
    raw_directory: str | Path,
) -> DurableTaskRecovery:
    """Recover complete and partial task candidate sets from valid raw sidecars."""

    raw_scan = scan_raw_shards(raw_directory)
    semantic_values = {
        shard.semantic_config_sha256 for shard in raw_scan.valid_shards
    }
    if semantic_values - {plan.semantic_config_sha256}:
        raise ValueError(
            "raw shards use a different semantic configuration; refusing mixed resume"
        )
    tasks = {task.task_id: task for task in plan.tasks}
    durable_indices: dict[str, set[int]] = {task_id: set() for task_id in tasks}
    for shard in raw_scan.valid_shards:
        for reference in shard.references:
            task = tasks.get(reference.task_id)
            if task is None:
                raise ValueError(
                    f"raw shard references a task outside the current plan: {reference.task_id}"
                )
            if reference.candidate_index in durable_indices[task.task_id]:
                raise ValueError(
                    "raw shards contain a duplicate task candidate index: "
                    f"{task.task_id}:{reference.candidate_index}"
                )
            expected_candidate_id = candidate_id(
                task_id=task.task_id,
                candidate_index=reference.candidate_index,
                latent_seed=latent_seed(
                    plan.base_seed,
                    task.task_id,
                    reference.candidate_index,
                ),
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=task.semantic_config_sha256,
            )
            if reference.candidate_id != expected_candidate_id:
                raise ValueError(
                    f"raw candidate differs from the current task plan: {reference.candidate_id}"
                )
            durable_indices[task.task_id].add(reference.candidate_index)

    durable_tasks: set[str] = set()
    partial_tasks: set[str] = set()
    pending_tasks: set[str] = set()
    pending_indices: dict[str, tuple[int, ...]] = {}
    extra_indices: dict[str, tuple[int, ...]] = {}
    durable_output: dict[str, tuple[int, ...]] = {}
    for task in plan.tasks:
        expected = set(range(task.candidate_budget))
        present = durable_indices[task.task_id]
        missing = tuple(sorted(expected - present))
        extra = tuple(sorted(present - expected))
        durable_output[task.task_id] = tuple(sorted(present))
        pending_indices[task.task_id] = missing
        extra_indices[task.task_id] = extra
        if not missing:
            durable_tasks.add(task.task_id)
        elif present:
            partial_tasks.add(task.task_id)
        else:
            pending_tasks.add(task.task_id)
    return DurableTaskRecovery(
        raw_scan=raw_scan,
        durable_task_ids=frozenset(durable_tasks),
        partial_task_ids=frozenset(partial_tasks),
        pending_task_ids=frozenset(pending_tasks),
        durable_candidate_indices=durable_output,
        pending_candidate_indices=pending_indices,
        extra_candidate_indices=extra_indices,
    )


def _task_id_for_shard_artifact(
    path: Path,
    tasks_by_index: Mapping[int, GenerationTask],
) -> str:
    match = _SHARD_ARTIFACT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected Pilot shard artifact name: {path.name}")
    shard_index = int(match.group("index"))
    task = tasks_by_index.get(shard_index)
    if task is None:
        raise ValueError(
            f"Pilot raw shard index is outside the current task plan: {shard_index}"
        )
    return task.task_id


def recover_paired_pilot_tasks(
    plan: TaskPlan,
    raw_directory: str | Path,
    *,
    latent_seed_source_tasks: Mapping[str, GenerationTask] | None = None,
) -> PairedPilotRecovery:
    """Recover only complete one-task shards under the paired Pilot seed contract.

    Missing, corrupt, orphaned, or valid-but-partial task shards are explicitly
    marked for whole-task rebuild. Unknown tasks, out-of-budget candidates, and
    candidates generated with a different latent contract are rejected.
    """

    raw_scan = scan_raw_shards(raw_directory)
    tasks_by_index = {task.task_index: task for task in plan.tasks}
    if len(tasks_by_index) != len(plan.tasks):
        raise ValueError("Pilot plan task indices must be unique")

    durable: set[str] = set()
    partial: set[str] = set()
    valid_indices: set[int] = set()
    for shard in raw_scan.valid_shards:
        task = tasks_by_index.get(shard.shard_index)
        if task is None:
            raise ValueError(
                "Pilot raw shard index is outside the current task plan: "
                f"{shard.shard_index}"
            )
        if shard.semantic_config_sha256 != plan.semantic_config_sha256:
            raise ValueError(
                "Pilot raw shard uses a different semantic configuration: "
                f"{shard.commit_path}"
            )
        if shard.execution_config_sha256 != plan.execution_config_sha256:
            raise ValueError(
                "Pilot raw shard uses a different execution configuration: "
                f"{shard.commit_path}"
            )
        valid_indices.add(shard.shard_index)
        references = shard.references
        if any(reference.task_id != task.task_id for reference in references):
            raise ValueError(
                "Pilot raw shard contains a task outside its deterministic shard: "
                f"{shard.commit_path}"
            )
        indices = [reference.candidate_index for reference in references]
        if len(indices) != len(set(indices)):
            raise ValueError(
                f"Pilot raw shard contains duplicate candidate indices: {shard.commit_path}"
            )
        expected_indices = set(range(task.candidate_budget))
        actual_indices = set(indices)
        extra_indices = sorted(actual_indices - expected_indices)
        if extra_indices:
            raise ValueError(
                "Pilot raw shard contains candidate indices outside the frozen budget: "
                f"task={task.task_id}, indices={extra_indices[:3]}"
            )
        expected_order = list(range(task.candidate_budget))
        if actual_indices == expected_indices and indices != expected_order:
            raise ValueError(
                "Pilot raw shard candidate order differs from the deterministic budget: "
                f"{shard.commit_path}"
            )
        seed_task = (
            task
            if latent_seed_source_tasks is None
            else latent_seed_source_tasks.get(task.task_id)
        )
        if seed_task is None:
            raise ValueError(
                "Pilot latent seed source is missing for task: "
                f"{task.task_id}"
            )
        for reference in references:
            expected_seed = paired_latent_seed(
                plan.base_seed,
                seed_task,
                reference.candidate_index,
            )
            if reference.latent_seed != expected_seed:
                raise ValueError(
                    "Pilot raw candidate uses a different paired latent seed: "
                    f"{reference.candidate_id}"
                )
            expected_candidate_id = candidate_id(
                task_id=task.task_id,
                candidate_index=reference.candidate_index,
                latent_seed=expected_seed,
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=task.semantic_config_sha256,
            )
            if reference.candidate_id != expected_candidate_id:
                raise ValueError(
                    "Pilot raw candidate differs from the paired task contract: "
                    f"{reference.candidate_id}"
                )
        if actual_indices == expected_indices:
            durable.add(task.task_id)
        else:
            partial.add(task.task_id)

    invalid = {
        _task_id_for_shard_artifact(issue.commit_path, tasks_by_index)
        for issue in raw_scan.invalid_shards
    }
    orphaned = {
        _task_id_for_shard_artifact(path, tasks_by_index)
        for path in raw_scan.orphaned_files
    }
    if durable & (partial | invalid | orphaned):
        raise ValueError("Pilot task has conflicting durable and rebuild artifacts")

    artifact_indices = valid_indices | {
        task.task_index
        for task in plan.tasks
        if task.task_id in invalid or task.task_id in orphaned
    }
    missing = {
        task.task_id
        for task in plan.tasks
        if task.task_index not in artifact_indices
    }
    rebuild = missing | partial | invalid | orphaned
    if (durable | rebuild) != {task.task_id for task in plan.tasks}:
        raise ValueError("Pilot recovery did not classify every task exactly once")
    return PairedPilotRecovery(
        raw_scan=raw_scan,
        durable_task_ids=frozenset(durable),
        rebuild_task_ids=frozenset(rebuild),
        missing_task_ids=frozenset(missing),
        partial_task_ids=frozenset(partial),
        invalid_task_ids=frozenset(invalid),
        orphaned_task_ids=frozenset(orphaned),
    )


def recovery_progress(recovery: DurableTaskRecovery) -> RecoveryScanProgress:
    scanned = len(recovery.raw_scan.valid_shards) + len(recovery.raw_scan.invalid_shards)
    return RecoveryScanProgress(
        total_shards=scanned,
        scanned_shards=scanned,
        valid_shards=len(recovery.raw_scan.valid_shards),
        invalid_shards=len(recovery.raw_scan.invalid_shards),
        orphaned_files=len(recovery.raw_scan.orphaned_files),
        durable_tasks=len(recovery.durable_task_ids),
        durable_candidates=recovery.durable_candidate_count,
    )


__all__ = [
    "TASK_PLAN_FILE_NAME",
    "TASK_PLAN_SCHEMA_VERSION",
    "TASK_PLAN_SUMMARY_NAME",
    "DurableTaskRecovery",
    "LoadedTaskPlan",
    "PairedPilotRecovery",
    "ProcessingProgress",
    "RecoveryScanProgress",
    "ScenarioTaskGroup",
    "TaskPlan",
    "TaskPlanArtifacts",
    "build_paired_pilot_task_plan",
    "build_pilot_task_plan",
    "load_task_plan",
    "recover_paired_pilot_tasks",
    "recover_durable_tasks",
    "recovery_progress",
    "write_task_plan",
]

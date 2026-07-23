"""Immutable coverage-first task plans for formal counterfactual generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.generation.contracts import (
    GenerationTask,
    canonical_json_bytes,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    build_generation_task,
    semantic_generation_config_sha256,
)
from skilldrive.seeds.records import SeedRecord, sort_seed_records


FORMAL_CONTRACT_VERSION = "formal_v1"
FORMAL_PLAN_SCHEMA_VERSION = 1
FORMAL_TASK_PLAN_FILE_NAME = "formal_task_plan.jsonl"
FORMAL_TASK_PLAN_SUMMARY_NAME = "formal_task_plan.summary.json"
FORMAL_TARGET_ACCEPTED_PER_SKILL = 300
FORMAL_RESUME_MODE = "auto"
FORMAL_PRODUCTION_EXPECTED_TASK_COUNT = 33_914
FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT = 5_000
FORMAL_PRODUCTION_SKILL_COUNT = 34

FormalPhase = Literal["coverage", "balance"]
FormalProfile = Literal["production", "fixture"]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PHASES: tuple[FormalPhase, ...] = ("coverage", "balance")
_PROFILES: tuple[FormalProfile, ...] = ("production", "fixture")
_REQUIRED_CONFIG_SHA256_KEYS = frozenset(
    {"generation_config", "filter_config", "performance_config"}
)
_REQUIRED_SOURCE_SHA256_KEYS = frozenset(
    {"generation_source", "filter_source"}
)
_TASK_ROW_FIELDS = {
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
    "phase",
    "phase_index",
    "shard_index",
}
_BINDING_FIELDS = {
    "contract_version",
    "profile",
    "checkpoint_sha256",
    "schema_sha256",
    "semantic_config_sha256",
    "filter_semantic_sha256",
    "generation_execution_sha256",
    "seed_manifest_sha256",
    "formal_train_boundary_audit_sha256",
    "config_sha256",
    "source_sha256",
    "base_seed",
    "candidate_budget",
    "tasks_per_shard",
    "expected_task_count",
    "expected_scenario_count",
    "expected_skill_ids",
    "target_accepted_per_skill",
    "resume_mode",
    "formal_train_only",
    "internal_validation_accessed",
    "final_validation_accessed",
}
_SUMMARY_FIELDS = {
    "schema_version",
    "kind",
    "formal_plan_id",
    "bindings",
    "counts",
    "task_plan",
}


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


def _hash_items(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
    name: str,
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{name} must map names to SHA-256 digests")
        key, digest = item
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key: {key}")
        seen.add(key)
        normalized.append((key, _sha256_text(digest, f"{name}.{key}")))
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def _skill_ids(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of unique skill IDs")
    normalized: list[str] = []
    for skill_id in value:
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        normalized.append(skill_id)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique skill IDs")
    return tuple(sorted(normalized))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_file(
    repository_root: str | Path,
    path: str | Path,
    name: str,
) -> tuple[Path, str]:
    root = Path(repository_root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{name} must remain inside repository_root") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must reference an existing file: {resolved}")
    return resolved, relative


def _verified_repository_sha256(
    repository_root: str | Path,
    path: str | Path,
    name: str,
    *,
    expected_sha256: str | None = None,
) -> str:
    resolved, _ = _repository_file(repository_root, path, name)
    digest = _sha256_file(resolved)
    if expected_sha256 is not None and digest != _sha256_text(
        expected_sha256,
        f"{name}.expected_sha256",
    ):
        raise ValueError(f"{name} SHA-256 differs from its frozen generation config")
    return digest


def _source_bundle_sha256(
    repository_root: str | Path,
    paths: Sequence[str | Path],
    name: str,
) -> str:
    if isinstance(paths, (str, bytes)) or not paths:
        raise ValueError(f"{name} must contain at least one source file")
    file_sha256: dict[str, str] = {}
    for index, path in enumerate(paths):
        resolved, relative = _repository_file(
            repository_root,
            path,
            f"{name}[{index}]",
        )
        if relative in file_sha256:
            raise ValueError(f"{name} contains duplicate source file: {relative}")
        file_sha256[relative] = _sha256_file(resolved)
    return canonical_sha256(
        {
            "version": 1,
            "files": dict(sorted(file_sha256.items())),
        }
    )


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


@dataclass(frozen=True)
class FormalPlanBindings:
    """All semantic inputs that make one ``formal_v1`` plan reusable or stale."""

    checkpoint_sha256: str
    schema_sha256: str
    semantic_config_sha256: str
    filter_semantic_sha256: str
    generation_execution_sha256: str
    seed_manifest_sha256: str
    formal_train_boundary_audit_sha256: str
    config_sha256: Mapping[str, str] | Sequence[tuple[str, str]]
    source_sha256: Mapping[str, str] | Sequence[tuple[str, str]]
    base_seed: int
    candidate_budget: int
    tasks_per_shard: int
    expected_task_count: int
    expected_scenario_count: int
    expected_skill_ids: Sequence[str]
    profile: FormalProfile = "production"
    target_accepted_per_skill: int = FORMAL_TARGET_ACCEPTED_PER_SKILL
    resume_mode: str = FORMAL_RESUME_MODE
    formal_train_only: bool = True
    internal_validation_accessed: bool = False
    final_validation_accessed: bool = False
    contract_version: str = FORMAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != FORMAL_CONTRACT_VERSION:
            raise ValueError(
                f"contract_version must remain {FORMAL_CONTRACT_VERSION!r}"
            )
        if self.profile not in _PROFILES:
            raise ValueError(f"unknown formal plan profile: {self.profile!r}")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256_text(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "schema_sha256",
            _sha256_text(self.schema_sha256, "schema_sha256"),
        )
        object.__setattr__(
            self,
            "semantic_config_sha256",
            _sha256_text(
                self.semantic_config_sha256,
                "semantic_config_sha256",
            ),
        )
        object.__setattr__(
            self,
            "filter_semantic_sha256",
            _sha256_text(
                self.filter_semantic_sha256,
                "filter_semantic_sha256",
            ),
        )
        object.__setattr__(
            self,
            "generation_execution_sha256",
            _sha256_text(
                self.generation_execution_sha256,
                "generation_execution_sha256",
            ),
        )
        object.__setattr__(
            self,
            "seed_manifest_sha256",
            _sha256_text(self.seed_manifest_sha256, "seed_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "formal_train_boundary_audit_sha256",
            _sha256_text(
                self.formal_train_boundary_audit_sha256,
                "formal_train_boundary_audit_sha256",
            ),
        )
        object.__setattr__(
            self,
            "config_sha256",
            _hash_items(self.config_sha256, "config_sha256"),
        )
        object.__setattr__(
            self,
            "source_sha256",
            _hash_items(self.source_sha256, "source_sha256"),
        )
        missing_configs = _REQUIRED_CONFIG_SHA256_KEYS - dict(self.config_sha256).keys()
        if missing_configs:
            raise ValueError(
                "config_sha256 is missing required hashes: "
                + ", ".join(sorted(missing_configs))
            )
        missing_sources = _REQUIRED_SOURCE_SHA256_KEYS - dict(self.source_sha256).keys()
        if missing_sources:
            raise ValueError(
                "source_sha256 is missing required hashes: "
                + ", ".join(sorted(missing_sources))
            )
        _nonnegative_integer(self.base_seed, "base_seed")
        _positive_integer(self.candidate_budget, "candidate_budget")
        _positive_integer(self.tasks_per_shard, "tasks_per_shard")
        _positive_integer(self.expected_task_count, "expected_task_count")
        _positive_integer(self.expected_scenario_count, "expected_scenario_count")
        if self.expected_scenario_count > self.expected_task_count:
            raise ValueError("expected_scenario_count cannot exceed expected_task_count")
        object.__setattr__(
            self,
            "expected_skill_ids",
            _skill_ids(self.expected_skill_ids, "expected_skill_ids"),
        )
        if self.profile == "production":
            if self.expected_task_count != FORMAL_PRODUCTION_EXPECTED_TASK_COUNT:
                raise ValueError(
                    "production profile requires exactly "
                    f"{FORMAL_PRODUCTION_EXPECTED_TASK_COUNT} formal tasks"
                )
            if (
                self.expected_scenario_count
                != FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT
            ):
                raise ValueError(
                    "production profile requires exactly "
                    f"{FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT} formal scenarios"
                )
            if len(self.expected_skill_ids) != FORMAL_PRODUCTION_SKILL_COUNT:
                raise ValueError(
                    "production profile requires exactly "
                    f"{FORMAL_PRODUCTION_SKILL_COUNT} formal skills"
                )
        if self.target_accepted_per_skill != FORMAL_TARGET_ACCEPTED_PER_SKILL:
            raise ValueError(
                "formal_v1 target_accepted_per_skill must remain "
                f"{FORMAL_TARGET_ACCEPTED_PER_SKILL}"
            )
        if self.resume_mode != FORMAL_RESUME_MODE:
            raise ValueError(f"formal_v1 resume_mode must remain {FORMAL_RESUME_MODE!r}")
        if not isinstance(self.formal_train_only, bool):
            raise ValueError("formal_train_only must be a boolean")
        if not self.formal_train_only:
            raise ValueError("formal_v1 requires formal_train_only=true")
        for name in (
            "internal_validation_accessed",
            "final_validation_accessed",
        ):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            if value:
                raise ValueError(f"formal_v1 requires {name}=false")

    @classmethod
    def from_generation_config(
        cls,
        config: CounterfactualGenerationConfig,
        *,
        repository_root: str | Path,
        generation_config_path: str | Path,
        filter_config_path: str | Path,
        performance_config_path: str | Path,
        detection_config_path: str | Path,
        filter_additional_paths: Sequence[str | Path],
        generation_source_paths: Sequence[str | Path],
        filter_source_paths: Sequence[str | Path],
        execution_config: Mapping[str, Any],
        tasks_per_shard: int,
    ) -> "FormalPlanBindings":
        """Compute and freeze the production model/config/source identity."""

        seed_manifest_sha256 = _verified_repository_sha256(
            repository_root,
            config.inputs.seed_manifest,
            "formal seed manifest",
            expected_sha256=config.inputs.seed_manifest_sha256,
        )
        boundary_audit_sha256 = _verified_repository_sha256(
            repository_root,
            config.inputs.leakage_audit,
            "Formal Train boundary audit",
            expected_sha256=config.inputs.leakage_audit_sha256,
        )
        config_sha256 = {
            "generation_config": _verified_repository_sha256(
                repository_root,
                generation_config_path,
                "generation config",
            ),
            "filter_config": _verified_repository_sha256(
                repository_root,
                filter_config_path,
                "filter config",
            ),
            "performance_config": _verified_repository_sha256(
                repository_root,
                performance_config_path,
                "performance config",
            ),
        }
        source_sha256 = {
            "generation_source": _source_bundle_sha256(
                repository_root,
                generation_source_paths,
                "generation_source_paths",
            ),
            "filter_source": _source_bundle_sha256(
                repository_root,
                filter_source_paths,
                "filter_source_paths",
            ),
        }
        filter_semantic_sha256 = build_filter_semantic_fingerprint(
            repository_root=repository_root,
            generation_config_path=generation_config_path,
            filter_config_path=filter_config_path,
            detection_config_path=detection_config_path,
            additional_paths=filter_additional_paths,
        ).semantic_sha256
        generation_execution_sha256 = canonical_sha256(execution_config)

        return cls(
            checkpoint_sha256=config.active_checkpoint.sha256,
            schema_sha256=config.active_checkpoint.schema_sha256,
            semantic_config_sha256=semantic_generation_config_sha256(config),
            filter_semantic_sha256=filter_semantic_sha256,
            generation_execution_sha256=generation_execution_sha256,
            seed_manifest_sha256=seed_manifest_sha256,
            formal_train_boundary_audit_sha256=boundary_audit_sha256,
            config_sha256=config_sha256,
            source_sha256=source_sha256,
            base_seed=config.sampling.base_seed,
            candidate_budget=config.sampling.formal_candidates_per_task,
            tasks_per_shard=tasks_per_shard,
            expected_task_count=FORMAL_PRODUCTION_EXPECTED_TASK_COUNT,
            expected_scenario_count=FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT,
            expected_skill_ids=config.formal_skill_ids,
            profile="production",
        )

    @classmethod
    def for_fixture(
        cls,
        config: CounterfactualGenerationConfig,
        *,
        config_sha256: Mapping[str, str],
        source_sha256: Mapping[str, str],
        tasks_per_shard: int,
        expected_task_count: int,
        expected_scenario_count: int,
        expected_skill_ids: Sequence[str],
        filter_semantic_sha256: str,
        generation_execution_sha256: str,
    ) -> "FormalPlanBindings":
        """Build an explicit small-scope binding for isolated contract tests."""

        return cls(
            checkpoint_sha256=config.active_checkpoint.sha256,
            schema_sha256=config.active_checkpoint.schema_sha256,
            semantic_config_sha256=semantic_generation_config_sha256(config),
            filter_semantic_sha256=filter_semantic_sha256,
            generation_execution_sha256=generation_execution_sha256,
            seed_manifest_sha256=config.inputs.seed_manifest_sha256,
            formal_train_boundary_audit_sha256=config.inputs.leakage_audit_sha256,
            config_sha256=config_sha256,
            source_sha256=source_sha256,
            base_seed=config.sampling.base_seed,
            candidate_budget=config.sampling.formal_candidates_per_task,
            tasks_per_shard=tasks_per_shard,
            expected_task_count=expected_task_count,
            expected_scenario_count=expected_scenario_count,
            expected_skill_ids=expected_skill_ids,
            profile="fixture",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "profile": self.profile,
            "checkpoint_sha256": self.checkpoint_sha256,
            "schema_sha256": self.schema_sha256,
            "semantic_config_sha256": self.semantic_config_sha256,
            "filter_semantic_sha256": self.filter_semantic_sha256,
            "generation_execution_sha256": self.generation_execution_sha256,
            "seed_manifest_sha256": self.seed_manifest_sha256,
            "formal_train_boundary_audit_sha256": (
                self.formal_train_boundary_audit_sha256
            ),
            "config_sha256": dict(self.config_sha256),
            "source_sha256": dict(self.source_sha256),
            "base_seed": self.base_seed,
            "candidate_budget": self.candidate_budget,
            "tasks_per_shard": self.tasks_per_shard,
            "expected_task_count": self.expected_task_count,
            "expected_scenario_count": self.expected_scenario_count,
            "expected_skill_ids": list(self.expected_skill_ids),
            "target_accepted_per_skill": self.target_accepted_per_skill,
            "resume_mode": self.resume_mode,
            "formal_train_only": self.formal_train_only,
            "internal_validation_accessed": self.internal_validation_accessed,
            "final_validation_accessed": self.final_validation_accessed,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalPlanBindings":
        if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
            raise ValueError("formal plan bindings have missing or unknown fields")
        return cls(**value)

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class FormalTask:
    """One generation task plus immutable formal scheduling coordinates."""

    task_id: str
    task_index: int
    seed_record_id: str
    scenario_id: str
    skill_id: str
    target_track_id: str
    proposal_mode: str
    condition_skill_id: str
    candidate_budget: int
    checkpoint_sha256: str
    semantic_config_sha256: str
    phase: FormalPhase
    phase_index: int
    shard_index: int

    def __post_init__(self) -> None:
        self.as_generation_task()
        if self.phase not in _PHASES:
            raise ValueError(f"unknown formal task phase: {self.phase!r}")
        _nonnegative_integer(self.phase_index, "phase_index")
        _nonnegative_integer(self.shard_index, "shard_index")

    @classmethod
    def from_generation_task(
        cls,
        task: GenerationTask,
        *,
        phase: FormalPhase,
        phase_index: int,
        shard_index: int,
    ) -> "FormalTask":
        return cls(
            task_id=task.task_id,
            task_index=task.task_index,
            seed_record_id=task.seed_record_id,
            scenario_id=task.scenario_id,
            skill_id=task.skill_id,
            target_track_id=task.target_track_id,
            proposal_mode=task.proposal_mode,
            condition_skill_id=task.condition_skill_id,
            candidate_budget=task.candidate_budget,
            checkpoint_sha256=task.checkpoint_sha256,
            semantic_config_sha256=task.semantic_config_sha256,
            phase=phase,
            phase_index=phase_index,
            shard_index=shard_index,
        )

    def as_generation_task(self) -> GenerationTask:
        return GenerationTask(
            task_id=self.task_id,
            task_index=self.task_index,
            seed_record_id=self.seed_record_id,
            scenario_id=self.scenario_id,
            skill_id=self.skill_id,
            target_track_id=self.target_track_id,
            proposal_mode=self.proposal_mode,
            condition_skill_id=self.condition_skill_id,
            candidate_budget=self.candidate_budget,
            checkpoint_sha256=self.checkpoint_sha256,
            semantic_config_sha256=self.semantic_config_sha256,
            status="pending",
        )


def _selection_hash(
    *,
    purpose: str,
    base_seed: int,
    task: GenerationTask,
) -> str:
    return canonical_sha256(
        {
            "contract_version": FORMAL_CONTRACT_VERSION,
            "purpose": purpose,
            "base_seed": base_seed,
            "scenario_id": task.scenario_id,
            "skill_id": task.skill_id,
            "task_id": task.task_id,
        }
    )


def _coverage_order_hash(*, base_seed: int, scenario_id: str) -> str:
    return canonical_sha256(
        {
            "contract_version": FORMAL_CONTRACT_VERSION,
            "purpose": "coverage_order",
            "base_seed": base_seed,
            "scenario_id": scenario_id,
        }
    )


def _ordered_phase_tasks(
    tasks: Iterable[GenerationTask],
    *,
    base_seed: int,
) -> tuple[tuple[GenerationTask, ...], tuple[GenerationTask, ...]]:
    by_scenario: dict[str, list[GenerationTask]] = defaultdict(list)
    all_tasks = tuple(tasks)
    for task in all_tasks:
        by_scenario[task.scenario_id].append(task)

    coverage = tuple(
        sorted(
            (
                min(
                    scenario_tasks,
                    key=lambda task: (
                        _selection_hash(
                            purpose="coverage_choice",
                            base_seed=base_seed,
                            task=task,
                        ),
                        task.task_id,
                    ),
                )
                for scenario_tasks in by_scenario.values()
            ),
            key=lambda task: (
                _coverage_order_hash(
                    base_seed=base_seed,
                    scenario_id=task.scenario_id,
                ),
                task.scenario_id,
                task.task_id,
            ),
        )
    )
    coverage_ids = {task.task_id for task in coverage}

    by_skill: dict[str, list[GenerationTask]] = defaultdict(list)
    for task in all_tasks:
        if task.task_id not in coverage_ids:
            by_skill[task.skill_id].append(task)
    for skill_id, skill_tasks in by_skill.items():
        by_skill[skill_id] = sorted(
            skill_tasks,
            key=lambda task: (
                _selection_hash(
                    purpose="balance_order",
                    base_seed=base_seed,
                    task=task,
                ),
                task.scenario_id,
                task.task_id,
            ),
        )

    balance: list[GenerationTask] = []
    skills = sorted(by_skill)
    round_index = 0
    while True:
        added = False
        for skill_id in skills:
            skill_tasks = by_skill[skill_id]
            if round_index < len(skill_tasks):
                balance.append(skill_tasks[round_index])
                added = True
        if not added:
            break
        round_index += 1
    return coverage, tuple(balance)


@dataclass(frozen=True)
class FormalTaskPlan:
    """Complete, coverage-first and semantically immutable formal workload."""

    bindings: FormalPlanBindings
    tasks: tuple[FormalTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, FormalPlanBindings):
            raise ValueError("bindings must be FormalPlanBindings")
        if not self.tasks:
            raise ValueError("formal task plan must contain at least one task")
        if any(not isinstance(task, FormalTask) for task in self.tasks):
            raise ValueError("formal task plan must contain only FormalTask values")
        if [task.task_index for task in self.tasks] != list(range(len(self.tasks))):
            raise ValueError("formal task indices must be contiguous and zero-based")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("formal task IDs must be unique")
        if len({task.seed_record_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("formal seed labels must be unique")
        if len(self.tasks) != self.bindings.expected_task_count:
            raise ValueError(
                "formal task count differs from plan bindings: "
                f"{len(self.tasks)} != {self.bindings.expected_task_count}"
            )
        actual_scenarios = {task.scenario_id for task in self.tasks}
        if len(actual_scenarios) != self.bindings.expected_scenario_count:
            raise ValueError(
                "formal scenario count differs from plan bindings: "
                f"{len(actual_scenarios)} != "
                f"{self.bindings.expected_scenario_count}"
            )
        actual_skill_ids = tuple(sorted({task.skill_id for task in self.tasks}))
        if actual_skill_ids != self.bindings.expected_skill_ids:
            raise ValueError("formal task skill set differs from plan bindings")

        generation_tasks = tuple(task.as_generation_task() for task in self.tasks)
        for task in generation_tasks:
            if task.checkpoint_sha256 != self.bindings.checkpoint_sha256:
                raise ValueError("formal task checkpoint differs from plan bindings")
            if task.semantic_config_sha256 != self.bindings.semantic_config_sha256:
                raise ValueError("formal task semantic config differs from plan bindings")
            if task.candidate_budget != self.bindings.candidate_budget:
                raise ValueError("formal task candidate budget differs from plan bindings")

        expected_coverage, expected_balance = _ordered_phase_tasks(
            generation_tasks,
            base_seed=self.bindings.base_seed,
        )
        expected_ids = tuple(
            task.task_id for task in (*expected_coverage, *expected_balance)
        )
        if tuple(task.task_id for task in self.tasks) != expected_ids:
            raise ValueError("formal task order differs from coverage-first balance contract")

        coverage_count = len(expected_coverage)
        for task_index, task in enumerate(self.tasks):
            expected_phase: FormalPhase = (
                "coverage" if task_index < coverage_count else "balance"
            )
            expected_phase_index = (
                task_index if expected_phase == "coverage" else task_index - coverage_count
            )
            if task.phase != expected_phase or task.phase_index != expected_phase_index:
                raise ValueError("formal task phase coordinates differ from plan order")
            if task.shard_index != task_index // self.bindings.tasks_per_shard:
                raise ValueError("formal task shard index differs from tasks_per_shard")

        coverage_scenarios = {
            task.scenario_id for task in self.tasks if task.phase == "coverage"
        }
        all_scenarios = actual_scenarios
        if coverage_scenarios != all_scenarios or coverage_count != len(all_scenarios):
            raise ValueError("coverage phase must contain exactly one task per scenario")
        if coverage_count != self.bindings.expected_scenario_count:
            raise ValueError("coverage task count differs from expected scenario count")

    @property
    def formal_plan_id(self) -> str:
        return canonical_sha256(
            {
                "schema_version": FORMAL_PLAN_SCHEMA_VERSION,
                "bindings_sha256": self.bindings.semantic_sha256,
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "phase": task.phase,
                        "phase_index": task.phase_index,
                        "shard_index": task.shard_index,
                    }
                    for task in self.tasks
                ],
            }
        )

    @property
    def total_candidates(self) -> int:
        return len(self.tasks) * self.bindings.candidate_budget

    @property
    def scenario_count(self) -> int:
        return len({task.scenario_id for task in self.tasks})

    @property
    def shard_count(self) -> int:
        return self.tasks[-1].shard_index + 1


@dataclass(frozen=True)
class FormalTaskPlanArtifacts:
    task_plan_path: Path
    summary_path: Path
    task_plan_sha256: str
    summary_sha256: str


def _validate_bindings_against_generation_config(
    bindings: FormalPlanBindings,
    config: CounterfactualGenerationConfig,
) -> None:
    if bindings.checkpoint_sha256 != config.active_checkpoint.sha256:
        raise ValueError("formal bindings checkpoint differs from generation config")
    if bindings.schema_sha256 != config.active_checkpoint.schema_sha256:
        raise ValueError("formal bindings schema differs from generation config")
    if bindings.semantic_config_sha256 != semantic_generation_config_sha256(config):
        raise ValueError("formal bindings semantic config differs from generation config")
    if bindings.seed_manifest_sha256 != config.inputs.seed_manifest_sha256:
        raise ValueError("formal bindings seed manifest differs from generation config")
    if (
        bindings.formal_train_boundary_audit_sha256
        != config.inputs.leakage_audit_sha256
    ):
        raise ValueError(
            "formal bindings Formal Train boundary audit differs from generation config"
        )
    if bindings.base_seed != config.sampling.base_seed:
        raise ValueError("formal bindings base seed differs from generation config")
    if bindings.candidate_budget != config.sampling.formal_candidates_per_task:
        raise ValueError("formal bindings candidate budget differs from generation config")
    if bindings.expected_skill_ids != _skill_ids(
        config.formal_skill_ids,
        "config.formal_skill_ids",
    ):
        raise ValueError("formal bindings expected skill IDs differ from generation config")


def build_formal_task_plan(
    records: Iterable[SeedRecord],
    config: CounterfactualGenerationConfig,
    *,
    bindings: FormalPlanBindings,
) -> FormalTaskPlan:
    """Build every unique formal seed label into a deterministic two-phase plan."""

    _validate_bindings_against_generation_config(bindings, config)

    ordered_records = sort_seed_records(records)
    if not ordered_records:
        raise ValueError("formal seed labels must not be empty")
    provisional = tuple(
        build_generation_task(
            task_index=0,
            record=record,
            config=config,
            candidate_budget=bindings.candidate_budget,
        )
        for record in ordered_records
    )
    if len({task.task_id for task in provisional}) != len(provisional):
        raise ValueError("formal seed labels produce duplicate generation task IDs")

    coverage, balance = _ordered_phase_tasks(
        provisional,
        base_seed=bindings.base_seed,
    )
    formal_tasks: list[FormalTask] = []
    for task_index, task in enumerate((*coverage, *balance)):
        phase: FormalPhase = "coverage" if task_index < len(coverage) else "balance"
        phase_index = task_index if phase == "coverage" else task_index - len(coverage)
        indexed = replace(task, task_index=task_index)
        formal_tasks.append(
            FormalTask.from_generation_task(
                indexed,
                phase=phase,
                phase_index=phase_index,
                shard_index=task_index // bindings.tasks_per_shard,
            )
        )
    return FormalTaskPlan(bindings=bindings, tasks=tuple(formal_tasks))


def _task_row(task: FormalTask) -> dict[str, Any]:
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
        "phase": task.phase,
        "phase_index": task.phase_index,
        "shard_index": task.shard_index,
    }


def _task_from_row(value: Any) -> FormalTask:
    if not isinstance(value, dict) or set(value) != _TASK_ROW_FIELDS:
        raise ValueError("formal task row has missing or unknown fields")
    return FormalTask(**value)


def _counts(plan: FormalTaskPlan) -> dict[str, Any]:
    by_skill = Counter(task.skill_id for task in plan.tasks)
    by_phase = Counter(task.phase for task in plan.tasks)
    return {
        "tasks": len(plan.tasks),
        "scenarios": plan.scenario_count,
        "candidates": plan.total_candidates,
        "shards": plan.shard_count,
        "by_phase": {phase: by_phase.get(phase, 0) for phase in _PHASES},
        "by_skill": dict(sorted(by_skill.items())),
    }


def _summary(
    plan: FormalTaskPlan,
    *,
    task_plan_size: int,
    task_plan_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_PLAN_SCHEMA_VERSION,
        "kind": "formal_task_plan_summary",
        "formal_plan_id": plan.formal_plan_id,
        "bindings": plan.bindings.to_dict(),
        "counts": _counts(plan),
        "task_plan": {
            "path": FORMAL_TASK_PLAN_FILE_NAME,
            "size_bytes": task_plan_size,
            "sha256": task_plan_sha256,
        },
    }


def write_formal_task_plan(
    directory: str | Path,
    plan: FormalTaskPlan,
    *,
    config: CounterfactualGenerationConfig,
) -> FormalTaskPlanArtifacts:
    """Atomically commit an immutable JSONL plan and idempotently resume publication."""

    _validate_bindings_against_generation_config(plan.bindings, config)
    root = Path(directory)
    task_plan_path = root / FORMAL_TASK_PLAN_FILE_NAME
    summary_path = root / FORMAL_TASK_PLAN_SUMMARY_NAME
    task_payload = b"".join(
        canonical_json_bytes(_task_row(task)) + b"\n" for task in plan.tasks
    )
    task_sha256 = _sha256_bytes(task_payload)
    summary_payload = canonical_json_bytes(
        _summary(
            plan,
            task_plan_size=len(task_payload),
            task_plan_sha256=task_sha256,
        ),
        indent=2,
    )

    if summary_path.exists():
        loaded = load_formal_task_plan(
            root,
            expected_bindings=plan.bindings,
            config=config,
        )
        if loaded != plan:
            raise ValueError("immutable formal task plan already contains different tasks")
    else:
        if task_plan_path.exists():
            if task_plan_path.read_bytes() != task_payload:
                raise ValueError(
                    "uncommitted formal task plan differs from the requested immutable plan"
                )
        else:
            _atomic_write(task_plan_path, task_payload)
        _atomic_write(summary_path, summary_payload)

    loaded = load_formal_task_plan(
        root,
        expected_bindings=plan.bindings,
        config=config,
    )
    if loaded != plan:
        raise ValueError("written formal task plan differs after verification")
    return FormalTaskPlanArtifacts(
        task_plan_path=task_plan_path,
        summary_path=summary_path,
        task_plan_sha256=_sha256_file(task_plan_path),
        summary_sha256=_sha256_file(summary_path),
    )


def _read_summary(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read formal task plan summary {path}: {error}") from error
    if not isinstance(value, dict) or set(value) != _SUMMARY_FIELDS:
        raise ValueError("formal task plan summary has missing or unknown fields")
    return value


def load_formal_task_plan(
    directory: str | Path,
    *,
    expected_bindings: FormalPlanBindings,
    config: CounterfactualGenerationConfig,
) -> FormalTaskPlan:
    """Load a committed plan and reject any task, artifact or semantic drift."""

    if not isinstance(expected_bindings, FormalPlanBindings):
        raise ValueError("expected_bindings must be FormalPlanBindings")
    _validate_bindings_against_generation_config(expected_bindings, config)
    root = Path(directory)
    summary = _read_summary(root / FORMAL_TASK_PLAN_SUMMARY_NAME)
    if summary.get("schema_version") != FORMAL_PLAN_SCHEMA_VERSION:
        raise ValueError("formal task plan schema version is incompatible")
    if summary.get("kind") != "formal_task_plan_summary":
        raise ValueError("formal task plan summary kind is invalid")

    stored_bindings = FormalPlanBindings.from_dict(summary.get("bindings"))
    if stored_bindings != expected_bindings:
        raise ValueError(
            "formal task plan semantic drift: stored bindings differ from expected bindings"
        )

    descriptor = summary.get("task_plan")
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("formal task plan file descriptor is invalid")
    if descriptor["path"] != FORMAL_TASK_PLAN_FILE_NAME:
        raise ValueError("formal task plan summary references an unexpected file")
    task_plan_path = root / FORMAL_TASK_PLAN_FILE_NAME
    try:
        size = task_plan_path.stat().st_size
    except OSError as error:
        raise ValueError(
            f"failed to inspect formal task plan {task_plan_path}: {error}"
        ) from error
    if size != descriptor["size_bytes"]:
        raise ValueError("formal task plan file size differs from its summary")
    if _sha256_file(task_plan_path) != _sha256_text(
        descriptor["sha256"],
        "task_plan.sha256",
    ):
        raise ValueError("formal task plan file SHA-256 differs from its summary")

    tasks: list[FormalTask] = []
    try:
        lines = task_plan_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"failed to read formal task plan {task_plan_path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"formal task plan contains a blank line at {line_number}")
        try:
            tasks.append(_task_from_row(json.loads(line)))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid formal task plan line {line_number}: {error}") from error

    plan = FormalTaskPlan(bindings=stored_bindings, tasks=tuple(tasks))
    if summary.get("counts") != _counts(plan):
        raise ValueError("formal task plan summary counts differ from task rows")
    if summary.get("formal_plan_id") != plan.formal_plan_id:
        raise ValueError("formal task plan ID differs from task rows")
    return plan


__all__ = [
    "FORMAL_CONTRACT_VERSION",
    "FORMAL_PLAN_SCHEMA_VERSION",
    "FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT",
    "FORMAL_PRODUCTION_EXPECTED_TASK_COUNT",
    "FORMAL_PRODUCTION_SKILL_COUNT",
    "FORMAL_RESUME_MODE",
    "FORMAL_TARGET_ACCEPTED_PER_SKILL",
    "FORMAL_TASK_PLAN_FILE_NAME",
    "FORMAL_TASK_PLAN_SUMMARY_NAME",
    "FormalPhase",
    "FormalProfile",
    "FormalPlanBindings",
    "FormalTask",
    "FormalTaskPlan",
    "FormalTaskPlanArtifacts",
    "build_formal_task_plan",
    "load_formal_task_plan",
    "write_formal_task_plan",
]

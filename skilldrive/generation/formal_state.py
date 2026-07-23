"""Sharded, hash-bound resume state for formal counterfactual generation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping, Sequence

from skilldrive.generation.contracts import (
    FilterRejection,
    candidate_id,
    canonical_json_bytes,
    canonical_sha256,
    filter_evaluation_id,
)
from skilldrive.filtering.contracts import FilterStage
from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
from skilldrive.generation.formal import (
    FORMAL_RESUME_MODE,
    FormalTask,
    FormalTaskPlan,
)
from skilldrive.generation.planning import latent_seed
from skilldrive.generation.storage import (
    FILTER_INDEX_SCHEMA_VERSION,
    RawCandidateReference,
    RawRecoveryIssue,
    RawShardCommit,
    RawShardError,
    scan_raw_shards,
    verify_raw_shard,
)


FORMAL_STATE_SCHEMA_VERSION = 1
FORMAL_STATE_SUMMARY_NAME = "formal_state.summary.json"
FORMAL_STATE_DIRECTORY_NAME = "state"
FORMAL_FAILURE_DIRECTORY_NAME = "failures"
FORMAL_INVALID_DIRECTORY_NAME = "invalid-generation"
FORMAL_PROGRESS_FILE_NAME = "progress.json"
FORMAL_TASK_STATE_STATUSES = (
    "pending",
    "generated",
    "filtered",
    "accepted",
    "rejected",
    "failed",
)
FormalTaskStateStatus = Literal[
    "pending",
    "generated",
    "filtered",
    "accepted",
    "rejected",
    "failed",
]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_STATE_COMMIT_PATTERN = re.compile(r"^shard-(?P<index>\d+)\.commit\.json$")
_RAW_ARTIFACT_PATTERN = re.compile(
    r"^shard-(?P<index>\d+)\.(?:commit\.json|npz|meta\.jsonl\.gz)$"
)
_INVALID_COMMIT_PATTERN = re.compile(
    r"^task-(?P<task>\d+)-candidate-(?P<candidate>\d+)\.commit\.json$"
)
_SUMMARY_FIELDS = {
    "schema_version",
    "kind",
    "bindings",
    "counts",
    "state_directory",
}
_STATE_COMMIT_FIELDS = {
    "schema_version",
    "kind",
    "formal_plan_id",
    "task_plan_sha256",
    "bindings_sha256",
    "shard_index",
    "tasks",
    "state_sha256",
}
_TASK_STATE_FIELDS = {
    "task_id",
    "task_index",
    "skill_id",
    "status",
    "raw",
    "invalid_candidates",
    "filter",
    "failure",
}
_RAW_REFERENCE_FIELDS = {
    "commit_path",
    "commit_sha256",
    "shard_index",
    "execution_config_sha256",
    "arrays_path",
    "arrays_sha256",
    "metadata_path",
    "metadata_sha256",
    "candidate_indices",
    "candidate_ids_sha256",
}
_FILTER_REFERENCE_FIELDS = {
    "commit_path",
    "commit_sha256",
    "formal_plan_id",
    "task_plan_sha256",
    "filter_config_sha256",
    "filter_contract_version",
    "raw_commit_sha256",
    "decision_sha256",
    "candidate_count",
    "accepted_count",
    "rejected_count",
    "stage_rejection_counts",
}
_INVALID_REFERENCE_FIELDS = {
    "candidate_index",
    "latent_seed",
    "reason_code",
    "sidecar_path",
    "sidecar_sha256",
}
_INVALID_SIDECAR_FIELDS = {
    "schema_version",
    "kind",
    "formal_plan_id",
    "task_plan_sha256",
    "bindings_sha256",
    "task_id",
    "task_index",
    "skill_id",
    "candidate_index",
    "latent_seed",
    "reason_code",
    "message",
}
_FAILURE_REFERENCE_FIELDS = {
    "sidecar_path",
    "sidecar_sha256",
    "stage",
    "retryable",
    "reason_code",
    "attempt",
}
_FAILURE_SIDECAR_FIELDS = {
    "schema_version",
    "kind",
    "formal_plan_id",
    "task_plan_sha256",
    "bindings_sha256",
    "task_id",
    "task_index",
    "skill_id",
    "stage",
    "retryable",
    "reason_code",
    "message",
    "attempt",
}
_STATE_PHASE = {
    "pending": 0,
    "generated": 1,
    "filtered": 2,
    "accepted": 3,
    "rejected": 3,
}


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and nonnegative")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _relative_artifact(path: str | Path, root: str | Path) -> str:
    artifact = Path(path).resolve()
    base = Path(root).resolve()
    try:
        return artifact.relative_to(base).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact must be inside the formal run directory: {path}") from error


def _artifact_path(root: Path, relative: Any, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} must be a safe relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"{name} must be a safe relative path")
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes the formal run directory") from error
    return path


def _count_items(
    value: Mapping[str, int] | Sequence[tuple[str, int]],
    name: str,
) -> tuple[tuple[str, int], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{name} must map names to counts")
        key, count = item
        key = _required_text(key, f"{name} key")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key: {key}")
        seen.add(key)
        normalized.append((key, _nonnegative_integer(count, f"{name}.{key}")))
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class FormalStateBindings:
    """Semantic identity shared by every state shard and failure sidecar."""

    formal_plan_id: str
    task_plan_sha256: str
    plan_bindings_sha256: str
    checkpoint_sha256: str
    generation_semantic_sha256: str
    generation_execution_config_sha256: str
    filter_config_sha256: str
    filter_source_sha256: str
    filter_contract_version: str | int
    base_seed: int
    resume_mode: str = FORMAL_RESUME_MODE

    def __post_init__(self) -> None:
        for name in (
            "formal_plan_id",
            "task_plan_sha256",
            "plan_bindings_sha256",
            "checkpoint_sha256",
            "generation_semantic_sha256",
            "generation_execution_config_sha256",
            "filter_config_sha256",
            "filter_source_sha256",
        ):
            object.__setattr__(self, name, _sha256_text(getattr(self, name), name))
        if isinstance(self.filter_contract_version, bool) or not isinstance(
            self.filter_contract_version,
            (str, int),
        ):
            raise ValueError("filter_contract_version must be a string or integer")
        if isinstance(self.filter_contract_version, str) and not self.filter_contract_version:
            raise ValueError("filter_contract_version must not be empty")
        _nonnegative_integer(self.base_seed, "base_seed")
        if self.resume_mode != FORMAL_RESUME_MODE:
            raise ValueError(f"formal resume mode must remain {FORMAL_RESUME_MODE!r}")

    @classmethod
    def from_plan(
        cls,
        plan: FormalTaskPlan,
        *,
        task_plan_sha256: str,
    ) -> "FormalStateBindings":
        source_sha256 = dict(plan.bindings.source_sha256)
        return cls(
            formal_plan_id=plan.formal_plan_id,
            task_plan_sha256=task_plan_sha256,
            plan_bindings_sha256=plan.bindings.semantic_sha256,
            checkpoint_sha256=plan.bindings.checkpoint_sha256,
            generation_semantic_sha256=plan.bindings.semantic_config_sha256,
            generation_execution_config_sha256=(
                plan.bindings.generation_execution_sha256
            ),
            filter_config_sha256=plan.bindings.filter_semantic_sha256,
            filter_source_sha256=source_sha256["filter_source"],
            filter_contract_version=FILTER_CONTRACT_VERSION,
            base_seed=plan.bindings.base_seed,
        )

    @property
    def semantic_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "formal_plan_id": self.formal_plan_id,
            "task_plan_sha256": self.task_plan_sha256,
            "plan_bindings_sha256": self.plan_bindings_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "generation_semantic_sha256": self.generation_semantic_sha256,
            "generation_execution_config_sha256": (
                self.generation_execution_config_sha256
            ),
            "filter_config_sha256": self.filter_config_sha256,
            "filter_source_sha256": self.filter_source_sha256,
            "filter_contract_version": self.filter_contract_version,
            "base_seed": self.base_seed,
            "resume_mode": self.resume_mode,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalStateBindings":
        fields = {
            "formal_plan_id",
            "task_plan_sha256",
            "plan_bindings_sha256",
            "checkpoint_sha256",
            "generation_semantic_sha256",
            "generation_execution_config_sha256",
            "filter_config_sha256",
            "filter_source_sha256",
            "filter_contract_version",
            "base_seed",
            "resume_mode",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("formal state bindings have missing or unknown fields")
        return cls(**value)


@dataclass(frozen=True)
class FormalRawReference:
    commit_path: str
    commit_sha256: str
    shard_index: int
    execution_config_sha256: str
    arrays_path: str
    arrays_sha256: str
    metadata_path: str
    metadata_sha256: str
    candidate_indices: Sequence[int]
    candidate_ids_sha256: str

    def __post_init__(self) -> None:
        _required_text(self.commit_path, "commit_path")
        _sha256_text(self.commit_sha256, "commit_sha256")
        _nonnegative_integer(self.shard_index, "shard_index")
        _sha256_text(self.execution_config_sha256, "execution_config_sha256")
        _required_text(self.arrays_path, "arrays_path")
        _sha256_text(self.arrays_sha256, "arrays_sha256")
        _required_text(self.metadata_path, "metadata_path")
        _sha256_text(self.metadata_sha256, "metadata_sha256")
        indices = tuple(self.candidate_indices)
        if not indices:
            raise ValueError("candidate_indices must not be empty")
        for index in indices:
            _nonnegative_integer(index, "candidate_indices item")
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("candidate_indices must be unique and sorted")
        object.__setattr__(self, "candidate_indices", indices)
        _sha256_text(self.candidate_ids_sha256, "candidate_ids_sha256")

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_indices)

    @classmethod
    def from_commit(
        cls,
        commit: RawShardCommit,
        *,
        task_id: str,
        artifact_root: str | Path,
    ) -> "FormalRawReference":
        references = _references_by_task(commit).get(task_id, ())
        if not references:
            raise ValueError(f"raw shard does not contain task: {task_id}")
        return _raw_reference_from_references(
            commit,
            references,
            artifact_root=artifact_root,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_path": self.commit_path,
            "commit_sha256": self.commit_sha256,
            "shard_index": self.shard_index,
            "execution_config_sha256": self.execution_config_sha256,
            "arrays_path": self.arrays_path,
            "arrays_sha256": self.arrays_sha256,
            "metadata_path": self.metadata_path,
            "metadata_sha256": self.metadata_sha256,
            "candidate_indices": list(self.candidate_indices),
            "candidate_ids_sha256": self.candidate_ids_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalRawReference":
        if not isinstance(value, dict) or set(value) != _RAW_REFERENCE_FIELDS:
            raise ValueError("formal raw reference has missing or unknown fields")
        return cls(**value)


def _references_by_task(
    commit: RawShardCommit,
) -> dict[str, tuple[RawCandidateReference, ...]]:
    grouped: dict[str, list[RawCandidateReference]] = defaultdict(list)
    for reference in commit.references:
        grouped[reference.task_id].append(reference)
    return {
        task_id: tuple(sorted(items, key=lambda item: item.candidate_index))
        for task_id, items in grouped.items()
    }


def _raw_metadata_by_candidate(
    commit: RawShardCommit,
) -> dict[str, Mapping[str, Any]]:
    try:
        with gzip.open(
            commit.metadata_path,
            "rt",
            encoding="utf-8",
            newline="",
        ) as handle:
            rows = [json.loads(line) for line in handle]
    except (OSError, EOFError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read verified raw metadata: {error}") from error
    if len(rows) != commit.candidate_count or any(
        not isinstance(row, dict) for row in rows
    ):
        raise ValueError("verified raw metadata count or row type differs")
    by_candidate = {row.get("candidate_id"): row for row in rows}
    expected_ids = {reference.candidate_id for reference in commit.references}
    if len(by_candidate) != len(rows) or set(by_candidate) != expected_ids:
        raise ValueError("verified raw metadata candidate IDs differ")
    return by_candidate


def _raw_reference_from_references(
    commit: RawShardCommit,
    references: Sequence[RawCandidateReference],
    *,
    artifact_root: str | Path,
    commit_sha256: str | None = None,
) -> FormalRawReference:
    materialized = tuple(references)
    if not materialized:
        raise ValueError("raw task reference must contain at least one candidate")
    return FormalRawReference(
        commit_path=_relative_artifact(commit.commit_path, artifact_root),
        commit_sha256=(
            _sha256_file(commit.commit_path)
            if commit_sha256 is None
            else _sha256_text(commit_sha256, "commit_sha256")
        ),
        shard_index=commit.shard_index,
        execution_config_sha256=commit.execution_config_sha256,
        arrays_path=_relative_artifact(commit.arrays_path, artifact_root),
        arrays_sha256=commit.arrays_sha256,
        metadata_path=_relative_artifact(commit.metadata_path, artifact_root),
        metadata_sha256=commit.metadata_sha256,
        candidate_indices=tuple(item.candidate_index for item in materialized),
        candidate_ids_sha256=canonical_sha256(
            [item.candidate_id for item in materialized]
        ),
    )


@dataclass(frozen=True)
class FormalInvalidCandidateReference:
    candidate_index: int
    latent_seed: int
    reason_code: str
    sidecar_path: str
    sidecar_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.candidate_index, "candidate_index")
        _nonnegative_integer(self.latent_seed, "latent_seed")
        _required_text(self.reason_code, "reason_code")
        _required_text(self.sidecar_path, "sidecar_path")
        _sha256_text(self.sidecar_sha256, "sidecar_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "latent_seed": self.latent_seed,
            "reason_code": self.reason_code,
            "sidecar_path": self.sidecar_path,
            "sidecar_sha256": self.sidecar_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalInvalidCandidateReference":
        if not isinstance(value, dict) or set(value) != _INVALID_REFERENCE_FIELDS:
            raise ValueError("invalid-generation reference has missing or unknown fields")
        return cls(**value)


@dataclass(frozen=True)
class FormalFilterReference:
    commit_path: str
    commit_sha256: str
    formal_plan_id: str
    task_plan_sha256: str
    filter_config_sha256: str
    filter_contract_version: str | int
    raw_commit_sha256: str
    decision_sha256: str
    candidate_count: int
    accepted_count: int
    rejected_count: int
    stage_rejection_counts: Mapping[str, int] | Sequence[tuple[str, int]]

    def __post_init__(self) -> None:
        _required_text(self.commit_path, "commit_path")
        for name in (
            "commit_sha256",
            "formal_plan_id",
            "task_plan_sha256",
            "filter_config_sha256",
            "raw_commit_sha256",
            "decision_sha256",
        ):
            _sha256_text(getattr(self, name), name)
        if isinstance(self.filter_contract_version, bool) or not isinstance(
            self.filter_contract_version,
            (str, int),
        ):
            raise ValueError("filter_contract_version must be a string or integer")
        if isinstance(self.filter_contract_version, str) and not self.filter_contract_version:
            raise ValueError("filter_contract_version must not be empty")
        _positive_integer(self.candidate_count, "candidate_count")
        _nonnegative_integer(self.accepted_count, "accepted_count")
        _nonnegative_integer(self.rejected_count, "rejected_count")
        if self.accepted_count + self.rejected_count != self.candidate_count:
            raise ValueError("accepted_count + rejected_count must equal candidate_count")
        counts = _count_items(self.stage_rejection_counts, "stage_rejection_counts")
        if sum(count for _, count in counts) != self.rejected_count:
            raise ValueError("stage_rejection_counts must sum to rejected_count")
        object.__setattr__(self, "stage_rejection_counts", counts)

    @classmethod
    def from_commit(
        cls,
        commit_path: str | Path,
        *,
        artifact_root: str | Path,
        plan: FormalTaskPlan,
        bindings: FormalStateBindings,
        task: FormalTask,
        raw: FormalRawReference,
    ) -> "FormalFilterReference":
        return build_formal_filter_references(
            commit_path,
            artifact_root=artifact_root,
            plan=plan,
            bindings=bindings,
            raw_by_task={task.task_id: raw},
        )[task.task_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit_path": self.commit_path,
            "commit_sha256": self.commit_sha256,
            "formal_plan_id": self.formal_plan_id,
            "task_plan_sha256": self.task_plan_sha256,
            "filter_config_sha256": self.filter_config_sha256,
            "filter_contract_version": self.filter_contract_version,
            "raw_commit_sha256": self.raw_commit_sha256,
            "decision_sha256": self.decision_sha256,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "stage_rejection_counts": dict(self.stage_rejection_counts),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalFilterReference":
        if not isinstance(value, dict) or set(value) != _FILTER_REFERENCE_FIELDS:
            raise ValueError("formal filter reference has missing or unknown fields")
        return cls(**value)


@dataclass(frozen=True)
class FormalFailureReference:
    sidecar_path: str
    sidecar_sha256: str
    stage: str
    retryable: bool
    reason_code: str
    attempt: int

    def __post_init__(self) -> None:
        _required_text(self.sidecar_path, "sidecar_path")
        _sha256_text(self.sidecar_sha256, "sidecar_sha256")
        _required_text(self.stage, "stage")
        if not isinstance(self.retryable, bool):
            raise ValueError("retryable must be a boolean")
        _required_text(self.reason_code, "reason_code")
        _positive_integer(self.attempt, "attempt")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sidecar_path": self.sidecar_path,
            "sidecar_sha256": self.sidecar_sha256,
            "stage": self.stage,
            "retryable": self.retryable,
            "reason_code": self.reason_code,
            "attempt": self.attempt,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalFailureReference":
        if not isinstance(value, dict) or set(value) != _FAILURE_REFERENCE_FIELDS:
            raise ValueError("formal failure reference has missing or unknown fields")
        return cls(**value)


@dataclass(frozen=True)
class FormalTaskState:
    task_id: str
    task_index: int
    skill_id: str
    status: FormalTaskStateStatus
    raw: FormalRawReference | None = None
    invalid_candidates: Sequence[FormalInvalidCandidateReference] = ()
    filter: FormalFilterReference | None = None
    failure: FormalFailureReference | None = None

    def __post_init__(self) -> None:
        _sha256_text(self.task_id, "task_id")
        _nonnegative_integer(self.task_index, "task_index")
        _required_text(self.skill_id, "skill_id")
        if self.status not in FORMAL_TASK_STATE_STATUSES:
            raise ValueError(f"unknown formal task state: {self.status!r}")
        if self.raw is not None and not isinstance(self.raw, FormalRawReference):
            raise ValueError("raw must be a FormalRawReference")
        invalid = tuple(self.invalid_candidates)
        if any(
            not isinstance(item, FormalInvalidCandidateReference) for item in invalid
        ):
            raise ValueError(
                "invalid_candidates must contain FormalInvalidCandidateReference values"
            )
        if tuple(sorted(invalid, key=lambda item: item.candidate_index)) != invalid:
            raise ValueError("invalid_candidates must be sorted by candidate_index")
        if len({item.candidate_index for item in invalid}) != len(invalid):
            raise ValueError("invalid_candidates contain duplicate candidate indices")
        object.__setattr__(self, "invalid_candidates", invalid)
        if self.filter is not None and not isinstance(
            self.filter, FormalFilterReference
        ):
            raise ValueError("filter must be a FormalFilterReference")
        if self.failure is not None and not isinstance(
            self.failure, FormalFailureReference
        ):
            raise ValueError("failure must be a FormalFailureReference")
        if self.filter is not None:
            if self.raw is None:
                raise ValueError("a filter reference requires a raw reference")
            if self.filter.candidate_count != self.raw.candidate_count:
                raise ValueError("filter candidate_count differs from raw candidate_count")
            if self.filter.raw_commit_sha256 != self.raw.commit_sha256:
                raise ValueError("filter reference differs from its raw commit")

        if self.status == "pending":
            if (
                self.raw is not None
                or self.filter is not None
                or self.failure is not None
            ):
                raise ValueError(
                    "pending task state can contain invalid candidates only"
                )
        elif self.status == "generated":
            if (
                (self.raw is None and not invalid)
                or self.filter is not None
                or self.failure is not None
            ):
                raise ValueError(
                    "generated task state requires raw or invalid candidates only"
                )
        elif self.status == "filtered":
            if self.raw is None or self.filter is None or self.failure is not None:
                raise ValueError("filtered task state requires raw and filter references")
        elif self.status == "accepted":
            if self.raw is None or self.filter is None or self.failure is not None:
                raise ValueError("accepted task state requires raw and filter references")
            if self.filter.accepted_count <= 0:
                raise ValueError("accepted task state requires accepted_count > 0")
        elif self.status == "rejected":
            if self.failure is not None:
                raise ValueError("rejected task state cannot contain task failure")
            if self.raw is None:
                if self.filter is not None or not invalid:
                    raise ValueError(
                        "all-invalid rejected state requires invalid candidates only"
                    )
            elif self.filter is None:
                raise ValueError("rejected raw candidates require a filter reference")
            elif self.filter.accepted_count != 0:
                raise ValueError("rejected task state requires accepted_count == 0")
        elif self.status == "failed" and self.failure is None:
            raise ValueError("failed task state requires a failure sidecar")

    @classmethod
    def pending(
        cls,
        task: FormalTask,
        *,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "pending",
            invalid_candidates=invalid_candidates,
        )

    @classmethod
    def generated(
        cls,
        task: FormalTask,
        raw: FormalRawReference | None = None,
        *,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "generated",
            raw=raw,
            invalid_candidates=invalid_candidates,
        )

    @classmethod
    def filtered(
        cls,
        task: FormalTask,
        raw: FormalRawReference,
        filter: FormalFilterReference,
        *,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "filtered",
            raw=raw,
            invalid_candidates=invalid_candidates,
            filter=filter,
        )

    @classmethod
    def accepted(
        cls,
        task: FormalTask,
        raw: FormalRawReference,
        filter: FormalFilterReference,
        *,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "accepted",
            raw=raw,
            invalid_candidates=invalid_candidates,
            filter=filter,
        )

    @classmethod
    def rejected(
        cls,
        task: FormalTask,
        raw: FormalRawReference | None = None,
        filter: FormalFilterReference | None = None,
        *,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "rejected",
            raw=raw,
            invalid_candidates=invalid_candidates,
            filter=filter,
        )

    @classmethod
    def failed(
        cls,
        task: FormalTask,
        failure: FormalFailureReference,
        *,
        raw: FormalRawReference | None = None,
        invalid_candidates: Sequence[FormalInvalidCandidateReference] = (),
        filter: FormalFilterReference | None = None,
    ) -> "FormalTaskState":
        return cls(
            task.task_id,
            task.task_index,
            task.skill_id,
            "failed",
            raw=raw,
            invalid_candidates=invalid_candidates,
            filter=filter,
            failure=failure,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_index": self.task_index,
            "skill_id": self.skill_id,
            "status": self.status,
            "raw": None if self.raw is None else self.raw.to_dict(),
            "invalid_candidates": [item.to_dict() for item in self.invalid_candidates],
            "filter": None if self.filter is None else self.filter.to_dict(),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FormalTaskState":
        if not isinstance(value, dict) or set(value) != _TASK_STATE_FIELDS:
            raise ValueError("formal task state has missing or unknown fields")
        raw = None if value["raw"] is None else FormalRawReference.from_dict(value["raw"])
        if not isinstance(value["invalid_candidates"], list):
            raise ValueError("invalid_candidates must be a list")
        invalid = tuple(
            FormalInvalidCandidateReference.from_dict(item)
            for item in value["invalid_candidates"]
        )
        filter_reference = (
            None
            if value["filter"] is None
            else FormalFilterReference.from_dict(value["filter"])
        )
        failure = (
            None
            if value["failure"] is None
            else FormalFailureReference.from_dict(value["failure"])
        )
        return cls(
            task_id=value["task_id"],
            task_index=value["task_index"],
            skill_id=value["skill_id"],
            status=value["status"],
            raw=raw,
            invalid_candidates=invalid,
            filter=filter_reference,
            failure=failure,
        )


@dataclass(frozen=True)
class FormalStateShardCommit:
    shard_index: int
    path: Path
    commit_sha256: str


@dataclass(frozen=True)
class FormalRunState:
    bindings: FormalStateBindings
    task_states: tuple[FormalTaskState, ...]
    state_commit_sha256: Sequence[tuple[int, str]]

    def __post_init__(self) -> None:
        if any(not isinstance(state, FormalTaskState) for state in self.task_states):
            raise ValueError("task_states must contain FormalTaskState values")
        commits: list[tuple[int, str]] = []
        seen: set[int] = set()
        for shard_index, digest in self.state_commit_sha256:
            _nonnegative_integer(shard_index, "state shard index")
            _sha256_text(digest, "state shard SHA-256")
            if shard_index in seen:
                raise ValueError("state_commit_sha256 contains duplicate shard indices")
            seen.add(shard_index)
            commits.append((shard_index, digest))
        object.__setattr__(self, "state_commit_sha256", tuple(sorted(commits)))

    @property
    def state_snapshot_sha256(self) -> str:
        return canonical_sha256(
            {
                "bindings_sha256": self.bindings.semantic_sha256,
                "state_commits": [list(item) for item in self.state_commit_sha256],
            }
        )


@dataclass(frozen=True)
class FormalRawRecovery:
    generated_task_states: tuple[FormalTaskState, ...]
    partial_task_states: tuple[FormalTaskState, ...]
    rebuild_task_ids: frozenset[str]
    pending_task_ids: frozenset[str]
    invalid_raw_shards: tuple[RawRecoveryIssue, ...]
    orphaned_files: tuple[Path, ...]


@dataclass(frozen=True)
class FormalResumePlan:
    generate_task_ids: tuple[str, ...]
    filter_task_ids: tuple[str, ...]
    finalize_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    terminal_failed_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class FormalProgressRuntime:
    updated_at_utc: str
    elapsed_seconds: float
    candidates_per_second: float
    accepted_per_second: float
    eta_seconds: float | None

    def __post_init__(self) -> None:
        _required_text(self.updated_at_utc, "updated_at_utc")
        object.__setattr__(
            self,
            "elapsed_seconds",
            _finite_nonnegative(self.elapsed_seconds, "elapsed_seconds"),
        )
        object.__setattr__(
            self,
            "candidates_per_second",
            _finite_nonnegative(
                self.candidates_per_second,
                "candidates_per_second",
            ),
        )
        object.__setattr__(
            self,
            "accepted_per_second",
            _finite_nonnegative(self.accepted_per_second, "accepted_per_second"),
        )
        if self.eta_seconds is not None:
            object.__setattr__(
                self,
                "eta_seconds",
                _finite_nonnegative(self.eta_seconds, "eta_seconds"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "updated_at_utc": self.updated_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "candidates_per_second": self.candidates_per_second,
            "accepted_per_second": self.accepted_per_second,
            "eta_seconds": self.eta_seconds,
        }


@dataclass(frozen=True)
class FormalProgressSnapshot:
    formal_plan_id: str
    task_plan_sha256: str
    bindings_sha256: str
    state_snapshot_sha256: str
    tasks: Mapping[str, int]
    candidates: Mapping[str, Any]
    by_skill: Mapping[str, Mapping[str, Any]]
    runtime: FormalProgressRuntime

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORMAL_STATE_SCHEMA_VERSION,
            "kind": "formal_progress",
            "formal_plan_id": self.formal_plan_id,
            "task_plan_sha256": self.task_plan_sha256,
            "bindings_sha256": self.bindings_sha256,
            "state_snapshot_sha256": self.state_snapshot_sha256,
            "tasks": dict(self.tasks),
            "candidates": dict(self.candidates),
            "by_skill": {
                skill_id: dict(values) for skill_id, values in self.by_skill.items()
            },
            "runtime": self.runtime.to_dict(),
        }


def _expected_bindings(
    plan: FormalTaskPlan,
    *,
    task_plan_sha256: str,
) -> FormalStateBindings:
    return FormalStateBindings.from_plan(
        plan,
        task_plan_sha256=task_plan_sha256,
    )


def _summary(plan: FormalTaskPlan, bindings: FormalStateBindings) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_state_summary",
        "bindings": bindings.to_dict(),
        "counts": {
            "tasks": len(plan.tasks),
            "shards": plan.shard_count,
            "tasks_per_shard": plan.bindings.tasks_per_shard,
        },
        "state_directory": FORMAL_STATE_DIRECTORY_NAME,
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def initialize_formal_state(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    task_plan_sha256: str,
) -> FormalStateBindings:
    """Create the immutable small run summary; task shards stay implicit-pending."""

    bindings = _expected_bindings(
        plan,
        task_plan_sha256=task_plan_sha256,
    )
    root = Path(directory)
    path = root / FORMAL_STATE_SUMMARY_NAME
    expected = _summary(plan, bindings)
    payload = canonical_json_bytes(expected, indent=2)
    if path.exists():
        _load_summary(root, plan=plan, expected_bindings=bindings)
    else:
        state_directory = root / FORMAL_STATE_DIRECTORY_NAME
        if state_directory.exists() and any(state_directory.glob("*.commit.json")):
            raise ValueError("formal state commits exist without their immutable summary")
        _atomic_write(path, payload)
    return bindings


def _load_summary(
    directory: Path,
    *,
    plan: FormalTaskPlan,
    expected_bindings: FormalStateBindings,
) -> None:
    value = _read_json(directory / FORMAL_STATE_SUMMARY_NAME, "formal state summary")
    if set(value) != _SUMMARY_FIELDS:
        raise ValueError("formal state summary has missing or unknown fields")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != FORMAL_STATE_SCHEMA_VERSION
    ):
        raise ValueError("formal state schema version is incompatible")
    if value.get("kind") != "formal_state_summary":
        raise ValueError("formal state summary kind is invalid")
    stored = FormalStateBindings.from_dict(value.get("bindings"))
    if stored != expected_bindings:
        raise ValueError("formal state semantic drift: stored bindings differ")
    counts = value.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "tasks",
        "shards",
        "tasks_per_shard",
    }:
        raise ValueError("formal state summary counts are invalid")
    counts = {
        name: _nonnegative_integer(counts[name], f"formal state counts.{name}")
        for name in ("tasks", "shards", "tasks_per_shard")
    }
    if counts != _summary(plan, expected_bindings)["counts"]:
        raise ValueError("formal state summary counts differ from the task plan")
    if value.get("state_directory") != FORMAL_STATE_DIRECTORY_NAME:
        raise ValueError("formal state summary references an unexpected directory")


def _tasks_for_shard(plan: FormalTaskPlan, shard_index: int) -> tuple[FormalTask, ...]:
    tasks_per_shard = plan.bindings.tasks_per_shard
    start = shard_index * tasks_per_shard
    end = min(start + tasks_per_shard, len(plan.tasks))
    return plan.tasks[start:end]


def _validate_task_states(
    plan: FormalTaskPlan,
    shard_index: int,
    states: Sequence[FormalTaskState],
) -> tuple[FormalTaskState, ...]:
    expected = _tasks_for_shard(plan, shard_index)
    materialized = tuple(states)
    if not expected:
        raise ValueError(f"formal state shard is outside the task plan: {shard_index}")
    if len(materialized) != len(expected):
        raise ValueError("formal state shard must contain every task in its plan shard")
    for task, state in zip(expected, materialized):
        if (
            state.task_id != task.task_id
            or state.task_index != task.task_index
            or state.skill_id != task.skill_id
        ):
            raise ValueError("formal task state differs from its task plan entry")
        raw_indices: set[int] = set()
        if state.raw is not None:
            if state.raw.shard_index != shard_index:
                raise ValueError("formal raw reference differs from the task shard")
            if (
                state.raw.execution_config_sha256
                != plan.bindings.generation_execution_sha256
            ):
                raise ValueError("formal raw execution config differs from plan binding")
            raw_indices = set(state.raw.candidate_indices)
        invalid_indices: set[int] = set()
        for invalid in state.invalid_candidates:
            if not 0 <= invalid.candidate_index < task.candidate_budget:
                raise ValueError("invalid-generation candidate index exceeds budget")
            expected_seed = latent_seed(
                plan.bindings.base_seed,
                task.task_id,
                invalid.candidate_index,
            )
            if invalid.latent_seed != expected_seed:
                raise ValueError("invalid-generation latent seed differs from task")
            invalid_indices.add(invalid.candidate_index)
        if raw_indices & invalid_indices:
            raise ValueError("raw and invalid-generation candidate indices overlap")
        covered = raw_indices | invalid_indices
        if any(not 0 <= index < task.candidate_budget for index in raw_indices):
            raise ValueError("formal raw candidate index exceeds frozen budget")
        if state.status == "pending" and covered == set(range(task.candidate_budget)):
            raise ValueError("pending task state already covers its candidate budget")
        if (
            state.status == "failed"
            and state.raw is not None
            and covered != set(range(task.candidate_budget))
        ):
            raise ValueError("failed task raw must cover its candidate budget")
        if state.status in ("generated", "filtered", "accepted", "rejected") and covered != set(
            range(task.candidate_budget)
        ):
            raise ValueError("terminal generation state does not cover candidate budget")
    return materialized


def _state_commit_payload(
    bindings: FormalStateBindings,
    shard_index: int,
    states: Sequence[FormalTaskState],
) -> dict[str, Any]:
    return {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_task_state_shard",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "bindings_sha256": bindings.semantic_sha256,
        "shard_index": shard_index,
        "tasks": [state.to_dict() for state in states],
    }


def _validate_state_transitions(
    previous: Sequence[FormalTaskState],
    current: Sequence[FormalTaskState],
) -> None:
    for old, new in zip(previous, current):
        if old == new:
            continue
        if old.status in ("accepted", "rejected"):
            raise ValueError(f"completed formal task state is immutable: {old.task_id}")
        if old.raw is not None and new.raw != old.raw:
            raise ValueError("formal task state cannot replace a durable raw reference")
        new_invalid_by_index = {
            item.candidate_index: item for item in new.invalid_candidates
        }
        if any(
            new_invalid_by_index.get(item.candidate_index) != item
            for item in old.invalid_candidates
        ):
            raise ValueError(
                "formal task state cannot discard or replace an invalid candidate"
            )
        if old.filter is not None and new.filter is None:
            raise ValueError("formal task state cannot discard a durable filter reference")
        if old.filter is not None or new.filter is not None:
            if old.invalid_candidates != new.invalid_candidates:
                raise ValueError(
                    "invalid-generation references cannot change after entering filter"
                )
            if old.filter is not None and new.filter != old.filter:
                raise ValueError("filtered task cannot replace its filter reference")
        if old.status not in ("pending", "failed") and (
            old.invalid_candidates != new.invalid_candidates
        ):
            raise ValueError("invalid-generation candidate identity changed")
        if old.status == "failed":
            old_failure = old.failure
            if old_failure is None:
                raise ValueError("failed formal task state lacks its failure reference")
            if not old_failure.retryable:
                raise ValueError(f"terminal formal task failure is immutable: {old.task_id}")
            if (
                new.status == "pending"
                and new.invalid_candidates == old.invalid_candidates
            ):
                raise ValueError(
                    "retryable failure must remain recorded until new output commits"
                )
            if new.status == "failed":
                new_failure = new.failure
                if new_failure is None:
                    raise ValueError("failed formal task state lacks its failure reference")
                if new_failure.attempt <= old_failure.attempt:
                    raise ValueError("a changed formal task failure must increase attempt")
            continue
        if old.status == new.status:
            continue
        if new.status == "failed":
            continue
        if _STATE_PHASE[new.status] < _STATE_PHASE[old.status]:
            raise ValueError(
                f"formal task state cannot move backward: {old.status} -> {new.status}"
            )


def commit_formal_state_shard(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    shard_index: int,
    task_states: Sequence[FormalTaskState],
) -> FormalStateShardCommit:
    """Atomically replace one small plan-aligned state shard."""

    _nonnegative_integer(shard_index, "shard_index")
    root = Path(directory)
    _load_summary(root, plan=plan, expected_bindings=bindings)
    states = _validate_task_states(plan, shard_index, task_states)
    base = _state_commit_payload(bindings, shard_index, states)
    value = {**base, "state_sha256": canonical_sha256(base)}
    payload = canonical_json_bytes(value, indent=2)
    path = (
        root
        / FORMAL_STATE_DIRECTORY_NAME
        / f"shard-{shard_index:05d}.commit.json"
    )
    if path.exists():
        stored_index, previous = _read_state_commit(
            path,
            plan=plan,
            bindings=bindings,
        )
        if stored_index != shard_index:
            raise ValueError("formal state shard index changed during update")
        _validate_state_transitions(previous, states)
    _verify_state_artifacts(root, plan, bindings, states)
    if not path.exists() or path.read_bytes() != payload:
        _atomic_write(path, payload)
    return FormalStateShardCommit(
        shard_index=shard_index,
        path=path,
        commit_sha256=_sha256_file(path),
    )


def commit_formal_state_shards(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    shard_states: Mapping[int, Sequence[FormalTaskState]],
) -> tuple[FormalStateShardCommit, ...]:
    """Commit multiple shards while verifying shared raw/filter artifacts once."""

    if not isinstance(shard_states, Mapping) or not shard_states:
        raise ValueError("shard_states must be a non-empty mapping")
    root = Path(directory)
    _load_summary(root, plan=plan, expected_bindings=bindings)
    prepared: list[tuple[int, tuple[FormalTaskState, ...], Path, bytes]] = []
    all_states: list[FormalTaskState] = []
    for shard_index, task_states in sorted(shard_states.items()):
        _nonnegative_integer(shard_index, "shard_index")
        states = _validate_task_states(plan, shard_index, task_states)
        base = _state_commit_payload(bindings, shard_index, states)
        value = {**base, "state_sha256": canonical_sha256(base)}
        path = (
            root
            / FORMAL_STATE_DIRECTORY_NAME
            / f"shard-{shard_index:05d}.commit.json"
        )
        if path.exists():
            stored_index, previous = _read_state_commit(
                path,
                plan=plan,
                bindings=bindings,
            )
            if stored_index != shard_index:
                raise ValueError("formal state shard index changed during update")
            _validate_state_transitions(previous, states)
        prepared.append(
            (shard_index, states, path, canonical_json_bytes(value, indent=2))
        )
        all_states.extend(states)

    _verify_state_artifacts(root, plan, bindings, all_states)
    commits: list[FormalStateShardCommit] = []
    for shard_index, _, path, payload in prepared:
        if not path.exists() or path.read_bytes() != payload:
            _atomic_write(path, payload)
        commits.append(
            FormalStateShardCommit(
                shard_index=shard_index,
                path=path,
                commit_sha256=_sha256_file(path),
            )
        )
    return tuple(commits)


def _read_state_commit(
    path: Path,
    *,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
) -> tuple[int, tuple[FormalTaskState, ...]]:
    value = _read_json(path, "formal state shard")
    if set(value) != _STATE_COMMIT_FIELDS:
        raise ValueError("formal state shard has missing or unknown fields")
    base = {key: item for key, item in value.items() if key != "state_sha256"}
    stored_sha = _sha256_text(value.get("state_sha256"), "state_sha256")
    if canonical_sha256(base) != stored_sha:
        raise ValueError(f"formal state shard SHA-256 differs: {path}")
    if (
        isinstance(value.get("schema_version"), bool)
        or value.get("schema_version") != FORMAL_STATE_SCHEMA_VERSION
    ):
        raise ValueError("formal state shard schema version is incompatible")
    if value.get("kind") != "formal_task_state_shard":
        raise ValueError("formal state shard kind is invalid")
    if (
        value.get("formal_plan_id") != bindings.formal_plan_id
        or value.get("task_plan_sha256") != bindings.task_plan_sha256
        or value.get("bindings_sha256") != bindings.semantic_sha256
    ):
        raise ValueError("formal state shard semantic drift")
    shard_index = _nonnegative_integer(value.get("shard_index"), "shard_index")
    match = _STATE_COMMIT_PATTERN.fullmatch(path.name)
    if match is None or int(match.group("index")) != shard_index:
        raise ValueError("formal state shard index differs from its file name")
    rows = value.get("tasks")
    if not isinstance(rows, list):
        raise ValueError("formal state shard tasks must be a list")
    states = tuple(FormalTaskState.from_dict(row) for row in rows)
    return shard_index, _validate_task_states(plan, shard_index, states)


def _validate_raw_task_reference(
    commit: RawShardCommit,
    task: FormalTask,
    raw: FormalRawReference,
    *,
    base_seed: int,
    references: Sequence[RawCandidateReference] | None = None,
    metadata_by_candidate: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    if commit.shard_index != task.shard_index or raw.shard_index != task.shard_index:
        raise ValueError("raw artifact shard differs from the formal task plan")
    references = (
        _references_by_task(commit).get(task.task_id, ())
        if references is None
        else tuple(references)
    )
    if len(references) != raw.candidate_count:
        raise ValueError("raw artifact candidate count differs from formal state")
    if tuple(item.candidate_index for item in references) != raw.candidate_indices:
        raise ValueError("raw artifact candidate indices differ from formal state")
    if canonical_sha256([item.candidate_id for item in references]) != raw.candidate_ids_sha256:
        raise ValueError("raw artifact candidate digest differs from formal state")
    metadata = (
        _raw_metadata_by_candidate(commit)
        if metadata_by_candidate is None
        else metadata_by_candidate
    )
    seen: set[int] = set()
    for reference in references:
        index = reference.candidate_index
        if index in seen or not 0 <= index < task.candidate_budget:
            raise ValueError("raw artifact candidate indices differ from the frozen budget")
        seen.add(index)
        expected_seed = latent_seed(base_seed, task.task_id, index)
        if reference.latent_seed != expected_seed:
            raise ValueError("raw artifact latent seed differs from the formal contract")
        expected_id = candidate_id(
            task_id=task.task_id,
            candidate_index=index,
            latent_seed=expected_seed,
            checkpoint_sha256=task.checkpoint_sha256,
            semantic_config_sha256=task.semantic_config_sha256,
        )
        if reference.candidate_id != expected_id:
            raise ValueError("raw artifact candidate ID differs from the formal contract")
        row = metadata.get(reference.candidate_id)
        if row is None:
            raise ValueError("raw artifact metadata candidate is missing")
        expected_metadata = {
            "scenario_id": task.scenario_id,
            "skill_id": task.skill_id,
            "proposal_mode": task.proposal_mode,
            "target_track_id": task.target_track_id,
            "checkpoint_sha256": task.checkpoint_sha256,
            "semantic_config_sha256": task.semantic_config_sha256,
        }
        for name, expected_value in expected_metadata.items():
            if row.get(name) != expected_value:
                raise ValueError(f"raw artifact {name} differs from the formal task")


def _commit_identity(
    path: Path,
    commit_sha256: str,
    *,
    artifact_name: str,
) -> tuple[str, str, int, int, int]:
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(f"{artifact_name} is missing: {path}") from error
    return (
        str(path.resolve()),
        commit_sha256,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


@dataclass(frozen=True)
class _VerifiedRawArtifact:
    commit: RawShardCommit
    references_by_task: Mapping[str, tuple[RawCandidateReference, ...]]
    metadata_by_candidate: Mapping[str, Mapping[str, Any]]


def _verify_raw_artifact(
    path: Path,
    *,
    commit_sha256: str,
    bindings: FormalStateBindings,
) -> _VerifiedRawArtifact:
    resolved = path.resolve()
    expected_sha256 = _sha256_text(commit_sha256, "raw commit SHA-256")
    try:
        actual_sha256 = _sha256_file(resolved)
    except OSError as error:
        raise ValueError(f"raw artifact is missing: {resolved}") from error
    if actual_sha256 != expected_sha256:
        raise ValueError(f"raw commit SHA-256 differs: {resolved}")
    try:
        commit = verify_raw_shard(
            resolved,
            expected_semantic_config_sha256=bindings.generation_semantic_sha256,
        )
    except RawShardError as error:
        raise ValueError(f"raw artifact is damaged: {resolved}: {error}") from error
    _validate_raw_commit_layout(commit)
    if commit.execution_config_sha256 != bindings.generation_execution_config_sha256:
        raise ValueError("raw artifact execution config drift")
    return _VerifiedRawArtifact(
        commit=commit,
        references_by_task=_references_by_task(commit),
        metadata_by_candidate=_raw_metadata_by_candidate(commit),
    )


def _verify_failure(
    root: Path,
    bindings: FormalStateBindings,
    task: FormalTask,
    reference: FormalFailureReference,
) -> None:
    path = _artifact_path(root, reference.sidecar_path, "failure.sidecar_path")
    try:
        digest = _sha256_file(path)
    except OSError as error:
        raise ValueError(f"failure sidecar is missing: {path}") from error
    if digest != reference.sidecar_sha256:
        raise ValueError(f"failure sidecar SHA-256 differs: {path}")
    value = _read_json(path, "formal failure sidecar")
    if set(value) != _FAILURE_SIDECAR_FIELDS:
        raise ValueError("formal failure sidecar has missing or unknown fields")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != FORMAL_STATE_SCHEMA_VERSION
    ):
        raise ValueError("formal failure sidecar schema version is incompatible")
    _nonnegative_integer(value["task_index"], "failure sidecar task_index")
    if not isinstance(value["retryable"], bool):
        raise ValueError("failure sidecar retryable must be a boolean")
    _positive_integer(value["attempt"], "failure sidecar attempt")
    expected = {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_task_failure",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "bindings_sha256": bindings.semantic_sha256,
        "task_id": task.task_id,
        "task_index": task.task_index,
        "skill_id": task.skill_id,
        "stage": reference.stage,
        "retryable": reference.retryable,
        "reason_code": reference.reason_code,
        "message": value.get("message"),
        "attempt": reference.attempt,
    }
    _required_text(value.get("message"), "failure message")
    if value != expected:
        raise ValueError("formal failure sidecar differs from its state reference")


@dataclass(frozen=True)
class _FilterTaskEvidence:
    accepted_ids: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    raw_commit_path: str
    raw_commit_sha256: str


@dataclass(frozen=True)
class _VerifiedFilterCommit:
    decision_sha256: str
    by_task: Mapping[str, _FilterTaskEvidence]


def _verified_file(path: Path, descriptor: Any, name: str) -> Path:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{name} descriptor is invalid")
    file_name = descriptor["path"]
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError(f"{name} path is invalid")
    file_path = path.parent / file_name
    expected_size = _nonnegative_integer(descriptor["size_bytes"], f"{name} size")
    expected_sha256 = _sha256_text(descriptor["sha256"], f"{name} SHA-256")
    try:
        if file_path.stat().st_size != expected_size:
            raise ValueError(f"{name} size differs: {file_path}")
        if _sha256_file(file_path) != expected_sha256:
            raise ValueError(f"{name} SHA-256 differs: {file_path}")
    except OSError as error:
        raise ValueError(f"{name} file is missing: {file_path}") from error
    return file_path


def _jsonl_rows(path: Path, name: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"failed to read {name}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{name} contains a blank line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name} line {line_number} is invalid: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{name} row must be an object")
        canonical_json_bytes(row)
        rows.append(row)
    return rows


def _verify_filter_commit(
    path: Path,
    *,
    root: Path,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    expected_commit_sha256: str | None = None,
    raw_artifact_cache: dict[
        tuple[str, str, int, int, int],
        _VerifiedRawArtifact,
    ]
    | None = None,
) -> _VerifiedFilterCommit:
    expected_bindings = FormalStateBindings.from_plan(
        plan,
        task_plan_sha256=bindings.task_plan_sha256,
    )
    if bindings != expected_bindings:
        raise ValueError("formal filter bindings differ from the task plan")
    path = path.resolve()
    try:
        actual_commit_sha256 = _sha256_file(path)
    except OSError as error:
        raise ValueError(f"filter artifact is missing: {path}") from error
    if expected_commit_sha256 is not None and actual_commit_sha256 != _sha256_text(
        expected_commit_sha256,
        "filter commit SHA-256",
    ):
        raise ValueError(f"filter artifact commit SHA-256 differs: {path}")
    value = _read_json(path, "formal filter commit")
    required = {
        "schema_version",
        "kind",
        "formal_plan_id",
        "task_plan_sha256",
        "filter_config_sha256",
        "filter_contract_version",
        "raw_commits",
        "decision_sha256",
        "counts",
        "task_statuses",
        "files",
    }
    if set(value) != required:
        raise ValueError("formal filter commit has missing or unknown fields")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != FILTER_INDEX_SCHEMA_VERSION
    ):
        raise ValueError("formal filter commit schema version is incompatible")
    if value["kind"] != "formal_filter_commit":
        raise ValueError("formal filter commit kind is invalid")
    expected_binding = (
        bindings.formal_plan_id,
        bindings.task_plan_sha256,
        bindings.filter_config_sha256,
        bindings.filter_contract_version,
    )
    filter_contract_version = value["filter_contract_version"]
    if isinstance(filter_contract_version, bool) or not isinstance(
        filter_contract_version, (str, int)
    ):
        raise ValueError("formal filter contract version is invalid")
    if (
        value["formal_plan_id"],
        value["task_plan_sha256"],
        value["filter_config_sha256"],
        filter_contract_version,
    ) != expected_binding:
        raise ValueError("formal filter commit semantic drift")

    raw_commits = value["raw_commits"]
    if not isinstance(raw_commits, list) or not raw_commits:
        raise ValueError("formal filter raw_commits must be a non-empty list")
    raw_by_path: dict[str, RawShardCommit] = {}
    raw_sha_by_path: dict[str, str] = {}
    seen_shard_indices: set[int] = set()
    for descriptor in raw_commits:
        if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
            raise ValueError("formal filter raw commit descriptor is invalid")
        relative = descriptor["path"]
        raw_path = _artifact_path(root, relative, "raw_commits.path")
        digest = _sha256_text(descriptor["sha256"], "raw_commits.sha256")
        raw_identity = _commit_identity(
            raw_path,
            digest,
            artifact_name="raw artifact",
        )
        verified_raw = (
            None
            if raw_artifact_cache is None
            else raw_artifact_cache.get(raw_identity)
        )
        if verified_raw is None:
            verified_raw = _verify_raw_artifact(
                raw_path,
                commit_sha256=digest,
                bindings=bindings,
            )
            if raw_artifact_cache is not None:
                raw_artifact_cache[raw_identity] = verified_raw
        commit = verified_raw.commit
        if commit.shard_index in seen_shard_indices:
            raise ValueError("formal filter contains duplicate raw shard indices")
        seen_shard_indices.add(commit.shard_index)
        expected_tasks = _tasks_for_shard(plan, commit.shard_index)
        expected_task_ids = {task.task_id for task in expected_tasks}
        references_by_task = _references_by_task(commit)
        if not expected_tasks or set(references_by_task) - expected_task_ids:
            raise ValueError("formal filter raw commit is outside the task plan")
        metadata_by_candidate = _raw_metadata_by_candidate(commit)
        for task in expected_tasks:
            references = references_by_task.get(task.task_id, ())
            if not references:
                continue
            raw_reference = _raw_reference_from_references(
                commit,
                references,
                artifact_root=root,
                commit_sha256=digest,
            )
            _validate_raw_task_reference(
                commit,
                task,
                raw_reference,
                base_seed=plan.bindings.base_seed,
                references=references,
                metadata_by_candidate=metadata_by_candidate,
            )
        if relative in raw_by_path:
            raise ValueError("formal filter contains duplicate raw commit descriptors")
        raw_by_path[relative] = commit
        raw_sha_by_path[relative] = digest
    references_by_raw = {
        relative: {reference.candidate_id: reference for reference in commit.references}
        for relative, commit in raw_by_path.items()
    }
    all_raw_references = [
        reference for commit in raw_by_path.values() for reference in commit.references
    ]
    expected_candidate_ids = {item.candidate_id for item in all_raw_references}
    if len(expected_candidate_ids) != len(all_raw_references):
        raise ValueError("formal filter raw commits contain duplicate candidate IDs")
    expected_task_ids = {item.task_id for item in all_raw_references}

    files = value["files"]
    if not isinstance(files, dict) or set(files) != {"accepted", "rejected"}:
        raise ValueError("formal filter files are invalid")
    accepted_path = _verified_file(path, files["accepted"], "formal filter accepted")
    rejected_path = _verified_file(path, files["rejected"], "formal filter rejected")
    accepted_rows = _jsonl_rows(accepted_path, "formal filter accepted")
    rejected_rows = _jsonl_rows(rejected_path, "formal filter rejected")
    decision_sha256 = canonical_sha256(
        {"accepted": accepted_rows, "rejected": rejected_rows}
    )
    if decision_sha256 != _sha256_text(value["decision_sha256"], "decision_sha256"):
        raise ValueError("formal filter decision digest differs")

    accepted_fields = {
        "candidate_id",
        "filter_evaluation_id",
        "task_id",
        "candidate_index",
        "latent_seed",
        "raw",
        "metrics",
    }
    rejected_fields = accepted_fields | {
        "rejection_reasons",
        "primary_rejection_reason",
        "first_failed_stage",
    }
    accepted_by_task: dict[str, list[str]] = defaultdict(list)
    rejected_by_task: dict[str, list[tuple[str, str]]] = defaultdict(list)
    raw_paths_by_task: dict[str, set[str]] = defaultdict(set)
    seen_candidates: set[str] = set()
    used_raw_paths: set[str] = set()
    for accepted, rows in ((True, accepted_rows), (False, rejected_rows)):
        for row in rows:
            if set(row) != (accepted_fields if accepted else rejected_fields):
                raise ValueError("formal filter row has missing or unknown fields")
            task_id = _sha256_text(row["task_id"], "filter row task_id")
            candidate_value = _sha256_text(
                row["candidate_id"], "filter row candidate_id"
            )
            if candidate_value in seen_candidates:
                raise ValueError("formal filter contains duplicate candidate IDs")
            seen_candidates.add(candidate_value)
            raw = row["raw"]
            if not isinstance(raw, dict) or set(raw) != {
                "commit",
                "arrays",
                "metadata",
                "offset",
            }:
                raise ValueError("formal filter raw reference fields are invalid")
            raw_path = _required_text(raw["commit"], "filter row raw commit")
            commit = raw_by_path.get(raw_path)
            if commit is None:
                raise ValueError("formal filter row references an unbound raw commit")
            used_raw_paths.add(raw_path)
            reference = references_by_raw[raw_path].get(candidate_value)
            if reference is None or reference.task_id != task_id:
                raise ValueError("formal filter row candidate differs from raw metadata")
            row_candidate_index = _nonnegative_integer(
                row["candidate_index"], "filter row candidate_index"
            )
            if row_candidate_index != reference.candidate_index:
                raise ValueError("formal filter row candidate_index differs from raw")
            row_latent_seed = _nonnegative_integer(
                row["latent_seed"], "filter row latent_seed"
            )
            if row_latent_seed != reference.latent_seed:
                raise ValueError("formal filter row latent_seed differs from raw")
            if row["filter_evaluation_id"] != filter_evaluation_id(
                candidate_id=candidate_value,
                filter_config_sha256=bindings.filter_config_sha256,
                filter_contract_version=bindings.filter_contract_version,
            ):
                raise ValueError("formal filter_evaluation_id differs from frozen contract")
            if raw["arrays"] != _relative_artifact(commit.arrays_path, root):
                raise ValueError("formal filter row raw arrays path differs")
            if raw["metadata"] != _relative_artifact(commit.metadata_path, root):
                raise ValueError("formal filter row raw metadata path differs")
            raw_offset = _nonnegative_integer(raw["offset"], "filter row raw offset")
            if raw_offset != reference.raw_offset:
                raise ValueError("formal filter row raw offset differs")
            if not isinstance(row["metrics"], dict):
                raise ValueError("formal filter row metrics must be an object")
            raw_paths_by_task[task_id].add(raw_path)
            if accepted:
                if row["metrics"].get("first_failed_stage") is not None:
                    raise ValueError("accepted formal filter row has failed stage")
                accepted_by_task[task_id].append(candidate_value)
            else:
                reasons = row["rejection_reasons"]
                if (
                    not isinstance(reasons, list)
                    or not reasons
                    or any(not isinstance(reason, str) or not reason for reason in reasons)
                    or row["primary_rejection_reason"] != reasons[0]
                ):
                    raise ValueError("formal filter rejection reasons are invalid")
                try:
                    tuple(FilterRejection(reason) for reason in reasons)
                except ValueError as error:
                    raise ValueError(
                        "formal filter rejection reason is outside the contract"
                    ) from error
                stage = _required_text(
                    row["first_failed_stage"], "first_failed_stage"
                )
                try:
                    FilterStage(stage)
                except ValueError as error:
                    raise ValueError(
                        "formal filter failed stage is outside the contract"
                    ) from error
                if row["metrics"].get("first_failed_stage") != stage:
                    raise ValueError("formal filter failed stage differs from metrics")
                rejected_by_task[task_id].append((candidate_value, stage))

    if seen_candidates != expected_candidate_ids:
        raise ValueError("formal filter decisions do not cover every raw candidate")
    if used_raw_paths != set(raw_by_path):
        raise ValueError("formal filter raw commit descriptors include unused commits")
    counts = value["counts"]
    if not isinstance(counts, dict) or set(counts) != {
        "accepted",
        "rejected",
        "tasks",
    }:
        raise ValueError("formal filter counts are invalid")
    counts = {
        name: _nonnegative_integer(counts[name], f"formal filter counts.{name}")
        for name in ("accepted", "rejected", "tasks")
    }
    expected_counts = {
        "accepted": len(accepted_rows),
        "rejected": len(rejected_rows),
        "tasks": len(expected_task_ids),
    }
    if counts != expected_counts:
        raise ValueError("formal filter counts differ from decision rows")
    task_statuses = value["task_statuses"]
    if (
        not isinstance(task_statuses, dict)
        or set(task_statuses) != expected_task_ids
        or any(status != "complete" for status in task_statuses.values())
    ):
        raise ValueError("formal filter task status must be complete")
    by_task: dict[str, _FilterTaskEvidence] = {}
    for task_id in sorted(expected_task_ids):
        raw_paths = raw_paths_by_task[task_id]
        if len(raw_paths) != 1:
            raise ValueError("formal filter task must reference exactly one raw commit")
        raw_path = next(iter(raw_paths))
        raw_commit = raw_by_path[raw_path]
        decision_ids = set(accepted_by_task[task_id]) | {
            candidate for candidate, _ in rejected_by_task[task_id]
        }
        expected_ids = {
            reference.candidate_id
            for reference in raw_commit.references
            if reference.task_id == task_id
        }
        if decision_ids != expected_ids:
            raise ValueError("formal filter decisions do not cover every raw candidate")
        by_task[task_id] = _FilterTaskEvidence(
            accepted_ids=tuple(sorted(accepted_by_task[task_id])),
            rejected=tuple(sorted(rejected_by_task[task_id])),
            raw_commit_path=raw_path,
            raw_commit_sha256=raw_sha_by_path[raw_path],
        )
    return _VerifiedFilterCommit(
        decision_sha256=decision_sha256,
        by_task=by_task,
    )


def _filter_reference_from_evidence(
    path: Path,
    evidence: _VerifiedFilterCommit,
    *,
    task: FormalTask,
    raw: FormalRawReference,
    bindings: FormalStateBindings,
    artifact_root: Path,
) -> FormalFilterReference:
    task_evidence = evidence.by_task.get(task.task_id)
    if task_evidence is None:
        raise ValueError("formal filter commit does not contain its task")
    accepted_ids = task_evidence.accepted_ids
    rejected_ids = tuple(candidate for candidate, _ in task_evidence.rejected)
    if (
        task_evidence.raw_commit_path != raw.commit_path
        or task_evidence.raw_commit_sha256 != raw.commit_sha256
    ):
        raise ValueError("formal filter task raw commit differs from state")
    stage_counts = Counter(stage for _, stage in task_evidence.rejected)
    return FormalFilterReference(
        commit_path=_relative_artifact(path, artifact_root),
        commit_sha256=_sha256_file(path),
        formal_plan_id=bindings.formal_plan_id,
        task_plan_sha256=bindings.task_plan_sha256,
        filter_config_sha256=bindings.filter_config_sha256,
        filter_contract_version=bindings.filter_contract_version,
        raw_commit_sha256=raw.commit_sha256,
        decision_sha256=evidence.decision_sha256,
        candidate_count=len(accepted_ids) + len(rejected_ids),
        accepted_count=len(accepted_ids),
        rejected_count=len(rejected_ids),
        stage_rejection_counts=stage_counts,
    )


def build_formal_filter_references(
    commit_path: str | Path,
    *,
    artifact_root: str | Path,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    raw_by_task: Mapping[str, FormalRawReference],
) -> dict[str, FormalFilterReference]:
    """Verify one filter commit once, then bind any number of task references."""

    path = Path(commit_path)
    root = Path(artifact_root)
    evidence = _verify_filter_commit(
        path,
        root=root,
        plan=plan,
        bindings=bindings,
    )
    tasks = {task.task_id: task for task in plan.tasks}
    result: dict[str, FormalFilterReference] = {}
    for task_id, raw in raw_by_task.items():
        task = tasks.get(task_id)
        if task is None:
            raise ValueError("formal filter reference task is outside the task plan")
        if not isinstance(raw, FormalRawReference):
            raise ValueError("raw_by_task values must be FormalRawReference values")
        result[task_id] = _filter_reference_from_evidence(
            path,
            evidence,
            task=task,
            raw=raw,
            bindings=bindings,
            artifact_root=root,
        )
    return result


def _verify_invalid_candidate(
    root: Path,
    bindings: FormalStateBindings,
    task: FormalTask,
    reference: FormalInvalidCandidateReference,
) -> None:
    path = _artifact_path(root, reference.sidecar_path, "invalid.sidecar_path")
    try:
        digest = _sha256_file(path)
    except OSError as error:
        raise ValueError(f"invalid-generation sidecar is missing: {path}") from error
    if digest != reference.sidecar_sha256:
        raise ValueError(f"invalid-generation sidecar SHA-256 differs: {path}")
    value = _read_json(path, "invalid-generation sidecar")
    if set(value) != _INVALID_SIDECAR_FIELDS:
        raise ValueError("invalid-generation sidecar has missing or unknown fields")
    if (
        isinstance(value["schema_version"], bool)
        or value["schema_version"] != FORMAL_STATE_SCHEMA_VERSION
    ):
        raise ValueError("invalid-generation sidecar schema version is incompatible")
    _nonnegative_integer(
        value["task_index"], "invalid-generation sidecar task_index"
    )
    _nonnegative_integer(
        value["candidate_index"], "invalid-generation sidecar candidate_index"
    )
    _nonnegative_integer(
        value["latent_seed"], "invalid-generation sidecar latent_seed"
    )
    expected = {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_invalid_generation_candidate",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "bindings_sha256": bindings.semantic_sha256,
        "task_id": task.task_id,
        "task_index": task.task_index,
        "skill_id": task.skill_id,
        "candidate_index": reference.candidate_index,
        "latent_seed": reference.latent_seed,
        "reason_code": reference.reason_code,
        "message": value.get("message"),
    }
    _required_text(value.get("message"), "invalid-generation message")
    if value != expected:
        raise ValueError("invalid-generation sidecar differs from state reference")


def _verify_state_artifacts(
    root: Path,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    states: Sequence[FormalTaskState],
) -> None:
    tasks = {task.task_id: task for task in plan.tasks}
    raw_cache: dict[tuple[str, str, int, int, int], _VerifiedRawArtifact] = {}
    state_raw_cache_keys: set[tuple[str, str, int, int, int]] = set()
    raw_sha_by_path: dict[Path, str] = {}
    state_task_ids_by_raw: dict[Path, set[str]] = defaultdict(set)
    filter_cache: dict[tuple[str, str, int, int, int], _VerifiedFilterCommit] = {}
    for state in states:
        task = tasks[state.task_id]
        if state.raw is not None:
            path = _artifact_path(root, state.raw.commit_path, "raw.commit_path")
            state_task_ids_by_raw[path].add(state.task_id)
            previous_sha = raw_sha_by_path.setdefault(path, state.raw.commit_sha256)
            if previous_sha != state.raw.commit_sha256:
                raise ValueError("formal state binds one raw path to multiple commits")
            cache_key = _commit_identity(
                path,
                state.raw.commit_sha256,
                artifact_name="raw artifact",
            )
            state_raw_cache_keys.add(cache_key)
            if cache_key not in raw_cache:
                raw_cache[cache_key] = _verify_raw_artifact(
                    path,
                    commit_sha256=state.raw.commit_sha256,
                    bindings=bindings,
                )
            verified_raw = raw_cache[cache_key]
            expected_raw = _raw_reference_from_references(
                verified_raw.commit,
                verified_raw.references_by_task.get(task.task_id, ()),
                artifact_root=root,
                commit_sha256=state.raw.commit_sha256,
            )
            if expected_raw != state.raw:
                raise ValueError("raw artifact reference differs from formal state")
            _validate_raw_task_reference(
                verified_raw.commit,
                task,
                state.raw,
                base_seed=plan.bindings.base_seed,
                references=verified_raw.references_by_task.get(task.task_id, ()),
                metadata_by_candidate=verified_raw.metadata_by_candidate,
            )
        for invalid in state.invalid_candidates:
            _verify_invalid_candidate(root, bindings, task, invalid)
        if state.filter is not None:
            path = _artifact_path(root, state.filter.commit_path, "filter.commit_path")
            cache_key = _commit_identity(
                path,
                state.filter.commit_sha256,
                artifact_name="filter artifact",
            )
            if cache_key not in filter_cache:
                filter_cache[cache_key] = _verify_filter_commit(
                    path,
                    root=root,
                    plan=plan,
                    bindings=bindings,
                    expected_commit_sha256=state.filter.commit_sha256,
                    raw_artifact_cache=raw_cache,
                )
            expected_filter = _filter_reference_from_evidence(
                path,
                filter_cache[cache_key],
                task=task,
                raw=state.raw,
                bindings=bindings,
                artifact_root=root,
            )
            if expected_filter != state.filter:
                raise ValueError("filter artifact reference differs from formal state")
        if state.failure is not None:
            _verify_failure(root, bindings, task, state.failure)
    for cache_key in state_raw_cache_keys:
        verified_raw = raw_cache[cache_key]
        path = Path(cache_key[0])
        if set(verified_raw.references_by_task) != state_task_ids_by_raw[path]:
            raise ValueError("raw artifact task coverage differs from formal state")


def load_formal_state(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    task_plan_sha256: str,
    resume_mode: str = FORMAL_RESUME_MODE,
) -> FormalRunState:
    """Load only committed state shards and reject corruption or semantic drift."""

    if resume_mode != FORMAL_RESUME_MODE:
        raise ValueError(f"formal resume mode must remain {FORMAL_RESUME_MODE!r}")
    root = Path(directory)
    bindings = _expected_bindings(
        plan,
        task_plan_sha256=task_plan_sha256,
    )
    _load_summary(root, plan=plan, expected_bindings=bindings)
    states = [FormalTaskState.pending(task) for task in plan.tasks]
    commit_hashes: list[tuple[int, str]] = []
    seen: set[int] = set()
    state_directory = root / FORMAL_STATE_DIRECTORY_NAME
    for path in sorted(state_directory.glob("*.commit.json")):
        if _STATE_COMMIT_PATTERN.fullmatch(path.name) is None:
            raise ValueError(f"unexpected formal state commit name: {path.name}")
        shard_index, shard_states = _read_state_commit(
            path,
            plan=plan,
            bindings=bindings,
        )
        if shard_index in seen:
            raise ValueError(f"duplicate formal state shard commit: {shard_index}")
        seen.add(shard_index)
        for state in shard_states:
            states[state.task_index] = state
        commit_hashes.append((shard_index, _sha256_file(path)))
    _verify_state_artifacts(root, plan, bindings, states)
    return FormalRunState(
        bindings=bindings,
        task_states=tuple(states),
        state_commit_sha256=tuple(commit_hashes),
    )


def open_formal_state(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    task_plan_sha256: str,
    raw_directory: str | Path | None = None,
    resume_mode: str = FORMAL_RESUME_MODE,
) -> FormalRunState:
    """Use the same ``resume=auto`` path for a first run and every restart."""

    if resume_mode != FORMAL_RESUME_MODE:
        raise ValueError(f"formal resume mode must remain {FORMAL_RESUME_MODE!r}")
    initialize_formal_state(
        directory,
        plan=plan,
        task_plan_sha256=task_plan_sha256,
    )
    loaded = load_formal_state(
        directory,
        plan=plan,
        task_plan_sha256=task_plan_sha256,
        resume_mode=resume_mode,
    )
    root = Path(directory)
    recovery = recover_generated_from_raw(
        plan,
        root / "raw" if raw_directory is None else raw_directory,
        artifact_root=root,
        bindings=loaded.bindings,
    )
    states = list(loaded.task_states)
    changed_shards: set[int] = set()
    for recovered in (
        *recovery.generated_task_states,
        *recovery.partial_task_states,
    ):
        old = states[recovered.task_index]
        if old.status in ("filtered", "accepted", "rejected"):
            if old.raw != recovered.raw or (
                old.invalid_candidates != recovered.invalid_candidates
            ):
                raise ValueError("durable raw recovery differs from filtered state")
            continue
        if old.status == "failed" and old.failure is not None:
            same_durable_generation = (
                old.raw == recovered.raw
                and old.invalid_candidates == recovered.invalid_candidates
            )
            if same_durable_generation:
                if (
                    not old.failure.retryable
                    or old.filter is not None
                    or recovered.status == "pending"
                ):
                    continue
            if not old.failure.retryable:
                raise ValueError("durable generation changed after terminal task failure")
            if old.filter is not None:
                raise ValueError("durable generation changed after filter-stage failure")
        if old != recovered:
            states[recovered.task_index] = recovered
            changed_shards.add(plan.tasks[recovered.task_index].shard_index)
    for shard_index in sorted(changed_shards):
        commit_formal_state_shard(
            root,
            plan=plan,
            bindings=loaded.bindings,
            shard_index=shard_index,
            task_states=tuple(
                states[task.task_index] for task in _tasks_for_shard(plan, shard_index)
            ),
        )
    if not changed_shards:
        return loaded
    return load_formal_state(
        root,
        plan=plan,
        task_plan_sha256=task_plan_sha256,
        resume_mode=resume_mode,
    )


def _raw_artifact_index(path: Path) -> int | None:
    match = _RAW_ARTIFACT_PATTERN.fullmatch(path.name)
    return None if match is None else int(match.group("index"))


def _validate_raw_commit_layout(commit: RawShardCommit) -> None:
    for path in (commit.commit_path, commit.arrays_path, commit.metadata_path):
        if _raw_artifact_index(path) != commit.shard_index:
            raise ValueError("raw artifact file name differs from its shard index")


def _validate_bound_task(
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    task: FormalTask,
) -> None:
    expected_bindings = FormalStateBindings.from_plan(
        plan,
        task_plan_sha256=bindings.task_plan_sha256,
    )
    if bindings != expected_bindings:
        raise ValueError("formal sidecar bindings differ from the task plan")
    if (
        task.task_index >= len(plan.tasks)
        or plan.tasks[task.task_index] != task
    ):
        raise ValueError("formal sidecar task is outside the frozen task plan")


def write_formal_candidate_invalid(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    task: FormalTask,
    candidate_index: int,
    reason_code: str,
    message: str,
) -> FormalInvalidCandidateReference:
    """Atomically record a deterministic candidate that cannot enter raw storage."""

    _nonnegative_integer(candidate_index, "candidate_index")
    if candidate_index >= task.candidate_budget:
        raise ValueError("candidate_index exceeds the frozen task budget")
    _required_text(reason_code, "reason_code")
    _required_text(message, "message")
    _validate_bound_task(plan, bindings, task)
    seed = latent_seed(bindings.base_seed, task.task_id, candidate_index)
    root = Path(directory)
    path = (
        root
        / FORMAL_INVALID_DIRECTORY_NAME
        / f"task-{task.task_index:05d}-candidate-{candidate_index:04d}.commit.json"
    )
    value = {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_invalid_generation_candidate",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "bindings_sha256": bindings.semantic_sha256,
        "task_id": task.task_id,
        "task_index": task.task_index,
        "skill_id": task.skill_id,
        "candidate_index": candidate_index,
        "latent_seed": seed,
        "reason_code": reason_code,
        "message": message,
    }
    payload = canonical_json_bytes(value, indent=2)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"invalid-generation candidate already differs: {path}")
    if not path.exists():
        _atomic_write(path, payload)
    return FormalInvalidCandidateReference(
        candidate_index=candidate_index,
        latent_seed=seed,
        reason_code=reason_code,
        sidecar_path=_relative_artifact(path, root),
        sidecar_sha256=_sha256_file(path),
    )


def _scan_invalid_candidates(
    root: Path,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
) -> dict[str, tuple[FormalInvalidCandidateReference, ...]]:
    tasks_by_index = {task.task_index: task for task in plan.tasks}
    grouped: dict[str, list[FormalInvalidCandidateReference]] = defaultdict(list)
    directory = root / FORMAL_INVALID_DIRECTORY_NAME
    for path in sorted(directory.glob("*.commit.json")):
        match = _INVALID_COMMIT_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected invalid-generation commit name: {path.name}")
        task = tasks_by_index.get(int(match.group("task")))
        if task is None:
            raise ValueError("invalid-generation sidecar task is outside formal plan")
        value = _read_json(path, "invalid-generation sidecar")
        reference = FormalInvalidCandidateReference(
            candidate_index=value.get("candidate_index"),
            latent_seed=value.get("latent_seed"),
            reason_code=value.get("reason_code"),
            sidecar_path=_relative_artifact(path, root),
            sidecar_sha256=_sha256_file(path),
        )
        if reference.candidate_index != int(match.group("candidate")):
            raise ValueError("invalid-generation candidate differs from file name")
        if reference.candidate_index >= task.candidate_budget:
            raise ValueError("invalid-generation candidate index exceeds budget")
        _verify_invalid_candidate(root, bindings, task, reference)
        grouped[task.task_id].append(reference)
    result: dict[str, tuple[FormalInvalidCandidateReference, ...]] = {}
    for task_id, items in grouped.items():
        ordered = tuple(sorted(items, key=lambda item: item.candidate_index))
        if len({item.candidate_index for item in ordered}) != len(ordered):
            raise ValueError("duplicate invalid-generation candidate sidecars")
        result[task_id] = ordered
    return result


def recover_generated_from_raw(
    plan: FormalTaskPlan,
    raw_directory: str | Path,
    *,
    artifact_root: str | Path,
    bindings: FormalStateBindings,
) -> FormalRawRecovery:
    """Recover complete raw/invalid candidate budgets without model inference."""

    root = Path(artifact_root)
    expected = FormalStateBindings.from_plan(
        plan,
        task_plan_sha256=bindings.task_plan_sha256,
    )
    if bindings != expected:
        raise ValueError("raw recovery bindings differ from formal task plan")
    scan = scan_raw_shards(
        raw_directory,
        expected_semantic_config_sha256=bindings.generation_semantic_sha256,
    )
    invalid_by_task = _scan_invalid_candidates(root, plan, bindings)
    tasks_by_shard = {
        shard_index: _tasks_for_shard(plan, shard_index)
        for shard_index in range(plan.shard_count)
    }
    rebuild_shards: set[int] = set()
    invalid = list(scan.invalid_shards)
    for issue in scan.invalid_shards:
        index = _raw_artifact_index(issue.commit_path)
        if index is not None:
            if index not in tasks_by_shard:
                raise ValueError(f"raw shard is outside the formal task plan: {index}")
            rebuild_shards.add(index)
    for path in scan.orphaned_files:
        index = _raw_artifact_index(path)
        if index is not None:
            if index not in tasks_by_shard:
                raise ValueError(f"raw artifact is outside the formal task plan: {index}")
            rebuild_shards.add(index)

    raw_by_shard: dict[int, RawShardCommit] = {}
    raw_metadata_by_shard: dict[int, Mapping[str, Mapping[str, Any]]] = {}
    for commit in scan.valid_shards:
        expected_tasks = tasks_by_shard.get(commit.shard_index)
        if expected_tasks is None:
            raise ValueError(
                f"raw shard is outside the formal task plan: {commit.shard_index}"
            )
        _validate_raw_commit_layout(commit)
        expected_task_ids = {task.task_id for task in expected_tasks}
        actual_task_ids = {reference.task_id for reference in commit.references}
        if actual_task_ids - expected_task_ids:
            raise ValueError("raw shard references tasks outside its formal plan shard")
        if commit.execution_config_sha256 != bindings.generation_execution_config_sha256:
            rebuild_shards.add(commit.shard_index)
            invalid.append(
                RawRecoveryIssue(
                    commit_path=commit.commit_path,
                    reason="raw execution config differs from formal plan binding",
                )
            )
            continue
        try:
            raw_metadata = _raw_metadata_by_candidate(commit)
        except ValueError as error:
            rebuild_shards.add(commit.shard_index)
            invalid.append(
                RawRecoveryIssue(commit_path=commit.commit_path, reason=str(error))
            )
            continue
        raw_by_shard[commit.shard_index] = commit
        raw_metadata_by_shard[commit.shard_index] = raw_metadata

    generated_states: list[FormalTaskState] = []
    partial_states: list[FormalTaskState] = []

    def preserve_invalid_candidates(tasks: Sequence[FormalTask]) -> None:
        for task in tasks:
            candidates = invalid_by_task.get(task.task_id, ())
            if not candidates:
                continue
            if {item.candidate_index for item in candidates} == set(
                range(task.candidate_budget)
            ):
                generated_states.append(
                    FormalTaskState.generated(
                        task,
                        invalid_candidates=candidates,
                    )
                )
            else:
                partial_states.append(
                    FormalTaskState.pending(
                        task,
                        invalid_candidates=candidates,
                    )
                )

    for shard_index, tasks in tasks_by_shard.items():
        if shard_index in rebuild_shards:
            preserve_invalid_candidates(tasks)
            continue
        commit = raw_by_shard.get(shard_index)
        if commit is None:
            preserve_invalid_candidates(tasks)
            continue
        references_by_task = _references_by_task(commit)
        commit_sha256 = _sha256_file(commit.commit_path)
        shard_states: list[FormalTaskState] = []
        shard_partial = False
        for task in tasks:
            references = references_by_task.get(task.task_id, ())
            raw = None
            if references:
                raw = _raw_reference_from_references(
                    commit,
                    references,
                    artifact_root=root,
                    commit_sha256=commit_sha256,
                )
                try:
                    _validate_raw_task_reference(
                        commit,
                        task,
                        raw,
                        base_seed=plan.bindings.base_seed,
                        references=references,
                        metadata_by_candidate=raw_metadata_by_shard[shard_index],
                    )
                except ValueError as error:
                    invalid.append(
                        RawRecoveryIssue(commit_path=commit.commit_path, reason=str(error))
                    )
                    shard_partial = True
                    break
            invalid_candidates = invalid_by_task.get(task.task_id, ())
            covered = set() if raw is None else set(raw.candidate_indices)
            invalid_indices = {item.candidate_index for item in invalid_candidates}
            if covered & invalid_indices:
                raise ValueError("raw and invalid-generation recovery indices overlap")
            covered.update(invalid_indices)
            if covered == set(range(task.candidate_budget)):
                shard_states.append(
                    FormalTaskState.generated(
                        task,
                        raw,
                        invalid_candidates=invalid_candidates,
                    )
                )
            elif covered:
                shard_partial = True
                break
            else:
                shard_partial = True
                break
        if shard_partial:
            rebuild_shards.add(shard_index)
            preserve_invalid_candidates(tasks)
        else:
            generated_states.extend(shard_states)

    recovered = tuple(
        sorted(generated_states, key=lambda item: item.task_index)
    )
    partial = tuple(sorted(partial_states, key=lambda item: item.task_index))
    recovered_ids = {state.task_id for state in recovered}
    rebuild_ids = frozenset(
        task.task_id
        for shard_index in rebuild_shards
        for task in tasks_by_shard[shard_index]
        if task.task_id not in recovered_ids
    )
    pending_ids = frozenset(
        task.task_id
        for task in plan.tasks
        if task.task_id not in recovered_ids and task.task_id not in rebuild_ids
    )
    return FormalRawRecovery(
        generated_task_states=recovered,
        partial_task_states=partial,
        rebuild_task_ids=rebuild_ids,
        pending_task_ids=pending_ids,
        invalid_raw_shards=tuple(invalid),
        orphaned_files=scan.orphaned_files,
    )


def write_formal_failure(
    directory: str | Path,
    *,
    plan: FormalTaskPlan,
    bindings: FormalStateBindings,
    task: FormalTask,
    stage: str,
    retryable: bool,
    reason_code: str,
    message: str,
    attempt: int,
) -> FormalFailureReference:
    """Write one immutable, task-bound failure sidecar before state commit."""

    _required_text(stage, "stage")
    if not isinstance(retryable, bool):
        raise ValueError("retryable must be a boolean")
    _required_text(reason_code, "reason_code")
    _required_text(message, "message")
    _positive_integer(attempt, "attempt")
    _validate_bound_task(plan, bindings, task)
    root = Path(directory)
    path = (
        root
        / FORMAL_FAILURE_DIRECTORY_NAME
        / f"task-{task.task_index:05d}-attempt-{attempt:03d}.json"
    )
    value = {
        "schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "kind": "formal_task_failure",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "bindings_sha256": bindings.semantic_sha256,
        "task_id": task.task_id,
        "task_index": task.task_index,
        "skill_id": task.skill_id,
        "stage": stage,
        "retryable": retryable,
        "reason_code": reason_code,
        "message": message,
        "attempt": attempt,
    }
    payload = canonical_json_bytes(value, indent=2)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"formal failure attempt already contains different data: {path}")
    if not path.exists():
        _atomic_write(path, payload)
    return FormalFailureReference(
        sidecar_path=_relative_artifact(path, root),
        sidecar_sha256=_sha256_file(path),
        stage=stage,
        retryable=retryable,
        reason_code=reason_code,
        attempt=attempt,
    )


def build_formal_resume_plan(state: FormalRunState) -> FormalResumePlan:
    """Classify ``--resume auto`` work while preserving durable raw/filter artifacts."""

    generate: list[str] = []
    filter_tasks: list[str] = []
    finalize: list[str] = []
    completed: list[str] = []
    terminal_failed: list[str] = []
    for task in state.task_states:
        if task.status == "pending":
            generate.append(task.task_id)
        elif task.status == "generated":
            (filter_tasks if task.raw is not None else finalize).append(task.task_id)
        elif task.status == "filtered":
            finalize.append(task.task_id)
        elif task.status in ("accepted", "rejected"):
            completed.append(task.task_id)
        elif task.status == "failed":
            if task.failure is not None and not task.failure.retryable:
                terminal_failed.append(task.task_id)
            elif task.filter is not None:
                finalize.append(task.task_id)
            elif task.raw is not None:
                filter_tasks.append(task.task_id)
            else:
                generate.append(task.task_id)
        else:
            raise ValueError(f"unknown formal resume state: {task.status!r}")
    return FormalResumePlan(
        generate_task_ids=tuple(generate),
        filter_task_ids=tuple(filter_tasks),
        finalize_task_ids=tuple(finalize),
        completed_task_ids=tuple(completed),
        terminal_failed_task_ids=tuple(terminal_failed),
    )


def build_formal_progress(
    state: FormalRunState,
    *,
    plan: FormalTaskPlan,
    runtime: FormalProgressRuntime,
) -> FormalProgressSnapshot:
    """Build the trustworthy progress.json payload from committed task state."""

    expected_ids = [task.task_id for task in plan.tasks]
    if [item.task_id for item in state.task_states] != expected_ids:
        raise ValueError("formal progress state differs from the task plan")
    if state.bindings.formal_plan_id != plan.formal_plan_id:
        raise ValueError("formal progress bindings differ from the task plan")

    statuses = Counter(item.status for item in state.task_states)
    raw_candidates = sum(
        item.raw.candidate_count for item in state.task_states if item.raw is not None
    )
    invalid_candidates = sum(len(item.invalid_candidates) for item in state.task_states)
    generated_candidates = raw_candidates + invalid_candidates
    accepted_candidates = sum(
        item.filter.accepted_count
        for item in state.task_states
        if item.filter is not None
    )
    rejected_candidates = invalid_candidates + sum(
        item.filter.rejected_count
        for item in state.task_states
        if item.filter is not None
    )
    stage_rejections: Counter[str] = Counter()
    by_skill_states: dict[str, list[FormalTaskState]] = defaultdict(list)
    for item in state.task_states:
        by_skill_states[item.skill_id].append(item)
        if item.filter is not None:
            stage_rejections.update(dict(item.filter.stage_rejection_counts))
        if item.invalid_candidates:
            stage_rejections["generation_invalid"] += len(item.invalid_candidates)

    task_counts = {"total": len(state.task_states)}
    task_counts.update(
        {status: statuses.get(status, 0) for status in FORMAL_TASK_STATE_STATUSES}
    )
    candidates = {
        "budget_total": plan.total_candidates,
        "generated": generated_candidates,
        "raw_stored": raw_candidates,
        "invalid_generation": invalid_candidates,
        "accepted": accepted_candidates,
        "rejected": rejected_candidates,
        "stage_rejection_counts": dict(sorted(stage_rejections.items())),
    }
    by_skill: dict[str, dict[str, Any]] = {}
    for skill_id, items in sorted(by_skill_states.items()):
        skill_statuses = Counter(item.status for item in items)
        by_skill[skill_id] = {
            "tasks_total": len(items),
            "tasks_by_status": {
                status: skill_statuses.get(status, 0)
                for status in FORMAL_TASK_STATE_STATUSES
            },
            "generated_candidates": sum(
                (0 if item.raw is None else item.raw.candidate_count)
                + len(item.invalid_candidates)
                for item in items
            ),
            "invalid_generation_candidates": sum(
                len(item.invalid_candidates) for item in items
            ),
            "accepted_candidates": sum(
                item.filter.accepted_count for item in items if item.filter is not None
            ),
            "rejected_candidates": sum(
                (0 if item.filter is None else item.filter.rejected_count)
                + len(item.invalid_candidates)
                for item in items
            ),
        }
    return FormalProgressSnapshot(
        formal_plan_id=state.bindings.formal_plan_id,
        task_plan_sha256=state.bindings.task_plan_sha256,
        bindings_sha256=state.bindings.semantic_sha256,
        state_snapshot_sha256=state.state_snapshot_sha256,
        tasks=task_counts,
        candidates=candidates,
        by_skill=by_skill,
        runtime=runtime,
    )


def write_formal_progress(
    directory: str | Path,
    progress: FormalProgressSnapshot,
) -> Path:
    """Atomically replace the small user-facing progress snapshot."""

    path = Path(directory) / FORMAL_PROGRESS_FILE_NAME
    _atomic_write(path, canonical_json_bytes(progress.to_dict(), indent=2))
    return path


__all__ = [
    "FORMAL_FAILURE_DIRECTORY_NAME",
    "FORMAL_INVALID_DIRECTORY_NAME",
    "FORMAL_PROGRESS_FILE_NAME",
    "FORMAL_STATE_DIRECTORY_NAME",
    "FORMAL_STATE_SCHEMA_VERSION",
    "FORMAL_STATE_SUMMARY_NAME",
    "FORMAL_TASK_STATE_STATUSES",
    "FormalFailureReference",
    "FormalFilterReference",
    "FormalInvalidCandidateReference",
    "FormalProgressRuntime",
    "FormalProgressSnapshot",
    "FormalRawRecovery",
    "FormalRawReference",
    "FormalResumePlan",
    "FormalRunState",
    "FormalStateBindings",
    "FormalStateShardCommit",
    "FormalTaskState",
    "FormalTaskStateStatus",
    "build_formal_progress",
    "build_formal_filter_references",
    "build_formal_resume_plan",
    "commit_formal_state_shard",
    "commit_formal_state_shards",
    "initialize_formal_state",
    "load_formal_state",
    "open_formal_state",
    "recover_generated_from_raw",
    "write_formal_failure",
    "write_formal_candidate_invalid",
    "write_formal_progress",
]

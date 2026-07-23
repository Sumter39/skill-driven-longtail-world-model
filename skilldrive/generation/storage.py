"""Atomic raw-overlay shards and lightweight filter indexes."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from skilldrive.generation.contracts import (
    FilterDecision,
    GeneratedCandidate,
    TaskStatus,
    candidate_id as make_candidate_id,
    canonical_json_bytes,
    canonical_sha256,
    filter_evaluation_id,
)


RAW_SHARD_SCHEMA_VERSION = 1
FILTER_INDEX_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RAW_METADATA_FIELDS = {
    "candidate_id",
    "task_id",
    "candidate_index",
    "latent_seed",
    "scenario_id",
    "skill_id",
    "proposal_mode",
    "target_track_id",
    "checkpoint_sha256",
    "semantic_config_sha256",
    "raw_offset",
    "metadata",
}


class RawShardError(ValueError):
    """A committed raw shard is missing, corrupt, or contract-incompatible."""


@dataclass(frozen=True)
class RawCandidateReference:
    candidate_id: str
    task_id: str
    candidate_index: int
    latent_seed: int
    raw_offset: int
    arrays_path: Path
    metadata_path: Path
    commit_path: Path


@dataclass(frozen=True)
class StoredRawCandidate:
    """One verified raw overlay reconstructed without re-running the model."""

    candidate_id: str
    task_id: str
    candidate_index: int
    latent_seed: int
    scenario_id: str
    skill_id: str
    proposal_mode: str
    target_track_id: str
    checkpoint_sha256: str
    semantic_config_sha256: str
    future_xy_global: np.ndarray
    metadata: Mapping[str, Any]
    reference: RawCandidateReference


@dataclass(frozen=True)
class RawShardCommit:
    shard_index: int
    semantic_config_sha256: str
    execution_config_sha256: str
    candidate_count: int
    arrays_sha256: str
    metadata_sha256: str
    candidate_ids_sha256: str
    arrays_path: Path
    metadata_path: Path
    commit_path: Path
    references: tuple[RawCandidateReference, ...]


@dataclass(frozen=True)
class RawRecoveryIssue:
    commit_path: Path
    reason: str


@dataclass(frozen=True)
class RawRecoveryScan:
    valid_shards: tuple[RawShardCommit, ...]
    invalid_shards: tuple[RawRecoveryIssue, ...]
    orphaned_files: tuple[Path, ...]

    @property
    def candidate_count(self) -> int:
        return sum(shard.candidate_count for shard in self.valid_shards)


@dataclass(frozen=True)
class FilterIndexSummary:
    filter_config_sha256: str
    filter_contract_version: str | int
    accepted_path: Path
    rejected_path: Path
    commit_path: Path
    accepted_count: int
    rejected_count: int
    task_statuses: Mapping[str, TaskStatus]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RawShardError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value, indent=2))


def _gzip_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row) + b"\n")
    return buffer.getvalue()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _raw_paths(directory: Path, shard_index: int) -> tuple[Path, Path, Path]:
    stem = f"shard-{shard_index:05d}"
    return (
        directory / f"{stem}.npz",
        directory / f"{stem}.meta.jsonl.gz",
        directory / f"{stem}.commit.json",
    )


def _safe_child(directory: Path, name: Any, field: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise RawShardError(f"{field} must contain one relative file name")
    return directory / name


def _candidate_metadata(candidate: GeneratedCandidate, raw_offset: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "task_id": candidate.task_id,
        "candidate_index": candidate.candidate_index,
        "latent_seed": candidate.latent_seed,
        "scenario_id": candidate.scenario_id,
        "skill_id": candidate.skill_id,
        "proposal_mode": candidate.proposal_mode,
        "target_track_id": candidate.overlay.target_track_id,
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "semantic_config_sha256": candidate.semantic_config_sha256,
        "raw_offset": raw_offset,
        "metadata": dict(candidate.metadata),
    }


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise RawShardError(
                        f"raw metadata contains a blank line at {line_number}: {path}"
                    )
                try:
                    value = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise RawShardError(
                        f"invalid raw metadata line {line_number}: {path}: {error}"
                    ) from error
                if not isinstance(value, dict) or set(value) != _RAW_METADATA_FIELDS:
                    raise RawShardError(
                        f"raw metadata line {line_number} has invalid fields: {path}"
                    )
                canonical_json_bytes(value)
                rows.append(value)
    except (OSError, EOFError) as error:
        raise RawShardError(f"failed to read raw metadata {path}: {error}") from error
    return rows


def _read_arrays(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"future_xy_global"}:
                raise RawShardError(f"invalid raw array fields: {path}")
            futures = payload["future_xy_global"]
    except RawShardError:
        raise
    except Exception as error:
        raise RawShardError(f"failed to read raw arrays {path}: {error}") from error
    if futures.dtype != np.float32 or futures.ndim != 3 or futures.shape[1:] != (60, 2):
        raise RawShardError(
            f"future_xy_global must have dtype float32 and shape [N, 60, 2]: {path}"
        )
    if not np.isfinite(futures).all():
        raise RawShardError(f"future_xy_global contains non-finite values: {path}")
    return futures


def _write_npz(path: Path, futures: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, future_xy_global=futures)
            handle.flush()
            os.fsync(handle.fileno())
        _read_arrays(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_raw_shard(
    directory: str | Path,
    shard_index: int,
    candidates: Sequence[GeneratedCandidate],
    *,
    semantic_config_sha256: str,
    execution_config_sha256: str,
) -> RawShardCommit:
    """Atomically commit arrays and metadata, publishing the sidecar last."""

    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("shard_index must be a nonnegative integer")
    if not candidates:
        raise ValueError("a raw shard must contain at least one candidate")
    semantic = _valid_sha256(semantic_config_sha256, "semantic_config_sha256")
    execution = _valid_sha256(execution_config_sha256, "execution_config_sha256")
    if any(candidate.semantic_config_sha256 != semantic for candidate in candidates):
        raise ValueError("candidate semantic_config_sha256 differs from the raw shard")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("raw shard candidate IDs must be unique")

    root = Path(directory)
    arrays_path, metadata_path, commit_path = _raw_paths(root, shard_index)
    futures = np.stack(
        [candidate.overlay.future_xy_global for candidate in candidates],
        axis=0,
    ).astype(np.float32, copy=False)
    metadata_rows = [
        _candidate_metadata(candidate, raw_offset)
        for raw_offset, candidate in enumerate(candidates)
    ]

    _write_npz(arrays_path, futures)
    _atomic_write_bytes(metadata_path, _gzip_jsonl(metadata_rows))
    loaded_metadata = _read_metadata(metadata_path)
    if len(loaded_metadata) != len(candidates):
        raise RawShardError("written raw metadata count differs from candidate count")

    commit = {
        "schema_version": RAW_SHARD_SCHEMA_VERSION,
        "kind": "raw_shard_commit",
        "shard_index": shard_index,
        "semantic_config_sha256": semantic,
        "execution_config_sha256": execution,
        "candidate_count": len(candidates),
        "candidate_ids_sha256": canonical_sha256(candidate_ids),
        "files": {
            "arrays": {
                "path": arrays_path.name,
                "size_bytes": arrays_path.stat().st_size,
                "sha256": _sha256(arrays_path),
            },
            "metadata": {
                "path": metadata_path.name,
                "size_bytes": metadata_path.stat().st_size,
                "sha256": _sha256(metadata_path),
            },
        },
    }
    _atomic_write_json(commit_path, commit)
    return verify_raw_shard(
        commit_path,
        expected_semantic_config_sha256=semantic,
    )


def _read_commit(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RawShardError(f"failed to read raw commit {path}: {error}") from error
    required = {
        "schema_version",
        "kind",
        "shard_index",
        "semantic_config_sha256",
        "execution_config_sha256",
        "candidate_count",
        "candidate_ids_sha256",
        "files",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RawShardError(f"raw commit has invalid fields: {path}")
    canonical_json_bytes(value)
    return value


def verify_raw_shard(
    commit_path: str | Path,
    *,
    expected_semantic_config_sha256: str | None = None,
) -> RawShardCommit:
    """Verify hashes, arrays, metadata, offsets, and semantic identity."""

    path = Path(commit_path)
    value = _read_commit(path)
    if value["schema_version"] != RAW_SHARD_SCHEMA_VERSION:
        raise RawShardError(f"unsupported raw shard schema version: {path}")
    if value["kind"] != "raw_shard_commit":
        raise RawShardError(f"invalid raw shard commit kind: {path}")
    shard_index = value["shard_index"]
    candidate_count = value["candidate_count"]
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
    ):
        raise RawShardError(f"raw shard index is invalid: {path}")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise RawShardError(f"raw candidate count is invalid: {path}")
    semantic = _valid_sha256(
        value["semantic_config_sha256"],
        "semantic_config_sha256",
    )
    execution = _valid_sha256(
        value["execution_config_sha256"],
        "execution_config_sha256",
    )
    if expected_semantic_config_sha256 is not None:
        expected = _valid_sha256(
            expected_semantic_config_sha256,
            "expected_semantic_config_sha256",
        )
        if semantic != expected:
            raise RawShardError(
                f"raw semantic configuration differs: shard={semantic}, expected={expected}"
            )
    candidate_ids_sha256 = _valid_sha256(
        value["candidate_ids_sha256"],
        "candidate_ids_sha256",
    )
    files = value["files"]
    if not isinstance(files, dict) or set(files) != {"arrays", "metadata"}:
        raise RawShardError(f"raw commit files are invalid: {path}")

    resolved: dict[str, tuple[Path, str]] = {}
    for name in ("arrays", "metadata"):
        descriptor = files[name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RawShardError(f"raw {name} descriptor is invalid: {path}")
        file_path = _safe_child(path.parent, descriptor["path"], f"files.{name}.path")
        size = descriptor["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RawShardError(f"raw {name} size is invalid: {path}")
        expected_sha = _valid_sha256(descriptor["sha256"], f"files.{name}.sha256")
        try:
            actual_size = file_path.stat().st_size
        except OSError as error:
            raise RawShardError(f"raw {name} file is missing: {file_path}") from error
        if actual_size != size:
            raise RawShardError(f"raw {name} size differs: {file_path}")
        if _sha256(file_path) != expected_sha:
            raise RawShardError(f"raw {name} SHA-256 differs: {file_path}")
        resolved[name] = (file_path, expected_sha)

    futures = _read_arrays(resolved["arrays"][0])
    metadata = _read_metadata(resolved["metadata"][0])
    if len(futures) != candidate_count or len(metadata) != candidate_count:
        raise RawShardError(f"raw candidate counts differ: {path}")
    candidate_ids: list[str] = []
    references: list[RawCandidateReference] = []
    for offset, row in enumerate(metadata):
        if row["raw_offset"] != offset:
            raise RawShardError(f"raw metadata offset differs at {offset}: {path}")
        candidate_value = _valid_sha256(row["candidate_id"], "candidate_id")
        task_value = _valid_sha256(row["task_id"], "task_id")
        checkpoint_value = _valid_sha256(
            row["checkpoint_sha256"],
            "checkpoint_sha256",
        )
        for name in ("scenario_id", "skill_id", "proposal_mode", "target_track_id"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise RawShardError(f"raw metadata {name} is invalid: {path}")
        if row["semantic_config_sha256"] != semantic:
            raise RawShardError(f"raw metadata semantic contract differs: {path}")
        try:
            expected_candidate_id = make_candidate_id(
                task_id=task_value,
                candidate_index=row["candidate_index"],
                latent_seed=row["latent_seed"],
                checkpoint_sha256=checkpoint_value,
                semantic_config_sha256=semantic,
            )
        except ValueError as error:
            raise RawShardError(f"raw candidate metadata is invalid: {path}: {error}") from error
        if candidate_value != expected_candidate_id:
            raise RawShardError(f"raw candidate ID differs from its metadata: {path}")
        candidate_ids.append(candidate_value)
        references.append(
            RawCandidateReference(
                candidate_id=candidate_value,
                task_id=task_value,
                candidate_index=row["candidate_index"],
                latent_seed=row["latent_seed"],
                raw_offset=offset,
                arrays_path=resolved["arrays"][0],
                metadata_path=resolved["metadata"][0],
                commit_path=path,
            )
        )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RawShardError(f"raw candidate IDs are duplicated: {path}")
    if canonical_sha256(candidate_ids) != candidate_ids_sha256:
        raise RawShardError(f"raw candidate ID digest differs: {path}")
    return RawShardCommit(
        shard_index=shard_index,
        semantic_config_sha256=semantic,
        execution_config_sha256=execution,
        candidate_count=candidate_count,
        arrays_sha256=resolved["arrays"][1],
        metadata_sha256=resolved["metadata"][1],
        candidate_ids_sha256=candidate_ids_sha256,
        arrays_path=resolved["arrays"][0],
        metadata_path=resolved["metadata"][0],
        commit_path=path,
        references=tuple(references),
    )


def scan_raw_shards(
    directory: str | Path,
    *,
    expected_semantic_config_sha256: str | None = None,
) -> RawRecoveryScan:
    """Classify committed shards and orphan files for deterministic recovery."""

    root = Path(directory)
    valid: list[RawShardCommit] = []
    invalid: list[RawRecoveryIssue] = []
    referenced: set[Path] = set()
    for commit_path in sorted(root.glob("shard-*.commit.json")):
        try:
            shard = verify_raw_shard(
                commit_path,
                expected_semantic_config_sha256=expected_semantic_config_sha256,
            )
        except RawShardError as error:
            invalid.append(RawRecoveryIssue(commit_path=commit_path, reason=str(error)))
            continue
        valid.append(shard)
        referenced.update((shard.arrays_path, shard.metadata_path, shard.commit_path))
    indices = [shard.shard_index for shard in valid]
    if len(set(indices)) != len(indices):
        raise RawShardError("multiple valid raw commits use the same shard index")
    possible_orphans = set(root.glob("shard-*.npz")) | set(
        root.glob("shard-*.meta.jsonl.gz")
    )
    orphaned = tuple(sorted(possible_orphans - referenced))
    return RawRecoveryScan(
        valid_shards=tuple(sorted(valid, key=lambda shard: shard.shard_index)),
        invalid_shards=tuple(invalid),
        orphaned_files=orphaned,
    )


def load_raw_shard_candidates(
    shard: RawShardCommit | str | Path,
    *,
    expected_semantic_config_sha256: str | None = None,
) -> tuple[StoredRawCandidate, ...]:
    """Load every candidate from one hash-verified committed raw shard."""

    commit = verify_raw_shard(
        shard.commit_path if isinstance(shard, RawShardCommit) else shard,
        expected_semantic_config_sha256=expected_semantic_config_sha256,
    )
    futures = _read_arrays(commit.arrays_path)
    metadata = _read_metadata(commit.metadata_path)
    if len(metadata) != len(commit.references):
        raise RawShardError("verified raw references and metadata counts differ")
    return tuple(
        StoredRawCandidate(
            candidate_id=row["candidate_id"],
            task_id=row["task_id"],
            candidate_index=row["candidate_index"],
            latent_seed=row["latent_seed"],
            scenario_id=row["scenario_id"],
            skill_id=row["skill_id"],
            proposal_mode=row["proposal_mode"],
            target_track_id=row["target_track_id"],
            checkpoint_sha256=row["checkpoint_sha256"],
            semantic_config_sha256=row["semantic_config_sha256"],
            future_xy_global=np.ascontiguousarray(futures[index].copy()),
            metadata=MappingProxyType(dict(row["metadata"])),
            reference=commit.references[index],
        )
        for index, row in enumerate(metadata)
    )


def _relative_reference(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return str(path.resolve())


def write_filter_indexes(
    directory: str | Path,
    raw_shards: Iterable[RawShardCommit | str | Path],
    decisions: Iterable[FilterDecision],
    *,
    filter_config_sha256: str,
    filter_contract_version: str | int,
) -> FilterIndexSummary:
    """Write accepted/rejected references without duplicating raw trajectories."""

    root = Path(directory)
    filter_config = _valid_sha256(filter_config_sha256, "filter_config_sha256")
    commits = [
        verify_raw_shard(
            value.commit_path if isinstance(value, RawShardCommit) else value
        )
        for value in raw_shards
    ]
    references: dict[str, RawCandidateReference] = {}
    for commit in commits:
        for reference in commit.references:
            if reference.candidate_id in references:
                raise ValueError(
                    f"duplicate raw candidate reference: {reference.candidate_id}"
                )
            references[reference.candidate_id] = reference
    materialized = list(decisions)
    decision_by_candidate: dict[str, FilterDecision] = {}
    for decision in materialized:
        if decision.candidate_id in decision_by_candidate:
            raise ValueError(f"duplicate filter decision: {decision.candidate_id}")
        expected_id = filter_evaluation_id(
            candidate_id=decision.candidate_id,
            filter_config_sha256=filter_config,
            filter_contract_version=filter_contract_version,
        )
        if decision.filter_evaluation_id != expected_id:
            raise ValueError(
                f"filter decision contract differs for {decision.candidate_id}"
            )
        decision_by_candidate[decision.candidate_id] = decision
    if set(decision_by_candidate) != set(references):
        missing = sorted(set(references) - set(decision_by_candidate))
        unexpected = sorted(set(decision_by_candidate) - set(references))
        raise ValueError(
            "filter decisions must cover every raw candidate exactly once: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    task_candidates: dict[str, int] = {}
    task_decisions: dict[str, int] = {}
    for candidate_value, reference in sorted(
        references.items(),
        key=lambda item: (item[1].task_id, item[1].raw_offset, item[0]),
    ):
        task_candidates[reference.task_id] = task_candidates.get(reference.task_id, 0) + 1
        decision = decision_by_candidate[candidate_value]
        task_decisions[reference.task_id] = task_decisions.get(reference.task_id, 0) + 1
        row = {
            "candidate_id": candidate_value,
            "filter_evaluation_id": decision.filter_evaluation_id,
            "task_id": reference.task_id,
            "candidate_index": reference.candidate_index,
            "latent_seed": reference.latent_seed,
            "raw": {
                "commit": _relative_reference(reference.commit_path, root),
                "arrays": _relative_reference(reference.arrays_path, root),
                "metadata": _relative_reference(reference.metadata_path, root),
                "offset": reference.raw_offset,
            },
            "metrics": dict(decision.metrics),
        }
        if decision.accepted:
            accepted_rows.append(row)
        else:
            rejected_rows.append(
                {
                    **row,
                    "rejection_reasons": list(decision.rejection_reasons),
                    "primary_rejection_reason": decision.rejection_reasons[0],
                    "first_failed_stage": decision.metrics.get("first_failed_stage"),
                }
            )
    task_statuses: dict[str, TaskStatus] = {
        task_id: (
            "complete"
            if task_decisions.get(task_id, 0) == candidate_count
            else "filter_committed"
        )
        for task_id, candidate_count in sorted(task_candidates.items())
    }

    accepted_path = root / "accepted.jsonl"
    rejected_path = root / "rejected.jsonl"
    commit_path = root / "filter-index.commit.json"
    _atomic_write_bytes(accepted_path, _jsonl(accepted_rows))
    _atomic_write_bytes(rejected_path, _jsonl(rejected_rows))
    commit = {
        "schema_version": FILTER_INDEX_SCHEMA_VERSION,
        "kind": "filter_index_commit",
        "filter_config_sha256": filter_config,
        "filter_contract_version": filter_contract_version,
        "counts": {
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "tasks": len(task_statuses),
        },
        "task_statuses": task_statuses,
        "files": {
            "accepted": {
                "path": accepted_path.name,
                "size_bytes": accepted_path.stat().st_size,
                "sha256": _sha256(accepted_path),
            },
            "rejected": {
                "path": rejected_path.name,
                "size_bytes": rejected_path.stat().st_size,
                "sha256": _sha256(rejected_path),
            },
        },
    }
    _atomic_write_json(commit_path, commit)
    return FilterIndexSummary(
        filter_config_sha256=filter_config,
        filter_contract_version=filter_contract_version,
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        commit_path=commit_path,
        accepted_count=len(accepted_rows),
        rejected_count=len(rejected_rows),
        task_statuses=task_statuses,
    )


__all__ = [
    "FILTER_INDEX_SCHEMA_VERSION",
    "RAW_SHARD_SCHEMA_VERSION",
    "FilterIndexSummary",
    "RawCandidateReference",
    "RawRecoveryIssue",
    "RawRecoveryScan",
    "RawShardCommit",
    "RawShardError",
    "StoredRawCandidate",
    "load_raw_shard_candidates",
    "scan_raw_shards",
    "verify_raw_shard",
    "write_filter_indexes",
    "write_raw_shard",
]

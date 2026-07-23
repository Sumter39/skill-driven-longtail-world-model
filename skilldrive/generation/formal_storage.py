"""Atomic filter indexes bound to one immutable formal task plan."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Iterable, Mapping

from skilldrive.generation.contracts import (
    FilterDecision,
    canonical_json_bytes,
    canonical_sha256,
    filter_evaluation_id,
)
from skilldrive.generation.formal_state import FormalStateBindings
from skilldrive.generation.storage import (
    FILTER_INDEX_SCHEMA_VERSION,
    RawCandidateReference,
    RawShardCommit,
    verify_raw_shard,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"formal artifact escapes its run root: {path}") from error


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


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def write_formal_filter_indexes(
    directory: str | Path,
    raw_shards: Iterable[RawShardCommit | str | Path],
    decisions: Iterable[FilterDecision],
    *,
    artifact_root: str | Path,
    bindings: FormalStateBindings,
) -> Path:
    """Write one complete, plan-bound filter commit for a task partition."""

    root = Path(artifact_root).resolve()
    output = Path(directory).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("formal filter output must remain inside artifact_root") from error

    commits = tuple(
        verify_raw_shard(
            value.commit_path if isinstance(value, RawShardCommit) else value,
            expected_semantic_config_sha256=bindings.generation_semantic_sha256,
        )
        for value in raw_shards
    )
    if not commits:
        raise ValueError("formal filter indexes require at least one raw shard")

    references: dict[str, RawCandidateReference] = {}
    for commit in commits:
        if commit.execution_config_sha256 != bindings.generation_execution_config_sha256:
            raise ValueError("formal raw execution config differs from state bindings")
        for reference in commit.references:
            if reference.candidate_id in references:
                raise ValueError("formal raw shards contain duplicate candidate IDs")
            references[reference.candidate_id] = reference

    materialized = tuple(decisions)
    by_candidate: dict[str, FilterDecision] = {}
    for decision in materialized:
        if decision.candidate_id in by_candidate:
            raise ValueError("formal filter decisions contain duplicate candidate IDs")
        expected_id = filter_evaluation_id(
            candidate_id=decision.candidate_id,
            filter_config_sha256=bindings.filter_config_sha256,
            filter_contract_version=bindings.filter_contract_version,
        )
        if decision.filter_evaluation_id != expected_id:
            raise ValueError("formal filter decision uses a different filter contract")
        by_candidate[decision.candidate_id] = decision
    if set(by_candidate) != set(references):
        raise ValueError("formal filter decisions must cover every raw candidate exactly once")

    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    task_ids: set[str] = set()
    for candidate_value, reference in sorted(
        references.items(),
        key=lambda item: (item[1].task_id, item[1].candidate_index, item[0]),
    ):
        decision = by_candidate[candidate_value]
        task_ids.add(reference.task_id)
        row: dict[str, object] = {
            "candidate_id": candidate_value,
            "filter_evaluation_id": decision.filter_evaluation_id,
            "task_id": reference.task_id,
            "candidate_index": reference.candidate_index,
            "latent_seed": reference.latent_seed,
            "raw": {
                "commit": _relative(reference.commit_path, root),
                "arrays": _relative(reference.arrays_path, root),
                "metadata": _relative(reference.metadata_path, root),
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

    accepted_path = output / "accepted.jsonl"
    rejected_path = output / "rejected.jsonl"
    _atomic_write(
        accepted_path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in accepted_rows),
    )
    _atomic_write(
        rejected_path,
        b"".join(canonical_json_bytes(row) + b"\n" for row in rejected_rows),
    )
    commit_path = output / "filter-index.commit.json"
    commit: Mapping[str, object] = {
        "schema_version": FILTER_INDEX_SCHEMA_VERSION,
        "kind": "formal_filter_commit",
        "formal_plan_id": bindings.formal_plan_id,
        "task_plan_sha256": bindings.task_plan_sha256,
        "filter_config_sha256": bindings.filter_config_sha256,
        "filter_contract_version": bindings.filter_contract_version,
        "raw_commits": [
            {
                "path": _relative(commit.commit_path, root),
                "sha256": _sha256(commit.commit_path),
            }
            for commit in sorted(commits, key=lambda item: item.shard_index)
        ],
        "decision_sha256": canonical_sha256(
            {"accepted": accepted_rows, "rejected": rejected_rows}
        ),
        "counts": {
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "tasks": len(task_ids),
        },
        "task_statuses": {task_id: "complete" for task_id in sorted(task_ids)},
        "files": {
            "accepted": _descriptor(accepted_path),
            "rejected": _descriptor(rejected_path),
        },
    }
    _atomic_write(commit_path, canonical_json_bytes(commit, indent=2))
    return commit_path


__all__ = ["write_formal_filter_indexes"]

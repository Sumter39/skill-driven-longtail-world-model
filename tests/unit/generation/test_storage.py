from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from skilldrive.generation.contracts import (
    FilterDecision,
    FilterRejection,
    GeneratedCandidate,
    GeneratedOverlay,
)
from skilldrive.generation.storage import (
    load_raw_shard_candidates,
    scan_raw_shards,
    verify_raw_shard,
    write_filter_indexes,
    write_raw_shard,
)


CHECKPOINT_SHA = "a" * 64
SEMANTIC_SHA = "b" * 64
EXECUTION_SHA = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(task_id: str, index: int) -> GeneratedCandidate:
    future = np.column_stack(
        (
            np.linspace(0.0, 10.0 + index, 60),
            np.full(60, float(index)),
        )
    )
    return GeneratedCandidate(
        task_id=task_id,
        candidate_index=index,
        latent_seed=100 + index,
        scenario_id="scene",
        skill_id="skill",
        proposal_mode="rule_guided_prior_search",
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256=SEMANTIC_SHA,
        overlay=GeneratedOverlay(
            target_track_id="target",
            future_xy_global=future,
        ),
        metadata={"candidate_rank": index},
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_recovery_scan_reports_corrupt_sidecar_and_orphaned_raw_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    commit = write_raw_shard(
        raw_dir,
        0,
        [_candidate("d" * 64, 0)],
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )
    commit.commit_path.write_text("{broken", encoding="utf-8")

    recovery = scan_raw_shards(
        raw_dir,
        expected_semantic_config_sha256=SEMANTIC_SHA,
    )
    assert recovery.valid_shards == ()
    assert len(recovery.invalid_shards) == 1
    assert "failed to read raw commit" in recovery.invalid_shards[0].reason
    assert set(recovery.orphaned_files) == {
        commit.arrays_path,
        commit.metadata_path,
    }


def test_verified_raw_candidates_can_be_reloaded_without_model_inference(
    tmp_path: Path,
) -> None:
    candidates = [_candidate("9" * 64, 0), _candidate("9" * 64, 1)]
    commit = write_raw_shard(
        tmp_path / "raw",
        0,
        candidates,
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )

    loaded = load_raw_shard_candidates(
        commit,
        expected_semantic_config_sha256=SEMANTIC_SHA,
    )

    assert [item.candidate_id for item in loaded] == [
        candidate.candidate_id for candidate in candidates
    ]
    assert [item.candidate_index for item in loaded] == [0, 1]
    np.testing.assert_array_equal(
        loaded[1].future_xy_global,
        candidates[1].overlay.future_xy_global,
    )
    assert loaded[1].metadata == {"candidate_rank": 1}
    assert loaded[1].reference.raw_offset == 1


def test_recovery_scan_rejects_raw_file_with_wrong_sidecar_hash(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    commit = write_raw_shard(
        raw_dir,
        0,
        [_candidate("4" * 64, 0)],
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )
    commit.arrays_path.write_bytes(b"corrupt")

    recovery = scan_raw_shards(raw_dir)
    assert recovery.valid_shards == ()
    assert len(recovery.invalid_shards) == 1
    assert "size differs" in recovery.invalid_shards[0].reason


def test_filter_only_reconfiguration_reuses_raw_shard(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    candidate = _candidate("e" * 64, 0)
    commit = write_raw_shard(
        raw_dir,
        0,
        [candidate],
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )
    raw_hashes = (_sha256(commit.arrays_path), _sha256(commit.metadata_path))
    raw_mtimes = (
        commit.arrays_path.stat().st_mtime_ns,
        commit.metadata_path.stat().st_mtime_ns,
    )

    first_filter = "1" * 64
    write_filter_indexes(
        tmp_path / "filter-v1",
        [commit],
        [
            FilterDecision.create(
                candidate_id=candidate.candidate_id,
                filter_config_sha256=first_filter,
                filter_contract_version=1,
                accepted=True,
            )
        ],
        filter_config_sha256=first_filter,
        filter_contract_version=1,
    )
    second_filter = "2" * 64
    write_filter_indexes(
        tmp_path / "filter-v2",
        [commit],
        [
            FilterDecision.create(
                candidate_id=candidate.candidate_id,
                filter_config_sha256=second_filter,
                filter_contract_version=2,
                accepted=False,
                rejection_reasons=(FilterRejection.RISK_OUT_OF_TARGET_RANGE,),
            )
        ],
        filter_config_sha256=second_filter,
        filter_contract_version=2,
    )

    assert raw_hashes == (_sha256(commit.arrays_path), _sha256(commit.metadata_path))
    assert raw_mtimes == (
        commit.arrays_path.stat().st_mtime_ns,
        commit.metadata_path.stat().st_mtime_ns,
    )
    assert verify_raw_shard(
        commit.commit_path,
        expected_semantic_config_sha256=SEMANTIC_SHA,
    ).execution_config_sha256 == EXECUTION_SHA


def test_mixed_accept_and_reject_candidates_leave_task_complete(tmp_path: Path) -> None:
    task_id = "f" * 64
    candidates = [_candidate(task_id, 0), _candidate(task_id, 1)]
    commit = write_raw_shard(
        tmp_path / "raw",
        0,
        candidates,
        semantic_config_sha256=SEMANTIC_SHA,
        execution_config_sha256=EXECUTION_SHA,
    )
    filter_sha = "3" * 64
    summary = write_filter_indexes(
        tmp_path / "filter",
        [commit],
        [
            FilterDecision.create(
                candidate_id=candidates[0].candidate_id,
                filter_config_sha256=filter_sha,
                filter_contract_version="v1",
                accepted=True,
                metrics={"score": 0.9},
            ),
            FilterDecision.create(
                candidate_id=candidates[1].candidate_id,
                filter_config_sha256=filter_sha,
                filter_contract_version="v1",
                accepted=False,
                rejection_reasons=(FilterRejection.COLLISION_PROXY_OVERLAP,),
                metrics={"score": 0.1},
            ),
        ],
        filter_config_sha256=filter_sha,
        filter_contract_version="v1",
    )

    assert summary.task_statuses == {task_id: "complete"}
    assert summary.accepted_count == 1
    assert summary.rejected_count == 1
    accepted = _read_jsonl(summary.accepted_path)
    rejected = _read_jsonl(summary.rejected_path)
    assert accepted[0]["raw"]["offset"] == 0
    assert rejected[0]["raw"]["offset"] == 1
    assert rejected[0]["rejection_reasons"] == [
        FilterRejection.COLLISION_PROXY_OVERLAP.value
    ]
    assert rejected[0]["primary_rejection_reason"] == (
        FilterRejection.COLLISION_PROXY_OVERLAP.value
    )
    assert rejected[0]["candidate_index"] == 1
    assert rejected[0]["latent_seed"] == 101
    for row in (*accepted, *rejected):
        assert set(row["raw"]) == {"commit", "arrays", "metadata", "offset"}
        assert "future_xy_global" not in row

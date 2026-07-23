from __future__ import annotations

import numpy as np
import pytest

from skilldrive.generation.contracts import (
    GeneratedCandidate,
    GeneratedOverlay,
    GenerationConfigFingerprints,
    GenerationTask,
    candidate_id,
    canonical_sha256,
)


CHECKPOINT_SHA = "a" * 64


def test_canonical_hash_and_candidate_id_ignore_execution_configuration() -> None:
    first = GenerationConfigFingerprints.from_configs(
        {"candidate_budget": 16, "checkpoint": CHECKPOINT_SHA},
        {"batch_size": 32, "workers": 2},
    )
    reordered = GenerationConfigFingerprints.from_configs(
        {"checkpoint": CHECKPOINT_SHA, "candidate_budget": 16},
        {"workers": 8, "batch_size": 64},
    )
    assert first.semantic_config_sha256 == reordered.semantic_config_sha256
    assert first.execution_config_sha256 != reordered.execution_config_sha256
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256(
        {"a": 1, "b": 2}
    )

    task_id = "b" * 64
    first_id = candidate_id(
        task_id=task_id,
        candidate_index=0,
        latent_seed=2026,
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256=first.semantic_config_sha256,
    )
    second_id = candidate_id(
        task_id=task_id,
        candidate_index=0,
        latent_seed=2026,
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256=reordered.semantic_config_sha256,
    )
    assert first_id == second_id


def test_generation_task_enforces_ordered_status_transitions() -> None:
    task = GenerationTask.create(
        task_index=0,
        seed_record_id="c" * 64,
        scenario_id="scene",
        skill_id="skill",
        target_track_id="target",
        proposal_mode="rule_guided_prior_search",
        condition_skill_id="<none>",
        candidate_budget=2,
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256="d" * 64,
    )
    assert task.status == "pending"
    task = task.transition("raw_committed")
    task = task.transition("filter_committed")
    task = task.transition("complete")
    assert task.status == "complete"
    assert task.transition("complete") is task
    with pytest.raises(ValueError, match="invalid task status transition"):
        task.transition("failed")


def test_generated_candidate_uses_finite_single_target_overlay() -> None:
    candidate = GeneratedCandidate(
        task_id="e" * 64,
        candidate_index=1,
        latent_seed=7,
        scenario_id="scene",
        skill_id="skill",
        proposal_mode="learned_conditioned_prior",
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256="f" * 64,
        overlay=GeneratedOverlay(
            target_track_id="target",
            future_xy_global=np.zeros((60, 2), dtype=np.float64),
        ),
        metadata={"requested_parameters": {}},
    )
    assert candidate.overlay.future_xy_global.dtype == np.float32
    assert candidate.candidate_id == candidate_id(
        task_id=candidate.task_id,
        candidate_index=1,
        latent_seed=7,
        checkpoint_sha256=CHECKPOINT_SHA,
        semantic_config_sha256="f" * 64,
    )


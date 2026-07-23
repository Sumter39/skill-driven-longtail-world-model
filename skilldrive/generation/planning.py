"""Deterministic task planning for counterfactual Prior generation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np

from skilldrive.data import PriorContextSpec
from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.generation.contracts import (
    MAX_LATENT_SEED,
    GenerationTask,
    canonical_sha256,
)
from skilldrive.seeds.records import SeedRecord


PilotEvaluationArm = Literal[
    "learned_conditioned",
    "learned_none_control",
    "rule_guided_none",
]


@dataclass(frozen=True)
class PilotEligibilityDecision:
    """One attempted Formal Train seed eligibility result."""

    skill_id: str
    scenario_id: str
    seed_record_id: str
    context_fingerprint: str
    candidate_rank: int
    eligible: bool
    cache_hit: bool
    failure_type: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True)
class PilotEligibilitySelection:
    """Stable per-skill selection after deterministic ineligible-seed replacement."""

    records: tuple[SeedRecord, ...]
    decisions: tuple[PilotEligibilityDecision, ...]


def seed_record_id(record: SeedRecord) -> str:
    """Fingerprint the complete normalized seed row, including evidence and requests."""

    return canonical_sha256(record.to_csv_row())


def semantic_generation_config_sha256(config: CounterfactualGenerationConfig) -> str:
    """Hash generation semantics while excluding budgets and execution tuning knobs."""

    return canonical_sha256(
        {
            "version": config.version,
            "contract_name": config.contract_name,
            "none_skill_id": config.none_skill_id,
            "checkpoint_sha256": config.active_checkpoint.sha256,
            "schema_sha256": config.active_checkpoint.schema_sha256,
            "seed_manifest_sha256": config.inputs.seed_manifest_sha256,
            "base_seed": config.sampling.base_seed,
            "skills": [
                {
                    "skill_id": item.skill_id,
                    "primary_generated_role": item.primary_generated_role,
                    "proposal_mode": item.proposal_mode,
                    "condition_skill_strategy": item.condition_skill_strategy,
                    "joint_generation_limited": item.joint_generation_limited,
                }
                for item in config.skills
            ],
        }
    )


def build_generation_task(
    *,
    task_index: int,
    record: SeedRecord,
    config: CounterfactualGenerationConfig,
    candidate_budget: int,
) -> GenerationTask:
    skill_config = config.skills_by_id.get(record.skill_id)
    if skill_config is None:
        raise ValueError(f"seed record references a non-formal skill: {record.skill_id}")
    try:
        target_track_id = record.role_track_ids[skill_config.primary_generated_role]
    except KeyError:
        raise ValueError(
            f"seed record for {record.skill_id} is missing primary role "
            f"{skill_config.primary_generated_role}"
        ) from None
    return GenerationTask.create(
        task_index=task_index,
        seed_record_id=seed_record_id(record),
        scenario_id=record.scenario_id,
        skill_id=record.skill_id,
        target_track_id=target_track_id,
        proposal_mode=skill_config.proposal_mode,
        condition_skill_id=skill_config.condition_skill_id(config.none_skill_id),
        candidate_budget=candidate_budget,
        checkpoint_sha256=config.active_checkpoint.sha256,
        semantic_config_sha256=semantic_generation_config_sha256(config),
    )


def prior_context_spec_for_task(
    task: GenerationTask,
    record: SeedRecord,
) -> PriorContextSpec:
    if task.scenario_id != record.scenario_id or task.skill_id != record.skill_id:
        raise ValueError("task and seed record identity differ")
    if task.seed_record_id != seed_record_id(record):
        raise ValueError("task seed_record_id differs from the seed record")
    all_context_tracks = tuple(
        sorted(set(record.role_track_ids.values()) - {task.target_track_id})
    )
    roles = (
        tuple(record.role_track_ids.items())
        if task.condition_skill_id != "<none>"
        else ()
    )
    return PriorContextSpec(
        scenario_id=task.scenario_id,
        target_track_id=task.target_track_id,
        condition_skill_id=task.condition_skill_id,
        required_context_track_ids=all_context_tracks,
        role_track_ids=roles,
    )


def prior_context_fingerprint(
    task: GenerationTask,
    record: SeedRecord,
) -> str:
    """Fingerprint every input that can change history-only Prior tensorization."""

    spec = prior_context_spec_for_task(task, record)
    return canonical_sha256(
        {
            "version": 1,
            "source_path": record.source_path,
            "scenario_id": spec.scenario_id,
            "target_track_id": spec.target_track_id,
            "condition_skill_id": spec.condition_skill_id,
            "required_context_track_ids": spec.required_context_track_ids,
            "role_track_ids": spec.role_track_ids,
        }
    )


def latent_seed(base_seed: int, task_id: str, candidate_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError("candidate_index must be a nonnegative integer")
    if candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    digest = hashlib.sha256(
        f"v1:{base_seed}:{task_id}:{candidate_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & MAX_LATENT_SEED


def latent_seeds_for_task(task: GenerationTask, *, base_seed: int) -> np.ndarray:
    return np.asarray(
        [
            latent_seed(base_seed, task.task_id, candidate_index)
            for candidate_index in range(task.candidate_budget)
        ],
        dtype=np.int64,
    )


def pilot_evaluation_arm(
    task: GenerationTask,
    *,
    none_skill_id: str = "<none>",
) -> PilotEvaluationArm:
    """Derive the Pilot arm without changing the stable task-plan schema."""

    if not isinstance(none_skill_id, str) or not none_skill_id:
        raise ValueError("none_skill_id must be a non-empty string")
    if task.proposal_mode == "learned_conditioned_prior":
        if task.condition_skill_id == task.skill_id:
            return "learned_conditioned"
        if task.condition_skill_id == none_skill_id:
            return "learned_none_control"
        raise ValueError(
            "learned Pilot task condition must be the requested skill or <none>"
        )
    if task.proposal_mode == "rule_guided_prior_search":
        if task.condition_skill_id != none_skill_id:
            raise ValueError("rule-guided Pilot task must use only the <none> condition")
        return "rule_guided_none"
    raise ValueError(f"unknown Pilot proposal mode: {task.proposal_mode!r}")


def latent_group_id(task: GenerationTask) -> str:
    """Identify tasks that must share reparameterization noise in Pilot controls."""

    return canonical_sha256(
        {
            "version": 1,
            "seed_record_id": task.seed_record_id,
            "scenario_id": task.scenario_id,
            "skill_id": task.skill_id,
            "target_track_id": task.target_track_id,
            "proposal_mode": task.proposal_mode,
            "checkpoint_sha256": task.checkpoint_sha256,
            "semantic_config_sha256": task.semantic_config_sha256,
        }
    )


def paired_latent_seed(
    base_seed: int,
    task: GenerationTask,
    candidate_index: int,
) -> int:
    """Return shared epsilon seed for one paired Pilot candidate index."""

    return latent_seed(base_seed, latent_group_id(task), candidate_index)


def paired_latent_seeds_for_task(
    task: GenerationTask,
    *,
    base_seed: int,
) -> np.ndarray:
    """Return deterministic Pilot seeds shared by both learned comparison arms."""

    return np.asarray(
        [
            paired_latent_seed(base_seed, task, candidate_index)
            for candidate_index in range(task.candidate_budget)
        ],
        dtype=np.int64,
    )


def select_pilot_records(
    records: Iterable[SeedRecord],
    *,
    formal_skill_ids: tuple[str, ...],
    per_skill: int,
    base_seed: int,
) -> list[SeedRecord]:
    ordered_by_skill = _ordered_pilot_records_by_skill(
        records,
        formal_skill_ids=formal_skill_ids,
        base_seed=base_seed,
    )
    if per_skill <= 0:
        raise ValueError("per_skill must be positive")
    return [
        record
        for skill_id in formal_skill_ids
        for record in ordered_by_skill[skill_id][:per_skill]
    ]


def _ordered_pilot_records_by_skill(
    records: Iterable[SeedRecord],
    *,
    formal_skill_ids: tuple[str, ...],
    base_seed: int,
) -> dict[str, list[SeedRecord]]:
    grouped: dict[str, list[SeedRecord]] = defaultdict(list)
    formal = set(formal_skill_ids)
    for record in records:
        if record.skill_id not in formal:
            raise ValueError(f"pilot record references a non-formal skill: {record.skill_id}")
        grouped[record.skill_id].append(record)
    missing = [skill_id for skill_id in formal_skill_ids if not grouped[skill_id]]
    if missing:
        raise ValueError(f"formal skills have no pilot records: {missing}")

    for skill_id in formal_skill_ids:
        grouped[skill_id] = sorted(
            grouped[skill_id],
            key=lambda record: canonical_sha256(
                {
                    "version": 1,
                    "base_seed": base_seed,
                    "seed_record_id": seed_record_id(record),
                }
            ),
        )
    return grouped


def select_eligible_pilot_records(
    records: Iterable[SeedRecord],
    *,
    formal_skill_ids: tuple[str, ...],
    per_skill: int,
    base_seed: int,
    context_fingerprint: Callable[[SeedRecord], str],
    validate_record: Callable[[SeedRecord], None],
) -> PilotEligibilitySelection:
    """Replace invalid seeds with later deterministic candidates from the same skill."""

    if per_skill <= 0:
        raise ValueError("per_skill must be positive")
    ordered_by_skill = _ordered_pilot_records_by_skill(
        records,
        formal_skill_ids=formal_skill_ids,
        base_seed=base_seed,
    )
    outcomes: dict[str, tuple[bool, str | None, str | None]] = {}
    selected: list[SeedRecord] = []
    decisions: list[PilotEligibilityDecision] = []

    for skill_id in formal_skill_ids:
        selected_for_skill = 0
        for candidate_rank, record in enumerate(ordered_by_skill[skill_id]):
            if selected_for_skill >= per_skill:
                break
            fingerprint = context_fingerprint(record)
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("Pilot context fingerprint must be a non-empty string")
            cache_hit = fingerprint in outcomes
            if cache_hit:
                eligible, failure_type, failure_message = outcomes[fingerprint]
            else:
                try:
                    validate_record(record)
                except ValueError as error:
                    eligible = False
                    failure_type = type(error).__name__
                    failure_message = str(error)
                else:
                    eligible = True
                    failure_type = None
                    failure_message = None
                outcomes[fingerprint] = (
                    eligible,
                    failure_type,
                    failure_message,
                )
            decisions.append(
                PilotEligibilityDecision(
                    skill_id=record.skill_id,
                    scenario_id=record.scenario_id,
                    seed_record_id=seed_record_id(record),
                    context_fingerprint=fingerprint,
                    candidate_rank=candidate_rank,
                    eligible=eligible,
                    cache_hit=cache_hit,
                    failure_type=failure_type,
                    failure_message=failure_message,
                )
            )
            if eligible:
                selected.append(record)
                selected_for_skill += 1

    return PilotEligibilitySelection(tuple(selected), tuple(decisions))


__all__ = [
    "PilotEligibilityDecision",
    "PilotEligibilitySelection",
    "PilotEvaluationArm",
    "build_generation_task",
    "latent_group_id",
    "latent_seed",
    "latent_seeds_for_task",
    "paired_latent_seed",
    "paired_latent_seeds_for_task",
    "pilot_evaluation_arm",
    "prior_context_fingerprint",
    "prior_context_spec_for_task",
    "seed_record_id",
    "select_eligible_pilot_records",
    "select_pilot_records",
    "semantic_generation_config_sha256",
]

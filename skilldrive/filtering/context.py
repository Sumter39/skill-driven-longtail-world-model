"""Unambiguous binding from raw overlays to tasks, seed rows, and source scenes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from skilldrive.data.coordinates import global_to_local
from skilldrive.generation.assembly import materialize_overlay_scenario
from skilldrive.generation.contracts import GenerationTask
from skilldrive.generation.planning import seed_record_id
from skilldrive.generation.storage import StoredRawCandidate
from skilldrive.schemas import Scenario, SkillSpec
from skilldrive.seeds.records import SeedRecord


@dataclass(frozen=True)
class BoundRawCandidate:
    raw: StoredRawCandidate
    task: GenerationTask
    seed_record: SeedRecord


@dataclass(frozen=True)
class CandidateEvaluationContext:
    raw: StoredRawCandidate
    task: GenerationTask
    seed_record: SeedRecord
    skill: SkillSpec
    source_scenario: Scenario
    generated_scenario: Scenario
    future_xy_local: np.ndarray
    anchor_origin_global: np.ndarray
    anchor_heading_global: float

    def __post_init__(self) -> None:
        local = np.asarray(self.future_xy_local, dtype=np.float64)
        origin = np.asarray(self.anchor_origin_global, dtype=np.float64)
        if local.shape != (60, 2) or not np.isfinite(local).all():
            raise ValueError("future_xy_local must be finite with shape (60, 2)")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("anchor_origin_global must be a finite (2,) point")
        if not math.isfinite(float(self.anchor_heading_global)):
            raise ValueError("anchor_heading_global must be finite")
        object.__setattr__(self, "future_xy_local", np.ascontiguousarray(local.copy()))
        object.__setattr__(self, "anchor_origin_global", origin.copy())
        object.__setattr__(self, "anchor_heading_global", float(self.anchor_heading_global))


def bind_raw_candidates(
    raw_candidates: Sequence[StoredRawCandidate],
    tasks: Iterable[GenerationTask],
    seed_records: Iterable[SeedRecord],
) -> tuple[BoundRawCandidate, ...]:
    task_rows = tuple(tasks)
    tasks_by_id = {task.task_id: task for task in task_rows}
    if len(tasks_by_id) != len(task_rows):
        raise ValueError("tasks must have unique task IDs")
    records_by_id: dict[str, SeedRecord] = {}
    for record in seed_records:
        record_id = seed_record_id(record)
        if record_id in records_by_id:
            raise ValueError(f"seed records contain a duplicate row: {record_id}")
        records_by_id[record_id] = record

    bound: list[BoundRawCandidate] = []
    for raw in raw_candidates:
        task = tasks_by_id.get(raw.task_id)
        if task is None:
            raise ValueError(f"raw candidate references an unknown task: {raw.task_id}")
        record = records_by_id.get(task.seed_record_id)
        if record is None:
            raise ValueError(
                f"task references an unknown seed record: {task.seed_record_id}"
            )
        expected = (
            task.scenario_id,
            task.skill_id,
            task.target_track_id,
            task.proposal_mode,
            task.checkpoint_sha256,
            task.semantic_config_sha256,
        )
        actual = (
            raw.scenario_id,
            raw.skill_id,
            raw.target_track_id,
            raw.proposal_mode,
            raw.checkpoint_sha256,
            raw.semantic_config_sha256,
        )
        if actual != expected:
            raise ValueError(f"raw candidate identity differs from task: {raw.candidate_id}")
        if record.scenario_id != task.scenario_id or record.skill_id != task.skill_id:
            raise ValueError(f"seed record identity differs from task: {task.task_id}")
        bound.append(BoundRawCandidate(raw=raw, task=task, seed_record=record))
    return tuple(bound)


def validate_bound_candidate_contract(
    bound: BoundRawCandidate,
    *,
    primary_generated_role: str,
) -> None:
    """Reject ambiguous or drifted raw metadata before quality filtering."""

    if not isinstance(primary_generated_role, str) or not primary_generated_role:
        raise ValueError("primary_generated_role must be a non-empty string")
    expected_target = bound.seed_record.role_track_ids.get(primary_generated_role)
    if expected_target != bound.task.target_track_id:
        raise ValueError("primary generated role does not resolve to the task target")
    metadata = bound.raw.metadata
    if metadata.get("condition_skill_id") != bound.task.condition_skill_id:
        raise ValueError("raw condition_skill_id differs from the generation task")
    if metadata.get("primary_generated_role") != primary_generated_role:
        raise ValueError("raw primary_generated_role differs from the frozen skill config")
    if metadata.get("requested_parameters") != bound.seed_record.sampled_parameters:
        raise ValueError("raw requested_parameters differ from the seed record")
    expected_mode = bound.seed_record.evidence.get("detection_mode")
    if metadata.get("detection_mode") != expected_mode:
        raise ValueError("raw detection_mode differs from the seed evidence")


def _anchor_frame(source_scenario: Scenario, target_track_id: str) -> tuple[np.ndarray, float]:
    target = next(
        (agent for agent in source_scenario.agents if agent.track_id == target_track_id),
        None,
    )
    if target is None or len(target.positions) < 50:
        raise ValueError("source target must contain frame 49")
    origin = np.asarray(target.positions[49], dtype=np.float64)
    if not np.isfinite(origin).all():
        raise ValueError("source target frame 49 position must be finite")
    heading = float(target.headings[49])
    if not math.isfinite(heading):
        velocity = target.velocities[49]
        if np.isfinite(velocity).all() and float(np.linalg.norm(velocity)) > 1e-6:
            heading = float(np.arctan2(velocity[1], velocity[0]))
        elif len(target.positions) >= 50:
            delta = target.positions[49] - target.positions[48]
            if np.isfinite(delta).all() and float(np.linalg.norm(delta)) > 1e-6:
                heading = float(np.arctan2(delta[1], delta[0]))
    if not math.isfinite(heading):
        raise ValueError("source target frame 49 heading cannot be resolved")
    return origin.copy(), heading


def build_candidate_evaluation_context(
    bound: BoundRawCandidate,
    *,
    skill: SkillSpec,
    source_scenario: Scenario,
) -> CandidateEvaluationContext:
    if source_scenario.scenario_id != bound.task.scenario_id:
        raise ValueError("loaded source scenario differs from the bound task")
    if skill.skill_id != bound.task.skill_id:
        raise ValueError("loaded skill differs from the bound task")
    origin, heading = _anchor_frame(source_scenario, bound.task.target_track_id)
    generated = materialize_overlay_scenario(
        source_scenario,
        bound.task.target_track_id,
        bound.raw.future_xy_global,
    )
    future_local = global_to_local(bound.raw.future_xy_global, origin, heading)
    return CandidateEvaluationContext(
        raw=bound.raw,
        task=bound.task,
        seed_record=bound.seed_record,
        skill=skill,
        source_scenario=source_scenario,
        generated_scenario=generated,
        future_xy_local=future_local,
        anchor_origin_global=origin,
        anchor_heading_global=heading,
    )


__all__ = [
    "BoundRawCandidate",
    "CandidateEvaluationContext",
    "bind_raw_candidates",
    "build_candidate_evaluation_context",
    "validate_bound_candidate_contract",
]

"""Small paired Prior capability contract for repair checkpoints."""

from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any, Iterable, Mapping

from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.generation.planning import pilot_evaluation_arm
from skilldrive.generation.scheduler import TaskPlan, build_paired_pilot_task_plan
from skilldrive.seeds.records import SeedRecord


REPAIR_SMOKE_CANDIDATES_PER_ARM = 8
REPAIR_SMOKE_SKILLS = (
    "slow_lead_blockage",
    "construction_object_lane_blockage",
)
REPAIR_SMOKE_ARMS = (
    "learned_conditioned",
    "learned_none_control",
    "rule_guided_none",
)
KINEMATIC_METRICS = (
    "seam_speed_mps",
    "maximum_speed_mps",
    "maximum_acceleration_mps2",
    "maximum_deceleration_mps2",
    "maximum_jerk_mps3",
    "maximum_curvature_per_m",
    "maximum_heading_rate_rad_s",
)


def build_repair_smoke_plan(
    records: Iterable[SeedRecord],
    config: CounterfactualGenerationConfig,
    *,
    execution_config: Mapping[str, Any],
) -> TaskPlan:
    """Build exactly three existing Pilot arms and 24 candidates."""

    selected = tuple(records)
    counts = Counter(record.skill_id for record in selected)
    expected = Counter({skill_id: 1 for skill_id in REPAIR_SMOKE_SKILLS})
    if counts != expected:
        raise ValueError(
            "repair smoke requires one seed for slow_lead_blockage and one for "
            "construction_object_lane_blockage"
        )
    plan = build_paired_pilot_task_plan(
        selected,
        config,
        execution_config=execution_config,
        per_skill=1,
        candidate_budget=REPAIR_SMOKE_CANDIDATES_PER_ARM,
        allow_missing_skills=True,
    )
    arms = Counter(
        pilot_evaluation_arm(task, none_skill_id=config.none_skill_id)
        for task in plan.tasks
    )
    if arms != Counter({arm: 1 for arm in REPAIR_SMOKE_ARMS}):
        raise RuntimeError(f"repair smoke arm contract mismatch: {dict(arms)}")
    if len(plan.tasks) != 3 or plan.total_candidates != 24:
        raise RuntimeError("repair smoke must contain exactly 3 tasks and 24 candidates")
    return plan


def _finite_numbers(rows: list[Mapping[str, Any]], metric: str) -> list[float]:
    result: list[float] = []
    for row in rows:
        value = row["kinematics"]["metrics"].get(metric)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            result.append(float(value))
    return result


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "minimum": None if not values else min(values),
        "median": None if not values else median(values),
        "maximum": None if not values else max(values),
    }


def _pair_outcome(conditioned_pass: bool, control_pass: bool) -> str:
    if conditioned_pass and control_pass:
        return "both_passed"
    if conditioned_pass:
        return "conditioned_only_passed"
    if control_pass:
        return "control_only_passed"
    return "neither_passed"


def summarize_repair_smoke_kinematics(
    evidence: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the fixed arms and verify the learned pair shares epsilon seeds."""

    rows = [dict(row) for row in evidence]
    if len(rows) != 24:
        raise ValueError("repair smoke requires exactly 24 kinematic evidence rows")
    by_arm: dict[str, list[Mapping[str, Any]]] = {
        arm: [] for arm in REPAIR_SMOKE_ARMS
    }
    for row in rows:
        arm = row.get("evaluation_arm")
        if arm not in by_arm:
            raise ValueError(f"unknown repair smoke evaluation arm: {arm!r}")
        kinematics = row.get("kinematics")
        if not isinstance(kinematics, Mapping):
            raise ValueError("repair smoke evidence is missing kinematics")
        if not isinstance(kinematics.get("passed"), bool):
            raise ValueError("repair smoke kinematics.passed must be boolean")
        by_arm[arm].append(row)
    if any(
        len(values) != 8
        or {int(row["candidate_index"]) for row in values} != set(range(8))
        for values in by_arm.values()
    ):
        raise ValueError("repair smoke requires exactly eight candidates per arm")
    if any(
        row.get("skill_id") != "slow_lead_blockage"
        for arm in ("learned_conditioned", "learned_none_control")
        for row in by_arm[arm]
    ) or any(
        row.get("skill_id") != "construction_object_lane_blockage"
        for row in by_arm["rule_guided_none"]
    ):
        raise ValueError("repair smoke arm and skill identities do not match")

    arm_summaries = []
    for arm in REPAIR_SMOKE_ARMS:
        arm_rows = by_arm[arm]
        rejection_counts = Counter(
            reason
            for row in arm_rows
            for reason in row["kinematics"].get("rejection_reasons", ())
        )
        passed = sum(bool(row["kinematics"]["passed"]) for row in arm_rows)
        arm_summaries.append(
            {
                "evaluation_arm": arm,
                "candidate_count": len(arm_rows),
                "kinematic_passed": passed,
                "kinematic_rejected": len(arm_rows) - passed,
                "rejection_reasons": dict(sorted(rejection_counts.items())),
                "metrics": {
                    metric: _numeric_summary(_finite_numbers(arm_rows, metric))
                    for metric in KINEMATIC_METRICS
                },
            }
        )

    conditioned = {
        int(row["candidate_index"]): row
        for row in by_arm["learned_conditioned"]
    }
    control = {
        int(row["candidate_index"]): row
        for row in by_arm["learned_none_control"]
    }
    outcomes = Counter()
    metric_deltas: dict[str, list[float]] = {
        metric: [] for metric in KINEMATIC_METRICS
    }
    pair_rows = []
    for candidate_index in range(8):
        conditioned_row = conditioned[candidate_index]
        control_row = control[candidate_index]
        if conditioned_row["latent_seed"] != control_row["latent_seed"]:
            raise ValueError("paired repair smoke candidates do not share epsilon seeds")
        if conditioned_row["latent_group_id"] != control_row["latent_group_id"]:
            raise ValueError("paired repair smoke candidates do not share a latent group")
        conditioned_pass = bool(conditioned_row["kinematics"]["passed"])
        control_pass = bool(control_row["kinematics"]["passed"])
        outcome = _pair_outcome(conditioned_pass, control_pass)
        outcomes[outcome] += 1
        for metric in KINEMATIC_METRICS:
            conditioned_value = conditioned_row["kinematics"]["metrics"].get(metric)
            control_value = control_row["kinematics"]["metrics"].get(metric)
            if all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (conditioned_value, control_value)
            ):
                metric_deltas[metric].append(
                    float(conditioned_value) - float(control_value)
                )
        pair_rows.append(
            {
                "candidate_index": candidate_index,
                "latent_seed": int(conditioned_row["latent_seed"]),
                "latent_group_id": conditioned_row["latent_group_id"],
                "conditioned_candidate_id": conditioned_row["candidate_id"],
                "control_candidate_id": control_row["candidate_id"],
                "outcome": outcome,
            }
        )

    return {
        "by_arm": arm_summaries,
        "paired_control": {
            "skill_id": "slow_lead_blockage",
            "pair_count": 8,
            "shared_epsilon_pair_count": 8,
            "epsilon_contract": "paired_standard_normal_epsilon_v1",
            "outcomes": {
                name: outcomes[name]
                for name in (
                    "both_passed",
                    "conditioned_only_passed",
                    "control_only_passed",
                    "neither_passed",
                )
            },
            "metric_delta_conditioned_minus_control": {
                metric: _numeric_summary(values)
                for metric, values in metric_deltas.items()
            },
            "pairs": pair_rows,
        },
    }


__all__ = [
    "KINEMATIC_METRICS",
    "REPAIR_SMOKE_ARMS",
    "REPAIR_SMOKE_CANDIDATES_PER_ARM",
    "REPAIR_SMOKE_SKILLS",
    "build_repair_smoke_plan",
    "summarize_repair_smoke_kinematics",
]

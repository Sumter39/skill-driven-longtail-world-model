"""Freeze the active Pilot capability boundary after checkpoint promotion."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from skilldrive.generation.capability import write_generation_capability_matrix
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.contracts import canonical_sha256
from skilldrive.generation.inference import (
    file_sha256,
    validate_active_checkpoint_promotion,
)
from skilldrive.generation.planning import pilot_evaluation_arm
from skilldrive.generation.scheduler import load_task_plan


PILOT_GATE_SCHEMA_VERSION = 1
PILOT_GATE_CONTRACT = "active_pilot_gate_v1"
FORMAL_ARMS = frozenset({"learned_conditioned", "rule_guided_none"})
CONTROL_ARM = "learned_none_control"


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{name} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{name} line {line_number} is invalid JSON: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def _resolved(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_label(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _expected_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 string") from error
    return value


def _verify_sha256(path: Path, expected: Any, name: str) -> str:
    expected_value = _expected_sha256(expected, f"{name} SHA-256")
    actual = file_sha256(path)
    if actual != expected_value:
        raise ValueError(f"{name} SHA-256 mismatch: {path}")
    return actual


def _finite_score(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return math.inf
    value = metrics.get("quality_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.inf
    score = float(value)
    return score if math.isfinite(score) else math.inf


def _row_identity(row: Mapping[str, Any], name: str) -> tuple[str, str]:
    task_id = row.get("task_id")
    candidate_id = row.get("candidate_id")
    if not isinstance(task_id, str) or not isinstance(candidate_id, str):
        raise ValueError(f"{name} row lacks task_id or candidate_id")
    return task_id, candidate_id


def _compact_case(
    row: Mapping[str, Any],
    *,
    disposition: str,
    evaluation_arm: str,
    decision_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    metrics = row.get("metrics")
    raw = row.get("raw")
    if not isinstance(metrics, Mapping) or not isinstance(raw, Mapping):
        raise ValueError("Pilot decision row lacks metrics or raw reference")
    raw_reference = {
        name: (
            value
            if name == "offset"
            else _path_label(repository_root, (decision_root / str(value)).resolve())
        )
        for name, value in raw.items()
    }
    result = {
        "disposition": disposition,
        "evaluation_arm": evaluation_arm,
        "candidate_id": row["candidate_id"],
        "task_id": row["task_id"],
        "candidate_index": row.get("candidate_index"),
        "latent_seed": row.get("latent_seed"),
        "scenario_id": metrics.get("scenario_id"),
        "seed_record_id": metrics.get("seed_record_id"),
        "skill_id": metrics.get("skill_id"),
        "target_track_id": metrics.get("target_track_id"),
        "quality_score": metrics.get("quality_score"),
        "raw": raw_reference,
    }
    if disposition == "rejected":
        result.update(
            {
                "first_failed_stage": row.get("first_failed_stage"),
                "primary_rejection_reason": row.get("primary_rejection_reason"),
            }
        )
    return result


def _representative_rejection(
    rows: Iterable[Mapping[str, Any]],
    *,
    arms_by_task: Mapping[str, str],
    decision_root: Path,
    repository_root: Path,
) -> dict[str, Any] | None:
    formal_rows = [
        row
        for row in rows
        if arms_by_task.get(str(row.get("task_id"))) in FORMAL_ARMS
    ]
    if not formal_rows:
        return None
    reasons = Counter(
        (
            str(row.get("first_failed_stage")),
            str(row.get("primary_rejection_reason")),
        )
        for row in formal_rows
    )
    representative_reason = min(
        reasons,
        key=lambda value: (-reasons[value], value[0], value[1]),
    )
    candidates = [
        row
        for row in formal_rows
        if (
            str(row.get("first_failed_stage")),
            str(row.get("primary_rejection_reason")),
        )
        == representative_reason
    ]
    selected = min(
        candidates,
        key=lambda row: (
            _finite_score(row),
            str(row.get("candidate_id")),
        ),
    )
    return _compact_case(
        selected,
        disposition="rejected",
        evaluation_arm=arms_by_task[str(selected["task_id"])],
        decision_root=decision_root,
        repository_root=repository_root,
    )


def _representative_acceptance(
    rows: Iterable[Mapping[str, Any]],
    *,
    arms_by_task: Mapping[str, str],
    decision_root: Path,
    repository_root: Path,
) -> dict[str, Any] | None:
    formal_rows = [
        row
        for row in rows
        if arms_by_task.get(str(row.get("task_id"))) in FORMAL_ARMS
    ]
    if not formal_rows:
        return None
    selected = min(
        formal_rows,
        key=lambda row: (
            _finite_score(row),
            str(row.get("metrics", {}).get("scenario_id", "")),
            str(row.get("candidate_id")),
        ),
    )
    return _compact_case(
        selected,
        disposition="accepted",
        evaluation_arm=arms_by_task[str(selected["task_id"])],
        decision_root=decision_root,
        repository_root=repository_root,
    )


def analyze_active_pilot(
    *,
    pilot_summary_path: str | Path,
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    output_root: str | Path | None = None,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate one completed active Pilot and freeze its capability boundary."""

    root = Path(repository_root).resolve()
    summary_path = _resolved(root, pilot_summary_path)
    config_path = _resolved(root, generation_config_path)
    summary = _read_json(summary_path, "active Pilot summary")
    config = load_counterfactual_config(config_path)
    promotion = validate_active_checkpoint_promotion(
        config.active_checkpoint,
        repository_root=root,
    )
    if promotion is None:
        raise ValueError("active Pilot gate requires a promoted repair checkpoint")
    if (
        summary.get("version") != 1
        or summary.get("stage") != "pilot"
        or summary.get("status") != "completed"
    ):
        raise ValueError("active Pilot summary is not complete")
    if summary.get("checkpoint_sha256") != config.active_checkpoint.sha256:
        raise ValueError("active Pilot checkpoint differs from active config")
    if (
        summary.get("validation_manifests_opened") is not False
        or summary.get("final_validation_accessed") is not False
    ):
        raise ValueError("active Pilot accessed a validation partition")
    if summary.get("raw_immutable_verified") is not True:
        raise ValueError("active Pilot did not verify raw immutability")

    outputs = summary.get("outputs")
    output_sha256 = summary.get("output_sha256")
    if not isinstance(outputs, Mapping) or not isinstance(output_sha256, Mapping):
        raise ValueError("active Pilot summary lacks output descriptors")
    task_plan_path = _resolved(root, str(outputs.get("task_plan", "")))
    accepted_path = _resolved(root, str(outputs.get("accepted", "")))
    rejected_path = _resolved(root, str(outputs.get("rejected", "")))
    filter_commit_path = _resolved(root, str(outputs.get("filter_commit", "")))
    _verify_sha256(task_plan_path, summary.get("task_plan_sha256"), "task plan")
    _verify_sha256(accepted_path, output_sha256.get("accepted"), "accepted index")
    _verify_sha256(rejected_path, output_sha256.get("rejected"), "rejected index")
    _verify_sha256(
        filter_commit_path,
        output_sha256.get("filter_commit"),
        "filter commit",
    )

    loaded = load_task_plan(
        task_plan_path.parent,
        expected_semantic_config_sha256=_expected_sha256(
            summary.get("generation_semantic_sha256"),
            "generation semantic SHA-256",
        ),
        current_execution_config_sha256=_expected_sha256(
            summary.get("generation_execution_sha256"),
            "generation execution SHA-256",
        ),
    )
    plan = loaded.plan
    if (
        plan.task_plan_id != summary.get("task_plan_id")
        or len(plan.tasks) != summary.get("task_count")
        or plan.total_candidates != summary.get("candidate_count")
    ):
        raise ValueError("active Pilot task-plan counts differ from summary")

    filter_commit = _read_json(filter_commit_path, "Pilot filter commit")
    counts = filter_commit.get("counts")
    task_statuses = filter_commit.get("task_statuses")
    if (
        filter_commit.get("kind") != "filter_index_commit"
        or not isinstance(counts, Mapping)
        or not isinstance(task_statuses, Mapping)
        or set(task_statuses) != {task.task_id for task in plan.tasks}
        or any(value != "complete" for value in task_statuses.values())
    ):
        raise ValueError("Pilot filter commit is incomplete")

    accepted = _read_jsonl(accepted_path, "accepted Pilot index")
    rejected = _read_jsonl(rejected_path, "rejected Pilot index")
    if (
        len(accepted) != summary.get("accepted_count")
        or len(rejected) != summary.get("rejected_count")
        or len(accepted) != counts.get("accepted")
        or len(rejected) != counts.get("rejected")
        or len(accepted) + len(rejected) != plan.total_candidates
    ):
        raise ValueError("Pilot filter decision counts are incomplete")
    candidate_ids: set[str] = set()
    tasks_by_id = {task.task_id: task for task in plan.tasks}
    arms_by_task = {
        task.task_id: pilot_evaluation_arm(task, none_skill_id=config.none_skill_id)
        for task in plan.tasks
    }
    for name, rows in (("accepted", accepted), ("rejected", rejected)):
        for row in rows:
            task_id, candidate_id = _row_identity(row, name)
            if task_id not in tasks_by_id:
                raise ValueError(f"{name} index references an unknown task")
            if candidate_id in candidate_ids:
                raise ValueError("Pilot decision candidate IDs are not unique")
            candidate_ids.add(candidate_id)
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError(f"{name} index lacks metrics")
            if metrics.get("skill_id") != tasks_by_id[task_id].skill_id:
                raise ValueError(f"{name} index skill differs from task plan")

    evidence = promotion.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("active promotion lacks Heldout evidence")
    heldout_descriptor = evidence.get("heldout_gate_summary")
    if not isinstance(heldout_descriptor, Mapping):
        raise ValueError("active promotion lacks Heldout gate summary")
    heldout_path = _resolved(root, str(heldout_descriptor.get("path", "")))
    heldout_sha256 = _verify_sha256(
        heldout_path,
        heldout_descriptor.get("sha256"),
        "Heldout gate summary",
    )
    heldout = _read_json(heldout_path, "Heldout gate summary")
    heldout_gates = heldout.get("ability_gates")
    if (
        heldout.get("status") != "passed"
        or heldout.get("checkpoint_sha256") != config.active_checkpoint.sha256
        or not isinstance(heldout_gates, Mapping)
        or not heldout_gates
        or not all(value is True for value in heldout_gates.values())
        or heldout.get("validation_manifests_opened") is not False
        or heldout.get("final_validation_accessed") is not False
    ):
        raise ValueError("active checkpoint Heldout gate did not pass")

    formal_skill_ids = tuple(config.formal_skill_ids)
    formal_skill_set = set(formal_skill_ids)
    without_eligible = summary.get("skills_without_eligible_seed_records")
    if not isinstance(without_eligible, list) or any(
        not isinstance(value, str) for value in without_eligible
    ):
        raise ValueError("Pilot missing-skill status is malformed")
    without_eligible_set = set(without_eligible)
    task_skills = {task.skill_id for task in plan.tasks}
    if task_skills & without_eligible_set:
        raise ValueError("Pilot skill cannot have tasks and be marked without eligibility")
    if task_skills | without_eligible_set != formal_skill_set:
        raise ValueError("Pilot does not account for every formal skill")

    accepted_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted_by_arm: Counter[str] = Counter()
    candidates_by_arm: Counter[str] = Counter()
    tasks_by_skill_arm: Counter[tuple[str, str]] = Counter()
    for task in plan.tasks:
        arm = arms_by_task[task.task_id]
        candidates_by_arm[arm] += task.candidate_budget
        tasks_by_skill_arm[(task.skill_id, arm)] += 1
    for row in accepted:
        task = tasks_by_id[str(row["task_id"])]
        accepted_by_skill[task.skill_id].append(row)
        accepted_by_arm[arms_by_task[task.task_id]] += 1
    for row in rejected:
        task = tasks_by_id[str(row["task_id"])]
        rejected_by_skill[task.skill_id].append(row)

    skill_rows: list[dict[str, Any]] = []
    review_cases: list[dict[str, Any]] = []
    proposal_modes = {item.skill_id: item.proposal_mode for item in config.skills}
    for skill_id in formal_skill_ids:
        accepted_rows = accepted_by_skill.get(skill_id, [])
        rejected_rows = rejected_by_skill.get(skill_id, [])
        accepted_counts = Counter(
            arms_by_task[str(row["task_id"])] for row in accepted_rows
        )
        task_counts = {
            arm: tasks_by_skill_arm[(skill_id, arm)]
            for arm in ("learned_conditioned", CONTROL_ARM, "rule_guided_none")
            if tasks_by_skill_arm[(skill_id, arm)]
        }
        formal_accepted = sum(accepted_counts[arm] for arm in FORMAL_ARMS)
        control_accepted = accepted_counts[CONTROL_ARM]
        if skill_id in without_eligible_set:
            support_status = "no_eligible_seed_record"
        elif formal_accepted:
            support_status = "formal_supported"
        elif control_accepted:
            support_status = "control_only_not_formal"
        else:
            support_status = "pilot_zero_accept"
        rejections = Counter(
            (
                str(row.get("first_failed_stage")),
                str(row.get("primary_rejection_reason")),
            )
            for row in rejected_rows
            if arms_by_task[str(row["task_id"])] in FORMAL_ARMS
        )
        skill_rows.append(
            {
                "skill_id": skill_id,
                "proposal_mode": proposal_modes[skill_id],
                "support_status": support_status,
                "task_count_by_arm": task_counts,
                "formal_accepted": formal_accepted,
                "control_accepted": control_accepted,
                "dominant_formal_rejection": (
                    None
                    if not rejections
                    else {
                        "first_failed_stage": min(
                            rejections,
                            key=lambda value: (
                                -rejections[value],
                                value[0],
                                value[1],
                            ),
                        )[0],
                        "primary_rejection_reason": min(
                            rejections,
                            key=lambda value: (
                                -rejections[value],
                                value[0],
                                value[1],
                            ),
                        )[1],
                        "count": max(rejections.values()),
                    }
                ),
            }
        )
        accepted_case = _representative_acceptance(
            accepted_rows,
            arms_by_task=arms_by_task,
            decision_root=filter_commit_path.parent,
            repository_root=root,
        )
        rejected_case = _representative_rejection(
            rejected_rows,
            arms_by_task=arms_by_task,
            decision_root=filter_commit_path.parent,
            repository_root=root,
        )
        review_cases.extend(
            value for value in (accepted_case, rejected_case) if value is not None
        )

    formal_accepted_count = sum(accepted_by_arm[arm] for arm in FORMAL_ARMS)
    gates = {
        "promoted_checkpoint_heldout_gate_passed": True,
        "complete_task_and_candidate_budget": True,
        "all_formal_skills_have_explicit_status": len(skill_rows)
        == len(formal_skill_ids),
        "formal_candidate_accepted": formal_accepted_count > 0,
        "learned_conditioned_candidate_accepted": accepted_by_arm[
            "learned_conditioned"
        ]
        > 0,
        "rule_guided_candidate_accepted": accepted_by_arm["rule_guided_none"] > 0,
        "conditioned_acceptance_exceeds_control": accepted_by_arm[
            "learned_conditioned"
        ]
        > accepted_by_arm[CONTROL_ARM],
        "validation_partitions_not_accessed": True,
    }
    failure_reasons = [name for name, passed in gates.items() if not passed]

    input_contract = {
        "contract": PILOT_GATE_CONTRACT,
        "pilot_summary_sha256": file_sha256(summary_path),
        "generation_config_sha256": file_sha256(config_path),
        "task_plan_sha256": file_sha256(task_plan_path),
        "filter_commit_sha256": file_sha256(filter_commit_path),
        "promotion_recommendation_sha256": config.active_checkpoint.promotion_recommendation_sha256,
        "heldout_gate_summary_sha256": heldout_sha256,
    }
    analysis_id = canonical_sha256(input_contract)
    destination_root = (
        _resolved(root, output_root)
        if output_root is not None
        else summary_path.parent / "ability-analysis-v1" / analysis_id
    )
    review_manifest = {
        "schema_version": PILOT_GATE_SCHEMA_VERSION,
        "kind": "active_pilot_review_manifest",
        "contract": PILOT_GATE_CONTRACT,
        "analysis_id": analysis_id,
        "case_count": len(review_cases),
        "cases": sorted(
            review_cases,
            key=lambda value: (
                str(value["skill_id"]),
                0 if value["disposition"] == "accepted" else 1,
                str(value["candidate_id"]),
            ),
        ),
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    review_path = destination_root / "pilot_review_manifest.json"
    if review_path.exists():
        if _read_json(review_path, "Pilot review manifest") != review_manifest:
            raise ValueError("existing Pilot review manifest differs from frozen analysis")
    else:
        write_generation_capability_matrix(review_path, review_manifest)

    analysis = {
        "schema_version": PILOT_GATE_SCHEMA_VERSION,
        "kind": "active_pilot_gate_analysis",
        "contract": PILOT_GATE_CONTRACT,
        "analysis_id": analysis_id,
        "status": "passed" if not failure_reasons else "failed",
        "failure_reasons": failure_reasons,
        "checkpoint_sha256": config.active_checkpoint.sha256,
        "pilot": {
            "summary": {
                "path": _path_label(root, summary_path),
                "sha256": input_contract["pilot_summary_sha256"],
            },
            "task_plan_id": plan.task_plan_id,
            "task_count": len(plan.tasks),
            "candidate_count": plan.total_candidates,
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "formal_accepted_count": formal_accepted_count,
            "formal_skill_count": len(formal_skill_ids),
            "formal_supported_skill_count": sum(
                row["support_status"] == "formal_supported" for row in skill_rows
            ),
        },
        "accepted_by_arm": {
            arm: accepted_by_arm[arm]
            for arm in ("learned_conditioned", CONTROL_ARM, "rule_guided_none")
        },
        "candidate_count_by_arm": {
            arm: candidates_by_arm[arm]
            for arm in ("learned_conditioned", CONTROL_ARM, "rule_guided_none")
        },
        "heldout_gate": {
            "path": _path_label(root, heldout_path),
            "sha256": heldout_sha256,
            "ability_gates": dict(heldout_gates),
        },
        "gates": gates,
        "skill_status": skill_rows,
        "inputs": input_contract,
        "outputs": {
            "review_manifest": _path_label(root, review_path),
            "review_manifest_sha256": file_sha256(review_path),
        },
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    analysis_path = destination_root / "pilot_gate_analysis.json"
    if analysis_path.exists():
        if _read_json(analysis_path, "Pilot gate analysis") != analysis:
            raise ValueError("existing Pilot gate analysis differs from frozen evidence")
    else:
        write_generation_capability_matrix(analysis_path, analysis)
    return {
        **analysis,
        "output_paths": {
            "analysis": _path_label(root, analysis_path),
            "review_manifest": _path_label(root, review_path),
        },
    }


__all__ = [
    "PILOT_GATE_CONTRACT",
    "PILOT_GATE_SCHEMA_VERSION",
    "analyze_active_pilot",
]

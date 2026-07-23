"""Deterministic balanced delivery selection and audit for formal accepted data."""

from __future__ import annotations

import json
import math
import os
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.inference import file_sha256
from skilldrive.generation.formal_review import _iter_jsonl, _read_json, _resolved


FORMAL_DELIVERY_SCHEMA_VERSION = 1
FORMAL_DELIVERY_CONTRACT = "formal_balanced_delivery_v1"


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_plan(run_root: Path) -> dict[str, dict[str, Any]]:
    path = run_root / "formal_task_plan.jsonl"
    tasks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or task_id in tasks:
                raise ValueError("formal task plan has duplicate or invalid task IDs")
            tasks[task_id] = row
    return tasks


def _quality_score(row: Mapping[str, Any]) -> float:
    metrics = row.get("metrics")
    value = metrics.get("quality_score") if isinstance(metrics, Mapping) else None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("accepted row has no finite quality_score")
    return float(value)


def _row_key(row: Mapping[str, Any]) -> tuple[float, str, int, str]:
    return (-_quality_score(row), str(row.get("scenario_id", "")), int(row.get("candidate_index", 0)), str(row.get("candidate_id", "")))


def _target_risk_value(row: Mapping[str, Any]) -> float | None:
    evidence = _stage_evidence(row, "target_risk")
    if evidence is not None:
        value = evidence.get("metrics", {}).get("evaluation", {}).get("value")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _stage_evidence(row: Mapping[str, Any], stage: str) -> Mapping[str, Any] | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    for evidence in metrics.get("stage_evidence", ()):
        if isinstance(evidence, Mapping) and evidence.get("stage") == stage:
            return evidence
    return None


def _accepted_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for directory in sorted(path for path in (run_root / "filter").iterdir() if path.is_dir()):
        path = directory / "accepted.jsonl"
        if path.is_file():
            rows.extend(_iter_jsonl(path))
    return rows


def _select_balanced(rows: list[dict[str, Any]], *, max_per_skill: int) -> list[dict[str, Any]]:
    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        skill = row.get("skill_id") or row.get("metrics", {}).get("skill_id")
        scenario = row.get("scenario_id") or row.get("metrics", {}).get("scenario_id")
        if not isinstance(skill, str) or not isinstance(scenario, str):
            raise ValueError("accepted row is missing skill or scenario identity")
        row["skill_id"] = skill
        row["scenario_id"] = scenario
        by_skill[skill].append(row)
    selected: list[dict[str, Any]] = []
    for skill in sorted(by_skill):
        scenario_counts: Counter[str] = Counter()
        selected_for_skill = 0
        candidates = sorted(by_skill[skill], key=_row_key)
        for row in candidates:
            scenario = row["scenario_id"]
            if scenario_counts[scenario] >= 3:
                continue
            selected.append(row)
            scenario_counts[scenario] += 1
            selected_for_skill += 1
            if selected_for_skill >= max_per_skill:
                break
    return selected


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def build_formal_delivery(
    *,
    run_root: str | Path,
    repository_root: str | Path = ".",
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    output_root: str | Path | None = None,
    max_per_skill: int = 300,
) -> dict[str, Any]:
    """Select at most ``max_per_skill`` accepted candidates per skill and audit them."""

    if isinstance(max_per_skill, bool) or not isinstance(max_per_skill, int) or max_per_skill <= 0:
        raise ValueError("max_per_skill must be a positive integer")
    root = Path(repository_root).resolve()
    run = Path(run_root).resolve()
    formal_summary_path = run / "summary.json"
    formal_summary = _read_json(formal_summary_path, "formal summary")
    if (
        formal_summary.get("kind") != "formal_counterfactual_summary"
        or formal_summary.get("status") != "completed"
        or formal_summary.get("validation_manifests_opened") is not False
        or formal_summary.get("final_validation_accessed") is not False
    ):
        raise ValueError("formal run is not a completed Formal Train artifact")
    config_path = _resolved(root, generation_config_path)
    config = load_counterfactual_config(config_path, repository_root=root)
    tasks = _task_plan(run)
    source_rows = _accepted_rows(run)
    selected = _select_balanced(source_rows, max_per_skill=max_per_skill)
    selected_ids = [row["candidate_id"] for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("balanced delivery contains duplicate candidate IDs")
    if len({row["filter_evaluation_id"] for row in selected}) != len(selected):
        raise ValueError("balanced delivery contains duplicate filter evaluations")

    delivery_rows: list[dict[str, Any]] = []
    by_skill: dict[str, list[dict[str, Any]]] = defaultdict(list)
    diversity_policy_counts: Counter[str] = Counter()
    for row in selected:
        task = tasks.get(row["task_id"])
        if task is None:
            raise ValueError("accepted row references unknown formal task")
        skill_id = row["skill_id"]
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        diversity = _stage_evidence(row, "diversity")
        if diversity is None or diversity.get("passed") is not True:
            raise ValueError("accepted delivery row did not pass the diversity stage")
        diversity_policy = diversity.get("metrics", {}).get("policy_source")
        if not isinstance(diversity_policy, str) or not diversity_policy:
            raise ValueError("accepted delivery row has no diversity policy source")
        diversity_policy_counts[diversity_policy] += 1
        delivery = {
            "candidate_id": row["candidate_id"],
            "filter_evaluation_id": row["filter_evaluation_id"],
            "task_id": row["task_id"],
            "candidate_index": row["candidate_index"],
            "scenario_id": row["scenario_id"],
            "skill_id": skill_id,
            "seed_record_id": metrics.get("seed_record_id"),
            "proposal_mode": task.get("proposal_mode"),
            "condition_skill_id": task.get("condition_skill_id"),
            "target_track_id": metrics.get("target_track_id"),
            "quality_score": _quality_score(row),
            "target_risk_value": _target_risk_value(row),
            "realized_parameter_bins": diversity.get("metrics", {}).get(
                "realized_parameter_bins", []
            ),
            "raw": row.get("raw"),
        }
        delivery_rows.append(delivery)
        by_skill[skill_id].append(delivery)

    output = _resolved(root, output_root) if output_root is not None else run / "review" / "formal_delivery_v1"
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "balanced_accepted.jsonl"
    index_payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in delivery_rows)
    _atomic_write(index_path, index_payload)

    risk_by_skill = {
        skill: _summarize(
            [float(row["target_risk_value"]) for row in values if row["target_risk_value"] is not None]
        )
        for skill, values in sorted(by_skill.items())
    }
    skill_counts = {skill: len(values) for skill, values in sorted(by_skill.items())}
    mode_counts = Counter(str(row["proposal_mode"]) for row in delivery_rows)
    role_counts = Counter(
        str(config.skills_by_id[row["skill_id"]].primary_generated_role)
        for row in delivery_rows
    )
    scenario_counts = Counter(row["scenario_id"] for row in delivery_rows)
    scenario_skill_counts = Counter(
        (row["scenario_id"], row["skill_id"]) for row in delivery_rows
    )
    parameter_bin_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in delivery_rows:
        for value in row["realized_parameter_bins"]:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("accepted delivery row has an invalid realized parameter bin")
            parameter, bin_index = value
            parameter_bin_counts[row["skill_id"]][f"{parameter}={bin_index}"] += 1
    scenario_histogram = Counter(scenario_counts.values())
    top_scenarios = sorted(scenario_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    audit = {
        "schema_version": FORMAL_DELIVERY_SCHEMA_VERSION,
        "kind": "formal_balanced_delivery",
        "contract": FORMAL_DELIVERY_CONTRACT,
        "status": "ready_for_manual_review",
        "formal_plan_id": formal_summary["formal_plan_id"],
        "formal_summary_sha256": file_sha256(formal_summary_path),
        "generation_config_sha256": file_sha256(config_path),
        "source_candidate_count": len(source_rows),
        "selected_candidate_count": len(delivery_rows),
        "max_per_skill": max_per_skill,
        "skill_counts": skill_counts,
        "proposal_mode_counts": dict(sorted(mode_counts.items())),
        "primary_generated_role_counts": dict(sorted(role_counts.items())),
        "unique_source_scenario_count": len(scenario_counts),
        "maximum_candidates_per_scenario": max(scenario_counts.values(), default=0),
        "maximum_candidates_per_scenario_skill": max(scenario_skill_counts.values(), default=0),
        "source_scenario_candidate_count_histogram": {
            str(count): scenarios for count, scenarios in sorted(scenario_histogram.items())
        },
        "top_source_scenarios": [
            {"scenario_id": scenario_id, "candidate_count": count}
            for scenario_id, count in top_scenarios
        ],
        "top_10_source_scenario_share": (
            sum(count for _, count in top_scenarios) / len(delivery_rows)
            if delivery_rows
            else 0.0
        ),
        "duplicate_candidate_ids": len(selected_ids) - len(set(selected_ids)),
        "duplicate_filter_evaluation_ids": len(selected) - len({row["filter_evaluation_id"] for row in selected}),
        "diversity_stage_passed_count": len(delivery_rows),
        "diversity_policy_counts": dict(sorted(diversity_policy_counts.items())),
        "realized_parameter_bin_counts_by_skill": {
            skill: dict(sorted(counts.items()))
            for skill, counts in sorted(parameter_bin_counts.items())
        },
        "quality_score_by_skill": {
            skill: _summarize([float(row["quality_score"]) for row in values])
            for skill, values in sorted(by_skill.items())
        },
        "target_risk_by_skill": risk_by_skill,
        "accepted_index": {
            "path": index_path.name,
            "sha256": file_sha256(index_path),
            "size_bytes": index_path.stat().st_size,
        },
        "formal_train_only": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "manual_review_status": "pending",
    }
    audit_path = output / "audit.json"
    _atomic_write(audit_path, json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {**audit, "output_path": audit_path.as_posix()}


__all__ = ["FORMAL_DELIVERY_CONTRACT", "FORMAL_DELIVERY_SCHEMA_VERSION", "build_formal_delivery"]

"""Evidence-based automatic review for completed formal review cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

from skilldrive.generation.formal_review import (
    REVIEW_CRITERIA,
    REVIEW_TEMPLATE_COLUMNS,
    _iter_jsonl,
    _read_json,
    _valid_png,
)
from skilldrive.generation.inference import file_sha256


AUTOMATED_REVIEWER = "codex-automated-evidence-v1"
CRITERION_STAGES = {
    "history_invariants": "history_invariants",
    "road_relation": "map",
    "motion_continuity": "kinematics",
    "skill_role": "skill_trigger",
    "target_risk": "target_risk",
    "parameter_realization": "parameter_realization",
    "background_interaction": "collision",
}


def _stage_result(row: Mapping[str, Any], stage: str) -> str:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return "uncertain"
    for evidence in metrics.get("stage_evidence", ()):
        if isinstance(evidence, Mapping) and evidence.get("stage") == stage:
            return "pass" if evidence.get("passed") is True else "fail"
    return "not_applicable"


def _image_passed(reference: Mapping[str, Any]) -> bool:
    path_value = reference.get("path")
    expected_sha256 = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha256, str):
        return False
    path = Path(path_value)
    if not _valid_png(path):
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return file_sha256(path) == expected_sha256
    except (OSError, ValueError):
        return False


def _load_selected_rows(run_root: Path, candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(run_root.joinpath("filter").glob("*/accepted.jsonl")) + sorted(
        run_root.joinpath("filter").glob("*/rejected.jsonl")
    ):
        for row in _iter_jsonl(path):
            candidate_id = row.get("candidate_id")
            if candidate_id not in candidate_ids:
                continue
            if candidate_id in rows:
                raise ValueError(f"duplicate selected candidate in formal filter output: {candidate_id}")
            rows[candidate_id] = row
    missing = candidate_ids - rows.keys()
    if missing:
        raise ValueError(f"formal filter output is missing selected candidates: {sorted(missing)[:3]}")
    return rows


def build_automated_review_csv(
    *,
    summary_path: str | Path,
    run_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic review CSV from filter evidence and image hashes."""

    summary_file = Path(summary_path).resolve()
    run = Path(run_root).resolve()
    summary = _read_json(summary_file, "formal review summary")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("formal review summary has no cases")
    candidate_ids = {case.get("candidate_id") for case in cases}
    if not all(isinstance(value, str) and value for value in candidate_ids):
        raise ValueError("formal review summary has invalid candidate IDs")
    filter_rows = _load_selected_rows(run, candidate_ids)
    output_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {"passed": 0, "failed": 0, "uncertain": 0}
    criterion_counts = {
        criterion: {"pass": 0, "fail": 0, "not_applicable": 0, "uncertain": 0}
        for criterion in REVIEW_CRITERIA
    }
    for case in cases:
        candidate_id = case["candidate_id"]
        row = filter_rows[candidate_id]
        if row.get("candidate_id") != candidate_id:
            raise ValueError(f"candidate identity mismatch: {candidate_id}")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"selected row has no metrics: {candidate_id}")
        criteria = {
            criterion: _stage_result(row, stage)
            for criterion, stage in CRITERION_STAGES.items()
        }
        criteria["visual_artifacts"] = (
            "pass"
            if _image_passed(case["source_png"]) and _image_passed(case["generated_png"])
            else "fail"
        )
        for criterion, value in criteria.items():
            criterion_counts[criterion][value] += 1
        if "fail" in criteria.values():
            status = "failed"
        elif "uncertain" in criteria.values():
            status = "uncertain"
        else:
            status = "passed"
        status_counts[status] += 1
        failed = [criterion for criterion, value in criteria.items() if value == "fail"]
        output_rows.append(
            {
                "review_rank": case["review_rank"],
                "case_name": case["case_name"],
                "disposition": case["disposition"],
                "skill_id": case["skill_id"],
                "scenario_id": case["scenario_id"],
                "candidate_id": candidate_id,
                "first_failed_stage": case.get("first_failed_stage") or "",
                "source_png": case["source_png"]["path"],
                "generated_png": case["generated_png"]["path"],
                **criteria,
                "review_status": status,
                "reviewer": AUTOMATED_REVIEWER,
                "issue_categories": ";".join(failed),
                "notes": (
                    "Automated from formal filter stage evidence and PNG hash/decode checks; "
                    f"first_failed_stage={case.get('first_failed_stage') or 'none'}"
                ),
            }
        )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else summary_file.parent / "automated_review.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "status": "automated_review_completed",
        "reviewer": AUTOMATED_REVIEWER,
        "case_count": len(output_rows),
        "reviewed_count": len(output_rows),
        "status_counts": status_counts,
        "criterion_counts": criterion_counts,
        "output_path": destination.as_posix(),
    }


__all__ = ["AUTOMATED_REVIEWER", "build_automated_review_csv"]

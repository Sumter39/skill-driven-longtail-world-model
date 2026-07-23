import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from skilldrive.generation.formal_review import (
    _select_accepted,
    _select_representatives,
    audit_formal_review,
    finalize_review_annotations,
    write_review_template,
)
from skilldrive.generation.inference import file_sha256


def _row(
    scenario: str,
    skill: str = "skill",
    task: str = "task",
    index: int = 0,
    candidate: str | None = None,
    stage: str | None = "map",
    score: float | None = None,
) -> dict:
    candidate = candidate or f"candidate-{scenario}-{index}"
    return {
        "candidate_id": candidate,
        "candidate_index": index,
        "task_id": task,
        "metrics": {
            "scenario_id": scenario,
            "skill_id": skill,
            "first_failed_stage": stage,
            "quality_score": score,
        },
    }


def test_representatives_cover_distinct_failure_stages_before_filling() -> None:
    rows = [
        _row("scene-map", stage="map"),
        _row("scene-kinematics", stage="kinematics"),
        _row("scene-map-2", stage="map"),
        _row("scene-risk", stage="target_risk"),
    ]

    selected = _select_representatives(rows, 3)

    assert [row["metrics"]["first_failed_stage"] for row in selected] == [
        "kinematics",
        "map",
        "target_risk",
    ]


def test_accepted_selection_is_quality_sorted_and_bounded() -> None:
    rows = [_row("scene-a", score=0.2), _row("scene-b", score=0.9), _row("scene-c", score=0.5)]

    selected = _select_accepted(rows, 2)

    assert [row["metrics"]["scenario_id"] for row in selected] == ["scene-b", "scene-c"]


def _review_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (8, 6), "white").save(source)
    Image.new("RGB", (8, 6), "black").save(generated)
    case = {
        "review_rank": 1,
        "case_name": "001-skill-accepted-candidate",
        "candidate_id": "candidate",
        "scenario_id": "scene",
        "skill_id": "skill",
        "disposition": "accepted",
        "first_failed_stage": None,
        "review_status": "pending",
        "reviewer": "",
        "notes": "",
    }
    manifest = {"cases": [case]}
    manifest_path = tmp_path / "review_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary = {
        "kind": "formal_generation_review",
        "manual_review_status": "pending",
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "review_manifest": {
            "path": manifest_path.as_posix(),
            "sha256": file_sha256(manifest_path),
        },
        "cases": [
            {
                **case,
                "source_png": {"path": source.as_posix(), "sha256": file_sha256(source)},
                "generated_png": {
                    "path": generated.as_posix(),
                    "sha256": file_sha256(generated),
                },
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_formal_review_audit_verifies_images_and_writes_template(tmp_path: Path) -> None:
    summary_path = _review_fixture(tmp_path)

    audit = audit_formal_review(summary_path=summary_path, repository_root=tmp_path)
    template = write_review_template(summary_path=summary_path)

    assert audit["status"] == "automated_audit_passed"
    assert audit["verified_image_count"] == 2
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["review_status"] == "pending"


def test_formal_review_audit_rejects_changed_image(tmp_path: Path) -> None:
    summary_path = _review_fixture(tmp_path)
    generated = tmp_path / "generated.png"
    Image.new("RGB", (8, 6), "red").save(generated)

    try:
        audit_formal_review(summary_path=summary_path, repository_root=tmp_path)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("changed review image was accepted")


def test_finalize_review_annotations_requires_reviewer_and_status(tmp_path: Path) -> None:
    summary_path = _review_fixture(tmp_path)
    template = write_review_template(summary_path=summary_path)
    with template.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["review_status"] = "passed"
    rows[0]["reviewer"] = "reviewer-a"
    for column in (
        "history_invariants",
        "road_relation",
        "motion_continuity",
        "skill_role",
        "target_risk",
        "parameter_realization",
        "background_interaction",
        "visual_artifacts",
    ):
        rows[0][column] = "pass"
    with template.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = finalize_review_annotations(
        summary_path=summary_path,
        annotations_path=template,
        minimum_reviews=1,
    )

    assert result["manual_review_status"] == "completed_minimum"
    assert result["manual_review_count"] == 1
    assert result["manual_review_criterion_counts"]["motion_continuity"] == {"pass": 1}


def test_finalize_review_annotations_enforces_minimum_count(tmp_path: Path) -> None:
    summary_path = _review_fixture(tmp_path)
    template = write_review_template(summary_path=summary_path)

    with pytest.raises(ValueError, match="at least 1"):
        finalize_review_annotations(
            summary_path=summary_path,
            annotations_path=template,
            minimum_reviews=1,
        )

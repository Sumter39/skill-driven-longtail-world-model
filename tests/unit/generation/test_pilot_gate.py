from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skilldrive.generation.inference import file_sha256
from skilldrive.generation.pilot_gate import analyze_active_pilot


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _decision(
    task_id: str,
    candidate_id: str,
    skill_id: str,
    *,
    score: float,
    rejected: bool = False,
) -> dict:
    row = {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "candidate_index": 0,
        "latent_seed": 1,
        "raw": {
            "commit": "raw.commit.json",
            "arrays": "raw.npz",
            "metadata": "raw.meta.jsonl.gz",
            "offset": 0,
        },
        "metrics": {
            "quality_score": score,
            "scenario_id": f"scenario-{skill_id}",
            "seed_record_id": "b" * 64,
            "skill_id": skill_id,
            "target_track_id": "target",
        },
    }
    if rejected:
        row.update(
            {
                "first_failed_stage": "map",
                "primary_rejection_reason": "map.outside_drivable_area",
            }
        )
    return row


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    checkpoint_sha256 = "a" * 64
    tasks = (
        SimpleNamespace(
            task_id="conditioned",
            skill_id="learned_skill",
            proposal_mode="learned_conditioned_prior",
            condition_skill_id="learned_skill",
            candidate_budget=1,
        ),
        SimpleNamespace(
            task_id="control",
            skill_id="learned_skill",
            proposal_mode="learned_conditioned_prior",
            condition_skill_id="<none>",
            candidate_budget=1,
        ),
        SimpleNamespace(
            task_id="rule",
            skill_id="rule_skill",
            proposal_mode="rule_guided_prior_search",
            condition_skill_id="<none>",
            candidate_budget=2,
        ),
    )
    plan = SimpleNamespace(tasks=tasks, task_plan_id="plan-v1", total_candidates=4)
    config = SimpleNamespace(
        active_checkpoint=SimpleNamespace(
            sha256=checkpoint_sha256,
            promotion_recommendation_sha256="c" * 64,
        ),
        none_skill_id="<none>",
        formal_skill_ids=("learned_skill", "rule_skill", "no_eligible_skill"),
        skills=(
            SimpleNamespace(
                skill_id="learned_skill",
                proposal_mode="learned_conditioned_prior",
            ),
            SimpleNamespace(
                skill_id="rule_skill",
                proposal_mode="rule_guided_prior_search",
            ),
            SimpleNamespace(
                skill_id="no_eligible_skill",
                proposal_mode="rule_guided_prior_search",
            ),
        ),
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_gate.load_counterfactual_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_gate.load_task_plan",
        lambda *args, **kwargs: SimpleNamespace(plan=plan),
    )

    heldout_path = tmp_path / "heldout.json"
    _write_json(
        heldout_path,
        {
            "status": "passed",
            "checkpoint_sha256": checkpoint_sha256,
            "ability_gates": {"complete": True, "conditioned": True},
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )
    promotion = {
        "evidence": {
            "heldout_gate_summary": {
                "path": heldout_path,
                "sha256": file_sha256(heldout_path),
            }
        }
    }
    monkeypatch.setattr(
        "skilldrive.generation.pilot_gate.validate_active_checkpoint_promotion",
        lambda *args, **kwargs: promotion,
    )

    task_plan_path = tmp_path / "task_plan.jsonl"
    task_plan_path.write_text("synthetic-plan\n", encoding="utf-8")
    accepted_path = tmp_path / "accepted.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    filter_commit_path = tmp_path / "filter-index.commit.json"
    _write_jsonl(
        accepted_path,
        [
            _decision("conditioned", "accepted-conditioned", "learned_skill", score=0.1),
            _decision("rule", "accepted-rule", "rule_skill", score=0.2),
        ],
    )
    _write_jsonl(
        rejected_path,
        [
            _decision(
                "control",
                "rejected-control",
                "learned_skill",
                score=0.5,
                rejected=True,
            ),
            _decision(
                "rule",
                "rejected-rule",
                "rule_skill",
                score=0.4,
                rejected=True,
            ),
        ],
    )
    _write_json(
        filter_commit_path,
        {
            "kind": "filter_index_commit",
            "counts": {"accepted": 2, "rejected": 2, "tasks": 3},
            "task_statuses": {
                "conditioned": "complete",
                "control": "complete",
                "rule": "complete",
            },
        },
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic: true\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {
            "version": 1,
            "stage": "pilot",
            "status": "completed",
            "checkpoint_sha256": checkpoint_sha256,
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
            "raw_immutable_verified": True,
            "outputs": {
                "task_plan": str(task_plan_path),
                "accepted": str(accepted_path),
                "rejected": str(rejected_path),
                "filter_commit": str(filter_commit_path),
            },
            "output_sha256": {
                "accepted": file_sha256(accepted_path),
                "rejected": file_sha256(rejected_path),
                "filter_commit": file_sha256(filter_commit_path),
            },
            "task_plan_sha256": file_sha256(task_plan_path),
            "generation_semantic_sha256": "d" * 64,
            "generation_execution_sha256": "e" * 64,
            "task_plan_id": "plan-v1",
            "task_count": 3,
            "candidate_count": 4,
            "accepted_count": 2,
            "rejected_count": 2,
            "skills_without_eligible_seed_records": ["no_eligible_skill"],
        },
    )
    summary_path.with_suffix(".config-path").write_text(
        str(config_path), encoding="utf-8"
    )
    return summary_path


def test_active_pilot_gate_freezes_support_boundary_and_review_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _fixture(tmp_path, monkeypatch)
    config_path = Path(summary_path.with_suffix(".config-path").read_text(encoding="utf-8"))

    result = analyze_active_pilot(
        pilot_summary_path=summary_path,
        generation_config_path=config_path,
        output_root=tmp_path / "analysis",
        repository_root=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["accepted_by_arm"] == {
        "learned_conditioned": 1,
        "learned_none_control": 0,
        "rule_guided_none": 1,
    }
    statuses = {row["skill_id"]: row["support_status"] for row in result["skill_status"]}
    assert statuses == {
        "learned_skill": "formal_supported",
        "rule_skill": "formal_supported",
        "no_eligible_skill": "no_eligible_seed_record",
    }
    manifest = json.loads(
        (tmp_path / "analysis" / "pilot_review_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [(row["skill_id"], row["disposition"]) for row in manifest["cases"]] == [
        ("learned_skill", "accepted"),
        ("rule_skill", "accepted"),
        ("rule_skill", "rejected"),
    ]


def test_active_pilot_gate_rejects_unaccounted_formal_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = _fixture(tmp_path, monkeypatch)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["skills_without_eligible_seed_records"] = []
    _write_json(summary_path, summary)
    config_path = Path(summary_path.with_suffix(".config-path").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="account for every formal skill"):
        analyze_active_pilot(
            pilot_summary_path=summary_path,
            generation_config_path=config_path,
            output_root=tmp_path / "analysis",
            repository_root=tmp_path,
        )

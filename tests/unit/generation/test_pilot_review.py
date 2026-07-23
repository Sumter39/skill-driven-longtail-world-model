from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skilldrive.generation.inference import file_sha256
from skilldrive.generation.pilot_review import PNG_SIGNATURE, render_active_pilot_review
from skilldrive.generation.planning import seed_record_id
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds.records import SeedRecord


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _record() -> SeedRecord:
    return SeedRecord(
        scenario_id="scene",
        skill_id="lead_sudden_stop",
        initiator_track_id="leader",
        responder_track_id="follower",
        role_track_ids={"stopping_leader": "leader", "follower": "follower"},
        trigger_score=0.5,
        seed_risk_metric="stopping_distance_margin",
        seed_risk_value=4.0,
        target_risk_definition={
            "metric": "stopping_distance_margin",
            "direction": "lower_is_riskier",
            "source": "semantic",
            "target_range": [0.5, 8.0],
        },
        source_path="train/scene/scenario_scene.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"deceleration_mps2": 3.0},
    )


def _scenario() -> Scenario:
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    positions = np.column_stack((np.arange(110, dtype=np.float64), np.zeros(110)))
    return Scenario(
        scenario_id="scene",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="leader",
        agents=[
            AgentTrack(
                track_id="leader",
                object_type="vehicle",
                positions=positions,
                velocities=np.tile([10.0, 0.0], (110, 1)),
                headings=np.zeros(110),
                observed_mask=observed,
                is_focal=True,
            )
        ],
        map_polylines=[],
    )


def test_pilot_review_resume_keeps_stable_summary_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    scenario = _scenario()
    data_root = tmp_path / "data"
    scenario_path = data_root / record.source_path
    scenario_path.parent.mkdir(parents=True)
    scenario_path.write_bytes(b"placeholder")
    seed_manifest = tmp_path / "seeds.csv"
    seed_manifest.write_text("placeholder\n", encoding="utf-8")
    config = SimpleNamespace(
        inputs=SimpleNamespace(seed_manifest=seed_manifest, data_root=data_root)
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.load_counterfactual_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.read_seed_records",
        lambda _: [record],
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.load_av2_scenario",
        lambda _: scenario,
    )
    raw_candidate = SimpleNamespace(
        candidate_id="candidate",
        scenario_id="scene",
        skill_id="lead_sudden_stop",
        target_track_id="leader",
        future_xy_global=scenario.agents[0].positions[50:110].copy(),
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.load_raw_shard_candidates",
        lambda _: (raw_candidate,),
    )

    def render(_, __, output_dir):
        path = Path(output_dir) / "review.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_SIGNATURE + b"valid")
        return path

    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.render_seed_review",
        render,
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "kind": "active_pilot_review_manifest",
            "analysis_id": "analysis",
            "case_count": 1,
            "cases": [
                {
                    "candidate_id": "candidate",
                    "task_id": "task",
                    "scenario_id": "scene",
                    "skill_id": "lead_sudden_stop",
                    "seed_record_id": seed_record_id(record),
                    "target_track_id": "leader",
                    "disposition": "accepted",
                    "evaluation_arm": "learned_conditioned",
                    "raw": {"commit": "raw.commit.json", "offset": 0},
                }
            ],
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )
    analysis_path = tmp_path / "analysis.json"
    _write_json(
        analysis_path,
        {
            "kind": "active_pilot_gate_analysis",
            "status": "passed",
            "analysis_id": "analysis",
            "outputs": {
                "review_manifest": str(manifest_path),
                "review_manifest_sha256": file_sha256(manifest_path),
            },
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )
    output_root = tmp_path / "review"

    first = render_active_pilot_review(
        gate_analysis_path=analysis_path,
        generation_config_path=tmp_path / "config.yaml",
        output_root=output_root,
        repository_root=tmp_path,
    )
    first_sha256 = file_sha256(output_root / "pilot_bev_review.json")
    second = render_active_pilot_review(
        gate_analysis_path=analysis_path,
        generation_config_path=tmp_path / "config.yaml",
        output_root=output_root,
        repository_root=tmp_path,
    )

    assert first["rendered_case_count"] == 1
    assert first["resumed_case_count"] == 0
    assert second["rendered_case_count"] == 0
    assert second["resumed_case_count"] == 1
    assert file_sha256(output_root / "pilot_bev_review.json") == first_sha256


def test_pilot_review_rejects_raw_candidate_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record()
    data_root = tmp_path / "data"
    scenario_path = data_root / record.source_path
    scenario_path.parent.mkdir(parents=True)
    scenario_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.load_counterfactual_config",
        lambda _: SimpleNamespace(
            inputs=SimpleNamespace(
                seed_manifest=tmp_path / "seeds.csv",
                data_root=data_root,
            )
        ),
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.read_seed_records",
        lambda _: [record],
    )
    monkeypatch.setattr(
        "skilldrive.generation.pilot_review.load_raw_shard_candidates",
        lambda _: (
            SimpleNamespace(
                candidate_id="different",
                scenario_id="scene",
                skill_id="lead_sudden_stop",
                target_track_id="leader",
            ),
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    _write_json(
        manifest_path,
        {
            "kind": "active_pilot_review_manifest",
            "analysis_id": "analysis",
            "case_count": 1,
            "cases": [
                {
                    "candidate_id": "candidate",
                    "task_id": "task",
                    "scenario_id": "scene",
                    "skill_id": "lead_sudden_stop",
                    "seed_record_id": seed_record_id(record),
                    "target_track_id": "leader",
                    "disposition": "accepted",
                    "evaluation_arm": "learned_conditioned",
                    "raw": {"commit": "raw.commit.json", "offset": 0},
                }
            ],
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )
    analysis_path = tmp_path / "analysis.json"
    _write_json(
        analysis_path,
        {
            "kind": "active_pilot_gate_analysis",
            "status": "passed",
            "analysis_id": "analysis",
            "outputs": {
                "review_manifest": str(manifest_path),
                "review_manifest_sha256": file_sha256(manifest_path),
            },
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )

    with pytest.raises(ValueError, match="raw candidate identity mismatch"):
        render_active_pilot_review(
            gate_analysis_path=analysis_path,
            generation_config_path=tmp_path / "config.yaml",
            output_root=tmp_path / "review",
            repository_root=tmp_path,
        )

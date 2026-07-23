from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.generation.run_counterfactual_pipeline as pipeline
import skilldrive.data as data_module
import skilldrive.data.av2_reader as av2_reader
import skilldrive.filtering.common as common_filter
import skilldrive.generation.inference as inference
from skilldrive.filtering.contracts import FilterCheck, FilterStage
from skilldrive.generation.contracts import FilterRejection
from skilldrive.seeds.records import SeedRecord


CONFIG = Path("configs/generation/counterfactual_v1.yaml")
FILTER_CONFIG = Path("configs/generation/filters_v1.yaml")


def _record(
    *,
    scenario_id: str,
    skill_id: str,
    primary_role: str,
    target_track_id: str,
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=target_track_id,
        responder_track_id=f"{scenario_id}-context",
        role_track_ids={
            primary_role: target_track_id,
            "context_actor": f"{scenario_id}-context",
        },
        trigger_score=0.5,
        seed_risk_metric="minimum_distance_m",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "minimum_distance_m",
            "target_range": [0.0, 2.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"test_parameter": 1.0},
    )


def test_repair_prior_smoke_runs_three_existing_arms_without_touching_legacy_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = (
        _record(
            scenario_id="slow-scene",
            skill_id="slow_lead_blockage",
            primary_role="slow_leader",
            target_track_id="slow-target",
        ),
        _record(
            scenario_id="construction-scene",
            skill_id="construction_object_lane_blockage",
            primary_role="responding_vehicle",
            target_track_id="construction-target",
        ),
    )
    output_root = tmp_path / "outputs"
    checkpoint = tmp_path / "repair-overfit.pt"
    manifest = tmp_path / "run_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint_sha = "c" * 64
    manifest_sha = "d" * 64
    captured: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "read_seed_records", lambda path: records)
    monkeypatch.setattr(
        pipeline,
        "_select_smoke_records",
        lambda config, values: list(records),
    )
    monkeypatch.setattr(
        pipeline,
        "_verified_hash",
        lambda path, expected, name: expected,
    )
    monkeypatch.setattr(
        pipeline,
        "_read_json",
        lambda path, name: {
            "leakage_check": {
                "candidate_pool_final_validation_overlap": 0,
                "candidate_pool_internal_validation_overlap": 0,
                "candidate_pool_outside_formal_train": 0,
                "selected_final_validation_overlap": 0,
                "selected_internal_validation_overlap": 0,
                "status": "passed",
            }
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_seed_source_path",
        lambda data_root, source_path: tmp_path / source_path,
    )
    monkeypatch.setattr(pipeline, "build_cvae_schema", lambda path: object())
    policy = SimpleNamespace(
        maximum_seam_speed_mps=20.0,
        maximum_speed_mps=40.0,
        maximum_acceleration_mps2=10.0,
        maximum_deceleration_mps2=10.0,
        maximum_jerk_mps3=20.0,
        maximum_curvature_per_m=1.0,
        maximum_heading_rate_rad_s=3.0,
        minimum_heading_speed_mps=0.5,
    )
    monkeypatch.setattr(
        pipeline,
        "load_filter_config",
        lambda path: SimpleNamespace(kinematics_by_type={"vehicle": policy}),
    )

    def load_repair_cvae(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            checkpoint_sha256=checkpoint_sha,
            run_manifest_sha256=manifest_sha,
            schema_sha256=kwargs["expected_schema_sha256"],
            manifest_stage="repair-overfit",
        )

    monkeypatch.setattr(inference, "load_repair_cvae", load_repair_cvae)

    def load_history(path):
        return SimpleNamespace(
            timestamps=np.arange(50, dtype=np.int64),
            metadata={"temporal_scope": "history_only"},
        )

    monkeypatch.setattr(av2_reader, "load_av2_history_scenario", load_history)
    monkeypatch.setattr(
        av2_reader,
        "load_av2_scenario",
        lambda path: SimpleNamespace(
            agents=(
                SimpleNamespace(track_id="slow-target", object_type="vehicle"),
                SimpleNamespace(
                    track_id="construction-target",
                    object_type="vehicle",
                ),
            )
        ),
    )

    def tensorize_prior_context(scenario, spec, schema):
        del scenario, schema
        return SimpleNamespace(
            target_track_id=spec.target_track_id,
            actor_track_ids=(spec.target_track_id,),
            anchor_origin_global=np.zeros(2, dtype=np.float32),
            anchor_heading_global=np.float32(0.0),
            condition_skill_id=spec.condition_skill_id,
        )

    monkeypatch.setattr(
        data_module,
        "tensorize_prior_context",
        tensorize_prior_context,
    )

    def generate_prior_batch(runtime, contexts, latent_seeds, **kwargs):
        del runtime, kwargs
        futures = np.zeros((3, 8, 60, 2), dtype=np.float32)
        for batch_index, context in enumerate(contexts):
            if context.target_track_id == "construction-target":
                marker = 3.0
            elif context.condition_skill_id == "<none>":
                marker = 2.0
            else:
                marker = 1.0
            for candidate_index in range(8):
                futures[batch_index, candidate_index, :, 0] = candidate_index
                futures[batch_index, candidate_index, :, 1] = marker
        return SimpleNamespace(
            future_position_local=futures,
            latent=np.zeros((3, 8, 16), dtype=np.float32),
            prior_mean=np.zeros((3, 16), dtype=np.float32),
            prior_logvar=np.zeros((3, 16), dtype=np.float32),
        )

    monkeypatch.setattr(inference, "generate_prior_batch", generate_prior_batch)

    def check_kinematics(scenario, target_track_id, future, limits):
        del scenario, target_track_id, limits
        candidate_index = int(round(float(future[0, 0])))
        marker = int(round(float(future[0, 1])))
        if marker == 1:
            passed = candidate_index % 2 == 0
        elif marker == 2:
            passed = candidate_index % 3 == 0
        else:
            passed = True
        return FilterCheck(
            stage=FilterStage.KINEMATICS,
            rejection_reasons=(
                () if passed else (FilterRejection.JERK_LIMIT_EXCEEDED,)
            ),
            metrics={
                "seam_speed_mps": float(marker),
                "maximum_speed_mps": float(candidate_index + marker),
                "maximum_acceleration_mps2": float(candidate_index + 1),
                "maximum_deceleration_mps2": float(candidate_index + 2),
                "maximum_jerk_mps3": float(candidate_index + marker + 3),
                "maximum_curvature_per_m": 0.1,
                "maximum_heading_rate_rad_s": 0.2,
            },
        )

    monkeypatch.setattr(common_filter, "check_kinematics", check_kinematics)

    summary = pipeline.run_repair_prior_smoke(
        config_path=CONFIG,
        filter_config_path=FILTER_CONFIG,
        output_root=output_root,
        device="cuda",
        repair_checkpoint_path=checkpoint,
        repair_checkpoint_sha256=checkpoint_sha,
        repair_run_manifest_path=manifest,
        repair_run_manifest_sha256=manifest_sha,
        repair_checkpoint_mode="diagnostic-overfit",
    )

    assert captured["checkpoint_mode"] == "diagnostic-overfit"
    assert summary["task_count"] == 3
    assert summary["candidate_count"] == 24
    assert summary["checkpoint_use"] == {
        "requested_mode": "diagnostic-overfit",
        "run_manifest_stage": "repair-overfit",
        "repair_contract": "cvae_generation_repair_v1",
        "diagnostic_only": True,
        "formal_active": False,
        "promotion_status": "not_promoted_by_smoke",
    }
    assert summary["kinematics"]["paired_control"]["pair_count"] == 8
    assert (
        summary["kinematics"]["paired_control"]["shared_epsilon_pair_count"]
        == 8
    )
    evidence_path = Path(summary["outputs"]["kinematic_evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["candidate_count"] == 24
    assert len(evidence["rows"]) == 24
    assert not (output_root / "pilot" / "smoke").exists()

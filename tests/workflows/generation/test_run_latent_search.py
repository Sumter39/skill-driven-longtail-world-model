from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import skilldrive.generation.latent_search_workflow as workflow
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.generation.contracts import GenerationTask
from skilldrive.generation.latent_search import LatentSearchTask
from skilldrive.generation.planning import seed_record_id, semantic_generation_config_sha256
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds.records import SeedRecord


CHECKPOINT_SHA = "a" * 64
RUN_MANIFEST_SHA = "b" * 64
SCHEMA_SHA = "c" * 64
FILTER_SHA = "f" * 64


def _record(skill_id: str, scenario_id: str) -> SeedRecord:
    target = f"target-{scenario_id}"
    other = f"other-{scenario_id}"
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=target,
        responder_track_id=other,
        role_track_ids={"target": target, "other": other},
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
        sampled_parameters={"target_gap_m": 1.0},
    )


def _generation_config(root: Path, records) -> CounterfactualGenerationConfig:
    skill_modes = (
        ("forced_lane_change_around_blockage", "rule_guided_prior_search", "none_skill_id"),
        ("jaywalking_pedestrian_crossing", "learned_conditioned_prior", "requested_skill_id"),
        ("slow_lead_blockage", "learned_conditioned_prior", "requested_skill_id"),
        ("construction_object_lane_blockage", "rule_guided_prior_search", "none_skill_id"),
    )
    skills = tuple(
        SkillGenerationConfig(
            skill_id=skill_id,
            primary_generated_role="target",
            proposal_mode=mode,
            condition_skill_strategy=strategy,
            joint_generation_limited=False,
        )
        for skill_id, mode, strategy in skill_modes
    )
    config = CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1_mock",
        formal_catalog=root / "configs" / "skills" / "catalog.yaml",
        candidate_catalog=root / "configs" / "skills" / "candidate_catalog.yaml",
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=root / "best.pt",
            sha256=CHECKPOINT_SHA,
            run_manifest=root / "run_manifest.json",
            run_manifest_sha256=RUN_MANIFEST_SHA,
            schema_sha256=SCHEMA_SHA,
        ),
        inputs=GenerationInputConfig(
            data_root=root / "data",
            seed_manifest=root / "seeds.csv",
            seed_manifest_sha256="d" * 64,
            training_cache_manifest=root / "cache.json",
            training_cache_manifest_sha256="e" * 64,
            leakage_audit=root / "audit.json",
            leakage_audit_sha256="9" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=1,
            pilot_candidates_per_task=16,
            formal_candidates_per_task=16,
        ),
        formal_skill_ids=tuple(item.skill_id for item in skills),
        candidate_skill_ids=(),
        skills=skills,
    )
    for record in records:
        path = config.inputs.data_root.joinpath(*Path(record.source_path).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return config


def _tasks(config: CounterfactualGenerationConfig, records) -> tuple[LatentSearchTask, ...]:
    by_skill = {record.skill_id: record for record in records}
    semantic = semantic_generation_config_sha256(config)
    rows = (
        ("forced_lane_change_deepest_funnel", "forced_lane_change_around_blockage", "rule_guided_none", "<none>"),
        ("jaywalking_condition_reverse", "jaywalking_pedestrian_crossing", "learned_conditioned", "jaywalking_pedestrian_crossing"),
        ("jaywalking_condition_reverse", "jaywalking_pedestrian_crossing", "learned_none_control", "<none>"),
        ("slow_lead_learned_failure", "slow_lead_blockage", "learned_conditioned", "slow_lead_blockage"),
        ("slow_lead_learned_failure", "slow_lead_blockage", "learned_none_control", "<none>"),
        ("construction_rule_deepest_funnel", "construction_object_lane_blockage", "rule_guided_none", "<none>"),
    )
    result = []
    for index, (representative, skill_id, arm, condition) in enumerate(rows):
        record = by_skill[skill_id]
        mode = config.skills_by_id[skill_id].proposal_mode
        task = GenerationTask.create(
            task_index=index,
            seed_record_id=seed_record_id(record),
            scenario_id=record.scenario_id,
            skill_id=skill_id,
            target_track_id=record.role_track_ids["target"],
            proposal_mode=mode,
            condition_skill_id=condition,
            candidate_budget=4096,
            checkpoint_sha256=CHECKPOINT_SHA,
            semantic_config_sha256=semantic,
        )
        result.append(LatentSearchTask(representative, arm, task))
    return tuple(result)


def _scenario(record: SeedRecord, *, history_only: bool) -> Scenario:
    steps = 50 if history_only else 110
    timestamps = np.arange(steps, dtype=np.int64) * 100_000_000
    positions = np.column_stack(
        (np.arange(steps, dtype=np.float64) * 0.1, np.zeros(steps))
    )
    target = AgentTrack(
        track_id=record.role_track_ids["target"],
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([1.0, 0.0], (steps, 1)),
        headings=np.zeros(steps),
        observed_mask=np.ones(steps, dtype=bool),
        is_focal=True,
    )
    return Scenario(
        scenario_id=record.scenario_id,
        city_name="city",
        timestamps=timestamps,
        focal_track_id=target.track_id,
        agents=[target],
        map_polylines=[],
        metadata={"temporal_scope": "history_only"} if history_only else {},
    )


@dataclass
class _MockValidation:
    candidate_input: Any

    def compact(self, *, cohort: str):
        raw = self.candidate_input.bound.raw
        task = self.candidate_input.bound.task
        checks = tuple(
            SimpleNamespace(
                check=SimpleNamespace(
                    stage=SimpleNamespace(value=stage),
                    passed=True,
                    metrics={},
                )
            )
            for stage in (
                "schema_finite",
                "history_invariants",
                "kinematics",
                "map",
                "collision",
                "target_risk",
                "skill_trigger",
                "parameter_realization",
            )
        )
        return SimpleNamespace(
            identity=SimpleNamespace(
                candidate_id=raw.candidate_id,
                task_id=task.task_id,
            ),
            cohort=cohort,
            checks=checks,
            quality_passed=True,
        )


def test_latent_search_mock_workflow_keeps_only_top_k_and_resumes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    records = (
        _record("forced_lane_change_around_blockage", "scene-forced"),
        _record("jaywalking_pedestrian_crossing", "scene-jay"),
        _record("slow_lead_blockage", "scene-slow"),
        _record("construction_object_lane_blockage", "scene-construction"),
    )
    generation_config = _generation_config(tmp_path, records)
    tasks = _tasks(generation_config, records)
    search_config_path = Path("configs/generation/latent_search_v1.yaml")
    generation_config_path = tmp_path / "generation.yaml"
    filter_config_path = tmp_path / "filters.yaml"
    detection_config_path = tmp_path / "detection.yaml"
    manifest_path = tmp_path / "representatives.json"
    for path in (
        generation_config_path,
        filter_config_path,
        detection_config_path,
        manifest_path,
    ):
        path.write_text("{}\n", encoding="utf-8")

    scenarios_by_path = {
        generation_config.inputs.data_root.joinpath(*Path(record.source_path).parts): record
        for record in records
    }
    calls = {"generation": 0, "filtering": 0}
    policy = SimpleNamespace(
        maximum_seam_speed_mps=20.0,
        maximum_speed_mps=20.0,
        maximum_acceleration_mps2=20.0,
        maximum_deceleration_mps2=20.0,
        maximum_jerk_mps3=200.0,
        maximum_curvature_per_m=2.0,
        maximum_heading_rate_rad_s=4.0,
        minimum_heading_speed_mps=0.2,
    )
    filter_config = SimpleNamespace(kinematics_by_type={"vehicle": policy})
    manifest = SimpleNamespace(
        sha256="1" * 64,
        pilot={
            "pilot_run_id": "2" * 64,
            "summary_sha256": "3" * 64,
            "checkpoint_sha256": CHECKPOINT_SHA,
        },
    )

    monkeypatch.setattr(workflow, "load_counterfactual_config", lambda _: generation_config)
    monkeypatch.setattr(workflow, "load_filter_config", lambda _: filter_config)
    monkeypatch.setattr(workflow, "load_detection_config", lambda _: object())
    monkeypatch.setattr(workflow, "load_latent_search_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(workflow, "build_latent_search_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(workflow, "read_seed_records", lambda _: records)
    monkeypatch.setattr(workflow, "build_cvae_schema", lambda _: object())
    monkeypatch.setattr(workflow, "load_configured_cvae", lambda **kwargs: object())
    monkeypatch.setattr(
        workflow,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(
            semantic_sha256=FILTER_SHA,
            file_sha256={"mock": "4" * 64},
        ),
    )
    monkeypatch.setattr(workflow, "load_skill", lambda _: SimpleNamespace())

    def load_history(path: Path):
        return _scenario(scenarios_by_path[path], history_only=True)

    def load_full(path: Path):
        return _scenario(scenarios_by_path[path], history_only=False)

    monkeypatch.setattr(workflow, "load_av2_history_scenario", load_history)
    monkeypatch.setattr(workflow, "load_av2_scenario", load_full)
    monkeypatch.setattr(
        workflow,
        "tensorize_prior_context",
        lambda scenario, spec, schema: SimpleNamespace(
            target_track_id=spec.target_track_id,
            anchor_origin_global=scenario.agents[0].positions[49].copy(),
            anchor_heading_global=0.0,
        ),
    )

    def generate(runtime, contexts, latent_seeds, *, use_bfloat16):
        calls["generation"] += 1
        sample_count = latent_seeds.shape[1]
        future = np.column_stack(
            (np.arange(1, 61, dtype=np.float32) * 0.1, np.zeros(60, dtype=np.float32))
        )
        return SimpleNamespace(
            future_position_local=np.repeat(
                future[None, None, :, :],
                sample_count,
                axis=1,
            )
        )

    monkeypatch.setattr(workflow, "generate_prior_batch", generate)
    monkeypatch.setattr(
        workflow,
        "validate_candidate",
        lambda candidate, **kwargs: (
            calls.__setitem__("filtering", calls["filtering"] + 1)
            or _MockValidation(candidate)
        ),
    )

    def finalize(validations, **kwargs):
        decisions = tuple(
            SimpleNamespace(
                candidate_id=item.identity.candidate_id,
                accepted=True,
                metrics={"first_failed_stage": None},
                rejection_reasons=(),
            )
            for item in validations
        )
        return SimpleNamespace(
            validations=tuple(validations),
            decisions=decisions,
            stage_execution_counts={"mock": len(validations)},
            stage_elapsed_seconds={"mock": 0.0},
        )

    monkeypatch.setattr(workflow, "finalize_candidate_validations", finalize)

    def write_indexes(root, raw_shards, decisions, **kwargs):
        root.mkdir(parents=True, exist_ok=True)
        accepted = root / "accepted.jsonl"
        rejected = root / "rejected.jsonl"
        commit = root / "filter-index.commit.json"
        accepted.write_text("\n".join(item.candidate_id for item in decisions) + "\n")
        rejected.write_text("")
        commit.write_text("{}\n")
        return SimpleNamespace(
            accepted_count=len(decisions),
            rejected_count=0,
            accepted_path=accepted,
            rejected_path=rejected,
            commit_path=commit,
        )

    monkeypatch.setattr(workflow, "write_filter_indexes", write_indexes)

    arguments = {
        "generation_config_path": generation_config_path,
        "filter_config_path": filter_config_path,
        "detection_config_path": detection_config_path,
        "latent_search_config_path": search_config_path,
        "representative_manifest_path": manifest_path,
        "output_root": tmp_path / "outputs",
        "device": "cuda",
        "progress_interval_seconds": 60.0,
        "repository_root": tmp_path,
    }
    first = workflow.run_latent_search_workflow(**arguments)
    assert first["candidate_count"] == 6 * 4096
    assert first["kinematic_passed_count"] == 6 * 4096
    assert first["raw_saved_count"] == 6 * 64
    assert first["accepted_count"] == 6 * 64
    assert first["raw_immutable_verified"] is True
    assert all(item["shared_latent_verified"] for item in first["paired_latent"])
    assert calls["generation"] == 6 * 8
    assert calls["filtering"] == 6 * 64

    second = workflow.run_latent_search_workflow(**arguments)
    assert second["run_id"] == first["run_id"]
    assert calls["generation"] == 6 * 8
    assert calls["filtering"] == 2 * 6 * 64

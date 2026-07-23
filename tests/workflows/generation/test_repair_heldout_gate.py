from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import skilldrive.generation.heldout_gate as gate_module
import skilldrive.data as skilldrive_data
import skilldrive.data.av2_reader as av2_reader
import skilldrive.filtering.pipeline as filter_pipeline
import skilldrive.generation.config as generation_config_module
import skilldrive.generation.inference as generation_inference
import skilldrive.skills.detection as skill_detection
import skilldrive.skills.loader as skill_loader
from scripts.generation.run_counterfactual_pipeline import build_parser
from skilldrive.generation import (
    FilterDecision,
    FilterRejection,
    GeneratedCandidate,
    GeneratedOverlay,
    GenerationTask,
    TaskPlan,
    canonical_sha256,
    latent_group_id,
    load_counterfactual_config,
    load_raw_shard_candidates,
    load_task_plan,
    paired_latent_seed,
    pilot_evaluation_arm,
    recover_paired_pilot_tasks,
    seed_record_id,
    write_filter_indexes,
    write_raw_shard,
    write_task_plan,
)
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
from skilldrive.seeds import SeedRecord, write_seed_records
from skilldrive.training.checkpoint import TrainingProgress, save_checkpoint


def _sha256(path: Path) -> str:
    return gate_module._file_sha256(path)  # noqa: SLF001


def _descriptor(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _source_plan(
    root: Path,
    *,
    slow_seed_record_id: str = "1" * 64,
    construction_seed_record_id: str = "2" * 64,
) -> tuple[Path, TaskPlan, Path]:
    source_root = root / "manifests" / "heldout"
    provisional = [
        GenerationTask.create(
            task_index=0,
            seed_record_id=construction_seed_record_id,
            scenario_id="construction-scene",
            skill_id="construction_object_lane_blockage",
            target_track_id="construction-target",
            proposal_mode="rule_guided_prior_search",
            condition_skill_id="<none>",
            candidate_budget=2,
            checkpoint_sha256="9" * 64,
            semantic_config_sha256="a" * 64,
        ),
        GenerationTask.create(
            task_index=0,
            seed_record_id=slow_seed_record_id,
            scenario_id="slow-scene",
            skill_id="slow_lead_blockage",
            target_track_id="slow-target",
            proposal_mode="learned_conditioned_prior",
            condition_skill_id="<none>",
            candidate_budget=2,
            checkpoint_sha256="9" * 64,
            semantic_config_sha256="a" * 64,
        ),
        GenerationTask.create(
            task_index=0,
            seed_record_id=slow_seed_record_id,
            scenario_id="slow-scene",
            skill_id="slow_lead_blockage",
            target_track_id="slow-target",
            proposal_mode="learned_conditioned_prior",
            condition_skill_id="slow_lead_blockage",
            candidate_budget=2,
            checkpoint_sha256="9" * 64,
            semantic_config_sha256="a" * 64,
        ),
    ]
    ordered = sorted(
        provisional,
        key=lambda task: (
            task.scenario_id,
            task.skill_id,
            task.seed_record_id,
            task.condition_skill_id,
            task.task_id,
        ),
    )
    plan = TaskPlan(
        semantic_config_sha256="a" * 64,
        execution_config_sha256="b" * 64,
        base_seed=2026,
        per_skill=1,
        candidate_budget=2,
        tasks=tuple(
            replace(task, task_index=index) for index, task in enumerate(ordered)
        ),
    )
    write_task_plan(source_root, plan)
    audit_path = root / "manifests" / "repair.audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_manifests_opened": False,
                "contract": {
                    "heldout_plan_requires_rebind_to_new_checkpoint": True
                },
                "integrity": {"pilot_validation_manifests_opened": False},
                "counts": {
                    "heldout_ability_tasks": 3,
                    "heldout_rule_tasks": 1,
                },
                "outputs": {
                    "heldout_task_plan": _descriptor(
                        source_root / "task_plan.jsonl", root
                    ),
                    "heldout_task_plan_summary": _descriptor(
                        source_root / "task_plan.summary.json", root
                    ),
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_root, plan, audit_path


def _formal_candidate(
    root: Path,
    *,
    stage: str = "repair-formal",
    role: str = "epoch_validation_candidate",
) -> tuple[Path, str, Path, str]:
    epoch_root = root / "model" / "formal" / "epoch_candidates"
    epoch_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "model" / "formal" / "run_manifest.json"
    config = load_counterfactual_config()
    manifest = {
        "stage": stage,
        "repair_contract": "cvae_generation_repair_v1",
        "schema_sha256": config.active_checkpoint.schema_sha256,
        "fingerprints": {"contract": "repair-formal-test"},
        "formal_selection": {
            "active_checkpoint_gate": (
                "heldout_generation_capability_gate_required"
            ),
            "epoch_candidate_directory": epoch_root.as_posix(),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    checkpoint_path = epoch_root / "epoch-0002-step-00000010.pt"
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_metadata = {
        "role": role,
        "active_checkpoint": False,
        "candidate_epoch": 2,
        "selection_status": (
            "unpromoted_epoch_candidate"
            if role == "epoch_validation_candidate"
            else "provisional_fde_candidate"
        ),
        "active_checkpoint_gate": "heldout_generation_capability_gate_required",
    }
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(2, 0, 10, 1.0, 1),
        fingerprints=manifest["fingerprints"],
        extra={
            "run_manifest_sha256": canonical_sha256(manifest),
            "checkpoint": checkpoint_metadata,
        },
    )
    (manifest_path.parent / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": stage,
                "stop_reason": "fixed_epoch_budget",
                "epoch_records_written": 2,
                "metrics_records_written": 2,
                "formal_selection": {
                    "frozen_epoch_budget": 2,
                    "active_checkpoint_selected": False,
                    "active_checkpoint_gate": (
                        "heldout_generation_capability_gate_required"
                    ),
                },
                "progress": {"epoch": 2},
                "final_evaluation": {"sample_count": 4},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metric_rows = [
        {
            "kind": "epoch",
            "stage": stage,
            "completed_epoch": True,
            "epoch": 0,
            "global_step": 5,
        },
        {
            "kind": "epoch",
            "stage": stage,
            "completed_epoch": True,
            "epoch": 1,
            "global_step": 10,
            "checkpoint_selection": {
                "active_checkpoint_gate": (
                    "heldout_generation_capability_gate_required"
                ),
                "epoch_candidate": checkpoint_path.as_posix(),
            },
            "validation": {
                "sample_count": 4,
                "prior": {"min_ade": 1.0, "min_fde": 2.0, "samples": 6},
                "constant_velocity": {"ade": 3.0, "fde": 4.0},
            },
            "validation_loss": {
                "total_loss": 1.5,
                "seam_velocity_loss": 0.01,
            },
        },
    ]
    (manifest_path.parent / "metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metric_rows),
        encoding="utf-8",
    )
    return (
        checkpoint_path,
        _sha256(checkpoint_path),
        manifest_path,
        _sha256(manifest_path),
    )


def _execute_record(
    *,
    skill_id: str,
    scenario_id: str,
    target_track_id: str,
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=target_track_id,
        responder_track_id=f"{scenario_id}-other",
        role_track_ids={
            "actor": target_track_id,
            "other": f"{scenario_id}-other",
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
        sampled_parameters={"target_gap_m": 1.0},
    )


def _execute_config(
    root: Path,
    records: tuple[SeedRecord, ...],
) -> CounterfactualGenerationConfig:
    seed_manifest = root / "manifests" / "seeds.csv"
    write_seed_records(seed_manifest, records)
    for record in records:
        source = root / "data" / Path(record.source_path)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    schema_sha = load_counterfactual_config().active_checkpoint.schema_sha256
    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1_test",
        formal_catalog=root / "configs" / "skills" / "catalog.yaml",
        candidate_catalog=root / "configs" / "skills" / "candidates.yaml",
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=root / "baseline.pt",
            sha256="9" * 64,
            run_manifest=root / "baseline-run.json",
            run_manifest_sha256="8" * 64,
            schema_sha256=schema_sha,
        ),
        inputs=GenerationInputConfig(
            data_root=root / "data",
            seed_manifest=seed_manifest,
            seed_manifest_sha256=_sha256(seed_manifest),
            training_cache_manifest=root / "cache.json",
            training_cache_manifest_sha256="7" * 64,
            leakage_audit=root / "audit.json",
            leakage_audit_sha256="6" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=1,
            pilot_candidates_per_task=2,
            formal_candidates_per_task=4,
        ),
        formal_skill_ids=(
            "construction_object_lane_blockage",
            "slow_lead_blockage",
        ),
        candidate_skill_ids=(),
        skills=(
            SkillGenerationConfig(
                skill_id="construction_object_lane_blockage",
                primary_generated_role="actor",
                proposal_mode="rule_guided_prior_search",
                condition_skill_strategy="none_skill_id",
                joint_generation_limited=False,
            ),
            SkillGenerationConfig(
                skill_id="slow_lead_blockage",
                primary_generated_role="actor",
                proposal_mode="learned_conditioned_prior",
                condition_skill_strategy="requested_skill_id",
                joint_generation_limited=False,
            ),
        ),
    )


def _rebind(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: CounterfactualGenerationConfig | None = None,
    slow_seed_record_id: str = "1" * 64,
    construction_seed_record_id: str = "2" * 64,
) -> tuple[dict[str, object], Path, Path, str, Path, str]:
    source_root, _, audit_path = _source_plan(
        root,
        slow_seed_record_id=slow_seed_record_id,
        construction_seed_record_id=construction_seed_record_id,
    )
    checkpoint, checkpoint_sha, manifest, manifest_sha = _formal_candidate(root)
    config = load_counterfactual_config() if config is None else config
    monkeypatch.setattr(
        gate_module,
        "load_counterfactual_config",
        lambda path, repository_root: config,
    )
    monkeypatch.setattr(
        gate_module,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(
            semantic_sha256="f" * 64,
            file_sha256={"filters.py": "e" * 64},
        ),
    )
    monkeypatch.setattr(
        gate_module,
        "_heldout_generation_source_sha256",
        lambda repository_root: {"generation.py": "d" * 64},
    )
    output_root = root / "outputs" / "heldout"
    result = gate_module.rebind_repair_heldout_plan(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        run_manifest_path=manifest,
        run_manifest_sha256=manifest_sha,
        config_path="config.yaml",
        filter_config_path="filters.yaml",
        detection_config_path="detection.yaml",
        source_plan_dir=source_root,
        repair_audit_path=audit_path,
        output_root=output_root,
        repository_root=root,
    )
    gate_root = output_root / checkpoint_sha
    return result, gate_root, checkpoint, checkpoint_sha, manifest, manifest_sha


class _ExecuteValidation:
    def __init__(self, candidate_input) -> None:
        self._candidate_input = candidate_input

    def compact(self, *, cohort: str):
        raw = self._candidate_input.bound.raw
        accepted = raw.candidate_index == 0 and cohort != "learned_none_control"
        return SimpleNamespace(
            identity=SimpleNamespace(
                candidate_id=raw.candidate_id,
                task_id=raw.task_id,
                candidate_index=raw.candidate_index,
                latent_seed=raw.latent_seed,
            ),
            cohort=cohort,
            quality_passed=accepted,
        )


def _install_execute_harness(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    records = (
        _execute_record(
            skill_id="slow_lead_blockage",
            scenario_id="slow-scene",
            target_track_id="slow-target",
        ),
        _execute_record(
            skill_id="construction_object_lane_blockage",
            scenario_id="construction-scene",
            target_track_id="construction-target",
        ),
    )
    config = _execute_config(root, records)
    rebound, gate_root, checkpoint, checkpoint_sha, manifest, manifest_sha = _rebind(
        root,
        monkeypatch,
        config=config,
        slow_seed_record_id=seed_record_id(records[0]),
        construction_seed_record_id=seed_record_id(records[1]),
    )
    calls: dict[str, object] = {
        "events": [],
        "model_loads": 0,
        "latent_batches": [],
        "filter_calls": 0,
    }

    def load_history(path: Path):
        calls["events"].append(("history", Path(path)))
        return SimpleNamespace(
            timestamps=np.arange(50, dtype=np.int64),
            metadata={"temporal_scope": "history_only"},
        )

    def load_full(path: Path):
        calls["events"].append(("full", Path(path)))
        return SimpleNamespace(source_path=Path(path))

    def tensorize_prior_context(scenario, spec, schema):
        return SimpleNamespace(
            target_track_id=spec.target_track_id,
            anchor_origin_global=np.zeros(2, dtype=np.float64),
            anchor_heading_global=0.0,
        )

    def load_repair_cvae(**kwargs):
        assert kwargs["checkpoint_mode"] == "formal"
        calls["model_loads"] += 1
        calls["events"].append(("model", kwargs["checkpoint_path"]))
        return SimpleNamespace(device=kwargs["device"])

    def generate_prior_batch(runtime, contexts, latent_seeds, **kwargs):
        seeds = np.asarray(latent_seeds, dtype=np.int64)
        calls["latent_batches"].append(seeds.copy())
        calls["events"].append(("generate", seeds.copy()))
        futures = np.empty((len(contexts), seeds.shape[1], 60, 2), dtype=np.float32)
        base = np.linspace(0.0, 6.0, 60, dtype=np.float32)
        for row in range(len(contexts)):
            for candidate_index in range(seeds.shape[1]):
                marker = np.float32(int(seeds[row, candidate_index]) % 997) / 997.0
                futures[row, candidate_index, :, 0] = base + marker
                futures[row, candidate_index, :, 1] = marker
        return SimpleNamespace(future_position_local=futures)

    def validate_candidate(candidate_input, **kwargs):
        calls["filter_calls"] += 1
        return _ExecuteValidation(candidate_input)

    def finalize_candidate_validations(
        compact_results,
        *,
        filter_semantic_sha256,
        **kwargs,
    ):
        compact = tuple(compact_results)
        decisions = tuple(
            FilterDecision.create(
                candidate_id=item.identity.candidate_id,
                filter_config_sha256=filter_semantic_sha256,
                filter_contract_version=FILTER_CONTRACT_VERSION,
                accepted=item.quality_passed,
                rejection_reasons=(
                    ()
                    if item.quality_passed
                    else (FilterRejection.JERK_LIMIT_EXCEEDED,)
                ),
                metrics={
                    "task_id": item.identity.task_id,
                    "candidate_index": item.identity.candidate_index,
                    "latent_seed": item.identity.latent_seed,
                    "first_failed_stage": (
                        None if item.quality_passed else "kinematics"
                    ),
                },
            )
            for item in compact
        )
        return SimpleNamespace(
            decisions=decisions,
            validations=compact,
            stage_execution_counts={"mock": len(compact)},
            stage_elapsed_seconds={"mock": 0.0},
        )

    monkeypatch.setattr(skilldrive_data, "build_cvae_schema", lambda path: object())
    monkeypatch.setattr(
        skilldrive_data,
        "tensorize_prior_context",
        tensorize_prior_context,
    )
    monkeypatch.setattr(av2_reader, "load_av2_history_scenario", load_history)
    monkeypatch.setattr(av2_reader, "load_av2_scenario", load_full)
    monkeypatch.setattr(generation_inference, "load_repair_cvae", load_repair_cvae)
    monkeypatch.setattr(
        generation_inference,
        "generate_prior_batch",
        generate_prior_batch,
    )
    monkeypatch.setattr(
        generation_config_module,
        "load_filter_config",
        lambda path: object(),
    )
    monkeypatch.setattr(filter_pipeline, "validate_candidate", validate_candidate)
    monkeypatch.setattr(
        filter_pipeline,
        "finalize_candidate_validations",
        finalize_candidate_validations,
    )
    monkeypatch.setattr(skill_detection, "load_detection_config", lambda path: object())
    monkeypatch.setattr(
        skill_loader,
        "load_skill",
        lambda path: SimpleNamespace(skill_id=Path(path).stem),
    )
    return {
        "root": root,
        "records": records,
        "config": config,
        "rebound": rebound,
        "gate_root": gate_root,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "manifest": manifest,
        "manifest_sha": manifest_sha,
        "source_root": root / "manifests" / "heldout",
        "audit_path": root / "manifests" / "repair.audit.json",
        "calls": calls,
        "execute_kwargs": {
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "run_manifest_path": manifest,
            "run_manifest_sha256": manifest_sha,
            "config_path": "config.yaml",
            "filter_config_path": "filters.yaml",
            "detection_config_path": "detection.yaml",
            "source_plan_dir": root / "manifests" / "heldout",
            "repair_audit_path": root / "manifests" / "repair.audit.json",
            "output_root": root / "outputs" / "heldout",
            "device": "cuda",
            "task_batch_size": 8,
            "progress_interval_seconds": 0.001,
            "repository_root": root,
        },
    }


def _load_frozen_task_plan(directory: Path) -> TaskPlan:
    summary = json.loads(
        (directory / "task_plan.summary.json").read_text(encoding="utf-8")
    )
    return load_task_plan(
        directory,
        expected_semantic_config_sha256=summary["semantic_config_sha256"],
        current_execution_config_sha256=summary["execution_config_sha256"],
    ).plan


def _write_complete_filter_result(root: Path, gate_root: Path) -> TaskPlan:
    summary = json.loads(
        (gate_root / "task_plan.summary.json").read_text(encoding="utf-8")
    )
    plan = load_task_plan(
        gate_root,
        expected_semantic_config_sha256=summary["semantic_config_sha256"],
        current_execution_config_sha256=summary["execution_config_sha256"],
    ).plan
    source_summary = json.loads(
        (root / "manifests" / "heldout" / "task_plan.summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_plan = load_task_plan(
        root / "manifests" / "heldout",
        expected_semantic_config_sha256=source_summary["semantic_config_sha256"],
        current_execution_config_sha256=source_summary["execution_config_sha256"],
    ).plan
    source_by_index = {task.task_index: task for task in source_plan.tasks}
    decisions = []
    commits = []
    for task in plan.tasks:
        candidates = []
        arm = pilot_evaluation_arm(task)
        group_id = latent_group_id(task)
        for candidate_index in range(task.candidate_budget):
            seed = paired_latent_seed(
                plan.base_seed,
                source_by_index[task.task_index],
                candidate_index,
            )
            candidate = GeneratedCandidate(
                task_id=task.task_id,
                candidate_index=candidate_index,
                latent_seed=seed,
                scenario_id=task.scenario_id,
                skill_id=task.skill_id,
                proposal_mode=task.proposal_mode,
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=task.semantic_config_sha256,
                overlay=GeneratedOverlay(
                    target_track_id=task.target_track_id,
                    future_xy_global=np.column_stack(
                        (
                            np.linspace(0.0, 6.0, 60, dtype=np.float32),
                            np.full(60, task.task_index, dtype=np.float32),
                        )
                    ),
                ),
                metadata={
                    "condition_skill_id": task.condition_skill_id,
                    "evaluation_arm": arm,
                    "latent_group_id": group_id,
                },
            )
            candidates.append(candidate)
            accepted = candidate_index == 0 and arm != "learned_none_control"
            decisions.append(
                FilterDecision.create(
                    candidate_id=candidate.candidate_id,
                    filter_config_sha256="f" * 64,
                    filter_contract_version=FILTER_CONTRACT_VERSION,
                    accepted=accepted,
                    rejection_reasons=(
                        ()
                        if accepted
                        else (FilterRejection.JERK_LIMIT_EXCEEDED,)
                    ),
                    metrics={
                        "first_failed_stage": None if accepted else "kinematics",
                        "skill_id": task.skill_id,
                    },
                )
            )
        commits.append(
            write_raw_shard(
                gate_root / "raw",
                task.task_index,
                candidates,
                semantic_config_sha256=plan.semantic_config_sha256,
                execution_config_sha256=plan.execution_config_sha256,
            )
        )
    write_filter_indexes(
        gate_root / "filter",
        commits,
        decisions,
        filter_config_sha256="f" * 64,
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )
    return plan


def _dev_evidence(
    gate_root: Path,
    *,
    checkpoint_sha: str,
    manifest_sha: str,
    passed: bool,
) -> Path:
    path = gate_root / "repair_dev_candidate_gate.json"
    value = {
        "schema_version": 1,
        "kind": "repair_dev_candidate_gate",
        "checkpoint_sha256": checkpoint_sha,
        "run_manifest_sha256": manifest_sha,
        "candidate_epoch": 2,
        "source_partition": "repair_dev_from_formal_train",
        "formal_training_complete": True,
        "metrics": {"min_fde_6": 1.0, "maximum_jerk_mps3": 10.0},
        "gates": [
            {
                "name": "repair_dev_motion",
                "passed": passed,
                "comparison": "value <= threshold",
                "value": 10.0,
                "threshold": 20.0,
                "source": "frozen_repair_dev_policy",
            }
        ],
        "status": "passed" if passed else "failed",
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def test_execute_runs_source_seeded_prior_filter_and_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_execute_harness(tmp_path, monkeypatch)
    gate_root = harness["gate_root"]
    frozen_paths = (
        gate_root / "rebind_contract.json",
        gate_root / "task_plan.jsonl",
        gate_root / "task_plan.summary.json",
        gate_root / "task_rebind.json",
    )
    frozen_before = {path: path.read_bytes() for path in frozen_paths}

    summary = gate_module.execute_repair_heldout_plan(
        **harness["execute_kwargs"]
    )

    assert summary["status"] == "completed"
    assert summary["task_count"] == 3
    assert summary["candidate_count"] == 6
    assert summary["accepted_count"] == 2
    assert summary["rejected_count"] == 4
    assert summary["formal_active"] is False
    assert summary["active_config_modified"] is False
    assert summary["cross_checkpoint_latent_source_frozen"] is True
    assert summary["validation_manifests_opened"] is False
    assert summary["final_validation_accessed"] is False
    assert {path: path.read_bytes() for path in frozen_paths} == frozen_before
    assert harness["calls"]["model_loads"] == 1

    rebound_summary = json.loads(
        (gate_root / "task_plan.summary.json").read_text(encoding="utf-8")
    )
    rebound_plan = load_task_plan(
        gate_root,
        expected_semantic_config_sha256=rebound_summary["semantic_config_sha256"],
        current_execution_config_sha256=rebound_summary["execution_config_sha256"],
    ).plan
    source_summary = json.loads(
        (harness["source_root"] / "task_plan.summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_plan = load_task_plan(
        harness["source_root"],
        expected_semantic_config_sha256=source_summary["semantic_config_sha256"],
        current_execution_config_sha256=source_summary["execution_config_sha256"],
    ).plan
    source_by_index = {task.task_index: task for task in source_plan.tasks}
    rebound_by_index = {task.task_index: task for task in rebound_plan.tasks}
    source_mapping = {
        task.task_id: source_by_index[task.task_index] for task in rebound_plan.tasks
    }
    recovery = recover_paired_pilot_tasks(
        rebound_plan,
        gate_root / "raw",
        latent_seed_source_tasks=source_mapping,
    )
    assert len(recovery.raw_scan.valid_shards) == 3
    seeds_by_arm = {}
    for shard in recovery.raw_scan.valid_shards:
        task = rebound_by_index[shard.shard_index]
        source_task = source_by_index[task.task_index]
        candidates = load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=rebound_plan.semantic_config_sha256,
        )
        assert [candidate.latent_seed for candidate in candidates] == [
            paired_latent_seed(source_plan.base_seed, source_task, index)
            for index in range(task.candidate_budget)
        ]
        assert all(
            candidate.metadata["source_task_id"] == source_task.task_id
            for candidate in candidates
        )
        seeds_by_arm[pilot_evaluation_arm(task)] = tuple(
            candidate.latent_seed for candidate in candidates
        )
    assert seeds_by_arm["learned_conditioned"] == seeds_by_arm[
        "learned_none_control"
    ]

    events = [name for name, _ in harness["calls"]["events"]]
    assert max(index for index, name in enumerate(events) if name == "generate") < min(
        index for index, name in enumerate(events) if name == "full"
    )

    dev_result = gate_module.build_repair_dev_candidate_evidence(
        checkpoint_path=harness["checkpoint"],
        checkpoint_sha256=harness["checkpoint_sha"],
        run_manifest_path=harness["manifest"],
        run_manifest_sha256=harness["manifest_sha"],
        config_path="config.yaml",
        output_root=tmp_path / "outputs" / "heldout",
        repository_root=tmp_path,
    )
    assert dev_result["evidence"]["status"] == "passed"
    dev = tmp_path / dev_result["outputs"]["repair_dev_candidate_gate"]
    recommendation = gate_module.aggregate_repair_heldout_gate(
        checkpoint_path=harness["checkpoint"],
        checkpoint_sha256=harness["checkpoint_sha"],
        run_manifest_path=harness["manifest"],
        run_manifest_sha256=harness["manifest_sha"],
        repair_dev_evidence_path=dev,
        repair_dev_evidence_sha256=_sha256(dev),
        config_path="config.yaml",
        output_root=tmp_path / "outputs" / "heldout",
        repository_root=tmp_path,
    )
    assert recommendation["recommendation"] == "recommend_promotion"


def test_execute_complete_resume_skips_model_filter_and_preserves_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_execute_harness(tmp_path, monkeypatch)
    first = gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])
    gate_root = harness["gate_root"]
    persisted = {
        path: path.read_bytes()
        for path in sorted((gate_root / "raw").iterdir())
        if path.is_file()
    }
    persisted.update(
        {
            path: path.read_bytes()
            for path in sorted((gate_root / "filter").iterdir())
            if path.is_file()
        }
    )
    before = {
        "model_loads": harness["calls"]["model_loads"],
        "filter_calls": harness["calls"]["filter_calls"],
        "events": len(harness["calls"]["events"]),
    }

    second = gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])

    assert second == first
    assert harness["calls"]["model_loads"] == before["model_loads"]
    assert harness["calls"]["filter_calls"] == before["filter_calls"]
    assert len(harness["calls"]["events"]) == before["events"]
    assert {path: path.read_bytes() for path in persisted} == persisted


def test_execute_rebuilds_invalid_filter_from_durable_raw_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_execute_harness(tmp_path, monkeypatch)
    gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])
    gate_root = harness["gate_root"]
    accepted = gate_root / "filter" / "accepted.jsonl"
    accepted.write_bytes(accepted.read_bytes() + b"\n")
    model_loads = harness["calls"]["model_loads"]
    filter_calls = harness["calls"]["filter_calls"]

    summary = gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])

    assert summary["status"] == "completed"
    assert harness["calls"]["model_loads"] == model_loads
    assert harness["calls"]["filter_calls"] == filter_calls + 6
    rebound_plan = _load_frozen_task_plan(gate_root)
    source_plan = _load_frozen_task_plan(harness["source_root"])
    source_by_index = {task.task_index: task for task in source_plan.tasks}
    accepted_rows, rejected_rows, _, _ = gate_module._verify_filter_result(  # noqa: SLF001
        gate_root=gate_root,
        plan=rebound_plan,
        expected_filter_sha256="f" * 64,
        latent_seed_source_tasks={
            task.task_id: source_by_index[task.task_index]
            for task in rebound_plan.tasks
        },
    )
    assert len(accepted_rows) == 2
    assert len(rejected_rows) == 4


def test_execute_invalidates_stale_filter_commit_before_raw_rebuild_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_execute_harness(tmp_path, monkeypatch)
    gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])
    gate_root = harness["gate_root"]
    plan_summary = json.loads(
        (gate_root / "task_plan.summary.json").read_text(encoding="utf-8")
    )
    plan = load_task_plan(
        gate_root,
        expected_semantic_config_sha256=plan_summary["semantic_config_sha256"],
        current_execution_config_sha256=plan_summary["execution_config_sha256"],
    ).plan
    broken = plan.tasks[-1]
    unaffected = {
        path: path.read_bytes()
        for path in (gate_root / "raw").iterdir()
        if path.is_file() and f"{broken.task_index:05d}" not in path.name
    }
    (gate_root / "raw" / f"shard-{broken.task_index:05d}.commit.json").unlink()

    def fail_filter(*args, **kwargs):
        raise RuntimeError("injected filtering failure")

    monkeypatch.setattr(filter_pipeline, "validate_candidate", fail_filter)
    with pytest.raises(RuntimeError, match="injected filtering failure"):
        gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])

    assert not (gate_root / "filter" / "filter-index.commit.json").exists()
    assert {path: path.read_bytes() for path in unaffected} == unaffected


def test_execute_rejects_tampered_adapter_before_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_execute_harness(tmp_path, monkeypatch)
    rebind_path = harness["gate_root"] / "rebind_contract.json"
    value = json.loads(rebind_path.read_text(encoding="utf-8"))
    value["execution_adapter"]["raw_directory"] = "../../escape"
    rebind_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter is not available"):
        gate_module.execute_repair_heldout_plan(**harness["execute_kwargs"])

    assert harness["calls"]["model_loads"] == 0
    assert not (tmp_path / "escape").exists()


def test_rebind_and_gate_recommendation_are_hash_bound_and_never_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rebound, gate_root, checkpoint, checkpoint_sha, manifest, manifest_sha = _rebind(
        tmp_path,
        monkeypatch,
    )
    assert rebound["status"] == "rebound_pending_execution"
    assert rebound["formal_active"] is False
    assert rebound["rebound"]["task_count"] == 3
    assert rebound["rebound"]["candidate_count"] == 6
    _write_complete_filter_result(tmp_path, gate_root)
    dev = _dev_evidence(
        gate_root,
        checkpoint_sha=checkpoint_sha,
        manifest_sha=manifest_sha,
        passed=True,
    )

    recommendation = gate_module.aggregate_repair_heldout_gate(
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        run_manifest_path=manifest,
        run_manifest_sha256=manifest_sha,
        repair_dev_evidence_path=dev,
        repair_dev_evidence_sha256=_sha256(dev),
        config_path="config.yaml",
        output_root=tmp_path / "outputs" / "heldout",
        repository_root=tmp_path,
    )

    assert recommendation["recommendation"] == "recommend_promotion"
    assert recommendation["formal_active"] is False
    assert recommendation["active_config_modified"] is False
    summary = json.loads(
        (tmp_path / recommendation["outputs"]["heldout_gate_summary"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["task_count"] == 3
    assert summary["candidate_count"] == 6
    assert summary["funnel"]["formal_accepted_count"] == 2
    assert summary["funnel"]["control_accepted_count"] == 0


def test_gate_refuses_incomplete_or_hash_changed_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, gate_root, checkpoint, checkpoint_sha, manifest, manifest_sha = _rebind(
        tmp_path,
        monkeypatch,
    )
    plan = _write_complete_filter_result(tmp_path, gate_root)
    (gate_root / "raw" / f"shard-{plan.tasks[-1].task_index:05d}.commit.json").unlink()
    dev = _dev_evidence(
        gate_root,
        checkpoint_sha=checkpoint_sha,
        manifest_sha=manifest_sha,
        passed=True,
    )

    with pytest.raises(ValueError, match="execution is incomplete"):
        gate_module.aggregate_repair_heldout_gate(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            run_manifest_path=manifest,
            run_manifest_sha256=manifest_sha,
            repair_dev_evidence_path=dev,
            repair_dev_evidence_sha256=_sha256(dev),
            config_path="config.yaml",
            output_root=tmp_path / "outputs" / "heldout",
            repository_root=tmp_path,
        )
    assert not (gate_root / "promotion_recommendation.json").exists()


@pytest.mark.parametrize("stage", ("repair-overfit", "formal"))
def test_rebind_rejects_non_repair_formal_manifest(
    tmp_path: Path,
    stage: str,
) -> None:
    checkpoint, checkpoint_sha, manifest, _ = _formal_candidate(
        tmp_path,
        stage=stage,
    )
    manifest_sha = _sha256(manifest)
    config = load_counterfactual_config()

    with pytest.raises(ValueError, match="only stage=repair-formal"):
        gate_module.validate_repair_formal_candidate(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            run_manifest_path=manifest,
            run_manifest_sha256=manifest_sha,
            expected_schema_sha256=config.active_checkpoint.schema_sha256,
            repository_root=tmp_path,
        )


def test_provisional_fde_best_is_not_an_epoch_gate_candidate(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, manifest, manifest_sha = _formal_candidate(
        tmp_path,
        role="provisional_fde_best",
    )
    config = load_counterfactual_config()

    with pytest.raises(ValueError, match="metadata mismatch"):
        gate_module.validate_repair_formal_candidate(
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            run_manifest_path=manifest,
            run_manifest_sha256=manifest_sha,
            expected_schema_sha256=config.active_checkpoint.schema_sha256,
            repository_root=tmp_path,
        )


def test_cli_exposes_rebind_and_gate_stages() -> None:
    parser = build_parser()
    assert parser.parse_args(["--stage", "repair-heldout-dev-evidence"]).stage == (
        "repair-heldout-dev-evidence"
    )
    assert parser.parse_args(["--stage", "repair-heldout-rebind"]).stage == (
        "repair-heldout-rebind"
    )
    assert parser.parse_args(["--stage", "repair-heldout-gate"]).stage == (
        "repair-heldout-gate"
    )

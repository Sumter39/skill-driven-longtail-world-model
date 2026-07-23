"""Run staged counterfactual generation workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np

from skilldrive.data import build_cvae_schema, cvae_schema_fingerprint
from skilldrive.generation import (
    build_generation_capability_matrix,
    load_counterfactual_config,
    load_filter_config,
    write_generation_capability_matrix,
)
from skilldrive.seeds import read_seed_records


DEFAULT_OUTPUT_ROOT = Path("outputs/generation/counterfactual_v1")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping: {path}")
    return value


def _verified_hash(path: Path, expected: str, name: str) -> str:
    actual = _file_sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _validate_frozen_leakage_audit(value: Mapping[str, Any]) -> None:
    leakage = value.get("leakage_check")
    if not isinstance(leakage, Mapping):
        raise ValueError("frozen seed audit is missing leakage_check")
    expected = {
        "candidate_pool_final_validation_overlap": 0,
        "candidate_pool_internal_validation_overlap": 0,
        "candidate_pool_outside_formal_train": 0,
        "selected_final_validation_overlap": 0,
        "selected_internal_validation_overlap": 0,
        "status": "passed",
    }
    mismatches = {
        key: (leakage.get(key), expected_value)
        for key, expected_value in expected.items()
        if leakage.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"frozen seed leakage audit failed: {mismatches}")


def _validate_seed_source_path(data_root: Path, source_path: str) -> Path:
    pure = PurePosixPath(source_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "train":
        raise ValueError(f"seed source_path must stay inside Formal Train: {source_path}")
    path = data_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(f"seed scenario file is missing: {path}")
    return path


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_paths": []}
    return {
        "commit": commit,
        "dirty": bool(status),
        "changed_paths": sorted(line[3:] for line in status if len(line) >= 4),
    }


def _select_smoke_records(config, records):
    from skilldrive.generation import seed_record_id, select_pilot_records

    pilot_records = select_pilot_records(
        records,
        formal_skill_ids=config.formal_skill_ids,
        per_skill=1,
        base_seed=config.sampling.base_seed,
    )
    by_skill = {record.skill_id: record for record in pilot_records}
    selected = [
        by_skill["slow_lead_blockage"],
        by_skill["construction_object_lane_blockage"],
    ]
    return sorted(
        selected,
        key=lambda record: (
            record.scenario_id,
            record.skill_id,
            seed_record_id(record),
        ),
    )


def run_audit(
    *,
    config_path: Path,
    filter_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    from skilldrive.generation.inference import (
        validate_active_checkpoint_promotion,
    )

    config = load_counterfactual_config(config_path)
    load_filter_config(filter_config_path)
    active = config.active_checkpoint
    inputs = config.inputs

    hashes = {
        "generation_config": _file_sha256(config_path),
        "filter_config": _file_sha256(filter_config_path),
        "checkpoint": _verified_hash(active.path, active.sha256, "active checkpoint"),
        "run_manifest": _verified_hash(
            active.run_manifest,
            active.run_manifest_sha256,
            "active run manifest",
        ),
        "seed_manifest": _verified_hash(
            inputs.seed_manifest,
            inputs.seed_manifest_sha256,
            "formal seed manifest",
        ),
        "training_cache_manifest": _verified_hash(
            inputs.training_cache_manifest,
            inputs.training_cache_manifest_sha256,
            "training cache manifest",
        ),
        "leakage_audit": _verified_hash(
            inputs.leakage_audit,
            inputs.leakage_audit_sha256,
            "frozen leakage audit",
        ),
    }
    promotion = validate_active_checkpoint_promotion(active)
    if promotion is not None:
        if (
            active.promotion_recommendation is None
            or active.promotion_recommendation_sha256 is None
        ):
            raise RuntimeError("validated repair promotion identity is missing")
        hashes["promotion_recommendation"] = _verified_hash(
            active.promotion_recommendation,
            active.promotion_recommendation_sha256,
            "active checkpoint promotion recommendation",
        )
    leakage_audit = _read_json(inputs.leakage_audit, "frozen leakage audit")
    _validate_frozen_leakage_audit(leakage_audit)

    schema = build_cvae_schema(config.formal_catalog.parent)
    schema_sha256 = cvae_schema_fingerprint(schema)
    if schema_sha256 != active.schema_sha256:
        raise ValueError("active schema SHA-256 mismatch")
    hashes["schema"] = schema_sha256

    records = read_seed_records(inputs.seed_manifest)
    unique_scenarios: dict[str, str] = {}
    validated_paths: dict[str, Path] = {}
    for record in records:
        scenario_path = validated_paths.get(record.source_path)
        if scenario_path is None:
            scenario_path = _validate_seed_source_path(
                inputs.data_root,
                record.source_path,
            )
            validated_paths[record.source_path] = scenario_path
        previous = unique_scenarios.setdefault(record.scenario_id, str(scenario_path))
        if previous != str(scenario_path):
            raise ValueError(f"scenario has inconsistent source paths: {record.scenario_id}")
    if len(records) != 33_914 or len(unique_scenarios) != 5_000:
        raise ValueError(
            "formal seed manifest count mismatch: "
            f"records={len(records)}, scenarios={len(unique_scenarios)}"
        )

    training_cache_manifest = _read_json(
        inputs.training_cache_manifest,
        "training cache manifest",
    )
    matrix = build_generation_capability_matrix(
        config=config,
        records=records,
        training_cache_manifest=training_cache_manifest,
        checkpoint_path=active.path.as_posix(),
        checkpoint_sha256=active.sha256,
        schema_sha256=schema_sha256,
    )
    pilot_root = output_root / "pilot"
    matrix_path = write_generation_capability_matrix(
        pilot_root / "generation_capability_matrix.json",
        matrix,
    )
    audit = {
        "version": 1,
        "status": "passed",
        "validation_manifests_opened": False,
        "formal_seed_records": len(records),
        "formal_seed_scenarios": len(unique_scenarios),
        "formal_skill_count": len(config.formal_skill_ids),
        "active_checkpoint_stage": active.run_manifest_stage,
        "active_repair_contract": active.repair_contract,
        "candidate_skill_count": len(config.candidate_skill_ids),
        "hashes": hashes,
        "frozen_leakage_check": dict(leakage_audit["leakage_check"]),
        "git": _git_state(),
        "outputs": {"generation_capability_matrix": matrix_path.as_posix()},
    }
    audit_path = write_generation_capability_matrix(pilot_root / "input_audit.json", audit)
    print(
        "stage A audit passed: "
        f"{len(records)} labels, {len(unique_scenarios)} scenarios, "
        f"{len(config.formal_skill_ids)} formal skills"
    )
    print(f"capability matrix: {matrix_path}")
    print(f"input audit: {audit_path}")
    return audit


def run_smoke(
    *,
    config_path: Path,
    filter_config_path: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Generate eight Prior candidates for one learned and one search-only task."""

    from skilldrive.data import tensorize_prior_context
    from skilldrive.data.av2_reader import load_av2_history_scenario
    from skilldrive.generation import (
        GeneratedCandidate,
        GeneratedOverlay,
        TaskPlan,
        build_generation_task,
        canonical_sha256,
        latent_seeds_for_task,
        local_futures_to_global,
        prior_context_spec_for_task,
        semantic_generation_config_sha256,
        write_task_plan,
        write_raw_shard,
    )
    from skilldrive.generation.inference import generate_prior_batch, load_configured_cvae

    run_audit(
        config_path=config_path,
        filter_config_path=filter_config_path,
        output_root=output_root,
    )
    config = load_counterfactual_config(config_path)
    records = read_seed_records(config.inputs.seed_manifest)
    selected = _select_smoke_records(config, records)

    execution_sha256 = canonical_sha256(
        {
            "version": 1,
            "device": device,
            "batch_size": len(selected),
            "candidate_budget": 8,
            "bfloat16": False,
        }
    )
    tasks = [
        build_generation_task(
            task_index=index,
            record=record,
            config=config,
            candidate_budget=8,
        )
        for index, record in enumerate(selected)
    ]
    smoke_root = output_root / "pilot" / "smoke"
    plan_artifacts = write_task_plan(
        smoke_root,
        TaskPlan(
            semantic_config_sha256=semantic_generation_config_sha256(config),
            execution_config_sha256=execution_sha256,
            base_seed=config.sampling.base_seed,
            per_skill=1,
            candidate_budget=8,
            tasks=tuple(tasks),
        ),
    )

    schema = build_cvae_schema(config.formal_catalog.parent)
    runtime = load_configured_cvae(
        active_checkpoint=config.active_checkpoint,
        schema=schema,
        device=device,
    )
    scenarios = [
        load_av2_history_scenario(
            _validate_seed_source_path(config.inputs.data_root, record.source_path)
        )
        for record in selected
    ]
    if any(
        len(scenario.timestamps) != 50
        or scenario.metadata.get("temporal_scope") != "history_only"
        for scenario in scenarios
    ):
        raise ValueError("smoke generation must receive history-only AV2 scenarios")
    contexts = [
        tensorize_prior_context(
            scenario,
            prior_context_spec_for_task(task, record),
            schema,
        )
        for scenario, task, record in zip(scenarios, tasks, selected)
    ]
    for task, record, context in zip(tasks, selected, contexts):
        primary_role = config.skills_by_id[task.skill_id].primary_generated_role
        expected_target = record.role_track_ids[primary_role]
        if task.target_track_id != expected_target or context.target_track_id != expected_target:
            raise ValueError(f"smoke primary role binding differs for {task.skill_id}")
        if context.actor_track_ids[0] != expected_target:
            raise ValueError(f"smoke target actor is not context slot zero for {task.skill_id}")
        if hasattr(context, "target_future") or hasattr(context, "target_future_mask"):
            raise ValueError("smoke Prior context must not contain target future tensors")
    seeds = [
        latent_seeds_for_task(task, base_seed=config.sampling.base_seed)
        for task in tasks
    ]
    latent_seed_matrix = np.stack(seeds)
    generated = generate_prior_batch(
        runtime,
        contexts,
        latent_seed_matrix,
        use_bfloat16=False,
    )

    candidates: list[GeneratedCandidate] = []
    semantic_sha256 = semantic_generation_config_sha256(config)
    for batch_index, (task, record, context) in enumerate(
        zip(tasks, selected, contexts)
    ):
        global_futures = local_futures_to_global(
            generated.future_position_local[batch_index],
            context.anchor_origin_global,
            float(context.anchor_heading_global),
        )
        for candidate_index in range(task.candidate_budget):
            candidates.append(
                GeneratedCandidate(
                    task_id=task.task_id,
                    candidate_index=candidate_index,
                    latent_seed=int(latent_seed_matrix[batch_index, candidate_index]),
                    scenario_id=task.scenario_id,
                    skill_id=task.skill_id,
                    proposal_mode=task.proposal_mode,
                    checkpoint_sha256=task.checkpoint_sha256,
                    semantic_config_sha256=semantic_sha256,
                    overlay=GeneratedOverlay(
                        target_track_id=task.target_track_id,
                        future_xy_global=global_futures[candidate_index],
                    ),
                    metadata={
                        "condition_skill_id": task.condition_skill_id,
                        "primary_generated_role": config.skills_by_id[
                            task.skill_id
                        ].primary_generated_role,
                        "requested_parameters": record.sampled_parameters,
                        "detection_mode": record.evidence["detection_mode"],
                        "latent": generated.latent[
                            batch_index, candidate_index
                        ].tolist(),
                    },
                )
            )
    raw_dir = smoke_root / "raw"
    commit = write_raw_shard(
        raw_dir,
        0,
        candidates,
        semantic_config_sha256=semantic_sha256,
        execution_config_sha256=execution_sha256,
    )
    summary = {
        "version": 1,
        "status": "passed",
        "stage": "smoke",
        "task_count": len(tasks),
        "candidate_count": len(candidates),
        "skills": [task.skill_id for task in tasks],
        "task_ids": [task.task_id for task in tasks],
        "candidate_ids_sha256": commit.candidate_ids_sha256,
        "raw_commit": commit.commit_path.as_posix(),
        "task_plan": plan_artifacts.task_plan_path.as_posix(),
        "task_plan_sha256": plan_artifacts.task_plan_sha256,
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "schema_sha256": runtime.schema_sha256,
        "history_only_inputs": True,
        "future_tensors_present": False,
        "role_bindings": {
            task.skill_id: {
                "primary_generated_role": config.skills_by_id[
                    task.skill_id
                ].primary_generated_role,
                "target_track_id": task.target_track_id,
            }
            for task in tasks
        },
        "validation_manifests_opened": False,
    }
    summary_path = write_generation_capability_matrix(
        output_root / "pilot" / "smoke" / "summary.json",
        summary,
    )
    print(
        "stage B smoke passed: "
        f"{len(tasks)} tasks, {len(candidates)} deterministic candidates"
    )
    print(f"raw commit: {commit.commit_path}")
    print(f"summary: {summary_path}")
    return summary


def run_repair_prior_smoke(
    *,
    config_path: Path,
    filter_config_path: Path,
    output_root: Path,
    device: str,
    repair_checkpoint_path: Path,
    repair_checkpoint_sha256: str,
    repair_run_manifest_path: Path,
    repair_run_manifest_sha256: str,
    repair_checkpoint_mode: str,
) -> dict[str, Any]:
    """Run the fixed 24-candidate repair Prior capability smoke."""

    from dataclasses import replace

    from skilldrive.data import tensorize_prior_context
    from skilldrive.data.av2_reader import (
        load_av2_history_scenario,
        load_av2_scenario,
    )
    from skilldrive.filtering.common import KinematicLimits, check_kinematics
    from skilldrive.generation import (
        GeneratedCandidate,
        GeneratedOverlay,
        canonical_sha256,
        latent_group_id,
        local_futures_to_global,
        paired_latent_seeds_for_task,
        pilot_evaluation_arm,
        prior_context_spec_for_task,
        seed_record_id,
        write_raw_shard,
        write_task_plan,
    )
    from skilldrive.generation.inference import (
        generate_prior_batch,
        load_repair_cvae,
    )
    from skilldrive.generation.repair_smoke import (
        build_repair_smoke_plan,
        summarize_repair_smoke_kinematics,
    )

    config = load_counterfactual_config(config_path)
    filter_config = load_filter_config(filter_config_path)
    input_hashes = {
        "generation_config": _file_sha256(config_path),
        "filter_config": _file_sha256(filter_config_path),
        "seed_manifest": _verified_hash(
            config.inputs.seed_manifest,
            config.inputs.seed_manifest_sha256,
            "formal seed manifest",
        ),
        "leakage_audit": _verified_hash(
            config.inputs.leakage_audit,
            config.inputs.leakage_audit_sha256,
            "frozen leakage audit",
        ),
    }
    _validate_frozen_leakage_audit(
        _read_json(config.inputs.leakage_audit, "frozen leakage audit")
    )
    records = read_seed_records(config.inputs.seed_manifest)
    selected = _select_smoke_records(config, records)

    schema = build_cvae_schema(config.formal_catalog.parent)
    runtime = load_repair_cvae(
        checkpoint_path=repair_checkpoint_path,
        run_manifest_path=repair_run_manifest_path,
        schema=schema,
        expected_checkpoint_sha256=repair_checkpoint_sha256,
        expected_run_manifest_sha256=repair_run_manifest_sha256,
        expected_schema_sha256=config.active_checkpoint.schema_sha256,
        device=device,
        checkpoint_mode=repair_checkpoint_mode,
    )
    input_hashes.update(
        checkpoint=runtime.checkpoint_sha256,
        run_manifest=runtime.run_manifest_sha256,
        schema=runtime.schema_sha256,
    )

    # The planner reads checkpoint identity from this immutable in-memory copy.
    # It is never written back to the formal active generation configuration.
    repair_generation_config = replace(
        config,
        active_checkpoint=replace(
            config.active_checkpoint,
            path=Path(repair_checkpoint_path),
            sha256=runtime.checkpoint_sha256,
            run_manifest=Path(repair_run_manifest_path),
            run_manifest_sha256=runtime.run_manifest_sha256,
            schema_sha256=runtime.schema_sha256,
        ),
    )
    execution_config = {
        "version": 1,
        "stage": "repair-prior-smoke",
        "device": device,
        "task_batch_size": 3,
        "candidate_budget": 8,
        "use_bfloat16": False,
        "latent_contract": "paired_standard_normal_epsilon_v1",
        "repair_checkpoint_mode": repair_checkpoint_mode,
        "repair_run_manifest_stage": runtime.manifest_stage,
        "repair_checkpoint_sha256": runtime.checkpoint_sha256,
        "repair_run_manifest_sha256": runtime.run_manifest_sha256,
        "filter_config_sha256": input_hashes["filter_config"],
    }
    plan = build_repair_smoke_plan(
        selected,
        repair_generation_config,
        execution_config=execution_config,
    )
    smoke_run_id = canonical_sha256(
        {
            "version": 1,
            "contract": "repair_prior_smoke_v1",
            "task_plan_id": plan.task_plan_id,
            "execution_config_sha256": plan.execution_config_sha256,
        }
    )
    smoke_root = output_root / "pilot" / "repair-prior-smoke-v1" / smoke_run_id
    plan_artifacts = write_task_plan(smoke_root, plan)

    records_by_id = {seed_record_id(record): record for record in selected}
    history_scenarios: dict[Path, Any] = {}
    task_records = []
    contexts = []
    latent_rows = []
    for task in plan.tasks:
        record = records_by_id.get(task.seed_record_id)
        if record is None:
            raise ValueError("repair smoke task references an unknown seed record")
        source_path = _validate_seed_source_path(
            config.inputs.data_root,
            record.source_path,
        )
        scenario = history_scenarios.get(source_path)
        if scenario is None:
            scenario = load_av2_history_scenario(source_path)
            if (
                len(scenario.timestamps) != 50
                or scenario.metadata.get("temporal_scope") != "history_only"
            ):
                raise ValueError(
                    "repair smoke Prior must receive history-only AV2 scenarios"
                )
            history_scenarios[source_path] = scenario
        context = tensorize_prior_context(
            scenario,
            prior_context_spec_for_task(task, record),
            schema,
        )
        if context.target_track_id != task.target_track_id:
            raise ValueError("repair smoke tensor target differs from the task target")
        if context.actor_track_ids[0] != task.target_track_id:
            raise ValueError("repair smoke target actor is not context slot zero")
        if hasattr(context, "target_future") or hasattr(
            context,
            "target_future_mask",
        ):
            raise ValueError("repair smoke Prior context contains future tensors")
        task_records.append(record)
        contexts.append(context)
        latent_rows.append(
            paired_latent_seeds_for_task(task, base_seed=plan.base_seed)
        )

    latent_seed_matrix = np.stack(latent_rows)
    generated = generate_prior_batch(
        runtime,
        contexts,
        latent_seed_matrix,
        use_bfloat16=False,
    )
    candidates: list[GeneratedCandidate] = []
    for batch_index, (task, record, context) in enumerate(
        zip(plan.tasks, task_records, contexts)
    ):
        global_futures = local_futures_to_global(
            generated.future_position_local[batch_index],
            context.anchor_origin_global,
            float(context.anchor_heading_global),
        )
        arm = pilot_evaluation_arm(
            task,
            none_skill_id=repair_generation_config.none_skill_id,
        )
        group_id = latent_group_id(task)
        primary_role = repair_generation_config.skills_by_id[
            task.skill_id
        ].primary_generated_role
        for candidate_index in range(task.candidate_budget):
            candidates.append(
                GeneratedCandidate(
                    task_id=task.task_id,
                    candidate_index=candidate_index,
                    latent_seed=int(
                        latent_seed_matrix[batch_index, candidate_index]
                    ),
                    scenario_id=task.scenario_id,
                    skill_id=task.skill_id,
                    proposal_mode=task.proposal_mode,
                    checkpoint_sha256=task.checkpoint_sha256,
                    semantic_config_sha256=task.semantic_config_sha256,
                    overlay=GeneratedOverlay(
                        target_track_id=task.target_track_id,
                        future_xy_global=global_futures[candidate_index],
                    ),
                    metadata={
                        "condition_skill_id": task.condition_skill_id,
                        "evaluation_arm": arm,
                        "latent_group_id": group_id,
                        "primary_generated_role": primary_role,
                        "requested_parameters": record.sampled_parameters,
                        "detection_mode": record.evidence["detection_mode"],
                    },
                )
            )
    raw_commit = write_raw_shard(
        smoke_root / "raw",
        0,
        candidates,
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )

    tasks_by_id = {task.task_id: task for task in plan.tasks}
    full_scenarios: dict[Path, Any] = {}
    evidence = []
    for candidate in candidates:
        task = tasks_by_id[candidate.task_id]
        record = records_by_id[task.seed_record_id]
        source_path = _validate_seed_source_path(
            config.inputs.data_root,
            record.source_path,
        )
        source_scenario = full_scenarios.get(source_path)
        if source_scenario is None:
            source_scenario = load_av2_scenario(source_path)
            full_scenarios[source_path] = source_scenario
        target = next(
            (
                agent
                for agent in source_scenario.agents
                if agent.track_id == task.target_track_id
            ),
            None,
        )
        if target is None:
            raise ValueError("repair smoke target is missing from the full scenario")
        policy = filter_config.kinematics_by_type.get(target.object_type.lower())
        if policy is None:
            passed = False
            rejection_reasons = ["kinematics.class_unsupported"]
            metrics = {"target_object_type": target.object_type.lower()}
        else:
            check = check_kinematics(
                source_scenario,
                task.target_track_id,
                candidate.overlay.future_xy_global,
                KinematicLimits(
                    maximum_seam_speed_mps=policy.maximum_seam_speed_mps,
                    maximum_speed_mps=policy.maximum_speed_mps,
                    maximum_acceleration_mps2=policy.maximum_acceleration_mps2,
                    maximum_deceleration_mps2=policy.maximum_deceleration_mps2,
                    maximum_jerk_mps3=policy.maximum_jerk_mps3,
                    maximum_curvature_per_m=policy.maximum_curvature_per_m,
                    maximum_heading_rate_rad_s=policy.maximum_heading_rate_rad_s,
                    minimum_heading_speed_mps=policy.minimum_heading_speed_mps,
                ),
            )
            passed = check.passed
            rejection_reasons = list(check.rejection_values)
            metrics = dict(check.metrics)
        evidence.append(
            {
                "candidate_id": candidate.candidate_id,
                "task_id": candidate.task_id,
                "scenario_id": candidate.scenario_id,
                "skill_id": candidate.skill_id,
                "evaluation_arm": candidate.metadata["evaluation_arm"],
                "candidate_index": candidate.candidate_index,
                "latent_seed": candidate.latent_seed,
                "latent_group_id": candidate.metadata["latent_group_id"],
                "kinematics": {
                    "passed": passed,
                    "rejection_reasons": rejection_reasons,
                    "metrics": metrics,
                },
            }
        )

    kinematic_summary = summarize_repair_smoke_kinematics(evidence)
    evidence_path = write_generation_capability_matrix(
        smoke_root / "kinematic_evidence.json",
        {
            "version": 1,
            "kind": "repair_prior_smoke_kinematic_evidence",
            "candidate_count": len(evidence),
            "rows": evidence,
        },
    )
    diagnostic_only = repair_checkpoint_mode == "diagnostic-overfit"
    summary = {
        "version": 1,
        "status": "completed",
        "stage": "repair-prior-smoke",
        "ability_gate_status": (
            "diagnostic_only"
            if diagnostic_only
            else "smoke_completed_pending_promotion"
        ),
        "smoke_run_id": smoke_run_id,
        "task_count": len(plan.tasks),
        "candidate_count": len(candidates),
        "candidate_ids_sha256": raw_commit.candidate_ids_sha256,
        "execution_config": execution_config,
        "checkpoint_use": {
            "requested_mode": repair_checkpoint_mode,
            "run_manifest_stage": runtime.manifest_stage,
            "repair_contract": "cvae_generation_repair_v1",
            "diagnostic_only": diagnostic_only,
            "formal_active": False,
            "promotion_status": "not_promoted_by_smoke",
        },
        "input_hashes": input_hashes,
        "history_only_prior_inputs": True,
        "full_scenarios_used_only_after_generation_for_kinematics": True,
        "kinematics": kinematic_summary,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "outputs": {
            "task_plan": plan_artifacts.task_plan_path.as_posix(),
            "raw_commit": raw_commit.commit_path.as_posix(),
            "kinematic_evidence": evidence_path.as_posix(),
        },
        "output_sha256": {
            "task_plan": plan_artifacts.task_plan_sha256,
            "raw_commit": _file_sha256(raw_commit.commit_path),
            "kinematic_evidence": _file_sha256(evidence_path),
        },
    }
    summary_path = write_generation_capability_matrix(
        smoke_root / "summary.json",
        summary,
    )
    print(
        "repair Prior smoke complete: "
        f"{len(plan.tasks)} tasks, {len(candidates)} candidates, "
        f"{sum(row['kinematics']['passed'] for row in evidence)} kinematic passes",
        flush=True,
    )
    print(f"repair smoke summary: {summary_path}", flush=True)
    return summary


def run_filter_smoke(
    *,
    config_path: Path,
    filter_config_path: Path,
    detection_config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Re-filter the committed 16-candidate smoke raw without model inference."""

    from collections import Counter

    from skilldrive.data.av2_reader import load_av2_scenario
    from skilldrive.filtering.context import bind_raw_candidates
    from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
    from skilldrive.filtering.pipeline import (
        FILTER_CONTRACT_VERSION,
        CandidateFilterInput,
        validate_candidates,
    )
    from skilldrive.generation import (
        TaskPlan,
        build_generation_task,
        load_raw_shard_candidates,
        load_task_plan,
        recover_durable_tasks,
        scan_raw_shards,
        semantic_generation_config_sha256,
        write_filter_indexes,
        write_task_plan,
    )
    from skilldrive.skills.detection import load_detection_config
    from skilldrive.skills.loader import load_skill

    run_audit(
        config_path=config_path,
        filter_config_path=filter_config_path,
        output_root=output_root,
    )
    config = load_counterfactual_config(config_path)
    filter_config = load_filter_config(filter_config_path)
    detection_config = load_detection_config(detection_config_path)
    generation_semantic = semantic_generation_config_sha256(config)
    smoke_root = output_root / "pilot" / "smoke"
    raw_dir = smoke_root / "raw"
    raw_scan = scan_raw_shards(
        raw_dir,
        expected_semantic_config_sha256=generation_semantic,
    )
    if raw_scan.invalid_shards or raw_scan.orphaned_files:
        raise ValueError(
            "filter-smoke requires a clean committed raw set: "
            f"invalid={len(raw_scan.invalid_shards)}, "
            f"orphaned={len(raw_scan.orphaned_files)}"
        )
    if raw_scan.candidate_count != 16:
        raise ValueError(
            f"filter-smoke requires exactly 16 raw candidates, got {raw_scan.candidate_count}"
        )
    execution_hashes = {
        shard.execution_config_sha256 for shard in raw_scan.valid_shards
    }
    if len(execution_hashes) != 1:
        raise ValueError("filter-smoke raw shards have inconsistent execution identities")
    execution_sha256 = next(iter(execution_hashes))

    records = read_seed_records(config.inputs.seed_manifest)
    selected_records = _select_smoke_records(config, records)
    if not (smoke_root / "task_plan.jsonl").is_file():
        tasks = tuple(
            build_generation_task(
                task_index=index,
                record=record,
                config=config,
                candidate_budget=8,
            )
            for index, record in enumerate(selected_records)
        )
        write_task_plan(
            smoke_root,
            TaskPlan(
                semantic_config_sha256=generation_semantic,
                execution_config_sha256=execution_sha256,
                base_seed=config.sampling.base_seed,
                per_skill=1,
                candidate_budget=8,
                tasks=tasks,
            ),
        )
    loaded_plan = load_task_plan(
        smoke_root,
        expected_semantic_config_sha256=generation_semantic,
        current_execution_config_sha256=execution_sha256,
    )
    recovery = recover_durable_tasks(loaded_plan.plan, raw_dir)
    recovery_errors = {
        "invalid_shards": len(recovery.raw_scan.invalid_shards),
        "orphaned_files": len(recovery.raw_scan.orphaned_files),
        "partial_tasks": len(recovery.partial_task_ids),
        "pending_tasks": len(recovery.pending_task_ids),
        "extra_candidate_tasks": sum(
            bool(indices) for indices in recovery.extra_candidate_indices.values()
        ),
    }
    expected_task_ids = {task.task_id for task in loaded_plan.plan.tasks}
    if any(recovery_errors.values()) or recovery.durable_task_ids != expected_task_ids:
        raise ValueError(
            "filter-smoke requires a complete raw candidate budget: "
            f"{recovery_errors}"
        )
    raw_scan = recovery.raw_scan
    raw_candidates = tuple(
        candidate
        for shard in raw_scan.valid_shards
        for candidate in load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=generation_semantic,
        )
    )
    bound = bind_raw_candidates(
        raw_candidates,
        loaded_plan.plan.tasks,
        selected_records,
    )

    raw_snapshot = {
        path.resolve(): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for shard in raw_scan.valid_shards
        for path in (shard.arrays_path, shard.metadata_path, shard.commit_path)
    }
    skill_directory = config.formal_catalog.parent
    skills = {
        skill_id: load_skill(skill_directory / f"{skill_id}.yaml")
        for skill_id in {item.task.skill_id for item in bound}
    }
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=Path.cwd(),
        generation_config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    source_cache = {}
    candidate_inputs = []
    for item in bound:
        source = source_cache.get(item.task.scenario_id)
        if source is None:
            source = load_av2_scenario(
                _validate_seed_source_path(
                    config.inputs.data_root,
                    item.seed_record.source_path,
                )
            )
            source_cache[item.task.scenario_id] = source
        candidate_inputs.append(
            CandidateFilterInput(
                bound=item,
                skill=skills[item.task.skill_id],
                source_scenario=source,
                primary_generated_role=config.skills_by_id[
                    item.task.skill_id
                ].primary_generated_role,
            )
        )
    batch = validate_candidates(
        candidate_inputs,
        filter_config=filter_config,
        detection_config=detection_config,
        filter_semantic_sha256=fingerprint.semantic_sha256,
    )
    decisions = batch.decisions
    quality_passed = sum(item.quality_passed for item in batch.validations)
    filter_root = smoke_root / "filter-evaluations" / fingerprint.semantic_sha256
    index = write_filter_indexes(
        filter_root,
        raw_scan.valid_shards,
        decisions,
        filter_config_sha256=fingerprint.semantic_sha256,
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )

    raw_snapshot_after = {
        path: {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in raw_snapshot
        if path.is_file()
    }
    raw_unchanged = raw_snapshot_after == raw_snapshot
    if not raw_unchanged:
        raise RuntimeError("filter-smoke modified committed raw files")
    primary_rejections = Counter(
        decision.rejection_reasons[0]
        for decision in decisions
        if not decision.accepted
    )
    first_failed_stages = Counter(
        decision.metrics["first_failed_stage"]
        for decision in decisions
        if not decision.accepted
    )
    summary = {
        "version": 1,
        "status": "passed",
        "stage": "filter-smoke",
        "candidate_count": len(decisions),
        "quality_passed_before_diversity": quality_passed,
        "accepted_count": index.accepted_count,
        "rejected_count": index.rejected_count,
        "primary_rejections": dict(sorted(primary_rejections.items())),
        "first_failed_stages": dict(sorted(first_failed_stages.items())),
        "filter_semantic_sha256": fingerprint.semantic_sha256,
        "filter_dependency_sha256": dict(fingerprint.file_sha256),
        "filter_contract_version": FILTER_CONTRACT_VERSION,
        "stage_execution_counts": dict(batch.stage_execution_counts),
        "stage_elapsed_seconds": dict(batch.stage_elapsed_seconds),
        "task_plan_id": loaded_plan.plan.task_plan_id,
        "task_plan_sha256": _file_sha256(smoke_root / "task_plan.jsonl"),
        "raw_immutable_verified": raw_unchanged,
        "raw_snapshot": {
            _path_label(path): identity
            for path, identity in sorted(
                raw_snapshot.items(), key=lambda item: item[0].as_posix()
            )
        },
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "outputs": {
            "accepted": index.accepted_path.as_posix(),
            "rejected": index.rejected_path.as_posix(),
            "commit": index.commit_path.as_posix(),
        },
    }
    summary_path = write_generation_capability_matrix(
        filter_root / "summary.json",
        summary,
    )
    print(
        "stage C filter smoke passed: "
        f"{index.accepted_count} accepted, {index.rejected_count} rejected"
    )
    print(f"filter summary: {summary_path}")
    return summary


def run_pilot(
    *,
    config_path: Path,
    filter_config_path: Path,
    detection_config_path: Path,
    output_root: Path,
    device: str,
    task_batch_size: int,
    progress_interval_seconds: float,
) -> dict[str, Any]:
    """Run the resumable paired 34-skill numeric Pilot and full filtering."""

    import math
    import time
    from collections import Counter
    from statistics import median

    from skilldrive.data import tensorize_prior_context
    from skilldrive.data.av2_reader import (
        load_av2_history_scenario,
        load_av2_scenario,
    )
    from skilldrive.filtering.context import bind_raw_candidates
    from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
    from skilldrive.filtering.pipeline import (
        FILTER_CONTRACT_VERSION,
        CandidateFilterInput,
        finalize_candidate_validations,
        validate_candidate,
    )
    from skilldrive.generation import (
        GeneratedCandidate,
        GeneratedOverlay,
        build_generation_task,
        build_paired_pilot_task_plan,
        canonical_sha256,
        latent_group_id,
        load_raw_shard_candidates,
        load_task_plan,
        local_futures_to_global,
        paired_latent_seeds_for_task,
        pilot_evaluation_arm,
        prior_context_fingerprint,
        prior_context_spec_for_task,
        recover_paired_pilot_tasks,
        seed_record_id,
        select_eligible_pilot_records,
        write_filter_indexes,
        write_raw_shard,
        write_task_plan,
    )
    from skilldrive.generation.inference import generate_prior_batch, load_configured_cvae
    from skilldrive.skills.detection import load_detection_config
    from skilldrive.skills.loader import load_skill

    pilot_started = time.perf_counter()
    if (
        isinstance(task_batch_size, bool)
        or not isinstance(task_batch_size, int)
        or task_batch_size <= 0
    ):
        raise ValueError("task_batch_size must be a positive integer")
    if (
        isinstance(progress_interval_seconds, bool)
        or not isinstance(progress_interval_seconds, (int, float))
        or not math.isfinite(float(progress_interval_seconds))
        or float(progress_interval_seconds) <= 0.0
    ):
        raise ValueError("progress_interval_seconds must be a positive finite number")

    run_audit(
        config_path=config_path,
        filter_config_path=filter_config_path,
        output_root=output_root,
    )
    config = load_counterfactual_config(config_path)
    filter_config = load_filter_config(filter_config_path)
    detection_config = load_detection_config(detection_config_path)
    records = read_seed_records(config.inputs.seed_manifest)

    schema = build_cvae_schema(config.formal_catalog.parent)
    history_scenarios: dict[Path, Any] = {}
    context_cache: dict[str, Any] = {}
    eligibility_started = time.perf_counter()
    last_eligibility_progress = eligibility_started
    eligibility_contexts_checked = 0
    eligibility_failures = 0
    records_by_id = {}
    formal_tasks_by_record = {}

    def formal_task_for_record(record):
        record_id = seed_record_id(record)
        existing_record = records_by_id.setdefault(record_id, record)
        if existing_record != record:
            raise ValueError("formal seed records have a SHA-256 identity collision")
        task = formal_tasks_by_record.get(record_id)
        if task is None:
            task = build_generation_task(
            task_index=0,
            record=record,
            config=config,
            candidate_budget=config.sampling.pilot_candidates_per_task,
        )
            formal_tasks_by_record[record_id] = task
        return task

    def tensorize_task_context(task, record):
        context_fingerprint = prior_context_fingerprint(task, record)
        cached = context_cache.get(context_fingerprint)
        if cached is not None:
            return cached
        source_path = _validate_seed_source_path(
            config.inputs.data_root,
            record.source_path,
        )
        scenario = history_scenarios.get(source_path)
        if scenario is None:
            scenario = load_av2_history_scenario(source_path)
            if (
                len(scenario.timestamps) != 50
                or scenario.metadata.get("temporal_scope") != "history_only"
            ):
                raise ValueError("Pilot generation must receive history-only scenes")
            history_scenarios[source_path] = scenario
        context = tensorize_prior_context(
            scenario,
            prior_context_spec_for_task(task, record),
            schema,
        )
        primary_role = config.skills_by_id[task.skill_id].primary_generated_role
        if record.role_track_ids.get(primary_role) != task.target_track_id:
            raise ValueError("Pilot primary role differs from the task target")
        if context.target_track_id != task.target_track_id:
            raise ValueError("Pilot tensor target differs from the task target")
        if hasattr(context, "target_future") or hasattr(
            context,
            "target_future_mask",
        ):
            raise ValueError("Pilot Prior context must not expose future tensors")
        context_cache[context_fingerprint] = context
        return context

    def validate_eligibility_record(record):
        nonlocal eligibility_contexts_checked
        nonlocal eligibility_failures
        nonlocal last_eligibility_progress
        try:
            return tensorize_task_context(
                formal_task_for_record(record),
                record,
            )
        except ValueError:
            eligibility_failures += 1
            raise
        finally:
            eligibility_contexts_checked += 1
            now = time.perf_counter()
            if now - last_eligibility_progress >= float(progress_interval_seconds):
                print(
                    "pilot eligibility: "
                    f"{eligibility_contexts_checked} unique contexts checked, "
                    f"{len(history_scenarios)} scenarios loaded, "
                    f"{eligibility_failures} ineligible",
                    flush=True,
                )
                last_eligibility_progress = now

    print(
        "pilot eligibility: validating deterministic Formal Train seed candidates",
        flush=True,
    )

    eligibility = select_eligible_pilot_records(
        records,
        formal_skill_ids=config.formal_skill_ids,
        per_skill=config.sampling.pilot_seed_records_per_skill,
        base_seed=config.sampling.base_seed,
        context_fingerprint=lambda record: prior_context_fingerprint(
            formal_task_for_record(record),
            record,
        ),
        validate_record=validate_eligibility_record,
    )
    print(
        "pilot eligibility complete: "
        f"{eligibility_contexts_checked} unique contexts checked, "
        f"{len(eligibility.records)} records selected, "
        f"{eligibility_failures} ineligible",
        flush=True,
    )
    selected_records = eligibility.records
    selected_record_ids = {seed_record_id(record) for record in selected_records}
    decision_by_record_id = {
        decision.seed_record_id: decision for decision in eligibility.decisions
    }
    input_by_skill = Counter(record.skill_id for record in records)
    selected_by_skill = Counter(record.skill_id for record in selected_records)
    attempted_by_skill = Counter(decision.skill_id for decision in eligibility.decisions)
    excluded_by_skill = Counter(
        decision.skill_id
        for decision in eligibility.decisions
        if not decision.eligible
    )
    failure_reasons_by_skill: dict[str, Counter[tuple[str, str]]] = {}
    for decision in eligibility.decisions:
        if decision.eligible:
            continue
        failure_reasons_by_skill.setdefault(decision.skill_id, Counter())[
            (
                decision.failure_type or "unknown",
                decision.failure_message or "unknown",
            )
        ] += 1
    skills_without_eligible_seed_records = [
        skill_id
        for skill_id in config.formal_skill_ids
        if not selected_by_skill[skill_id]
    ]
    eligibility_audit = {
        "version": 1,
        "kind": "pilot_seed_eligibility_audit",
        "status": (
            "completed"
            if not skills_without_eligible_seed_records
            else "completed_with_ineligible_skills"
        ),
        "qualification_contract": (
            "formal_train_history_only_load_plus_tensorize_prior_context_v1"
        ),
        "seed_manifest_sha256": config.inputs.seed_manifest_sha256,
        "schema_sha256": config.active_checkpoint.schema_sha256,
        "base_seed": config.sampling.base_seed,
        "per_skill_cap": config.sampling.pilot_seed_records_per_skill,
        "formal_skill_ids": list(config.formal_skill_ids),
        "formal_train_only": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "selection_stable": True,
        "skills_without_eligible_seed_records": skills_without_eligible_seed_records,
        "counts": {
            "input_records": len(records),
            "attempted_records": len(eligibility.decisions),
            "unique_contexts_evaluated": len(
                {
                    decision.context_fingerprint
                    for decision in eligibility.decisions
                }
            ),
            "context_cache_hits": sum(
                decision.cache_hit for decision in eligibility.decisions
            ),
            "selected_records": len(selected_records),
            "excluded_records": sum(
                not decision.eligible for decision in eligibility.decisions
            ),
            "formal_skills_covered": len(selected_by_skill),
            "formal_skills_without_eligible_seed_records": len(
                skills_without_eligible_seed_records
            ),
        },
        "by_skill": [
            {
                "skill_id": skill_id,
                "available_records": input_by_skill[skill_id],
                "attempted_records": attempted_by_skill[skill_id],
                "selected_records": selected_by_skill[skill_id],
                "excluded_records": excluded_by_skill[skill_id],
                "per_skill_cap": config.sampling.pilot_seed_records_per_skill,
                "status": (
                    "selected"
                    if selected_by_skill[skill_id]
                    else "no_eligible_seed_records"
                ),
                "failure_reasons": [
                    {
                        "failure_type": failure_type,
                        "failure_message": failure_message,
                        "record_count": count,
                    }
                    for (failure_type, failure_message), count in sorted(
                        failure_reasons_by_skill.get(skill_id, {}).items()
                    )
                ],
            }
            for skill_id in config.formal_skill_ids
        ],
        "selected": [
            {
                "skill_id": record.skill_id,
                "scenario_id": record.scenario_id,
                "seed_record_id": record_id,
                "target_track_id": formal_tasks_by_record[
                    record_id
                ].target_track_id,
                "source_path": record.source_path,
                "context_fingerprint": decision_by_record_id[
                    record_id
                ].context_fingerprint,
                "candidate_rank": decision_by_record_id[record_id].candidate_rank,
                "context_cache_hit": decision_by_record_id[record_id].cache_hit,
            }
            for record in selected_records
            for record_id in (seed_record_id(record),)
        ],
        "excluded": [
            {
                "skill_id": decision.skill_id,
                "scenario_id": decision.scenario_id,
                "seed_record_id": decision.seed_record_id,
                "target_track_id": formal_tasks_by_record[
                    decision.seed_record_id
                ].target_track_id,
                "source_path": records_by_id[decision.seed_record_id].source_path,
                "context_fingerprint": decision.context_fingerprint,
                "candidate_rank": decision.candidate_rank,
                "context_cache_hit": decision.cache_hit,
                "failure_type": decision.failure_type,
                "failure_message": decision.failure_message,
            }
            for decision in eligibility.decisions
            if not decision.eligible
        ],
    }
    eligibility_sha256 = canonical_sha256(eligibility_audit)

    generation_dependencies = {
        *Path("skilldrive/generation").glob("*.py"),
        Path("skilldrive/data/av2_reader.py"),
        Path("skilldrive/data/coordinates.py"),
        Path("skilldrive/data/cvae_samples.py"),
        Path("skilldrive/models/conditional_cvae.py"),
        Path("scripts/generation/run_counterfactual_pipeline.py"),
    }
    execution_config = {
        "version": 2,
        "device": device,
        "task_batch_size": task_batch_size,
        "candidate_budget": config.sampling.pilot_candidates_per_task,
        "use_bfloat16": False,
        "raw_shard_policy": "one_task_per_shard",
        "latent_contract": "paired_standard_normal_epsilon_v1",
        "pilot_seed_eligibility_sha256": eligibility_sha256,
        "source_sha256": {
            path.as_posix(): _file_sha256(path)
            for path in sorted(generation_dependencies)
        },
    }
    plan = build_paired_pilot_task_plan(
        selected_records,
        config,
        execution_config=execution_config,
        allow_missing_skills=True,
    )
    if {task.seed_record_id for task in plan.tasks} != selected_record_ids:
        raise RuntimeError("Pilot plan differs from the stable eligible seed selection")
    pilot_run_id = canonical_sha256(
        {
            "version": 1,
            "task_plan_id": plan.task_plan_id,
            "execution_config_sha256": plan.execution_config_sha256,
        }
    )
    pilot_root = output_root / "pilot" / "skill-pilot-v1" / pilot_run_id
    eligibility_audit_path = pilot_root / "eligibility_audit.json"
    if eligibility_audit_path.exists():
        stored_eligibility = _read_json(
            eligibility_audit_path,
            "Pilot eligibility audit",
        )
        if stored_eligibility != eligibility_audit:
            raise ValueError(
                "Pilot eligibility audit changed; use a new Pilot contract"
            )
    else:
        write_generation_capability_matrix(
            eligibility_audit_path,
            eligibility_audit,
        )
    if canonical_sha256(
        _read_json(eligibility_audit_path, "Pilot eligibility audit")
    ) != eligibility_sha256:
        raise ValueError("Pilot eligibility audit SHA-256 mismatch")
    plan_path = pilot_root / "task_plan.jsonl"
    plan_summary_path = pilot_root / "task_plan.summary.json"
    if plan_path.exists() or plan_summary_path.exists():
        if not plan_path.is_file() or not plan_summary_path.is_file():
            raise ValueError("Pilot task plan is partially present")
        loaded = load_task_plan(
            pilot_root,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
            current_execution_config_sha256=plan.execution_config_sha256,
        )
        if loaded.execution_config_changed or loaded.plan != plan:
            raise ValueError(
                "Pilot task plan or execution config changed; use a new Pilot contract"
            )
        plan = loaded.plan
    else:
        write_task_plan(pilot_root, plan)

    arms_by_task = {
        task.task_id: pilot_evaluation_arm(
            task,
            none_skill_id=config.none_skill_id,
        )
        for task in plan.tasks
    }
    raw_dir = pilot_root / "raw"
    recovery = recover_paired_pilot_tasks(plan, raw_dir)
    tasks_to_generate = [
        task for task in plan.tasks if task.task_id in recovery.rebuild_task_ids
    ]
    print(
        "pilot resume: "
        f"{len(recovery.durable_task_ids)}/{len(plan.tasks)} task shards durable, "
        f"{len(tasks_to_generate)} to generate",
        flush=True,
    )

    generation_started = time.perf_counter()
    newly_generated_candidates = 0
    if tasks_to_generate:
        runtime = load_configured_cvae(
            active_checkpoint=config.active_checkpoint,
            schema=schema,
            device=device,
        )
        last_progress = generation_started
        for offset in range(0, len(tasks_to_generate), task_batch_size):
            batch_tasks = tasks_to_generate[offset : offset + task_batch_size]
            contexts = []
            latent_rows = []
            batch_records = []
            for task in batch_tasks:
                record = records_by_id.get(task.seed_record_id)
                if record is None:
                    raise ValueError(
                        f"Pilot task references an unknown seed row: {task.seed_record_id}"
                    )
                context = tensorize_task_context(task, record)
                contexts.append(context)
                latent_rows.append(
                    paired_latent_seeds_for_task(
                        task,
                        base_seed=plan.base_seed,
                    )
                )
                batch_records.append(record)

            latent_seed_matrix = np.stack(latent_rows)
            generated = generate_prior_batch(
                runtime,
                contexts,
                latent_seed_matrix,
                use_bfloat16=False,
            )
            for batch_index, (task, record, context) in enumerate(
                zip(batch_tasks, batch_records, contexts)
            ):
                global_futures = local_futures_to_global(
                    generated.future_position_local[batch_index],
                    context.anchor_origin_global,
                    float(context.anchor_heading_global),
                )
                arm = arms_by_task[task.task_id]
                primary_role = config.skills_by_id[
                    task.skill_id
                ].primary_generated_role
                candidates = [
                    GeneratedCandidate(
                        task_id=task.task_id,
                        candidate_index=candidate_index,
                        latent_seed=int(
                            latent_seed_matrix[batch_index, candidate_index]
                        ),
                        scenario_id=task.scenario_id,
                        skill_id=task.skill_id,
                        proposal_mode=task.proposal_mode,
                        checkpoint_sha256=task.checkpoint_sha256,
                        semantic_config_sha256=task.semantic_config_sha256,
                        overlay=GeneratedOverlay(
                            target_track_id=task.target_track_id,
                            future_xy_global=global_futures[candidate_index],
                        ),
                        metadata={
                            "condition_skill_id": task.condition_skill_id,
                            "evaluation_arm": arm,
                            "latent_group_id": latent_group_id(task),
                            "primary_generated_role": primary_role,
                            "requested_parameters": record.sampled_parameters,
                            "detection_mode": record.evidence["detection_mode"],
                        },
                    )
                    for candidate_index in range(task.candidate_budget)
                ]
                write_raw_shard(
                    raw_dir,
                    task.task_index,
                    candidates,
                    semantic_config_sha256=plan.semantic_config_sha256,
                    execution_config_sha256=plan.execution_config_sha256,
                )
                newly_generated_candidates += len(candidates)

            now = time.perf_counter()
            completed = min(offset + len(batch_tasks), len(tasks_to_generate))
            if (
                now - last_progress >= float(progress_interval_seconds)
                or completed == len(tasks_to_generate)
            ):
                elapsed = max(now - generation_started, 1e-9)
                rate = newly_generated_candidates / elapsed
                remaining = (
                    len(tasks_to_generate) * plan.candidate_budget
                    - newly_generated_candidates
                )
                eta = None if rate <= 0.0 else remaining / rate
                print(
                    "pilot generation: "
                    f"{completed}/{len(tasks_to_generate)} tasks, "
                    f"{newly_generated_candidates}/"
                    f"{len(tasks_to_generate) * plan.candidate_budget} candidates, "
                    f"{rate:.1f} candidates/s, ETA "
                    f"{('--:--' if eta is None else f'{int(eta // 60):02d}:{int(eta % 60):02d}')}",
                    flush=True,
                )
                last_progress = now
    generation_elapsed = time.perf_counter() - generation_started
    history_scenarios.clear()
    context_cache.clear()

    recovery = recover_paired_pilot_tasks(plan, raw_dir)
    if recovery.rebuild_task_ids or recovery.durable_task_ids != {
        task.task_id for task in plan.tasks
    }:
        raise RuntimeError("Pilot generation ended without a complete durable task set")
    raw_commits = tuple(
        sorted(recovery.raw_scan.valid_shards, key=lambda item: item.shard_index)
    )
    if (
        len(raw_commits) != len(plan.tasks)
        or sum(item.candidate_count for item in raw_commits) != plan.total_candidates
    ):
        raise RuntimeError("Pilot raw shard totals differ from the frozen task plan")

    raw_snapshot = {
        path.resolve(): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for shard in raw_commits
        for path in (shard.arrays_path, shard.metadata_path, shard.commit_path)
    }
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=Path.cwd(),
        generation_config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    skill_directory = config.formal_catalog.parent
    skills = {
        skill_id: load_skill(skill_directory / f"{skill_id}.yaml")
        for skill_id in config.formal_skill_ids
    }
    tasks_by_index = {task.task_index: task for task in plan.tasks}
    compact_results = []
    filtering_started = time.perf_counter()
    last_progress = filtering_started
    filtered_count = 0
    current_scenario_id = None
    current_source = None
    for shard in raw_commits:
        task = tasks_by_index[shard.shard_index]
        record = records_by_id[task.seed_record_id]
        if task.scenario_id != current_scenario_id:
            current_source = load_av2_scenario(
                _validate_seed_source_path(
                    config.inputs.data_root,
                    record.source_path,
                )
            )
            current_scenario_id = task.scenario_id
        raw_candidates = load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
        )
        bound = bind_raw_candidates(raw_candidates, [task], [record])
        arm = arms_by_task[task.task_id]
        cohort = "learned_none_control" if arm == "learned_none_control" else "formal"
        primary_role = config.skills_by_id[task.skill_id].primary_generated_role
        for item in bound:
            validation = validate_candidate(
                CandidateFilterInput(
                    bound=item,
                    skill=skills[task.skill_id],
                    source_scenario=current_source,
                    primary_generated_role=primary_role,
                ),
                filter_config=filter_config,
                detection_config=detection_config,
            )
            compact_results.append(validation.compact(cohort=cohort))
            filtered_count += 1
        now = time.perf_counter()
        if (
            now - last_progress >= float(progress_interval_seconds)
            or filtered_count == plan.total_candidates
        ):
            elapsed = max(now - filtering_started, 1e-9)
            rate = filtered_count / elapsed
            remaining = plan.total_candidates - filtered_count
            eta = None if rate <= 0.0 else remaining / rate
            quality_passed = sum(item.quality_passed for item in compact_results)
            print(
                "pilot filtering: "
                f"{filtered_count}/{plan.total_candidates} candidates, "
                f"{quality_passed} quality survivors, {rate:.1f} candidates/s, "
                f"ETA {('--:--' if eta is None else f'{int(eta // 60):02d}:{int(eta % 60):02d}')}",
                flush=True,
            )
            last_progress = now

    batch = finalize_candidate_validations(
        compact_results,
        filter_config=filter_config,
        filter_semantic_sha256=fingerprint.semantic_sha256,
    )
    filtering_elapsed = time.perf_counter() - filtering_started
    filter_root = pilot_root / "filter-evaluations" / fingerprint.semantic_sha256
    index = write_filter_indexes(
        filter_root,
        raw_commits,
        batch.decisions,
        filter_config_sha256=fingerprint.semantic_sha256,
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )

    raw_snapshot_after = {
        path: {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in raw_snapshot
        if path.is_file()
    }
    if raw_snapshot_after != raw_snapshot:
        raise RuntimeError("Pilot filtering modified committed raw files")

    task_by_id = {task.task_id: task for task in plan.tasks}
    compact_by_id = {item.identity.candidate_id: item for item in batch.validations}
    per_skill_arm: dict[tuple[str, str], dict[str, Any]] = {}
    for task in plan.tasks:
        key = (task.skill_id, arms_by_task[task.task_id])
        entry = per_skill_arm.setdefault(
            key,
            {
                "skill_id": task.skill_id,
                "evaluation_arm": arms_by_task[task.task_id],
                "task_count": 0,
                "candidate_count": 0,
                "quality_passed": 0,
                "accepted": 0,
                "skill_trigger_passed": 0,
                "first_failed_stages": Counter(),
                "primary_rejections": Counter(),
                "risk_values": [],
                "parameter_absolute_errors": {},
            },
        )
        entry["task_count"] += 1
    for decision in batch.decisions:
        task = task_by_id[decision.metrics["task_id"]]
        key = (task.skill_id, arms_by_task[task.task_id])
        entry = per_skill_arm[key]
        entry["candidate_count"] += 1
        compact = compact_by_id[decision.candidate_id]
        if compact.quality_passed:
            entry["quality_passed"] += 1
        for timed in compact.checks:
            if timed.check.stage.value == "target_risk":
                evaluation = timed.check.metrics.get("evaluation", {})
                value = evaluation.get("value")
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    entry["risk_values"].append(float(value))
            elif timed.check.stage.value == "skill_trigger" and timed.check.passed:
                entry["skill_trigger_passed"] += 1
            elif timed.check.stage.value == "parameter_realization":
                parameters = timed.check.metrics.get("parameters", {})
                if isinstance(parameters, Mapping):
                    for parameter_name, parameter in parameters.items():
                        if not isinstance(parameter_name, str) or not isinstance(
                            parameter, Mapping
                        ):
                            continue
                        error = parameter.get("absolute_error")
                        if (
                            isinstance(error, (int, float))
                            and not isinstance(error, bool)
                            and math.isfinite(float(error))
                        ):
                            entry["parameter_absolute_errors"].setdefault(
                                parameter_name,
                                [],
                            ).append(float(error))
        if decision.accepted:
            entry["accepted"] += 1
        else:
            entry["first_failed_stages"][decision.metrics["first_failed_stage"]] += 1
            entry["primary_rejections"][decision.rejection_reasons[0]] += 1

    skill_rows = []
    for key in sorted(per_skill_arm):
        entry = per_skill_arm[key]
        candidates = entry["candidate_count"]
        risk_values = entry["risk_values"]
        parameter_errors_by_name = entry["parameter_absolute_errors"]
        skill_rows.append(
            {
                "skill_id": entry["skill_id"],
                "evaluation_arm": entry["evaluation_arm"],
                "task_count": entry["task_count"],
                "candidate_count": candidates,
                "quality_passed": entry["quality_passed"],
                "quality_pass_rate": (
                    0.0 if not candidates else entry["quality_passed"] / candidates
                ),
                "accepted": entry["accepted"],
                "accept_rate": 0.0 if not candidates else entry["accepted"] / candidates,
                "skill_trigger_passed": entry["skill_trigger_passed"],
                "skill_trigger_pass_rate": (
                    0.0 if not candidates else entry["skill_trigger_passed"] / candidates
                ),
                "risk_values": {
                    "count": len(risk_values),
                    "minimum": None if not risk_values else min(risk_values),
                    "median": None if not risk_values else median(risk_values),
                    "maximum": None if not risk_values else max(risk_values),
                },
                "parameter_absolute_errors": {
                    parameter_name: {
                        "count": len(parameter_errors),
                        "median": median(parameter_errors),
                        "maximum": max(parameter_errors),
                    }
                    for parameter_name, parameter_errors in sorted(
                        parameter_errors_by_name.items()
                    )
                },
                "first_failed_stages": dict(
                    sorted(entry["first_failed_stages"].items())
                ),
                "primary_rejections": dict(
                    sorted(entry["primary_rejections"].items())
                ),
            }
        )

    paired: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in batch.validations:
        task = task_by_id[item.identity.task_id]
        arm = arms_by_task[task.task_id]
        if arm not in {"learned_conditioned", "learned_none_control"}:
            continue
        pair = paired.setdefault(
            (
                task.skill_id,
                item.identity.seed_record_id,
                item.identity.candidate_index,
            ),
            {},
        )
        pair[arm] = item
    paired_by_skill: dict[str, dict[str, Any]] = {}
    decision_by_id = {item.candidate_id: item for item in batch.decisions}
    for (skill_id, _, _), pair in paired.items():
        conditioned = pair.get("learned_conditioned")
        control = pair.get("learned_none_control")
        if conditioned is None or control is None:
            raise RuntimeError("learned Pilot comparison pair is incomplete")
        skill_pair = paired_by_skill.setdefault(
            skill_id,
            {
                "skill_id": skill_id,
                "shared_latent_pairs": 0,
                "both_quality_pass": 0,
                "conditioned_only_quality_pass": 0,
                "control_only_quality_pass": 0,
                "neither_quality_pass": 0,
                "conditioned_accepted": 0,
                "control_accepted": 0,
                "conditioned_trigger_passed": 0,
                "control_trigger_passed": 0,
                "conditioned_only_trigger_passed": 0,
                "control_only_trigger_passed": 0,
                "risk_deltas": [],
            },
        )
        skill_pair["shared_latent_pairs"] += 1
        conditioned_pass = conditioned.quality_passed
        control_pass = control.quality_passed
        if conditioned_pass and control_pass:
            skill_pair["both_quality_pass"] += 1
        elif conditioned_pass:
            skill_pair["conditioned_only_quality_pass"] += 1
        elif control_pass:
            skill_pair["control_only_quality_pass"] += 1
        else:
            skill_pair["neither_quality_pass"] += 1
        if decision_by_id[conditioned.identity.candidate_id].accepted:
            skill_pair["conditioned_accepted"] += 1
        if decision_by_id[control.identity.candidate_id].accepted:
            skill_pair["control_accepted"] += 1

        def stage_passed(value, stage: str) -> bool:
            return any(
                timed.check.stage.value == stage and timed.check.passed
                for timed in value.checks
            )

        conditioned_trigger_passed = stage_passed(conditioned, "skill_trigger")
        control_trigger_passed = stage_passed(control, "skill_trigger")
        if conditioned_trigger_passed:
            skill_pair["conditioned_trigger_passed"] += 1
        if control_trigger_passed:
            skill_pair["control_trigger_passed"] += 1
        if conditioned_trigger_passed and not control_trigger_passed:
            skill_pair["conditioned_only_trigger_passed"] += 1
        elif control_trigger_passed and not conditioned_trigger_passed:
            skill_pair["control_only_trigger_passed"] += 1

        def risk_value(value):
            for timed in value.checks:
                if timed.check.stage.value == "target_risk":
                    evaluation = timed.check.metrics.get("evaluation", {})
                    result = evaluation.get("value")
                    if isinstance(result, (int, float)) and math.isfinite(float(result)):
                        return float(result)
            return None

        conditioned_risk = risk_value(conditioned)
        control_risk = risk_value(control)
        if conditioned_risk is not None and control_risk is not None:
            skill_pair["risk_deltas"].append(conditioned_risk - control_risk)

    paired_skill_rows = []
    for skill_id in sorted(paired_by_skill):
        entry = paired_by_skill[skill_id]
        risk_deltas = entry.pop("risk_deltas")
        paired_skill_rows.append(
            {
                **entry,
                "risk_delta_conditioned_minus_control": {
                    "count": len(risk_deltas),
                    "median": None if not risk_deltas else median(risk_deltas),
                },
            }
        )

    summary = {
        "version": 1,
        "status": "completed",
        "ability_gate_status": "pending_analysis",
        "stage": "pilot",
        "skills_without_eligible_seed_records": skills_without_eligible_seed_records,
        "pilot_run_id": pilot_run_id,
        "task_plan_id": plan.task_plan_id,
        "task_count": len(plan.tasks),
        "candidate_count": plan.total_candidates,
        "durable_task_count": len(recovery.durable_task_ids),
        "resumed_task_count": len(plan.tasks) - len(tasks_to_generate),
        "newly_generated_task_count": len(tasks_to_generate),
        "newly_generated_candidate_count": newly_generated_candidates,
        "quality_passed_before_diversity": sum(
            item.quality_passed for item in batch.validations
        ),
        "accepted_count": index.accepted_count,
        "rejected_count": index.rejected_count,
        "task_plan_sha256": _file_sha256(plan_path),
        "task_plan_summary_sha256": _file_sha256(plan_summary_path),
        "raw_commit_set_sha256": canonical_sha256(
            [
                {
                    "shard_index": shard.shard_index,
                    "commit_sha256": _file_sha256(shard.commit_path),
                }
                for shard in raw_commits
            ]
        ),
        "raw_snapshot_sha256": canonical_sha256(
            {
                _path_label(path): identity
                for path, identity in sorted(
                    raw_snapshot.items(), key=lambda item: item[0].as_posix()
                )
            }
        ),
        "generation_semantic_sha256": plan.semantic_config_sha256,
        "generation_execution_sha256": plan.execution_config_sha256,
        "pilot_seed_eligibility_sha256": eligibility_sha256,
        "pilot_seed_eligibility_artifact_sha256": _file_sha256(
            eligibility_audit_path
        ),
        "filter_semantic_sha256": fingerprint.semantic_sha256,
        "filter_dependency_sha256": dict(fingerprint.file_sha256),
        "filter_contract_version": FILTER_CONTRACT_VERSION,
        "checkpoint_sha256": config.active_checkpoint.sha256,
        "execution_config": execution_config,
        "stage_execution_counts": dict(batch.stage_execution_counts),
        "stage_elapsed_seconds": dict(batch.stage_elapsed_seconds),
        "timing_seconds": {
            "generation": generation_elapsed,
            "filtering_and_diversity": filtering_elapsed,
            "end_to_end": time.perf_counter() - pilot_started,
        },
        "paired_control": {
            "by_skill": paired_skill_rows,
        },
        "by_skill_and_arm": skill_rows,
        "raw_immutable_verified": True,
        "raw_file_count": len(raw_snapshot),
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "outputs": {
            "eligibility_audit": eligibility_audit_path.as_posix(),
            "task_plan": plan_path.as_posix(),
            "raw": raw_dir.as_posix(),
            "accepted": index.accepted_path.as_posix(),
            "rejected": index.rejected_path.as_posix(),
            "filter_commit": index.commit_path.as_posix(),
        },
        "output_sha256": {
            "eligibility_audit": _file_sha256(eligibility_audit_path),
            "accepted": _file_sha256(index.accepted_path),
            "rejected": _file_sha256(index.rejected_path),
            "filter_commit": _file_sha256(index.commit_path),
        },
    }
    summary_path = write_generation_capability_matrix(
        filter_root / "summary.json",
        summary,
    )
    print(
        "stage D paired Pilot complete: "
        f"{len(plan.tasks)} tasks, {plan.total_candidates} candidates, "
        f"{index.accepted_count} accepted",
        flush=True,
    )
    print(f"pilot summary: {summary_path}", flush=True)
    return summary


def run_latent_search(
    *,
    config_path: Path,
    filter_config_path: Path,
    detection_config_path: Path,
    latent_search_config_path: Path,
    representative_manifest_path: Path,
    output_root: Path,
    device: str,
    progress_interval_seconds: float,
) -> dict[str, Any]:
    """Run the frozen four-representative, six-arm latent Top-K experiment."""

    from skilldrive.generation.latent_search_workflow import (
        run_latent_search_workflow,
    )

    run_audit(
        config_path=config_path,
        filter_config_path=filter_config_path,
        output_root=output_root,
    )
    return run_latent_search_workflow(
        generation_config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
        latent_search_config_path=latent_search_config_path,
        representative_manifest_path=representative_manifest_path,
        output_root=output_root,
        device=device,
        progress_interval_seconds=progress_interval_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "audit",
            "smoke",
            "repair-smoke",
            "repair-heldout-dev-evidence",
            "repair-heldout-rebind",
            "repair-heldout-execute",
            "repair-heldout-gate",
            "filter-smoke",
            "pilot",
            "latent-search",
        ),
        required=True,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/counterfactual_v1.yaml"),
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=Path("configs/generation/filters_v1.yaml"),
    )
    parser.add_argument(
        "--latent-search-config",
        type=Path,
        default=Path("configs/generation/latent_search_v1.yaml"),
    )
    parser.add_argument(
        "--representative-manifest",
        type=Path,
        default=Path("manifests/generation/latent_search_representatives_v1.json"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repair-checkpoint-path", type=Path)
    parser.add_argument("--repair-checkpoint-sha256")
    parser.add_argument("--repair-run-manifest-path", type=Path)
    parser.add_argument("--repair-run-manifest-sha256")
    parser.add_argument(
        "--repair-checkpoint-mode",
        choices=("diagnostic-overfit", "formal"),
    )
    parser.add_argument(
        "--repair-heldout-source-plan",
        type=Path,
        default=Path("manifests/generation/repair_v1/heldout_ability"),
    )
    parser.add_argument(
        "--repair-audit",
        type=Path,
        default=Path("manifests/splits/formal_train_repair_v1.audit.json"),
    )
    parser.add_argument(
        "--repair-heldout-output-root",
        type=Path,
        default=Path(
            "outputs/generation/counterfactual_v1/pilot/repair-heldout-gate-v1"
        ),
    )
    parser.add_argument("--repair-dev-evidence", type=Path)
    parser.add_argument("--repair-dev-evidence-sha256")
    parser.add_argument("--resume", choices=("auto",), default="auto")
    parser.add_argument("--task-batch-size", type=int, default=8)
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument(
        "--detection-config",
        type=Path,
        default=Path("configs/seed_detection.yaml"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "audit":
        run_audit(
            config_path=args.config,
            filter_config_path=args.filter_config,
            output_root=args.output_root,
        )
    elif args.stage == "smoke":
        run_smoke(
            config_path=args.config,
            filter_config_path=args.filter_config,
            output_root=args.output_root,
            device=args.device,
        )
    elif args.stage == "repair-smoke":
        required = {
            "--repair-checkpoint-path": args.repair_checkpoint_path,
            "--repair-checkpoint-sha256": args.repair_checkpoint_sha256,
            "--repair-run-manifest-path": args.repair_run_manifest_path,
            "--repair-run-manifest-sha256": args.repair_run_manifest_sha256,
            "--repair-checkpoint-mode": args.repair_checkpoint_mode,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "repair-smoke requires explicit hash-bound checkpoint arguments: "
                f"{missing}"
            )
        run_repair_prior_smoke(
            config_path=args.config,
            filter_config_path=args.filter_config,
            output_root=args.output_root,
            device=args.device,
            repair_checkpoint_path=args.repair_checkpoint_path,
            repair_checkpoint_sha256=args.repair_checkpoint_sha256,
            repair_run_manifest_path=args.repair_run_manifest_path,
            repair_run_manifest_sha256=args.repair_run_manifest_sha256,
            repair_checkpoint_mode=args.repair_checkpoint_mode,
        )
    elif args.stage == "repair-heldout-dev-evidence":
        from skilldrive.generation.heldout_gate import (
            build_repair_dev_candidate_evidence,
        )

        required = {
            "--repair-checkpoint-path": args.repair_checkpoint_path,
            "--repair-checkpoint-sha256": args.repair_checkpoint_sha256,
            "--repair-run-manifest-path": args.repair_run_manifest_path,
            "--repair-run-manifest-sha256": args.repair_run_manifest_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or args.repair_checkpoint_mode != "formal":
            raise ValueError(
                "repair-heldout-dev-evidence requires a hash-bound "
                f"repair-formal epoch checkpoint; missing={missing}"
            )
        summary = build_repair_dev_candidate_evidence(
            checkpoint_path=args.repair_checkpoint_path,
            checkpoint_sha256=args.repair_checkpoint_sha256,
            run_manifest_path=args.repair_run_manifest_path,
            run_manifest_sha256=args.repair_run_manifest_sha256,
            config_path=args.config,
            output_root=args.repair_heldout_output_root,
        )
        print(
            "repair dev candidate evidence: "
            f"{summary['evidence']['status']} -> "
            f"{summary['outputs']['repair_dev_candidate_gate']}"
        )
    elif args.stage == "repair-heldout-rebind":
        from skilldrive.generation.heldout_gate import (
            rebind_repair_heldout_plan,
        )

        required = {
            "--repair-checkpoint-path": args.repair_checkpoint_path,
            "--repair-checkpoint-sha256": args.repair_checkpoint_sha256,
            "--repair-run-manifest-path": args.repair_run_manifest_path,
            "--repair-run-manifest-sha256": args.repair_run_manifest_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or args.repair_checkpoint_mode != "formal":
            raise ValueError(
                "repair-heldout-rebind requires a hash-bound "
                f"repair-formal epoch checkpoint; missing={missing}"
            )
        summary = rebind_repair_heldout_plan(
            checkpoint_path=args.repair_checkpoint_path,
            checkpoint_sha256=args.repair_checkpoint_sha256,
            run_manifest_path=args.repair_run_manifest_path,
            run_manifest_sha256=args.repair_run_manifest_sha256,
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            source_plan_dir=args.repair_heldout_source_plan,
            repair_audit_path=args.repair_audit,
            output_root=args.repair_heldout_output_root,
            device=args.device,
            task_batch_size=args.task_batch_size,
        )
        print(f"repair heldout rebound: {summary['outputs']['rebind_contract']}")
    elif args.stage == "repair-heldout-execute":
        from skilldrive.generation.heldout_gate import (
            execute_repair_heldout_plan,
        )

        required = {
            "--repair-checkpoint-path": args.repair_checkpoint_path,
            "--repair-checkpoint-sha256": args.repair_checkpoint_sha256,
            "--repair-run-manifest-path": args.repair_run_manifest_path,
            "--repair-run-manifest-sha256": args.repair_run_manifest_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or args.repair_checkpoint_mode != "formal":
            raise ValueError(
                "repair-heldout-execute requires a rebound hash-bound "
                f"repair-formal epoch checkpoint; missing={missing}"
            )
        summary = execute_repair_heldout_plan(
            checkpoint_path=args.repair_checkpoint_path,
            checkpoint_sha256=args.repair_checkpoint_sha256,
            run_manifest_path=args.repair_run_manifest_path,
            run_manifest_sha256=args.repair_run_manifest_sha256,
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            source_plan_dir=args.repair_heldout_source_plan,
            repair_audit_path=args.repair_audit,
            output_root=args.repair_heldout_output_root,
            device=args.device,
            task_batch_size=args.task_batch_size,
            progress_interval_seconds=args.progress_interval_seconds,
        )
        print(
            "repair heldout execution: "
            f"{summary['status']} -> {summary['outputs']['execution_summary']}"
        )
    elif args.stage == "repair-heldout-gate":
        from skilldrive.generation.heldout_gate import (
            aggregate_repair_heldout_gate,
        )

        required = {
            "--repair-checkpoint-path": args.repair_checkpoint_path,
            "--repair-checkpoint-sha256": args.repair_checkpoint_sha256,
            "--repair-run-manifest-path": args.repair_run_manifest_path,
            "--repair-run-manifest-sha256": args.repair_run_manifest_sha256,
            "--repair-dev-evidence": args.repair_dev_evidence,
            "--repair-dev-evidence-sha256": args.repair_dev_evidence_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing or args.repair_checkpoint_mode != "formal":
            raise ValueError(
                "repair-heldout-gate requires completed hash-bound formal evidence; "
                f"missing={missing}"
            )
        summary = aggregate_repair_heldout_gate(
            checkpoint_path=args.repair_checkpoint_path,
            checkpoint_sha256=args.repair_checkpoint_sha256,
            run_manifest_path=args.repair_run_manifest_path,
            run_manifest_sha256=args.repair_run_manifest_sha256,
            repair_dev_evidence_path=args.repair_dev_evidence,
            repair_dev_evidence_sha256=args.repair_dev_evidence_sha256,
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            output_root=args.repair_heldout_output_root,
        )
        print(
            "repair heldout recommendation: "
            f"{summary['recommendation']} -> "
            f"{summary['outputs']['promotion_recommendation']}"
        )
    elif args.stage == "filter-smoke":
        run_filter_smoke(
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            output_root=args.output_root,
        )
    elif args.stage == "pilot":
        run_pilot(
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            output_root=args.output_root,
            device=args.device,
            task_batch_size=args.task_batch_size,
            progress_interval_seconds=args.progress_interval_seconds,
        )
    elif args.stage == "latent-search":
        run_latent_search(
            config_path=args.config,
            filter_config_path=args.filter_config,
            detection_config_path=args.detection_config,
            latent_search_config_path=args.latent_search_config,
            representative_manifest_path=args.representative_manifest,
            output_root=args.output_root,
            device=args.device,
            progress_interval_seconds=args.progress_interval_seconds,
        )


if __name__ == "__main__":
    main()

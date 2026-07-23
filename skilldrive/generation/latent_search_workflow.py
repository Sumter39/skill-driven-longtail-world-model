"""Execute the fixed stage-D latent Top-K search without trajectory repair."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from skilldrive.data import build_cvae_schema, tensorize_prior_context
from skilldrive.data.av2_reader import load_av2_history_scenario, load_av2_scenario
from skilldrive.filtering.common import KinematicLimits
from skilldrive.filtering.context import bind_raw_candidates
from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.filtering.pipeline import (
    FILTER_CONTRACT_VERSION,
    CandidateFilterInput,
    CompactCandidateValidationResult,
    finalize_candidate_validations,
    validate_candidate,
)
from skilldrive.generation.assembly import local_futures_to_global
from skilldrive.generation.capability import write_generation_capability_matrix
from skilldrive.generation.config import (
    CounterfactualFilterConfig,
    CounterfactualGenerationConfig,
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import (
    GeneratedCandidate,
    GeneratedOverlay,
    canonical_sha256,
)
from skilldrive.generation.inference import generate_prior_batch, load_configured_cvae
from skilldrive.generation.latent_search import (
    LatentSearchConfig,
    LatentSearchManifest,
    LatentSearchTask,
    build_latent_search_tasks,
    latent_search_plan_id,
    latent_search_plan_payload,
    load_latent_search_config,
    load_latent_search_manifest,
)
from skilldrive.generation.planning import (
    latent_group_id,
    paired_latent_seed,
    prior_context_spec_for_task,
    seed_record_id,
)
from skilldrive.generation.search import (
    KinematicCandidateScores,
    KinematicTopKAccumulator,
    score_kinematic_candidates,
)
from skilldrive.generation.storage import (
    RawShardCommit,
    load_raw_shard_candidates,
    verify_raw_shard,
    write_filter_indexes,
    write_raw_shard,
)
from skilldrive.schemas import Scenario
from skilldrive.seeds import read_seed_records
from skilldrive.skills.detection import load_detection_config
from skilldrive.skills.loader import load_skill


LATENT_SEARCH_OUTPUT_CONTRACT = "latent-search-v1"
_SEARCH_EVIDENCE_VERSION = 1


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _validate_progress_interval(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError("progress_interval_seconds must be a positive finite number")
    return float(value)


def _source_path(
    generation_config: CounterfactualGenerationConfig,
    source_path: str,
) -> Path:
    pure = PurePosixPath(source_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "train":
        raise ValueError(f"latent-search source must stay inside Formal Train: {source_path}")
    path = generation_config.inputs.data_root.joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(f"latent-search source scenario is missing: {path}")
    return path


def _kinematic_limits(
    scenario: Scenario,
    target_track_id: str,
    filter_config: CounterfactualFilterConfig,
) -> KinematicLimits:
    target = next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )
    if target is None:
        raise ValueError(f"latent-search target is missing: {target_track_id}")
    policy = filter_config.kinematics_by_type.get(target.object_type.lower())
    if policy is None:
        raise ValueError(
            "latent-search cannot rank an unsupported kinematic class: "
            f"{target.object_type.lower()}"
        )
    return KinematicLimits(
        maximum_seam_speed_mps=policy.maximum_seam_speed_mps,
        maximum_speed_mps=policy.maximum_speed_mps,
        maximum_acceleration_mps2=policy.maximum_acceleration_mps2,
        maximum_deceleration_mps2=policy.maximum_deceleration_mps2,
        maximum_jerk_mps3=policy.maximum_jerk_mps3,
        maximum_curvature_per_m=policy.maximum_curvature_per_m,
        maximum_heading_rate_rad_s=policy.maximum_heading_rate_rad_s,
        minimum_heading_speed_mps=policy.minimum_heading_speed_mps,
    )


def _seed_vector(
    task: LatentSearchTask,
    *,
    base_seed: int,
    start: int,
    count: int,
) -> np.ndarray:
    return np.asarray(
        [
            paired_latent_seed(base_seed, task.task, candidate_index)
            for candidate_index in range(start, start + count)
        ],
        dtype=np.int64,
    )


def _score_row(scores: KinematicCandidateScores, row: int) -> dict[str, Any]:
    def finite_or_none(value: Any) -> float | None:
        number = float(value)
        return number if math.isfinite(number) else None

    return {
        "passed": bool(scores.passed[row]),
        "finite_kinematics": bool(scores.finite_kinematics[row]),
        "normalized_violation_score": finite_or_none(
            scores.normalized_violation_score[row]
        ),
        "rejection_reasons": [item.value for item in scores.rejection_reasons[row]],
        "seam_speed_mps": finite_or_none(scores.seam_speed_mps[row]),
        "maximum_speed_mps": finite_or_none(scores.maximum_speed_mps[row]),
        "maximum_speed_future_index": int(scores.maximum_speed_future_index[row]),
        "maximum_acceleration_mps2": finite_or_none(
            scores.maximum_acceleration_mps2[row]
        ),
        "maximum_acceleration_future_index": int(
            scores.maximum_acceleration_future_index[row]
        ),
        "maximum_deceleration_mps2": finite_or_none(
            scores.maximum_deceleration_mps2[row]
        ),
        "maximum_deceleration_future_index": int(
            scores.maximum_deceleration_future_index[row]
        ),
        "maximum_jerk_mps3": finite_or_none(scores.maximum_jerk_mps3[row]),
        "maximum_jerk_future_index": int(scores.maximum_jerk_future_index[row]),
        "maximum_curvature_per_m": finite_or_none(
            scores.maximum_curvature_per_m[row]
        ),
        "maximum_curvature_future_index": int(
            scores.maximum_curvature_future_index[row]
        ),
        "maximum_heading_rate_rad_s": finite_or_none(
            scores.maximum_heading_rate_rad_s[row]
        ),
        "maximum_heading_rate_future_index": int(
            scores.maximum_heading_rate_future_index[row]
        ),
        "low_speed_heading_suppressed_steps": int(
            scores.low_speed_heading_suppressed_steps[row]
        ),
    }


def _raw_snapshot(shards: Sequence[RawShardCommit]) -> dict[Path, dict[str, Any]]:
    return {
        path.resolve(): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for shard in shards
        for path in (shard.arrays_path, shard.metadata_path, shard.commit_path)
    }


def _task_evidence_path(root: Path, task: LatentSearchTask) -> Path:
    return root / "search-evidence" / f"task-{task.task.task_index:05d}.json"


def _raw_commit_path(root: Path, task: LatentSearchTask) -> Path:
    return root / "raw" / f"shard-{task.task.task_index:05d}.commit.json"


def _load_json(path: Path) -> dict[str, Any]:
    import json

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {path}")
    return value


def _load_completed_task(
    run_root: Path,
    task: LatentSearchTask,
    *,
    run_id: str,
    plan_id: str,
    execution_config_sha256: str,
    config: LatentSearchConfig,
) -> tuple[dict[str, Any], RawShardCommit] | None:
    evidence_path = _task_evidence_path(run_root, task)
    commit_path = _raw_commit_path(run_root, task)
    if not evidence_path.exists():
        return None
    if not commit_path.is_file():
        raise ValueError("latent-search evidence exists without its raw commit")
    evidence = _load_json(evidence_path)
    expected = {
        "version": _SEARCH_EVIDENCE_VERSION,
        "kind": "latent_search_task_evidence",
        "status": "completed",
        "run_id": run_id,
        "plan_id": plan_id,
        "execution_config_sha256": execution_config_sha256,
        "task_index": task.task.task_index,
        "task_id": task.task.task_id,
        "representative_id": task.representative_id,
        "evaluation_arm": task.evaluation_arm,
        "candidate_budget": config.candidate_budget_per_arm,
        "generation_chunk_size": config.generation_chunk_size,
        "kinematic_top_k": config.kinematic_top_k,
        "raw_policy": "top_k_only",
        "latent_contract": config.latent_contract,
        "external_smoothing_applied": False,
        "filter_threshold_overrides_applied": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    mismatches = {
        name: (evidence.get(name), value)
        for name, value in expected.items()
        if evidence.get(name) != value
    }
    if mismatches:
        raise ValueError(f"latent-search task evidence contract changed: {mismatches}")
    shard = verify_raw_shard(
        commit_path,
        expected_semantic_config_sha256=task.task.semantic_config_sha256,
    )
    if shard.execution_config_sha256 != execution_config_sha256:
        raise ValueError("latent-search raw shard uses another execution contract")
    if shard.candidate_count != config.kinematic_top_k:
        raise ValueError("latent-search raw shard does not contain exactly Top-K")
    if evidence.get("raw_commit_sha256") != _file_sha256(commit_path):
        raise ValueError("latent-search task evidence raw commit hash changed")
    raw = load_raw_shard_candidates(shard)
    indices = [item.candidate_index for item in raw]
    seeds = [item.latent_seed for item in raw]
    if (
        len(indices) != config.kinematic_top_k
        or len(set(indices)) != len(indices)
        or any(not 0 <= index < config.candidate_budget_per_arm for index in indices)
    ):
        raise ValueError("latent-search Top-K evidence has invalid candidate indices")
    if indices != evidence.get("top_k_candidate_indices"):
        raise ValueError("latent-search raw candidate order differs from evidence")
    if seeds != evidence.get("top_k_latent_seeds"):
        raise ValueError("latent-search raw latent seeds differ from evidence")
    for item in raw:
        expected_seed = paired_latent_seed(
            config.base_seed,
            task.task,
            item.candidate_index,
        )
        if item.latent_seed != expected_seed:
            raise ValueError("latent-search raw candidate cannot be regenerated by seed")
    full_seed_digest = canonical_sha256(
        [
            paired_latent_seed(config.base_seed, task.task, candidate_index)
            for candidate_index in range(config.candidate_budget_per_arm)
        ]
    )
    if evidence.get("latent_seed_sequence_sha256") != full_seed_digest:
        raise ValueError("latent-search full latent seed sequence differs from evidence")
    passed_count = evidence.get("kinematic_passed_count")
    if (
        isinstance(passed_count, bool)
        or not isinstance(passed_count, int)
        or not 0 <= passed_count <= config.candidate_budget_per_arm
    ):
        raise ValueError("latent-search kinematic pass count is invalid")
    if evidence.get("kinematic_pass_rate") != (
        passed_count / config.candidate_budget_per_arm
    ):
        raise ValueError("latent-search kinematic pass rate differs from its count")
    return evidence, shard


def _candidate_metadata(
    *,
    task: LatentSearchTask,
    primary_generated_role: str,
    record,
    search_rank: int,
    kinematic_evidence: Mapping[str, Any],
    config: LatentSearchConfig,
) -> dict[str, Any]:
    return {
        "condition_skill_id": task.task.condition_skill_id,
        "evaluation_arm": task.evaluation_arm,
        "representative_id": task.representative_id,
        "latent_group_id": latent_group_id(task.task),
        "primary_generated_role": primary_generated_role,
        "requested_parameters": record.sampled_parameters,
        "detection_mode": record.evidence["detection_mode"],
        "latent_search": {
            "contract": config.contract_name,
            "candidate_budget": config.candidate_budget_per_arm,
            "kinematic_top_k": config.kinematic_top_k,
            "search_rank": search_rank,
            "kinematic_evidence": dict(kinematic_evidence),
            "raw_policy": config.raw_policy,
            "external_smoothing_applied": False,
            "interpolation_repair_applied": False,
            "filter_threshold_overrides_applied": False,
        },
    }


def _generate_task_top_k(
    *,
    task: LatentSearchTask,
    record,
    context,
    history_scenario: Scenario,
    runtime,
    filter_config: CounterfactualFilterConfig,
    generation_config: CounterfactualGenerationConfig,
    config: LatentSearchConfig,
    run_root: Path,
    run_id: str,
    plan_id: str,
    execution_config_sha256: str,
    progress_interval_seconds: float,
    repository_root: Path,
) -> tuple[dict[str, Any], RawShardCommit]:
    accumulator = KinematicTopKAccumulator(config.kinematic_top_k)
    retained: dict[int, tuple[np.ndarray, int, dict[str, Any]]] = {}
    passed_count = 0
    limits = _kinematic_limits(history_scenario, task.task.target_track_id, filter_config)
    started = time.perf_counter()
    last_progress = started
    for start in range(0, config.candidate_budget_per_arm, config.generation_chunk_size):
        count = min(
            config.generation_chunk_size,
            config.candidate_budget_per_arm - start,
        )
        candidate_indices = np.arange(start, start + count, dtype=np.int64)
        latent_seeds = _seed_vector(
            task,
            base_seed=config.base_seed,
            start=start,
            count=count,
        )
        generated = generate_prior_batch(
            runtime,
            [context],
            latent_seeds[None, :],
            use_bfloat16=config.use_bfloat16,
        )
        global_futures = local_futures_to_global(
            generated.future_position_local[0],
            context.anchor_origin_global,
            float(context.anchor_heading_global),
        ).astype(np.float32)
        scores = score_kinematic_candidates(
            history_scenario,
            task.task.target_track_id,
            global_futures,
            limits,
            latent_seeds,
            candidate_indices=candidate_indices,
            top_k=config.kinematic_top_k,
        )
        passed_count += int(np.count_nonzero(scores.passed))
        accumulator.update(scores)
        row_by_index = {
            int(candidate_index): row
            for row, candidate_index in enumerate(scores.candidate_indices)
        }
        for candidate_index in scores.top_k_indices:
            index = int(candidate_index)
            row = row_by_index[index]
            retained[index] = (
                np.ascontiguousarray(global_futures[row].copy()),
                int(scores.latent_seeds[row]),
                _score_row(scores, row),
            )
        keep = {int(value) for value in accumulator.top_k_indices}
        retained = {index: value for index, value in retained.items() if index in keep}
        now = time.perf_counter()
        completed = start + count
        if (
            now - last_progress >= progress_interval_seconds
            or completed == config.candidate_budget_per_arm
        ):
            elapsed = max(now - started, 1e-9)
            rate = completed / elapsed
            remaining = config.candidate_budget_per_arm - completed
            eta = None if rate <= 0.0 else remaining / rate
            print(
                "latent-search generation: "
                f"task {task.task.task_index + 1}/6 {task.representative_id}/"
                f"{task.evaluation_arm}, {completed}/{config.candidate_budget_per_arm}, "
                f"{passed_count} kinematic passes, {rate:.1f} candidates/s, ETA "
                f"{('--:--' if eta is None else f'{int(eta // 60):02d}:{int(eta % 60):02d}')}",
                flush=True,
            )
            last_progress = now
    top_k_indices = [int(value) for value in accumulator.top_k_indices]
    if len(top_k_indices) != config.kinematic_top_k or set(top_k_indices) != set(retained):
        raise RuntimeError("latent-search Top-K accumulator lost retained candidates")
    primary_role = generation_config.skills_by_id[task.task.skill_id].primary_generated_role
    candidates = []
    for search_rank, candidate_index in enumerate(top_k_indices):
        future, latent_seed, kinematic_evidence = retained[candidate_index]
        candidates.append(
            GeneratedCandidate(
                task_id=task.task.task_id,
                candidate_index=candidate_index,
                latent_seed=latent_seed,
                scenario_id=task.task.scenario_id,
                skill_id=task.task.skill_id,
                proposal_mode=task.task.proposal_mode,
                checkpoint_sha256=task.task.checkpoint_sha256,
                semantic_config_sha256=task.task.semantic_config_sha256,
                overlay=GeneratedOverlay(
                    target_track_id=task.task.target_track_id,
                    future_xy_global=future,
                ),
                metadata=_candidate_metadata(
                    task=task,
                    primary_generated_role=primary_role,
                    record=record,
                    search_rank=search_rank,
                    kinematic_evidence=kinematic_evidence,
                    config=config,
                ),
            )
        )

    commit_path = _raw_commit_path(run_root, task)
    if commit_path.is_file():
        shard = verify_raw_shard(
            commit_path,
            expected_semantic_config_sha256=task.task.semantic_config_sha256,
        )
        if shard.execution_config_sha256 != execution_config_sha256:
            raise ValueError("existing latent-search raw shard has another execution hash")
        stored = load_raw_shard_candidates(shard)
        if [item.candidate_id for item in stored] != [item.candidate_id for item in candidates]:
            raise ValueError("regenerated Top-K differs from the existing raw shard")
        for existing, regenerated in zip(stored, candidates):
            if not np.array_equal(
                existing.future_xy_global,
                regenerated.overlay.future_xy_global,
            ):
                raise ValueError("regenerated Top-K trajectory differs from durable raw")
    else:
        shard = write_raw_shard(
            run_root / "raw",
            task.task.task_index,
            candidates,
            semantic_config_sha256=task.task.semantic_config_sha256,
            execution_config_sha256=execution_config_sha256,
        )
    latent_seed_sequence = [
        paired_latent_seed(config.base_seed, task.task, candidate_index)
        for candidate_index in range(config.candidate_budget_per_arm)
    ]
    evidence = {
        "version": _SEARCH_EVIDENCE_VERSION,
        "kind": "latent_search_task_evidence",
        "status": "completed",
        "run_id": run_id,
        "plan_id": plan_id,
        "execution_config_sha256": execution_config_sha256,
        "task_index": task.task.task_index,
        "task_id": task.task.task_id,
        "representative_id": task.representative_id,
        "evaluation_arm": task.evaluation_arm,
        "scenario_id": task.task.scenario_id,
        "skill_id": task.task.skill_id,
        "seed_record_id": task.task.seed_record_id,
        "target_track_id": task.task.target_track_id,
        "condition_skill_id": task.task.condition_skill_id,
        "candidate_budget": config.candidate_budget_per_arm,
        "generation_chunk_size": config.generation_chunk_size,
        "kinematic_top_k": config.kinematic_top_k,
        "candidate_count": config.candidate_budget_per_arm,
        "kinematic_passed_count": passed_count,
        "kinematic_pass_rate": passed_count / config.candidate_budget_per_arm,
        "top_k_kinematic_passed_count": sum(
            bool(retained[index][2]["passed"]) for index in top_k_indices
        ),
        "top_k_candidate_indices": top_k_indices,
        "top_k_latent_seeds": [retained[index][1] for index in top_k_indices],
        "latent_seed_sequence_sha256": canonical_sha256(latent_seed_sequence),
        "latent_contract": config.latent_contract,
        "raw_policy": config.raw_policy,
        "raw_commit_path": _path_label(shard.commit_path, repository_root),
        "raw_commit_sha256": _file_sha256(shard.commit_path),
        "raw_candidate_count": shard.candidate_count,
        "all_candidates_reproducible_from_seed": True,
        "source_future_accessed_during_search": False,
        "external_smoothing_applied": False,
        "interpolation_repair_applied": False,
        "filter_threshold_overrides_applied": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    evidence_path = write_generation_capability_matrix(
        _task_evidence_path(run_root, task),
        evidence,
    )
    if _load_json(evidence_path) != evidence:
        raise RuntimeError("latent-search task evidence changed while writing")
    return evidence, shard


def _filter_funnel(
    validations: Sequence[CompactCandidateValidationResult],
    decisions,
) -> dict[str, Any]:
    passed_by_stage = Counter()
    for validation in validations:
        for timed in validation.checks:
            if timed.check.passed:
                passed_by_stage[timed.check.stage.value] += 1
    accepted = sum(item.accepted for item in decisions)
    passed_by_stage["diversity"] = accepted
    return {
        "input_top_k": len(validations),
        "passed_by_stage": {
            stage: passed_by_stage[stage]
            for stage in (
                "schema_finite",
                "history_invariants",
                "kinematics",
                "map",
                "collision",
                "target_risk",
                "skill_trigger",
                "parameter_realization",
                "diversity",
            )
        },
        "quality_passed_before_diversity": sum(
            item.quality_passed for item in validations
        ),
        "accepted": accepted,
        "first_failed_stages": dict(
            sorted(
                Counter(
                    item.metrics["first_failed_stage"]
                    for item in decisions
                    if not item.accepted
                ).items()
            )
        ),
        "primary_rejections": dict(
            sorted(
                Counter(
                    item.rejection_reasons[0]
                    for item in decisions
                    if not item.accepted
                ).items()
            )
        ),
    }


def run_latent_search_workflow(
    *,
    generation_config_path: Path,
    filter_config_path: Path,
    detection_config_path: Path,
    latent_search_config_path: Path,
    representative_manifest_path: Path,
    output_root: Path,
    device: str,
    progress_interval_seconds: float,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Generate 4096 Prior candidates per arm, retain Top-K, then fully filter."""

    repository = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    progress_interval = _validate_progress_interval(progress_interval_seconds)
    started = time.perf_counter()
    generation_config = load_counterfactual_config(generation_config_path)
    filter_config = load_filter_config(filter_config_path)
    detection_config = load_detection_config(detection_config_path)
    search_config = load_latent_search_config(latent_search_config_path)
    manifest = load_latent_search_manifest(
        representative_manifest_path,
        config=search_config,
        repository_root=repository,
    )
    tasks = build_latent_search_tasks(
        manifest,
        config=search_config,
        generation_config=generation_config,
        repository_root=repository,
    )
    plan_payload = latent_search_plan_payload(
        tasks,
        config=search_config,
        manifest=manifest,
    )
    plan_id = latent_search_plan_id(tasks, config=search_config, manifest=manifest)
    filter_fingerprint = build_filter_semantic_fingerprint(
        repository_root=repository,
        generation_config_path=generation_config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    source_paths = {
        *Path("skilldrive/generation").glob("*.py"),
        Path("skilldrive/data/av2_reader.py"),
        Path("skilldrive/data/coordinates.py"),
        Path("skilldrive/data/cvae_samples.py"),
        Path("skilldrive/models/conditional_cvae.py"),
        Path("scripts/generation/run_counterfactual_pipeline.py"),
    }
    execution_config = {
        "version": 1,
        "device": device,
        "use_bfloat16": search_config.use_bfloat16,
        "candidate_budget_per_arm": search_config.candidate_budget_per_arm,
        "generation_chunk_size": search_config.generation_chunk_size,
        "kinematic_top_k": search_config.kinematic_top_k,
        "raw_policy": search_config.raw_policy,
        "latent_contract": search_config.latent_contract,
        "source_sha256": {
            path.as_posix(): _file_sha256(path)
            for path in sorted(source_paths)
            if path.is_file()
        },
    }
    execution_config_sha256 = canonical_sha256(execution_config)
    run_id = canonical_sha256(
        {
            "version": 1,
            "plan_id": plan_id,
            "execution_config_sha256": execution_config_sha256,
            "filter_semantic_sha256": filter_fingerprint.semantic_sha256,
            "latent_search_config_sha256": _file_sha256(latent_search_config_path),
            "representative_manifest_sha256": manifest.sha256,
        }
    )
    run_root = (
        output_root / "pilot" / LATENT_SEARCH_OUTPUT_CONTRACT / run_id
    )
    plan_path = run_root / "task_plan.json"
    if plan_path.exists():
        if _load_json(plan_path) != plan_payload:
            raise ValueError("latent-search durable task plan differs from current plan")
    else:
        write_generation_capability_matrix(plan_path, plan_payload)

    records = read_seed_records(generation_config.inputs.seed_manifest)
    records_by_id = {seed_record_id(record): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")
    selected_records = {}
    source_paths_by_task = {}
    for item in tasks:
        record = records_by_id.get(item.task.seed_record_id)
        if record is None:
            raise ValueError("latent-search task references an unknown formal seed")
        if (record.scenario_id, record.skill_id) != (
            item.task.scenario_id,
            item.task.skill_id,
        ):
            raise ValueError("latent-search seed identity differs from its task")
        selected_records[item.task.task_id] = record
        source_paths_by_task[item.task.task_id] = _source_path(
            generation_config,
            record.source_path,
        )

    learned_pairs: dict[str, list[LatentSearchTask]] = {}
    for item in tasks:
        if item.evaluation_arm in {"learned_conditioned", "learned_none_control"}:
            learned_pairs.setdefault(item.representative_id, []).append(item)
    for representative_id, pair in learned_pairs.items():
        if len(pair) != 2:
            raise ValueError(f"learned latent-search pair is incomplete: {representative_id}")
        left = _seed_vector(
            pair[0],
            base_seed=search_config.base_seed,
            start=0,
            count=search_config.candidate_budget_per_arm,
        )
        right = _seed_vector(
            pair[1],
            base_seed=search_config.base_seed,
            start=0,
            count=search_config.candidate_budget_per_arm,
        )
        if not np.array_equal(left, right):
            raise RuntimeError("learned conditioned/control arms do not share latent epsilon")

    completed: dict[str, tuple[dict[str, Any], RawShardCommit]] = {}
    for item in tasks:
        durable = _load_completed_task(
            run_root,
            item,
            run_id=run_id,
            plan_id=plan_id,
            execution_config_sha256=execution_config_sha256,
            config=search_config,
        )
        if durable is not None:
            completed[item.task.task_id] = durable
    print(
        "latent-search resume: "
        f"{len(completed)}/{len(tasks)} task arms durable, "
        f"{len(tasks) - len(completed)} to generate",
        flush=True,
    )

    schema = build_cvae_schema(generation_config.formal_catalog.parent)
    runtime = None
    history_cache: dict[Path, Scenario] = {}
    context_cache = {}
    generation_started = time.perf_counter()
    for item in tasks:
        if item.task.task_id in completed:
            continue
        if runtime is None:
            runtime = load_configured_cvae(
                active_checkpoint=generation_config.active_checkpoint,
                schema=schema,
                device=device,
            )
        record = selected_records[item.task.task_id]
        source_path = source_paths_by_task[item.task.task_id]
        history = history_cache.get(source_path)
        if history is None:
            history = load_av2_history_scenario(source_path)
            if len(history.timestamps) != 50 or history.metadata.get("temporal_scope") != "history_only":
                raise ValueError("latent-search Prior path must load history-only scenarios")
            history_cache[source_path] = history
        context = context_cache.get(item.task.task_id)
        if context is None:
            context = tensorize_prior_context(
                history,
                prior_context_spec_for_task(item.task, record),
                schema,
            )
            if context.target_track_id != item.task.target_track_id:
                raise ValueError("latent-search tensor target differs from the task")
            if hasattr(context, "target_future") or hasattr(context, "target_future_mask"):
                raise ValueError("latent-search Prior context exposed future tensors")
            context_cache[item.task.task_id] = context
        completed[item.task.task_id] = _generate_task_top_k(
            task=item,
            record=record,
            context=context,
            history_scenario=history,
            runtime=runtime,
            filter_config=filter_config,
            generation_config=generation_config,
            config=search_config,
            run_root=run_root,
            run_id=run_id,
            plan_id=plan_id,
            execution_config_sha256=execution_config_sha256,
            progress_interval_seconds=progress_interval,
            repository_root=repository,
        )
    generation_elapsed = time.perf_counter() - generation_started
    history_cache.clear()
    context_cache.clear()
    if len(completed) != len(tasks):
        raise RuntimeError("latent-search did not complete every task arm")

    evidence_by_task = {task_id: value[0] for task_id, value in completed.items()}
    raw_shards = tuple(completed[item.task.task_id][1] for item in tasks)
    snapshot_before = _raw_snapshot(raw_shards)
    skills = {
        item.task.skill_id: load_skill(
            generation_config.formal_catalog.parent / f"{item.task.skill_id}.yaml"
        )
        for item in tasks
    }
    validations: list[CompactCandidateValidationResult] = []
    validation_task_ids: list[str] = []
    source_cache: dict[Path, Scenario] = {}
    filtering_started = time.perf_counter()
    filtered_count = 0
    last_progress = filtering_started
    for item, shard in zip(tasks, raw_shards):
        record = selected_records[item.task.task_id]
        source_path = source_paths_by_task[item.task.task_id]
        source = source_cache.get(source_path)
        if source is None:
            source = load_av2_scenario(source_path)
            source_cache[source_path] = source
        raw = load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=item.task.semantic_config_sha256,
        )
        bound = bind_raw_candidates(raw, [item.task], [record])
        primary_role = generation_config.skills_by_id[
            item.task.skill_id
        ].primary_generated_role
        cohort = f"{item.representative_id}/{item.evaluation_arm}"
        for candidate in bound:
            validation = validate_candidate(
                CandidateFilterInput(
                    bound=candidate,
                    skill=skills[item.task.skill_id],
                    source_scenario=source,
                    primary_generated_role=primary_role,
                ),
                filter_config=filter_config,
                detection_config=detection_config,
            )
            validations.append(validation.compact(cohort=cohort))
            validation_task_ids.append(item.task.task_id)
            filtered_count += 1
        now = time.perf_counter()
        total_top_k = len(tasks) * search_config.kinematic_top_k
        if now - last_progress >= progress_interval or filtered_count == total_top_k:
            elapsed = max(now - filtering_started, 1e-9)
            rate = filtered_count / elapsed
            remaining = total_top_k - filtered_count
            eta = None if rate <= 0.0 else remaining / rate
            print(
                "latent-search filtering: "
                f"{filtered_count}/{total_top_k} Top-K candidates, "
                f"{rate:.1f} candidates/s, ETA "
                f"{('--:--' if eta is None else f'{int(eta // 60):02d}:{int(eta % 60):02d}')}",
                flush=True,
            )
            last_progress = now
    batch = finalize_candidate_validations(
        validations,
        filter_config=filter_config,
        filter_semantic_sha256=filter_fingerprint.semantic_sha256,
    )
    filtering_elapsed = time.perf_counter() - filtering_started
    filter_root = run_root / "filter-evaluations" / filter_fingerprint.semantic_sha256
    index = write_filter_indexes(
        filter_root,
        raw_shards,
        batch.decisions,
        filter_config_sha256=filter_fingerprint.semantic_sha256,
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )
    snapshot_after = _raw_snapshot(raw_shards)
    if snapshot_after != snapshot_before:
        raise RuntimeError("latent-search full filtering modified committed Top-K raw")

    per_task = []
    decision_by_id = {item.candidate_id: item for item in batch.decisions}
    for item in tasks:
        task_validations = tuple(
            value
            for value, task_id in zip(validations, validation_task_ids)
            if task_id == item.task.task_id
        )
        task_decisions = tuple(
            decision_by_id[value.identity.candidate_id] for value in task_validations
        )
        evidence = evidence_by_task[item.task.task_id]
        per_task.append(
            {
                "task_index": item.task.task_index,
                "task_id": item.task.task_id,
                "representative_id": item.representative_id,
                "evaluation_arm": item.evaluation_arm,
                "scenario_id": item.task.scenario_id,
                "skill_id": item.task.skill_id,
                "seed_record_id": item.task.seed_record_id,
                "candidate_count": evidence["candidate_count"],
                "kinematic_passed_count": evidence["kinematic_passed_count"],
                "kinematic_pass_rate": evidence["kinematic_pass_rate"],
                "top_k_count": evidence["raw_candidate_count"],
                "top_k_full_filter_funnel": _filter_funnel(
                    task_validations,
                    task_decisions,
                ),
                "accepted_candidate_ids": [
                    decision.candidate_id
                    for decision in task_decisions
                    if decision.accepted
                ],
                "search_evidence_path": _path_label(
                    _task_evidence_path(run_root, item), repository
                ),
                "search_evidence_sha256": _file_sha256(
                    _task_evidence_path(run_root, item)
                ),
            }
        )
    paired_latent = []
    for representative_id, pair in sorted(learned_pairs.items()):
        hashes = {
            evidence_by_task[item.task.task_id]["latent_seed_sequence_sha256"]
            for item in pair
        }
        if len(hashes) != 1:
            raise RuntimeError("learned latent-search pair seed sequence drifted")
        paired_latent.append(
            {
                "representative_id": representative_id,
                "task_ids": [item.task.task_id for item in pair],
                "candidate_pairs": search_config.candidate_budget_per_arm,
                "latent_seed_sequence_sha256": next(iter(hashes)),
                "shared_latent_verified": True,
            }
        )

    overall_funnel = _filter_funnel(batch.validations, batch.decisions)
    summary = {
        "version": 1,
        "kind": "latent_search_summary",
        "status": "completed",
        "stage": "latent-search",
        "run_id": run_id,
        "plan_id": plan_id,
        "pilot": dict(manifest.pilot),
        "checkpoint_sha256": generation_config.active_checkpoint.sha256,
        "generation_config_sha256": _file_sha256(generation_config_path),
        "latent_search_config_sha256": _file_sha256(latent_search_config_path),
        "representative_manifest_sha256": manifest.sha256,
        "execution_config_sha256": execution_config_sha256,
        "execution_config": execution_config,
        "filter_semantic_sha256": filter_fingerprint.semantic_sha256,
        "filter_dependency_sha256": dict(filter_fingerprint.file_sha256),
        "filter_contract_version": FILTER_CONTRACT_VERSION,
        "task_arm_count": len(tasks),
        "representative_count": len(search_config.representatives),
        "candidate_count": len(tasks) * search_config.candidate_budget_per_arm,
        "kinematic_passed_count": sum(
            value["kinematic_passed_count"] for value in evidence_by_task.values()
        ),
        "kinematic_pass_rate": sum(
            value["kinematic_passed_count"] for value in evidence_by_task.values()
        )
        / (len(tasks) * search_config.candidate_budget_per_arm),
        "raw_saved_count": sum(shard.candidate_count for shard in raw_shards),
        "raw_policy": "top_k_only",
        "all_candidates_reproducible_from_seed": True,
        "top_k_full_filter_funnel": overall_funnel,
        "accepted_count": index.accepted_count,
        "rejected_count": index.rejected_count,
        "by_task_arm": per_task,
        "paired_latent": paired_latent,
        "stage_execution_counts": dict(batch.stage_execution_counts),
        "stage_elapsed_seconds": dict(batch.stage_elapsed_seconds),
        "timing_seconds": {
            "generation_and_kinematic_search": generation_elapsed,
            "top_k_full_filtering": filtering_elapsed,
            "end_to_end": time.perf_counter() - started,
        },
        "raw_immutable_verified": True,
        "raw_snapshot_sha256": canonical_sha256(
            {
                _path_label(path, repository): identity
                for path, identity in sorted(
                    snapshot_before.items(), key=lambda item: item[0].as_posix()
                )
            }
        ),
        "external_smoothing_applied": False,
        "interpolation_repair_applied": False,
        "filter_threshold_overrides_applied": False,
        "formal_train_only": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "outputs": {
            "task_plan": _path_label(plan_path, repository),
            "raw": _path_label(run_root / "raw", repository),
            "accepted": _path_label(index.accepted_path, repository),
            "rejected": _path_label(index.rejected_path, repository),
            "filter_commit": _path_label(index.commit_path, repository),
        },
    }
    summary_path = write_generation_capability_matrix(filter_root / "summary.json", summary)
    print(
        "stage D latent-search complete: "
        f"{summary['candidate_count']} generated, "
        f"{summary['kinematic_passed_count']} kinematic passes, "
        f"{index.accepted_count} accepted",
        flush=True,
    )
    print(f"latent-search summary: {summary_path}", flush=True)
    return summary


__all__ = ["LATENT_SEARCH_OUTPUT_CONTRACT", "run_latent_search_workflow"]

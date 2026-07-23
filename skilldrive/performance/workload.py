"""Deterministic fixed Pilot workloads with hash-bound raw inputs."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.contracts import (
    GenerationTask,
    candidate_id,
    canonical_json_bytes,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    paired_latent_seed,
    seed_record_id,
    semantic_generation_config_sha256,
)
from skilldrive.generation.scheduler import TaskPlan, load_task_plan
from skilldrive.generation.storage import verify_raw_shard
from skilldrive.performance.config import PerformanceBenchmarkConfig
from skilldrive.seeds import read_seed_records


WORKLOAD_SCHEMA_VERSION = 1
WORKLOAD_KIND = "counterfactual_performance_fixed_workload"
SELECTION_CONTRACT = "deterministic_atomic_condition_pairs_v1"
_TASK_FIELDS = (
    "task_id",
    "task_index",
    "seed_record_id",
    "scenario_id",
    "skill_id",
    "target_track_id",
    "proposal_mode",
    "condition_skill_id",
    "candidate_budget",
    "checkpoint_sha256",
    "semantic_config_sha256",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def generation_task_to_row(task: GenerationTask) -> dict[str, Any]:
    return {name: getattr(task, name) for name in _TASK_FIELDS}


def generation_task_from_row(value: Any) -> GenerationTask:
    if not isinstance(value, Mapping) or set(value) != set(_TASK_FIELDS):
        raise ValueError("performance task row has missing or unknown fields")
    return GenerationTask(status="pending", **dict(value))


def _unit_key(task: GenerationTask) -> tuple[str, str, str, str, str]:
    return (
        task.seed_record_id,
        task.scenario_id,
        task.skill_id,
        task.target_track_id,
        task.proposal_mode,
    )


def select_fixed_tasks(
    tasks: Sequence[GenerationTask],
    *,
    max_tasks: int,
    base_seed: int,
) -> tuple[GenerationTask, ...]:
    """Select at most ``max_tasks`` while retaining learned pairs atomically."""

    if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or not 1 <= max_tasks <= 512:
        raise ValueError("max_tasks must be an integer from 1 to 512")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not tasks:
        raise ValueError("fixed workload requires at least one Pilot task")

    grouped: dict[tuple[str, str, str, str, str], list[GenerationTask]] = {}
    for task in tasks:
        if not isinstance(task, GenerationTask):
            raise TypeError("fixed workload tasks must be GenerationTask values")
        grouped.setdefault(_unit_key(task), []).append(task)

    units: list[tuple[GenerationTask, ...]] = []
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda item: item.task_index))
        proposal_mode = key[-1]
        if proposal_mode == "learned_conditioned_prior":
            expected_conditions = {"<none>", ordered[0].skill_id}
            if len(ordered) != 2 or {item.condition_skill_id for item in ordered} != expected_conditions:
                raise ValueError(
                    "learned Pilot task must have one conditioned/control atomic pair"
                )
        elif proposal_mode == "rule_guided_prior_search":
            if len(ordered) != 1 or ordered[0].condition_skill_id != "<none>":
                raise ValueError("rule-guided Pilot task must be one <none> singleton")
        else:
            raise ValueError(f"unsupported Pilot proposal mode: {proposal_mode}")
        if len({item.candidate_budget for item in ordered}) != 1:
            raise ValueError("atomic task unit has inconsistent candidate budgets")
        units.append(ordered)

    units.sort(
        key=lambda unit: (
            canonical_sha256(
                {
                    "contract": SELECTION_CONTRACT,
                    "base_seed": base_seed,
                    "task_ids": sorted(item.task_id for item in unit),
                }
            ),
            tuple(item.task_id for item in unit),
        )
    )
    selected: list[GenerationTask] = []
    for unit in units:
        if len(selected) + len(unit) <= max_tasks:
            selected.extend(unit)
    if not selected:
        raise ValueError("max_tasks is too small for every available atomic task unit")
    return tuple(sorted(selected, key=lambda item: item.task_index))


def validate_active_pilot_summary(
    summary: Mapping[str, Any],
    *,
    active_checkpoint_sha256: str,
    generation_semantic_sha256: str,
) -> tuple[str, str]:
    expected = {
        "status": "completed",
        "stage": "pilot",
        "checkpoint_sha256": active_checkpoint_sha256,
        "generation_semantic_sha256": generation_semantic_sha256,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("configured Pilot summary differs from active frozen semantics")
    execution = summary.get("generation_execution_sha256")
    if not isinstance(execution, str) or len(execution) != 64:
        raise ValueError("active Pilot summary lacks generation execution identity")
    pilot_filter_semantic = summary.get("filter_semantic_sha256")
    if not isinstance(pilot_filter_semantic, str) or len(pilot_filter_semantic) != 64:
        raise ValueError("active Pilot summary lacks filter semantic identity")
    return execution, pilot_filter_semantic


def validate_active_task_plan(
    plan: TaskPlan,
    *,
    generation_semantic_sha256: str,
    generation_execution_sha256: str,
    active_checkpoint_sha256: str,
) -> None:
    if plan.semantic_config_sha256 != generation_semantic_sha256:
        raise ValueError("Pilot task plan does not match active generation semantics")
    if plan.execution_config_sha256 != generation_execution_sha256:
        raise ValueError("Pilot task plan execution identity changed")
    if any(task.checkpoint_sha256 != active_checkpoint_sha256 for task in plan.tasks):
        raise ValueError("Pilot task plan mixes a non-active checkpoint")


def validate_shared_latent_pairs(
    tasks: Sequence[GenerationTask],
    latent_sequences: Mapping[str, tuple[int, ...]],
) -> None:
    units: dict[tuple[str, str, str, str, str], list[GenerationTask]] = {}
    for task in tasks:
        units.setdefault(_unit_key(task), []).append(task)
    for unit in units.values():
        if unit[0].proposal_mode != "learned_conditioned_prior":
            continue
        if len(unit) != 2:
            raise ValueError("conditioned/control workload pair is incomplete")
        first, second = sorted(unit, key=lambda item: item.condition_skill_id)
        if (
            first.task_id not in latent_sequences
            or second.task_id not in latent_sequences
            or latent_sequences[first.task_id] != latent_sequences[second.task_id]
        ):
            raise ValueError("conditioned/control raw candidates do not share latent seeds")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"benchmark input escapes repository root: {path}") from error


def _bind_file(
    root: Path,
    path: Path,
    bindings: dict[str, str],
    *,
    expected_sha256: str | None = None,
) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"benchmark input file is missing: {resolved}")
    label = _relative(root, resolved)
    actual = file_sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"benchmark input SHA-256 mismatch: {label}")
    previous = bindings.setdefault(label, actual)
    if previous != actual:
        raise ValueError(f"benchmark input identity changed while preparing: {label}")
    return label


def _summary_output_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"active Pilot summary output {name} is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"active Pilot summary output {name} is not repository-relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"active Pilot summary output {name} escapes repository") from error
    return resolved


def prepare_fixed_workload(
    config: PerformanceBenchmarkConfig,
    *,
    config_path: str | Path,
    repository_root: str | Path = ".",
) -> tuple[Path, dict[str, Any]]:
    """Prepare and atomically publish one hash-bound legacy CPU-filter workload."""

    root = Path(repository_root).resolve()
    performance_source = Path(config_path).resolve()
    generation_source = (root / config.inputs.generation_config).resolve()
    filter_source = (root / config.inputs.filter_config).resolve()
    detection_source = (root / config.inputs.detection_config).resolve()
    generation = load_counterfactual_config(generation_source, repository_root=root)
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=root,
        generation_config_path=config.inputs.generation_config,
        filter_config_path=config.inputs.filter_config,
        detection_config_path=config.inputs.detection_config,
    )
    pilot_execution_summary_path = (root / config.inputs.pilot_summary).resolve()
    pilot_execution_summary = _read_json(
        pilot_execution_summary_path,
        "active Pilot execution summary",
    )
    current_generation_semantic = semantic_generation_config_sha256(generation)
    generation_execution_sha256, pilot_filter_semantic_sha256 = (
        validate_active_pilot_summary(
            pilot_execution_summary,
            active_checkpoint_sha256=generation.active_checkpoint.sha256,
            generation_semantic_sha256=current_generation_semantic,
        )
    )
    if pilot_execution_summary_path.parent.name != pilot_filter_semantic_sha256:
        raise ValueError("configured Pilot summary path differs from its filter identity")
    outputs = pilot_execution_summary.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("active Pilot summary outputs are missing")
    plan_path = _summary_output_path(root, outputs.get("task_plan"), "task_plan")
    raw_root = _summary_output_path(root, outputs.get("raw"), "raw")
    pilot_root = plan_path.parent
    if raw_root != pilot_root / "raw":
        raise ValueError("active Pilot summary raw directory differs from task plan root")
    plan_summary_path = pilot_root / "task_plan.summary.json"
    _read_json(plan_summary_path, "Pilot task plan summary")
    loaded = load_task_plan(
        pilot_root,
        expected_semantic_config_sha256=current_generation_semantic,
        current_execution_config_sha256=generation_execution_sha256,
    )
    plan = loaded.plan
    validate_active_task_plan(
        plan,
        generation_semantic_sha256=current_generation_semantic,
        generation_execution_sha256=generation_execution_sha256,
        active_checkpoint_sha256=generation.active_checkpoint.sha256,
    )
    if (
        plan.task_plan_id != pilot_execution_summary.get("task_plan_id")
        or plan.semantic_config_sha256
        != pilot_execution_summary.get("generation_semantic_sha256")
        or plan.execution_config_sha256 != generation_execution_sha256
        or file_sha256(plan_path)
        != pilot_execution_summary.get("task_plan_sha256")
        or file_sha256(plan_summary_path)
        != pilot_execution_summary.get("task_plan_summary_sha256")
    ):
        raise ValueError("active Pilot task plan identity differs from its summary")
    raw_commit_identity = []
    raw_commit_paths = sorted(raw_root.glob("shard-*.commit.json"))
    for commit_path in raw_commit_paths:
        commit_value = _read_json(commit_path, "Pilot raw commit")
        shard_index = commit_value.get("shard_index")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise ValueError("active Pilot raw commit has an invalid shard index")
        raw_commit_identity.append(
            {
                "shard_index": shard_index,
                "commit_sha256": file_sha256(commit_path),
            }
        )
    if (
        len(raw_commit_identity) != len(plan.tasks)
        or canonical_sha256(raw_commit_identity)
        != pilot_execution_summary.get("raw_commit_set_sha256")
    ):
        raise ValueError("active Pilot raw commit set differs from its summary")
    selected = select_fixed_tasks(
        plan.tasks,
        max_tasks=config.workload.max_tasks,
        base_seed=plan.base_seed,
    )

    seed_path = (root / generation.inputs.seed_manifest).resolve()
    records = read_seed_records(seed_path)
    records_by_id = {seed_record_id(record): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")

    bindings: dict[str, str] = {}
    _bind_file(root, performance_source, bindings)
    _bind_file(root, generation_source, bindings)
    _bind_file(root, filter_source, bindings)
    _bind_file(root, detection_source, bindings)
    _bind_file(root, pilot_execution_summary_path, bindings)
    _bind_file(root, plan_path, bindings)
    _bind_file(root, plan_summary_path, bindings)
    output_sha256 = pilot_execution_summary.get("output_sha256")
    if not isinstance(output_sha256, Mapping):
        raise ValueError("active Pilot summary output SHA-256 bindings are missing")
    for name in ("eligibility_audit", "accepted", "rejected", "filter_commit"):
        output_path = _summary_output_path(root, outputs.get(name), name)
        expected = output_sha256.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"active Pilot summary lacks {name} SHA-256")
        _bind_file(root, output_path, bindings, expected_sha256=expected)
    _bind_file(
        root,
        (root / generation.active_checkpoint.path).resolve(),
        bindings,
        expected_sha256=generation.active_checkpoint.sha256,
    )
    _bind_file(
        root,
        (root / generation.active_checkpoint.run_manifest).resolve(),
        bindings,
        expected_sha256=generation.active_checkpoint.run_manifest_sha256,
    )
    if generation.active_checkpoint.promotion_recommendation is not None:
        _bind_file(
            root,
            (root / generation.active_checkpoint.promotion_recommendation).resolve(),
            bindings,
            expected_sha256=generation.active_checkpoint.promotion_recommendation_sha256,
        )
    for relative, expected in (
        (generation.inputs.seed_manifest, generation.inputs.seed_manifest_sha256),
        (
            generation.inputs.training_cache_manifest,
            generation.inputs.training_cache_manifest_sha256,
        ),
        (generation.inputs.leakage_audit, generation.inputs.leakage_audit_sha256),
    ):
        _bind_file(root, (root / relative).resolve(), bindings, expected_sha256=expected)

    for relative, expected in fingerprint.file_sha256.items():
        _bind_file(root, root / relative, bindings, expected_sha256=expected)
    for relative in (
        "skilldrive/performance/__init__.py",
        "skilldrive/performance/config.py",
        "skilldrive/performance/workload.py",
        "skilldrive/performance/benchmark.py",
        "scripts/generation/run_performance_benchmark.py",
    ):
        _bind_file(root, root / relative, bindings)

    task_rows: list[dict[str, Any]] = []
    proposal_counts: Counter[str] = Counter()
    selected_scenarios: set[str] = set()
    selected_candidates = 0
    latent_sequences: dict[str, tuple[int, ...]] = {}
    data_root = (root / generation.inputs.data_root).resolve()
    for task in selected:
        record = records_by_id.get(task.seed_record_id)
        if record is None:
            raise ValueError(f"Pilot task references an unknown seed: {task.task_id}")
        skill_config = generation.skills_by_id.get(task.skill_id)
        if (
            skill_config is None
            or record.scenario_id != task.scenario_id
            or record.skill_id != task.skill_id
            or record.role_track_ids.get(skill_config.primary_generated_role)
            != task.target_track_id
            or skill_config.proposal_mode != task.proposal_mode
        ):
            raise ValueError("selected Pilot task identity differs from its formal seed")
        source_path = (data_root / record.source_path).resolve()
        try:
            source_path.relative_to(data_root)
        except ValueError as error:
            raise ValueError("benchmark source path escapes Formal Train data root") from error
        _bind_file(root, source_path, bindings)
        map_paths = tuple(sorted(source_path.parent.glob("log_map_archive_*.json")))
        if not map_paths:
            raise FileNotFoundError(
                f"benchmark source map is missing beside scenario: {source_path}"
            )
        for map_path in map_paths:
            _bind_file(root, map_path, bindings)

        commit_path = raw_root / f"shard-{task.task_index:05d}.commit.json"
        commit = verify_raw_shard(
            commit_path,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
        )
        if commit.execution_config_sha256 != plan.execution_config_sha256:
            raise ValueError("selected raw shard execution identity differs from Pilot plan")
        if (
            len(commit.references) != task.candidate_budget
            or any(reference.task_id != task.task_id for reference in commit.references)
            or [reference.candidate_index for reference in commit.references]
            != list(range(task.candidate_budget))
        ):
            raise ValueError("selected raw shard does not contain one complete Pilot task")
        for reference in commit.references:
            expected_seed = paired_latent_seed(
                plan.base_seed,
                task,
                reference.candidate_index,
            )
            expected_candidate = candidate_id(
                task_id=task.task_id,
                candidate_index=reference.candidate_index,
                latent_seed=expected_seed,
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=task.semantic_config_sha256,
            )
            if (
                reference.latent_seed != expected_seed
                or reference.candidate_id != expected_candidate
            ):
                raise ValueError("selected raw candidate differs from Pilot latent contract")
        latent_sequences[task.task_id] = tuple(
            reference.latent_seed for reference in commit.references
        )
        for raw_path in (commit.commit_path, commit.arrays_path, commit.metadata_path):
            _bind_file(root, raw_path, bindings)
        task_rows.append(
            {
                "task": generation_task_to_row(task),
                "source_path": record.source_path,
                "raw_commit": _relative(root, commit.commit_path),
            }
        )
        proposal_counts[task.proposal_mode] += 1
        selected_scenarios.add(task.scenario_id)
        selected_candidates += task.candidate_budget

    validate_shared_latent_pairs(selected, latent_sequences)

    value: dict[str, Any] = {
        "schema_version": WORKLOAD_SCHEMA_VERSION,
        "kind": WORKLOAD_KIND,
        "selection_contract": SELECTION_CONTRACT,
        "pilot": {
            "root": _relative(root, pilot_root),
            "summary": _relative(root, pilot_execution_summary_path),
            "summary_sha256": file_sha256(pilot_execution_summary_path),
            "task_plan_id": plan.task_plan_id,
            "task_plan_sha256": file_sha256(plan_path),
            "task_plan_summary_sha256": file_sha256(plan_summary_path),
            "semantic_config_sha256": plan.semantic_config_sha256,
            "execution_config_sha256": plan.execution_config_sha256,
            "base_seed": plan.base_seed,
            "checkpoint_sha256": generation.active_checkpoint.sha256,
            "filter_semantic_sha256": pilot_filter_semantic_sha256,
        },
        "counts": {
            "tasks": len(selected),
            "candidates": selected_candidates,
            "scenarios": len(selected_scenarios),
            "by_proposal_mode": dict(sorted(proposal_counts.items())),
        },
        "maximum_tasks": config.workload.max_tasks,
        "conditioned_control_shared_latents_verified": True,
        "filter_semantic_sha256": fingerprint.semantic_sha256,
        "filter_dependency_sha256": dict(fingerprint.file_sha256),
        "input_sha256": dict(sorted(bindings.items())),
        "tasks": task_rows,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    workload_id = canonical_sha256(value)
    value["workload_id"] = workload_id
    destination = root / config.output_root / "workloads" / workload_id / "workload.json"
    _atomic_write(destination, canonical_json_bytes(value, indent=2))
    loaded_value = load_fixed_workload(destination, repository_root=root)
    if loaded_value != value:
        raise ValueError("written performance workload differs after verification")
    return destination, value


def load_fixed_workload(
    path: str | Path,
    *,
    repository_root: str | Path = ".",
    verify_inputs: bool = True,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    source = Path(path).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError("performance workload path escapes repository root") from error
    value = _read_json(source, "performance fixed workload")
    if value.get("schema_version") != WORKLOAD_SCHEMA_VERSION:
        raise ValueError("performance workload schema version is invalid")
    if value.get("kind") != WORKLOAD_KIND:
        raise ValueError("performance workload kind is invalid")
    if value.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("performance workload selection contract is invalid")
    if value.get("conditioned_control_shared_latents_verified") is not True:
        raise ValueError("performance workload lacks shared-latent pair verification")
    if value.get("validation_manifests_opened") is not False:
        raise ValueError("performance workload opened validation manifests")
    if value.get("final_validation_accessed") is not False:
        raise ValueError("performance workload accessed Final Validation")
    expected_id = value.get("workload_id")
    identity_value = dict(value)
    identity_value.pop("workload_id", None)
    if expected_id != canonical_sha256(identity_value):
        raise ValueError("performance workload ID differs from its content")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("performance workload tasks are missing")
    materialized = []
    for entry in tasks:
        if not isinstance(entry, Mapping) or set(entry) != {
            "task",
            "source_path",
            "raw_commit",
        }:
            raise ValueError("performance workload task entry is invalid")
        for name in ("source_path", "raw_commit"):
            path_value = entry[name]
            if (
                not isinstance(path_value, str)
                or not path_value
                or Path(path_value).is_absolute()
                or ".." in Path(path_value).parts
            ):
                raise ValueError(f"performance workload {name} is invalid")
        materialized.append(generation_task_from_row(entry["task"]))
    counts = value.get("counts")
    if (
        not isinstance(counts, Mapping)
        or counts.get("tasks") != len(materialized)
        or counts.get("candidates")
        != sum(task.candidate_budget for task in materialized)
    ):
        raise ValueError("performance workload counts differ from task rows")
    if verify_inputs:
        bindings = value.get("input_sha256")
        if not isinstance(bindings, Mapping) or not bindings:
            raise ValueError("performance workload input SHA-256 bindings are missing")
        for relative, expected in sorted(bindings.items()):
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise ValueError("performance workload input binding is invalid")
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("performance workload input path is invalid")
            if file_sha256(root / relative) != expected:
                raise ValueError(f"performance workload input changed: {relative}")
    return value


__all__ = [
    "SELECTION_CONTRACT",
    "WORKLOAD_KIND",
    "WORKLOAD_SCHEMA_VERSION",
    "file_sha256",
    "generation_task_from_row",
    "generation_task_to_row",
    "load_fixed_workload",
    "prepare_fixed_workload",
    "select_fixed_tasks",
    "validate_active_pilot_summary",
    "validate_active_task_plan",
    "validate_shared_latent_pairs",
]

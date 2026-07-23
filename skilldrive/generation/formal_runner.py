"""Resumable formal Prior generation and skill-partitioned filtering."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from skilldrive.data import build_cvae_schema, tensorize_prior_context
from skilldrive.data.av2_reader import load_av2_history_scenario
from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
from skilldrive.generation.assembly import local_futures_to_global
from skilldrive.generation.config import load_counterfactual_config, load_filter_config
from skilldrive.generation.contracts import GeneratedCandidate, GeneratedOverlay, canonical_json_bytes, canonical_sha256
from skilldrive.generation.formal import (
    FormalPlanBindings,
    FormalTaskPlan,
    build_formal_task_plan,
    load_formal_task_plan,
    write_formal_task_plan,
)
from skilldrive.generation.formal_state import (
    FormalTaskState,
    build_formal_filter_references,
    commit_formal_state_shards,
    load_formal_state,
    open_formal_state,
    write_formal_candidate_invalid,
)
from skilldrive.generation.formal_storage import write_formal_filter_indexes
from skilldrive.generation.inference import generate_prior_batch, load_configured_cvae
from skilldrive.generation.planning import (
    latent_group_id,
    latent_seeds_for_task,
    prior_context_fingerprint,
    prior_context_spec_for_task,
    seed_record_id,
    semantic_generation_config_sha256,
)
from skilldrive.generation.storage import scan_raw_shards, verify_raw_shard
from skilldrive.performance.parallel_filter import run_parallel_filter_workload
from skilldrive.seeds import read_seed_records
from skilldrive.skills.detection import load_detection_config


DEFAULT_FORMAL_EXECUTION_CONFIG = Path("configs/generation/formal_execution_v1.yaml")
DEFAULT_FORMAL_OUTPUT_ROOT = Path("outputs/generation/counterfactual_v1/formal")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _read_yaml(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"formal execution config must be a mapping: {path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(value, indent=2))
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    seconds = max(0.0, seconds)
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def _execution_values(root: Path, path: Path) -> dict[str, Any]:
    value = _read_yaml(path)
    expected = {"version", "contract_name", "inputs", "execution", "output"}
    if set(value) != expected:
        raise ValueError("formal execution config has missing or unknown fields")
    execution = value["execution"]
    if not isinstance(execution, Mapping):
        raise ValueError("formal execution config.execution must be a mapping")
    required = {
        "device",
        "task_batch_size",
        "filter_workers",
        "map_batch_size",
        "tasks_per_shard",
        "progress_interval_seconds",
        "use_bfloat16",
    }
    if set(execution) != required:
        raise ValueError("formal execution config.execution fields differ")
    output = value["output"]
    if not isinstance(output, Mapping) or set(output) != {"root"}:
        raise ValueError("formal execution config.output is invalid")
    inputs = value["inputs"]
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "generation_config",
        "filter_config",
        "detection_config",
        "performance_config",
    }:
        raise ValueError("formal execution config.inputs is invalid")
    numbers = ("task_batch_size", "filter_workers", "map_batch_size", "tasks_per_shard")
    for name in numbers:
        item = execution[name]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"formal execution {name} must be a positive integer")
    if execution["map_batch_size"] not in (8, 16, 32):
        raise ValueError("formal execution map_batch_size must be 8, 16, or 32")
    return {
        "version": value["version"],
        "contract_name": value["contract_name"],
        "inputs": {name: str(inputs[name]) for name in inputs},
        "execution": dict(execution),
        "output": {"root": str(output["root"])},
        "config_sha256": _sha256(path),
    }


def _generation_sources(root: Path) -> tuple[Path, ...]:
    return tuple(
        root / relative
        for relative in (
            "skilldrive/generation/assembly.py",
            "skilldrive/generation/config.py",
            "skilldrive/generation/contracts.py",
            "skilldrive/generation/formal.py",
            "skilldrive/generation/formal_runner.py",
            "skilldrive/generation/formal_state.py",
            "skilldrive/generation/formal_storage.py",
            "skilldrive/generation/inference.py",
            "skilldrive/generation/planning.py",
            "skilldrive/generation/storage.py",
            "skilldrive/data/av2_reader.py",
            "skilldrive/data/coordinates.py",
            "skilldrive/data/cvae_samples.py",
            "skilldrive/models/conditional_cvae.py",
            "skilldrive/training/checkpoint.py",
        )
    )


def _validate_formal_records(records: Sequence[Any], config: Any) -> None:
    if len(records) != 33_914:
        raise ValueError(f"formal seed manifest must contain 33914 rows, got {len(records)}")
    ids = {seed_record_id(record) for record in records}
    if len(ids) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")
    scenarios = {record.scenario_id for record in records}
    if len(scenarios) != 5_000:
        raise ValueError(f"formal seed manifest must cover 5000 scenarios, got {len(scenarios)}")
    if {record.skill_id for record in records} != set(config.formal_skill_ids):
        raise ValueError("formal seed manifest skill set differs from formal catalog")
    if any(not str(record.source_path).replace("\\", "/").startswith("train/") for record in records):
        raise ValueError("formal generation source paths must all belong to Formal Train")


def prepare_formal_run(
    *,
    repository_root: str | Path = ".",
    execution_config_path: str | Path = DEFAULT_FORMAL_EXECUTION_CONFIG,
) -> tuple[Path, FormalTaskPlan]:
    root = Path(repository_root).resolve()
    execution_path = _resolved(root, execution_config_path)
    values = _execution_values(root, execution_path)
    inputs = values["inputs"]
    generation_path = _resolved(root, inputs["generation_config"])
    filter_path = _resolved(root, inputs["filter_config"])
    detection_path = _resolved(root, inputs["detection_config"])
    performance_path = _resolved(root, inputs["performance_config"])
    config = load_counterfactual_config(generation_path, repository_root=root)
    records = tuple(read_seed_records(root / config.inputs.seed_manifest))
    _validate_formal_records(records, config)
    fingerprint = build_filter_semantic_fingerprint(
        repository_root=root,
        generation_config_path=generation_path,
        filter_config_path=filter_path,
        detection_config_path=detection_path,
    )
    execution = values["execution"]
    execution_contract = {
        "version": 1,
        "device": execution["device"],
        "task_batch_size": execution["task_batch_size"],
        "filter_workers": execution["filter_workers"],
        "map_batch_size": execution["map_batch_size"],
        "tasks_per_shard": execution["tasks_per_shard"],
        "use_bfloat16": execution["use_bfloat16"],
        "filter_partition": "skill_id",
        "resume_mode": "auto",
        "bev_rendering": "excluded",
    }
    filter_sources = tuple(root / relative for relative in fingerprint.file_sha256)
    bindings = FormalPlanBindings.from_generation_config(
        config,
        repository_root=root,
        generation_config_path=generation_path,
        filter_config_path=filter_path,
        performance_config_path=performance_path,
        detection_config_path=detection_path,
        filter_additional_paths=(),
        generation_source_paths=_generation_sources(root),
        filter_source_paths=filter_sources,
        execution_config=execution_contract,
        tasks_per_shard=execution["tasks_per_shard"],
    )
    plan = build_formal_task_plan(records, config, bindings=bindings)
    output_root = _resolved(root, values["output"]["root"])
    run_root = output_root / plan.formal_plan_id
    artifacts = write_formal_task_plan(run_root, plan, config=config)
    open_formal_state(
        run_root,
        plan=plan,
        task_plan_sha256=artifacts.task_plan_sha256,
    )
    _write_json(
        run_root / "execution_contract.json",
        {
            "kind": "formal_execution_contract",
            "execution_config": values,
            "execution": execution_contract,
            "formal_plan_id": plan.formal_plan_id,
            "task_plan_sha256": artifacts.task_plan_sha256,
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
        },
    )
    return run_root, plan


def _task_rows(
    plan: FormalTaskPlan,
    records_by_id: Mapping[str, Any],
    root: Path,
    tasks: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for task in plan.tasks if tasks is None else tasks:
        record = records_by_id[task.seed_record_id]
        raw_path = root / "raw" / f"shard-{task.shard_index:05d}.commit.json"
        rows.append(
            {
                "task": {
                    name: getattr(task.as_generation_task(), name)
                    for name in (
                        "task_id", "task_index", "seed_record_id", "scenario_id",
                        "skill_id", "target_track_id", "proposal_mode",
                        "condition_skill_id", "candidate_budget", "checkpoint_sha256",
                        "semantic_config_sha256",
                    )
                },
                "source_path": record.source_path,
                "raw_commit": str(raw_path),
            }
        )
    return rows


def _write_progress(
    run_root: Path,
    *,
    stage: str,
    completed: int,
    total: int,
    candidates: int,
    candidate_total: int,
    started: float,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate = candidates / elapsed
    remaining_candidates = max(0, candidate_total - candidates)
    _write_json(
        run_root / "progress.json",
        {
            "kind": "formal_runtime_progress",
            "stage": stage,
            "completed_tasks": completed,
            "total_tasks": total,
            "completed_candidates": candidates,
            "total_candidates": candidate_total,
            "elapsed_seconds": elapsed,
            "candidates_per_second": rate,
            "eta_seconds": None if rate <= 0 else remaining_candidates / rate,
            "updated_at_utc": _utc_now(),
        },
    )
    print(
        f"formal {stage}: {completed}/{total} tasks, {candidates} candidates, "
        f"{rate:.1f} candidates/s, ETA {_eta(None if rate <= 0 else remaining_candidates / rate)}",
        flush=True,
    )


def _generate(
    *,
    root: Path,
    run_root: Path,
    plan: FormalTaskPlan,
    config: Any,
    records_by_id: Mapping[str, Any],
    execution: Mapping[str, Any],
    state_tasks: Sequence[FormalTaskState],
    bindings: Any,
) -> None:
    raw_dir = run_root / "raw"
    completed_ids = {
        state.task_id
        for state in state_tasks
        if state.status in ("generated", "accepted", "rejected")
    }
    if len(completed_ids) == len(plan.tasks):
        print("formal generation: all tasks are durable; skipped", flush=True)
        return
    scan = scan_raw_shards(raw_dir, expected_semantic_config_sha256=plan.bindings.semantic_config_sha256)
    durable = {
        shard.shard_index
        for shard in scan.valid_shards
        if shard.execution_config_sha256 == plan.bindings.generation_execution_sha256
    }
    pending = [
        task
        for task in plan.tasks
        if task.shard_index not in durable and task.task_id not in completed_ids
    ]
    if not pending:
        print("formal generation: all raw shards are durable; skipped", flush=True)
        return
    schema = build_cvae_schema(_resolved(root, config.formal_catalog).parent)
    runtime = load_configured_cvae(
        active_checkpoint=config.active_checkpoint,
        schema=schema,
        device=execution["device"],
        repository_root=root,
    )
    # Grouping by scenario keeps each AV2 parquet/map pair hot while raw shards
    # remain task-indexed and independently resumable.
    pending.sort(key=lambda task: (task.scenario_id, task.task_index))
    started = time.perf_counter()
    generated = 0
    last_report = started
    batch_size = execution["task_batch_size"]
    for offset in range(0, len(pending), batch_size):
        history_cache: dict[str, Any] = {}
        context_cache: dict[str, Any] = {}
        batch_tasks = pending[offset : offset + batch_size]
        contexts = []
        latent_rows = []
        batch_records = []
        valid_tasks = []
        for task in batch_tasks:
            record = records_by_id[task.seed_record_id]
            source = str((_resolved(root, config.inputs.data_root) / record.source_path).resolve())
            try:
                history = history_cache.get(source)
                if history is None:
                    history = load_av2_history_scenario(source)
                    history_cache[source] = history
                fingerprint = prior_context_fingerprint(task, record)
                context = context_cache.get(fingerprint)
                if context is None:
                    context = tensorize_prior_context(
                        history,
                        prior_context_spec_for_task(task, record),
                        schema,
                    )
                    context_cache[fingerprint] = context
            except ValueError as error:
                for candidate_index in range(task.candidate_budget):
                    write_formal_candidate_invalid(
                        run_root,
                        plan=plan,
                        bindings=bindings,
                        task=task,
                        candidate_index=candidate_index,
                        reason_code="prior_context_invalid",
                        message=str(error),
                    )
                generated += task.candidate_budget
                continue
            contexts.append(context)
            latent_rows.append(latent_seeds_for_task(task, base_seed=plan.bindings.base_seed))
            batch_records.append(record)
            valid_tasks.append(task)
        if contexts:
            latent_matrix = np.stack(latent_rows)
            generated_batch = generate_prior_batch(
                runtime,
                contexts,
                latent_matrix,
                use_bfloat16=bool(execution["use_bfloat16"]),
            )
        for index, (task, record, context) in enumerate(zip(valid_tasks, batch_records, contexts, strict=True)):
            futures = local_futures_to_global(
                generated_batch.future_position_local[index],
                context.anchor_origin_global,
                float(context.anchor_heading_global),
            )
            skill_config = config.skills_by_id[task.skill_id]
            candidates = tuple(
                GeneratedCandidate(
                    task_id=task.task_id,
                    candidate_index=candidate_index,
                    latent_seed=int(latent_matrix[index, candidate_index]),
                    scenario_id=task.scenario_id,
                    skill_id=task.skill_id,
                    proposal_mode=task.proposal_mode,
                    checkpoint_sha256=task.checkpoint_sha256,
                    semantic_config_sha256=task.semantic_config_sha256,
                    overlay=GeneratedOverlay(
                        target_track_id=task.target_track_id,
                        future_xy_global=futures[candidate_index],
                    ),
                    metadata={
                        "condition_skill_id": task.condition_skill_id,
                        "latent_group_id": latent_group_id(task),
                        "primary_generated_role": skill_config.primary_generated_role,
                        "requested_parameters": record.sampled_parameters,
                        "detection_mode": record.evidence["detection_mode"],
                    },
                )
                for candidate_index in range(task.candidate_budget)
            )
            from skilldrive.generation.storage import write_raw_shard

            write_raw_shard(
                raw_dir,
                task.shard_index,
                candidates,
                semantic_config_sha256=plan.bindings.semantic_config_sha256,
                execution_config_sha256=plan.bindings.generation_execution_sha256,
            )
            generated += len(candidates)
        history_cache.clear()
        context_cache.clear()
        now = time.perf_counter()
        completed = min(offset + len(batch_tasks), len(pending))
        if now - last_report >= float(execution["progress_interval_seconds"]) or completed == len(pending):
            _write_progress(
                run_root,
                stage="generation",
                completed=completed,
                total=len(pending),
                candidates=generated,
                candidate_total=len(pending) * plan.bindings.candidate_budget,
                started=started,
            )
            last_report = now
def _filter_skill(
    *,
    root: Path,
    run_root: Path,
    plan: FormalTaskPlan,
    config: Any,
    records_by_id: Mapping[str, Any],
    bindings: Any,
    skill_id: str,
    execution: Mapping[str, Any],
    filter_config: Any,
    detection_config: Any,
    state_tasks: list[FormalTaskState],
) -> tuple[FormalTaskState, ...]:
    task_subset = tuple(task for task in plan.tasks if task.skill_id == skill_id)
    raw_tasks = tuple(
        task for task in task_subset if state_tasks[task.task_index].raw is not None
    )
    invalid_tasks = tuple(task for task in task_subset if task not in raw_tasks)
    filter_dir = run_root / "filter" / skill_id
    commit_path = filter_dir / "filter-index.commit.json"
    raw_by_task = {
        task.task_id: state_tasks[task.task_index].raw for task in raw_tasks
    }
    updates_by_shard: dict[int, list[FormalTaskState]] = defaultdict(list)
    for task in invalid_tasks:
        old = state_tasks[task.task_index]
        if old.invalid_candidates:
            updates_by_shard[task.shard_index].append(
                FormalTaskState.rejected(task, invalid_candidates=old.invalid_candidates)
            )
    if not raw_tasks:
        if updates_by_shard:
            commit_formal_state_shards(
                run_root,
                plan=plan,
                bindings=bindings,
                shard_states={index: tuple(items) for index, items in updates_by_shard.items()},
            )
            for items in updates_by_shard.values():
                for item in items:
                    state_tasks[item.task_index] = item
        print(f"formal filtering {skill_id}: all tasks invalid at Prior input", flush=True)
        return tuple(state_tasks)
    if commit_path.is_file():
        try:
            refs = build_formal_filter_references(
                commit_path,
                artifact_root=run_root,
                plan=plan,
                bindings=bindings,
                raw_by_task=raw_by_task,
            )
        except ValueError:
            commit_path.unlink(missing_ok=True)
        else:
            if set(refs) == {task.task_id for task in raw_tasks}:
                if all(
                    state_tasks[task.task_index].status in ("accepted", "rejected")
                    for task in raw_tasks
                ):
                    print(f"formal filtering {skill_id}: durable commit verified; skipped", flush=True)
                    return tuple(state_tasks)

    raw_shards = tuple(
        verify_raw_shard(
            run_root / "raw" / f"shard-{task.shard_index:05d}.commit.json",
            expected_semantic_config_sha256=plan.bindings.semantic_config_sha256,
        )
        for task in raw_tasks
    )
    workload = {
        "filter_semantic_sha256": bindings.filter_config_sha256,
        "counts": {
            "tasks": len(raw_tasks),
            "candidates": sum(task.candidate_budget for task in raw_tasks),
            "scenarios": len({task.scenario_id for task in raw_tasks}),
        },
        "tasks": _task_rows(plan, records_by_id, run_root, raw_tasks),
    }
    result = run_parallel_filter_workload(
        workload,
        repository_root=root,
        generation_config=config,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=execution["filter_workers"],
        map_batch_size=execution["map_batch_size"],
    )
    commit_path = write_formal_filter_indexes(
        filter_dir,
        raw_shards,
        result.batch.decisions,
        artifact_root=run_root,
        bindings=bindings,
    )
    refs = build_formal_filter_references(
        commit_path,
        artifact_root=run_root,
        plan=plan,
        bindings=bindings,
        raw_by_task=raw_by_task,
    )
    for task in raw_tasks:
        reference = refs[task.task_id]
        old = state_tasks[task.task_index]
        new = FormalTaskState.accepted(task, old.raw, reference) if reference.accepted_count else FormalTaskState.rejected(task, old.raw, reference)
        updates_by_shard[task.shard_index].append(new)
    updates: dict[int, tuple[FormalTaskState, ...]] = {}
    for shard_index, updated in updates_by_shard.items():
        start = shard_index * plan.bindings.tasks_per_shard
        current = list(state_tasks[start : start + plan.bindings.tasks_per_shard])
        replacements = {item.task_id: item for item in updated}
        updates[shard_index] = tuple(
            replacements.get(item.task_id, item) for item in current
        )
    commit_formal_state_shards(
        run_root,
        plan=plan,
        bindings=bindings,
        shard_states=updates,
    )
    for updated in updates.values():
        for item in updated:
            state_tasks[item.task_index] = item
    print(
        f"formal filtering {skill_id}: "
        f"{sum(decision.accepted for decision in result.batch.decisions)} accepted / "
        f"{len(result.batch.decisions)} candidates, {result.timings['total_seconds']:.1f}s",
        flush=True,
    )
    return tuple(state_tasks)


def run_formal(
    *,
    repository_root: str | Path = ".",
    execution_config_path: str | Path = DEFAULT_FORMAL_EXECUTION_CONFIG,
) -> Path:
    root = Path(repository_root).resolve()
    values = _execution_values(root, _resolved(root, execution_config_path))
    run_root, plan = prepare_formal_run(
        repository_root=root,
        execution_config_path=execution_config_path,
    )
    config = load_counterfactual_config(
        _resolved(root, values["inputs"]["generation_config"]),
        repository_root=root,
    )
    records = tuple(read_seed_records(root / config.inputs.seed_manifest))
    records_by_id = {seed_record_id(record): record for record in records}
    filter_config = load_filter_config(
        _resolved(root, values["inputs"]["filter_config"])
    )
    detection_config = load_detection_config(
        _resolved(root, values["inputs"]["detection_config"])
    )
    started = time.perf_counter()
    state = open_formal_state(
        run_root,
        plan=plan,
        task_plan_sha256=_sha256(run_root / "formal_task_plan.jsonl"),
    )
    _generate(
        root=root,
        run_root=run_root,
        plan=plan,
        config=config,
        records_by_id=records_by_id,
        execution=values["execution"],
        state_tasks=state.task_states,
        bindings=state.bindings,
    )
    state = open_formal_state(run_root, plan=plan, task_plan_sha256=_sha256(run_root / "formal_task_plan.jsonl"))
    state_tasks = list(state.task_states)
    if any(item.status not in ("generated", "accepted", "rejected") for item in state_tasks):
        raise RuntimeError("formal generation ended with non-durable tasks")
    filter_started = time.perf_counter()
    filter_candidate_total = sum(
        item.raw.candidate_count for item in state_tasks if item.raw is not None
    )
    for skill_id in config.formal_skill_ids:
        bindings = state.bindings
        state_tasks = list(_filter_skill(
            root=root,
            run_root=run_root,
            plan=plan,
            config=config,
            records_by_id=records_by_id,
            bindings=bindings,
            skill_id=skill_id,
            execution=values["execution"],
            filter_config=filter_config,
            detection_config=detection_config,
            state_tasks=state_tasks,
        ))
        _write_progress(
            run_root,
            stage=f"filter:{skill_id}",
            completed=sum(item.status in ("accepted", "rejected") for item in state_tasks),
            total=len(state_tasks),
            candidates=sum(
                item.filter.candidate_count for item in state_tasks if item.filter is not None
            ),
            candidate_total=filter_candidate_total,
            started=filter_started,
        )
    final = load_formal_state(run_root, plan=plan, task_plan_sha256=_sha256(run_root / "formal_task_plan.jsonl"))
    if any(item.status not in ("accepted", "rejected") for item in final.task_states):
        raise RuntimeError("formal filtering did not finish every task")
    _write_json(
        run_root / "summary.json",
        {
            "kind": "formal_counterfactual_summary",
            "status": "completed",
            "formal_plan_id": plan.formal_plan_id,
            "task_count": len(plan.tasks),
            "candidate_count": plan.total_candidates,
            "accepted_count": sum(item.filter.accepted_count for item in final.task_states if item.filter),
            "rejected_count": sum(item.filter.rejected_count for item in final.task_states if item.filter),
            "elapsed_seconds": time.perf_counter() - started,
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
            "bev_rendering": "not_run",
        },
    )
    print(f"formal run complete: {run_root}", flush=True)
    return run_root


__all__ = ["prepare_formal_run", "run_formal"]

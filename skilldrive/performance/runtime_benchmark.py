"""Fixed GPU-generation and end-to-end counterfactual benchmarks."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from skilldrive.filtering.pipeline import DEFAULT_MAP_BATCH_SIZE, MAP_BATCH_SIZES
from skilldrive.generation.config import (
    CounterfactualGenerationConfig,
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import (
    GeneratedCandidate,
    GeneratedOverlay,
    GenerationTask,
    canonical_json_bytes,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    latent_group_id,
    paired_latent_seeds_for_task,
    pilot_evaluation_arm,
    prior_context_fingerprint,
    prior_context_spec_for_task,
    seed_record_id,
)
from skilldrive.generation.storage import write_filter_indexes, write_raw_shard
from skilldrive.performance.config import PerformanceBenchmarkConfig
from skilldrive.performance.workload import (
    file_sha256,
    generation_task_from_row,
    generation_task_to_row,
    load_fixed_workload,
)
from skilldrive.seeds import read_seed_records
from skilldrive.skills.detection import load_detection_config


SCHEMA_VERSION = 1
GPU_RUNNER = "gpu_generation_fixed_v1"
E2E_RUNNER = "end_to_end_fixed_v1"
TASKS = 512
CANDIDATES_PER_TASK = 16
CANDIDATES = TASKS * CANDIDATES_PER_TASK
REPEATS = 3


@dataclass(frozen=True)
class FixedInputs:
    workload: Mapping[str, Any]
    generation: CounterfactualGenerationConfig
    tasks: tuple[GenerationTask, ...]
    records: tuple[Any, ...]
    source_paths: tuple[Path, ...]
    source_values: tuple[str, ...]
    latent_seeds: tuple[np.ndarray, ...]
    task_order_sha256: str
    latent_seed_sha256: str


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _positive(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_or_none(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "expected_semantic_decision_sha256 must be a lowercase SHA-256 digest"
        )
    return value


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise ValueError("benchmark distribution must be finite and non-empty")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "range": ordered[-1] - ordered[0],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _load_inputs(
    config: PerformanceBenchmarkConfig,
    *,
    workload_path: Path,
    root: Path,
) -> FixedInputs:
    workload = load_fixed_workload(workload_path, repository_root=root)
    counts = workload["counts"]
    if counts["tasks"] != TASKS or counts["candidates"] != CANDIDATES:
        raise ValueError("runtime benchmarks require the fixed 512-task workload")
    generation = load_counterfactual_config(
        root / config.inputs.generation_config,
        repository_root=root,
    )
    pilot = workload["pilot"]
    if pilot["checkpoint_sha256"] != generation.active_checkpoint.sha256:
        raise ValueError("fixed workload checkpoint is no longer active")
    records = read_seed_records(root / generation.inputs.seed_manifest)
    by_id = {seed_record_id(record): record for record in records}
    base_seed = int(pilot["base_seed"])
    tasks: list[GenerationTask] = []
    selected_records: list[Any] = []
    sources: list[Path] = []
    source_values: list[str] = []
    latent_rows: list[np.ndarray] = []
    for entry in workload["tasks"]:
        task = generation_task_from_row(entry["task"])
        record = by_id[task.seed_record_id]
        source_value = str(entry["source_path"])
        if record.source_path != source_value or task.candidate_budget != CANDIDATES_PER_TASK:
            raise ValueError("fixed workload task no longer matches its seed")
        tasks.append(task)
        selected_records.append(record)
        sources.append((root / generation.inputs.data_root / source_value).resolve())
        source_values.append(source_value)
        latent_rows.append(
            np.asarray(
                paired_latent_seeds_for_task(task, base_seed=base_seed),
                dtype=np.int64,
            )
        )
    task_ids = [task.task_id for task in tasks]
    latent_payload = [
        {"task_id": task.task_id, "seeds": values.tolist()}
        for task, values in zip(tasks, latent_rows)
    ]
    return FixedInputs(
        workload=workload,
        generation=generation,
        tasks=tuple(tasks),
        records=tuple(selected_records),
        source_paths=tuple(sources),
        source_values=tuple(source_values),
        latent_seeds=tuple(latent_rows),
        task_order_sha256=canonical_sha256(task_ids),
        latent_seed_sha256=canonical_sha256(latent_payload),
    )


def _prepare_contexts(
    inputs: FixedInputs,
    schema: Any,
    *,
    limit: int | None = None,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    from skilldrive.data import tensorize_prior_context
    from skilldrive.data.av2_reader import load_av2_history_scenario

    started = time.perf_counter()
    scenarios: dict[Path, Any] = {}
    cache: dict[str, Any] = {}
    contexts: list[Any] = []
    count = len(inputs.tasks) if limit is None else min(limit, len(inputs.tasks))
    for task, record, source in zip(
        inputs.tasks[:count],
        inputs.records[:count],
        inputs.source_paths[:count],
    ):
        fingerprint = prior_context_fingerprint(task, record)
        context = cache.get(fingerprint)
        if context is None:
            scenario = scenarios.get(source)
            if scenario is None:
                scenario = load_av2_history_scenario(source)
                if scenario.metadata.get("temporal_scope") != "history_only":
                    raise ValueError("runtime benchmark generation read future frames")
                scenarios[source] = scenario
            context = tensorize_prior_context(
                scenario,
                prior_context_spec_for_task(task, record),
                schema,
            )
            cache[fingerprint] = context
        contexts.append(context)
    return tuple(contexts), {
        "elapsed_seconds": time.perf_counter() - started,
        "scenario_load_count": len(scenarios),
        "unique_context_count": len(cache),
    }


def _load_runtime(inputs: FixedInputs, schema: Any, *, device: str, root: Path):
    from skilldrive.generation.inference import load_configured_cvae

    started = time.perf_counter()
    runtime = load_configured_cvae(
        active_checkpoint=inputs.generation.active_checkpoint,
        schema=schema,
        device=device,
        repository_root=root,
    )
    if runtime.device.type != "cuda":
        raise ValueError("runtime benchmarks require a CUDA device")
    return runtime, time.perf_counter() - started


def _environment(runtime: Any) -> dict[str, Any]:
    import torch

    index = runtime.device.index
    if index is None:
        index = torch.cuda.current_device()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(runtime.device),
        "gpu_name": torch.cuda.get_device_name(index),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
    }


def _gpu_batch(
    runtime: Any,
    contexts: Sequence[Any],
    seeds: np.ndarray,
    *,
    use_bfloat16: bool,
    copy_output: bool,
) -> dict[str, Any]:
    """Measure model kernels only; input preparation and D2H stay separate."""

    import torch

    from skilldrive.generation.inference import stack_prior_contexts, standard_normal_from_seeds

    wall_started = time.perf_counter()
    preparation_started = time.perf_counter()
    batch = stack_prior_contexts(contexts, device=runtime.device)
    noise = torch.as_tensor(
        standard_normal_from_seeds(seeds, latent_dim=runtime.model.latent_dim),
        device=runtime.device,
    )
    torch.cuda.synchronize(runtime.device)
    preparation_seconds = time.perf_counter() - preparation_started
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_bfloat16
    ):
        output = runtime.model.sample_prior_from_noise(batch, noise)
    end.record()
    end.synchronize()
    transfer_started = time.perf_counter()
    future = output.future_position_local.float().cpu().numpy() if copy_output else None
    transfer_seconds = time.perf_counter() - transfer_started
    return {
        "future": future,
        "gpu_seconds": float(start.elapsed_time(end)) / 1000.0,
        "preparation_seconds": preparation_seconds,
        "transfer_seconds": transfer_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
    }


def _warmup(
    runtime: Any,
    contexts: Sequence[Any],
    inputs: FixedInputs,
    *,
    batch_size: int,
    iterations: int,
    use_bfloat16: bool,
) -> dict[str, Any]:
    size = min(batch_size, TASKS)
    seeds = np.stack(inputs.latent_seeds[:size])
    started = time.perf_counter()
    gpu_seconds = 0.0
    for _ in range(iterations):
        gpu_seconds += _gpu_batch(
            runtime,
            contexts[:size],
            seeds,
            use_bfloat16=use_bfloat16,
            copy_output=False,
        )["gpu_seconds"]
    return {
        "iterations": iterations,
        "tasks_per_iteration": size,
        "gpu_seconds": gpu_seconds,
        "wall_seconds": time.perf_counter() - started,
        "included_in_repeats": False,
    }


def _gpu_pass(
    runtime: Any,
    contexts: Sequence[Any],
    inputs: FixedInputs,
    *,
    repeat_index: int,
    batch_size: int,
    use_bfloat16: bool,
) -> dict[str, Any]:
    import torch

    torch.cuda.reset_peak_memory_stats(runtime.device)
    digest = hashlib.sha256()
    totals = {"gpu_seconds": 0.0, "preparation_seconds": 0.0, "transfer_seconds": 0.0}
    started = time.perf_counter()
    for offset in range(0, TASKS, batch_size):
        size = min(batch_size, TASKS - offset)
        result = _gpu_batch(
            runtime,
            contexts[offset : offset + size],
            np.stack(inputs.latent_seeds[offset : offset + size]),
            use_bfloat16=use_bfloat16,
            copy_output=True,
        )
        future = result["future"]
        if future.shape != (size, CANDIDATES_PER_TASK, 60, 2) or not np.isfinite(future).all():
            raise ValueError("GPU benchmark produced invalid trajectories")
        digest.update(np.ascontiguousarray(future, dtype=np.float32).tobytes())
        for name in totals:
            totals[name] += float(result[name])
    wall = time.perf_counter() - started
    return {
        "repeat_index": repeat_index,
        "task_count": TASKS,
        "candidate_count": CANDIDATES,
        **totals,
        "measured_wall_seconds": wall,
        "tasks_per_gpu_second": TASKS / totals["gpu_seconds"],
        "candidates_per_gpu_second": CANDIDATES / totals["gpu_seconds"],
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device)),
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime.device)),
        "output_sha256": digest.hexdigest(),
    }


def _sources(root: Path, *, e2e: bool) -> dict[str, str]:
    paths = [
        "skilldrive/performance/runtime_benchmark.py",
        "scripts/generation/run_runtime_benchmark.py",
        "skilldrive/data/cvae_samples.py",
        "skilldrive/generation/inference.py",
        "skilldrive/models/conditional_cvae.py",
        "skilldrive/training/checkpoint.py",
    ]
    if e2e:
        paths.append("skilldrive/performance/parallel_filter.py")
    return {path: file_sha256(root / path) for path in paths}


def _contract(
    runner: str,
    inputs: FixedInputs,
    config: PerformanceBenchmarkConfig,
    *,
    config_path: Path,
    workload_path: Path,
    environment: Mapping[str, Any],
    source_sha256: Mapping[str, str],
    options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": runner,
        "workload_id": inputs.workload["workload_id"],
        "workload_sha256": file_sha256(workload_path),
        "workload_input_manifest_sha256": canonical_sha256(inputs.workload["input_sha256"]),
        "performance_config_sha256": file_sha256(config_path),
        "checkpoint_sha256": inputs.generation.active_checkpoint.sha256,
        "task_order_sha256": inputs.task_order_sha256,
        "latent_seed_sha256": inputs.latent_seed_sha256,
        "task_count": TASKS,
        "candidate_count": CANDIDATES,
        "repeats": REPEATS,
        "formal_candidate_count": config.benchmark.formal_candidate_count,
        "environment": dict(environment),
        "source_sha256": dict(source_sha256),
        **dict(options),
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }


def _result_root(
    root: Path,
    config: PerformanceBenchmarkConfig,
    inputs: FixedInputs,
    runner: str,
    benchmark_id: str,
) -> Path:
    return (
        root
        / config.output_root
        / "results"
        / inputs.workload["workload_id"]
        / runner
        / benchmark_id
    )


def _aggregate_gpu(repeats: Sequence[Mapping[str, Any]], formal_count: int) -> dict[str, Any]:
    if len(repeats) != REPEATS or len({item["output_sha256"] for item in repeats}) != 1:
        raise ValueError("fixed GPU repeats changed output")
    rates = [float(item["candidates_per_gpu_second"]) for item in repeats]
    return {
        "gpu_seconds": _distribution([item["gpu_seconds"] for item in repeats]),
        "candidates_per_gpu_second": _distribution(rates),
        "tasks_per_gpu_second": _distribution([item["tasks_per_gpu_second"] for item in repeats]),
        "peak_gpu_memory_allocated_bytes": _distribution(
            [item["peak_gpu_memory_allocated_bytes"] for item in repeats]
        ),
        "peak_gpu_memory_reserved_bytes": _distribution(
            [item["peak_gpu_memory_reserved_bytes"] for item in repeats]
        ),
        "formal_projection_hours": _distribution([formal_count / rate / 3600.0 for rate in rates]),
        "output_sha256": repeats[0]["output_sha256"],
    }


def run_gpu_generation_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    config_path: str | Path,
    workload_path: str | Path,
    repository_root: str | Path = ".",
    device: str = "cuda",
    task_batch_size: int = 32,
    warmup_iterations: int = 2,
    use_bfloat16: bool = False,
) -> tuple[Path, dict[str, Any]]:
    batch_size = _positive(task_batch_size, "task_batch_size")
    warmups = _positive(warmup_iterations, "warmup_iterations")
    root, config_path, workload_path = (
        Path(repository_root).resolve(),
        Path(config_path).resolve(),
        Path(workload_path).resolve(),
    )
    preflight_started = time.perf_counter()
    inputs = _load_inputs(config, workload_path=workload_path, root=root)
    from skilldrive.data import build_cvae_schema

    schema = build_cvae_schema(inputs.generation.formal_catalog.parent)
    runtime, model_load_seconds = _load_runtime(inputs, schema, device=device, root=root)
    contexts, context_stats = _prepare_contexts(inputs, schema)
    environment = _environment(runtime)
    preflight_seconds = time.perf_counter() - preflight_started
    warmup = _warmup(
        runtime,
        contexts,
        inputs,
        batch_size=batch_size,
        iterations=warmups,
        use_bfloat16=use_bfloat16,
    )
    contract = _contract(
        GPU_RUNNER,
        inputs,
        config,
        config_path=config_path,
        workload_path=workload_path,
        environment=environment,
        source_sha256=_sources(root, e2e=False),
        options={
            "task_batch_size": batch_size,
            "warmup_iterations": warmups,
            "use_bfloat16": use_bfloat16,
            "measurement_scope": (
                "cuda_events:scene_encoding+prior_sampling+decode;"
                "input_preparation_and_output_transfer_reported_separately_v1"
            ),
        },
    )
    benchmark_id = canonical_sha256(contract)
    result_root = _result_root(root, config, inputs, GPU_RUNNER, benchmark_id)
    repeats = []
    for index in range(REPEATS):
        repeat = _gpu_pass(
            runtime,
            contexts,
            inputs,
            repeat_index=index,
            batch_size=batch_size,
            use_bfloat16=use_bfloat16,
        )
        repeats.append(repeat)
        _atomic_write(result_root / f"repeat-{index + 1:02d}.json", repeat)
        print(
            f"GPU generation repeat {index + 1}/3: "
            f"{repeat['gpu_seconds']:.3f}s, "
            f"{repeat['candidates_per_gpu_second']:.1f} candidates/s",
            flush=True,
        )
    aggregate = _aggregate_gpu(repeats, config.benchmark.formal_candidate_count)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "counterfactual_gpu_generation_benchmark_summary",
        "status": "completed",
        "benchmark_id": benchmark_id,
        "benchmark_contract": contract,
        "initialization": {
            "preflight_seconds": preflight_seconds,
            "model_load_seconds": model_load_seconds,
            "context_preparation": context_stats,
        },
        "warmup": warmup,
        "aggregate": aggregate,
        "correctness": {
            "task_order_sha256": inputs.task_order_sha256,
            "latent_seed_sha256": inputs.latent_seed_sha256,
            "output_sha256": aggregate["output_sha256"],
            "all_outputs_finite": True,
        },
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    path = result_root / "summary.json"
    _atomic_write(path, summary)
    return path, summary


def _metadata(inputs: FixedInputs, index: int) -> dict[str, Any]:
    task, record = inputs.tasks[index], inputs.records[index]
    primary_role = inputs.generation.skills_by_id[task.skill_id].primary_generated_role
    return {
        "condition_skill_id": task.condition_skill_id,
        "evaluation_arm": pilot_evaluation_arm(
            task,
            none_skill_id=inputs.generation.none_skill_id,
        ),
        "latent_group_id": latent_group_id(task),
        "primary_generated_role": primary_role,
        "requested_parameters": record.sampled_parameters,
        "detection_mode": record.evidence["detection_mode"],
    }


def _filter_generated(
    workload: Mapping[str, Any],
    *,
    root: Path,
    generation: Any,
    filter_config: Any,
    detection_config: Any,
    workers: int,
    map_batch_size: int,
):
    from skilldrive.performance.parallel_filter import run_parallel_filter_workload

    return run_parallel_filter_workload(
        workload,
        repository_root=root,
        generation_config=generation,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=workers,
        map_batch_size=map_batch_size,
    )


def _e2e_repeat(
    runtime: Any,
    schema: Any,
    inputs: FixedInputs,
    *,
    repeat_index: int,
    result_root: Path,
    root: Path,
    batch_size: int,
    workers: int,
    map_batch_size: int,
    use_bfloat16: bool,
    filter_config: Any,
    detection_config: Any,
    execution_sha256: str,
) -> dict[str, Any]:
    import torch

    from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
    from skilldrive.generation.assembly import local_futures_to_global

    repeat_root = result_root / f"repeat-{repeat_index + 1:02d}"
    total_started = time.perf_counter()
    contexts, context_stats = _prepare_contexts(inputs, schema)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    digest = hashlib.sha256()
    commits, entries = [], []
    gpu_seconds = preparation_seconds = transfer_seconds = raw_seconds = 0.0
    generation_started = time.perf_counter()
    for offset in range(0, TASKS, batch_size):
        size = min(batch_size, TASKS - offset)
        result = _gpu_batch(
            runtime,
            contexts[offset : offset + size],
            np.stack(inputs.latent_seeds[offset : offset + size]),
            use_bfloat16=use_bfloat16,
            copy_output=True,
        )
        gpu_seconds += result["gpu_seconds"]
        preparation_seconds += result["preparation_seconds"]
        transfer_seconds += result["transfer_seconds"]
        for local_index in range(size):
            index = offset + local_index
            task, context = inputs.tasks[index], contexts[index]
            futures = local_futures_to_global(
                result["future"][local_index],
                context.anchor_origin_global,
                float(context.anchor_heading_global),
            )
            digest.update(np.ascontiguousarray(futures, dtype=np.float32).tobytes())
            metadata = _metadata(inputs, index)
            candidates = [
                GeneratedCandidate(
                    task_id=task.task_id,
                    candidate_index=candidate_index,
                    latent_seed=int(inputs.latent_seeds[index][candidate_index]),
                    scenario_id=task.scenario_id,
                    skill_id=task.skill_id,
                    proposal_mode=task.proposal_mode,
                    checkpoint_sha256=task.checkpoint_sha256,
                    semantic_config_sha256=task.semantic_config_sha256,
                    overlay=GeneratedOverlay(task.target_track_id, futures[candidate_index]),
                    metadata=metadata,
                )
                for candidate_index in range(CANDIDATES_PER_TASK)
            ]
            raw_started = time.perf_counter()
            commit = write_raw_shard(
                repeat_root / "raw",
                task.task_index,
                candidates,
                semantic_config_sha256=task.semantic_config_sha256,
                execution_config_sha256=execution_sha256,
            )
            raw_seconds += time.perf_counter() - raw_started
            commits.append(commit)
            entries.append(
                {
                    "task": generation_task_to_row(task),
                    "source_path": inputs.source_values[index],
                    "raw_commit": commit.commit_path.resolve().relative_to(root).as_posix(),
                }
            )
    generation_seconds = time.perf_counter() - generation_started
    generated_workload = {**inputs.workload, "tasks": entries}
    filter_started = time.perf_counter()
    filtered = _filter_generated(
        generated_workload,
        root=root,
        generation=inputs.generation,
        filter_config=filter_config,
        detection_config=detection_config,
        workers=workers,
        map_batch_size=map_batch_size,
    )
    filter_seconds = time.perf_counter() - filter_started
    index_started = time.perf_counter()
    index = write_filter_indexes(
        repeat_root / "filter",
        commits,
        filtered.batch.decisions,
        filter_config_sha256=inputs.workload["filter_semantic_sha256"],
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )
    index_seconds = time.perf_counter() - index_started
    total_seconds = time.perf_counter() - total_started
    return {
        "repeat_index": repeat_index,
        "task_count": TASKS,
        "candidate_count": CANDIDATES,
        "accepted_count": index.accepted_count,
        "rejected_count": index.rejected_count,
        "quality_passed_before_diversity": sum(
            value.quality_passed for value in filtered.batch.validations
        ),
        "read_and_tensorize_seconds": context_stats["elapsed_seconds"],
        "generation_wall_seconds": generation_seconds,
        "generation_gpu_seconds": gpu_seconds,
        "generation_batch_preparation_seconds": preparation_seconds,
        "generation_output_transfer_seconds": transfer_seconds,
        "raw_serialization_seconds": raw_seconds,
        "filter_wall_seconds": filter_seconds,
        "filter_index_serialization_seconds": index_seconds,
        "end_to_end_seconds": total_seconds,
        "candidates_per_second": CANDIDATES / total_seconds,
        "accepted_per_second": index.accepted_count / total_seconds,
        "peak_gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime.device)),
        "filter_workers_requested": filtered.requested_worker_count,
        "filter_workers_effective": filtered.effective_worker_count,
        "map_batch_size": map_batch_size,
        "filter_timings": dict(filtered.timings),
        "stage_execution_counts": dict(filtered.stage_execution_counts),
        "stage_rejection_counts": dict(filtered.stage_rejection_counts),
        "output_sha256": digest.hexdigest(),
        "decision_sha256": filtered.decision_sha256,
        "semantic_decision_sha256": filtered.semantic_decision_sha256,
        "bev_rendering_included": False,
    }


def _aggregate_e2e(
    repeats: Sequence[Mapping[str, Any]],
    formal_count: int,
    expected_semantic_decision: str | None,
) -> dict[str, Any]:
    stable = (
        "accepted_count",
        "rejected_count",
        "quality_passed_before_diversity",
        "output_sha256",
        "decision_sha256",
        "semantic_decision_sha256",
        "stage_execution_counts",
        "stage_rejection_counts",
    )
    if len(repeats) != REPEATS or any(
        len({canonical_sha256(item[name]) for item in repeats}) != 1 for name in stable
    ):
        raise ValueError("fixed end-to-end repeats changed correctness results")
    decision = repeats[0]["decision_sha256"]
    semantic_decision = repeats[0]["semantic_decision_sha256"]
    if (
        expected_semantic_decision is not None
        and semantic_decision != expected_semantic_decision
    ):
        raise ValueError(
            "end-to-end semantic decision SHA differs from the reference"
        )
    rates = [float(item["candidates_per_second"]) for item in repeats]
    return {
        "end_to_end_seconds": _distribution([item["end_to_end_seconds"] for item in repeats]),
        "candidates_per_second": _distribution(rates),
        "accepted_per_second": _distribution([item["accepted_per_second"] for item in repeats]),
        "generation_gpu_seconds": _distribution(
            [item["generation_gpu_seconds"] for item in repeats]
        ),
        "filter_wall_seconds": _distribution([item["filter_wall_seconds"] for item in repeats]),
        "formal_projection_hours": _distribution([formal_count / rate / 3600.0 for rate in rates]),
        "accepted_count": repeats[0]["accepted_count"],
        "rejected_count": repeats[0]["rejected_count"],
        "output_sha256": repeats[0]["output_sha256"],
        "decision_sha256": decision,
        "semantic_decision_sha256": semantic_decision,
        "stage_execution_counts": repeats[0]["stage_execution_counts"],
        "stage_rejection_counts": repeats[0]["stage_rejection_counts"],
    }


def run_end_to_end_benchmark(
    config: PerformanceBenchmarkConfig,
    *,
    config_path: str | Path,
    workload_path: str | Path,
    repository_root: str | Path = ".",
    device: str = "cuda",
    task_batch_size: int = 32,
    warmup_iterations: int = 2,
    use_bfloat16: bool = False,
    filter_workers: int = 1,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
    expected_semantic_decision_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    batch_size = _positive(task_batch_size, "task_batch_size")
    warmups = _positive(warmup_iterations, "warmup_iterations")
    workers = _positive(filter_workers, "filter_workers")
    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")
    expected = _sha256_or_none(expected_semantic_decision_sha256)
    root, config_path, workload_path = (
        Path(repository_root).resolve(),
        Path(config_path).resolve(),
        Path(workload_path).resolve(),
    )
    preflight_started = time.perf_counter()
    inputs = _load_inputs(config, workload_path=workload_path, root=root)
    filter_config = load_filter_config(root / config.inputs.filter_config)
    detection_config = load_detection_config(root / config.inputs.detection_config)
    from skilldrive.data import build_cvae_schema

    schema = build_cvae_schema(inputs.generation.formal_catalog.parent)
    runtime, model_load_seconds = _load_runtime(inputs, schema, device=device, root=root)
    warmup_contexts, warmup_context_stats = _prepare_contexts(
        inputs,
        schema,
        limit=batch_size,
    )
    environment = _environment(runtime)
    preflight_seconds = time.perf_counter() - preflight_started
    warmup = _warmup(
        runtime,
        warmup_contexts,
        inputs,
        batch_size=batch_size,
        iterations=warmups,
        use_bfloat16=use_bfloat16,
    )
    contract = _contract(
        E2E_RUNNER,
        inputs,
        config,
        config_path=config_path,
        workload_path=workload_path,
        environment=environment,
        source_sha256=_sources(root, e2e=True),
        options={
            "task_batch_size": batch_size,
            "warmup_iterations": warmups,
            "use_bfloat16": use_bfloat16,
            "filter_workers": workers,
            "filter_worker_semantics": "1=single_worker;>1=scenario_parallel",
            "map_batch_size": map_batch_size,
            "expected_semantic_decision_sha256": expected,
            "bev_rendering_included": False,
            "measurement_scope": (
                "read+future_free_tensorize+gpu_prior+raw_overlay_commit+"
                "chunked_batch_map_filter+filter_index_commit_v2"
            ),
        },
    )
    benchmark_id = canonical_sha256(contract)
    result_root = _result_root(root, config, inputs, E2E_RUNNER, benchmark_id)
    execution_sha256 = canonical_sha256({"benchmark_id": benchmark_id, "raw": "one_task"})
    repeats = []
    for index in range(REPEATS):
        repeat = _e2e_repeat(
            runtime,
            schema,
            inputs,
            repeat_index=index,
            result_root=result_root,
            root=root,
            batch_size=batch_size,
            workers=workers,
            map_batch_size=map_batch_size,
            use_bfloat16=use_bfloat16,
            filter_config=filter_config,
            detection_config=detection_config,
            execution_sha256=execution_sha256,
        )
        repeats.append(repeat)
        _atomic_write(result_root / f"repeat-{index + 1:02d}.json", repeat)
        print(
            f"end-to-end repeat {index + 1}/3: "
            f"{repeat['end_to_end_seconds']:.3f}s, "
            f"{repeat['candidates_per_second']:.1f} candidates/s",
            flush=True,
        )
    aggregate = _aggregate_e2e(repeats, config.benchmark.formal_candidate_count, expected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "kind": "counterfactual_end_to_end_benchmark_summary",
        "status": "completed",
        "benchmark_id": benchmark_id,
        "benchmark_contract": contract,
        "initialization": {
            "preflight_seconds": preflight_seconds,
            "model_load_seconds": model_load_seconds,
            "warmup_context_preparation": warmup_context_stats,
        },
        "warmup": warmup,
        "aggregate": aggregate,
        "correctness": {
            "task_order_sha256": inputs.task_order_sha256,
            "latent_seed_sha256": inputs.latent_seed_sha256,
            "output_sha256": aggregate["output_sha256"],
            "decision_sha256": aggregate["decision_sha256"],
            "semantic_decision_sha256": aggregate["semantic_decision_sha256"],
            "expected_semantic_decision_sha256": expected,
            "expected_semantic_decision_matched": None if expected is None else True,
            "accepted_count": aggregate["accepted_count"],
            "rejected_count": aggregate["rejected_count"],
            "stage_execution_counts": aggregate["stage_execution_counts"],
            "stage_rejection_counts": aggregate["stage_rejection_counts"],
        },
        "bev_rendering_included": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    path = result_root / "summary.json"
    _atomic_write(path, summary)
    return path, summary


__all__ = [
    "CANDIDATES",
    "CANDIDATES_PER_TASK",
    "E2E_RUNNER",
    "GPU_RUNNER",
    "REPEATS",
    "SCHEMA_VERSION",
    "TASKS",
    "run_end_to_end_benchmark",
    "run_gpu_generation_benchmark",
]

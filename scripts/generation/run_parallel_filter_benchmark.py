"""Run three fixed-workload scenario-parallel CPU filter repeats."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

from skilldrive.generation.config import (
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.filtering.pipeline import DEFAULT_MAP_BATCH_SIZE, MAP_BATCH_SIZES
from skilldrive.generation.contracts import canonical_json_bytes, canonical_sha256
from skilldrive.performance.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    aggregate_repeat_results,
)
from skilldrive.performance.config import (
    DEFAULT_PERFORMANCE_CONFIG,
    load_performance_config,
)
from skilldrive.performance.parallel_filter import run_parallel_filter_workload
from skilldrive.performance.workload import file_sha256, load_fixed_workload
from skilldrive.skills.detection import load_detection_config


_WORKER_COUNTS = (1, 2, 4, 8, 12, 16, 20)


def _sha256(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return normalized


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


def run_benchmark(
    *,
    repository_root: str | Path,
    config_path: str | Path,
    workload_path: str | Path,
    workers: int,
    expected_decision_sha256: str,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
) -> tuple[Path, dict[str, Any]]:
    """Run exactly three repeats and publish only after all repeats pass."""

    if workers not in _WORKER_COUNTS:
        raise ValueError("workers must be one of 1, 2, 4, 8, 12, 16, or 20")
    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")
    expected = _sha256(expected_decision_sha256, "expected_decision_sha256")
    root = Path(repository_root).resolve()
    config_file = Path(config_path).resolve()
    workload_file = Path(workload_path).resolve()
    config = load_performance_config(config_file)
    workload = load_fixed_workload(workload_file, repository_root=root)
    generation = load_counterfactual_config(
        root / config.inputs.generation_config,
        repository_root=root,
    )
    filter_config = load_filter_config(root / config.inputs.filter_config)
    detection_config = load_detection_config(root / config.inputs.detection_config)

    repeats = []
    for repeat_index in range(3):
        result = run_parallel_filter_workload(
            workload,
            repository_root=root,
            generation_config=generation,
            filter_config=filter_config,
            detection_config=detection_config,
            worker_count=workers,
            map_batch_size=map_batch_size,
        )
        if result.effective_worker_count != workers:
            raise ValueError(
                "parallel repeat did not use every requested worker: "
                f"requested={workers}, effective={result.effective_worker_count}"
            )
        if result.decision_sha256 != expected:
            raise ValueError(
                "parallel decision_sha256 differs from expected: "
                f"expected={expected}, actual={result.decision_sha256}"
            )
        candidate_count = len(result.batch.decisions)
        if candidate_count != int(workload["counts"]["candidates"]):
            raise ValueError("parallel repeat differs from fixed workload counts")
        accepted = sum(item.accepted for item in result.batch.decisions)
        stable_worker = result.timings["worker_execution_seconds"]
        stable_total = result.timings["stable_total_seconds"]
        repeats.append(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "kind": "counterfactual_performance_repeat",
                "runner": "cpu_filter_parallel",
                "repeat_index": repeat_index,
                "map_batch_size": map_batch_size,
                "task_count": int(workload["counts"]["tasks"]),
                "candidate_count": candidate_count,
                "accepted_count": accepted,
                "rejected_count": candidate_count - accepted,
                "quality_passed_before_diversity": sum(
                    item.quality_passed for item in result.batch.validations
                ),
                "elapsed_seconds": stable_total,
                "candidates_per_second": candidate_count / stable_total,
                "accepted_per_second": accepted / stable_total,
                "decision_sha256": result.decision_sha256,
                "semantic_decision_sha256": result.semantic_decision_sha256,
                "stage_execution_counts": dict(result.stage_execution_counts),
                "stage_rejection_counts": dict(result.stage_rejection_counts),
                "stage_elapsed_seconds": dict(result.batch.stage_elapsed_seconds),
                "worker_startup_seconds": result.timings["worker_startup_seconds"],
                "stable_worker_seconds": stable_worker,
                "stable_total_seconds": stable_total,
                "wall_total_seconds": result.timings["total_seconds"],
                "requested_worker_count": result.requested_worker_count,
                "effective_worker_count": result.effective_worker_count,
                "worker_pids": list(result.worker_pids),
                "scenario_load_count": result.scenario_load_count,
                "prepared_map_count": result.prepared_map_count,
                "timings": dict(result.timings),
                "validation_manifests_opened": False,
                "final_validation_accessed": False,
            }
        )
        print(
            f"parallel repeat {repeat_index + 1}/3 workers={workers}: "
            f"startup={result.timings['worker_startup_seconds']:.3f}s, "
            f"stable_worker={stable_worker:.3f}s, "
            f"stable_total={stable_total:.3f}s",
            flush=True,
        )

    aggregate = aggregate_repeat_results(
        repeats,
        formal_candidate_count=config.benchmark.formal_candidate_count,
    )
    contract = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "runner": "cpu_filter_parallel",
        "workload_id": workload["workload_id"],
        "workload_sha256": file_sha256(workload_file),
        "filter_semantic_sha256": workload["filter_semantic_sha256"],
        "performance_config_sha256": file_sha256(config_file),
        "runner_source_sha256": {
            relative: file_sha256(root / relative)
            for relative in (
                "skilldrive/performance/parallel_filter.py",
                "scripts/generation/run_parallel_filter_benchmark.py",
            )
        },
        "workers": workers,
        "map_batch_size": map_batch_size,
        "repeats": 3,
        "expected_decision_sha256": expected,
        "measurement_scope": (
            "spawn_startup_plus_chunked_batch_map_worker_plus_global_finalize_v2"
        ),
    }
    benchmark_id = canonical_sha256(contract)
    output_root = (
        root
        / config.output_root
        / "results"
        / str(workload["workload_id"])
        / f"cpu_filter_parallel_w{workers}_mb{map_batch_size}"
        / benchmark_id
    )
    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "kind": "counterfactual_performance_benchmark_summary",
        "status": "completed",
        "benchmark_id": benchmark_id,
        "benchmark_contract": contract,
        "semantic_decision_sha256": repeats[0]["semantic_decision_sha256"],
        "aggregate": aggregate,
        "worker_startup_seconds": [item["worker_startup_seconds"] for item in repeats],
        "stable_worker_seconds": [item["stable_worker_seconds"] for item in repeats],
        "stable_total_seconds": [item["stable_total_seconds"] for item in repeats],
        "wall_total_seconds": [item["wall_total_seconds"] for item in repeats],
        "effective_worker_count": [item["effective_worker_count"] for item in repeats],
        "worker_pids": [item["worker_pids"] for item in repeats],
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    for repeat in repeats:
        _atomic_write(
            output_root / f"repeat-{int(repeat['repeat_index']) + 1:02d}.json",
            repeat,
        )
    summary_path = output_root / "summary.json"
    _atomic_write(summary_path, summary)
    return summary_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE_CONFIG)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=_WORKER_COUNTS, required=True)
    parser.add_argument(
        "--map-batch-size",
        type=int,
        choices=sorted(MAP_BATCH_SIZES),
        default=DEFAULT_MAP_BATCH_SIZE,
    )
    parser.add_argument("--expected-decision-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    workload = args.workload if args.workload.is_absolute() else root / args.workload
    path, summary = run_benchmark(
        repository_root=root,
        config_path=config,
        workload_path=workload,
        workers=args.workers,
        expected_decision_sha256=args.expected_decision_sha256,
        map_batch_size=args.map_batch_size,
    )
    projection = summary["aggregate"]["formal_projection"]
    print(
        f"parallel benchmark complete: p50={projection['hours_p50']:.3f}h, "
        f"p95={projection['hours_p95']:.3f}h -> {path}"
    )


if __name__ == "__main__":
    main()

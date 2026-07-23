"""Run fixed GPU-generation or end-to-end performance benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.filtering.pipeline import DEFAULT_MAP_BATCH_SIZE, MAP_BATCH_SIZES
from skilldrive.performance.config import (
    DEFAULT_PERFORMANCE_CONFIG,
    load_performance_config,
)
from skilldrive.performance.runtime_benchmark import (
    run_end_to_end_benchmark,
    run_gpu_generation_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("gpu-generation", "end-to-end"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE_CONFIG)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-batch-size", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--use-bfloat16", action="store_true")
    parser.add_argument(
        "--filter-workers",
        type=int,
        default=1,
        help="End-to-end only: 1 uses one spawned worker; >1 enables scenario parallelism.",
    )
    parser.add_argument(
        "--map-batch-size",
        type=int,
        choices=sorted(MAP_BATCH_SIZES),
        default=DEFAULT_MAP_BATCH_SIZE,
        help="End-to-end only: number of pre-map survivors per vectorized map query.",
    )
    parser.add_argument(
        "--expected-semantic-decision-sha256",
        help=(
            "End-to-end only: require the candidate outcomes and rejection stages "
            "to match a semantic reference SHA."
        ),
    )
    return parser


def _resolved(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    config_path = _resolved(root, args.config)
    workload_path = _resolved(root, args.workload)
    config = load_performance_config(config_path)
    common = {
        "config_path": config_path,
        "workload_path": workload_path,
        "repository_root": root,
        "device": args.device,
        "task_batch_size": args.task_batch_size,
        "warmup_iterations": args.warmup_iterations,
        "use_bfloat16": args.use_bfloat16,
    }
    if args.stage == "gpu-generation":
        path, summary = run_gpu_generation_benchmark(config, **common)
        aggregate = summary["aggregate"]
        print(
            "GPU generation benchmark complete: "
            f"p50={aggregate['gpu_seconds']['p50']:.3f}s, "
            f"p50={aggregate['candidates_per_gpu_second']['p50']:.1f} candidates/s "
            f"-> {path}"
        )
        return

    path, summary = run_end_to_end_benchmark(
        config,
        **common,
        filter_workers=args.filter_workers,
        map_batch_size=args.map_batch_size,
        expected_semantic_decision_sha256=args.expected_semantic_decision_sha256,
    )
    aggregate = summary["aggregate"]
    print(
        "End-to-end benchmark complete (BEV excluded): "
        f"p50={aggregate['end_to_end_seconds']['p50']:.3f}s, "
        f"p50={aggregate['candidates_per_second']['p50']:.1f} candidates/s "
        f"-> {path}"
    )


if __name__ == "__main__":
    main()

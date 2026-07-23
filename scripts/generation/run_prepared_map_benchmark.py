"""Run the prepared-map CPU filter benchmark on an existing fixed workload."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.performance.config import (
    DEFAULT_PERFORMANCE_CONFIG,
    load_performance_config,
)
from skilldrive.performance.prepared_benchmark import (
    run_cpu_filter_prepared_map_benchmark,
)
from skilldrive.filtering.pipeline import DEFAULT_MAP_BATCH_SIZE, MAP_BATCH_SIZES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE_CONFIG)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument(
        "--map-batch-size",
        type=int,
        choices=sorted(MAP_BATCH_SIZES),
        default=DEFAULT_MAP_BATCH_SIZE,
    )
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--expected-decision-sha256",
        help=(
            "optional legacy/reference decision digest; every prepared-map repeat "
            "must match it"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    workload_path = (
        args.workload.resolve()
        if args.workload.is_absolute()
        else (repository_root / args.workload).resolve()
    )
    config = load_performance_config(config_path)
    path, summary = run_cpu_filter_prepared_map_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=repository_root,
        expected_decision_sha256=args.expected_decision_sha256,
        map_batch_size=args.map_batch_size,
    )
    projection = summary["aggregate"]["formal_projection"]
    digest = summary["aggregate"]["decision_sha256"]
    print(
        "cpu-filter prepared-map benchmark complete: "
        f"decision_sha256={digest}, "
        f"formal projection p50={projection['hours_p50']:.3f}h, "
        f"p95={projection['hours_p95']:.3f}h -> {path}"
    )


if __name__ == "__main__":
    main()

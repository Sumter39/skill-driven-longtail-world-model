"""Prepare and run fixed counterfactual-generation performance benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.performance import (
    DEFAULT_PERFORMANCE_CONFIG,
    load_performance_config,
    prepare_fixed_workload,
    run_cpu_filter_legacy_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "cpu-filter-legacy"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PERFORMANCE_CONFIG)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository_root = args.repository_root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (repository_root / args.config).resolve()
    )
    config = load_performance_config(config_path)
    if args.stage == "prepare":
        path, workload = prepare_fixed_workload(
            config,
            config_path=config_path,
            repository_root=repository_root,
        )
        print(
            "performance workload prepared: "
            f"{workload['counts']['tasks']} tasks, "
            f"{workload['counts']['candidates']} candidates -> {path}"
        )
        return

    if args.workload is None:
        raise ValueError("cpu-filter-legacy requires --workload")
    workload_path = (
        args.workload.resolve()
        if args.workload.is_absolute()
        else (repository_root / args.workload).resolve()
    )
    path, summary = run_cpu_filter_legacy_benchmark(
        config,
        config_path=config_path,
        workload_path=workload_path,
        repository_root=repository_root,
    )
    projection = summary["aggregate"]["formal_projection"]
    print(
        "cpu-filter legacy benchmark complete: "
        f"formal projection p50={projection['hours_p50']:.3f}h, "
        f"p95={projection['hours_p95']:.3f}h -> {path}"
    )


if __name__ == "__main__":
    main()

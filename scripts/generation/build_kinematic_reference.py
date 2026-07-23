"""Build the resumable AV2 Formal Train kinematic reference."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from skilldrive.filtering.reference_kinematics import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    ReferenceProgress,
    build_kinematic_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan only manifests/splits/formal_train.csv and build deterministic "
            "per-class kinematic reference statistics."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="AV2 motion-forecasting root containing train/ (project-relative by default)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Reference output directory (project-relative by default)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parquet worker processes; changing this does not change outputs",
    )
    parser.add_argument(
        "--max-new-shards",
        type=int,
        default=None,
        help="Optional bounded run; later invocation resumes the same full contract",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard only this reference output and rebuild it from shard zero",
    )
    return parser


class _ProgressPrinter:
    def __init__(self) -> None:
        self.started_at: float | None = None
        self.initial_scenarios: int | None = None
        self.last_length = 0

    def __call__(self, value: ReferenceProgress) -> None:
        if value.phase == "preflight":
            print(
                "kinematics preflight: validating Formal Train manifest, filter "
                "policy, output ownership, and resumable shards",
                flush=True,
            )
            return
        if value.phase != "scan":
            raise ValueError(f"unknown reference progress phase: {value.phase}")
        if self.initial_scenarios is None:
            self.initial_scenarios = value.completed_scenarios
            self.started_at = time.monotonic()
        assert self.started_at is not None
        elapsed = max(time.monotonic() - self.started_at, 1.0e-9)
        new_scenarios = value.completed_scenarios - self.initial_scenarios
        rate = new_scenarios / elapsed
        remaining = value.total_scenarios - value.completed_scenarios
        eta = None if rate <= 0.0 else remaining / rate
        width = 30
        fraction = (
            1.0
            if value.total_scenarios == 0
            else value.completed_scenarios / value.total_scenarios
        )
        filled = min(width, int(fraction * width))
        line = (
            f"kinematics [{'#' * filled}{'-' * (width - filled)}] "
            f"{fraction * 100:6.2f}%  "
            f"{value.completed_scenarios}/{value.total_scenarios} scenarios  "
            f"{value.completed_shards}/{value.total_shards} shards  "
            f"{rate:5.1f} scenarios/s  ETA {_duration(eta)}"
        )
        if sys.stdout.isatty():
            padding = " " * max(0, self.last_length - len(line))
            print(f"\r{line}{padding}", end="", flush=True)
            self.last_length = len(line)
        else:
            print(line, flush=True)

    def finish(self) -> None:
        if sys.stdout.isatty() and self.last_length:
            print()


def _duration(seconds: float | None) -> str:
    if seconds is None or not float(seconds) >= 0.0:
        return "--:--"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[2]
    printer = _ProgressPrinter()
    try:
        result = build_kinematic_reference(
            project_root,
            data_root=args.data_root,
            output_root=args.output_root,
            workers=args.workers,
            max_new_shards=args.max_new_shards,
            restart=args.restart,
            progress=printer,
        )
    finally:
        printer.finish()
    if result.complete:
        print(f"complete reference: {result.summary_path}")
    else:
        print(
            "bounded run stopped cleanly; rerun the same command to resume "
            f"({result.completed_shards}/{result.total_shards} shards complete)"
        )


if __name__ == "__main__":
    main()

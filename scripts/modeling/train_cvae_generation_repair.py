"""Run the isolated cvae_generation_repair_v1 training contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, TextIO

from scripts.modeling.train_cvae import run_training


DEFAULT_REPAIR_CONFIG = Path("configs/models/cvae_generation_repair_v1.yaml")
REPAIR_MODES = ("overfit", "benchmark", "formal")


def run_repair_training(
    *,
    mode: str,
    config_path: str | Path = DEFAULT_REPAIR_CONFIG,
    project_root: str | Path = ".",
    resume: str = "auto",
    max_steps: int | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_root: str | Path | None = None,
    amp: bool | None = None,
    prefetch_factor: int | None = None,
    benchmark_repeats: int | None = None,
    tf32: bool | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Dispatch one repair-only mode; no legacy stage is reachable here."""

    if mode not in REPAIR_MODES:
        raise ValueError(f"mode must be one of {REPAIR_MODES}")
    return run_training(
        config_path=config_path,
        stage=f"repair-{mode}",
        project_root=project_root,
        resume=resume,
        max_steps=max_steps,
        max_epochs=max_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        cache_root=cache_root,
        amp=amp,
        prefetch_factor=prefetch_factor,
        benchmark_repeats=benchmark_repeats,
        allow_tf32=tf32,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        progress_stream=progress_stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit, benchmark, or train cvae_generation_repair_v1 using only "
            "audited Formal Train repair views."
        )
    )
    parser.add_argument("--mode", choices=REPAIR_MODES, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_REPAIR_CONFIG)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--amp", choices=("on", "off"))
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--benchmark-repeats", type=int)
    parser.add_argument("--tf32", choices=("on", "off"))
    parser.add_argument("--pin-memory", choices=("on", "off"))
    parser.add_argument("--persistent-workers", choices=("on", "off"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_repair_training(
        mode=args.mode,
        config_path=args.config,
        resume=args.resume,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_root=args.cache_root,
        amp=None if args.amp is None else args.amp == "on",
        prefetch_factor=args.prefetch_factor,
        benchmark_repeats=args.benchmark_repeats,
        tf32=None if args.tf32 is None else args.tf32 == "on",
        pin_memory=(
            None if args.pin_memory is None else args.pin_memory == "on"
        ),
        persistent_workers=(
            None
            if args.persistent_workers is None
            else args.persistent_workers == "on"
        ),
    )
    print(
        f"CVAE repair {args.mode} complete: "
        f"step={summary.get('progress', {}).get('global_step', 'benchmark')}"
    )


if __name__ == "__main__":
    main()

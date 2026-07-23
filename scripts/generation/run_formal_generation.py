"""Prepare, resume, or verify the formal counterfactual generation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skilldrive.generation.formal_runner import (
    DEFAULT_FORMAL_EXECUTION_CONFIG,
    prepare_formal_run,
    run_formal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--execution-config", type=Path, default=DEFAULT_FORMAL_EXECUTION_CONFIG)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.repository_root.resolve()
    if args.command == "prepare":
        run_root, plan = prepare_formal_run(
            repository_root=root,
            execution_config_path=args.execution_config,
        )
        print(f"formal plan prepared: {run_root}")
        print(f"tasks={len(plan.tasks)} candidates={plan.total_candidates}")
        return
    if args.command == "run":
        run_root = run_formal(
            repository_root=root,
            execution_config_path=args.execution_config,
        )
        print(f"formal run output: {run_root}")
        return

    from skilldrive.generation.formal_runner import _sha256, _execution_values, _resolved
    values = _execution_values(root, _resolved(root, args.execution_config))
    output_root = _resolved(root, values["output"]["root"])
    summaries = []
    for path in sorted(output_root.glob("*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise SystemExit(f"no formal summary found under {output_root}")
    print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

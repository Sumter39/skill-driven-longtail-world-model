"""Render source/generated BEV pairs for the frozen active Pilot review set."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.pilot_review import render_active_pilot_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-analysis", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/counterfactual_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_active_pilot_review(
        gate_analysis_path=args.gate_analysis,
        generation_config_path=args.config,
        output_root=args.output_root,
    )
    print(
        "active Pilot BEV review: "
        f"{result['case_count']} cases, "
        f"{result['rendered_case_count']} rendered, "
        f"{result['resumed_case_count']} resumed",
        flush=True,
    )
    print(f"review summary: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

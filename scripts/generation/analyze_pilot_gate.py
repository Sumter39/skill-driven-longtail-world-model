"""Freeze the active checkpoint Pilot capability boundary and review cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.pilot_gate import analyze_active_pilot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/counterfactual_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze_active_pilot(
        pilot_summary_path=args.pilot_summary,
        generation_config_path=args.config,
        output_root=args.output_root,
    )
    print(
        "active Pilot gate: "
        f"{result['status']}, "
        f"{result['pilot']['formal_accepted_count']} formal accepted, "
        f"{result['pilot']['formal_supported_skill_count']} supported skills",
        flush=True,
    )
    print(f"analysis: {result['output_paths']['analysis']}", flush=True)
    print(
        f"review manifest: {result['output_paths']['review_manifest']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

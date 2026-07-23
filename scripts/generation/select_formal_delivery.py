"""Select and audit the balanced accepted subset from a completed formal run."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.formal_delivery import build_formal_delivery


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/generation/counterfactual_v1.yaml"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-per-skill", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_formal_delivery(
        run_root=args.run_root,
        repository_root=args.repository_root,
        generation_config_path=args.generation_config,
        output_root=args.output_root,
        max_per_skill=args.max_per_skill,
    )
    print(
        f"formal delivery: {result['selected_candidate_count']} candidates, "
        f"{len(result['skill_counts'])} skills",
        flush=True,
    )
    print(f"delivery audit: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

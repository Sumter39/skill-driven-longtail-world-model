"""Build and render the deterministic representative review for a formal run."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.formal_review import render_formal_review


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
    parser.add_argument("--per-disposition", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_formal_review(
        run_root=args.run_root,
        repository_root=args.repository_root,
        generation_config_path=args.generation_config,
        output_root=args.output_root,
        per_disposition=args.per_disposition,
    )
    print(
        f"formal review: {result['case_count']} cases, "
        f"{result['rendered_case_count']} rendered, "
        f"{result['resumed_case_count']} resumed",
        flush=True,
    )
    print(f"review summary: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

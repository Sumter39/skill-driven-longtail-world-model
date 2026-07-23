"""Generate the evidence-based automatic review CSV for formal cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.automated_review import build_automated_review_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_automated_review_csv(
        summary_path=args.summary,
        run_root=args.run_root,
        output_path=args.output,
    )
    print(
        f"automated formal review: {result['reviewed_count']} cases, "
        f"statuses={result['status_counts']}",
        flush=True,
    )
    print(f"automated review CSV: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

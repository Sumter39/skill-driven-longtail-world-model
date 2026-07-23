"""Validate and finalize manually annotated formal review cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.formal_review import finalize_review_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-reviews", type=int, default=100)
    args = parser.parse_args()
    result = finalize_review_annotations(
        summary_path=args.summary,
        annotations_path=args.annotations,
        output_path=args.output,
        minimum_reviews=args.minimum_reviews,
    )
    print(
        f"manual review finalized: {result['manual_review_count']} cases, "
        f"status={result['manual_review_status']}",
        flush=True,
    )
    print(f"reviewed summary: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

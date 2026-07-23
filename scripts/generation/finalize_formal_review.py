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
    parser.add_argument(
        "--review-method",
        choices=("manual", "automated_evidence"),
        default="manual",
    )
    args = parser.parse_args()
    result = finalize_review_annotations(
        summary_path=args.summary,
        annotations_path=args.annotations,
        output_path=args.output,
        minimum_reviews=args.minimum_reviews,
        review_method=args.review_method,
    )
    print(
        f"review finalized: {result['review_count']} cases, "
        f"method={result['review_method']}, status={result['review_status']}",
        flush=True,
    )
    print(f"reviewed summary: {result['output_path']}", flush=True)


if __name__ == "__main__":
    main()

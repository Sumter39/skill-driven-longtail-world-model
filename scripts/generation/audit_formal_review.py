"""Audit rendered formal review artifacts and create the manual review CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation.formal_review import audit_formal_review, write_review_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--review-template", type=Path)
    args = parser.parse_args()
    result = audit_formal_review(
        summary_path=args.summary,
        repository_root=args.repository_root,
        output_path=args.audit_output,
    )
    template = write_review_template(
        summary_path=args.summary,
        output_path=args.review_template,
    )
    print(
        f"formal review audit: {result['case_count']} cases, "
        f"{result['verified_image_count']} images verified",
        flush=True,
    )
    print(f"audit: {result['output_path']}", flush=True)
    print(f"manual review template: {template}", flush=True)


if __name__ == "__main__":
    main()

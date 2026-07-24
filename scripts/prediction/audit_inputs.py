"""Audit the immutable inputs required by the downstream prediction Goal."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.prediction.audit import audit_prediction_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument(
        "--archive",
        type=Path,
        action="append",
        default=[],
        help="Optional local archive to hash; may be passed more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/prediction/input_audit_v1.json"),
    )
    args = parser.parse_args()
    payload = audit_prediction_inputs(
        cache_root=args.cache_root,
        formal_run_root=args.formal_run_root,
        output_path=args.output,
        archive_paths=tuple(args.archive),
    )
    print(
        "input audit complete: "
        f"formal={payload['manifests']['formal_train']['scenario_count']} scenarios, "
        f"internal={payload['manifests']['internal_validation']['scenario_count']} scenarios, "
        f"final={payload['manifests']['final_validation']['scenario_count']} scenarios, "
        f"accepted={payload['formal_generation']['delivery_count']}"
    )


if __name__ == "__main__":
    main()

"""Prepare resumable E1/E2/E3 downstream prediction views."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.prediction.preparation import build_prediction_augmentation_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--formal-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("manifests/prediction/augmentation_bundle_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    manifest = build_prediction_augmentation_bundle(
        data_root=args.data_root,
        formal_run_root=args.formal_run_root,
        output_root=args.output_root,
        manifest_output=args.manifest_output,
        seed=args.seed,
        resume=not args.no_resume,
    )
    print(
        "prediction augmentation bundle complete: "
        f"contexts={manifest['context_count']} "
        f"e1={manifest['arms']['e1']['count']} "
        f"e2={manifest['arms']['e2']['count']} "
        f"e3={manifest['arms']['e3']['count']} "
        f"e1_partial_rows={manifest['e1_partial_future_rows']} "
        f"e1_masked_points={manifest['e1_masked_future_points']}"
    )


if __name__ == "__main__":
    main()

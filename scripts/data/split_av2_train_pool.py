"""Split the downloaded AV2 Train pool into reproducible logical manifests."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from skilldrive.data.manifests import ManifestRow, read_manifest, write_manifest
from skilldrive.data.subsets import select_ids


def build_splits(
    rows: list[ManifestRow],
    *,
    train_count: int,
    internal_validation_count: int,
    development_train_count: int,
    development_validation_count: int,
    seed: int,
) -> dict[str, list[ManifestRow]]:
    by_id = {row.scenario_id: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("input manifest contains duplicate scenario IDs")
    if len(rows) != train_count + internal_validation_count:
        raise ValueError(
            "input manifest size must equal train_count + internal_validation_count"
        )

    all_ids = sorted(by_id)
    internal_validation_ids = set(
        select_ids(all_ids, internal_validation_count, seed)
    )
    train_ids = [scenario_id for scenario_id in all_ids if scenario_id not in internal_validation_ids]
    development_train_ids = set(select_ids(train_ids, development_train_count, seed + 1))
    development_validation_ids = set(
        select_ids(sorted(internal_validation_ids), development_validation_count, seed + 2)
    )

    def make_rows(ids: list[str] | set[str], split: str, reason: str) -> list[ManifestRow]:
        return [
            replace(by_id[scenario_id], split=split, selected_reason=reason)
            for scenario_id in sorted(ids)
        ]

    return {
        "formal_train": make_rows(train_ids, "train", f"train_pool_split_seed_{seed}"),
        "internal_validation": make_rows(
            internal_validation_ids,
            "internal_validation",
            f"train_pool_split_seed_{seed}",
        ),
        "development_train": make_rows(
            development_train_ids,
            "development_train",
            f"development_subset_seed_{seed + 1}",
        ),
        "development_validation": make_rows(
            development_validation_ids,
            "development_validation",
            f"development_subset_seed_{seed + 2}",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        default=Path("manifests/acquisition/formal_train_pool.csv"),
    )
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument("--internal-validation-count", type=int, default=2000)
    parser.add_argument("--development-train-count", type=int, default=500)
    parser.add_argument("--development-validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    outputs = {
        "formal_train": args.manifest_root / "splits" / "formal_train.csv",
        "internal_validation": (
            args.manifest_root / "splits" / "internal_validation.csv"
        ),
        "development_train": (
            args.manifest_root / "development" / "development_train.csv"
        ),
        "development_validation": (
            args.manifest_root / "development" / "development_validation.csv"
        ),
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output manifests already exist: {names}; pass --force to replace them")

    splits = build_splits(
        read_manifest(args.pool_manifest),
        train_count=args.train_count,
        internal_validation_count=args.internal_validation_count,
        development_train_count=args.development_train_count,
        development_validation_count=args.development_validation_count,
        seed=args.seed,
    )
    for name, path in outputs.items():
        write_manifest(path, splits[name])
        print(f"wrote {len(splits[name])} scenarios to {path}")


if __name__ == "__main__":
    main()

"""Verify every AV2 scenario referenced by a manifest."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from skilldrive.data.manifests import ManifestRow, read_manifest


def verify_row(row: ManifestRow, data_root: Path) -> list[str]:
    scenario_path = data_root / row.source_path
    scenario_directory = scenario_path.parent
    map_path = scenario_directory / f"log_map_archive_{row.scenario_id}.json"
    errors: list[str] = []

    for label, path in (("scenario", scenario_path), ("map", map_path)):
        if not path.is_file():
            errors.append(f"{row.scenario_id}: missing {label} file: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{row.scenario_id}: zero-byte {label} file: {path}")

    if errors:
        return errors

    try:
        pq.ParquetFile(scenario_path).metadata
    except Exception as error:
        errors.append(f"{row.scenario_id}: invalid parquet: {error}")

    try:
        with map_path.open(encoding="utf-8") as handle:
            json.load(handle)
    except Exception as error:
        errors.append(f"{row.scenario_id}: invalid map JSON: {error}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/av2/motion-forecasting"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(verify_row, row, args.data_root): row for row in rows}
        for completed, future in enumerate(as_completed(futures), start=1):
            errors.extend(future.result())
            if completed % 1_000 == 0 or completed == len(rows):
                print(f"verified {completed}/{len(rows)} scenarios", flush=True)

    if errors:
        print(f"verification failed with {len(errors)} errors")
        for error in errors[:100]:
            print(error)
        if len(errors) > 100:
            print(f"... {len(errors) - 100} additional errors omitted")
        raise SystemExit(1)
    print(f"verification passed: {len(rows)} scenarios are complete and readable")


if __name__ == "__main__":
    main()

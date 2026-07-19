"""CSV scene manifests and split-leakage checks."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FIELDNAMES = ["scenario_id", "split", "source_path", "city_name", "selected_reason"]


@dataclass(frozen=True)
class ManifestRow:
    scenario_id: str
    split: str
    source_path: str
    city_name: str
    selected_reason: str


def write_manifest(path: str | Path, rows: Iterable[ManifestRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(f"manifest columns must be {FIELDNAMES}, got {reader.fieldnames}")
        return [ManifestRow(**row) for row in reader]


def assert_disjoint(*groups: Iterable[ManifestRow]) -> None:
    seen: set[str] = set()
    for index, group in enumerate(groups):
        current = {row.scenario_id for row in group}
        overlap = seen & current
        if overlap:
            values = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"scenario leakage detected before group {index}: {values}")
        seen.update(current)

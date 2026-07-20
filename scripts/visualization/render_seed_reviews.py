"""Render a deterministic, resumable review set from a candidate seed CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.schemas import Scenario
from skilldrive.seeds import SEED_CSV_FIELDS, SeedRecord, read_seed_records
from skilldrive.visualization import (
    render_seed_review,
    seed_review_filename,
    select_stratified_review_records,
)


DEFAULT_OUTPUT_DIR = Path("outputs/seed_detection/review")
REVIEW_INDEX_FIELDS = (
    "review_rank",
    "output_png",
    "seed_risk_is_proxy",
    *SEED_CSV_FIELDS,
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

ScenarioLoader = Callable[[str | Path], Scenario]
ReviewRenderer = Callable[[Scenario, SeedRecord, str | Path], Path]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_entry(rank: int, record: SeedRecord) -> dict[str, object]:
    return {
        "review_rank": rank,
        "output_png": seed_review_filename(record),
        "scenario_id": record.scenario_id,
        "skill_id": record.skill_id,
        "initiator_track_id": record.initiator_track_id,
        "responder_track_id": record.responder_track_id,
        "role_track_ids": dict(sorted(record.role_track_ids.items())),
        "trigger_score": record.trigger_score,
        "seed_risk_metric": record.seed_risk_metric,
        "seed_risk_value": record.seed_risk_value,
        "seed_risk_is_proxy": record.seed_risk_is_proxy,
        "target_risk_definition": record.target_risk_definition,
        "source_path": record.source_path,
        "evidence": record.evidence,
        "sampled_parameters": record.sampled_parameters,
    }


def _index_payloads(
    selected: Sequence[SeedRecord],
    *,
    available_count: int,
    target_count: int,
) -> tuple[bytes, bytes]:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=REVIEW_INDEX_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    entries: list[dict[str, object]] = []
    for rank, record in enumerate(selected, start=1):
        writer.writerow(
            {
                "review_rank": str(rank),
                "output_png": seed_review_filename(record),
                "seed_risk_is_proxy": str(record.seed_risk_is_proxy).lower(),
                **record.to_csv_row(),
            }
        )
        entries.append(_json_entry(rank, record))

    canonical_entries = json.dumps(
        entries,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    document = {
        "schema_version": 2,
        "available_candidates": available_count,
        "target_count": target_count,
        "selected_reviews": len(selected),
        "shortfall": max(0, target_count - len(selected)),
        "unique_scenarios": len({record.scenario_id for record in selected}),
        "skill_counts": dict(sorted(Counter(record.skill_id for record in selected).items())),
        "selection_sha256": hashlib.sha256(canonical_entries).hexdigest(),
        "reviews": entries,
    }
    json_payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return csv_buffer.getvalue().encode("utf-8"), json_payload


def _check_existing_indices(
    payloads: Sequence[tuple[Path, bytes]],
    *,
    restart: bool,
) -> bool:
    matching_index_exists = False
    if restart:
        return False
    for path, expected in payloads:
        if not path.exists():
            continue
        if not path.is_file() or path.read_bytes() != expected:
            raise ValueError(
                f"existing review index differs from the current selection: {path}; "
                "rerun with --restart"
            )
        matching_index_exists = True
    return matching_index_exists


def _normalized_source_path(record: SeedRecord) -> PurePosixPath:
    source = PurePosixPath(record.source_path.replace("\\", "/"))
    if (
        source.is_absolute()
        or not source.parts
        or source.suffix.lower() != ".parquet"
        or any(part in {"", ".", ".."} for part in source.parts)
        or ":" in source.parts[0]
    ):
        raise ValueError(
            f"{record.scenario_id} source_path must be a relative parquet path "
            f"inside --data-root, got {record.source_path}"
        )
    return source


def _resolve_source_path(data_root: Path, record: SeedRecord) -> Path:
    root = data_root.resolve()
    source = _normalized_source_path(record)
    resolved = root.joinpath(*source.parts).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"{record.scenario_id} source_path escapes --data-root: {record.source_path}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"scenario file not found for {record.scenario_id}: {resolved}"
        )
    return resolved


def _valid_png(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
            return False
        with path.open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def _remove_stale_review_pngs(
    output_dir: Path,
    selected_names: set[str],
) -> int:
    if not output_dir.exists():
        return 0
    removed = 0
    for path in output_dir.glob("*.png"):
        if path.is_file() and path.name not in selected_names:
            path.unlink()
            removed += 1
    return removed


def run_review_rendering(
    *,
    candidate_csv: Path,
    data_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_count: int = 100,
    restart: bool = False,
    scenario_loader: ScenarioLoader = load_av2_scenario,
    renderer: ReviewRenderer = render_seed_review,
) -> dict[str, object]:
    """Select, render, and index one deterministic candidate review set."""

    records = read_seed_records(candidate_csv)
    if not records:
        raise ValueError(f"candidate CSV contains no records: {candidate_csv}")
    selected = select_stratified_review_records(records, target_count=target_count)

    output_names = [seed_review_filename(record) for record in selected]
    if len(output_names) != len(set(output_names)):
        raise ValueError("selected candidates produced duplicate review filenames")

    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"review output path is not a directory: {output_dir}")
    index_csv = output_dir / "review_index.csv"
    index_json = output_dir / "review_index.json"
    csv_payload, json_payload = _index_payloads(
        selected,
        available_count=len(records),
        target_count=target_count,
    )
    indexed_run = _check_existing_indices(
        ((index_csv, csv_payload), (index_json, json_payload)),
        restart=restart,
    )

    by_scenario: dict[str, list[SeedRecord]] = {}
    for record in selected:
        _normalized_source_path(record)
        by_scenario.setdefault(record.scenario_id, []).append(record)

    pending: dict[str, list[SeedRecord]] = {}
    source_paths: dict[str, Path] = {}
    skipped = 0
    for scenario_id, scenario_records in by_scenario.items():
        sources = {_normalized_source_path(record).as_posix() for record in scenario_records}
        if len(sources) != 1:
            raise ValueError(
                f"candidate records for {scenario_id} disagree on source_path: "
                f"{', '.join(sorted(sources))}"
            )
        scenario_pending = [
            record
            for record in scenario_records
            if restart
            or not indexed_run
            or not _valid_png(output_dir / seed_review_filename(record))
        ]
        skipped += len(scenario_records) - len(scenario_pending)
        if scenario_pending:
            pending[scenario_id] = scenario_pending
            source_paths[scenario_id] = _resolve_source_path(data_root, scenario_records[0])

    removed_stale = (
        _remove_stale_review_pngs(output_dir, set(output_names))
        if restart
        else 0
    )
    _atomic_write(index_csv, csv_payload)
    _atomic_write(index_json, json_payload)

    rendered = 0
    for scenario_id, scenario_records in pending.items():
        source_path = source_paths[scenario_id]
        try:
            scenario = scenario_loader(source_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load scenario {scenario_id} from {source_path}: {exc}"
            ) from exc
        if not isinstance(scenario, Scenario):
            raise TypeError(f"scenario loader returned {type(scenario).__name__} for {scenario_id}")
        if scenario.scenario_id != scenario_id:
            raise ValueError(
                f"loaded scenario_id={scenario.scenario_id} while candidate expects {scenario_id}"
            )
        for record in scenario_records:
            expected_output = output_dir / seed_review_filename(record)
            try:
                actual_output = Path(renderer(scenario, record, output_dir))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to render {record.skill_id} candidate in {scenario_id}: {exc}"
                ) from exc
            if actual_output.resolve() != expected_output.resolve():
                raise RuntimeError(
                    f"renderer returned unexpected path for {scenario_id}: {actual_output}"
                )
            if not _valid_png(expected_output):
                raise RuntimeError(f"renderer did not produce a valid PNG: {expected_output}")
            rendered += 1

    return {
        "available_candidates": len(records),
        "selected_reviews": len(selected),
        "unique_scenarios": len(by_scenario),
        "rendered_this_run": rendered,
        "resumed_reviews": skipped,
        "removed_stale_reviews": removed_stale,
        "output_dir": str(output_dir),
        "index_csv": str(index_csv),
        "index_json": str(index_json),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a stratified BEV review set from a seed candidate CSV."
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/av2/motion-forecasting"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Rerender all selected candidates and replace the existing indices.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        summary = run_review_rendering(
            candidate_csv=args.candidate_csv,
            data_root=args.data_root,
            output_dir=args.output_dir,
            target_count=args.target_count,
            restart=args.restart,
        )
    except KeyboardInterrupt:
        print("\nreview rendering interrupted; rerun the same command to resume", flush=True)
        raise SystemExit(130) from None
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        f"review complete: {summary['selected_reviews']} selected, "
        f"{summary['rendered_this_run']} rendered, "
        f"{summary['resumed_reviews']} resumed, output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()

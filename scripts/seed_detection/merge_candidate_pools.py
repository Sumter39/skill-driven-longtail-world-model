"""Merge disjoint formal seed scans into one candidate pool and checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from skilldrive.seeds import (
    SeedRecord,
    read_seed_records,
    sort_seed_records,
    write_seed_records,
)


CHECKPOINT_VERSION = 2
DEFAULT_EXPECTED_SCENARIOS = 20_000
_ACTOR_DISTRIBUTION_FIELDS = {
    "initiator",
    "responder",
    "pair",
    "by_role",
    "role_combination",
}


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validated_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _read_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_metadata(handle, path: Path) -> dict[str, Any]:
    raw = handle.readline()
    if not raw:
        raise ValueError(f"checkpoint is empty: {path}")
    metadata = _read_json_object(raw, label=f"checkpoint metadata: {path}")
    if metadata.get("kind") != "metadata":
        raise ValueError(f"invalid checkpoint metadata: {path}")
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint schema version must be {CHECKPOINT_VERSION}: {path}"
        )
    if metadata.get("scan_kind") != "formal":
        raise ValueError(f"checkpoint must come from a formal Train scan: {path}")
    _required_text(metadata.get("manifest_sha256"), f"manifest_sha256 in {path}")
    return metadata


def _read_entry(
    raw: bytes,
    *,
    path: Path,
    line_number: int,
    expected_index: int,
) -> dict[str, Any]:
    entry = _read_json_object(raw, label=f"checkpoint line {line_number}: {path}")
    if entry.get("kind") != "scenario":
        raise ValueError(f"invalid checkpoint line {line_number}: {path}")
    _required_text(
        entry.get("scenario_id"),
        f"scenario_id on checkpoint line {line_number}: {path}",
    )
    _required_text(
        entry.get("city_name"),
        f"city_name on checkpoint line {line_number}: {path}",
    )
    if entry.get("manifest_index") != expected_index:
        raise ValueError(
            f"checkpoint manifest_index must be {expected_index} on line "
            f"{line_number}: {path}"
        )
    if not isinstance(entry.get("records"), list):
        raise ValueError(f"checkpoint records missing on line {line_number}: {path}")
    if not isinstance(entry.get("rejection_counts"), dict):
        raise ValueError(
            f"checkpoint rejection_counts missing on line {line_number}: {path}"
        )
    distribution = entry.get("actor_distribution")
    if not isinstance(distribution, dict) or set(distribution) != _ACTOR_DISTRIBUTION_FIELDS:
        raise ValueError(
            f"checkpoint actor_distribution has invalid fields on line "
            f"{line_number}: {path}"
        )
    return entry


def _entry_skill_ids(entry: Mapping[str, Any], *, path: Path) -> set[str]:
    skill_ids: set[str] = set()
    for row in entry["records"]:
        if not isinstance(row, dict):
            raise ValueError(f"checkpoint record must be an object: {path}")
        skill_ids.add(_required_text(row.get("skill_id"), f"record skill_id in {path}"))
    for reason in entry["rejection_counts"]:
        reason_text = _required_text(reason, f"rejection key in {path}")
        skill_id, separator, _ = reason_text.partition(":")
        if not separator:
            raise ValueError(f"rejection key must start with skill_id: {reason_text}")
        skill_ids.add(_required_text(skill_id, f"rejection skill_id in {path}"))
    return skill_ids


def _metadata_skill_ids(
    metadata: Mapping[str, Any],
    derived_skill_ids: set[str],
    *,
    path: Path,
) -> list[str]:
    stored = metadata.get("selected_skill_ids")
    if stored is None:
        if not derived_skill_ids:
            raise ValueError(f"cannot infer selected skill IDs from legacy checkpoint: {path}")
        return sorted(derived_skill_ids)
    if (
        not isinstance(stored, list)
        or not stored
        or any(not isinstance(skill_id, str) or not skill_id for skill_id in stored)
        or len(set(stored)) != len(stored)
    ):
        raise ValueError(f"invalid selected_skill_ids in checkpoint metadata: {path}")
    if derived_skill_ids != set(stored):
        raise ValueError(
            f"checkpoint entries do not match selected_skill_ids in {path}: "
            f"derived={sorted(derived_skill_ids)} stored={stored}"
        )
    return list(stored)


def _inspect_checkpoint(
    path: Path,
    *,
    expected_scenarios: int,
) -> tuple[dict[str, Any], list[str]]:
    derived_skill_ids: set[str] = set()
    seen_scenario_ids: set[str] = set()
    scenario_count = 0
    with path.open("rb") as handle:
        metadata = _read_metadata(handle, path)
        declared_count = metadata.get("manifest_scenario_count")
        if declared_count is not None and declared_count != expected_scenarios:
            raise ValueError(
                f"checkpoint declares {declared_count} scenarios; "
                f"expected {expected_scenarios}: {path}"
            )
        for expected_index, raw in enumerate(handle):
            entry = _read_entry(
                raw,
                path=path,
                line_number=expected_index + 2,
                expected_index=expected_index,
            )
            scenario_id = entry["scenario_id"]
            if scenario_id in seen_scenario_ids:
                raise ValueError(f"duplicate scenario in checkpoint: {scenario_id}")
            seen_scenario_ids.add(scenario_id)
            derived_skill_ids.update(_entry_skill_ids(entry, path=path))
            scenario_count += 1
    if scenario_count != expected_scenarios:
        raise ValueError(
            f"checkpoint covers {scenario_count} scenarios; "
            f"expected {expected_scenarios}: {path}"
        )
    return metadata, _metadata_skill_ids(
        metadata,
        derived_skill_ids,
        path=path,
    )


def _records_by_scenario(
    records: Iterable[SeedRecord],
) -> dict[str, list[SeedRecord]]:
    grouped: dict[str, list[SeedRecord]] = {}
    for record in records:
        grouped.setdefault(record.scenario_id, []).append(record)
    return {
        scenario_id: sort_seed_records(group)
        for scenario_id, group in grouped.items()
    }


def _entry_records(entry: Mapping[str, Any], *, path: Path) -> list[SeedRecord]:
    try:
        return sort_seed_records(
            SeedRecord.from_csv_row(row) for row in entry["records"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid checkpoint records for {entry.get('scenario_id')} in {path}: {exc}"
        ) from exc


def _merge_count_trees(
    values: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be a mapping")
        for raw_key, item in value.items():
            key = _required_text(raw_key, f"{label} key")
            current = merged.get(key)
            if isinstance(item, Mapping):
                if current is not None and not isinstance(current, dict):
                    raise ValueError(f"{label}.{key} mixes mappings and counts")
                merged[key] = _merge_count_trees(
                    [current or {}, item],
                    label=f"{label}.{key}",
                )
            else:
                count = _validated_count(item, f"{label}.{key}")
                if isinstance(current, dict):
                    raise ValueError(f"{label}.{key} mixes mappings and counts")
                merged[key] = (current or 0) + count
    return {key: merged[key] for key in sorted(merged)}


def _merged_metadata(
    metadata: Sequence[Mapping[str, Any]],
    selected_skill_ids: Sequence[str],
    excluded_skill_ids: Sequence[str],
    candidate_paths: Sequence[Path],
    checkpoint_paths: Sequence[Path],
    expected_scenarios: int,
) -> dict[str, Any]:
    skill_payload = json.dumps(
        {
            "selected_skill_ids": list(selected_skill_ids),
            "source_skills_sha256": [item.get("skills_sha256") for item in metadata],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    result = {
        "kind": "metadata",
        "version": CHECKPOINT_VERSION,
        "scan_kind": "formal",
        "manifest_sha256": metadata[0]["manifest_sha256"],
        "manifest_scenario_count": expected_scenarios,
        "selected_skill_ids": list(selected_skill_ids),
        "excluded_zero_hit_skill_ids": list(excluded_skill_ids),
        "skills_sha256": hashlib.sha256(skill_payload).hexdigest(),
        "merge_kind": "candidate_pool_union",
        "source_candidate_pools": [str(path) for path in candidate_paths],
        "source_checkpoints": [str(path) for path in checkpoint_paths],
    }
    for key in (
        "config_sha256",
        "data_root",
        "internal_validation_sha256",
        "final_validation_sha256",
    ):
        values = [item.get(key) for item in metadata]
        if values[0] is not None and len(set(values)) == 1:
            result[key] = values[0]
    return result


def _validate_excluded_skill_ids(
    values: Sequence[str],
    available_skill_ids: Sequence[str],
) -> list[str]:
    excluded = list(values)
    if any(not isinstance(skill_id, str) or not skill_id for skill_id in excluded):
        raise ValueError("excluded_skill_ids must contain non-empty strings")
    if len(set(excluded)) != len(excluded):
        raise ValueError("excluded_skill_ids must not contain duplicates")
    unknown = sorted(set(excluded) - set(available_skill_ids))
    if unknown:
        raise ValueError(f"excluded skill IDs are not present in source scans: {unknown}")
    return excluded


def _without_excluded_rejections(
    rejection_counts: Mapping[str, Any],
    excluded_skill_ids: set[str],
) -> dict[str, Any]:
    return {
        reason: count
        for reason, count in rejection_counts.items()
        if reason.partition(":")[0] not in excluded_skill_ids
    }


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _validate_paths(
    candidate_paths: Sequence[Path],
    checkpoint_paths: Sequence[Path],
    output_csv: Path,
    output_checkpoint: Path,
) -> None:
    if len(candidate_paths) != 2 or len(checkpoint_paths) != 2:
        raise ValueError("exactly two candidate pools and two checkpoints are required")
    inputs = [*candidate_paths, *checkpoint_paths]
    if len({path.resolve() for path in inputs}) != len(inputs):
        raise ValueError("candidate pool and checkpoint input paths must be distinct")
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"merge input not found: {path}")
    output_paths = {output_csv.resolve(), output_checkpoint.resolve()}
    if len(output_paths) != 2:
        raise ValueError("output CSV and checkpoint paths must be distinct")
    overlap = output_paths & {path.resolve() for path in inputs}
    if overlap:
        raise ValueError("merge outputs must not overwrite source candidate pools or checkpoints")


def merge_candidate_pools(
    *,
    candidate_paths: Sequence[Path],
    checkpoint_paths: Sequence[Path],
    output_csv: Path,
    output_checkpoint: Path,
    output_summary_json: Path | None = None,
    excluded_skill_ids: Sequence[str] = (),
    expected_scenarios: int = DEFAULT_EXPECTED_SCENARIOS,
) -> dict[str, Any]:
    """Validate and merge two complete, skill-disjoint formal scans.

    Zero-hit rules may be excluded from the merged formal catalog without
    rescanning. Exclusion is rejected if either source CSV contains a record
    for the requested rule.
    """

    if (
        isinstance(expected_scenarios, bool)
        or not isinstance(expected_scenarios, int)
        or expected_scenarios <= 0
    ):
        raise ValueError("expected_scenarios must be a positive integer")
    candidate_paths = [Path(path) for path in candidate_paths]
    checkpoint_paths = [Path(path) for path in checkpoint_paths]
    output_csv = Path(output_csv)
    output_checkpoint = Path(output_checkpoint)
    output_summary_json = (
        None if output_summary_json is None else Path(output_summary_json)
    )
    _validate_paths(candidate_paths, checkpoint_paths, output_csv, output_checkpoint)
    if output_summary_json is not None and output_summary_json.resolve() in {
        *(path.resolve() for path in candidate_paths),
        *(path.resolve() for path in checkpoint_paths),
        output_csv.resolve(),
        output_checkpoint.resolve(),
    }:
        raise ValueError("summary JSON path must be distinct from merge inputs and outputs")

    checkpoint_info = [
        _inspect_checkpoint(path, expected_scenarios=expected_scenarios)
        for path in checkpoint_paths
    ]
    metadata = [item[0] for item in checkpoint_info]
    manifest_fingerprints = {item["manifest_sha256"] for item in metadata}
    if len(manifest_fingerprints) != 1:
        raise ValueError("checkpoint manifest fingerprints differ")

    source_skill_ids = [item[1] for item in checkpoint_info]
    overlap = set(source_skill_ids[0]) & set(source_skill_ids[1])
    if overlap:
        raise ValueError(f"checkpoint selected skill IDs overlap: {sorted(overlap)}")
    source_selected_skill_ids = [*source_skill_ids[0], *source_skill_ids[1]]
    excluded_skill_ids = _validate_excluded_skill_ids(
        excluded_skill_ids,
        source_selected_skill_ids,
    )
    excluded_skill_id_set = set(excluded_skill_ids)
    selected_skill_ids = [
        skill_id
        for skill_id in source_selected_skill_ids
        if skill_id not in excluded_skill_id_set
    ]
    if not selected_skill_ids:
        raise ValueError("at least one source skill must remain after exclusions")

    source_records = [read_seed_records(path) for path in candidate_paths]
    for path, records, allowed_ids in zip(
        candidate_paths,
        source_records,
        source_skill_ids,
    ):
        unknown = {record.skill_id for record in records} - set(allowed_ids)
        if unknown:
            raise ValueError(
                f"candidate pool contains skills outside its checkpoint in {path}: "
                f"{sorted(unknown)}"
            )
    excluded_record_skills = sorted(
        {
            record.skill_id
            for records in source_records
            for record in records
            if record.skill_id in excluded_skill_id_set
        }
    )
    if excluded_record_skills:
        raise ValueError(
            "excluded skills must have zero candidate records: "
            f"{excluded_record_skills}"
        )
    merged_records = sort_seed_records(
        record for records in source_records for record in records
    )
    expected_by_source = [_records_by_scenario(records) for records in source_records]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = _temporary_path(output_csv)
    checkpoint_temporary = _temporary_path(output_checkpoint)
    for temporary in (csv_temporary, checkpoint_temporary):
        if temporary.exists():
            temporary.unlink()

    try:
        merged_metadata = _merged_metadata(
            metadata,
            selected_skill_ids,
            excluded_skill_ids,
            candidate_paths,
            checkpoint_paths,
            expected_scenarios,
        )
        with (
            checkpoint_paths[0].open("rb") as first,
            checkpoint_paths[1].open("rb") as second,
            checkpoint_temporary.open("wb") as output,
        ):
            _read_metadata(first, checkpoint_paths[0])
            _read_metadata(second, checkpoint_paths[1])
            output.write(_json_line(merged_metadata))
            scenario_count = 0
            for expected_index, pair in enumerate(
                zip_longest(first, second, fillvalue=None)
            ):
                first_raw, second_raw = pair
                if first_raw is None or second_raw is None:
                    raise ValueError("checkpoint scenario counts differ")
                entries = [
                    _read_entry(
                        raw,
                        path=path,
                        line_number=expected_index + 2,
                        expected_index=expected_index,
                    )
                    for raw, path in zip(pair, checkpoint_paths)
                ]
                scenario_ids = [entry["scenario_id"] for entry in entries]
                if scenario_ids[0] != scenario_ids[1]:
                    raise ValueError(
                        f"checkpoint scenario order differs at index {expected_index}: "
                        f"{scenario_ids}"
                    )
                city_names = [entry["city_name"] for entry in entries]
                if city_names[0] != city_names[1]:
                    raise ValueError(
                        f"checkpoint city_name differs for {scenario_ids[0]}: "
                        f"{city_names}"
                    )

                records = [
                    _entry_records(entry, path=path)
                    for entry, path in zip(entries, checkpoint_paths)
                ]
                for source_index, checkpoint_records in enumerate(records):
                    expected_records = expected_by_source[source_index].pop(
                        scenario_ids[0], []
                    )
                    if checkpoint_records != expected_records:
                        raise ValueError(
                            f"checkpoint candidate records differ from "
                            f"{candidate_paths[source_index]} for {scenario_ids[0]}"
                        )
                combined_records = sort_seed_records(
                    record for group in records for record in group
                )
                elapsed_values = [
                    _nonnegative_number(
                        entry.get("elapsed_seconds", 0.0),
                        f"elapsed_seconds for {scenario_ids[0]}",
                    )
                    for entry in entries
                ]
                peak_values = [
                    _nonnegative_number(
                        entry.get("peak_memory_mib", 0.0),
                        f"peak_memory_mib for {scenario_ids[0]}",
                    )
                    for entry in entries
                ]
                merged_entry = {
                    "kind": "scenario",
                    "manifest_index": expected_index,
                    "scenario_id": scenario_ids[0],
                    "city_name": city_names[0],
                    "elapsed_seconds": sum(elapsed_values),
                    "peak_memory_mib": max(peak_values),
                    "records": [record.to_csv_row() for record in combined_records],
                    "rejection_counts": _merge_count_trees(
                        [
                            _without_excluded_rejections(
                                entry["rejection_counts"],
                                excluded_skill_id_set,
                            )
                            for entry in entries
                        ],
                        label=f"rejection_counts for {scenario_ids[0]}",
                    ),
                    "actor_distribution": _merge_count_trees(
                        [entry["actor_distribution"] for entry in entries],
                        label=f"actor_distribution for {scenario_ids[0]}",
                    ),
                }
                output.write(_json_line(merged_entry))
                scenario_count += 1
            if scenario_count != expected_scenarios:
                raise ValueError(
                    f"merged checkpoint covers {scenario_count} scenarios; "
                    f"expected {expected_scenarios}"
                )
            output.flush()
            os.fsync(output.fileno())

        leftovers = [sorted(group)[:5] for group in expected_by_source if group]
        if leftovers:
            raise ValueError(
                f"candidate pools contain scenarios outside their checkpoints: {leftovers}"
            )

        write_seed_records(csv_temporary, merged_records)
        with csv_temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(checkpoint_temporary, output_checkpoint)
        os.replace(csv_temporary, output_csv)
    finally:
        for temporary in (csv_temporary, checkpoint_temporary):
            if temporary.exists():
                temporary.unlink()

    skill_hits = Counter(record.skill_id for record in merged_records)
    skill_scenario_hits: Counter[str] = Counter()
    for records in _records_by_scenario(merged_records).values():
        skill_scenario_hits.update({record.skill_id for record in records})
    summary = {
        "schema_version": 1,
        "status": "complete",
        "summary_kind": "merged_formal_candidate_pool",
        "merge": {
            "method": "skill_disjoint_candidate_pool_union",
            "source_candidate_pools": [str(path) for path in candidate_paths],
            "source_checkpoints": [str(path) for path in checkpoint_paths],
            "selected_skill_ids": selected_skill_ids,
            "excluded_zero_hit_skill_ids": excluded_skill_ids,
        },
        "counts": {
            "processed_scenarios": expected_scenarios,
            "candidates": len(merged_records),
            "unique_candidate_scenarios": len(
                {record.scenario_id for record in merged_records}
            ),
            "covered_skills": len(skill_hits),
            "configured_skills": len(selected_skill_ids),
        },
        "skill_hits": {
            skill_id: skill_hits.get(skill_id, 0) for skill_id in selected_skill_ids
        },
        "skill_scenario_hits": {
            skill_id: skill_scenario_hits.get(skill_id, 0)
            for skill_id in selected_skill_ids
        },
        "outputs": {
            "candidate_csv": str(output_csv),
            "candidate_csv_sha256": _sha256(output_csv),
            "checkpoint": str(output_checkpoint),
            "checkpoint_sha256": _sha256(output_checkpoint),
        },
    }
    if output_summary_json is not None:
        _atomic_write_json(output_summary_json, summary)

    return {
        "scenario_count": expected_scenarios,
        "candidate_count": len(merged_records),
        "selected_skill_ids": selected_skill_ids,
        "excluded_skill_ids": excluded_skill_ids,
        "candidate_csv": str(output_csv),
        "checkpoint": str(output_checkpoint),
        "summary_json": (
            None if output_summary_json is None else str(output_summary_json)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge two complete, skill-disjoint formal seed scans."
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        action="append",
        required=True,
        help="Source candidate CSV. Provide exactly twice, in checkpoint order.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Source complete checkpoint. Provide exactly twice.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path)
    parser.add_argument(
        "--exclude-skill-id",
        action="append",
        default=[],
        help=(
            "Exclude a zero-hit rule from the merged formal catalog and "
            "checkpoint. Repeat for multiple rules."
        ),
    )
    parser.add_argument(
        "--expected-scenarios",
        type=int,
        default=DEFAULT_EXPECTED_SCENARIOS,
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if len(args.candidate_pool) != 2 or len(args.checkpoint) != 2:
        parser.error("provide exactly two --candidate-pool and two --checkpoint values")
    result = merge_candidate_pools(
        candidate_paths=args.candidate_pool,
        checkpoint_paths=args.checkpoint,
        output_csv=args.output_csv,
        output_checkpoint=args.output_checkpoint,
        output_summary_json=args.output_summary_json,
        excluded_skill_ids=args.exclude_skill_id,
        expected_scenarios=args.expected_scenarios,
    )
    print(
        f"merge complete: {result['scenario_count']} scenarios, "
        f"{result['candidate_count']} candidates, "
        f"{len(result['selected_skill_ids'])} skills"
    )


if __name__ == "__main__":
    main()

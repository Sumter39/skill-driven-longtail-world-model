"""Select the final deterministic 5,000-scenario formal seed set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from skilldrive.data.manifests import ManifestRow, read_manifest
from skilldrive.seeds import (
    SEED_CSV_FIELDS,
    SeedRecord,
    read_seed_records,
    sort_seed_records,
    write_seed_records,
)
from skilldrive.seeds.selection import select_seed_records


DEFAULT_FORMAL_MANIFEST = Path("manifests/splits/formal_train.csv")
DEFAULT_INTERNAL_VALIDATION = Path("manifests/splits/internal_validation.csv")
DEFAULT_FINAL_VALIDATION = Path("manifests/splits/final_validation.csv")
DEFAULT_CANDIDATE_POOL = Path("outputs/seed_detection/formal_candidate_pool.csv")
DEFAULT_CHECKPOINT = Path("outputs/seed_detection/formal_pool_summary.checkpoint.jsonl")
DEFAULT_OUTPUT_CSV = Path("manifests/seeds/formal_candidates.csv")
DEFAULT_SUMMARY_JSON = Path("outputs/seed_detection/formal_summary.json")
DEFAULT_TARGET_SCENARIOS = 5_000
DEFAULT_SEED = 2026
CHECKPOINT_VERSION = 2

_MANIFEST_CONTRACTS = {
    "formal_train.csv": ("train", "train", 20_000),
    "internal_validation.csv": ("internal_validation", "train", 2_000),
    "final_validation.csv": ("validation", "val", 5_000),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(rows: Sequence[ManifestRow]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_write_seed_csv(path: Path, records: Sequence[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_seed_records(temporary, records)
        if read_seed_records(temporary) != list(records):
            raise RuntimeError("selected seed CSV failed its schema round trip")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)


def _required_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _validate_manifest(path: Path, expected_name: str) -> list[ManifestRow]:
    if path.name != expected_name:
        raise ValueError(f"expected {expected_name}, got {path.name}")
    rows = read_manifest(path)
    if not rows:
        raise ValueError(f"{expected_name} must contain at least one scenario")

    expected_split, source_split, canonical_count = _MANIFEST_CONTRACTS[expected_name]
    project_root = Path(__file__).resolve().parents[2]
    canonical_path = project_root / "manifests" / "splits" / expected_name
    if path.resolve() == canonical_path.resolve() and len(rows) != canonical_count:
        raise ValueError(
            f"canonical {expected_name} must contain {canonical_count} scenarios, "
            f"got {len(rows)}"
        )

    seen: set[str] = set()
    for row in rows:
        for field in ("scenario_id", "split", "source_path", "city_name", "selected_reason"):
            _required_text(getattr(row, field), f"{expected_name} {field}")
        if row.scenario_id in seen:
            raise ValueError(f"duplicate scenario_id in {expected_name}: {row.scenario_id}")
        seen.add(row.scenario_id)
        if row.split != expected_split:
            raise ValueError(
                f"{expected_name} may only contain split={expected_split}, "
                f"got {row.split} for {row.scenario_id}"
            )
        source = PurePosixPath(row.source_path.replace("\\", "/"))
        expected_parts = (
            source_split,
            row.scenario_id,
            f"scenario_{row.scenario_id}.parquet",
        )
        if source.is_absolute() or source.parts != expected_parts:
            raise ValueError(
                f"{row.scenario_id} source_path must be {'/'.join(expected_parts)}, "
                f"got {row.source_path}"
            )
    return rows


def _validate_manifests(
    formal_path: Path,
    internal_path: Path,
    final_path: Path,
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    formal = _validate_manifest(formal_path, "formal_train.csv")
    internal = _validate_manifest(internal_path, "internal_validation.csv")
    final = _validate_manifest(final_path, "final_validation.csv")
    named_ids = {
        "formal_train": {row.scenario_id for row in formal},
        "internal_validation": {row.scenario_id for row in internal},
        "final_validation": {row.scenario_id for row in final},
    }
    pairs = (
        ("formal_train", "internal_validation"),
        ("formal_train", "final_validation"),
        ("internal_validation", "final_validation"),
    )
    for first, second in pairs:
        overlap = named_ids[first] & named_ids[second]
        if overlap:
            raise ValueError(
                f"scenario leakage between {first} and {second}: "
                f"{', '.join(sorted(overlap)[:5])}"
            )
    return formal, internal, final


def _parse_checkpoint(
    path: Path | None,
    *,
    formal_manifest_sha256: str,
    formal_ids: set[str],
    selected_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None:
        return {}, {"status": "disabled", "path": None}
    if not path.exists():
        raise FileNotFoundError(f"formal scan checkpoint not found: {path}")

    entries: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    scenario_count = 0
    with path.open("rb") as handle:
        metadata_line = handle.readline()
        if not metadata_line:
            raise ValueError(f"checkpoint is empty: {path}")
        try:
            metadata = json.loads(metadata_line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint metadata: {path}") from exc
        if not isinstance(metadata, dict) or metadata.get("kind") != "metadata":
            raise ValueError(f"invalid checkpoint metadata: {path}")
        if metadata.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"checkpoint schema version must be {CHECKPOINT_VERSION}")
        if metadata.get("scan_kind") != "formal":
            raise ValueError("checkpoint must come from a formal Train scan")
        if metadata.get("manifest_sha256") != formal_manifest_sha256:
            raise ValueError("checkpoint manifest fingerprint differs from formal_train.csv")

        for line_number, raw in enumerate(handle, start=2):
            try:
                entry = json.loads(raw.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid checkpoint line {line_number}: {path}") from exc
            if not isinstance(entry, dict) or entry.get("kind") != "scenario":
                raise ValueError(f"invalid checkpoint line {line_number}: {path}")
            scenario_id = entry.get("scenario_id")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError(f"invalid checkpoint scenario_id on line {line_number}")
            if scenario_id in seen:
                raise ValueError(f"duplicate scenario in checkpoint: {scenario_id}")
            if scenario_id not in formal_ids:
                raise ValueError(f"checkpoint contains scenario outside formal_train: {scenario_id}")
            seen.add(scenario_id)
            scenario_count += 1
            if scenario_id in selected_ids:
                entries[scenario_id] = entry
    if scenario_count != len(formal_ids):
        raise ValueError(
            f"checkpoint covers {scenario_count} formal scenarios; "
            f"expected {len(formal_ids)}"
        )

    return entries, {
        "status": "available",
        "path": str(path),
        "scenario_count": scenario_count,
        "sha256": _sha256(path),
    }


def _records_by_scenario(records: Iterable[SeedRecord]) -> dict[str, list[SeedRecord]]:
    grouped: dict[str, list[SeedRecord]] = {}
    for record in records:
        grouped.setdefault(record.scenario_id, []).append(record)
    return {
        scenario_id: sort_seed_records(group)
        for scenario_id, group in grouped.items()
    }


def _validated_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _merge_actor_distributions(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {name: Counter() for name in ("initiator", "responder", "pair")}
    role_totals: dict[str, Counter[str]] = {}
    role_combinations: Counter[str] = Counter()
    for entry in entries:
        distribution = entry.get("actor_distribution")
        if not isinstance(distribution, Mapping):
            raise ValueError(
                f"checkpoint actor_distribution is missing for {entry.get('scenario_id')}"
            )
        for name, counter in totals.items():
            values = distribution.get(name, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"checkpoint actor_distribution.{name} must be a mapping")
            for key, value in values.items():
                _required_text(key, f"actor_distribution.{name} key")
                counter[key] += _validated_count(value, f"actor_distribution.{name}.{key}")
        by_role = distribution.get("by_role", {})
        if not isinstance(by_role, Mapping):
            raise ValueError("checkpoint actor_distribution.by_role must be a mapping")
        for role, values in by_role.items():
            _required_text(role, "actor_distribution.by_role key")
            if not isinstance(values, Mapping):
                raise ValueError(f"actor_distribution.by_role.{role} must be a mapping")
            target = role_totals.setdefault(role, Counter())
            for actor_type, value in values.items():
                _required_text(actor_type, f"actor_distribution.by_role.{role} key")
                target[actor_type] += _validated_count(
                    value,
                    f"actor_distribution.by_role.{role}.{actor_type}",
                )
        combinations = distribution.get("role_combination", {})
        if not isinstance(combinations, Mapping):
            raise ValueError("checkpoint actor_distribution.role_combination must be a mapping")
        for name, value in combinations.items():
            _required_text(name, "actor_distribution.role_combination key")
            role_combinations[name] += _validated_count(
                value,
                f"actor_distribution.role_combination.{name}",
            )

    result: dict[str, Any] = {
        name: dict(sorted(counter.items())) for name, counter in totals.items()
    }
    result["by_role"] = {
        role: dict(sorted(counter.items()))
        for role, counter in sorted(role_totals.items())
    }
    result["role_combination"] = dict(sorted(role_combinations.items()))
    return result


def _checkpoint_enrichment(
    entries: Mapping[str, Mapping[str, Any]],
    selected_by_scenario: Mapping[str, list[SeedRecord]],
    formal_by_id: Mapping[str, ManifestRow],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    available: list[Mapping[str, Any]] = []
    city_counts: Counter[str] = Counter()
    city_label_counts: Counter[str] = Counter()
    missing_city: list[str] = []
    checkpoint_matches = 0
    for scenario_id, expected_records in sorted(selected_by_scenario.items()):
        entry = entries.get(scenario_id)
        if entry is None:
            city = formal_by_id[scenario_id].city_name
            if city and city != "unknown_until_loaded":
                city_counts[city] += 1
                city_label_counts[city] += len(expected_records)
            else:
                missing_city.append(scenario_id)
            continue
        checkpoint_rows = entry.get("records")
        if not isinstance(checkpoint_rows, list):
            raise ValueError(f"checkpoint records missing for {scenario_id}")
        checkpoint_records = sort_seed_records(
            SeedRecord.from_csv_row(row) for row in checkpoint_rows
        )
        if checkpoint_records != expected_records:
            raise ValueError(
                f"checkpoint candidate records differ from formal_candidate_pool.csv "
                f"for {scenario_id}"
            )
        city = entry.get("city_name")
        _required_text(city, f"checkpoint city_name for {scenario_id}")
        city_counts[city] += 1
        city_label_counts[city] += len(expected_records)
        available.append(entry)
        checkpoint_matches += 1

    coverage = {
        "selected_scenarios_matched_in_checkpoint": checkpoint_matches,
        "selected_scenarios_with_city": len(selected_by_scenario) - len(missing_city),
        "selected_scenarios_without_city": len(missing_city),
        "missing_city_scenario_ids_sample": missing_city[:10],
    }
    cities = {
        "selected_scenarios": dict(sorted(city_counts.items())),
        "selected_labels": dict(sorted(city_label_counts.items())),
    }
    actors = _merge_actor_distributions(available) if available else {
        "initiator": {},
        "responder": {},
        "pair": {},
        "by_role": {},
        "role_combination": {},
    }
    return coverage, cities, actors


def _target_risk_definitions(records: Sequence[SeedRecord]) -> dict[str, dict[str, Any]]:
    definitions: dict[str, dict[str, Any]] = {}
    for record in records:
        previous = definitions.setdefault(record.skill_id, record.target_risk_definition)
        if previous != record.target_risk_definition:
            raise ValueError(
                f"inconsistent target_risk_definition for skill {record.skill_id}"
            )
    return {skill_id: definitions[skill_id] for skill_id in sorted(definitions)}


def run_selection(
    *,
    formal_manifest_path: Path = DEFAULT_FORMAL_MANIFEST,
    internal_validation_manifest: Path = DEFAULT_INTERNAL_VALIDATION,
    final_validation_manifest: Path = DEFAULT_FINAL_VALIDATION,
    candidate_pool_path: Path = DEFAULT_CANDIDATE_POOL,
    checkpoint_path: Path | None = DEFAULT_CHECKPOINT,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    target_scenario_count: int = DEFAULT_TARGET_SCENARIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Validate a formal candidate pool and select exactly the requested scenarios."""

    if (
        isinstance(target_scenario_count, bool)
        or not isinstance(target_scenario_count, int)
        or target_scenario_count < 1
    ):
        raise ValueError("target_scenario_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if candidate_pool_path.resolve() == output_csv.resolve():
        raise ValueError("candidate pool and final output CSV must be different files")
    if summary_json.resolve() in {
        candidate_pool_path.resolve(),
        output_csv.resolve(),
    }:
        raise ValueError("summary JSON must be separate from the candidate CSV files")
    formal, internal, final = _validate_manifests(
        formal_manifest_path,
        internal_validation_manifest,
        final_validation_manifest,
    )
    formal_by_id = {row.scenario_id: row for row in formal}
    formal_ids = set(formal_by_id)
    internal_ids = {row.scenario_id for row in internal}
    final_ids = {row.scenario_id for row in final}

    pool_records = read_seed_records(candidate_pool_path)
    if not pool_records:
        raise ValueError("formal candidate pool must contain at least one record")
    pool_by_scenario = _records_by_scenario(pool_records)
    pool_ids = set(pool_by_scenario)
    internal_overlap = pool_ids & internal_ids
    final_overlap = pool_ids & final_ids
    if internal_overlap:
        raise ValueError(
            "formal candidate pool leaks internal_validation scenarios: "
            + ", ".join(sorted(internal_overlap)[:5])
        )
    if final_overlap:
        raise ValueError(
            "formal candidate pool leaks final_validation scenarios: "
            + ", ".join(sorted(final_overlap)[:5])
        )
    outside_formal = pool_ids - formal_ids
    if outside_formal:
        raise ValueError(
            "formal candidate pool contains scenarios outside formal_train: "
            + ", ".join(sorted(outside_formal)[:5])
        )
    for record in pool_records:
        expected_source = formal_by_id[record.scenario_id].source_path
        if record.source_path != expected_source:
            raise ValueError(
                f"candidate source_path differs from formal_train for {record.scenario_id}"
            )
    _target_risk_definitions(pool_records)

    if len(pool_ids) < target_scenario_count:
        raise ValueError(
            f"formal candidate pool has only {len(pool_ids)} unique scenarios; "
            f"cannot select {target_scenario_count}"
        )
    selected = select_seed_records(
        pool_records,
        target_scenario_count,
        seed=seed,
    )
    reversed_selection = select_seed_records(
        reversed(pool_records),
        target_scenario_count,
        seed=seed,
    )
    if selected != reversed_selection:
        raise RuntimeError("seed selection depends on candidate input order")
    selected_by_scenario = _records_by_scenario(selected)
    selected_ids = set(selected_by_scenario)
    if len(selected_ids) != target_scenario_count:
        raise RuntimeError(
            f"selection returned {len(selected_ids)} unique scenarios, "
            f"expected {target_scenario_count}"
        )
    expected_selected = sort_seed_records(
        record for record in pool_records if record.scenario_id in selected_ids
    )
    if selected != expected_selected:
        raise RuntimeError("selection did not retain every label for selected scenarios")
    if selected_ids & internal_ids or selected_ids & final_ids:
        raise RuntimeError("selected scenarios overlap an excluded validation manifest")

    checkpoint_entries, checkpoint_audit = _parse_checkpoint(
        checkpoint_path,
        formal_manifest_sha256=_manifest_fingerprint(formal),
        formal_ids=formal_ids,
        selected_ids=selected_ids,
    )
    enrichment_coverage, city_distribution, actor_distribution = _checkpoint_enrichment(
        checkpoint_entries,
        selected_by_scenario,
        formal_by_id,
    )

    _atomic_write_seed_csv(output_csv, selected)
    if read_seed_records(output_csv) != selected:
        raise RuntimeError("final seed CSV failed deterministic schema verification")

    skill_hits = Counter(record.skill_id for record in selected)
    skill_scenario_hits: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for scenario_records in selected_by_scenario.values():
        skill_scenario_hits.update({record.skill_id for record in scenario_records})
        for record in scenario_records:
            relation_counts[
                "proxy_metric" if record.seed_risk_is_proxy else "target_metric_observation"
            ] += 1

    manifest_inputs = {
        "formal_train": (formal_manifest_path, formal),
        "internal_validation": (internal_validation_manifest, internal),
        "final_validation": (final_validation_manifest, final),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "selection": {
            "target_unique_scenarios": target_scenario_count,
            "seed": seed,
            "method": "skill_seed_metric_risk_quartile_round_robin",
            "retain_all_labels_for_selected_scenarios": True,
        },
        "inputs": {
            name: {
                "path": str(path),
                "scenario_count": len(rows),
                "sha256": _sha256(path),
            }
            for name, (path, rows) in manifest_inputs.items()
        },
        "candidate_pool": {
            "path": str(candidate_pool_path),
            "record_count": len(pool_records),
            "unique_scenario_count": len(pool_ids),
            "sha256": _sha256(candidate_pool_path),
        },
        "checkpoint_enrichment": {
            **checkpoint_audit,
            **enrichment_coverage,
        },
        "outputs": {
            "candidate_csv": str(output_csv),
            "summary_json": str(summary_json),
            "candidate_csv_sha256": _sha256(output_csv),
        },
        "counts": {
            "selected_records": len(selected),
            "selected_unique_scenarios": len(selected_ids),
        },
        "leakage_check": {
            "status": "passed",
            "candidate_pool_outside_formal_train": 0,
            "candidate_pool_internal_validation_overlap": 0,
            "candidate_pool_final_validation_overlap": 0,
            "selected_internal_validation_overlap": 0,
            "selected_final_validation_overlap": 0,
        },
        "schema_check": {
            "status": "passed",
            "candidate_csv_fields": list(SEED_CSV_FIELDS),
            "round_trip_record_count": len(selected),
        },
        "determinism_check": {
            "status": "passed",
            "reversed_input_selection_matches": True,
            "canonical_record_order": True,
        },
        "skill_distribution": {
            "records": dict(sorted(skill_hits.items())),
            "scenarios": dict(sorted(skill_scenario_hits.items())),
        },
        "seed_risk_relation_counts": {
            name: relation_counts.get(name, 0)
            for name in ("target_metric_observation", "proxy_metric")
        },
        "target_risk_definitions": _target_risk_definitions(selected),
        "city_distribution": city_distribution,
        "actor_distribution": actor_distribution,
    }
    _atomic_write_json(summary_json, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select exactly 5,000 unique formal seed scenarios from the scan pool."
    )
    parser.add_argument("--formal-manifest", type=Path, default=DEFAULT_FORMAL_MANIFEST)
    parser.add_argument(
        "--internal-validation-manifest",
        type=Path,
        default=DEFAULT_INTERNAL_VALIDATION,
    )
    parser.add_argument(
        "--final-validation-manifest",
        type=Path,
        default=DEFAULT_FINAL_VALIDATION,
    )
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--target-scenarios",
        type=int,
        default=DEFAULT_TARGET_SCENARIOS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_selection(
        formal_manifest_path=args.formal_manifest,
        internal_validation_manifest=args.internal_validation_manifest,
        final_validation_manifest=args.final_validation_manifest,
        candidate_pool_path=args.candidate_pool,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        target_scenario_count=args.target_scenarios,
        seed=args.seed,
    )
    print(
        "selection complete: "
        f"{summary['counts']['selected_unique_scenarios']} scenarios, "
        f"{summary['counts']['selected_records']} labels, "
        f"output={summary['outputs']['candidate_csv']}"
    )


if __name__ == "__main__":
    main()

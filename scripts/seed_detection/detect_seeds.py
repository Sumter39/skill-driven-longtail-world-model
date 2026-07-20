"""Scan approved AV2 Train manifests for deterministic skill seed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tracemalloc
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import yaml

from skilldrive.data.av2_reader import (
    load_av2_scenario,
    preload_av2_dependencies,
    preload_av2_worker_dependencies,
)
from skilldrive.data.manifests import ManifestRow, read_manifest
from skilldrive.schemas import Scenario, SkillSpec
from skilldrive.seeds import SeedRecord, sort_seed_records, write_seed_records
from skilldrive.skills.detection import (
    DetectionConfig,
    DetectionRun,
    load_detection_config,
)
from skilldrive.skills.loader import load_skill


CHECKPOINT_VERSION = 2
ALLOWED_MANIFESTS = {
    "development_train.csv": ("development", "development_train"),
    "formal_train.csv": ("formal", "train"),
}
CANONICAL_MANIFESTS = {
    "development_train.csv": Path("manifests/development/development_train.csv"),
    "formal_train.csv": Path("manifests/splits/formal_train.csv"),
}
DEFAULT_OUTPUTS = {
    "development": (
        Path("outputs/seed_detection/development_candidate_pool.csv"),
        Path("outputs/seed_detection/development_summary.json"),
    ),
    "formal": (
        Path("outputs/seed_detection/formal_candidate_pool.csv"),
        Path("outputs/seed_detection/formal_pool_summary.json"),
    ),
}
EXPECTED_SCENARIO_COUNTS = {"development": 500, "formal": 20_000}
DEFAULT_EXCLUSION_MANIFESTS = {
    "internal_validation": Path("manifests/splits/internal_validation.csv"),
    "final_validation": Path("manifests/splits/final_validation.csv"),
}
PIPELINE_FILES = (
    "scripts/seed_detection/detect_seeds.py",
    "skilldrive/data/av2_reader.py",
    "skilldrive/data/manifests.py",
    "skilldrive/schemas/core.py",
    "skilldrive/seeds/records.py",
    "skilldrive/seeds/sampling.py",
    "skilldrive/skills/detection.py",
    "skilldrive/skills/geometry.py",
    "skilldrive/skills/loader.py",
    "skilldrive/skills/registry.py",
)

ScenarioLoader = Callable[[str | Path], Scenario]
Detector = Callable[[Scenario, Sequence[SkillSpec], DetectionConfig], DetectionRun]
_ORIGINAL_AV2_SCENARIO_LOADER = load_av2_scenario


class _PeakMemoryTracker:
    """Read peak resident memory on WSL, with a portable Python fallback."""

    def __init__(self) -> None:
        try:
            import resource
        except ImportError:
            resource = None
        self._resource = resource
        if resource is None:
            tracemalloc.start()

    @property
    def method(self) -> str:
        return "peak_rss" if self._resource is not None else "tracemalloc"

    def peak_mib(self) -> float:
        if self._resource is not None:
            value = float(
                self._resource.getrusage(self._resource.RUSAGE_SELF).ru_maxrss
            )
            if sys.platform == "darwin":
                value /= 1024.0
            return value / 1024.0
        return tracemalloc.get_traced_memory()[1] / (1024.0 * 1024.0)

    def close(self) -> None:
        if self._resource is None and tracemalloc.is_tracing():
            tracemalloc.stop()


def _scan_kind(manifest_path: Path) -> str:
    try:
        return ALLOWED_MANIFESTS[manifest_path.name][0]
    except KeyError:
        allowed = ", ".join(sorted(ALLOWED_MANIFESTS))
        raise ValueError(
            f"manifest must be one of {allowed}; validation manifests are forbidden"
        ) from None


def _validate_manifest_scope(
    manifest_path: Path, rows: Sequence[ManifestRow]
) -> str:
    kind = _scan_kind(manifest_path)
    expected_split = ALLOWED_MANIFESTS[manifest_path.name][1]
    if not rows:
        raise ValueError("manifest must contain at least one scenario")
    project_root = Path(__file__).resolve().parents[2]
    canonical_path = project_root / CANONICAL_MANIFESTS[manifest_path.name]
    expected_count = EXPECTED_SCENARIO_COUNTS[kind]
    if manifest_path.resolve() == canonical_path.resolve() and len(rows) != expected_count:
        raise ValueError(
            f"canonical {manifest_path.name} must contain {expected_count} scenarios, "
            f"got {len(rows)}"
        )

    seen: set[str] = set()
    for row in rows:
        if row.scenario_id in seen:
            raise ValueError(f"duplicate scenario_id in manifest: {row.scenario_id}")
        seen.add(row.scenario_id)
        if row.split != expected_split:
            raise ValueError(
                f"{manifest_path.name} may only contain split={expected_split}, "
                f"got {row.split} for {row.scenario_id}"
            )
        source = PurePosixPath(row.source_path.replace("\\", "/"))
        expected_parts = (
            "train",
            row.scenario_id,
            f"scenario_{row.scenario_id}.parquet",
        )
        if source.is_absolute() or source.parts != expected_parts:
            raise ValueError(
                f"{row.scenario_id} source_path must be "
                f"{'/'.join(expected_parts)}, got {row.source_path}"
            )
    return kind


def _validate_exclusion_manifests(
    rows: Sequence[ManifestRow],
    exclusion_manifests: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    candidate_ids = {row.scenario_id for row in rows}
    audit: dict[str, dict[str, Any]] = {}
    for name, path in sorted(exclusion_manifests.items()):
        excluded_rows = read_manifest(path)
        excluded_ids = {row.scenario_id for row in excluded_rows}
        overlap = candidate_ids & excluded_ids
        if overlap:
            raise ValueError(
                f"scenario leakage with {name}: {', '.join(sorted(overlap)[:5])}"
            )
        audit[name] = {
            "path": str(path),
            "scenario_count": len(excluded_ids),
            "sha256": _manifest_fingerprint(excluded_rows),
        }
    return audit


def _load_confirmed_skills(
    skill_dir: Path,
    selected_skill_ids: Sequence[str] | None = None,
) -> list[SkillSpec]:
    catalog_path = skill_dir / "catalog.yaml"
    with catalog_path.open(encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle)
    if not isinstance(catalog, dict) or catalog.get("status") != "user_confirmed":
        raise ValueError("skill catalog must have status=user_confirmed")
    families = catalog.get("families")
    if not isinstance(families, dict):
        raise ValueError("skill catalog families must be a mapping")

    entries: list[Mapping[str, Any]] = []
    for family_name, family_entries in families.items():
        if not isinstance(family_entries, list):
            raise ValueError(f"skill catalog family {family_name} must be a list")
        for entry in family_entries:
            if not isinstance(entry, dict):
                raise ValueError("skill catalog entries must be mappings")
            entries.append(entry)

    catalog_skill_ids = [entry.get("skill_id") for entry in entries]
    if (
        len(catalog_skill_ids) < 30
        or any(not isinstance(skill_id, str) or not skill_id for skill_id in catalog_skill_ids)
        or len(set(catalog_skill_ids)) != len(catalog_skill_ids)
    ):
        raise ValueError("confirmed skill catalog must contain at least 30 unique skills")

    if selected_skill_ids is None:
        skill_ids = catalog_skill_ids
    else:
        requested = list(selected_skill_ids)
        if not requested:
            raise ValueError("selected_skill_ids must contain at least one skill ID")
        if any(not isinstance(skill_id, str) or not skill_id for skill_id in requested):
            raise ValueError("selected_skill_ids must contain non-empty strings")
        if len(set(requested)) != len(requested):
            raise ValueError("selected_skill_ids must not contain duplicates")
        unknown = sorted(set(requested) - set(catalog_skill_ids))
        if unknown:
            raise ValueError(
                f"selected skill IDs are not in the confirmed catalog: {unknown}"
            )
        requested_set = set(requested)
        skill_ids = [
            skill_id for skill_id in catalog_skill_ids if skill_id in requested_set
        ]

    skills = [load_skill(skill_dir / f"{skill_id}.yaml") for skill_id in skill_ids]
    if [skill.skill_id for skill in skills] != skill_ids:
        raise ValueError("skill YAML IDs do not match the confirmed catalog")
    return skills


def _hash_payloads(payloads: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_fingerprint(rows: Sequence[ManifestRow]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _skills_fingerprint(directory: Path, skill_ids: Sequence[str]) -> str:
    if not skill_ids:
        raise ValueError("skill_ids must contain at least one skill ID")
    paths = [directory / f"{skill_id}.yaml" for skill_id in skill_ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing selected skill YAML files: {missing}")
    return _hash_payloads(
        [(path.relative_to(directory).as_posix(), path.read_bytes()) for path in paths]
    )


def _pipeline_fingerprint(project_root: Path) -> str:
    payloads: list[tuple[str, bytes]] = []
    for relative_path in PIPELINE_FILES:
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing seed detection pipeline file: {path}")
        payloads.append((relative_path, path.read_bytes()))
    return _hash_payloads(payloads)


def _checkpoint_metadata(
    *,
    kind: str,
    rows: Sequence[ManifestRow],
    data_root: Path,
    config_path: Path,
    skill_dir: Path,
    selected_skill_ids: Sequence[str],
    exclusion_audit: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    metadata = {
        "kind": "metadata",
        "version": CHECKPOINT_VERSION,
        "scan_kind": kind,
        "manifest_sha256": _manifest_fingerprint(rows),
        "manifest_scenario_count": len(rows),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "selected_skill_ids": list(selected_skill_ids),
        "skills_sha256": _skills_fingerprint(skill_dir, selected_skill_ids),
        "pipeline_sha256": _pipeline_fingerprint(project_root),
        "data_root": str(data_root.resolve()),
    }
    for name, audit in exclusion_audit.items():
        metadata[f"{name}_sha256"] = audit["sha256"]
    return metadata


def _json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")


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


def _load_checkpoint(
    path: Path,
    expected_metadata: Mapping[str, Any],
    *,
    restart: bool,
) -> dict[str, dict[str, Any]]:
    if restart and path.exists():
        path.unlink()
    if not path.exists():
        _atomic_write_bytes(path, _json_line(expected_metadata))
        return {}

    entries: dict[str, dict[str, Any]] = {}
    repair_offset: int | None = None
    append_newline = False
    with path.open("rb") as handle:
        metadata_line = handle.readline()
        try:
            stored_metadata = json.loads(metadata_line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint metadata: {path}") from exc
        if stored_metadata != dict(expected_metadata):
            raise ValueError(
                "checkpoint inputs differ from the current manifest, configuration, "
                "skills, data root, or pipeline; rerun with --restart"
            )

        line_number = 1
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            has_newline = line.endswith(b"\n")
            try:
                value = json.loads(line.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not has_newline:
                    repair_offset = line_start
                    break
                raise ValueError(f"invalid checkpoint line {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"checkpoint line {line_number} must be an object")
            if value.get("kind") != "scenario" or not isinstance(
                value.get("scenario_id"), str
            ):
                raise ValueError(f"invalid scenario checkpoint line {line_number}")
            scenario_id = value["scenario_id"]
            if scenario_id in entries:
                raise ValueError(f"duplicate scenario in checkpoint: {scenario_id}")
            entries[scenario_id] = value
            if not has_newline:
                append_newline = True

    if repair_offset is not None:
        with path.open("r+b") as handle:
            handle.truncate(repair_offset)
    elif append_newline:
        with path.open("ab") as handle:
            handle.write(b"\n")
    return entries


def _atomic_write_seed_csv(path: Path, records: Sequence[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write_seed_records(temporary, records)
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


def _normalize_records(
    records: Sequence[SeedRecord],
    *,
    row: ManifestRow,
    target_risk_definitions: Mapping[str, Mapping[str, Any]],
) -> list[SeedRecord]:
    normalized: list[SeedRecord] = []
    for record in records:
        if not isinstance(record, SeedRecord):
            raise TypeError("detect_scenario must return SeedRecord instances")
        if record.scenario_id != row.scenario_id:
            raise ValueError(
                f"detector returned scenario_id={record.scenario_id} while scanning "
                f"{row.scenario_id}"
            )
        if record.skill_id not in target_risk_definitions:
            raise ValueError(f"detector returned unknown skill_id: {record.skill_id}")
        if record.target_risk_definition != target_risk_definitions[record.skill_id]:
            raise ValueError(
                f"detector returned target_risk_definition that differs from "
                f"{record.skill_id} YAML"
            )
        normalized.append(replace(record, source_path=row.source_path))
    return sort_seed_records(normalized)


def _validated_rejections(values: Mapping[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for reason, raw_count in values.items():
        if not isinstance(reason, str) or not reason:
            raise ValueError("rejection reason must be a non-empty string")
        count = int(raw_count)
        if isinstance(raw_count, bool) or count != raw_count or count < 0:
            raise ValueError(f"invalid rejection count for {reason}: {raw_count}")
        if count:
            result[reason] = count
    return dict(sorted(result.items()))


def _actor_distribution(
    scenario: Scenario, records: Sequence[SeedRecord]
) -> dict[str, Any]:
    actor_types = {
        agent.track_id: agent.object_type.lower() for agent in scenario.agents
    }
    initiators: Counter[str] = Counter()
    responders: Counter[str] = Counter()
    pairs: Counter[str] = Counter()
    roles: dict[str, Counter[str]] = {}
    role_combinations: Counter[str] = Counter()
    for record in records:
        try:
            initiator_type = actor_types[record.initiator_track_id]
            responder_type = actor_types[record.responder_track_id]
        except KeyError as exc:
            raise ValueError(
                f"candidate references missing actor {exc.args[0]} in {scenario.scenario_id}"
            ) from None
        initiators[initiator_type] += 1
        responders[responder_type] += 1
        pairs[f"{initiator_type}|{responder_type}"] += 1
        combination: list[str] = []
        for role, track_id in sorted(record.role_track_ids.items()):
            try:
                actor_type = actor_types[track_id]
            except KeyError:
                raise ValueError(
                    f"candidate role {role} references missing actor {track_id} "
                    f"in {scenario.scenario_id}"
                ) from None
            roles.setdefault(role, Counter())[actor_type] += 1
            combination.append(f"{role}={actor_type}")
        role_combinations["|".join(combination)] += 1
    return {
        "initiator": dict(sorted(initiators.items())),
        "responder": dict(sorted(responders.items())),
        "pair": dict(sorted(pairs.items())),
        "by_role": {
            role: dict(sorted(counts.items())) for role, counts in sorted(roles.items())
        },
        "role_combination": dict(sorted(role_combinations.items())),
    }


def _scenario_entry(
    *,
    manifest_index: int,
    row: ManifestRow,
    scenario: Scenario,
    run: DetectionRun,
    records: Sequence[SeedRecord],
    elapsed_seconds: float,
    peak_memory_mib: float,
) -> dict[str, Any]:
    return {
        "kind": "scenario",
        "manifest_index": manifest_index,
        "scenario_id": row.scenario_id,
        "city_name": scenario.city_name,
        "records": [record.to_csv_row() for record in records],
        "rejection_counts": _validated_rejections(run.rejection_counts),
        "actor_distribution": _actor_distribution(scenario, records),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "peak_memory_mib": round(peak_memory_mib, 3),
    }


def _scan_scenario_entry(
    manifest_index: int,
    row: ManifestRow,
    *,
    data_root: Path,
    skills: Sequence[SkillSpec],
    config: DetectionConfig,
    target_risk_definitions: Mapping[str, Mapping[str, Any]],
    scenario_loader: ScenarioLoader,
    detector: Detector,
    memory: _PeakMemoryTracker,
) -> dict[str, Any]:
    scenario_started = time.perf_counter()
    source = PurePosixPath(row.source_path.replace("\\", "/"))
    source_path = data_root / Path(*source.parts)
    scenario = scenario_loader(source_path)
    if scenario.scenario_id != row.scenario_id:
        raise ValueError(
            f"loaded scenario_id={scenario.scenario_id} while manifest expects "
            f"{row.scenario_id}"
        )
    scenario.metadata["source_path"] = row.source_path
    run = detector(scenario, skills, config)
    if not isinstance(run, DetectionRun):
        raise TypeError("detect_scenario must return DetectionRun")
    records = _normalize_records(
        run.records,
        row=row,
        target_risk_definitions=target_risk_definitions,
    )
    return _scenario_entry(
        manifest_index=manifest_index,
        row=row,
        scenario=scenario,
        run=run,
        records=records,
        elapsed_seconds=time.perf_counter() - scenario_started,
        peak_memory_mib=memory.peak_mib(),
    )


_WORKER_CONTEXT: tuple[
    Path,
    Sequence[SkillSpec],
    DetectionConfig,
    Mapping[str, Mapping[str, Any]],
    ScenarioLoader,
    Detector,
] | None = None
_WORKER_MEMORY: _PeakMemoryTracker | None = None


def _initialize_scan_worker(
    data_root: Path,
    skills: Sequence[SkillSpec],
    config: DetectionConfig,
    target_risk_definitions: Mapping[str, Mapping[str, Any]],
    scenario_loader: ScenarioLoader,
    detector: Detector,
) -> None:
    global _WORKER_CONTEXT, _WORKER_MEMORY
    if scenario_loader is _ORIGINAL_AV2_SCENARIO_LOADER:
        preload_av2_dependencies()
    _WORKER_CONTEXT = (
        data_root,
        skills,
        config,
        target_risk_definitions,
        scenario_loader,
        detector,
    )
    _WORKER_MEMORY = _PeakMemoryTracker()


def _scan_worker_task(task: tuple[int, ManifestRow]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None or _WORKER_MEMORY is None:
        raise RuntimeError("parallel scan worker was not initialized")
    manifest_index, row = task
    data_root, skills, config, target_risk_definitions, scenario_loader, detector = (
        _WORKER_CONTEXT
    )
    return _scan_scenario_entry(
        manifest_index,
        row,
        data_root=data_root,
        skills=skills,
        config=config,
        target_risk_definitions=target_risk_definitions,
        scenario_loader=scenario_loader,
        detector=detector,
        memory=_WORKER_MEMORY,
    )


def _entry_records(entry: Mapping[str, Any]) -> list[SeedRecord]:
    rows = entry.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"checkpoint records missing for {entry.get('scenario_id')}")
    return [SeedRecord.from_csv_row(row) for row in rows]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _numeric_distribution(values: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "p25": round(_percentile(ordered, 0.25), 6),
        "median": round(_percentile(ordered, 0.5), 6),
        "p75": round(_percentile(ordered, 0.75), 6),
        "max": round(ordered[-1], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }


def _merge_actor_distributions(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    totals = {name: Counter() for name in ("initiator", "responder", "pair")}
    role_totals: dict[str, Counter[str]] = {}
    role_combinations: Counter[str] = Counter()
    for entry in entries:
        distribution = entry.get("actor_distribution", {})
        for name, counter in totals.items():
            counter.update(distribution.get(name, {}))
        for role, counts in distribution.get("by_role", {}).items():
            role_totals.setdefault(role, Counter()).update(counts)
        role_combinations.update(distribution.get("role_combination", {}))
    result: dict[str, Any] = {
        name: dict(sorted(counter.items())) for name, counter in totals.items()
    }
    result["by_role"] = {
        role: dict(sorted(counts.items()))
        for role, counts in sorted(role_totals.items())
    }
    result["role_combination"] = dict(sorted(role_combinations.items()))
    return result


def _build_summary(
    *,
    kind: str,
    manifest_path: Path,
    data_root: Path,
    config_path: Path,
    skill_dir: Path,
    output_csv: Path,
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
    all_rows: Sequence[ManifestRow],
    selected_rows: Sequence[ManifestRow],
    selected_entries: Sequence[Mapping[str, Any]],
    records: Sequence[SeedRecord],
    skills: Sequence[SkillSpec],
    config: DetectionConfig,
    exclusion_audit: Mapping[str, Mapping[str, Any]],
    resumed_count: int,
    current_run_elapsed_seconds: float,
    current_peak_memory_mib: float,
    peak_memory_method: str,
    workers: int,
) -> dict[str, Any]:
    skill_ids = [skill.skill_id for skill in skills]
    skill_hits = Counter(record.skill_id for record in records)
    skill_scenario_hits: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    scanned_cities: Counter[str] = Counter()
    candidate_cities: Counter[str] = Counter()
    seed_risk_by_metric: dict[str, list[float]] = {}
    seed_risk_by_skill_and_metric: dict[str, dict[str, list[float]]] = {
        skill_id: {} for skill_id in skill_ids
    }
    seed_risk_relation_counts: Counter[str] = Counter()
    seed_risk_relation_by_skill: dict[str, Counter[str]] = {
        skill_id: Counter() for skill_id in skill_ids
    }
    records_by_scenario: dict[str, list[SeedRecord]] = {}
    for record in records:
        records_by_scenario.setdefault(record.scenario_id, []).append(record)

    for entry in selected_entries:
        entry_records = records_by_scenario.get(str(entry["scenario_id"]), [])
        city = str(entry["city_name"])
        scanned_cities[city] += 1
        candidate_cities[city] += len(entry_records)
        rejection_counts.update(entry.get("rejection_counts", {}))
        skill_scenario_hits.update({record.skill_id for record in entry_records})
        for record in entry_records:
            seed_risk_by_metric.setdefault(record.seed_risk_metric, []).append(
                record.seed_risk_value
            )
            seed_risk_by_skill_and_metric[record.skill_id].setdefault(
                record.seed_risk_metric,
                [],
            ).append(record.seed_risk_value)
            relation = (
                "proxy_metric"
                if record.seed_risk_is_proxy
                else "target_metric_observation"
            )
            seed_risk_relation_counts[relation] += 1
            seed_risk_relation_by_skill[record.skill_id][relation] += 1

    scenario_processing_seconds = sum(
        float(entry.get("elapsed_seconds", 0.0)) for entry in selected_entries
    )
    current_run_entries = selected_entries[resumed_count:]
    current_run_scenario_seconds = sum(
        float(entry.get("elapsed_seconds", 0.0)) for entry in current_run_entries
    )
    current_run_scenario_count = len(current_run_entries)
    ideal_balanced_seconds = (
        current_run_scenario_seconds / workers
        if current_run_scenario_count and current_run_scenario_seconds > 0
        else 0.0
    )
    steady_state_rate = (
        current_run_scenario_count / ideal_balanced_seconds
        if ideal_balanced_seconds > 0
        else None
    )
    peak_memory_mib = max(
        current_peak_memory_mib,
        max(
            (float(entry.get("peak_memory_mib", 0.0)) for entry in selected_entries),
            default=0.0,
        ),
    )
    unique_candidate_scenarios = len({record.scenario_id for record in records})
    summary = {
        "schema_version": 2,
        "status": "complete" if len(selected_rows) == len(all_rows) else "partial",
        "scan_kind": kind,
        "inputs": {
            "manifest": str(manifest_path),
            "data_root": str(data_root),
            "config": str(config_path),
            "skills_dir": str(skill_dir),
            "manifest_scenario_count": len(all_rows),
            "selected_scenario_count": len(selected_rows),
            "selected_skill_ids": skill_ids,
            "global_seed": config.global_seed,
            "workers": workers,
            "fingerprints": {
                key: value
                for key, value in metadata.items()
                if key.endswith("_sha256")
            },
        },
        "outputs": {
            "candidate_csv": str(output_csv),
            "checkpoint": str(checkpoint_path),
        },
        "leakage_check": {
            "status": "passed",
            "overlap_count": 0,
            "excluded_manifests": dict(exclusion_audit),
        },
        "counts": {
            "processed_scenarios": len(selected_entries),
            "resumed_scenarios_this_run": resumed_count,
            "new_scenarios_this_run": len(selected_entries) - resumed_count,
            "candidates": len(records),
            "unique_candidate_scenarios": unique_candidate_scenarios,
        },
        "skill_hits": {
            skill_id: skill_hits.get(skill_id, 0) for skill_id in skill_ids
        },
        "skill_scenario_hits": {
            skill_id: skill_scenario_hits.get(skill_id, 0) for skill_id in skill_ids
        },
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "city_distribution": {
            "scanned_scenarios": dict(sorted(scanned_cities.items())),
            "candidates": dict(sorted(candidate_cities.items())),
        },
        "actor_distribution": _merge_actor_distributions(selected_entries),
        "target_risk_definitions": {
            skill.skill_id: skill.risk_definition for skill in skills
        },
        "seed_risk_distribution": {
            "by_metric": {
                name: _numeric_distribution(values)
                for name, values in sorted(seed_risk_by_metric.items())
            },
            "by_skill_and_metric": {
                skill_id: {
                    name: _numeric_distribution(values)
                    for name, values in sorted(
                        seed_risk_by_skill_and_metric[skill_id].items()
                    )
                }
                for skill_id in skill_ids
            },
            "relation_counts": {
                name: seed_risk_relation_counts.get(name, 0)
                for name in ("target_metric_observation", "proxy_metric")
            },
            "relation_by_skill": {
                skill_id: {
                    name: seed_risk_relation_by_skill[skill_id].get(name, 0)
                    for name in ("target_metric_observation", "proxy_metric")
                }
                for skill_id in skill_ids
            },
        },
        "performance": {
            "scenario_processing_seconds": round(scenario_processing_seconds, 6),
            "scenario_processing_seconds_semantics": (
                "sum_of_per_scenario_wall_seconds; may exceed end-to-end wall time "
                "when workers run concurrently"
            ),
            "current_run_scenario_elapsed_seconds_sum": round(
                current_run_scenario_seconds,
                6,
            ),
            "current_run_scenario_count": current_run_scenario_count,
            "current_run_ideal_balanced_scenario_wall_seconds": round(
                ideal_balanced_seconds,
                6,
            ),
            "current_run_estimated_steady_state_scenarios_per_second": (
                None if steady_state_rate is None else round(steady_state_rate, 6)
            ),
            "current_run_wall_seconds": round(current_run_elapsed_seconds, 6),
            "current_run_wall_minus_ideal_balanced_scenario_seconds": None,
            "peak_memory_mib": round(peak_memory_mib, 3),
            "peak_memory_method": peak_memory_method,
        },
    }
    return summary


def _progress_line(
    *,
    total: int,
    completed: int,
    new_completed: int,
    candidates: int,
    elapsed_seconds: float,
) -> str:
    width = 30
    fraction = 1.0 if total == 0 else completed / total
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    speed = new_completed / elapsed_seconds if elapsed_seconds > 0 else 0.0
    return (
        f"scan [{bar}] {fraction * 100:6.2f}%  {completed}/{total} scenarios  "
        f"{candidates} candidates  {speed:.2f} scenarios/s"
    )


def run_scan(
    *,
    manifest_path: Path,
    data_root: Path,
    skill_dir: Path,
    config_path: Path,
    output_csv: Path,
    summary_json: Path,
    checkpoint_path: Path | None = None,
    skill_ids: Sequence[str] | None = None,
    limit: int | None = None,
    progress_every: int = 10,
    restart: bool = False,
    workers: int = 1,
    confirm_formal_scan: bool = False,
    internal_validation_manifest: Path = DEFAULT_EXCLUSION_MANIFESTS[
        "internal_validation"
    ],
    final_validation_manifest: Path = DEFAULT_EXCLUSION_MANIFESTS["final_validation"],
    scenario_loader: ScenarioLoader | None = None,
    detector: Detector | None = None,
) -> dict[str, Any]:
    """Run or resume one deterministic manifest scan."""

    started = time.perf_counter()
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers > 1 and (scenario_loader is not None or detector is not None):
        raise ValueError(
            "workers > 1 requires the default scenario_loader and detector"
        )
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if progress_every <= 0:
        raise ValueError("progress_every must be a positive integer")

    kind = _scan_kind(manifest_path)
    if kind == "formal" and not confirm_formal_scan:
        raise PermissionError(
            "formal Train scanning requires user confirmation and "
            "--confirm-formal-scan"
        )
    rows = read_manifest(manifest_path)
    kind = _validate_manifest_scope(manifest_path, rows)
    selected_rows = rows if limit is None else rows[:limit]
    if len(selected_rows) < len(rows) or skill_ids is not None:
        default_csv, default_summary = DEFAULT_OUTPUTS[kind]
        uses_canonical_output = (
            output_csv.resolve() == default_csv.resolve()
            or summary_json.resolve() == default_summary.resolve()
        )
        if uses_canonical_output:
            reason = (
                "partial scans"
                if len(selected_rows) < len(rows)
                else "skill-subset scans"
            )
            raise ValueError(
                f"{reason} cannot overwrite the canonical candidate CSV or summary; "
                "provide explicit smoke output paths"
            )
    exclusion_manifests = {
        "internal_validation": internal_validation_manifest,
        "final_validation": final_validation_manifest,
    }
    exclusion_audit = _validate_exclusion_manifests(rows, exclusion_manifests)
    skills = _load_confirmed_skills(skill_dir, skill_ids)
    selected_skill_ids = [skill.skill_id for skill in skills]
    config = load_detection_config(config_path)
    target_risk_definitions = {
        skill.skill_id: skill.risk_definition for skill in skills
    }
    if scenario_loader is None:
        scenario_loader = load_av2_scenario
    if detector is None:
        from skilldrive.skills.detection import detect_scenario

        detector = detect_scenario

    if checkpoint_path is None:
        checkpoint_path = summary_json.with_suffix(".checkpoint.jsonl")
    metadata = _checkpoint_metadata(
        kind=kind,
        rows=rows,
        data_root=data_root,
        config_path=config_path,
        skill_dir=skill_dir,
        selected_skill_ids=selected_skill_ids,
        exclusion_audit=exclusion_audit,
    )
    entries = _load_checkpoint(checkpoint_path, metadata, restart=restart)
    authoritative_ids = {row.scenario_id for row in rows}
    unexpected_ids = set(entries) - authoritative_ids
    if unexpected_ids:
        raise ValueError(
            f"checkpoint contains scenarios outside the manifest: "
            f"{sorted(unexpected_ids)[:5]}"
        )
    expected_checkpoint_ids = [row.scenario_id for row in rows[: len(entries)]]
    if list(entries) != expected_checkpoint_ids:
        raise ValueError("checkpoint scenarios must be a manifest-order prefix")
    for manifest_index, scenario_id in enumerate(expected_checkpoint_ids):
        if entries[scenario_id].get("manifest_index") != manifest_index:
            raise ValueError(
                f"checkpoint manifest_index differs for {scenario_id}: "
                f"expected {manifest_index}"
            )

    selected_ids = {row.scenario_id for row in selected_rows}
    resumed_count = len(selected_ids & set(entries))
    completed = resumed_count
    new_completed = 0
    candidates = sum(
        len(entries[scenario_id].get("records", []))
        for scenario_id in selected_ids
        if scenario_id in entries
    )
    print(
        "\r"
        + _progress_line(
            total=len(selected_rows),
            completed=completed,
            new_completed=new_completed,
            candidates=candidates,
            elapsed_seconds=max(time.perf_counter() - started, 1e-9),
        ),
        end="",
        flush=True,
    )

    memory = _PeakMemoryTracker()
    progress_line_open = True
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("ab", buffering=0) as checkpoint:
            pending_tasks = [
                (manifest_index, row)
                for manifest_index, row in enumerate(selected_rows)
                if row.scenario_id not in entries
            ]

            def commit_entry(
                manifest_index: int,
                row: ManifestRow,
                entry: dict[str, Any],
            ) -> None:
                nonlocal completed, new_completed, candidates
                if entry.get("manifest_index") != manifest_index:
                    raise ValueError("worker returned an unexpected manifest_index")
                if entry.get("scenario_id") != row.scenario_id:
                    raise ValueError("worker returned an unexpected scenario_id")
                checkpoint.write(_json_line(entry))
                entries[row.scenario_id] = entry
                completed += 1
                new_completed += 1
                candidates += len(entry.get("records", []))
                if completed % progress_every == 0 or completed == len(selected_rows):
                    os.fsync(checkpoint.fileno())
                    print(
                        "\r"
                        + _progress_line(
                            total=len(selected_rows),
                            completed=completed,
                            new_completed=new_completed,
                            candidates=candidates,
                            elapsed_seconds=max(time.perf_counter() - started, 1e-9),
                        ),
                        end="",
                        flush=True,
                    )

            if workers == 1:
                if scenario_loader is _ORIGINAL_AV2_SCENARIO_LOADER:
                    preload_av2_dependencies()
                for manifest_index, row in pending_tasks:
                    entry = _scan_scenario_entry(
                        manifest_index,
                        row,
                        data_root=data_root,
                        skills=skills,
                        config=config,
                        target_risk_definitions=target_risk_definitions,
                        scenario_loader=scenario_loader,
                        detector=detector,
                        memory=memory,
                    )
                    commit_entry(manifest_index, row, entry)
            elif pending_tasks:
                if scenario_loader is _ORIGINAL_AV2_SCENARIO_LOADER:
                    preload_av2_worker_dependencies()
                executor = ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_initialize_scan_worker,
                    initargs=(
                        data_root,
                        skills,
                        config,
                        target_risk_definitions,
                        scenario_loader,
                        detector,
                    ),
                )
                in_flight: deque[
                    tuple[int, ManifestRow, Future[dict[str, Any]]]
                ] = deque()
                next_task = 0
                window = 2 * workers
                try:
                    while next_task < len(pending_tasks) and len(in_flight) < window:
                        manifest_index, row = pending_tasks[next_task]
                        in_flight.append(
                            (
                                manifest_index,
                                row,
                                executor.submit(_scan_worker_task, (manifest_index, row)),
                            )
                        )
                        next_task += 1
                    while in_flight:
                        manifest_index, row, future = in_flight[0]
                        entry = future.result()
                        in_flight.popleft()
                        commit_entry(manifest_index, row, entry)
                        if next_task < len(pending_tasks):
                            next_index, next_row = pending_tasks[next_task]
                            in_flight.append(
                                (
                                    next_index,
                                    next_row,
                                    executor.submit(
                                        _scan_worker_task,
                                        (next_index, next_row),
                                    ),
                                )
                            )
                            next_task += 1
                finally:
                    for _, _, future in in_flight:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
        print(flush=True)
        progress_line_open = False

        selected_entries = [entries[row.scenario_id] for row in selected_rows]
        records: list[SeedRecord] = []
        for entry in selected_entries:
            records.extend(_entry_records(entry))
            entry.pop("records", None)
        records = sort_seed_records(records)
        _atomic_write_seed_csv(output_csv, records)
        memory_method = memory.method
        summary = _build_summary(
            kind=kind,
            manifest_path=manifest_path,
            data_root=data_root,
            config_path=config_path,
            skill_dir=skill_dir,
            output_csv=output_csv,
            checkpoint_path=checkpoint_path,
            metadata=metadata,
            all_rows=rows,
            selected_rows=selected_rows,
            selected_entries=selected_entries,
            records=records,
            skills=skills,
            config=config,
            exclusion_audit=exclusion_audit,
            resumed_count=resumed_count,
            current_run_elapsed_seconds=0.0,
            current_peak_memory_mib=memory.peak_mib(),
            peak_memory_method=memory_method,
            workers=workers,
        )
        summary["performance"]["current_run_wall_seconds"] = round(
            time.perf_counter() - started, 6
        )
        summary["performance"][
            "current_run_wall_minus_ideal_balanced_scenario_seconds"
        ] = round(
            max(
                0.0,
                summary["performance"]["current_run_wall_seconds"]
                - summary["performance"][
                    "current_run_ideal_balanced_scenario_wall_seconds"
                ],
            ),
            6,
        )
        summary["performance"]["peak_memory_mib"] = round(
            max(summary["performance"]["peak_memory_mib"], memory.peak_mib()), 3
        )
        _atomic_write_json(summary_json, summary)
        return summary
    finally:
        if progress_line_open:
            print(flush=True)
        memory.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect skill seed candidates from approved AV2 Train manifests."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/av2/motion-forecasting"),
    )
    parser.add_argument("--skills-dir", type=Path, default=Path("configs/skills"))
    parser.add_argument(
        "--skill-id",
        dest="skill_ids",
        action="append",
        help=(
            "Scan only this confirmed skill ID. Repeat for multiple skills; "
            "catalog order is preserved."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/seed_detection.yaml")
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--internal-validation-manifest",
        type=Path,
        default=DEFAULT_EXCLUSION_MANIFESTS["internal_validation"],
    )
    parser.add_argument(
        "--final-validation-manifest",
        type=Path,
        default=DEFAULT_EXCLUSION_MANIFESTS["final_validation"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of independent scenario worker processes (this machine: 10).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Discard the compatible checkpoint and scan from the beginning.",
    )
    parser.add_argument(
        "--confirm-formal-scan",
        action="store_true",
        help="Required for formal_train.csv after the user approves the rules.",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        kind = _scan_kind(args.manifest)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None and (
        args.output_csv is None or args.summary_json is None
    ):
        parser.error("--limit requires explicit --output-csv and --summary-json paths")
    default_csv, default_summary = DEFAULT_OUTPUTS[kind]
    try:
        summary = run_scan(
            manifest_path=args.manifest,
            data_root=args.data_root,
            skill_dir=args.skills_dir,
            config_path=args.config,
            output_csv=args.output_csv or default_csv,
            summary_json=args.summary_json or default_summary,
            checkpoint_path=args.checkpoint,
            skill_ids=args.skill_ids,
            limit=args.limit,
            progress_every=args.progress_every,
            restart=args.restart,
            workers=args.workers,
            confirm_formal_scan=args.confirm_formal_scan,
            internal_validation_manifest=args.internal_validation_manifest,
            final_validation_manifest=args.final_validation_manifest,
        )
    except KeyboardInterrupt:
        print("\nscan interrupted; rerun the same command to resume", flush=True)
        raise SystemExit(130) from None
    print(
        f"scan complete: {summary['counts']['processed_scenarios']} scenarios, "
        f"{summary['counts']['candidates']} candidates, "
        f"output={summary['outputs']['candidate_csv']}"
    )


if __name__ == "__main__":
    main()

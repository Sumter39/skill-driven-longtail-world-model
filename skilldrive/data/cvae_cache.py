"""Deterministic, resumable scenario caches for conditional CVAE training."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, TextIO

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.data.cvae_samples import (
    ACTOR_FEATURE_DIM,
    ACTOR_RADIUS_M,
    ANCHOR_INDEX,
    CVAESchema,
    FUTURE_STEPS,
    HISTORY_STEPS,
    MAP_FEATURE_DIM,
    MAP_RADIUS_M,
    MAX_ACTORS,
    MAX_MAP_POINTS,
    MAX_MAP_POLYLINES,
    MINIMUM_TARGET_HISTORY_STEPS,
    MapClipStatistics,
    SampleSpec,
    TensorizedSample,
    build_cvae_schema,
    make_base_sample_spec,
    observed_sample_specs,
    tensorize_scenario,
)
from skilldrive.data.manifests import ManifestRow, read_manifest
from skilldrive.schemas import Scenario
from skilldrive.seeds import iter_seed_records
from skilldrive.training.config import CVAEConfig


CACHE_VERSION = 5
SHARD_DIRECTORY = "shards"
CACHE_MANIFEST_NAME = "cache_manifest.json"
SAMPLE_INDEX_NAME = "sample_index.jsonl"

ScenarioLoader = Callable[[str | Path], Scenario]
ValidationLabeler = Callable[[Scenario, CVAESchema], Iterable[SampleSpec]]

_TENSOR_FIELDS = (
    "actor_history",
    "actor_time_mask",
    "actor_mask",
    "actor_type_id",
    "actor_role_id",
    "map_polylines",
    "map_point_mask",
    "map_polyline_mask",
    "map_type_id",
    "target_actor_index",
    "skill_id",
    "skill_supervision_mask",
    "skill_parameters",
    "parameter_mask",
    "target_future",
    "target_future_mask",
    "anchor_origin_global",
    "anchor_heading_global",
)
_MAP_TENSOR_FIELDS = (
    "map_polylines",
    "map_point_mask",
    "map_polyline_mask",
    "map_type_id",
)
_SAMPLE_TENSOR_FIELDS = tuple(
    name for name in _TENSOR_FIELDS if name not in _MAP_TENSOR_FIELDS
)
_TENSOR_DTYPES = {
    "actor_history": torch.float32,
    "actor_time_mask": torch.bool,
    "actor_mask": torch.bool,
    "actor_type_id": torch.int64,
    "actor_role_id": torch.int64,
    "map_polylines": torch.float32,
    "map_point_mask": torch.bool,
    "map_polyline_mask": torch.bool,
    "map_type_id": torch.int64,
    "target_actor_index": torch.int64,
    "skill_id": torch.int64,
    "skill_supervision_mask": torch.bool,
    "skill_parameters": torch.float32,
    "parameter_mask": torch.bool,
    "target_future": torch.float32,
    "target_future_mask": torch.bool,
    "anchor_origin_global": torch.float32,
    "anchor_heading_global": torch.float32,
}

_MAP_CLIP_STAT_FIELDS = (
    "eligible_polylines",
    "retained_polylines",
    "dropped_polylines_due_to_limit",
    "original_in_radius_points",
    "retained_in_radius_points",
    "resampled_polylines_due_to_point_limit",
    "excess_input_points_over_point_limit",
)
_MAP_CLIP_LIMITS = {
    "radius_m": MAP_RADIUS_M,
    "max_polylines": MAX_MAP_POLYLINES,
    "max_points_per_polyline": MAX_MAP_POINTS,
}

_PARTITIONS = {
    "development_train": ("development_train", True),
    "development_validation": ("development_validation", False),
    "formal_train": ("train", True),
    "internal_validation": ("internal_validation", False),
}
_SPLITS = {
    "development": ("development_train", "development_validation"),
    "formal": ("formal_train", "internal_validation"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any, *, indent: int | None = None) -> bytes:
    suffix = "\n" if indent is not None else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
            indent=indent,
            allow_nan=False,
        )
        + suffix
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_bytes(path, _canonical_json(value, indent=2))


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _schema_fingerprint(schema: CVAESchema) -> str:
    value = {
        "formal_skills": [skill.to_dict() for skill in schema.formal_skills],
        "formal_skill_ids": list(schema.formal_skill_ids),
        "candidate_skill_ids": list(schema.candidate_skill_ids),
        "skill_vocabulary": list(schema.skill_vocabulary.tokens),
        "actor_type_vocabulary": list(schema.actor_type_vocabulary.tokens),
        "role_vocabulary": list(schema.role_vocabulary.tokens),
        "map_type_vocabulary": list(schema.map_type_vocabulary.tokens),
        "parameter_schema": {
            "version": schema.parameter_schema.version,
            "dimension": schema.parameter_schema.dimension,
            "definitions": [asdict(item) for item in schema.parameter_schema.definitions],
        },
    }
    return _hash_value(value)


def _cache_config_fingerprint(config: CVAEConfig) -> str:
    return _hash_value(
        {
            "tensorization": asdict(config.tensorization),
            "shard_size": config.cache.shard_size,
            "actor_feature_dim": config.model.actor_feature_dim,
            "map_feature_dim": config.model.map_feature_dim,
        }
    )


def cvae_schema_fingerprint(schema: CVAESchema) -> str:
    """Return the stable cache contract fingerprint for one CVAE schema."""

    return _schema_fingerprint(schema)


def _manifest_rows_fingerprint(rows: Iterable[ManifestRow]) -> str:
    return _hash_value([asdict(row) for row in rows])


def _validate_config_contract(config: CVAEConfig) -> None:
    expected = {
        "tensorization.schema_version": (config.tensorization.schema_version, 1),
        "tensorization.history_steps": (config.tensorization.history_steps, HISTORY_STEPS),
        "tensorization.future_steps": (config.tensorization.future_steps, FUTURE_STEPS),
        "tensorization.anchor_frame": (config.tensorization.anchor_frame, ANCHOR_INDEX),
        "tensorization.minimum_history_steps": (
            config.tensorization.minimum_history_steps,
            MINIMUM_TARGET_HISTORY_STEPS,
        ),
        "tensorization.max_actors": (config.tensorization.max_actors, MAX_ACTORS),
        "tensorization.actor_radius_m": (
            config.tensorization.actor_radius_m,
            ACTOR_RADIUS_M,
        ),
        "tensorization.max_map_polylines": (
            config.tensorization.max_map_polylines,
            MAX_MAP_POLYLINES,
        ),
        "tensorization.map_points_per_polyline": (
            config.tensorization.map_points_per_polyline,
            MAX_MAP_POINTS,
        ),
        "tensorization.map_radius_m": (config.tensorization.map_radius_m, MAP_RADIUS_M),
        "model.actor_feature_dim": (config.model.actor_feature_dim, ACTOR_FEATURE_DIM),
        "model.map_feature_dim": (config.model.map_feature_dim, MAP_FEATURE_DIM),
        "cache.shard_size": (config.cache.shard_size, 64),
    }
    mismatches = [
        f"{name}={actual!r}, expected {wanted!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if not config.tensorization.require_complete_future:
        mismatches.append("tensorization.require_complete_future=False, expected True")
    if mismatches:
        raise ValueError("CVAE cache configuration differs from cvae_samples: " + "; ".join(mismatches))


def _spec_to_dict(spec: SampleSpec) -> dict[str, Any]:
    return {
        "scenario_id": spec.scenario_id,
        "target_track_id": spec.target_track_id,
        "skill_id": spec.skill_id,
        "skill_supervision_mask": spec.skill_supervision_mask,
        "responder_track_id": spec.responder_track_id,
        "role_track_ids": [list(item) for item in spec.role_track_ids],
        "trigger_score": spec.trigger_score,
    }


def _spec_from_dict(value: Mapping[str, Any]) -> SampleSpec:
    required = {
        "scenario_id",
        "target_track_id",
        "skill_id",
        "skill_supervision_mask",
        "responder_track_id",
        "role_track_ids",
        "trigger_score",
    }
    if set(value) != required:
        raise ValueError("cached SampleSpec has missing or unknown fields")
    roles = value["role_track_ids"]
    if not isinstance(roles, list):
        raise ValueError("cached SampleSpec role_track_ids must be a list")
    return SampleSpec(
        scenario_id=value["scenario_id"],
        target_track_id=value["target_track_id"],
        skill_id=value["skill_id"],
        skill_supervision_mask=value["skill_supervision_mask"],
        responder_track_id=value["responder_track_id"],
        role_track_ids=tuple(tuple(item) for item in roles),
        trigger_score=value["trigger_score"],
    )


def _labeler_identity(labeler: ValidationLabeler | None) -> str:
    if labeler is None:
        return "base_only"
    explicit = getattr(labeler, "cache_identity", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    module = getattr(labeler, "__module__", "unknown")
    name = getattr(labeler, "__qualname__", getattr(labeler, "__name__", "callable"))
    return f"{module}.{name}"


def _stream_training_specs(
    pool_path: Path,
    scenario_ids: set[str],
    schema: CVAESchema,
) -> tuple[dict[str, tuple[SampleSpec, ...]], dict[str, int]]:
    records_by_scenario: dict[str, list[Any]] = defaultdict(list)
    counts: Counter[str] = Counter()
    skills_by_id = {skill.skill_id: skill for skill in schema.formal_skills}
    candidate_ids = set(schema.candidate_skill_ids)
    for record in iter_seed_records(pool_path):
        counts["pool_records"] += 1
        skill = skills_by_id.get(record.skill_id)
        if record.skill_id in candidate_ids or skill is None:
            raise ValueError(f"training pool contains a non-formal skill: {record.skill_id}")
        contract_fields = {
            "detection_mode": (
                record.evidence.get("detection_mode"),
                skill.detection.get("mode"),
            ),
            "detection_thresholds": (
                record.evidence.get("detection_thresholds"),
                skill.detection.get("thresholds"),
            ),
            "feasibility": (
                record.evidence.get("feasibility"),
                skill.data_support.get("feasibility"),
            ),
            "target_risk_definition": (
                record.target_risk_definition,
                skill.risk_definition,
            ),
        }
        for field, (actual, expected) in contract_fields.items():
            if actual != expected:
                raise ValueError(
                    "training pool record "
                    f"{record.scenario_id}/{record.skill_id} field {field} differs "
                    f"from current formal SkillSpec: actual={actual!r}, "
                    f"expected={expected!r}"
                )
        if record.scenario_id not in scenario_ids:
            continue
        counts["selected_scenario_records"] += 1
        mode = record.evidence.get("detection_mode")
        if mode == "compatible_seed":
            counts["compatible_ignored"] += 1
            continue
        if mode != "observed_trigger":
            raise ValueError(f"unknown detection_mode in training pool: {mode!r}")
        counts["observed_records"] += 1
        records_by_scenario[record.scenario_id].append(record)

    specs = {
        scenario_id: observed_sample_specs(records, schema)
        for scenario_id, records in records_by_scenario.items()
    }
    counts["observed_specs"] = sum(len(values) for values in specs.values())
    return specs, dict(sorted(counts.items()))


def _validation_specs(
    scenario: Scenario,
    schema: CVAESchema,
    labeler: ValidationLabeler | None,
) -> tuple[SampleSpec, ...]:
    if labeler is None:
        return ()
    specs = tuple(labeler(scenario, schema))
    seen: set[str] = set()
    ordered: list[SampleSpec] = []
    for spec in sorted(specs, key=lambda item: item.sort_key):
        if not isinstance(spec, SampleSpec):
            raise TypeError("validation labeler must return SampleSpec instances")
        if spec.scenario_id != scenario.scenario_id:
            raise ValueError("validation labeler returned a different scenario_id")
        if not spec.skill_supervision_mask:
            raise ValueError("validation labeler may only add frozen observed samples")
        if spec.skill_id not in schema.formal_skill_ids:
            raise ValueError("validation labeler returned a non-formal skill")
        if spec.sample_id not in seen:
            seen.add(spec.sample_id)
            ordered.append(spec)
    return tuple(ordered)


def _cleanup_partition(cache_dir: Path) -> None:
    shard_dir = cache_dir / SHARD_DIRECTORY
    if shard_dir.exists():
        for path in shard_dir.glob("shard-*"):
            if path.is_file():
                path.unlink()
    for name in (CACHE_MANIFEST_NAME, SAMPLE_INDEX_NAME):
        path = cache_dir / name
        if path.exists():
            path.unlink()


def _sample_tensor(sample: TensorizedSample, name: str) -> torch.Tensor:
    value = getattr(sample, name)
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else torch.as_tensor(value)
    if tensor.dtype != _TENSOR_DTYPES[name]:
        raise ValueError(
            f"tensorized field {name} has dtype {tensor.dtype}, "
            f"expected {_TENSOR_DTYPES[name]}"
        )
    return tensor


def _stack_tensorized_samples(
    samples: list[TensorizedSample],
    fields: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if not samples:
        return {
            name: torch.empty((0,), dtype=dtype)
            for name, dtype in _TENSOR_DTYPES.items()
            if name in fields
        }
    return {
        name: torch.stack([_sample_tensor(sample, name) for sample in samples])
        for name in fields
    }


def _map_context_digest(sample: TensorizedSample) -> str:
    digest = hashlib.sha256()
    for name in _MAP_TENSOR_FIELDS:
        tensor = _sample_tensor(sample, name).contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(_canonical_json(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _same_map_context(left: TensorizedSample, right: TensorizedSample) -> bool:
    return all(
        torch.equal(_sample_tensor(left, name), _sample_tensor(right, name))
        for name in _MAP_TENSOR_FIELDS
    )


def _deduplicate_map_contexts(
    samples: list[TensorizedSample],
) -> tuple[torch.Tensor, list[TensorizedSample]]:
    unique_contexts: list[TensorizedSample] = []
    digest_buckets: dict[str, list[int]] = defaultdict(list)
    indices: list[int] = []
    for sample in samples:
        digest = _map_context_digest(sample)
        context_index = next(
            (
                index
                for index in digest_buckets[digest]
                if _same_map_context(unique_contexts[index], sample)
            ),
            None,
        )
        if context_index is None:
            context_index = len(unique_contexts)
            unique_contexts.append(sample)
            digest_buckets[digest].append(context_index)
        indices.append(context_index)
    return torch.tensor(indices, dtype=torch.int64), unique_contexts


def _map_clip_statistics_to_dict(value: MapClipStatistics) -> dict[str, int]:
    return {name: int(getattr(value, name)) for name in _MAP_CLIP_STAT_FIELDS}


def _validated_map_clip_sample(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_MAP_CLIP_STAT_FIELDS):
        raise ValueError("invalid sample map clip statistics")
    try:
        statistics = MapClipStatistics(
            **{name: value[name] for name in _MAP_CLIP_STAT_FIELDS}
        )
    except TypeError as exc:
        raise ValueError("invalid sample map clip statistics") from exc
    return _map_clip_statistics_to_dict(statistics)


def _map_clip_group_summary(values: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    maxima = {name: 0 for name in _MAP_CLIP_STAT_FIELDS}
    samples = 0
    polyline_limit_hits = 0
    point_limit_hits = 0
    for value in values:
        statistics = _validated_map_clip_sample(value)
        samples += 1
        totals.update(statistics)
        for name, count in statistics.items():
            maxima[name] = max(maxima[name], count)
        polyline_limit_hits += statistics["dropped_polylines_due_to_limit"] > 0
        point_limit_hits += (
            statistics["resampled_polylines_due_to_point_limit"] > 0
        )
    return {
        "samples": samples,
        "totals": {name: totals[name] for name in _MAP_CLIP_STAT_FIELDS},
        "maxima": maxima,
        "samples_hitting_polyline_limit": polyline_limit_hits,
        "samples_hitting_point_limit": point_limit_hits,
    }


def _map_clip_statistics(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(samples)
    by_skill_values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_values: list[Mapping[str, Any]] = []
    for sample in materialized:
        spec = _spec_from_dict(sample["spec"])
        statistics = _validated_map_clip_sample(sample.get("map_clip_statistics"))
        all_values.append(statistics)
        by_skill_values[spec.skill_id].append(statistics)
    return {
        "limits": dict(_MAP_CLIP_LIMITS),
        **_map_clip_group_summary(all_values),
        "by_skill": {
            skill_id: _map_clip_group_summary(by_skill_values[skill_id])
            for skill_id in sorted(by_skill_values)
        },
    }


def _sample_spec_statistics(
    counts_by_skill: Mapping[str, Counter[str]],
    rejection_reasons_by_skill: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    by_skill: dict[str, dict[str, Any]] = {}
    for skill_id in sorted(counts_by_skill):
        counts = counts_by_skill[skill_id]
        entry = {
            "candidate": counts["candidate"],
            "retained": counts["retained"],
            "rejected": counts["rejected"],
            "rejection_reasons": dict(
                sorted(rejection_reasons_by_skill.get(skill_id, Counter()).items())
            ),
        }
        if entry["candidate"] != entry["retained"] + entry["rejected"]:
            raise ValueError(f"inconsistent SampleSpec counts for skill {skill_id}")
        if sum(entry["rejection_reasons"].values()) != entry["rejected"]:
            raise ValueError(f"inconsistent SampleSpec rejection reasons for skill {skill_id}")
        totals.update({name: entry[name] for name in ("candidate", "retained", "rejected")})
        by_skill[skill_id] = entry
    return {
        "totals": {
            name: totals[name] for name in ("candidate", "retained", "rejected")
        },
        "by_skill": by_skill,
    }


def _validated_sample_spec_statistics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"totals", "by_skill"}:
        raise ValueError("invalid SampleSpec statistics")
    totals = value["totals"]
    by_skill = value["by_skill"]
    count_names = {"candidate", "retained", "rejected"}
    if not isinstance(totals, dict) or set(totals) != count_names:
        raise ValueError("invalid SampleSpec total counts")
    if not isinstance(by_skill, dict):
        raise ValueError("invalid per-skill SampleSpec statistics")
    accumulated: Counter[str] = Counter()
    for skill_id, entry in by_skill.items():
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("invalid SampleSpec statistics skill ID")
        if not isinstance(entry, dict) or set(entry) != count_names | {"rejection_reasons"}:
            raise ValueError(f"invalid SampleSpec statistics for skill {skill_id}")
        counts = {name: entry[name] for name in count_names}
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in counts.values()
        ):
            raise ValueError(f"invalid SampleSpec counts for skill {skill_id}")
        reasons = entry["rejection_reasons"]
        if not isinstance(reasons, dict) or any(
            not isinstance(reason, str)
            or not reason
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            for reason, count in reasons.items()
        ):
            raise ValueError(f"invalid SampleSpec rejection reasons for skill {skill_id}")
        if counts["candidate"] != counts["retained"] + counts["rejected"]:
            raise ValueError(f"inconsistent SampleSpec counts for skill {skill_id}")
        if sum(reasons.values()) != counts["rejected"]:
            raise ValueError(f"inconsistent SampleSpec rejection reasons for skill {skill_id}")
        accumulated.update(counts)
    if any(
        isinstance(totals[name], bool)
        or not isinstance(totals[name], int)
        or totals[name] < 0
        or totals[name] != accumulated[name]
        for name in count_names
    ):
        raise ValueError("inconsistent SampleSpec total counts")
    return value


def _aggregate_sample_spec_statistics(
    values: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    counts_by_skill: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_reasons_by_skill: dict[str, Counter[str]] = defaultdict(Counter)
    for value in values:
        validated = _validated_sample_spec_statistics(value)
        for skill_id, entry in validated["by_skill"].items():
            counts_by_skill[skill_id].update(
                {name: entry[name] for name in ("candidate", "retained", "rejected")}
            )
            rejection_reasons_by_skill[skill_id].update(entry["rejection_reasons"])
    return _sample_spec_statistics(counts_by_skill, rejection_reasons_by_skill)


def _load_shard(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        value = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "version",
        "partition",
        "sample_ids",
        "scenario_ids",
        "target_track_ids",
        "map_context_indices",
        "tensors",
        "map_contexts",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"invalid CVAE shard payload: {path}")
    if value["version"] != CACHE_VERSION:
        raise ValueError(f"invalid CVAE shard version: {path}")
    identifiers = (
        value["sample_ids"],
        value["scenario_ids"],
        value["target_track_ids"],
    )
    if any(
        not isinstance(items, list)
        or any(not isinstance(item, str) or not item for item in items)
        for items in identifiers
    ):
        raise ValueError(f"invalid CVAE shard identifiers: {path}")
    sample_count = len(value["sample_ids"])
    if len(set(value["sample_ids"])) != sample_count or any(
        len(items) != sample_count for items in identifiers[1:]
    ):
        raise ValueError(f"inconsistent CVAE shard identifiers: {path}")
    tensors = value["tensors"]
    if not isinstance(tensors, dict) or set(tensors) != set(_SAMPLE_TENSOR_FIELDS):
        raise ValueError(f"invalid CVAE shard tensor fields: {path}")
    for name in _SAMPLE_TENSOR_FIELDS:
        tensor = tensors[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim < 1
            or tensor.shape[0] != sample_count
            or tensor.dtype != _TENSOR_DTYPES[name]
        ):
            raise ValueError(f"invalid CVAE shard tensor {name}: {path}")
    map_contexts = value["map_contexts"]
    if not isinstance(map_contexts, dict) or set(map_contexts) != set(_MAP_TENSOR_FIELDS):
        raise ValueError(f"invalid CVAE shard map context fields: {path}")
    map_context_count: int | None = None
    for name in _MAP_TENSOR_FIELDS:
        tensor = map_contexts[name]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise ValueError(f"invalid CVAE shard map context {name}: {path}")
        if map_context_count is None:
            map_context_count = tensor.shape[0]
        if tensor.shape[0] != map_context_count or tensor.dtype != _TENSOR_DTYPES[name]:
            raise ValueError(f"invalid CVAE shard map context {name}: {path}")
    map_context_count = 0 if map_context_count is None else map_context_count
    indices = value["map_context_indices"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.int64
        or indices.ndim != 1
        or indices.shape[0] != sample_count
        or map_context_count > sample_count
        or (sample_count > 0 and map_context_count == 0)
        or (
            sample_count > 0
            and (int(indices.min()) < 0 or int(indices.max()) >= map_context_count)
        )
    ):
        raise ValueError(f"invalid CVAE shard map context indices: {path}")
    return value


def _verified_sidecar(
    shard_path: Path,
    sidecar_path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        sidecar = _read_json(sidecar_path)
        for key, value in expected.items():
            if sidecar.get(key) != value:
                return None
        if sidecar.get("shard_sha256") != _sha256(shard_path):
            return None
        payload = _load_shard(shard_path)
        if payload["partition"] != expected["partition"]:
            return None
        retained_scenario_ids = list(dict.fromkeys(payload["scenario_ids"]))
        if retained_scenario_ids != sidecar.get("retained_scenario_ids"):
            return None
        samples = sidecar.get("samples")
        if not isinstance(samples, list) or len(samples) != len(payload["sample_ids"]):
            return None
        for offset, sample in enumerate(samples):
            if (
                sample.get("offset") != offset
                or sample.get("sample_id") != payload["sample_ids"][offset]
                or sample.get("scenario_id") != payload["scenario_ids"][offset]
                or sample.get("target_track_id") != payload["target_track_ids"][offset]
            ):
                return None
            _spec_from_dict(sample["spec"])
            map_statistics = _validated_map_clip_sample(
                sample.get("map_clip_statistics")
            )
            context_index = int(payload["map_context_indices"][offset])
            retained_polylines = int(
                payload["map_contexts"]["map_polyline_mask"][context_index].sum()
            )
            if map_statistics["retained_polylines"] != retained_polylines:
                return None
        counts = sidecar.get("counts")
        if not isinstance(counts, dict):
            return None
        map_context_count = len(payload["map_contexts"][_MAP_TENSOR_FIELDS[0]])
        if (
            counts.get("retained_samples") != len(samples)
            or counts.get("map_contexts") != map_context_count
            or counts.get("deduplicated_map_sample_copies")
            != len(samples) - map_context_count
        ):
            return None
        statistics = _validated_sample_spec_statistics(
            sidecar.get("sample_spec_statistics")
        )
        retained_by_skill = Counter(
            sample["spec"]["skill_id"] for sample in samples
        )
        if statistics["totals"]["retained"] != len(samples):
            return None
        if statistics["totals"]["rejected"] != counts.get("rejected_samples"):
            return None
        if any(
            entry["retained"] != retained_by_skill[skill_id]
            for skill_id, entry in statistics["by_skill"].items()
        ):
            return None
        if any(
            skill_id not in statistics["by_skill"]
            for skill_id in retained_by_skill
        ):
            return None
        if sidecar.get("map_clip_statistics") != _map_clip_statistics(samples):
            return None
        return sidecar
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _build_shard(
    *,
    rows: list[ManifestRow],
    shard_path: Path,
    sidecar_path: Path,
    expected: Mapping[str, Any],
    partition: str,
    is_training: bool,
    data_root: Path,
    schema: CVAESchema,
    observed_specs: Mapping[str, tuple[SampleSpec, ...]],
    scenario_loader: ScenarioLoader,
    validation_labeler: ValidationLabeler | None,
) -> dict[str, Any]:
    retained_ids: list[str] = []
    tensorized_samples: list[TensorizedSample] = []
    samples: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    rejection_counts: Counter[str] = Counter()
    sample_spec_counts: dict[str, Counter[str]] = defaultdict(Counter)
    sample_spec_rejection_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_samples = 0

    for row in rows:
        try:
            scenario = scenario_loader(data_root / row.source_path)
            if not isinstance(scenario, Scenario):
                raise TypeError("scenario_loader must return Scenario")
            if scenario.scenario_id != row.scenario_id:
                raise ValueError("loaded Scenario ID differs from manifest")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rejection_counts[reason] += 1
            rejections.append({"scenario_id": row.scenario_id, "reason": reason})
            continue

        extras = (
            observed_specs.get(row.scenario_id, ())
            if is_training
            else _validation_specs(scenario, schema, validation_labeler)
        )
        candidate_specs = (make_base_sample_spec(scenario), *extras)
        valid_samples: list[tuple[SampleSpec, TensorizedSample]] = []
        seen_samples: set[str] = set()
        for spec in candidate_specs:
            if spec.sample_id in seen_samples:
                continue
            seen_samples.add(spec.sample_id)
            sample_spec_counts[spec.skill_id]["candidate"] += 1
            try:
                tensorized = tensorize_scenario(scenario, spec, schema)
            except ValueError as exc:
                reason = f"sample ValueError: {exc}"
                rejection_counts[reason] += 1
                sample_spec_counts[spec.skill_id]["rejected"] += 1
                sample_spec_rejection_reasons[spec.skill_id][reason] += 1
                rejected_samples += 1
                continue
            sample_spec_counts[spec.skill_id]["retained"] += 1
            valid_samples.append((spec, tensorized))

        if not valid_samples:
            reason = "no valid base or observed samples"
            rejection_counts[reason] += 1
            rejections.append({"scenario_id": row.scenario_id, "reason": reason})
            continue
        retained_ids.append(scenario.scenario_id)
        for spec, tensorized in valid_samples:
            offset = len(tensorized_samples)
            tensorized_samples.append(tensorized)
            samples.append(
                {
                    "offset": offset,
                    "sample_id": tensorized.sample_id,
                    "scenario_id": tensorized.scenario_id,
                    "target_track_id": tensorized.target_track_id,
                    "spec": _spec_to_dict(spec),
                    "map_clip_statistics": _map_clip_statistics_to_dict(
                        tensorized.map_clip_statistics
                    ),
                }
            )

    map_context_indices, map_contexts = _deduplicate_map_contexts(tensorized_samples)
    _atomic_torch_save(
        shard_path,
        {
            "version": CACHE_VERSION,
            "partition": partition,
            "sample_ids": [sample.sample_id for sample in tensorized_samples],
            "scenario_ids": [sample.scenario_id for sample in tensorized_samples],
            "target_track_ids": [
                sample.target_track_id for sample in tensorized_samples
            ],
            "map_context_indices": map_context_indices,
            "tensors": _stack_tensorized_samples(
                tensorized_samples,
                _SAMPLE_TENSOR_FIELDS,
            ),
            "map_contexts": _stack_tensorized_samples(
                map_contexts,
                _MAP_TENSOR_FIELDS,
            ),
        },
    )
    sidecar = {
        **expected,
        "shard_path": shard_path.name,
        "shard_sha256": _sha256(shard_path),
        "retained_scenario_ids": retained_ids,
        "samples": samples,
        "rejections": rejections,
        "counts": {
            "input_scenarios": len(rows),
            "retained_scenarios": len(retained_ids),
            "rejected_scenarios": len(rows) - len(retained_ids),
            "retained_samples": len(samples),
            "rejected_samples": rejected_samples,
            "map_contexts": len(map_contexts),
            "deduplicated_map_sample_copies": len(samples) - len(map_contexts),
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sample_spec_statistics": _sample_spec_statistics(
            sample_spec_counts,
            sample_spec_rejection_reasons,
        ),
        "map_clip_statistics": _map_clip_statistics(samples),
    }
    _atomic_write_json(sidecar_path, sidecar)
    return sidecar


def _progress_line(
    partition: str,
    completed: int,
    total: int,
    skipped_shards: int,
    elapsed: float,
) -> str:
    width = 24
    fraction = 1.0 if total == 0 else completed / total
    filled = min(width, int(fraction * width))
    speed = completed / elapsed if elapsed > 0 else 0.0
    remaining = 0.0 if speed <= 0 else max(total - completed, 0) / speed
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"prepare {partition} [{bar}] {fraction * 100:6.2f}% "
        f"{completed}/{total} scenarios {speed:.2f} scenarios/s "
        f"ETA {remaining:.1f}s skipped_shards={skipped_shards}"
    )


def prepare_cvae_partition(
    config: CVAEConfig,
    partition: str,
    *,
    project_root: str | Path = ".",
    schema: CVAESchema | None = None,
    scenario_loader: ScenarioLoader = load_av2_scenario,
    validation_labeler: ValidationLabeler | None = None,
    limit: int | None = None,
    force: bool = False,
    progress_stream: TextIO | None = None,
    pool_sha256: str | None = None,
) -> dict[str, Any]:
    """Build or resume one manifest-ordered scenario cache partition."""
    if partition not in _PARTITIONS:
        raise ValueError(f"unknown CVAE cache partition: {partition}")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("limit must be a positive integer")
    _validate_config_contract(config)
    root = Path(project_root)
    schema = schema or build_cvae_schema(root / config.data.skill_dir)
    schema_sha256 = _schema_fingerprint(schema)
    manifest_path = root / getattr(config.data.manifests, partition)
    candidate_pool_path = root / config.data.formal_candidate_pool
    pool_sha256 = pool_sha256 or _sha256(candidate_pool_path)
    rows = read_manifest(manifest_path)
    expected_split, is_training = _PARTITIONS[partition]
    if len({row.scenario_id for row in rows}) != len(rows):
        raise ValueError(f"{partition} manifest contains duplicate scenario IDs")
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"{partition} manifest rows must use split={expected_split}")

    cache_dir = root / config.cache.root / partition
    shard_dir = cache_dir / SHARD_DIRECTORY
    input_hashes = {
        "config_sha256": _cache_config_fingerprint(config),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_rows_sha256": _manifest_rows_fingerprint(rows),
        "schema_sha256": schema_sha256,
        "candidate_pool_sha256": pool_sha256,
        "labeler": "training_pool_observed_only" if is_training else _labeler_identity(validation_labeler),
    }
    manifest_file = cache_dir / CACHE_MANIFEST_NAME
    if force:
        _cleanup_partition(cache_dir)
    elif manifest_file.exists():
        existing = _read_json(manifest_file)
        if existing.get("version") != CACHE_VERSION:
            raise ValueError(
                "existing CVAE cache version is incompatible; rerun with force=True"
            )
        if existing.get("inputs") != input_hashes:
            raise ValueError("cache inputs differ from existing cache; rerun with force=True")
        previous = int(existing.get("counts", {}).get("processed_manifest_scenarios", 0))
        if limit is not None:
            limit = max(limit, previous)

    selected_rows = rows if limit is None else rows[:limit]
    selected_ids = {row.scenario_id for row in selected_rows}
    if is_training:
        observed_by_scenario, label_counts = _stream_training_specs(
            candidate_pool_path,
            selected_ids,
            schema,
        )
    else:
        observed_by_scenario = {}
        label_counts = {"validation_labeler_specs": 0}

    stream = progress_stream or sys.stdout
    started = time.perf_counter()
    sidecars: list[dict[str, Any]] = []
    skipped_shards = 0
    completed = 0
    print(
        "\r" + _progress_line(partition, completed, len(selected_rows), skipped_shards, 1e-9),
        end="",
        flush=True,
        file=stream,
    )
    for shard_index, start in enumerate(range(0, len(selected_rows), config.cache.shard_size)):
        shard_rows = selected_rows[start : start + config.cache.shard_size]
        shard_name = f"shard-{shard_index:05d}.pt"
        sidecar_name = f"shard-{shard_index:05d}.json"
        shard_path = shard_dir / shard_name
        sidecar_path = shard_dir / sidecar_name
        expected = {
            "version": CACHE_VERSION,
            "partition": partition,
            "shard_index": shard_index,
            "input_scenario_ids": [row.scenario_id for row in shard_rows],
            **input_hashes,
        }
        sidecar = None if force else _verified_sidecar(shard_path, sidecar_path, expected)
        if sidecar is None:
            sidecar = _build_shard(
                rows=shard_rows,
                shard_path=shard_path,
                sidecar_path=sidecar_path,
                expected=expected,
                partition=partition,
                is_training=is_training,
                data_root=root / config.data.root,
                schema=schema,
                observed_specs=observed_by_scenario,
                scenario_loader=scenario_loader,
                validation_labeler=validation_labeler,
            )
        else:
            skipped_shards += 1
        sidecars.append(sidecar)
        completed += len(shard_rows)
        print(
            "\r"
            + _progress_line(
                partition,
                completed,
                len(selected_rows),
                skipped_shards,
                max(time.perf_counter() - started, 1e-9),
            ),
            end="",
            flush=True,
            file=stream,
        )
    print(file=stream)

    index_lines: list[bytes] = []
    shard_summaries: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    for sidecar in sidecars:
        relative_shard = f"{SHARD_DIRECTORY}/{sidecar['shard_path']}"
        for sample in sidecar["samples"]:
            index_lines.append(
                _canonical_json(
                    {
                        "sample_id": sample["sample_id"],
                        "scenario_id": sample["scenario_id"],
                        "target_track_id": sample["target_track_id"],
                        "shard": relative_shard,
                        "offset": sample["offset"],
                        "spec": sample["spec"],
                    }
                )
                + b"\n"
            )
        totals.update(sidecar["counts"])
        rejection_counts.update(sidecar["rejection_counts"])
        shard_summaries.append(
            {
                "shard_index": sidecar["shard_index"],
                "path": relative_shard,
                "sidecar": f"{SHARD_DIRECTORY}/shard-{sidecar['shard_index']:05d}.json",
                "sha256": sidecar["shard_sha256"],
                "counts": sidecar["counts"],
                "sample_spec_statistics": sidecar["sample_spec_statistics"],
                "map_clip_statistics": sidecar["map_clip_statistics"],
            }
        )
    retained_observed_samples = sum(
        1
        for sidecar in sidecars
        for sample in sidecar["samples"]
        if sample["spec"]["skill_supervision_mask"]
    )
    label_counts["retained_observed_samples"] = retained_observed_samples
    label_counts["retained_base_samples"] = len(index_lines) - retained_observed_samples
    if not is_training:
        label_counts["validation_labeler_specs"] = retained_observed_samples
    index_path = cache_dir / SAMPLE_INDEX_NAME
    _atomic_write_bytes(index_path, b"".join(index_lines))
    counts = {
        "manifest_scenarios": len(rows),
        "processed_manifest_scenarios": len(selected_rows),
        **dict(sorted(totals.items())),
    }
    cache_manifest = {
        "version": CACHE_VERSION,
        "status": "complete" if len(selected_rows) == len(rows) else "partial",
        "partition": partition,
        "inputs": input_hashes,
        "shard_size": config.cache.shard_size,
        "in_memory_shards": config.cache.in_memory_shards,
        "counts": counts,
        "label_counts": label_counts,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sample_spec_statistics": _aggregate_sample_spec_statistics(
            sidecar["sample_spec_statistics"] for sidecar in sidecars
        ),
        "map_clip_statistics": _map_clip_statistics(
            sample
            for sidecar in sidecars
            for sample in sidecar["samples"]
        ),
        "shards": shard_summaries,
        "sample_index": {
            "path": SAMPLE_INDEX_NAME,
            "sha256": _sha256(index_path),
            "records": len(index_lines),
        },
        "resume": {
            "verified_skipped_shards": skipped_shards,
            "rebuilt_shards": len(sidecars) - skipped_shards,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    _atomic_write_json(manifest_file, cache_manifest)
    return cache_manifest


def prepare_cvae_split(
    config: CVAEConfig,
    split: str,
    *,
    project_root: str | Path = ".",
    schema: CVAESchema | None = None,
    scenario_loader: ScenarioLoader = load_av2_scenario,
    validation_labeler: ValidationLabeler | None = None,
    limit: int | None = None,
    force: bool = False,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Prepare the train/validation pair for one development or formal split."""
    if split not in _SPLITS:
        raise ValueError(f"split must be one of {sorted(_SPLITS)}")
    root = Path(project_root)
    schema = schema or build_cvae_schema(root / config.data.skill_dir)
    pool_sha256 = _sha256(root / config.data.formal_candidate_pool)
    partitions = {
        partition: prepare_cvae_partition(
            config,
            partition,
            project_root=root,
            schema=schema,
            scenario_loader=scenario_loader,
            validation_labeler=validation_labeler,
            limit=limit,
            force=force,
            progress_stream=progress_stream,
            pool_sha256=pool_sha256,
        )
        for partition in _SPLITS[split]
    }
    return {"split": split, "partitions": partitions}


class CVAECachedDataset(Dataset[dict[str, Any]]):
    """Read indexed tensor samples while keeping a small LRU of mapped shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        schema: CVAESchema | None = None,
        in_memory_shards: int | None = None,
        sample_index_path: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.schema = schema or build_cvae_schema()
        self.cache_manifest = _read_json(self.cache_dir / CACHE_MANIFEST_NAME)
        if self.cache_manifest.get("version") != CACHE_VERSION:
            raise ValueError("CVAE cache_manifest version is incompatible")
        source_index_path = self.cache_dir / self.cache_manifest["sample_index"]["path"]
        if _sha256(source_index_path) != self.cache_manifest["sample_index"]["sha256"]:
            raise ValueError("CVAE sample index hash differs from cache_manifest")
        source_entries = [
            json.loads(line)
            for line in source_index_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.source_sample_index_path = source_index_path
        if sample_index_path is None:
            self.sample_index_path = source_index_path
            self.entries = source_entries
        else:
            view_path = Path(sample_index_path)
            view_entries = [
                json.loads(line)
                for line in view_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            source_by_offset: dict[tuple[str, int], dict[str, Any]] = {}
            for entry in source_entries:
                shard = entry.get("shard")
                offset = entry.get("offset")
                if not isinstance(shard, str) or isinstance(offset, bool) or not isinstance(
                    offset, int
                ):
                    raise ValueError("source CVAE sample index has an invalid shard offset")
                key = (shard, offset)
                if key in source_by_offset:
                    raise ValueError("source CVAE sample index has duplicate shard offsets")
                source_by_offset[key] = entry
            seen: set[tuple[str, int]] = set()
            for entry in view_entries:
                shard = entry.get("shard")
                offset = entry.get("offset")
                if not isinstance(shard, str) or isinstance(offset, bool) or not isinstance(
                    offset, int
                ):
                    raise ValueError("explicit CVAE sample view has an invalid shard offset")
                key = (shard, offset)
                source = source_by_offset.get(key)
                if source is None or source != entry:
                    raise ValueError(
                        "explicit CVAE sample view contains an entry outside the source index"
                    )
                if key in seen:
                    raise ValueError("explicit CVAE sample view contains duplicate entries")
                seen.add(key)
            self.sample_index_path = view_path
            self.entries = view_entries
        self.sample_index_sha256 = _sha256(self.sample_index_path)
        maximum = (
            self.cache_manifest["in_memory_shards"]
            if in_memory_shards is None
            else in_memory_shards
        )
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError("in_memory_shards must be a positive integer")
        self.in_memory_shards = maximum
        self._shards: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._verified_shards: set[str] = set()
        self._shard_hashes = {
            shard["path"]: shard["sha256"] for shard in self.cache_manifest["shards"]
        }

    def __len__(self) -> int:
        return len(self.entries)

    def _shard(self, shard_name: str) -> dict[str, Any]:
        payload = self._shards.get(shard_name)
        if payload is None:
            shard_path = self.cache_dir / shard_name
            if shard_name not in self._verified_shards:
                if self._shard_hashes.get(shard_name) != _sha256(shard_path):
                    raise ValueError(
                        f"CVAE shard hash differs from cache_manifest: {shard_name}"
                    )
                self._verified_shards.add(shard_name)
            payload = _load_shard(shard_path)
            if payload["partition"] != self.cache_manifest["partition"]:
                raise ValueError(f"CVAE shard partition differs: {shard_name}")
            self._shards[shard_name] = payload
            while len(self._shards) > self.in_memory_shards:
                self._shards.popitem(last=False)
        else:
            self._shards.move_to_end(shard_name)
        return payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        payload = self._shard(entry["shard"])
        offset = entry.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise ValueError("CVAE sample index offset must be an integer")
        if offset < 0 or offset >= len(payload["sample_ids"]):
            raise ValueError("CVAE sample index offset is outside its shard")
        identifiers = {
            "sample_id": payload["sample_ids"][offset],
            "scenario_id": payload["scenario_ids"][offset],
            "target_track_id": payload["target_track_ids"][offset],
        }
        if any(entry.get(name) != value for name, value in identifiers.items()):
            raise ValueError("CVAE sample index identifiers differ from its shard")
        result: dict[str, Any] = {
            **identifiers,
        }
        map_context_index = int(payload["map_context_indices"][offset])
        for name in _TENSOR_FIELDS:
            if name in _MAP_TENSOR_FIELDS:
                result[name] = payload["map_contexts"][name][map_context_index]
            else:
                result[name] = payload["tensors"][name][offset]
        return result


class ShardShuffleSampler(Sampler[int]):
    """Shuffle shards and samples reproducibly while preserving disk locality."""

    def __init__(self, dataset: CVAECachedDataset, *, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self.stop_index: int | None = None
        groups: dict[str, list[int]] = defaultdict(list)
        for index, entry in enumerate(dataset.entries):
            groups[entry["shard"]].append(index)
        self.groups = tuple(
            (name, tuple(indices)) for name, indices in sorted(groups.items())
        )

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def set_range(self, start_index: int = 0, stop_index: int | None = None) -> None:
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError("start_index must be a nonnegative integer")
        if start_index < 0:
            raise ValueError("start_index must be a nonnegative integer")
        if stop_index is not None:
            if isinstance(stop_index, bool) or not isinstance(stop_index, int):
                raise ValueError("stop_index must be an integer when present")
            if stop_index < start_index:
                raise ValueError("stop_index must not precede start_index")
        self.start_index = min(start_index, len(self.dataset))
        self.stop_index = (
            None if stop_index is None else min(stop_index, len(self.dataset))
        )

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        shard_order = torch.randperm(len(self.groups), generator=generator).tolist()

        def ordered_indices():
            for shard_index in shard_order:
                _, indices = self.groups[shard_index]
                sample_order = torch.randperm(len(indices), generator=generator).tolist()
                yield from (indices[position] for position in sample_order)

        yield from itertools.islice(
            ordered_indices(),
            self.start_index,
            self.stop_index,
        )

    def __len__(self) -> int:
        stop = len(self.dataset) if self.stop_index is None else self.stop_index
        return max(stop - self.start_index, 0)


class ObservedSkillBalanceSampler(Sampler[int]):
    """Balance observed skills deterministically without dropping base samples.

    Every source sample appears once. Observed skills are then repeated toward the
    most frequent observed skill in the current training view, while no individual
    sample may appear more than ``max_repeats_per_sample`` times in one epoch.
    Expanded occurrences remain shard-grouped to preserve cache locality.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int,
        max_repeats_per_sample: int = 8,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("observed-skill sampler requires a non-empty dataset")
        if (
            isinstance(max_repeats_per_sample, bool)
            or not isinstance(max_repeats_per_sample, int)
            or not 1 <= max_repeats_per_sample <= 8
        ):
            raise ValueError("max_repeats_per_sample must be an integer from 1 to 8")
        entries = getattr(dataset, "entries", None)
        if not isinstance(entries, list) or len(entries) != len(dataset):
            raise ValueError("observed-skill sampler requires indexed dataset entries")
        self.dataset = dataset
        self.seed = int(seed)
        self.max_repeats_per_sample = max_repeats_per_sample
        self.epoch = 0
        self.start_index = 0
        self.stop_index: int | None = None
        base_indices: list[int] = []
        observed: dict[str, list[int]] = defaultdict(list)
        shard_by_index: list[str] = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError("dataset entry must be a mapping")
            shard = entry.get("shard")
            spec = entry.get("spec")
            if not isinstance(shard, str) or not isinstance(spec, Mapping):
                raise ValueError("dataset entry must contain shard and spec mappings")
            skill_id = spec.get("skill_id")
            supervised = spec.get("skill_supervision_mask")
            if not isinstance(skill_id, str) or not isinstance(supervised, bool):
                raise ValueError("dataset sample spec has an invalid skill contract")
            if skill_id == "<none>" and not supervised:
                base_indices.append(index)
            elif skill_id != "<none>" and supervised:
                observed[skill_id].append(index)
            else:
                raise ValueError(
                    "balanced training accepts only base or observed-skill samples"
                )
            shard_by_index.append(shard)
        if not base_indices:
            raise ValueError("balanced training requires at least one base sample")
        if not observed:
            raise ValueError("balanced training requires observed-skill samples")
        self.base_indices = tuple(base_indices)
        self.observed_groups = tuple(
            (skill_id, tuple(indices)) for skill_id, indices in sorted(observed.items())
        )
        self.shard_by_index = tuple(shard_by_index)
        target = max(len(indices) for _, indices in self.observed_groups)
        self.target_observed_exposure = target
        self.observed_exposure = {
            skill_id: min(target, len(indices) * max_repeats_per_sample)
            for skill_id, indices in self.observed_groups
        }
        self.epoch_size = len(self.base_indices) + sum(self.observed_exposure.values())

    @property
    def contract(self) -> dict[str, Any]:
        return {
            "strategy": "observed_skill_balance_v1",
            "target": "most_frequent_observed",
            "seed": self.seed,
            "source_samples": len(self.dataset),
            "base_samples": len(self.base_indices),
            "observed_source_by_skill": {
                skill_id: len(indices) for skill_id, indices in self.observed_groups
            },
            "observed_epoch_exposure_by_skill": dict(self.observed_exposure),
            "target_observed_exposure": self.target_observed_exposure,
            "max_repeats_per_sample": self.max_repeats_per_sample,
            "epoch_samples": self.epoch_size,
        }

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def set_range(self, start_index: int = 0, stop_index: int | None = None) -> None:
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise ValueError("start_index must be a nonnegative integer")
        if start_index < 0:
            raise ValueError("start_index must be a nonnegative integer")
        if stop_index is not None:
            if isinstance(stop_index, bool) or not isinstance(stop_index, int):
                raise ValueError("stop_index must be an integer when present")
            if stop_index < start_index:
                raise ValueError("stop_index must not precede start_index")
        self.start_index = min(start_index, self.epoch_size)
        self.stop_index = (
            None if stop_index is None else min(stop_index, self.epoch_size)
        )

    def _epoch_order(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        occurrences: list[int] = list(self.base_indices)
        for skill_id, indices in self.observed_groups:
            desired = self.observed_exposure[skill_id]
            full_repeats, remainder = divmod(desired, len(indices))
            if full_repeats > self.max_repeats_per_sample:
                raise AssertionError("sampler repeat cap was exceeded")
            occurrences.extend(index for index in indices for _ in range(full_repeats))
            if remainder:
                order = torch.randperm(len(indices), generator=generator).tolist()
                occurrences.extend(indices[position] for position in order[:remainder])

        by_shard: dict[str, list[int]] = defaultdict(list)
        for index in occurrences:
            by_shard[self.shard_by_index[index]].append(index)
        shard_names = sorted(by_shard)
        shard_order = torch.randperm(len(shard_names), generator=generator).tolist()
        result: list[int] = []
        for position in shard_order:
            shard_occurrences = by_shard[shard_names[position]]
            order = torch.randperm(len(shard_occurrences), generator=generator).tolist()
            result.extend(shard_occurrences[index] for index in order)
        if len(result) != self.epoch_size:
            raise AssertionError("balanced sampler epoch size changed")
        return result

    def exposure(self) -> dict[str, Any]:
        """Describe exactly the currently selected epoch range."""

        stop = self.epoch_size if self.stop_index is None else self.stop_index
        indices = self._epoch_order()[self.start_index:stop]
        base = 0
        observed: Counter[str] = Counter()
        sample_repeats: Counter[int] = Counter(indices)
        for index in indices:
            spec = self.dataset.entries[index]["spec"]
            if spec["skill_supervision_mask"]:
                observed[spec["skill_id"]] += 1
            else:
                base += 1
        return {
            "epoch": self.epoch,
            "range_start": self.start_index,
            "range_stop": stop,
            "samples": len(indices),
            "base": base,
            "observed_by_skill": dict(sorted(observed.items())),
            "maximum_sample_repeats_in_range": max(sample_repeats.values(), default=0),
        }

    def __iter__(self):
        stop = self.epoch_size if self.stop_index is None else self.stop_index
        yield from self._epoch_order()[self.start_index:stop]

    def __len__(self) -> int:
        stop = self.epoch_size if self.stop_index is None else self.stop_index
        return max(stop - self.start_index, 0)


__all__ = [
    "CACHE_MANIFEST_NAME",
    "CACHE_VERSION",
    "CVAECachedDataset",
    "ObservedSkillBalanceSampler",
    "ShardShuffleSampler",
    "cvae_schema_fingerprint",
    "SAMPLE_INDEX_NAME",
    "ScenarioLoader",
    "ValidationLabeler",
    "prepare_cvae_partition",
    "prepare_cvae_split",
]

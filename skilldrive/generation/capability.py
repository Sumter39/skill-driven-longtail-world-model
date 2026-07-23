"""Build the frozen per-skill generation capability matrix for stage A."""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.seeds.records import SeedRecord


def _training_retained_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    statistics = manifest.get("sample_spec_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError("training cache manifest is missing sample_spec_statistics")
    by_skill = statistics.get("by_skill")
    if not isinstance(by_skill, Mapping):
        raise ValueError("training cache manifest is missing by_skill statistics")
    retained: dict[str, int] = {}
    for skill_id, value in by_skill.items():
        if not isinstance(skill_id, str) or not isinstance(value, Mapping):
            raise ValueError("training by_skill statistics are malformed")
        count = value.get("retained")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"training retained count is invalid for {skill_id}")
        retained[skill_id] = count
    return retained


def build_generation_capability_matrix(
    *,
    config: CounterfactualGenerationConfig,
    records: Iterable[SeedRecord],
    training_cache_manifest: Mapping[str, Any],
    checkpoint_path: str,
    checkpoint_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    """Summarize evidence without claiming unverified generation control."""

    retained = _training_retained_counts(training_cache_manifest)
    formal_ids = set(config.formal_skill_ids)
    record_count: Counter[str] = Counter()
    scenario_ids: dict[str, set[str]] = {skill_id: set() for skill_id in formal_ids}
    target_track_ids: dict[str, set[str]] = {skill_id: set() for skill_id in formal_ids}
    detection_modes: dict[str, Counter[str]] = {
        skill_id: Counter() for skill_id in formal_ids
    }
    parameter_names: dict[str, set[str]] = {skill_id: set() for skill_id in formal_ids}
    configs = config.skills_by_id

    for record in records:
        if record.skill_id not in formal_ids:
            raise ValueError(f"seed record references a non-formal skill: {record.skill_id}")
        skill_config = configs[record.skill_id]
        try:
            target_track_id = record.role_track_ids[skill_config.primary_generated_role]
        except KeyError:
            raise ValueError(
                f"seed record for {record.skill_id} is missing primary role "
                f"{skill_config.primary_generated_role}"
            ) from None
        mode = record.evidence.get("detection_mode")
        if mode not in {"observed_trigger", "compatible_seed"}:
            raise ValueError(
                f"seed record for {record.skill_id} has invalid detection_mode: {mode!r}"
            )
        record_count[record.skill_id] += 1
        scenario_ids[record.skill_id].add(record.scenario_id)
        target_track_ids[record.skill_id].add(target_track_id)
        detection_modes[record.skill_id][mode] += 1
        parameter_names[record.skill_id].update(record.sampled_parameters)

    missing_records = [skill_id for skill_id in config.formal_skill_ids if not record_count[skill_id]]
    if missing_records:
        raise ValueError(f"formal skills have no seed records: {missing_records}")

    entries: list[dict[str, Any]] = []
    for skill_config in config.skills:
        skill_id = skill_config.skill_id
        observed_training_samples = retained.get(skill_id, 0)
        if skill_config.proposal_mode == "learned_conditioned_prior":
            if observed_training_samples <= 0:
                raise ValueError(
                    f"learned-conditioned skill has no retained training samples: {skill_id}"
                )
            support_status = "direct"
        elif observed_training_samples != 0:
            raise ValueError(
                f"rule-guided skill unexpectedly has retained training samples: {skill_id}"
            )
        elif skill_config.joint_generation_limited:
            support_status = "single_target_limited"
        else:
            support_status = "search_only"

        entries.append(
            {
                "skill_id": skill_id,
                "observed_training_samples": observed_training_samples,
                "seed_records": record_count[skill_id],
                "unique_seed_scenarios": len(scenario_ids[skill_id]),
                "unique_primary_target_tracks": len(target_track_ids[skill_id]),
                "detection_modes": dict(sorted(detection_modes[skill_id].items())),
                "primary_generated_role": skill_config.primary_generated_role,
                "proposal_mode": skill_config.proposal_mode,
                "condition_skill_id": skill_config.condition_skill_id(
                    config.none_skill_id
                ),
                "parameter_conditioning_supported": False,
                "requested_parameter_fields": sorted(parameter_names[skill_id]),
                "joint_generation_required": skill_config.joint_generation_limited,
                "current_support_status": support_status,
            }
        )

    return {
        "version": 1,
        "status": "initial_evidence_only",
        "active_checkpoint": {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256,
            "schema_sha256": schema_sha256,
        },
        "formal_skill_count": len(entries),
        "seed_record_count": sum(record_count.values()),
        "unique_seed_scenarios": len(
            {scenario_id for values in scenario_ids.values() for scenario_id in values}
        ),
        "skills": entries,
    }


def write_generation_capability_matrix(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write one canonical UTF-8 capability matrix."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "build_generation_capability_matrix",
    "write_generation_capability_matrix",
]

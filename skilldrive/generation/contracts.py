"""Stable identifiers and small schemas for generated trajectory overlays."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Literal, Mapping

import numpy as np


OVERLAY_FUTURE_STEPS = 60
MAX_LATENT_SEED = 2**63 - 1
TASK_STATUSES = (
    "pending",
    "raw_committed",
    "filter_committed",
    "complete",
    "failed",
)
TaskStatus = Literal[
    "pending",
    "raw_committed",
    "filter_committed",
    "complete",
    "failed",
]


class FilterRejection(str, Enum):
    """Stable machine-readable reasons emitted by the frozen filter contract."""

    INVALID_FUTURE_SHAPE = "schema.invalid_future_shape"
    INVALID_FUTURE_DTYPE = "schema.invalid_future_dtype"
    INVALID_SCENARIO_LENGTH = "schema.invalid_scenario_length"
    TARGET_TRACK_MISSING = "schema.target_track_missing"
    INVALID_TARGET_LENGTH = "schema.invalid_target_length"
    NON_MONOTONIC_TIMESTAMPS = "schema.non_monotonic_timestamps"
    INVALID_SAMPLE_PERIOD = "schema.invalid_sample_period"
    TARGET_ANCHOR_NOT_OBSERVED = "schema.target_anchor_not_observed"
    NON_FINITE_GENERATED_POSITIONS = "finite.generated_positions"
    NON_FINITE_TARGET_ANCHOR = "finite.target_anchor"
    NON_FINITE_DERIVED_KINEMATICS = "finite.derived_kinematics"
    HISTORY_TIMESTAMPS_CHANGED = "history.timestamps_changed"
    HISTORY_TARGET_CHANGED = "history.target_changed"
    BACKGROUND_TRACK_CHANGED = "history.background_track_changed"
    MAP_CHANGED = "history.map_changed"
    METADATA_CHANGED = "history.metadata_changed"
    COORDINATE_ROUND_TRIP_EXCEEDED = "history.coordinate_round_trip_exceeded"
    SEAM_SPEED_LIMIT_EXCEEDED = "kinematics.seam_speed_limit_exceeded"
    SPEED_LIMIT_EXCEEDED = "kinematics.speed_limit_exceeded"
    ACCELERATION_LIMIT_EXCEEDED = "kinematics.acceleration_limit_exceeded"
    DECELERATION_LIMIT_EXCEEDED = "kinematics.deceleration_limit_exceeded"
    JERK_LIMIT_EXCEEDED = "kinematics.jerk_limit_exceeded"
    CURVATURE_LIMIT_EXCEEDED = "kinematics.curvature_limit_exceeded"
    HEADING_RATE_LIMIT_EXCEEDED = "kinematics.heading_rate_limit_exceeded"
    KINEMATIC_CLASS_UNSUPPORTED = "kinematics.class_unsupported"
    DRIVABLE_AREA_UNAVAILABLE = "map.drivable_area_unavailable"
    OUTSIDE_DRIVABLE_AREA = "map.outside_drivable_area"
    LANE_GEOMETRY_UNAVAILABLE = "map.lane_geometry_unavailable"
    LANE_ASSIGNMENT_INSUFFICIENT = "map.lane_assignment_insufficient"
    LANE_TYPE_INCOMPATIBLE = "map.lane_type_incompatible"
    LANE_CONNECTIVITY_VIOLATION = "map.lane_connectivity_violation"
    LANE_DIRECTION_VIOLATION = "map.lane_direction_violation"
    COLLISION_PROXY_UNAVAILABLE = "collision.class_proxy_unavailable"
    COLLISION_PROXY_OVERLAP = "collision.class_proxy_overlap"
    RISK_METRIC_UNAVAILABLE = "risk.metric_unavailable"
    RISK_METRIC_MISMATCH = "risk.metric_mismatch"
    RISK_NON_FINITE = "risk.non_finite"
    RISK_OUT_OF_TARGET_RANGE = "risk.out_of_target_range"
    SKILL_ROLE_CONTRACT_MISMATCH = "skill.role_contract_mismatch"
    OBSERVED_ROLE_CONTRACT_MISMATCH = "skill.observed_role_contract_mismatch"
    OBSERVED_SKILL_NOT_REDETECTED = "skill.observed_exact_roles_not_redetected"
    SKILL_TRIGGER_NOT_REALIZED = "skill.trigger_not_realized"
    NOVELTY_REFERENCE_UNAVAILABLE = "skill.novelty_reference_unavailable"
    NOVELTY_INSUFFICIENT = "skill.novelty_insufficient"
    PARAMETER_NON_FINITE = "parameter.non_finite"
    PARAMETER_OUT_OF_TOLERANCE = "parameter.out_of_tolerance"
    DIVERSITY_SCENARIO_SKILL_LIMIT = "diversity.scenario_skill_limit"
    DIVERSITY_TRAJECTORY_TOO_SIMILAR = "diversity.trajectory_too_similar"
    DIVERSITY_GLOBAL_DUPLICATE = "diversity.global_duplicate"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    "pending": frozenset(("raw_committed", "failed")),
    "raw_committed": frozenset(("filter_committed", "failed")),
    "filter_committed": frozenset(("complete", "failed")),
    "complete": frozenset(),
    "failed": frozenset(),
}


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256_text(value: Any, name: str) -> str:
    text = _required_text(value, name).lower()
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _json_value(value: Any, name: str = "value") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} cannot contain non-finite numbers")
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} cannot contain non-finite numbers")
        return number
    if isinstance(value, (list, tuple)):
        return [_json_value(item, name) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings")
            result[key] = _json_value(item, name)
        return result
    raise ValueError(f"{name} must contain only JSON-compatible values")


def canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    """Serialize JSON-compatible data deterministically and reject NaN/Inf."""

    normalized = _json_value(value)
    suffix = "\n" if indent is not None else ""
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
            indent=indent,
            allow_nan=False,
        )
        + suffix
    ).encode("utf-8")


def canonical_json(value: Any) -> str:
    """Return canonical JSON text for IDs and JSONL records."""

    return canonical_json_bytes(value).decode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash one canonical JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class GenerationConfigFingerprints:
    """Keep semantic identity separate from performance execution settings."""

    semantic_config_sha256: str
    execution_config_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_config_sha256",
            _sha256_text(self.semantic_config_sha256, "semantic_config_sha256"),
        )
        object.__setattr__(
            self,
            "execution_config_sha256",
            _sha256_text(self.execution_config_sha256, "execution_config_sha256"),
        )

    @classmethod
    def from_configs(
        cls,
        semantic_config: Mapping[str, Any],
        execution_config: Mapping[str, Any],
    ) -> "GenerationConfigFingerprints":
        return cls(
            semantic_config_sha256=canonical_sha256(semantic_config),
            execution_config_sha256=canonical_sha256(execution_config),
        )


def generation_task_id(
    *,
    seed_record_id: str,
    scenario_id: str,
    skill_id: str,
    target_track_id: str,
    proposal_mode: str,
    condition_skill_id: str,
    checkpoint_sha256: str,
    semantic_config_sha256: str,
) -> str:
    """Build a task ID without candidate budget or execution tuning knobs."""

    return canonical_sha256(
        {
            "version": 1,
            "seed_record_id": _sha256_text(seed_record_id, "seed_record_id"),
            "scenario_id": _required_text(scenario_id, "scenario_id"),
            "skill_id": _required_text(skill_id, "skill_id"),
            "target_track_id": _required_text(target_track_id, "target_track_id"),
            "proposal_mode": _required_text(proposal_mode, "proposal_mode"),
            "condition_skill_id": _required_text(
                condition_skill_id,
                "condition_skill_id",
            ),
            "checkpoint_sha256": _sha256_text(
                checkpoint_sha256,
                "checkpoint_sha256",
            ),
            "semantic_config_sha256": _sha256_text(
                semantic_config_sha256,
                "semantic_config_sha256",
            ),
        }
    )


@dataclass(frozen=True)
class GenerationTask:
    """One deterministic seed/skill/target request before candidate sampling."""

    task_id: str
    task_index: int
    seed_record_id: str
    scenario_id: str
    skill_id: str
    target_track_id: str
    proposal_mode: str
    condition_skill_id: str
    candidate_budget: int
    checkpoint_sha256: str
    semantic_config_sha256: str
    status: TaskStatus = "pending"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _sha256_text(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "seed_record_id",
            _sha256_text(self.seed_record_id, "seed_record_id"),
        )
        for name in (
            "scenario_id",
            "skill_id",
            "target_track_id",
            "proposal_mode",
            "condition_skill_id",
        ):
            _required_text(getattr(self, name), name)
        if isinstance(self.task_index, bool) or not isinstance(self.task_index, int):
            raise ValueError("task_index must be a nonnegative integer")
        if self.task_index < 0:
            raise ValueError("task_index must be a nonnegative integer")
        if (
            isinstance(self.candidate_budget, bool)
            or not isinstance(self.candidate_budget, int)
            or self.candidate_budget <= 0
        ):
            raise ValueError("candidate_budget must be a positive integer")
        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256_text(self.checkpoint_sha256, "checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "semantic_config_sha256",
            _sha256_text(self.semantic_config_sha256, "semantic_config_sha256"),
        )
        if self.status not in TASK_STATUSES:
            raise ValueError(f"unknown task status: {self.status!r}")
        expected = generation_task_id(
            seed_record_id=self.seed_record_id,
            scenario_id=self.scenario_id,
            skill_id=self.skill_id,
            target_track_id=self.target_track_id,
            proposal_mode=self.proposal_mode,
            condition_skill_id=self.condition_skill_id,
            checkpoint_sha256=self.checkpoint_sha256,
            semantic_config_sha256=self.semantic_config_sha256,
        )
        if self.task_id != expected:
            raise ValueError("task_id differs from the canonical task contract")

    @classmethod
    def create(
        cls,
        *,
        task_index: int,
        seed_record_id: str,
        scenario_id: str,
        skill_id: str,
        target_track_id: str,
        proposal_mode: str,
        condition_skill_id: str,
        candidate_budget: int,
        checkpoint_sha256: str,
        semantic_config_sha256: str,
        status: TaskStatus = "pending",
    ) -> "GenerationTask":
        task_id = generation_task_id(
            seed_record_id=seed_record_id,
            scenario_id=scenario_id,
            skill_id=skill_id,
            target_track_id=target_track_id,
            proposal_mode=proposal_mode,
            condition_skill_id=condition_skill_id,
            checkpoint_sha256=checkpoint_sha256,
            semantic_config_sha256=semantic_config_sha256,
        )
        return cls(
            task_id=task_id,
            task_index=task_index,
            seed_record_id=seed_record_id,
            scenario_id=scenario_id,
            skill_id=skill_id,
            target_track_id=target_track_id,
            proposal_mode=proposal_mode,
            condition_skill_id=condition_skill_id,
            candidate_budget=candidate_budget,
            checkpoint_sha256=checkpoint_sha256,
            semantic_config_sha256=semantic_config_sha256,
            status=status,
        )

    def transition(self, status: TaskStatus) -> "GenerationTask":
        """Return an idempotently advanced task while rejecting invalid jumps."""

        if status == self.status:
            return self
        if status not in TASK_STATUSES:
            raise ValueError(f"unknown task status: {status!r}")
        if status not in _TASK_TRANSITIONS[self.status]:
            raise ValueError(f"invalid task status transition: {self.status} -> {status}")
        return replace(self, status=status)


def candidate_id(
    *,
    task_id: str,
    candidate_index: int,
    latent_seed: int,
    checkpoint_sha256: str,
    semantic_config_sha256: str,
) -> str:
    """Identify one raw trajectory independently of execution configuration."""

    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise ValueError("candidate_index must be a nonnegative integer")
    if candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    if isinstance(latent_seed, bool) or not isinstance(latent_seed, int):
        raise ValueError("latent_seed must be an int64-compatible nonnegative integer")
    if not 0 <= latent_seed <= MAX_LATENT_SEED:
        raise ValueError("latent_seed must be an int64-compatible nonnegative integer")
    return canonical_sha256(
        {
            "version": 1,
            "task_id": _sha256_text(task_id, "task_id"),
            "candidate_index": candidate_index,
            "latent_seed": latent_seed,
            "checkpoint_sha256": _sha256_text(
                checkpoint_sha256,
                "checkpoint_sha256",
            ),
            "semantic_config_sha256": _sha256_text(
                semantic_config_sha256,
                "semantic_config_sha256",
            ),
        }
    )


def filter_evaluation_id(
    *,
    candidate_id: str,
    filter_config_sha256: str,
    filter_contract_version: str | int,
) -> str:
    """Identify one filter decision without changing the raw candidate ID."""

    if isinstance(filter_contract_version, bool) or not isinstance(
        filter_contract_version,
        (str, int),
    ):
        raise ValueError("filter_contract_version must be a string or integer")
    if isinstance(filter_contract_version, str) and not filter_contract_version:
        raise ValueError("filter_contract_version must not be empty")
    return canonical_sha256(
        {
            "version": 1,
            "candidate_id": _sha256_text(candidate_id, "candidate_id"),
            "filter_config_sha256": _sha256_text(
                filter_config_sha256,
                "filter_config_sha256",
            ),
            "filter_contract_version": filter_contract_version,
        }
    )


@dataclass(frozen=True)
class GeneratedOverlay:
    """One single-target, 60-step global trajectory replacement."""

    target_track_id: str
    future_xy_global: np.ndarray

    def __post_init__(self) -> None:
        _required_text(self.target_track_id, "target_track_id")
        positions = np.asarray(self.future_xy_global, dtype=np.float32)
        if positions.shape != (OVERLAY_FUTURE_STEPS, 2):
            raise ValueError(
                "future_xy_global must have shape "
                f"{(OVERLAY_FUTURE_STEPS, 2)}, got {positions.shape}"
            )
        if not np.isfinite(positions).all():
            raise ValueError("future_xy_global must contain only finite values")
        object.__setattr__(self, "future_xy_global", np.ascontiguousarray(positions.copy()))


Overlay = GeneratedOverlay


@dataclass(frozen=True)
class GeneratedCandidate:
    """Metadata and overlay for one deterministic Prior proposal."""

    task_id: str
    candidate_index: int
    latent_seed: int
    scenario_id: str
    skill_id: str
    proposal_mode: str
    checkpoint_sha256: str
    semantic_config_sha256: str
    overlay: GeneratedOverlay
    metadata: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = field(init=False)

    def __post_init__(self) -> None:
        task_id_value = _sha256_text(self.task_id, "task_id")
        checkpoint = _sha256_text(self.checkpoint_sha256, "checkpoint_sha256")
        semantic = _sha256_text(
            self.semantic_config_sha256,
            "semantic_config_sha256",
        )
        for name in ("scenario_id", "skill_id", "proposal_mode"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.overlay, GeneratedOverlay):
            raise TypeError("overlay must be a GeneratedOverlay")
        normalized_metadata = _json_value(self.metadata, "metadata")
        object.__setattr__(self, "task_id", task_id_value)
        object.__setattr__(self, "checkpoint_sha256", checkpoint)
        object.__setattr__(self, "semantic_config_sha256", semantic)
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(
            self,
            "candidate_id",
            candidate_id(
                task_id=task_id_value,
                candidate_index=self.candidate_index,
                latent_seed=self.latent_seed,
                checkpoint_sha256=checkpoint,
                semantic_config_sha256=semantic,
            ),
        )


@dataclass(frozen=True)
class FilterDecision:
    """One accepted or rejected evaluation of an existing raw candidate."""

    candidate_id: str
    filter_evaluation_id: str
    accepted: bool
    rejection_reasons: tuple[str | FilterRejection, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _sha256_text(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(
            self,
            "filter_evaluation_id",
            _sha256_text(self.filter_evaluation_id, "filter_evaluation_id"),
        )
        if not isinstance(self.accepted, bool):
            raise ValueError("accepted must be a boolean")
        reasons: list[str] = []
        for reason in self.rejection_reasons:
            value = reason.value if isinstance(reason, FilterRejection) else reason
            text = _required_text(value, "rejection_reasons item")
            try:
                normalized = FilterRejection(text).value
            except ValueError as error:
                raise ValueError(f"unknown filter rejection reason: {text}") from error
            reasons.append(normalized)
        normalized_reasons = tuple(reasons)
        if self.accepted and normalized_reasons:
            raise ValueError("accepted decisions cannot contain rejection reasons")
        if not self.accepted and not normalized_reasons:
            raise ValueError("rejected decisions must contain a rejection reason")
        object.__setattr__(self, "rejection_reasons", normalized_reasons)
        object.__setattr__(self, "metrics", _json_value(self.metrics, "metrics"))

    @classmethod
    def create(
        cls,
        *,
        candidate_id: str,
        filter_config_sha256: str,
        filter_contract_version: str | int,
        accepted: bool,
        rejection_reasons: tuple[str | FilterRejection, ...] = (),
        metrics: Mapping[str, Any] | None = None,
    ) -> "FilterDecision":
        return cls(
            candidate_id=candidate_id,
            filter_evaluation_id=filter_evaluation_id(
                candidate_id=candidate_id,
                filter_config_sha256=filter_config_sha256,
                filter_contract_version=filter_contract_version,
            ),
            accepted=accepted,
            rejection_reasons=rejection_reasons,
            metrics={} if metrics is None else metrics,
        )


__all__ = [
    "MAX_LATENT_SEED",
    "OVERLAY_FUTURE_STEPS",
    "TASK_STATUSES",
    "FilterDecision",
    "FilterRejection",
    "GeneratedCandidate",
    "GeneratedOverlay",
    "GenerationConfigFingerprints",
    "GenerationTask",
    "Overlay",
    "TaskStatus",
    "candidate_id",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "filter_evaluation_id",
    "generation_task_id",
]

"""Validated, deterministic CSV records for detected scenario seeds."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


SEED_CSV_FIELDS = (
    "scenario_id",
    "skill_id",
    "initiator_track_id",
    "responder_track_id",
    "role_track_ids_json",
    "trigger_score",
    "seed_risk_metric",
    "seed_risk_value",
    "target_risk_definition_json",
    "source_path",
    "evidence_json",
    "sampled_parameters_json",
)

TARGET_RISK_DEFINITION_FIELDS = {
    "metric",
    "target_range",
    "source",
    "direction",
}
TARGET_RISK_SOURCES = {"semantic", "train_statistics", "reference"}
TARGET_RISK_DIRECTIONS = {"lower_is_riskier", "higher_is_riskier"}


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} cannot contain non-finite numbers")
        return value
    if isinstance(value, list):
        return [_json_value(item, name) for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{name} keys must be non-empty strings")
            result[key] = _json_value(item, name)
        return result
    raise ValueError(f"{name} must contain only JSON-compatible values")


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object")
    return _json_value(value, name)


def _role_track_ids(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or len(value) < 2:
        raise ValueError("role_track_ids must map at least two roles to tracks")
    result: dict[str, str] = {}
    for role, track_id in value.items():
        result[_required_text(role, "role_track_ids role")] = _required_text(
            track_id,
            "role_track_ids track_id",
        )
    if len(set(result.values())) != len(result):
        raise ValueError("role_track_ids must reference distinct tracks")
    return result


def _target_risk_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != TARGET_RISK_DEFINITION_FIELDS:
        raise ValueError(
            "target_risk_definition must contain exactly metric, target_range, "
            "source, and direction"
        )
    metric = _required_text(value["metric"], "target_risk_definition.metric")
    target_range = value["target_range"]
    if not isinstance(target_range, list) or len(target_range) != 2:
        raise ValueError("target_risk_definition.target_range must contain two numbers")
    low = _finite_number(target_range[0], "target_risk_definition.target_range")
    high = _finite_number(target_range[1], "target_risk_definition.target_range")
    if low < 0 or low > high:
        raise ValueError("target_risk_definition.target_range must be ordered and nonnegative")
    source = _required_text(value["source"], "target_risk_definition.source")
    if source not in TARGET_RISK_SOURCES:
        raise ValueError("target_risk_definition.source is unknown")
    direction = _required_text(value["direction"], "target_risk_definition.direction")
    if direction not in TARGET_RISK_DIRECTIONS:
        raise ValueError("target_risk_definition.direction is unknown")
    return {
        "metric": metric,
        "target_range": [low, high],
        "source": source,
        "direction": direction,
    }


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _parse_json_object(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value, parse_constant=_reject_json_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    return _json_object(parsed, name)


@dataclass(frozen=True)
class SeedRecord:
    """One skill match for one ordered initiator/responder pair."""

    scenario_id: str
    skill_id: str
    initiator_track_id: str
    responder_track_id: str
    role_track_ids: dict[str, str]
    trigger_score: float
    seed_risk_metric: str
    seed_risk_value: float
    target_risk_definition: dict[str, Any]
    source_path: str
    evidence: dict[str, Any]
    sampled_parameters: dict[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "scenario_id",
            "skill_id",
            "initiator_track_id",
            "responder_track_id",
            "seed_risk_metric",
            "source_path",
        ):
            _required_text(getattr(self, name), name)
        if self.initiator_track_id == self.responder_track_id:
            raise ValueError("initiator_track_id and responder_track_id must differ")
        role_track_ids = _role_track_ids(self.role_track_ids)
        if self.initiator_track_id not in role_track_ids.values():
            raise ValueError("role_track_ids must include initiator_track_id")
        if self.responder_track_id not in role_track_ids.values():
            raise ValueError("role_track_ids must include responder_track_id")
        object.__setattr__(self, "role_track_ids", role_track_ids)
        trigger_score = _finite_number(self.trigger_score, "trigger_score")
        if not 0.0 <= trigger_score <= 1.0:
            raise ValueError("trigger_score must be between 0 and 1")
        object.__setattr__(self, "trigger_score", trigger_score)
        object.__setattr__(
            self,
            "seed_risk_value",
            _finite_number(self.seed_risk_value, "seed_risk_value"),
        )
        object.__setattr__(
            self,
            "target_risk_definition",
            _target_risk_definition(self.target_risk_definition),
        )
        object.__setattr__(self, "evidence", _json_object(self.evidence, "evidence"))
        object.__setattr__(
            self,
            "sampled_parameters",
            _json_object(self.sampled_parameters, "sampled_parameters"),
        )

    @property
    def seed_risk_is_proxy(self) -> bool:
        return self.seed_risk_metric != self.target_risk_definition["metric"]

    @property
    def unique_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.scenario_id,
            self.skill_id,
            self.initiator_track_id,
            self.responder_track_id,
            _canonical_json(self.role_track_ids),
        )

    def to_csv_row(self) -> dict[str, str]:
        return {
            "scenario_id": self.scenario_id,
            "skill_id": self.skill_id,
            "initiator_track_id": self.initiator_track_id,
            "responder_track_id": self.responder_track_id,
            "role_track_ids_json": _canonical_json(self.role_track_ids),
            "trigger_score": repr(self.trigger_score),
            "seed_risk_metric": self.seed_risk_metric,
            "seed_risk_value": repr(self.seed_risk_value),
            "target_risk_definition_json": _canonical_json(
                self.target_risk_definition
            ),
            "source_path": self.source_path,
            "evidence_json": _canonical_json(self.evidence),
            "sampled_parameters_json": _canonical_json(self.sampled_parameters),
        }

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "SeedRecord":
        if set(row) != set(SEED_CSV_FIELDS):
            raise ValueError("seed CSV row has missing or unknown fields")
        try:
            trigger_score = float(row["trigger_score"])
            seed_risk_value = float(row["seed_risk_value"])
        except (TypeError, ValueError) as exc:
            raise ValueError("seed CSV numeric fields must be valid numbers") from exc
        return cls(
            scenario_id=row["scenario_id"],
            skill_id=row["skill_id"],
            initiator_track_id=row["initiator_track_id"],
            responder_track_id=row["responder_track_id"],
            role_track_ids=_parse_json_object(
                row["role_track_ids_json"],
                "role_track_ids_json",
            ),
            trigger_score=trigger_score,
            seed_risk_metric=row["seed_risk_metric"],
            seed_risk_value=seed_risk_value,
            target_risk_definition=_parse_json_object(
                row["target_risk_definition_json"],
                "target_risk_definition_json",
            ),
            source_path=row["source_path"],
            evidence=_parse_json_object(row["evidence_json"], "evidence_json"),
            sampled_parameters=_parse_json_object(
                row["sampled_parameters_json"],
                "sampled_parameters_json",
            ),
        )


def sort_seed_records(records: Iterable[SeedRecord]) -> list[SeedRecord]:
    ordered = sorted(records, key=lambda record: record.unique_key)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.unique_key == current.unique_key:
            raise ValueError(f"duplicate seed record key: {current.unique_key!r}")
    return ordered


def write_seed_records(path: str | Path, records: Iterable[SeedRecord]) -> Path:
    """Write a canonical UTF-8 CSV sorted by the record unique key."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sort_seed_records(records)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(record.to_csv_row() for record in ordered)
    return output


def read_seed_records(path: str | Path) -> list[SeedRecord]:
    """Read and validate a seed CSV, returning records in canonical order."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SEED_CSV_FIELDS:
            raise ValueError("seed CSV header does not match the required fields")
        records: list[SeedRecord] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                records.append(SeedRecord.from_csv_row(row))
            except ValueError as exc:
                raise ValueError(f"invalid seed CSV row {line_number}: {exc}") from exc
    return sort_seed_records(records)

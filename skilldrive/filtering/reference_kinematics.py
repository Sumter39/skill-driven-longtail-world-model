"""Deterministic AV2 Formal Train kinematic reference statistics.

The scanner deliberately has no validation-manifest input.  Its only manifest is
``manifests/splits/formal_train.csv`` below the supplied project root, and every
source path must have the canonical AV2 Train layout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

REFERENCE_VERSION = 3
FORMAL_TRAIN_MANIFEST = Path("manifests/splits/formal_train.csv")
FILTER_CONFIG = Path("configs/generation/filters_v1.yaml")
DEFAULT_DATA_ROOT = Path("data/av2/motion-forecasting")
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/generation/counterfactual_v1/kinematic_reference"
)
FORMAL_TRAIN_SCENARIO_COUNT = 20_000
DEFAULT_SHARD_SIZE = 250
DEFAULT_SAMPLE_PERIOD_S = 0.1
REFERENCE_TIMESTEPS = tuple(range(48, 110))
_OWNER_SENTINEL_NAME = ".kinematic-reference-owner.json"
_OWNER_SENTINEL = {
    "kind": "skilldrive_kinematic_reference_output",
    "owner_format_version": 1,
}
_MANIFEST_COLUMNS = [
    "scenario_id",
    "split",
    "source_path",
    "city_name",
    "selected_reason",
]

OBJECT_TYPES = ("vehicle", "bus", "motorcyclist", "cyclist", "pedestrian")
METRIC_NAMES = (
    "speed_mps",
    "positive_acceleration_mps2",
    "deceleration_mps2",
    "jerk_mps3",
    "yaw_rate_radps",
    "curvature_inv_m",
)
PARQUET_COLUMNS = (
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
    "start_timestamp",
)
QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999)

# A fixed logarithmic histogram keeps the complete 20k-scene output small while
# making worker count and resume boundaries irrelevant to percentile results.
HISTOGRAM_MIN_POSITIVE = 1.0e-6
HISTOGRAM_MAX_VALUE = 1.0e4
HISTOGRAM_BIN_COUNT = 2048
_LOG_BIN_WIDTH = math.log(
    HISTOGRAM_MAX_VALUE / HISTOGRAM_MIN_POSITIVE
) / HISTOGRAM_BIN_COUNT


@dataclass(frozen=True)
class TrackKinematicSamples:
    """Candidate-equivalent samples for one AV2 timestep 48..109 window."""

    speed_mps: np.ndarray
    positive_acceleration_mps2: np.ndarray
    deceleration_mps2: np.ndarray
    jerk_mps3: np.ndarray
    yaw_rate_radps: np.ndarray
    curvature_inv_m: np.ndarray
    valid_reference_window: bool
    invalid_position_rows: int
    invalid_velocity_rows: int
    invalid_anchor_heading_rows: int
    low_speed_turn_transitions_suppressed: int

    def metric_values(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in METRIC_NAMES}


@dataclass(frozen=True)
class ReferenceProgress:
    phase: str
    completed_shards: int
    total_shards: int
    completed_scenarios: int
    total_scenarios: int
    new_shards: int


@dataclass(frozen=True)
class ReferenceBuildResult:
    complete: bool
    completed_shards: int
    total_shards: int
    completed_scenarios: int
    total_scenarios: int
    new_shards: int
    output_root: Path
    summary_path: Path | None


@dataclass(frozen=True)
class _FormalTrainRow:
    scenario_id: str
    split: str
    source_path: str


@dataclass
class _MetricAccumulator:
    count: int = 0
    value_sum: float = 0.0
    value_sum_squares: float = 0.0
    minimum: float | None = None
    maximum: float | None = None
    zero_count: int = 0
    underflow_count: int = 0
    overflow_count: int = 0
    bins: dict[int, int] = field(default_factory=dict)

    def add(self, values: Any) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(array):
            return
        if not np.isfinite(array).all():
            raise ValueError("kinematic metric samples must all be finite")
        if np.any(array < 0.0):
            raise ValueError("kinematic reference metrics must be non-negative")

        self.count += int(len(array))
        self.value_sum += float(np.sum(array, dtype=np.float64))
        self.value_sum_squares += float(
            np.sum(array * array, dtype=np.float64)
        )
        local_minimum = float(np.min(array))
        local_maximum = float(np.max(array))
        self.minimum = (
            local_minimum
            if self.minimum is None
            else min(self.minimum, local_minimum)
        )
        self.maximum = (
            local_maximum
            if self.maximum is None
            else max(self.maximum, local_maximum)
        )

        self.zero_count += int(np.count_nonzero(array == 0.0))
        positive = array[array > 0.0]
        if not len(positive):
            return
        underflow = positive < HISTOGRAM_MIN_POSITIVE
        overflow = positive > HISTOGRAM_MAX_VALUE
        self.underflow_count += int(np.count_nonzero(underflow))
        self.overflow_count += int(np.count_nonzero(overflow))
        regular = positive[~underflow & ~overflow]
        if not len(regular):
            return
        indices = np.floor(
            np.log(regular / HISTOGRAM_MIN_POSITIVE) / _LOG_BIN_WIDTH
        ).astype(np.int64)
        indices = np.clip(indices, 0, HISTOGRAM_BIN_COUNT - 1)
        unique, counts = np.unique(indices, return_counts=True)
        for index, count in zip(unique.tolist(), counts.tolist(), strict=True):
            self.bins[int(index)] = self.bins.get(int(index), 0) + int(count)

    def merge(self, other: _MetricAccumulator) -> None:
        self.count += other.count
        self.value_sum += other.value_sum
        self.value_sum_squares += other.value_sum_squares
        if other.minimum is not None:
            self.minimum = (
                other.minimum
                if self.minimum is None
                else min(self.minimum, other.minimum)
            )
        if other.maximum is not None:
            self.maximum = (
                other.maximum
                if self.maximum is None
                else max(self.maximum, other.maximum)
            )
        self.zero_count += other.zero_count
        self.underflow_count += other.underflow_count
        self.overflow_count += other.overflow_count
        for index, count in sorted(other.bins.items()):
            self.bins[index] = self.bins.get(index, 0) + count

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.value_sum,
            "sum_squares": self.value_sum_squares,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "zero_count": self.zero_count,
            "underflow_count": self.underflow_count,
            "overflow_count": self.overflow_count,
            "bins": [[index, count] for index, count in sorted(self.bins.items())],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> _MetricAccumulator:
        value = cls(
            count=int(payload["count"]),
            value_sum=float(payload["sum"]),
            value_sum_squares=float(payload["sum_squares"]),
            minimum=(
                None if payload["minimum"] is None else float(payload["minimum"])
            ),
            maximum=(
                None if payload["maximum"] is None else float(payload["maximum"])
            ),
            zero_count=int(payload["zero_count"]),
            underflow_count=int(payload["underflow_count"]),
            overflow_count=int(payload["overflow_count"]),
        )
        pairs = payload["bins"]
        if not isinstance(pairs, list):
            raise ValueError("histogram bins must be a list")
        previous = -1
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each histogram bin must be [index, count]")
            index, count = int(pair[0]), int(pair[1])
            if not 0 <= index < HISTOGRAM_BIN_COUNT or index <= previous:
                raise ValueError("histogram bin indices must be sorted and in range")
            if count <= 0:
                raise ValueError("histogram bin counts must be positive")
            value.bins[index] = count
            previous = index
        represented = (
            value.zero_count
            + value.underflow_count
            + value.overflow_count
            + sum(value.bins.values())
        )
        if represented != value.count:
            raise ValueError(
                f"histogram represents {represented} values but count is {value.count}"
            )
        return value

    def _quantile_upper_bound(self, probability: float) -> float | None:
        if not self.count:
            return None
        rank = max(1, math.ceil(probability * self.count))
        cumulative = self.zero_count
        if rank <= cumulative:
            return 0.0
        cumulative += self.underflow_count
        if rank <= cumulative:
            return HISTOGRAM_MIN_POSITIVE
        for index, count in sorted(self.bins.items()):
            cumulative += count
            if rank <= cumulative:
                upper = HISTOGRAM_MIN_POSITIVE * math.exp(
                    (index + 1) * _LOG_BIN_WIDTH
                )
                return min(upper, HISTOGRAM_MAX_VALUE)
        # Values above the fixed range are deliberately not assigned a fake
        # percentile.  The exact maximum and overflow count remain available.
        return None

    def summary(self) -> dict[str, Any]:
        mean = None if not self.count else self.value_sum / self.count
        standard_deviation = None
        if mean is not None:
            variance = max(
                0.0,
                self.value_sum_squares / self.count - mean * mean,
            )
            standard_deviation = math.sqrt(variance)
        return {
            "sample_count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": mean,
            "population_standard_deviation": standard_deviation,
            "zero_count": self.zero_count,
            "underflow_count": self.underflow_count,
            "overflow_count": self.overflow_count,
            "quantile_upper_bounds": {
                _quantile_label(probability): self._quantile_upper_bound(probability)
                for probability in QUANTILES
            },
        }


@dataclass
class _CategoryAccumulator:
    track_count: int = 0
    row_count: int = 0
    reference_window_count: int = 0
    metrics: dict[str, _MetricAccumulator] = field(
        default_factory=lambda: {
            name: _MetricAccumulator() for name in METRIC_NAMES
        }
    )
    window_max_metrics: dict[str, _MetricAccumulator] = field(
        default_factory=lambda: {
            name: _MetricAccumulator() for name in METRIC_NAMES
        }
    )

    def merge(self, other: _CategoryAccumulator) -> None:
        self.track_count += other.track_count
        self.row_count += other.row_count
        self.reference_window_count += other.reference_window_count
        for name in METRIC_NAMES:
            self.metrics[name].merge(other.metrics[name])
            self.window_max_metrics[name].merge(other.window_max_metrics[name])

    def to_payload(self) -> dict[str, Any]:
        return {
            "track_count": self.track_count,
            "row_count": self.row_count,
            "reference_window_count": self.reference_window_count,
            "metrics": {
                name: self.metrics[name].to_payload() for name in METRIC_NAMES
            },
            "window_max_metrics": {
                name: self.window_max_metrics[name].to_payload()
                for name in METRIC_NAMES
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> _CategoryAccumulator:
        metrics = payload["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
            raise ValueError("category payload has an invalid metric set")
        window_max_metrics = payload["window_max_metrics"]
        if not isinstance(window_max_metrics, Mapping) or set(
            window_max_metrics
        ) != set(METRIC_NAMES):
            raise ValueError("category payload has an invalid window-max metric set")
        return cls(
            track_count=int(payload["track_count"]),
            row_count=int(payload["row_count"]),
            reference_window_count=int(payload["reference_window_count"]),
            metrics={
                name: _MetricAccumulator.from_payload(metrics[name])
                for name in METRIC_NAMES
            },
            window_max_metrics={
                name: _MetricAccumulator.from_payload(window_max_metrics[name])
                for name in METRIC_NAMES
            },
        )


@dataclass
class _ReferenceAccumulator:
    scenario_count: int = 0
    quality_counts: Counter[str] = field(default_factory=Counter)
    categories: dict[str, _CategoryAccumulator] = field(
        default_factory=lambda: {
            object_type: _CategoryAccumulator() for object_type in OBJECT_TYPES
        }
    )

    def merge(self, other: _ReferenceAccumulator) -> None:
        self.scenario_count += other.scenario_count
        self.quality_counts.update(other.quality_counts)
        for object_type in OBJECT_TYPES:
            self.categories[object_type].merge(other.categories[object_type])

    def to_payload(self) -> dict[str, Any]:
        return {
            "scenario_count": self.scenario_count,
            "quality_counts": dict(sorted(self.quality_counts.items())),
            "categories": {
                object_type: self.categories[object_type].to_payload()
                for object_type in OBJECT_TYPES
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> _ReferenceAccumulator:
        categories = payload["categories"]
        if not isinstance(categories, Mapping) or set(categories) != set(OBJECT_TYPES):
            raise ValueError("reference payload has an invalid object-type set")
        quality = payload["quality_counts"]
        if not isinstance(quality, Mapping):
            raise ValueError("quality_counts must be a mapping")
        return cls(
            scenario_count=int(payload["scenario_count"]),
            quality_counts=Counter({str(key): int(value) for key, value in quality.items()}),
            categories={
                object_type: _CategoryAccumulator.from_payload(categories[object_type])
                for object_type in OBJECT_TYPES
            },
        )

    def summary(self) -> dict[str, Any]:
        return {
            "scenario_count": self.scenario_count,
            "quality_counts": dict(sorted(self.quality_counts.items())),
            "categories": {
                object_type: {
                    "track_count": category.track_count,
                    "row_count": category.row_count,
                    "reference_window_count": category.reference_window_count,
                    "point_distributions": {
                        name: category.metrics[name].summary()
                        for name in METRIC_NAMES
                    },
                    "window_max_distributions": {
                        name: category.window_max_metrics[name].summary()
                        for name in METRIC_NAMES
                    },
                }
                for object_type, category in self.categories.items()
            },
        }


def derive_track_kinematic_samples(
    timestep: Any,
    position_xy: Any,
    velocity_xy: Any,
    heading_rad: Any,
    *,
    low_speed_threshold_mps: float,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    scenario_start_timestamp_ns: int = 0,
) -> TrackKinematicSamples:
    """Match ``common.derive_future_kinematics`` on AV2 timesteps 48..109."""

    if (
        isinstance(sample_period_s, bool)
        or not isinstance(sample_period_s, (int, float))
        or not math.isfinite(float(sample_period_s))
        or float(sample_period_s) <= 0.0
    ):
        raise ValueError("sample_period_s must be a positive finite number")
    if isinstance(scenario_start_timestamp_ns, bool) or not isinstance(
        scenario_start_timestamp_ns, int
    ):
        raise ValueError("scenario_start_timestamp_ns must be an integer")
    if (
        isinstance(low_speed_threshold_mps, bool)
        or not isinstance(low_speed_threshold_mps, (int, float))
        or not math.isfinite(float(low_speed_threshold_mps))
        or float(low_speed_threshold_mps) <= 0.0
    ):
        raise ValueError("low_speed_threshold_mps must be a positive finite number")

    steps = np.asarray(timestep, dtype=np.int64).reshape(-1)
    position = np.asarray(position_xy, dtype=np.float64)
    velocity = np.asarray(velocity_xy, dtype=np.float64)
    heading = np.asarray(heading_rad, dtype=np.float64).reshape(-1)
    expected_steps = np.asarray(REFERENCE_TIMESTEPS, dtype=np.int64)
    if (
        position.shape != (len(steps), 2)
        or velocity.shape != (len(steps), 2)
        or len(heading) != len(steps)
    ):
        raise ValueError(
            "timestep, position_xy, velocity_xy, and heading_rad lengths must match"
        )
    if not np.array_equal(steps, expected_steps):
        raise ValueError("reference track window must contain timesteps 48 through 109")

    required_position = np.vstack(
        (
            position[1],
            position[2:].astype(np.float32).astype(np.float64),
        )
    )
    required_velocity = velocity[:2]
    invalid_position_rows = int(
        np.count_nonzero(~np.isfinite(required_position).all(axis=1))
    )
    invalid_velocity_rows = int(
        np.count_nonzero(~np.isfinite(required_velocity).all(axis=1))
    )
    invalid_anchor_heading_rows = int(not math.isfinite(float(heading[1])))
    if invalid_position_rows or invalid_velocity_rows:
        empty = np.empty(0, dtype=np.float64)
        return TrackKinematicSamples(
            speed_mps=empty,
            positive_acceleration_mps2=empty,
            deceleration_mps2=empty,
            jerk_mps3=empty,
            yaw_rate_radps=empty,
            curvature_inv_m=empty,
            valid_reference_window=False,
            invalid_position_rows=invalid_position_rows,
            invalid_velocity_rows=invalid_velocity_rows,
            invalid_anchor_heading_rows=invalid_anchor_heading_rows,
            low_speed_turn_transitions_suppressed=0,
        )

    period_ns = int(round(float(sample_period_s) * 1_000_000_000.0))
    if period_ns <= 0 or not math.isclose(
        period_ns / 1_000_000_000.0,
        float(sample_period_s),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("sample_period_s must be representable as integer nanoseconds")
    timestamps_ns = (
        int(scenario_start_timestamp_ns) + steps.astype(np.int64) * period_ns
    )
    timestamps_s = (
        timestamps_ns[1:].astype(np.float64) - float(timestamps_ns[1])
    ) / 1_000_000_000.0
    elapsed = np.diff(timestamps_s)
    history_elapsed = float(timestamps_ns[1] - timestamps_ns[0]) / 1_000_000_000.0
    future_velocity = np.diff(required_position, axis=0) / elapsed[:, None]
    previous_velocity = required_velocity[1]
    velocity_sequence = np.vstack((previous_velocity, future_velocity))
    acceleration = np.diff(velocity_sequence, axis=0) / elapsed[:, None]
    previous_acceleration = (
        required_velocity[1] - required_velocity[0]
    ) / history_elapsed
    acceleration_sequence = np.vstack((previous_acceleration, acceleration))
    jerk = np.linalg.norm(
        np.diff(acceleration_sequence, axis=0) / elapsed[:, None], axis=1
    )

    speed = np.linalg.norm(future_velocity, axis=1)
    previous_speed = float(np.linalg.norm(previous_velocity))
    speed_sequence = np.concatenate(([previous_speed], speed))
    tangential_acceleration = np.diff(speed_sequence) / elapsed
    positive_acceleration = np.maximum(tangential_acceleration, 0.0)
    deceleration = np.maximum(-tangential_acceleration, 0.0)

    previous_heading = float(heading[1])
    if not math.isfinite(previous_heading):
        previous_heading = (
            float(np.arctan2(previous_velocity[1], previous_velocity[0]))
            if previous_speed > 0.0
            else float("nan")
        )
    yaw_rate = np.zeros(len(future_velocity), dtype=np.float64)
    curvature = np.zeros(len(future_velocity), dtype=np.float64)
    suppressed = np.zeros(len(future_velocity), dtype=bool)
    threshold = float(low_speed_threshold_mps)
    for index, item in enumerate(future_velocity):
        current_speed = float(speed[index])
        if current_speed >= threshold:
            current_heading = float(np.arctan2(item[1], item[0]))
            if math.isfinite(previous_heading) and previous_speed >= threshold:
                delta = float(
                    (current_heading - previous_heading + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
                yaw_rate[index] = abs(delta) / float(elapsed[index])
                curvature[index] = yaw_rate[index] / (
                    (previous_speed + current_speed) / 2.0
                )
            previous_heading = current_heading
        else:
            suppressed[index] = True
        previous_speed = current_speed

    return TrackKinematicSamples(
        speed_mps=speed,
        positive_acceleration_mps2=positive_acceleration,
        deceleration_mps2=deceleration,
        jerk_mps3=jerk,
        yaw_rate_radps=yaw_rate,
        curvature_inv_m=curvature,
        valid_reference_window=True,
        invalid_position_rows=0,
        invalid_velocity_rows=0,
        invalid_anchor_heading_rows=invalid_anchor_heading_rows,
        low_speed_turn_transitions_suppressed=int(np.count_nonzero(suppressed)),
    )


def load_formal_train_rows(
    project_root: str | Path,
    *,
    expected_scenario_count: int = FORMAL_TRAIN_SCENARIO_COUNT,
) -> tuple[Path, list[_FormalTrainRow]]:
    """Read and validate the sole allowed manifest for this reference scan."""

    if (
        isinstance(expected_scenario_count, bool)
        or not isinstance(expected_scenario_count, int)
        or expected_scenario_count <= 0
    ):
        raise ValueError("expected_scenario_count must be a positive integer")
    root = Path(project_root).resolve()
    manifest_path = root / FORMAL_TRAIN_MANIFEST
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _MANIFEST_COLUMNS:
            raise ValueError(
                f"manifest columns must be {_MANIFEST_COLUMNS}, got {reader.fieldnames}"
            )
        rows = [
            _FormalTrainRow(
                scenario_id=row["scenario_id"],
                split=row["split"],
                source_path=row["source_path"],
            )
            for row in reader
        ]
    if len(rows) != expected_scenario_count:
        raise ValueError(
            f"Formal Train manifest must contain {expected_scenario_count} scenarios, "
            f"got {len(rows)}"
        )

    seen: set[str] = set()
    for row in rows:
        if not row.scenario_id or row.scenario_id in seen:
            raise ValueError(f"duplicate or empty scenario_id: {row.scenario_id!r}")
        seen.add(row.scenario_id)
        if row.split != "train":
            raise ValueError(
                f"Formal Train manifest may only contain split=train, got "
                f"{row.split!r} for {row.scenario_id}"
            )
        if "\\" in row.source_path:
            raise ValueError("Formal Train source_path must use POSIX separators")
        source = PurePosixPath(row.source_path)
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
    return manifest_path, sorted(rows, key=lambda row: row.scenario_id)


def _load_minimum_heading_speed_policy(
    project_root: Path,
) -> tuple[Path, dict[str, float]]:
    config_path = project_root / FILTER_CONFIG
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("filter configuration must be a mapping")
    kinematics = payload.get("kinematics")
    actor_types = (
        kinematics.get("actor_types") if isinstance(kinematics, Mapping) else None
    )
    if not isinstance(actor_types, Mapping):
        raise ValueError("filter configuration must define kinematics.actor_types")

    policy: dict[str, float] = {}
    for object_type in OBJECT_TYPES:
        actor = actor_types.get(object_type)
        value = (
            actor.get("minimum_heading_speed_mps")
            if isinstance(actor, Mapping)
            else None
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                "filter configuration must define a positive finite "
                f"minimum_heading_speed_mps for {object_type}"
            )
        policy[object_type] = float(value)
    return config_path, policy


def scan_scenario_parquet(
    scenario_id: str,
    scenario_path: str | Path,
    *,
    minimum_heading_speed_mps_by_actor_type: Mapping[str, float],
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
) -> dict[str, Any]:
    """Read the frozen Parquet columns and return candidate-equivalent statistics."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to build the AV2 reference") from error

    source_path = Path(scenario_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"missing Formal Train scenario: {source_path}")
    table = pq.ParquetFile(source_path, memory_map=True).read(
        columns=list(PARQUET_COLUMNS),
        use_threads=False,
    )
    if tuple(table.column_names) != PARQUET_COLUMNS:
        raise ValueError(
            f"unexpected Parquet columns for {scenario_id}: {table.column_names}"
        )

    track_ids = np.asarray(table["track_id"].to_pylist(), dtype=object)
    object_types = np.asarray(
        [str(value).lower() for value in table["object_type"].to_pylist()],
        dtype=object,
    )
    timestep = np.asarray(table["timestep"].to_numpy(), dtype=np.int64)
    position = np.column_stack(
        (
            np.asarray(table["position_x"].to_numpy(), dtype=np.float64),
            np.asarray(table["position_y"].to_numpy(), dtype=np.float64),
        )
    )
    heading = np.asarray(table["heading"].to_numpy(), dtype=np.float64)
    velocity = np.column_stack(
        (
            np.asarray(table["velocity_x"].to_numpy(), dtype=np.float64),
            np.asarray(table["velocity_y"].to_numpy(), dtype=np.float64),
        )
    )
    start_timestamps = np.asarray(
        table["start_timestamp"].to_numpy(), dtype=np.float64
    )
    if (
        not len(start_timestamps)
        or not np.isfinite(start_timestamps).all()
        or not np.all(start_timestamps == start_timestamps[0])
    ):
        raise ValueError(f"invalid start_timestamp column for {scenario_id}")
    scenario_start_timestamp_ns = int(start_timestamps[0])

    aggregate = _ReferenceAccumulator(scenario_count=1)
    aggregate.quality_counts["parquet_rows"] = len(table)
    eligible = np.isin(object_types, OBJECT_TYPES)
    aggregate.quality_counts["ignored_object_type_rows"] = int(
        np.count_nonzero(~eligible)
    )
    eligible_indices = np.flatnonzero(eligible)
    aggregate.quality_counts["eligible_object_type_rows"] = int(
        len(eligible_indices)
    )
    if not len(eligible_indices):
        return {
            "scenario_id": scenario_id,
            "statistics": aggregate.to_payload(),
        }

    order = np.lexsort(
        (timestep[eligible_indices], track_ids[eligible_indices])
    )
    sorted_indices = eligible_indices[order]
    sorted_track_ids = track_ids[sorted_indices]
    boundaries = np.flatnonzero(sorted_track_ids[1:] != sorted_track_ids[:-1]) + 1
    groups = np.split(sorted_indices, boundaries)
    pending_metrics = {
        object_type: {name: [] for name in METRIC_NAMES}
        for object_type in OBJECT_TYPES
    }
    pending_window_max_metrics = {
        object_type: {name: [] for name in METRIC_NAMES}
        for object_type in OBJECT_TYPES
    }
    expected_steps = np.asarray(REFERENCE_TIMESTEPS, dtype=np.int64)

    for indices in groups:
        types = set(object_types[indices].tolist())
        if len(types) != 1:
            raise ValueError(
                f"track {track_ids[indices[0]]!r} changes object_type in {scenario_id}"
            )
        object_type = types.pop()
        category = aggregate.categories[object_type]
        category.track_count += 1
        category.row_count += int(len(indices))
        track_steps = timestep[indices]
        reference_indices = indices[
            (track_steps >= REFERENCE_TIMESTEPS[0])
            & (track_steps <= REFERENCE_TIMESTEPS[-1])
        ]
        if not np.array_equal(timestep[reference_indices], expected_steps):
            aggregate.quality_counts["incomplete_reference_window_tracks"] += 1
            continue
        samples = derive_track_kinematic_samples(
            timestep[reference_indices],
            position[reference_indices],
            velocity[reference_indices],
            heading[reference_indices],
            sample_period_s=sample_period_s,
            scenario_start_timestamp_ns=scenario_start_timestamp_ns,
            low_speed_threshold_mps=minimum_heading_speed_mps_by_actor_type[
                object_type
            ],
        )
        aggregate.quality_counts["invalid_reference_position_rows"] += (
            samples.invalid_position_rows
        )
        aggregate.quality_counts["invalid_reference_velocity_rows"] += (
            samples.invalid_velocity_rows
        )
        aggregate.quality_counts["invalid_anchor_heading_rows"] += (
            samples.invalid_anchor_heading_rows
        )
        if not samples.valid_reference_window:
            aggregate.quality_counts["invalid_reference_window_tracks"] += 1
            continue
        category.reference_window_count += 1
        for name, values in samples.metric_values().items():
            pending_metrics[object_type][name].append(values)
            pending_window_max_metrics[object_type][name].append(float(np.max(values)))
        aggregate.quality_counts["low_speed_turn_transitions_suppressed"] += (
            samples.low_speed_turn_transitions_suppressed
        )

    for object_type, metrics in pending_metrics.items():
        category = aggregate.categories[object_type]
        for name, arrays in metrics.items():
            if arrays:
                category.metrics[name].add(np.concatenate(arrays))
        for name, maxima in pending_window_max_metrics[object_type].items():
            if maxima:
                category.window_max_metrics[name].add(maxima)

    aggregate.quality_counts["eligible_tracks"] = len(groups)
    aggregate.quality_counts["reference_window_tracks"] = sum(
        category.reference_window_count for category in aggregate.categories.values()
    )
    return {"scenario_id": scenario_id, "statistics": aggregate.to_payload()}


def build_kinematic_reference(
    project_root: str | Path,
    *,
    data_root: str | Path | None = None,
    output_root: str | Path | None = None,
    workers: int = 1,
    expected_scenario_count: int = FORMAL_TRAIN_SCENARIO_COUNT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    max_new_shards: int | None = None,
    restart: bool = False,
    progress: Callable[[ReferenceProgress], None] | None = None,
) -> ReferenceBuildResult:
    """Build or resume the deterministic sharded Formal Train reference."""

    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    if (
        isinstance(expected_scenario_count, bool)
        or not isinstance(expected_scenario_count, int)
        or expected_scenario_count <= 0
    ):
        raise ValueError("expected_scenario_count must be a positive integer")
    if max_new_shards is not None and (
        isinstance(max_new_shards, bool)
        or not isinstance(max_new_shards, int)
        or max_new_shards <= 0
    ):
        raise ValueError("max_new_shards must be None or a positive integer")
    if (
        isinstance(sample_period_s, bool)
        or not isinstance(sample_period_s, (int, float))
        or not math.isfinite(float(sample_period_s))
        or float(sample_period_s) <= 0.0
    ):
        raise ValueError("sample_period_s must be a positive finite number")

    root = Path(project_root).resolve()
    if progress is not None:
        progress(
            ReferenceProgress(
                phase="preflight",
                completed_shards=0,
                total_shards=math.ceil(expected_scenario_count / shard_size),
                completed_scenarios=0,
                total_scenarios=expected_scenario_count,
                new_shards=0,
            )
        )
    resolved_data_root = _resolve_path(
        root,
        DEFAULT_DATA_ROOT if data_root is None else Path(data_root),
    )
    resolved_output_root = _resolve_managed_output_root(
        root,
        DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root),
    )
    manifest_path, rows = load_formal_train_rows(
        root,
        expected_scenario_count=expected_scenario_count,
    )
    filter_config_path, minimum_heading_speed_policy = (
        _load_minimum_heading_speed_policy(root)
    )
    tasks = [
        (
            row.scenario_id,
            _train_source_path(resolved_data_root, row),
        )
        for row in rows
    ]
    contract = _build_contract(
        root=root,
        manifest_path=manifest_path,
        filter_config_path=filter_config_path,
        rows=rows,
        data_root=resolved_data_root,
        shard_size=shard_size,
        sample_period_s=sample_period_s,
        minimum_heading_speed_policy=minimum_heading_speed_policy,
    )
    contract_bytes = _json_bytes(contract, pretty=True)
    contract_sha256 = _sha256(contract_bytes)
    _prepare_output_root(
        resolved_output_root,
        contract_bytes=contract_bytes,
        restart=restart,
    )

    shards = [tasks[index : index + shard_size] for index in range(0, len(tasks), shard_size)]
    validity = [
        _valid_shard(
            resolved_output_root,
            shard_index=index,
            tasks=shard_tasks,
            contract_sha256=contract_sha256,
        )
        for index, shard_tasks in enumerate(shards)
    ]
    completed_scenarios = sum(
        len(shard_tasks)
        for shard_tasks, valid in zip(shards, validity, strict=True)
        if valid
    )
    new_shards = 0
    if progress is not None:
        progress(
            ReferenceProgress(
                phase="scan",
                completed_shards=sum(validity),
                total_shards=len(shards),
                completed_scenarios=completed_scenarios,
                total_scenarios=len(tasks),
                new_shards=0,
            )
        )

    executor: ProcessPoolExecutor | None = None
    try:
        if workers > 1 and any(not valid for valid in validity):
            executor = ProcessPoolExecutor(max_workers=workers)
        for shard_index, shard_tasks in enumerate(shards):
            if validity[shard_index]:
                continue
            if max_new_shards is not None and new_shards >= max_new_shards:
                break
            worker_arguments = [
                (
                    scenario_id,
                    str(path),
                    sample_period_s,
                    minimum_heading_speed_policy,
                )
                for scenario_id, path in shard_tasks
            ]
            if executor is None:
                results = [_scan_worker(argument) for argument in worker_arguments]
            else:
                chunksize = max(
                    1,
                    min(16, len(worker_arguments) // max(1, workers * 4)),
                )
                results = list(
                    executor.map(_scan_worker, worker_arguments, chunksize=chunksize)
                )
            _write_shard(
                resolved_output_root,
                shard_index=shard_index,
                tasks=shard_tasks,
                results=results,
                contract_sha256=contract_sha256,
            )
            validity[shard_index] = True
            new_shards += 1
            completed_scenarios += len(shard_tasks)
            if progress is not None:
                progress(
                    ReferenceProgress(
                        phase="scan",
                        completed_shards=sum(validity),
                        total_shards=len(shards),
                        completed_scenarios=completed_scenarios,
                        total_scenarios=len(tasks),
                        new_shards=new_shards,
                    )
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if not all(validity):
        _remove_final_outputs(resolved_output_root)
        return ReferenceBuildResult(
            complete=False,
            completed_shards=sum(validity),
            total_shards=len(shards),
            completed_scenarios=completed_scenarios,
            total_scenarios=len(tasks),
            new_shards=new_shards,
            output_root=resolved_output_root,
            summary_path=None,
        )

    aggregate = _ReferenceAccumulator()
    for shard_index in range(len(shards)):
        shard_payload = _read_json(_shard_path(resolved_output_root, shard_index))
        aggregate.merge(
            _ReferenceAccumulator.from_payload(shard_payload["statistics"])
        )
    if aggregate.scenario_count != len(tasks):
        raise ValueError(
            f"final aggregate contains {aggregate.scenario_count} scenarios, "
            f"expected {len(tasks)}"
        )
    _write_final_outputs(
        resolved_output_root,
        contract=contract,
        contract_sha256=contract_sha256,
        aggregate=aggregate,
    )
    return ReferenceBuildResult(
        complete=True,
        completed_shards=len(shards),
        total_shards=len(shards),
        completed_scenarios=len(tasks),
        total_scenarios=len(tasks),
        new_shards=new_shards,
        output_root=resolved_output_root,
        summary_path=resolved_output_root / "summary.json",
    )


def _scan_worker(
    argument: tuple[str, str, float, Mapping[str, float]],
) -> dict[str, Any]:
    scenario_id, path, sample_period_s, minimum_heading_speed_policy = argument
    return scan_scenario_parquet(
        scenario_id,
        path,
        minimum_heading_speed_mps_by_actor_type=minimum_heading_speed_policy,
        sample_period_s=sample_period_s,
    )


def _resolve_path(project_root: Path, value: Path) -> Path:
    return (project_root / value).resolve() if not value.is_absolute() else value.resolve()


def _resolve_managed_output_root(project_root: Path, value: Path) -> Path:
    resolved = _resolve_path(project_root, value)
    outputs_root = (project_root / "outputs").resolve()
    if (
        outputs_root == project_root
        or not outputs_root.is_relative_to(project_root)
        or resolved in {project_root, outputs_root}
        or not resolved.is_relative_to(outputs_root)
    ):
        raise ValueError(
            "kinematic reference output_root must be a dedicated directory below "
            f"{outputs_root}"
        )
    return resolved


def _train_source_path(data_root: Path, row: _FormalTrainRow) -> Path:
    source = PurePosixPath(row.source_path)
    return data_root.joinpath(*source.parts)


def _project_relative(project_root: Path, value: Path) -> str:
    try:
        return value.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(value.resolve())


def _build_contract(
    *,
    root: Path,
    manifest_path: Path,
    filter_config_path: Path,
    rows: Sequence[_FormalTrainRow],
    data_root: Path,
    shard_size: int,
    sample_period_s: float,
    minimum_heading_speed_policy: Mapping[str, float],
) -> dict[str, Any]:
    if not math.isfinite(float(sample_period_s)) or float(sample_period_s) <= 0.0:
        raise ValueError("sample_period_s must be a positive finite number")
    return {
        "kind": "formal_train_kinematic_reference_contract",
        "version": REFERENCE_VERSION,
        "manifest_path": FORMAL_TRAIN_MANIFEST.as_posix(),
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "filter_config_path": FILTER_CONFIG.as_posix(),
        "filter_config_sha256": _sha256(filter_config_path.read_bytes()),
        "scenario_ids_sha256": _scenario_ids_sha256(
            [row.scenario_id for row in rows]
        ),
        "scenario_count": len(rows),
        "data_root": _project_relative(root, data_root),
        "source_split": "train",
        "temporal_scope": "candidate_equivalent_timesteps_48_through_109",
        "parquet_columns": list(PARQUET_COLUMNS),
        "object_types": list(OBJECT_TYPES),
        "metric_names": list(METRIC_NAMES),
        "distribution_scopes": {
            "point_distributions": (
                "all 60 candidate-equivalent values from each complete window"
            ),
            "window_max_distributions": (
                "one maximum over the 60 candidate-equivalent values per complete window"
            ),
        },
        "sample_period_s": float(sample_period_s),
        "candidate_future_dtype": "float32",
        "minimum_heading_speed_mps_by_actor_type": {
            object_type: float(minimum_heading_speed_policy[object_type])
            for object_type in OBJECT_TYPES
        },
        "shard_size": shard_size,
        "histogram": _histogram_contract(),
        "quantiles": list(QUANTILES),
        "implementation_sha256": _sha256(Path(__file__).read_bytes()),
        "validation_manifests_opened": False,
    }


def _histogram_contract() -> dict[str, Any]:
    return {
        "kind": "fixed_logarithmic_nonnegative",
        "minimum_positive": HISTOGRAM_MIN_POSITIVE,
        "maximum_value": HISTOGRAM_MAX_VALUE,
        "regular_bin_count": HISTOGRAM_BIN_COUNT,
        "relative_bin_width_upper_bound": math.exp(_LOG_BIN_WIDTH) - 1.0,
        "quantile_method": "conservative_histogram_bin_upper_bound",
        "overflow_quantile_value": None,
    }


def _metric_definitions(sample_period_s: float) -> dict[str, str]:
    period = f"{sample_period_s:g} s"
    return {
        "speed_mps": (
            "Euclidean norm of position-difference velocity for timesteps "
            f"50..109 using {period}."
        ),
        "positive_acceleration_mps2": (
            "Non-negative part of candidate-equivalent speed change, including "
            f"the stored-velocity timestep 49 seam, divided by {period}."
        ),
        "deceleration_mps2": (
            "Non-negative magnitude of candidate-equivalent speed decrease, "
            f"including zero samples, divided by {period}."
        ),
        "jerk_mps3": (
            "Magnitude of candidate-equivalent vector-acceleration change; the "
            "first sample uses stored velocities at timesteps 48 and 49."
        ),
        "yaw_rate_radps": (
            "Absolute wrapped motion-direction change per second; suppressed "
            "low-speed steps contribute zero exactly as the production filter."
        ),
        "curvature_inv_m": (
            "Absolute motion-direction yaw rate divided by endpoint mean speed; "
            "suppressed steps contribute zero exactly as the production filter."
        ),
    }


def _prepare_output_root(
    output_root: Path,
    *,
    contract_bytes: bytes,
    restart: bool,
) -> None:
    contract_path = output_root / "contract.json"
    sentinel_path = output_root / _OWNER_SENTINEL_NAME
    if restart and output_root.exists():
        entries = list(output_root.iterdir())
        if entries:
            if not sentinel_path.is_file() or _read_json(sentinel_path) != _OWNER_SENTINEL:
                raise ValueError(
                    "refusing to restart an output directory not owned by the "
                    "kinematic reference builder"
                )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if sentinel_path.exists():
        if not sentinel_path.is_file() or _read_json(sentinel_path) != _OWNER_SENTINEL:
            raise ValueError("kinematic reference output has an invalid owner sentinel")
    else:
        unexpected_before_ownership = list(output_root.iterdir())
        if unexpected_before_ownership:
            raise ValueError(
                "kinematic reference output is not empty and has no owner sentinel"
            )
        _atomic_write(sentinel_path, _json_bytes(_OWNER_SENTINEL, pretty=True))
    (output_root / "shards").mkdir(parents=True, exist_ok=True)
    if contract_path.exists():
        if contract_path.read_bytes() != contract_bytes:
            raise ValueError(
                "kinematic reference inputs or implementation changed; rerun with restart=True"
            )
        return
    expected_names = {_OWNER_SENTINEL_NAME, "shards"}
    unexpected = [
        path for path in output_root.iterdir() if path.name not in expected_names
    ]
    if unexpected or any((output_root / "shards").iterdir()):
        raise ValueError(
            "kinematic reference output has artifacts but no contract; rerun with restart=True"
        )
    _atomic_write(contract_path, contract_bytes)


def _shard_path(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard-{shard_index:05d}.json"


def _shard_commit_path(output_root: Path, shard_index: int) -> Path:
    return output_root / "shards" / f"shard-{shard_index:05d}.commit.json"


def _valid_shard(
    output_root: Path,
    *,
    shard_index: int,
    tasks: Sequence[tuple[str, Path]],
    contract_sha256: str,
) -> bool:
    shard_path = _shard_path(output_root, shard_index)
    commit_path = _shard_commit_path(output_root, shard_index)
    if not shard_path.is_file() or not commit_path.is_file():
        return False
    try:
        payload_bytes = shard_path.read_bytes()
        commit = _read_json(commit_path)
        scenario_ids = [scenario_id for scenario_id, _ in tasks]
        if commit != {
            "kind": "formal_train_kinematic_reference_shard_commit",
            "version": REFERENCE_VERSION,
            "shard_index": shard_index,
            "contract_sha256": contract_sha256,
            "scenario_count": len(tasks),
            "scenario_ids_sha256": _scenario_ids_sha256(scenario_ids),
            "payload_size_bytes": len(payload_bytes),
            "payload_sha256": _sha256(payload_bytes),
        }:
            return False
        payload = json.loads(payload_bytes.decode("utf-8"))
        if (
            payload.get("kind") != "formal_train_kinematic_reference_shard"
            or payload.get("version") != REFERENCE_VERSION
            or payload.get("shard_index") != shard_index
            or payload.get("contract_sha256") != contract_sha256
            or payload.get("scenario_ids") != scenario_ids
        ):
            return False
        statistics = _ReferenceAccumulator.from_payload(payload["statistics"])
        return statistics.scenario_count == len(tasks)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _write_shard(
    output_root: Path,
    *,
    shard_index: int,
    tasks: Sequence[tuple[str, Path]],
    results: Sequence[Mapping[str, Any]],
    contract_sha256: str,
) -> None:
    scenario_ids = [scenario_id for scenario_id, _ in tasks]
    result_ids = [str(result["scenario_id"]) for result in results]
    if result_ids != scenario_ids:
        raise ValueError("worker results do not match deterministic shard order")
    aggregate = _ReferenceAccumulator()
    for result in results:
        aggregate.merge(_ReferenceAccumulator.from_payload(result["statistics"]))
    payload = {
        "kind": "formal_train_kinematic_reference_shard",
        "version": REFERENCE_VERSION,
        "shard_index": shard_index,
        "contract_sha256": contract_sha256,
        "scenario_ids": scenario_ids,
        "statistics": aggregate.to_payload(),
    }
    payload_bytes = _json_bytes(payload, pretty=False)
    shard_path = _shard_path(output_root, shard_index)
    commit_path = _shard_commit_path(output_root, shard_index)
    _atomic_write(shard_path, payload_bytes)
    commit = {
        "kind": "formal_train_kinematic_reference_shard_commit",
        "version": REFERENCE_VERSION,
        "shard_index": shard_index,
        "contract_sha256": contract_sha256,
        "scenario_count": len(tasks),
        "scenario_ids_sha256": _scenario_ids_sha256(scenario_ids),
        "payload_size_bytes": len(payload_bytes),
        "payload_sha256": _sha256(payload_bytes),
    }
    _atomic_write(commit_path, _json_bytes(commit, pretty=True))


def _write_final_outputs(
    output_root: Path,
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    aggregate: _ReferenceAccumulator,
) -> None:
    aggregate_payload = {
        "kind": "formal_train_kinematic_reference_aggregate",
        "version": REFERENCE_VERSION,
        "contract_sha256": contract_sha256,
        "statistics": aggregate.to_payload(),
    }
    summary_payload = {
        "kind": "formal_train_kinematic_reference_summary",
        "version": REFERENCE_VERSION,
        "contract_sha256": contract_sha256,
        "source": {
            "manifest_path": contract["manifest_path"],
            "manifest_sha256": contract["manifest_sha256"],
            "filter_config_path": contract["filter_config_path"],
            "filter_config_sha256": contract["filter_config_sha256"],
            "scenario_ids_sha256": contract["scenario_ids_sha256"],
            "scenario_count": contract["scenario_count"],
            "source_split": "train",
            "validation_manifests_opened": False,
        },
        "derivation": {
            "sample_period_s": contract["sample_period_s"],
            "candidate_future_dtype": contract["candidate_future_dtype"],
            "distribution_scopes": contract["distribution_scopes"],
            "minimum_heading_speed_mps_by_actor_type": contract[
                "minimum_heading_speed_mps_by_actor_type"
            ],
            "metric_definitions": _metric_definitions(contract["sample_period_s"]),
            "histogram": _histogram_contract(),
            "hard_boundary_status": (
                "reference_distribution_only; hard boundaries require an explicit frozen policy"
            ),
        },
        "statistics": aggregate.summary(),
    }
    aggregate_bytes = _json_bytes(aggregate_payload, pretty=False)
    summary_bytes = _json_bytes(summary_payload, pretty=True)
    aggregate_path = output_root / "aggregate.json"
    summary_path = output_root / "summary.json"
    _atomic_write(aggregate_path, aggregate_bytes)
    _atomic_write(summary_path, summary_bytes)
    final_commit = {
        "kind": "formal_train_kinematic_reference_final_commit",
        "version": REFERENCE_VERSION,
        "contract_sha256": contract_sha256,
        "aggregate_size_bytes": len(aggregate_bytes),
        "aggregate_sha256": _sha256(aggregate_bytes),
        "summary_size_bytes": len(summary_bytes),
        "summary_sha256": _sha256(summary_bytes),
    }
    _atomic_write(
        output_root / "final.commit.json",
        _json_bytes(final_commit, pretty=True),
    )


def _remove_final_outputs(output_root: Path) -> None:
    for name in ("aggregate.json", "summary.json", "final.commit.json"):
        path = output_root / name
        if path.exists():
            path.unlink()


def _quantile_label(probability: float) -> str:
    percent = probability * 100.0
    return f"p{percent:g}".replace(".", "_")


def _scenario_ids_sha256(scenario_ids: Iterable[str]) -> str:
    return _sha256(
        json.dumps(
            list(scenario_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    )


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


__all__ = [
    "DEFAULT_DATA_ROOT",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_SAMPLE_PERIOD_S",
    "DEFAULT_SHARD_SIZE",
    "FORMAL_TRAIN_MANIFEST",
    "FORMAL_TRAIN_SCENARIO_COUNT",
    "METRIC_NAMES",
    "OBJECT_TYPES",
    "PARQUET_COLUMNS",
    "REFERENCE_TIMESTEPS",
    "ReferenceBuildResult",
    "ReferenceProgress",
    "TrackKinematicSamples",
    "build_kinematic_reference",
    "derive_track_kinematic_samples",
    "load_formal_train_rows",
    "scan_scenario_parquet",
]

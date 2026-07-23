"""Strict, immutable configuration for the conditional CVAE pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_CVAE_CONFIG = Path("configs/models/cvae_baseline.yaml")


@dataclass(frozen=True)
class ManifestConfig:
    development_train: Path
    development_validation: Path
    formal_train: Path
    internal_validation: Path
    final_validation: Path


@dataclass(frozen=True)
class DataConfig:
    root: Path
    skill_dir: Path
    detection_config: Path
    formal_candidate_pool: Path
    manifests: ManifestConfig


@dataclass(frozen=True)
class TensorizationConfig:
    schema_version: int
    history_steps: int
    future_steps: int
    anchor_frame: int
    sample_period_s: float
    minimum_history_steps: int
    require_complete_future: bool
    max_actors: int
    actor_radius_m: float
    max_map_polylines: int
    map_points_per_polyline: int
    map_radius_m: float
    map_types: tuple[str, ...]


@dataclass(frozen=True)
class CacheConfig:
    root: Path
    shard_size: int
    in_memory_shards: int
    resume: bool


@dataclass(frozen=True)
class ModelConfig:
    actor_feature_dim: int
    actor_type_embedding_dim: int
    actor_role_embedding_dim: int
    history_hidden_dim: int
    map_feature_dim: int
    map_type_embedding_dim: int
    map_hidden_dim: int
    interaction_hidden_dim: int
    interaction_layers: int
    interaction_heads: int
    skill_embedding_dim: int
    parameter_hidden_dim: int
    latent_dim: int
    decoder_hidden_dim: int
    dropout: float


@dataclass(frozen=True)
class LossConfig:
    reconstruction: str
    endpoint_weight: float
    kl_max_weight: float
    kl_warmup_steps: int
    map_soft_weight: float
    collision_soft_weight: float


@dataclass(frozen=True)
class RepairSplitConfig:
    audit: Path
    train_sample_index: Path
    development_sample_index: Path


@dataclass(frozen=True)
class RepairModelConfig:
    decoder_initial_delta_mode: str


@dataclass(frozen=True)
class MotionLossConfig:
    seam_velocity_weight: float
    velocity_weight: float
    acceleration_weight: float
    jerk_weight: float


@dataclass(frozen=True)
class ConditionRankingConfig:
    weight: float
    margin_per_latent_dim: float


@dataclass(frozen=True)
class ObservedSkillSamplerConfig:
    strategy: str
    target: str
    max_repeats_per_sample: int


@dataclass(frozen=True)
class GenerationRepairConfig:
    contract: str
    source_cache_partition: str
    split: RepairSplitConfig
    model: RepairModelConfig
    motion_loss: MotionLossConfig
    condition_ranking: ConditionRankingConfig
    sampler: ObservedSkillSamplerConfig


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    device: str
    amp: bool
    allow_tf32: bool
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    prefetch_factor: int
    persistent_workers: bool
    pin_memory: bool
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    development_max_epochs: int
    formal_max_epochs: int
    early_stopping_patience: int
    validation_every_epochs: int
    checkpoint_every_steps: int
    prior_samples: int
    best_metric: str


@dataclass(frozen=True)
class OverfitConfig:
    skill_id: str
    sample_count: int
    batch_size: int
    max_steps: int
    learning_rate: float


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_steps: int
    measured_steps: int
    repeats: int
    worker_candidates: tuple[int, ...]
    batch_size_candidates: tuple[int, ...]


@dataclass(frozen=True)
class OutputConfig:
    root: Path
    development: Path
    benchmarks: Path
    formal: Path


@dataclass(frozen=True)
class CVAEConfig:
    version: int
    data: DataConfig
    tensorization: TensorizationConfig
    cache: CacheConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    overfit: OverfitConfig
    benchmark: BenchmarkConfig
    outputs: OutputConfig
    repair: GenerationRepairConfig | None = None

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation of this configuration."""
        value = _canonical_value(self)
        if not isinstance(value, dict):  # pragma: no cover - guarded by the dataclass type
            raise TypeError("CVAEConfig did not canonicalize to a mapping")
        if self.repair is None:
            # Preserve the exact v1 baseline fingerprint and checkpoint contract.
            value.pop("repair", None)
        return value

    @property
    def fingerprint(self) -> str:
        """Return the SHA256 of the canonical JSON configuration."""
        payload = json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration cannot contain non-finite numbers")
        return value
    raise TypeError(f"unsupported canonical configuration value: {type(value).__name__}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _section(
    value: Any,
    name: str,
    expected_keys: Sequence[str],
) -> Mapping[str, Any]:
    mapping = _mapping(value, name)
    expected = set(expected_keys)
    missing = expected - set(mapping)
    unknown = set(mapping) - expected
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
    return mapping


def _integer(
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = mapping[key]
    name = f"{prefix}.{key}"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _number(
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    minimum: float | None = None,
    minimum_inclusive: bool = True,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float:
    value = mapping[key]
    name = f"{prefix}.{key}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = number < minimum if minimum_inclusive else number <= minimum
        if invalid:
            relation = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        invalid = number > maximum if maximum_inclusive else number >= maximum
        if invalid:
            relation = "at most" if maximum_inclusive else "less than"
            raise ValueError(f"{name} must be {relation} {maximum}")
    return number


def _boolean(mapping: Mapping[str, Any], key: str, prefix: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{prefix}.{key} must be a boolean")
    return value


def _string(
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    choices: set[str] | None = None,
) -> str:
    value = mapping[key]
    name = f"{prefix}.{key}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if choices is not None and value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value


def _relative_path(mapping: Mapping[str, Any], key: str, prefix: str) -> Path:
    value = _string(mapping, key, prefix)
    name = f"{prefix}.{key}"
    if "\\" in value:
        raise ValueError(f"{name} must use repository-relative POSIX separators")
    if re.match(r"^[A-Za-z]:[/\\]", value):
        raise ValueError(f"{name} must be repository-relative")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be repository-relative without '..'")
    return path


def _string_tuple(mapping: Mapping[str, Any], key: str, prefix: str) -> tuple[str, ...]:
    value = mapping[key]
    name = f"{prefix}.{key}"
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty strings")
        result.append(item)
    if not result:
        raise ValueError(f"{name} must be a non-empty sequence of strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _integer_tuple(
    mapping: Mapping[str, Any],
    key: str,
    prefix: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    value = mapping[key]
    name = f"{prefix}.{key}"
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{name} must contain integers")
        if item < minimum:
            raise ValueError(f"{name} values must be at least {minimum}")
        result.append(item)
    if not result:
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _parse_manifests(value: Any) -> ManifestConfig:
    keys = (
        "development_train",
        "development_validation",
        "formal_train",
        "internal_validation",
        "final_validation",
    )
    section = _section(value, "data.manifests", keys)
    manifests = ManifestConfig(
        **{
            key: _relative_path(section, key, "data.manifests")
            for key in keys
        }
    )
    values = {key: getattr(manifests, key) for key in keys}
    if len(set(values.values())) != len(values):
        final = manifests.final_validation
        aliases = [name for name, path in values.items() if name != "final_validation" and path == final]
        if aliases:
            raise ValueError(
                "final_validation cannot be used for training or model selection: "
                f"{aliases}"
            )
        raise ValueError("data.manifests paths must be unique")
    for key, path in values.items():
        expected_name = f"{key}.csv"
        if path.name != expected_name:
            raise ValueError(f"data.manifests.{key} must reference {expected_name}")
    return manifests


def _parse_data(value: Any) -> DataConfig:
    prefix = "data"
    section = _section(
        value,
        prefix,
        ("root", "skill_dir", "detection_config", "formal_candidate_pool", "manifests"),
    )
    return DataConfig(
        root=_relative_path(section, "root", prefix),
        skill_dir=_relative_path(section, "skill_dir", prefix),
        detection_config=_relative_path(section, "detection_config", prefix),
        formal_candidate_pool=_relative_path(section, "formal_candidate_pool", prefix),
        manifests=_parse_manifests(section["manifests"]),
    )


def _parse_tensorization(value: Any) -> TensorizationConfig:
    prefix = "tensorization"
    keys = (
        "schema_version",
        "history_steps",
        "future_steps",
        "anchor_frame",
        "sample_period_s",
        "minimum_history_steps",
        "require_complete_future",
        "max_actors",
        "actor_radius_m",
        "max_map_polylines",
        "map_points_per_polyline",
        "map_radius_m",
        "map_types",
    )
    section = _section(value, prefix, keys)
    history_steps = _integer(section, "history_steps", prefix, minimum=1)
    minimum_history_steps = _integer(
        section,
        "minimum_history_steps",
        prefix,
        minimum=1,
        maximum=history_steps,
    )
    anchor_frame = _integer(
        section,
        "anchor_frame",
        prefix,
        minimum=0,
        maximum=history_steps - 1,
    )
    return TensorizationConfig(
        schema_version=_integer(section, "schema_version", prefix, minimum=1),
        history_steps=history_steps,
        future_steps=_integer(section, "future_steps", prefix, minimum=1),
        anchor_frame=anchor_frame,
        sample_period_s=_number(
            section,
            "sample_period_s",
            prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        minimum_history_steps=minimum_history_steps,
        require_complete_future=_boolean(section, "require_complete_future", prefix),
        max_actors=_integer(section, "max_actors", prefix, minimum=1),
        actor_radius_m=_number(
            section, "actor_radius_m", prefix, minimum=0.0, minimum_inclusive=False
        ),
        max_map_polylines=_integer(section, "max_map_polylines", prefix, minimum=1),
        map_points_per_polyline=_integer(
            section, "map_points_per_polyline", prefix, minimum=2
        ),
        map_radius_m=_number(
            section, "map_radius_m", prefix, minimum=0.0, minimum_inclusive=False
        ),
        map_types=_string_tuple(section, "map_types", prefix),
    )


def _parse_cache(value: Any) -> CacheConfig:
    prefix = "cache"
    section = _section(value, prefix, ("root", "shard_size", "in_memory_shards", "resume"))
    return CacheConfig(
        root=_relative_path(section, "root", prefix),
        shard_size=_integer(section, "shard_size", prefix, minimum=1),
        in_memory_shards=_integer(section, "in_memory_shards", prefix, minimum=1),
        resume=_boolean(section, "resume", prefix),
    )


def _parse_model(value: Any) -> ModelConfig:
    prefix = "model"
    keys = tuple(field.name for field in fields(ModelConfig))
    section = _section(value, prefix, keys)
    integer_keys = keys[:-1]
    values = {
        key: _integer(section, key, prefix, minimum=1)
        for key in integer_keys
    }
    dropout = _number(
        section,
        "dropout",
        prefix,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if values["interaction_hidden_dim"] % values["interaction_heads"]:
        raise ValueError(
            "model.interaction_hidden_dim must be divisible by model.interaction_heads"
        )
    return ModelConfig(**values, dropout=dropout)


def _parse_loss(value: Any) -> LossConfig:
    prefix = "loss"
    keys = tuple(field.name for field in fields(LossConfig))
    section = _section(value, prefix, keys)
    return LossConfig(
        reconstruction=_string(
            section, "reconstruction", prefix, choices={"smooth_l1"}
        ),
        endpoint_weight=_number(section, "endpoint_weight", prefix, minimum=0.0),
        kl_max_weight=_number(section, "kl_max_weight", prefix, minimum=0.0),
        kl_warmup_steps=_integer(section, "kl_warmup_steps", prefix, minimum=0),
        map_soft_weight=_number(section, "map_soft_weight", prefix, minimum=0.0),
        collision_soft_weight=_number(
            section, "collision_soft_weight", prefix, minimum=0.0
        ),
    )


def _parse_repair_split(value: Any) -> RepairSplitConfig:
    prefix = "repair.split"
    keys = ("audit", "train_sample_index", "development_sample_index")
    section = _section(value, prefix, keys)
    result = RepairSplitConfig(
        audit=_relative_path(section, "audit", prefix),
        train_sample_index=_relative_path(section, "train_sample_index", prefix),
        development_sample_index=_relative_path(
            section,
            "development_sample_index",
            prefix,
        ),
    )
    if result.train_sample_index == result.development_sample_index:
        raise ValueError("repair train and development sample indexes must differ")
    forbidden = ("internal_validation", "final_validation")
    for field in fields(RepairSplitConfig):
        path = getattr(result, field.name).as_posix()
        if any(name in path for name in forbidden):
            raise ValueError(
                f"repair.split.{field.name} must stay inside Formal Train"
            )
    return result


def _parse_repair(value: Any) -> GenerationRepairConfig:
    prefix = "repair"
    section = _section(
        value,
        prefix,
        (
            "contract",
            "source_cache_partition",
            "split",
            "model",
            "motion_loss",
            "condition_ranking",
            "sampler",
        ),
    )
    contract = _string(section, "contract", prefix)
    if contract != "cvae_generation_repair_v1":
        raise ValueError("repair.contract must equal cvae_generation_repair_v1")
    source_partition = _string(section, "source_cache_partition", prefix)
    if source_partition != "formal_train":
        raise ValueError("repair.source_cache_partition must equal formal_train")

    model_prefix = "repair.model"
    model_section = _section(
        section["model"],
        model_prefix,
        ("decoder_initial_delta_mode",),
    )
    model = RepairModelConfig(
        decoder_initial_delta_mode=_string(
            model_section,
            "decoder_initial_delta_mode",
            model_prefix,
            choices={"history_velocity"},
        )
    )

    motion_prefix = "repair.motion_loss"
    motion_keys = tuple(field.name for field in fields(MotionLossConfig))
    motion_section = _section(section["motion_loss"], motion_prefix, motion_keys)
    motion = MotionLossConfig(
        **{
            key: _number(motion_section, key, motion_prefix, minimum=0.0)
            for key in motion_keys
        }
    )
    if not any(getattr(motion, key) > 0.0 for key in motion_keys):
        raise ValueError("repair.motion_loss must enable at least one component")

    ranking_prefix = "repair.condition_ranking"
    ranking_section = _section(
        section["condition_ranking"],
        ranking_prefix,
        ("weight", "margin_per_latent_dim"),
    )
    ranking = ConditionRankingConfig(
        weight=_number(
            ranking_section,
            "weight",
            ranking_prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        margin_per_latent_dim=_number(
            ranking_section,
            "margin_per_latent_dim",
            ranking_prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
    )

    sampler_prefix = "repair.sampler"
    sampler_section = _section(
        section["sampler"],
        sampler_prefix,
        ("strategy", "target", "max_repeats_per_sample"),
    )
    sampler = ObservedSkillSamplerConfig(
        strategy=_string(
            sampler_section,
            "strategy",
            sampler_prefix,
            choices={"observed_skill_balance_v1"},
        ),
        target=_string(
            sampler_section,
            "target",
            sampler_prefix,
            choices={"most_frequent_observed"},
        ),
        max_repeats_per_sample=_integer(
            sampler_section,
            "max_repeats_per_sample",
            sampler_prefix,
            minimum=1,
            maximum=8,
        ),
    )
    return GenerationRepairConfig(
        contract=contract,
        source_cache_partition=source_partition,
        split=_parse_repair_split(section["split"]),
        model=model,
        motion_loss=motion,
        condition_ranking=ranking,
        sampler=sampler,
    )


def _parse_training(value: Any) -> TrainingConfig:
    prefix = "training"
    keys = tuple(field.name for field in fields(TrainingConfig))
    section = _section(value, prefix, keys)
    num_workers = _integer(section, "num_workers", prefix, minimum=0)
    persistent_workers = _boolean(section, "persistent_workers", prefix)
    if num_workers == 0 and persistent_workers:
        raise ValueError("training.persistent_workers requires training.num_workers > 0")
    return TrainingConfig(
        seed=_integer(section, "seed", prefix, minimum=0),
        device=_string(section, "device", prefix, choices={"cpu", "cuda"}),
        amp=_boolean(section, "amp", prefix),
        allow_tf32=_boolean(section, "allow_tf32", prefix),
        batch_size=_integer(section, "batch_size", prefix, minimum=1),
        gradient_accumulation_steps=_integer(
            section, "gradient_accumulation_steps", prefix, minimum=1
        ),
        num_workers=num_workers,
        prefetch_factor=_integer(section, "prefetch_factor", prefix, minimum=1),
        persistent_workers=persistent_workers,
        pin_memory=_boolean(section, "pin_memory", prefix),
        learning_rate=_number(
            section,
            "learning_rate",
            prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        weight_decay=_number(section, "weight_decay", prefix, minimum=0.0),
        gradient_clip_norm=_number(
            section,
            "gradient_clip_norm",
            prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
        development_max_epochs=_integer(
            section, "development_max_epochs", prefix, minimum=1
        ),
        formal_max_epochs=_integer(section, "formal_max_epochs", prefix, minimum=1),
        early_stopping_patience=_integer(
            section, "early_stopping_patience", prefix, minimum=0
        ),
        validation_every_epochs=_integer(
            section, "validation_every_epochs", prefix, minimum=1
        ),
        checkpoint_every_steps=_integer(
            section, "checkpoint_every_steps", prefix, minimum=1
        ),
        prior_samples=_integer(section, "prior_samples", prefix, minimum=1),
        best_metric=_string(
            section,
            "best_metric",
            prefix,
            choices={"ade", "fde", "min_ade_6", "min_fde_6"},
        ),
    )


def _parse_overfit(value: Any) -> OverfitConfig:
    prefix = "overfit"
    section = _section(
        value,
        prefix,
        ("skill_id", "sample_count", "batch_size", "max_steps", "learning_rate"),
    )
    return OverfitConfig(
        skill_id=_string(section, "skill_id", prefix),
        sample_count=_integer(section, "sample_count", prefix, minimum=1),
        batch_size=_integer(section, "batch_size", prefix, minimum=1),
        max_steps=_integer(section, "max_steps", prefix, minimum=1),
        learning_rate=_number(
            section,
            "learning_rate",
            prefix,
            minimum=0.0,
            minimum_inclusive=False,
        ),
    )


def _parse_benchmark(value: Any) -> BenchmarkConfig:
    prefix = "benchmark"
    section = _section(
        value,
        prefix,
        ("warmup_steps", "measured_steps", "repeats", "worker_candidates", "batch_size_candidates"),
    )
    return BenchmarkConfig(
        warmup_steps=_integer(section, "warmup_steps", prefix, minimum=0),
        measured_steps=_integer(section, "measured_steps", prefix, minimum=1),
        repeats=_integer(section, "repeats", prefix, minimum=1),
        worker_candidates=_integer_tuple(
            section, "worker_candidates", prefix, minimum=0
        ),
        batch_size_candidates=_integer_tuple(
            section, "batch_size_candidates", prefix, minimum=1
        ),
    )


def _parse_outputs(value: Any) -> OutputConfig:
    prefix = "outputs"
    section = _section(value, prefix, ("root", "development", "benchmarks", "formal"))
    config = OutputConfig(
        root=_relative_path(section, "root", prefix),
        development=_relative_path(section, "development", prefix),
        benchmarks=_relative_path(section, "benchmarks", prefix),
        formal=_relative_path(section, "formal", prefix),
    )
    for key in ("development", "benchmarks", "formal"):
        path = getattr(config, key)
        try:
            path.relative_to(config.root)
        except ValueError as exc:
            raise ValueError(f"outputs.{key} must be contained by outputs.root") from exc
    return config


def load_cvae_config(path: str | Path = DEFAULT_CVAE_CONFIG) -> CVAEConfig:
    """Load and strictly validate one conditional CVAE YAML configuration."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration: {source}") from exc
    base_root_keys = (
        "version",
        "data",
        "tensorization",
        "cache",
        "model",
        "loss",
        "training",
        "overfit",
        "benchmark",
        "outputs",
    )
    raw_mapping = _mapping(raw, "configuration")
    raw_version = raw_mapping.get("version")
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("configuration.version must be an integer")
    root_keys = (
        *base_root_keys,
        *(("repair",) if raw_version == 2 else ()),
    )
    root = _section(raw_mapping, "configuration", root_keys)
    version = _integer(root, "version", "configuration", minimum=1)
    if version not in {1, 2}:
        raise ValueError("configuration.version must equal 1 or 2")
    return CVAEConfig(
        version=version,
        data=_parse_data(root["data"]),
        tensorization=_parse_tensorization(root["tensorization"]),
        cache=_parse_cache(root["cache"]),
        model=_parse_model(root["model"]),
        loss=_parse_loss(root["loss"]),
        training=_parse_training(root["training"]),
        overfit=_parse_overfit(root["overfit"]),
        benchmark=_parse_benchmark(root["benchmark"]),
        outputs=_parse_outputs(root["outputs"]),
        repair=None if version == 1 else _parse_repair(root["repair"]),
    )

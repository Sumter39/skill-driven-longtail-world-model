"""Train, resume, or benchmark the conditional CVAE."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from skilldrive.data import (
    CVAECachedDataset,
    ShardShuffleSampler,
    build_cvae_schema,
    cvae_schema_fingerprint,
    read_manifest,
)
from skilldrive.data.cvae_samples import CVAESchema
from skilldrive.models import ConditionalCVAE
from skilldrive.training import (
    DEFAULT_CVAE_CONFIG,
    TrainingProgress,
    load_checkpoint,
    load_cvae_config,
    save_checkpoint,
)
from skilldrive.training.config import CVAEConfig, LossConfig, TrainingConfig
from skilldrive.training.trainer import (
    BenchmarkResult,
    EvaluationResult,
    benchmark_training,
    evaluate,
    train_epoch,
)


STAGES = ("overfit", "development", "benchmark", "formal")
VALIDATION_SEED_OFFSET = 100_000
BENCHMARK_SAMPLER_STRATEGY = "shard_shuffle_epoch_cycle_v1"


class _DatasetView(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)
        self.entries = [dataset.entries[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[self.indices[index]]


class _RepeatedDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset, *, repeats: int) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if len(dataset) == 0:
            raise ValueError("cannot repeat an empty dataset")
        self.dataset = dataset
        self.repeats = repeats
        self.entries = list(dataset.entries) * repeats

    def __len__(self) -> int:
        return len(self.dataset) * self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.dataset[index % len(self.dataset)]


class _DeterministicBenchmarkSampler(Sampler[int]):
    """Yield a finite full-batch stream while cycling deterministic shard epochs."""

    def __init__(
        self,
        dataset: CVAECachedDataset,
        *,
        seed: int,
        num_samples: int,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("benchmark sampler requires a non-empty dataset")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise ValueError("benchmark sampler num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("benchmark sampler num_samples must be a positive integer")
        self.dataset = dataset
        self.seed = int(seed)
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        epoch_sampler = ShardShuffleSampler(self.dataset, seed=self.seed)
        remaining = self.num_samples
        epoch = 0
        while remaining > 0:
            epoch_sampler.set_epoch(epoch)
            take = min(remaining, len(self.dataset))
            yield from itertools.islice(iter(epoch_sampler), take)
            remaining -= take
            epoch += 1


def _benchmark_sample_stream_metadata(
    sampler: _DeterministicBenchmarkSampler,
    *,
    warmup_samples: int,
) -> dict[str, Any]:
    if warmup_samples < 0 or warmup_samples > len(sampler):
        raise ValueError("warmup_samples must be inside the benchmark sample stream")
    measured_digest = hashlib.sha256()
    for position, index in enumerate(sampler):
        if position >= warmup_samples:
            measured_digest.update(f"{index}\n".encode("ascii"))
    contract = {
        "strategy": BENCHMARK_SAMPLER_STRATEGY,
        "seed": sampler.seed,
        "dataset_samples": len(sampler.dataset),
        "stream_samples": len(sampler),
        "warmup_samples": warmup_samples,
        "measured_samples": len(sampler) - warmup_samples,
    }
    return {
        **contract,
        "contract_sha256": _hash_value(contract),
        "measured_order_sha256": measured_digest.hexdigest(),
    }


class _MaterializedDataset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset) -> None:
        if len(dataset) == 0:
            raise ValueError("cannot materialize an empty dataset")
        self.entries = list(dataset.entries)
        self.samples = tuple(dataset[index] for index in range(len(dataset)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cache_contract(
    dataset: Dataset,
    *,
    expected_partition: str,
    manifest_path: Path,
    schema_sha256: str,
    candidate_pool_path: Path,
) -> None:
    if expected_partition == "final_validation":
        raise ValueError("Final Validation is forbidden for CVAE training and selection")
    cache_manifest = dataset.cache_manifest
    if cache_manifest.get("status") != "complete":
        raise ValueError(f"{expected_partition} cache must be complete")
    if cache_manifest.get("partition") != expected_partition:
        raise ValueError(
            f"cache partition is {cache_manifest.get('partition')!r}; "
            f"expected {expected_partition!r}"
        )
    rows = read_manifest(manifest_path)
    expected_split = "train" if expected_partition == "formal_train" else expected_partition
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"{manifest_path} contains rows outside split {expected_split}")
    counts = cache_manifest.get("counts") or {}
    expected_count = len(rows)
    if counts.get("manifest_scenarios") != expected_count:
        raise ValueError(
            f"{expected_partition} cache covers {counts.get('manifest_scenarios')} manifest "
            f"scenarios; current manifest contains {expected_count}"
        )
    if counts.get("processed_manifest_scenarios") != expected_count:
        raise ValueError(
            f"{expected_partition} cache processed "
            f"{counts.get('processed_manifest_scenarios')} of {expected_count} scenarios"
        )
    inputs = cache_manifest.get("inputs") or {}
    expected_inputs = {
        "manifest_sha256": _sha256(manifest_path),
        "schema_sha256": schema_sha256,
        "candidate_pool_sha256": _sha256(candidate_pool_path),
    }
    mismatches = {
        name: (inputs.get(name), expected)
        for name, expected in expected_inputs.items()
        if inputs.get(name) != expected
    }
    if mismatches:
        details = "; ".join(
            f"{name}: cache={actual!r}, current={expected!r}"
            for name, (actual, expected) in sorted(mismatches.items())
        )
        raise ValueError(f"{expected_partition} cache input fingerprint mismatch: {details}")


def _hash_value(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_fingerprint(cache_dir: Path) -> str:
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    semantic = {
        "version": manifest.get("version"),
        "status": manifest.get("status"),
        "partition": manifest.get("partition"),
        "inputs": manifest.get("inputs"),
        "counts": manifest.get("counts"),
        "label_counts": manifest.get("label_counts"),
        "rejection_counts": manifest.get("rejection_counts"),
        "sample_index": manifest.get("sample_index"),
        "shard_size": manifest.get("shard_size"),
        "shards": manifest.get("shards"),
    }
    return _hash_value(semantic)


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
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


def model_kwargs_from_config(
    config: CVAEConfig,
    schema: CVAESchema,
) -> dict[str, Any]:
    """Build the complete model constructor contract from config and vocabularies."""
    model = config.model
    return {
        "actor_feature_dim": model.actor_feature_dim,
        "map_feature_dim": model.map_feature_dim,
        "num_actor_types": len(schema.actor_type_vocabulary.tokens),
        "num_actor_roles": len(schema.role_vocabulary.tokens),
        "num_map_types": len(schema.map_type_vocabulary.tokens),
        "num_skills": len(schema.skill_vocabulary.tokens),
        "parameter_dim": schema.parameter_schema.dimension,
        "actor_type_embedding_dim": model.actor_type_embedding_dim,
        "actor_role_embedding_dim": model.actor_role_embedding_dim,
        "history_hidden_dim": model.history_hidden_dim,
        "map_type_embedding_dim": model.map_type_embedding_dim,
        "map_hidden_dim": model.map_hidden_dim,
        "interaction_hidden_dim": model.interaction_hidden_dim,
        "interaction_layers": model.interaction_layers,
        "interaction_heads": model.interaction_heads,
        "skill_embedding_dim": model.skill_embedding_dim,
        "parameter_hidden_dim": model.parameter_hidden_dim,
        "latent_dim": model.latent_dim,
        "decoder_hidden_dim": model.decoder_hidden_dim,
        "future_steps": config.tensorization.future_steps,
        "dropout": model.dropout,
    }


def build_model_from_config(config: CVAEConfig, schema: CVAESchema) -> ConditionalCVAE:
    return ConditionalCVAE(**model_kwargs_from_config(config, schema))


def evaluation_to_dict(result: EvaluationResult, prior_samples: int) -> dict[str, Any]:
    return {
        "sample_count": result.sample_count,
        "posterior": {
            "ade": result.posterior.ade,
            "fde": result.posterior.fde,
            "kl": result.posterior_kl,
        },
        "prior": {
            "samples": prior_samples,
            "min_ade": result.prior.ade,
            "min_fde": result.prior.fde,
        },
        "constant_velocity": {
            "ade": result.constant_velocity.ade,
            "fde": result.constant_velocity.fde,
        },
    }


def validation_loss_to_dict(
    result: EvaluationResult,
    loss_config: LossConfig,
) -> dict[str, Any]:
    loss = result.validation_loss
    if loss is None:
        raise ValueError("evaluation did not compute validation loss")
    return {
        "total_loss": loss.total_loss,
        "reconstruction_loss": loss.reconstruction_loss,
        "endpoint_loss": loss.endpoint_loss,
        "kl_loss": loss.kl_loss,
        "kl_weight": loss.kl_weight,
        "endpoint_weight": loss_config.endpoint_weight,
        "map_soft_loss": loss.map_soft_loss,
        "map_soft_weight": loss_config.map_soft_weight,
        "collision_soft_loss": loss.collision_soft_loss,
        "collision_soft_weight": loss_config.collision_soft_weight,
        "sample_count": loss.sums.sample_count,
        "valid_point_count": loss.sums.valid_point_count,
    }


def _read_metric_evidence(metrics_path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "initial_evaluation": None,
        "initial_validation_loss": None,
        "final_evaluation": None,
        "final_validation_loss": None,
    }
    if not metrics_path.is_file():
        return evidence
    for line_number, raw_line in enumerate(
        metrics_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid metrics JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError(f"metrics line {line_number} must contain a mapping")
        if record.get("kind") == "initial_evaluation":
            evidence["initial_evaluation"] = record.get("validation")
            evidence["initial_validation_loss"] = record.get("validation_loss")
        elif record.get("kind") == "epoch" and record.get("validation") is not None:
            evidence["final_evaluation"] = record["validation"]
            evidence["final_validation_loss"] = record.get("validation_loss")
    return evidence


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stage_paths(
    config: CVAEConfig,
    stage: str,
    root: Path,
    cache_root_override: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    if cache_root_override is None:
        cache_root = root / config.cache.root
    else:
        cache_root = Path(cache_root_override)
        if not cache_root.is_absolute():
            cache_root = root / cache_root
    if stage == "overfit":
        train_cache = cache_root / "development_train"
        validation_cache = train_cache
    elif stage in {"development", "benchmark"}:
        train_cache = cache_root / "development_train"
        validation_cache = cache_root / "development_validation"
    else:
        train_cache = cache_root / "formal_train"
        validation_cache = cache_root / "internal_validation"
    if stage == "overfit":
        output = root / config.outputs.development / "overfit"
    elif stage == "development":
        output = root / config.outputs.development
    elif stage == "benchmark":
        output = root / config.outputs.benchmarks
    else:
        output = root / config.outputs.formal
    return train_cache, validation_cache, output


def _manifest_path(config: CVAEConfig, partition: str, root: Path) -> Path:
    allowed = {
        "development_train": config.data.manifests.development_train,
        "development_validation": config.data.manifests.development_validation,
        "formal_train": config.data.manifests.formal_train,
        "internal_validation": config.data.manifests.internal_validation,
    }
    try:
        relative = allowed[partition]
    except KeyError:
        raise ValueError(f"unsupported training cache partition: {partition}") from None
    return root / relative


def _validation_seed(training: TrainingConfig) -> int:
    return training.seed + VALIDATION_SEED_OFFSET


def _effective_training(
    config: CVAEConfig,
    stage: str,
    batch_size: int | None,
    num_workers: int | None,
    amp: bool | None = None,
    prefetch_factor: int | None = None,
) -> tuple[TrainingConfig, float]:
    training = config.training
    default_batch = config.overfit.batch_size if stage == "overfit" else training.batch_size
    effective_batch = default_batch if batch_size is None else batch_size
    effective_workers = training.num_workers if num_workers is None else num_workers
    if effective_batch <= 0:
        raise ValueError("batch_size override must be positive")
    if effective_workers < 0:
        raise ValueError("num_workers override must be nonnegative")
    effective_prefetch = (
        training.prefetch_factor if prefetch_factor is None else prefetch_factor
    )
    if effective_prefetch <= 0:
        raise ValueError("prefetch_factor override must be positive")
    training = replace(
        training,
        amp=training.amp if amp is None else amp,
        batch_size=effective_batch,
        num_workers=effective_workers,
        prefetch_factor=effective_prefetch,
        persistent_workers=training.persistent_workers and effective_workers > 0,
    )
    learning_rate = (
        config.overfit.learning_rate if stage == "overfit" else training.learning_rate
    )
    return training, learning_rate


def _loader(
    dataset: Dataset,
    *,
    training: TrainingConfig,
    sampler: Any | None,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": training.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": training.num_workers,
        "pin_memory": training.pin_memory,
        "drop_last": drop_last,
        "generator": generator,
    }
    if training.num_workers > 0:
        kwargs.update(
            prefetch_factor=training.prefetch_factor,
            persistent_workers=training.persistent_workers,
        )
    return DataLoader(**kwargs)


def _overfit_view(dataset: CVAECachedDataset, config: CVAEConfig) -> _DatasetView:
    indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry["spec"]["skill_id"] == config.overfit.skill_id
        and entry["spec"]["skill_supervision_mask"]
    ][: config.overfit.sample_count]
    if len(indices) < config.overfit.sample_count:
        raise ValueError(
            f"overfit cache contains {len(indices)} {config.overfit.skill_id} samples; "
            f"expected {config.overfit.sample_count}"
        )
    return _DatasetView(dataset, indices)


def _base_view(dataset: Dataset) -> _DatasetView:
    indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry["spec"]["skill_id"] == "<none>"
        and not entry["spec"]["skill_supervision_mask"]
    ]
    if not indices:
        raise ValueError("validation cache contains no <none> base samples")
    return _DatasetView(dataset, indices)


def _observed_view(dataset: Dataset) -> _DatasetView:
    indices = [
        index
        for index, entry in enumerate(dataset.entries)
        if entry["spec"]["skill_supervision_mask"]
        and entry["spec"]["skill_id"] != "<none>"
    ]
    return _DatasetView(dataset, indices)


def _fingerprints(
    *,
    config: CVAEConfig,
    schema: CVAESchema,
    stage: str,
    training: TrainingConfig,
    learning_rate: float,
    train_cache: Path,
    validation_cache: Path,
) -> dict[str, str]:
    model_kwargs = model_kwargs_from_config(config, schema)
    optimizer_contract = {
        "learning_rate": learning_rate,
        "weight_decay": training.weight_decay,
        "batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "amp": training.amp,
        "allow_tf32": training.allow_tf32,
        "loss": {
            "reconstruction": config.loss.reconstruction,
            "endpoint_weight": config.loss.endpoint_weight,
            "kl_max_weight": config.loss.kl_max_weight,
            "kl_warmup_steps": config.loss.kl_warmup_steps,
            "map_soft_weight": config.loss.map_soft_weight,
            "collision_soft_weight": config.loss.collision_soft_weight,
        },
    }
    pipeline_contract = {
        "tensorization": asdict(config.tensorization),
        "model": asdict(config.model),
        "loss": asdict(config.loss),
    }
    schedule_contract = {
        "seed": training.seed,
        "development_max_epochs": training.development_max_epochs,
        "formal_max_epochs": training.formal_max_epochs,
        "early_stopping_patience": training.early_stopping_patience,
        "validation_every_epochs": training.validation_every_epochs,
        "checkpoint_every_steps": training.checkpoint_every_steps,
        "prior_samples": training.prior_samples,
        "best_metric": training.best_metric,
        "overfit_max_steps": config.overfit.max_steps,
    }
    return {
        "config": _hash_value(pipeline_contract),
        "model": _hash_value(model_kwargs),
        "optimizer": _hash_value(optimizer_contract),
        "schedule": _hash_value(schedule_contract),
        "stage": _hash_value(stage),
        "train_cache": _cache_fingerprint(train_cache),
        "validation_cache": _cache_fingerprint(validation_cache),
    }


def _make_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler | None:
    if device.type != "cuda" or not amp or torch.cuda.is_bf16_supported():
        return None
    return torch.amp.GradScaler(device.type, enabled=True)


def _resume_path(value: str, latest: Path, root: Path) -> Path | None:
    if value == "none":
        return None
    if value == "auto":
        return latest if latest.is_file() else None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _checkpoint_extra(
    generator: torch.Generator,
    *,
    stage: str,
    epochs_without_improvement: int,
    timing: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "epochs_without_improvement": epochs_without_improvement,
        "training_generator_state": generator.get_state(),
        "timing": dict(timing),
    }


def _progress_line(
    *,
    stage: str,
    epoch: int,
    next_batch: int,
    total_batches: int,
    completed_batches: int,
    planned_batches: int,
    processed_batches: int,
    global_step: int,
    loss: float,
    processed_samples: int,
    elapsed: float,
) -> str:
    overall_fraction = (
        1.0 if planned_batches == 0 else completed_batches / planned_batches
    )
    speed = processed_samples / elapsed if elapsed > 0 else 0.0
    batch_speed = processed_batches / elapsed if elapsed > 0 else 0.0
    eta = (
        0.0
        if batch_speed <= 0
        else max(planned_batches - completed_batches, 0) / batch_speed
    )
    return (
        f"{stage} [{min(overall_fraction, 1.0) * 100:6.2f}%] epoch {epoch + 1} "
        f"batch {next_batch}/{total_batches} step {global_step} "
        f"loss {loss:.6f} +{processed_samples} samples "
        f"{speed:.2f} samples/s elapsed {elapsed:.1f}s ETA {eta:.1f}s"
    )


def _run_benchmark(
    *,
    config: CVAEConfig,
    schema: CVAESchema,
    training: TrainingConfig,
    learning_rate: float,
    train_dataset: CVAECachedDataset,
    output_dir: Path,
    max_steps: int | None,
    progress_stream: TextIO,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measured_steps = config.benchmark.measured_steps if max_steps is None else max_steps
    microbatches = (
        config.benchmark.warmup_steps + measured_steps
    ) * training.gradient_accumulation_steps
    stream_samples = microbatches * training.batch_size
    warmup_samples = (
        config.benchmark.warmup_steps
        * training.gradient_accumulation_steps
        * training.batch_size
    )
    sampler = _DeterministicBenchmarkSampler(
        train_dataset,
        seed=training.seed,
        num_samples=stream_samples,
    )
    sampler_metadata = _benchmark_sample_stream_metadata(
        sampler,
        warmup_samples=warmup_samples,
    )
    cache_dir = str(Path(train_dataset.cache_dir).parent.resolve())
    cache_path_id = _hash_value(cache_dir)[:8]
    cache_fingerprint = _cache_fingerprint(Path(train_dataset.cache_dir))
    cache_content_id = cache_fingerprint[:8]
    resolved_device = torch.device(training.device)
    benchmark_contract = {
        "cache_fingerprint": cache_fingerprint,
        "schema_fingerprint": cvae_schema_fingerprint(schema),
        "model": model_kwargs_from_config(config, schema),
        "loss": asdict(config.loss),
        "seed": training.seed,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "sampler_strategy": BENCHMARK_SAMPLER_STRATEGY,
        "dataset_samples": len(train_dataset),
        "warmup_steps": config.benchmark.warmup_steps,
        "measured_steps": measured_steps,
        "repeats": config.benchmark.repeats,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_type": resolved_device.type,
        "device_name": (
            torch.cuda.get_device_name(resolved_device)
            if resolved_device.type == "cuda"
            else "cpu"
        ),
        "source_sha256": {
            "train_cvae": _sha256(Path(__file__)),
            "trainer": _sha256(Path(benchmark_training.__code__.co_filename)),
        },
    }
    benchmark_contract_id = _hash_value(benchmark_contract)
    candidate_id = (
        f"batch-{training.batch_size}_workers-{training.num_workers}_"
        f"prefetch-{training.prefetch_factor}_"
        f"amp-{int(training.amp)}_tf32-{int(training.allow_tf32)}_"
        f"steps-{measured_steps}_fullbatches-1_"
        f"stream-{sampler_metadata['contract_sha256'][:8]}_"
        f"cache-{cache_path_id}-{cache_content_id}"
    )
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = candidate_dir / "metrics.jsonl"
    _atomic_write(metrics_path, b"")
    results: list[dict[str, Any]] = []
    for repeat in range(config.benchmark.repeats):
        _seed_everything(training.seed)
        if torch.device(training.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(training.device)
        model_setup_started = time.perf_counter()
        model = build_model_from_config(config, schema).to(training.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=training.weight_decay,
        )
        model_setup_seconds = time.perf_counter() - model_setup_started
        loader_setup_started = time.perf_counter()
        loader = _loader(
            train_dataset,
            training=training,
            sampler=sampler,
            generator=torch.Generator().manual_seed(training.seed + 1),
            drop_last=True,
        )
        loader_setup_seconds = time.perf_counter() - loader_setup_started
        generator = torch.Generator(device=torch.device(training.device)).manual_seed(
            training.seed
        )
        result: BenchmarkResult = benchmark_training(
            model,
            optimizer,
            loader,
            device=training.device,
            loss_config=config.loss,
            training_config=training,
            global_step=0,
            generator=generator,
            warmup_steps=config.benchmark.warmup_steps,
            measured_steps=measured_steps,
            scaler=_make_scaler(torch.device(training.device), training.amp),
        )
        if result.measured_samples != sampler_metadata["measured_samples"]:
            raise RuntimeError(
                "benchmark measured sample count differs from deterministic stream"
            )
        value = {
            "kind": "benchmark",
            "repeat": repeat,
            "configuration": {
                "batch_size": training.batch_size,
                "num_workers": training.num_workers,
                "prefetch_factor": training.prefetch_factor,
                "persistent_workers": training.persistent_workers,
                "pin_memory": training.pin_memory,
                "amp": training.amp,
                "allow_tf32": training.allow_tf32,
                "warmup_steps": config.benchmark.warmup_steps,
                "measured_steps": measured_steps,
                "cache_dir": cache_dir,
                "cache_fingerprint": cache_fingerprint,
                "benchmark_contract_id": benchmark_contract_id,
                "sampler": sampler_metadata,
            },
            "model_setup_seconds": model_setup_seconds,
            "loader_setup_seconds": loader_setup_seconds,
            "first_batch_seconds": result.startup_seconds,
            "warmup_seconds": result.warmup_seconds,
            "p50_step_seconds": result.p50_step_seconds,
            "p95_step_seconds": result.p95_step_seconds,
            "samples_per_second": result.samples_per_second,
            "data_wait_fraction": result.data_wait_fraction,
            "measured_samples": result.measured_samples,
            "cpu_metrics_available": result.cpu_metrics_available,
            "cpu_busy_percent": result.cpu_busy_percent,
            "cpu_iowait_percent": result.cpu_iowait_percent,
            "gpu_utilization_available": result.gpu_utilization_available,
            "gpu_utilization_mean_percent": result.gpu_utilization_mean_percent,
            "gpu_utilization_p50_percent": result.gpu_utilization_p50_percent,
            "gpu_utilization_p95_percent": result.gpu_utilization_p95_percent,
            "gpu_utilization_sample_count": result.gpu_utilization_sample_count,
            "monitor_overhead_seconds": result.monitor_overhead_seconds,
            "peak_vram_mib": (
                torch.cuda.max_memory_allocated(training.device) / 1024**2
                if torch.device(training.device).type == "cuda"
                else 0.0
            ),
        }
        results.append(value)
        _append_jsonl(metrics_path, value)
        print(
            f"benchmark repeat {repeat + 1}/{config.benchmark.repeats}: "
            f"{result.samples_per_second:.2f} samples/s",
            file=progress_stream,
        )
    throughputs = [value["samples_per_second"] for value in results]
    step_p50s = [value["p50_step_seconds"] for value in results]
    step_p95s = [value["p95_step_seconds"] for value in results]

    def median_present(name: str) -> float | None:
        values = [
            float(value[name])
            for value in results
            if value.get(name) is not None
        ]
        return statistics.median(values) if values else None

    median_throughput = statistics.median(throughputs)
    summary = {
        "stage": "benchmark",
        "candidate_id": candidate_id,
        "benchmark_contract_id": benchmark_contract_id,
        "benchmark_contract": benchmark_contract,
        "configuration": results[0]["configuration"],
        "median_samples_per_second": median_throughput,
        "throughput_range": [min(throughputs), max(throughputs)],
        "throughput_relative_range_percent": (
            (max(throughputs) - min(throughputs)) / median_throughput * 100.0
            if median_throughput > 0.0
            else None
        ),
        "median_p50_step_seconds": statistics.median(step_p50s),
        "median_p95_step_seconds": statistics.median(step_p95s),
        "median_data_wait_fraction": median_present("data_wait_fraction"),
        "median_model_setup_seconds": median_present("model_setup_seconds"),
        "median_loader_setup_seconds": median_present("loader_setup_seconds"),
        "median_first_batch_seconds": median_present("first_batch_seconds"),
        "median_warmup_seconds": median_present("warmup_seconds"),
        "median_monitor_overhead_seconds": median_present(
            "monitor_overhead_seconds"
        ),
        "median_peak_vram_mib": median_present("peak_vram_mib"),
        "median_cpu_busy_percent": median_present("cpu_busy_percent"),
        "median_cpu_iowait_percent": median_present("cpu_iowait_percent"),
        "median_gpu_utilization_mean_percent": median_present(
            "gpu_utilization_mean_percent"
        ),
        "median_gpu_utilization_p50_percent": median_present(
            "gpu_utilization_p50_percent"
        ),
        "median_gpu_utilization_p95_percent": median_present(
            "gpu_utilization_p95_percent"
        ),
        "results": results,
    }
    _atomic_json(candidate_dir / "summary.json", summary)
    index_path = output_dir / "summary.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.is_file()
        else {"stage": "benchmark"}
    )
    contracts = index.setdefault("contracts", {})
    contract_entry = contracts.setdefault(
        benchmark_contract_id,
        {"benchmark_contract": benchmark_contract, "candidates": {}},
    )
    if contract_entry.get("benchmark_contract") != benchmark_contract:
        raise ValueError("benchmark contract hash collision or index corruption")
    contract_entry.setdefault("candidates", {})[candidate_id] = summary
    index["active_contract_id"] = benchmark_contract_id
    _atomic_json(index_path, index)
    return summary


def run_training(
    *,
    config_path: str | Path = DEFAULT_CVAE_CONFIG,
    stage: str,
    project_root: str | Path = ".",
    resume: str = "auto",
    max_steps: int | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_root: str | Path | None = None,
    amp: bool | None = None,
    prefetch_factor: int | None = None,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Run one resumable training stage using already prepared caches."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_epochs is not None and max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    root = Path(project_root)
    config = load_cvae_config(config_path)
    schema = build_cvae_schema(root / config.data.skill_dir)
    training, learning_rate = _effective_training(
        config,
        stage,
        batch_size,
        num_workers,
        amp,
        prefetch_factor,
    )
    if training.best_metric != "min_fde_6" or training.prior_samples != 6:
        raise ValueError("training CLI requires best_metric=min_fde_6 and prior_samples=6")
    device = torch.device(training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = training.allow_tf32
        torch.backends.cudnn.allow_tf32 = training.allow_tf32

    train_cache, validation_cache, output_dir = _stage_paths(
        config,
        stage,
        root,
        cache_root,
    )
    train_partition = "formal_train" if stage == "formal" else "development_train"
    validation_partition = (
        "internal_validation"
        if stage == "formal"
        else ("development_train" if stage == "overfit" else "development_validation")
    )
    raw_train_dataset = CVAECachedDataset(train_cache, schema=schema)
    raw_validation_dataset = (
        raw_train_dataset
        if validation_cache == train_cache
        else CVAECachedDataset(validation_cache, schema=schema)
    )
    schema_sha256 = cvae_schema_fingerprint(schema)
    candidate_pool_path = root / config.data.formal_candidate_pool
    _validate_cache_contract(
        raw_train_dataset,
        expected_partition=train_partition,
        manifest_path=_manifest_path(config, train_partition, root),
        schema_sha256=schema_sha256,
        candidate_pool_path=candidate_pool_path,
    )
    if raw_validation_dataset is not raw_train_dataset:
        _validate_cache_contract(
            raw_validation_dataset,
            expected_partition=validation_partition,
            manifest_path=_manifest_path(config, validation_partition, root),
            schema_sha256=schema_sha256,
            candidate_pool_path=candidate_pool_path,
        )

    if stage == "overfit":
        if max_epochs is not None:
            raise ValueError("overfit stage is controlled by max_steps, not max_epochs")
        if max_steps is not None and max_steps > config.overfit.max_steps:
            raise ValueError("overfit max_steps cannot exceed the configured overfit limit")
        step_limit = config.overfit.max_steps if max_steps is None else max_steps
        validation_dataset = _MaterializedDataset(
            _overfit_view(raw_train_dataset, config)
        )
        required_samples = (
            config.overfit.max_steps
            * training.gradient_accumulation_steps
            * training.batch_size
        )
        repeats = math.ceil(required_samples / len(validation_dataset))
        train_dataset: Dataset = _RepeatedDataset(validation_dataset, repeats=repeats)
        epoch_limit = 1
    else:
        step_limit = max_steps
        train_dataset = raw_train_dataset
        validation_dataset = _base_view(raw_validation_dataset)
        configured_epochs = (
            config.training.development_max_epochs
            if stage in {"development", "benchmark"}
            else config.training.formal_max_epochs
        )
        epoch_limit = configured_epochs if max_epochs is None else max_epochs
    stream = progress_stream or sys.stdout
    if stage == "benchmark":
        if resume not in {"auto", "none"}:
            raise ValueError("benchmark stage does not load training checkpoints")
        return _run_benchmark(
            config=config,
            schema=schema,
            training=training,
            learning_rate=learning_rate,
            train_dataset=train_dataset,
            output_dir=output_dir,
            max_steps=max_steps,
            progress_stream=stream,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    run_manifest_path = output_dir / "run_manifest.json"
    fingerprints = _fingerprints(
        config=config,
        schema=schema,
        stage=stage,
        training=training,
        learning_rate=learning_rate,
        train_cache=train_cache,
        validation_cache=validation_cache,
    )
    run_manifest = {
        "stage": stage,
        "train_partition": train_partition,
        "validation_partition": validation_partition,
        "validation_seed": _validation_seed(training),
        "fingerprints": fingerprints,
        "model": model_kwargs_from_config(config, schema),
        "training": {
            "batch_size": training.batch_size,
            "num_workers": training.num_workers,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "amp": training.amp,
            "prefetch_factor": training.prefetch_factor,
            "learning_rate": learning_rate,
            "device": str(device),
            "validation_every_epochs": training.validation_every_epochs,
        },
    }
    resume_path = _resume_path(resume, latest_path, root)
    if resume == "none":
        for path in (
            latest_path,
            best_path,
            metrics_path,
            summary_path,
            run_manifest_path,
            output_dir / "evaluation.json",
        ):
            path.unlink(missing_ok=True)
        shutil.rmtree(output_dir / "diagnostics", ignore_errors=True)
    elif run_manifest_path.exists():
        stored = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if stored.get("fingerprints") != fingerprints:
            raise ValueError("existing run_manifest fingerprint mismatch")
    _atomic_json(run_manifest_path, run_manifest)
    if resume_path is None:
        _atomic_write(metrics_path, b"")

    _seed_everything(training.seed)
    model = build_model_from_config(config, schema).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training.weight_decay,
    )
    scaler = _make_scaler(device, training.amp)
    training_generator = torch.Generator(device=device).manual_seed(training.seed)
    progress = TrainingProgress(0, 0, 0, None, None)
    epochs_without_improvement = 0
    timing = {
        "training_seconds": 0.0,
        "validation_seconds": 0.0,
        "checkpoint_seconds": 0.0,
    }
    if resume_path is not None:
        checkpoint_started = time.perf_counter()
        progress, extra = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            expected_fingerprints=fingerprints,
            map_location=device,
        )
        checkpoint_elapsed = time.perf_counter() - checkpoint_started
        if extra.get("stage") != stage or "training_generator_state" not in extra:
            raise ValueError("checkpoint is missing stage or training Generator state")
        training_generator.set_state(extra["training_generator_state"].detach().cpu())
        epochs_without_improvement = int(extra.get("epochs_without_improvement", 0))
        stored_timing = extra.get("timing", {})
        if not isinstance(stored_timing, Mapping):
            raise ValueError("checkpoint timing must be a mapping")
        for key in timing:
            value = stored_timing.get(key, 0.0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"checkpoint {key} must be finite and nonnegative")
            timing[key] = float(value)
        timing["checkpoint_seconds"] += checkpoint_elapsed
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampler = ShardShuffleSampler(train_dataset, seed=training.seed)
    train_loader = _loader(
        train_dataset,
        training=training,
        sampler=sampler,
        generator=torch.Generator().manual_seed(training.seed + 1),
    )
    validation_loader = _loader(
        validation_dataset,
        training=training,
        sampler=None,
        generator=torch.Generator().manual_seed(training.seed + 2),
    )
    total_batches = len(train_loader)
    if total_batches == 0 or len(validation_loader) == 0:
        raise ValueError("training and validation caches must both contain samples")
    planned_batches = epoch_limit * total_batches
    if step_limit is not None:
        planned_batches = min(
            planned_batches,
            step_limit * training.gradient_accumulation_steps,
        )
    started = time.perf_counter()
    stop_reason = "max_epochs"
    epoch_records = 0
    metrics_records = 0
    evidence = _read_metric_evidence(metrics_path) if resume_path is not None else {
        "initial_evaluation": None,
        "initial_validation_loss": None,
        "final_evaluation": None,
        "final_validation_loss": None,
    }

    def run_validation(global_step: int) -> tuple[EvaluationResult, float]:
        validation_started = time.perf_counter()
        evaluation = evaluate(
            model,
            validation_loader,
            device=device,
            prior_samples=training.prior_samples,
            sample_period_s=config.tensorization.sample_period_s,
            evaluation_seed=_validation_seed(training),
            amp=training.amp,
            loss_config=config.loss,
            global_step=global_step,
        )
        elapsed = time.perf_counter() - validation_started
        timing["validation_seconds"] += elapsed
        return evaluation, elapsed

    def save_run_checkpoint(
        path: Path,
        checkpoint_progress: TrainingProgress,
    ) -> None:
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            progress=checkpoint_progress,
            fingerprints=fingerprints,
            extra=_checkpoint_extra(
                training_generator,
                stage=stage,
                epochs_without_improvement=epochs_without_improvement,
                timing=timing,
            ),
        )
        timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started

    if resume == "none" and stage == "overfit":
        initial_evaluation, initial_elapsed = run_validation(progress.global_step)
        evidence["initial_evaluation"] = evaluation_to_dict(
            initial_evaluation,
            training.prior_samples,
        )
        evidence["initial_validation_loss"] = validation_loss_to_dict(
            initial_evaluation,
            config.loss,
        )
        _append_jsonl(
            metrics_path,
            {
                "kind": "initial_evaluation",
                "stage": stage,
                "global_step": progress.global_step,
                "validation": evidence["initial_evaluation"],
                "validation_loss": evidence["initial_validation_loss"],
                "elapsed_seconds": initial_elapsed,
            },
        )
        metrics_records += 1

    while progress.epoch < epoch_limit:
        if step_limit is not None and progress.global_step >= step_limit:
            stop_reason = "max_steps"
            break
        epoch = progress.epoch
        sampler.set_epoch(epoch)
        start_batch = progress.next_batch_index
        if start_batch >= total_batches:
            progress = replace(progress, epoch=epoch + 1, next_batch_index=0)
            continue
        remaining_batches = total_batches - start_batch
        if step_limit is not None:
            remaining_steps = step_limit - progress.global_step
            remaining_batches = min(
                remaining_batches,
                remaining_steps * training.gradient_accumulation_steps,
            )
        end_batch = start_batch + remaining_batches
        sampler.set_range(
            start_batch * training.batch_size,
            min(end_batch * training.batch_size, len(train_dataset)),
        )
        epoch_started = time.perf_counter()
        training_segment_started = epoch_started
        batches = iter(train_loader)
        processed_samples = 0
        last_progress_print = 0.0

        def on_optimizer_step(result, microbatch_count: int) -> None:
            nonlocal last_progress_print, processed_samples, training_segment_started
            now = time.perf_counter()
            timing["training_seconds"] += now - training_segment_started
            training_segment_started = now
            processed_samples += result.sums.sample_count
            next_batch = start_batch + microbatch_count
            elapsed = max(time.perf_counter() - epoch_started, 1e-9)
            reached_current_limit = (
                step_limit is not None and result.next_global_step >= step_limit
            )
            if (
                now - last_progress_print >= 5.0
                or next_batch >= total_batches
                or reached_current_limit
            ):
                print(
                    "\r"
                    + _progress_line(
                        stage=stage,
                        epoch=epoch,
                        next_batch=next_batch,
                        total_batches=total_batches,
                        completed_batches=epoch * total_batches + next_batch,
                        planned_batches=planned_batches,
                        processed_batches=microbatch_count,
                        global_step=result.next_global_step,
                        loss=result.total_loss,
                        processed_samples=processed_samples,
                        elapsed=elapsed,
                    ),
                    end="",
                    flush=True,
                    file=stream,
                )
                last_progress_print = now
            if (
                result.next_global_step % training.checkpoint_every_steps == 0
                and next_batch < total_batches
            ):
                save_run_checkpoint(
                    latest_path,
                    TrainingProgress(
                        epoch=epoch,
                        next_batch_index=next_batch,
                        global_step=result.next_global_step,
                        best_metric=progress.best_metric,
                        best_epoch=progress.best_epoch,
                    ),
                )
                training_segment_started = time.perf_counter()

        epoch_result = train_epoch(
            model,
            optimizer,
            batches,
            device=device,
            loss_config=config.loss,
            training_config=training,
            global_step=progress.global_step,
            generator=training_generator,
            scaler=scaler,
            on_optimizer_step=on_optimizer_step,
        )
        timing["training_seconds"] += time.perf_counter() - training_segment_started
        print(file=stream)
        next_batch = start_batch + epoch_result.microbatch_count
        completed_epoch = next_batch >= total_batches
        reached_step_limit = (
            step_limit is not None and epoch_result.next_global_step >= step_limit
        )
        reached_epoch_limit = completed_epoch and epoch + 1 >= epoch_limit
        should_validate = (
            reached_step_limit
            if stage == "overfit"
            else completed_epoch
            and (
                (epoch + 1) % training.validation_every_epochs == 0
                or reached_epoch_limit
            )
        )
        evaluation_dict: dict[str, Any] | None = None
        validation_loss_dict: dict[str, Any] | None = None
        improved = False
        best_metric = progress.best_metric
        best_epoch = progress.best_epoch
        if should_validate:
            evaluation, _ = run_validation(epoch_result.next_global_step)
            evaluation_dict = evaluation_to_dict(
                evaluation,
                training.prior_samples,
            )
            validation_loss_dict = validation_loss_to_dict(evaluation, config.loss)
            evidence["final_evaluation"] = evaluation_dict
            evidence["final_validation_loss"] = validation_loss_dict
            metric = evaluation.prior.fde
            improved = best_metric is None or metric < best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
            elif completed_epoch:
                epochs_without_improvement += 1
        next_progress = TrainingProgress(
            epoch=epoch + 1 if completed_epoch else epoch,
            next_batch_index=0 if completed_epoch else next_batch,
            global_step=epoch_result.next_global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        record = {
            "kind": "epoch",
            "stage": stage,
            "epoch": epoch,
            "completed_epoch": completed_epoch,
            "global_step": epoch_result.next_global_step,
            "next_batch_index": next_progress.next_batch_index,
            "train": {
                "mean_optimizer_loss": epoch_result.mean_optimizer_loss,
                "reconstruction_loss": epoch_result.sums.reconstruction_loss,
                "endpoint_loss": epoch_result.sums.endpoint_loss,
                "kl_loss": epoch_result.sums.kl_loss,
                "optimizer_steps": epoch_result.optimizer_steps,
                "microbatches": epoch_result.microbatch_count,
            },
            "validation": evaluation_dict,
            "validation_loss": validation_loss_dict,
            "best_prior_min_fde_6": best_metric,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        _append_jsonl(metrics_path, record)
        epoch_records += 1
        metrics_records += 1
        if improved:
            save_run_checkpoint(
                best_path,
                next_progress,
            )
        save_run_checkpoint(
            latest_path,
            next_progress,
        )
        progress = next_progress
        if step_limit is not None and progress.global_step >= step_limit:
            stop_reason = "max_steps"
            break
        if (
            stage != "overfit"
            and should_validate
            and training.early_stopping_patience > 0
            and epochs_without_improvement >= training.early_stopping_patience
        ):
            stop_reason = "early_stopping"
            break

    elapsed_seconds = time.perf_counter() - started
    peak_vram_mib = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )
    summary = {
        "stage": stage,
        "status": "complete",
        "stop_reason": stop_reason,
        "progress": {
            "epoch": progress.epoch,
            "next_batch_index": progress.next_batch_index,
            "global_step": progress.global_step,
            "best_metric": progress.best_metric,
            "best_epoch": progress.best_epoch,
        },
        "epoch_records_written": epoch_records,
        "metrics_records_written": metrics_records,
        "initial_evaluation": evidence["initial_evaluation"],
        "initial_validation_loss": evidence["initial_validation_loss"],
        "final_evaluation": evidence["final_evaluation"],
        "final_validation_loss": evidence["final_validation_loss"],
        "peak_vram_mib": peak_vram_mib,
        "timing": {
            **timing,
            "elapsed_seconds_this_invocation": elapsed_seconds,
        },
        "elapsed_seconds": elapsed_seconds,
        "outputs": {
            "latest": str(latest_path),
            "best": str(best_path),
            "metrics": str(metrics_path),
            "run_manifest": str(run_manifest_path),
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or benchmark the conditional CVAE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CVAE_CONFIG)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--amp", choices=("on", "off"))
    parser.add_argument("--prefetch-factor", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_training(
        config_path=args.config,
        stage=args.stage,
        resume=args.resume,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_root=args.cache_root,
        amp=(None if args.amp is None else args.amp == "on"),
        prefetch_factor=args.prefetch_factor,
    )
    print(
        f"CVAE {args.stage} complete: "
        f"step={summary.get('progress', {}).get('global_step', 'benchmark')}",
    )


if __name__ == "__main__":
    main()

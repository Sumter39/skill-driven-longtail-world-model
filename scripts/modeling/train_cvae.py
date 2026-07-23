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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, TextIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from skilldrive.data import (
    CVAECachedDataset,
    ObservedSkillBalanceSampler,
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
from skilldrive.training.config import (
    CVAEConfig,
    GenerationRepairConfig,
    LossConfig,
    TrainingConfig,
)
from skilldrive.training.trainer import (
    BenchmarkResult,
    EvaluationResult,
    benchmark_training,
    evaluate,
    train_epoch,
)


REPAIR_STAGES = ("repair-overfit", "repair-benchmark", "repair-formal")
STAGES = ("overfit", "development", "benchmark", "formal", *REPAIR_STAGES)
VALIDATION_SEED_OFFSET = 100_000
BENCHMARK_SAMPLER_STRATEGY = "shard_shuffle_epoch_cycle_v1"
REPAIR_BENCHMARK_MEASURED_SAMPLES = 53_760
REPAIR_BENCHMARK_WARMUP_SEED_OFFSET = 1_000_000
REPAIR_BENCHMARK_MEASUREMENT_CONTRACT = "repair_fixed_measurement_v1"
REPAIR_SOURCE_PATHS = {
    "train_cvae": Path("scripts/modeling/train_cvae.py"),
    "trainer": Path("skilldrive/training/trainer.py"),
    "conditional_cvae": Path("skilldrive/models/conditional_cvae.py"),
    "motion_losses": Path("skilldrive/training/motion_losses.py"),
    "condition_ranking": Path("skilldrive/training/condition_ranking.py"),
    "cvae_cache": Path("skilldrive/data/cvae_cache.py"),
}
REPAIR_FORMAL_SELECTION_CONTRACT = {
    "epoch_candidates": "one_checkpoint_per_repair_dev_validation_v1",
    "provisional_best": "repair_dev_min_fde_6_only",
    "fde_early_stopping": False,
    "active_checkpoint_gate": "heldout_generation_capability_gate_required",
}
REPAIR_OVERFIT_SAMPLE_COUNT = 64
REPAIR_OVERFIT_BASE_COUNT = 32
REPAIR_OVERFIT_OBSERVED_COUNT = 32
REPAIR_OVERFIT_FOCUS_OBSERVED_COUNT = 16
REPAIR_OVERFIT_OTHER_OBSERVED_COUNT = 16
REPAIR_OVERFIT_OBSERVED_SKILL_COUNT = 13


def _is_repair_stage(stage: str) -> bool:
    return stage in REPAIR_STAGES


def _is_overfit_stage(stage: str) -> bool:
    return stage in {"overfit", "repair-overfit"}


def _is_benchmark_stage(stage: str) -> bool:
    return stage in {"benchmark", "repair-benchmark"}


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


class _DeterministicFullCycleSampler(Sampler[int]):
    """Repeat one deterministic permutation so every full window covers the set."""

    def __init__(self, dataset: Dataset, *, seed: int, num_samples: int) -> None:
        if len(dataset) == 0:
            raise ValueError("cycle sampler requires a non-empty dataset")
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise ValueError("cycle sampler num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("cycle sampler num_samples must be a positive integer")
        entries = getattr(dataset, "entries", None)
        if not isinstance(entries, list) or len(entries) != len(dataset):
            raise ValueError("cycle sampler requires indexed dataset entries")
        self.dataset = dataset
        self.seed = int(seed)
        self.num_samples = num_samples
        self.epoch = 0
        self.start_index = 0
        self.stop_index: int | None = None

    def _permutation(self) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def _full_stream(self):
        permutation = self._permutation()
        for position in range(self.num_samples):
            yield permutation[position % len(permutation)]

    @property
    def contract(self) -> dict[str, Any]:
        base = 0
        observed: dict[str, int] = {}
        for entry in self.dataset.entries:
            spec = entry["spec"]
            if spec["skill_supervision_mask"]:
                skill_id = spec["skill_id"]
                observed[skill_id] = observed.get(skill_id, 0) + 1
            else:
                base += 1
        return {
            "strategy": "deterministic_full_cycle_v1",
            "seed": self.seed,
            "cycle_samples": len(self.dataset),
            "stream_samples": self.num_samples,
            "base_per_cycle": base,
            "observed_per_cycle_by_skill": dict(sorted(observed.items())),
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
        self.start_index = min(start_index, self.num_samples)
        self.stop_index = None if stop_index is None else min(stop_index, self.num_samples)

    def exposure(self) -> dict[str, Any]:
        stop = self.num_samples if self.stop_index is None else self.stop_index
        indices = itertools.islice(self._full_stream(), self.start_index, stop)
        base = 0
        observed: dict[str, int] = {}
        samples = 0
        for index in indices:
            samples += 1
            spec = self.dataset.entries[index]["spec"]
            if spec["skill_supervision_mask"]:
                skill_id = spec["skill_id"]
                observed[skill_id] = observed.get(skill_id, 0) + 1
            else:
                base += 1
        return {
            "epoch": self.epoch,
            "range_start": self.start_index,
            "range_stop": stop,
            "samples": samples,
            "base": base,
            "observed_by_skill": dict(sorted(observed.items())),
        }

    def __iter__(self):
        stop = self.num_samples if self.stop_index is None else self.stop_index
        yield from itertools.islice(self._full_stream(), self.start_index, stop)

    def __len__(self) -> int:
        stop = self.num_samples if self.stop_index is None else self.stop_index
        return max(stop - self.start_index, 0)


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
        self.strategy = BENCHMARK_SAMPLER_STRATEGY

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


class _DeterministicRepairBenchmarkSampler(Sampler[int]):
    """Cycle complete deterministic balanced epochs for repair benchmarks."""

    def __init__(
        self,
        dataset: CVAECachedDataset,
        *,
        seed: int,
        num_samples: int,
        max_repeats_per_sample: int,
    ) -> None:
        if isinstance(num_samples, bool) or not isinstance(num_samples, int):
            raise ValueError("benchmark sampler num_samples must be a positive integer")
        if num_samples <= 0:
            raise ValueError("benchmark sampler num_samples must be a positive integer")
        self.dataset = dataset
        self.seed = int(seed)
        self.num_samples = num_samples
        self.max_repeats_per_sample = max_repeats_per_sample
        self.strategy = "observed_skill_balance_epoch_cycle_v1"
        self.epoch_contract = ObservedSkillBalanceSampler(
            self.dataset,
            seed=self.seed,
            max_repeats_per_sample=self.max_repeats_per_sample,
        ).contract

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        sampler = ObservedSkillBalanceSampler(
            self.dataset,
            seed=self.seed,
            max_repeats_per_sample=self.max_repeats_per_sample,
        )
        remaining = self.num_samples
        epoch = 0
        while remaining > 0:
            sampler.set_epoch(epoch)
            sampler.set_range()
            take = min(remaining, sampler.epoch_size)
            yield from itertools.islice(iter(sampler), take)
            remaining -= take
            epoch += 1


class _ConcatenatedBenchmarkSampler(Sampler[int]):
    """Yield an independent warmup stream before one canonical measured stream."""

    def __init__(self, warmup: Sampler[int] | None, measured: Sampler[int]) -> None:
        self.warmup = warmup
        self.measured = measured

    def __len__(self) -> int:
        warmup_samples = 0 if self.warmup is None else len(self.warmup)
        return warmup_samples + len(self.measured)

    def __iter__(self):
        if self.warmup is not None:
            yield from self.warmup
        yield from self.measured


@dataclass(frozen=True)
class _BenchmarkSamplingPlan:
    loader_sampler: Sampler[int]
    measured_steps: int
    measured_samples: int
    warmup_samples: int
    sampler_metadata: Mapping[str, Any]
    warmup_sampler_metadata: Mapping[str, Any] | None
    measurement_sample_contract: Mapping[str, Any] | None


def _benchmark_sample_stream_metadata(
    sampler: _DeterministicBenchmarkSampler | _DeterministicRepairBenchmarkSampler,
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
        "strategy": sampler.strategy,
        "seed": sampler.seed,
        "dataset_samples": len(sampler.dataset),
        "stream_samples": len(sampler),
        "warmup_samples": warmup_samples,
        "measured_samples": len(sampler) - warmup_samples,
    }
    if isinstance(sampler, _DeterministicRepairBenchmarkSampler):
        contract["max_repeats_per_sample"] = sampler.max_repeats_per_sample
        contract["balanced_epoch_contract"] = sampler.epoch_contract
    return {
        **contract,
        "contract_sha256": _hash_value(contract),
        "measured_order_sha256": measured_digest.hexdigest(),
    }


def _benchmark_sampling_plan(
    *,
    config: CVAEConfig,
    training: TrainingConfig,
    train_dataset: CVAECachedDataset,
    max_steps: int | None,
) -> _BenchmarkSamplingPlan:
    samples_per_step = (
        training.batch_size * training.gradient_accumulation_steps
    )
    warmup_samples = config.benchmark.warmup_steps * samples_per_step
    if config.repair is None:
        measured_steps = (
            config.benchmark.measured_steps if max_steps is None else max_steps
        )
        stream_samples = warmup_samples + measured_steps * samples_per_step
        sampler = _DeterministicBenchmarkSampler(
            train_dataset,
            seed=training.seed,
            num_samples=stream_samples,
        )
        metadata = _benchmark_sample_stream_metadata(
            sampler,
            warmup_samples=warmup_samples,
        )
        return _BenchmarkSamplingPlan(
            loader_sampler=sampler,
            measured_steps=measured_steps,
            measured_samples=metadata["measured_samples"],
            warmup_samples=warmup_samples,
            sampler_metadata=metadata,
            warmup_sampler_metadata=None,
            measurement_sample_contract=None,
        )

    if REPAIR_BENCHMARK_MEASURED_SAMPLES % samples_per_step:
        raise ValueError(
            "repair benchmark fixed measured sample count "
            f"{REPAIR_BENCHMARK_MEASURED_SAMPLES} must be divisible by "
            f"batch_size * gradient_accumulation_steps ({samples_per_step})"
        )
    measured_steps = REPAIR_BENCHMARK_MEASURED_SAMPLES // samples_per_step
    if max_steps is not None and max_steps != measured_steps:
        raise ValueError(
            "repair benchmark uses a fixed 53,760-sample measurement; "
            f"max_steps must be omitted or equal {measured_steps}"
        )
    max_repeats = config.repair.sampler.max_repeats_per_sample
    measured_sampler = _DeterministicRepairBenchmarkSampler(
        train_dataset,
        seed=training.seed,
        num_samples=REPAIR_BENCHMARK_MEASURED_SAMPLES,
        max_repeats_per_sample=max_repeats,
    )
    measured_metadata = _benchmark_sample_stream_metadata(
        measured_sampler,
        warmup_samples=0,
    )
    measurement_payload = {
        "contract": REPAIR_BENCHMARK_MEASUREMENT_CONTRACT,
        "sampler": measured_metadata,
    }
    measurement_contract = {
        **measurement_payload,
        "contract_sha256": _hash_value(measurement_payload),
    }
    warmup_sampler: _DeterministicRepairBenchmarkSampler | None = None
    warmup_metadata: Mapping[str, Any] | None = None
    if warmup_samples:
        warmup_sampler = _DeterministicRepairBenchmarkSampler(
            train_dataset,
            seed=training.seed + REPAIR_BENCHMARK_WARMUP_SEED_OFFSET,
            num_samples=warmup_samples,
            max_repeats_per_sample=max_repeats,
        )
        warmup_metadata = _benchmark_sample_stream_metadata(
            warmup_sampler,
            warmup_samples=0,
        )
    return _BenchmarkSamplingPlan(
        loader_sampler=_ConcatenatedBenchmarkSampler(
            warmup_sampler,
            measured_sampler,
        ),
        measured_steps=measured_steps,
        measured_samples=REPAIR_BENCHMARK_MEASURED_SAMPLES,
        warmup_samples=warmup_samples,
        sampler_metadata=measured_metadata,
        warmup_sampler_metadata=warmup_metadata,
        measurement_sample_contract=measurement_contract,
    )


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


def _repair_source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        name: _sha256(root / relative_path)
        for name, relative_path in REPAIR_SOURCE_PATHS.items()
    }


def _repair_overfit_identity(
    dataset: Dataset,
    sampler: _DeterministicFullCycleSampler,
    *,
    expected_sample_count: int,
    focus_skill_id: str,
) -> dict[str, Any]:
    """Describe the exact fixed repair-overfit cohort and repeated sample stream."""

    entries = getattr(dataset, "entries", None)
    if not isinstance(entries, list) or len(entries) != expected_sample_count:
        raise ValueError(
            "repair overfit materialization must contain the configured sample count"
        )
    sample_ids: list[str] = []
    base_count = 0
    observed_by_skill: dict[str, int] = {}
    for entry in entries:
        sample_id = entry.get("sample_id") if isinstance(entry, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("repair overfit samples require non-empty sample_id values")
        sample_ids.append(sample_id)
        spec = entry.get("spec")
        if not isinstance(spec, Mapping):
            raise ValueError("repair overfit samples require a spec mapping")
        if spec.get("skill_supervision_mask"):
            skill_id = spec.get("skill_id")
            if not isinstance(skill_id, str) or skill_id == "<none>":
                raise ValueError("repair overfit observed sample has an invalid skill_id")
            observed_by_skill[skill_id] = observed_by_skill.get(skill_id, 0) + 1
        else:
            base_count += 1
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("repair overfit sample_id values must be unique")
    if sampler.dataset is not dataset:
        raise ValueError("repair overfit sampler must bind the materialized cohort")
    if sampler.epoch != 0:
        raise ValueError("repair overfit stream identity must be frozen at epoch zero")
    if (
        base_count != REPAIR_OVERFIT_BASE_COUNT
        or sum(observed_by_skill.values()) != REPAIR_OVERFIT_OBSERVED_COUNT
        or len(observed_by_skill) != REPAIR_OVERFIT_OBSERVED_SKILL_COUNT
        or observed_by_skill.get(focus_skill_id)
        != REPAIR_OVERFIT_FOCUS_OBSERVED_COUNT
        or sum(
            count
            for skill_id, count in observed_by_skill.items()
            if skill_id != focus_skill_id
        )
        != REPAIR_OVERFIT_OTHER_OBSERVED_COUNT
    ):
        raise ValueError(
            "repair overfit identity requires 32 base, 16 focus, and 16 other "
            "observed samples covering all 13 training skills"
        )
    permutation = sampler._permutation()
    cycle_sample_ids = [sample_ids[index] for index in permutation]
    stream_contract = {
        "strategy": sampler.contract["strategy"],
        "cycle_order_sample_ids_sha256": _hash_value(cycle_sample_ids),
        "stream_samples": sampler.num_samples,
        "repeat_rule": "repeat_epoch_zero_cycle_then_truncate_v1",
    }
    return {
        "contract": "repair_overfit_fixed_stream_v2",
        "selected_sample_count": len(sample_ids),
        "base_sample_count": base_count,
        "observed_sample_count": sum(observed_by_skill.values()),
        "focus_skill_id": focus_skill_id,
        "focus_observed_sample_count": observed_by_skill[focus_skill_id],
        "other_observed_sample_count": sum(
            count
            for skill_id, count in observed_by_skill.items()
            if skill_id != focus_skill_id
        ),
        "observed_by_skill": dict(sorted(observed_by_skill.items())),
        "selected_sample_ids": sample_ids,
        "selected_sample_ids_sha256": _hash_value(sample_ids),
        "cycle_order_sample_ids_sha256": stream_contract[
            "cycle_order_sample_ids_sha256"
        ],
        "stream_samples": sampler.num_samples,
        "stream_identity_sha256": _hash_value(stream_contract),
    }


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


def _repair_view_datasets(
    config: CVAEConfig,
    *,
    root: Path,
    source_cache: Path,
    schema: CVAESchema,
) -> tuple[CVAECachedDataset, CVAECachedDataset, dict[str, Any]]:
    """Load audited, disjoint Formal Train views without opening Validation."""

    repair = config.repair
    if repair is None:
        raise ValueError("repair stage requires repair configuration")
    audit_path = root / repair.split.audit
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict) or audit.get("status") != "complete":
        raise ValueError("repair split audit must be a complete JSON object")
    if audit.get("validation_manifests_opened") is not False:
        raise ValueError("repair split audit must prove Validation was not opened")
    integrity = audit.get("integrity")
    required_integrity = {
        "scenario_overlap": 0,
        "sample_offset_overlap": 0,
        "sample_offset_union_matches_v5_cache": True,
    }
    if not isinstance(integrity, Mapping) or any(
        integrity.get(name) != expected
        for name, expected in required_integrity.items()
    ):
        raise ValueError("repair split audit integrity gates are not satisfied")
    sources = audit.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("repair split audit sources are missing")
    source_paths = {
        "formal_train_v5_cache_manifest": source_cache / "cache_manifest.json",
        "formal_train_v5_sample_index": source_cache / "sample_index.jsonl",
    }
    for name, path in source_paths.items():
        descriptor = sources.get(name)
        if not isinstance(descriptor, Mapping) or descriptor.get("sha256") != _sha256(
            path
        ):
            raise ValueError(f"repair split audit source differs for {name}")

    train_index = root / repair.split.train_sample_index
    development_index = root / repair.split.development_sample_index
    outputs = audit.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("repair split audit outputs are missing")
    for name, path in (
        ("repair_train_sample_index", train_index),
        ("repair_dev_sample_index", development_index),
    ):
        descriptor = outputs.get(name)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"repair split audit is missing {name}")
        if descriptor.get("sha256") != _sha256(path):
            raise ValueError(f"repair split audit hash differs for {name}")
        if descriptor.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"repair split audit size differs for {name}")

    train = CVAECachedDataset(
        source_cache,
        schema=schema,
        sample_index_path=train_index,
    )
    development = CVAECachedDataset(
        source_cache,
        schema=schema,
        sample_index_path=development_index,
    )
    train_offsets = {(entry["shard"], entry["offset"]) for entry in train.entries}
    development_offsets = {
        (entry["shard"], entry["offset"]) for entry in development.entries
    }
    source_offsets = {
        (entry["shard"], entry["offset"])
        for entry in (
            json.loads(line)
            for line in train.source_sample_index_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line
        )
    }
    if train_offsets & development_offsets:
        raise ValueError("repair train and development views overlap")
    if train_offsets | development_offsets != source_offsets:
        raise ValueError("repair views do not form the complete Formal Train cache")
    train_scenarios = {entry["scenario_id"] for entry in train.entries}
    development_scenarios = {
        entry["scenario_id"] for entry in development.entries
    }
    if train_scenarios & development_scenarios:
        raise ValueError("repair train and development scenarios overlap")
    counts = audit.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("repair split audit counts are missing")
    if counts.get("repair_train_samples") != len(train):
        raise ValueError("repair train view count differs from audit")
    if counts.get("repair_dev_samples") != len(development):
        raise ValueError("repair development view count differs from audit")
    return train, development, audit


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


def _ensure_immutable_run_manifest(
    path: Path,
    expected: Mapping[str, Any],
    *,
    resuming: bool,
) -> None:
    """Create one run contract, or require exact equality before resuming it."""

    expected_value = dict(expected)
    if not path.is_file():
        if resuming:
            raise ValueError("cannot resume without the immutable run_manifest")
        _atomic_json(path, expected_value)
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read immutable run_manifest: {error}") from error
    if stored != expected_value:
        stored_mapping = stored if isinstance(stored, Mapping) else {}
        differing = sorted(
            key
            for key in set(stored_mapping) | set(expected_value)
            if stored_mapping.get(key) != expected_value.get(key)
        )
        raise ValueError(
            "immutable run_manifest mismatch: " + ", ".join(differing)
        )


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
    kwargs = {
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
    if config.repair is not None:
        kwargs.update(
            decoder_initial_delta_mode=(
                config.repair.model.decoder_initial_delta_mode
            ),
            sample_period_s=config.tensorization.sample_period_s,
        )
    return kwargs


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
    repair_config: GenerationRepairConfig | None = None,
) -> dict[str, Any]:
    loss = result.validation_loss
    if loss is None:
        raise ValueError("evaluation did not compute validation loss")
    value = {
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
    if repair_config is not None:
        value.update(
            seam_velocity_loss=loss.seam_velocity_loss,
            seam_velocity_weight=repair_config.motion_loss.seam_velocity_weight,
            velocity_loss=loss.velocity_loss,
            velocity_weight=repair_config.motion_loss.velocity_weight,
            acceleration_loss=loss.acceleration_loss,
            acceleration_weight=repair_config.motion_loss.acceleration_weight,
            jerk_loss=loss.jerk_loss,
            jerk_weight=repair_config.motion_loss.jerk_weight,
            condition_ranking_loss=loss.condition_ranking_loss,
            condition_ranking_weight=repair_config.condition_ranking.weight,
            condition_ranking_margin_per_latent_dim=(
                repair_config.condition_ranking.margin_per_latent_dim
            ),
            observed_condition_count=loss.sums.observed_condition_count,
            correct_condition_kl=loss.correct_condition_kl,
            none_condition_kl=loss.none_condition_kl,
        )
    return value


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
    elif stage == "formal":
        train_cache = cache_root / "formal_train"
        validation_cache = cache_root / "internal_validation"
    else:
        if config.repair is None:
            raise ValueError("repair stages require a versioned repair configuration")
        train_cache = cache_root / config.repair.source_cache_partition
        validation_cache = train_cache
    if stage in {"overfit", "repair-overfit"}:
        output = root / config.outputs.development / "overfit"
    elif stage == "development":
        output = root / config.outputs.development
    elif stage in {"benchmark", "repair-benchmark"}:
        output = root / config.outputs.benchmarks
    elif stage == "formal":
        output = root / config.outputs.formal
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
    allow_tf32: bool | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
) -> tuple[TrainingConfig, float]:
    training = config.training
    default_batch = (
        config.overfit.batch_size if _is_overfit_stage(stage) else training.batch_size
    )
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
    effective_persistent = (
        training.persistent_workers
        if persistent_workers is None
        else persistent_workers
    )
    if effective_workers == 0:
        if persistent_workers is True:
            raise ValueError("persistent_workers override requires num_workers > 0")
        effective_persistent = False
    training = replace(
        training,
        amp=training.amp if amp is None else amp,
        allow_tf32=training.allow_tf32 if allow_tf32 is None else allow_tf32,
        batch_size=effective_batch,
        num_workers=effective_workers,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
        pin_memory=training.pin_memory if pin_memory is None else pin_memory,
    )
    learning_rate = (
        config.overfit.learning_rate
        if _is_overfit_stage(stage)
        else training.learning_rate
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


def _repair_overfit_view(dataset: CVAECachedDataset, config: CVAEConfig) -> _DatasetView:
    """Select the fixed 32-base/32-observed cohort from repair_train."""

    if config.overfit.sample_count != REPAIR_OVERFIT_SAMPLE_COUNT:
        raise ValueError("repair overfit requires exactly 64 samples")
    base_indices: list[int] = []
    observed_groups: dict[str, list[int]] = {}
    for index, entry in enumerate(dataset.entries):
        spec = entry["spec"]
        if spec["skill_supervision_mask"]:
            observed_groups.setdefault(spec["skill_id"], []).append(index)
        else:
            base_indices.append(index)
    if len(base_indices) < REPAIR_OVERFIT_BASE_COUNT:
        raise ValueError("repair overfit requires at least 32 base samples")
    if len(observed_groups) != REPAIR_OVERFIT_OBSERVED_SKILL_COUNT:
        raise ValueError(
            "repair overfit requires observed samples from all 13 training skills"
        )
    focus_skill_id = config.overfit.skill_id
    if focus_skill_id not in observed_groups:
        raise ValueError("repair overfit focus skill has no observed samples")
    other_skill_ids = sorted(set(observed_groups) - {focus_skill_id})
    if len(other_skill_ids) != REPAIR_OVERFIT_OBSERVED_SKILL_COUNT - 1:
        raise ValueError("repair overfit requires exactly 12 non-focus observed skills")
    ordered_base = sorted(
        base_indices,
        key=lambda index: hashlib.sha256(
            (
                f"repair-overfit-v3|{config.training.seed}|<base>|"
                f"{dataset.entries[index]['sample_id']}"
            ).encode("utf-8")
        ).hexdigest(),
    )
    ordered_observed: dict[str, list[int]] = {}
    for label, indices in observed_groups.items():
        ordered_observed[label] = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                (
                    f"repair-overfit-v3|{config.training.seed}|{label}|"
                    f"{dataset.entries[index]['sample_id']}"
                ).encode("utf-8")
            ).hexdigest(),
        )
    focus_indices = ordered_observed[focus_skill_id]
    if len(focus_indices) < REPAIR_OVERFIT_FOCUS_OBSERVED_COUNT:
        raise ValueError("repair overfit focus skill requires at least 16 samples")
    selected_other: list[int] = []
    positions = {label: 0 for label in other_skill_ids}
    while len(selected_other) < REPAIR_OVERFIT_OTHER_OBSERVED_COUNT:
        progressed = False
        for label in other_skill_ids:
            position = positions[label]
            if position >= len(ordered_observed[label]):
                continue
            selected_other.append(ordered_observed[label][position])
            positions[label] = position + 1
            progressed = True
            if len(selected_other) == REPAIR_OVERFIT_OTHER_OBSERVED_COUNT:
                break
        if not progressed:
            raise ValueError("repair overfit non-focus skills contain fewer than 16 samples")
    selected_observed = (
        focus_indices[:REPAIR_OVERFIT_FOCUS_OBSERVED_COUNT] + selected_other
    )
    selected = ordered_base[:REPAIR_OVERFIT_BASE_COUNT] + selected_observed
    if len(set(selected)) != REPAIR_OVERFIT_SAMPLE_COUNT:
        raise AssertionError("repair overfit cohort contains duplicate source samples")
    return _DatasetView(dataset, selected)


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
    train_sample_index: Path | None = None,
    validation_sample_index: Path | None = None,
    sampler_contract: Mapping[str, Any] | None = None,
    overfit_identity: Mapping[str, Any] | None = None,
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
    if config.repair is not None:
        repair_contract = config.to_canonical_dict()["repair"]
        optimizer_contract["repair"] = repair_contract
        pipeline_contract["repair"] = repair_contract
        schedule_contract["repair_sampler"] = repair_contract["sampler"]
    result = {
        "config": _hash_value(pipeline_contract),
        "model": _hash_value(model_kwargs),
        "optimizer": _hash_value(optimizer_contract),
        "schedule": _hash_value(schedule_contract),
        "stage": _hash_value(stage),
        "train_cache": _cache_fingerprint(train_cache),
        "validation_cache": _cache_fingerprint(validation_cache),
    }
    if train_sample_index is not None or validation_sample_index is not None:
        if train_sample_index is None or validation_sample_index is None:
            raise ValueError("both repair sample-index fingerprints are required")
        result.update(
            train_sample_index=_sha256(train_sample_index),
            validation_sample_index=_sha256(validation_sample_index),
        )
    if config.repair is not None:
        if sampler_contract is None:
            raise ValueError("repair fingerprints require the complete sampler contract")
        result["repair_sampler_contract"] = _hash_value(dict(sampler_contract))
        for name, digest in _repair_source_fingerprints().items():
            result[f"repair_source_{name}"] = digest
        if stage == "repair-overfit":
            if overfit_identity is None:
                raise ValueError(
                    "repair-overfit fingerprints require the fixed cohort and stream identity"
                )
            selected_sample_ids = overfit_identity.get("selected_sample_ids")
            if (
                not isinstance(selected_sample_ids, list)
                or len(selected_sample_ids) != REPAIR_OVERFIT_SAMPLE_COUNT
                or any(not isinstance(value, str) for value in selected_sample_ids)
            ):
                raise ValueError("repair-overfit cohort identity is invalid")
            result["repair_overfit_samples"] = _hash_value(selected_sample_ids)
            stream_identity = overfit_identity.get("stream_identity_sha256")
            if not isinstance(stream_identity, str) or len(stream_identity) != 64:
                raise ValueError("repair-overfit stream identity is invalid")
            result["repair_overfit_stream"] = stream_identity
        elif overfit_identity is not None:
            raise ValueError("overfit identity is only valid for repair-overfit")
    return result


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
    repair_contract: str | None = None,
    run_manifest_sha256: str | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = {
        "stage": stage,
        "epochs_without_improvement": epochs_without_improvement,
        "training_generator_state": generator.get_state(),
        "timing": dict(timing),
    }
    if repair_contract is not None:
        value["repair_contract"] = repair_contract
    if run_manifest_sha256 is not None:
        value["run_manifest_sha256"] = run_manifest_sha256
    if checkpoint_metadata is not None:
        value["checkpoint"] = dict(checkpoint_metadata)
    return value


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
    benchmark_repeats: int,
    run_training_started: float,
    progress_stream: TextIO,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_stage = "repair-benchmark" if config.repair is not None else "benchmark"
    sampling = _benchmark_sampling_plan(
        config=config,
        training=training,
        train_dataset=train_dataset,
        max_steps=max_steps,
    )
    measured_steps = sampling.measured_steps
    sampler_metadata = sampling.sampler_metadata
    cache_dir = str(Path(train_dataset.cache_dir).parent.resolve())
    cache_path_id = _hash_value(cache_dir)[:8]
    cache_fingerprint = _cache_fingerprint(Path(train_dataset.cache_dir))
    cache_content_id = cache_fingerprint[:8]
    resolved_device = torch.device(training.device)
    repeat_state_contract = (
        "continuous_model_optimizer_loader_v1"
        if config.repair is not None
        else "fresh_model_optimizer_loader_per_repeat_v1"
    )
    benchmark_contract = {
        "cache_fingerprint": cache_fingerprint,
        "schema_fingerprint": cvae_schema_fingerprint(schema),
        "model": model_kwargs_from_config(config, schema),
        "loss": asdict(config.loss),
        "seed": training.seed,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "sampler_strategy": sampler_metadata["strategy"],
        "dataset_samples": len(train_dataset),
        "warmup_steps": config.benchmark.warmup_steps,
        "repeats": benchmark_repeats,
        "repeat_state_contract": repeat_state_contract,
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
    if config.repair is None:
        benchmark_contract["measured_steps"] = measured_steps
    else:
        if sampling.measurement_sample_contract is None:  # pragma: no cover
            raise AssertionError("repair measurement contract disappeared")
        benchmark_contract.update(
            sample_index_sha256=train_dataset.sample_index_sha256,
            repair=config.to_canonical_dict()["repair"],
            measurement_sample_contract=sampling.measurement_sample_contract,
            source_sha256=_repair_source_fingerprints(),
        )
    benchmark_contract_id = _hash_value(benchmark_contract)
    candidate_contract = {
        "benchmark_contract_id": benchmark_contract_id,
        "repeat_state_contract": repeat_state_contract,
        "batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "num_workers": training.num_workers,
        "prefetch_factor": training.prefetch_factor,
        "persistent_workers": training.persistent_workers,
        "pin_memory": training.pin_memory,
        "amp": training.amp,
        "allow_tf32": training.allow_tf32,
        "warmup_steps": config.benchmark.warmup_steps,
        "warmup_samples": sampling.warmup_samples,
        "warmup_sampler": sampling.warmup_sampler_metadata,
        "measured_steps": measured_steps,
        "measured_samples": sampling.measured_samples,
        "sampler": sampler_metadata,
    }
    candidate_contract_id = _hash_value(candidate_contract)
    measurement_id = (
        sampling.measurement_sample_contract["contract_sha256"][:8]
        if sampling.measurement_sample_contract is not None
        else sampler_metadata["contract_sha256"][:8]
    )
    candidate_id = (
        f"b{training.batch_size}-w{training.num_workers}-"
        f"pf{training.prefetch_factor}-pw{int(training.persistent_workers)}-"
        f"pin{int(training.pin_memory)}-a{int(training.amp)}-"
        f"t{int(training.allow_tf32)}-m{measurement_id}-"
        f"bc{benchmark_contract_id[:8]}-cc{candidate_contract_id[:8]}-"
        f"k{cache_path_id}-{cache_content_id}"
    )
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = candidate_dir / "metrics.jsonl"
    _atomic_write(metrics_path, b"")
    run_training_setup_seconds = time.perf_counter() - run_training_started

    def build_runtime():
        _seed_everything(training.seed)
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(training.device)
        runtime_setup_started = time.perf_counter()
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
            sampler=sampling.loader_sampler,
            generator=torch.Generator().manual_seed(training.seed + 1),
            drop_last=True,
        )
        loader_setup_seconds = time.perf_counter() - loader_setup_started
        generator = torch.Generator(device=resolved_device).manual_seed(training.seed)
        scaler = _make_scaler(resolved_device, training.amp)
        runtime_setup_seconds = time.perf_counter() - runtime_setup_started
        return (
            model,
            optimizer,
            loader,
            generator,
            scaler,
            model_setup_seconds,
            loader_setup_seconds,
            runtime_setup_seconds,
        )

    shared_runtime = build_runtime() if config.repair is not None else None
    shared_setup = None
    if shared_runtime is not None:
        shared_setup = {
            "model_optimizer_seconds": shared_runtime[5],
            "data_loader_seconds": shared_runtime[6],
            "runtime_total_seconds": shared_runtime[7],
            "scope": "one shared runtime before all repeat windows",
        }
    results: list[dict[str, Any]] = []
    next_global_step = 0
    for repeat in range(benchmark_repeats):
        runtime = shared_runtime if shared_runtime is not None else build_runtime()
        (
            model,
            optimizer,
            loader,
            generator,
            scaler,
            model_setup_seconds,
            loader_setup_seconds,
            _,
        ) = runtime
        repeat_global_step = next_global_step if shared_runtime is not None else 0
        result: BenchmarkResult = benchmark_training(
            model,
            optimizer,
            loader,
            device=training.device,
            loss_config=config.loss,
            training_config=training,
            global_step=repeat_global_step,
            generator=generator,
            warmup_steps=config.benchmark.warmup_steps,
            measured_steps=measured_steps,
            scaler=scaler,
            repair_config=config.repair,
            sample_period_s=config.tensorization.sample_period_s,
        )
        if shared_runtime is not None:
            next_global_step = result.next_global_step
        if result.measured_samples != sampler_metadata["measured_samples"]:
            raise RuntimeError(
                "benchmark measured sample count differs from deterministic stream"
            )
        value = {
            "kind": "benchmark",
            "repeat": repeat,
            "repeat_state_contract": repeat_state_contract,
            "global_step_start": repeat_global_step,
            "global_step_end": result.next_global_step,
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
                "candidate_contract_id": candidate_contract_id,
                "candidate_contract": candidate_contract,
                "sampler": sampler_metadata,
                "warmup_sampler": sampling.warmup_sampler_metadata,
                "measurement_sample_contract": sampling.measurement_sample_contract,
            },
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
        if shared_runtime is None:
            value.update(
                model_setup_seconds=model_setup_seconds,
                loader_setup_seconds=loader_setup_seconds,
            )
        results.append(value)
        _append_jsonl(metrics_path, value)
        print(
            f"benchmark repeat {repeat + 1}/{benchmark_repeats}: "
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
        "stage": benchmark_stage,
        "candidate_id": candidate_id,
        "benchmark_contract_id": benchmark_contract_id,
        "benchmark_contract": benchmark_contract,
        "candidate_contract_id": candidate_contract_id,
        "candidate_contract": candidate_contract,
        "repeat_state_contract": repeat_state_contract,
        "shared_setup": shared_setup,
        "run_training_setup_seconds": run_training_setup_seconds,
        "run_training_setup_scope": {
            "starts_at": "run_training entry",
            "ends_before": "first benchmark repeat model setup",
            "excludes": ["Python process startup", "module imports including torch"],
        },
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
    if shared_setup is None:
        summary.update(
            median_model_setup_seconds=median_present("model_setup_seconds"),
            median_loader_setup_seconds=median_present("loader_setup_seconds"),
        )
    _atomic_json(candidate_dir / "summary.json", summary)
    index_path = output_dir / "summary.json"
    index = (
        json.loads(index_path.read_text(encoding="utf-8"))
        if index_path.is_file()
        else {"stage": benchmark_stage}
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
    benchmark_repeats: int | None = None,
    allow_tf32: bool | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    """Run one resumable training stage using already prepared caches."""
    run_training_started = time.perf_counter()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if max_epochs is not None and max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if benchmark_repeats is not None and benchmark_repeats <= 0:
        raise ValueError("benchmark_repeats must be positive")
    root = Path(project_root)
    config = load_cvae_config(config_path)
    if config.repair is not None and not _is_repair_stage(stage):
        raise ValueError(
            "a generation-repair configuration may only run repair stages"
        )
    if _is_repair_stage(stage) and config.repair is None:
        raise ValueError("repair stage requires cvae_generation_repair_v1 config")
    schema = build_cvae_schema(root / config.data.skill_dir)
    training, learning_rate = _effective_training(
        config,
        stage,
        batch_size,
        num_workers,
        amp,
        prefetch_factor,
        allow_tf32,
        pin_memory,
        persistent_workers,
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
    schema_sha256 = cvae_schema_fingerprint(schema)
    candidate_pool_path = root / config.data.formal_candidate_pool
    repair_audit: dict[str, Any] | None = None
    if _is_repair_stage(stage):
        raw_train_dataset, raw_validation_dataset, repair_audit = (
            _repair_view_datasets(
                config,
                root=root,
                source_cache=train_cache,
                schema=schema,
            )
        )
        if stage == "repair-overfit":
            train_partition = "repair_train_overfit_subset"
            validation_partition = "repair_train_overfit_subset"
        elif stage == "repair-benchmark":
            train_partition = "repair_train"
            validation_partition = "not_used"
        else:
            train_partition = "repair_train"
            validation_partition = "repair_dev"
        _validate_cache_contract(
            raw_train_dataset,
            expected_partition="formal_train",
            manifest_path=_manifest_path(config, "formal_train", root),
            schema_sha256=schema_sha256,
            candidate_pool_path=candidate_pool_path,
        )
    else:
        train_partition = "formal_train" if stage == "formal" else "development_train"
        validation_partition = (
            "internal_validation"
            if stage == "formal"
            else (
                "development_train"
                if stage == "overfit"
                else "development_validation"
            )
        )
        raw_train_dataset = CVAECachedDataset(train_cache, schema=schema)
        raw_validation_dataset = (
            raw_train_dataset
            if validation_cache == train_cache
            else CVAECachedDataset(validation_cache, schema=schema)
        )
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

    repair_overfit_stream_samples: int | None = None
    formal_epoch_budget: int | None = None
    if _is_overfit_stage(stage):
        if max_epochs is not None:
            raise ValueError("overfit stage is controlled by max_steps, not max_epochs")
        if max_steps is not None and max_steps > config.overfit.max_steps:
            raise ValueError("overfit max_steps cannot exceed the configured overfit limit")
        step_limit = config.overfit.max_steps if max_steps is None else max_steps
        overfit_view = (
            _repair_overfit_view(raw_train_dataset, config)
            if stage == "repair-overfit"
            else _overfit_view(raw_train_dataset, config)
        )
        validation_dataset = _MaterializedDataset(overfit_view)
        required_samples = (
            config.overfit.max_steps
            * training.gradient_accumulation_steps
            * training.batch_size
        )
        if stage == "repair-overfit":
            train_dataset = validation_dataset
            repair_overfit_stream_samples = required_samples
        else:
            repeats = math.ceil(required_samples / len(validation_dataset))
            train_dataset = _RepeatedDataset(validation_dataset, repeats=repeats)
        epoch_limit = 1
    elif stage == "repair-formal":
        step_limit = max_steps
        train_dataset = raw_train_dataset
        validation_dataset = raw_validation_dataset
        formal_epoch_budget = config.training.formal_max_epochs
        if max_epochs is not None and max_epochs > formal_epoch_budget:
            raise ValueError(
                "repair-formal max_epochs cannot exceed the frozen formal epoch budget"
            )
        epoch_limit = formal_epoch_budget if max_epochs is None else max_epochs
    elif stage == "repair-benchmark":
        if max_epochs is not None:
            raise ValueError("benchmark stage is controlled by max_steps, not max_epochs")
        step_limit = max_steps
        train_dataset = raw_train_dataset
        validation_dataset = raw_validation_dataset
        epoch_limit = config.training.formal_max_epochs
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
    if _is_benchmark_stage(stage):
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
            benchmark_repeats=(
                config.benchmark.repeats
                if benchmark_repeats is None
                else benchmark_repeats
            ),
            run_training_started=run_training_started,
            progress_stream=stream,
        )

    repair_overfit_identity: dict[str, Any] | None = None
    if stage == "repair-formal":
        if config.repair is None:  # pragma: no cover - guarded above
            raise AssertionError("repair config disappeared")
        sampler = ObservedSkillBalanceSampler(
            train_dataset,
            seed=training.seed,
            max_repeats_per_sample=(
                config.repair.sampler.max_repeats_per_sample
            ),
        )
        epoch_sample_count = sampler.epoch_size
        sampler_contract: Mapping[str, Any] = sampler.contract
    elif stage == "repair-overfit":
        if repair_overfit_stream_samples is None:  # pragma: no cover - guarded above
            raise AssertionError("repair overfit stream size disappeared")
        sampler = _DeterministicFullCycleSampler(
            train_dataset,
            seed=training.seed,
            num_samples=repair_overfit_stream_samples,
        )
        epoch_sample_count = repair_overfit_stream_samples
        sampler_contract = sampler.contract
        repair_overfit_identity = _repair_overfit_identity(
            train_dataset,
            sampler,
            expected_sample_count=config.overfit.sample_count,
            focus_skill_id=config.overfit.skill_id,
        )
    else:
        sampler = ShardShuffleSampler(train_dataset, seed=training.seed)
        epoch_sample_count = len(train_dataset)
        sampler_contract = {
            "strategy": "shard_shuffle_v1",
            "seed": training.seed,
            "epoch_samples": epoch_sample_count,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    run_manifest_path = output_dir / "run_manifest.json"
    epoch_candidates_dir = output_dir / "epoch_candidates"
    fingerprints = _fingerprints(
        config=config,
        schema=schema,
        stage=stage,
        training=training,
        learning_rate=learning_rate,
        train_cache=train_cache,
        validation_cache=validation_cache,
        train_sample_index=(
            raw_train_dataset.sample_index_path
            if _is_repair_stage(stage)
            else None
        ),
        validation_sample_index=(
            raw_validation_dataset.sample_index_path
            if _is_repair_stage(stage)
            else None
        ),
        sampler_contract=sampler_contract if config.repair is not None else None,
        overfit_identity=repair_overfit_identity,
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
    if config.repair is not None:
        if repair_audit is None:  # pragma: no cover - guarded by stage setup
            raise AssertionError("repair audit disappeared")
        repair_training_contract = asdict(training)
        repair_training_contract["learning_rate"] = learning_rate
        repair_training_contract["sampler"] = dict(sampler_contract)
        run_manifest.update(
            immutable_contract="repair_run_manifest_v1",
            repair_contract=config.repair.contract,
            repair_split_audit_sha256=_sha256(root / config.repair.split.audit),
            repair_source_sha256={
                name: fingerprints[f"repair_source_{name}"]
                for name in REPAIR_SOURCE_PATHS
            },
            schema_sha256=schema_sha256,
        )
        run_manifest["training"] = repair_training_contract
        if repair_overfit_identity is not None:
            run_manifest["repair_overfit"] = repair_overfit_identity
        if stage == "repair-formal":
            if formal_epoch_budget is None:  # pragma: no cover - guarded above
                raise AssertionError("formal epoch budget disappeared")
            run_manifest["formal_selection"] = {
                **REPAIR_FORMAL_SELECTION_CONTRACT,
                "frozen_epoch_budget": formal_epoch_budget,
                "validation_every_epochs": training.validation_every_epochs,
                "epoch_candidate_directory": str(epoch_candidates_dir),
                "provisional_best_path": str(best_path),
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
        if stage == "repair-formal":
            shutil.rmtree(epoch_candidates_dir, ignore_errors=True)
    if config.repair is not None:
        _ensure_immutable_run_manifest(
            run_manifest_path,
            run_manifest,
            resuming=resume_path is not None,
        )
    else:
        if resume != "none" and run_manifest_path.exists():
            stored = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if stored.get("fingerprints") != fingerprints:
                raise ValueError("existing run_manifest fingerprint mismatch")
        _atomic_json(run_manifest_path, run_manifest)
    run_manifest_sha256 = _hash_value(run_manifest)
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
        if config.repair is not None and extra.get("repair_contract") != config.repair.contract:
            raise ValueError("legacy checkpoint cannot resume the repair contract")
        if (
            config.repair is not None
            and extra.get("run_manifest_sha256") != run_manifest_sha256
        ):
            raise ValueError("checkpoint immutable run_manifest fingerprint mismatch")
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
    stop_reason = (
        "fixed_epoch_budget"
        if stage == "repair-formal" and epoch_limit == formal_epoch_budget
        else (
            "invocation_epoch_limit"
            if stage == "repair-formal"
            else "max_epochs"
        )
    )
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
            repair_config=config.repair,
        )
        elapsed = time.perf_counter() - validation_started
        timing["validation_seconds"] += elapsed
        return evaluation, elapsed

    def save_run_checkpoint(
        path: Path,
        checkpoint_progress: TrainingProgress,
        *,
        role: str,
        candidate_epoch: int | None = None,
    ) -> None:
        checkpoint_metadata: dict[str, Any] | None = None
        if config.repair is not None:
            checkpoint_metadata = {
                "role": role,
                "active_checkpoint": False,
            }
            if candidate_epoch is not None:
                checkpoint_metadata["candidate_epoch"] = candidate_epoch
            if stage == "repair-formal":
                checkpoint_metadata.update(
                    selection_metric="repair_dev.min_fde_6",
                    selection_status=(
                        "provisional_fde_candidate"
                        if role == "provisional_fde_best"
                        else (
                            "unpromoted_epoch_candidate"
                            if role == "epoch_validation_candidate"
                            else "resume_state_not_selectable"
                        )
                    ),
                    active_checkpoint_gate=(
                        REPAIR_FORMAL_SELECTION_CONTRACT["active_checkpoint_gate"]
                    ),
                )
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
                repair_contract=(
                    None if config.repair is None else config.repair.contract
                ),
                run_manifest_sha256=(
                    run_manifest_sha256 if config.repair is not None else None
                ),
                checkpoint_metadata=checkpoint_metadata,
            ),
        )
        timing["checkpoint_seconds"] += time.perf_counter() - checkpoint_started

    if resume == "none" and _is_overfit_stage(stage):
        initial_evaluation, initial_elapsed = run_validation(progress.global_step)
        evidence["initial_evaluation"] = evaluation_to_dict(
            initial_evaluation,
            training.prior_samples,
        )
        evidence["initial_validation_loss"] = validation_loss_to_dict(
            initial_evaluation,
            config.loss,
            config.repair,
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
            min(end_batch * training.batch_size, epoch_sample_count),
        )
        sampler_exposure = (
            sampler.exposure()
            if isinstance(
                sampler,
                (ObservedSkillBalanceSampler, _DeterministicFullCycleSampler),
            )
            else None
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
                    role="latest_resume_state",
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
            repair_config=config.repair,
            sample_period_s=config.tensorization.sample_period_s,
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
            if _is_overfit_stage(stage)
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
            validation_loss_dict = validation_loss_to_dict(
                evaluation,
                config.loss,
                config.repair,
            )
            evidence["final_evaluation"] = evaluation_dict
            evidence["final_validation_loss"] = validation_loss_dict
            metric = evaluation.prior.fde
            improved = best_metric is None or metric < best_metric
            if improved:
                best_metric = metric
                best_epoch = epoch
                epochs_without_improvement = 0
            elif completed_epoch and stage != "repair-formal":
                epochs_without_improvement += 1
        next_progress = TrainingProgress(
            epoch=epoch + 1 if completed_epoch else epoch,
            next_batch_index=0 if completed_epoch else next_batch,
            global_step=epoch_result.next_global_step,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        train_record = {
            "mean_optimizer_loss": epoch_result.mean_optimizer_loss,
            "reconstruction_loss": epoch_result.sums.reconstruction_loss,
            "endpoint_loss": epoch_result.sums.endpoint_loss,
            "kl_loss": epoch_result.sums.kl_loss,
            "optimizer_steps": epoch_result.optimizer_steps,
            "microbatches": epoch_result.microbatch_count,
        }
        if config.repair is not None:
            train_record.update(
                seam_velocity_loss=epoch_result.sums.seam_velocity_loss,
                velocity_loss=epoch_result.sums.velocity_loss,
                acceleration_loss=epoch_result.sums.acceleration_loss,
                jerk_loss=epoch_result.sums.jerk_loss,
                condition_ranking_loss=(
                    epoch_result.sums.condition_ranking_loss
                ),
                observed_condition_count=(
                    epoch_result.sums.observed_condition_count
                ),
                correct_condition_kl=epoch_result.sums.correct_condition_kl,
                none_condition_kl=epoch_result.sums.none_condition_kl,
                sampler_exposure=sampler_exposure,
            )
        epoch_candidate_path: Path | None = None
        if stage == "repair-formal" and should_validate:
            if not completed_epoch:
                raise AssertionError("repair-formal validation must complete an epoch")
            epoch_candidate_path = epoch_candidates_dir / (
                f"epoch-{epoch + 1:04d}-step-{next_progress.global_step:08d}.pt"
            )
        record = {
            "kind": "epoch",
            "stage": stage,
            "epoch": epoch,
            "completed_epoch": completed_epoch,
            "global_step": epoch_result.next_global_step,
            "next_batch_index": next_progress.next_batch_index,
            "train": train_record,
            "validation": evaluation_dict,
            "validation_loss": validation_loss_dict,
            "best_prior_min_fde_6": best_metric,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        if stage == "repair-formal":
            record["checkpoint_selection"] = {
                "epoch_candidate": (
                    None if epoch_candidate_path is None else str(epoch_candidate_path)
                ),
                "best_is_provisional_fde_candidate": True,
                "active_checkpoint_gate": REPAIR_FORMAL_SELECTION_CONTRACT[
                    "active_checkpoint_gate"
                ],
            }
        _append_jsonl(metrics_path, record)
        epoch_records += 1
        metrics_records += 1
        if epoch_candidate_path is not None:
            save_run_checkpoint(
                epoch_candidate_path,
                next_progress,
                role="epoch_validation_candidate",
                candidate_epoch=epoch + 1,
            )
        if improved:
            save_run_checkpoint(
                best_path,
                next_progress,
                role=(
                    "provisional_fde_best"
                    if stage == "repair-formal"
                    else "best_min_fde_6"
                ),
            )
        save_run_checkpoint(
            latest_path,
            next_progress,
            role="latest_resume_state",
        )
        progress = next_progress
        if step_limit is not None and progress.global_step >= step_limit:
            stop_reason = "max_steps"
            break
        if (
            not _is_overfit_stage(stage)
            and stage != "repair-formal"
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
    repair_formal_complete = (
        stage != "repair-formal"
        or (
            formal_epoch_budget is not None
            and progress.epoch >= formal_epoch_budget
            and progress.next_batch_index == 0
        )
    )
    epoch_candidate_paths = (
        sorted(str(path) for path in epoch_candidates_dir.glob("epoch-*.pt"))
        if stage == "repair-formal"
        else []
    )
    summary = {
        "stage": stage,
        "status": "complete" if repair_formal_complete else "paused",
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
    if stage == "repair-formal":
        summary.update(
            formal_selection={
                **REPAIR_FORMAL_SELECTION_CONTRACT,
                "frozen_epoch_budget": formal_epoch_budget,
                "best_checkpoint_status": "provisional_fde_candidate",
                "active_checkpoint_selected": False,
                "epoch_candidate_count": len(epoch_candidate_paths),
            }
        )
        summary["outputs"].update(
            epoch_candidates=str(epoch_candidates_dir),
            epoch_candidate_checkpoints=epoch_candidate_paths,
        )
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
    parser.add_argument("--benchmark-repeats", type=int)
    parser.add_argument("--tf32", choices=("on", "off"))
    parser.add_argument("--pin-memory", choices=("on", "off"))
    parser.add_argument("--persistent-workers", choices=("on", "off"))
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
        benchmark_repeats=args.benchmark_repeats,
        allow_tf32=(None if args.tf32 is None else args.tf32 == "on"),
        pin_memory=(
            None if args.pin_memory is None else args.pin_memory == "on"
        ),
        persistent_workers=(
            None
            if args.persistent_workers is None
            else args.persistent_workers == "on"
        ),
    )
    print(
        f"CVAE {args.stage} complete: "
        f"step={summary.get('progress', {}).get('global_step', 'benchmark')}",
    )


if __name__ == "__main__":
    main()

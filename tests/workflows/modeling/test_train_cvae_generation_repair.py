from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import scripts.modeling.train_cvae as train_module
import scripts.modeling.train_cvae_generation_repair as repair_entry
from scripts.modeling.train_cvae import (
    _DeterministicFullCycleSampler,
    _ensure_immutable_run_manifest,
    _fingerprints,
    _repair_overfit_identity,
    _repair_overfit_view,
    _repair_view_datasets,
    _stage_paths,
)
from skilldrive.training import load_cvae_config
from skilldrive.training.trainer import (
    BenchmarkResult,
    LossSums,
    OptimizerStepResult,
    TrainEpochResult,
)


CONFIG = Path("configs/models/cvae_generation_repair_v1.yaml")


class _EntriesDataset(Dataset[int]):
    def __init__(self) -> None:
        self.entries = []
        labels = [("<none>", 100), ("slow_lead_blockage", 20)] + [
            (f"skill-{index:02d}", 4) for index in range(12)
        ]
        for label, count in labels:
            for index in range(count):
                supervised = label != "<none>"
                self.entries.append(
                    {
                        "sample_id": f"{label}-{index}",
                        "shard": f"shard-{index // 10}.pt",
                        "spec": {
                            "skill_id": label,
                            "skill_supervision_mask": supervised,
                        },
                    }
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> int:
        return index


class _MixedOverfitDataset(Dataset[int]):
    def __init__(self) -> None:
        observed = []
        for index in range(16):
            observed.append(
                {
                    "sample_id": f"focus-{index}",
                    "shard": "observed-shard.pt",
                    "spec": {
                        "skill_id": "slow_lead_blockage",
                        "skill_supervision_mask": True,
                    },
                }
            )
        for index in range(16):
            observed.append(
                {
                    "sample_id": f"other-{index}",
                    "shard": "observed-shard.pt",
                    "spec": {
                        "skill_id": f"skill-{index % 12:02d}",
                        "skill_supervision_mask": True,
                    },
                }
            )
        base = [
            {
                "sample_id": f"base-{index}",
                "shard": "base-shard.pt",
                "spec": {
                    "skill_id": "<none>",
                    "skill_supervision_mask": False,
                },
            }
            for index in range(32)
        ]
        self.entries = observed + base

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> int:
        return index


class _FormalToyDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, sample_index_path: Path, *, development: bool = False) -> None:
        labels = (
            [("<none>", False)] * 4
            if development
            else [
                ("<none>", False),
                ("<none>", False),
                ("<none>", False),
                ("<none>", False),
                ("skill-a", True),
                ("skill-a", True),
                ("skill-b", True),
                ("skill-b", True),
            ]
        )
        self.sample_index_path = sample_index_path
        self.entries = [
            {
                "sample_id": f"{'dev' if development else 'train'}-{index}",
                "shard": f"shard-{index // 4}.pt",
                "spec": {
                    "skill_id": skill_id,
                    "skill_supervision_mask": supervised,
                },
            }
            for index, (skill_id, supervised) in enumerate(labels)
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor(float(index))}


class _ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_repair_overfit_selects_32_base_with_16_focus_and_16_other_observed() -> None:
    config = load_cvae_config(CONFIG)
    dataset = _EntriesDataset()
    view = _repair_overfit_view(dataset, config)
    repeated = _repair_overfit_view(dataset, config)

    base = [
        entry for entry in view.entries if not entry["spec"]["skill_supervision_mask"]
    ]
    observed = [
        entry for entry in view.entries if entry["spec"]["skill_supervision_mask"]
    ]
    assert len(view) == 64
    assert len(set(view.indices)) == 64
    assert view.indices == repeated.indices
    assert len(base) == 32
    assert len(observed) == 32
    observed_counts = {}
    for entry in observed:
        skill_id = entry["spec"]["skill_id"]
        observed_counts[skill_id] = observed_counts.get(skill_id, 0) + 1
    assert observed_counts[config.overfit.skill_id] == 16
    assert sum(
        count
        for skill_id, count in observed_counts.items()
        if skill_id != config.overfit.skill_id
    ) == 16
    assert set(observed_counts) - {config.overfit.skill_id} == {
        f"skill-{index:02d}" for index in range(12)
    }


def test_repair_overfit_cycle_never_degenerates_to_a_shard_prefix() -> None:
    dataset = _MixedOverfitDataset()
    sampler = _DeterministicFullCycleSampler(
        dataset,
        seed=2026,
        num_samples=64 * 50,
    )
    full = list(sampler)

    assert all(
        len(set(full[start : start + 64])) == 64
        for start in range(len(full) - 63)
    )
    assert sum(index >= 32 for index in full) == 32 * 50
    assert sum(index < 32 for index in full) == 32 * 50
    for start in range(0, len(full), 64):
        cycle = full[start : start + 64]
        labels = {
            dataset.entries[index]["spec"]["skill_id"]
            for index in cycle
            if dataset.entries[index]["spec"]["skill_supervision_mask"]
        }
        assert labels == {
            "slow_lead_blockage",
            *(f"skill-{index:02d}" for index in range(12)),
        }
        assert sum(
            dataset.entries[index]["spec"]["skill_id"] == "slow_lead_blockage"
            for index in cycle
        ) == 16

    sampler.set_range(37, 211)
    assert list(sampler) == full[37:211]
    sampler.set_range()
    sampler.set_epoch(1)
    second = list(sampler)
    assert second != full
    assert all(
        len(set(second[start : start + 64])) == 64
        for start in range(len(second) - 63)
    )


def test_repair_overfit_identity_records_exact_cohort_and_fixed_stream() -> None:
    config = load_cvae_config(CONFIG)
    view = _repair_overfit_view(_EntriesDataset(), config)
    sampler = _DeterministicFullCycleSampler(
        view,
        seed=config.training.seed,
        num_samples=config.overfit.max_steps * config.overfit.batch_size,
    )

    identity = _repair_overfit_identity(
        view,
        sampler,
        expected_sample_count=64,
        focus_skill_id=config.overfit.skill_id,
    )

    assert identity["selected_sample_count"] == 64
    assert identity["base_sample_count"] == 32
    assert identity["observed_sample_count"] == 32
    assert len(identity["observed_by_skill"]) == 13
    assert identity["focus_skill_id"] == config.overfit.skill_id
    assert identity["focus_observed_sample_count"] == 16
    assert identity["other_observed_sample_count"] == 16
    assert len(identity["selected_sample_ids"]) == 64
    assert len(identity["stream_identity_sha256"]) == 64


def test_repair_benchmark_uses_one_canonical_measured_order_across_batches() -> None:
    config = load_cvae_config(CONFIG)
    dataset = _EntriesDataset()
    plans = []
    for batch_size in (384, 512):
        training = replace(
            config.training,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
        )
        plans.append(
            train_module._benchmark_sampling_plan(
                config=config,
                training=training,
                train_dataset=dataset,
                max_steps=None,
            )
        )

    first, second = plans
    assert first.measured_steps == 140
    assert second.measured_steps == 105
    assert first.measured_samples == second.measured_samples == 53_760
    assert first.sampler_metadata["measured_order_sha256"] == second.sampler_metadata[
        "measured_order_sha256"
    ]
    assert first.measurement_sample_contract == second.measurement_sample_contract
    first_stream = list(first.loader_sampler)
    second_stream = list(second.loader_sampler)
    assert first_stream[first.warmup_samples :] == second_stream[second.warmup_samples :]
    assert first.warmup_sampler_metadata is not None
    assert first.warmup_sampler_metadata["seed"] != first.sampler_metadata["seed"]


def test_repair_benchmark_rejects_non_divisible_batch() -> None:
    config = load_cvae_config(CONFIG)
    training = replace(
        config.training,
        batch_size=500,
        gradient_accumulation_steps=1,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        train_module._benchmark_sampling_plan(
            config=config,
            training=training,
            train_dataset=_EntriesDataset(),
            max_steps=None,
        )


def test_repair_benchmark_candidate_contract_prevents_pin_or_worker_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG)
    dataset = _EntriesDataset()
    dataset.cache_dir = tmp_path / "formal_train"
    dataset.sample_index_sha256 = "sample-index"
    monkeypatch.setattr(train_module, "_cache_fingerprint", lambda path: "c" * 64)
    monkeypatch.setattr(
        train_module,
        "cvae_schema_fingerprint",
        lambda schema: "schema",
    )
    monkeypatch.setattr(
        train_module,
        "model_kwargs_from_config",
        lambda config, schema: {"model": "scalar"},
    )
    monkeypatch.setattr(
        train_module,
        "_repair_source_fingerprints",
        lambda: {name: name * 64 for name in train_module.REPAIR_SOURCE_PATHS},
    )
    calls = {"model": 0, "loader": 0, "benchmark": 0}
    global_step_starts = []
    runtime_ids = []

    def fake_build_model(config, schema):
        del config, schema
        calls["model"] += 1
        return _ScalarModel()

    original_loader = train_module._loader

    def counting_loader(*args, **kwargs):
        calls["loader"] += 1
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(train_module, "build_model_from_config", fake_build_model)
    monkeypatch.setattr(train_module, "_loader", counting_loader)

    def fake_benchmark(
        model,
        optimizer,
        loader,
        *,
        global_step,
        generator,
        warmup_steps,
        measured_steps,
        **kwargs,
    ):
        del kwargs
        calls["benchmark"] += 1
        global_step_starts.append(global_step)
        runtime_ids.append(
            (id(model), id(optimizer), id(loader), id(generator))
        )
        return BenchmarkResult(
            startup_seconds=0.01,
            warmup_seconds=0.02,
            step_seconds=(0.1,),
            data_wait_seconds=(0.0,),
            p50_step_seconds=0.1,
            p95_step_seconds=0.1,
            samples_per_second=100.0,
            measured_samples=train_module.REPAIR_BENCHMARK_MEASURED_SAMPLES,
            next_global_step=global_step + warmup_steps + measured_steps,
            cpu_metrics_available=False,
            cpu_busy_percent=None,
            cpu_iowait_percent=None,
            gpu_utilization_available=False,
            gpu_utilization_mean_percent=None,
            gpu_utilization_p50_percent=None,
            gpu_utilization_p95_percent=None,
            gpu_utilization_sample_count=0,
            monitor_overhead_seconds=0.0,
        )

    monkeypatch.setattr(train_module, "benchmark_training", fake_benchmark)
    output_dir = tmp_path / "benchmarks"
    summaries = []
    for pin_memory, persistent_workers in (
        (False, False),
        (True, False),
        (False, True),
    ):
        training = replace(
            config.training,
            device="cpu",
            amp=False,
            batch_size=384,
            num_workers=1,
            prefetch_factor=1,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        summaries.append(
            train_module._run_benchmark(
                config=config,
                schema=object(),
                training=training,
                learning_rate=config.training.learning_rate,
                train_dataset=dataset,
                output_dir=output_dir,
                max_steps=None,
                benchmark_repeats=3,
                run_training_started=time.perf_counter(),
                progress_stream=io.StringIO(),
            )
        )

    assert len({summary["candidate_id"] for summary in summaries}) == 3
    assert all(len(summary["candidate_id"]) <= 120 for summary in summaries)
    assert "-pw0-pin0-" in summaries[0]["candidate_id"]
    assert "-pw0-pin1-" in summaries[1]["candidate_id"]
    assert "-pw1-pin0-" in summaries[2]["candidate_id"]
    assert len({summary["candidate_contract_id"] for summary in summaries}) == 3
    assert len({summary["benchmark_contract_id"] for summary in summaries}) == 1
    assert calls == {"model": 3, "loader": 3, "benchmark": 9}
    window_steps = config.benchmark.warmup_steps + 140
    assert global_step_starts == [0, window_steps, 2 * window_steps] * 3
    assert all(
        len(set(runtime_ids[start : start + 3])) == 1
        for start in range(0, len(runtime_ids), 3)
    )
    assert all(len(summary["results"]) == 3 for summary in summaries)
    assert all(
        summary["repeat_state_contract"]
        == "continuous_model_optimizer_loader_v1"
        for summary in summaries
    )
    assert all(summary["shared_setup"] is not None for summary in summaries)
    assert all(
        "model_setup_seconds" not in result
        and "loader_setup_seconds" not in result
        for summary in summaries
        for result in summary["results"]
    )
    assert all(
        (output_dir / summary["candidate_id"] / "summary.json").is_file()
        for summary in summaries
    )
    index = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    active = index["contracts"][index["active_contract_id"]]
    assert len(active["candidates"]) == 3
    assert summaries[0]["run_training_setup_seconds"] >= 0.0
    assert summaries[0]["run_training_setup_scope"]["excludes"] == [
        "Python process startup",
        "module imports including torch",
    ]


def test_training_cli_forwards_benchmark_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return {"stage": kwargs["stage"]}

    monkeypatch.setattr(train_module, "run_training", fake_run_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cvae.py",
            "--config",
            str(CONFIG),
            "--stage",
            "repair-benchmark",
            "--benchmark-repeats",
            "2",
            "--tf32",
            "on",
            "--pin-memory",
            "off",
            "--persistent-workers",
            "on",
        ],
    )

    train_module.main()

    assert captured["benchmark_repeats"] == 2
    assert captured["allow_tf32"] is True
    assert captured["pin_memory"] is False
    assert captured["persistent_workers"] is True


def test_effective_training_applies_runtime_backend_and_loader_overrides() -> None:
    config = load_cvae_config(CONFIG)
    training, _ = train_module._effective_training(
        config,
        "repair-benchmark",
        384,
        4,
        True,
        1,
        True,
        False,
        False,
    )

    assert training.allow_tf32 is True
    assert training.pin_memory is False
    assert training.persistent_workers is False
    with pytest.raises(ValueError, match="requires num_workers > 0"):
        train_module._effective_training(
            config,
            "repair-benchmark",
            384,
            0,
            True,
            1,
            False,
            True,
            True,
        )


@pytest.mark.parametrize("mode", ["overfit", "benchmark", "formal"])
def test_repair_entry_dispatches_only_explicit_repair_stages(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return {"stage": kwargs["stage"]}

    monkeypatch.setattr(repair_entry, "run_training", fake_run_training)
    result = repair_entry.run_repair_training(mode=mode, max_steps=2)

    assert captured["stage"] == f"repair-{mode}"
    assert captured["config_path"] == repair_entry.DEFAULT_REPAIR_CONFIG
    assert result["stage"] == f"repair-{mode}"


def test_repair_wrapper_forwards_benchmark_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run_training(**kwargs):
        captured.update(kwargs)
        return {"stage": kwargs["stage"]}

    monkeypatch.setattr(repair_entry, "run_training", fake_run_training)
    repair_entry.run_repair_training(
        mode="benchmark",
        benchmark_repeats=2,
        tf32=True,
        pin_memory=False,
        persistent_workers=True,
    )

    assert captured["benchmark_repeats"] == 2
    assert captured["allow_tf32"] is True
    assert captured["pin_memory"] is False
    assert captured["persistent_workers"] is True


def test_repair_wrapper_cli_forwards_benchmark_runtime_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_run_repair_training(**kwargs):
        captured.update(kwargs)
        return {"stage": "repair-benchmark"}

    monkeypatch.setattr(repair_entry, "run_repair_training", fake_run_repair_training)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_cvae_generation_repair.py",
            "--mode",
            "benchmark",
            "--benchmark-repeats",
            "2",
            "--tf32",
            "on",
            "--pin-memory",
            "off",
            "--persistent-workers",
            "on",
        ],
    )

    repair_entry.main()

    assert captured["benchmark_repeats"] == 2
    assert captured["tf32"] is True
    assert captured["pin_memory"] is False
    assert captured["persistent_workers"] is True


@pytest.mark.parametrize("legacy_stage", ["overfit", "development", "benchmark", "formal"])
def test_repair_yaml_rejects_every_legacy_stage_before_data_access(
    legacy_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG)
    monkeypatch.setattr(train_module, "load_cvae_config", lambda path: config)
    monkeypatch.setattr(
        train_module,
        "build_cvae_schema",
        lambda path: (_ for _ in ()).throw(AssertionError("data access reached")),
    )

    with pytest.raises(ValueError, match="only run repair stages"):
        train_module.run_training(config_path=CONFIG, stage=legacy_stage)


@pytest.mark.parametrize(
    ("stage", "output_suffix"),
    [
        ("repair-overfit", "development/overfit"),
        ("repair-benchmark", "benchmarks"),
        ("repair-formal", "formal"),
    ],
)
def test_repair_stage_paths_use_only_formal_train_source_cache(
    stage: str,
    output_suffix: str,
    tmp_path: Path,
) -> None:
    config = load_cvae_config(CONFIG)
    train, development, output = _stage_paths(config, stage, tmp_path)

    expected = tmp_path / config.cache.root / "formal_train"
    assert train == expected
    assert development == expected
    assert output.as_posix().endswith(output_suffix)
    assert "internal_validation" not in train.as_posix()
    assert "final_validation" not in development.as_posix()


def test_repair_view_audit_loads_disjoint_complete_formal_train_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG)
    assert config.repair is not None
    source_cache = tmp_path / config.cache.root / "formal_train"
    source_cache.mkdir(parents=True)
    source_manifest = source_cache / "cache_manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    source_entries = [
        {
            "sample_id": f"sample-{index}",
            "scenario_id": f"scenario-{index}",
            "target_track_id": f"track-{index}",
            "shard": "shards/shard-00000.pt",
            "offset": index,
            "spec": {"skill_id": "<none>", "skill_supervision_mask": False},
        }
        for index in range(3)
    ]
    source_index = source_cache / "sample_index.jsonl"
    source_index.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in source_entries),
        encoding="utf-8",
    )
    train_index = tmp_path / config.repair.split.train_sample_index
    development_index = tmp_path / config.repair.split.development_sample_index
    train_index.parent.mkdir(parents=True, exist_ok=True)
    train_index.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in source_entries[:2]),
        encoding="utf-8",
    )
    development_index.write_text(
        json.dumps(source_entries[2], sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def descriptor(path: Path) -> dict[str, object]:
        return {
            "path": path.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    audit_path = tmp_path / config.repair.split.audit
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "validation_manifests_opened": False,
                "integrity": {
                    "scenario_overlap": 0,
                    "sample_offset_overlap": 0,
                    "sample_offset_union_matches_v5_cache": True,
                },
                "sources": {
                    "formal_train_v5_cache_manifest": descriptor(source_manifest),
                    "formal_train_v5_sample_index": descriptor(source_index),
                },
                "outputs": {
                    "repair_train_sample_index": descriptor(train_index),
                    "repair_dev_sample_index": descriptor(development_index),
                },
                "counts": {
                    "repair_train_samples": 2,
                    "repair_dev_samples": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeCachedDataset(Dataset[int]):
        def __init__(self, cache_dir, *, schema, sample_index_path):
            del schema
            self.cache_dir = Path(cache_dir)
            self.sample_index_path = Path(sample_index_path)
            self.source_sample_index_path = self.cache_dir / "sample_index.jsonl"
            self.entries = [
                json.loads(line)
                for line in self.sample_index_path.read_text(encoding="utf-8").splitlines()
            ]

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, index):
            return index

    monkeypatch.setattr(train_module, "CVAECachedDataset", FakeCachedDataset)
    train, development, audit = _repair_view_datasets(
        config,
        root=tmp_path,
        source_cache=source_cache,
        schema=object(),
    )

    assert len(train) == 2
    assert len(development) == 1
    assert audit["validation_manifests_opened"] is False


def test_repair_fingerprints_include_both_view_indexes_and_canonical_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG)
    train_index = tmp_path / "train.jsonl"
    development_index = tmp_path / "dev.jsonl"
    train_index.write_text("train\n", encoding="utf-8")
    development_index.write_text("dev\n", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "model_kwargs_from_config",
        lambda config, schema: {"model": "repair"},
    )
    monkeypatch.setattr(train_module, "_cache_fingerprint", lambda path: "cache")

    sampler_contract = {
        "strategy": "observed_skill_balance_v1",
        "target": "most_frequent_observed",
        "seed": 2026,
        "source_samples": 100,
        "base_samples": 50,
        "observed_source_by_skill": {"skill-a": 50},
        "observed_epoch_exposure_by_skill": {"skill-a": 50},
        "target_observed_exposure": 50,
        "max_repeats_per_sample": 8,
        "epoch_samples": 100,
    }
    fingerprints = _fingerprints(
        config=config,
        schema=object(),
        stage="repair-formal",
        training=config.training,
        learning_rate=config.training.learning_rate,
        train_cache=tmp_path / "formal_train",
        validation_cache=tmp_path / "formal_train",
        train_sample_index=train_index,
        validation_sample_index=development_index,
        sampler_contract=sampler_contract,
    )
    changed = _fingerprints(
        config=config,
        schema=object(),
        stage="repair-formal",
        training=config.training,
        learning_rate=config.training.learning_rate,
        train_cache=tmp_path / "formal_train",
        validation_cache=tmp_path / "formal_train",
        train_sample_index=train_index,
        validation_sample_index=development_index,
        sampler_contract={**sampler_contract, "max_repeats_per_sample": 7},
    )

    assert fingerprints["train_sample_index"] == _sha256(train_index)
    assert fingerprints["validation_sample_index"] == _sha256(development_index)
    assert len(fingerprints["config"]) == 64
    assert len(fingerprints["repair_sampler_contract"]) == 64
    assert fingerprints["repair_sampler_contract"] != changed[
        "repair_sampler_contract"
    ]
    for name in train_module.REPAIR_SOURCE_PATHS:
        assert len(fingerprints[f"repair_source_{name}"]) == 64


def test_repair_yaml_disables_fde_early_stopping() -> None:
    config = load_cvae_config(CONFIG)

    assert config.training.early_stopping_patience == 0


def test_resume_requires_the_complete_immutable_run_manifest(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    original = {
        "immutable_contract": "repair_run_manifest_v1",
        "fingerprints": {"repair_sampler_contract": "a" * 64},
        "training": {"sampler": {"max_repeats_per_sample": 8}},
    }
    _ensure_immutable_run_manifest(path, original, resuming=False)
    original_bytes = path.read_bytes()

    with pytest.raises(ValueError, match="immutable run_manifest mismatch"):
        _ensure_immutable_run_manifest(
            path,
            {
                **original,
                "training": {"sampler": {"max_repeats_per_sample": 7}},
            },
            resuming=True,
        )
    assert path.read_bytes() == original_bytes

    path.unlink()
    with pytest.raises(ValueError, match="cannot resume without"):
        _ensure_immutable_run_manifest(path, original, resuming=True)
    assert not path.exists()


def test_repair_formal_resume_matches_continuous_and_retains_epoch_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG)
    config = replace(
        config,
        training=replace(
            config.training,
            device="cpu",
            amp=False,
            batch_size=4,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
            formal_max_epochs=3,
            early_stopping_patience=1,
            checkpoint_every_steps=1000,
        ),
        outputs=replace(
            config.outputs,
            root=Path("artifacts"),
            development=Path("artifacts/development"),
            benchmarks=Path("artifacts/benchmarks"),
            formal=Path("artifacts/formal"),
        ),
    )
    assert config.repair is not None

    roots = (tmp_path / "continuous", tmp_path / "resumed")
    for root in roots:
        audit_path = root / config.repair.split.audit
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text("{}\n", encoding="utf-8")

    def fake_views(config, *, root, source_cache, schema):
        del config, source_cache, schema
        return (
            _FormalToyDataset(root / "train.index.jsonl"),
            _FormalToyDataset(root / "dev.index.jsonl", development=True),
            {"status": "complete"},
        )

    def fake_fingerprints(**kwargs):
        sampler_digest = train_module._hash_value(kwargs["sampler_contract"])
        return {
            "contract": "fixed",
            "repair_sampler_contract": sampler_digest,
            **{
                f"repair_source_{name}": hashlib.sha256(name.encode()).hexdigest()
                for name in train_module.REPAIR_SOURCE_PATHS
            },
        }

    def fake_train_epoch(
        model,
        optimizer,
        batches,
        *,
        global_step,
        generator,
        on_optimizer_step,
        **kwargs,
    ):
        del kwargs
        aggregate = LossSums()
        microbatch_count = 0
        losses = []
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            target = batch["value"].float().mean() / 10.0 + torch.rand(
                (), generator=generator
            )
            loss = (model.weight - target).square()
            loss.backward()
            optimizer.step()
            microbatch_count += 1
            global_step += 1
            sample_count = int(batch["value"].shape[0])
            batch_sums = LossSums(
                reconstruction_sum=float(loss.detach()) * sample_count,
                reconstruction_element_count=sample_count,
                endpoint_sum=float(loss.detach()) * sample_count,
                endpoint_element_count=sample_count,
                kl_sum=0.0,
                sample_count=sample_count,
                valid_point_count=sample_count,
                valid_sample_count=sample_count,
            )
            aggregate = aggregate + batch_sums
            losses.append(float(loss.detach()))
            on_optimizer_step(
                OptimizerStepResult(
                    next_global_step=global_step,
                    total_loss=losses[-1],
                    kl_weight=0.0,
                    gradient_norm=0.0,
                    microbatch_count=1,
                    sums=batch_sums,
                ),
                microbatch_count,
            )
        return TrainEpochResult(
            next_global_step=global_step,
            optimizer_steps=microbatch_count,
            microbatch_count=microbatch_count,
            mean_optimizer_loss=sum(losses) / len(losses),
            sums=aggregate,
        )

    monkeypatch.setattr(train_module, "load_cvae_config", lambda path: config)
    monkeypatch.setattr(train_module, "build_cvae_schema", lambda path: object())
    monkeypatch.setattr(train_module, "cvae_schema_fingerprint", lambda schema: "schema")
    monkeypatch.setattr(train_module, "_repair_view_datasets", fake_views)
    monkeypatch.setattr(train_module, "_validate_cache_contract", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "_fingerprints", fake_fingerprints)
    monkeypatch.setattr(
        train_module,
        "model_kwargs_from_config",
        lambda config, schema: {"model": "scalar"},
    )
    monkeypatch.setattr(
        train_module,
        "build_model_from_config",
        lambda config, schema: _ScalarModel(),
    )
    monkeypatch.setattr(train_module, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *args, **kwargs: SimpleNamespace(
            prior=SimpleNamespace(fde=1.0)
        ),
    )
    monkeypatch.setattr(
        train_module,
        "evaluation_to_dict",
        lambda result, prior_samples: {
            "prior_samples": prior_samples,
            "min_fde_6": result.prior.fde,
        },
    )
    monkeypatch.setattr(
        train_module,
        "validation_loss_to_dict",
        lambda *args, **kwargs: {"total_loss": 0.0},
    )

    continuous = train_module.run_training(
        config_path=CONFIG,
        stage="repair-formal",
        project_root=roots[0],
        resume="none",
        progress_stream=io.StringIO(),
    )
    paused = train_module.run_training(
        config_path=CONFIG,
        stage="repair-formal",
        project_root=roots[1],
        resume="none",
        max_epochs=1,
        progress_stream=io.StringIO(),
    )
    resumed = train_module.run_training(
        config_path=CONFIG,
        stage="repair-formal",
        project_root=roots[1],
        resume="auto",
        progress_stream=io.StringIO(),
    )

    assert paused["status"] == "paused"
    assert paused["stop_reason"] == "invocation_epoch_limit"
    assert continuous["status"] == resumed["status"] == "complete"
    assert continuous["stop_reason"] == resumed["stop_reason"] == "fixed_epoch_budget"
    assert len(continuous["outputs"]["epoch_candidate_checkpoints"]) == 3
    assert len(resumed["outputs"]["epoch_candidate_checkpoints"]) == 3
    assert resumed["formal_selection"]["active_checkpoint_selected"] is False
    assert resumed["formal_selection"]["fde_early_stopping"] is False

    continuous_latest = torch.load(
        continuous["outputs"]["latest"], map_location="cpu", weights_only=False
    )
    resumed_latest = torch.load(
        resumed["outputs"]["latest"], map_location="cpu", weights_only=False
    )
    assert continuous_latest["progress"] == resumed_latest["progress"]
    assert torch.equal(
        continuous_latest["model"]["weight"],
        resumed_latest["model"]["weight"],
    )
    assert torch.equal(
        continuous_latest["extra"]["training_generator_state"],
        resumed_latest["extra"]["training_generator_state"],
    )

    candidate_epochs = []
    for path in resumed["outputs"]["epoch_candidate_checkpoints"]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        candidate_epochs.append(payload["extra"]["checkpoint"]["candidate_epoch"])
        assert payload["extra"]["checkpoint"]["active_checkpoint"] is False
    assert candidate_epochs == [1, 2, 3]

    best = torch.load(resumed["outputs"]["best"], map_location="cpu", weights_only=False)
    assert best["extra"]["checkpoint"]["role"] == "provisional_fde_best"
    assert best["extra"]["checkpoint"]["selection_status"] == (
        "provisional_fde_candidate"
    )
    assert best["extra"]["checkpoint"]["active_checkpoint"] is False

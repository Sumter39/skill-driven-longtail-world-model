from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import scripts.modeling.train_cvae as train_cvae_module
from scripts.modeling.evaluate_cvae import _endpoint_diversity
from scripts.modeling.train_cvae import (
    _DeterministicBenchmarkSampler,
    _MaterializedDataset,
    _RepeatedDataset,
    _base_view,
    _benchmark_sample_stream_metadata,
    _cache_fingerprint,
    _observed_view,
    _progress_line,
    _stage_paths,
    _validate_cache_contract,
)
from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.models import CVAEOutput
from skilldrive.training import load_cvae_config


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/models/cvae_baseline.yaml"


class _ToyDataset(Dataset[int]):
    def __init__(self) -> None:
        self.calls = 0
        self.entries = [
            {
                "shard": "shard-0.pt",
                "spec": {"skill_id": "<none>", "skill_supervision_mask": False},
            },
            {
                "shard": "shard-0.pt",
                "spec": {
                    "skill_id": "slow_lead_blockage",
                    "skill_supervision_mask": True,
                },
            },
            {
                "shard": "shard-1.pt",
                "spec": {
                    "skill_id": "short_headway_following",
                    "skill_supervision_mask": True,
                },
            },
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> int:
        self.calls += 1
        return index


class _SamplerDataset(Dataset[int]):
    def __init__(self, sample_count: int, shard_size: int = 64) -> None:
        self.entries = [
            {"shard": f"shard-{index // shard_size:05d}.pt"}
            for index in range(sample_count)
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> int:
        return index


class _TrainingEvidenceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, sample_count: int = 32) -> None:
        self.entries = [
            {
                "shard": "shard-00000.pt",
                "spec": {
                    "skill_id": "slow_lead_blockage",
                    "skill_supervision_mask": True,
                },
            }
            for _ in range(sample_count)
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        del index
        history = torch.zeros(1, 2, 6)
        history[0, -1, 2] = 1.0
        return {
            "actor_history": history,
            "actor_time_mask": torch.ones(1, 2, dtype=torch.bool),
            "actor_mask": torch.ones(1, dtype=torch.bool),
            "target_actor_index": torch.tensor(0),
            "target_future": torch.zeros(2, 2),
            "target_future_mask": torch.ones(2, dtype=torch.bool),
        }


class _TrainingEvidenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> CVAEOutput:
        del generator
        target = batch["target_future"]
        prediction = torch.ones_like(target) * self.scale
        zeros = torch.zeros(target.shape[0], 1, device=target.device)
        return CVAEOutput(
            future_delta=prediction,
            future_position_local=prediction,
            prior_mean=zeros,
            prior_logvar=zeros,
            posterior_mean=zeros + self.scale * 0.0,
            posterior_logvar=zeros,
            latent=zeros,
        )

    def sample_prior(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
        generator: torch.Generator,
    ) -> CVAEOutput:
        del generator
        target = batch["target_future"]
        prediction = (
            torch.ones(
                target.shape[0],
                num_samples,
                target.shape[1],
                target.shape[2],
                device=target.device,
            )
            * self.scale
        )
        zeros = torch.zeros(target.shape[0], 1, device=target.device)
        return CVAEOutput(
            future_delta=prediction,
            future_position_local=prediction,
            prior_mean=zeros,
            prior_logvar=zeros,
            posterior_mean=None,
            posterior_logvar=None,
            latent=zeros.unsqueeze(1).expand(-1, num_samples, -1),
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_views_keep_base_and_observed_samples_separate() -> None:
    dataset = _ToyDataset()

    base = _base_view(dataset)
    observed = _observed_view(dataset)

    assert list(base) == [0]
    assert list(observed) == [1, 2]
    assert [entry["spec"]["skill_id"] for entry in observed.entries] == [
        "slow_lead_blockage",
        "short_headway_following",
    ]


def test_repeated_dataset_preserves_entries_and_cycles_samples() -> None:
    dataset = _ToyDataset()
    repeated = _RepeatedDataset(dataset, repeats=3)

    assert len(repeated) == 9
    assert list(repeated) == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert repeated.entries == dataset.entries * 3


def test_benchmark_sampler_cycles_complete_shuffled_epochs_deterministically() -> None:
    dataset = _SamplerDataset(23, shard_size=5)
    sampler = _DeterministicBenchmarkSampler(
        dataset,
        seed=2026,
        num_samples=23 * 2 + 7,
    )

    first = list(sampler)
    second = list(sampler)

    assert len(first) == 53
    assert first == second
    assert sorted(first[:23]) == list(range(23))
    assert sorted(first[23:46]) == list(range(23))
    assert first[:23] != first[23:46]


@pytest.mark.parametrize("batch_size", [512, 640])
def test_benchmark_sampler_keeps_full_batches_without_permanent_tail_loss(
    batch_size: int,
) -> None:
    dataset = _SamplerDataset(772)
    total_samples = (20 + 200) * batch_size
    sampler = _DeterministicBenchmarkSampler(
        dataset,
        seed=2026,
        num_samples=total_samples,
    )
    indices = list(sampler)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )
    batches = iter(loader)
    first_two_batches = torch.cat((next(batches), next(batches))).tolist()

    assert len(indices) == total_samples
    assert len(indices) % batch_size == 0
    assert len(loader) == 220
    assert sorted(indices[: len(dataset)]) == list(range(len(dataset)))
    assert set(indices) == set(range(len(dataset)))
    assert set(first_two_batches) == set(range(len(dataset)))


def test_benchmark_sampler_metadata_hashes_the_fixed_measured_order() -> None:
    dataset = _SamplerDataset(19, shard_size=4)
    first = _DeterministicBenchmarkSampler(dataset, seed=7, num_samples=80)
    second = _DeterministicBenchmarkSampler(dataset, seed=7, num_samples=80)
    different = _DeterministicBenchmarkSampler(dataset, seed=8, num_samples=80)

    first_metadata = _benchmark_sample_stream_metadata(first, warmup_samples=20)
    second_metadata = _benchmark_sample_stream_metadata(second, warmup_samples=20)
    different_metadata = _benchmark_sample_stream_metadata(
        different,
        warmup_samples=20,
    )

    assert first_metadata == second_metadata
    assert first_metadata["measured_samples"] == 60
    assert first_metadata["contract_sha256"] != different_metadata["contract_sha256"]
    assert (
        first_metadata["measured_order_sha256"]
        != different_metadata["measured_order_sha256"]
    )


def test_materialized_dataset_reads_source_once_then_reuses_samples() -> None:
    dataset = _ToyDataset()
    materialized = _MaterializedDataset(dataset)

    assert dataset.calls == 3
    assert list(materialized) == [0, 1, 2]
    assert list(materialized) == [0, 1, 2]
    assert dataset.calls == 3


def test_endpoint_diversity_reports_real_separation() -> None:
    futures = torch.zeros(2, 3, 4, 2)
    futures[0, 1, -1, 0] = 0.2

    result = _endpoint_diversity(futures)

    assert result["maximum_endpoint_separation_m"] == pytest.approx(0.2)
    assert result["conditions_above_threshold"] == 1
    assert result["condition_count"] == 2


def test_resume_eta_uses_only_batches_processed_after_resume() -> None:
    line = _progress_line(
        stage="formal",
        epoch=0,
        next_batch=81,
        total_batches=100,
        completed_batches=81,
        planned_batches=100,
        processed_batches=1,
        global_step=81,
        loss=1.0,
        processed_samples=16,
        elapsed=1.0,
    )

    assert "formal" in line
    assert "batch 81/100" in line
    assert "ETA 19.0s" in line


def test_formal_stage_uses_internal_validation_and_never_final_validation() -> None:
    config = load_cvae_config(CONFIG_PATH)
    train, validation, _ = _stage_paths(config, "formal", REPO_ROOT)

    assert train == REPO_ROOT / config.cache.root / "formal_train"
    assert validation == REPO_ROOT / config.cache.root / "internal_validation"
    assert "final_validation" not in str(train)
    assert "final_validation" not in str(validation)


def test_cache_contract_rejects_partial_and_wrong_partition(tmp_path: Path) -> None:
    manifest_path = tmp_path / "formal_train.csv"
    write_manifest(
        manifest_path,
        [
            ManifestRow(
                scenario_id="scenario-1",
                split="train",
                source_path="train/scenario-1/scenario_scenario-1.parquet",
                city_name="test-city",
                selected_reason="test",
            )
        ],
    )
    candidate_pool = tmp_path / "formal_candidate_pool.csv"
    candidate_pool.write_text("header\n", encoding="utf-8")

    dataset = _ToyDataset()
    dataset.cache_manifest = {
        "status": "complete",
        "partition": "formal_train",
        "counts": {
            "manifest_scenarios": 1,
            "processed_manifest_scenarios": 1,
        },
        "inputs": {
            "manifest_sha256": _sha256(manifest_path),
            "schema_sha256": "schema-fingerprint",
            "candidate_pool_sha256": _sha256(candidate_pool),
        },
    }

    _validate_cache_contract(
        dataset,
        expected_partition="formal_train",
        manifest_path=manifest_path,
        schema_sha256="schema-fingerprint",
        candidate_pool_path=candidate_pool,
    )

    dataset.cache_manifest["status"] = "partial"
    with pytest.raises(ValueError, match="must be complete"):
        _validate_cache_contract(
            dataset,
            expected_partition="formal_train",
            manifest_path=manifest_path,
            schema_sha256="schema-fingerprint",
            candidate_pool_path=candidate_pool,
        )

    dataset.cache_manifest["status"] = "complete"
    dataset.cache_manifest["partition"] = "final_validation"
    with pytest.raises(ValueError, match="partition"):
        _validate_cache_contract(
            dataset,
            expected_partition="formal_train",
            manifest_path=manifest_path,
            schema_sha256="schema-fingerprint",
            candidate_pool_path=candidate_pool,
        )


def test_cache_fingerprint_ignores_runtime_timing_but_tracks_semantics(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = {
        "version": 1,
        "status": "complete",
        "partition": "development_train",
        "inputs": {"schema_sha256": "schema"},
        "counts": {"retained_samples": 2},
        "sample_index": {"sha256": "index", "records": 2},
        "shard_size": 64,
        "shards": [{"path": "shard.pt", "sha256": "shard"}],
        "elapsed_seconds": 10.0,
        "resume": {"verified_skipped_shards": 0},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    first = _cache_fingerprint(cache_dir)

    manifest["elapsed_seconds"] = 999.0
    manifest["resume"] = {"verified_skipped_shards": 8}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _cache_fingerprint(cache_dir) == first

    manifest["sample_index"]["sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _cache_fingerprint(cache_dir) != first


def test_overfit_records_initial_and_final_loss_timing_and_resume_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_cvae_config(CONFIG_PATH)
    config = replace(
        config,
        training=replace(
            config.training,
            device="cpu",
            amp=False,
            batch_size=32,
            num_workers=0,
            persistent_workers=False,
            pin_memory=False,
        ),
        overfit=replace(
            config.overfit,
            sample_count=32,
            batch_size=32,
            max_steps=1,
        ),
        outputs=replace(
            config.outputs,
            root=Path("artifacts"),
            development=Path("artifacts/development"),
            benchmarks=Path("artifacts/benchmarks"),
            formal=Path("artifacts/formal"),
        ),
    )
    dataset = _TrainingEvidenceDataset()
    monkeypatch.setattr(train_cvae_module, "load_cvae_config", lambda path: config)
    monkeypatch.setattr(train_cvae_module, "build_cvae_schema", lambda path: object())
    monkeypatch.setattr(
        train_cvae_module,
        "cvae_schema_fingerprint",
        lambda schema: "schema",
    )
    monkeypatch.setattr(
        train_cvae_module,
        "CVAECachedDataset",
        lambda path, schema: dataset,
    )
    monkeypatch.setattr(
        train_cvae_module,
        "_validate_cache_contract",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        train_cvae_module,
        "_fingerprints",
        lambda **kwargs: {"contract": "fixed"},
    )
    monkeypatch.setattr(
        train_cvae_module,
        "model_kwargs_from_config",
        lambda config, schema: {"model": "toy"},
    )
    monkeypatch.setattr(
        train_cvae_module,
        "build_model_from_config",
        lambda config, schema: _TrainingEvidenceModel(),
    )

    summary = train_cvae_module.run_training(
        config_path=CONFIG_PATH,
        stage="overfit",
        project_root=tmp_path,
        resume="none",
        max_steps=1,
        progress_stream=io.StringIO(),
    )
    metrics_path = Path(summary["outputs"]["metrics"])
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["kind"] for record in records] == [
        "initial_evaluation",
        "epoch",
    ]
    assert records[0]["validation"]["sample_count"] == 32
    assert records[0]["validation_loss"]["sample_count"] == 32
    assert records[1]["validation_loss"]["valid_point_count"] == 64
    assert summary["initial_evaluation"] == records[0]["validation"]
    assert summary["initial_validation_loss"] == records[0]["validation_loss"]
    assert summary["final_evaluation"] == records[1]["validation"]
    assert summary["final_validation_loss"] == records[1]["validation_loss"]
    assert summary["peak_vram_mib"] == 0.0
    assert summary["timing"]["training_seconds"] > 0.0
    assert summary["timing"]["validation_seconds"] > 0.0
    assert summary["timing"]["checkpoint_seconds"] > 0.0

    resumed = train_cvae_module.run_training(
        config_path=CONFIG_PATH,
        stage="overfit",
        project_root=tmp_path,
        resume="auto",
        max_steps=1,
        progress_stream=io.StringIO(),
    )

    assert resumed["initial_evaluation"] == summary["initial_evaluation"]
    assert resumed["initial_validation_loss"] == summary["initial_validation_loss"]
    assert resumed["final_evaluation"] == summary["final_evaluation"]
    assert resumed["final_validation_loss"] == summary["final_validation_loss"]
    assert resumed["timing"]["checkpoint_seconds"] > 0.0
    assert len(metrics_path.read_text(encoding="utf-8").splitlines()) == 2

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.modeling.prepare_cvae_data import run_preparation
import skilldrive.data.cvae_cache as cvae_cache
from skilldrive.data.cvae_cache import (
    CACHE_VERSION,
    CVAECachedDataset,
    ShardShuffleSampler,
)
from skilldrive.data.cvae_samples import (
    FUTURE_STEPS,
    HISTORY_STEPS,
    MAX_MAP_POINTS,
    MAX_MAP_POLYLINES,
    SampleSpec,
    build_cvae_schema,
    make_base_sample_spec,
    tensorize_scenario,
)
from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.seeds import SeedRecord, write_seed_records


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CONFIG = REPO_ROOT / "configs/models/cvae_baseline.yaml"
SKILL_CONTRACTS = {
    skill_id: yaml.safe_load(
        (REPO_ROOT / f"configs/skills/{skill_id}.yaml").read_text(encoding="utf-8")
    )
    for skill_id in ("slow_lead_blockage", "cut_in_then_brake")
}


def _agent(
    track_id: str,
    *,
    offset_y: float,
    is_focal: bool = False,
) -> AgentTrack:
    steps = HISTORY_STEPS + FUTURE_STEPS
    positions = np.column_stack(
        (np.arange(steps, dtype=np.float64), np.full(steps, offset_y))
    )
    return AgentTrack(
        track_id=track_id,
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([1.0, 0.0], (steps, 1)),
        headings=np.zeros(steps),
        observed_mask=np.arange(steps) < HISTORY_STEPS,
        is_focal=is_focal,
    )


def _scenario(
    scenario_id: str,
    *,
    target_offset_y: float = 0.0,
    dense_map: bool = False,
) -> Scenario:
    if dense_map:
        points = np.column_stack(
            (
                np.linspace(0.0, 98.0, MAX_MAP_POINTS + 5),
                np.full(MAX_MAP_POINTS + 5, target_offset_y),
            )
        )
        map_polylines = [
            MapPolyline(
                polyline_id=f"lane-{scenario_id}-{index:03d}",
                polyline_type="lane_centerline",
                points=points.copy(),
            )
            for index in range(MAX_MAP_POLYLINES + 2)
        ]
    else:
        map_polylines = [
            MapPolyline(
                polyline_id=f"lane-{scenario_id}",
                polyline_type="lane_centerline",
                points=np.array([[0.0, 0.0], [120.0, 0.0]]),
            )
        ]
    return Scenario(
        scenario_id=scenario_id,
        city_name="test-city",
        timestamps=np.arange(HISTORY_STEPS + FUTURE_STEPS, dtype=np.int64),
        focal_track_id="target",
        agents=[
            _agent("target", offset_y=target_offset_y, is_focal=True),
            _agent("responder", offset_y=target_offset_y + 5.0),
        ],
        map_polylines=map_polylines,
    )


def _row(scenario_id: str, split: str) -> ManifestRow:
    return ManifestRow(
        scenario_id=scenario_id,
        split=split,
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        city_name="test-city",
        selected_reason="test",
    )


def _record(
    scenario_id: str,
    skill_id: str = "slow_lead_blockage",
) -> SeedRecord:
    contract = SKILL_CONTRACTS[skill_id]
    roles = (
        {"slow_leader": "target", "follower": "responder"}
        if skill_id == "slow_lead_blockage"
        else {"cut_in_braking_vehicle": "target", "responding_vehicle": "responder"}
    )
    mode = contract["detection"]["mode"]
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id="target",
        responder_track_id="responder",
        role_track_ids=roles,
        trigger_score=0.9,
        seed_risk_metric="minimum_distance",
        seed_risk_value=3.0,
        target_risk_definition=contract["risk_definition"],
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={
            "detection_mode": mode,
            "detection_thresholds": contract["detection"]["thresholds"],
            "feasibility": contract["data_support"]["feasibility"],
            "missing_generation_conditions": (
                [] if mode == "observed_trigger" else ["target_behavior"]
            ),
        },
        sampled_parameters={"unused": 1.0},
    )


def _assert_cached_sample_matches_tensorized(actual, expected) -> None:
    assert actual["sample_id"] == expected.sample_id
    assert actual["scenario_id"] == expected.scenario_id
    assert actual["target_track_id"] == expected.target_track_id
    for name, value in actual.items():
        if name in {"sample_id", "scenario_id", "target_track_id"}:
            continue
        expected_value = torch.as_tensor(getattr(expected, name))
        assert value.dtype == expected_value.dtype, name
        assert torch.equal(value, expected_value), name


def _write_project(
    tmp_path: Path,
    *,
    train_count: int,
    validation_count: int,
    records: list[SeedRecord] | None = None,
) -> Path:
    raw: dict[str, Any] = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["root"] = "data"
    raw["data"]["formal_candidate_pool"] = "pool.csv"
    raw["data"]["manifests"] = {
        "development_train": "manifests/development/development_train.csv",
        "development_validation": "manifests/development/development_validation.csv",
        "formal_train": "manifests/splits/formal_train.csv",
        "internal_validation": "manifests/splits/internal_validation.csv",
        "final_validation": "manifests/splits/final_validation.csv",
    }
    raw["cache"]["root"] = "cache"
    raw["outputs"] = {
        "root": "outputs/modeling/cvae_baseline",
        "development": "outputs/modeling/cvae_baseline/development",
        "benchmarks": "outputs/modeling/cvae_baseline/benchmarks",
        "formal": "outputs/modeling/cvae_baseline/formal",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_manifest(
        tmp_path / raw["data"]["manifests"]["development_train"],
        [_row(f"train-{index:03d}", "development_train") for index in range(train_count)],
    )
    write_manifest(
        tmp_path / raw["data"]["manifests"]["development_validation"],
        [
            _row(f"validation-{index:03d}", "development_validation")
            for index in range(validation_count)
        ],
    )
    write_seed_records(tmp_path / "pool.csv", records or [])
    return config_path


@pytest.fixture(scope="module")
def cvae_schema():
    return build_cvae_schema(REPO_ROOT / "configs/skills")


def test_prepare_builds_atomic_shards_index_hashes_and_default_collatable_dataset(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(
        tmp_path,
        train_count=65,
        validation_count=2,
        records=[
            _record("train-000"),
            _record("train-001", "cut_in_then_brake"),
        ],
    )
    loaded: list[str] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id)

    progress = io.StringIO()
    summary = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=progress,
    )

    train = summary["partitions"]["development_train"]
    validation = summary["partitions"]["development_validation"]
    assert train["status"] == "complete"
    assert train["counts"]["retained_scenarios"] == 65
    assert train["counts"]["retained_samples"] == 66
    assert train["label_counts"]["observed_records"] == 1
    assert train["label_counts"]["compatible_ignored"] == 1
    assert len(train["shards"]) == 2
    assert validation["counts"]["retained_samples"] == 2
    assert len(loaded) == 67
    assert "65/65 scenarios" in progress.getvalue()
    assert "ETA" in progress.getvalue()

    train_dir = tmp_path / "cache/development_train"
    first_shard = torch.load(
        train_dir / train["shards"][0]["path"],
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    assert set(first_shard) == {
        "version",
        "partition",
        "sample_ids",
        "scenario_ids",
        "target_track_ids",
        "map_context_indices",
        "tensors",
        "map_contexts",
    }
    assert first_shard["version"] == CACHE_VERSION
    assert len(first_shard["sample_ids"]) == 65
    assert len(set(first_shard["scenario_ids"])) == 64
    assert all(
        tensor.shape[0] == len(first_shard["sample_ids"])
        for tensor in first_shard["tensors"].values()
    )
    assert first_shard["map_context_indices"].tolist() == [0] * 65
    assert all(tensor.shape[0] == 1 for tensor in first_shard["map_contexts"].values())
    assert train["counts"]["map_contexts"] == 2
    assert train["counts"]["deduplicated_map_sample_copies"] == 64
    assert train["shards"][0]["counts"]["map_contexts"] == 1
    assert train["shards"][0]["counts"]["deduplicated_map_sample_copies"] == 64
    sidecar = json.loads(
        (train_dir / train["shards"][0]["sidecar"]).read_text(encoding="utf-8")
    )
    for name in (
        "config_sha256",
        "manifest_sha256",
        "manifest_rows_sha256",
        "schema_sha256",
        "candidate_pool_sha256",
        "shard_sha256",
    ):
        assert len(sidecar[name]) == 64
    assert sidecar["counts"]["map_contexts"] == 1
    assert sidecar["counts"]["deduplicated_map_sample_copies"] == 64

    index = [
        json.loads(line)
        for line in (train_dir / "sample_index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(index) == 66
    assert sum(item["spec"]["skill_supervision_mask"] for item in index) == 1
    first_shard_entries = [
        item for item in index if item["shard"] == train["shards"][0]["path"]
    ]
    assert [item["offset"] for item in first_shard_entries] == list(range(65))

    dataset = CVAECachedDataset(train_dir, schema=cvae_schema, in_memory_shards=1)
    base_scenario = _scenario("train-000")
    expected_base = tensorize_scenario(
        base_scenario,
        make_base_sample_spec(base_scenario),
        cvae_schema,
    )
    expected_observed = tensorize_scenario(
        base_scenario,
        SampleSpec(
            scenario_id="train-000",
            target_track_id="target",
            skill_id="slow_lead_blockage",
            skill_supervision_mask=True,
            responder_track_id="responder",
            role_track_ids=(
                ("slow_leader", "target"),
                ("follower", "responder"),
            ),
            trigger_score=0.9,
        ),
        cvae_schema,
    )
    _assert_cached_sample_matches_tensorized(dataset[0], expected_base)
    _assert_cached_sample_matches_tensorized(dataset[1], expected_observed)
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    assert batch["actor_history"].shape[:3] == (2, 32, 50)
    assert batch["target_future"].shape == (2, 60, 2)
    assert all(
        isinstance(value, torch.Tensor)
        for key, value in batch.items()
        if key not in {"sample_id", "scenario_id", "target_track_id"}
    )
    assert len(batch["sample_id"]) == 2
    assert len(batch["scenario_id"]) == 2
    assert len(batch["target_track_id"]) == 2

    sampler = ShardShuffleSampler(dataset, seed=2026)
    first_order = list(sampler)
    assert first_order == list(sampler)
    ordered_shards = [dataset.entries[index]["shard"] for index in first_order]
    assert sum(
        first != second
        for first, second in zip(ordered_shards, ordered_shards[1:])
    ) <= len({entry["shard"] for entry in dataset.entries}) - 1
    sampler.set_epoch(1)
    assert list(sampler) != first_order
    second_order = list(sampler)
    sampler.set_range(2, 5)
    assert list(sampler) == second_order[2:5]
    assert len(sampler) == 3
    sampler.set_range()
    assert list(sampler) == second_order
    dataset[0]
    dataset[len(dataset) - 1]
    assert len(dataset._shards) == 1


def test_verified_shards_are_skipped_and_corrupt_shard_is_rebuilt(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(tmp_path, train_count=65, validation_count=1)
    loaded: list[str] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id)

    first = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert len(loaded) == 66
    loaded.clear()

    second = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == []
    assert second["partitions"]["development_train"]["resume"] == {
        "verified_skipped_shards": 2,
        "rebuilt_shards": 0,
    }

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["training"]["batch_size"] *= 2
    config_path.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    training_only_change = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == []
    assert training_only_change["partitions"]["development_train"]["resume"] == {
        "verified_skipped_shards": 2,
        "rebuilt_shards": 0,
    }

    train = first["partitions"]["development_train"]
    shard_path = tmp_path / "cache/development_train" / train["shards"][1]["path"]
    dataset = CVAECachedDataset(
        tmp_path / "cache/development_train",
        schema=cvae_schema,
    )
    shard_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="shard hash differs"):
        dataset[len(dataset) - 1]
    third = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == ["train-064"]
    train_resume = third["partitions"]["development_train"]["resume"]
    assert train_resume == {
        "verified_skipped_shards": 1,
        "rebuilt_shards": 1,
    }


def test_partial_prefix_resumes_complete_shards_before_finishing_manifest(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(tmp_path, train_count=70, validation_count=1)
    loaded: list[str] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id)

    partial = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        limit=64,
        progress_stream=io.StringIO(),
    )
    assert partial["partitions"]["development_train"]["status"] == "partial"
    assert len(loaded) == 65
    loaded.clear()

    complete = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == [f"train-{index:03d}" for index in range(64, 70)]
    train = complete["partitions"]["development_train"]
    assert train["status"] == "complete"
    assert train["resume"]["verified_skipped_shards"] == 1
    assert train["counts"]["retained_scenarios"] == 70


def test_validation_labeler_is_injectable_but_default_validation_is_base_only(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(tmp_path, train_count=1, validation_count=1)

    def loader(path: str | Path) -> Scenario:
        return _scenario(Path(path).parent.name)

    def labeler(scenario: Scenario, schema) -> tuple[SampleSpec, ...]:
        assert "slow_lead_blockage" in schema.formal_skill_ids
        return (
            SampleSpec(
                scenario_id=scenario.scenario_id,
                target_track_id="target",
                skill_id="slow_lead_blockage",
                skill_supervision_mask=True,
                responder_track_id="responder",
                role_track_ids=(
                    ("slow_leader", "target"),
                    ("follower", "responder"),
                ),
            ),
        )

    base = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert base["partitions"]["development_validation"]["counts"]["retained_samples"] == 1

    labeled = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        validation_labeler=labeler,
        force=True,
        progress_stream=io.StringIO(),
    )
    assert labeled["partitions"]["development_validation"]["counts"]["retained_samples"] == 2


def test_distinct_anchor_map_content_uses_distinct_contexts_and_getitem_is_equivalent(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(tmp_path, train_count=2, validation_count=1)

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        target_offset_y = 10.0 if scenario_id == "train-001" else 0.0
        return _scenario(scenario_id, target_offset_y=target_offset_y)

    summary = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    train = summary["partitions"]["development_train"]
    shard_path = tmp_path / "cache/development_train" / train["shards"][0]["path"]
    payload = torch.load(
        shard_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    assert payload["map_context_indices"].tolist() == [0, 1]
    assert all(tensor.shape[0] == 2 for tensor in payload["map_contexts"].values())
    assert train["counts"]["map_contexts"] == 2
    assert train["counts"]["deduplicated_map_sample_copies"] == 0

    dataset = CVAECachedDataset(
        tmp_path / "cache/development_train",
        schema=cvae_schema,
    )
    for index, offset_y in enumerate((0.0, 10.0)):
        scenario = _scenario(f"train-{index:03d}", target_offset_y=offset_y)
        expected = tensorize_scenario(
            scenario,
            make_base_sample_spec(scenario),
            cvae_schema,
        )
        _assert_cached_sample_matches_tensorized(dataset[index], expected)


def test_map_clip_statistics_are_aggregated_by_shard_cache_and_skill_and_verified_on_resume(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(
        tmp_path,
        train_count=2,
        validation_count=1,
        records=[_record("train-001")],
    )
    loaded: list[str] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id, dense_map=scenario_id == "train-001")

    first = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    dense = {
        "eligible_polylines": 130,
        "retained_polylines": 128,
        "dropped_polylines_due_to_limit": 2,
        "original_in_radius_points": 3250,
        "retained_in_radius_points": 3200,
        "resampled_polylines_due_to_point_limit": 128,
        "excess_input_points_over_point_limit": 640,
    }
    expected = {
        "limits": {
            "radius_m": 100.0,
            "max_polylines": 128,
            "max_points_per_polyline": 20,
        },
        "samples": 3,
        "totals": {
            "eligible_polylines": 261,
            "retained_polylines": 257,
            "dropped_polylines_due_to_limit": 4,
            "original_in_radius_points": 6502,
            "retained_in_radius_points": 6402,
            "resampled_polylines_due_to_point_limit": 256,
            "excess_input_points_over_point_limit": 1280,
        },
        "maxima": dense,
        "samples_hitting_polyline_limit": 2,
        "samples_hitting_point_limit": 2,
        "by_skill": {
            "<none>": {
                "samples": 2,
                "totals": {
                    "eligible_polylines": 131,
                    "retained_polylines": 129,
                    "dropped_polylines_due_to_limit": 2,
                    "original_in_radius_points": 3252,
                    "retained_in_radius_points": 3202,
                    "resampled_polylines_due_to_point_limit": 128,
                    "excess_input_points_over_point_limit": 640,
                },
                "maxima": dense,
                "samples_hitting_polyline_limit": 1,
                "samples_hitting_point_limit": 1,
            },
            "slow_lead_blockage": {
                "samples": 1,
                "totals": dense,
                "maxima": dense,
                "samples_hitting_polyline_limit": 1,
                "samples_hitting_point_limit": 1,
            },
        },
    }
    train = first["partitions"]["development_train"]
    assert train["map_clip_statistics"] == expected
    assert train["shards"][0]["map_clip_statistics"] == expected
    sidecar_path = (
        tmp_path / "cache/development_train" / train["shards"][0]["sidecar"]
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["map_clip_statistics"] == expected
    assert sidecar["samples"][0]["map_clip_statistics"] == {
        "eligible_polylines": 1,
        "retained_polylines": 1,
        "dropped_polylines_due_to_limit": 0,
        "original_in_radius_points": 2,
        "retained_in_radius_points": 2,
        "resampled_polylines_due_to_point_limit": 0,
        "excess_input_points_over_point_limit": 0,
    }
    assert sidecar["samples"][1]["map_clip_statistics"] == dense
    assert sidecar["samples"][2]["map_clip_statistics"] == dense

    loaded.clear()
    resumed = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == []
    assert resumed["partitions"]["development_train"]["resume"] == {
        "verified_skipped_shards": 1,
        "rebuilt_shards": 0,
    }
    assert resumed["partitions"]["development_train"]["map_clip_statistics"] == expected

    sidecar["samples"][0]["map_clip_statistics"]["eligible_polylines"] = 2
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rebuilt = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    assert loaded == ["train-000", "train-001"]
    assert rebuilt["partitions"]["development_train"]["resume"] == {
        "verified_skipped_shards": 0,
        "rebuilt_shards": 1,
    }
    assert rebuilt["partitions"]["development_train"]["map_clip_statistics"] == expected


@pytest.mark.parametrize(
    "field",
    (
        "detection_mode",
        "detection_thresholds",
        "feasibility",
        "target_risk_definition",
    ),
)
def test_training_pool_skill_contract_drift_is_rejected_without_final_validation_access(
    tmp_path: Path,
    cvae_schema,
    field: str,
) -> None:
    record = _record("outside-selected-manifests")
    if field == "target_risk_definition":
        record = replace(
            record,
            target_risk_definition={
                **record.target_risk_definition,
                "target_range": [4.0, 16.0],
            },
        )
    else:
        evidence = dict(record.evidence)
        evidence[field] = {
            "drifted_threshold": {"value": 1.0, "source": "semantic"}
        } if field == "detection_thresholds" else "drifted"
        record = replace(record, evidence=evidence)
    config_path = _write_project(
        tmp_path,
        train_count=1,
        validation_count=1,
        records=[record],
    )
    final_validation = tmp_path / "manifests/splits/final_validation.csv"
    assert not final_validation.exists()

    with pytest.raises(ValueError, match=rf"field {field} differs"):
        run_preparation(
            config_path=config_path,
            split="development",
            project_root=tmp_path,
            schema=cvae_schema,
            scenario_loader=lambda path: pytest.fail(f"unexpected scenario read: {path}"),
            progress_stream=io.StringIO(),
        )
    assert not final_validation.exists()


def test_sample_spec_statistics_are_per_skill_and_stable_across_resume(
    tmp_path: Path,
    cvae_schema,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_project(tmp_path, train_count=1, validation_count=1)

    def loader(path: str | Path) -> Scenario:
        return _scenario(Path(path).parent.name)

    def labeler(scenario: Scenario, schema) -> tuple[SampleSpec, ...]:
        assert {"slow_lead_blockage", "lead_sudden_stop"}.issubset(
            schema.formal_skill_ids
        )
        return (
            SampleSpec(
                scenario_id=scenario.scenario_id,
                target_track_id="target",
                skill_id="slow_lead_blockage",
                skill_supervision_mask=True,
                responder_track_id="responder",
                role_track_ids=(("slow_leader", "target"), ("follower", "responder")),
            ),
            SampleSpec(
                scenario_id=scenario.scenario_id,
                target_track_id="target",
                skill_id="lead_sudden_stop",
                skill_supervision_mask=True,
                responder_track_id="responder",
                role_track_ids=(("stopping_leader", "target"), ("follower", "responder")),
            ),
        )

    original_tensorize = cvae_cache.tensorize_scenario

    def tensorize_with_one_rejection(scenario, spec, schema):
        if spec.skill_id == "slow_lead_blockage":
            raise ValueError("forced skill-specific rejection")
        return original_tensorize(scenario, spec, schema)

    monkeypatch.setattr(cvae_cache, "tensorize_scenario", tensorize_with_one_rejection)
    first = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        validation_labeler=labeler,
        progress_stream=io.StringIO(),
    )
    expected_statistics = {
        "totals": {"candidate": 3, "retained": 2, "rejected": 1},
        "by_skill": {
            "<none>": {
                "candidate": 1,
                "retained": 1,
                "rejected": 0,
                "rejection_reasons": {},
            },
            "lead_sudden_stop": {
                "candidate": 1,
                "retained": 1,
                "rejected": 0,
                "rejection_reasons": {},
            },
            "slow_lead_blockage": {
                "candidate": 1,
                "retained": 0,
                "rejected": 1,
                "rejection_reasons": {
                    "sample ValueError: forced skill-specific rejection": 1
                },
            },
        },
    }
    validation = first["partitions"]["development_validation"]
    assert validation["sample_spec_statistics"] == expected_statistics
    assert validation["shards"][0]["sample_spec_statistics"] == expected_statistics
    sidecar_path = (
        tmp_path
        / "cache/development_validation"
        / validation["shards"][0]["sidecar"]
    )
    sidecar_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_bytes)
    assert sidecar["sample_spec_statistics"] == expected_statistics

    second = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        validation_labeler=labeler,
        progress_stream=io.StringIO(),
    )
    resumed_validation = second["partitions"]["development_validation"]
    assert resumed_validation["resume"] == {
        "verified_skipped_shards": 1,
        "rebuilt_shards": 0,
    }
    assert resumed_validation["sample_spec_statistics"] == expected_statistics
    assert sidecar_path.read_bytes() == sidecar_bytes


def test_old_cache_version_is_rejected_until_forced_rebuild(
    tmp_path: Path,
    cvae_schema,
) -> None:
    config_path = _write_project(tmp_path, train_count=1, validation_count=1)

    def loader(path: str | Path) -> Scenario:
        return _scenario(Path(path).parent.name)

    run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        progress_stream=io.StringIO(),
    )
    cache_dir = tmp_path / "cache/development_train"
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = CACHE_VERSION - 1
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cache_manifest version is incompatible"):
        CVAECachedDataset(cache_dir, schema=cvae_schema)
    with pytest.raises(ValueError, match="cache version is incompatible"):
        run_preparation(
            config_path=config_path,
            split="development",
            project_root=tmp_path,
            schema=cvae_schema,
            scenario_loader=loader,
            progress_stream=io.StringIO(),
        )

    rebuilt = run_preparation(
        config_path=config_path,
        split="development",
        project_root=tmp_path,
        schema=cvae_schema,
        scenario_loader=loader,
        force=True,
        progress_stream=io.StringIO(),
    )
    assert rebuilt["partitions"]["development_train"]["version"] == CACHE_VERSION
    assert len(CVAECachedDataset(cache_dir, schema=cvae_schema)) == 1

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from skilldrive.filtering.common import derive_future_kinematics
from skilldrive.filtering.reference_kinematics import (
    METRIC_NAMES,
    PARQUET_COLUMNS,
    REFERENCE_TIMESTEPS,
    ReferenceProgress,
    build_kinematic_reference,
    derive_track_kinematic_samples,
    load_formal_train_rows,
    scan_scenario_parquet,
)
from skilldrive.schemas import AgentTrack, Scenario
from scripts.generation.build_kinematic_reference import _ProgressPrinter


HEADING_SPEED_POLICY = {
    "vehicle": 0.5,
    "bus": 0.5,
    "motorcyclist": 0.5,
    "cyclist": 0.3,
    "pedestrian": 0.2,
}
FIELDNAMES = ["scenario_id", "split", "source_path", "city_name", "selected_reason"]


def _write_manifest(project_root: Path, scenario_ids: list[str]) -> Path:
    path = project_root / "manifests/splits/formal_train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for scenario_id in scenario_ids:
            writer.writerow(
                {
                    "scenario_id": scenario_id,
                    "split": "train",
                    "source_path": (
                        f"train/{scenario_id}/scenario_{scenario_id}.parquet"
                    ),
                    "city_name": "PIT",
                    "selected_reason": "unit_test",
                }
            )
    return path


def _write_filter_config(
    project_root: Path,
    *,
    policy: dict[str, float] | None = None,
) -> Path:
    values = HEADING_SPEED_POLICY if policy is None else policy
    path = project_root / "configs/generation/filters_v1.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kinematics": {
            "actor_types": {
                object_type: {"minimum_heading_speed_mps": value}
                for object_type, value in values.items()
            }
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_scenario(
    project_root: Path,
    scenario_id: str,
    *,
    speed_offset: float = 0.0,
) -> Path:
    path = (
        project_root
        / "data/av2/motion-forecasting/train"
        / scenario_id
        / f"scenario_{scenario_id}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    timestep = np.arange(110, dtype=np.int64)
    vehicle_speed = 1.0 + speed_offset
    pedestrian_speed = 0.25
    vehicle_x = timestep.astype(np.float64) * 0.1 * vehicle_speed
    pedestrian_x = timestep.astype(np.float64) * 0.1 * pedestrian_speed
    table = pa.table(
        {
            "track_id": ["vehicle-track"] * 110 + ["pedestrian-track"] * 110,
            "object_type": ["vehicle"] * 110 + ["pedestrian"] * 110,
            "timestep": np.concatenate((timestep, timestep)),
            "position_x": np.concatenate((vehicle_x, pedestrian_x)),
            "position_y": np.zeros(220, dtype=np.float64),
            "heading": np.zeros(220, dtype=np.float64),
            "velocity_x": np.concatenate(
                (
                    np.full(110, vehicle_speed, dtype=np.float64),
                    np.full(110, pedestrian_speed, dtype=np.float64),
                )
            ),
            "velocity_y": np.zeros(220, dtype=np.float64),
            "start_timestamp": np.full(220, 315_990_334_959_769_000.0),
            "unused_large_column": ["not-read"] * 220,
        }
    )
    pq.write_table(table, path)
    return path


def _output_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_track_derivation_matches_production_future_kinematics() -> None:
    timestamps = np.arange(110, dtype=np.int64) * 100_000_000
    positions = np.zeros((110, 2), dtype=np.float64)
    positions[:50, 0] = np.arange(50, dtype=np.float64) * 0.1
    speeds = np.linspace(1.0, 2.5, 60)
    angles = np.linspace(0.0, 0.6, 60)
    future_velocity = np.column_stack(
        (speeds * np.cos(angles), speeds * np.sin(angles))
    )
    positions[50:] = positions[49] + np.cumsum(future_velocity * 0.1, axis=0)
    velocities = np.tile([1.0, 0.0], (110, 1)).astype(np.float64)
    velocities[48] = [0.8, 0.0]
    velocities[49] = [1.0, 0.0]
    headings = np.zeros(110, dtype=np.float64)
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    scenario = Scenario(
        scenario_id="scene",
        city_name="PIT",
        timestamps=timestamps,
        focal_track_id="target",
        agents=[
            AgentTrack(
                track_id="target",
                object_type="vehicle",
                positions=positions,
                velocities=velocities,
                headings=headings,
                observed_mask=observed,
                is_focal=True,
            )
        ],
        map_polylines=[],
    )
    production = derive_future_kinematics(
        scenario,
        "target",
        positions[50:].astype(np.float32),
        minimum_heading_speed_mps=0.5,
    )
    reference = derive_track_kinematic_samples(
        REFERENCE_TIMESTEPS,
        positions[48:],
        velocities[48:],
        headings[48:],
        low_speed_threshold_mps=0.5,
    )

    assert reference.valid_reference_window
    np.testing.assert_array_equal(reference.speed_mps, production.speed_mps)
    np.testing.assert_array_equal(
        reference.positive_acceleration_mps2,
        np.maximum(production.tangential_acceleration_mps2, 0.0),
    )
    np.testing.assert_array_equal(reference.deceleration_mps2, production.deceleration_mps2)
    np.testing.assert_array_equal(reference.jerk_mps3, production.jerk_mps3)
    np.testing.assert_array_equal(reference.yaw_rate_radps, production.heading_rate_rad_s)
    np.testing.assert_array_equal(reference.curvature_inv_m, production.curvature_per_m)


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("nan"), True])
def test_track_derivation_requires_positive_heading_speed_threshold(
    threshold: float,
) -> None:
    positions = np.zeros((62, 2), dtype=np.float64)
    velocities = np.zeros((62, 2), dtype=np.float64)
    headings = np.zeros(62, dtype=np.float64)

    with pytest.raises(ValueError, match="positive finite"):
        derive_track_kinematic_samples(
            REFERENCE_TIMESTEPS,
            positions,
            velocities,
            headings,
            low_speed_threshold_mps=threshold,
        )


@pytest.mark.parametrize(
    ("split", "source_path", "message"),
    [
        ("val", "train/scene/scenario_scene.parquet", "split=train"),
        ("train", "val/scene/scenario_scene.parquet", "source_path must be"),
        ("train", "../train/scene/scenario_scene.parquet", "source_path must be"),
    ],
)
def test_formal_manifest_rejects_any_non_train_scope(
    tmp_path: Path,
    split: str,
    source_path: str,
    message: str,
) -> None:
    manifest = _write_manifest(tmp_path, ["scene"])
    rows = list(csv.DictReader(manifest.read_text(encoding="utf-8").splitlines()))
    rows[0]["split"] = split
    rows[0]["source_path"] = source_path
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        load_formal_train_rows(tmp_path, expected_scenario_count=1)


def test_scenario_scan_uses_parquet_file_and_only_the_frozen_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_scenario(tmp_path, "scene")
    requested: list[tuple[str, ...]] = []
    original = pq.ParquetFile

    class RecordingParquetFile:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.inner = original(*args, **kwargs)

        def read(self, *args: object, **kwargs: object) -> pa.Table:
            requested.append(tuple(kwargs["columns"]))
            return self.inner.read(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", RecordingParquetFile)
    result = scan_scenario_parquet(
        "scene",
        path,
        minimum_heading_speed_mps_by_actor_type=HEADING_SPEED_POLICY,
    )

    assert requested == [PARQUET_COLUMNS]
    assert result["statistics"]["scenario_count"] == 1
    assert result["statistics"]["quality_counts"]["eligible_tracks"] == 2
    assert result["statistics"]["quality_counts"]["reference_window_tracks"] == 2
    vehicle = result["statistics"]["categories"]["vehicle"]
    assert vehicle["track_count"] == 1
    assert vehicle["reference_window_count"] == 1
    assert vehicle["metrics"]["speed_mps"]["count"] == 60
    assert all(
        vehicle["window_max_metrics"][name]["count"] == 1
        for name in METRIC_NAMES
    )
    assert vehicle["window_max_metrics"]["speed_mps"]["maximum"] == pytest.approx(
        1.0,
        abs=1.0e-5,
    )


def test_build_is_byte_deterministic_for_one_and_multiple_workers(
    tmp_path: Path,
) -> None:
    scenario_ids = ["scene-a", "scene-b", "scene-c"]
    _write_manifest(tmp_path, scenario_ids)
    _write_filter_config(tmp_path)
    for index, scenario_id in enumerate(scenario_ids):
        _write_scenario(tmp_path, scenario_id, speed_offset=index * 0.25)

    one_root = tmp_path / "outputs/one-worker"
    multi_root = tmp_path / "outputs/multi-worker"
    one = build_kinematic_reference(
        tmp_path,
        output_root=one_root,
        workers=1,
        expected_scenario_count=3,
        shard_size=2,
    )
    multi = build_kinematic_reference(
        tmp_path,
        output_root=multi_root,
        workers=2,
        expected_scenario_count=3,
        shard_size=2,
    )

    assert one.complete and multi.complete
    assert _output_bytes(one_root) == _output_bytes(multi_root)
    summary = json.loads((one_root / "summary.json").read_text(encoding="ascii"))
    assert summary["source"]["source_split"] == "train"
    assert summary["source"]["validation_manifests_opened"] is False
    assert summary["statistics"]["scenario_count"] == 3
    assert summary["derivation"]["minimum_heading_speed_mps_by_actor_type"] == (
        HEADING_SPEED_POLICY
    )
    vehicle = summary["statistics"]["categories"]["vehicle"]
    assert set(vehicle) == {
        "track_count",
        "row_count",
        "reference_window_count",
        "point_distributions",
        "window_max_distributions",
    }
    assert vehicle["point_distributions"]["speed_mps"]["sample_count"] == 180
    assert vehicle["window_max_distributions"]["speed_mps"]["sample_count"] == 3
    assert summary["derivation"]["distribution_scopes"] == {
        "point_distributions": (
            "all 60 candidate-equivalent values from each complete window"
        ),
        "window_max_distributions": (
            "one maximum over the 60 candidate-equivalent values per complete window"
        ),
    }


def test_bounded_run_resumes_valid_shards_and_repairs_corruption(
    tmp_path: Path,
) -> None:
    scenario_ids = ["scene-a", "scene-b", "scene-c"]
    _write_manifest(tmp_path, scenario_ids)
    _write_filter_config(tmp_path)
    for index, scenario_id in enumerate(scenario_ids):
        _write_scenario(tmp_path, scenario_id, speed_offset=index * 0.25)

    output_root = tmp_path / "outputs/resume"
    partial = build_kinematic_reference(
        tmp_path,
        output_root=output_root,
        workers=1,
        expected_scenario_count=3,
        shard_size=2,
        max_new_shards=1,
    )
    first_shard = output_root / "shards/shard-00000.json"
    first_mtime = first_shard.stat().st_mtime_ns
    assert not partial.complete
    assert partial.completed_shards == 1
    assert not (output_root / "summary.json").exists()

    complete = build_kinematic_reference(
        tmp_path,
        output_root=output_root,
        workers=2,
        expected_scenario_count=3,
        shard_size=2,
    )
    assert complete.complete
    assert complete.new_shards == 1
    assert first_shard.stat().st_mtime_ns == first_mtime
    expected = _output_bytes(output_root)

    corrupt = output_root / "shards/shard-00001.json"
    corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")
    repaired = build_kinematic_reference(
        tmp_path,
        output_root=output_root,
        workers=1,
        expected_scenario_count=3,
        shard_size=2,
    )

    assert repaired.complete
    assert repaired.new_shards == 1
    assert _output_bytes(output_root) == expected


def test_restart_rejects_project_outputs_root_and_external_paths(
    tmp_path: Path,
) -> None:
    outputs_root = tmp_path / "outputs"
    outputs_root.mkdir()
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    markers = []
    for index, path in enumerate((tmp_path, outputs_root, external)):
        marker = path / f"keep-{index}.txt"
        marker.write_text("keep", encoding="utf-8")
        markers.append(marker)

    for path in (tmp_path, outputs_root, external):
        with pytest.raises(ValueError, match="dedicated directory below"):
            build_kinematic_reference(
                tmp_path,
                output_root=path,
                expected_scenario_count=1,
                restart=True,
            )

    assert all(marker.read_text(encoding="utf-8") == "keep" for marker in markers)


def test_restart_requires_owner_sentinel_and_preserves_unmanaged_files(
    tmp_path: Path,
) -> None:
    scenario_id = "scene"
    _write_manifest(tmp_path, [scenario_id])
    _write_filter_config(tmp_path)
    _write_scenario(tmp_path, scenario_id)
    output_root = tmp_path / "outputs/unmanaged"
    output_root.mkdir(parents=True)
    marker = output_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="not owned"):
        build_kinematic_reference(
            tmp_path,
            output_root=output_root,
            expected_scenario_count=1,
            restart=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_owned_output_can_restart_without_preserving_stale_artifacts(
    tmp_path: Path,
) -> None:
    scenario_id = "scene"
    _write_manifest(tmp_path, [scenario_id])
    _write_filter_config(tmp_path)
    _write_scenario(tmp_path, scenario_id)
    output_root = tmp_path / "outputs/owned"
    first = build_kinematic_reference(
        tmp_path,
        output_root=output_root,
        expected_scenario_count=1,
    )
    assert first.complete
    stale = output_root / "stale.txt"
    stale.write_text("stale", encoding="utf-8")

    restarted = build_kinematic_reference(
        tmp_path,
        output_root=output_root,
        expected_scenario_count=1,
        restart=True,
    )

    assert restarted.complete
    assert not stale.exists()
    assert (output_root / ".kinematic-reference-owner.json").is_file()


def test_filter_policy_rejects_nonpositive_actor_threshold(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, ["scene"])
    invalid = dict(HEADING_SPEED_POLICY)
    invalid["pedestrian"] = 0.0
    _write_filter_config(tmp_path, policy=invalid)

    with pytest.raises(ValueError, match="pedestrian"):
        build_kinematic_reference(
            tmp_path,
            output_root=tmp_path / "outputs/reference",
            expected_scenario_count=1,
        )


def test_progress_reports_preflight_before_lazy_missing_file_failure(
    tmp_path: Path,
) -> None:
    _write_manifest(tmp_path, ["missing-scene"])
    _write_filter_config(tmp_path)
    values: list[ReferenceProgress] = []

    with pytest.raises(FileNotFoundError, match="missing Formal Train scenario"):
        build_kinematic_reference(
            tmp_path,
            output_root=tmp_path / "outputs/reference",
            expected_scenario_count=1,
            progress=values.append,
        )

    assert [value.phase for value in values] == ["preflight", "scan"]


def test_progress_printer_eta_starts_at_first_scan_update(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    moments = iter((100.0, 100.0, 110.0))
    monkeypatch.setattr(
        "scripts.generation.build_kinematic_reference.time.monotonic",
        lambda: next(moments),
    )
    printer = _ProgressPrinter()
    printer(ReferenceProgress("preflight", 0, 2, 0, 500, 0))
    printer(ReferenceProgress("scan", 0, 2, 0, 500, 0))
    printer(ReferenceProgress("scan", 1, 2, 250, 500, 1))

    output = capsys.readouterr().out
    assert "kinematics preflight" in output
    assert "25.0 scenarios/s" in output
    assert "ETA 00:10" in output


def test_filtering_lazy_public_exports_remain_import_compatible() -> None:
    import skilldrive.filtering as filtering

    expected = {
        "FilterCheck",
        "FilterDecision",
        "FilterRejection",
        "FilterStage",
        "FutureKinematics",
        "KinematicLimits",
        "ProxyCollisionContact",
        "ProxyCollisionReport",
        "check_kinematics",
        "check_proxy_collisions",
        "check_schema_and_finite",
        "derive_future_kinematics",
        "detect_synchronized_proxy_collisions",
        "oriented_boxes_overlap",
        "validate_observed_skill",
    }
    assert set(filtering.__all__) == expected
    assert all(getattr(filtering, name) is not None for name in expected)

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import scripts.seed_detection.detect_seeds as detect_seeds
import skilldrive.skills.detection as skill_detection
from scripts.seed_detection.detect_seeds import DEFAULT_OUTPUTS, run_scan
from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds import SeedRecord, read_seed_records
from skilldrive.skills.detection import DetectionRun


SKILL_DIR = Path("configs/skills")
CONFIG_PATH = Path("configs/seed_detection.yaml")


def test_formal_scan_defaults_write_an_ignored_candidate_pool() -> None:
    assert DEFAULT_OUTPUTS["development"] == (
        Path("outputs/seed_detection/development_candidate_pool.csv"),
        Path("outputs/seed_detection/development_summary.json"),
    )
    assert DEFAULT_OUTPUTS["formal"] == (
        Path("outputs/seed_detection/formal_candidate_pool.csv"),
        Path("outputs/seed_detection/formal_pool_summary.json"),
    )


def _manifest_row(scenario_id: str, split: str = "development_train") -> ManifestRow:
    return ManifestRow(
        scenario_id=scenario_id,
        split=split,
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        city_name="unknown_until_loaded",
        selected_reason="test",
    )


def _scenario(scenario_id: str) -> Scenario:
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [1.0, 0.0]])
    focal = AgentTrack(
        track_id="vehicle-id",
        object_type="vehicle",
        positions=positions,
        velocities=velocities,
        headings=np.zeros(2),
        observed_mask=np.ones(2, dtype=bool),
        is_focal=True,
    )
    responder = AgentTrack(
        track_id="pedestrian-id",
        object_type="pedestrian",
        positions=positions + np.array([0.0, 2.0]),
        velocities=velocities,
        headings=np.zeros(2),
        observed_mask=np.ones(2, dtype=bool),
    )
    return Scenario(
        scenario_id=scenario_id,
        city_name=f"city-{scenario_id[-1]}",
        timestamps=np.arange(2, dtype=np.int64),
        focal_track_id=focal.track_id,
        agents=[focal, responder],
        map_polylines=[],
    )


def _detector(scenario, skills, config) -> DetectionRun:
    target_risk_definition = skills[0].risk_definition
    target_metric_observed = scenario.scenario_id.endswith("a")
    record = SeedRecord(
        scenario_id=scenario.scenario_id,
        skill_id=skills[0].skill_id,
        initiator_track_id="vehicle-id",
        responder_track_id="pedestrian-id",
        role_track_ids={
            "initiator": "vehicle-id",
            "responder": "pedestrian-id",
        },
        trigger_score=0.75,
        seed_risk_metric=(
            target_risk_definition["metric"]
            if target_metric_observed
            else "minimum_trajectory_distance"
        ),
        seed_risk_value=2.0 if target_metric_observed else 4.0,
        target_risk_definition=target_risk_definition,
        source_path="will-be-normalized",
        evidence={"matched": True},
        sampled_parameters={"test_parameter": config.global_seed},
    )
    return DetectionRun(
        records=[record],
        rejection_counts=Counter({"no_compatible_pair": len(skills) - 1}),
    )


def _parallel_loader(path: str | Path) -> Scenario:
    scenario_id = Path(path).parent.name
    if scenario_id.endswith("a"):
        time.sleep(0.1)
    return _scenario(scenario_id)


def _parallel_detector(scenario, skills, config) -> DetectionRun:
    return _detector(scenario, skills, config)


def _failing_parallel_detector(scenario, skills, config) -> DetectionRun:
    if scenario.scenario_id.endswith("c"):
        raise RuntimeError("parallel detector failure")
    return _detector(scenario, skills, config)


def _parallel_arguments(tmp_path: Path, rows: list[ManifestRow]) -> dict[str, object]:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, rows)
    internal = tmp_path / "internal_validation.csv"
    final = tmp_path / "final_validation.csv"
    write_manifest(internal, [])
    write_manifest(final, [])
    return {
        "manifest_path": manifest,
        "data_root": tmp_path / "data",
        "skill_dir": SKILL_DIR,
        "config_path": CONFIG_PATH,
        "output_csv": tmp_path / "candidates.csv",
        "summary_json": tmp_path / "summary.json",
        "checkpoint_path": tmp_path / "checkpoint.jsonl",
        "progress_every": 1,
        "internal_validation_manifest": internal,
        "final_validation_manifest": final,
    }


def _checkpoint_scenario_ids(path: Path) -> list[str]:
    values = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    return [value["scenario_id"] for value in values[1:]]


def test_scan_resumes_after_partial_checkpoint_and_builds_summary(
    tmp_path, capsys
) -> None:
    manifest = tmp_path / "development_train.csv"
    rows = [_manifest_row("scene-b"), _manifest_row("scene-a")]
    write_manifest(manifest, rows)
    loaded: list[str] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id)

    output_csv = tmp_path / "candidates.csv"
    summary_json = tmp_path / "summary.json"
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = run_scan(
        manifest_path=manifest,
        data_root=tmp_path / "data",
        skill_dir=SKILL_DIR,
        config_path=CONFIG_PATH,
        output_csv=output_csv,
        summary_json=summary_json,
        checkpoint_path=checkpoint,
        limit=1,
        progress_every=1,
        scenario_loader=loader,
        detector=_detector,
    )
    assert first["counts"]["processed_scenarios"] == 1
    assert first["status"] == "partial"
    assert loaded == ["scene-b"]

    with checkpoint.open("ab") as handle:
        handle.write(b'{"kind":')

    second = run_scan(
        manifest_path=manifest,
        data_root=tmp_path / "data",
        skill_dir=SKILL_DIR,
        config_path=CONFIG_PATH,
        output_csv=output_csv,
        summary_json=summary_json,
        checkpoint_path=checkpoint,
        limit=2,
        progress_every=1,
        scenario_loader=loader,
        detector=_detector,
    )

    assert loaded == ["scene-b", "scene-a"]
    assert second["status"] == "complete"
    assert second["counts"] == {
        "processed_scenarios": 2,
        "resumed_scenarios_this_run": 1,
        "new_scenarios_this_run": 1,
        "candidates": 2,
        "unique_candidate_scenarios": 2,
    }
    assert second["skill_hits"]["lead_hard_brake"] == 2
    assert second["skill_scenario_hits"]["lead_hard_brake"] == 2
    assert second["rejection_reasons"] == {"no_compatible_pair": 58}
    assert second["actor_distribution"]["pair"] == {"vehicle|pedestrian": 2}
    assert second["actor_distribution"]["by_role"] == {
        "initiator": {"vehicle": 2},
        "responder": {"pedestrian": 2},
    }
    assert second["schema_version"] == 2
    assert second["target_risk_definitions"]["lead_hard_brake"] == {
        "metric": "time_to_collision",
        "target_range": [1.0, 4.0],
        "source": "reference",
        "direction": "lower_is_riskier",
    }
    assert second["seed_risk_distribution"]["by_metric"]["time_to_collision"] == {
        "count": 1,
        "min": 2.0,
        "p25": 2.0,
        "median": 2.0,
        "p75": 2.0,
        "max": 2.0,
        "mean": 2.0,
    }
    assert second["seed_risk_distribution"]["by_skill_and_metric"][
        "lead_hard_brake"
    ] == {
        "minimum_trajectory_distance": {
            "count": 1,
            "min": 4.0,
            "p25": 4.0,
            "median": 4.0,
            "p75": 4.0,
            "max": 4.0,
            "mean": 4.0,
        },
        "time_to_collision": {
            "count": 1,
            "min": 2.0,
            "p25": 2.0,
            "median": 2.0,
            "p75": 2.0,
            "max": 2.0,
            "mean": 2.0,
        },
    }
    assert second["seed_risk_distribution"]["relation_counts"] == {
        "target_metric_observation": 1,
        "proxy_metric": 1,
    }
    assert second["seed_risk_distribution"]["relation_by_skill"][
        "lead_hard_brake"
    ] == {
        "target_metric_observation": 1,
        "proxy_metric": 1,
    }
    assert "risk_distribution" not in second
    assert summary_json.is_file()
    records = read_seed_records(output_csv)
    candidate_bytes = output_csv.read_bytes()
    assert [record.scenario_id for record in records] == ["scene-a", "scene-b"]
    assert [record.seed_risk_is_proxy for record in records] == [False, True]
    assert {record.source_path for record in records} == {
        row.source_path for row in rows
    }
    progress = capsys.readouterr().out
    assert "0/1 scenarios" in progress
    assert "1/2 scenarios" in progress
    assert "2/2 scenarios" in progress

    metadata = json.loads(checkpoint.read_text(encoding="ascii").splitlines()[0])
    assert metadata["version"] == 2

    run_scan(
        manifest_path=manifest,
        data_root=tmp_path / "data",
        skill_dir=SKILL_DIR,
        config_path=CONFIG_PATH,
        output_csv=output_csv,
        summary_json=summary_json,
        checkpoint_path=checkpoint,
        limit=2,
        progress_every=1,
        restart=True,
        scenario_loader=loader,
        detector=_detector,
    )
    assert output_csv.read_bytes() == candidate_bytes


def test_checkpoint_rejects_changed_configuration(tmp_path) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a")])
    config = tmp_path / "seed_detection.yaml"
    config.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.jsonl"
    arguments = {
        "manifest_path": manifest,
        "data_root": tmp_path / "data",
        "skill_dir": SKILL_DIR,
        "config_path": config,
        "output_csv": tmp_path / "candidates.csv",
        "summary_json": tmp_path / "summary.json",
        "checkpoint_path": checkpoint,
        "limit": 1,
        "progress_every": 1,
        "scenario_loader": lambda path: _scenario(Path(path).parent.name),
        "detector": _detector,
    }
    run_scan(**arguments)
    config.write_text(
        config.read_text(encoding="utf-8").replace("global_seed: 2026", "global_seed: 2027"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint inputs differ"):
        run_scan(**arguments)


def test_checkpoint_rejects_legacy_schema_version(tmp_path) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a")])
    checkpoint = tmp_path / "checkpoint.jsonl"
    arguments = {
        "manifest_path": manifest,
        "data_root": tmp_path / "data",
        "skill_dir": SKILL_DIR,
        "config_path": CONFIG_PATH,
        "output_csv": tmp_path / "candidates.csv",
        "summary_json": tmp_path / "summary.json",
        "checkpoint_path": checkpoint,
        "scenario_loader": lambda path: _scenario("scene-a"),
        "detector": _detector,
    }
    run_scan(**arguments)
    lines = checkpoint.read_bytes().splitlines(keepends=True)
    metadata = json.loads(lines[0].decode("ascii"))
    metadata["version"] = 1
    checkpoint.write_bytes(detect_seeds._json_line(metadata) + b"".join(lines[1:]))

    with pytest.raises(ValueError, match="checkpoint inputs differ"):
        run_scan(**arguments)


def test_checkpoint_must_be_a_manifest_order_prefix(tmp_path: Path) -> None:
    manifest = tmp_path / "development_train.csv"
    rows = [_manifest_row("scene-a"), _manifest_row("scene-b")]
    write_manifest(manifest, rows)
    checkpoint = tmp_path / "checkpoint.jsonl"
    arguments = {
        "manifest_path": manifest,
        "data_root": tmp_path / "data",
        "skill_dir": SKILL_DIR,
        "config_path": CONFIG_PATH,
        "output_csv": tmp_path / "candidates.csv",
        "summary_json": tmp_path / "summary.json",
        "checkpoint_path": checkpoint,
        "limit": 1,
        "scenario_loader": lambda path: _scenario("scene-a"),
        "detector": _detector,
    }
    run_scan(**arguments)
    lines = checkpoint.read_text(encoding="ascii").splitlines()
    entry = json.loads(lines[1])
    entry["scenario_id"] = "scene-b"
    checkpoint.write_text(
        lines[0] + "\n" + json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="manifest-order prefix"):
        run_scan(**arguments)


def test_scan_rejects_target_risk_definition_that_differs_from_yaml(tmp_path) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a")])

    def mismatched_detector(scenario, skills, config) -> DetectionRun:
        run = _detector(scenario, skills, config)
        record = replace(
            run.records[0],
            target_risk_definition={
                **run.records[0].target_risk_definition,
                "target_range": [2.0, 5.0],
            },
        )
        return DetectionRun(records=[record], rejection_counts=run.rejection_counts)

    with pytest.raises(ValueError, match="differs from lead_hard_brake YAML"):
        run_scan(
            manifest_path=manifest,
            data_root=tmp_path / "data",
            skill_dir=SKILL_DIR,
            config_path=CONFIG_PATH,
            output_csv=tmp_path / "candidates.csv",
            summary_json=tmp_path / "summary.json",
            scenario_loader=lambda path: _scenario("scene-a"),
            detector=mismatched_detector,
        )


def test_formal_scan_requires_explicit_confirmation(tmp_path) -> None:
    manifest = tmp_path / "formal_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a", split="train")])

    with pytest.raises(PermissionError, match="confirm-formal-scan"):
        run_scan(
            manifest_path=manifest,
            data_root=tmp_path / "data",
            skill_dir=SKILL_DIR,
            config_path=CONFIG_PATH,
            output_csv=tmp_path / "candidates.csv",
            summary_json=tmp_path / "summary.json",
            scenario_loader=lambda path: _scenario("scene-a"),
            detector=_detector,
        )


def test_validation_manifest_is_rejected_before_loading(tmp_path) -> None:
    manifest = tmp_path / "development_validation.csv"
    write_manifest(manifest, [_manifest_row("scene-a", split="development_validation")])

    with pytest.raises(ValueError, match="validation manifests are forbidden"):
        run_scan(
            manifest_path=manifest,
            data_root=tmp_path / "data",
            skill_dir=SKILL_DIR,
            config_path=CONFIG_PATH,
            output_csv=tmp_path / "candidates.csv",
            summary_json=tmp_path / "summary.json",
            scenario_loader=lambda path: _scenario("scene-a"),
            detector=_detector,
        )


def test_candidate_manifest_must_be_disjoint_from_validation(tmp_path) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a")])
    internal_validation = tmp_path / "internal_validation.csv"
    write_manifest(
        internal_validation,
        [_manifest_row("scene-a", split="internal_validation")],
    )
    final_validation = tmp_path / "final_validation.csv"
    write_manifest(final_validation, [])

    with pytest.raises(ValueError, match="scenario leakage"):
        run_scan(
            manifest_path=manifest,
            data_root=tmp_path / "data",
            skill_dir=SKILL_DIR,
            config_path=CONFIG_PATH,
            output_csv=tmp_path / "candidates.csv",
            summary_json=tmp_path / "summary.json",
            limit=1,
            internal_validation_manifest=internal_validation,
            final_validation_manifest=final_validation,
            scenario_loader=lambda path: _scenario("scene-a"),
            detector=_detector,
        )


def test_partial_scan_cannot_overwrite_canonical_outputs(tmp_path) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a"), _manifest_row("scene-b")])

    with pytest.raises(ValueError, match="partial scans cannot overwrite"):
        run_scan(
            manifest_path=manifest,
            data_root=tmp_path / "data",
            skill_dir=SKILL_DIR,
            config_path=CONFIG_PATH,
            output_csv=DEFAULT_OUTPUTS["development"][0],
            summary_json=tmp_path / "summary.json",
            limit=1,
            scenario_loader=lambda path: _scenario("scene-a"),
            detector=_detector,
        )


def test_checkpoint_rejects_changed_pipeline_fingerprint(
    tmp_path, monkeypatch
) -> None:
    manifest = tmp_path / "development_train.csv"
    write_manifest(manifest, [_manifest_row("scene-a")])
    checkpoint = tmp_path / "checkpoint.jsonl"
    arguments = {
        "manifest_path": manifest,
        "data_root": tmp_path / "data",
        "skill_dir": SKILL_DIR,
        "config_path": CONFIG_PATH,
        "output_csv": tmp_path / "candidates.csv",
        "summary_json": tmp_path / "summary.json",
        "checkpoint_path": checkpoint,
        "scenario_loader": lambda path: _scenario("scene-a"),
        "detector": _detector,
    }
    monkeypatch.setattr(detect_seeds, "_pipeline_fingerprint", lambda root: "first")
    run_scan(**arguments)
    monkeypatch.setattr(detect_seeds, "_pipeline_fingerprint", lambda root: "second")

    with pytest.raises(ValueError, match="checkpoint inputs differ"):
        run_scan(**arguments)


@pytest.mark.parametrize("workers", [0, -1, 1.5, True])
def test_workers_must_be_a_positive_integer(tmp_path: Path, workers: object) -> None:
    arguments = _parallel_arguments(tmp_path, [_manifest_row("scene-a")])
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        run_scan(**arguments, workers=workers)  # type: ignore[arg-type]


def test_parallel_scan_rejects_custom_loader_or_detector(tmp_path: Path) -> None:
    arguments = _parallel_arguments(tmp_path, [_manifest_row("scene-a")])
    with pytest.raises(ValueError, match="default scenario_loader and detector"):
        run_scan(**arguments, workers=2, scenario_loader=_parallel_loader)
    with pytest.raises(ValueError, match="default scenario_loader and detector"):
        run_scan(**arguments, workers=2, detector=_parallel_detector)


def test_parallel_default_av2_loader_preloads_before_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _parallel_arguments(tmp_path, [_manifest_row("scene-a")])
    events: list[str] = []

    def fake_preload() -> None:
        events.append("preload")

    class StoppingExecutor:
        def __init__(self, *args, **kwargs) -> None:
            events.append("executor")
            raise RuntimeError("stop after executor construction")

    monkeypatch.setattr(
        detect_seeds,
        "preload_av2_worker_dependencies",
        fake_preload,
    )
    monkeypatch.setattr(detect_seeds, "ProcessPoolExecutor", StoppingExecutor)

    with pytest.raises(RuntimeError, match="stop after executor construction"):
        run_scan(**arguments, workers=2)

    assert events == ["preload", "executor"]


def test_parallel_scan_matches_serial_and_writes_manifest_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [_manifest_row(f"scene-{suffix}") for suffix in ("a", "b", "c", "d")]
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    serial_dir.mkdir()
    parallel_dir.mkdir()
    serial_arguments = _parallel_arguments(serial_dir, rows)
    parallel_arguments = _parallel_arguments(parallel_dir, rows)

    serial = run_scan(
        **serial_arguments,
        workers=1,
        scenario_loader=_parallel_loader,
        detector=_parallel_detector,
    )
    monkeypatch.setattr(detect_seeds, "load_av2_scenario", _parallel_loader)
    monkeypatch.setattr(
        detect_seeds,
        "preload_av2_worker_dependencies",
        lambda: pytest.fail("patched default loader must not preload real AV2"),
    )
    monkeypatch.setattr(skill_detection, "detect_scenario", _parallel_detector)
    parallel = run_scan(**parallel_arguments, workers=2)

    serial_csv = Path(serial_arguments["output_csv"])
    parallel_csv = Path(parallel_arguments["output_csv"])
    assert parallel_csv.read_bytes() == serial_csv.read_bytes()
    assert read_seed_records(parallel_csv) == read_seed_records(serial_csv)
    assert parallel["counts"] == serial["counts"]
    assert parallel["skill_hits"] == serial["skill_hits"]
    assert parallel["skill_scenario_hits"] == serial["skill_scenario_hits"]
    assert parallel["rejection_reasons"] == serial["rejection_reasons"]
    assert parallel["city_distribution"] == serial["city_distribution"]
    assert parallel["actor_distribution"] == serial["actor_distribution"]
    assert parallel["target_risk_definitions"] == serial["target_risk_definitions"]
    assert parallel["seed_risk_distribution"] == serial["seed_risk_distribution"]
    assert serial["inputs"]["workers"] == 1
    assert parallel["inputs"]["workers"] == 2
    performance = parallel["performance"]
    assert performance["current_run_scenario_count"] == len(rows)
    assert performance["current_run_scenario_elapsed_seconds_sum"] > 0
    assert performance["current_run_ideal_balanced_scenario_wall_seconds"] > 0
    assert performance["current_run_estimated_steady_state_scenarios_per_second"] > 0
    assert performance["current_run_wall_minus_ideal_balanced_scenario_seconds"] >= 0
    assert _checkpoint_scenario_ids(Path(parallel_arguments["checkpoint_path"])) == [
        row.scenario_id for row in rows
    ]


def test_parallel_failure_keeps_ordered_prefix_and_can_resume(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [_manifest_row(f"scene-{suffix}") for suffix in ("a", "b", "c", "d")]
    arguments = _parallel_arguments(tmp_path, rows)
    monkeypatch.setattr(detect_seeds, "load_av2_scenario", _parallel_loader)
    monkeypatch.setattr(skill_detection, "detect_scenario", _failing_parallel_detector)

    with pytest.raises(RuntimeError, match="parallel detector failure"):
        run_scan(**arguments, workers=2)
    assert _checkpoint_scenario_ids(Path(arguments["checkpoint_path"])) == [
        "scene-a",
        "scene-b",
    ]

    monkeypatch.setattr(skill_detection, "detect_scenario", _parallel_detector)
    summary = run_scan(**arguments, workers=2)

    assert summary["counts"]["resumed_scenarios_this_run"] == 2
    assert summary["counts"]["new_scenarios_this_run"] == 2
    assert _checkpoint_scenario_ids(Path(arguments["checkpoint_path"])) == [
        row.scenario_id for row in rows
    ]
    assert {record.scenario_id for record in read_seed_records(Path(arguments["output_csv"]))} == {
        row.scenario_id for row in rows
    }

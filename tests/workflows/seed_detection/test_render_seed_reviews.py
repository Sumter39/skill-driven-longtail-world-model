from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.visualization.render_seed_reviews import run_review_rendering
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds import SeedRecord, write_seed_records
from skilldrive.visualization import seed_review_filename


TARGET_RISK_DEFINITION = {
    "metric": "minimum_distance",
    "target_range": [1.0, 4.0],
    "source": "semantic",
    "direction": "lower_is_riskier",
}


def _record(
    scenario_id: str,
    skill_id: str,
    score: float,
    *,
    source_path: str | None = None,
    seed_risk_metric: str = "minimum_distance",
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id="initiator",
        responder_track_id="responder",
        role_track_ids={"initiator": "initiator", "responder": "responder"},
        trigger_score=score,
        seed_risk_metric=seed_risk_metric,
        seed_risk_value=3.0,
        target_risk_definition=TARGET_RISK_DEFINITION,
        source_path=(
            source_path
            or f"train/{scenario_id}/scenario_{scenario_id}.parquet"
        ),
        evidence={"matched": True, "gap_m": 3.0},
        sampled_parameters={"offset_m": 1.0},
    )


def _scenario(scenario_id: str) -> Scenario:
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [1.0, 0.0]])
    actors = [
        AgentTrack(
            track_id=track_id,
            object_type="vehicle",
            positions=positions + offset,
            velocities=velocities,
            headings=np.zeros(2),
            observed_mask=np.ones(2, dtype=bool),
            is_focal=track_id == "initiator",
        )
        for track_id, offset in (
            ("initiator", np.array([0.0, 0.0])),
            ("responder", np.array([0.0, 2.0])),
        )
    ]
    return Scenario(
        scenario_id=scenario_id,
        city_name="test-city",
        timestamps=np.arange(2, dtype=np.int64),
        focal_track_id="initiator",
        agents=actors,
        map_polylines=[],
    )


def _prepare_source(data_root: Path, record: SeedRecord) -> None:
    source = data_root.joinpath(*record.source_path.split("/"))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"test parquet placeholder")


def test_review_batch_loads_each_scenario_once_resumes_and_writes_indices(
    tmp_path: Path,
) -> None:
    records = [
        _record(
            "scene-a",
            "skill_a",
            0.9,
            seed_risk_metric="minimum_trajectory_distance",
        ),
        _record("scene-b", "skill_b", 0.8),
        _record("scene-a", "skill_c", 0.7),
    ]
    candidate_csv = write_seed_records(tmp_path / "candidates.csv", records)
    data_root = tmp_path / "data"
    for record in records:
        _prepare_source(data_root, record)

    loaded: list[str] = []
    rendered: list[tuple[str, str]] = []

    def loader(path: str | Path) -> Scenario:
        scenario_id = Path(path).parent.name
        loaded.append(scenario_id)
        return _scenario(scenario_id)

    def renderer(scenario: Scenario, record: SeedRecord, output_dir: str | Path) -> Path:
        rendered.append((scenario.scenario_id, record.skill_id))
        output = Path(output_dir) / seed_review_filename(record)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x89PNG\r\n\x1a\nreview")
        return output

    output_dir = tmp_path / "review"
    first = run_review_rendering(
        candidate_csv=candidate_csv,
        data_root=data_root,
        output_dir=output_dir,
        target_count=3,
        scenario_loader=loader,
        renderer=renderer,
    )

    assert loaded == ["scene-a", "scene-b"]
    assert len(rendered) == 3
    assert first["rendered_this_run"] == 3
    with (output_dir / "review_index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["skill_id"] for row in rows] == ["skill_a", "skill_b", "skill_c"]
    assert {row["seed_risk_is_proxy"] for row in rows} == {"false", "true"}
    assert json.loads(rows[0]["target_risk_definition_json"]) == TARGET_RISK_DEFINITION
    index = json.loads((output_dir / "review_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == 2
    assert index["selected_reviews"] == 3
    assert index["unique_scenarios"] == 2
    assert index["skill_counts"] == {"skill_a": 1, "skill_b": 1, "skill_c": 1}
    assert index["reviews"][0]["seed_risk_is_proxy"] is True
    assert index["reviews"][0]["target_risk_definition"] == TARGET_RISK_DEFINITION
    csv_bytes = (output_dir / "review_index.csv").read_bytes()
    json_bytes = (output_dir / "review_index.json").read_bytes()

    second = run_review_rendering(
        candidate_csv=candidate_csv,
        data_root=data_root,
        output_dir=output_dir,
        target_count=3,
        scenario_loader=loader,
        renderer=renderer,
    )
    assert second["rendered_this_run"] == 0
    assert second["resumed_reviews"] == 3
    assert loaded == ["scene-a", "scene-b"]

    stale_png = output_dir / "stale.png"
    stale_png.write_bytes(b"\x89PNG\r\n\x1a\nstale")
    unrelated_file = output_dir / "notes.txt"
    unrelated_file.write_text("keep", encoding="utf-8")

    third = run_review_rendering(
        candidate_csv=candidate_csv,
        data_root=data_root,
        output_dir=output_dir,
        target_count=3,
        restart=True,
        scenario_loader=loader,
        renderer=renderer,
    )
    assert third["rendered_this_run"] == 3
    assert third["removed_stale_reviews"] == 1
    assert loaded == ["scene-a", "scene-b", "scene-a", "scene-b"]
    assert not stale_png.exists()
    assert unrelated_file.read_text(encoding="utf-8") == "keep"
    assert (output_dir / "review_index.csv").read_bytes() == csv_bytes
    assert (output_dir / "review_index.json").read_bytes() == json_bytes


def test_review_batch_requires_restart_when_selection_content_changes(tmp_path: Path) -> None:
    record = _record("scene-a", "skill_a", 0.8)
    candidate_csv = write_seed_records(tmp_path / "candidates.csv", [record])
    data_root = tmp_path / "data"
    _prepare_source(data_root, record)

    def renderer(scenario: Scenario, candidate: SeedRecord, output_dir: str | Path) -> Path:
        output = Path(output_dir) / seed_review_filename(candidate)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x89PNG\r\n\x1a\nreview")
        return output

    arguments = {
        "candidate_csv": candidate_csv,
        "data_root": data_root,
        "output_dir": tmp_path / "review",
        "scenario_loader": lambda path: _scenario("scene-a"),
        "renderer": renderer,
    }
    run_review_rendering(**arguments)
    write_seed_records(
        candidate_csv,
        [
            SeedRecord(
                **{
                    **record.__dict__,
                    "trigger_score": 0.9,
                    "evidence": {"matched": True, "gap_m": 2.0},
                }
            )
        ],
    )

    with pytest.raises(ValueError, match="rerun with --restart"):
        run_review_rendering(**arguments)


@pytest.mark.parametrize(
    ("source_path", "message"),
    [
        ("../outside.parquet", "relative parquet path"),
        ("train/scene-a/missing.parquet", "scenario file not found"),
    ],
)
def test_review_batch_reports_invalid_or_missing_source_paths(
    tmp_path: Path,
    source_path: str,
    message: str,
) -> None:
    record = _record("scene-a", "skill_a", 0.8, source_path=source_path)
    candidate_csv = write_seed_records(tmp_path / "candidates.csv", [record])

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        run_review_rendering(
            candidate_csv=candidate_csv,
            data_root=tmp_path / "data",
            output_dir=tmp_path / "review",
            scenario_loader=lambda path: _scenario("scene-a"),
        )

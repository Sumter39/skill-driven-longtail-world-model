from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.seed_detection.select_formal_seeds as select_formal_seeds
from scripts.seed_detection.select_formal_seeds import run_selection
from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.seeds import SeedRecord, read_seed_records, write_seed_records


TARGET_RISK_DEFINITION = {
    "metric": "time_to_collision",
    "target_range": [1.0, 4.0],
    "source": "reference",
    "direction": "lower_is_riskier",
}


def _manifest_row(
    scenario_id: str,
    *,
    split: str = "train",
    source_split: str = "train",
) -> ManifestRow:
    return ManifestRow(
        scenario_id=scenario_id,
        split=split,
        source_path=(
            f"{source_split}/{scenario_id}/scenario_{scenario_id}.parquet"
        ),
        city_name="unknown_until_loaded",
        selected_reason="test",
    )


def _write_manifests(tmp_path: Path, scenario_ids: list[str]) -> dict[str, Path]:
    formal = tmp_path / "formal_train.csv"
    internal = tmp_path / "internal_validation.csv"
    final = tmp_path / "final_validation.csv"
    write_manifest(formal, [_manifest_row(scenario_id) for scenario_id in scenario_ids])
    write_manifest(
        internal,
        [_manifest_row("internal-only", split="internal_validation")],
    )
    write_manifest(
        final,
        [_manifest_row("final-only", split="validation", source_split="val")],
    )
    return {"formal": formal, "internal": internal, "final": final}


def _record(
    scenario_id: str,
    skill_id: str,
    risk_value: float,
    *,
    risk_metric: str = "time_to_collision",
) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"initiator-{skill_id}",
        responder_track_id=f"responder-{skill_id}",
        role_track_ids={
            "initiator": f"initiator-{skill_id}",
            "responder": f"responder-{skill_id}",
        },
        trigger_score=0.8,
        seed_risk_metric=risk_metric,
        seed_risk_value=risk_value,
        target_risk_definition=TARGET_RISK_DEFINITION,
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"risk_value": risk_value},
        sampled_parameters={"amount": 1.0},
    )


def _actor_distribution(record_count: int) -> dict[str, object]:
    return {
        "initiator": {"vehicle": record_count},
        "responder": {"vehicle": record_count},
        "pair": {"vehicle|vehicle": record_count},
        "by_role": {
            "initiator": {"vehicle": record_count},
            "responder": {"vehicle": record_count},
        },
        "role_combination": {
            "initiator=vehicle|responder=vehicle": record_count,
        },
    }


def _write_checkpoint(
    path: Path,
    formal_rows: list[ManifestRow],
    records: list[SeedRecord],
) -> None:
    by_scenario: dict[str, list[SeedRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_id, []).append(record)
    lines = [
        {
            "kind": "metadata",
            "version": 2,
            "scan_kind": "formal",
            "manifest_sha256": select_formal_seeds._manifest_fingerprint(formal_rows),
        }
    ]
    for index, scenario_id in enumerate(sorted(by_scenario)):
        scenario_records = by_scenario[scenario_id]
        lines.append(
            {
                "kind": "scenario",
                "scenario_id": scenario_id,
                "city_name": f"city-{index % 2}",
                "records": [record.to_csv_row() for record in scenario_records],
                "actor_distribution": _actor_distribution(len(scenario_records)),
            }
        )
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in lines
        ),
        encoding="ascii",
    )


def _arguments(
    tmp_path: Path,
    manifests: dict[str, Path],
    pool: Path,
    *,
    checkpoint: Path | None = None,
    target: int = 3,
) -> dict[str, object]:
    return {
        "formal_manifest_path": manifests["formal"],
        "internal_validation_manifest": manifests["internal"],
        "final_validation_manifest": manifests["final"],
        "candidate_pool_path": pool,
        "checkpoint_path": checkpoint,
        "output_csv": tmp_path / "formal_candidates.csv",
        "summary_json": tmp_path / "formal_summary.json",
        "target_scenario_count": target,
        "seed": 2026,
    }


def test_selects_exact_scenarios_retains_all_labels_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    scenario_ids = [f"scene-{index}" for index in range(6)]
    manifests = _write_manifests(tmp_path, scenario_ids)
    records = [
        record
        for index, scenario_id in enumerate(scenario_ids)
        for record in (
            _record(scenario_id, "common", float(index)),
            _record(
                scenario_id,
                f"skill-{index % 2}",
                float(index + 10),
                risk_metric="minimum_distance",
            ),
        )
    ]
    pool = tmp_path / "formal_candidate_pool.csv"
    write_seed_records(pool, records)
    checkpoint = tmp_path / "formal_candidate_pool.checkpoint.jsonl"
    formal_rows = select_formal_seeds.read_manifest(manifests["formal"])
    _write_checkpoint(checkpoint, formal_rows, records)
    arguments = _arguments(tmp_path, manifests, pool, checkpoint=checkpoint)

    first = run_selection(**arguments)
    output = Path(arguments["output_csv"])
    summary_path = Path(arguments["summary_json"])
    selected = read_seed_records(output)
    selected_ids = {record.scenario_id for record in selected}

    assert len(selected_ids) == 3
    assert selected == sorted(
        [record for record in records if record.scenario_id in selected_ids],
        key=lambda record: record.unique_key,
    )
    assert all(
        sum(record.scenario_id == scenario_id for record in selected) == 2
        for scenario_id in selected_ids
    )
    assert first["counts"] == {
        "selected_records": 6,
        "selected_unique_scenarios": 3,
    }
    assert first["leakage_check"]["status"] == "passed"
    assert first["schema_check"]["candidate_csv_fields"] == list(
        select_formal_seeds.SEED_CSV_FIELDS
    )
    assert first["checkpoint_enrichment"]["selected_scenarios_matched_in_checkpoint"] == 3
    assert first["checkpoint_enrichment"]["selected_scenarios_with_city"] == 3
    assert first["checkpoint_enrichment"]["selected_scenarios_without_city"] == 0
    assert sum(first["city_distribution"]["selected_scenarios"].values()) == 3
    assert first["actor_distribution"]["initiator"] == {"vehicle": 6}
    assert first["seed_risk_relation_counts"] == {
        "target_metric_observation": 3,
        "proxy_metric": 3,
    }

    output_bytes = output.read_bytes()
    summary_bytes = summary_path.read_bytes()
    second = run_selection(**arguments)
    assert second == first
    assert output.read_bytes() == output_bytes
    assert summary_path.read_bytes() == summary_bytes


def test_explicitly_disabled_checkpoint_keeps_selection_valid_without_fake_metadata(
    tmp_path: Path,
) -> None:
    scenario_ids = ["scene-a", "scene-b", "scene-c"]
    manifests = _write_manifests(tmp_path, scenario_ids)
    pool = tmp_path / "formal_candidate_pool.csv"
    write_seed_records(
        pool,
        [_record(scenario_id, "skill", float(index)) for index, scenario_id in enumerate(scenario_ids)],
    )

    summary = run_selection(
        **_arguments(
            tmp_path,
            manifests,
            pool,
            checkpoint=None,
            target=2,
        )
    )

    assert summary["checkpoint_enrichment"]["status"] == "disabled"
    assert summary["checkpoint_enrichment"]["selected_scenarios_matched_in_checkpoint"] == 0
    assert summary["checkpoint_enrichment"]["selected_scenarios_with_city"] == 0
    assert summary["checkpoint_enrichment"]["selected_scenarios_without_city"] == 2
    assert summary["city_distribution"] == {
        "selected_scenarios": {},
        "selected_labels": {},
    }
    assert summary["actor_distribution"]["by_role"] == {}


def test_missing_checkpoint_path_is_rejected(tmp_path: Path) -> None:
    scenario_ids = ["scene-a", "scene-b"]
    manifests = _write_manifests(tmp_path, scenario_ids)
    pool = tmp_path / "formal_candidate_pool.csv"
    write_seed_records(
        pool,
        [_record(scenario_id, "skill", float(index)) for index, scenario_id in enumerate(scenario_ids)],
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_selection(
            **_arguments(
                tmp_path,
                manifests,
                pool,
                checkpoint=tmp_path / "missing.checkpoint.jsonl",
                target=1,
            )
        )


@pytest.mark.parametrize(
    ("candidate_id", "message"),
    [
        ("outside-formal", "outside formal_train"),
        ("internal-only", "leaks internal_validation"),
        ("final-only", "leaks final_validation"),
    ],
)
def test_rejects_candidate_pool_leakage_or_out_of_scope_scenarios(
    tmp_path: Path,
    candidate_id: str,
    message: str,
) -> None:
    manifests = _write_manifests(tmp_path, ["formal-only"])
    pool = tmp_path / "formal_candidate_pool.csv"
    source_split = "val" if candidate_id == "final-only" else "train"
    record = _record(candidate_id, "skill", 1.0)
    if source_split == "val":
        record = SeedRecord(
            **{
                **record.__dict__,
                "source_path": f"val/{candidate_id}/scenario_{candidate_id}.parquet",
            }
        )
    write_seed_records(pool, [record])

    with pytest.raises(ValueError, match=message):
        run_selection(**_arguments(tmp_path, manifests, pool, target=1))


def test_rejects_invalid_pool_schema_and_insufficient_unique_scenarios(
    tmp_path: Path,
) -> None:
    manifests = _write_manifests(tmp_path, ["scene-a", "scene-b"])
    invalid_pool = tmp_path / "invalid_pool.csv"
    invalid_pool.write_text("scenario_id,unknown\nscene-a,x\n", encoding="utf-8")
    invalid_arguments = _arguments(tmp_path, manifests, invalid_pool, target=1)

    with pytest.raises(ValueError, match="header"):
        run_selection(**invalid_arguments)

    pool = tmp_path / "formal_candidate_pool.csv"
    write_seed_records(pool, [_record("scene-a", "skill", 1.0)])
    arguments = _arguments(tmp_path, manifests, pool, target=2)
    with pytest.raises(ValueError, match="only 1 unique scenarios"):
        run_selection(**arguments)
    assert not Path(arguments["output_csv"]).exists()
    assert not Path(arguments["summary_json"]).exists()


def test_rejects_manifest_and_checkpoint_contract_drift(tmp_path: Path) -> None:
    manifests = _write_manifests(tmp_path, ["scene-a", "scene-b"])
    pool = tmp_path / "formal_candidate_pool.csv"
    records = [_record("scene-a", "skill", 1.0), _record("scene-b", "skill", 2.0)]
    write_seed_records(pool, records)

    wrong_rows = [
        _manifest_row("scene-a", split="development_train"),
        _manifest_row("scene-b"),
    ]
    write_manifest(manifests["formal"], wrong_rows)
    with pytest.raises(ValueError, match="split=train"):
        run_selection(**_arguments(tmp_path, manifests, pool, target=1))

    write_manifest(manifests["formal"], [_manifest_row("scene-a"), _manifest_row("scene-b")])
    checkpoint = tmp_path / "formal_candidate_pool.checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "kind": "metadata",
                "version": 2,
                "scan_kind": "formal",
                "manifest_sha256": "stale",
            }
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="fingerprint differs"):
        run_selection(
            **_arguments(tmp_path, manifests, pool, checkpoint=checkpoint, target=1)
        )


def test_checkpoint_streaming_retains_only_selected_entries(tmp_path: Path) -> None:
    formal_rows = [_manifest_row(f"scene-{name}") for name in ("a", "b", "c")]
    records = [_record(row.scenario_id, "skill", float(index)) for index, row in enumerate(formal_rows)]
    checkpoint = tmp_path / "formal_pool_summary.checkpoint.jsonl"
    _write_checkpoint(checkpoint, formal_rows, records)

    entries, audit = select_formal_seeds._parse_checkpoint(
        checkpoint,
        formal_manifest_sha256=select_formal_seeds._manifest_fingerprint(formal_rows),
        formal_ids={row.scenario_id for row in formal_rows},
        selected_ids={"scene-b"},
    )

    assert set(entries) == {"scene-b"}
    assert audit["scenario_count"] == 3


def test_checkpoint_must_cover_the_complete_formal_manifest(tmp_path: Path) -> None:
    formal_rows = [_manifest_row("scene-a"), _manifest_row("scene-b")]
    checkpoint = tmp_path / "formal_pool_summary.checkpoint.jsonl"
    _write_checkpoint(checkpoint, formal_rows, [_record("scene-a", "skill", 1.0)])

    with pytest.raises(ValueError, match="covers 1 formal scenarios; expected 2"):
        select_formal_seeds._parse_checkpoint(
            checkpoint,
            formal_manifest_sha256=select_formal_seeds._manifest_fingerprint(formal_rows),
            formal_ids={row.scenario_id for row in formal_rows},
            selected_ids={"scene-a"},
        )


@pytest.mark.parametrize(
    ("scenario_ids", "message"),
    [
        (["scene-a", "scene-b", "scene-b"], "duplicate scenario"),
        (["scene-a", "outside-formal"], "outside formal_train"),
    ],
)
def test_checkpoint_streaming_still_rejects_unselected_duplicate_or_outside_ids(
    tmp_path: Path,
    scenario_ids: list[str],
    message: str,
) -> None:
    formal_rows = [_manifest_row("scene-a"), _manifest_row("scene-b")]
    metadata = {
        "kind": "metadata",
        "version": 2,
        "scan_kind": "formal",
        "manifest_sha256": select_formal_seeds._manifest_fingerprint(formal_rows),
    }
    checkpoint = tmp_path / "formal_pool_summary.checkpoint.jsonl"
    checkpoint.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in (
                metadata,
                *(
                    {
                        "kind": "scenario",
                        "scenario_id": scenario_id,
                        "city_name": "unused",
                        "records": [],
                        "actor_distribution": _actor_distribution(0),
                    }
                    for scenario_id in scenario_ids
                ),
            )
        ),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match=message):
        select_formal_seeds._parse_checkpoint(
            checkpoint,
            formal_manifest_sha256=select_formal_seeds._manifest_fingerprint(formal_rows),
            formal_ids={row.scenario_id for row in formal_rows},
            selected_ids=set(),
        )


def test_default_paths_and_target_match_the_formal_delivery_contract() -> None:
    assert select_formal_seeds.DEFAULT_CANDIDATE_POOL == Path(
        "outputs/seed_detection/formal_candidate_pool.csv"
    )
    assert select_formal_seeds.DEFAULT_OUTPUT_CSV == Path(
        "manifests/seeds/formal_candidates.csv"
    )
    assert select_formal_seeds.DEFAULT_SUMMARY_JSON == Path(
        "outputs/seed_detection/formal_summary.json"
    )
    assert select_formal_seeds.DEFAULT_CHECKPOINT == Path(
        "outputs/seed_detection/formal_pool_summary.checkpoint.jsonl"
    )
    assert select_formal_seeds.DEFAULT_TARGET_SCENARIOS == 5_000

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.seed_detection.merge_candidate_pools import merge_candidate_pools
from scripts.seed_detection.select_formal_seeds import _parse_checkpoint
from skilldrive.seeds import SeedRecord, read_seed_records, write_seed_records


def _record(scenario_id: str, skill_id: str, suffix: str) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"initiator-{suffix}",
        responder_track_id=f"responder-{suffix}",
        role_track_ids={
            "initiator": f"initiator-{suffix}",
            "responder": f"responder-{suffix}",
        },
        trigger_score=0.8,
        seed_risk_metric="minimum_gap",
        seed_risk_value=2.0,
        target_risk_definition={
            "metric": "minimum_gap",
            "target_range": [1.0, 4.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=(
            f"train/{scenario_id}/scenario_{scenario_id}.parquet"
        ),
        evidence={"matched": True},
        sampled_parameters={"test": suffix},
    )


def _actor_distribution(count: int) -> dict[str, object]:
    if count == 0:
        return {
            "initiator": {},
            "responder": {},
            "pair": {},
            "by_role": {},
            "role_combination": {},
        }
    return {
        "initiator": {"vehicle": count},
        "responder": {"vehicle": count},
        "pair": {"vehicle|vehicle": count},
        "by_role": {
            "initiator": {"vehicle": count},
            "responder": {"vehicle": count},
        },
        "role_combination": {"initiator=vehicle|responder=vehicle": count},
    }


def _entry(
    index: int,
    scenario_id: str,
    *,
    records: list[SeedRecord],
    rejection_counts: dict[str, int],
    city_name: str = "austin",
    elapsed_seconds: float = 1.0,
    peak_memory_mib: float = 10.0,
) -> dict[str, object]:
    return {
        "kind": "scenario",
        "manifest_index": index,
        "scenario_id": scenario_id,
        "city_name": city_name,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_mib": peak_memory_mib,
        "records": [record.to_csv_row() for record in records],
        "rejection_counts": rejection_counts,
        "actor_distribution": _actor_distribution(len(records)),
    }


def _write_checkpoint(
    path: Path,
    entries: list[dict[str, object]],
    *,
    manifest_sha256: str = "manifest-a",
    selected_skill_ids: list[str] | None = None,
) -> None:
    metadata: dict[str, object] = {
        "kind": "metadata",
        "version": 2,
        "scan_kind": "formal",
        "manifest_sha256": manifest_sha256,
        "manifest_scenario_count": len(entries),
        "skills_sha256": "skills",
    }
    if selected_skill_ids is not None:
        metadata["selected_skill_ids"] = selected_skill_ids
    values = [metadata, *entries]
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="ascii",
    )


def _sources(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    scene_a = "scene-a"
    scene_b = "scene-b"
    old_record = _record(scene_a, "old_skill", "old")
    new_record = _record(scene_a, "new_skill", "new")
    candidate_paths = [tmp_path / "old.csv", tmp_path / "new.csv"]
    write_seed_records(candidate_paths[0], [old_record])
    write_seed_records(candidate_paths[1], [new_record])
    checkpoint_paths = [tmp_path / "old.checkpoint.jsonl", tmp_path / "new.checkpoint.jsonl"]
    _write_checkpoint(
        checkpoint_paths[0],
        [
            _entry(0, scene_a, records=[old_record], rejection_counts={}),
            _entry(
                1,
                scene_b,
                records=[],
                rejection_counts={"old_skill:no_rule_match": 1},
                elapsed_seconds=1.5,
            ),
        ],
    )
    _write_checkpoint(
        checkpoint_paths[1],
        [
            _entry(
                0,
                scene_a,
                records=[new_record],
                rejection_counts={},
                elapsed_seconds=2.0,
                peak_memory_mib=20.0,
            ),
            _entry(
                1,
                scene_b,
                records=[],
                rejection_counts={"new_skill:no_rule_match": 1},
                elapsed_seconds=2.5,
                peak_memory_mib=20.0,
            ),
        ],
        selected_skill_ids=["new_skill"],
    )
    return candidate_paths, checkpoint_paths


def test_merges_disjoint_candidate_pools_and_complete_checkpoints(
    tmp_path: Path,
) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    output_csv = tmp_path / "merged.csv"
    output_checkpoint = tmp_path / "merged.checkpoint.jsonl"
    output_summary = tmp_path / "merged.summary.json"

    result = merge_candidate_pools(
        candidate_paths=candidate_paths,
        checkpoint_paths=checkpoint_paths,
        output_csv=output_csv,
        output_checkpoint=output_checkpoint,
        output_summary_json=output_summary,
        expected_scenarios=2,
    )

    records = read_seed_records(output_csv)
    assert result["scenario_count"] == 2
    assert result["candidate_count"] == 2
    assert result["selected_skill_ids"] == ["old_skill", "new_skill"]
    assert result["summary_json"] == str(output_summary)
    assert [record.skill_id for record in records] == ["new_skill", "old_skill"]

    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["summary_kind"] == "merged_formal_candidate_pool"
    assert summary["counts"] == {
        "processed_scenarios": 2,
        "candidates": 2,
        "unique_candidate_scenarios": 1,
        "covered_skills": 2,
        "configured_skills": 2,
    }
    assert summary["skill_hits"] == {"old_skill": 1, "new_skill": 1}

    lines = output_checkpoint.read_text(encoding="ascii").splitlines()
    metadata = json.loads(lines[0])
    first = json.loads(lines[1])
    second = json.loads(lines[2])
    assert metadata["manifest_sha256"] == "manifest-a"
    assert metadata["manifest_scenario_count"] == 2
    assert metadata["selected_skill_ids"] == ["old_skill", "new_skill"]
    assert metadata["merge_kind"] == "candidate_pool_union"
    assert first["scenario_id"] == "scene-a"
    assert len(first["records"]) == 2
    assert first["elapsed_seconds"] == 3.0
    assert first["peak_memory_mib"] == 20.0
    assert first["actor_distribution"]["pair"] == {"vehicle|vehicle": 2}
    assert second["rejection_counts"] == {
        "new_skill:no_rule_match": 1,
        "old_skill:no_rule_match": 1,
    }

    entries, audit = _parse_checkpoint(
        output_checkpoint,
        formal_manifest_sha256="manifest-a",
        formal_ids={"scene-a", "scene-b"},
        selected_ids={"scene-a", "scene-b"},
    )
    assert set(entries) == {"scene-a", "scene-b"}
    assert audit["scenario_count"] == 2


def test_excludes_zero_hit_skills_from_formal_checkpoint_and_summary(
    tmp_path: Path,
) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    values = [
        json.loads(line)
        for line in checkpoint_paths[0].read_text(encoding="ascii").splitlines()
    ]
    for entry in values[1:]:
        entry["rejection_counts"]["retired_skill:no_rule_match"] = 1
    checkpoint_paths[0].write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="ascii",
    )
    output_csv = tmp_path / "merged.csv"
    output_checkpoint = tmp_path / "merged.checkpoint.jsonl"
    output_summary = tmp_path / "merged.summary.json"

    result = merge_candidate_pools(
        candidate_paths=candidate_paths,
        checkpoint_paths=checkpoint_paths,
        output_csv=output_csv,
        output_checkpoint=output_checkpoint,
        output_summary_json=output_summary,
        excluded_skill_ids=["retired_skill"],
        expected_scenarios=2,
    )

    assert result["selected_skill_ids"] == ["old_skill", "new_skill"]
    assert result["excluded_skill_ids"] == ["retired_skill"]
    summary = json.loads(output_summary.read_text(encoding="utf-8"))
    assert summary["counts"]["configured_skills"] == 2
    assert summary["merge"]["excluded_zero_hit_skill_ids"] == ["retired_skill"]
    checkpoint = [
        json.loads(line)
        for line in output_checkpoint.read_text(encoding="ascii").splitlines()
    ]
    assert checkpoint[0]["selected_skill_ids"] == ["old_skill", "new_skill"]
    assert checkpoint[0]["excluded_zero_hit_skill_ids"] == ["retired_skill"]
    assert all(
        not any(reason.startswith("retired_skill:") for reason in entry["rejection_counts"])
        for entry in checkpoint[1:]
    )


def test_rejects_excluding_a_skill_with_candidate_records(tmp_path: Path) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)

    with pytest.raises(ValueError, match="must have zero candidate records"):
        merge_candidate_pools(
            candidate_paths=candidate_paths,
            checkpoint_paths=checkpoint_paths,
            output_csv=tmp_path / "merged.csv",
            output_checkpoint=tmp_path / "merged.checkpoint.jsonl",
            excluded_skill_ids=["old_skill"],
            expected_scenarios=2,
        )


def test_rejects_manifest_fingerprint_mismatch_without_touching_outputs(
    tmp_path: Path,
) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    new_lines = checkpoint_paths[1].read_text(encoding="ascii").splitlines()
    metadata = json.loads(new_lines[0])
    metadata["manifest_sha256"] = "manifest-b"
    checkpoint_paths[1].write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        + "\n"
        + "\n".join(new_lines[1:])
        + "\n",
        encoding="ascii",
    )
    output_csv = tmp_path / "merged.csv"
    output_checkpoint = tmp_path / "merged.checkpoint.jsonl"
    output_csv.write_bytes(b"existing-csv")
    output_checkpoint.write_bytes(b"existing-checkpoint")

    with pytest.raises(ValueError, match="manifest fingerprints differ"):
        merge_candidate_pools(
            candidate_paths=candidate_paths,
            checkpoint_paths=checkpoint_paths,
            output_csv=output_csv,
            output_checkpoint=output_checkpoint,
            expected_scenarios=2,
        )

    assert output_csv.read_bytes() == b"existing-csv"
    assert output_checkpoint.read_bytes() == b"existing-checkpoint"


@pytest.mark.parametrize("difference", ["scenario_order", "city_name"])
def test_rejects_checkpoint_scenario_or_city_mismatch(
    tmp_path: Path,
    difference: str,
) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    values = [
        json.loads(line)
        for line in checkpoint_paths[1].read_text(encoding="ascii").splitlines()
    ]
    if difference == "scenario_order":
        values[1]["scenario_id"], values[2]["scenario_id"] = (
            values[2]["scenario_id"],
            values[1]["scenario_id"],
        )
    else:
        values[1]["city_name"] = "miami"
    checkpoint_paths[1].write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="ascii",
    )

    expected_error = (
        "scenario order differs"
        if difference == "scenario_order"
        else "city_name differs"
    )
    with pytest.raises(ValueError, match=expected_error):
        merge_candidate_pools(
            candidate_paths=candidate_paths,
            checkpoint_paths=checkpoint_paths,
            output_csv=tmp_path / "merged.csv",
            output_checkpoint=tmp_path / "merged.checkpoint.jsonl",
            expected_scenarios=2,
        )


def test_rejects_candidate_checkpoint_drift_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    write_seed_records(candidate_paths[1], [])
    output_csv = tmp_path / "merged.csv"
    output_checkpoint = tmp_path / "merged.checkpoint.jsonl"
    output_csv.write_bytes(b"existing-csv")
    output_checkpoint.write_bytes(b"existing-checkpoint")

    with pytest.raises(ValueError, match="checkpoint candidate records differ"):
        merge_candidate_pools(
            candidate_paths=candidate_paths,
            checkpoint_paths=checkpoint_paths,
            output_csv=output_csv,
            output_checkpoint=output_checkpoint,
            expected_scenarios=2,
        )

    assert output_csv.read_bytes() == b"existing-csv"
    assert output_checkpoint.read_bytes() == b"existing-checkpoint"
    assert not (tmp_path / ".merged.csv.tmp").exists()
    assert not (tmp_path / ".merged.checkpoint.jsonl.tmp").exists()


def test_rejects_overlapping_skill_scans(tmp_path: Path) -> None:
    candidate_paths, checkpoint_paths = _sources(tmp_path)
    old_record = _record("scene-a", "old_skill", "other")
    write_seed_records(candidate_paths[1], [old_record])
    _write_checkpoint(
        checkpoint_paths[1],
        [
            _entry(0, "scene-a", records=[old_record], rejection_counts={}),
            _entry(
                1,
                "scene-b",
                records=[],
                rejection_counts={"old_skill:no_rule_match": 1},
            ),
        ],
        selected_skill_ids=["old_skill"],
    )

    with pytest.raises(ValueError, match="selected skill IDs overlap"):
        merge_candidate_pools(
            candidate_paths=candidate_paths,
            checkpoint_paths=checkpoint_paths,
            output_csv=tmp_path / "merged.csv",
            output_checkpoint=tmp_path / "merged.checkpoint.jsonl",
            expected_scenarios=2,
        )

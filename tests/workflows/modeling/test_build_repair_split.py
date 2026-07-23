from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.modeling.build_repair_split import (
    FORMAL_CACHE_VERSION,
    build_repair_split,
)
from skilldrive.data.manifests import ManifestRow, read_manifest, write_manifest
from skilldrive.generation.contracts import GenerationTask
from skilldrive.generation.scheduler import TaskPlan, load_task_plan, write_task_plan


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _task(
    *,
    scenario_id: str,
    skill_id: str,
    proposal_mode: str,
    condition_skill_id: str,
    semantic_sha256: str,
) -> GenerationTask:
    return GenerationTask.create(
        task_index=0,
        seed_record_id=_sha256_bytes(f"seed:{scenario_id}:{skill_id}".encode()),
        scenario_id=scenario_id,
        skill_id=skill_id,
        target_track_id=f"target-{scenario_id}",
        proposal_mode=proposal_mode,
        condition_skill_id=condition_skill_id,
        candidate_budget=4,
        checkpoint_sha256=_sha256_bytes(b"checkpoint"),
        semantic_config_sha256=semantic_sha256,
    )


def _write_fixture(tmp_path: Path, *, validation_opened: bool = False) -> dict:
    formal_path = tmp_path / "manifests/splits/formal_train.csv"
    scenarios = [
        "a1",
        "a2",
        "b1",
        "b2",
        "b3",
        "b4",
        "r1",
        "r2",
        "g1",
        "g2",
        "g3",
        "g4",
    ]
    write_manifest(
        formal_path,
        [
            ManifestRow(
                scenario_id=scenario_id,
                split="train",
                source_path=(
                    f"train/{scenario_id}/scenario_{scenario_id}.parquet"
                ),
                city_name="test-city",
                selected_reason="fixture",
            )
            for scenario_id in scenarios
        ],
    )

    semantic_sha256 = _sha256_bytes(b"semantic")
    execution_sha256 = _sha256_bytes(b"execution")
    tasks: list[GenerationTask] = []
    for scenario_id in ("a1", "a2"):
        tasks.extend(
            [
                _task(
                    scenario_id=scenario_id,
                    skill_id="learned_a",
                    proposal_mode="learned_conditioned_prior",
                    condition_skill_id=condition,
                    semantic_sha256=semantic_sha256,
                )
                for condition in ("learned_a", "<none>")
            ]
        )
    for scenario_id in ("b1", "b2", "b3", "b4"):
        tasks.extend(
            [
                _task(
                    scenario_id=scenario_id,
                    skill_id="learned_b",
                    proposal_mode="learned_conditioned_prior",
                    condition_skill_id=condition,
                    semantic_sha256=semantic_sha256,
                )
                for condition in ("learned_b", "<none>")
            ]
        )
    for scenario_id, skill_id in (("r1", "rule_a"), ("r2", "rule_b")):
        tasks.append(
            _task(
                scenario_id=scenario_id,
                skill_id=skill_id,
                proposal_mode="rule_guided_prior_search",
                condition_skill_id="<none>",
                semantic_sha256=semantic_sha256,
            )
        )
    tasks = [
        replace(task, task_index=index)
        for index, task in enumerate(
            sorted(
                tasks,
                key=lambda task: (
                    task.scenario_id,
                    task.skill_id,
                    task.seed_record_id,
                    task.condition_skill_id,
                    task.task_id,
                ),
            )
        )
    ]
    pilot_dir = tmp_path / "pilot"
    source_plan = TaskPlan(
        semantic_config_sha256=semantic_sha256,
        execution_config_sha256=execution_sha256,
        base_seed=2026,
        per_skill=2,
        candidate_budget=4,
        tasks=tuple(tasks),
    )
    write_task_plan(pilot_dir, source_plan)
    (pilot_dir / "eligibility_audit.json").write_text(
        json.dumps(
            {
                "selection_stable": True,
                "formal_train_only": True,
                "validation_manifests_opened": validation_opened,
                "final_validation_accessed": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache/formal_train"
    cache_dir.mkdir(parents=True)
    positive = {
        "a1": "learned_a",
        "a2": "learned_a",
        "b1": "learned_b",
        "b2": "learned_b",
        "b3": "learned_b",
        "b4": "learned_b",
    }
    index_rows: list[dict] = []
    offset = 0
    for scenario_id in scenarios:
        index_rows.append(
            {
                "offset": offset,
                "sample_id": _sha256_bytes(f"base:{scenario_id}".encode()),
                "scenario_id": scenario_id,
                "shard": "shards/shard-00000.pt",
                "spec": {
                    "scenario_id": scenario_id,
                    "skill_id": "<none>",
                    "skill_supervision_mask": False,
                    "target_track_id": f"target-{scenario_id}",
                },
                "target_track_id": f"target-{scenario_id}",
            }
        )
        offset += 1
        if scenario_id in positive:
            skill_id = positive[scenario_id]
            index_rows.append(
                {
                    "offset": offset,
                    "sample_id": _sha256_bytes(
                        f"positive:{scenario_id}:{skill_id}".encode()
                    ),
                    "scenario_id": scenario_id,
                    "shard": "shards/shard-00000.pt",
                    "spec": {
                        "scenario_id": scenario_id,
                        "skill_id": skill_id,
                        "skill_supervision_mask": True,
                        "target_track_id": f"target-{scenario_id}",
                    },
                    "target_track_id": f"target-{scenario_id}",
                }
            )
            offset += 1
    index_path = cache_dir / "sample_index.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in index_rows
        ),
        encoding="utf-8",
    )
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "version": FORMAL_CACHE_VERSION,
                "status": "complete",
                "partition": "formal_train",
                "inputs": {"manifest_sha256": _sha256_file(formal_path)},
                "sample_index": {
                    "path": "sample_index.jsonl",
                    "sha256": _sha256_file(index_path),
                    "records": len(index_rows),
                },
                "shards": [
                    {
                        "path": "shards/shard-00000.pt",
                        "sha256": _sha256_bytes(b"unused-test-shard"),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "formal_path": formal_path,
        "cache_dir": cache_dir,
        "pilot_dir": pilot_dir,
        "source_plan": source_plan,
    }


def _run(tmp_path: Path, fixture: dict) -> dict:
    return build_repair_split(
        project_root=tmp_path,
        formal_manifest=fixture["formal_path"],
        cache_dir=fixture["cache_dir"],
        pilot_dir=fixture["pilot_dir"],
        output_manifest="out/formal_train_repair_v1.csv",
        output_audit="out/formal_train_repair_v1.audit.json",
        output_index_dir="out/cache_view",
        heldout_task_dir="out/heldout",
        dev_size=7,
        expected_train_size=5,
        expected_learned_skill_count=2,
        expected_rule_task_count=2,
    )


def test_build_repair_split_is_deterministic_disjoint_and_keeps_rule_tasks(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    first = _run(tmp_path, fixture)

    assert first["validation_manifests_opened"] is False
    assert first["counts"] == {
        "formal_scenarios": 12,
        "repair_train_scenarios": 5,
        "repair_dev_scenarios": 7,
        "source_cache_samples": 18,
        "repair_train_samples": 8,
        "repair_dev_samples": 10,
        "source_positive_samples": 6,
        "repair_train_positive_samples": 3,
        "repair_dev_positive_samples": 3,
        "source_pilot_tasks": 14,
        "heldout_ability_tasks": 8,
        "source_rule_tasks": 2,
        "heldout_rule_tasks": 2,
        "source_cache_shards": 1,
        "repair_train_shards_touched": 1,
        "repair_dev_shards_touched": 1,
        "shared_shards": 1,
    }
    assert first["learned_skill_distribution"]["learned_a"][
        "repair_train_positive_samples"
    ] == 1
    assert first["learned_skill_distribution"]["learned_a"][
        "repair_dev_positive_samples"
    ] == 1
    assert first["learned_skill_distribution"]["learned_b"][
        "repair_train_positive_samples"
    ] == 2
    assert first["learned_skill_distribution"]["learned_b"][
        "repair_dev_positive_samples"
    ] == 2

    manifest_rows = read_manifest(tmp_path / "out/formal_train_repair_v1.csv")
    train_ids = {row.scenario_id for row in manifest_rows if row.split == "repair_train"}
    dev_ids = {row.scenario_id for row in manifest_rows if row.split == "repair_dev"}
    assert len(train_ids) == 5
    assert len(dev_ids) == 7
    assert not train_ids & dev_ids
    assert {
        row.scenario_id
        for row in manifest_rows
        if row.selected_reason == "repair_v1:background_topup_no_positive"
    }.isdisjoint({"a1", "a2", "b1", "b2", "b3", "b4"})

    heldout_summary = json.loads(
        (tmp_path / "out/heldout/task_plan.summary.json").read_text(
            encoding="utf-8"
        )
    )
    heldout = load_task_plan(
        tmp_path / "out/heldout",
        expected_semantic_config_sha256=heldout_summary[
            "semantic_config_sha256"
        ],
        current_execution_config_sha256=heldout_summary[
            "execution_config_sha256"
        ],
    ).plan
    assert len(heldout.tasks) == 8
    assert {
        task.skill_id
        for task in heldout.tasks
        if task.proposal_mode == "rule_guided_prior_search"
    } == {"rule_a", "rule_b"}
    for skill_id in ("learned_a", "learned_b"):
        skill_tasks = [task for task in heldout.tasks if task.skill_id == skill_id]
        groups: dict[str, list[GenerationTask]] = {}
        for task in skill_tasks:
            groups.setdefault(task.scenario_id, []).append(task)
        assert all(
            {task.condition_skill_id for task in tasks} == {skill_id, "<none>"}
            for tasks in groups.values()
        )

    output_paths = [
        tmp_path / "out/formal_train_repair_v1.csv",
        tmp_path / "out/formal_train_repair_v1.audit.json",
        tmp_path / "out/cache_view/train.sample_index.jsonl",
        tmp_path / "out/cache_view/dev.sample_index.jsonl",
        tmp_path / "out/heldout/task_plan.jsonl",
        tmp_path / "out/heldout/task_plan.summary.json",
    ]
    first_hashes = [_sha256_file(path) for path in output_paths]
    second = _run(tmp_path, fixture)
    assert second == first
    assert [_sha256_file(path) for path in output_paths] == first_hashes


def test_build_repair_split_rejects_validation_access_evidence(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, validation_opened=True)
    with pytest.raises(
        ValueError,
        match="validation_manifests_opened=false",
    ):
        _run(tmp_path, fixture)


def test_build_repair_split_rejects_an_incomplete_learned_pair(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    plan = fixture["source_plan"]
    removed = next(
        task
        for task in plan.tasks
        if task.skill_id == "learned_a" and task.condition_skill_id == "<none>"
    )
    tasks = tuple(
        replace(task, task_index=index)
        for index, task in enumerate(task for task in plan.tasks if task != removed)
    )
    write_task_plan(
        fixture["pilot_dir"],
        TaskPlan(
            semantic_config_sha256=plan.semantic_config_sha256,
            execution_config_sha256=plan.execution_config_sha256,
            base_seed=plan.base_seed,
            per_skill=plan.per_skill,
            candidate_budget=plan.candidate_budget,
            tasks=tasks,
        ),
    )
    with pytest.raises(ValueError, match="conditioned/control pair"):
        _run(tmp_path, fixture)

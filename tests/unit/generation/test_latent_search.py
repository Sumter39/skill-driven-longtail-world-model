from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import skilldrive.generation.latent_search as latent_search_module
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.generation.contracts import GenerationTask, canonical_sha256
from skilldrive.generation.latent_search import (
    build_latent_search_manifest,
    build_latent_search_tasks,
    load_latent_search_config,
    load_latent_search_manifest,
)
from skilldrive.generation.planning import (
    paired_latent_seed,
    pilot_evaluation_arm,
    semantic_generation_config_sha256,
)


CHECKPOINT_SHA = "a" * 64
RUN_MANIFEST_SHA = "b" * 64
SCHEMA_SHA = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _generation_config(root: Path) -> CounterfactualGenerationConfig:
    skills = (
        SkillGenerationConfig(
            skill_id="forced_lane_change_around_blockage",
            primary_generated_role="target",
            proposal_mode="rule_guided_prior_search",
            condition_skill_strategy="none_skill_id",
            joint_generation_limited=False,
        ),
        SkillGenerationConfig(
            skill_id="jaywalking_pedestrian_crossing",
            primary_generated_role="target",
            proposal_mode="learned_conditioned_prior",
            condition_skill_strategy="requested_skill_id",
            joint_generation_limited=False,
        ),
        SkillGenerationConfig(
            skill_id="slow_lead_blockage",
            primary_generated_role="target",
            proposal_mode="learned_conditioned_prior",
            condition_skill_strategy="requested_skill_id",
            joint_generation_limited=False,
        ),
        SkillGenerationConfig(
            skill_id="construction_object_lane_blockage",
            primary_generated_role="target",
            proposal_mode="rule_guided_prior_search",
            condition_skill_strategy="none_skill_id",
            joint_generation_limited=False,
        ),
    )
    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1_test",
        formal_catalog=root / "configs" / "skills" / "catalog.yaml",
        candidate_catalog=root / "configs" / "skills" / "candidate_catalog.yaml",
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=root / "best.pt",
            sha256=CHECKPOINT_SHA,
            run_manifest=root / "run_manifest.json",
            run_manifest_sha256=RUN_MANIFEST_SHA,
            schema_sha256=SCHEMA_SHA,
        ),
        inputs=GenerationInputConfig(
            data_root=root / "data",
            seed_manifest=root / "seeds.csv",
            seed_manifest_sha256="d" * 64,
            training_cache_manifest=root / "cache.json",
            training_cache_manifest_sha256="e" * 64,
            leakage_audit=root / "audit.json",
            leakage_audit_sha256="f" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=1,
            pilot_candidates_per_task=2,
            formal_candidates_per_task=2,
        ),
        formal_skill_ids=tuple(item.skill_id for item in skills),
        candidate_skill_ids=(),
        skills=skills,
    )


def _tasks(config: CounterfactualGenerationConfig) -> tuple[GenerationTask, ...]:
    semantic = semantic_generation_config_sha256(config)
    rows = (
        ("1" * 64, "scene-forced", "forced_lane_change_around_blockage", "rule_guided_prior_search", "<none>"),
        ("2" * 64, "scene-jay", "jaywalking_pedestrian_crossing", "learned_conditioned_prior", "jaywalking_pedestrian_crossing"),
        ("2" * 64, "scene-jay", "jaywalking_pedestrian_crossing", "learned_conditioned_prior", "<none>"),
        ("3" * 64, "scene-slow", "slow_lead_blockage", "learned_conditioned_prior", "slow_lead_blockage"),
        ("3" * 64, "scene-slow", "slow_lead_blockage", "learned_conditioned_prior", "<none>"),
        ("4" * 64, "scene-construction", "construction_object_lane_blockage", "rule_guided_prior_search", "<none>"),
    )
    return tuple(
        GenerationTask.create(
            task_index=index,
            seed_record_id=seed_record,
            scenario_id=scenario,
            skill_id=skill,
            target_track_id=f"target-{scenario}",
            proposal_mode=proposal_mode,
            condition_skill_id=condition,
            candidate_budget=2,
            checkpoint_sha256=CHECKPOINT_SHA,
            semantic_config_sha256=semantic,
        )
        for index, (seed_record, scenario, skill, proposal_mode, condition) in enumerate(rows)
    )


def _row(task: GenerationTask, candidate_index: int, *, failed: str | None):
    stage_order = (
        "schema_finite",
        "history_invariants",
        "kinematics",
        "map",
        "collision",
        "target_risk",
        "skill_trigger",
        "parameter_realization",
        "diversity",
    )
    evidence = []
    for stage in stage_order[:-1]:
        passed = stage != failed
        evidence.append({"stage": stage, "passed": passed, "metrics": {}})
        if not passed:
            break
    return {
        "candidate_id": canonical_sha256([task.task_id, candidate_index]),
        "metrics": {
            "task_id": task.task_id,
            "candidate_index": candidate_index,
            "scenario_id": task.scenario_id,
            "skill_id": task.skill_id,
            "first_failed_stage": failed,
            "quality_score": None if failed else 0.1 + candidate_index,
            "stage_evidence": evidence,
        },
    }


def _pilot_artifacts(root: Path, config: CounterfactualGenerationConfig):
    pilot_root = root / "outputs" / "pilot-run"
    tasks = _tasks(config)
    task_plan = pilot_root / "task_plan.jsonl"
    task_plan.parent.mkdir(parents=True, exist_ok=True)
    task_plan.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "task_index": task.task_index,
                    "seed_record_id": task.seed_record_id,
                    "scenario_id": task.scenario_id,
                    "skill_id": task.skill_id,
                    "target_track_id": task.target_track_id,
                    "proposal_mode": task.proposal_mode,
                    "condition_skill_id": task.condition_skill_id,
                    "candidate_budget": task.candidate_budget,
                    "checkpoint_sha256": task.checkpoint_sha256,
                    "semantic_config_sha256": task.semantic_config_sha256,
                },
                sort_keys=True,
            )
            for task in tasks
        )
        + "\n",
        encoding="utf-8",
    )
    task_plan_id = "5" * 64
    task_plan_summary = pilot_root / "task_plan.summary.json"
    _write_json(
        task_plan_summary,
        {"task_plan_id": task_plan_id, "base_seed": 2026},
    )

    accepted_rows = [
        _row(tasks[2], 0, failed=None),
    ]
    rejected_rows = [
        _row(tasks[0], 0, failed="skill_trigger"),
        _row(tasks[0], 1, failed="kinematics"),
        _row(tasks[1], 0, failed="kinematics"),
        _row(tasks[1], 1, failed="kinematics"),
        _row(tasks[2], 1, failed="kinematics"),
        _row(tasks[3], 0, failed="kinematics"),
        _row(tasks[3], 1, failed="kinematics"),
        _row(tasks[4], 0, failed="kinematics"),
        _row(tasks[4], 1, failed="kinematics"),
        _row(tasks[5], 0, failed="parameter_realization"),
        _row(tasks[5], 1, failed="kinematics"),
    ]
    accepted = pilot_root / "filter" / "accepted.jsonl"
    rejected = pilot_root / "filter" / "rejected.jsonl"
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in accepted_rows) + "\n",
        encoding="utf-8",
    )
    rejected.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rejected_rows) + "\n",
        encoding="utf-8",
    )
    summary = pilot_root / "filter" / "summary.json"
    _write_json(
        summary,
        {
            "stage": "pilot",
            "status": "completed",
            "pilot_run_id": "6" * 64,
            "task_plan_id": task_plan_id,
            "task_plan_sha256": _sha256(task_plan),
            "task_plan_summary_sha256": _sha256(task_plan_summary),
            "checkpoint_sha256": CHECKPOINT_SHA,
            "generation_semantic_sha256": semantic_generation_config_sha256(config),
            "generation_execution_sha256": "7" * 64,
            "filter_semantic_sha256": "8" * 64,
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
            "outputs": {
                "task_plan": task_plan.relative_to(root).as_posix(),
                "accepted": accepted.relative_to(root).as_posix(),
                "rejected": rejected.relative_to(root).as_posix(),
            },
        },
    )
    return summary, tasks


def test_config_is_frozen_to_4096_512_and_top64(tmp_path: Path) -> None:
    source = Path("configs/generation/latent_search_v1.yaml")
    config = load_latent_search_config(source)
    assert config.candidate_budget_per_arm == 4096
    assert config.generation_chunk_size == 512
    assert config.kinematic_top_k == 64
    assert sum(len(item.required_arms) for item in config.representatives) == 6

    changed = tmp_path / "changed.yaml"
    changed.write_text(
        source.read_text(encoding="utf-8").replace(
            "candidate_budget_per_arm: 4096",
            "candidate_budget_per_arm: 1024",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen at 4096"):
        load_latent_search_config(changed)


def test_manifest_selection_and_expansion_are_deterministic(tmp_path: Path) -> None:
    generation_config = _generation_config(tmp_path)
    pilot_summary, pilot_tasks = _pilot_artifacts(tmp_path, generation_config)
    search_config = load_latent_search_config(
        Path("configs/generation/latent_search_v1.yaml")
    )
    output = tmp_path / "manifests" / "representatives.json"
    manifest = build_latent_search_manifest(
        pilot_summary_path=pilot_summary,
        output_path=output,
        config=search_config,
        repository_root=tmp_path,
    )

    assert [item.seed_record_id for item in manifest.representatives] == [
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
    ]
    assert manifest.representatives[0].pilot_evidence_by_arm[
        "rule_guided_none"
    ]["deepest_reached_stage"] == "skill_trigger"
    assert manifest.representatives[2].selection_basis.startswith(
        "both arms fail every Pilot candidate at kinematics; Pilot rows lack"
    )
    assert manifest.representatives[3].pilot_evidence_by_arm[
        "rule_guided_none"
    ]["deepest_reached_stage"] == "parameter_realization"
    tasks = build_latent_search_tasks(
        manifest,
        config=search_config,
        generation_config=generation_config,
        repository_root=tmp_path,
    )
    assert len(tasks) == 6
    assert all(item.task.candidate_budget == 4096 for item in tasks)
    assert {item.task.task_id for item in tasks} == {
        item.task_id for item in pilot_tasks
    }

    jay = [item for item in tasks if item.representative_id == "jaywalking_condition_reverse"]
    assert len(jay) == 2
    for index in (0, 15, 4095):
        assert paired_latent_seed(2026, jay[0].task, index) == paired_latent_seed(
            2026, jay[1].task, index
        )

    loaded = load_latent_search_manifest(
        output,
        config=search_config,
        repository_root=tmp_path,
    )
    assert loaded.sha256 == manifest.sha256


def test_manifest_rejects_changed_pilot_summary(tmp_path: Path) -> None:
    generation_config = _generation_config(tmp_path)
    pilot_summary, _ = _pilot_artifacts(tmp_path, generation_config)
    search_config = load_latent_search_config(
        Path("configs/generation/latent_search_v1.yaml")
    )
    output = tmp_path / "manifests" / "representatives.json"
    build_latent_search_manifest(
        pilot_summary_path=pilot_summary,
        output_path=output,
        config=search_config,
        repository_root=tmp_path,
    )
    value = json.loads(pilot_summary.read_text(encoding="utf-8"))
    value["status"] = "changed"
    _write_json(pilot_summary, value)

    with pytest.raises(ValueError, match="artifact changed"):
        load_latent_search_manifest(
            output,
            config=search_config,
            repository_root=tmp_path,
        )


def test_selection_prefers_deepest_funnel_and_stable_slow_pair(
    tmp_path: Path,
) -> None:
    generation_config = _generation_config(tmp_path)
    search_config = load_latent_search_config(
        Path("configs/generation/latent_search_v1.yaml")
    )
    base_tasks = _tasks(generation_config)
    semantic = semantic_generation_config_sha256(generation_config)

    def extra_task(
        *,
        index: int,
        seed: str,
        scenario: str,
        skill: str,
        condition: str,
        proposal_mode: str,
    ) -> GenerationTask:
        return GenerationTask.create(
            task_index=index,
            seed_record_id=seed,
            scenario_id=scenario,
            skill_id=skill,
            target_track_id=f"target-{scenario}",
            proposal_mode=proposal_mode,
            condition_skill_id=condition,
            candidate_budget=2,
            checkpoint_sha256=CHECKPOINT_SHA,
            semantic_config_sha256=semantic,
        )

    forced_shallow = extra_task(
        index=10,
        seed="7" * 64,
        scenario="scene-forced-shallow",
        skill="forced_lane_change_around_blockage",
        condition="<none>",
        proposal_mode="rule_guided_prior_search",
    )
    construction_shallow = extra_task(
        index=11,
        seed="8" * 64,
        scenario="scene-construction-shallow",
        skill="construction_object_lane_blockage",
        condition="<none>",
        proposal_mode="rule_guided_prior_search",
    )
    slow_other_conditioned = extra_task(
        index=12,
        seed="9" * 64,
        scenario="scene-slow-other",
        skill="slow_lead_blockage",
        condition="slow_lead_blockage",
        proposal_mode="learned_conditioned_prior",
    )
    slow_other_control = extra_task(
        index=13,
        seed="9" * 64,
        scenario="scene-slow-other",
        skill="slow_lead_blockage",
        condition="<none>",
        proposal_mode="learned_conditioned_prior",
    )
    all_tasks = (
        *base_tasks,
        forced_shallow,
        construction_shallow,
        slow_other_conditioned,
        slow_other_control,
    )

    def stats(task: GenerationTask, failures: tuple[str | None, ...]):
        result = latent_search_module._PilotTaskStats(  # noqa: SLF001
            task=task,
            evaluation_arm=pilot_evaluation_arm(task),
        )
        for candidate_index, failure in enumerate(failures):
            result.add(
                _row(task, candidate_index, failed=failure),
                accepted=failure is None,
            )
        return result

    failures = {
        base_tasks[0].task_id: ("skill_trigger", "kinematics"),
        base_tasks[1].task_id: ("kinematics", "kinematics"),
        base_tasks[2].task_id: (None, "kinematics"),
        base_tasks[3].task_id: ("kinematics", "kinematics"),
        base_tasks[4].task_id: ("kinematics", "kinematics"),
        base_tasks[5].task_id: ("parameter_realization", "kinematics"),
        forced_shallow.task_id: ("kinematics", "kinematics"),
        construction_shallow.task_id: ("map", "kinematics"),
        slow_other_conditioned.task_id: ("kinematics", "kinematics"),
        slow_other_control.task_id: ("kinematics", "kinematics"),
    }
    stats_by_task = {
        task.task_id: stats(task, failures[task.task_id]) for task in all_tasks
    }
    selected = latent_search_module._select_representatives(  # noqa: SLF001
        search_config,
        stats_by_task,
    )

    assert selected[0].seed_record_id == base_tasks[0].seed_record_id
    assert selected[3].seed_record_id == base_tasks[5].seed_record_id
    slow_pairs = (
        (base_tasks[3], base_tasks[4]),
        (slow_other_conditioned, slow_other_control),
    )
    expected_slow = min(
        slow_pairs,
        key=lambda pair: (
            tuple(sorted(item.task_id for item in pair)),
            pair[0].seed_record_id,
        ),
    )
    assert selected[2].seed_record_id == expected_slow[0].seed_record_id
    assert "stable lexicographic paired task IDs" in selected[2].selection_basis

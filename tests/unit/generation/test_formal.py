from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import skilldrive.generation.formal as formal_module
from skilldrive.filtering.fingerprint import FilterSemanticFingerprint
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.generation.contracts import canonical_sha256
from skilldrive.generation.formal import (
    FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT,
    FORMAL_PRODUCTION_EXPECTED_TASK_COUNT,
    FORMAL_PRODUCTION_SKILL_COUNT,
    FORMAL_TASK_PLAN_FILE_NAME,
    FormalPlanBindings,
    build_formal_task_plan,
    load_formal_task_plan,
    write_formal_task_plan,
)
from skilldrive.seeds.records import SeedRecord


def _config() -> CounterfactualGenerationConfig:
    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1",
        formal_catalog=Path("configs/skills/catalog.yaml"),
        candidate_catalog=Path("configs/skills/candidate_catalog.yaml"),
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=Path("best.pt"),
            sha256="a" * 64,
            run_manifest=Path("run_manifest.json"),
            run_manifest_sha256="b" * 64,
            schema_sha256="c" * 64,
        ),
        inputs=GenerationInputConfig(
            data_root=Path("data"),
            seed_manifest=Path("seeds.csv"),
            seed_manifest_sha256="d" * 64,
            training_cache_manifest=Path("cache.json"),
            training_cache_manifest_sha256="e" * 64,
            leakage_audit=Path("audit.json"),
            leakage_audit_sha256="f" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=2,
            pilot_candidates_per_task=2,
            formal_candidates_per_task=4,
        ),
        formal_skill_ids=("learned", "search"),
        candidate_skill_ids=(),
        skills=(
            SkillGenerationConfig(
                skill_id="learned",
                primary_generated_role="actor",
                proposal_mode="learned_conditioned_prior",
                condition_skill_strategy="requested_skill_id",
                joint_generation_limited=False,
            ),
            SkillGenerationConfig(
                skill_id="search",
                primary_generated_role="actor",
                proposal_mode="rule_guided_prior_search",
                condition_skill_strategy="none_skill_id",
                joint_generation_limited=False,
            ),
        ),
    )


def _record(skill_id: str, scenario_id: str) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"{scenario_id}-actor",
        responder_track_id=f"{scenario_id}-other",
        role_track_ids={
            "actor": f"{scenario_id}-actor",
            "other": f"{scenario_id}-other",
        },
        trigger_score=0.5,
        seed_risk_metric="metric",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "metric",
            "target_range": [0.0, 2.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"value": 1.0},
    )


def _records() -> list[SeedRecord]:
    return [
        _record("learned", "scene-a"),
        _record("search", "scene-a"),
        _record("learned", "scene-b"),
        _record("search", "scene-b"),
        _record("learned", "scene-c"),
        _record("search", "scene-c"),
        _record("learned", "scene-d"),
    ]


def _bindings(
    config: CounterfactualGenerationConfig,
    *,
    tasks_per_shard: int = 3,
) -> FormalPlanBindings:
    return FormalPlanBindings.for_fixture(
        config,
        config_sha256={
            "generation_config": "1" * 64,
            "filter_config": "2" * 64,
            "performance_config": "6" * 64,
        },
        source_sha256={
            "formal_seed_manifest": "3" * 64,
            "generation_source": "4" * 64,
            "filter_source": "5" * 64,
        },
        tasks_per_shard=tasks_per_shard,
        expected_task_count=7,
        expected_scenario_count=4,
        expected_skill_ids=config.formal_skill_ids,
        filter_semantic_sha256="7" * 64,
        generation_execution_sha256="8" * 64,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_inputs(
    tmp_path: Path,
    config: CounterfactualGenerationConfig,
) -> tuple[CounterfactualGenerationConfig, dict[str, object]]:
    seed_manifest = tmp_path / "formal-seeds.csv"
    leakage_audit = tmp_path / "formal-train-boundary.json"
    generation_config = tmp_path / "generation.yaml"
    filter_config = tmp_path / "filter.yaml"
    performance_config = tmp_path / "performance.json"
    detection_config = tmp_path / "detection.yaml"
    generation_source = tmp_path / "generation.py"
    filter_source = tmp_path / "filter.py"
    filter_additional = tmp_path / "filter-extra.py"
    for path, payload in (
        (seed_manifest, "formal seeds\n"),
        (leakage_audit, '{"formal_train_only":true}\n'),
        (generation_config, "contract: formal_v1\n"),
        (filter_config, "contract: filters_v1\n"),
        (performance_config, '{"profile":"frozen"}\n'),
        (detection_config, "version: 1\n"),
        (generation_source, "GENERATION = 1\n"),
        (filter_source, "FILTER = 1\n"),
        (filter_additional, "FILTER_EXTRA = 1\n"),
    ):
        path.write_text(payload, encoding="utf-8")
    updated = replace(
        config,
        inputs=replace(
            config.inputs,
            seed_manifest=seed_manifest,
            seed_manifest_sha256=_file_sha256(seed_manifest),
            leakage_audit=leakage_audit,
            leakage_audit_sha256=_file_sha256(leakage_audit),
        ),
    )
    return updated, {
        "repository_root": tmp_path,
        "generation_config_path": generation_config,
        "filter_config_path": filter_config,
        "performance_config_path": performance_config,
        "detection_config_path": detection_config,
        "filter_additional_paths": (filter_additional,),
        "generation_source_paths": (generation_source,),
        "filter_source_paths": (filter_source,),
        "execution_config": {"batch_size": 32, "workers": 4},
        "tasks_per_shard": 256,
    }


def test_formal_plan_is_input_order_invariant_coverage_first_and_balanced() -> None:
    config = _config()
    bindings = _bindings(config, tasks_per_shard=3)
    first = build_formal_task_plan(_records(), config, bindings=bindings)
    shuffled = build_formal_task_plan(reversed(_records()), config, bindings=bindings)

    assert first == shuffled
    assert first.formal_plan_id == shuffled.formal_plan_id
    assert len(first.tasks) == 7
    assert [task.phase for task in first.tasks] == [
        "coverage",
        "coverage",
        "coverage",
        "coverage",
        "balance",
        "balance",
        "balance",
    ]
    assert [task.phase_index for task in first.tasks] == [0, 1, 2, 3, 0, 1, 2]
    assert [task.shard_index for task in first.tasks] == [0, 0, 0, 1, 1, 1, 2]
    assert len({task.scenario_id for task in first.tasks[:4]}) == 4
    assert {task.scenario_id for task in first.tasks[:4]} == {
        task.scenario_id for task in first.tasks
    }
    balance_skills = [task.skill_id for task in first.tasks[4:]]
    remaining_per_skill = Counter(balance_skills)
    expected_round_robin = [
        skill_id
        for round_index in range(max(remaining_per_skill.values()))
        for skill_id in sorted(remaining_per_skill)
        if round_index < remaining_per_skill[skill_id]
    ]
    assert balance_skills == expected_round_robin


def test_formal_plan_coverage_phase_contains_all_5000_scenarios() -> None:
    config = _config()
    records = [_record("learned", f"scene-{index:04d}") for index in range(5000)]
    records.extend(
        _record("search", f"scene-{index:04d}") for index in range(3)
    )

    plan = build_formal_task_plan(
        reversed(records),
        config,
        bindings=FormalPlanBindings.for_fixture(
            config,
            config_sha256={
                "generation_config": "1" * 64,
                "filter_config": "2" * 64,
                "performance_config": "6" * 64,
            },
            source_sha256={
                "formal_seed_manifest": "3" * 64,
                "generation_source": "4" * 64,
                "filter_source": "5" * 64,
            },
            tasks_per_shard=256,
            expected_task_count=len(records),
            expected_scenario_count=5000,
            expected_skill_ids=config.formal_skill_ids,
            filter_semantic_sha256="7" * 64,
            generation_execution_sha256="8" * 64,
        ),
    )

    coverage = plan.tasks[:5000]
    balance = plan.tasks[5000:]
    assert plan.scenario_count == 5000
    assert len(coverage) == 5000
    assert len({task.scenario_id for task in coverage}) == 5000
    assert all(task.phase == "coverage" for task in coverage)
    assert len(balance) == 3
    assert all(task.phase == "balance" for task in balance)
    assert len({task.task_id for task in plan.tasks}) == len(records)


def test_formal_bindings_freeze_resume_target_and_validation_boundary() -> None:
    bindings = _bindings(_config())

    assert bindings.resume_mode == "auto"
    assert bindings.target_accepted_per_skill == 300
    assert bindings.internal_validation_accessed is False
    assert bindings.final_validation_accessed is False
    assert bindings.formal_train_only is True
    assert bindings.filter_semantic_sha256 == "7" * 64
    assert bindings.generation_execution_sha256 == "8" * 64
    assert bindings.seed_manifest_sha256 == _config().inputs.seed_manifest_sha256
    assert (
        bindings.formal_train_boundary_audit_sha256
        == _config().inputs.leakage_audit_sha256
    )
    assert bindings.to_dict()["internal_validation_accessed"] is False
    assert bindings.to_dict()["final_validation_accessed"] is False

    with pytest.raises(ValueError, match="resume_mode"):
        replace(bindings, resume_mode="never")
    with pytest.raises(ValueError, match="target_accepted_per_skill"):
        replace(bindings, target_accepted_per_skill=299)
    with pytest.raises(ValueError, match="internal_validation_accessed=false"):
        replace(bindings, internal_validation_accessed=True)
    with pytest.raises(ValueError, match="final_validation_accessed=false"):
        replace(bindings, final_validation_accessed=True)
    with pytest.raises(ValueError, match="formal_train_only=true"):
        replace(bindings, formal_train_only=False)
    with pytest.raises(ValueError, match="filter_semantic_sha256"):
        replace(bindings, filter_semantic_sha256="not-a-hash")
    with pytest.raises(ValueError, match="generation_execution_sha256"):
        replace(bindings, generation_execution_sha256="not-a-hash")


def test_production_bindings_freeze_official_scale_and_34_skill_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_ids = tuple(f"formal-skill-{index:02d}" for index in range(34))
    config = replace(
        _config(),
        formal_skill_ids=skill_ids,
        skills=tuple(
            SkillGenerationConfig(
                skill_id=skill_id,
                primary_generated_role="actor",
                proposal_mode="rule_guided_prior_search",
                condition_skill_strategy="none_skill_id",
                joint_generation_limited=False,
            )
            for skill_id in skill_ids
        ),
    )

    config, production_inputs = _production_inputs(tmp_path, config)
    filter_calls: list[dict[str, object]] = []

    def fake_filter_fingerprint(**kwargs):
        filter_calls.append(kwargs)
        return FilterSemanticFingerprint(
            semantic_sha256="7" * 64,
            file_sha256={},
        )

    monkeypatch.setattr(
        formal_module,
        "build_filter_semantic_fingerprint",
        fake_filter_fingerprint,
    )
    bindings = FormalPlanBindings.from_generation_config(
        config,
        **production_inputs,
    )

    assert bindings.profile == "production"
    assert bindings.expected_task_count == FORMAL_PRODUCTION_EXPECTED_TASK_COUNT
    assert (
        bindings.expected_scenario_count
        == FORMAL_PRODUCTION_EXPECTED_SCENARIO_COUNT
    )
    assert len(bindings.expected_skill_ids) == FORMAL_PRODUCTION_SKILL_COUNT
    assert bindings.expected_skill_ids == tuple(sorted(skill_ids))
    assert bindings.filter_semantic_sha256 == "7" * 64
    assert bindings.generation_execution_sha256 == canonical_sha256(
        production_inputs["execution_config"]
    )
    assert filter_calls[-1]["detection_config_path"] == production_inputs[
        "detection_config_path"
    ]
    assert filter_calls[-1]["additional_paths"] == production_inputs[
        "filter_additional_paths"
    ]

    changed_execution = dict(production_inputs)
    changed_execution["execution_config"] = {"batch_size": 64, "workers": 4}
    changed_bindings = FormalPlanBindings.from_generation_config(
        config,
        **changed_execution,
    )
    assert (
        changed_bindings.generation_execution_sha256
        != bindings.generation_execution_sha256
    )
    assert changed_bindings.filter_semantic_sha256 == bindings.filter_semantic_sha256

    with pytest.raises(ValueError, match="exactly 34 formal skills"):
        FormalPlanBindings.from_generation_config(
            replace(
                config,
                formal_skill_ids=_config().formal_skill_ids,
                skills=_config().skills,
            ),
            **production_inputs,
        )

    config.inputs.seed_manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed manifest SHA-256"):
        FormalPlanBindings.from_generation_config(
            config,
            **production_inputs,
        )


def test_formal_bindings_require_named_config_and_source_hashes() -> None:
    config = _config()
    fixture_arguments = {
        "tasks_per_shard": 3,
        "expected_task_count": 7,
        "expected_scenario_count": 4,
        "expected_skill_ids": config.formal_skill_ids,
        "filter_semantic_sha256": "7" * 64,
        "generation_execution_sha256": "8" * 64,
    }
    with pytest.raises(ValueError, match="performance_config"):
        FormalPlanBindings.for_fixture(
            config,
            config_sha256={
                "generation_config": "1" * 64,
                "filter_config": "2" * 64,
            },
            source_sha256={
                "generation_source": "4" * 64,
                "filter_source": "5" * 64,
            },
            **fixture_arguments,
        )
    with pytest.raises(ValueError, match="filter_source"):
        FormalPlanBindings.for_fixture(
            config,
            config_sha256={
                "generation_config": "1" * 64,
                "filter_config": "2" * 64,
                "performance_config": "6" * 64,
            },
            source_sha256={"generation_source": "4" * 64},
            **fixture_arguments,
        )


def test_formal_plan_rejects_manifest_boundary_and_actual_scope_drift() -> None:
    config = _config()
    records = _records()
    bindings = _bindings(config)

    with pytest.raises(ValueError, match="seed manifest"):
        build_formal_task_plan(
            records,
            config,
            bindings=replace(bindings, seed_manifest_sha256="9" * 64),
        )
    with pytest.raises(ValueError, match="boundary audit"):
        build_formal_task_plan(
            records,
            config,
            bindings=replace(
                bindings,
                formal_train_boundary_audit_sha256="9" * 64,
            ),
        )
    with pytest.raises(ValueError, match="task count"):
        build_formal_task_plan(
            records,
            config,
            bindings=replace(bindings, expected_task_count=8),
        )
    with pytest.raises(ValueError, match="scenario count"):
        build_formal_task_plan(
            records,
            config,
            bindings=replace(bindings, expected_scenario_count=5),
        )

    learned_only = [record for record in records if record.skill_id == "learned"]
    learned_only_bindings = FormalPlanBindings.for_fixture(
        config,
        config_sha256=dict(bindings.config_sha256),
        source_sha256=dict(bindings.source_sha256),
        tasks_per_shard=3,
        expected_task_count=len(learned_only),
        expected_scenario_count=4,
        expected_skill_ids=config.formal_skill_ids,
        filter_semantic_sha256=bindings.filter_semantic_sha256,
        generation_execution_sha256=bindings.generation_execution_sha256,
    )
    with pytest.raises(ValueError, match="skill set"):
        build_formal_task_plan(
            learned_only,
            config,
            bindings=learned_only_bindings,
        )


def test_formal_plan_atomic_round_trip_hash_and_semantic_drift_rejection(
    tmp_path: Path,
) -> None:
    config = _config()
    bindings = _bindings(config)
    plan = build_formal_task_plan(_records(), config, bindings=bindings)
    artifacts = write_formal_task_plan(tmp_path, plan, config=config)
    repeated = write_formal_task_plan(tmp_path, plan, config=config)
    loaded = load_formal_task_plan(
        tmp_path,
        expected_bindings=bindings,
        config=config,
    )

    assert loaded == plan
    assert loaded.formal_plan_id == plan.formal_plan_id
    assert repeated.task_plan_sha256 == artifacts.task_plan_sha256
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["formal_plan_id"] == plan.formal_plan_id
    assert summary["task_plan"]["sha256"] == artifacts.task_plan_sha256
    assert summary["counts"]["by_phase"] == {"balance": 3, "coverage": 4}
    assert summary["bindings"]["filter_semantic_sha256"] == "7" * 64
    assert summary["bindings"]["generation_execution_sha256"] == "8" * 64
    changed_filter = replace(bindings, filter_semantic_sha256="9" * 64)
    changed_execution = replace(bindings, generation_execution_sha256="a" * 64)
    assert replace(plan, bindings=changed_filter).formal_plan_id != plan.formal_plan_id
    assert replace(plan, bindings=changed_execution).formal_plan_id != plan.formal_plan_id

    changed_source_hashes = dict(bindings.source_sha256)
    changed_source_hashes["formal_seed_manifest"] = "9" * 64
    changed_sources = replace(bindings, source_sha256=changed_source_hashes)
    assert replace(plan, bindings=changed_sources).formal_plan_id != plan.formal_plan_id
    with pytest.raises(ValueError, match="semantic drift"):
        load_formal_task_plan(
            tmp_path,
            expected_bindings=changed_sources,
            config=config,
        )

    artifacts.task_plan_path.write_bytes(
        artifacts.task_plan_path.read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="size differs"):
        load_formal_task_plan(
            tmp_path,
            expected_bindings=bindings,
            config=config,
        )


def test_formal_plan_rejects_candidate_budget_drift_from_generation_config() -> None:
    config = _config()
    drifted = replace(
        _bindings(config),
        candidate_budget=config.sampling.formal_candidates_per_task + 1,
    )

    with pytest.raises(ValueError, match="candidate budget differs"):
        build_formal_task_plan(_records(), config, bindings=drifted)


def test_formal_plan_rejects_duplicate_seed_labels() -> None:
    config = _config()
    record = _record("learned", "scene-a")
    with pytest.raises(ValueError, match="duplicate seed record key"):
        build_formal_task_plan(
            [record, record],
            config,
            bindings=_bindings(config),
        )


def test_formal_plan_file_name_is_versioned_for_formal_use() -> None:
    assert FORMAL_TASK_PLAN_FILE_NAME == "formal_task_plan.jsonl"

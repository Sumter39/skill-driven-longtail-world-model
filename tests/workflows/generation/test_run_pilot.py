from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.generation.run_counterfactual_pipeline as pipeline_module
import skilldrive.data as skilldrive_data
import skilldrive.data.av2_reader as av2_reader
import skilldrive.filtering.fingerprint as filter_fingerprint
import skilldrive.filtering.pipeline as filter_pipeline
import skilldrive.generation.inference as generation_inference
import skilldrive.skills.detection as skill_detection
import skilldrive.skills.loader as skill_loader
from skilldrive.generation import (
    FilterDecision,
    GeneratedCandidate,
    GeneratedOverlay,
    canonical_sha256,
    latent_group_id,
    load_raw_shard_candidates,
    load_task_plan,
    paired_latent_seed,
    pilot_evaluation_arm,
    recover_paired_pilot_tasks,
    seed_record_id,
    select_pilot_records,
    write_raw_shard,
)
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
)
from skilldrive.seeds.records import SeedRecord


CHECKPOINT_SHA = "a" * 64
RUN_MANIFEST_SHA = "b" * 64
SCHEMA_SHA = "c" * 64
SEED_MANIFEST_SHA = "d" * 64
FILTER_SHA = "f" * 64
LEARNED_SKILL = "learned_skill"
RULE_SKILL = "rule_skill"
CANDIDATE_BUDGET = 2
TASK_BATCH_SIZE = 8


@dataclass
class _PilotHarness:
    root: Path
    config: CounterfactualGenerationConfig
    records: tuple[SeedRecord, ...]
    config_path: Path
    filter_config_path: Path
    detection_config_path: Path
    output_root: Path
    calls: dict[str, Any]

    @property
    def arguments(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "filter_config_path": self.filter_config_path,
            "detection_config_path": self.detection_config_path,
            "output_root": self.output_root,
            "device": "cuda",
            "task_batch_size": TASK_BATCH_SIZE,
            "progress_interval_seconds": 0.001,
        }


class _MockValidation:
    def __init__(self, candidate_input, calls: dict[str, Any]) -> None:
        self._candidate_input = candidate_input
        self._calls = calls

    def compact(self, *, cohort: str):
        bound = self._candidate_input.bound
        raw = bound.raw
        task = bound.task
        arm = raw.metadata["evaluation_arm"]
        trigger_passed = (
            arm == "rule_guided_none"
            or (arm == "learned_conditioned" and raw.candidate_index == 0)
            or (arm == "learned_none_control" and raw.candidate_index == 1)
        )
        risk_base = {
            "learned_conditioned": 10.0,
            "learned_none_control": 2.0,
            "rule_guided_none": 5.0,
        }[arm]

        def timed_check(stage: str, passed: bool, metrics: dict[str, Any]):
            return SimpleNamespace(
                check=SimpleNamespace(
                    stage=SimpleNamespace(value=stage),
                    passed=passed,
                    metrics=metrics,
                )
            )

        checks = [
            timed_check(
                "target_risk",
                True,
                {"evaluation": {"value": risk_base + raw.candidate_index}},
            ),
            timed_check("skill_trigger", trigger_passed, {}),
        ]
        if trigger_passed:
            checks.append(
                timed_check(
                    "parameter_realization",
                    True,
                    {
                        "parameters": {
                            "reaction_time_s": {
                                "absolute_error": 0.1 * (raw.candidate_index + 1)
                            },
                            "target_gap_m": {
                                "absolute_error": 0.25 * (raw.candidate_index + 1)
                            },
                        }
                    },
                )
            )
        result = SimpleNamespace(
            identity=SimpleNamespace(
                candidate_id=raw.candidate_id,
                task_id=raw.task_id,
                candidate_index=raw.candidate_index,
                latent_seed=raw.latent_seed,
                scenario_id=raw.scenario_id,
                skill_id=raw.skill_id,
                target_track_id=raw.target_track_id,
                seed_record_id=task.seed_record_id,
                proposal_mode=raw.proposal_mode,
                checkpoint_sha256=raw.checkpoint_sha256,
                semantic_config_sha256=raw.semantic_config_sha256,
            ),
            cohort=cohort,
            checks=tuple(checks),
            quality_passed=trigger_passed,
        )
        self._calls["compact_cohorts"].append(
            (raw.candidate_id, task.task_id, cohort)
        )
        return result


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
        seed_risk_metric="minimum_distance_m",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "minimum_distance_m",
            "target_range": [0.0, 2.0],
            "source": "semantic",
            "direction": "lower_is_riskier",
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"target_gap_m": 1.0},
    )


def _config(root: Path) -> CounterfactualGenerationConfig:
    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1_test",
        formal_catalog=root / "cfg" / "catalog.yaml",
        candidate_catalog=root / "cfg" / "candidate_catalog.yaml",
        none_skill_id="<none>",
        active_checkpoint=ActiveCheckpointConfig(
            path=root / "model.pt",
            sha256=CHECKPOINT_SHA,
            run_manifest=root / "run_manifest.json",
            run_manifest_sha256=RUN_MANIFEST_SHA,
            schema_sha256=SCHEMA_SHA,
        ),
        inputs=GenerationInputConfig(
            data_root=root / "data",
            seed_manifest=root / "seeds.csv",
            seed_manifest_sha256=SEED_MANIFEST_SHA,
            training_cache_manifest=root / "cache.json",
            training_cache_manifest_sha256="e" * 64,
            leakage_audit=root / "audit.json",
            leakage_audit_sha256="9" * 64,
        ),
        sampling=SamplingConfig(
            base_seed=2026,
            pilot_seed_records_per_skill=1,
            pilot_candidates_per_task=CANDIDATE_BUDGET,
            formal_candidates_per_task=4,
        ),
        formal_skill_ids=(LEARNED_SKILL, RULE_SKILL),
        candidate_skill_ids=(),
        skills=(
            SkillGenerationConfig(
                skill_id=LEARNED_SKILL,
                primary_generated_role="actor",
                proposal_mode="learned_conditioned_prior",
                condition_skill_strategy="requested_skill_id",
                joint_generation_limited=False,
            ),
            SkillGenerationConfig(
                skill_id=RULE_SKILL,
                primary_generated_role="actor",
                proposal_mode="rule_guided_prior_search",
                condition_skill_strategy="none_skill_id",
                joint_generation_limited=False,
            ),
        ),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _install_harness(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _PilotHarness:
    config = _config(root)
    records = (
        _record(LEARNED_SKILL, "scene-learned"),
        _record(RULE_SKILL, "scene-rule"),
    )
    for record in records:
        source = config.inputs.data_root.joinpath(*Path(record.source_path).parts)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()

    calls: dict[str, Any] = {
        "audit": 0,
        "gpu_loads": 0,
        "inference_batches": [],
        "history_paths": [],
        "full_paths": [],
        "prior_specs": [],
        "compact_cohorts": [],
        "finalized_cohorts": [],
    }
    config_path = root / "g.yaml"
    filter_config_path = root / "f.yaml"
    detection_config_path = root / "d.yaml"
    output_root = root / "o"

    def run_audit(**kwargs):
        calls["audit"] += 1
        return {"status": "passed"}

    def load_history(path: Path):
        calls["history_paths"].append(Path(path))
        return SimpleNamespace(
            timestamps=np.arange(50, dtype=np.int64),
            metadata={"temporal_scope": "history_only"},
        )

    def load_full(path: Path):
        calls["full_paths"].append(Path(path))
        return SimpleNamespace(source_path=Path(path))

    def tensorize_prior_context(scenario, spec, schema):
        calls["prior_specs"].append(spec)
        return SimpleNamespace(
            target_track_id=spec.target_track_id,
            anchor_origin_global=np.zeros(2, dtype=np.float64),
            anchor_heading_global=0.0,
        )

    def load_active_cvae(**kwargs):
        calls["gpu_loads"] += 1
        return SimpleNamespace(device=kwargs["device"])

    def generate_prior_batch(runtime, contexts, latent_seeds, **kwargs):
        seeds = np.asarray(latent_seeds, dtype=np.int64)
        calls["inference_batches"].append(seeds.copy())
        futures = np.empty(
            (len(contexts), seeds.shape[1], 60, 2),
            dtype=np.float32,
        )
        base = np.linspace(0.0, 6.0, 60, dtype=np.float32)
        for row in range(len(contexts)):
            for candidate_index in range(seeds.shape[1]):
                offset = np.float32(int(seeds[row, candidate_index]) % 997) / 997.0
                futures[row, candidate_index, :, 0] = base + offset
                futures[row, candidate_index, :, 1] = offset
        return SimpleNamespace(future_position_local=futures)

    def validate_candidate(candidate_input, **kwargs):
        return _MockValidation(candidate_input, calls)

    def finalize_candidate_validations(
        compact_results,
        *,
        filter_semantic_sha256,
        **kwargs,
    ):
        compact = tuple(compact_results)
        calls["finalized_cohorts"].append(
            tuple((item.identity.task_id, item.cohort) for item in compact)
        )
        decisions = tuple(
            FilterDecision.create(
                candidate_id=item.identity.candidate_id,
                filter_config_sha256=filter_semantic_sha256,
                filter_contract_version=filter_pipeline.FILTER_CONTRACT_VERSION,
                accepted=item.quality_passed,
                rejection_reasons=(
                    () if item.quality_passed else ("skill.trigger_not_realized",)
                ),
                metrics={
                    "task_id": item.identity.task_id,
                    "candidate_index": item.identity.candidate_index,
                    "latent_seed": item.identity.latent_seed,
                    "cohort": item.cohort,
                    "first_failed_stage": (
                        None if item.quality_passed else "skill_trigger"
                    ),
                },
            )
            for item in compact
        )
        return SimpleNamespace(
            decisions=decisions,
            validations=compact,
            stage_execution_counts={"mock_filter": len(compact)},
            stage_elapsed_seconds={"mock_filter": 0.0},
        )

    monkeypatch.setattr(pipeline_module, "run_audit", run_audit)
    monkeypatch.setattr(
        pipeline_module,
        "load_counterfactual_config",
        lambda path: config,
    )
    monkeypatch.setattr(pipeline_module, "load_filter_config", lambda path: object())
    monkeypatch.setattr(pipeline_module, "read_seed_records", lambda path: records)
    monkeypatch.setattr(pipeline_module, "build_cvae_schema", lambda path: object())
    monkeypatch.setattr(av2_reader, "load_av2_history_scenario", load_history)
    monkeypatch.setattr(av2_reader, "load_av2_scenario", load_full)
    monkeypatch.setattr(
        skilldrive_data,
        "tensorize_prior_context",
        tensorize_prior_context,
    )
    monkeypatch.setattr(generation_inference, "load_configured_cvae", load_active_cvae)
    monkeypatch.setattr(generation_inference, "generate_prior_batch", generate_prior_batch)
    monkeypatch.setattr(
        filter_fingerprint,
        "build_filter_semantic_fingerprint",
        lambda **kwargs: SimpleNamespace(
            semantic_sha256=FILTER_SHA,
            file_sha256={"filters.yaml": FILTER_SHA},
        ),
    )
    monkeypatch.setattr(filter_pipeline, "validate_candidate", validate_candidate)
    monkeypatch.setattr(
        filter_pipeline,
        "finalize_candidate_validations",
        finalize_candidate_validations,
    )
    monkeypatch.setattr(skill_detection, "load_detection_config", lambda path: object())
    monkeypatch.setattr(
        skill_loader,
        "load_skill",
        lambda path: SimpleNamespace(skill_id=Path(path).stem),
    )
    return _PilotHarness(
        root=root,
        config=config,
        records=records,
        config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
        output_root=output_root,
        calls=calls,
    )


def _load_persisted_plan(summary: dict[str, Any]):
    plan_root = Path(summary["outputs"]["task_plan"]).parent
    return load_task_plan(
        plan_root,
        expected_semantic_config_sha256=summary["generation_semantic_sha256"],
        current_execution_config_sha256=summary["generation_execution_sha256"],
    ).plan


def _candidate_for_task(
    plan,
    task,
    record: SeedRecord,
    config: CounterfactualGenerationConfig,
    candidate_index: int,
    *,
    marker: float,
) -> GeneratedCandidate:
    skill = config.skills_by_id[task.skill_id]
    return GeneratedCandidate(
        task_id=task.task_id,
        candidate_index=candidate_index,
        latent_seed=paired_latent_seed(plan.base_seed, task, candidate_index),
        scenario_id=task.scenario_id,
        skill_id=task.skill_id,
        proposal_mode=task.proposal_mode,
        checkpoint_sha256=task.checkpoint_sha256,
        semantic_config_sha256=task.semantic_config_sha256,
        overlay=GeneratedOverlay(
            target_track_id=task.target_track_id,
            future_xy_global=np.full((60, 2), marker, dtype=np.float32),
        ),
        metadata={
            "condition_skill_id": task.condition_skill_id,
            "evaluation_arm": pilot_evaluation_arm(
                task,
                none_skill_id=config.none_skill_id,
            ),
            "latent_group_id": latent_group_id(task),
            "primary_generated_role": skill.primary_generated_role,
            "requested_parameters": record.sampled_parameters,
            "detection_mode": record.evidence["detection_mode"],
        },
    )


@pytest.fixture
def short_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("rp")


def test_pilot_persists_paired_identity_and_separates_control_cohort(
    short_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(short_root, monkeypatch)

    summary = pipeline_module.run_pilot(**harness.arguments)
    plan = _load_persisted_plan(summary)
    recovery = recover_paired_pilot_tasks(plan, Path(summary["outputs"]["raw"]))
    eligibility_audit_path = Path(summary["outputs"]["eligibility_audit"])
    eligibility_audit = json.loads(eligibility_audit_path.read_text(encoding="utf-8"))

    assert eligibility_audit["formal_train_only"] is True
    assert eligibility_audit["validation_manifests_opened"] is False
    assert eligibility_audit["final_validation_accessed"] is False
    assert eligibility_audit["selection_stable"] is True
    assert eligibility_audit["skills_without_eligible_seed_records"] == []
    assert eligibility_audit["counts"] == {
        "input_records": 2,
        "attempted_records": 2,
        "unique_contexts_evaluated": 2,
        "context_cache_hits": 0,
        "selected_records": 2,
        "excluded_records": 0,
        "formal_skills_covered": 2,
        "formal_skills_without_eligible_seed_records": 0,
    }
    assert summary["pilot_seed_eligibility_sha256"] == canonical_sha256(
        eligibility_audit
    )
    assert summary["execution_config"]["pilot_seed_eligibility_sha256"] == summary[
        "pilot_seed_eligibility_sha256"
    ]
    assert summary["output_sha256"]["eligibility_audit"] == pipeline_module._file_sha256(
        eligibility_audit_path
    )

    assert Counter(pilot_evaluation_arm(task) for task in plan.tasks) == {
        "learned_conditioned": 1,
        "learned_none_control": 1,
        "rule_guided_none": 1,
    }
    learned_tasks = [task for task in plan.tasks if task.skill_id == LEARNED_SKILL]
    conditioned = next(
        task
        for task in learned_tasks
        if pilot_evaluation_arm(task) == "learned_conditioned"
    )
    control = next(
        task
        for task in learned_tasks
        if pilot_evaluation_arm(task) == "learned_none_control"
    )
    rule = next(task for task in plan.tasks if task.skill_id == RULE_SKILL)
    assert conditioned.task_id != control.task_id
    assert conditioned.condition_skill_id == LEARNED_SKILL
    assert control.condition_skill_id == "<none>"
    assert rule.condition_skill_id == "<none>"

    stored_by_task = {
        task.task_id: load_raw_shard_candidates(
            next(
                shard
                for shard in recovery.raw_scan.valid_shards
                if shard.shard_index == task.task_index
            ),
            expected_semantic_config_sha256=plan.semantic_config_sha256,
        )
        for task in plan.tasks
    }
    for candidate_index in range(CANDIDATE_BUDGET):
        conditioned_raw = stored_by_task[conditioned.task_id][candidate_index]
        control_raw = stored_by_task[control.task_id][candidate_index]
        assert conditioned_raw.latent_seed == control_raw.latent_seed
        assert conditioned_raw.candidate_id != control_raw.candidate_id
        np.testing.assert_array_equal(
            conditioned_raw.future_xy_global,
            control_raw.future_xy_global,
        )
    assert all(
        raw.metadata["condition_skill_id"] == "<none>"
        for raw in stored_by_task[rule.task_id]
    )

    specs_by_condition = Counter(spec.condition_skill_id for spec in harness.calls["prior_specs"])
    assert specs_by_condition == {LEARNED_SKILL: 1, "<none>": 2}
    assert all(
        not spec.role_track_ids
        for spec in harness.calls["prior_specs"]
        if spec.condition_skill_id == "<none>"
    )

    task_arm = {task.task_id: pilot_evaluation_arm(task) for task in plan.tasks}
    observed_cohorts = {
        task_id: {cohort}
        for task_id, cohort in harness.calls["finalized_cohorts"][0]
    }
    assert observed_cohorts[control.task_id] == {"learned_none_control"}
    assert observed_cohorts[conditioned.task_id] == {"formal"}
    assert observed_cohorts[rule.task_id] == {"formal"}
    accepted = _read_jsonl(Path(summary["outputs"]["accepted"]))
    assert {
        row["task_id"]: row["metrics"]["cohort"] for row in accepted
    } == {
        task_id: (
            "learned_none_control"
            if arm == "learned_none_control"
            else "formal"
        )
        for task_id, arm in task_arm.items()
    }

    rows = {
        (row["skill_id"], row["evaluation_arm"]): row
        for row in summary["by_skill_and_arm"]
    }
    conditioned_row = rows[(LEARNED_SKILL, "learned_conditioned")]
    control_row = rows[(LEARNED_SKILL, "learned_none_control")]
    rule_row = rows[(RULE_SKILL, "rule_guided_none")]
    assert conditioned_row["parameter_absolute_errors"] == {
        "reaction_time_s": {"count": 1, "median": 0.1, "maximum": 0.1},
        "target_gap_m": {"count": 1, "median": 0.25, "maximum": 0.25},
    }
    assert control_row["parameter_absolute_errors"] == {
        "reaction_time_s": {"count": 1, "median": 0.2, "maximum": 0.2},
        "target_gap_m": {"count": 1, "median": 0.5, "maximum": 0.5},
    }
    assert rule_row["parameter_absolute_errors"] == {
        "reaction_time_s": {
            "count": 2,
            "median": pytest.approx(0.15),
            "maximum": 0.2,
        },
        "target_gap_m": {"count": 2, "median": 0.375, "maximum": 0.5},
    }
    assert conditioned_row["first_failed_stages"] == {"skill_trigger": 1}
    assert conditioned_row["primary_rejections"] == {
        "skill.trigger_not_realized": 1
    }

    assert "risk_delta_conditioned_minus_control_median" not in summary[
        "paired_control"
    ]
    assert summary["paired_control"]["by_skill"] == [
        {
            "skill_id": LEARNED_SKILL,
            "shared_latent_pairs": 2,
            "both_quality_pass": 0,
            "conditioned_only_quality_pass": 1,
            "control_only_quality_pass": 1,
            "neither_quality_pass": 0,
            "conditioned_accepted": 1,
            "control_accepted": 1,
            "conditioned_trigger_passed": 1,
            "control_trigger_passed": 1,
            "conditioned_only_trigger_passed": 1,
            "control_only_trigger_passed": 1,
            "risk_delta_conditioned_minus_control": {
                "count": 2,
                "median": 8.0,
            },
        }
    ]


def test_pilot_eligibility_replaces_invalid_seed_before_freezing_plan(
    short_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(short_root, monkeypatch)
    learned_records = tuple(
        _record(LEARNED_SKILL, f"scene-learned-{index}") for index in range(3)
    )
    rule_record = _record(RULE_SKILL, "scene-rule")
    records = (*learned_records, rule_record)
    for record in learned_records:
        source = harness.config.inputs.data_root.joinpath(*Path(record.source_path).parts)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    ordered = [
        record
        for record in select_pilot_records(
            records,
            formal_skill_ids=harness.config.formal_skill_ids,
            per_skill=len(records),
            base_seed=harness.config.sampling.base_seed,
        )
        if record.skill_id == LEARNED_SKILL
    ]
    invalid_record, replacement_record = ordered[:2]
    monkeypatch.setattr(pipeline_module, "read_seed_records", lambda path: records)

    def tensorize_with_invalid_seed(scenario, spec, schema):
        harness.calls["prior_specs"].append(spec)
        if spec.scenario_id == invalid_record.scenario_id:
            raise ValueError("target has fewer than 30 valid history steps")
        return SimpleNamespace(
            target_track_id=spec.target_track_id,
            anchor_origin_global=np.zeros(2, dtype=np.float64),
            anchor_heading_global=0.0,
        )

    monkeypatch.setattr(
        skilldrive_data,
        "tensorize_prior_context",
        tensorize_with_invalid_seed,
    )

    summary = pipeline_module.run_pilot(**harness.arguments)
    plan = _load_persisted_plan(summary)
    planned_seed_ids = {task.seed_record_id for task in plan.tasks}
    audit = json.loads(
        Path(summary["outputs"]["eligibility_audit"]).read_text(encoding="utf-8")
    )

    assert seed_record_id(invalid_record) not in planned_seed_ids
    assert seed_record_id(replacement_record) in planned_seed_ids
    assert {task.skill_id for task in plan.tasks} == {LEARNED_SKILL, RULE_SKILL}
    assert len(plan.tasks) == 3
    assert audit["counts"]["formal_skills_covered"] == 2
    assert audit["counts"]["selected_records"] == 2
    assert audit["counts"]["excluded_records"] == 1
    assert audit["excluded"] == [
        {
            "skill_id": LEARNED_SKILL,
            "scenario_id": invalid_record.scenario_id,
            "seed_record_id": seed_record_id(invalid_record),
            "target_track_id": invalid_record.role_track_ids["actor"],
            "source_path": invalid_record.source_path,
            "context_fingerprint": audit["excluded"][0]["context_fingerprint"],
            "candidate_rank": 0,
            "context_cache_hit": False,
            "failure_type": "ValueError",
            "failure_message": "target has fewer than 30 valid history steps",
        }
    ]


def test_pilot_records_zero_eligible_skill_without_fabricating_task(
    short_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(short_root, monkeypatch)
    learned_record = next(
        record for record in harness.records if record.skill_id == LEARNED_SKILL
    )

    def tensorize_without_learned_seed(scenario, spec, schema):
        harness.calls["prior_specs"].append(spec)
        if spec.scenario_id == learned_record.scenario_id:
            raise ValueError("target has fewer than 30 valid history steps")
        return SimpleNamespace(
            target_track_id=spec.target_track_id,
            anchor_origin_global=np.zeros(2, dtype=np.float64),
            anchor_heading_global=0.0,
        )

    monkeypatch.setattr(
        skilldrive_data,
        "tensorize_prior_context",
        tensorize_without_learned_seed,
    )

    summary = pipeline_module.run_pilot(**harness.arguments)
    plan = _load_persisted_plan(summary)
    audit = json.loads(
        Path(summary["outputs"]["eligibility_audit"]).read_text(encoding="utf-8")
    )

    assert [task.skill_id for task in plan.tasks] == [RULE_SKILL]
    assert seed_record_id(learned_record) not in {
        task.seed_record_id for task in plan.tasks
    }
    assert summary["task_count"] == 1
    assert summary["skills_without_eligible_seed_records"] == [LEARNED_SKILL]
    assert audit["status"] == "completed_with_ineligible_skills"
    assert audit["skills_without_eligible_seed_records"] == [LEARNED_SKILL]
    assert audit["counts"]["formal_skills_covered"] == 1
    assert audit["counts"]["formal_skills_without_eligible_seed_records"] == 1
    learned_row = next(
        row for row in audit["by_skill"] if row["skill_id"] == LEARNED_SKILL
    )
    assert learned_row["selected_records"] == 0
    assert learned_row["status"] == "no_eligible_seed_records"
    assert learned_row["failure_reasons"] == [
        {
            "failure_type": "ValueError",
            "failure_message": "target has fewer than 30 valid history steps",
            "record_count": 1,
        }
    ]


def test_pilot_resume_skips_gpu_and_keeps_durable_outputs_deterministic(
    short_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(short_root, monkeypatch)

    first = pipeline_module.run_pilot(**harness.arguments)
    plan = _load_persisted_plan(first)
    raw_dir = Path(first["outputs"]["raw"])
    recovery = recover_paired_pilot_tasks(plan, raw_dir)
    assert len(recovery.raw_scan.valid_shards) == len(plan.tasks) == 3
    assert all(shard.candidate_count == CANDIDATE_BUDGET for shard in recovery.raw_scan.valid_shards)
    assert all(
        len(
            {
                raw.task_id
                for raw in load_raw_shard_candidates(
                    shard,
                    expected_semantic_config_sha256=plan.semantic_config_sha256,
                )
            }
        )
        == 1
        for shard in recovery.raw_scan.valid_shards
    )
    durable_payloads = {
        path: path.read_bytes()
        for shard in recovery.raw_scan.valid_shards
        for path in (shard.arrays_path, shard.metadata_path, shard.commit_path)
    }
    output_payloads = {
        name: Path(first["outputs"][name]).read_bytes()
        for name in ("accepted", "rejected", "filter_commit")
    }

    def unexpected_gpu_call(*args, **kwargs):
        pytest.fail("a fully durable Pilot resume must not load or invoke the model")

    monkeypatch.setattr(
        generation_inference,
        "load_configured_cvae",
        unexpected_gpu_call,
    )
    monkeypatch.setattr(generation_inference, "generate_prior_batch", unexpected_gpu_call)
    second = pipeline_module.run_pilot(**harness.arguments)

    assert second["resumed_task_count"] == second["task_count"] == 3
    assert second["newly_generated_task_count"] == 0
    assert second["newly_generated_candidate_count"] == 0
    assert harness.calls["gpu_loads"] == 1
    assert len(harness.calls["inference_batches"]) == 1
    assert {path: path.read_bytes() for path in durable_payloads} == durable_payloads
    assert {
        name: Path(second["outputs"][name]).read_bytes()
        for name in output_payloads
    } == output_payloads

    stable_summary_fields = (
        "pilot_run_id",
        "skills_without_eligible_seed_records",
        "task_plan_id",
        "task_count",
        "candidate_count",
        "durable_task_count",
        "task_plan_sha256",
        "task_plan_summary_sha256",
        "raw_commit_set_sha256",
        "raw_snapshot_sha256",
        "generation_semantic_sha256",
        "generation_execution_sha256",
        "pilot_seed_eligibility_sha256",
        "pilot_seed_eligibility_artifact_sha256",
        "filter_semantic_sha256",
        "filter_dependency_sha256",
        "filter_contract_version",
        "checkpoint_sha256",
        "execution_config",
        "paired_control",
        "by_skill_and_arm",
        "outputs",
        "output_sha256",
    )
    assert {name: first[name] for name in stable_summary_fields} == {
        name: second[name] for name in stable_summary_fields
    }
    assert first["task_count"] == 3
    assert first["status"] == "completed"
    assert first["ability_gate_status"] == "pending_analysis"
    assert first["task_plan_id"] == plan.task_plan_id
    assert Path(first["outputs"]["task_plan"]).parent.name == first["pilot_run_id"]
    assert first["candidate_count"] == 3 * CANDIDATE_BUDGET
    assert first["durable_task_count"] == 3
    assert first["newly_generated_task_count"] == 3
    assert first["newly_generated_candidate_count"] == 3 * CANDIDATE_BUDGET
    assert first["validation_manifests_opened"] is False
    assert first["final_validation_accessed"] is False
    assert first["raw_immutable_verified"] is True
    assert first["raw_file_count"] == 3 * first["task_count"]
    assert first["task_plan_sha256"] == pipeline_module._file_sha256(
        Path(first["outputs"]["task_plan"])
    )
    for name in ("accepted", "rejected", "filter_commit"):
        assert first["output_sha256"][name] == pipeline_module._file_sha256(
            Path(first["outputs"][name])
        )


def test_pilot_rebuilds_entire_partial_and_invalid_tasks(
    short_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _install_harness(short_root, monkeypatch)
    initial = pipeline_module.run_pilot(**harness.arguments)
    plan = _load_persisted_plan(initial)
    pilot_root = Path(initial["outputs"]["task_plan"]).parent
    records_by_id = {seed_record_id(record): record for record in harness.records}
    raw_dir = pilot_root / "raw"

    durable_task, partial_task, invalid_task = plan.tasks
    initial_recovery = recover_paired_pilot_tasks(plan, raw_dir)
    durable_commit = next(
        shard
        for shard in initial_recovery.raw_scan.valid_shards
        if shard.shard_index == durable_task.task_index
    )
    partial_commit = write_raw_shard(
        raw_dir,
        partial_task.task_index,
        [
            _candidate_for_task(
                plan,
                partial_task,
                records_by_id[partial_task.seed_record_id],
                harness.config,
                0,
                marker=-123.0,
            )
        ],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )
    invalid_commit = write_raw_shard(
        raw_dir,
        invalid_task.task_index,
        [
            _candidate_for_task(
                plan,
                invalid_task,
                records_by_id[invalid_task.seed_record_id],
                harness.config,
                index,
                marker=-456.0 - index,
            )
            for index in range(CANDIDATE_BUDGET)
        ],
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
    )
    invalid_commit.arrays_path.write_bytes(
        invalid_commit.arrays_path.read_bytes() + b"corrupt"
    )
    durable_identity = {
        path: path.read_bytes()
        for path in (
            durable_commit.arrays_path,
            durable_commit.metadata_path,
            durable_commit.commit_path,
        )
    }
    before = recover_paired_pilot_tasks(plan, raw_dir)
    assert before.durable_task_ids == {durable_task.task_id}
    assert before.partial_task_ids == {partial_task.task_id}
    assert before.invalid_task_ids == {invalid_task.task_id}

    gpu_loads_before = harness.calls["gpu_loads"]
    generated_tasks_before = sum(
        len(batch) for batch in harness.calls["inference_batches"]
    )
    summary = pipeline_module.run_pilot(**harness.arguments)

    assert summary["resumed_task_count"] == 1
    assert summary["newly_generated_task_count"] == 2
    assert summary["newly_generated_candidate_count"] == 2 * CANDIDATE_BUDGET
    assert harness.calls["gpu_loads"] == gpu_loads_before + 1
    assert (
        sum(len(batch) for batch in harness.calls["inference_batches"])
        == generated_tasks_before + 2
    )
    assert {path: path.read_bytes() for path in durable_identity} == durable_identity

    after = recover_paired_pilot_tasks(plan, raw_dir)
    assert after.rebuild_task_ids == set()
    assert after.durable_task_ids == {task.task_id for task in plan.tasks}
    partial_shard = next(
        shard
        for shard in after.raw_scan.valid_shards
        if shard.shard_index == partial_task.task_index
    )
    rebuilt_partial = load_raw_shard_candidates(
        partial_shard,
        expected_semantic_config_sha256=plan.semantic_config_sha256,
    )
    assert [item.candidate_index for item in rebuilt_partial] == [0, 1]
    assert not np.all(rebuilt_partial[0].future_xy_global == -123.0)

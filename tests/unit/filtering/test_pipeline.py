from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest

import skilldrive.filtering.pipeline as pipeline
from skilldrive.filtering.context import (
    BoundRawCandidate,
    CandidateEvaluationContext,
    bind_raw_candidates,
)
from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.filtering.prepared_map import (
    PreparedMapVerificationSession,
    prepare_map_geometry,
)
from skilldrive.filtering.pipeline import CandidateFilterInput
from skilldrive.filtering.risk import RiskEvaluation, RiskStatus
from skilldrive.generation.config import (
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import GeneratedCandidate, GeneratedOverlay
from skilldrive.generation.planning import (
    build_generation_task,
    semantic_generation_config_sha256,
)
from skilldrive.generation.storage import load_raw_shard_candidates, write_raw_shard
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.seeds.records import SeedRecord
from skilldrive.skills.detection import load_detection_config
from skilldrive.skills.loader import load_skill


def _walk_compact_values(root):
    stack = [root]
    seen: set[int] = set()
    while stack:
        value = stack.pop()
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        yield value
        if value is None or isinstance(
            value,
            (str, bytes, bool, int, float, np.ndarray),
        ):
            continue
        if is_dataclass(value):
            stack.extend(getattr(value, field.name) for field in fields(value))
        elif isinstance(value, Mapping):
            stack.extend(value.keys())
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)


def _record() -> SeedRecord:
    return SeedRecord(
        scenario_id="scene",
        skill_id="slow_lead_blockage",
        initiator_track_id="leader",
        responder_track_id="follower",
        role_track_ids={"slow_leader": "leader", "follower": "follower"},
        trigger_score=0.5,
        seed_risk_metric="minimum_longitudinal_gap",
        seed_risk_value=20.0,
        target_risk_definition={
            "metric": "minimum_longitudinal_gap",
            "direction": "lower_is_riskier",
            "source": "semantic",
            "target_range": [3.0, 15.0],
        },
        source_path="train/scene/scenario_scene.parquet",
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"leader_speed_scale": 0.5},
    )


def _source() -> Scenario:
    time = np.arange(110, dtype=np.float64) * 0.1
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    agents = []
    for track_id, x, speed in (("leader", 20.0, 1.0), ("follower", 0.0, 2.0)):
        positions = np.column_stack((x + speed * time, np.zeros(110)))
        agents.append(
            AgentTrack(
                track_id=track_id,
                object_type="vehicle",
                positions=positions,
                velocities=np.tile([speed, 0.0], (110, 1)),
                headings=np.zeros(110),
                observed_mask=observed.copy(),
                is_focal=track_id == "leader",
            )
        )
    return Scenario(
        scenario_id="scene",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="leader",
        agents=agents,
        map_polylines=[],
    )


def _inputs(tmp_path, offsets: tuple[float, ...]) -> tuple[CandidateFilterInput, ...]:
    config = load_counterfactual_config()
    record = _record()
    task = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=len(offsets),
    )
    source = _source()
    base_future = source.agents[0].positions[50:].copy()
    metadata = {
        "condition_skill_id": task.condition_skill_id,
        "primary_generated_role": "slow_leader",
        "requested_parameters": record.sampled_parameters,
        "detection_mode": "observed_trigger",
    }
    candidates = []
    for index, offset in enumerate(offsets):
        future = base_future.copy()
        future[:, 1] += offset
        candidates.append(
            GeneratedCandidate(
                task_id=task.task_id,
                candidate_index=index,
                latent_seed=100 + index,
                scenario_id=task.scenario_id,
                skill_id=task.skill_id,
                proposal_mode=task.proposal_mode,
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=semantic_generation_config_sha256(config),
                overlay=GeneratedOverlay(
                    target_track_id=task.target_track_id,
                    future_xy_global=future,
                ),
                metadata=metadata,
            )
        )
    commit = write_raw_shard(
        tmp_path / "raw",
        0,
        candidates,
        semantic_config_sha256=task.semantic_config_sha256,
        execution_config_sha256="e" * 64,
    )
    raw = load_raw_shard_candidates(commit)
    bound = bind_raw_candidates(raw, [task], [record])
    skill = load_skill("configs/skills/slow_lead_blockage.yaml")
    return tuple(
        CandidateFilterInput(
            bound=item,
            skill=skill,
            source_scenario=source,
            primary_generated_role="slow_leader",
        )
        for item in bound
    )


def _pass(stage: FilterStage, metrics=None):
    def check(*args, **kwargs):
        return FilterCheck(stage=stage, metrics={} if metrics is None else metrics)

    return check


def _computed_risk(candidate: CandidateFilterInput) -> RiskEvaluation:
    return RiskEvaluation(
        skill_id=candidate.skill.skill_id,
        metric=candidate.skill.risk_definition["metric"],
        unit="m",
        formula_version="test.v1",
        status=RiskStatus.COMPUTED,
        value=9.0,
        role_track_ids=candidate.bound.seed_record.role_track_ids,
    )


def _patch_quality_gates(monkeypatch, candidate: CandidateFilterInput) -> None:
    monkeypatch.setattr(
        pipeline,
        "check_history_and_coordinates",
        _pass(FilterStage.HISTORY_INVARIANTS),
    )
    monkeypatch.setattr(pipeline, "check_kinematics", _pass(FilterStage.KINEMATICS))
    monkeypatch.setattr(pipeline, "check_map_compliance", _pass(FilterStage.MAP))
    monkeypatch.setattr(
        pipeline,
        "check_proxy_collisions",
        _pass(FilterStage.COLLISION),
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_risk_context",
        lambda **kwargs: kwargs["generated_scenario"],
    )
    monkeypatch.setattr(
        pipeline,
        "evaluate_skill_risk",
        lambda *args, **kwargs: _computed_risk(candidate),
    )
    monkeypatch.setattr(
        pipeline,
        "_skill_trigger_check",
        _pass(FilterStage.SKILL_TRIGGER),
    )
    monkeypatch.setattr(
        pipeline,
        "check_parameter_realization",
        _pass(
            FilterStage.PARAMETER_REALIZATION,
            {
                "parameters": {
                    "leader_speed_scale": {
                        "status": "computed",
                        "realized": 0.52,
                        "absolute_tolerance": 0.15,
                    }
                }
            },
        ),
    )


def test_validate_candidate_passes_explicit_prepared_map_to_map_gate(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (1.0,))[0]
    prepared = prepare_map_geometry(candidate.source_scenario)
    session = PreparedMapVerificationSession(candidate.source_scenario, prepared)
    candidate = replace(
        candidate,
        prepared_map=prepared,
        map_verification_session=session,
    )
    _patch_quality_gates(monkeypatch, candidate)
    captured = []

    def map_check(*args, **kwargs):
        captured.append(
            (kwargs.get("prepared_map"), kwargs.get("verification_session"))
        )
        return FilterCheck(stage=FilterStage.MAP)

    monkeypatch.setattr(pipeline, "check_map_compliance", map_check)
    result = pipeline.validate_candidate(
        candidate,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
    )

    assert result.quality_passed
    assert captured == [(prepared, session)]
    session.finalize()


def test_verification_session_preserves_complete_filter_decision(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (0.0,))[0]
    source = candidate.source_scenario
    source.map_polylines.extend(
        (
            MapPolyline(
                polyline_id="area",
                polyline_type="drivable_area",
                points=np.array(
                    [
                        [-10.0, -5.0],
                        [100.0, -5.0],
                        [100.0, 5.0],
                        [-10.0, 5.0],
                        [-10.0, -5.0],
                    ]
                ),
            ),
            MapPolyline(
                polyline_id="lane",
                polyline_type="lane_centerline",
                points=np.array([[-10.0, 0.0], [100.0, 0.0]]),
                direction="vehicle",
                lane_id="lane",
            ),
        )
    )
    prepared = prepare_map_geometry(source)
    session = PreparedMapVerificationSession(source, prepared)
    scalar = replace(candidate, prepared_map=prepared)
    optimized = replace(
        candidate,
        prepared_map=prepared,
        map_verification_session=session,
    )
    real_map_check = pipeline.check_map_compliance
    _patch_quality_gates(monkeypatch, candidate)
    monkeypatch.setattr(pipeline, "check_map_compliance", real_map_check)
    filter_config = load_filter_config()
    detection_config = load_detection_config("configs/seed_detection.yaml")

    scalar_result = pipeline.validate_candidate(
        scalar,
        filter_config=filter_config,
        detection_config=detection_config,
    )
    optimized_result = pipeline.validate_candidate(
        optimized,
        filter_config=filter_config,
        detection_config=detection_config,
    )
    session.finalize()
    diversity = FilterCheck(stage=FilterStage.DIVERSITY)
    scalar_decision = scalar_result.to_filter_decision(
        filter_semantic_sha256="f" * 64,
        diversity_check=diversity,
    )
    optimized_decision = optimized_result.to_filter_decision(
        filter_semantic_sha256="f" * 64,
        diversity_check=diversity,
    )

    assert scalar_result.quality_passed
    assert optimized_result.quality_passed
    assert scalar_decision == optimized_decision


def test_individual_batch_chunks_pre_map_survivors_and_preserves_elapsed_total(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = list(_inputs(tmp_path, tuple(float(index) for index in range(20))))
    for index in range(0, len(candidates), 2):
        candidate = candidates[index]
        invalid_raw = replace(
            candidate.bound.raw,
            future_xy_global=candidate.bound.raw.future_xy_global.astype(np.float64),
        )
        candidates[index] = replace(
            candidate,
            bound=BoundRawCandidate(
                raw=invalid_raw,
                task=candidate.bound.task,
                seed_record=candidate.bound.seed_record,
            ),
        )
    source = candidates[0].source_scenario
    prepared = prepare_map_geometry(source)
    session = PreparedMapVerificationSession(source, prepared)
    candidates = [
        replace(
            candidate,
            prepared_map=prepared,
            map_verification_session=session,
        )
        for candidate in candidates
    ]
    _patch_quality_gates(monkeypatch, candidates[0])
    map_batch_sizes = []

    def batch_map(scenarios, target_ids, skill_ids, policy, **kwargs):
        assert len(scenarios) == len(target_ids) == len(skill_ids)
        map_batch_sizes.append(len(scenarios))
        return tuple(FilterCheck(stage=FilterStage.MAP) for _ in scenarios)

    monkeypatch.setattr(pipeline, "check_map_compliance_batch", batch_map)
    tick = 0

    def perf_counter():
        nonlocal tick
        tick += 1
        return float(tick)

    monkeypatch.setattr(pipeline.time, "perf_counter", perf_counter)
    results = pipeline.validate_candidate_individual_batch(
        candidates,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
        map_batch_size=8,
    )
    session.finalize()

    assert map_batch_sizes == [8, 2]
    assert [item.candidate.bound.raw.candidate_id for item in results] == [
        item.bound.raw.candidate_id for item in candidates
    ]
    assert all(
        result.first_failed.check.stage is FilterStage.SCHEMA_FINITE
        for result in results[::2]
    )
    assert all(result.quality_passed for result in results[1::2])
    assert math.isclose(
        sum(
            check.elapsed_seconds
            for result in results
            for check in result.checks
            if check.check.stage is FilterStage.MAP
        ),
        2.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("map_batch_size", (8, 16, 32))
def test_individual_batch_randomized_decisions_match_scalar(
    tmp_path,
    monkeypatch,
    map_batch_size,
) -> None:
    rng = np.random.default_rng(2026)
    candidates = _inputs(tmp_path, tuple(rng.uniform(-7.0, 7.0, size=35)))
    source = candidates[0].source_scenario
    source.map_polylines.extend(
        (
            MapPolyline(
                polyline_id="area",
                polyline_type="drivable_area",
                points=np.array(
                    [
                        [-10.0, -5.0],
                        [100.0, -5.0],
                        [100.0, 5.0],
                        [-10.0, 5.0],
                        [-10.0, -5.0],
                    ]
                ),
            ),
            MapPolyline(
                polyline_id="lane",
                polyline_type="lane_centerline",
                points=np.array([[-10.0, 0.0], [100.0, 0.0]]),
                direction="vehicle",
                lane_id="lane",
            ),
        )
    )
    prepared = prepare_map_geometry(source)
    scalar_session = PreparedMapVerificationSession(source, prepared)
    batch_session = PreparedMapVerificationSession(source, prepared)
    scalar_inputs = tuple(
        replace(
            candidate,
            prepared_map=prepared,
            map_verification_session=scalar_session,
        )
        for candidate in candidates
    )
    batch_inputs = tuple(
        replace(
            candidate,
            prepared_map=prepared,
            map_verification_session=batch_session,
        )
        for candidate in candidates
    )
    real_map_check = pipeline.check_map_compliance
    _patch_quality_gates(monkeypatch, candidates[0])
    monkeypatch.setattr(pipeline, "check_map_compliance", real_map_check)
    filter_config = load_filter_config()
    detection_config = load_detection_config("configs/seed_detection.yaml")

    scalar_results = tuple(
        pipeline.validate_candidate(
            candidate,
            filter_config=filter_config,
            detection_config=detection_config,
        )
        for candidate in scalar_inputs
    )
    scalar_session.finalize()
    batch_results = pipeline.validate_candidate_individual_batch(
        batch_inputs,
        filter_config=filter_config,
        detection_config=detection_config,
        map_batch_size=map_batch_size,
    )
    batch_session.finalize()
    scalar_batch = pipeline.finalize_candidate_validations(
        tuple(result.compact(cohort="formal") for result in scalar_results),
        filter_config=filter_config,
        filter_semantic_sha256="f" * 64,
    )
    optimized_batch = pipeline.finalize_candidate_validations(
        tuple(result.compact(cohort="formal") for result in batch_results),
        filter_config=filter_config,
        filter_semantic_sha256="f" * 64,
    )

    assert scalar_batch.decisions == optimized_batch.decisions
    assert scalar_batch.stage_execution_counts == optimized_batch.stage_execution_counts
    assert [item.identity.candidate_id for item in optimized_batch.validations] == [
        item.bound.raw.candidate_id for item in candidates
    ]


def test_schema_rejection_happens_before_overlay_and_has_stable_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (0.0,))[0]
    invalid_raw = replace(
        candidate.bound.raw,
        future_xy_global=candidate.bound.raw.future_xy_global.astype(np.float64),
    )
    invalid = replace(
        candidate,
        bound=BoundRawCandidate(
            raw=invalid_raw,
            task=candidate.bound.task,
            seed_record=candidate.bound.seed_record,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_candidate_evaluation_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("overlay must not be built after schema rejection")
        ),
    )

    validation = pipeline.validate_candidate(
        invalid,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
    )
    compact = validation.compact(cohort="formal")
    decision = validation.to_filter_decision(
        filter_semantic_sha256="f" * 64,
        diversity_check=None,
    )

    assert compact.diversity_candidate is None
    assert not compact.quality_passed
    assert decision.rejection_reasons == (
        FilterRejection.INVALID_FUTURE_DTYPE.value,
    )
    assert decision.metrics["evaluated_stages"] == ["schema_finite"]
    assert decision.metrics["skipped_stages"] == [
        stage.value for stage in pipeline._STAGE_ORDER[1:]
    ]
    assert "elapsed_seconds" not in repr(decision.metrics)


def test_batch_runs_diversity_only_for_individual_quality_survivors(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _inputs(tmp_path, (0.0, 3.0))
    _patch_quality_gates(monkeypatch, candidates[0])

    def map_check(scenario, *args, **kwargs):
        target = next(agent for agent in scenario.agents if agent.track_id == "leader")
        rejected = float(np.mean(target.positions[50:, 1])) > 1.0
        return FilterCheck(
            stage=FilterStage.MAP,
            rejection_reasons=(
                (FilterRejection.OUTSIDE_DRIVABLE_AREA,) if rejected else ()
            ),
        )

    monkeypatch.setattr(pipeline, "check_map_compliance", map_check)
    original_diversity = pipeline.apply_diversity_filter
    seen: list[str] = []

    def diversity(items, policy):
        seen.extend(item.candidate_id for item in items)
        return original_diversity(items, policy)

    monkeypatch.setattr(pipeline, "apply_diversity_filter", diversity)
    batch = pipeline.validate_candidates(
        candidates,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
        filter_semantic_sha256="f" * 64,
    )

    assert seen == [candidates[0].bound.raw.candidate_id]
    assert [decision.accepted for decision in batch.decisions] == [True, False]
    assert batch.decisions[1].metrics["first_failed_stage"] == "map"
    assert batch.stage_execution_counts["diversity"] == 1
    assert batch.decisions[0].metrics["evaluated_stages"] == [
        stage.value for stage in pipeline._STAGE_ORDER
    ]


def test_risk_context_is_prepared_after_history_map_and_collision(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (0.0,))[0]
    events: list[str] = []

    def passed(stage: FilterStage, event: str):
        def check(*args, **kwargs):
            events.append(event)
            return FilterCheck(stage=stage)

        return check

    monkeypatch.setattr(
        pipeline,
        "check_history_and_coordinates",
        passed(FilterStage.HISTORY_INVARIANTS, "history"),
    )
    monkeypatch.setattr(
        pipeline,
        "check_kinematics",
        passed(FilterStage.KINEMATICS, "kinematics"),
    )
    monkeypatch.setattr(
        pipeline,
        "check_map_compliance",
        passed(FilterStage.MAP, "map"),
    )
    monkeypatch.setattr(
        pipeline,
        "check_proxy_collisions",
        passed(FilterStage.COLLISION, "collision"),
    )

    def prepare(**kwargs):
        events.append("risk_context")
        return kwargs["generated_scenario"]

    def risk(*args, **kwargs):
        events.append("risk")
        return _computed_risk(candidate)

    monkeypatch.setattr(pipeline, "prepare_risk_context", prepare)
    monkeypatch.setattr(pipeline, "evaluate_skill_risk", risk)
    monkeypatch.setattr(
        pipeline,
        "_skill_trigger_check",
        passed(FilterStage.SKILL_TRIGGER, "skill"),
    )
    monkeypatch.setattr(
        pipeline,
        "check_parameter_realization",
        passed(FilterStage.PARAMETER_REALIZATION, "parameter"),
    )

    result = pipeline.validate_candidate(
        candidate,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
    )

    assert result.quality_passed
    assert events == [
        "history",
        "kinematics",
        "map",
        "collision",
        "risk_context",
        "risk",
        "skill",
        "parameter",
    ]


def test_compact_result_drops_scenario_context_and_retains_audit_data(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (0.0,))[0]
    _patch_quality_gates(monkeypatch, candidate)

    validation = pipeline.validate_candidate(
        candidate,
        filter_config=load_filter_config(),
        detection_config=load_detection_config("configs/seed_detection.yaml"),
    )
    compact = validation.compact(cohort="formal")

    forbidden = (
        Scenario,
        CandidateEvaluationContext,
        CandidateFilterInput,
        BoundRawCandidate,
    )
    assert not any(
        isinstance(value, forbidden) for value in _walk_compact_values(compact)
    )
    assert compact.identity.candidate_id == candidate.bound.raw.candidate_id
    assert compact.quality_passed
    assert compact.diversity_candidate is not None
    assert tuple(compact.stage_elapsed_seconds) == tuple(
        stage.value for stage in pipeline._INDIVIDUAL_STAGE_ORDER
    )
    assert compact.runtime_seconds == pytest.approx(
        sum(compact.stage_elapsed_seconds.values())
    )
    parameter_metrics = compact.checks[-1].check.metrics["parameters"]
    with pytest.raises(TypeError):
        parameter_metrics["new_parameter"] = {}


def test_finalize_runs_global_diversity_once_per_isolated_cohort(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _inputs(tmp_path, (0.0, 0.0))
    _patch_quality_gates(monkeypatch, candidates[0])
    filter_config = load_filter_config()
    detection_config = load_detection_config("configs/seed_detection.yaml")
    cohorts = ("formal", "learned_none_control")
    compact = tuple(
        pipeline.validate_candidate(
            candidate,
            filter_config=filter_config,
            detection_config=detection_config,
        ).compact(cohort=cohort)
        for candidate, cohort in zip(candidates, cohorts, strict=True)
    )
    calls: list[tuple[str, ...]] = []
    original_diversity = pipeline.apply_diversity_filter

    def diversity(items, policy):
        calls.append(tuple(item.candidate_id for item in items))
        return original_diversity(items, policy)

    monkeypatch.setattr(pipeline, "apply_diversity_filter", diversity)
    batch = pipeline.finalize_candidate_validations(
        compact,
        filter_config=filter_config,
        filter_semantic_sha256="f" * 64,
    )

    assert calls == [
        (compact[0].identity.candidate_id,),
        (compact[1].identity.candidate_id,),
    ]
    assert all(decision.accepted for decision in batch.decisions)
    assert [decision.metrics["diversity_cohort"] for decision in batch.decisions] == [
        "formal",
        "learned_none_control",
    ]
    assert {decision.candidate_id for decision in batch.decisions} == {
        item.identity.candidate_id for item in compact
    }


def test_finalize_decisions_are_deterministic_across_input_order_and_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    candidates = _inputs(tmp_path, (0.0, 0.0))
    _patch_quality_gates(monkeypatch, candidates[0])
    filter_config = load_filter_config()
    detection_config = load_detection_config("configs/seed_detection.yaml")
    compact = tuple(
        pipeline.validate_candidate(
            candidate,
            filter_config=filter_config,
            detection_config=detection_config,
        ).compact(cohort="formal")
        for candidate in candidates
    )
    slower_reversed = tuple(
        replace(
            item,
            checks=tuple(
                replace(check, elapsed_seconds=check.elapsed_seconds + 10.0)
                for check in item.checks
            ),
        )
        for item in reversed(compact)
    )

    first = pipeline.finalize_candidate_validations(
        compact,
        filter_config=filter_config,
        filter_semantic_sha256="f" * 64,
    )
    second = pipeline.finalize_candidate_validations(
        slower_reversed,
        filter_config=filter_config,
        filter_semantic_sha256="f" * 64,
    )
    first_by_id = {item.candidate_id: item for item in first.decisions}
    second_by_id = {item.candidate_id: item for item in second.decisions}

    assert first_by_id == second_by_id
    assert first.stage_elapsed_seconds != second.stage_elapsed_seconds
    assert sum(item.accepted for item in first.decisions) == 1


def test_finalize_rejects_duplicate_ids_and_invalid_diversity_coverage(
    tmp_path,
    monkeypatch,
) -> None:
    candidate = _inputs(tmp_path, (0.0,))[0]
    _patch_quality_gates(monkeypatch, candidate)
    filter_config = load_filter_config()
    compact = pipeline.validate_candidate(
        candidate,
        filter_config=filter_config,
        detection_config=load_detection_config("configs/seed_detection.yaml"),
    ).compact(cohort="formal")

    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        pipeline.finalize_candidate_validations(
            (compact, compact),
            filter_config=filter_config,
            filter_semantic_sha256="f" * 64,
        )

    with pytest.raises(ValueError, match="must cover every individual stage"):
        replace(
            compact,
            checks=compact.checks[:-1],
            quality_score=None,
            diversity_candidate=None,
        )

    monkeypatch.setattr(pipeline, "apply_diversity_filter", lambda items, policy: {})
    with pytest.raises(ValueError, match="do not exactly cover cohort"):
        pipeline.finalize_candidate_validations(
            (compact,),
            filter_config=filter_config,
            filter_semantic_sha256="f" * 64,
        )

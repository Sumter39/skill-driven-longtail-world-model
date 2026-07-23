from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import skilldrive.performance.parallel_filter as parallel_filter
from skilldrive.filtering.contracts import FilterCheck, FilterStage
from skilldrive.filtering.diversity import DiversityCandidate
from skilldrive.filtering.pipeline import (
    CandidateFilterIdentity,
    CandidateFilterInput,
    CompactCandidateValidationResult,
    TimedFilterCheck,
    validate_candidate,
)
from skilldrive.generation.config import (
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.generation.contracts import (
    GeneratedCandidate,
    GeneratedOverlay,
    GenerationTask,
)
from skilldrive.generation.planning import build_generation_task
from skilldrive.generation.storage import write_raw_shard
from skilldrive.performance.parallel_filter import (
    CompactValidationWire,
    ParallelFilterWorkerError,
    run_parallel_filter_workload,
)
from skilldrive.performance.workload import generation_task_to_row
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario
from skilldrive.seeds.records import SeedRecord, write_seed_records
from skilldrive.skills.detection import load_detection_config


_INDIVIDUAL_STAGES = tuple(
    stage for stage in FilterStage if stage is not FilterStage.DIVERSITY
)


def _load_json_scenario(path: str | Path) -> Scenario:
    return Scenario.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _failing_scenario_loader(path: str | Path) -> Scenario:
    if "scenario-1" in Path(path).as_posix():
        raise RuntimeError("synthetic worker failure")
    return _load_json_scenario(path)


@dataclass(frozen=True)
class _SyntheticValidation:
    candidate: CandidateFilterInput

    def compact(self, *, cohort: str) -> CompactCandidateValidationResult:
        prepared = self.candidate.prepared_map
        if prepared is None or prepared.scenario_id != self.candidate.source_scenario.scenario_id:
            raise AssertionError("worker did not reuse the scenario PreparedMap")
        session = self.candidate.map_verification_session
        if session is None or session.prepared_map is not prepared:
            raise AssertionError("worker did not reuse one map verification session")
        raw = self.candidate.bound.raw
        task = self.candidate.bound.task
        score = float(raw.candidate_index)
        checks = tuple(
            TimedFilterCheck(
                FilterCheck(stage=stage, metrics={"synthetic_stage": stage.value}),
                elapsed_seconds=(index + 1) / 10_000.0,
            )
            for index, stage in enumerate(_INDIVIDUAL_STAGES)
        )
        diversity = DiversityCandidate(
            candidate_id=raw.candidate_id,
            scenario_id=raw.scenario_id,
            skill_id=raw.skill_id,
            future_xy_local=raw.future_xy_global.astype(np.float64),
            target_risk_value=5.0,
            quality_score=score,
            realized_parameter_bins=(("speed", 1),),
        )
        return CompactCandidateValidationResult(
            identity=CandidateFilterIdentity(
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
            checks=checks,
            quality_score=score,
            diversity_candidate=diversity,
        )


def _synthetic_validator(candidate: CandidateFilterInput, **kwargs) -> _SyntheticValidation:
    return _SyntheticValidation(candidate)


def _map_mutating_validator(
    candidate: CandidateFilterInput,
    **kwargs,
) -> _SyntheticValidation:
    candidate.source_scenario.map_polylines[0].points[0, 0] += 1.0
    return _SyntheticValidation(candidate)


def _source(scenario_id: str, offset: float) -> Scenario:
    time = np.arange(110, dtype=np.float64) * 0.1
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    agents = []
    for track_id, start, speed in (
        ("leader", offset + 20.0, 1.0),
        ("follower", offset, 2.0),
    ):
        positions = np.column_stack((start + speed * time, np.zeros(110)))
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
        scenario_id=scenario_id,
        city_name="synthetic",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="leader",
        agents=agents,
        map_polylines=[
            MapPolyline(
                polyline_id=f"{scenario_id}-area",
                polyline_type="drivable_area",
                points=np.array(
                    [
                        [offset - 10.0, -5.0],
                        [offset + 100.0, -5.0],
                        [offset + 100.0, 5.0],
                        [offset - 10.0, 5.0],
                        [offset - 10.0, -5.0],
                    ]
                ),
            ),
            MapPolyline(
                polyline_id=f"{scenario_id}-lane",
                polyline_type="lane_centerline",
                points=np.array([[offset - 10.0, 0.0], [offset + 100.0, 0.0]]),
                direction="vehicle",
                lane_id=f"{scenario_id}-lane",
            ),
        ],
    )


def _record(scenario_id: str, source_path: str) -> SeedRecord:
    return SeedRecord(
        scenario_id=scenario_id,
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
        source_path=source_path,
        evidence={"detection_mode": "observed_trigger"},
        sampled_parameters={"leader_speed_scale": 0.5},
    )


def _control_task(conditioned: GenerationTask, *, task_index: int) -> GenerationTask:
    return GenerationTask.create(
        task_index=task_index,
        seed_record_id=conditioned.seed_record_id,
        scenario_id=conditioned.scenario_id,
        skill_id=conditioned.skill_id,
        target_track_id=conditioned.target_track_id,
        proposal_mode=conditioned.proposal_mode,
        condition_skill_id="<none>",
        candidate_budget=conditioned.candidate_budget,
        checkpoint_sha256=conditioned.checkpoint_sha256,
        semantic_config_sha256=conditioned.semantic_config_sha256,
    )


def _build_workload(tmp_path: Path, *, scenario_count: int = 4):
    base = load_counterfactual_config()
    inputs = replace(
        base.inputs,
        data_root=Path("data"),
        seed_manifest=Path("seeds.csv"),
    )
    generation = replace(
        base,
        formal_catalog=Path("configs/skills/catalog.yaml"),
        inputs=inputs,
    )
    skill_root = tmp_path / "configs" / "skills"
    skill_root.mkdir(parents=True)
    shutil.copy2(
        Path("configs/skills/slow_lead_blockage.yaml"),
        skill_root / "slow_lead_blockage.yaml",
    )

    records = []
    rows = []
    for scenario_index in range(scenario_count):
        scenario_id = f"scenario-{scenario_index}"
        source_path = f"train/{scenario_id}/scenario.json"
        record = _record(scenario_id, source_path)
        records.append(record)
        source = _source(scenario_id, offset=float(scenario_index * 20))
        source_file = tmp_path / "data" / source_path
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            json.dumps(source.to_dict(), sort_keys=True),
            encoding="utf-8",
        )

        conditioned = build_generation_task(
            task_index=scenario_index * 2,
            record=record,
            config=generation,
            candidate_budget=1,
        )
        tasks = (
            conditioned,
            _control_task(conditioned, task_index=scenario_index * 2 + 1),
        )
        future = source.agents[0].positions[50:].copy()
        for task in tasks:
            candidate = GeneratedCandidate(
                task_id=task.task_id,
                candidate_index=0,
                latent_seed=123,
                scenario_id=task.scenario_id,
                skill_id=task.skill_id,
                proposal_mode=task.proposal_mode,
                checkpoint_sha256=task.checkpoint_sha256,
                semantic_config_sha256=task.semantic_config_sha256,
                overlay=GeneratedOverlay(
                    target_track_id=task.target_track_id,
                    future_xy_global=future,
                ),
                metadata={
                    "condition_skill_id": task.condition_skill_id,
                    "primary_generated_role": "slow_leader",
                    "requested_parameters": record.sampled_parameters,
                    "detection_mode": "observed_trigger",
                },
            )
            commit = write_raw_shard(
                tmp_path / "raw" / f"task-{task.task_index}",
                task.task_index,
                [candidate],
                semantic_config_sha256=task.semantic_config_sha256,
                execution_config_sha256="e" * 64,
            )
            rows.append(
                {
                    "task": generation_task_to_row(task),
                    "source_path": source_path,
                    "raw_commit": commit.commit_path.relative_to(tmp_path).as_posix(),
                }
            )
    write_seed_records(tmp_path / "seeds.csv", records)
    workload = {
        "counts": {
            "tasks": len(rows),
            "candidates": len(rows),
            "scenarios": scenario_count,
        },
        "filter_semantic_sha256": "f" * 64,
        "tasks": rows,
    }
    return (
        workload,
        generation,
        load_filter_config(),
        load_detection_config("configs/seed_detection.yaml"),
    )


def _decision_rows(result) -> list[tuple]:
    return [
        (
            decision.candidate_id,
            decision.filter_evaluation_id,
            decision.accepted,
            decision.rejection_reasons,
            dict(decision.metrics),
        )
        for decision in sorted(
            result.batch.decisions,
            key=lambda item: item.candidate_id,
        )
    ]


def test_wire_dto_is_pickle_safe_and_restores_compact_result() -> None:
    identity = CandidateFilterIdentity(
        candidate_id="a" * 64,
        task_id="b" * 64,
        candidate_index=0,
        latent_seed=1,
        scenario_id="scenario",
        skill_id="skill",
        target_track_id="track",
        seed_record_id="c" * 64,
        proposal_mode="learned_conditioned_prior",
        checkpoint_sha256="d" * 64,
        semantic_config_sha256="e" * 64,
    )
    checks = tuple(
        TimedFilterCheck(
            FilterCheck(
                stage=stage,
                metrics={"nested": MappingProxyType({"values": (stage.value, 1)})},
            ),
            elapsed_seconds=0.01,
        )
        for stage in _INDIVIDUAL_STAGES
    )
    diversity = DiversityCandidate(
        candidate_id=identity.candidate_id,
        scenario_id=identity.scenario_id,
        skill_id=identity.skill_id,
        future_xy_local=np.zeros((60, 2)),
        target_risk_value=1.0,
        quality_score=0.0,
        realized_parameter_bins=(),
    )
    compact = CompactCandidateValidationResult(
        identity=identity,
        cohort="formal",
        checks=checks,
        quality_score=0.0,
        diversity_candidate=diversity,
    )
    wire = parallel_filter._to_wire(compact, 3)

    restored_wire = pickle.loads(pickle.dumps(wire))
    assert isinstance(restored_wire, CompactValidationWire)
    restored = parallel_filter._from_wire(restored_wire)

    assert restored.identity == identity
    assert restored.quality_passed
    assert restored.cohort == "formal"
    assert restored.diversity_candidate is not None
    np.testing.assert_array_equal(
        restored.diversity_candidate.future_xy_local,
        np.zeros((60, 2)),
    )


def test_one_and_two_workers_match_with_shuffled_input_and_isolated_cohorts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, generation, filter_config, detection_config = _build_workload(
        tmp_path,
        scenario_count=2,
    )
    finalize_calls = 0
    real_finalize = parallel_filter.finalize_candidate_validations

    def tracking_finalize(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(
        parallel_filter,
        "finalize_candidate_validations",
        tracking_finalize,
    )
    serial = run_parallel_filter_workload(
        workload,
        repository_root=tmp_path,
        generation_config=generation,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=1,
        scenario_loader=_load_json_scenario,
        candidate_validator=_synthetic_validator,
    )
    shuffled = dict(workload)
    shuffled["tasks"] = list(reversed(workload["tasks"]))
    parallel = run_parallel_filter_workload(
        shuffled,
        repository_root=tmp_path,
        generation_config=generation,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=2,
        scenario_loader=_load_json_scenario,
        candidate_validator=_synthetic_validator,
    )
    assert finalize_calls == 2
    assert serial.decision_sha256 == parallel.decision_sha256
    assert _decision_rows(serial) == _decision_rows(parallel)
    assert serial.validation_order == parallel.validation_order
    assert serial.stage_execution_counts == parallel.stage_execution_counts
    assert serial.stage_rejection_counts == parallel.stage_rejection_counts
    assert serial.scenario_load_count == parallel.scenario_load_count == 2
    assert serial.prepared_map_count == parallel.prepared_map_count == 2
    assert serial.map_batch_size == parallel.map_batch_size == 16
    for result in (serial, parallel):
        assert result.timings["map_integrity_finalize_seconds"] >= 0.0
        assert result.timings["map_subsystem_seconds"] == pytest.approx(
            result.timings["prepared_map_seconds"]
            + result.batch.stage_elapsed_seconds[FilterStage.MAP.value]
            + result.timings["map_integrity_finalize_seconds"]
        )
    assert serial.effective_worker_count == 1
    assert parallel.requested_worker_count == 2
    assert 1 <= parallel.effective_worker_count <= 2
    assert len(parallel.worker_pids) == parallel.effective_worker_count
    assert all(decision.accepted for decision in parallel.batch.decisions)
    cohorts = {
        decision.metrics["task_id"]: decision.metrics["diversity_cohort"]
        for decision in parallel.batch.decisions
    }
    for entry in workload["tasks"]:
        task = entry["task"]
        expected = (
            "learned_none_control"
            if task["condition_skill_id"] == "<none>"
            else "formal"
        )
        assert cohorts[task["task_id"]] == expected


def test_worker_failure_never_finalizes_or_writes_a_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, generation, filter_config, detection_config = _build_workload(
        tmp_path,
        scenario_count=2,
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    finalize_calls = 0

    def forbidden_finalize(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        raise AssertionError("global finalize must not run after a worker failure")

    monkeypatch.setattr(
        parallel_filter,
        "finalize_candidate_validations",
        forbidden_finalize,
    )
    with pytest.raises(ParallelFilterWorkerError, match="scenario-1"):
        run_parallel_filter_workload(
            workload,
            repository_root=tmp_path,
            generation_config=generation,
            filter_config=filter_config,
            detection_config=detection_config,
            worker_count=2,
            scenario_loader=_failing_scenario_loader,
            candidate_validator=_synthetic_validator,
        )

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file()}
    assert finalize_calls == 0
    assert after == before
    assert not list(tmp_path.rglob("*filter*.commit.json"))


def test_default_batch_worker_matches_scalar_filter_reference(tmp_path: Path) -> None:
    workload, generation, filter_config, detection_config = _build_workload(
        tmp_path,
        scenario_count=1,
    )
    scalar = run_parallel_filter_workload(
        workload,
        repository_root=tmp_path,
        generation_config=generation,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=1,
        scenario_loader=_load_json_scenario,
        candidate_validator=validate_candidate,
    )
    batched = run_parallel_filter_workload(
        workload,
        repository_root=tmp_path,
        generation_config=generation,
        filter_config=filter_config,
        detection_config=detection_config,
        worker_count=1,
        map_batch_size=8,
        scenario_loader=_load_json_scenario,
    )

    assert scalar.decision_sha256 == batched.decision_sha256
    assert _decision_rows(scalar) == _decision_rows(batched)
    assert scalar.stage_execution_counts == batched.stage_execution_counts
    assert scalar.stage_rejection_counts == batched.stage_rejection_counts
    assert batched.map_batch_size == 8


def test_parallel_filter_rejects_unsupported_map_batch_size(tmp_path: Path) -> None:
    workload, generation, filter_config, detection_config = _build_workload(
        tmp_path,
        scenario_count=1,
    )
    with pytest.raises(ValueError, match="map_batch_size"):
        run_parallel_filter_workload(
            workload,
            repository_root=tmp_path,
            generation_config=generation,
            filter_config=filter_config,
            detection_config=detection_config,
            worker_count=1,
            map_batch_size=12,
        )


def test_map_integrity_failure_never_reaches_global_finalize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, generation, filter_config, detection_config = _build_workload(
        tmp_path,
        scenario_count=1,
    )
    finalize_calls = 0

    def forbidden_finalize(*args, **kwargs):
        nonlocal finalize_calls
        finalize_calls += 1
        raise AssertionError("global finalize must not run after map mutation")

    monkeypatch.setattr(
        parallel_filter,
        "finalize_candidate_validations",
        forbidden_finalize,
    )
    with pytest.raises(ParallelFilterWorkerError, match="scenario-0"):
        run_parallel_filter_workload(
            workload,
            repository_root=tmp_path,
            generation_config=generation,
            filter_config=filter_config,
            detection_config=detection_config,
            worker_count=1,
            scenario_loader=_load_json_scenario,
            candidate_validator=_map_mutating_validator,
        )

    assert finalize_calls == 0
    assert not list(tmp_path.rglob("*filter*.commit.json"))

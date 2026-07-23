from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from skilldrive.filtering.context import (
    BoundRawCandidate,
    bind_raw_candidates,
    build_candidate_evaluation_context,
    validate_bound_candidate_contract,
)
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.contracts import GeneratedCandidate, GeneratedOverlay
from skilldrive.generation.planning import (
    build_generation_task,
    semantic_generation_config_sha256,
)
from skilldrive.generation.storage import load_raw_shard_candidates, write_raw_shard
from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds.records import SeedRecord
from skilldrive.skills.loader import load_skill


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


def test_raw_task_seed_and_source_bind_without_heuristic_matching(tmp_path) -> None:
    config = load_counterfactual_config()
    record = _record()
    task = build_generation_task(
        task_index=0,
        record=record,
        config=config,
        candidate_budget=1,
    )
    source = _source()
    future = source.agents[0].positions[50:].copy()
    candidate = GeneratedCandidate(
        task_id=task.task_id,
        candidate_index=0,
        latent_seed=7,
        scenario_id=task.scenario_id,
        skill_id=task.skill_id,
        proposal_mode=task.proposal_mode,
        checkpoint_sha256=task.checkpoint_sha256,
        semantic_config_sha256=semantic_generation_config_sha256(config),
        overlay=GeneratedOverlay(target_track_id=task.target_track_id, future_xy_global=future),
        metadata={
            "condition_skill_id": task.condition_skill_id,
            "primary_generated_role": "slow_leader",
            "requested_parameters": record.sampled_parameters,
            "detection_mode": "observed_trigger",
        },
    )
    commit = write_raw_shard(
        tmp_path / "raw",
        0,
        [candidate],
        semantic_config_sha256=task.semantic_config_sha256,
        execution_config_sha256="e" * 64,
    )
    raw = load_raw_shard_candidates(commit)
    bound = bind_raw_candidates(raw, [task], [record])
    context = build_candidate_evaluation_context(
        bound[0],
        skill=load_skill("configs/skills/slow_lead_blockage.yaml"),
        source_scenario=source,
    )

    assert context.task.seed_record_id == task.seed_record_id
    assert context.raw.latent_seed == 7
    assert context.generated_scenario is not source
    np.testing.assert_allclose(context.future_xy_local[:, 1], 0.0, atol=1e-12)

    validate_bound_candidate_contract(
        bound[0],
        primary_generated_role="slow_leader",
    )
    invalid = BoundRawCandidate(
        raw=replace(
            bound[0].raw,
            metadata={
                **dict(bound[0].raw.metadata),
                "condition_skill_id": "<none>",
            },
        ),
        task=bound[0].task,
        seed_record=bound[0].seed_record,
    )
    with pytest.raises(ValueError, match="condition_skill_id"):
        validate_bound_candidate_contract(
            invalid,
            primary_generated_role="slow_leader",
        )

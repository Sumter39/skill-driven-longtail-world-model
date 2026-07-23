from __future__ import annotations

from pathlib import Path

import pytest

from skilldrive.generation.capability import build_generation_capability_matrix
from skilldrive.generation.config import (
    ActiveCheckpointConfig,
    CounterfactualGenerationConfig,
    GenerationInputConfig,
    SamplingConfig,
    SkillGenerationConfig,
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
            path=Path("outputs/modeling/cvae_baseline/formal/best.pt"),
            sha256="a" * 64,
            run_manifest=Path("outputs/modeling/cvae_baseline/formal/run_manifest.json"),
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
            pilot_seed_records_per_skill=16,
            pilot_candidates_per_task=16,
            formal_candidates_per_task=16,
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
                joint_generation_limited=True,
            ),
        ),
    )


def _record(skill_id: str, scenario_id: str, *, include_role: bool = True) -> SeedRecord:
    roles = {"actor": f"{scenario_id}-a", "other": f"{scenario_id}-b"}
    if not include_role:
        roles = {"wrong": f"{scenario_id}-a", "other": f"{scenario_id}-b"}
    return SeedRecord(
        scenario_id=scenario_id,
        skill_id=skill_id,
        initiator_track_id=f"{scenario_id}-a",
        responder_track_id=f"{scenario_id}-b",
        role_track_ids=roles,
        trigger_score=0.5,
        seed_risk_metric="metric",
        seed_risk_value=1.0,
        target_risk_definition={
            "metric": "metric",
            "direction": "lower_is_riskier",
            "source": "semantic",
            "target_range": [0.0, 2.0],
        },
        source_path=f"train/{scenario_id}/scenario_{scenario_id}.parquet",
        evidence={
            "detection_mode": (
                "observed_trigger" if skill_id == "learned" else "compatible_seed"
            )
        },
        sampled_parameters={"gap": 1.0},
    )


def test_build_generation_capability_matrix_separates_direct_and_search() -> None:
    result = build_generation_capability_matrix(
        config=_config(),
        records=[_record("learned", "one"), _record("search", "two")],
        training_cache_manifest={
            "sample_spec_statistics": {
                "by_skill": {
                    "learned": {"retained": 3},
                    "search": {"retained": 0},
                }
            }
        },
        checkpoint_path="outputs/model/best.pt",
        checkpoint_sha256="a" * 64,
        schema_sha256="b" * 64,
    )

    entries = {entry["skill_id"]: entry for entry in result["skills"]}
    assert entries["learned"]["current_support_status"] == "direct"
    assert entries["learned"]["condition_skill_id"] == "learned"
    assert entries["search"]["current_support_status"] == "single_target_limited"
    assert entries["search"]["condition_skill_id"] == "<none>"
    assert entries["search"]["parameter_conditioning_supported"] is False


def test_build_generation_capability_matrix_rejects_missing_primary_role() -> None:
    with pytest.raises(ValueError, match="missing primary role"):
        build_generation_capability_matrix(
            config=_config(),
            records=[
                _record("learned", "one", include_role=False),
                _record("search", "two"),
            ],
            training_cache_manifest={
                "sample_spec_statistics": {
                    "by_skill": {
                        "learned": {"retained": 3},
                        "search": {"retained": 0},
                    }
                }
            },
            checkpoint_path="outputs/model/best.pt",
            checkpoint_sha256="a" * 64,
            schema_sha256="b" * 64,
        )

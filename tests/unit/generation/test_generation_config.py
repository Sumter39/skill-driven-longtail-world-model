from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from skilldrive.generation.config import (
    COLLISION_PROXY_ACTOR_TYPES,
    FILTER_STAGES,
    FINITE_VALUE_FIELDS,
    LEARNED_CONDITIONED_SKILLS,
    load_counterfactual_config,
    load_filter_config,
)
from skilldrive.skills.loader import load_skill


GENERATION_CONFIG_PATH = Path("configs/generation/counterfactual_v1.yaml")
FILTER_CONFIG_PATH = Path("configs/generation/filters_v1.yaml")


def _raw(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write(tmp_path: Path, name: str, value: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_loads_frozen_34_plus_5_generation_partition() -> None:
    config = load_counterfactual_config()
    by_skill = config.skills_by_id

    assert config.active_checkpoint.path.as_posix().endswith(
        "epoch-0045-step-00004095.pt"
    )
    assert len(config.active_checkpoint.sha256) == 64
    assert len(config.active_checkpoint.run_manifest_sha256) == 64
    assert len(config.active_checkpoint.schema_sha256) == 64
    assert config.active_checkpoint.run_manifest_stage == "repair-formal"
    assert config.active_checkpoint.repair_contract == "cvae_generation_repair_v1"
    assert config.active_checkpoint.promotion_recommendation is not None
    assert len(config.active_checkpoint.promotion_recommendation_sha256 or "") == 64
    assert config.inputs.seed_manifest.as_posix() == "manifests/seeds/formal_candidates.csv"
    assert len(config.inputs.seed_manifest_sha256) == 64
    assert config.sampling.base_seed == 2026
    assert config.sampling.pilot_candidates_per_task == 16
    assert len(config.formal_skill_ids) == len(config.skills) == 34
    assert len(config.candidate_skill_ids) == 5
    assert set(config.formal_skill_ids).isdisjoint(config.candidate_skill_ids)
    assert tuple(by_skill) == config.formal_skill_ids
    assert {
        skill_id
        for skill_id, item in by_skill.items()
        if item.proposal_mode == "learned_conditioned_prior"
    } == LEARNED_CONDITIONED_SKILLS
    assert sum(
        item.proposal_mode == "rule_guided_prior_search" for item in config.skills
    ) == 21

    for item in config.skills:
        expected_strategy = (
            "requested_skill_id"
            if item.skill_id in LEARNED_CONDITIONED_SKILLS
            else "none_skill_id"
        )
        assert item.condition_skill_strategy == expected_strategy
        expected_condition = (
            item.skill_id if expected_strategy == "requested_skill_id" else "<none>"
        )
        assert item.condition_skill_id(config.none_skill_id) == expected_condition


def test_primary_roles_and_joint_limited_set_match_stage_a_contract() -> None:
    config = load_counterfactual_config()
    by_skill = config.skills_by_id

    assert by_skill["short_headway_following"].primary_generated_role == "close_follower"
    assert (
        by_skill["forced_lane_change_around_blockage"].primary_generated_role
        == "avoiding_vehicle"
    )
    assert (
        by_skill["construction_object_lane_blockage"].primary_generated_role
        == "responding_vehicle"
    )
    assert by_skill["static_object_avoidance"].primary_generated_role == "avoiding_vehicle"
    assert by_skill["group_pedestrian_crossing"].proposal_mode == (
        "rule_guided_prior_search"
    )

    assert {
        item.skill_id for item in config.skills if item.joint_generation_limited
    } == {
        "chain_braking",
        "multi_vehicle_gap_squeeze",
        "mutual_yield_deadlock",
        "simultaneous_lane_change_conflict",
    }

    skill_dir = config.formal_catalog.parent
    for item in config.skills:
        skill = load_skill(skill_dir / f"{item.skill_id}.yaml")
        assert item.primary_generated_role in skill.actors["generated_roles"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["skills"].pop("lead_sudden_stop"),
            "missing formal skill entries",
        ),
        (
            lambda value: value["skills"].update(
                {
                    "wrong_way_vehicle": copy.deepcopy(
                        value["skills"]["lead_sudden_stop"]
                    )
                }
            ),
            "non-formal entries",
        ),
        (
            lambda value: value["skills"]["lead_sudden_stop"].update(
                {"primary_generated_role": "missing_role"}
            ),
            "primary_generated_role",
        ),
        (
            lambda value: value["skills"]["lead_sudden_stop"].update(
                {"proposal_mode": "rule_guided_prior_search"}
            ),
            "must be learned_conditioned_prior",
        ),
        (
            lambda value: value["skills"]["group_pedestrian_crossing"].update(
                {"proposal_mode": "learned_conditioned_prior"}
            ),
            "must be rule_guided_prior_search",
        ),
        (
            lambda value: value["skills"]["lead_sudden_stop"].update(
                {"condition_skill_strategy": "none_skill_id"}
            ),
            "must be requested_skill_id",
        ),
        (
            lambda value: value["skills"]["chain_braking"].update(
                {"joint_generation_limited": "yes"}
            ),
            "must be boolean",
        ),
        (
            lambda value: value["active_checkpoint"].update({"sha256": "bad"}),
            "SHA-256",
        ),
        (
            lambda value: value["inputs"].update(
                {"seed_manifest": "../formal_candidates.csv"}
            ),
            "repository-relative path",
        ),
        (
            lambda value: value["sampling"].update(
                {"pilot_candidates_per_task": 0}
            ),
            "positive integer",
        ),
    ],
)
def test_rejects_generation_contract_drift(tmp_path: Path, mutation, message: str) -> None:
    value = _raw(GENERATION_CONFIG_PATH)
    mutation(value)
    path = _write(tmp_path, "counterfactual.yaml", value)

    with pytest.raises(ValueError, match=message):
        load_counterfactual_config(path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "run_manifest_stage",
        "repair_contract",
        "promotion_recommendation",
        "promotion_recommendation_sha256",
    ],
)
def test_rejects_incomplete_active_repair_contract(
    tmp_path: Path,
    missing_field: str,
) -> None:
    value = _raw(GENERATION_CONFIG_PATH)
    value["active_checkpoint"].pop(missing_field)
    path = _write(tmp_path, "counterfactual.yaml", value)

    with pytest.raises(ValueError, match="active_checkpoint has missing"):
        load_counterfactual_config(path)


@pytest.mark.parametrize("partition", ["formal_count", "candidate_count", "overlap"])
def test_rejects_catalog_partition_drift(tmp_path: Path, partition: str) -> None:
    skill_directory = tmp_path / "configs" / "skills"
    shutil.copytree(Path("configs/skills"), skill_directory)
    generation = _raw(GENERATION_CONFIG_PATH)
    generation_path = _write(tmp_path, "counterfactual.yaml", generation)
    formal_path = skill_directory / "catalog.yaml"
    candidate_path = skill_directory / "candidate_catalog.yaml"
    formal = _raw(formal_path)
    candidate = _raw(candidate_path)

    if partition == "formal_count":
        formal["families"]["longitudinal_interaction"].pop()
        formal_path.write_text(yaml.safe_dump(formal, sort_keys=False), encoding="utf-8")
    elif partition == "candidate_count":
        candidate["skills"].pop()
        candidate_path.write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )
    else:
        candidate["skills"][0]["skill_id"] = "lead_sudden_stop"
        candidate_path.write_text(
            yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="34 unique|5 unique|overlap"):
        load_counterfactual_config(generation_path, repository_root=tmp_path)


def test_loads_complete_filter_policy_contract() -> None:
    config = load_filter_config()

    assert config.finite_fields == FINITE_VALUE_FIELDS
    assert config.filter_stages == FILTER_STAGES
    assert config.footprint_ground_truth is False
    assert set(config.footprints_by_type) == COLLISION_PROXY_ACTOR_TYPES
    assert set(config.kinematics_by_type) == {
        "bus",
        "cyclist",
        "motorcyclist",
        "pedestrian",
        "vehicle",
    }
    assert config.kinematics_by_type["vehicle"].maximum_deceleration_mps2 == 12.0
    assert config.map_policy.minimum_inside_fraction == 0.95
    assert config.novelty_policy.minimum_rms_displacement_m == 0.5
    assert config.parameter_policy.unavailable_action == "audit_only"
    assert config.diversity_policy.maximum_per_scenario_skill == 3
    assert all(
        proxy.length_m > 0 and proxy.width_m > 0
        for proxy in config.footprint_proxies
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"skill_thresholds": {}}),
            "unknown keys",
        ),
        (
            lambda value: value["finite_values"]["required"].reverse(),
            "frozen v1 finite fields",
        ),
        (
            lambda value: value["filter_order"]["stages"].reverse(),
            "frozen v1 order",
        ),
        (
            lambda value: value["class_footprint_proxies"]["actor_types"].pop(
                "construction"
            ),
            "cover all actor types",
        ),
        (
            lambda value: value["class_footprint_proxies"]["actor_types"][
                "vehicle"
            ].update({"length_m": 0.0}),
            "positive finite number",
        ),
        (
            lambda value: value["class_footprint_proxies"].update(
                {"ground_truth": True}
            ),
            "cannot be marked as ground truth",
        ),
    ],
)
def test_rejects_filter_contract_drift(tmp_path: Path, mutation, message: str) -> None:
    value = _raw(FILTER_CONFIG_PATH)
    mutation(value)
    path = _write(tmp_path, "filters.yaml", value)

    with pytest.raises(ValueError, match=message):
        load_filter_config(path)

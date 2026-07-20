from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skilldrive.skills.registry import (
    DETECTION_STRATEGIES,
    SHARED_CAPABILITIES,
    SKILL_RULES,
    get_skill_detection_rule,
)


SKILL_DIR = Path("configs/skills")


def _catalog_entries() -> list[dict[str, str]]:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    return [entry for family in catalog["families"].values() for entry in family]


def _candidate_entries() -> list[dict[str, object]]:
    catalog = yaml.safe_load(
        (SKILL_DIR / "candidate_catalog.yaml").read_text(encoding="utf-8")
    )
    return catalog["skills"]


def test_registry_matches_formal_and_candidate_rule_catalogs() -> None:
    entries = _catalog_entries()
    candidate_entries = _candidate_entries()
    formal_ids = {entry["skill_id"] for entry in entries}
    candidate_ids = {entry["skill_id"] for entry in candidate_entries}

    assert len(entries) == len(formal_ids) == 34
    assert len(candidate_entries) == len(candidate_ids) == 5
    assert formal_ids.isdisjoint(candidate_ids)
    assert len(SKILL_RULES) == 39
    assert set(SKILL_RULES) == formal_ids | candidate_ids
    assert {entry["feasibility"] for entry in entries} <= {"A", "B"}
    assert all(key == rule.skill_id for key, rule in SKILL_RULES.items())


def test_every_rule_has_known_strategy_capabilities_and_role_count() -> None:
    used_strategies = set()
    for skill_id, rule in SKILL_RULES.items():
        used_strategies.add(rule.strategy)
        assert rule.strategy in DETECTION_STRATEGIES
        assert rule.required_capabilities
        assert set(rule.required_capabilities) <= SHARED_CAPABILITIES

        skill = yaml.safe_load(
            (SKILL_DIR / f"{skill_id}.yaml").read_text(encoding="utf-8")
        )
        assert rule.primary_actor_count == len(skill["actors"]["generated_roles"])

    assert used_strategies == DETECTION_STRATEGIES


def test_registry_lookup_is_explicit_for_unknown_skill() -> None:
    assert get_skill_detection_rule("lead_hard_brake") is SKILL_RULES["lead_hard_brake"]
    with pytest.raises(KeyError, match="unknown skill_id: missing_skill"):
        get_skill_detection_rule("missing_skill")


def test_stopped_reentry_contract_uses_ordered_three_vehicle_roles() -> None:
    skill = yaml.safe_load(
        (SKILL_DIR / "stopped_vehicle_reentry.yaml").read_text(encoding="utf-8")
    )
    rule = get_skill_detection_rule("stopped_vehicle_reentry")

    assert skill["actors"]["generated_roles"] == [
        "reentering_vehicle",
        "front_main_flow_vehicle",
        "rear_main_flow_vehicle",
    ]
    assert skill["detection"]["conditions"] == [
        "currently_stopped",
        "front_and_rear_main_flow_vehicles_present",
        "reentry_space_available",
    ]
    assert skill["detection"]["thresholds"] == {
        "stopped_speed_mps": {"value": 0.5, "source": "semantic"},
        "minimum_stopped_duration_s": {"value": 1.0, "source": "semantic"},
        "minimum_moving_speed_mps": {"value": 1.0, "source": "semantic"},
        "minimum_vehicle_center_distance_m": {"value": 2.0, "source": "semantic"},
        "maximum_lateral_reentry_distance_m": {"value": 5.0, "source": "semantic"},
        "maximum_front_gap_m": {"value": 55.0, "source": "semantic"},
        "maximum_rear_gap_m": {"value": 55.0, "source": "semantic"},
    }
    assert rule.primary_actor_count == 3
    assert {
        "longitudinal_order",
        "minimum_distance",
        "multi_actor_relation",
    } <= set(rule.required_capabilities)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        SKILL_RULES["missing_skill"] = SKILL_RULES["lead_hard_brake"]  # type: ignore[index]

"""Exact-role re-detection for skills with real v5 training supervision."""

from __future__ import annotations

from typing import Mapping

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.schemas import Scenario, SkillSpec
from skilldrive.skills.detection import DetectionConfig, detect_scenario
from skilldrive.filtering.roles import validate_role_contract


def validate_observed_skill(
    scenario: Scenario,
    skill: SkillSpec,
    role_track_ids: Mapping[str, str],
    detection_config: DetectionConfig,
) -> FilterCheck:
    """Re-run one observed detector on only the requested roles and require exact IDs."""

    if skill.detection["mode"] != "observed_trigger":
        raise ValueError(
            f"observed validator requires observed_trigger detection mode: {skill.skill_id}"
        )
    generated_roles = tuple(skill.actors["generated_roles"])
    normalized_roles = {
        str(role): str(track_id) for role, track_id in role_track_ids.items()
    }
    agents = {agent.track_id: agent for agent in scenario.agents}
    role_contract = validate_role_contract(scenario, skill, normalized_roles)
    if not role_contract.passed:
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.OBSERVED_ROLE_CONTRACT_MISMATCH,),
            metrics=dict(role_contract.metrics),
        )

    ordered_track_ids = tuple(normalized_roles[role] for role in generated_roles)
    restricted = Scenario(
        scenario_id=scenario.scenario_id,
        city_name=scenario.city_name,
        timestamps=scenario.timestamps.copy(),
        focal_track_id=ordered_track_ids[0],
        agents=[agents[track_id] for track_id in ordered_track_ids],
        map_polylines=list(scenario.map_polylines),
        metadata=dict(scenario.metadata),
    )
    run = detect_scenario(restricted, [skill], detection_config)
    match = next(
        (
            record
            for record in run.records
            if record.skill_id == skill.skill_id
            and record.role_track_ids == normalized_roles
        ),
        None,
    )
    if match is None:
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.OBSERVED_SKILL_NOT_REDETECTED,),
            metrics={
                "requested_role_track_ids": normalized_roles,
                "detector_rejections": dict(sorted(run.rejection_counts.items())),
            },
        )
    return FilterCheck(
        stage=FilterStage.SKILL_TRIGGER,
        metrics={
            "requested_role_track_ids": normalized_roles,
            "trigger_score": match.trigger_score,
            "risk_metric": match.seed_risk_metric,
            "risk_value": match.seed_risk_value,
            "evidence": match.evidence,
        },
    )


__all__ = ["validate_observed_skill"]

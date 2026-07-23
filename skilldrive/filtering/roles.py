"""Exact generated-role contracts shared by all skill validators."""

from __future__ import annotations

from typing import Mapping

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.schemas import Scenario, SkillSpec


def validate_role_contract(
    scenario: Scenario,
    skill: SkillSpec,
    role_track_ids: Mapping[str, str],
) -> FilterCheck:
    required_roles = tuple(str(role) for role in skill.actors["generated_roles"])
    normalized = {str(role): str(track_id) for role, track_id in role_track_ids.items()}
    agents = {agent.track_id: agent for agent in scenario.agents}
    missing_roles = sorted(set(required_roles) - set(normalized))
    unexpected_roles = sorted(set(normalized) - set(required_roles))
    missing_tracks = sorted(
        track_id for track_id in normalized.values() if track_id not in agents
    )
    duplicate_track_ids = len(set(normalized.values())) != len(normalized)
    passed = not (
        missing_roles
        or unexpected_roles
        or missing_tracks
        or duplicate_track_ids
    )
    return FilterCheck(
        stage=FilterStage.SKILL_TRIGGER,
        rejection_reasons=(
            () if passed else (FilterRejection.SKILL_ROLE_CONTRACT_MISMATCH,)
        ),
        metrics={
            "required_roles": list(required_roles),
            "requested_role_track_ids": normalized,
            "missing_roles": missing_roles,
            "unexpected_roles": unexpected_roles,
            "missing_track_ids": missing_tracks,
            "duplicate_track_ids": duplicate_track_ids,
        },
    )


__all__ = ["validate_role_contract"]

"""Novelty checks against the original target future for observed-trigger skills."""

from __future__ import annotations

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.config import NoveltyFilterPolicy
from skilldrive.schemas import Scenario


def check_observed_future_novelty(
    source_scenario: Scenario,
    target_track_id: str,
    future_xy_global: np.ndarray,
    policy: NoveltyFilterPolicy,
) -> FilterCheck:
    target = next(
        (agent for agent in source_scenario.agents if agent.track_id == target_track_id),
        None,
    )
    generated = np.asarray(future_xy_global, dtype=np.float64)
    if target is None or generated.shape != (60, 2):
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.NOVELTY_REFERENCE_UNAVAILABLE,),
            metrics={"reference_status": "target_or_generated_future_unavailable"},
        )
    original = np.asarray(target.positions[50:110], dtype=np.float64)
    valid = np.isfinite(original).all(axis=1) & np.isfinite(generated).all(axis=1)
    if len(original) != 60 or not valid.all():
        return FilterCheck(
            stage=FilterStage.SKILL_TRIGGER,
            rejection_reasons=(FilterRejection.NOVELTY_REFERENCE_UNAVAILABLE,),
            metrics={
                "reference_status": "original_future_incomplete",
                "valid_reference_steps": int(valid.sum()),
            },
        )
    displacement = np.linalg.norm(generated - original, axis=1)
    rms = float(np.sqrt(np.mean(displacement * displacement)))
    endpoint = float(displacement[-1])
    passed = (
        rms >= policy.minimum_rms_displacement_m
        or endpoint >= policy.minimum_endpoint_displacement_m
    )
    return FilterCheck(
        stage=FilterStage.SKILL_TRIGGER,
        rejection_reasons=(
            () if passed else (FilterRejection.NOVELTY_INSUFFICIENT,)
        ),
        metrics={
            "policy_source": policy.source,
            "rms_displacement_m": rms,
            "endpoint_displacement_m": endpoint,
            "minimum_rms_displacement_m": policy.minimum_rms_displacement_m,
            "minimum_endpoint_displacement_m": policy.minimum_endpoint_displacement_m,
            "acceptance_rule": "rms_or_endpoint",
        },
    )


__all__ = ["check_observed_future_novelty"]

from __future__ import annotations

import numpy as np

from skilldrive.filtering.contracts import FilterRejection
from skilldrive.filtering.diversity import DiversityCandidate, apply_diversity_filter
from skilldrive.generation.config import load_filter_config


def _candidate(
    candidate_id: str,
    *,
    scenario_id: str = "scene",
    offset: float = 0.0,
    quality: float = 0.0,
    realized_parameter_bins: tuple[tuple[str, int], ...] = (),
) -> DiversityCandidate:
    future = np.column_stack(
        (
            np.linspace(0.0, 10.0, 60),
            np.full(60, offset),
        )
    )
    return DiversityCandidate(
        candidate_id=candidate_id,
        scenario_id=scenario_id,
        skill_id="skill",
        future_xy_local=future,
        target_risk_value=2.0,
        quality_score=quality,
        realized_parameter_bins=realized_parameter_bins,
    )


def test_near_identical_candidate_is_rejected_before_group_limit() -> None:
    policy = load_filter_config().diversity_policy
    results = apply_diversity_filter(
        [_candidate("a", offset=0.0), _candidate("b", offset=0.1)],
        policy,
    )

    assert results["a"].passed
    assert results["b"].rejection_reasons == (
        FilterRejection.DIVERSITY_TRAJECTORY_TOO_SIMILAR,
    )


def test_same_scene_skill_keeps_at_most_three_distinct_candidates() -> None:
    policy = load_filter_config().diversity_policy
    candidates = [
        _candidate(str(index), offset=float(index) * 2.0)
        for index in range(4)
    ]
    results = apply_diversity_filter(candidates, policy)

    assert all(results[str(index)].passed for index in range(3))
    assert results["3"].rejection_reasons == (
        FilterRejection.DIVERSITY_SCENARIO_SKILL_LIMIT,
    )


def test_cross_scene_duplicate_uses_local_trajectory_and_risk_signature() -> None:
    policy = load_filter_config().diversity_policy
    results = apply_diversity_filter(
        [_candidate("a", scenario_id="first"), _candidate("b", scenario_id="second")],
        policy,
    )

    assert results["a"].passed
    assert results["b"].rejection_reasons == (
        FilterRejection.DIVERSITY_GLOBAL_DUPLICATE,
    )


def test_cross_scene_signature_includes_realized_parameter_bins() -> None:
    policy = load_filter_config().diversity_policy
    results = apply_diversity_filter(
        [
            _candidate(
                "a",
                scenario_id="first",
                realized_parameter_bins=(("speed_scale", 3),),
            ),
            _candidate(
                "b",
                scenario_id="second",
                realized_parameter_bins=(("speed_scale", 4),),
            ),
        ],
        policy,
    )

    assert results["a"].passed
    assert results["b"].passed


def test_cross_scene_duplicate_keeps_lower_quality_score() -> None:
    policy = load_filter_config().diversity_policy
    results = apply_diversity_filter(
        [
            _candidate("worse", scenario_id="a-scene", quality=0.5),
            _candidate("better", scenario_id="z-scene", quality=0.1),
        ],
        policy,
    )

    assert results["better"].passed
    assert results["worse"].rejection_reasons == (
        FilterRejection.DIVERSITY_GLOBAL_DUPLICATE,
    )

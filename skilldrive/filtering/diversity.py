"""Deterministic within-scene and cross-scene trajectory diversity selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.config import DiversityFilterPolicy


@dataclass(frozen=True)
class DiversityCandidate:
    candidate_id: str
    scenario_id: str
    skill_id: str
    future_xy_local: np.ndarray
    target_risk_value: float
    quality_score: float
    realized_parameter_bins: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        future = np.asarray(self.future_xy_local, dtype=np.float64)
        if future.shape != (60, 2) or not np.isfinite(future).all():
            raise ValueError("future_xy_local must be finite with shape (60, 2)")
        for name in ("target_risk_value", "quality_score"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("candidate_id", "scenario_id", "skill_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        bins = tuple(self.realized_parameter_bins)
        for item in bins:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                    "realized_parameter_bins must contain (name, integer) pairs"
                )
            name, value = item
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, int)
            ):
                raise ValueError(
                    "realized_parameter_bins must contain (name, integer) pairs"
                )
        if bins != tuple(sorted(bins)) or len({name for name, _ in bins}) != len(bins):
            raise ValueError("realized_parameter_bins must be sorted with unique names")
        object.__setattr__(self, "future_xy_local", np.ascontiguousarray(future.copy()))
        object.__setattr__(self, "realized_parameter_bins", bins)


def _pair_distances(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = first - second
    rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    endpoint = float(np.linalg.norm(delta[-1]))
    return rms, endpoint


def _global_signature(
    candidate: DiversityCandidate,
    policy: DiversityFilterPolicy,
) -> tuple[object, ...]:
    sampled = candidate.future_xy_local[[14, 29, 44, 59]]
    trajectory_bins = tuple(
        int(round(float(value) / policy.global_endpoint_bin_m))
        for value in sampled.reshape(-1)
    )
    risk_bin = int(round(candidate.target_risk_value / policy.global_risk_bin))
    return (
        candidate.skill_id,
        *trajectory_bins,
        risk_bin,
        candidate.realized_parameter_bins,
    )


def apply_diversity_filter(
    candidates: Sequence[DiversityCandidate],
    policy: DiversityFilterPolicy,
) -> Mapping[str, FilterCheck]:
    """Select deterministic representatives after all individual quality gates pass."""

    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("diversity candidates must have unique candidate IDs")
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.skill_id,
            item.quality_score,
            item.scenario_id,
            item.candidate_id,
        ),
    )
    accepted_by_group: dict[tuple[str, str], list[DiversityCandidate]] = {}
    global_signatures: dict[tuple[object, ...], str] = {}
    results: dict[str, FilterCheck] = {}
    for candidate in ordered:
        group_key = (candidate.scenario_id, candidate.skill_id)
        group = accepted_by_group.setdefault(group_key, [])
        closest_id: str | None = None
        closest_rms: float | None = None
        closest_endpoint: float | None = None
        for previous in group:
            rms, endpoint = _pair_distances(
                candidate.future_xy_local,
                previous.future_xy_local,
            )
            if closest_rms is None or (rms, endpoint, previous.candidate_id) < (
                closest_rms,
                closest_endpoint if closest_endpoint is not None else float("inf"),
                closest_id or "",
            ):
                closest_id = previous.candidate_id
                closest_rms = rms
                closest_endpoint = endpoint
        reasons: list[FilterRejection] = []
        if (
            closest_rms is not None
            and closest_endpoint is not None
            and closest_rms < policy.minimum_pairwise_rms_m
            and closest_endpoint < policy.minimum_endpoint_distance_m
        ):
            reasons.append(FilterRejection.DIVERSITY_TRAJECTORY_TOO_SIMILAR)
        elif len(group) >= policy.maximum_per_scenario_skill:
            reasons.append(FilterRejection.DIVERSITY_SCENARIO_SKILL_LIMIT)

        signature = _global_signature(candidate, policy)
        duplicate_id = global_signatures.get(signature)
        if not reasons and duplicate_id is not None:
            reasons.append(FilterRejection.DIVERSITY_GLOBAL_DUPLICATE)

        if not reasons:
            group.append(candidate)
            global_signatures[signature] = candidate.candidate_id
        results[candidate.candidate_id] = FilterCheck(
            stage=FilterStage.DIVERSITY,
            rejection_reasons=tuple(reasons),
            metrics={
                "policy_source": policy.source,
                "accepted_rank_within_scenario_skill": (
                    len(group) if not reasons else None
                ),
                "closest_accepted_candidate_id": closest_id,
                "closest_pairwise_rms_m": closest_rms,
                "closest_endpoint_distance_m": closest_endpoint,
                "global_duplicate_candidate_id": duplicate_id,
                "realized_parameter_bins": [
                    [name, value] for name, value in candidate.realized_parameter_bins
                ],
            },
        )
    return results


__all__ = ["DiversityCandidate", "apply_diversity_filter"]

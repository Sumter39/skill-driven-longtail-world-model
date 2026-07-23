"""Auditable parameter realization checks for generated single-target futures."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from skilldrive.filtering.common import FutureKinematics
from skilldrive.filtering.contracts import FilterCheck, FilterRejection, FilterStage
from skilldrive.generation.config import ParameterFilterPolicy
from skilldrive.schemas import Scenario


_PRIMARY_SPEED_PARAMETERS = frozenset(
    {
        "creep_speed_mps",
        "crossing_speed_mps",
        "cyclist_speed_mps",
        "filtering_speed_mps",
        "pedestrian_speed_mps",
        "turn_speed_mps",
    }
)
_PRIMARY_SCALE_PARAMETERS = frozenset(
    {
        "first_speed_scale",
        "follower_speed_scale",
        "leader_speed_scale",
        "merge_speed_scale",
        "speed_scale",
    }
)
_RISK_TIME_PARAMETERS = frozenset(
    {"accepted_gap_s", "arrival_time_gap_s", "target_headway_s"}
)
_RISK_DISTANCE_PARAMETERS = frozenset({"object_buffer_m"})


def _target(scenario: Scenario, target_track_id: str):
    return next(
        (agent for agent in scenario.agents if agent.track_id == target_track_id),
        None,
    )


def _finite_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not len(finite) else float(finite.mean())


def _tolerance_key(parameter_name: str) -> str | None:
    if parameter_name.endswith("_mps2"):
        return "meters_per_second_squared"
    if parameter_name.endswith("_mps"):
        return "meters_per_second"
    if parameter_name.endswith("_deg") or parameter_name.endswith("_deg_s"):
        return "degrees"
    if parameter_name.endswith("_scale"):
        return "scale"
    if parameter_name.endswith("_s"):
        return "seconds"
    if parameter_name.endswith("_m"):
        return "meters"
    return None


def _realized_value(
    parameter_name: str,
    source_scenario: Scenario,
    generated_scenario: Scenario,
    target_track_id: str,
    kinematics: FutureKinematics,
    risk_metric: str | None,
    risk_value: float | None,
) -> tuple[float | None, str]:
    target = _target(generated_scenario, target_track_id)
    source_target = _target(source_scenario, target_track_id)
    if target is None or source_target is None:
        return None, "target_track_unavailable"

    if parameter_name in _PRIMARY_SPEED_PARAMETERS:
        value = _finite_mean(kinematics.speed_mps)
        return value, "generated_future_mean_speed"
    if parameter_name in _PRIMARY_SCALE_PARAMETERS:
        original_speed = np.linalg.norm(source_target.velocities[50:110], axis=1)
        original = _finite_mean(original_speed)
        generated = _finite_mean(kinematics.speed_mps)
        if original is None or generated is None or original <= 1e-6:
            return None, "original_future_speed_unavailable"
        return generated / original, "generated_to_original_mean_speed_ratio"
    if parameter_name == "peak_deceleration_mps2":
        return float(np.max(kinematics.deceleration_mps2)), "generated_peak_deceleration"
    if parameter_name == "acceleration_mps2":
        positive = np.maximum(kinematics.tangential_acceleration_mps2, 0.0)
        return float(np.max(positive)), "generated_peak_positive_acceleration"
    if parameter_name == "turn_radius_m":
        maximum = float(np.max(kinematics.curvature_per_m))
        if maximum <= 1e-9:
            return None, "generated_curvature_too_small"
        return 1.0 / maximum, "inverse_generated_peak_curvature"
    if parameter_name == "angular_speed_deg_s":
        return float(np.degrees(np.max(kinematics.heading_rate_rad_s))), "generated_peak_heading_rate"
    if parameter_name in _RISK_TIME_PARAMETERS:
        if risk_value is None or risk_metric is None or "time" not in risk_metric and "headway" not in risk_metric:
            return None, "target_risk_is_not_a_time_metric"
        return risk_value, "generated_target_risk_time"
    if parameter_name in _RISK_DISTANCE_PARAMETERS:
        if risk_value is None or risk_metric not in {
            "minimum_object_clearance",
            "minimum_combined_clearance",
        }:
            return None, "target_risk_is_not_a_clearance_metric"
        return risk_value, "generated_target_risk_clearance"
    return None, "no_frozen_realization_extractor"


def check_parameter_realization(
    *,
    requested_parameters: Mapping[str, Any],
    source_scenario: Scenario,
    generated_scenario: Scenario,
    target_track_id: str,
    kinematics: FutureKinematics,
    risk_metric: str | None,
    risk_value: float | None,
    policy: ParameterFilterPolicy,
) -> FilterCheck:
    """Evaluate only parameters with a frozen extractor; report all others honestly."""

    evidence: dict[str, dict[str, Any]] = {}
    reasons: list[FilterRejection] = []
    for name, requested in sorted(requested_parameters.items()):
        if isinstance(requested, bool) or not isinstance(requested, (int, float)):
            evidence[name] = {
                "status": "unavailable",
                "requested": requested,
                "reason": "non_numeric_or_choice_parameter",
            }
            continue
        requested_value = float(requested)
        if not math.isfinite(requested_value):
            reasons.append(FilterRejection.PARAMETER_NON_FINITE)
            evidence[name] = {
                "status": "invalid",
                "requested": None,
                "reason": "requested_value_non_finite",
            }
            continue
        realized, source = _realized_value(
            name,
            source_scenario,
            generated_scenario,
            target_track_id,
            kinematics,
            risk_metric,
            risk_value,
        )
        if realized is None or not math.isfinite(realized):
            evidence[name] = {
                "status": "unavailable",
                "requested": requested_value,
                "reason": source,
            }
            if policy.unavailable_action == "reject":
                reasons.append(FilterRejection.PARAMETER_OUT_OF_TOLERANCE)
            continue
        tolerance_key = _tolerance_key(name)
        if tolerance_key is None:
            evidence[name] = {
                "status": "unavailable",
                "requested": requested_value,
                "reason": "parameter_unit_has_no_frozen_tolerance",
            }
            if policy.unavailable_action == "reject":
                reasons.append(FilterRejection.PARAMETER_OUT_OF_TOLERANCE)
            continue
        tolerance = float(policy.absolute_tolerances[tolerance_key])
        absolute_error = abs(realized - requested_value)
        within = absolute_error <= tolerance
        evidence[name] = {
            "status": "computed",
            "requested": requested_value,
            "realized": realized,
            "absolute_error": absolute_error,
            "relative_error": (
                None if abs(requested_value) <= 1e-12 else absolute_error / abs(requested_value)
            ),
            "absolute_tolerance": tolerance,
            "tolerance_key": tolerance_key,
            "within_tolerance": within,
            "realization_source": source,
            "conditioning_claim": "rule_search_realization_not_direct_parameter_control",
        }
        if not within and policy.out_of_tolerance_action == "reject":
            reasons.append(FilterRejection.PARAMETER_OUT_OF_TOLERANCE)

    return FilterCheck(
        stage=FilterStage.PARAMETER_REALIZATION,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        metrics={
            "policy_source": policy.source,
            "unavailable_action": policy.unavailable_action,
            "out_of_tolerance_action": policy.out_of_tolerance_action,
            "parameters": evidence,
        },
    )


__all__ = ["check_parameter_realization"]

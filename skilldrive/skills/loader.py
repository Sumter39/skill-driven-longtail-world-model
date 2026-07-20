"""Load and strictly validate executable skill specifications from YAML."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from skilldrive.schemas import SkillSpec


REQUIRED_FIELDS = {
    "skill_id",
    "name_zh",
    "family",
    "definition",
    "source",
    "data_support",
    "seed_requirements",
    "trigger",
    "detection",
    "actors",
    "parameters",
    "generation_operators",
    "constraints",
    "risk_definition",
    "expected_behavior",
    "validation_metrics",
    "known_limitations",
    "output_labels",
}

ALLOWED_FAMILIES = {
    "longitudinal_interaction",
    "lane_change_interaction",
    "merge_topology",
    "intersection_interaction",
    "vulnerable_road_user",
    "atypical_composite",
}

ALLOWED_SOURCES = {
    "course_example",
    "traffic_rule",
    "safety_metric",
    "literature",
    "train_pattern",
}

ALLOWED_FEASIBILITY = {"A", "B"}
ALLOWED_DETECTION_MODES = {"observed_trigger", "compatible_seed"}
EXPECTED_MODE_BY_FEASIBILITY = {"A": "observed_trigger", "B": "compatible_seed"}
ALLOWED_THRESHOLD_SOURCES = {"semantic", "train_statistics", "reference"}
ALLOWED_RISK_DIRECTIONS = {"lower_is_riskier", "higher_is_riskier"}

ALLOWED_TRACK_REQUIREMENTS = {
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "static",
    "construction",
}

ALLOWED_ACTOR_TYPES = {
    "vehicle",
    "bus",
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "static",
    "construction",
}

# These names describe either a directly stored map layer or a deterministic
# relation derived from the lane graph exposed by the AV2 adapter.
ALLOWED_MAP_REQUIREMENTS = {
    "lane_centerline",
    "lane_successor",
    "adjacent_lane",
    "converging_lane",
    "diverging_lane",
    "intersection_lane",
    "pedestrian_crossing",
    "drivable_area",
    "bike_lane",
    "lane_direction",
}

ALLOWED_GENERATION_OPERATORS = {
    "LONGITUDINAL",
    "LANE_CHANGE",
    "MERGE",
    "CONFLICT_POINT",
    "YIELD_PRIORITY",
    "VRU_CROSSING",
    "BLOCKAGE_RESPONSE",
    "MULTI_AGENT",
}

ALLOWED_CONSTRAINTS = {
    "allow_safe_stop",
    "blocking_actor_near_stationary",
    "class_specific_speed_limit",
    "construction_actor_static",
    "continuous_crossing_motion",
    "continuous_entry_motion",
    "forbid_unavoidable_collision",
    "low_speed_motion",
    "maintain_role_consistency",
    "nonnegative_speed",
    "preserve_lane_direction",
    "preserve_path_direction",
    "remain_in_drivable_area",
    "remain_near_lane_centerline",
    "respect_object_buffer",
    "role_assignment_fixed",
    "shared_target_lane",
    "smooth_acceleration",
    "smooth_lateral_motion",
    "smooth_turning_motion",
    "static_actor_stationary",
    "sustained_opposite_direction",
    "vehicle_remains_in_drivable_area",
}

ALLOWED_RISK_METRICS = {
    "conflict_area_intrusion_margin",
    "conflict_area_occupancy_overlap",
    "conflict_point_time_gap",
    "convergence_time_gap",
    "first_intrusion_time_to_collision",
    "head_on_time_to_collision",
    "lateral_time_to_collision",
    "minimum_combined_clearance",
    "minimum_crossing_time_to_collision",
    "minimum_front_rear_time_to_collision",
    "minimum_longitudinal_gap",
    "minimum_object_clearance",
    "minimum_stage_time_to_collision",
    "newly_exposed_time_to_collision",
    "post_cut_in_time_to_collision",
    "post_encroachment_time",
    "rear_time_to_collision",
    "stopping_distance_margin",
    "time_headway",
    "time_to_collision",
}

# Both human-readable trigger conditions and executable seed-detection
# conditions are enumerated so misspellings fail at load time.
ALLOWED_CONDITIONS = {
    "adjacent_lane",
    "adjacent_lane_available",
    "adjacent_vehicle_cuts_in",
    "approach_to_safety_buffer",
    "away_from_crosswalk",
    "begins_moving",
    "bike_vehicle_paths_merge",
    "blockage_ahead",
    "close_longitudinal_position",
    "competing_arrival",
    "competing_vehicles_present",
    "conflicting_vehicle_present",
    "construction_actor_near_path",
    "converging_lanes",
    "crossing_flow_approaches",
    "crossing_flow_present",
    "crossing_or_merging_vehicle",
    "crossing_paths",
    "crossing_vehicle_path",
    "crossing_vehicle_present",
    "currently_stopped",
    "cyclist_crosses_vehicle_path",
    "delayed_entry",
    "delayed_braking_response",
    "diverging_topology",
    "drivable_entry_available",
    "enters_main_flow",
    "explicit_priority_role",
    "follower_approaching",
    "follower_closing",
    "front_and_rear_main_flow_vehicles_present",
    "front_gap_small",
    "filtering_gap_available",
    "group_members_present",
    "holding_space_available",
    "initiator_brakes",
    "inside_conflict_area",
    "insufficient_gap",
    "intersection_conflict_area_available",
    "lane_change",
    "lane_successors_converge",
    "late_lateral_crossing",
    "lateral_crossing_feasible",
    "lateral_entry",
    "lead_vehicle_cuts_out",
    "leader_decelerating",
    "left_turn_path",
    "main_flow_vehicle_present",
    "motorcyclist_between_vehicles",
    "multi_direction_gap_closing",
    "moving_to_stopped",
    "near_intersection_entry",
    "newly_exposed_slow_vehicle",
    "non_congested_speed",
    "oncoming_vehicle_present",
    "opposing_through_path",
    "overlapping_arrival_window",
    "overlapping_lane_change_window",
    "pedestrian_enters_drivable_area",
    "pedestrian_near_crosswalk",
    "pedestrian_near_drivable_boundary",
    "pedestrian_on_crosswalk",
    "pedestrian_group_crossing",
    "positive_relative_speed",
    "post_cut_in_braking_space",
    "predicted_buffer_intrusion",
    "previously_stopped",
    "priority_roles_assignable",
    "rear_gap_small",
    "rear_vehicle_closing",
    "reentry_space_available",
    "response_space_available",
    "right_turn_path",
    "safe_creep_space",
    "safety_buffer_violation_predicted",
    "same_lane",
    "same_or_successor_lane",
    "shared_conflict_area",
    "shared_conflict_point",
    "shared_target_lane",
    "shared_target_lane_available",
    "short_post_cut_in_delay",
    "slow_vehicle_ahead",
    "small_arrival_time_gap",
    "stable_conflict_point",
    "stable_lane_match",
    "static_actor_near_path",
    "sustained_creep_speed",
    "sustained_low_speed",
    "sustained_low_speed_or_stop",
    "sustained_opposite_heading",
    "sustained_short_headway",
    "target_gap_ahead",
    "three_vehicle_queue",
    "turning_vehicle",
    "two_lane_change_paths_available",
    "vehicle_approaching",
    "vehicle_path_conflict",
    "yielding_actor_continues",
}

ALLOWED_DETECTION_THRESHOLDS = {
    "maximum_arrival_time_gap_s",
    "maximum_blockage_distance_m",
    "maximum_blocker_speed_mps",
    "maximum_boundary_distance_m",
    "maximum_competing_vehicle_gap_m",
    "maximum_conflict_distance_m",
    "maximum_convergence_distance_m",
    "maximum_creep_speed_mps",
    "maximum_crossing_angle_deg",
    "maximum_crossing_arrival_s",
    "maximum_crosswalk_distance_m",
    "maximum_distance_to_diverge_m",
    "maximum_entry_distance_m",
    "maximum_front_gap_m",
    "maximum_filtering_vehicle_gap_m",
    "maximum_group_heading_difference_deg",
    "maximum_group_member_distance_m",
    "maximum_initial_gap_m",
    "maximum_lateral_reentry_distance_m",
    "maximum_leader_speed_mps",
    "maximum_longitudinal_gap_m",
    "maximum_main_flow_gap_m",
    "maximum_merge_heading_difference_deg",
    "maximum_motorcyclist_vehicle_distance_m",
    "maximum_object_path_distance_m",
    "maximum_object_speed_mps",
    "maximum_oncoming_distance_m",
    "maximum_pair_gap_m",
    "maximum_queue_gap_m",
    "maximum_rear_gap_m",
    "maximum_slow_vehicle_speed_mps",
    "maximum_target_gap_m",
    "maximum_time_headway_s",
    "maximum_time_to_collision_s",
    "maximum_vehicle_arrival_s",
    "minimum_adjacent_lane_length_m",
    "minimum_blockage_distance_m",
    "minimum_closing_speed_mps",
    "minimum_combined_closing_speed_mps",
    "minimum_conflict_area_length_m",
    "minimum_crossing_angle_deg",
    "minimum_crossing_vehicle_speed_mps",
    "minimum_current_separation_m",
    "minimum_deceleration_mps2",
    "minimum_duration_s",
    "minimum_follower_speed_mps",
    "minimum_lateral_displacement_m",
    "minimum_low_speed_duration_s",
    "minimum_moving_speed_mps",
    "minimum_opposing_heading_difference_deg",
    "minimum_opposite_heading_duration_s",
    "minimum_opposite_heading_difference_deg",
    "minimum_pair_gap_m",
    "minimum_post_cut_in_braking_distance_m",
    "minimum_prior_speed_mps",
    "minimum_queue_gap_m",
    "minimum_relative_speed_mps",
    "minimum_shared_target_length_m",
    "minimum_stopped_duration_s",
    "minimum_target_gap_m",
    "minimum_turn_heading_change_deg",
    "minimum_vehicle_center_distance_m",
    "minimum_crosswalk_clearance_m",
    "stopped_speed_mps",
}

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def _mapping(value: Any, name: str, fields: set[str], *, non_empty: bool = True) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise ValueError(f"{name} must be a {qualifier}mapping")
    missing = fields - set(value)
    unknown = set(value) - fields
    if missing:
        raise ValueError(f"{name} has missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text_list(value: Any, name: str, *, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = [_text(item, f"{name} item") for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    if allowed is not None:
        unknown = set(result) - allowed
        if unknown:
            raise ValueError(f"{name} has unknown values: {sorted(unknown)}")
    return result


def _number(value: Any, name: str, *, nonnegative: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _range(value: Any, name: str, *, nonnegative: bool = False) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    low = _number(value[0], f"{name} lower bound", nonnegative=nonnegative)
    high = _number(value[1], f"{name} upper bound", nonnegative=nonnegative)
    if low > high:
        raise ValueError(f"{name} has an invalid range")
    return [low, high]


def _threshold_source(value: Any, name: str) -> str:
    source = _text(value, name)
    if source not in ALLOWED_THRESHOLD_SOURCES:
        raise ValueError(f"{name} has an unknown threshold source")
    return source


def _validate_parameter(name: str, value: Any) -> None:
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ValueError("parameter names must be lower-case identifiers")
    if not isinstance(value, Mapping):
        raise ValueError(f"parameter {name} must be a mapping")
    has_range = "range" in value
    has_choices = "choices" in value
    expected = {"source", "range" if has_range else "choices"}
    if has_range == has_choices or set(value) != expected:
        raise ValueError(f"parameter {name} must contain source and exactly one of range or choices")
    _threshold_source(value["source"], f"parameter {name}.source")
    if has_range:
        _range(value["range"], f"parameter {name}.range")
        return
    choices = value["choices"]
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"parameter {name}.choices must be a non-empty list")
    for choice in choices:
        if choice is None or isinstance(choice, (str, bool, int)):
            continue
        if isinstance(choice, float) and math.isfinite(choice):
            continue
        raise ValueError(f"parameter {name}.choices must contain JSON scalar values")


def validate_skill_dict(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("skill YAML must contain one mapping")
    _mapping(data, "skill", REQUIRED_FIELDS)

    for name in ("skill_id", "name_zh", "definition"):
        _text(data[name], name)
    family = _text(data["family"], "family")
    if family not in ALLOWED_FAMILIES:
        raise ValueError(f"family has an unknown value: {family}")

    _text_list(data["source"], "source", allowed=ALLOWED_SOURCES)
    for name in ("expected_behavior", "validation_metrics", "known_limitations", "output_labels"):
        _text_list(data[name], name)
    _text_list(
        data["generation_operators"],
        "generation_operators",
        allowed=ALLOWED_GENERATION_OPERATORS,
    )

    support = _mapping(
        data["data_support"],
        "data_support",
        {"feasibility", "required_tracks", "required_map", "development_evidence"},
    )
    feasibility = _text(support["feasibility"], "data_support.feasibility")
    if feasibility not in ALLOWED_FEASIBILITY:
        raise ValueError("data_support.feasibility must be A or B")
    _text_list(
        support["required_tracks"],
        "data_support.required_tracks",
        allowed=ALLOWED_TRACK_REQUIREMENTS,
    )
    _text_list(
        support["required_map"],
        "data_support.required_map",
        allowed=ALLOWED_MAP_REQUIREMENTS,
    )
    _text(support["development_evidence"], "data_support.development_evidence")

    seed = _mapping(
        data["seed_requirements"],
        "seed_requirements",
        {"description", "threshold_source", "minimum_history_steps"},
    )
    _text(seed["description"], "seed_requirements.description")
    _threshold_source(seed["threshold_source"], "seed_requirements.threshold_source")
    history = seed["minimum_history_steps"]
    if isinstance(history, bool) or not isinstance(history, int) or history <= 0:
        raise ValueError("seed_requirements.minimum_history_steps must be a positive integer")

    trigger = _mapping(
        data["trigger"],
        "trigger",
        {"description", "threshold_source", "conditions"},
    )
    _text(trigger["description"], "trigger.description")
    _threshold_source(trigger["threshold_source"], "trigger.threshold_source")
    _text_list(trigger["conditions"], "trigger.conditions", allowed=ALLOWED_CONDITIONS)

    detection = _mapping(
        data["detection"],
        "detection",
        {"mode", "conditions", "thresholds"},
    )
    mode = _text(detection["mode"], "detection.mode")
    if mode not in ALLOWED_DETECTION_MODES:
        raise ValueError("detection.mode must be observed_trigger or compatible_seed")
    if mode != EXPECTED_MODE_BY_FEASIBILITY[feasibility]:
        raise ValueError(f"detection.mode {mode} does not match feasibility {feasibility}")
    _text_list(detection["conditions"], "detection.conditions", allowed=ALLOWED_CONDITIONS)
    thresholds = detection["thresholds"]
    if not isinstance(thresholds, Mapping) or not thresholds:
        raise ValueError("detection.thresholds must be a non-empty mapping")
    for name, threshold in thresholds.items():
        if name not in ALLOWED_DETECTION_THRESHOLDS:
            raise ValueError(f"detection.thresholds has an unknown name: {name}")
        item = _mapping(
            threshold,
            f"detection.thresholds.{name}",
            {"value", "source"},
        )
        _number(item["value"], f"detection.thresholds.{name}.value", nonnegative=True)
        _threshold_source(item["source"], f"detection.thresholds.{name}.source")

    actors = _mapping(
        data["actors"],
        "actors",
        {"initiator_types", "responder_types", "generated_roles"},
    )
    _text_list(
        actors["initiator_types"],
        "actors.initiator_types",
        allowed=ALLOWED_ACTOR_TYPES,
    )
    _text_list(
        actors["responder_types"],
        "actors.responder_types",
        allowed=ALLOWED_ACTOR_TYPES,
    )
    roles = _text_list(actors["generated_roles"], "actors.generated_roles")
    if len(roles) < 2:
        raise ValueError("actors.generated_roles must contain at least two roles")

    parameters = data["parameters"]
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("parameters must be a non-empty mapping")
    for name, parameter in parameters.items():
        _validate_parameter(name, parameter)

    constraints = data["constraints"]
    if not isinstance(constraints, Mapping) or not constraints:
        raise ValueError("constraints must be a non-empty mapping")
    unknown_constraints = set(constraints) - ALLOWED_CONSTRAINTS
    if unknown_constraints:
        raise ValueError(f"constraints has unknown fields: {sorted(unknown_constraints)}")
    for name, value in constraints.items():
        if not isinstance(value, bool):
            raise ValueError(f"constraint {name} must be boolean")

    risk = _mapping(
        data["risk_definition"],
        "risk_definition",
        {"metric", "target_range", "source", "direction"},
    )
    metric = _text(risk["metric"], "risk_definition.metric")
    if metric not in ALLOWED_RISK_METRICS:
        raise ValueError(f"risk_definition.metric has an unknown value: {metric}")
    _range(risk["target_range"], "risk_definition.target_range")
    _threshold_source(risk["source"], "risk_definition.source")
    direction = _text(risk["direction"], "risk_definition.direction")
    if direction not in ALLOWED_RISK_DIRECTIONS:
        raise ValueError("risk_definition.direction must describe how risk changes")


def load_skill(path: str | Path) -> SkillSpec:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("skill YAML must contain one mapping")
    validate_skill_dict(data)
    return SkillSpec.from_dict(data)

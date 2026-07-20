"""Explicit mapping from implemented rules to reusable detection strategies.

Formal membership is defined by ``configs/skills/catalog.yaml``. This registry
also keeps executable mappings for candidate rules so they remain testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


SHARED_CAPABILITIES = frozenset(
    {
        "actor_type_filter",
        "boundary_crossing",
        "conflict_point",
        "crosswalk_relation",
        "drivable_area_relation",
        "heading_alignment",
        "intersection_relation",
        "kinematics",
        "lane_assignment",
        "lane_change",
        "lane_topology",
        "longitudinal_order",
        "minimum_distance",
        "multi_actor_relation",
        "post_encroachment_time",
        "stationary_detection",
        "time_headway",
        "time_to_collision",
        "track_history",
        "turn_classification",
    }
)

DETECTION_STRATEGIES = frozenset(
    {
        "blockage_avoidance",
        "conflict_point_pair",
        "cut_in_then_brake",
        "diverge_crossing",
        "intersection_occupancy",
        "lane_change_gap",
        "lane_change_pair",
        "longitudinal_pair",
        "longitudinal_triple",
        "merge_pair",
        "merge_triple",
        "simultaneous_lane_change",
        "static_blockage",
        "stopped_reentry",
        "three_vehicle_reveal",
        "vru_vehicle_conflict",
        "wrong_way_pair",
    }
)


@dataclass(frozen=True)
class SkillDetectionRule:
    """Dispatch metadata for one implemented rule.

    ``strategy`` names the reusable detector family. ``required_capabilities``
    lists the shared computations that family must provide, while
    ``primary_actor_count`` follows the skill YAML's generated role count.
    """

    skill_id: str
    strategy: str
    required_capabilities: tuple[str, ...]
    primary_actor_count: int


def _rule(
    skill_id: str,
    strategy: str,
    required_capabilities: tuple[str, ...],
    primary_actor_count: int = 2,
) -> SkillDetectionRule:
    if strategy not in DETECTION_STRATEGIES:
        raise ValueError(f"unknown detection strategy: {strategy}")
    if not required_capabilities:
        raise ValueError(f"{skill_id} needs at least one shared capability")
    unknown = set(required_capabilities) - SHARED_CAPABILITIES
    if unknown:
        raise ValueError(f"{skill_id} has unknown shared capabilities: {sorted(unknown)}")
    if primary_actor_count < 1:
        raise ValueError(f"{skill_id} primary_actor_count must be positive")
    return SkillDetectionRule(
        skill_id=skill_id,
        strategy=strategy,
        required_capabilities=required_capabilities,
        primary_actor_count=primary_actor_count,
    )


_SKILL_RULES = {
    # Longitudinal vehicle interaction.
    rule.skill_id: rule
    for rule in (
        _rule(
            "lead_hard_brake",
            "longitudinal_pair",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "time_to_collision",
            ),
        ),
        _rule(
            "lead_sudden_stop",
            "longitudinal_pair",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "stationary_detection",
                "time_to_collision",
            ),
        ),
        _rule(
            "slow_lead_blockage",
            "longitudinal_pair",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "time_headway",
            ),
        ),
        _rule(
            "short_headway_following",
            "longitudinal_pair",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "time_headway",
            ),
        ),
        _rule(
            "rear_vehicle_rapid_approach",
            "longitudinal_pair",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "time_to_collision",
            ),
        ),
        _rule(
            "chain_braking",
            "longitudinal_triple",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "multi_actor_relation",
                "time_to_collision",
            ),
            primary_actor_count=3,
        ),
        # Lane-change interaction.
        _rule(
            "adjacent_vehicle_cut_in",
            "lane_change_pair",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "time_to_collision",
            ),
        ),
        _rule(
            "cut_out_reveals_slow_vehicle",
            "three_vehicle_reveal",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "kinematics",
                "multi_actor_relation",
            ),
            primary_actor_count=3,
        ),
        _rule(
            "narrow_gap_lane_change",
            "lane_change_gap",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "minimum_distance",
                "time_to_collision",
            ),
            primary_actor_count=3,
        ),
        _rule(
            "simultaneous_lane_change_conflict",
            "simultaneous_lane_change",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "minimum_distance",
                "multi_actor_relation",
            ),
        ),
        _rule(
            "forced_lane_change_around_blockage",
            "blockage_avoidance",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "drivable_area_relation",
                "stationary_detection",
            ),
        ),
        _rule(
            "late_lane_change_before_diverge",
            "diverge_crossing",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "minimum_distance",
            ),
        ),
        # Merge and lane-topology interaction.
        _rule(
            "ramp_merge_small_gap",
            "merge_pair",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "lane_drop_merge_competition",
            "merge_pair",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "conflict_point",
                "post_encroachment_time",
                "multi_actor_relation",
            ),
        ),
        _rule(
            "merge_without_yield",
            "merge_pair",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "conflict_point",
                "post_encroachment_time",
                "time_to_collision",
            ),
        ),
        _rule(
            "diverge_lane_crossing_conflict",
            "diverge_crossing",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "conflict_point",
                "time_to_collision",
            ),
        ),
        _rule(
            "bike_lane_vehicle_merge_conflict",
            "merge_pair",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "lane_topology",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "zipper_merge_multi_vehicle",
            "merge_triple",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "longitudinal_order",
                "multi_actor_relation",
                "post_encroachment_time",
            ),
            primary_actor_count=3,
        ),
        # Intersection vehicle interaction.
        _rule(
            "unprotected_left_turn_conflict",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "turn_classification",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "right_turn_vehicle_conflict",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "turn_classification",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "crossing_path_conflict",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "intersection_creep_conflict",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "kinematics",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "intersection_blocking_vehicle",
            "intersection_occupancy",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "stationary_detection",
                "conflict_point",
                "minimum_distance",
            ),
        ),
        _rule(
            "mutual_yield_deadlock",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "kinematics",
                "conflict_point",
                "multi_actor_relation",
            ),
        ),
        # Vulnerable road-user interaction.
        _rule(
            "crosswalk_pedestrian_crossing",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "crosswalk_relation",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "jaywalking_pedestrian_crossing",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "crosswalk_relation",
                "drivable_area_relation",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "roadside_pedestrian_emergence",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "drivable_area_relation",
                "boundary_crossing",
                "conflict_point",
                "time_to_collision",
            ),
        ),
        _rule(
            "cyclist_crossing",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "turning_vehicle_crosswalk_conflict",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "crosswalk_relation",
                "turn_classification",
                "conflict_point",
                "post_encroachment_time",
            ),
        ),
        _rule(
            "group_pedestrian_crossing",
            "vru_vehicle_conflict",
            (
                "track_history",
                "actor_type_filter",
                "conflict_point",
                "post_encroachment_time",
                "multi_actor_relation",
            ),
            primary_actor_count=3,
        ),
        _rule(
            "cyclist_vehicle_merge",
            "lane_change_gap",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "conflict_point",
                "post_encroachment_time",
            ),
            primary_actor_count=3,
        ),
        # Atypical, obstruction, and composite interaction.
        _rule(
            "wrong_way_vehicle",
            "wrong_way_pair",
            (
                "track_history",
                "lane_assignment",
                "heading_alignment",
                "time_to_collision",
            ),
        ),
        _rule(
            "stopped_vehicle_reentry",
            "stopped_reentry",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "longitudinal_order",
                "multi_actor_relation",
                "stationary_detection",
                "kinematics",
                "minimum_distance",
                "time_to_collision",
            ),
            primary_actor_count=3,
        ),
        _rule(
            "construction_object_lane_blockage",
            "static_blockage",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "drivable_area_relation",
                "stationary_detection",
                "minimum_distance",
            ),
        ),
        _rule(
            "static_object_avoidance",
            "static_blockage",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "drivable_area_relation",
                "stationary_detection",
                "minimum_distance",
            ),
        ),
        _rule(
            "cut_in_then_brake",
            "cut_in_then_brake",
            (
                "track_history",
                "lane_assignment",
                "lane_topology",
                "lane_change",
                "longitudinal_order",
                "kinematics",
                "time_to_collision",
            ),
        ),
        _rule(
            "abrupt_u_turn_conflict",
            "conflict_point_pair",
            (
                "track_history",
                "lane_assignment",
                "intersection_relation",
                "heading_alignment",
                "conflict_point",
                "time_to_collision",
            ),
        ),
        _rule(
            "multi_vehicle_gap_squeeze",
            "longitudinal_triple",
            (
                "track_history",
                "lane_assignment",
                "longitudinal_order",
                "kinematics",
                "multi_actor_relation",
                "minimum_distance",
            ),
            primary_actor_count=3,
        ),
        _rule(
            "motorcyclist_filtering_conflict",
            "lane_change_gap",
            (
                "track_history",
                "actor_type_filter",
                "lane_assignment",
                "minimum_distance",
                "multi_actor_relation",
            ),
            primary_actor_count=3,
        ),
    )
}

SKILL_RULES: Mapping[str, SkillDetectionRule] = MappingProxyType(_SKILL_RULES)


def get_skill_detection_rule(skill_id: str) -> SkillDetectionRule:
    """Return dispatch metadata for ``skill_id`` with a useful unknown-ID error."""

    try:
        return SKILL_RULES[skill_id]
    except KeyError:
        raise KeyError(f"unknown skill_id: {skill_id}") from None


__all__ = [
    "DETECTION_STRATEGIES",
    "SHARED_CAPABILITIES",
    "SKILL_RULES",
    "SkillDetectionRule",
    "get_skill_detection_rule",
]

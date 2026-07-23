from __future__ import annotations

import pytest

from skilldrive.generation.repair_smoke import (
    summarize_repair_smoke_kinematics,
)


def _evidence_rows() -> list[dict[str, object]]:
    rows = []
    for arm_index, arm in enumerate(
        ("learned_conditioned", "learned_none_control", "rule_guided_none")
    ):
        for candidate_index in range(8):
            if arm == "learned_conditioned":
                passed = candidate_index % 2 == 0
            elif arm == "learned_none_control":
                passed = candidate_index % 3 == 0
            else:
                passed = True
            rows.append(
                {
                    "candidate_id": f"{arm_index + 1:x}" * 64,
                    "task_id": f"{arm_index + 4:x}" * 64,
                    "scenario_id": f"scene-{arm_index}",
                    "skill_id": (
                        "construction_object_lane_blockage"
                        if arm == "rule_guided_none"
                        else "slow_lead_blockage"
                    ),
                    "evaluation_arm": arm,
                    "candidate_index": candidate_index,
                    "latent_seed": (
                        candidate_index
                        if arm != "rule_guided_none"
                        else 100 + candidate_index
                    ),
                    "latent_group_id": (
                        "a" * 64 if arm != "rule_guided_none" else "b" * 64
                    ),
                    "kinematics": {
                        "passed": passed,
                        "rejection_reasons": (
                            []
                            if passed
                            else ["kinematics.jerk_limit_exceeded"]
                        ),
                        "metrics": {
                            "seam_speed_mps": float(candidate_index + arm_index),
                            "maximum_speed_mps": float(candidate_index + 2),
                            "maximum_acceleration_mps2": float(candidate_index + 3),
                            "maximum_deceleration_mps2": float(candidate_index + 4),
                            "maximum_jerk_mps3": float(candidate_index + 5),
                            "maximum_curvature_per_m": 0.1,
                            "maximum_heading_rate_rad_s": 0.2,
                        },
                    },
                }
            )
    return rows


def test_repair_smoke_summary_reports_fixed_arms_and_paired_outcomes() -> None:
    summary = summarize_repair_smoke_kinematics(_evidence_rows())

    by_arm = {row["evaluation_arm"]: row for row in summary["by_arm"]}
    assert set(by_arm) == {
        "learned_conditioned",
        "learned_none_control",
        "rule_guided_none",
    }
    assert all(row["candidate_count"] == 8 for row in by_arm.values())
    assert summary["paired_control"]["pair_count"] == 8
    assert summary["paired_control"]["shared_epsilon_pair_count"] == 8
    assert summary["paired_control"]["outcomes"] == {
        "both_passed": 2,
        "conditioned_only_passed": 2,
        "control_only_passed": 1,
        "neither_passed": 3,
    }
    assert (
        summary["paired_control"]["metric_delta_conditioned_minus_control"][
            "seam_speed_mps"
        ]["median"]
        == pytest.approx(-1.0)
    )


def test_repair_smoke_summary_rejects_unpaired_epsilon_seed() -> None:
    rows = _evidence_rows()
    control = next(
        row
        for row in rows
        if row["evaluation_arm"] == "learned_none_control"
        and row["candidate_index"] == 4
    )
    control["latent_seed"] = 999

    with pytest.raises(ValueError, match="share epsilon"):
        summarize_repair_smoke_kinematics(rows)

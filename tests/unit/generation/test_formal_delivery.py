from skilldrive.generation.formal_delivery import _select_balanced


def _accepted(skill: str, scenario: str, index: int, score: float) -> dict:
    return {
        "candidate_id": f"{skill}-{scenario}-{index}",
        "candidate_index": index,
        "filter_evaluation_id": f"filter-{skill}-{scenario}-{index}",
        "task_id": f"task-{skill}-{scenario}",
        "metrics": {
            "skill_id": skill,
            "scenario_id": scenario,
            "quality_score": score,
        },
    }


def test_balanced_selection_caps_skill_and_scenario_counts() -> None:
    rows = [
        _accepted("skill-a", "scene-1", index, 1.0 - index / 10.0)
        for index in range(5)
    ] + [
        _accepted("skill-a", "scene-2", 0, 0.4),
        _accepted("skill-b", "scene-3", 0, 0.8),
    ]

    selected = _select_balanced(rows, max_per_skill=4)

    skill_a = [row for row in selected if row["skill_id"] == "skill-a"]
    assert len(skill_a) == 4
    assert sum(row["scenario_id"] == "scene-1" for row in skill_a) == 3
    assert any(row["scenario_id"] == "scene-2" for row in skill_a)
    assert [row for row in selected if row["skill_id"] == "skill-b"]

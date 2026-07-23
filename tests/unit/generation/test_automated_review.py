from skilldrive.generation.automated_review import _stage_result


def test_stage_result_uses_filter_evidence_and_short_circuit_boundary() -> None:
    row = {
        "metrics": {
            "stage_evidence": [
                {"stage": "history_invariants", "passed": True},
                {"stage": "kinematics", "passed": False},
            ]
        }
    }

    assert _stage_result(row, "history_invariants") == "pass"
    assert _stage_result(row, "kinematics") == "fail"
    assert _stage_result(row, "map") == "not_applicable"

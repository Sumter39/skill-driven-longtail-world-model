import torch

from skilldrive.prediction.evaluation import (
    paired_bootstrap_delta,
    per_sample_prediction_errors,
    summarize_prediction_rows,
)


def test_per_sample_errors_choose_best_mode_and_respect_mask():
    target = torch.zeros(2, 3, 2)
    predictions = torch.ones(2, 2, 3, 2)
    predictions[:, 1] = 0.0
    mask = torch.tensor([[True, True, False], [True, False, False]])
    errors = per_sample_prediction_errors(predictions, target, mask)
    assert errors["min_ade"].tolist() == [0.0, 0.0]
    assert errors["min_fde"].tolist() == [0.0, 0.0]
    assert errors["miss"].tolist() == [0.0, 0.0]


def test_paired_bootstrap_groups_multiple_labels_by_scenario():
    baseline = [
        {"scenario_id": "a", "min_ade": 1.0, "min_fde": 2.0, "miss": 1.0},
        {"scenario_id": "a", "min_ade": 3.0, "min_fde": 4.0, "miss": 1.0},
        {"scenario_id": "b", "min_ade": 2.0, "min_fde": 2.0, "miss": 0.0},
    ]
    candidate = [
        {"scenario_id": "a", "min_ade": 0.0, "min_fde": 1.0, "miss": 0.0},
        {"scenario_id": "a", "min_ade": 2.0, "min_fde": 3.0, "miss": 0.0},
        {"scenario_id": "b", "min_ade": 1.0, "min_fde": 1.0, "miss": 0.0},
    ]
    result = paired_bootstrap_delta(baseline, candidate, repetitions=100, seed=1)
    assert result["scenario_count"] == 2
    assert result["metrics"]["minADE"]["mean_delta"] == -1.0
    assert summarize_prediction_rows(baseline)["sample_count"] == 3

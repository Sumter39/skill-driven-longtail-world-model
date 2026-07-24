import pytest
import torch

from skilldrive.prediction.metrics import multimodal_prediction_sums


def test_metrics_use_best_modes_and_last_valid_time():
    target = torch.zeros(1, 3, 2)
    predictions = torch.tensor(
        [[
            [[1.0, 0.0], [9.0, 0.0], [3.0, 0.0]],
            [[2.0, 0.0], [9.0, 0.0], [1.0, 0.0]],
        ]]
    )
    mask = torch.tensor([[True, False, True]])
    result = multimodal_prediction_sums(predictions, target, mask)
    assert result.min_ade == pytest.approx(1.5)
    assert result.min_fde == pytest.approx(1.0)
    assert result.miss_rate == 0.0

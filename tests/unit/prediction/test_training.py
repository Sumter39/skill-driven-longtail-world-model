import torch

from skilldrive.prediction.model import PredictionOutput
from skilldrive.prediction.training import LocalityInterleaveSampler, prediction_loss


class _Real:
    entries = [
        {"shard": "a"},
        {"shard": "a"},
        {"shard": "b"},
        {"shard": "b"},
    ]


def test_prediction_loss_selects_lowest_error_mode():
    trajectories = torch.ones(2, 3, 4, 2)
    trajectories[:, 1] = 0.0
    output = PredictionOutput(trajectories, torch.zeros(2, 3))
    loss = prediction_loss(
        output, torch.zeros(2, 4, 2), torch.ones(2, 4, dtype=torch.bool)
    )
    assert loss.winning_modes.tolist() == [1, 1]
    assert loss.regression.item() == 0.0
    assert torch.isfinite(loss.total)


def test_locality_sampler_is_deterministic_and_complete():
    sampler = LocalityInterleaveSampler(_Real(), 2, seed=3, epoch=1)
    first = list(sampler)
    second = list(LocalityInterleaveSampler(_Real(), 2, seed=3, epoch=1))
    assert first == second
    assert sorted(first) == list(range(6))
    sampler.set_epoch(2)
    assert sorted(sampler) == list(range(6))
    assert list(sampler) != first

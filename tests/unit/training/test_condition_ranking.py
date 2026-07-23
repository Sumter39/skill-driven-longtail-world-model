from __future__ import annotations

import pytest
import torch
from torch import nn

from skilldrive.models import CVAEOutput
from skilldrive.training.condition_ranking import (
    make_none_condition_batch,
    observed_condition_prior_ranking_loss,
)


class _NonePrior(nn.Module):
    def __init__(self, mean: float = 1.0) -> None:
        super().__init__()
        self.none_mean = nn.Parameter(torch.tensor(mean))
        self.last_batch = None

    def prior_parameters(self, batch):
        self.last_batch = batch
        batch_size = batch["skill_id"].shape[0]
        mean = self.none_mean.expand(batch_size, 2)
        return mean, torch.zeros_like(mean)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "actor_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "actor_role_id": torch.tensor([[4, 5, 0], [2, 1, 0]]),
        "target_actor_index": torch.tensor([1, 0]),
        "skill_id": torch.tensor([3, 0]),
        "skill_supervision_mask": torch.tensor([True, False]),
        "skill_parameters": torch.tensor([[2.0, 3.0], [0.0, 0.0]]),
        "parameter_mask": torch.tensor([[True, True], [False, False]]),
    }


def _output(
    correct_mean: torch.Tensor,
    *,
    posterior_mean: torch.Tensor | None = None,
    posterior_logvar: torch.Tensor | None = None,
) -> CVAEOutput:
    prior_mean = torch.stack((correct_mean.expand(2), torch.zeros(2)))
    zeros = torch.zeros_like(prior_mean)
    posterior_mean_value = (
        zeros
        if posterior_mean is None
        else torch.stack((posterior_mean.expand(2), torch.zeros(2)))
    )
    posterior_logvar_value = (
        zeros
        if posterior_logvar is None
        else torch.stack((posterior_logvar.expand(2), torch.zeros(2)))
    )
    future = torch.zeros(2, 2, 2)
    return CVAEOutput(
        future_delta=future,
        future_position_local=future,
        prior_mean=prior_mean,
        prior_logvar=zeros,
        posterior_mean=posterior_mean_value,
        posterior_logvar=posterior_logvar_value,
        latent=zeros,
    )


def test_none_condition_replaces_only_condition_fields_with_base_vocabulary() -> None:
    original = _batch()
    converted = make_none_condition_batch(original)

    assert converted["skill_id"].tolist() == [0, 0]
    assert converted["skill_supervision_mask"].tolist() == [False, False]
    assert converted["actor_role_id"].tolist() == [[1, 2, 0], [2, 1, 0]]
    assert not converted["parameter_mask"].any()
    assert torch.equal(converted["skill_parameters"], torch.zeros(2, 2))
    assert torch.equal(converted["actor_mask"], original["actor_mask"])
    assert converted["actor_role_id"] is not original["actor_role_id"]


def test_observed_ranking_anchors_q_and_updates_both_prior_branches() -> None:
    model = _NonePrior(mean=1.0)
    correct_mean = nn.Parameter(torch.tensor(2.0))
    posterior_mean = nn.Parameter(torch.tensor(0.0))
    posterior_logvar = nn.Parameter(torch.tensor(0.0))
    result = observed_condition_prior_ranking_loss(
        model,
        _output(
            correct_mean,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
        ),
        _batch(),
        margin_per_latent_dim=0.25,
    )

    assert result.observed_count == 1
    assert result.correct_kl_sum.item() == pytest.approx(2.0)
    assert result.none_kl_sum.item() == pytest.approx(0.5)
    assert result.mean.item() == pytest.approx(1.75)
    result.mean.backward()
    assert correct_mean.grad is not None
    assert correct_mean.grad.item() == pytest.approx(2.0)
    assert model.none_mean.grad is not None
    assert model.none_mean.grad.item() == pytest.approx(-1.0)
    assert posterior_mean.grad is None
    assert posterior_logvar.grad is None
    assert model.last_batch["skill_id"].tolist() == [0]


def test_observed_ranking_sends_only_noncontiguous_observed_rows_to_none_prior() -> None:
    first = _batch()
    batch = {name: torch.cat((value, value), dim=0) for name, value in first.items()}
    batch["skill_id"] = torch.tensor([0, 3, 0, 4])
    batch["skill_supervision_mask"] = torch.tensor([False, True, False, True])
    batch["actor_role_id"] = torch.tensor(
        [[8, 2, 0], [4, 5, 0], [9, 2, 0], [6, 7, 0]]
    )
    batch["target_actor_index"] = torch.tensor([1, 1, 1, 1])
    zeros = torch.zeros(4, 2)
    future = torch.zeros(4, 2, 2)
    output = CVAEOutput(
        future_delta=future,
        future_position_local=future,
        prior_mean=zeros,
        prior_logvar=zeros,
        posterior_mean=zeros,
        posterior_logvar=zeros,
        latent=zeros,
    )
    model = _NonePrior()

    result = observed_condition_prior_ranking_loss(
        model,
        output,
        batch,
        margin_per_latent_dim=0.1,
    )

    assert result.observed_count == 2
    assert model.last_batch["skill_id"].tolist() == [0, 0]
    assert model.last_batch["actor_role_id"].tolist() == [[1, 2, 0], [1, 2, 0]]
    assert model.last_batch["actor_mask"].shape[0] == 2


def test_ranking_without_observed_rows_does_not_evaluate_none_prior() -> None:
    batch = _batch()
    batch["skill_id"].zero_()
    batch["skill_supervision_mask"].zero_()
    model = _NonePrior()

    result = observed_condition_prior_ranking_loss(
        model,
        _output(torch.tensor(0.0)),
        batch,
        margin_per_latent_dim=0.1,
    )

    assert result.observed_count == 0
    assert result.loss_sum.item() == 0.0
    assert model.last_batch is None


def test_compatible_seed_condition_is_rejected_before_none_prior_evaluation() -> None:
    batch = _batch()
    batch["skill_id"][1] = 7
    model = _NonePrior()
    with pytest.raises(ValueError, match="compatible-seed"):
        observed_condition_prior_ranking_loss(
            model,
            _output(torch.tensor(0.0)),
            batch,
            margin_per_latent_dim=0.1,
        )
    assert model.last_batch is None

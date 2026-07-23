"""Observed-condition prior ranking for the repaired conditional CVAE."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from skilldrive.models import CVAEOutput


class PriorParameterModel(Protocol):
    def prior_parameters(
        self,
        context_batch: Mapping[str, Tensor],
    ) -> tuple[Tensor, Tensor]: ...


@dataclass(frozen=True)
class ConditionRankingLossSum:
    loss_sum: Tensor
    observed_count: int
    correct_kl_sum: Tensor
    none_kl_sum: Tensor

    @property
    def mean(self) -> Tensor:
        if self.observed_count == 0:
            return self.loss_sum
        return self.loss_sum / self.observed_count


def _required_tensor(batch: Mapping[str, Any], name: str) -> Tensor:
    try:
        value = batch[name]
    except KeyError:
        raise KeyError(f"batch is missing required tensor: {name}") from None
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
    return value


def _select_tensor_rows(
    batch: Mapping[str, Any],
    selection: Tensor,
) -> dict[str, Tensor]:
    """Select batch-aligned tensors while dropping unused non-tensor metadata."""

    batch_size = selection.shape[0]
    return {
        name: (
            value[selection]
            if value.ndim > 0 and value.shape[0] == batch_size
            else value
        )
        for name, value in batch.items()
        if isinstance(value, Tensor)
    }


def make_none_condition_batch(
    batch: Mapping[str, Any],
    *,
    none_skill_index: int = 0,
    padding_role_index: int = 0,
    context_role_index: int = 1,
    target_role_index: int = 2,
) -> dict[str, Any]:
    """Replace only the skill, generated roles, and parameters with base context."""

    actor_mask = _required_tensor(batch, "actor_mask")
    actor_role_id = _required_tensor(batch, "actor_role_id")
    target_actor_index = _required_tensor(batch, "target_actor_index")
    skill_id = _required_tensor(batch, "skill_id")
    skill_parameters = _required_tensor(batch, "skill_parameters")
    parameter_mask = _required_tensor(batch, "parameter_mask")
    supervision = _required_tensor(batch, "skill_supervision_mask")
    if actor_mask.ndim != 2 or actor_mask.dtype is not torch.bool:
        raise ValueError("actor_mask must be a boolean [B, A] tensor")
    if actor_role_id.shape != actor_mask.shape:
        raise ValueError("actor_role_id must match actor_mask")
    batch_size, actor_count = actor_mask.shape
    if target_actor_index.shape != (batch_size,):
        raise ValueError("target_actor_index must have shape [B]")
    target = target_actor_index.to(dtype=torch.long)
    if bool(((target < 0) | (target >= actor_count)).any()):
        raise ValueError("target_actor_index is outside the actor dimension")
    rows = torch.arange(batch_size, device=actor_mask.device)
    if not bool(actor_mask[rows, target].all()):
        raise ValueError("target_actor_index must reference a valid actor")
    if skill_id.shape != (batch_size,) or supervision.shape != (batch_size,):
        raise ValueError("skill tensors must have shape [B]")
    if supervision.dtype is not torch.bool:
        raise ValueError("skill_supervision_mask must have boolean dtype")
    if skill_parameters.ndim != 2 or skill_parameters.shape[0] != batch_size:
        raise ValueError("skill_parameters must have shape [B, P]")
    if parameter_mask.shape != skill_parameters.shape or parameter_mask.dtype is not torch.bool:
        raise ValueError("parameter_mask must match skill_parameters and be boolean")

    roles = torch.full_like(actor_role_id, padding_role_index)
    roles = torch.where(
        actor_mask,
        torch.full_like(roles, context_role_index),
        roles,
    )
    roles[rows, target] = target_role_index
    result = dict(batch)
    result.update(
        actor_role_id=roles,
        skill_id=torch.full_like(skill_id, none_skill_index),
        skill_supervision_mask=torch.zeros_like(supervision),
        skill_parameters=torch.zeros_like(skill_parameters),
        parameter_mask=torch.zeros_like(parameter_mask),
    )
    return result


def _normalized_kl_per_sample(
    posterior_mean: Tensor,
    posterior_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    tensors = (posterior_mean, posterior_logvar, prior_mean, prior_logvar)
    if posterior_mean.ndim != 2 or any(value.shape != posterior_mean.shape for value in tensors[1:]):
        raise ValueError("posterior and prior parameters must share shape [B, latent_dim]")
    if posterior_mean.shape[1] <= 0:
        raise ValueError("latent_dim must be positive")
    value = 0.5 * (
        prior_logvar
        - posterior_logvar
        + (
            posterior_logvar.exp()
            + (posterior_mean - prior_mean).square()
        )
        / prior_logvar.exp()
        - 1.0
    )
    return value.mean(dim=-1)


def observed_condition_prior_ranking_loss(
    model: PriorParameterModel,
    output: CVAEOutput,
    batch: Mapping[str, Any],
    *,
    margin_per_latent_dim: float,
    none_skill_index: int = 0,
    padding_role_index: int = 0,
    context_role_index: int = 1,
    target_role_index: int = 2,
) -> ConditionRankingLossSum:
    """Rank the correct prior closer to a fixed posterior anchor than the none prior.

    The posterior ``q`` is evidence supplied by the reconstruction path and does
    not receive ranking gradients.  While the hinge is active, both prior
    branches remain trainable: the correct prior is pulled toward ``q`` and the
    ``<none>`` prior acts as a negative branch that is pushed relatively away.
    """

    if (
        isinstance(margin_per_latent_dim, bool)
        or not isinstance(margin_per_latent_dim, (int, float))
        or not math.isfinite(float(margin_per_latent_dim))
        or margin_per_latent_dim <= 0.0
    ):
        raise ValueError("margin_per_latent_dim must be a positive finite number")
    if output.posterior_mean is None or output.posterior_logvar is None:
        raise ValueError("condition ranking requires posterior Gaussian parameters")
    supervision = _required_tensor(batch, "skill_supervision_mask")
    skill_id = _required_tensor(batch, "skill_id")
    if supervision.shape != (output.prior_mean.shape[0],) or supervision.dtype is not torch.bool:
        raise ValueError("skill_supervision_mask must be boolean with shape [B]")
    if skill_id.shape != supervision.shape:
        raise ValueError("skill_id must have shape [B]")
    if bool((skill_id[supervision] == none_skill_index).any()):
        raise ValueError("observed supervision cannot use the none skill")
    if bool((skill_id[~supervision] != none_skill_index).any()):
        raise ValueError(
            "unsupervised compatible-seed conditions cannot enter ranking batches"
        )
    observed_count = int(supervision.sum().item())
    if observed_count == 0:
        zero = output.prior_mean.sum() * 0.0
        return ConditionRankingLossSum(zero, 0, zero, zero)

    observed_batch = _select_tensor_rows(batch, supervision)
    none_batch = make_none_condition_batch(
        observed_batch,
        none_skill_index=none_skill_index,
        padding_role_index=padding_role_index,
        context_role_index=context_role_index,
        target_role_index=target_role_index,
    )
    none_mean, none_logvar = model.prior_parameters(none_batch)
    posterior_mean = output.posterior_mean.detach()[supervision]
    posterior_logvar = output.posterior_logvar.detach()[supervision]
    correct_kl = _normalized_kl_per_sample(
        posterior_mean,
        posterior_logvar,
        output.prior_mean[supervision],
        output.prior_logvar[supervision],
    )
    none_kl = _normalized_kl_per_sample(
        posterior_mean,
        posterior_logvar,
        none_mean,
        none_logvar,
    )
    loss_sum = F.relu(
        float(margin_per_latent_dim) + correct_kl - none_kl
    ).sum()
    values = (loss_sum, correct_kl.sum(), none_kl.sum())
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("condition prior ranking contains NaN or Inf")
    return ConditionRankingLossSum(
        loss_sum=loss_sum,
        observed_count=observed_count,
        correct_kl_sum=correct_kl.sum(),
        none_kl_sum=none_kl.sum(),
    )


__all__ = [
    "ConditionRankingLossSum",
    "PriorParameterModel",
    "make_none_condition_batch",
    "observed_condition_prior_ranking_loss",
]

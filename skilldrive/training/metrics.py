"""Masked trajectory metrics used by CVAE training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DisplacementSums:
    """Additive displacement-error statistics for exact cross-batch aggregation."""

    ade_error_sum: float
    valid_point_count: int
    fde_error_sum: float
    valid_sample_count: int

    @property
    def ade(self) -> float:
        if self.valid_point_count == 0:
            raise ValueError("ADE is undefined without valid future points")
        return self.ade_error_sum / self.valid_point_count

    @property
    def fde(self) -> float:
        if self.valid_sample_count == 0:
            raise ValueError("FDE is undefined without valid future samples")
        return self.fde_error_sum / self.valid_sample_count

    def __add__(self, other: "DisplacementSums") -> "DisplacementSums":
        if not isinstance(other, DisplacementSums):
            return NotImplemented
        return DisplacementSums(
            ade_error_sum=self.ade_error_sum + other.ade_error_sum,
            valid_point_count=self.valid_point_count + other.valid_point_count,
            fde_error_sum=self.fde_error_sum + other.fde_error_sum,
            valid_sample_count=self.valid_sample_count + other.valid_sample_count,
        )


def _validate_target(target: Tensor, mask: Tensor) -> None:
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError(f"target must have shape [B, T, 2], got {tuple(target.shape)}")
    if mask.shape != target.shape[:2]:
        raise ValueError(
            f"mask must have shape {tuple(target.shape[:2])}, got {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise ValueError("mask must have boolean dtype")
    if not bool(mask.any()):
        raise ValueError("trajectory metrics require at least one valid future point")
    if not torch.isfinite(target[mask]).all():
        raise ValueError("valid target positions must be finite")


def _last_valid_indices(mask: Tensor) -> tuple[Tensor, Tensor]:
    valid_samples = mask.any(dim=1)
    time_indices = torch.arange(mask.shape[1], device=mask.device)
    last_indices = torch.where(mask, time_indices.unsqueeze(0), -1).max(dim=1).values
    return valid_samples, last_indices


def displacement_sums(prediction: Tensor, target: Tensor, mask: Tensor) -> DisplacementSums:
    """Return additive ADE/FDE statistics for one deterministic prediction."""

    _validate_target(target, mask)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction must have shape {tuple(target.shape)}, got {tuple(prediction.shape)}"
        )
    if not torch.isfinite(prediction[mask]).all():
        raise ValueError("valid predicted positions must be finite")

    distances = torch.linalg.vector_norm(prediction - target, dim=-1)
    valid_samples, last_indices = _last_valid_indices(mask)
    sample_indices = torch.arange(target.shape[0], device=target.device)[valid_samples]
    final_distances = distances[
        sample_indices,
        last_indices[valid_samples],
    ]
    return DisplacementSums(
        ade_error_sum=float(distances[mask].sum().detach().cpu()),
        valid_point_count=int(mask.sum().item()),
        fde_error_sum=float(final_distances.sum().detach().cpu()),
        valid_sample_count=int(valid_samples.sum().item()),
    )


def multimodal_displacement_sums(
    predictions: Tensor,
    target: Tensor,
    mask: Tensor,
) -> DisplacementSums:
    """Return minADE@K and minFDE@K sums for prior samples.

    ADE and FDE select their best mode independently for each sample.
    """

    _validate_target(target, mask)
    if predictions.ndim != 4 or predictions.shape[0] != target.shape[0]:
        raise ValueError(
            "predictions must have shape [B, K, T, 2] with the same batch as target"
        )
    if predictions.shape[2:] != target.shape[1:]:
        raise ValueError(
            f"predictions must end with shape {tuple(target.shape[1:])}, "
            f"got {tuple(predictions.shape[2:])}"
        )
    if predictions.shape[1] <= 0:
        raise ValueError("predictions must contain at least one mode")

    expanded_mask = mask[:, None, :]
    valid_values = expanded_mask.expand(predictions.shape[:3])
    if not torch.isfinite(predictions[valid_values]).all():
        raise ValueError("valid predicted positions must be finite")

    distances = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    valid_counts = mask.sum(dim=1)
    valid_samples, last_indices = _last_valid_indices(mask)
    sample_indices = torch.arange(target.shape[0], device=target.device)[valid_samples]

    ade_per_mode = (distances * expanded_mask).sum(dim=-1) / valid_counts.clamp_min(1)[:, None]
    min_ade = ade_per_mode[valid_samples].min(dim=1).values

    final_per_mode = distances[
        sample_indices,
        :,
        last_indices[valid_samples],
    ]
    min_fde = final_per_mode.min(dim=1).values
    return DisplacementSums(
        ade_error_sum=float(min_ade.sum().detach().cpu()),
        valid_point_count=int(valid_samples.sum().item()),
        fde_error_sum=float(min_fde.sum().detach().cpu()),
        valid_sample_count=int(valid_samples.sum().item()),
    )


def constant_velocity_prediction(
    last_position: Tensor,
    last_velocity: Tensor,
    *,
    future_steps: int,
    sample_period_s: float,
) -> Tensor:
    """Extrapolate a constant-velocity future in the current coordinate frame."""

    if last_position.ndim != 2 or last_position.shape[-1] != 2:
        raise ValueError("last_position must have shape [B, 2]")
    if last_velocity.shape != last_position.shape:
        raise ValueError("last_velocity must have the same shape as last_position")
    if isinstance(future_steps, bool) or not isinstance(future_steps, int) or future_steps <= 0:
        raise ValueError("future_steps must be a positive integer")
    if sample_period_s <= 0:
        raise ValueError("sample_period_s must be positive")
    steps = torch.arange(
        1,
        future_steps + 1,
        dtype=last_position.dtype,
        device=last_position.device,
    )
    return last_position[:, None] + last_velocity[:, None] * (
        steps[None, :, None] * sample_period_s
    )


def gaussian_kl_divergence(
    posterior_mean: Tensor,
    posterior_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    """Mean KL divergence from a diagonal posterior to a diagonal prior."""

    tensors = (posterior_mean, posterior_logvar, prior_mean, prior_logvar)
    if any(tensor.shape != posterior_mean.shape for tensor in tensors[1:]):
        raise ValueError("all Gaussian parameter tensors must have the same shape")
    if posterior_mean.ndim != 2:
        raise ValueError("Gaussian parameters must have shape [B, latent_dim]")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("Gaussian parameters must be finite")

    posterior_variance = posterior_logvar.exp()
    prior_variance = prior_logvar.exp()
    values = 0.5 * (
        prior_logvar
        - posterior_logvar
        + (posterior_variance + (posterior_mean - prior_mean).square()) / prior_variance
        - 1.0
    )
    return values.sum(dim=-1).mean()


__all__ = [
    "DisplacementSums",
    "constant_velocity_prediction",
    "displacement_sums",
    "gaussian_kl_divergence",
    "multimodal_displacement_sums",
]

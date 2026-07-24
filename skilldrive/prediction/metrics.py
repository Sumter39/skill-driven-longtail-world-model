"""Metrics and deterministic baseline for multimodal trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PredictionMetricSums:
    ade_sum: float
    fde_sum: float
    sample_count: int
    miss_count: int

    def __add__(self, other: "PredictionMetricSums") -> "PredictionMetricSums":
        return PredictionMetricSums(
            self.ade_sum + other.ade_sum,
            self.fde_sum + other.fde_sum,
            self.sample_count + other.sample_count,
            self.miss_count + other.miss_count,
        )

    @property
    def min_ade(self) -> float:
        return self.ade_sum / self.sample_count if self.sample_count else float("nan")

    @property
    def min_fde(self) -> float:
        return self.fde_sum / self.sample_count if self.sample_count else float("nan")

    @property
    def miss_rate(self) -> float:
        return self.miss_count / self.sample_count if self.sample_count else float("nan")


def _validate(predictions: Tensor, target: Tensor, mask: Tensor) -> None:
    if predictions.ndim != 4 or predictions.shape[0] != target.shape[0]:
        raise ValueError("predictions must have shape [B, K, T, 2]")
    if target.ndim != 3 or target.shape[-1] != 2 or predictions.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes do not align")
    if mask.shape != target.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("target mask must have shape [B, T] and boolean dtype")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every metric sample needs at least one future point")
    valid = mask[:, None, :].expand_as(predictions[..., 0])
    if not bool(torch.isfinite(predictions[valid]).all()) or not bool(torch.isfinite(target[mask]).all()):
        raise ValueError("valid prediction and target values must be finite")


def multimodal_prediction_sums(
    predictions: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    miss_threshold_m: float = 2.0,
) -> PredictionMetricSums:
    """Return additive minADE@K, minFDE@K and 2m endpoint miss statistics."""

    if miss_threshold_m <= 0:
        raise ValueError("miss_threshold_m must be positive")
    _validate(predictions, target, mask)
    distances = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    counts = mask.sum(dim=1).clamp_min(1)
    ade = (distances * mask[:, None]).sum(dim=-1) / counts[:, None]
    time_indices = torch.arange(mask.shape[1], device=mask.device)
    fde_indices = torch.where(mask, time_indices[None], -1).max(dim=1).values
    rows = torch.arange(target.shape[0], device=target.device)
    fde = distances[rows[:, None], torch.arange(predictions.shape[1], device=target.device)[None, :], fde_indices[:, None]]
    min_ade = ade.min(dim=1).values
    min_fde = fde.min(dim=1).values
    return PredictionMetricSums(
        ade_sum=float(min_ade.sum().detach().cpu()),
        fde_sum=float(min_fde.sum().detach().cpu()),
        sample_count=int(target.shape[0]),
        miss_count=int((min_fde > miss_threshold_m).sum().item()),
    )


def constant_velocity_prediction(
    actor_history: Tensor,
    actor_time_mask: Tensor,
    actor_mask: Tensor,
    target_actor_index: Tensor,
    *,
    future_steps: int = 60,
    sample_period_s: float = 0.1,
) -> Tensor:
    """Extrapolate the target's last valid local velocity."""

    if actor_history.ndim != 4 or actor_history.shape[-1] < 4:
        raise ValueError("actor_history must have shape [B, A, H, F>=4]")
    if actor_time_mask.shape != actor_history.shape[:3] or actor_time_mask.dtype is not torch.bool:
        raise ValueError("actor_time_mask has an invalid shape or dtype")
    if actor_mask.shape != actor_history.shape[:2] or actor_mask.dtype is not torch.bool:
        raise ValueError("actor_mask has an invalid shape or dtype")
    rows = torch.arange(actor_history.shape[0], device=actor_history.device)
    indices = target_actor_index.to(dtype=torch.long, device=actor_history.device)
    history_mask = actor_time_mask[rows, indices] & actor_mask[rows, indices, None]
    if not bool(history_mask.any(dim=1).all()):
        raise ValueError("constant velocity requires one valid target history point")
    time_indices = torch.arange(history_mask.shape[1], device=history_mask.device)
    last = torch.where(history_mask, time_indices[None, :], -1).max(dim=1).values
    velocity = actor_history[rows, indices, last, 2:4]
    if not bool(torch.isfinite(velocity).all()):
        raise ValueError("target history velocity must be finite")
    steps = torch.arange(1, future_steps + 1, device=velocity.device, dtype=velocity.dtype)
    return velocity[:, None, None, :] * (steps[None, None, :, None] * sample_period_s)


__all__ = ["PredictionMetricSums", "constant_velocity_prediction", "multimodal_prediction_sums"]

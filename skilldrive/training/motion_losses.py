"""Masked robust motion-consistency losses for local CVAE trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


SAMPLE_PERIOD_S = 0.1


@dataclass(frozen=True)
class MaskedSmoothL1Sum:
    """One differentiable Smooth L1 sum and its scalar-element denominator."""

    loss_sum: Tensor
    element_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.loss_sum, Tensor) or self.loss_sum.ndim != 0:
            raise ValueError("loss_sum must be a scalar torch.Tensor")
        if (
            isinstance(self.element_count, bool)
            or not isinstance(self.element_count, int)
            or self.element_count < 0
        ):
            raise ValueError("element_count must be a nonnegative integer")

    @property
    def mean(self) -> Tensor:
        """Return a differentiable zero when this component has no valid elements."""

        if self.element_count == 0:
            return self.loss_sum
        return self.loss_sum / self.element_count


@dataclass(frozen=True)
class MotionLossSums:
    """Masked seam, velocity, acceleration, and jerk loss contracts."""

    seam_velocity: MaskedSmoothL1Sum
    velocity: MaskedSmoothL1Sum
    acceleration: MaskedSmoothL1Sum
    jerk: MaskedSmoothL1Sum


@dataclass(frozen=True)
class MotionLossElementCounts:
    """Scalar x/y element counts for one optimizer accumulation group."""

    seam_velocity: int
    velocity: int
    acceleration: int
    jerk: int

    def __post_init__(self) -> None:
        for name in ("seam_velocity", "velocity", "acceleration", "jerk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class _MotionBatchContext:
    target: Tensor
    future_mask: Tensor
    final_history_velocity: Tensor
    previous_history_velocity: Tensor
    final_history_mask: Tensor
    history_acceleration_mask: Tensor
    seam_mask: Tensor
    velocity_mask: Tensor
    acceleration_mask: Tensor
    jerk_mask: Tensor


def _required_tensor(batch: Mapping[str, Any], name: str) -> Tensor:
    try:
        value = batch[name]
    except KeyError:
        raise KeyError(f"batch is missing required tensor: {name}") from None
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
    return value


def _boolean_mask(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype is not torch.bool:
        raise ValueError(f"{name} must have boolean dtype")
    return value


def _positive_finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _selected_finite(values: Tensor, mask: Tensor, name: str) -> None:
    selected = values[mask]
    if selected.numel() and not bool(torch.isfinite(selected).all()):
        raise FloatingPointError(f"valid {name} values must be finite")


def _smooth_l1_sum(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    beta: float,
) -> MaskedSmoothL1Sum:
    if prediction.shape != target.shape or prediction.shape[:-1] != mask.shape:
        raise ValueError("masked Smooth L1 tensors do not align")
    element_count = int(mask.sum().item()) * prediction.shape[-1]
    if element_count == 0:
        return MaskedSmoothL1Sum(prediction.sum() * 0.0, 0)
    loss_sum = F.smooth_l1_loss(
        prediction[mask],
        target[mask],
        reduction="sum",
        beta=beta,
    )
    if not bool(torch.isfinite(loss_sum)):
        raise FloatingPointError("motion Smooth L1 loss contains NaN or Inf")
    return MaskedSmoothL1Sum(loss_sum, element_count)


def _motion_batch_context(batch: Mapping[str, Any]) -> _MotionBatchContext:
    target = _required_tensor(batch, "target_future")
    if (
        target.ndim != 3
        or target.shape[-1] != 2
        or not target.is_floating_point()
    ):
        raise ValueError("target_future must have floating shape [B, T, 2]")
    batch_size, future_steps, _ = target.shape
    if future_steps <= 0:
        raise ValueError("future trajectories must contain at least one timestep")
    future_mask = _boolean_mask(
        _required_tensor(batch, "target_future_mask"),
        (batch_size, future_steps),
        "target_future_mask",
    )
    if future_mask.device != target.device:
        raise ValueError("target_future_mask must share the target_future device")

    actor_history = _required_tensor(batch, "actor_history")
    if (
        actor_history.ndim != 4
        or actor_history.shape[0] != batch_size
        or actor_history.shape[-1] < 4
        or not actor_history.is_floating_point()
    ):
        raise ValueError(
            "actor_history must have floating shape [B, A, H, F] with F >= 4"
        )
    if actor_history.device != target.device:
        raise ValueError("actor_history must share the target_future device")
    _, actor_count, history_steps, _ = actor_history.shape
    if history_steps <= 0:
        raise ValueError("actor_history must contain at least one history timestep")
    actor_time_mask = _boolean_mask(
        _required_tensor(batch, "actor_time_mask"),
        (batch_size, actor_count, history_steps),
        "actor_time_mask",
    )
    actor_mask = _boolean_mask(
        _required_tensor(batch, "actor_mask"),
        (batch_size, actor_count),
        "actor_mask",
    )
    if actor_time_mask.device != target.device or actor_mask.device != target.device:
        raise ValueError("actor masks must share the target_future device")
    target_actor_index = _required_tensor(batch, "target_actor_index")
    if tuple(target_actor_index.shape) != (batch_size,):
        raise ValueError(f"target_actor_index must have shape {(batch_size,)}")
    if target_actor_index.dtype is torch.bool or target_actor_index.is_floating_point():
        raise ValueError("target_actor_index must have integer dtype")
    target_actor_index = target_actor_index.to(device=target.device, dtype=torch.long)
    if bool(((target_actor_index < 0) | (target_actor_index >= actor_count)).any()):
        raise ValueError("target_actor_index is outside the actor dimension")

    batch_indices = torch.arange(batch_size, device=target.device)
    target_history_mask = (
        actor_time_mask[batch_indices, target_actor_index]
        & actor_mask[batch_indices, target_actor_index].unsqueeze(-1)
    )
    target_history_velocity = actor_history[
        batch_indices,
        target_actor_index,
        :,
        2:4,
    ]
    final_history_mask = target_history_mask[:, -1]
    if history_steps >= 2:
        previous_history_velocity = target_history_velocity[:, -2]
        history_acceleration_mask = final_history_mask & target_history_mask[:, -2]
    else:
        previous_history_velocity = torch.zeros_like(target_history_velocity[:, -1])
        history_acceleration_mask = torch.zeros_like(final_history_mask)

    previous_future_mask = torch.cat(
        (
            torch.ones((batch_size, 1), dtype=torch.bool, device=target.device),
            future_mask[:, :-1],
        ),
        dim=1,
    )
    velocity_mask = future_mask & previous_future_mask
    seam_mask = velocity_mask[:, 0] & final_history_mask
    previous_velocity_mask = torch.cat(
        (final_history_mask[:, None], velocity_mask[:, :-1]),
        dim=1,
    )
    acceleration_mask = velocity_mask & previous_velocity_mask
    previous_acceleration_mask = torch.cat(
        (history_acceleration_mask[:, None], acceleration_mask[:, :-1]),
        dim=1,
    )
    jerk_mask = acceleration_mask & previous_acceleration_mask
    return _MotionBatchContext(
        target=target,
        future_mask=future_mask,
        final_history_velocity=target_history_velocity[:, -1],
        previous_history_velocity=previous_history_velocity,
        final_history_mask=final_history_mask,
        history_acceleration_mask=history_acceleration_mask,
        seam_mask=seam_mask,
        velocity_mask=velocity_mask,
        acceleration_mask=acceleration_mask,
        jerk_mask=jerk_mask,
    )


def motion_loss_element_counts(
    batch: Mapping[str, Any],
) -> MotionLossElementCounts:
    """Return motion-loss denominators without decoding or differencing trajectories."""

    context = _motion_batch_context(batch)
    return MotionLossElementCounts(
        seam_velocity=int(context.seam_mask.sum().item()) * 2,
        velocity=int(context.velocity_mask.sum().item()) * 2,
        acceleration=int(context.acceleration_mask.sum().item()) * 2,
        jerk=int(context.jerk_mask.sum().item()) * 2,
    )


def compute_motion_loss_sums(
    future_position_local: Tensor,
    batch: Mapping[str, Any],
    *,
    sample_period_s: float = SAMPLE_PERIOD_S,
    seam_velocity_beta_mps: float = 1.0,
    velocity_beta_mps: float = 1.0,
    acceleration_beta_mps2: float = 2.0,
    jerk_beta_mps3: float = 20.0,
) -> MotionLossSums:
    """Compute robust motion losses with the same 49-to-future differencing seam.

    The prediction and target are local positions with frame 49 at the origin.
    Velocity prepends the frame-49 target velocity, acceleration prepends the
    frame-48-to-49 target acceleration, and masks require every contributing
    state to be valid. Each returned count is the number of scalar x/y elements.
    """

    if not isinstance(future_position_local, Tensor):
        raise TypeError("future_position_local must be a torch.Tensor")
    if (
        future_position_local.ndim != 3
        or future_position_local.shape[-1] != 2
        or not future_position_local.is_floating_point()
    ):
        raise ValueError("future_position_local must have floating shape [B, T, 2]")
    dt = _positive_finite(sample_period_s, "sample_period_s")
    seam_beta = _positive_finite(
        seam_velocity_beta_mps,
        "seam_velocity_beta_mps",
    )
    velocity_beta = _positive_finite(velocity_beta_mps, "velocity_beta_mps")
    acceleration_beta = _positive_finite(
        acceleration_beta_mps2,
        "acceleration_beta_mps2",
    )
    jerk_beta = _positive_finite(jerk_beta_mps3, "jerk_beta_mps3")

    context = _motion_batch_context(batch)
    target = context.target
    if target.shape != future_position_local.shape:
        raise ValueError("target_future must match future_position_local floating shape")
    if target.device != future_position_local.device:
        raise ValueError("target_future and future_position_local must share a device")
    batch_size = target.shape[0]
    future_mask = context.future_mask

    calculation_dtype = (
        torch.float32
        if future_position_local.dtype in {torch.float16, torch.bfloat16}
        else future_position_local.dtype
    )
    prediction_values = future_position_local.to(dtype=calculation_dtype)
    target_values = target.to(dtype=calculation_dtype)
    final_history_velocity = context.final_history_velocity.to(
        dtype=calculation_dtype
    )
    previous_history_velocity = context.previous_history_velocity.to(
        dtype=calculation_dtype
    )
    final_history_mask = context.final_history_mask
    _selected_finite(
        final_history_velocity,
        final_history_mask,
        "target frame-49 velocity",
    )
    safe_final_velocity = torch.where(
        final_history_mask.unsqueeze(-1),
        final_history_velocity,
        torch.zeros_like(final_history_velocity),
    )

    _selected_finite(target_values, future_mask, "target_future")
    _selected_finite(prediction_values, future_mask, "predicted future")
    safe_target = torch.where(
        future_mask.unsqueeze(-1),
        target_values,
        torch.zeros_like(target_values),
    )
    safe_prediction = torch.where(
        future_mask.unsqueeze(-1),
        prediction_values,
        torch.zeros_like(prediction_values),
    )
    anchor = prediction_values.new_zeros((batch_size, 1, 2))
    prediction_velocity = torch.diff(
        torch.cat((anchor, safe_prediction), dim=1),
        dim=1,
    ) / dt
    target_velocity = torch.diff(
        torch.cat((anchor, safe_target), dim=1),
        dim=1,
    ) / dt
    seam_velocity = _smooth_l1_sum(
        prediction_velocity[:, 0],
        safe_final_velocity,
        context.seam_mask,
        beta=seam_beta,
    )
    velocity = _smooth_l1_sum(
        prediction_velocity,
        target_velocity,
        context.velocity_mask,
        beta=velocity_beta,
    )

    prediction_acceleration = torch.diff(
        torch.cat((safe_final_velocity[:, None, :], prediction_velocity), dim=1),
        dim=1,
    ) / dt
    target_acceleration = torch.diff(
        torch.cat((safe_final_velocity[:, None, :], target_velocity), dim=1),
        dim=1,
    ) / dt
    acceleration = _smooth_l1_sum(
        prediction_acceleration,
        target_acceleration,
        context.acceleration_mask,
        beta=acceleration_beta,
    )

    _selected_finite(
        previous_history_velocity,
        context.history_acceleration_mask,
        "target frame-48 velocity",
    )
    safe_previous_velocity = torch.where(
        context.history_acceleration_mask.unsqueeze(-1),
        previous_history_velocity,
        torch.zeros_like(previous_history_velocity),
    )
    history_acceleration = (
        safe_final_velocity - safe_previous_velocity
    ) / dt

    prediction_jerk = torch.diff(
        torch.cat((history_acceleration[:, None, :], prediction_acceleration), dim=1),
        dim=1,
    ) / dt
    target_jerk = torch.diff(
        torch.cat((history_acceleration[:, None, :], target_acceleration), dim=1),
        dim=1,
    ) / dt
    jerk = _smooth_l1_sum(
        prediction_jerk,
        target_jerk,
        context.jerk_mask,
        beta=jerk_beta,
    )

    return MotionLossSums(
        seam_velocity=seam_velocity,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
    )


__all__ = [
    "MaskedSmoothL1Sum",
    "MotionLossElementCounts",
    "MotionLossSums",
    "SAMPLE_PERIOD_S",
    "compute_motion_loss_sums",
    "motion_loss_element_counts",
]

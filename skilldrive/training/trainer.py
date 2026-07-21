"""In-memory training, evaluation, and stable-step benchmarking for the CVAE."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from skilldrive.models import CVAEOutput, ConditionalCVAE
from skilldrive.training.config import LossConfig, TrainingConfig
from skilldrive.training.metrics import (
    DisplacementSums,
    constant_velocity_prediction,
    displacement_sums,
    gaussian_kl_divergence,
    multimodal_displacement_sums,
)
from skilldrive.training.system_monitor import SystemMonitor


Batch = Mapping[str, Any]

_MAP_SOFT_MARGIN_M = 5.0
_COLLISION_SOFT_RADIUS_M = 2.0
_COLLISION_TIE_BREAK_M = 1e-3
_FUTURE_STEP_SECONDS = 0.1


@dataclass(frozen=True)
class LossSums:
    reconstruction_sum: float = 0.0
    reconstruction_element_count: int = 0
    endpoint_sum: float = 0.0
    endpoint_element_count: int = 0
    kl_sum: float = 0.0
    sample_count: int = 0
    valid_point_count: int = 0
    valid_sample_count: int = 0

    def __add__(self, other: "LossSums") -> "LossSums":
        if not isinstance(other, LossSums):
            return NotImplemented
        return LossSums(
            reconstruction_sum=self.reconstruction_sum + other.reconstruction_sum,
            reconstruction_element_count=(
                self.reconstruction_element_count + other.reconstruction_element_count
            ),
            endpoint_sum=self.endpoint_sum + other.endpoint_sum,
            endpoint_element_count=self.endpoint_element_count + other.endpoint_element_count,
            kl_sum=self.kl_sum + other.kl_sum,
            sample_count=self.sample_count + other.sample_count,
            valid_point_count=self.valid_point_count + other.valid_point_count,
            valid_sample_count=self.valid_sample_count + other.valid_sample_count,
        )

    @property
    def reconstruction_loss(self) -> float:
        if self.reconstruction_element_count <= 0:
            raise ValueError("reconstruction loss requires valid future elements")
        return self.reconstruction_sum / self.reconstruction_element_count

    @property
    def endpoint_loss(self) -> float:
        if self.endpoint_element_count <= 0:
            raise ValueError("endpoint loss requires valid future samples")
        return self.endpoint_sum / self.endpoint_element_count

    @property
    def kl_loss(self) -> float:
        if self.sample_count <= 0:
            raise ValueError("KL loss requires at least one sample")
        return self.kl_sum / self.sample_count


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    reconstruction: Tensor
    endpoint: Tensor
    kl: Tensor
    kl_weight: float
    sums: LossSums


@dataclass(frozen=True)
class OptimizerStepResult:
    next_global_step: int
    total_loss: float
    kl_weight: float
    gradient_norm: float
    microbatch_count: int
    sums: LossSums


@dataclass(frozen=True)
class TrainEpochResult:
    next_global_step: int
    optimizer_steps: int
    microbatch_count: int
    mean_optimizer_loss: float
    sums: LossSums


@dataclass(frozen=True)
class ValidationLossResult:
    total_loss: float
    reconstruction_loss: float
    endpoint_loss: float
    kl_loss: float
    kl_weight: float
    map_soft_loss: float
    collision_soft_loss: float
    sums: LossSums


@dataclass(frozen=True)
class EvaluationResult:
    posterior: DisplacementSums
    prior: DisplacementSums
    constant_velocity: DisplacementSums
    kl_sum: float
    sample_count: int
    validation_loss: ValidationLossResult | None = None

    @property
    def posterior_kl(self) -> float:
        if self.sample_count <= 0:
            raise ValueError("evaluation KL requires at least one sample")
        return self.kl_sum / self.sample_count


@dataclass(frozen=True)
class BenchmarkResult:
    startup_seconds: float
    warmup_seconds: float
    step_seconds: tuple[float, ...]
    data_wait_seconds: tuple[float, ...]
    p50_step_seconds: float
    p95_step_seconds: float
    samples_per_second: float
    measured_samples: int
    next_global_step: int
    cpu_metrics_available: bool
    cpu_busy_percent: float | None
    cpu_iowait_percent: float | None
    gpu_utilization_available: bool
    gpu_utilization_mean_percent: float | None
    gpu_utilization_p50_percent: float | None
    gpu_utilization_p95_percent: float | None
    gpu_utilization_sample_count: int
    monitor_overhead_seconds: float

    @property
    def data_wait_fraction(self) -> float:
        total = sum(self.step_seconds)
        return 0.0 if total <= 0 else sum(self.data_wait_seconds) / total


@dataclass(frozen=True)
class _TensorLossSums:
    reconstruction_sum: Tensor
    reconstruction_element_count: int
    endpoint_sum: Tensor
    endpoint_element_count: int
    kl_sum: Tensor
    sample_count: int
    valid_point_count: int
    valid_sample_count: int

    def detached(self) -> LossSums:
        return LossSums(
            reconstruction_sum=float(self.reconstruction_sum.detach().cpu()),
            reconstruction_element_count=self.reconstruction_element_count,
            endpoint_sum=float(self.endpoint_sum.detach().cpu()),
            endpoint_element_count=self.endpoint_element_count,
            kl_sum=float(self.kl_sum.detach().cpu()),
            sample_count=self.sample_count,
            valid_point_count=self.valid_point_count,
            valid_sample_count=self.valid_sample_count,
        )


@dataclass(frozen=True)
class _AmpPolicy:
    enabled: bool
    dtype: torch.dtype
    use_scaler: bool


def move_batch_to_device(
    batch: Batch,
    device: str | torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, Any]:
    """Move tensor values while preserving non-tensor metadata unchanged."""

    resolved = torch.device(device)
    return {
        key: value.to(device=resolved, non_blocking=non_blocking)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }


def kl_anneal_weight(global_step: int, maximum: float, warmup_steps: int) -> float:
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    if maximum < 0 or not math.isfinite(maximum):
        raise ValueError("maximum KL weight must be finite and nonnegative")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if warmup_steps == 0:
        return float(maximum)
    return float(maximum) * min(1.0, global_step / warmup_steps)


def _required_tensor(batch: Batch, name: str) -> Tensor:
    try:
        value = batch[name]
    except KeyError:
        raise KeyError(f"batch is missing required tensor: {name}") from None
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
    return value


def _last_valid_indices(mask: Tensor) -> tuple[Tensor, Tensor]:
    if mask.ndim != 2 or mask.dtype is not torch.bool:
        raise ValueError("target_future_mask must be a boolean [B, T] tensor")
    valid_samples = mask.any(dim=1)
    if not bool(valid_samples.all()):
        raise ValueError("every sample must contain at least one valid future point")
    indices = torch.arange(mask.shape[1], device=mask.device)
    last = torch.where(mask, indices.unsqueeze(0), -1).max(dim=1).values
    return valid_samples, last


def _loss_sums(output: CVAEOutput, batch: Batch) -> _TensorLossSums:
    target = _required_tensor(batch, "target_future")
    mask = _required_tensor(batch, "target_future_mask")
    if output.posterior_mean is None or output.posterior_logvar is None:
        raise ValueError("training loss requires posterior Gaussian parameters")
    if output.future_position_local.shape != target.shape:
        raise ValueError("predicted and target future shapes must match")
    if mask.shape != target.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("target_future_mask must match target_future and be boolean")
    valid_samples, last_indices = _last_valid_indices(mask)
    if not torch.isfinite(target[mask]).all():
        raise FloatingPointError("valid target_future values must be finite")
    if not torch.isfinite(output.future_position_local[mask]).all():
        raise FloatingPointError("valid predicted future values must be finite")

    reconstruction_sum = F.smooth_l1_loss(
        output.future_position_local[mask],
        target[mask],
        reduction="sum",
    )
    batch_indices = torch.arange(target.shape[0], device=target.device)[valid_samples]
    endpoint_sum = F.smooth_l1_loss(
        output.future_position_local[batch_indices, last_indices[valid_samples]],
        target[batch_indices, last_indices[valid_samples]],
        reduction="sum",
    )
    kl_mean = gaussian_kl_divergence(
        output.posterior_mean,
        output.posterior_logvar,
        output.prior_mean,
        output.prior_logvar,
    )
    values = (reconstruction_sum, endpoint_sum, kl_mean)
    if not all(bool(torch.isfinite(value)) for value in values):
        raise FloatingPointError("CVAE loss contains NaN or Inf")
    valid_points = int(mask.sum().item())
    valid_sample_count = int(valid_samples.sum().item())
    sample_count = int(target.shape[0])
    return _TensorLossSums(
        reconstruction_sum=reconstruction_sum,
        reconstruction_element_count=valid_points * target.shape[-1],
        endpoint_sum=endpoint_sum,
        endpoint_element_count=valid_sample_count * target.shape[-1],
        kl_sum=kl_mean * sample_count,
        sample_count=sample_count,
        valid_point_count=valid_points,
        valid_sample_count=valid_sample_count,
    )


def _map_soft_context(
    prediction: Tensor,
    batch: Batch,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return map points, their mask, and future points with usable map context."""

    map_polylines = _required_tensor(batch, "map_polylines")
    map_point_mask = _required_tensor(batch, "map_point_mask")
    map_polyline_mask = _required_tensor(batch, "map_polyline_mask")
    future_mask = _required_tensor(batch, "target_future_mask")
    if map_polylines.ndim != 4 or map_polylines.shape[-1] < 2:
        raise ValueError("map_polylines must have shape [B, P, Q, F] with F >= 2")
    if (
        map_point_mask.shape != map_polylines.shape[:3]
        or map_point_mask.dtype is not torch.bool
    ):
        raise ValueError("map_point_mask must match map_polylines and be boolean")
    if (
        map_polyline_mask.shape != map_polylines.shape[:2]
        or map_polyline_mask.dtype is not torch.bool
    ):
        raise ValueError("map_polyline_mask must match map_polylines and be boolean")
    if map_polylines.shape[0] != prediction.shape[0]:
        raise ValueError("map_polylines and prediction batch sizes must match")
    if future_mask.shape != prediction.shape[:2] or future_mask.dtype is not torch.bool:
        raise ValueError("target_future_mask must match prediction and be boolean")

    point_mask = map_point_mask & map_polyline_mask.unsqueeze(-1)
    eligible_future_mask = future_mask & point_mask.flatten(1).any(dim=1).unsqueeze(1)
    return map_polylines, point_mask, eligible_future_mask


def _map_soft_penalty_sum(prediction: Tensor, batch: Batch) -> tuple[Tensor, int]:
    """Return map-support penalty sum and the number of evaluable future points."""

    map_polylines, point_mask, eligible_future_mask = _map_soft_context(
        prediction,
        batch,
    )
    valid_map_values = map_polylines[..., :2][point_mask]
    if valid_map_values.numel() and not bool(torch.isfinite(valid_map_values).all()):
        raise FloatingPointError("valid map positions must be finite")
    eligible_count = int(eligible_future_mask.sum().item())
    if eligible_count == 0:
        return prediction.new_zeros(()), 0

    has_context = eligible_future_mask.any(dim=1)
    eligible_prediction = prediction[has_context].float()
    selected_future_mask = eligible_future_mask[has_context]
    eligible_point_mask = point_mask[has_context].flatten(1)
    eligible_map_points = (
        map_polylines[has_context, ..., :2].flatten(1, 2).float()
    )
    safe_prediction = torch.where(
        selected_future_mask.unsqueeze(-1),
        eligible_prediction,
        torch.zeros_like(eligible_prediction),
    )
    safe_map_points = torch.where(
        eligible_point_mask.unsqueeze(-1),
        eligible_map_points,
        torch.zeros_like(eligible_map_points),
    )
    distances = torch.cdist(safe_prediction, safe_map_points)
    distances = distances.masked_fill(~eligible_point_mask.unsqueeze(1), float("inf"))
    nearest_distance = distances.min(dim=-1).values[selected_future_mask]
    penalty = F.relu(nearest_distance - _MAP_SOFT_MARGIN_M).square().sum()
    if not bool(torch.isfinite(penalty)):
        raise FloatingPointError("map soft loss contains NaN or Inf")
    return penalty, eligible_count


def _collision_soft_context(
    prediction: Tensor,
    batch: Batch,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return collision masks and target indices after validating batch structure."""

    actor_history = _required_tensor(batch, "actor_history")
    actor_time_mask = _required_tensor(batch, "actor_time_mask")
    actor_mask = _required_tensor(batch, "actor_mask")
    target_actor_index = _required_tensor(batch, "target_actor_index").to(dtype=torch.long)
    future_mask = _required_tensor(batch, "target_future_mask")
    if actor_history.ndim != 4 or actor_history.shape[-1] < 4:
        raise ValueError("actor_history must have shape [B, A, H, F] with F >= 4")
    if (
        actor_time_mask.shape != actor_history.shape[:3]
        or actor_time_mask.dtype is not torch.bool
    ):
        raise ValueError("actor_time_mask must match actor_history and be boolean")
    if actor_mask.shape != actor_history.shape[:2] or actor_mask.dtype is not torch.bool:
        raise ValueError("actor_mask must match actor_history and be boolean")
    if target_actor_index.shape != (actor_history.shape[0],):
        raise ValueError("target_actor_index must have shape [B]")
    if bool(
        ((target_actor_index < 0) | (target_actor_index >= actor_history.shape[1])).any()
    ):
        raise ValueError("target_actor_index is outside the actor dimension")
    if actor_history.shape[0] != prediction.shape[0]:
        raise ValueError("actor_history and prediction batch sizes must match")
    if future_mask.shape != prediction.shape[:2] or future_mask.dtype is not torch.bool:
        raise ValueError("target_future_mask must match prediction and be boolean")

    effective_actor_time_mask = actor_time_mask & actor_mask.unsqueeze(-1)
    effective_actor_mask = effective_actor_time_mask.any(dim=-1)
    batch_indices = torch.arange(actor_history.shape[0], device=actor_history.device)
    if not bool(effective_actor_mask[batch_indices, target_actor_index].all()):
        raise ValueError("target_actor_index must reference an actor with valid history")
    other_actor_mask = effective_actor_mask.clone()
    other_actor_mask.scatter_(1, target_actor_index.unsqueeze(1), False)
    eligible_future_mask = future_mask & other_actor_mask.any(dim=1).unsqueeze(1)
    return (
        actor_history,
        effective_actor_time_mask,
        other_actor_mask,
        eligible_future_mask,
        target_actor_index,
    )


def _collision_soft_penalty_sum(
    prediction: Tensor,
    batch: Batch,
) -> tuple[Tensor, int]:
    """Return constant-velocity collision penalty sum and evaluable point count."""

    (
        actor_history,
        effective_actor_time_mask,
        other_actor_mask,
        eligible_future_mask,
        target_actor_index,
    ) = _collision_soft_context(prediction, batch)
    valid_history_values = actor_history[..., :2][effective_actor_time_mask]
    if valid_history_values.numel() and not bool(torch.isfinite(valid_history_values).all()):
        raise FloatingPointError("valid actor history positions must be finite")
    valid_velocity_values = actor_history[..., 2:4][effective_actor_time_mask]
    if valid_velocity_values.numel() and not bool(torch.isfinite(valid_velocity_values).all()):
        raise FloatingPointError("valid actor history velocities must be finite")
    eligible_count = int(eligible_future_mask.sum().item())
    if eligible_count == 0:
        return prediction.new_zeros(()), 0

    history_indices = torch.arange(actor_history.shape[2], device=actor_history.device)
    last_indices = torch.where(
        effective_actor_time_mask,
        history_indices.view(1, 1, -1),
        -1,
    ).max(dim=-1).values
    gather_indices = last_indices.clamp_min(0).unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, 1, 4
    )
    last_states = actor_history[..., :4].gather(2, gather_indices).squeeze(2)
    effective_actor_mask = effective_actor_time_mask.any(dim=-1)
    last_states = torch.where(
        effective_actor_mask.unsqueeze(-1),
        last_states,
        torch.zeros_like(last_states),
    )
    last_positions = last_states[..., :2].float()
    last_velocities = last_states[..., 2:4].float()

    future_offsets = torch.arange(
        prediction.shape[1],
        device=prediction.device,
        dtype=torch.float32,
    )
    steps_from_last_history = (
        actor_history.shape[2] - last_indices.clamp_min(0)
    ).float()
    future_times = (
        steps_from_last_history.unsqueeze(1) + future_offsets.view(1, -1, 1)
    ) * _FUTURE_STEP_SECONDS
    actor_future = last_positions.unsqueeze(1) + (
        last_velocities.unsqueeze(1) * future_times.unsqueeze(-1)
    )

    batch_indices = torch.arange(actor_history.shape[0], device=actor_history.device)
    target_velocity = last_velocities[batch_indices, target_actor_index]
    relative_velocity = target_velocity.unsqueeze(1) - last_velocities
    relative_speed = torch.linalg.vector_norm(relative_velocity, dim=-1, keepdim=True)
    velocity_direction = relative_velocity / relative_speed.clamp_min(1e-12)
    slot_indices = torch.arange(
        actor_history.shape[1],
        device=actor_history.device,
        dtype=torch.float32,
    )
    fixed_angles = slot_indices * (math.pi * (3.0 - math.sqrt(5.0)))
    fixed_direction = torch.stack(
        (torch.cos(fixed_angles), torch.sin(fixed_angles)),
        dim=-1,
    ).unsqueeze(0)
    tie_direction = torch.where(
        relative_speed > 1e-6,
        velocity_direction,
        fixed_direction,
    )

    has_context = eligible_future_mask.any(dim=1)
    eligible_prediction = prediction[has_context].float()
    selected_future_mask = eligible_future_mask[has_context]
    eligible_actor_mask = other_actor_mask[has_context]
    eligible_actor_future = actor_future[has_context]
    eligible_tie_direction = tie_direction[has_context]
    safe_prediction = torch.where(
        selected_future_mask.unsqueeze(-1),
        eligible_prediction,
        torch.zeros_like(eligible_prediction),
    )
    relative_position = safe_prediction.unsqueeze(2) - eligible_actor_future
    exact_overlap = relative_position.square().sum(dim=-1) == 0.0
    stabilized_relative_position = relative_position + (
        exact_overlap.unsqueeze(-1)
        * eligible_tie_direction.unsqueeze(1)
        * _COLLISION_TIE_BREAK_M
    )
    distances = torch.linalg.vector_norm(stabilized_relative_position, dim=-1)
    distances = distances.masked_fill(~eligible_actor_mask.unsqueeze(1), float("inf"))
    nearest_distance = distances.min(dim=-1).values[selected_future_mask]
    penalty = F.relu(_COLLISION_SOFT_RADIUS_M - nearest_distance).square().sum()
    if not bool(torch.isfinite(penalty)):
        raise FloatingPointError("collision soft loss contains NaN or Inf")
    return penalty, eligible_count


def _map_soft_eligible_count(batch: Batch) -> int:
    target = _required_tensor(batch, "target_future")
    _, _, eligible_future_mask = _map_soft_context(target, batch)
    return int(eligible_future_mask.sum().item())


def _collision_soft_eligible_count(batch: Batch) -> int:
    target = _required_tensor(batch, "target_future")
    _, _, _, eligible_future_mask, _ = _collision_soft_context(target, batch)
    return int(eligible_future_mask.sum().item())


def compute_cvae_loss(
    output: CVAEOutput,
    batch: Batch,
    loss_config: LossConfig,
    *,
    global_step: int,
) -> LossBreakdown:
    """Compute one masked batch loss without changing optimizer state."""

    _validate_loss_config(loss_config)
    tensor_sums = _loss_sums(output, batch)
    reconstruction = (
        tensor_sums.reconstruction_sum / tensor_sums.reconstruction_element_count
    )
    endpoint = tensor_sums.endpoint_sum / tensor_sums.endpoint_element_count
    kl = tensor_sums.kl_sum / tensor_sums.sample_count
    kl_weight = kl_anneal_weight(
        global_step,
        loss_config.kl_max_weight,
        loss_config.kl_warmup_steps,
    )
    total = reconstruction + loss_config.endpoint_weight * endpoint + kl_weight * kl
    if loss_config.map_soft_weight > 0.0:
        map_penalty, map_count = _map_soft_penalty_sum(
            output.future_position_local,
            batch,
        )
        if map_count:
            total = total + loss_config.map_soft_weight * map_penalty / map_count
    if loss_config.collision_soft_weight > 0.0:
        collision_penalty, collision_count = _collision_soft_penalty_sum(
            output.future_position_local,
            batch,
        )
        if collision_count:
            total = total + (
                loss_config.collision_soft_weight
                * collision_penalty
                / collision_count
            )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("total CVAE loss contains NaN or Inf")
    return LossBreakdown(
        total=total,
        reconstruction=reconstruction,
        endpoint=endpoint,
        kl=kl,
        kl_weight=kl_weight,
        sums=tensor_sums.detached(),
    )


def _batch_counts(batch: Batch) -> tuple[int, int, int]:
    target = _required_tensor(batch, "target_future")
    mask = _required_tensor(batch, "target_future_mask")
    if target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("target_future must have shape [B, T, 2]")
    if mask.shape != target.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("target_future_mask must match target_future and be boolean")
    valid_samples, _ = _last_valid_indices(mask)
    return int(mask.sum().item()), int(valid_samples.sum().item()), int(target.shape[0])


def _validate_loss_config(loss_config: LossConfig) -> None:
    if loss_config.reconstruction != "smooth_l1":
        raise ValueError("only smooth_l1 reconstruction is implemented")
    for name in ("map_soft_weight", "collision_soft_weight"):
        value = getattr(loss_config, name)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")


def _amp_policy(device: torch.device, amp: bool) -> _AmpPolicy:
    if not amp or device.type != "cuda":
        return _AmpPolicy(enabled=False, dtype=torch.float32, use_scaler=False)
    if torch.cuda.is_bf16_supported():
        return _AmpPolicy(enabled=True, dtype=torch.bfloat16, use_scaler=False)
    return _AmpPolicy(enabled=True, dtype=torch.float16, use_scaler=True)


def _active_scaler(
    device: torch.device,
    policy: _AmpPolicy,
    scaler: torch.amp.GradScaler | None,
) -> torch.amp.GradScaler | None:
    if not policy.use_scaler:
        return None
    return scaler or torch.amp.GradScaler(device.type, enabled=True)


def train_optimizer_step(
    model: ConditionalCVAE,
    optimizer: Optimizer,
    microbatches: Sequence[Batch],
    *,
    device: str | torch.device,
    loss_config: LossConfig,
    global_step: int,
    gradient_clip_norm: float,
    generator: torch.Generator,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    non_blocking: bool = False,
) -> OptimizerStepResult:
    """Consume exactly one accumulation group and perform one optimizer step."""

    if not microbatches:
        raise ValueError("an optimizer step requires at least one microbatch")
    if gradient_clip_norm <= 0 or not math.isfinite(gradient_clip_norm):
        raise ValueError("gradient_clip_norm must be finite and positive")
    _validate_loss_config(loss_config)
    resolved = torch.device(device)
    amp_policy = _amp_policy(resolved, amp)
    active_scaler = _active_scaler(resolved, amp_policy, scaler)
    counts = [_batch_counts(batch) for batch in microbatches]
    total_points = sum(item[0] for item in counts)
    total_valid_samples = sum(item[1] for item in counts)
    total_samples = sum(item[2] for item in counts)
    reconstruction_elements = total_points * 2
    endpoint_elements = total_valid_samples * 2
    kl_weight = kl_anneal_weight(
        global_step,
        loss_config.kl_max_weight,
        loss_config.kl_warmup_steps,
    )
    total_map_points = (
        sum(_map_soft_eligible_count(batch) for batch in microbatches)
        if loss_config.map_soft_weight > 0.0
        else 0
    )
    total_collision_points = (
        sum(_collision_soft_eligible_count(batch) for batch in microbatches)
        if loss_config.collision_soft_weight > 0.0
        else 0
    )

    optimizer.zero_grad(set_to_none=True)
    aggregate = LossSums()
    map_soft_sum = 0.0
    collision_soft_sum = 0.0
    observed_map_points = 0
    observed_collision_points = 0
    try:
        for raw_batch in microbatches:
            batch = move_batch_to_device(
                raw_batch,
                resolved,
                non_blocking=non_blocking,
            )
            with torch.autocast(
                device_type=resolved.type,
                dtype=amp_policy.dtype,
                enabled=amp_policy.enabled,
            ):
                output = model.forward_train(batch, generator)
                tensor_sums = _loss_sums(output, batch)
                loss = (
                    tensor_sums.reconstruction_sum / reconstruction_elements
                    + loss_config.endpoint_weight
                    * tensor_sums.endpoint_sum
                    / endpoint_elements
                    + kl_weight * tensor_sums.kl_sum / total_samples
                )
                if total_map_points:
                    map_penalty, map_count = _map_soft_penalty_sum(
                        output.future_position_local,
                        batch,
                    )
                    loss = loss + (
                        loss_config.map_soft_weight
                        * map_penalty
                        / total_map_points
                    )
                    map_soft_sum += float(map_penalty.detach().cpu())
                    observed_map_points += map_count
                if total_collision_points:
                    collision_penalty, collision_count = _collision_soft_penalty_sum(
                        output.future_position_local,
                        batch,
                    )
                    loss = loss + (
                        loss_config.collision_soft_weight
                        * collision_penalty
                        / total_collision_points
                    )
                    collision_soft_sum += float(collision_penalty.detach().cpu())
                    observed_collision_points += collision_count
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("accumulated CVAE loss contains NaN or Inf")
            if active_scaler is None:
                loss.backward()
            else:
                active_scaler.scale(loss).backward()
            aggregate = aggregate + tensor_sums.detached()

        if observed_map_points != total_map_points:
            raise RuntimeError("map soft eligible point count changed during optimizer step")
        if observed_collision_points != total_collision_points:
            raise RuntimeError(
                "collision soft eligible point count changed during optimizer step"
            )
        if active_scaler is not None:
            active_scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("CVAE optimizer step produced no gradients")
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError("CVAE gradients contain NaN or Inf")
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm_tensor)):
            raise FloatingPointError("CVAE gradient norm contains NaN or Inf")
        if active_scaler is None:
            optimizer.step()
        else:
            active_scaler.step(optimizer)
            active_scaler.update()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError("CVAE parameters contain NaN or Inf after optimizer step")
    finally:
        optimizer.zero_grad(set_to_none=True)

    total_loss = (
        aggregate.reconstruction_loss
        + loss_config.endpoint_weight * aggregate.endpoint_loss
        + kl_weight * aggregate.kl_loss
    )
    if total_map_points:
        total_loss += loss_config.map_soft_weight * map_soft_sum / total_map_points
    if total_collision_points:
        total_loss += (
            loss_config.collision_soft_weight
            * collision_soft_sum
            / total_collision_points
        )
    return OptimizerStepResult(
        next_global_step=global_step + 1,
        total_loss=total_loss,
        kl_weight=kl_weight,
        gradient_norm=float(gradient_norm_tensor.detach().cpu()),
        microbatch_count=len(microbatches),
        sums=aggregate,
    )


def train_epoch(
    model: ConditionalCVAE,
    optimizer: Optimizer,
    batches: Iterable[Batch],
    *,
    device: str | torch.device,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    global_step: int,
    generator: torch.Generator,
    scaler: torch.amp.GradScaler | None = None,
    on_optimizer_step: Callable[[OptimizerStepResult, int], None] | None = None,
) -> TrainEpochResult:
    """Train over one iterable, including one final partial accumulation group."""

    model.train()
    resolved = torch.device(device)
    active_scaler = _active_scaler(
        resolved,
        _amp_policy(resolved, training_config.amp),
        scaler,
    )
    pending: list[Batch] = []
    aggregate = LossSums()
    optimizer_losses: list[float] = []
    microbatch_count = 0
    step = global_step

    def flush() -> None:
        nonlocal pending, aggregate, step
        result = train_optimizer_step(
            model,
            optimizer,
            pending,
            device=resolved,
            loss_config=loss_config,
            global_step=step,
            gradient_clip_norm=training_config.gradient_clip_norm,
            generator=generator,
            amp=training_config.amp,
            scaler=active_scaler,
            non_blocking=training_config.pin_memory,
        )
        step = result.next_global_step
        aggregate = aggregate + result.sums
        optimizer_losses.append(result.total_loss)
        pending = []
        if on_optimizer_step is not None:
            on_optimizer_step(result, microbatch_count)

    for batch in batches:
        pending.append(batch)
        microbatch_count += 1
        if len(pending) == training_config.gradient_accumulation_steps:
            flush()
    if pending:
        flush()
    if not optimizer_losses:
        raise ValueError("train_epoch received no batches")
    return TrainEpochResult(
        next_global_step=step,
        optimizer_steps=len(optimizer_losses),
        microbatch_count=microbatch_count,
        mean_optimizer_loss=sum(optimizer_losses) / len(optimizer_losses),
        sums=aggregate,
    )


def _target_last_state(batch: Batch) -> tuple[Tensor, Tensor]:
    history = _required_tensor(batch, "actor_history")
    time_mask = _required_tensor(batch, "actor_time_mask")
    target_index = _required_tensor(batch, "target_actor_index").to(dtype=torch.long)
    if history.ndim != 4 or history.shape[-1] < 4:
        raise ValueError("actor_history must contain local position and velocity features")
    if time_mask.shape != history.shape[:3] or time_mask.dtype is not torch.bool:
        raise ValueError("actor_time_mask must match actor_history and be boolean")
    batch_indices = torch.arange(history.shape[0], device=history.device)
    target_mask = time_mask[batch_indices, target_index]
    _, last_indices = _last_valid_indices(target_mask)
    target_history = history[batch_indices, target_index]
    last = target_history[batch_indices, last_indices]
    return last[:, :2], last[:, 2:4]


def evaluate(
    model: ConditionalCVAE,
    batches: Iterable[Batch],
    *,
    device: str | torch.device,
    prior_samples: int,
    sample_period_s: float,
    evaluation_seed: int,
    amp: bool = False,
    loss_config: LossConfig | None = None,
    global_step: int = 0,
) -> EvaluationResult:
    """Evaluate with a fresh local generator that never consumes training RNG."""

    if prior_samples <= 0:
        raise ValueError("prior_samples must be positive")
    if loss_config is not None:
        _validate_loss_config(loss_config)
    resolved = torch.device(device)
    amp_policy = _amp_policy(resolved, amp)
    evaluation_generator = torch.Generator(device=resolved).manual_seed(evaluation_seed)
    posterior_total = DisplacementSums(0.0, 0, 0.0, 0)
    prior_total = DisplacementSums(0.0, 0, 0.0, 0)
    constant_total = DisplacementSums(0.0, 0, 0.0, 0)
    loss_total = LossSums()
    map_soft_sum = 0.0
    collision_soft_sum = 0.0
    map_soft_count = 0
    collision_soft_count = 0
    batch_count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for raw_batch in batches:
                batch_count += 1
                batch = move_batch_to_device(raw_batch, resolved)
                target = _required_tensor(batch, "target_future")
                mask = _required_tensor(batch, "target_future_mask")
                with torch.autocast(
                    device_type=resolved.type,
                    dtype=amp_policy.dtype,
                    enabled=amp_policy.enabled,
                ):
                    posterior = model.forward_train(batch, evaluation_generator)
                    prior = model.sample_prior(batch, prior_samples, evaluation_generator)
                    tensor_sums = _loss_sums(posterior, batch)
                    if loss_config is not None and loss_config.map_soft_weight > 0.0:
                        map_penalty, current_map_count = _map_soft_penalty_sum(
                            posterior.future_position_local,
                            batch,
                        )
                        map_soft_sum += float(map_penalty.detach().cpu())
                        map_soft_count += current_map_count
                    if (
                        loss_config is not None
                        and loss_config.collision_soft_weight > 0.0
                    ):
                        collision_penalty, current_collision_count = (
                            _collision_soft_penalty_sum(
                                posterior.future_position_local,
                                batch,
                            )
                        )
                        collision_soft_sum += float(collision_penalty.detach().cpu())
                        collision_soft_count += current_collision_count
                loss_total = loss_total + tensor_sums.detached()
                posterior_total = posterior_total + displacement_sums(
                    posterior.future_position_local.float(),
                    target.float(),
                    mask,
                )
                prior_total = prior_total + multimodal_displacement_sums(
                    prior.future_position_local.float(),
                    target.float(),
                    mask,
                )
                last_position, last_velocity = _target_last_state(batch)
                constant = constant_velocity_prediction(
                    last_position.float(),
                    last_velocity.float(),
                    future_steps=target.shape[1],
                    sample_period_s=sample_period_s,
                )
                constant_total = constant_total + displacement_sums(
                    constant,
                    target.float(),
                    mask,
                )
    finally:
        model.train(was_training)
    if batch_count == 0:
        raise ValueError("evaluate received no batches")
    validation_loss = None
    if loss_config is not None:
        kl_weight = kl_anneal_weight(
            global_step,
            loss_config.kl_max_weight,
            loss_config.kl_warmup_steps,
        )
        map_soft_loss = 0.0 if map_soft_count == 0 else map_soft_sum / map_soft_count
        collision_soft_loss = (
            0.0
            if collision_soft_count == 0
            else collision_soft_sum / collision_soft_count
        )
        total_loss = (
            loss_total.reconstruction_loss
            + loss_config.endpoint_weight * loss_total.endpoint_loss
            + kl_weight * loss_total.kl_loss
            + loss_config.map_soft_weight * map_soft_loss
            + loss_config.collision_soft_weight * collision_soft_loss
        )
        validation_loss = ValidationLossResult(
            total_loss=total_loss,
            reconstruction_loss=loss_total.reconstruction_loss,
            endpoint_loss=loss_total.endpoint_loss,
            kl_loss=loss_total.kl_loss,
            kl_weight=kl_weight,
            map_soft_loss=map_soft_loss,
            collision_soft_loss=collision_soft_loss,
            sums=loss_total,
        )
    return EvaluationResult(
        posterior=posterior_total,
        prior=prior_total,
        constant_velocity=constant_total,
        kl_sum=loss_total.kl_sum,
        sample_count=loss_total.sample_count,
        validation_loss=validation_loss,
    )


class _RepeatingBatches:
    def __init__(self, batches: Iterable[Batch]) -> None:
        self.batches = batches
        self.iterator = iter(batches)

    def next(self) -> Batch:
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.batches)
            try:
                return next(self.iterator)
            except StopIteration:
                raise ValueError("benchmark batches must be non-empty and re-iterable") from None


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_training(
    model: ConditionalCVAE,
    optimizer: Optimizer,
    batches: Iterable[Batch],
    *,
    device: str | torch.device,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    global_step: int,
    generator: torch.Generator,
    warmup_steps: int,
    measured_steps: int,
    scaler: torch.amp.GradScaler | None = None,
) -> BenchmarkResult:
    """Measure fixed optimizer steps after startup and warmup are excluded."""

    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("benchmark requires nonnegative warmup and positive measured steps")
    resolved = torch.device(device)
    active_scaler = _active_scaler(
        resolved,
        _amp_policy(resolved, training_config.amp),
        scaler,
    )
    stream = _RepeatingBatches(batches)

    def next_group() -> list[Batch]:
        return [
            stream.next()
            for _ in range(training_config.gradient_accumulation_steps)
        ]

    startup_start = time.perf_counter()
    first_group = next_group()
    startup_seconds = time.perf_counter() - startup_start
    step = global_step
    model.train()

    warmup_start = time.perf_counter()
    pending_first = first_group
    for index in range(warmup_steps):
        group = pending_first if index == 0 else next_group()
        result = train_optimizer_step(
            model,
            optimizer,
            group,
            device=resolved,
            loss_config=loss_config,
            global_step=step,
            gradient_clip_norm=training_config.gradient_clip_norm,
            generator=generator,
            amp=training_config.amp,
            scaler=active_scaler,
            non_blocking=training_config.pin_memory,
        )
        step = result.next_global_step
    _synchronize(resolved)
    warmup_seconds = time.perf_counter() - warmup_start

    step_seconds: list[float] = []
    wait_seconds: list[float] = []
    measured_samples = 0
    gpu_index = None
    if resolved.type == "cuda":
        gpu_index = resolved.index
        if gpu_index is None:
            gpu_index = torch.cuda.current_device()
    monitor = SystemMonitor(gpu_index=gpu_index)
    monitor_overhead_start = time.perf_counter()
    monitor.start()
    monitor.begin_window()
    monitor_overhead_seconds = time.perf_counter() - monitor_overhead_start
    try:
        for index in range(measured_steps):
            wait_start = time.perf_counter()
            group = first_group if warmup_steps == 0 and index == 0 else next_group()
            data_wait = time.perf_counter() - wait_start
            _synchronize(resolved)
            step_start = time.perf_counter()
            result = train_optimizer_step(
                model,
                optimizer,
                group,
                device=resolved,
                loss_config=loss_config,
                global_step=step,
                gradient_clip_norm=training_config.gradient_clip_norm,
                generator=generator,
                amp=training_config.amp,
                scaler=active_scaler,
                non_blocking=training_config.pin_memory,
            )
            _synchronize(resolved)
            compute_seconds = time.perf_counter() - step_start
            step = result.next_global_step
            wait_seconds.append(data_wait)
            step_seconds.append(data_wait + compute_seconds)
            measured_samples += result.sums.sample_count
    finally:
        monitor_overhead_start = time.perf_counter()
        try:
            system_metrics = monitor.end_window()
        finally:
            try:
                monitor.stop()
            finally:
                monitor_overhead_seconds += (
                    time.perf_counter() - monitor_overhead_start
                )
    total_seconds = sum(step_seconds)
    return BenchmarkResult(
        startup_seconds=startup_seconds,
        warmup_seconds=warmup_seconds,
        step_seconds=tuple(step_seconds),
        data_wait_seconds=tuple(wait_seconds),
        p50_step_seconds=_percentile(step_seconds, 0.50),
        p95_step_seconds=_percentile(step_seconds, 0.95),
        samples_per_second=measured_samples / total_seconds,
        measured_samples=measured_samples,
        next_global_step=step,
        cpu_metrics_available=system_metrics.cpu_metrics_available,
        cpu_busy_percent=system_metrics.cpu_busy_percent,
        cpu_iowait_percent=system_metrics.cpu_iowait_percent,
        gpu_utilization_available=system_metrics.gpu_utilization_available,
        gpu_utilization_mean_percent=system_metrics.gpu_utilization_mean_percent,
        gpu_utilization_p50_percent=system_metrics.gpu_utilization_p50_percent,
        gpu_utilization_p95_percent=system_metrics.gpu_utilization_p95_percent,
        gpu_utilization_sample_count=system_metrics.gpu_utilization_sample_count,
        monitor_overhead_seconds=monitor_overhead_seconds,
    )


__all__ = [
    "BenchmarkResult",
    "EvaluationResult",
    "LossBreakdown",
    "LossSums",
    "OptimizerStepResult",
    "TrainEpochResult",
    "benchmark_training",
    "compute_cvae_loss",
    "evaluate",
    "kl_anneal_weight",
    "move_batch_to_device",
    "train_epoch",
    "train_optimizer_step",
]

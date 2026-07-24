"""Lightweight multimodal trajectory predictors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from skilldrive.prediction.metrics import constant_velocity_prediction


@dataclass(frozen=True)
class PredictionOutput:
    trajectories: Tensor
    logits: Tensor


def _tensor(batch: Mapping[str, Tensor], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"prediction batch is missing tensor {name}")
    return value


def _mask(batch: Mapping[str, Tensor], name: str, shape: tuple[int, ...]) -> Tensor:
    value = _tensor(batch, name)
    if value.shape != shape or value.dtype is not torch.bool:
        raise ValueError(f"{name} must have boolean shape {shape}")
    return value


def _masked_gru(inputs: Tensor, mask: Tensor, cell: nn.GRUCell) -> Tensor:
    if inputs.shape[:-1] != mask.shape:
        raise ValueError("masked GRU inputs and mask do not align")
    flat_inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
    flat_mask = mask.reshape(-1, mask.shape[-1])
    hidden = inputs.new_zeros((flat_inputs.shape[0], cell.hidden_size))
    for step in range(flat_inputs.shape[1]):
        candidate = cell(flat_inputs[:, step], hidden)
        hidden = torch.where(flat_mask[:, step, None], candidate, hidden)
    return hidden.reshape(*inputs.shape[:-2], cell.hidden_size)


class _MultiModalHead(nn.Module):
    def __init__(self, hidden_dim: int, *, num_modes: int, future_steps: int) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.mode_embedding = nn.Embedding(num_modes, hidden_dim)
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )
        self.logit_head = nn.Linear(hidden_dim, 1)

    def forward(self, context: Tensor, baseline: Tensor) -> PredictionOutput:
        mode_ids = torch.arange(self.num_modes, device=context.device)
        conditioned = context[:, None, :] + self.mode_embedding(mode_ids)[None]
        residual = self.trajectory_head(conditioned).reshape(
            context.shape[0], self.num_modes, self.future_steps, 2
        )
        logits = self.logit_head(conditioned).squeeze(-1)
        return PredictionOutput(trajectories=baseline + residual, logits=logits)


class LSTMTrajectoryPredictor(nn.Module):
    """Target-history-only recurrent baseline."""

    def __init__(
        self,
        *,
        actor_feature_dim: int = 6,
        hidden_dim: int = 128,
        num_modes: int = 6,
        future_steps: int = 60,
        sample_period_s: float = 0.1,
    ) -> None:
        super().__init__()
        self.actor_feature_dim = actor_feature_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.sample_period_s = sample_period_s
        self.history_cell = nn.GRUCell(actor_feature_dim, hidden_dim)
        self.head = _MultiModalHead(hidden_dim, num_modes=num_modes, future_steps=future_steps)

    def forward(self, batch: Mapping[str, Tensor]) -> PredictionOutput:
        actor_history = _tensor(batch, "actor_history")
        if actor_history.ndim != 4 or actor_history.shape[-1] != self.actor_feature_dim:
            raise ValueError("actor_history has an invalid shape")
        batch_size, actor_count, history_steps, _ = actor_history.shape
        actor_time_mask = _mask(
            batch, "actor_time_mask", (batch_size, actor_count, history_steps)
        )
        actor_mask = _mask(batch, "actor_mask", (batch_size, actor_count))
        target_actor_index = _tensor(batch, "target_actor_index").to(
            device=actor_history.device, dtype=torch.long
        )
        if target_actor_index.shape != (batch_size,):
            raise ValueError("target_actor_index has an invalid shape")
        rows = torch.arange(batch_size, device=actor_history.device)
        target_history = actor_history[rows, target_actor_index]
        target_mask = actor_time_mask[rows, target_actor_index] & actor_mask[
            rows, target_actor_index, None
        ]
        context = _masked_gru(target_history[:, None], target_mask[:, None], self.history_cell)[:, 0]
        baseline = constant_velocity_prediction(
            actor_history,
            actor_time_mask,
            actor_mask,
            target_actor_index,
            future_steps=self.future_steps,
            sample_period_s=self.sample_period_s,
        )
        return self.head(context, baseline)


class VectorTrajectoryPredictor(nn.Module):
    """GRU and polyline encoders with lightweight scene interaction."""

    def __init__(
        self,
        *,
        actor_feature_dim: int = 6,
        map_feature_dim: int = 4,
        num_actor_types: int = 11,
        num_map_types: int = 4,
        hidden_dim: int = 128,
        type_embedding_dim: int = 16,
        interaction_layers: int = 2,
        interaction_heads: int = 4,
        num_modes: int = 6,
        future_steps: int = 60,
        dropout: float = 0.1,
        sample_period_s: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % interaction_heads:
            raise ValueError("hidden_dim must be divisible by interaction_heads")
        self.actor_feature_dim = actor_feature_dim
        self.map_feature_dim = map_feature_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.sample_period_s = sample_period_s

        self.actor_history_cell = nn.GRUCell(actor_feature_dim, hidden_dim)
        self.actor_type_embedding = nn.Embedding(num_actor_types, type_embedding_dim)
        self.actor_projection = nn.Linear(hidden_dim + type_embedding_dim, hidden_dim)
        self.map_point_mlp = nn.Sequential(
            nn.Linear(map_feature_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.map_type_embedding = nn.Embedding(num_map_types, type_embedding_dim)
        self.map_projection = nn.Linear(hidden_dim + type_embedding_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=interaction_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.interaction_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=interaction_layers, enable_nested_tensor=False
        )
        self.head = _MultiModalHead(hidden_dim, num_modes=num_modes, future_steps=future_steps)

    def _context(self, batch: Mapping[str, Tensor]) -> Tensor:
        actor_history = _tensor(batch, "actor_history")
        if actor_history.ndim != 4 or actor_history.shape[-1] != self.actor_feature_dim:
            raise ValueError("actor_history has an invalid shape")
        batch_size, actor_count, history_steps, _ = actor_history.shape
        actor_time_mask = _mask(
            batch, "actor_time_mask", (batch_size, actor_count, history_steps)
        )
        actor_mask = _mask(batch, "actor_mask", (batch_size, actor_count))
        effective_actor_time = actor_time_mask & actor_mask[:, :, None]
        effective_actor = effective_actor_time.any(dim=-1)
        actor_values = actor_history[effective_actor_time]
        if actor_values.numel() and not bool(torch.isfinite(actor_values).all()):
            raise ValueError("valid actor history contains non-finite values")
        actor_hidden = _masked_gru(
            actor_history, effective_actor_time, self.actor_history_cell
        )
        actor_type_id = _tensor(batch, "actor_type_id").to(
            device=actor_history.device, dtype=torch.long
        )
        if actor_type_id.shape != (batch_size, actor_count):
            raise ValueError("actor_type_id has an invalid shape")
        actor_tokens = self.actor_projection(
            torch.cat((actor_hidden, self.actor_type_embedding(actor_type_id)), dim=-1)
        )
        actor_tokens = torch.where(
            effective_actor[:, :, None], actor_tokens, torch.zeros_like(actor_tokens)
        )

        map_polylines = _tensor(batch, "map_polylines")
        if map_polylines.ndim != 4 or map_polylines.shape[0] != batch_size or map_polylines.shape[-1] != self.map_feature_dim:
            raise ValueError("map_polylines has an invalid shape")
        _, polyline_count, point_count, _ = map_polylines.shape
        map_point_mask = _mask(
            batch, "map_point_mask", (batch_size, polyline_count, point_count)
        )
        map_polyline_mask = _mask(
            batch, "map_polyline_mask", (batch_size, polyline_count)
        )
        effective_points = map_point_mask & map_polyline_mask[:, :, None]
        effective_map = effective_points.any(dim=-1)
        valid_map = map_polylines[effective_points]
        if valid_map.numel() and not bool(torch.isfinite(valid_map).all()):
            raise ValueError("valid map values contain non-finite values")
        safe_map = torch.where(
            effective_points[:, :, :, None], map_polylines, torch.zeros_like(map_polylines)
        )
        point_features = self.map_point_mlp(safe_map).masked_fill(
            ~effective_points[:, :, :, None], -torch.inf
        )
        map_hidden = point_features.max(dim=2).values
        map_hidden = torch.where(
            effective_map[:, :, None], map_hidden, torch.zeros_like(map_hidden)
        )
        map_type_id = _tensor(batch, "map_type_id").to(
            device=map_polylines.device, dtype=torch.long
        )
        if map_type_id.shape != (batch_size, polyline_count):
            raise ValueError("map_type_id has an invalid shape")
        map_tokens = self.map_projection(
            torch.cat((map_hidden, self.map_type_embedding(map_type_id)), dim=-1)
        )
        map_tokens = torch.where(
            effective_map[:, :, None], map_tokens, torch.zeros_like(map_tokens)
        )

        token_mask = torch.cat((effective_actor, effective_map), dim=1)
        tokens = torch.cat((actor_tokens, map_tokens), dim=1)
        encoded = self.interaction_encoder(tokens, src_key_padding_mask=~token_mask)
        target_actor_index = _tensor(batch, "target_actor_index").to(
            device=actor_history.device, dtype=torch.long
        )
        if target_actor_index.shape != (batch_size,):
            raise ValueError("target_actor_index has an invalid shape")
        rows = torch.arange(batch_size, device=actor_history.device)
        if not bool(effective_actor[rows, target_actor_index].all()):
            raise ValueError("target_actor_index must reference valid history")
        return encoded[rows, target_actor_index]

    def forward(self, batch: Mapping[str, Tensor]) -> PredictionOutput:
        context = self._context(batch)
        actor_history = _tensor(batch, "actor_history")
        actor_time_mask = _tensor(batch, "actor_time_mask")
        actor_mask = _tensor(batch, "actor_mask")
        target_actor_index = _tensor(batch, "target_actor_index")
        baseline = constant_velocity_prediction(
            actor_history,
            actor_time_mask,
            actor_mask,
            target_actor_index,
            future_steps=self.future_steps,
            sample_period_s=self.sample_period_s,
        )
        return self.head(context, baseline)


__all__ = ["LSTMTrajectoryPredictor", "PredictionOutput", "VectorTrajectoryPredictor"]

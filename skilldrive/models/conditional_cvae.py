"""Single-target conditional CVAE for local-frame trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class CVAEOutput:
    """Model outputs for posterior training or prior sampling."""

    future_delta: Tensor
    future_position_local: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor
    posterior_mean: Tensor | None
    posterior_logvar: Tensor | None
    latent: Tensor


class ConditionalCVAE(nn.Module):
    """Encode one vector scene and generate one target actor's future trajectory."""

    def __init__(
        self,
        *,
        actor_feature_dim: int = 6,
        map_feature_dim: int = 4,
        num_actor_types: int = 11,
        num_actor_roles: int = 64,
        num_map_types: int = 4,
        num_skills: int = 35,
        parameter_dim: int = 106,
        actor_type_embedding_dim: int = 16,
        actor_role_embedding_dim: int = 16,
        history_hidden_dim: int = 128,
        map_type_embedding_dim: int = 16,
        map_hidden_dim: int = 128,
        interaction_hidden_dim: int = 128,
        interaction_layers: int = 2,
        interaction_heads: int = 4,
        skill_embedding_dim: int = 32,
        parameter_hidden_dim: int = 32,
        latent_dim: int = 16,
        decoder_hidden_dim: int = 128,
        future_steps: int = 60,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = {
            "actor_feature_dim": actor_feature_dim,
            "map_feature_dim": map_feature_dim,
            "num_actor_types": num_actor_types,
            "num_actor_roles": num_actor_roles,
            "num_map_types": num_map_types,
            "num_skills": num_skills,
            "parameter_dim": parameter_dim,
            "actor_type_embedding_dim": actor_type_embedding_dim,
            "actor_role_embedding_dim": actor_role_embedding_dim,
            "history_hidden_dim": history_hidden_dim,
            "map_type_embedding_dim": map_type_embedding_dim,
            "map_hidden_dim": map_hidden_dim,
            "interaction_hidden_dim": interaction_hidden_dim,
            "interaction_layers": interaction_layers,
            "interaction_heads": interaction_heads,
            "skill_embedding_dim": skill_embedding_dim,
            "parameter_hidden_dim": parameter_hidden_dim,
            "latent_dim": latent_dim,
            "decoder_hidden_dim": decoder_hidden_dim,
            "future_steps": future_steps,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {invalid}")
        if interaction_hidden_dim % interaction_heads:
            raise ValueError("interaction_hidden_dim must be divisible by interaction_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.actor_feature_dim = actor_feature_dim
        self.map_feature_dim = map_feature_dim
        self.parameter_dim = parameter_dim
        self.latent_dim = latent_dim
        self.future_steps = future_steps

        self.actor_history_cell = nn.GRUCell(actor_feature_dim, history_hidden_dim)
        self.actor_type_embedding = nn.Embedding(num_actor_types, actor_type_embedding_dim)
        self.actor_role_embedding = nn.Embedding(num_actor_roles, actor_role_embedding_dim)
        self.actor_token_projection = nn.Sequential(
            nn.Linear(
                history_hidden_dim
                + actor_type_embedding_dim
                + actor_role_embedding_dim,
                interaction_hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(interaction_hidden_dim),
        )

        self.map_point_mlp = nn.Sequential(
            nn.Linear(map_feature_dim, map_hidden_dim),
            nn.GELU(),
            nn.Linear(map_hidden_dim, map_hidden_dim),
            nn.GELU(),
        )
        self.map_type_embedding = nn.Embedding(num_map_types, map_type_embedding_dim)
        self.map_token_projection = nn.Sequential(
            nn.Linear(map_hidden_dim + map_type_embedding_dim, interaction_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(interaction_hidden_dim),
        )

        interaction_layer = nn.TransformerEncoderLayer(
            d_model=interaction_hidden_dim,
            nhead=interaction_heads,
            dim_feedforward=interaction_hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.interaction_encoder = nn.TransformerEncoder(
            interaction_layer,
            num_layers=interaction_layers,
        )

        self.skill_embedding = nn.Embedding(num_skills, skill_embedding_dim)
        self.parameter_encoder = nn.Sequential(
            nn.Linear(parameter_dim * 2, parameter_hidden_dim),
            nn.GELU(),
            nn.Linear(parameter_hidden_dim, parameter_hidden_dim),
            nn.GELU(),
        )
        condition_dim = (
            interaction_hidden_dim + skill_embedding_dim + parameter_hidden_dim
        )
        self.condition_dim = condition_dim

        self.prior_head = nn.Sequential(
            nn.Linear(condition_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, latent_dim * 2),
        )
        self.future_encoder_cell = nn.GRUCell(2, history_hidden_dim)
        self.posterior_head = nn.Sequential(
            nn.Linear(condition_dim + history_hidden_dim, decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(decoder_hidden_dim, latent_dim * 2),
        )

        decoder_input_dim = condition_dim + latent_dim
        self.decoder_initial = nn.Linear(decoder_input_dim, decoder_hidden_dim)
        self.decoder_cell = nn.GRUCell(
            decoder_input_dim + 2,
            decoder_hidden_dim,
        )
        self.delta_head = nn.Linear(decoder_hidden_dim, 2)

    @staticmethod
    def _tensor(batch: Mapping[str, Tensor], name: str) -> Tensor:
        try:
            value = batch[name]
        except KeyError:
            raise KeyError(f"batch is missing required tensor: {name}") from None
        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
        return value

    @staticmethod
    def _boolean_mask(batch: Mapping[str, Tensor], name: str, shape: tuple[int, ...]) -> Tensor:
        mask = ConditionalCVAE._tensor(batch, name)
        if tuple(mask.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
        if mask.dtype is not torch.bool:
            raise ValueError(f"{name} must have boolean dtype")
        return mask

    @staticmethod
    def _optional_ids(
        batch: Mapping[str, Tensor],
        name: str,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> Tensor:
        value = batch.get(name)
        if value is None:
            return torch.zeros(shape, dtype=torch.long, device=device)
        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{name!r}] must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        return value.to(device=device, dtype=torch.long)

    @staticmethod
    def _masked_gru(inputs: Tensor, mask: Tensor, cell: nn.GRUCell) -> Tensor:
        """Run a GRUCell without updating hidden state on any masked timestep."""

        if inputs.shape[:-1] != mask.shape:
            raise ValueError("masked GRU input and mask shapes do not align")
        prefix = inputs.shape[:-2]
        time_steps = inputs.shape[-2]
        flat_inputs = inputs.reshape(-1, time_steps, inputs.shape[-1])
        flat_mask = mask.reshape(-1, time_steps)
        hidden = inputs.new_zeros((flat_inputs.shape[0], cell.hidden_size))
        for timestep in range(time_steps):
            valid = flat_mask[:, timestep, None]
            step_input = torch.where(
                valid,
                flat_inputs[:, timestep],
                torch.zeros_like(flat_inputs[:, timestep]),
            )
            candidate = cell(step_input, hidden)
            hidden = torch.where(valid, candidate, hidden)
        return hidden.reshape(*prefix, cell.hidden_size)

    @staticmethod
    def _gaussian_parameters(values: Tensor) -> tuple[Tensor, Tensor]:
        mean, logvar = values.chunk(2, dim=-1)
        return mean, logvar.clamp(min=-10.0, max=10.0)

    @staticmethod
    def _sample_gaussian(
        mean: Tensor,
        logvar: Tensor,
        *,
        generator: torch.Generator | None,
        num_samples: int | None = None,
    ) -> Tensor:
        shape = mean.shape if num_samples is None else (*mean.shape[:-1], num_samples, mean.shape[-1])
        noise = torch.randn(
            shape,
            dtype=mean.dtype,
            device=mean.device,
            generator=generator,
        )
        if num_samples is None:
            return mean + noise * torch.exp(0.5 * logvar)
        return mean.unsqueeze(-2) + noise * torch.exp(0.5 * logvar).unsqueeze(-2)

    def _encode_context(self, batch: Mapping[str, Tensor]) -> Tensor:
        actor_history = self._tensor(batch, "actor_history")
        if actor_history.ndim != 4 or actor_history.shape[-1] != self.actor_feature_dim:
            raise ValueError(
                "actor_history must have shape [B, A, H, actor_feature_dim]"
            )
        batch_size, actor_count, history_steps, _ = actor_history.shape
        actor_time_mask = self._boolean_mask(
            batch,
            "actor_time_mask",
            (batch_size, actor_count, history_steps),
        )
        actor_mask = self._boolean_mask(
            batch,
            "actor_mask",
            (batch_size, actor_count),
        )
        effective_actor_time_mask = actor_time_mask & actor_mask.unsqueeze(-1)
        effective_actor_mask = effective_actor_time_mask.any(dim=-1)
        valid_actor_values = actor_history[effective_actor_time_mask]
        if valid_actor_values.numel() and not torch.isfinite(valid_actor_values).all():
            raise ValueError("valid actor_history values must be finite")

        actor_hidden = self._masked_gru(
            actor_history,
            effective_actor_time_mask,
            self.actor_history_cell,
        )
        actor_type_id = self._optional_ids(
            batch,
            "actor_type_id",
            (batch_size, actor_count),
            actor_history.device,
        )
        actor_role_id = self._optional_ids(
            batch,
            "actor_role_id",
            (batch_size, actor_count),
            actor_history.device,
        )
        actor_tokens = self.actor_token_projection(
            torch.cat(
                (
                    actor_hidden,
                    self.actor_type_embedding(actor_type_id),
                    self.actor_role_embedding(actor_role_id),
                ),
                dim=-1,
            )
        )
        actor_tokens = torch.where(
            effective_actor_mask.unsqueeze(-1),
            actor_tokens,
            torch.zeros_like(actor_tokens),
        )

        map_polylines = self._tensor(batch, "map_polylines")
        if map_polylines.ndim != 4 or map_polylines.shape[-1] != self.map_feature_dim:
            raise ValueError(
                "map_polylines must have shape [B, P, Q, map_feature_dim]"
            )
        if map_polylines.shape[0] != batch_size:
            raise ValueError("actor_history and map_polylines batch sizes must match")
        _, polyline_count, point_count, _ = map_polylines.shape
        map_point_mask = self._boolean_mask(
            batch,
            "map_point_mask",
            (batch_size, polyline_count, point_count),
        )
        map_polyline_mask = self._boolean_mask(
            batch,
            "map_polyline_mask",
            (batch_size, polyline_count),
        )
        effective_map_point_mask = map_point_mask & map_polyline_mask.unsqueeze(-1)
        effective_map_mask = effective_map_point_mask.any(dim=-1)
        valid_map_values = map_polylines[effective_map_point_mask]
        if valid_map_values.numel() and not torch.isfinite(valid_map_values).all():
            raise ValueError("valid map_polylines values must be finite")

        safe_map_points = torch.where(
            effective_map_point_mask.unsqueeze(-1),
            map_polylines,
            torch.zeros_like(map_polylines),
        )
        point_features = self.map_point_mlp(safe_map_points)
        point_features = point_features.masked_fill(
            ~effective_map_point_mask.unsqueeze(-1),
            -torch.inf,
        )
        map_hidden = point_features.max(dim=2).values
        map_hidden = torch.where(
            effective_map_mask.unsqueeze(-1),
            map_hidden,
            torch.zeros_like(map_hidden),
        )
        map_type_id = self._optional_ids(
            batch,
            "map_type_id",
            (batch_size, polyline_count),
            map_polylines.device,
        )
        map_tokens = self.map_token_projection(
            torch.cat((map_hidden, self.map_type_embedding(map_type_id)), dim=-1)
        )
        map_tokens = torch.where(
            effective_map_mask.unsqueeze(-1),
            map_tokens,
            torch.zeros_like(map_tokens),
        )

        token_mask = torch.cat((effective_actor_mask, effective_map_mask), dim=1)
        if not bool(token_mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid actor or map token")
        tokens = torch.cat((actor_tokens, map_tokens), dim=1)
        encoded_tokens = self.interaction_encoder(
            tokens,
            src_key_padding_mask=~token_mask,
        )

        target_actor_index = self._tensor(batch, "target_actor_index")
        if tuple(target_actor_index.shape) != (batch_size,):
            raise ValueError(
                f"target_actor_index must have shape {(batch_size,)}, "
                f"got {tuple(target_actor_index.shape)}"
            )
        target_actor_index = target_actor_index.to(
            device=actor_history.device,
            dtype=torch.long,
        )
        if bool(((target_actor_index < 0) | (target_actor_index >= actor_count)).any()):
            raise ValueError("target_actor_index is outside the actor dimension")
        batch_indices = torch.arange(batch_size, device=actor_history.device)
        if not bool(effective_actor_mask[batch_indices, target_actor_index].all()):
            raise ValueError("target_actor_index must reference an actor with valid history")
        target_context = encoded_tokens[batch_indices, target_actor_index]

        skill_id = self._tensor(batch, "skill_id")
        if tuple(skill_id.shape) != (batch_size,):
            raise ValueError(f"skill_id must have shape {(batch_size,)}")
        skill_condition = self.skill_embedding(
            skill_id.to(device=actor_history.device, dtype=torch.long)
        )

        skill_parameters = self._tensor(batch, "skill_parameters")
        if tuple(skill_parameters.shape) != (batch_size, self.parameter_dim):
            raise ValueError(
                "skill_parameters must have shape "
                f"{(batch_size, self.parameter_dim)}, got {tuple(skill_parameters.shape)}"
            )
        parameter_mask = self._boolean_mask(
            batch,
            "parameter_mask",
            (batch_size, self.parameter_dim),
        )
        valid_parameters = skill_parameters[parameter_mask]
        if valid_parameters.numel() and not torch.isfinite(valid_parameters).all():
            raise ValueError("valid skill_parameters values must be finite")
        masked_parameters = torch.where(
            parameter_mask,
            skill_parameters,
            torch.zeros_like(skill_parameters),
        )
        parameter_condition = self.parameter_encoder(
            torch.cat((masked_parameters, parameter_mask.to(skill_parameters.dtype)), dim=-1)
        )
        return torch.cat((target_context, skill_condition, parameter_condition), dim=-1)

    def _decode(self, condition: Tensor, latent: Tensor) -> tuple[Tensor, Tensor]:
        if condition.ndim != 2 or latent.ndim != 2:
            raise ValueError("condition and latent must both be rank-two tensors")
        if condition.shape[0] != latent.shape[0]:
            raise ValueError("condition and latent batch sizes must match")
        decoder_condition = torch.cat((condition, latent), dim=-1)
        hidden = torch.tanh(self.decoder_initial(decoder_condition))
        previous_delta = condition.new_zeros((condition.shape[0], 2))
        deltas: list[Tensor] = []
        for _ in range(self.future_steps):
            hidden = self.decoder_cell(
                torch.cat((previous_delta, decoder_condition), dim=-1),
                hidden,
            )
            previous_delta = self.delta_head(hidden)
            deltas.append(previous_delta)
        future_delta = torch.stack(deltas, dim=1)
        return future_delta, future_delta.cumsum(dim=1)

    def forward_train(
        self,
        batch: Mapping[str, Tensor],
        generator: torch.Generator | None = None,
    ) -> CVAEOutput:
        """Use the target future only in the posterior training path."""

        condition = self._encode_context(batch)
        prior_mean, prior_logvar = self._gaussian_parameters(self.prior_head(condition))

        target_future = self._tensor(batch, "target_future")
        expected_shape = (condition.shape[0], self.future_steps, 2)
        if tuple(target_future.shape) != expected_shape:
            raise ValueError(
                f"target_future must have shape {expected_shape}, "
                f"got {tuple(target_future.shape)}"
            )
        target_future_mask = self._boolean_mask(
            batch,
            "target_future_mask",
            expected_shape[:2],
        )
        if not bool(target_future_mask.any(dim=1).all()):
            raise ValueError("every training sample must contain a valid future point")
        valid_future = target_future[target_future_mask]
        if not torch.isfinite(valid_future).all():
            raise ValueError("valid target_future values must be finite")
        future_hidden = self._masked_gru(
            target_future,
            target_future_mask,
            self.future_encoder_cell,
        )
        posterior_mean, posterior_logvar = self._gaussian_parameters(
            self.posterior_head(torch.cat((condition, future_hidden), dim=-1))
        )
        latent = self._sample_gaussian(
            posterior_mean,
            posterior_logvar,
            generator=generator,
        )
        future_delta, future_position_local = self._decode(condition, latent)
        return CVAEOutput(
            future_delta=future_delta,
            future_position_local=future_position_local,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            posterior_mean=posterior_mean,
            posterior_logvar=posterior_logvar,
            latent=latent,
        )

    def sample_prior(
        self,
        context_batch: Mapping[str, Tensor],
        num_samples: int,
        generator: torch.Generator | None = None,
    ) -> CVAEOutput:
        """Sample only from the condition prior without reading any target future."""

        if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        condition = self._encode_context(context_batch)
        prior_mean, prior_logvar = self._gaussian_parameters(self.prior_head(condition))
        latent = self._sample_gaussian(
            prior_mean,
            prior_logvar,
            generator=generator,
            num_samples=num_samples,
        )
        batch_size = condition.shape[0]
        expanded_condition = condition[:, None, :].expand(-1, num_samples, -1)
        flat_delta, flat_position = self._decode(
            expanded_condition.reshape(batch_size * num_samples, -1),
            latent.reshape(batch_size * num_samples, self.latent_dim),
        )
        return CVAEOutput(
            future_delta=flat_delta.reshape(
                batch_size,
                num_samples,
                self.future_steps,
                2,
            ),
            future_position_local=flat_position.reshape(
                batch_size,
                num_samples,
                self.future_steps,
                2,
            ),
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            posterior_mean=None,
            posterior_logvar=None,
            latent=latent,
        )


__all__ = ["CVAEOutput", "ConditionalCVAE"]

from __future__ import annotations

from pathlib import Path

import torch

from skilldrive.models import CVAEOutput, ConditionalCVAE
from skilldrive.training.checkpoint import (
    TrainingProgress,
    load_checkpoint,
    save_checkpoint,
)


def _model() -> ConditionalCVAE:
    return ConditionalCVAE(
        actor_feature_dim=4,
        map_feature_dim=3,
        num_actor_types=4,
        num_actor_roles=4,
        num_map_types=4,
        num_skills=5,
        parameter_dim=3,
        actor_type_embedding_dim=4,
        actor_role_embedding_dim=4,
        history_hidden_dim=12,
        map_type_embedding_dim=4,
        map_hidden_dim=12,
        interaction_hidden_dim=16,
        interaction_layers=2,
        interaction_heads=4,
        skill_embedding_dim=4,
        parameter_hidden_dim=8,
        latent_dim=4,
        decoder_hidden_dim=16,
        future_steps=6,
        dropout=0.0,
    )


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    actor_history = torch.randn(2, 3, 5, 4, generator=generator)
    actor_time_mask = torch.tensor(
        [
            [[True, False, True, True, True], [True] * 5, [False] * 5],
            [[True] * 5, [True, True, False, True, True], [False] * 5],
        ]
    )
    map_polylines = torch.randn(2, 2, 4, 3, generator=generator)
    map_point_mask = torch.tensor(
        [
            [[True, False, True, True], [False] * 4],
            [[True] * 4, [True, True, False, False]],
        ]
    )
    target_future = torch.randn(2, 6, 2, generator=generator).cumsum(dim=1)
    return {
        "actor_history": actor_history,
        "actor_time_mask": actor_time_mask,
        "actor_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "actor_type_id": torch.tensor([[1, 2, 0], [2, 1, 0]]),
        "actor_role_id": torch.tensor([[1, 2, 0], [2, 1, 0]]),
        "map_polylines": map_polylines,
        "map_point_mask": map_point_mask,
        "map_polyline_mask": torch.tensor([[True, False], [True, True]]),
        "map_type_id": torch.tensor([[1, 0], [2, 3]]),
        "target_actor_index": torch.tensor([0, 1]),
        "skill_id": torch.tensor([1, 2]),
        "skill_supervision_mask": torch.tensor([True, True]),
        "skill_parameters": torch.tensor([[0.2, 0.0, 0.8], [0.4, 0.6, 0.0]]),
        "parameter_mask": torch.tensor([[True, False, True], [True, True, False]]),
        "target_future": target_future,
        "target_future_mask": torch.ones(2, 6, dtype=torch.bool),
    }


def _context(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in batch.items()
        if key not in {"target_future", "target_future_mask"}
    }


def _assert_finite(output: CVAEOutput) -> None:
    tensors = [
        output.future_delta,
        output.future_position_local,
        output.prior_mean,
        output.prior_logvar,
        output.latent,
    ]
    if output.posterior_mean is not None:
        tensors.append(output.posterior_mean)
    if output.posterior_logvar is not None:
        tensors.append(output.posterior_logvar)
    assert all(torch.isfinite(tensor).all() for tensor in tensors)


def test_forward_train_and_prior_sampling_shapes_are_finite() -> None:
    model = _model().eval()
    batch = _batch()

    train_output = model.forward_train(
        batch,
        torch.Generator().manual_seed(3),
    )
    assert train_output.future_delta.shape == (2, 6, 2)
    assert train_output.future_position_local.shape == (2, 6, 2)
    assert train_output.prior_mean.shape == (2, 4)
    assert train_output.posterior_mean is not None
    assert train_output.posterior_mean.shape == (2, 4)
    _assert_finite(train_output)

    prior_output = model.sample_prior(
        _context(batch),
        3,
        torch.Generator().manual_seed(4),
    )
    assert prior_output.future_delta.shape == (2, 3, 6, 2)
    assert prior_output.future_position_local.shape == (2, 3, 6, 2)
    assert prior_output.latent.shape == (2, 3, 4)
    assert prior_output.posterior_mean is None
    assert prior_output.posterior_logvar is None
    _assert_finite(prior_output)


def test_masked_history_map_points_and_parameters_do_not_change_output() -> None:
    model = _model().eval()
    original = _batch()
    changed = {key: value.clone() for key, value in original.items()}
    changed["actor_history"][0, 0, 1] = 10000.0
    changed["actor_history"][:, 2] = -10000.0
    changed["actor_time_mask"][:, 2] = True
    changed["map_polylines"][0, 0, 1] = 20000.0
    changed["map_polylines"][0, 1] = -20000.0
    changed["map_point_mask"][0, 1] = True
    changed["skill_parameters"][0, 1] = 30000.0
    changed["skill_parameters"][1, 2] = -30000.0

    first = model.forward_train(original, torch.Generator().manual_seed(5))
    second = model.forward_train(changed, torch.Generator().manual_seed(5))

    assert torch.equal(first.future_position_local, second.future_position_local)
    assert torch.equal(first.prior_mean, second.prior_mean)
    assert torch.equal(first.posterior_mean, second.posterior_mean)


def test_prior_sampling_does_not_read_target_future() -> None:
    model = _model().eval()
    first_batch = _batch()
    second_batch = {key: value.clone() for key, value in first_batch.items()}
    second_batch["target_future"] = torch.full_like(second_batch["target_future"], 1e6)
    second_batch["target_future_mask"][:] = False

    first = model.sample_prior(first_batch, 2, torch.Generator().manual_seed(6))
    second = model.sample_prior(second_batch, 2, torch.Generator().manual_seed(6))

    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.future_position_local, second.future_position_local)


def test_prior_sampling_is_reproducible_and_different_latents_change_trajectory() -> None:
    model = _model().eval()
    context = _context(_batch())
    first = model.sample_prior(context, 3, torch.Generator().manual_seed(7))
    second = model.sample_prior(context, 3, torch.Generator().manual_seed(7))

    assert torch.equal(first.latent, second.latent)
    assert torch.equal(first.future_position_local, second.future_position_local)
    assert not torch.allclose(
        first.future_position_local[:, 0],
        first.future_position_local[:, 1],
    )


def test_checkpoint_reload_preserves_fixed_prior_output(tmp_path: Path) -> None:
    model = _model().eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context = _context(_batch())
    checkpoint = tmp_path / "model.pt"
    fingerprints = {"config": "fixed-cvae", "data": "fixed-input"}

    with torch.inference_mode():
        expected = model.sample_prior(
            context,
            3,
            torch.Generator().manual_seed(29),
        )
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(0, 0, 0, None, None),
        fingerprints=fingerprints,
    )

    restored_model = _model().eval()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    load_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_fingerprints=fingerprints,
        restore_rng=False,
    )
    with torch.inference_mode():
        actual = restored_model.sample_prior(
            context,
            3,
            torch.Generator().manual_seed(29),
        )

    assert torch.equal(actual.latent, expected.latent)
    maximum_absolute_error = max(
        float((actual_tensor - expected_tensor).abs().max())
        for actual_tensor, expected_tensor in (
            (actual.future_delta, expected.future_delta),
            (actual.future_position_local, expected.future_position_local),
            (actual.prior_mean, expected.prior_mean),
            (actual.prior_logvar, expected.prior_logvar),
        )
    )
    assert maximum_absolute_error <= 1e-6


def test_forward_train_supports_finite_backward_pass() -> None:
    model = _model().train()
    output = model.forward_train(_batch(), torch.Generator().manual_seed(8))
    assert output.posterior_mean is not None
    assert output.posterior_logvar is not None
    loss = (
        output.future_position_local.square().mean()
        + output.prior_mean.square().mean()
        + output.prior_logvar.square().mean()
        + output.posterior_mean.square().mean()
        + output.posterior_logvar.square().mean()
    )
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(bool(gradient.abs().sum() > 0) for gradient in gradients)

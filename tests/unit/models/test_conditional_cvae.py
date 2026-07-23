from __future__ import annotations

from pathlib import Path

import pytest
import torch

from skilldrive.models import CVAEOutput, ConditionalCVAE
from skilldrive.training.checkpoint import (
    TrainingProgress,
    load_checkpoint,
    save_checkpoint,
)


def _model(**changes: object) -> ConditionalCVAE:
    parameters: dict[str, object] = {
        "actor_feature_dim": 4,
        "map_feature_dim": 3,
        "num_actor_types": 4,
        "num_actor_roles": 4,
        "num_map_types": 4,
        "num_skills": 5,
        "parameter_dim": 3,
        "actor_type_embedding_dim": 4,
        "actor_role_embedding_dim": 4,
        "history_hidden_dim": 12,
        "map_type_embedding_dim": 4,
        "map_hidden_dim": 12,
        "interaction_hidden_dim": 16,
        "interaction_layers": 2,
        "interaction_heads": 4,
        "skill_embedding_dim": 4,
        "parameter_hidden_dim": 8,
        "latent_dim": 4,
        "decoder_hidden_dim": 16,
        "future_steps": 6,
        "dropout": 0.0,
    }
    parameters.update(changes)
    return ConditionalCVAE(**parameters)


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


def _first_decoder_previous_delta(
    model: ConditionalCVAE,
    call,
) -> torch.Tensor:
    decoder_inputs: list[torch.Tensor] = []

    def capture(_module, args) -> None:
        decoder_inputs.append(args[0][:, :2].detach().clone())

    handle = model.decoder_cell.register_forward_pre_hook(capture)
    try:
        call()
    finally:
        handle.remove()
    assert decoder_inputs
    return decoder_inputs[0]


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


def test_public_prior_parameters_match_prior_sampling_without_decoding() -> None:
    model = _model().eval()
    context = _context(_batch())

    mean, logvar = model.prior_parameters(context)
    sampled = model.sample_prior_from_noise(
        context,
        torch.zeros(2, 1, model.latent_dim),
    )

    assert mean.shape == (2, model.latent_dim)
    assert logvar.shape == (2, model.latent_dim)
    assert torch.equal(mean, sampled.prior_mean)
    assert torch.equal(logvar, sampled.prior_logvar)


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


def test_history_velocity_initializes_train_and_all_prior_decoder_paths() -> None:
    model = _model(
        decoder_initial_delta_mode="history_velocity",
        sample_period_s=0.25,
    ).eval()
    batch = _batch()
    context = _context(batch)
    batch_indices = torch.arange(2)
    expected = (
        batch["actor_history"][
            batch_indices,
            batch["target_actor_index"],
            -1,
            2:4,
        ]
        * 0.25
    )

    train_initial = _first_decoder_previous_delta(
        model,
        lambda: model.forward_train(batch, torch.Generator().manual_seed(41)),
    )
    torch.testing.assert_close(train_initial, expected, rtol=0.0, atol=0.0)

    prior_initial = _first_decoder_previous_delta(
        model,
        lambda: model.sample_prior(
            context,
            3,
            torch.Generator().manual_seed(43),
        ),
    )
    expected_prior = expected[:, None, :].expand(-1, 3, -1).reshape(-1, 2)
    torch.testing.assert_close(prior_initial, expected_prior, rtol=0.0, atol=0.0)

    explicit_initial = _first_decoder_previous_delta(
        model,
        lambda: model.sample_prior_from_noise(
            context,
            torch.zeros(2, 2, model.latent_dim),
        ),
    )
    expected_explicit = expected[:, None, :].expand(-1, 2, -1).reshape(-1, 2)
    torch.testing.assert_close(explicit_initial, expected_explicit, rtol=0.0, atol=0.0)


def test_explicit_zero_mode_preserves_default_decoder_semantics() -> None:
    default = _model().eval()
    explicit_zero = _model(
        decoder_initial_delta_mode="zero",
        sample_period_s=0.25,
    ).eval()
    explicit_zero.load_state_dict(default.state_dict())
    context = _context(_batch())
    noise = torch.randn(2, 3, default.latent_dim, generator=torch.Generator().manual_seed(47))

    expected = default.sample_prior_from_noise(context, noise)
    actual = explicit_zero.sample_prior_from_noise(context, noise)

    assert torch.equal(actual.latent, expected.latent)
    assert torch.equal(actual.future_delta, expected.future_delta)
    assert torch.equal(actual.future_position_local, expected.future_position_local)


def test_history_velocity_requires_valid_final_target_history() -> None:
    model = _model(decoder_initial_delta_mode="history_velocity").eval()
    batch = _batch()
    batch["actor_time_mask"][1, 1, -1] = False

    with pytest.raises(ValueError, match="valid final target history"):
        model.sample_prior(
            _context(batch),
            2,
            torch.Generator().manual_seed(53),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"decoder_initial_delta_mode": "unsupported"}, "decoder_initial_delta_mode"),
        ({"sample_period_s": 0.0}, "sample_period_s"),
        ({"sample_period_s": float("nan")}, "sample_period_s"),
        (
            {
                "actor_feature_dim": 3,
                "decoder_initial_delta_mode": "history_velocity",
            },
            "actor_feature_dim",
        ),
    ],
)
def test_decoder_initialization_contract_rejects_invalid_configuration(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _model(**changes)


def test_explicit_prior_noise_is_batch_partition_independent() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    context = _context(_batch())
    noise = torch.randn(2, 3, model.latent_dim, generator=torch.Generator().manual_seed(31))

    whole = model.sample_prior_from_noise(context, noise)
    partitions = [
        model.sample_prior_from_noise(
            {name: value[index : index + 1] for name, value in context.items()},
            noise[index : index + 1],
        )
        for index in range(2)
    ]

    for name in (
        "future_delta",
        "future_position_local",
        "prior_mean",
        "prior_logvar",
        "latent",
    ):
        partitioned = torch.cat([getattr(output, name) for output in partitions], dim=0)
        torch.testing.assert_close(
            getattr(whole, name),
            partitioned,
            rtol=0.0,
            atol=1e-6,
        )


def test_explicit_prior_noise_matches_generator_sampling() -> None:
    model = _model().eval()
    context = _context(_batch())
    sampled = model.sample_prior(
        context,
        3,
        torch.Generator().manual_seed(37),
    )
    noise = torch.randn(
        2,
        3,
        model.latent_dim,
        generator=torch.Generator().manual_seed(37),
    )
    explicit = model.sample_prior_from_noise(context, noise)

    assert torch.equal(sampled.latent, explicit.latent)
    assert torch.equal(sampled.future_position_local, explicit.future_position_local)


@pytest.mark.parametrize(
    ("noise", "message"),
    [
        (torch.zeros(2, 4), "shape"),
        (torch.zeros(1, 2, 4), "shape"),
        (torch.zeros(2, 0, 4), "at least one sample"),
        (torch.zeros(2, 2, 5), "shape"),
        (torch.zeros(2, 2, 4, dtype=torch.int64), "floating-point"),
        (torch.full((2, 2, 4), float("nan")), "finite"),
    ],
)
def test_explicit_prior_noise_rejects_invalid_values(
    noise: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _model().eval().sample_prior_from_noise(_context(_batch()), noise)


def test_explicit_prior_noise_requires_tensor() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        _model().eval().sample_prior_from_noise(  # type: ignore[arg-type]
            _context(_batch()),
            [[[0.0]]],
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

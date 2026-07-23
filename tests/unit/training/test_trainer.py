from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

import skilldrive.training.trainer as trainer_module
from skilldrive.models import CVAEOutput
from skilldrive.training import load_cvae_config
from skilldrive.training.config import LossConfig, TrainingConfig
from skilldrive.training.system_monitor import SystemMetrics
from skilldrive.training.trainer import (
    _amp_policy,
    benchmark_training,
    compute_cvae_loss,
    evaluate,
    kl_anneal_weight,
    move_batch_to_device,
    train_epoch,
    train_optimizer_step,
)


class _ToyCVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward_train(
        self,
        batch: dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> CVAEOutput:
        prediction = batch.get("posterior_prediction", batch["prediction_template"])
        prediction = prediction * self.scale
        batch_size = prediction.shape[0]
        posterior_mean = batch.get(
            "posterior_mean",
            torch.zeros(batch_size, 1, device=prediction.device),
        )
        zeros = torch.zeros_like(posterior_mean)
        return CVAEOutput(
            future_delta=prediction,
            future_position_local=prediction,
            prior_mean=zeros,
            prior_logvar=zeros,
            posterior_mean=posterior_mean + self.scale * 0.0,
            posterior_logvar=zeros,
            latent=torch.zeros_like(posterior_mean),
        )

    def sample_prior(
        self,
        batch: dict[str, torch.Tensor],
        num_samples: int,
        generator: torch.Generator,
    ) -> CVAEOutput:
        predictions = batch.get("prior_predictions")
        if predictions is None:
            noise = torch.randn(
                batch["target_future"].shape[0],
                num_samples,
                batch["target_future"].shape[1],
                2,
                generator=generator,
                device=self.scale.device,
            )
            predictions = noise * self.scale
        else:
            predictions = predictions[:, :num_samples] * self.scale
        batch_size = predictions.shape[0]
        zeros = torch.zeros(batch_size, 1, device=predictions.device)
        return CVAEOutput(
            future_delta=predictions,
            future_position_local=predictions,
            prior_mean=zeros,
            prior_logvar=zeros,
            posterior_mean=None,
            posterior_logvar=None,
            latent=torch.zeros(batch_size, num_samples, 1, device=predictions.device),
        )

    def prior_parameters(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(batch["skill_id"].shape[0], 1, device=self.scale.device)
        return zeros, zeros


def _loss_config(**changes: object) -> LossConfig:
    base = LossConfig(
        reconstruction="smooth_l1",
        endpoint_weight=1.0,
        kl_max_weight=0.1,
        kl_warmup_steps=10,
        map_soft_weight=0.0,
        collision_soft_weight=0.0,
    )
    return replace(base, **changes)


def _training_config(accumulation: int = 1) -> TrainingConfig:
    return TrainingConfig(
        seed=2026,
        device="cpu",
        amp=False,
        allow_tf32=False,
        batch_size=2,
        gradient_accumulation_steps=accumulation,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        learning_rate=0.01,
        weight_decay=0.0,
        gradient_clip_norm=5.0,
        development_max_epochs=1,
        formal_max_epochs=1,
        early_stopping_patience=0,
        validation_every_epochs=1,
        checkpoint_every_steps=1,
        prior_samples=2,
        best_metric="min_fde_6",
    )


def _batch(
    prediction: float = 1.0,
    *,
    batch_size: int = 1,
    future_steps: int = 2,
) -> dict[str, torch.Tensor]:
    target = torch.zeros(batch_size, future_steps, 2)
    template = torch.zeros_like(target)
    template[..., 0] = prediction
    history = torch.zeros(batch_size, 1, 2, 6)
    history[:, 0, -1, 2] = 1.0
    return {
        "prediction_template": template,
        "actor_history": history,
        "actor_time_mask": torch.ones(batch_size, 1, 2, dtype=torch.bool),
        "actor_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "target_actor_index": torch.zeros(batch_size, dtype=torch.long),
        "target_future": target,
        "target_future_mask": torch.ones(batch_size, future_steps, dtype=torch.bool),
    }


def _batch_with_auxiliary_context(
    *,
    prediction_x: float,
    map_x: float = 0.0,
    actor_x: float = 0.0,
    actor_velocity_x: float = 0.0,
    target_velocity_x: float = 1.0,
    future_steps: int = 2,
) -> dict[str, torch.Tensor]:
    batch = _batch(prediction=prediction_x, future_steps=future_steps)
    batch["target_future"] = batch["prediction_template"].clone()
    batch["map_polylines"] = torch.tensor(
        [[[[map_x, 0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    batch["map_point_mask"] = torch.ones(1, 1, 1, dtype=torch.bool)
    batch["map_polyline_mask"] = torch.ones(1, 1, dtype=torch.bool)
    history = torch.zeros(1, 2, 2, 6)
    history[:, 0, -1, 2] = target_velocity_x
    history[:, 1, :, 0] = actor_x
    history[:, 1, -1, 2] = actor_velocity_x
    batch["actor_history"] = history
    batch["actor_time_mask"] = torch.ones(1, 2, 2, dtype=torch.bool)
    batch["actor_mask"] = torch.ones(1, 2, dtype=torch.bool)
    return batch


def _without_auxiliary_context(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result = {key: value.clone() for key, value in batch.items()}
    result["map_point_mask"].zero_()
    result["map_polyline_mask"].zero_()
    result["actor_mask"][:, 1] = False
    return result


def _concatenate_batches(
    *batches: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: torch.cat([batch[key] for batch in batches], dim=0)
        for key in batches[0]
    }


def test_batch_move_and_kl_schedule_are_explicit() -> None:
    batch = {"tensor": torch.ones(2), "sample_id": "scene"}
    moved = move_batch_to_device(batch, "cpu")

    assert moved["tensor"].device.type == "cpu"
    assert moved["sample_id"] == "scene"
    assert kl_anneal_weight(0, 0.1, 10) == 0.0
    assert kl_anneal_weight(5, 0.1, 10) == pytest.approx(0.05)
    assert kl_anneal_weight(20, 0.1, 10) == pytest.approx(0.1)
    assert kl_anneal_weight(0, 0.1, 0) == pytest.approx(0.1)


def test_amp_policy_prefers_bfloat16_and_falls_back_to_scaled_float16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    preferred = _amp_policy(torch.device("cuda"), True)
    assert preferred.enabled
    assert preferred.dtype is torch.bfloat16
    assert not preferred.use_scaler

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    fallback = _amp_policy(torch.device("cuda"), True)
    assert fallback.enabled
    assert fallback.dtype is torch.float16
    assert fallback.use_scaler

    cpu = _amp_policy(torch.device("cpu"), True)
    assert not cpu.enabled
    assert not cpu.use_scaler


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_optimizer_step_stays_finite() -> None:
    model = _ToyCVAE().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = train_optimizer_step(
        model,
        optimizer,
        [_batch()],
        device="cuda",
        loss_config=_loss_config(),
        global_step=1,
        gradient_clip_norm=5.0,
        generator=torch.Generator(device="cuda").manual_seed(11),
        amp=True,
    )

    assert math.isfinite(result.total_loss)
    assert math.isfinite(result.gradient_norm)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_masked_loss_matches_hand_calculation() -> None:
    model = _ToyCVAE()
    batch = _batch()
    batch["prediction_template"][0, 0, 0] = 1.0
    batch["prediction_template"][0, 1, 0] = 3.0
    batch["posterior_mean"] = torch.ones(1, 1)
    output = model.forward_train(batch, torch.Generator().manual_seed(1))

    loss = compute_cvae_loss(
        output,
        batch,
        _loss_config(endpoint_weight=2.0),
        global_step=5,
    )

    assert loss.reconstruction.item() == pytest.approx(0.75)
    assert loss.endpoint.item() == pytest.approx(1.25)
    assert loss.kl.item() == pytest.approx(0.5)
    assert loss.total.item() == pytest.approx(3.275)


@pytest.mark.parametrize(
    ("config_changes", "prediction_x", "expected_loss", "gradient_sign"),
    [
        ({"map_soft_weight": 1.0}, 10.0, 25.0, 1),
        ({"collision_soft_weight": 1.0}, 1.0, 1.0, -1),
    ],
)
def test_auxiliary_soft_losses_are_finite_and_backpropagate(
    config_changes: dict[str, float],
    prediction_x: float,
    expected_loss: float,
    gradient_sign: int,
) -> None:
    model = _ToyCVAE()
    batch = _batch_with_auxiliary_context(prediction_x=prediction_x)
    config = _loss_config(
        endpoint_weight=0.0,
        kl_max_weight=0.0,
        **config_changes,
    )
    output = model.forward_train(batch, torch.Generator().manual_seed(12))

    loss = compute_cvae_loss(output, batch, config, global_step=0)
    loss.total.backward()

    assert loss.total.item() == pytest.approx(expected_loss)
    assert model.scale.grad is not None
    assert bool(torch.isfinite(model.scale.grad))
    assert math.copysign(1.0, model.scale.grad.item()) == gradient_sign

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    result = train_optimizer_step(
        model,
        optimizer,
        [batch],
        device="cpu",
        loss_config=config,
        global_step=0,
        gradient_clip_norm=1000.0,
        generator=torch.Generator().manual_seed(13),
    )
    assert math.isfinite(result.total_loss)
    assert result.gradient_norm > 0.0


def test_auxiliary_soft_losses_respect_future_masks() -> None:
    model = _ToyCVAE()
    batch = _batch_with_auxiliary_context(prediction_x=1.0)
    batch["target_future_mask"][0, 1] = False
    batch["target_future"][0, 1] = float("nan")
    batch["prediction_template"][0, 1, 0] = 1000.0
    output = model.forward_train(batch, torch.Generator().manual_seed(14))

    loss = compute_cvae_loss(
        output,
        batch,
        _loss_config(
            endpoint_weight=0.0,
            kl_max_weight=0.0,
            map_soft_weight=1.0,
            collision_soft_weight=1.0,
        ),
        global_step=0,
    )
    loss.total.backward()

    assert loss.total.item() == pytest.approx(1.0)
    assert model.scale.grad is not None
    assert bool(torch.isfinite(model.scale.grad))


def test_auxiliary_soft_losses_are_zero_without_context() -> None:
    model = _ToyCVAE()
    batch = _batch_with_auxiliary_context(prediction_x=1.0)
    batch["map_polylines"] = torch.full((1, 1, 1, 4), float("nan"))
    batch["map_point_mask"] = torch.zeros(1, 1, 1, dtype=torch.bool)
    batch["map_polyline_mask"] = torch.zeros(1, 1, dtype=torch.bool)
    batch["actor_mask"][0, 1] = False
    batch["actor_history"][0, 1, :, :2] = float("nan")
    output = model.forward_train(batch, torch.Generator().manual_seed(15))

    loss = compute_cvae_loss(
        output,
        batch,
        _loss_config(
            endpoint_weight=0.0,
            kl_max_weight=0.0,
            map_soft_weight=1.0,
            collision_soft_weight=1.0,
        ),
        global_step=0,
    )
    loss.total.backward()

    assert loss.total.item() == pytest.approx(0.0)
    assert model.scale.grad is not None
    assert bool(torch.isfinite(model.scale.grad))


def test_collision_soft_loss_distinguishes_actors_driving_in_and_away() -> None:
    prediction = torch.zeros(1, 2, 2)
    driving_in = _batch_with_auxiliary_context(
        prediction_x=0.0,
        actor_x=2.5,
        actor_velocity_x=-5.0,
        target_velocity_x=0.0,
    )
    driving_away = _batch_with_auxiliary_context(
        prediction_x=0.0,
        actor_x=1.5,
        actor_velocity_x=5.0,
        target_velocity_x=0.0,
    )

    entering_sum, entering_count = trainer_module._collision_soft_penalty_sum(
        prediction,
        driving_in,
    )
    leaving_sum, leaving_count = trainer_module._collision_soft_penalty_sum(
        prediction,
        driving_away,
    )

    assert entering_count == leaving_count == 2
    assert (entering_sum / entering_count).item() == pytest.approx(0.125)
    assert leaving_sum.item() == pytest.approx(0.0)


def test_exact_collision_has_finite_nonzero_repulsion_gradient() -> None:
    prediction = torch.zeros(1, 2, 2, requires_grad=True)
    batch = _batch_with_auxiliary_context(
        prediction_x=0.0,
        actor_x=0.0,
        actor_velocity_x=0.0,
        target_velocity_x=0.0,
    )

    penalty_sum, eligible_count = trainer_module._collision_soft_penalty_sum(
        prediction,
        batch,
    )
    loss = penalty_sum / eligible_count
    loss.backward()

    assert eligible_count == 2
    assert math.isfinite(loss.item())
    assert loss.item() > 0.0
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool((torch.linalg.vector_norm(prediction.grad, dim=-1) > 0.0).all())


@pytest.mark.parametrize(
    ("config_changes", "prediction_x", "expected_loss"),
    [
        ({"map_soft_weight": 1.0}, 10.0, 25.0),
        ({"collision_soft_weight": 1.0}, 1.0, 1.0),
    ],
)
def test_auxiliary_normalization_is_invariant_to_microbatch_partition(
    config_changes: dict[str, float],
    prediction_x: float,
    expected_loss: float,
) -> None:
    context = _batch_with_auxiliary_context(prediction_x=prediction_x)
    no_context = _without_auxiliary_context(context)
    combined = _concatenate_batches(context, no_context)
    config = _loss_config(
        endpoint_weight=0.0,
        kl_max_weight=0.0,
        **config_changes,
    )
    combined_model = _ToyCVAE()
    split_model = _ToyCVAE()
    combined_optimizer = torch.optim.SGD(combined_model.parameters(), lr=0.001)
    split_optimizer = torch.optim.SGD(split_model.parameters(), lr=0.001)

    combined_loss = compute_cvae_loss(
        combined_model.forward_train(combined, torch.Generator().manual_seed(18)),
        combined,
        config,
        global_step=0,
    )

    combined_result = train_optimizer_step(
        combined_model,
        combined_optimizer,
        [combined],
        device="cpu",
        loss_config=config,
        global_step=0,
        gradient_clip_norm=1000.0,
        generator=torch.Generator().manual_seed(18),
    )
    split_result = train_optimizer_step(
        split_model,
        split_optimizer,
        [context, no_context],
        device="cpu",
        loss_config=config,
        global_step=0,
        gradient_clip_norm=1000.0,
        generator=torch.Generator().manual_seed(18),
    )

    assert combined_loss.total.item() == pytest.approx(expected_loss)
    assert combined_result.total_loss == pytest.approx(expected_loss)
    assert split_result.total_loss == pytest.approx(expected_loss)
    assert combined_result.gradient_norm == pytest.approx(split_result.gradient_norm)
    assert combined_model.scale.item() == pytest.approx(split_model.scale.item())


def test_zero_auxiliary_weights_skip_all_auxiliary_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_call(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("zero auxiliary weights must skip auxiliary computation")

    monkeypatch.setattr(trainer_module, "_map_soft_penalty_sum", unexpected_call)
    monkeypatch.setattr(trainer_module, "_collision_soft_penalty_sum", unexpected_call)
    model = _ToyCVAE()
    batch = _batch()
    batch["prediction_template"][0, 0, 0] = 1.0
    batch["prediction_template"][0, 1, 0] = 3.0
    batch["posterior_mean"] = torch.ones(1, 1)
    config = _loss_config(endpoint_weight=2.0)
    output = model.forward_train(batch, torch.Generator().manual_seed(16))

    loss = compute_cvae_loss(output, batch, config, global_step=5)
    assert loss.total.item() == pytest.approx(3.275)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    result = train_optimizer_step(
        model,
        optimizer,
        [batch],
        device="cpu",
        loss_config=config,
        global_step=5,
        gradient_clip_norm=5.0,
        generator=torch.Generator().manual_seed(17),
    )
    assert result.total_loss == pytest.approx(3.275)


def test_optimizer_step_accumulates_by_elements_not_batch_means() -> None:
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    first = _batch(prediction=3.0, future_steps=1)
    second = _batch(prediction=1.0, future_steps=3)

    result = train_optimizer_step(
        model,
        optimizer,
        [first, second],
        device="cpu",
        loss_config=_loss_config(endpoint_weight=0.0, kl_max_weight=0.0),
        global_step=0,
        gradient_clip_norm=5.0,
        generator=torch.Generator().manual_seed(2),
    )

    assert result.microbatch_count == 2
    assert result.sums.valid_point_count == 4
    assert result.sums.reconstruction_loss == pytest.approx(0.5)
    assert result.next_global_step == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_epoch_flushes_final_partial_accumulation_group() -> None:
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    result = train_epoch(
        model,
        optimizer,
        [_batch(), _batch(), _batch()],
        device="cpu",
        loss_config=_loss_config(kl_max_weight=0.0),
        training_config=_training_config(accumulation=2),
        global_step=4,
        generator=torch.Generator().manual_seed(3),
    )

    assert result.optimizer_steps == 2
    assert result.microbatch_count == 3
    assert result.next_global_step == 6
    assert result.sums.sample_count == 3


def test_evaluate_reports_exact_posterior_prior_and_constant_velocity_metrics() -> None:
    model = _ToyCVAE()
    batch = _batch()
    batch["target_future"] = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    batch["posterior_prediction"] = torch.tensor([[[2.0, 0.0], [4.0, 0.0]]])
    batch["prior_predictions"] = torch.tensor(
        [[[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [2.0, 0.0]]]]
    )
    training_generator = torch.Generator().manual_seed(99)
    state = training_generator.get_state().clone()

    result = evaluate(
        model,
        [batch],
        device="cpu",
        prior_samples=2,
        sample_period_s=1.0,
        evaluation_seed=10,
    )

    assert result.posterior.ade == pytest.approx(1.5)
    assert result.posterior.fde == pytest.approx(2.0)
    assert result.prior.ade == pytest.approx(0.0)
    assert result.prior.fde == pytest.approx(0.0)
    assert result.constant_velocity.ade == pytest.approx(0.0)
    assert result.constant_velocity.fde == pytest.approx(0.0)
    assert result.posterior_kl == pytest.approx(0.0)
    assert torch.equal(training_generator.get_state(), state)


def test_evaluate_reports_deterministic_globally_normalized_validation_loss() -> None:
    model = _ToyCVAE()
    short = _batch()
    short["target_future_mask"] = torch.tensor([[True, False]])
    short["posterior_prediction"] = torch.tensor(
        [[[1.0, 0.0], [99.0, 99.0]]]
    )
    short["posterior_mean"] = torch.tensor([[1.0]])
    long = _batch()
    long["posterior_prediction"] = torch.tensor(
        [[[3.0, 0.0], [3.0, 0.0]]]
    )
    long["posterior_mean"] = torch.tensor([[2.0]])
    loss_config = _loss_config(endpoint_weight=2.0)

    first = evaluate(
        model,
        [short, long],
        device="cpu",
        prior_samples=2,
        sample_period_s=1.0,
        evaluation_seed=10,
        loss_config=loss_config,
        global_step=5,
    )
    second = evaluate(
        model,
        [short, long],
        device="cpu",
        prior_samples=2,
        sample_period_s=1.0,
        evaluation_seed=10,
        loss_config=loss_config,
        global_step=5,
    )

    assert first.validation_loss is not None
    assert second.validation_loss is not None
    assert first.validation_loss == second.validation_loss
    assert first.validation_loss.reconstruction_loss == pytest.approx(5.5 / 6.0)
    assert first.validation_loss.endpoint_loss == pytest.approx(3.0 / 4.0)
    assert first.validation_loss.kl_loss == pytest.approx(1.25)
    assert first.validation_loss.kl_weight == pytest.approx(0.05)
    assert first.validation_loss.total_loss == pytest.approx(
        5.5 / 6.0 + 2.0 * 3.0 / 4.0 + 0.05 * 1.25
    )
    assert first.validation_loss.sums.valid_point_count == 3
    assert first.validation_loss.sums.reconstruction_element_count == 6


def test_nonfinite_loss_is_rejected_before_optimizer_step() -> None:
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _batch()
    batch["prediction_template"][0, 0, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="predicted future"):
        train_optimizer_step(
            model,
            optimizer,
            [batch],
            device="cpu",
            loss_config=_loss_config(),
            global_step=0,
            gradient_clip_norm=5.0,
            generator=torch.Generator().manual_seed(4),
        )


def test_nonfinite_gradient_is_rejected_by_the_fused_norm_check() -> None:
    model = _ToyCVAE()
    model.scale.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    with pytest.raises(FloatingPointError, match="gradients contain"):
        train_optimizer_step(
            model,
            optimizer,
            [_batch()],
            device="cpu",
            loss_config=_loss_config(),
            global_step=0,
            gradient_clip_norm=5.0,
            generator=torch.Generator().manual_seed(4),
        )


def test_nonfinite_parameter_is_rejected_after_optimizer_step() -> None:
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_step = optimizer.step

    def corrupt_parameter(*args: object, **kwargs: object) -> object:
        result = original_step(*args, **kwargs)
        with torch.no_grad():
            model.scale.fill_(float("inf"))
        return result

    optimizer.step = corrupt_parameter  # type: ignore[method-assign]
    with pytest.raises(FloatingPointError, match="parameters contain"):
        train_optimizer_step(
            model,
            optimizer,
            [_batch()],
            device="cpu",
            loss_config=_loss_config(),
            global_step=0,
            gradient_clip_norm=5.0,
            generator=torch.Generator().manual_seed(4),
        )


def test_benchmark_excludes_startup_and_returns_fixed_stable_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeMonitor:
        def __init__(self, *, gpu_index: int | None) -> None:
            assert gpu_index is None

        def start(self) -> None:
            events.append("monitor_start")

        def begin_window(self) -> None:
            events.append("window_start")

        def end_window(self) -> SystemMetrics:
            events.append("window_end")
            return SystemMetrics(
                cpu_metrics_available=True,
                cpu_busy_percent=75.0,
                cpu_iowait_percent=5.0,
                gpu_utilization_available=False,
                gpu_utilization_mean_percent=None,
                gpu_utilization_p50_percent=None,
                gpu_utilization_p95_percent=None,
                gpu_utilization_sample_count=0,
            )

        def stop(self) -> None:
            events.append("monitor_stop")

    real_optimizer_step = trainer_module.train_optimizer_step

    def tracked_optimizer_step(*args: object, **kwargs: object):
        events.append("step")
        return real_optimizer_step(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "SystemMonitor", FakeMonitor)
    monkeypatch.setattr(trainer_module, "train_optimizer_step", tracked_optimizer_step)
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    result = benchmark_training(
        model,
        optimizer,
        [_batch()],
        device="cpu",
        loss_config=_loss_config(kl_max_weight=0.0),
        training_config=_training_config(),
        global_step=0,
        generator=torch.Generator().manual_seed(5),
        warmup_steps=1,
        measured_steps=3,
    )

    assert result.startup_seconds >= 0.0
    assert result.warmup_seconds >= 0.0
    assert len(result.step_seconds) == 3
    assert len(result.data_wait_seconds) == 3
    assert result.measured_samples == 3
    assert result.next_global_step == 4
    assert 0.0 <= result.data_wait_fraction <= 1.0
    assert result.p50_step_seconds > 0.0
    assert result.p95_step_seconds >= result.p50_step_seconds
    assert result.samples_per_second > 0.0
    assert result.cpu_metrics_available
    assert result.cpu_busy_percent == pytest.approx(75.0)
    assert result.cpu_iowait_percent == pytest.approx(5.0)
    assert not result.gpu_utilization_available
    assert result.gpu_utilization_mean_percent is None
    assert result.gpu_utilization_p50_percent is None
    assert result.gpu_utilization_p95_percent is None
    assert result.gpu_utilization_sample_count == 0
    assert result.monitor_overhead_seconds >= 0.0
    assert events == [
        "step",
        "monitor_start",
        "window_start",
        "step",
        "step",
        "step",
        "window_end",
        "monitor_stop",
    ]


def test_repair_optimizer_step_records_motion_and_observed_ranking_losses() -> None:
    repair = load_cvae_config(
        Path("configs/models/cvae_generation_repair_v1.yaml")
    ).repair
    assert repair is not None
    model = _ToyCVAE()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _batch(prediction=0.5, batch_size=2, future_steps=4)
    batch.update(
        actor_role_id=torch.tensor([[3], [2]]),
        skill_id=torch.tensor([1, 0]),
        skill_supervision_mask=torch.tensor([True, False]),
        skill_parameters=torch.tensor([[1.0], [0.0]]),
        parameter_mask=torch.tensor([[True], [False]]),
    )

    result = train_optimizer_step(
        model,
        optimizer,
        [batch],
        device="cpu",
        loss_config=_loss_config(),
        global_step=1,
        gradient_clip_norm=5.0,
        generator=torch.Generator().manual_seed(1),
        repair_config=repair,
        sample_period_s=0.1,
    )

    assert result.sums.observed_condition_count == 1
    assert result.sums.condition_ranking_loss == pytest.approx(
        repair.condition_ranking.margin_per_latent_dim
    )
    assert result.sums.seam_velocity_element_count == 4
    assert result.sums.velocity_element_count == 16
    assert result.sums.acceleration_element_count > 0
    assert result.sums.jerk_element_count > 0
    assert math.isfinite(result.total_loss)

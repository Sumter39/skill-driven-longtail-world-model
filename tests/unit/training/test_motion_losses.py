from __future__ import annotations

import pytest
import torch

from skilldrive.training.motion_losses import (
    compute_motion_loss_sums,
    motion_loss_element_counts,
)


def _batch(
    *,
    target_future: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if target_future is None:
        target_future = torch.tensor([[[0.2, 0.0], [0.4, 0.0], [0.6, 0.0]]])
    actor_history = torch.zeros(1, 2, 2, 4)
    actor_history[0, 1, 0, 2:4] = torch.tensor([2.0, 0.0])
    actor_history[0, 1, 1, 2:4] = torch.tensor([2.0, 0.0])
    return {
        "actor_history": actor_history,
        "actor_time_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "actor_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_actor_index": torch.tensor([1]),
        "target_future": target_future,
        "target_future_mask": torch.ones(1, 3, dtype=torch.bool),
    }


def test_constant_velocity_motion_has_zero_losses_and_full_counts() -> None:
    batch = _batch()
    result = compute_motion_loss_sums(batch["target_future"].clone(), batch)

    for component in (
        result.seam_velocity,
        result.velocity,
        result.acceleration,
        result.jerk,
    ):
        assert component.loss_sum.item() == pytest.approx(0.0)
        assert component.mean.item() == pytest.approx(0.0)

    assert result.seam_velocity.element_count == 2
    assert result.velocity.element_count == 6
    assert result.acceleration.element_count == 6
    assert result.jerk.element_count == 6


def test_first_future_velocity_jump_is_seen_by_all_seam_derivative_losses() -> None:
    batch = _batch()
    prediction = batch["target_future"].clone()
    prediction[0, 0, 0] = 0.0

    result = compute_motion_loss_sums(prediction, batch)

    assert result.seam_velocity.loss_sum.item() == pytest.approx(1.5)
    assert result.velocity.loss_sum.item() > 0.0
    assert result.acceleration.loss_sum.item() > 0.0
    assert result.jerk.loss_sum.item() > 0.0


def test_future_masks_require_adjacent_positions_for_derivatives() -> None:
    batch = _batch()
    batch["target_future_mask"] = torch.tensor([[True, False, True]])
    batch["target_future"][0, 1] = float("nan")
    prediction = batch["target_future"].clone()

    result = compute_motion_loss_sums(prediction, batch)
    counts = motion_loss_element_counts(batch)

    assert result.seam_velocity.element_count == 2
    assert result.velocity.element_count == 2
    assert result.acceleration.element_count == 2
    assert result.jerk.element_count == 2
    assert counts.seam_velocity == result.seam_velocity.element_count
    assert counts.velocity == result.velocity.element_count
    assert counts.acceleration == result.acceleration.element_count
    assert counts.jerk == result.jerk.element_count
    assert all(
        torch.isfinite(component.loss_sum)
        for component in (
            result.seam_velocity,
            result.velocity,
            result.acceleration,
            result.jerk,
        )
    )


def test_missing_frame48_masks_only_history_dependent_first_jerk() -> None:
    batch = _batch()
    batch["actor_time_mask"][0, 1, 0] = False
    batch["actor_history"][0, 1, 0, 2:4] = float("nan")

    result = compute_motion_loss_sums(batch["target_future"].clone(), batch)

    assert result.seam_velocity.element_count == 2
    assert result.velocity.element_count == 6
    assert result.acceleration.element_count == 6
    assert result.jerk.element_count == 4
    assert result.jerk.loss_sum.item() == pytest.approx(0.0)


def test_robust_motion_losses_keep_outlier_gradients_finite() -> None:
    batch = _batch()
    prediction = torch.full_like(batch["target_future"], 1.0e6, requires_grad=True)

    result = compute_motion_loss_sums(prediction, batch)
    total = sum(
        component.mean
        for component in (
            result.seam_velocity,
            result.velocity,
            result.acceleration,
            result.jerk,
        )
    )
    total.backward()

    assert torch.isfinite(total)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert bool(prediction.grad.abs().sum() > 0.0)


def test_valid_nonfinite_motion_inputs_are_rejected() -> None:
    batch = _batch()
    prediction = batch["target_future"].clone()
    prediction[0, 1, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="predicted future"):
        compute_motion_loss_sums(prediction, batch)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("sample_period_s", 0.0),
        ("seam_velocity_beta_mps", 0.0),
        ("velocity_beta_mps", float("nan")),
        ("acceleration_beta_mps2", -1.0),
        ("jerk_beta_mps3", float("inf")),
    ],
)
def test_motion_loss_contract_rejects_nonpositive_or_nonfinite_scales(
    keyword: str,
    value: float,
) -> None:
    batch = _batch()
    with pytest.raises(ValueError, match=keyword):
        compute_motion_loss_sums(
            batch["target_future"].clone(),
            batch,
            **{keyword: value},
        )

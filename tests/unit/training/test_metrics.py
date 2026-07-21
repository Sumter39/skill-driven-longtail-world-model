import pytest
import torch

from skilldrive.training.metrics import (
    DisplacementSums,
    constant_velocity_prediction,
    displacement_sums,
    gaussian_kl_divergence,
    multimodal_displacement_sums,
)


def test_displacement_sums_use_all_valid_points_and_each_last_valid_frame() -> None:
    target = torch.zeros((2, 3, 2))
    prediction = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0], [9.0, 9.0]],
            [[0.0, 0.0], [6.0, 8.0], [0.0, 5.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, True, True]])

    result = displacement_sums(prediction, target, mask)

    assert result.ade_error_sum == pytest.approx(20.0)
    assert result.valid_point_count == 4
    assert result.ade == pytest.approx(5.0)
    assert result.fde_error_sum == pytest.approx(5.0)
    assert result.valid_sample_count == 2
    assert result.fde == pytest.approx(2.5)


def test_multimodal_metrics_select_ade_and_fde_modes_independently() -> None:
    target = torch.zeros((1, 2, 2))
    predictions = torch.tensor(
        [
            [
                [[0.0, 0.0], [10.0, 0.0]],
                [[6.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    mask = torch.ones((1, 2), dtype=torch.bool)

    result = multimodal_displacement_sums(predictions, target, mask)

    assert result.ade == pytest.approx(3.0)
    assert result.fde == pytest.approx(0.0)
    assert result.valid_point_count == 1
    assert result.valid_sample_count == 1


def test_displacement_sums_add_without_averaging_batch_means() -> None:
    first = DisplacementSums(ade_error_sum=9.0, valid_point_count=3, fde_error_sum=4.0, valid_sample_count=1)
    second = DisplacementSums(ade_error_sum=1.0, valid_point_count=1, fde_error_sum=6.0, valid_sample_count=2)

    total = first + second

    assert total.ade == pytest.approx(2.5)
    assert total.fde == pytest.approx(10.0 / 3.0)


def test_constant_velocity_prediction_uses_one_based_future_steps() -> None:
    prediction = constant_velocity_prediction(
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[2.0, -1.0]]),
        future_steps=3,
        sample_period_s=0.5,
    )

    torch.testing.assert_close(
        prediction,
        torch.tensor([[[1.0, 0.5], [2.0, 0.0], [3.0, -0.5]]]),
    )


def test_gaussian_kl_is_zero_for_identical_unit_gaussians() -> None:
    zeros = torch.zeros((3, 4))
    result = gaussian_kl_divergence(zeros, zeros, zeros, zeros)
    assert result.item() == pytest.approx(0.0)


def test_gaussian_kl_matches_shifted_unit_gaussians() -> None:
    posterior_mean = torch.tensor([[1.0, 2.0]])
    zeros = torch.zeros_like(posterior_mean)
    result = gaussian_kl_divergence(posterior_mean, zeros, zeros, zeros)
    assert result.item() == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("prediction", "target", "mask", "message"),
    [
        (torch.zeros((1, 2, 3)), torch.zeros((1, 2, 2)), torch.ones((1, 2), dtype=torch.bool), "prediction"),
        (torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2)), torch.ones((1, 3), dtype=torch.bool), "mask"),
        (torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2)), torch.zeros((1, 2), dtype=torch.bool), "at least one"),
    ],
)
def test_displacement_metrics_reject_invalid_contracts(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        displacement_sums(prediction, target, mask)

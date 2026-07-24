import torch
import pytest

from skilldrive.prediction.data import _tensor_sample_from_mapping


def _sample():
    sample = {
        "actor_history": torch.zeros(1, 1, 2, 6),
        "actor_time_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "actor_mask": torch.ones(1, 1, dtype=torch.bool),
        "actor_type_id": torch.zeros(1, 1, dtype=torch.long),
        "actor_role_id": torch.zeros(1, 1, dtype=torch.long),
        "map_polylines": torch.zeros(1, 1, 2, 4),
        "map_point_mask": torch.ones(1, 1, 2, dtype=torch.bool),
        "map_polyline_mask": torch.ones(1, 1, dtype=torch.bool),
        "map_type_id": torch.zeros(1, 1, dtype=torch.long),
        "target_actor_index": torch.zeros(1, dtype=torch.long),
        "target_future": torch.zeros(60, 2),
        "target_future_mask": torch.ones(60, dtype=torch.bool),
        "anchor_origin_global": torch.zeros(2),
        "anchor_heading_global": torch.tensor(0.0),
    }
    return sample


def test_partial_future_mask_is_allowed_when_valid_points_are_finite():
    sample = _sample()
    sample["target_future_mask"][20:] = False
    sample["target_future"][20:] = float("nan")
    _tensor_sample_from_mapping(sample)


def test_invalid_value_in_valid_future_is_rejected():
    sample = _sample()
    sample["target_future"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="valid target future"):
        _tensor_sample_from_mapping(sample)

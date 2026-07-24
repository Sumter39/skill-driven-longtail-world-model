import torch

from skilldrive.prediction.model import LSTMTrajectoryPredictor, VectorTrajectoryPredictor


def _batch(batch_size=2):
    return {
        "actor_history": torch.zeros(batch_size, 3, 5, 6),
        "actor_time_mask": torch.ones(batch_size, 3, 5, dtype=torch.bool),
        "actor_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "actor_type_id": torch.zeros(batch_size, 3, dtype=torch.long),
        "map_polylines": torch.zeros(batch_size, 4, 3, 4),
        "map_point_mask": torch.ones(batch_size, 4, 3, dtype=torch.bool),
        "map_polyline_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "map_type_id": torch.zeros(batch_size, 4, dtype=torch.long),
        "target_actor_index": torch.zeros(batch_size, dtype=torch.long),
    }


def test_predictors_return_six_finite_modes():
    for model in (
        LSTMTrajectoryPredictor(hidden_dim=16),
        VectorTrajectoryPredictor(
            hidden_dim=16, type_embedding_dim=4, interaction_layers=1, interaction_heads=2
        ),
    ):
        output = model(_batch())
        assert output.trajectories.shape == (2, 6, 60, 2)
        assert output.logits.shape == (2, 6)
        assert torch.isfinite(output.trajectories).all()
        assert torch.isfinite(output.logits).all()

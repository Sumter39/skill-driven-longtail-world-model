from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.modeling.diagnose_cvae import (
    MAX_DIAGNOSTIC_SAMPLES,
    infer_diagnostic_record,
    render_diagnostic_plot,
    select_diagnostic_indices,
)
from skilldrive.models import ConditionalCVAE


def _entry(index: int, *, observed: bool) -> dict:
    skill_id = "slow_lead_blockage" if observed else "<none>"
    return {
        "sample_id": f"sample-{index:02d}-{'observed' if observed else 'base'}",
        "scenario_id": f"scenario-{index:02d}",
        "target_track_id": "target",
        "spec": {
            "skill_id": skill_id,
            "skill_supervision_mask": observed,
        },
    }


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
        interaction_layers=1,
        interaction_heads=4,
        skill_embedding_dim=4,
        parameter_hidden_dim=8,
        latent_dim=4,
        decoder_hidden_dim=16,
        future_steps=6,
        dropout=0.0,
    )


def _sample() -> dict:
    generator = torch.Generator().manual_seed(17)
    actor_history = torch.randn(3, 5, 4, generator=generator)
    map_polylines = torch.randn(2, 4, 3, generator=generator)
    return {
        "sample_id": "sample-00-observed",
        "scenario_id": "scenario-00",
        "target_track_id": "target",
        "actor_history": actor_history,
        "actor_time_mask": torch.tensor(
            [[True, True, True, True, True], [True] * 5, [False] * 5]
        ),
        "actor_mask": torch.tensor([True, True, False]),
        "actor_type_id": torch.tensor([1, 2, 0]),
        "actor_role_id": torch.tensor([1, 2, 0]),
        "map_polylines": map_polylines,
        "map_point_mask": torch.tensor(
            [[True, True, True, True], [True, True, False, False]]
        ),
        "map_polyline_mask": torch.tensor([True, True]),
        "map_type_id": torch.tensor([1, 2]),
        "target_actor_index": torch.tensor(0),
        "skill_id": torch.tensor(1),
        "skill_supervision_mask": torch.tensor(True),
        "skill_parameters": torch.tensor([0.2, 0.0, 0.8]),
        "parameter_mask": torch.tensor([True, False, True]),
        "target_future": torch.randn(6, 2, generator=generator).cumsum(dim=0),
        "target_future_mask": torch.ones(6, dtype=torch.bool),
        "anchor_origin_global": torch.tensor([10.0, 20.0]),
        "anchor_heading_global": torch.tensor(0.25),
    }


def test_fixed_selection_is_reproducible_balanced_and_bounded() -> None:
    entries = [
        *[_entry(index, observed=False) for index in range(12)],
        *[_entry(index + 12, observed=True) for index in range(12)],
    ]

    first = select_diagnostic_indices(entries, sample_count=8, seed=2026)
    second = select_diagnostic_indices(entries, sample_count=8, seed=2026)

    assert first == second
    assert len(first) == len(set(first)) == 8
    assert sum(entries[index]["spec"]["skill_supervision_mask"] for index in first) == 4
    with pytest.raises(ValueError, match="between 1 and 16"):
        select_diagnostic_indices(
            entries,
            sample_count=MAX_DIAGNOSTIC_SAMPLES + 1,
            seed=2026,
        )


def test_inference_record_uses_model_posterior_and_prior_reproducibly() -> None:
    torch.manual_seed(11)
    model = _model().eval()
    sample = _sample()
    entry = _entry(0, observed=True)

    first = infer_diagnostic_record(
        model,
        sample,
        entry,
        device=torch.device("cpu"),
        prior_samples=3,
        seed=2026,
    )
    second = infer_diagnostic_record(
        model,
        sample,
        entry,
        device=torch.device("cpu"),
        prior_samples=3,
        seed=2026,
    )

    assert first == second
    assert first["inference"]["posterior"]["method"] == "ConditionalCVAE.forward_train"
    assert first["inference"]["prior"]["method"] == "ConditionalCVAE.sample_prior"
    assert len(first["trajectories_local_xy"]["prior_samples"]) == 3
    assert first["inference"]["posterior"]["trajectory_sha256"] != first["inference"][
        "prior"
    ]["trajectory_sha256"]
    priors = torch.tensor(first["trajectories_local_xy"]["prior_samples"])
    assert not torch.equal(priors[0], priors[1])
    batch = {
        name: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
        for name, value in sample.items()
    }
    expected_posterior = model.forward_train(
        batch,
        torch.Generator().manual_seed(
            first["inference"]["posterior"]["generator_seed"]
        ),
    )
    expected_prior = model.sample_prior(
        batch,
        3,
        torch.Generator().manual_seed(first["inference"]["prior"]["generator_seed"]),
    )
    torch.testing.assert_close(
        torch.tensor(first["trajectories_local_xy"]["posterior_reconstruction"]),
        expected_posterior.future_position_local[0],
    )
    torch.testing.assert_close(priors, expected_prior.future_position_local[0])


def test_plot_is_rendered_from_recorded_model_trajectories(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = _model().eval()
    sample = _sample()
    record = infer_diagnostic_record(
        model,
        sample,
        _entry(0, observed=True),
        device=torch.device("cpu"),
        prior_samples=3,
        seed=2026,
    )
    output = tmp_path / "diagnostic.png"

    render_diagnostic_plot(
        sample,
        record,
        output,
        checkpoint_sha256="a" * 64,
    )

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000

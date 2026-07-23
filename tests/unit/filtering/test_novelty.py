from __future__ import annotations

import numpy as np

from skilldrive.filtering.contracts import FilterRejection
from skilldrive.filtering.novelty import check_observed_future_novelty
from skilldrive.generation.config import load_filter_config
from skilldrive.schemas import AgentTrack, Scenario


def _scenario() -> Scenario:
    positions = np.column_stack(
        (np.arange(110, dtype=np.float64) * 0.1, np.zeros(110))
    )
    observed = np.zeros(110, dtype=bool)
    observed[:50] = True
    target = AgentTrack(
        track_id="target",
        object_type="vehicle",
        positions=positions,
        velocities=np.tile([1.0, 0.0], (110, 1)),
        headings=np.zeros(110),
        observed_mask=observed,
        is_focal=True,
    )
    return Scenario(
        scenario_id="novelty",
        city_name="city",
        timestamps=np.arange(110, dtype=np.int64) * 100_000_000,
        focal_track_id="target",
        agents=[target],
        map_polylines=[],
    )


def test_original_future_copy_is_not_a_counterfactual() -> None:
    scenario = _scenario()
    policy = load_filter_config().novelty_policy
    result = check_observed_future_novelty(
        scenario,
        "target",
        scenario.agents[0].positions[50:].copy(),
        policy,
    )

    assert result.rejection_reasons == (FilterRejection.NOVELTY_INSUFFICIENT,)


def test_endpoint_or_rms_difference_can_satisfy_novelty() -> None:
    scenario = _scenario()
    future = scenario.agents[0].positions[50:].copy()
    future[-1, 1] += 1.1
    result = check_observed_future_novelty(
        scenario,
        "target",
        future,
        load_filter_config().novelty_policy,
    )

    assert result.passed

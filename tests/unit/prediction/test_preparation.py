from types import SimpleNamespace

import numpy as np

from skilldrive.prediction.preparation import _original_future_local


def test_original_future_returns_mask_for_missing_points():
    positions = np.full((110, 2), np.nan, dtype=np.float64)
    positions[50:70] = np.arange(40, dtype=np.float64).reshape(20, 2)
    scenario = SimpleNamespace(
        scenario_id="scenario",
        agents=[SimpleNamespace(track_id="target", positions=positions)],
    )
    future, mask = _original_future_local(
        scenario, "target", np.zeros(2, dtype=np.float64), 0.0
    )
    assert future.shape == (60, 2)
    assert mask.shape == (60,)
    assert mask[:20].all()
    assert not mask[20:].any()
    assert np.isfinite(future).all()

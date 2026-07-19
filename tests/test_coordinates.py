import numpy as np

from skilldrive.data.coordinates import global_to_local, local_to_global, to_focal_frame, wrap_angle
from skilldrive.schemas import Scenario


def test_coordinate_round_trip() -> None:
    points = np.array([[10.0, -3.0], [25.5, 4.0], [-1.0, 8.0]])
    origin = np.array([7.0, -2.0])
    heading = 0.73
    recovered = local_to_global(global_to_local(points, origin, heading), origin, heading)
    np.testing.assert_allclose(recovered, points, atol=1e-10)


def test_heading_frame_orientation() -> None:
    point = np.array([[0.0, 2.0]])
    local = global_to_local(point, np.zeros(2), np.pi / 2)
    np.testing.assert_allclose(local, [[2.0, 0.0]], atol=1e-10)


def test_wrap_angle_range() -> None:
    wrapped = wrap_angle(np.array([-4 * np.pi, -np.pi, np.pi, 3 * np.pi]))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_scenario_focal_frame_places_focal_at_origin(synthetic_scenario: Scenario) -> None:
    local = to_focal_frame(synthetic_scenario)
    focal = local.agents[0]
    last_observed = np.flatnonzero(focal.observed_mask)[-1]
    np.testing.assert_allclose(focal.positions[last_observed], [0.0, 0.0], atol=1e-10)
    assert local.metadata["coordinate_frame"] == "focal_agent_at_last_observation"

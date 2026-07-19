from pathlib import Path

import numpy as np
import pytest

from skilldrive.data.av2_reader import _resample_polyline, discover_map_path


def test_resampled_polyline_preserves_endpoints() -> None:
    source = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    result = _resample_polyline(source, count=9)
    np.testing.assert_allclose(result[0], source[0])
    np.testing.assert_allclose(result[-1], source[-1])
    assert result.shape == (9, 2)


def test_discover_map_path(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_example.parquet"
    scenario.touch()
    expected = tmp_path / "log_map_archive_example.json"
    expected.touch()
    assert discover_map_path(scenario) == expected


def test_discover_map_path_rejects_ambiguity(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_example.parquet"
    scenario.touch()
    (tmp_path / "log_map_archive_a.json").touch()
    (tmp_path / "log_map_archive_b.json").touch()
    with pytest.raises(FileNotFoundError, match="expected one"):
        discover_map_path(scenario)


def test_installed_av2_api_has_required_entrypoints() -> None:
    pytest.importorskip("av2")
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, Track
    from av2.map.lane_segment import LaneSegment
    from av2.map.map_api import ArgoverseStaticMap

    assert hasattr(scenario_serialization, "load_argoverse_scenario_parquet")
    assert hasattr(ArgoverseStaticMap, "from_json")
    assert hasattr(ArgoverseStaticMap, "get_scenario_lane_segments")
    assert {"scenario_id", "timestamps_ns", "focal_track_id", "tracks"} <= set(
        ArgoverseScenario.__dataclass_fields__
    )
    assert {"track_id", "object_type", "object_states"} <= set(Track.__dataclass_fields__)
    assert {"timestep", "observed", "position", "velocity", "heading"} <= set(
        ObjectState.__dataclass_fields__
    )
    assert {"id", "left_lane_boundary", "right_lane_boundary", "is_intersection"} <= set(
        LaneSegment.__dataclass_fields__
    )

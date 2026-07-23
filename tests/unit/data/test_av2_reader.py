from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import skilldrive.data.av2_reader as av2_reader
from skilldrive.data.av2_reader import (
    _map_polylines,
    _resample_polyline,
    discover_map_path,
    load_av2_history_scenario,
    load_av2_scenario,
)


def test_preload_av2_dependencies_imports_pandas_and_required_av2_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario_serialization = SimpleNamespace()

    class FakeArgoverseStaticMap:
        pass

    modules = {
        "pandas": SimpleNamespace(DataFrame=object()),
        "av2.datasets.motion_forecasting.scenario_serialization": scenario_serialization,
        "av2.map.map_api": SimpleNamespace(
            ArgoverseStaticMap=FakeArgoverseStaticMap
        ),
    }
    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        return modules[name]

    monkeypatch.setattr(av2_reader, "import_module", fake_import_module)

    loaded_serialization, loaded_static_map = (
        av2_reader.preload_av2_dependencies()
    )

    assert imported == [
        "pandas",
        "av2.datasets.motion_forecasting.scenario_serialization",
        "av2.map.map_api",
    ]
    assert loaded_serialization is scenario_serialization
    assert loaded_static_map is FakeArgoverseStaticMap


def test_preload_av2_worker_dependencies_only_initializes_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pandas = SimpleNamespace(DataFrame=object())
    imported: list[str] = []

    def fake_import_module(name: str):
        imported.append(name)
        return pandas

    monkeypatch.setattr(av2_reader, "import_module", fake_import_module)

    av2_reader.preload_av2_worker_dependencies()

    assert imported == ["pandas"]


def test_preload_av2_worker_dependencies_rejects_incomplete_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        av2_reader,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="optional AV2 dependency is not installed"):
        av2_reader.preload_av2_worker_dependencies()


def test_preload_av2_dependencies_preserves_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(av2_reader, "import_module", missing_dependency)

    with pytest.raises(RuntimeError, match="optional AV2 dependency is not installed"):
        av2_reader.preload_av2_dependencies()


def test_load_av2_scenario_reuses_preloaded_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_scenario = SimpleNamespace(
        scenario_id="scenario-id",
        city_name="PIT",
        timestamps_ns=[0],
        focal_track_id="focal-id",
        tracks=[
            SimpleNamespace(
                track_id="focal-id",
                object_type=SimpleNamespace(value="VEHICLE"),
                object_states=[
                    SimpleNamespace(
                        timestep=0,
                        position=[1.0, 2.0],
                        velocity=[3.0, 4.0],
                        heading=0.25,
                        observed=True,
                    )
                ],
            )
        ],
    )
    static_map = SimpleNamespace(
        get_scenario_lane_segments=lambda: [],
        get_scenario_ped_crossings=lambda: [],
        get_scenario_vector_drivable_areas=lambda: [],
    )
    loaded_paths: list[Path] = []
    mapped_paths: list[Path] = []
    scenario_serialization = SimpleNamespace(
        load_argoverse_scenario_parquet=lambda path: (
            loaded_paths.append(path) or raw_scenario
        )
    )

    class FakeArgoverseStaticMap:
        @classmethod
        def from_json(cls, path: Path):
            mapped_paths.append(path)
            return static_map

    preload_calls = 0

    def fake_preload():
        nonlocal preload_calls
        preload_calls += 1
        return scenario_serialization, FakeArgoverseStaticMap

    monkeypatch.setattr(av2_reader, "preload_av2_dependencies", fake_preload)
    scenario_path = tmp_path / "scenario.parquet"
    map_path = tmp_path / "map.json"

    scenario = load_av2_scenario(scenario_path, map_path)

    assert preload_calls == 1
    assert loaded_paths == [scenario_path]
    assert mapped_paths == [map_path]
    assert scenario.scenario_id == "scenario-id"
    assert scenario.focal_track_id == "focal-id"
    assert scenario.metadata["map_path"] == str(map_path)
    np.testing.assert_allclose(scenario.agents[0].positions, [[1.0, 2.0]])


def test_load_av2_history_scenario_does_not_access_future_state_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PoisonFutureState:
        timestep = 50

        @property
        def position(self):
            raise AssertionError("future position was accessed")

        @property
        def velocity(self):
            raise AssertionError("future velocity was accessed")

        @property
        def heading(self):
            raise AssertionError("future heading was accessed")

        @property
        def observed(self):
            raise AssertionError("future observed flag was accessed")

    raw_scenario = SimpleNamespace(
        scenario_id="scenario-id",
        city_name="PIT",
        timestamps_ns=list(range(110)),
        focal_track_id="focal-id",
        tracks=[
            SimpleNamespace(
                track_id="focal-id",
                object_type=SimpleNamespace(value="VEHICLE"),
                object_states=[
                    SimpleNamespace(
                        timestep=0,
                        position=[1.0, 2.0],
                        velocity=[3.0, 4.0],
                        heading=0.25,
                        observed=True,
                    ),
                    PoisonFutureState(),
                ],
            )
        ],
    )
    scenario_serialization = SimpleNamespace(
        load_argoverse_scenario_parquet=lambda path: raw_scenario
    )

    class FakeArgoverseStaticMap:
        @classmethod
        def from_json(cls, path: Path):
            return SimpleNamespace(
                get_scenario_lane_segments=lambda: [],
                get_scenario_ped_crossings=lambda: [],
                get_scenario_vector_drivable_areas=lambda: [],
            )

    monkeypatch.setattr(
        av2_reader,
        "preload_av2_dependencies",
        lambda: (scenario_serialization, FakeArgoverseStaticMap),
    )

    scenario = load_av2_history_scenario(
        tmp_path / "scenario.parquet",
        tmp_path / "map.json",
    )

    assert len(scenario.timestamps) == 50
    assert len(scenario.agents[0].positions) == 50
    assert scenario.metadata["temporal_scope"] == "history_only"
    assert scenario.metadata["timestamp_count"] == 50


def test_load_av2_scenario_discovers_map_once_and_reuses_resolved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_scenario = SimpleNamespace(
        scenario_id="scenario-id",
        city_name="PIT",
        timestamps_ns=[0],
        focal_track_id="focal-id",
        tracks=[
            SimpleNamespace(
                track_id="focal-id",
                object_type=SimpleNamespace(value="VEHICLE"),
                object_states=[
                    SimpleNamespace(
                        timestep=0,
                        position=[0.0, 0.0],
                        velocity=[0.0, 0.0],
                        heading=0.0,
                        observed=True,
                    )
                ],
            )
        ],
    )
    scenario_serialization = SimpleNamespace(
        load_argoverse_scenario_parquet=lambda path: raw_scenario
    )
    mapped_paths: list[Path] = []

    class FakeArgoverseStaticMap:
        @classmethod
        def from_json(cls, path: Path):
            mapped_paths.append(path)
            return SimpleNamespace()

    monkeypatch.setattr(
        av2_reader,
        "preload_av2_dependencies",
        lambda: (scenario_serialization, FakeArgoverseStaticMap),
    )
    monkeypatch.setattr(av2_reader, "_map_polylines", lambda static_map: [])
    scenario_path = tmp_path / "scenario.parquet"
    expected_map_path = tmp_path / "log_map_archive.json"
    discovered_sources: list[Path] = []

    def fake_discover(source: Path) -> Path:
        discovered_sources.append(source)
        return expected_map_path

    monkeypatch.setattr(av2_reader, "discover_map_path", fake_discover)

    scenario = load_av2_scenario(scenario_path)

    assert discovered_sources == [scenario_path]
    assert mapped_paths == [expected_map_path]
    assert scenario.metadata["map_path"] == str(expected_map_path)


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


def test_discover_map_path_prefers_standard_pair_without_directory_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = tmp_path / "scenario_example.parquet"
    scenario.touch()
    expected = tmp_path / "log_map_archive_example.json"
    expected.touch()
    (tmp_path / "log_map_archive_stale.json").touch()

    def reject_glob(self: Path, pattern: str):
        raise AssertionError(f"unexpected glob for {self}: {pattern}")

    monkeypatch.setattr(Path, "glob", reject_glob)

    assert discover_map_path(scenario) == expected


def test_discover_map_path_falls_back_for_nonstandard_map_name(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_example.parquet"
    scenario.touch()
    expected = tmp_path / "log_map_archive_legacy.json"
    expected.touch()

    assert discover_map_path(scenario) == expected


def test_discover_map_path_rejects_ambiguity(tmp_path: Path) -> None:
    scenario = tmp_path / "scenario_example.parquet"
    scenario.touch()
    (tmp_path / "log_map_archive_a.json").touch()
    (tmp_path / "log_map_archive_b.json").touch()
    with pytest.raises(FileNotFoundError, match="expected one"):
        discover_map_path(scenario)


def test_map_polylines_expose_av2_map_geometry_and_topology() -> None:
    left = SimpleNamespace(xyz=np.array([[0.0, 1.0, 0.0], [4.0, 1.0, 0.0]]))
    right = SimpleNamespace(xyz=np.array([[0.0, -1.0, 0.0], [4.0, -1.0, 0.0]]))
    lane = SimpleNamespace(
        id=101,
        left_lane_boundary=left,
        right_lane_boundary=right,
        is_intersection=True,
        lane_type=SimpleNamespace(value="VEHICLE"),
        left_mark_type=SimpleNamespace(value="DASHED_WHITE"),
        right_mark_type=SimpleNamespace(value="SOLID_WHITE"),
        predecessors=[99],
        successors=[103, 104],
        left_neighbor_id=100,
        right_neighbor_id=None,
    )
    crossing = SimpleNamespace(
        id=201,
        edge1=SimpleNamespace(xyz=np.array([[1.0, -2.0, 0.0], [1.0, 2.0, 0.0]])),
        edge2=SimpleNamespace(xyz=np.array([[3.0, -2.0, 0.0], [3.0, 2.0, 0.0]])),
    )
    area = SimpleNamespace(
        id=301,
        area_boundary=[
            SimpleNamespace(x=-1.0, y=-3.0),
            SimpleNamespace(x=5.0, y=-3.0),
            SimpleNamespace(x=5.0, y=3.0),
            SimpleNamespace(x=-1.0, y=3.0),
        ],
    )
    static_map = SimpleNamespace(
        get_scenario_lane_segments=lambda: [lane],
        get_scenario_ped_crossings=lambda: [crossing],
        get_scenario_vector_drivable_areas=lambda: [area],
    )

    polylines = _map_polylines(static_map)
    by_type = {polyline.polyline_type: polyline for polyline in polylines}

    centerline = by_type["lane_centerline"]
    assert centerline.lane_id == "101"
    assert centerline.predecessor_ids == ["99"]
    assert centerline.successor_ids == ["103", "104"]
    assert centerline.left_neighbor_id == "100"
    assert centerline.right_neighbor_id is None
    assert centerline.left_mark_type == "dashed_white"
    assert centerline.right_mark_type == "solid_white"
    assert by_type["lane_boundary_left"].mark_type == "dashed_white"
    assert by_type["lane_boundary_right"].mark_type == "solid_white"

    crosswalk = by_type["pedestrian_crossing"]
    drivable_area = by_type["drivable_area"]
    np.testing.assert_allclose(crosswalk.points[0], crosswalk.points[-1])
    np.testing.assert_allclose(drivable_area.points[0], drivable_area.points[-1])


def _find_local_av2_scenario() -> Path | None:
    sample = Path(
        "data/sample/av2/0a1e6f0a-1817-4a98-b02e-db8c9327d151/"
        "scenario_0a1e6f0a-1817-4a98-b02e-db8c9327d151.parquet"
    )
    if sample.exists():
        return sample
    train_root = Path("data/av2/motion-forecasting/train")
    return next(train_root.glob("*/scenario_*.parquet"), None)


def test_real_av2_sample_exposes_all_available_map_features() -> None:
    pytest.importorskip("av2")
    from av2.map.map_api import ArgoverseStaticMap

    source = _find_local_av2_scenario()
    if source is None:
        pytest.skip("no local AV2 sample is available")

    map_path = discover_map_path(source)
    static_map = ArgoverseStaticMap.from_json(map_path)
    scenario = load_av2_scenario(source, map_path)

    lane_centers = [
        polyline
        for polyline in scenario.map_polylines
        if polyline.polyline_type == "lane_centerline"
    ]
    crosswalks = [
        polyline
        for polyline in scenario.map_polylines
        if polyline.polyline_type == "pedestrian_crossing"
    ]
    drivable_areas = [
        polyline
        for polyline in scenario.map_polylines
        if polyline.polyline_type == "drivable_area"
    ]

    assert len(lane_centers) == len(static_map.get_scenario_lane_segments())
    assert len(crosswalks) == len(static_map.get_scenario_ped_crossings())
    assert len(drivable_areas) == len(static_map.get_scenario_vector_drivable_areas())

    raw_lane = static_map.get_scenario_lane_segments()[0]
    centerline = next(item for item in lane_centers if item.lane_id == str(raw_lane.id))
    assert centerline.predecessor_ids == [str(value) for value in raw_lane.predecessors]
    assert centerline.successor_ids == [str(value) for value in raw_lane.successors]
    assert centerline.left_neighbor_id == (
        None if raw_lane.left_neighbor_id is None else str(raw_lane.left_neighbor_id)
    )
    assert centerline.right_neighbor_id == (
        None if raw_lane.right_neighbor_id is None else str(raw_lane.right_neighbor_id)
    )
    assert centerline.left_mark_type == str(raw_lane.left_mark_type.value).lower()
    assert centerline.right_mark_type == str(raw_lane.right_mark_type.value).lower()


def test_installed_av2_api_has_required_entrypoints() -> None:
    pytest.importorskip("av2")
    from av2.datasets.motion_forecasting import scenario_serialization
    from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, Track
    from av2.map.lane_segment import LaneSegment
    from av2.map.map_api import ArgoverseStaticMap

    assert hasattr(scenario_serialization, "load_argoverse_scenario_parquet")
    assert hasattr(ArgoverseStaticMap, "from_json")
    assert hasattr(ArgoverseStaticMap, "get_scenario_lane_segments")
    assert hasattr(ArgoverseStaticMap, "get_scenario_ped_crossings")
    assert hasattr(ArgoverseStaticMap, "get_scenario_vector_drivable_areas")
    assert {"scenario_id", "timestamps_ns", "focal_track_id", "tracks"} <= set(
        ArgoverseScenario.__dataclass_fields__
    )
    assert {"track_id", "object_type", "object_states"} <= set(Track.__dataclass_fields__)
    assert {"timestep", "observed", "position", "velocity", "heading"} <= set(
        ObjectState.__dataclass_fields__
    )
    assert {
        "id",
        "left_lane_boundary",
        "right_lane_boundary",
        "is_intersection",
        "left_mark_type",
        "right_mark_type",
        "predecessors",
        "successors",
        "left_neighbor_id",
        "right_neighbor_id",
    } <= set(LaneSegment.__dataclass_fields__)

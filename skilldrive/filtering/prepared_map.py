"""Reusable vector geometry for repeated map-compliance queries."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from skilldrive.schemas import Scenario


_POINT_QUERY_CHUNK_SIZE = 128
_LANE_AABB_EPSILON_M = 1e-8


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PreparedDrivablePolygon:
    vertices: np.ndarray
    edge_starts: np.ndarray
    edge_deltas: np.ndarray
    edge_squared_lengths: np.ndarray
    minimum_xy: np.ndarray | None
    maximum_xy: np.ndarray | None

    @property
    def valid(self) -> bool:
        return self.minimum_xy is not None


@dataclass(frozen=True)
class PreparedLaneGeometry:
    lane_id: str
    lane_type: str
    predecessor_ids: tuple[str, ...]
    successor_ids: tuple[str, ...]
    left_neighbor_id: str | None
    right_neighbor_id: str | None
    segment_starts: np.ndarray
    segment_deltas: np.ndarray
    segment_lengths: np.ndarray
    segment_squared_lengths: np.ndarray
    segment_headings: np.ndarray
    minimum_xy: np.ndarray
    maximum_xy: np.ndarray


@dataclass(frozen=True)
class PreparedMapGeometry:
    """Immutable geometry and identity evidence for one source scenario map."""

    scenario_id: str
    map_sha256: str
    drivable_area_count: int
    drivable_polygons: tuple[PreparedDrivablePolygon, ...]
    lanes: tuple[PreparedLaneGeometry, ...]


def _update_text(digest, value: object) -> None:
    encoded = ("<none>" if value is None else str(value)).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def map_geometry_sha256(scenario: Scenario) -> str:
    """Hash every map field whose change can affect compliance semantics."""

    digest = sha256()
    digest.update(len(scenario.map_polylines).to_bytes(8, "little"))
    for polyline in scenario.map_polylines:
        for value in (
            polyline.polyline_id,
            polyline.polyline_type,
            polyline.direction,
            polyline.is_intersection,
            polyline.lane_id,
            polyline.mark_type,
            polyline.left_mark_type,
            polyline.right_mark_type,
            polyline.left_neighbor_id,
            polyline.right_neighbor_id,
        ):
            _update_text(digest, value)
        for values in (polyline.predecessor_ids, polyline.successor_ids):
            digest.update(len(values).to_bytes(8, "little"))
            for value in values:
                _update_text(digest, value)
        points = np.ascontiguousarray(polyline.points, dtype="<f8")
        digest.update(len(points).to_bytes(8, "little"))
        digest.update(points.tobytes())
    return digest.hexdigest()


def _prepare_polygon(points: np.ndarray) -> PreparedDrivablePolygon:
    vertices = np.asarray(points, dtype=np.float64)
    finite = vertices[np.isfinite(vertices).all(axis=1)]
    if len(finite) >= 1 and np.allclose(finite[0], finite[-1]):
        finite = finite[:-1]
    finite = _readonly_float64(finite)
    if len(finite) < 3:
        empty = _readonly_float64(np.empty((0, 2), dtype=np.float64))
        squared = _readonly_float64(np.empty(0, dtype=np.float64))
        return PreparedDrivablePolygon(
            vertices=finite,
            edge_starts=empty,
            edge_deltas=empty,
            edge_squared_lengths=squared,
            minimum_xy=None,
            maximum_xy=None,
        )
    starts = _readonly_float64(np.roll(finite, 1, axis=0))
    deltas = _readonly_float64(finite - starts)
    squared = _readonly_float64(np.sum(deltas * deltas, axis=1))
    minimum = _readonly_float64(np.min(finite, axis=0))
    maximum = _readonly_float64(np.max(finite, axis=0))
    return PreparedDrivablePolygon(
        vertices=finite,
        edge_starts=starts,
        edge_deltas=deltas,
        edge_squared_lengths=squared,
        minimum_xy=minimum,
        maximum_xy=maximum,
    )


def _prepare_lane(polyline) -> PreparedLaneGeometry:
    points = np.asarray(polyline.points, dtype=np.float64)
    starts = _readonly_float64(points[:-1])
    deltas = _readonly_float64(points[1:] - points[:-1])
    lengths = _readonly_float64(np.linalg.norm(deltas, axis=1))
    squared = _readonly_float64(lengths**2)
    headings = np.full(len(deltas), np.nan, dtype=np.float64)
    nondegenerate = lengths > 1e-12
    headings[nondegenerate] = np.arctan2(
        deltas[nondegenerate, 1],
        deltas[nondegenerate, 0],
    )
    return PreparedLaneGeometry(
        lane_id=str(polyline.lane_id),
        lane_type=str(polyline.direction).strip().lower(),
        predecessor_ids=tuple(polyline.predecessor_ids),
        successor_ids=tuple(polyline.successor_ids),
        left_neighbor_id=polyline.left_neighbor_id,
        right_neighbor_id=polyline.right_neighbor_id,
        segment_starts=starts,
        segment_deltas=deltas,
        segment_lengths=lengths,
        segment_squared_lengths=squared,
        segment_headings=_readonly_float64(headings),
        minimum_xy=_readonly_float64(np.min(points, axis=0)),
        maximum_xy=_readonly_float64(np.max(points, axis=0)),
    )


def prepare_map_geometry(scenario: Scenario) -> PreparedMapGeometry:
    """Prepare one source map for repeated single or batched compliance checks."""

    declared_polygons = [
        polyline
        for polyline in scenario.map_polylines
        if polyline.polyline_type == "drivable_area"
    ]
    indexed_lanes = [
        (index, lane)
        for index, lane in enumerate(scenario.map_polylines)
        if lane.polyline_type == "lane_centerline"
        and lane.lane_id
        and len(lane.points) >= 2
        and np.isfinite(lane.points).all()
    ]
    indexed_lanes.sort(key=lambda item: (str(item[1].lane_id), item[0]))
    return PreparedMapGeometry(
        scenario_id=scenario.scenario_id,
        map_sha256=map_geometry_sha256(scenario),
        drivable_area_count=len(declared_polygons),
        drivable_polygons=tuple(
            _prepare_polygon(polyline.points) for polyline in declared_polygons
        ),
        lanes=tuple(_prepare_lane(lane) for _, lane in indexed_lanes),
    )


def require_compatible_map(
    scenarios: Sequence[Scenario],
    prepared_map: PreparedMapGeometry,
) -> None:
    """Reject a prepared map when any query scenario has different map content."""

    if not isinstance(prepared_map, PreparedMapGeometry):
        raise TypeError("prepared_map must be PreparedMapGeometry")
    fingerprints: dict[tuple[int, ...], str] = {}
    for scenario in scenarios:
        if scenario.scenario_id != prepared_map.scenario_id:
            raise ValueError("prepared_map scenario_id differs from query scenario")
        identity = tuple(id(polyline) for polyline in scenario.map_polylines)
        fingerprint = fingerprints.get(identity)
        if fingerprint is None:
            fingerprint = map_geometry_sha256(scenario)
            fingerprints[identity] = fingerprint
        if fingerprint != prepared_map.map_sha256:
            raise ValueError("prepared_map geometry differs from query scenario")


class PreparedMapVerificationSession:
    """Reuse one verified source-map identity until an atomic scenario result closes."""

    def __init__(
        self,
        source_scenario: Scenario,
        prepared_map: PreparedMapGeometry,
    ) -> None:
        if not isinstance(source_scenario, Scenario):
            raise TypeError("source_scenario must be a Scenario")
        if not isinstance(prepared_map, PreparedMapGeometry):
            raise TypeError("prepared_map must be PreparedMapGeometry")
        if source_scenario.scenario_id != prepared_map.scenario_id:
            raise ValueError("prepared_map scenario_id differs from source scenario")
        self._source_scenario = source_scenario
        self._prepared_map = prepared_map
        self._source_map_identity = tuple(
            id(polyline) for polyline in source_scenario.map_polylines
        )
        self._closed = False

    @property
    def prepared_map(self) -> PreparedMapGeometry:
        return self._prepared_map

    @property
    def closed(self) -> bool:
        return self._closed

    def verify_query(
        self,
        scenario: Scenario,
        prepared_map: PreparedMapGeometry,
    ) -> None:
        """Skip hashing only for generated scenes sharing the exact source map objects."""

        if self._closed:
            raise RuntimeError("prepared-map verification session is closed")
        if prepared_map is not self._prepared_map:
            raise ValueError("verification session is bound to a different prepared_map")
        shared_source_map = (
            scenario.scenario_id == self._source_scenario.scenario_id
            and tuple(id(polyline) for polyline in scenario.map_polylines)
            == self._source_map_identity
        )
        if not shared_source_map:
            require_compatible_map((scenario,), prepared_map)

    def finalize(self) -> None:
        """Rehash the authoritative source before any scenario result may escape."""

        if self._closed:
            raise RuntimeError("prepared-map verification session is closed")
        try:
            require_compatible_map((self._source_scenario,), self._prepared_map)
        finally:
            self._closed = True


def points_in_drivable_area(
    prepared_map: PreparedMapGeometry,
    positions: np.ndarray,
) -> np.ndarray:
    """Vectorized legacy-equivalent polygon membership for finite positions."""

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if not np.isfinite(points).all():
        raise ValueError("positions must contain finite values")
    inside = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), _POINT_QUERY_CHUNK_SIZE):
        stop = min(start + _POINT_QUERY_CHUNK_SIZE, len(points))
        query = points[start:stop]
        chunk_inside = inside[start:stop]
        for polygon in prepared_map.drivable_polygons:
            if not polygon.valid or np.all(chunk_inside):
                continue
            assert polygon.minimum_xy is not None and polygon.maximum_xy is not None
            candidate_indices = np.flatnonzero(
                ~chunk_inside
                & np.all(query >= polygon.minimum_xy - 1e-8, axis=1)
                & np.all(query <= polygon.maximum_xy + 1e-8, axis=1)
            )
            if not len(candidate_indices):
                continue
            chunk_inside[candidate_indices] = _points_in_polygon(
                polygon,
                query[candidate_indices],
            )
    return inside


def _points_in_polygon(
    polygon: PreparedDrivablePolygon,
    query: np.ndarray,
) -> np.ndarray:
    relative = query[:, None, :] - polygon.edge_starts[None, :, :]
    squared = polygon.edge_squared_lengths
    nondegenerate = squared > 1e-18
    fractions = np.zeros((len(query), len(squared)), dtype=np.float64)
    if np.any(nondegenerate):
        fractions[:, nondegenerate] = (
            np.sum(
                relative[:, nondegenerate, :]
                * polygon.edge_deltas[None, nondegenerate, :],
                axis=2,
            )
            / squared[nondegenerate]
        )
    on_extent = (fractions >= -1e-10) & (fractions <= 1.0 + 1e-10)
    clipped = np.clip(fractions, 0.0, 1.0)
    projections = (
        polygon.edge_starts[None, :, :]
        + clipped[:, :, None] * polygon.edge_deltas[None, :, :]
    )
    boundary = on_extent & (
        np.linalg.norm(query[:, None, :] - projections, axis=2) <= 1e-8
    )
    if np.any(~nondegenerate):
        boundary[:, ~nondegenerate] = (
            np.linalg.norm(relative[:, ~nondegenerate, :], axis=2) <= 1e-8
        )

    previous = polygon.edge_starts
    current = polygon.vertices
    y_crosses = (current[None, :, 1] > query[:, None, 1]) != (
        previous[None, :, 1] > query[:, None, 1]
    )
    x_crossing = np.zeros_like(fractions)
    np.divide(
        (previous[None, :, 0] - current[None, :, 0])
        * (query[:, None, 1] - current[None, :, 1]),
        previous[None, :, 1] - current[None, :, 1],
        out=x_crossing,
        where=y_crosses,
    )
    x_crossing += current[None, :, 0]
    ray_inside = np.logical_xor.reduce(
        y_crosses & (query[:, None, 0] < x_crossing),
        axis=1,
    )
    return np.any(boundary, axis=1) | ray_inside


def _project_points_to_lane(
    query: np.ndarray,
    lane: PreparedLaneGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    relative = query[:, None, :] - lane.segment_starts[None, :, :]
    fractions = np.zeros(
        (len(query), len(lane.segment_squared_lengths)),
        dtype=np.float64,
    )
    nondegenerate = lane.segment_lengths > 1e-12
    if np.any(nondegenerate):
        fractions[:, nondegenerate] = np.clip(
            np.sum(
                relative[:, nondegenerate, :]
                * lane.segment_deltas[None, nondegenerate, :],
                axis=2,
            )
            / lane.segment_squared_lengths[nondegenerate],
            0.0,
            1.0,
        )
    projected = (
        lane.segment_starts[None, :, :]
        + fractions[:, :, None] * lane.segment_deltas[None, :, :]
    )
    segment_distances = np.linalg.norm(query[:, None, :] - projected, axis=2)
    selected_segments = np.argmin(segment_distances, axis=1)
    lane_distances = segment_distances[np.arange(len(query)), selected_segments]
    return lane_distances, selected_segments


def _project_points_to_lanes(
    prepared_map: PreparedMapGeometry,
    positions: np.ndarray,
    *,
    maximum_distance_m: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    lane_indices = np.full(len(points), -1, dtype=np.int64)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    headings = np.full(len(points), np.nan, dtype=np.float64)
    if maximum_distance_m is not None and (
        not np.isfinite(maximum_distance_m) or maximum_distance_m < 0.0
    ):
        raise ValueError("maximum_distance_m must be finite and non-negative")
    finite_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    if not len(finite_indices) or not prepared_map.lanes:
        return lane_indices, distances, headings
    for start in range(0, len(finite_indices), _POINT_QUERY_CHUNK_SIZE):
        stop = min(start + _POINT_QUERY_CHUNK_SIZE, len(finite_indices))
        output_indices = finite_indices[start:stop]
        query = points[output_indices]
        best_distances = np.full(len(query), np.inf, dtype=np.float64)
        best_lanes = np.full(len(query), -1, dtype=np.int64)
        best_headings = np.full(len(query), np.nan, dtype=np.float64)
        for lane_index, lane in enumerate(prepared_map.lanes):
            if maximum_distance_m is None:
                query_indices = np.arange(len(query))
            else:
                margin = maximum_distance_m + _LANE_AABB_EPSILON_M
                query_indices = np.flatnonzero(
                    np.all(query >= lane.minimum_xy - margin, axis=1)
                    & np.all(query <= lane.maximum_xy + margin, axis=1)
                )
                if not len(query_indices):
                    continue
            lane_distances, selected_segments = _project_points_to_lane(
                query[query_indices],
                lane,
            )
            improved = lane_distances < best_distances[query_indices]
            if maximum_distance_m is not None:
                improved &= lane_distances <= maximum_distance_m
            improved_indices = query_indices[improved]
            best_distances[improved_indices] = lane_distances[improved]
            best_lanes[improved_indices] = lane_index
            best_headings[improved_indices] = lane.segment_headings[
                selected_segments[improved]
            ]
        lane_indices[output_indices] = best_lanes
        distances[output_indices] = best_distances
        headings[output_indices] = best_headings
    return lane_indices, distances, headings


def project_points_to_lanes(
    prepared_map: PreparedMapGeometry,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the global nearest lane for every finite point."""

    return _project_points_to_lanes(
        prepared_map,
        positions,
        maximum_distance_m=None,
    )


def project_points_to_lanes_within_distance(
    prepared_map: PreparedMapGeometry,
    positions: np.ndarray,
    maximum_distance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the nearest lane only when it is within the filtering threshold."""

    return _project_points_to_lanes(
        prepared_map,
        positions,
        maximum_distance_m=maximum_distance_m,
    )


__all__ = [
    "PreparedDrivablePolygon",
    "PreparedLaneGeometry",
    "PreparedMapGeometry",
    "PreparedMapVerificationSession",
    "map_geometry_sha256",
    "points_in_drivable_area",
    "prepare_map_geometry",
    "project_points_to_lanes",
    "project_points_to_lanes_within_distance",
    "require_compatible_map",
]

"""Deterministic NumPy sample construction for the conditional CVAE baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from skilldrive.data.coordinates import global_to_local, wrap_angle
from skilldrive.schemas import AgentTrack, MapPolyline, Scenario, SkillSpec
from skilldrive.seeds.records import SeedRecord
from skilldrive.skills import load_skill


NONE_SKILL_ID = "<none>"
PADDING_TOKEN = "<padding>"
CONTEXT_ROLE = "<context>"
BASE_TARGET_ROLE = "<target>"

HISTORY_STEPS = 50
FUTURE_STEPS = 60
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
ANCHOR_INDEX = HISTORY_STEPS - 1
MINIMUM_TARGET_HISTORY_STEPS = 30
MAX_ACTORS = 32
ACTOR_RADIUS_M = 100.0
MAX_MAP_POLYLINES = 128
MAX_MAP_POINTS = 20
MAP_RADIUS_M = 100.0
SAMPLE_PERIOD_S = 0.1
# One nominal lane width; larger center offsets are not same-lane following.
SHORT_HEADWAY_SAME_LANE_MAX_LATERAL_OFFSET_M = 3.75

ACTOR_FEATURE_DIM = 6
MAP_FEATURE_DIM = 4

ACTOR_TYPE_TOKENS = (
    PADDING_TOKEN,
    "unknown",
    "vehicle",
    "pedestrian",
    "motorcyclist",
    "cyclist",
    "bus",
    "static",
    "background",
    "construction",
    "riderless_bicycle",
)
MAP_TYPE_TOKENS = (
    PADDING_TOKEN,
    "lane_centerline",
    "pedestrian_crossing",
    "drivable_area",
)
ALLOWED_MAP_TYPES = frozenset(MAP_TYPE_TOKENS[1:])


@dataclass(frozen=True)
class TokenVocabulary:
    """A small, stable integer vocabulary."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("vocabulary tokens must be non-empty and unique")

    def encode(self, token: str, *, unknown_token: str | None = None) -> int:
        try:
            return self.tokens.index(token)
        except ValueError:
            if unknown_token is None:
                raise ValueError(f"unknown vocabulary token: {token}") from None
            try:
                return self.tokens.index(unknown_token)
            except ValueError:
                raise ValueError(
                    f"unknown fallback token {unknown_token!r} is not in the vocabulary"
                ) from None


@dataclass(frozen=True)
class ParameterDefinition:
    """One skill-local parameter and its dense-vector slots."""

    skill_id: str
    name: str
    kind: str
    indices: tuple[int, ...]
    lower: float | None = None
    upper: float | None = None
    choices: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ParameterSchema:
    """Versioned dense encoding derived only from the 34 formal skill YAML files."""

    version: int
    formal_skill_ids: tuple[str, ...]
    definitions: tuple[ParameterDefinition, ...]
    dimension: int

    def encode(
        self,
        skill_id: str,
        parameters: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros(self.dimension, dtype=np.float32)
        mask = np.zeros(self.dimension, dtype=bool)
        if skill_id == NONE_SKILL_ID:
            if parameters:
                raise ValueError("the <none> skill cannot have parameters")
            return values, mask
        if skill_id not in self.formal_skill_ids:
            raise ValueError(f"parameters may only reference a formal skill: {skill_id}")
        if parameters is None:
            return values, mask

        definitions = [item for item in self.definitions if item.skill_id == skill_id]
        expected = {item.name for item in definitions}
        actual = set(parameters)
        if actual != expected:
            raise ValueError(
                f"parameter names differ for {skill_id}: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )

        for definition in definitions:
            value = parameters[definition.name]
            if definition.kind == "continuous":
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"{skill_id}.{definition.name} must be a finite number"
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(
                        f"{skill_id}.{definition.name} must be a finite number"
                    )
                assert definition.lower is not None and definition.upper is not None
                if not definition.lower <= number <= definition.upper:
                    raise ValueError(f"{skill_id}.{definition.name} is outside its range")
                span = definition.upper - definition.lower
                values[definition.indices[0]] = (
                    0.0 if span == 0.0 else (number - definition.lower) / span
                )
                mask[definition.indices[0]] = True
            elif definition.kind == "categorical":
                try:
                    selected = definition.choices.index(value)
                except ValueError:
                    raise ValueError(
                        f"{skill_id}.{definition.name} is not an allowed choice"
                    ) from None
                values[list(definition.indices)] = 0.0
                values[definition.indices[selected]] = 1.0
                mask[list(definition.indices)] = True
            else:
                raise RuntimeError(f"unknown parameter kind: {definition.kind}")
        return values, mask


@dataclass(frozen=True)
class CVAESchema:
    """Formal skill, role, actor, map, and parameter vocabularies."""

    formal_skills: tuple[SkillSpec, ...]
    formal_skill_ids: tuple[str, ...]
    candidate_skill_ids: tuple[str, ...]
    skill_vocabulary: TokenVocabulary
    actor_type_vocabulary: TokenVocabulary
    role_vocabulary: TokenVocabulary
    map_type_vocabulary: TokenVocabulary
    parameter_schema: ParameterSchema


@dataclass(frozen=True)
class SampleSpec:
    """One deterministic base or observed-skill training example."""

    scenario_id: str
    target_track_id: str
    skill_id: str = NONE_SKILL_ID
    skill_supervision_mask: bool = False
    responder_track_id: str | None = None
    role_track_ids: tuple[tuple[str, str], ...] = ()
    trigger_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.target_track_id:
            raise ValueError("scenario_id and target_track_id must be non-empty")
        canonical_roles = tuple(sorted((str(role), str(track)) for role, track in self.role_track_ids))
        if len({role for role, _ in canonical_roles}) != len(canonical_roles):
            raise ValueError("role names must be unique")
        if len({track for _, track in canonical_roles}) != len(canonical_roles):
            raise ValueError("role tracks must be distinct")
        object.__setattr__(self, "role_track_ids", canonical_roles)
        if self.skill_supervision_mask:
            if self.skill_id == NONE_SKILL_ID:
                raise ValueError("a supervised sample must have a formal skill_id")
            if self.responder_track_id is None:
                raise ValueError("a supervised sample must have a responder")
            role_tracks = {track for _, track in canonical_roles}
            if self.target_track_id not in role_tracks:
                raise ValueError("supervised roles must include the target track")
            if self.responder_track_id not in role_tracks:
                raise ValueError("supervised roles must include the responder track")
        elif self.skill_id != NONE_SKILL_ID or self.responder_track_id is not None or canonical_roles:
            raise ValueError("an unsupervised sample must use only the <none> condition")
        if not math.isfinite(float(self.trigger_score)):
            raise ValueError("trigger_score must be finite")

    @property
    def sample_id(self) -> str:
        payload = json.dumps(
            {
                "scenario_id": self.scenario_id,
                "target_track_id": self.target_track_id,
                "skill_id": self.skill_id,
                "responder_track_id": self.responder_track_id,
                "role_track_ids": self.role_track_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.scenario_id,
            self.skill_id != NONE_SKILL_ID,
            self.skill_id,
            self.target_track_id,
            self.responder_track_id or "",
            self.role_track_ids,
        )


@dataclass(frozen=True)
class PriorContextSpec:
    """One future-free Prior condition for an arbitrary target actor."""

    scenario_id: str
    target_track_id: str
    condition_skill_id: str = NONE_SKILL_ID
    required_context_track_ids: tuple[str, ...] = ()
    role_track_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.target_track_id or not self.condition_skill_id:
            raise ValueError(
                "scenario_id, target_track_id, and condition_skill_id must be non-empty"
            )
        required = tuple(sorted(str(track_id) for track_id in self.required_context_track_ids))
        if any(not track_id for track_id in required):
            raise ValueError("required context track IDs must be non-empty")
        if len(set(required)) != len(required):
            raise ValueError("required context track IDs must be unique")
        if self.target_track_id in required:
            raise ValueError("required context tracks must not repeat the target track")
        object.__setattr__(self, "required_context_track_ids", required)

        canonical_roles = tuple(
            sorted((str(role), str(track)) for role, track in self.role_track_ids)
        )
        if any(not role or not track for role, track in canonical_roles):
            raise ValueError("role names and role track IDs must be non-empty")
        if len({role for role, _ in canonical_roles}) != len(canonical_roles):
            raise ValueError("role names must be unique")
        if len({track for _, track in canonical_roles}) != len(canonical_roles):
            raise ValueError("role tracks must be distinct")
        if self.condition_skill_id == NONE_SKILL_ID:
            if canonical_roles:
                raise ValueError("the <none> Prior condition must use only base actor roles")
        elif self.target_track_id not in {track for _, track in canonical_roles}:
            raise ValueError("a formal Prior condition must assign a role to the target track")
        object.__setattr__(self, "role_track_ids", canonical_roles)


@dataclass(frozen=True)
class MapClipStatistics:
    """Exact, non-tensor diagnostics for one sample's map clipping pipeline."""

    eligible_polylines: int
    retained_polylines: int
    dropped_polylines_due_to_limit: int
    original_in_radius_points: int
    retained_in_radius_points: int
    resampled_polylines_due_to_point_limit: int
    excess_input_points_over_point_limit: int

    def __post_init__(self) -> None:
        values = {
            name: getattr(self, name)
            for name in (
                "eligible_polylines",
                "retained_polylines",
                "dropped_polylines_due_to_limit",
                "original_in_radius_points",
                "retained_in_radius_points",
                "resampled_polylines_due_to_point_limit",
                "excess_input_points_over_point_limit",
            )
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("map clip statistics must be nonnegative integers")
        if self.retained_polylines != min(self.eligible_polylines, MAX_MAP_POLYLINES):
            raise ValueError("retained map polyline count is inconsistent")
        if (
            self.dropped_polylines_due_to_limit
            != self.eligible_polylines - self.retained_polylines
        ):
            raise ValueError("dropped map polyline count is inconsistent")
        if self.original_in_radius_points < self.eligible_polylines:
            raise ValueError("original in-radius map point count is inconsistent")
        if not (
            self.retained_polylines
            <= self.retained_in_radius_points
            <= self.original_in_radius_points
        ):
            raise ValueError("retained in-radius map point count is inconsistent")
        if (
            self.resampled_polylines_due_to_point_limit
            > self.retained_polylines
        ):
            raise ValueError("resampled map polyline count is inconsistent")
        if (
            self.resampled_polylines_due_to_point_limit == 0
        ) != (self.excess_input_points_over_point_limit == 0):
            raise ValueError("excess map point count is inconsistent")
        if (
            self.excess_input_points_over_point_limit
            < self.resampled_polylines_due_to_point_limit
        ):
            raise ValueError("excess map point count is inconsistent")


@dataclass(frozen=True)
class TensorizedSample:
    """Fixed-shape NumPy arrays consumed by the later Torch Dataset."""

    sample_id: str
    scenario_id: str
    target_track_id: str
    actor_track_ids: tuple[str, ...]
    map_polyline_ids: tuple[str, ...]
    map_clip_statistics: MapClipStatistics
    actor_history: np.ndarray
    actor_time_mask: np.ndarray
    actor_mask: np.ndarray
    actor_type_id: np.ndarray
    actor_role_id: np.ndarray
    map_polylines: np.ndarray
    map_point_mask: np.ndarray
    map_polyline_mask: np.ndarray
    map_type_id: np.ndarray
    target_actor_index: np.int64
    skill_id: np.int64
    skill_supervision_mask: np.bool_
    skill_parameters: np.ndarray
    parameter_mask: np.ndarray
    target_future: np.ndarray
    target_future_mask: np.ndarray
    anchor_origin_global: np.ndarray
    anchor_heading_global: np.float32


@dataclass(frozen=True)
class TensorizedPriorContext:
    """Fixed-shape model context that contains no target-future fields."""

    scenario_id: str
    target_track_id: str
    actor_track_ids: tuple[str, ...]
    map_polyline_ids: tuple[str, ...]
    map_clip_statistics: MapClipStatistics
    actor_history: np.ndarray
    actor_time_mask: np.ndarray
    actor_mask: np.ndarray
    actor_type_id: np.ndarray
    actor_role_id: np.ndarray
    map_polylines: np.ndarray
    map_point_mask: np.ndarray
    map_polyline_mask: np.ndarray
    map_type_id: np.ndarray
    target_actor_index: np.int64
    skill_id: np.int64
    skill_parameters: np.ndarray
    parameter_mask: np.ndarray
    anchor_origin_global: np.ndarray
    anchor_heading_global: np.float32


def _load_catalog(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"catalog must be a mapping: {path}")
    return data


def _catalog_skill_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    families = catalog.get("families")
    if not isinstance(families, Mapping) or not families:
        raise ValueError("formal catalog must contain non-empty families")
    skill_ids: list[str] = []
    for entries in families.values():
        if not isinstance(entries, list):
            raise ValueError("formal catalog family entries must be lists")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("skill_id"), str):
                raise ValueError("formal catalog entries must contain skill_id")
            skill_ids.append(entry["skill_id"])
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("formal catalog skill IDs must be unique")
    return tuple(skill_ids)


def _candidate_skill_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    entries = catalog.get("skills")
    if not isinstance(entries, list):
        raise ValueError("candidate catalog must contain a skills list")
    skill_ids = tuple(entry.get("skill_id") for entry in entries if isinstance(entry, Mapping))
    if any(not isinstance(skill_id, str) or not skill_id for skill_id in skill_ids):
        raise ValueError("candidate catalog entries must contain skill_id")
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("candidate catalog skill IDs must be unique")
    return skill_ids


def _build_parameter_schema(skills: tuple[SkillSpec, ...]) -> ParameterSchema:
    definitions: list[ParameterDefinition] = []
    next_index = 0
    for skill in skills:
        for name in sorted(skill.parameters):
            spec = skill.parameters[name]
            if "range" in spec:
                low, high = spec["range"]
                definitions.append(
                    ParameterDefinition(
                        skill_id=skill.skill_id,
                        name=name,
                        kind="continuous",
                        indices=(next_index,),
                        lower=float(low),
                        upper=float(high),
                    )
                )
                next_index += 1
            elif "choices" in spec:
                choices = tuple(spec["choices"])
                indices = tuple(range(next_index, next_index + len(choices)))
                definitions.append(
                    ParameterDefinition(
                        skill_id=skill.skill_id,
                        name=name,
                        kind="categorical",
                        indices=indices,
                        choices=choices,
                    )
                )
                next_index += len(choices)
            else:
                raise ValueError(
                    f"{skill.skill_id}.{name} must define either range or choices"
                )
    return ParameterSchema(
        version=1,
        formal_skill_ids=tuple(skill.skill_id for skill in skills),
        definitions=tuple(definitions),
        dimension=next_index,
    )


def build_cvae_schema(skill_dir: str | Path = "configs/skills") -> CVAESchema:
    """Build the frozen 34-formal/5-candidate conditioning contract."""

    directory = Path(skill_dir)
    formal_catalog = _load_catalog(directory / "catalog.yaml")
    if formal_catalog.get("status") != "user_confirmed":
        raise ValueError("formal catalog must have status=user_confirmed")
    formal_skill_ids = _catalog_skill_ids(formal_catalog)
    if len(formal_skill_ids) != 34:
        raise ValueError(f"CVAE baseline requires 34 formal skills, got {len(formal_skill_ids)}")

    candidate_name = formal_catalog.get("candidate_catalog")
    if not isinstance(candidate_name, str) or not candidate_name:
        raise ValueError("formal catalog must reference candidate_catalog")
    candidate_catalog = _load_catalog(directory / candidate_name)
    candidate_skill_ids = _candidate_skill_ids(candidate_catalog)
    if len(candidate_skill_ids) != 5:
        raise ValueError(
            f"CVAE baseline requires 5 excluded candidate skills, got {len(candidate_skill_ids)}"
        )
    overlap = set(formal_skill_ids) & set(candidate_skill_ids)
    if overlap:
        raise ValueError(f"formal and candidate skills overlap: {sorted(overlap)}")

    skills = tuple(load_skill(directory / f"{skill_id}.yaml") for skill_id in formal_skill_ids)
    if tuple(skill.skill_id for skill in skills) != formal_skill_ids:
        raise ValueError("formal skill YAML IDs do not match catalog order")
    role_tokens = sorted(
        {
            role
            for skill in skills
            for role in skill.actors.get("generated_roles", [])
        }
    )
    return CVAESchema(
        formal_skills=skills,
        formal_skill_ids=formal_skill_ids,
        candidate_skill_ids=candidate_skill_ids,
        skill_vocabulary=TokenVocabulary((NONE_SKILL_ID, *formal_skill_ids)),
        actor_type_vocabulary=TokenVocabulary(ACTOR_TYPE_TOKENS),
        role_vocabulary=TokenVocabulary(
            (PADDING_TOKEN, CONTEXT_ROLE, BASE_TARGET_ROLE, *role_tokens)
        ),
        map_type_vocabulary=TokenVocabulary(MAP_TYPE_TOKENS),
        parameter_schema=_build_parameter_schema(skills),
    )


def make_base_sample_spec(scenario: Scenario) -> SampleSpec:
    """Create the one allowed unconditional prior sample for a scenario."""

    return SampleSpec(
        scenario_id=scenario.scenario_id,
        target_track_id=scenario.focal_track_id,
    )


def observed_sample_specs(
    records: Iterable[SeedRecord],
    schema: CVAESchema,
) -> tuple[SampleSpec, ...]:
    """Keep observed formal labels, skip compatible seeds, and deduplicate stably."""

    formal_ids = set(schema.formal_skill_ids)
    candidate_ids = set(schema.candidate_skill_ids)
    grouped: dict[tuple[str, str, str, str], list[SeedRecord]] = {}
    for record in records:
        if record.skill_id in candidate_ids:
            raise ValueError(f"candidate skill cannot enter CVAE samples: {record.skill_id}")
        if record.skill_id not in formal_ids:
            raise ValueError(f"seed record references an unknown formal skill: {record.skill_id}")
        mode = record.evidence.get("detection_mode")
        if mode == "compatible_seed":
            continue
        if mode != "observed_trigger":
            raise ValueError(f"unknown detection_mode for {record.skill_id}: {mode!r}")
        missing = record.evidence.get("missing_generation_conditions")
        if missing not in (None, []):
            raise ValueError("observed_trigger records cannot have missing generation conditions")
        key = (
            record.scenario_id,
            record.skill_id,
            record.initiator_track_id,
            record.responder_track_id,
        )
        grouped.setdefault(key, []).append(record)

    specs: list[SampleSpec] = []
    for records_for_key in grouped.values():
        selected = min(
            records_for_key,
            key=lambda record: (-record.trigger_score, record.unique_key),
        )
        target_track_id = selected.initiator_track_id
        if selected.skill_id == "short_headway_following":
            leader_id = selected.role_track_ids.get("leader")
            follower_id = selected.role_track_ids.get("close_follower")
            if not leader_id or not follower_id:
                raise ValueError(
                    "short_headway_following supervision requires leader and "
                    "close_follower roles"
                )
            if (
                leader_id != selected.initiator_track_id
                or follower_id != selected.responder_track_id
            ):
                raise ValueError(
                    "short_headway_following roles must map leader to initiator and "
                    "close_follower to responder"
                )
            target_track_id = follower_id
        specs.append(
            SampleSpec(
                scenario_id=selected.scenario_id,
                target_track_id=target_track_id,
                skill_id=selected.skill_id,
                skill_supervision_mask=True,
                responder_track_id=selected.responder_track_id,
                role_track_ids=tuple(selected.role_track_ids.items()),
                trigger_score=selected.trigger_score,
            )
        )
    return tuple(sorted(specs, key=lambda spec: spec.sort_key))


def _validate_scenario(scenario: Scenario, spec: SampleSpec) -> dict[str, AgentTrack]:
    if scenario.scenario_id != spec.scenario_id:
        raise ValueError("sample scenario_id does not match the loaded Scenario")
    if len(scenario.timestamps) != TOTAL_STEPS:
        raise ValueError(f"Scenario must contain exactly {TOTAL_STEPS} timestamps")
    if np.any(np.diff(scenario.timestamps) <= 0):
        raise ValueError("Scenario timestamps must be strictly increasing")
    agents = {agent.track_id: agent for agent in scenario.agents}
    for agent in scenario.agents:
        if len(agent.positions) != TOTAL_STEPS:
            raise ValueError(f"agent {agent.track_id} does not contain {TOTAL_STEPS} states")
    if spec.target_track_id not in agents:
        raise ValueError(f"target track is missing: {spec.target_track_id}")
    if spec.responder_track_id is not None and spec.responder_track_id not in agents:
        raise ValueError(f"responder track is missing: {spec.responder_track_id}")
    for _, track_id in spec.role_track_ids:
        if track_id not in agents:
            raise ValueError(f"role track is missing: {track_id}")
    return agents


def _validate_prior_scenario(
    scenario: Scenario,
    spec: PriorContextSpec,
) -> dict[str, AgentTrack]:
    if scenario.scenario_id != spec.scenario_id:
        raise ValueError("Prior context scenario_id does not match the loaded Scenario")
    if len(scenario.timestamps) < HISTORY_STEPS:
        raise ValueError(f"Prior context requires at least {HISTORY_STEPS} timestamps")
    history_timestamps = scenario.timestamps[:HISTORY_STEPS]
    if np.any(np.diff(history_timestamps) <= 0):
        raise ValueError("Prior context history timestamps must be strictly increasing")
    agents = {agent.track_id: agent for agent in scenario.agents}
    for agent in scenario.agents:
        if len(agent.positions) < HISTORY_STEPS:
            raise ValueError(
                f"Prior context actor {agent.track_id} has fewer than "
                f"{HISTORY_STEPS} states"
            )
    required_tracks = {
        spec.target_track_id,
        *spec.required_context_track_ids,
        *(track_id for _, track_id in spec.role_track_ids),
    }
    missing = sorted(required_tracks - set(agents))
    if missing:
        raise ValueError(f"Prior context tracks are missing: {missing}")
    return agents


def _skill_threshold(schema: CVAESchema, skill_id: str, name: str) -> float:
    skill = next(
        (item for item in schema.formal_skills if item.skill_id == skill_id),
        None,
    )
    if skill is None:
        raise ValueError(f"unknown formal skill for future supervision: {skill_id}")
    thresholds = skill.detection.get("thresholds")
    item = thresholds.get(name) if isinstance(thresholds, Mapping) else None
    value = item.get("value") if isinstance(item, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{skill_id} detection threshold {name} is invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{skill_id} detection threshold {name} must be finite")
    return number


def _longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _validate_short_headway_future(
    spec: SampleSpec,
    schema: CVAESchema,
    agents: Mapping[str, AgentTrack],
) -> None:
    roles = dict(spec.role_track_ids)
    leader_id = roles.get("leader")
    follower_id = roles.get("close_follower")
    if not leader_id or not follower_id:
        raise ValueError(
            "short_headway_following supervision requires leader and close_follower roles"
        )
    if spec.target_track_id != follower_id:
        raise ValueError(
            "short_headway_following supervision target must be the close_follower"
        )
    leader = agents[leader_id]
    follower = agents[follower_id]
    future = slice(HISTORY_STEPS, TOTAL_STEPS)
    leader_positions = leader.positions[future]
    follower_positions = follower.positions[future]
    follower_velocities = follower.velocities[future]
    follower_headings = follower.headings[future]
    valid = (
        np.isfinite(leader_positions).all(axis=1)
        & np.isfinite(follower_positions).all(axis=1)
        & np.isfinite(follower_velocities).all(axis=1)
        & np.isfinite(follower_headings)
    )
    speed = np.linalg.norm(follower_velocities, axis=1)
    forward = np.column_stack(
        (np.cos(follower_headings), np.sin(follower_headings))
    )
    lateral = np.column_stack(
        (-np.sin(follower_headings), np.cos(follower_headings))
    )
    relative_positions = leader_positions - follower_positions
    gap = np.sum(relative_positions * forward, axis=1)
    lateral_offset = np.abs(np.sum(relative_positions * lateral, axis=1))
    headway = np.full(FUTURE_STEPS, np.inf, dtype=np.float64)
    moving = valid & (speed > 1e-6)
    headway[moving] = gap[moving] / speed[moving]

    minimum_speed = _skill_threshold(
        schema,
        spec.skill_id,
        "minimum_follower_speed_mps",
    )
    maximum_headway = _skill_threshold(
        schema,
        spec.skill_id,
        "maximum_time_headway_s",
    )
    minimum_duration = _skill_threshold(
        schema,
        spec.skill_id,
        "minimum_duration_s",
    )
    matches = (
        moving
        & (speed >= minimum_speed)
        & (headway > 0.0)
        & (headway <= maximum_headway)
        & (lateral_offset <= SHORT_HEADWAY_SAME_LANE_MAX_LATERAL_OFFSET_M)
    )
    duration = _longest_true_run(matches) * SAMPLE_PERIOD_S
    if duration + 1e-9 < minimum_duration:
        raise ValueError(
            "short_headway_following event is not sustained in prediction frames 50-109"
        )


def _validate_observed_future_supervision(
    spec: SampleSpec,
    schema: CVAESchema,
    agents: Mapping[str, AgentTrack],
) -> None:
    if not spec.skill_supervision_mask:
        return
    if spec.skill_id == "short_headway_following":
        _validate_short_headway_future(spec, schema, agents)


def _filled_history_headings(agent: AgentTrack) -> np.ndarray:
    """Fill headings using frames 0-49 only, never any prediction frame."""

    headings = agent.headings[:HISTORY_STEPS].astype(np.float64, copy=True)
    missing = ~np.isfinite(headings)
    velocities = agent.velocities[:HISTORY_STEPS]
    velocity_valid = np.isfinite(velocities).all(axis=1)
    speeds = np.linalg.norm(np.where(velocity_valid[:, None], velocities, 0.0), axis=1)
    use_velocity = missing & velocity_valid & (speeds > 1e-6)
    headings[use_velocity] = np.arctan2(
        velocities[use_velocity, 1],
        velocities[use_velocity, 0],
    )

    positions = agent.positions[:HISTORY_STEPS]
    deltas = positions[1:] - positions[:-1]
    delta_valid = np.isfinite(deltas).all(axis=1)
    delta_norm = np.linalg.norm(np.where(delta_valid[:, None], deltas, 0.0), axis=1)
    indices = np.flatnonzero(missing[1:] & delta_valid & (delta_norm > 1e-6)) + 1
    headings[indices] = np.arctan2(deltas[indices - 1, 1], deltas[indices - 1, 0])
    return headings


def _history_mask(agent: AgentTrack, filled_headings: np.ndarray) -> np.ndarray:
    return (
        agent.observed_mask[:HISTORY_STEPS]
        & np.isfinite(agent.positions[:HISTORY_STEPS]).all(axis=1)
        & np.isfinite(agent.velocities[:HISTORY_STEPS]).all(axis=1)
        & np.isfinite(filled_headings[:HISTORY_STEPS])
    )


def _resolve_anchor(target: AgentTrack) -> tuple[np.ndarray, float, np.ndarray]:
    if not target.observed_mask[ANCHOR_INDEX]:
        raise ValueError("target must be observed at frame 49")
    origin = target.positions[ANCHOR_INDEX].astype(np.float64, copy=True)
    if not np.isfinite(origin).all():
        raise ValueError("target position at frame 49 must be finite")
    headings = _filled_history_headings(target)
    heading = float(headings[ANCHOR_INDEX])
    if not math.isfinite(heading):
        raise ValueError("target heading cannot be resolved from frames 0-49")
    history_mask = _history_mask(target, headings)
    if int(history_mask.sum()) < MINIMUM_TARGET_HISTORY_STEPS:
        raise ValueError("target has fewer than 30 valid history steps")
    if not history_mask[ANCHOR_INDEX]:
        raise ValueError("target history features at frame 49 must be valid")
    return origin, heading, history_mask


def _latest_history_position(agent: AgentTrack) -> tuple[int, np.ndarray] | None:
    valid = (
        agent.observed_mask[:HISTORY_STEPS]
        & np.isfinite(agent.positions[:HISTORY_STEPS]).all(axis=1)
    )
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    index = int(indices[-1])
    return index, agent.positions[index]


def _selected_agents(
    scenario: Scenario,
    spec: SampleSpec,
    agents: Mapping[str, AgentTrack],
    origin: np.ndarray,
) -> tuple[list[AgentTrack], dict[str, str]]:
    role_by_track = {track_id: role for role, track_id in spec.role_track_ids}
    explicit_ids = [spec.target_track_id]
    if (
        spec.responder_track_id is not None
        and spec.responder_track_id != spec.target_track_id
    ):
        explicit_ids.append(spec.responder_track_id)
    additional_roles = sorted(
        (
            (role, track_id)
            for role, track_id in spec.role_track_ids
            if track_id not in set(explicit_ids)
        ),
        key=lambda item: (item[0], item[1]),
    )
    explicit_ids.extend(track_id for _, track_id in additional_roles)
    if len(explicit_ids) != len(set(explicit_ids)):
        raise ValueError("explicit actor roles must reference distinct tracks")
    if len(explicit_ids) > MAX_ACTORS:
        raise ValueError("explicit actor roles exceed the actor limit")
    for track_id in explicit_ids:
        if _latest_history_position(agents[track_id]) is None:
            raise ValueError(f"explicit actor has no valid history: {track_id}")

    explicit_set = set(explicit_ids)
    neighbors: list[tuple[float, str, AgentTrack]] = []
    for agent in scenario.agents:
        if agent.track_id in explicit_set:
            continue
        latest = _latest_history_position(agent)
        if latest is None:
            continue
        _, position = latest
        distance = float(np.linalg.norm(position - origin))
        if distance > ACTOR_RADIUS_M:
            continue
        neighbors.append((distance, agent.track_id, agent))
    neighbors.sort(key=lambda item: (item[0], item[1]))
    selected_ids = explicit_ids + [
        agent.track_id
        for _, _, agent in neighbors[: MAX_ACTORS - len(explicit_ids)]
    ]
    return [agents[track_id] for track_id in selected_ids], role_by_track


def _selected_prior_agents(
    scenario: Scenario,
    spec: PriorContextSpec,
    agents: Mapping[str, AgentTrack],
    origin: np.ndarray,
) -> tuple[list[AgentTrack], dict[str, str]]:
    role_by_track = {track_id: role for role, track_id in spec.role_track_ids}
    explicit_context_ids = set(spec.required_context_track_ids)
    explicit_context_ids.update(role_by_track)
    explicit_context_ids.discard(spec.target_track_id)
    explicit_ids = [spec.target_track_id, *sorted(explicit_context_ids)]
    if len(explicit_ids) > MAX_ACTORS:
        raise ValueError("explicit Prior context actors exceed the actor limit")
    for track_id in explicit_ids:
        if _latest_history_position(agents[track_id]) is None:
            raise ValueError(f"explicit Prior context actor has no valid history: {track_id}")

    explicit_set = set(explicit_ids)
    neighbors: list[tuple[float, str, AgentTrack]] = []
    for agent in scenario.agents:
        if agent.track_id in explicit_set:
            continue
        latest = _latest_history_position(agent)
        if latest is None:
            continue
        _, position = latest
        distance = float(np.linalg.norm(position - origin))
        if distance > ACTOR_RADIUS_M:
            continue
        neighbors.append((distance, agent.track_id, agent))
    neighbors.sort(key=lambda item: (item[0], item[1]))
    selected_ids = explicit_ids + [
        agent.track_id
        for _, _, agent in neighbors[: MAX_ACTORS - len(explicit_ids)]
    ]
    return [agents[track_id] for track_id in selected_ids], role_by_track


def _actor_arrays(
    selected: list[AgentTrack],
    spec: SampleSpec | PriorContextSpec,
    schema: CVAESchema,
    origin: np.ndarray,
    heading: float,
    role_by_track: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    history = np.zeros((MAX_ACTORS, HISTORY_STEPS, ACTOR_FEATURE_DIM), dtype=np.float32)
    time_mask = np.zeros((MAX_ACTORS, HISTORY_STEPS), dtype=bool)
    actor_mask = np.zeros(MAX_ACTORS, dtype=bool)
    type_ids = np.zeros(MAX_ACTORS, dtype=np.int64)
    role_ids = np.zeros(MAX_ACTORS, dtype=np.int64)
    track_ids = [""] * MAX_ACTORS

    zero_origin = np.zeros(2, dtype=np.float64)
    for slot, agent in enumerate(selected):
        filled_headings = _filled_history_headings(agent)
        valid = _history_mask(agent, filled_headings)
        local_positions = global_to_local(
            agent.positions[:HISTORY_STEPS], origin, heading
        )
        local_velocities = global_to_local(
            agent.velocities[:HISTORY_STEPS], zero_origin, heading
        )
        relative_headings = wrap_angle(filled_headings[:HISTORY_STEPS] - heading)
        features = np.column_stack(
            (
                local_positions,
                local_velocities,
                np.sin(relative_headings),
                np.cos(relative_headings),
            )
        )
        # The mask is computed from raw values above before any NaN is replaced.
        history[slot] = np.where(valid[:, None], features, 0.0).astype(np.float32)
        time_mask[slot] = valid
        actor_mask[slot] = bool(valid.any())
        type_ids[slot] = schema.actor_type_vocabulary.encode(
            agent.object_type.lower(), unknown_token="unknown"
        )
        if agent.track_id in role_by_track:
            role = role_by_track[agent.track_id]
        elif agent.track_id == spec.target_track_id:
            role = BASE_TARGET_ROLE
        else:
            role = CONTEXT_ROLE
        role_ids[slot] = schema.role_vocabulary.encode(role)
        track_ids[slot] = agent.track_id
    return history, time_mask, actor_mask, type_ids, role_ids, tuple(track_ids)


def _fit_polyline(points: np.ndarray) -> np.ndarray:
    if len(points) <= MAX_MAP_POINTS:
        return points.astype(np.float64, copy=True)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-9:
        return points[:1].astype(np.float64, copy=True)
    targets = np.linspace(0.0, cumulative[-1], MAX_MAP_POINTS)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(2)]
    )


def _polyline_tangents(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((len(points), 2), dtype=np.float64)
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 1e-9)


def _map_arrays(
    polylines: list[MapPolyline],
    schema: CVAESchema,
    origin: np.ndarray,
    heading: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    MapClipStatistics,
]:
    polyline_ids = [polyline.polyline_id for polyline in polylines]
    if len(polyline_ids) != len(set(polyline_ids)):
        raise ValueError("map polyline IDs must be unique")

    eligible: list[tuple[float, int, str, np.ndarray]] = []
    for polyline in polylines:
        if polyline.polyline_type not in ALLOWED_MAP_TYPES:
            continue
        finite = np.isfinite(polyline.points).all(axis=1)
        finite_points = polyline.points[finite]
        if not len(finite_points):
            continue
        distances = np.linalg.norm(finite_points - origin, axis=1)
        inside = distances <= MAP_RADIUS_M
        if not inside.any():
            continue
        points = finite_points[inside].astype(np.float64, copy=True)
        type_id = schema.map_type_vocabulary.encode(polyline.polyline_type)
        eligible.append(
            (float(distances[inside].min()), type_id, polyline.polyline_id, points)
        )
    eligible.sort(key=lambda item: (item[0], item[1], item[2]))
    retained = eligible[:MAX_MAP_POLYLINES]
    original_in_radius_points = sum(len(item[3]) for item in eligible)
    retained_in_radius_points = sum(len(item[3]) for item in retained)
    resampled_polylines = sum(len(item[3]) > MAX_MAP_POINTS for item in retained)
    excess_input_points = sum(
        max(len(item[3]) - MAX_MAP_POINTS, 0) for item in retained
    )
    statistics = MapClipStatistics(
        eligible_polylines=len(eligible),
        retained_polylines=len(retained),
        dropped_polylines_due_to_limit=len(eligible) - len(retained),
        original_in_radius_points=original_in_radius_points,
        retained_in_radius_points=retained_in_radius_points,
        resampled_polylines_due_to_point_limit=resampled_polylines,
        excess_input_points_over_point_limit=excess_input_points,
    )

    features = np.zeros(
        (MAX_MAP_POLYLINES, MAX_MAP_POINTS, MAP_FEATURE_DIM), dtype=np.float32
    )
    point_mask = np.zeros((MAX_MAP_POLYLINES, MAX_MAP_POINTS), dtype=bool)
    polyline_mask = np.zeros(MAX_MAP_POLYLINES, dtype=bool)
    type_ids = np.zeros(MAX_MAP_POLYLINES, dtype=np.int64)
    selected_ids = [""] * MAX_MAP_POLYLINES
    for slot, (_, type_id, polyline_id, raw_points) in enumerate(retained):
        points = _fit_polyline(raw_points)
        local_points = global_to_local(points, origin, heading)
        tangents = _polyline_tangents(local_points)
        count = len(local_points)
        raw_valid = np.isfinite(local_points).all(axis=1) & np.isfinite(tangents).all(axis=1)
        combined = np.column_stack((local_points, tangents))
        # As with actors, create the raw validity mask before replacing NaNs.
        features[slot, :count] = np.where(raw_valid[:, None], combined, 0.0).astype(
            np.float32
        )
        point_mask[slot, :count] = raw_valid
        polyline_mask[slot] = bool(raw_valid.any())
        type_ids[slot] = type_id
        selected_ids[slot] = polyline_id
    return (
        features,
        point_mask,
        polyline_mask,
        type_ids,
        tuple(selected_ids),
        statistics,
    )


def tensorize_prior_context(
    scenario: Scenario,
    spec: PriorContextSpec,
    schema: CVAESchema,
) -> TensorizedPriorContext:
    """Tensorize frames 0-49 for Prior inference without any future fields."""

    if spec.condition_skill_id not in schema.skill_vocabulary.tokens:
        raise ValueError(
            "Prior condition skill is not in the formal CVAE vocabulary: "
            f"{spec.condition_skill_id}"
        )
    agents = _validate_prior_scenario(scenario, spec)
    target = agents[spec.target_track_id]
    origin, heading, _ = _resolve_anchor(target)
    selected, role_by_track = _selected_prior_agents(
        scenario,
        spec,
        agents,
        origin,
    )
    (
        actor_history,
        actor_time_mask,
        actor_mask,
        actor_type_id,
        actor_role_id,
        actor_track_ids,
    ) = _actor_arrays(selected, spec, schema, origin, heading, role_by_track)
    (
        map_polylines,
        map_point_mask,
        map_polyline_mask,
        map_type_id,
        map_polyline_ids,
        map_clip_statistics,
    ) = _map_arrays(scenario.map_polylines, schema, origin, heading)
    parameter_values, parameter_mask = schema.parameter_schema.encode(
        spec.condition_skill_id,
        None,
    )
    return TensorizedPriorContext(
        scenario_id=scenario.scenario_id,
        target_track_id=spec.target_track_id,
        actor_track_ids=actor_track_ids,
        map_polyline_ids=map_polyline_ids,
        map_clip_statistics=map_clip_statistics,
        actor_history=actor_history,
        actor_time_mask=actor_time_mask,
        actor_mask=actor_mask,
        actor_type_id=actor_type_id,
        actor_role_id=actor_role_id,
        map_polylines=map_polylines,
        map_point_mask=map_point_mask,
        map_polyline_mask=map_polyline_mask,
        map_type_id=map_type_id,
        target_actor_index=np.int64(0),
        skill_id=np.int64(schema.skill_vocabulary.encode(spec.condition_skill_id)),
        skill_parameters=parameter_values,
        parameter_mask=parameter_mask,
        anchor_origin_global=origin.astype(np.float32),
        anchor_heading_global=np.float32(heading),
    )


def tensorize_scenario(
    scenario: Scenario,
    spec: SampleSpec,
    schema: CVAESchema,
) -> TensorizedSample:
    """Tensorize one arbitrary target at frame 49 without future leakage."""

    if spec.skill_id not in schema.skill_vocabulary.tokens:
        raise ValueError(f"sample skill is not in the formal CVAE vocabulary: {spec.skill_id}")
    agents = _validate_scenario(scenario, spec)
    _validate_observed_future_supervision(spec, schema, agents)
    target = agents[spec.target_track_id]
    origin, heading, _ = _resolve_anchor(target)
    selected, role_by_track = _selected_agents(scenario, spec, agents, origin)
    (
        actor_history,
        actor_time_mask,
        actor_mask,
        actor_type_id,
        actor_role_id,
        actor_track_ids,
    ) = _actor_arrays(selected, spec, schema, origin, heading, role_by_track)
    (
        map_polylines,
        map_point_mask,
        map_polyline_mask,
        map_type_id,
        map_polyline_ids,
        map_clip_statistics,
    ) = _map_arrays(scenario.map_polylines, schema, origin, heading)

    future_positions = target.positions[HISTORY_STEPS:TOTAL_STEPS]
    future_mask = np.isfinite(future_positions).all(axis=1)
    if not future_mask.all():
        raise ValueError("target future must contain 60 finite positions")
    local_future = global_to_local(future_positions, origin, heading)
    target_future = np.where(future_mask[:, None], local_future, 0.0).astype(np.float32)

    parameter_values, parameter_mask = schema.parameter_schema.encode(spec.skill_id, None)
    return TensorizedSample(
        sample_id=spec.sample_id,
        scenario_id=scenario.scenario_id,
        target_track_id=spec.target_track_id,
        actor_track_ids=actor_track_ids,
        map_polyline_ids=map_polyline_ids,
        map_clip_statistics=map_clip_statistics,
        actor_history=actor_history,
        actor_time_mask=actor_time_mask,
        actor_mask=actor_mask,
        actor_type_id=actor_type_id,
        actor_role_id=actor_role_id,
        map_polylines=map_polylines,
        map_point_mask=map_point_mask,
        map_polyline_mask=map_polyline_mask,
        map_type_id=map_type_id,
        target_actor_index=np.int64(0),
        skill_id=np.int64(schema.skill_vocabulary.encode(spec.skill_id)),
        skill_supervision_mask=np.bool_(spec.skill_supervision_mask),
        skill_parameters=parameter_values,
        parameter_mask=parameter_mask,
        target_future=target_future,
        target_future_mask=future_mask,
        anchor_origin_global=origin.astype(np.float32),
        anchor_heading_global=np.float32(heading),
    )


__all__ = [
    "ACTOR_FEATURE_DIM",
    "ACTOR_RADIUS_M",
    "ANCHOR_INDEX",
    "CVAESchema",
    "FUTURE_STEPS",
    "HISTORY_STEPS",
    "MAP_FEATURE_DIM",
    "MapClipStatistics",
    "MAX_ACTORS",
    "MAX_MAP_POINTS",
    "MAX_MAP_POLYLINES",
    "MAP_RADIUS_M",
    "SAMPLE_PERIOD_S",
    "NONE_SKILL_ID",
    "ParameterDefinition",
    "ParameterSchema",
    "PriorContextSpec",
    "SampleSpec",
    "TensorizedPriorContext",
    "TensorizedSample",
    "TokenVocabulary",
    "build_cvae_schema",
    "make_base_sample_spec",
    "observed_sample_specs",
    "tensorize_prior_context",
    "tensorize_scenario",
]

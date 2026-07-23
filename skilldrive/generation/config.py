"""Strict, immutable configuration for counterfactual generation stage A."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from skilldrive.skills.loader import ALLOWED_ACTOR_TYPES, load_skill


DEFAULT_COUNTERFACTUAL_CONFIG = Path("configs/generation/counterfactual_v1.yaml")
DEFAULT_FILTER_CONFIG = Path("configs/generation/filters_v1.yaml")

PROPOSAL_MODES = frozenset(
    {"learned_conditioned_prior", "rule_guided_prior_search"}
)
CONDITION_SKILL_STRATEGIES = frozenset({"requested_skill_id", "none_skill_id"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

LEARNED_CONDITIONED_SKILLS = frozenset(
    {
        "bike_lane_vehicle_merge_conflict",
        "crossing_path_conflict",
        "crosswalk_pedestrian_crossing",
        "cyclist_crossing",
        "cyclist_vehicle_merge",
        "jaywalking_pedestrian_crossing",
        "lead_sudden_stop",
        "ramp_merge_small_gap",
        "right_turn_vehicle_conflict",
        "short_headway_following",
        "slow_lead_blockage",
        "turning_vehicle_crosswalk_conflict",
        "unprotected_left_turn_conflict",
    }
)
OBSERVED_WITHOUT_TRAINING_SAMPLES = frozenset({"group_pedestrian_crossing"})

FINITE_VALUE_FIELDS = (
    "generated_positions",
    "derived_velocity",
    "derived_acceleration",
    "derived_heading",
    "risk_metrics",
    "distance_metrics",
    "realized_parameters",
)
FILTER_STAGES = (
    "schema_finite",
    "history_invariants",
    "kinematics",
    "map",
    "collision",
    "target_risk",
    "skill_trigger",
    "parameter_realization",
    "diversity",
)

COLLISION_PROXY_ACTOR_TYPES = frozenset(
    {
        *ALLOWED_ACTOR_TYPES,
        "background",
        "riderless_bicycle",
        "unknown",
    }
)
KINEMATIC_ACTOR_TYPES = frozenset(
    {"vehicle", "bus", "motorcyclist", "cyclist", "pedestrian"}
)
PARAMETER_ACTIONS = frozenset({"audit_only", "reject"})
PARAMETER_TOLERANCE_KEYS = (
    "seconds",
    "meters",
    "meters_per_second",
    "meters_per_second_squared",
    "degrees",
    "scale",
)


@dataclass(frozen=True)
class ActiveCheckpointConfig:
    path: Path
    sha256: str
    run_manifest: Path
    run_manifest_sha256: str
    schema_sha256: str
    run_manifest_stage: str = "formal"
    repair_contract: str | None = None
    promotion_recommendation: Path | None = None
    promotion_recommendation_sha256: str | None = None


@dataclass(frozen=True)
class GenerationInputConfig:
    data_root: Path
    seed_manifest: Path
    seed_manifest_sha256: str
    training_cache_manifest: Path
    training_cache_manifest_sha256: str
    leakage_audit: Path
    leakage_audit_sha256: str


@dataclass(frozen=True)
class SamplingConfig:
    base_seed: int
    pilot_seed_records_per_skill: int
    pilot_candidates_per_task: int
    formal_candidates_per_task: int


@dataclass(frozen=True)
class SkillGenerationConfig:
    skill_id: str
    primary_generated_role: str
    proposal_mode: str
    condition_skill_strategy: str
    joint_generation_limited: bool

    def condition_skill_id(self, none_skill_id: str) -> str:
        if self.condition_skill_strategy == "requested_skill_id":
            return self.skill_id
        return none_skill_id


@dataclass(frozen=True)
class CounterfactualGenerationConfig:
    version: int
    contract_name: str
    formal_catalog: Path
    candidate_catalog: Path
    none_skill_id: str
    active_checkpoint: ActiveCheckpointConfig
    inputs: GenerationInputConfig
    sampling: SamplingConfig
    formal_skill_ids: tuple[str, ...]
    candidate_skill_ids: tuple[str, ...]
    skills: tuple[SkillGenerationConfig, ...]

    @property
    def skills_by_id(self) -> Mapping[str, SkillGenerationConfig]:
        return MappingProxyType({item.skill_id: item for item in self.skills})


@dataclass(frozen=True)
class FootprintProxy:
    actor_type: str
    length_m: float
    width_m: float


@dataclass(frozen=True)
class KinematicClassPolicy:
    actor_type: str
    maximum_seam_speed_mps: float
    maximum_speed_mps: float
    maximum_acceleration_mps2: float
    maximum_deceleration_mps2: float
    maximum_jerk_mps3: float
    maximum_curvature_per_m: float
    maximum_heading_rate_rad_s: float
    minimum_heading_speed_mps: float


@dataclass(frozen=True)
class MapFilterPolicy:
    source: str
    required_drivable_actor_types: tuple[str, ...]
    minimum_inside_fraction: float
    lane_required_actor_types: tuple[str, ...]
    minimum_lane_assignment_fraction: float
    maximum_lane_distance_m: float
    maximum_heading_error_deg: float
    direction_exempt_skills: tuple[str, ...]


@dataclass(frozen=True)
class NoveltyFilterPolicy:
    source: str
    minimum_rms_displacement_m: float
    minimum_endpoint_displacement_m: float


@dataclass(frozen=True)
class ParameterFilterPolicy:
    source: str
    unavailable_action: str
    out_of_tolerance_action: str
    absolute_tolerances: Mapping[str, float]


@dataclass(frozen=True)
class DiversityFilterPolicy:
    source: str
    maximum_per_scenario_skill: int
    minimum_pairwise_rms_m: float
    minimum_endpoint_distance_m: float
    global_endpoint_bin_m: float
    global_risk_bin: float


@dataclass(frozen=True)
class CounterfactualFilterConfig:
    version: int
    contract_name: str
    finite_source: str
    finite_fields: tuple[str, ...]
    footprint_source: str
    footprint_basis: str
    footprint_ground_truth: bool
    footprint_proxies: tuple[FootprintProxy, ...]
    kinematic_source: str
    kinematic_basis: str
    kinematic_policies: tuple[KinematicClassPolicy, ...]
    map_policy: MapFilterPolicy
    novelty_policy: NoveltyFilterPolicy
    parameter_policy: ParameterFilterPolicy
    diversity_policy: DiversityFilterPolicy
    order_source: str
    filter_stages: tuple[str, ...]

    @property
    def footprints_by_type(self) -> Mapping[str, FootprintProxy]:
        return MappingProxyType({item.actor_type: item for item in self.footprint_proxies})

    @property
    def kinematics_by_type(self) -> Mapping[str, KinematicClassPolicy]:
        return MappingProxyType({item.actor_type: item for item in self.kinematic_policies})


def _load_yaml(path: Path, name: str) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must contain one string-keyed mapping: {path}")
    return value


def _section(value: Any, name: str, expected_keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    expected = set(expected_keys)
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ValueError(f"{name} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _sha256(value: Any, name: str) -> str:
    text = _text(value, name).lower()
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


def _text_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of unique strings")
    values = tuple(_text(item, f"{name} item") for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be a list of unique strings")
    return values


def _choice(value: Any, name: str, choices: frozenset[str]) -> str:
    text = _text(value, name)
    if text not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return text


def _repository_path(value: Any, name: str, repository_root: Path) -> tuple[Path, Path]:
    text = _text(value, name)
    relative = Path(text)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise ValueError(f"{name} must be a repository-relative path without '..'")
    return relative, repository_root / relative


def _formal_catalog_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    if catalog.get("status") != "user_confirmed":
        raise ValueError("formal catalog must have status=user_confirmed")
    families = catalog.get("families")
    if not isinstance(families, Mapping):
        raise ValueError("formal catalog families must be a mapping")
    skill_ids: list[str] = []
    for family, entries in families.items():
        _text(family, "formal catalog family")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"formal catalog family {family} must be a non-empty list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("formal catalog entries must be mappings")
            skill_ids.append(_text(entry.get("skill_id"), "formal catalog skill_id"))
    if len(skill_ids) != 34 or len(set(skill_ids)) != 34:
        raise ValueError("formal catalog must contain exactly 34 unique skills")
    return tuple(skill_ids)


def _candidate_catalog_ids(catalog: Mapping[str, Any]) -> tuple[str, ...]:
    if catalog.get("status") != "candidate_no_formal_seed":
        raise ValueError("candidate catalog must have status=candidate_no_formal_seed")
    entries = catalog.get("skills")
    if not isinstance(entries, list):
        raise ValueError("candidate catalog skills must be a list")
    skill_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("candidate catalog entries must be mappings")
        skill_ids.append(_text(entry.get("skill_id"), "candidate catalog skill_id"))
    if len(skill_ids) != 5 or len(set(skill_ids)) != 5:
        raise ValueError("candidate catalog must contain exactly 5 unique skills")
    return tuple(skill_ids)


def load_counterfactual_config(
    path: str | Path = DEFAULT_COUNTERFACTUAL_CONFIG,
    *,
    repository_root: str | Path = ".",
) -> CounterfactualGenerationConfig:
    source = Path(path)
    root = Path(repository_root)
    raw = _section(
        _load_yaml(source, "counterfactual generation config"),
        "counterfactual generation config",
        (
            "version",
            "contract_name",
            "formal_catalog",
            "candidate_catalog",
            "none_skill_id",
            "active_checkpoint",
            "inputs",
            "sampling",
            "skills",
        ),
    )
    if raw["version"] != 1:
        raise ValueError("counterfactual generation config version must be 1")
    if raw["contract_name"] != "counterfactual_v1":
        raise ValueError("counterfactual generation contract_name must be counterfactual_v1")
    none_skill_id = _text(raw["none_skill_id"], "none_skill_id")
    if none_skill_id != "<none>":
        raise ValueError("none_skill_id must be <none>")

    active_value = raw["active_checkpoint"]
    if not isinstance(active_value, Mapping):
        raise ValueError("active_checkpoint must be a mapping")
    active_base_fields = {
        "path",
        "sha256",
        "run_manifest",
        "run_manifest_sha256",
        "schema_sha256",
    }
    active_repair_fields = {
        *active_base_fields,
        "run_manifest_stage",
        "repair_contract",
        "promotion_recommendation",
        "promotion_recommendation_sha256",
    }
    if set(active_value) == active_base_fields:
        active_raw = dict(active_value)
        run_manifest_stage = "formal"
        repair_contract = None
        promotion_relative = None
        promotion_sha256 = None
    elif set(active_value) == active_repair_fields:
        active_raw = dict(active_value)
        run_manifest_stage = _choice(
            active_raw["run_manifest_stage"],
            "active_checkpoint.run_manifest_stage",
            frozenset({"repair-formal"}),
        )
        repair_contract = _text(
            active_raw["repair_contract"],
            "active_checkpoint.repair_contract",
        )
        if repair_contract != "cvae_generation_repair_v1":
            raise ValueError(
                "active_checkpoint.repair_contract must be "
                "cvae_generation_repair_v1"
            )
        promotion_relative, _ = _repository_path(
            active_raw["promotion_recommendation"],
            "active_checkpoint.promotion_recommendation",
            root,
        )
        promotion_sha256 = _sha256(
            active_raw["promotion_recommendation_sha256"],
            "active_checkpoint.promotion_recommendation_sha256",
        )
    else:
        repair_only_fields = active_repair_fields - active_base_fields
        expected_fields = (
            active_repair_fields
            if set(active_value) & repair_only_fields
            else active_base_fields
        )
        missing = sorted(expected_fields - set(active_value))
        unknown = sorted(set(active_value) - active_repair_fields)
        raise ValueError(
            "active_checkpoint has missing or unknown keys: "
            f"missing={missing}, unknown={unknown}"
        )
    checkpoint_relative, _ = _repository_path(
        active_raw["path"], "active_checkpoint.path", root
    )
    run_manifest_relative, _ = _repository_path(
        active_raw["run_manifest"], "active_checkpoint.run_manifest", root
    )
    active_checkpoint = ActiveCheckpointConfig(
        path=checkpoint_relative,
        sha256=_sha256(active_raw["sha256"], "active_checkpoint.sha256"),
        run_manifest=run_manifest_relative,
        run_manifest_sha256=_sha256(
            active_raw["run_manifest_sha256"],
            "active_checkpoint.run_manifest_sha256",
        ),
        schema_sha256=_sha256(
            active_raw["schema_sha256"], "active_checkpoint.schema_sha256"
        ),
        run_manifest_stage=run_manifest_stage,
        repair_contract=repair_contract,
        promotion_recommendation=promotion_relative,
        promotion_recommendation_sha256=promotion_sha256,
    )
    inputs_raw = _section(
        raw["inputs"],
        "inputs",
        (
            "data_root",
            "seed_manifest",
            "seed_manifest_sha256",
            "training_cache_manifest",
            "training_cache_manifest_sha256",
            "leakage_audit",
            "leakage_audit_sha256",
        ),
    )
    data_root_relative, _ = _repository_path(inputs_raw["data_root"], "inputs.data_root", root)
    seed_manifest_relative, _ = _repository_path(
        inputs_raw["seed_manifest"], "inputs.seed_manifest", root
    )
    training_cache_relative, _ = _repository_path(
        inputs_raw["training_cache_manifest"],
        "inputs.training_cache_manifest",
        root,
    )
    leakage_audit_relative, _ = _repository_path(
        inputs_raw["leakage_audit"], "inputs.leakage_audit", root
    )
    inputs = GenerationInputConfig(
        data_root=data_root_relative,
        seed_manifest=seed_manifest_relative,
        seed_manifest_sha256=_sha256(
            inputs_raw["seed_manifest_sha256"], "inputs.seed_manifest_sha256"
        ),
        training_cache_manifest=training_cache_relative,
        training_cache_manifest_sha256=_sha256(
            inputs_raw["training_cache_manifest_sha256"],
            "inputs.training_cache_manifest_sha256",
        ),
        leakage_audit=leakage_audit_relative,
        leakage_audit_sha256=_sha256(
            inputs_raw["leakage_audit_sha256"], "inputs.leakage_audit_sha256"
        ),
    )
    sampling_raw = _section(
        raw["sampling"],
        "sampling",
        (
            "base_seed",
            "pilot_seed_records_per_skill",
            "pilot_candidates_per_task",
            "formal_candidates_per_task",
        ),
    )
    sampling = SamplingConfig(
        base_seed=_positive_integer(sampling_raw["base_seed"], "sampling.base_seed"),
        pilot_seed_records_per_skill=_positive_integer(
            sampling_raw["pilot_seed_records_per_skill"],
            "sampling.pilot_seed_records_per_skill",
        ),
        pilot_candidates_per_task=_positive_integer(
            sampling_raw["pilot_candidates_per_task"],
            "sampling.pilot_candidates_per_task",
        ),
        formal_candidates_per_task=_positive_integer(
            sampling_raw["formal_candidates_per_task"],
            "sampling.formal_candidates_per_task",
        ),
    )

    formal_relative, formal_path = _repository_path(
        raw["formal_catalog"], "formal_catalog", root
    )
    candidate_relative, candidate_path = _repository_path(
        raw["candidate_catalog"], "candidate_catalog", root
    )
    formal_catalog = _load_yaml(formal_path, "formal catalog")
    candidate_catalog = _load_yaml(candidate_path, "candidate catalog")
    if formal_catalog.get("candidate_catalog") != candidate_path.name:
        raise ValueError("formal catalog candidate_catalog reference does not match config")
    formal_skill_ids = _formal_catalog_ids(formal_catalog)
    candidate_skill_ids = _candidate_catalog_ids(candidate_catalog)
    overlap = set(formal_skill_ids) & set(candidate_skill_ids)
    if overlap:
        raise ValueError(f"formal and candidate skill partitions overlap: {sorted(overlap)}")
    if not LEARNED_CONDITIONED_SKILLS <= set(formal_skill_ids):
        raise ValueError("formal catalog is missing a v5 learned-conditioned skill")
    if not OBSERVED_WITHOUT_TRAINING_SAMPLES <= set(formal_skill_ids):
        raise ValueError("formal catalog is missing the observed zero-training skill")

    skill_entries = raw["skills"]
    if not isinstance(skill_entries, Mapping) or not all(
        isinstance(key, str) for key in skill_entries
    ):
        raise ValueError("skills must be a string-keyed mapping")
    missing = set(formal_skill_ids) - set(skill_entries)
    unknown = set(skill_entries) - set(formal_skill_ids)
    if missing:
        raise ValueError(f"skills is missing formal skill entries: {sorted(missing)}")
    if unknown:
        raise ValueError(f"skills contains non-formal entries: {sorted(unknown)}")

    parsed: list[SkillGenerationConfig] = []
    skill_directory = formal_path.parent
    for skill_id in formal_skill_ids:
        entry = _section(
            skill_entries[skill_id],
            f"skills.{skill_id}",
            (
                "primary_generated_role",
                "proposal_mode",
                "condition_skill_strategy",
                "joint_generation_limited",
            ),
        )
        primary_role = _text(
            entry["primary_generated_role"],
            f"skills.{skill_id}.primary_generated_role",
        )
        proposal_mode = _text(entry["proposal_mode"], f"skills.{skill_id}.proposal_mode")
        if proposal_mode not in PROPOSAL_MODES:
            raise ValueError(f"skills.{skill_id}.proposal_mode is unknown")
        condition_strategy = _text(
            entry["condition_skill_strategy"],
            f"skills.{skill_id}.condition_skill_strategy",
        )
        if condition_strategy not in CONDITION_SKILL_STRATEGIES:
            raise ValueError(f"skills.{skill_id}.condition_skill_strategy is unknown")

        expected_mode = (
            "learned_conditioned_prior"
            if skill_id in LEARNED_CONDITIONED_SKILLS
            else "rule_guided_prior_search"
        )
        if proposal_mode != expected_mode:
            raise ValueError(
                f"skills.{skill_id}.proposal_mode must be {expected_mode} for v1"
            )
        expected_strategy = (
            "requested_skill_id"
            if proposal_mode == "learned_conditioned_prior"
            else "none_skill_id"
        )
        if condition_strategy != expected_strategy:
            raise ValueError(
                f"skills.{skill_id}.condition_skill_strategy must be {expected_strategy}"
            )

        skill = load_skill(skill_directory / f"{skill_id}.yaml")
        generated_roles = skill.actors["generated_roles"]
        if primary_role not in generated_roles:
            raise ValueError(
                f"skills.{skill_id}.primary_generated_role is not in actors.generated_roles"
            )
        detection_mode = skill.detection["mode"]
        if skill_id in LEARNED_CONDITIONED_SKILLS and detection_mode != "observed_trigger":
            raise ValueError(f"learned-conditioned skill {skill_id} is not observed_trigger")
        if (
            skill_id not in LEARNED_CONDITIONED_SKILLS
            and skill_id not in OBSERVED_WITHOUT_TRAINING_SAMPLES
            and detection_mode != "compatible_seed"
        ):
            raise ValueError(f"rule-guided skill {skill_id} is not compatible_seed")

        parsed.append(
            SkillGenerationConfig(
                skill_id=skill_id,
                primary_generated_role=primary_role,
                proposal_mode=proposal_mode,
                condition_skill_strategy=condition_strategy,
                joint_generation_limited=_boolean(
                    entry["joint_generation_limited"],
                    f"skills.{skill_id}.joint_generation_limited",
                ),
            )
        )

    return CounterfactualGenerationConfig(
        version=1,
        contract_name="counterfactual_v1",
        formal_catalog=formal_relative,
        candidate_catalog=candidate_relative,
        none_skill_id=none_skill_id,
        active_checkpoint=active_checkpoint,
        inputs=inputs,
        sampling=sampling,
        formal_skill_ids=formal_skill_ids,
        candidate_skill_ids=candidate_skill_ids,
        skills=tuple(parsed),
    )


def load_filter_config(path: str | Path = DEFAULT_FILTER_CONFIG) -> CounterfactualFilterConfig:
    source = Path(path)
    raw = _section(
        _load_yaml(source, "counterfactual filter config"),
        "counterfactual filter config",
        (
            "version",
            "contract_name",
            "finite_values",
            "class_footprint_proxies",
            "kinematics",
            "map_policy",
            "novelty",
            "parameter_realization",
            "diversity",
            "filter_order",
        ),
    )
    if raw["version"] != 1:
        raise ValueError("counterfactual filter config version must be 1")
    if raw["contract_name"] != "filters_v1":
        raise ValueError("counterfactual filter contract_name must be filters_v1")

    finite = _section(raw["finite_values"], "finite_values", ("source", "required"))
    finite_source = _text(finite["source"], "finite_values.source")
    required = finite["required"]
    if not isinstance(required, list) or tuple(required) != FINITE_VALUE_FIELDS:
        raise ValueError("finite_values.required must match the frozen v1 finite fields")

    footprints = _section(
        raw["class_footprint_proxies"],
        "class_footprint_proxies",
        ("source", "basis", "ground_truth", "actor_types"),
    )
    footprint_source = _text(footprints["source"], "class_footprint_proxies.source")
    footprint_basis = _text(footprints["basis"], "class_footprint_proxies.basis")
    ground_truth = _boolean(
        footprints["ground_truth"], "class_footprint_proxies.ground_truth"
    )
    if ground_truth:
        raise ValueError("class footprint proxies cannot be marked as ground truth")
    actor_types = footprints["actor_types"]
    if (
        not isinstance(actor_types, Mapping)
        or set(actor_types) != COLLISION_PROXY_ACTOR_TYPES
    ):
        raise ValueError(
            "class_footprint_proxies.actor_types must cover all actor types defined by AV2"
        )
    parsed_footprints: list[FootprintProxy] = []
    for actor_type in sorted(COLLISION_PROXY_ACTOR_TYPES):
        item = _section(
            actor_types[actor_type],
            f"class_footprint_proxies.actor_types.{actor_type}",
            ("length_m", "width_m"),
        )
        parsed_footprints.append(
            FootprintProxy(
                actor_type=actor_type,
                length_m=_positive_number(
                    item["length_m"],
                    f"class_footprint_proxies.actor_types.{actor_type}.length_m",
                ),
                width_m=_positive_number(
                    item["width_m"],
                    f"class_footprint_proxies.actor_types.{actor_type}.width_m",
                ),
            )
        )

    kinematics = _section(
        raw["kinematics"],
        "kinematics",
        ("source", "basis", "actor_types"),
    )
    kinematic_source = _text(kinematics["source"], "kinematics.source")
    kinematic_basis = _text(kinematics["basis"], "kinematics.basis")
    kinematic_actor_types = kinematics["actor_types"]
    if (
        not isinstance(kinematic_actor_types, Mapping)
        or set(kinematic_actor_types) != KINEMATIC_ACTOR_TYPES
    ):
        raise ValueError("kinematics.actor_types must cover all generated actor classes")
    parsed_kinematics: list[KinematicClassPolicy] = []
    kinematic_fields = (
        "maximum_seam_speed_mps",
        "maximum_speed_mps",
        "maximum_acceleration_mps2",
        "maximum_deceleration_mps2",
        "maximum_jerk_mps3",
        "maximum_curvature_per_m",
        "maximum_heading_rate_rad_s",
        "minimum_heading_speed_mps",
    )
    for actor_type in sorted(KINEMATIC_ACTOR_TYPES):
        item = _section(
            kinematic_actor_types[actor_type],
            f"kinematics.actor_types.{actor_type}",
            kinematic_fields,
        )
        parsed_kinematics.append(
            KinematicClassPolicy(
                actor_type=actor_type,
                **{
                    field: _positive_number(
                        item[field],
                        f"kinematics.actor_types.{actor_type}.{field}",
                    )
                    for field in kinematic_fields
                },
            )
        )

    map_raw = _section(
        raw["map_policy"],
        "map_policy",
        (
            "source",
            "required_drivable_actor_types",
            "minimum_inside_fraction",
            "lane_required_actor_types",
            "minimum_lane_assignment_fraction",
            "maximum_lane_distance_m",
            "maximum_heading_error_deg",
            "direction_exempt_skills",
        ),
    )
    required_drivable_actor_types = _text_tuple(
        map_raw["required_drivable_actor_types"],
        "map_policy.required_drivable_actor_types",
    )
    lane_required_actor_types = _text_tuple(
        map_raw["lane_required_actor_types"],
        "map_policy.lane_required_actor_types",
    )
    if not set(required_drivable_actor_types) <= KINEMATIC_ACTOR_TYPES:
        raise ValueError("map_policy.required_drivable_actor_types contains unknown types")
    if not set(lane_required_actor_types) <= KINEMATIC_ACTOR_TYPES:
        raise ValueError("map_policy.lane_required_actor_types contains unknown types")
    map_policy = MapFilterPolicy(
        source=_text(map_raw["source"], "map_policy.source"),
        required_drivable_actor_types=required_drivable_actor_types,
        minimum_inside_fraction=_fraction(
            map_raw["minimum_inside_fraction"],
            "map_policy.minimum_inside_fraction",
        ),
        lane_required_actor_types=lane_required_actor_types,
        minimum_lane_assignment_fraction=_fraction(
            map_raw["minimum_lane_assignment_fraction"],
            "map_policy.minimum_lane_assignment_fraction",
        ),
        maximum_lane_distance_m=_positive_number(
            map_raw["maximum_lane_distance_m"],
            "map_policy.maximum_lane_distance_m",
        ),
        maximum_heading_error_deg=_positive_number(
            map_raw["maximum_heading_error_deg"],
            "map_policy.maximum_heading_error_deg",
        ),
        direction_exempt_skills=_text_tuple(
            map_raw["direction_exempt_skills"],
            "map_policy.direction_exempt_skills",
        ),
    )

    novelty_raw = _section(
        raw["novelty"],
        "novelty",
        ("source", "minimum_rms_displacement_m", "minimum_endpoint_displacement_m"),
    )
    novelty_policy = NoveltyFilterPolicy(
        source=_text(novelty_raw["source"], "novelty.source"),
        minimum_rms_displacement_m=_positive_number(
            novelty_raw["minimum_rms_displacement_m"],
            "novelty.minimum_rms_displacement_m",
        ),
        minimum_endpoint_displacement_m=_positive_number(
            novelty_raw["minimum_endpoint_displacement_m"],
            "novelty.minimum_endpoint_displacement_m",
        ),
    )

    parameter_raw = _section(
        raw["parameter_realization"],
        "parameter_realization",
        (
            "source",
            "unavailable_action",
            "out_of_tolerance_action",
            "absolute_tolerances",
        ),
    )
    tolerance_raw = _section(
        parameter_raw["absolute_tolerances"],
        "parameter_realization.absolute_tolerances",
        PARAMETER_TOLERANCE_KEYS,
    )
    parameter_policy = ParameterFilterPolicy(
        source=_text(parameter_raw["source"], "parameter_realization.source"),
        unavailable_action=_choice(
            parameter_raw["unavailable_action"],
            "parameter_realization.unavailable_action",
            PARAMETER_ACTIONS,
        ),
        out_of_tolerance_action=_choice(
            parameter_raw["out_of_tolerance_action"],
            "parameter_realization.out_of_tolerance_action",
            PARAMETER_ACTIONS,
        ),
        absolute_tolerances=MappingProxyType(
            {
                key: _positive_number(
                    tolerance_raw[key],
                    f"parameter_realization.absolute_tolerances.{key}",
                )
                for key in PARAMETER_TOLERANCE_KEYS
            }
        ),
    )

    diversity_raw = _section(
        raw["diversity"],
        "diversity",
        (
            "source",
            "maximum_per_scenario_skill",
            "minimum_pairwise_rms_m",
            "minimum_endpoint_distance_m",
            "global_endpoint_bin_m",
            "global_risk_bin",
        ),
    )
    diversity_policy = DiversityFilterPolicy(
        source=_text(diversity_raw["source"], "diversity.source"),
        maximum_per_scenario_skill=_positive_integer(
            diversity_raw["maximum_per_scenario_skill"],
            "diversity.maximum_per_scenario_skill",
        ),
        minimum_pairwise_rms_m=_positive_number(
            diversity_raw["minimum_pairwise_rms_m"],
            "diversity.minimum_pairwise_rms_m",
        ),
        minimum_endpoint_distance_m=_positive_number(
            diversity_raw["minimum_endpoint_distance_m"],
            "diversity.minimum_endpoint_distance_m",
        ),
        global_endpoint_bin_m=_positive_number(
            diversity_raw["global_endpoint_bin_m"],
            "diversity.global_endpoint_bin_m",
        ),
        global_risk_bin=_positive_number(
            diversity_raw["global_risk_bin"],
            "diversity.global_risk_bin",
        ),
    )

    order = _section(raw["filter_order"], "filter_order", ("source", "stages"))
    order_source = _text(order["source"], "filter_order.source")
    stages = order["stages"]
    if not isinstance(stages, list) or tuple(stages) != FILTER_STAGES:
        raise ValueError("filter_order.stages must match the frozen v1 order")

    return CounterfactualFilterConfig(
        version=1,
        contract_name="filters_v1",
        finite_source=finite_source,
        finite_fields=FINITE_VALUE_FIELDS,
        footprint_source=footprint_source,
        footprint_basis=footprint_basis,
        footprint_ground_truth=ground_truth,
        footprint_proxies=tuple(parsed_footprints),
        kinematic_source=kinematic_source,
        kinematic_basis=kinematic_basis,
        kinematic_policies=tuple(parsed_kinematics),
        map_policy=map_policy,
        novelty_policy=novelty_policy,
        parameter_policy=parameter_policy,
        diversity_policy=diversity_policy,
        order_source=order_source,
        filter_stages=FILTER_STAGES,
    )


__all__ = [
    "ActiveCheckpointConfig",
    "CONDITION_SKILL_STRATEGIES",
    "COLLISION_PROXY_ACTOR_TYPES",
    "DEFAULT_COUNTERFACTUAL_CONFIG",
    "DEFAULT_FILTER_CONFIG",
    "FILTER_STAGES",
    "FINITE_VALUE_FIELDS",
    "KINEMATIC_ACTOR_TYPES",
    "LEARNED_CONDITIONED_SKILLS",
    "OBSERVED_WITHOUT_TRAINING_SAMPLES",
    "PROPOSAL_MODES",
    "CounterfactualFilterConfig",
    "CounterfactualGenerationConfig",
    "DiversityFilterPolicy",
    "FootprintProxy",
    "GenerationInputConfig",
    "KinematicClassPolicy",
    "MapFilterPolicy",
    "NoveltyFilterPolicy",
    "ParameterFilterPolicy",
    "SkillGenerationConfig",
    "SamplingConfig",
    "load_counterfactual_config",
    "load_filter_config",
]

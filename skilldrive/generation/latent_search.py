"""Frozen representative-task contract for stage-D latent search."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from skilldrive.generation.config import CounterfactualGenerationConfig
from skilldrive.generation.contracts import (
    GenerationTask,
    canonical_json_bytes,
    canonical_sha256,
)
from skilldrive.generation.planning import (
    PilotEvaluationArm,
    pilot_evaluation_arm,
    semantic_generation_config_sha256,
)


LATENT_SEARCH_CONFIG_VERSION = 1
LATENT_SEARCH_MANIFEST_VERSION = 1
LATENT_SEARCH_CANDIDATE_BUDGET = 4096
LATENT_SEARCH_CHUNK_SIZE = 512
LATENT_SEARCH_TOP_K = 64

_FILTER_STAGE_ORDER = (
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
_EXPECTED_REPRESENTATIVES = (
    (
        "forced_lane_change_deepest_funnel",
        "forced_lane_change_around_blockage",
        ("rule_guided_none",),
        "deepest_formal_funnel",
    ),
    (
        "jaywalking_condition_reverse",
        "jaywalking_pedestrian_crossing",
        ("learned_conditioned", "learned_none_control"),
        "none_control_outperforms_conditioned",
    ),
    (
        "slow_lead_learned_failure",
        "slow_lead_blockage",
        ("learned_conditioned", "learned_none_control"),
        "both_arms_fail_kinematics",
    ),
    (
        "construction_rule_deepest_funnel",
        "construction_object_lane_blockage",
        ("rule_guided_none",),
        "deepest_formal_funnel",
    ),
)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256_text(value: Any, name: str) -> str:
    text = _required_text(value, name)
    if len(text) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    try:
        int(text, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 string") from error
    if text.lower() != text:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return text


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _strict_mapping(value: Any, name: str, fields: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} must contain exactly {tuple(fields)}")
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping: {path}")
    canonical_json_bytes(value)
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical_json_bytes(value, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _repo_relative(path: Path, repository_root: Path) -> str:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact must stay inside the repository: {resolved}") from error


def _resolve_repo_path(value: Any, repository_root: Path, name: str) -> Path:
    text = _required_text(value, name)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be a repository-relative path")
    resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} escapes the repository") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is missing: {resolved}")
    return resolved


@dataclass(frozen=True)
class LatentSearchRepresentativeSpec:
    representative_id: str
    skill_id: str
    required_arms: tuple[PilotEvaluationArm, ...]
    evidence_contract: str

    def __post_init__(self) -> None:
        for name in ("representative_id", "skill_id", "evidence_contract"):
            _required_text(getattr(self, name), name)
        arms = tuple(self.required_arms)
        if not arms or len(set(arms)) != len(arms):
            raise ValueError("required_arms must contain unique Pilot arms")
        object.__setattr__(self, "required_arms", arms)


@dataclass(frozen=True)
class LatentSearchConfig:
    version: int
    contract_name: str
    base_seed: int
    candidate_budget_per_arm: int
    generation_chunk_size: int
    kinematic_top_k: int
    use_bfloat16: bool
    raw_policy: str
    latent_contract: str
    external_smoothing: bool
    interpolation_repair: bool
    filter_threshold_overrides: bool
    representatives: tuple[LatentSearchRepresentativeSpec, ...]

    def __post_init__(self) -> None:
        if self.version != LATENT_SEARCH_CONFIG_VERSION:
            raise ValueError("unsupported latent-search config version")
        if self.contract_name != "latent_search_v1":
            raise ValueError("latent-search contract_name must be latent_search_v1")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise ValueError("base_seed must be a nonnegative integer")
        if self.base_seed < 0:
            raise ValueError("base_seed must be a nonnegative integer")
        expected_numbers = {
            "candidate_budget_per_arm": LATENT_SEARCH_CANDIDATE_BUDGET,
            "generation_chunk_size": LATENT_SEARCH_CHUNK_SIZE,
            "kinematic_top_k": LATENT_SEARCH_TOP_K,
        }
        for name, expected in expected_numbers.items():
            actual = _positive_integer(getattr(self, name), name)
            if actual != expected:
                raise ValueError(f"{name} must remain frozen at {expected}")
        if self.candidate_budget_per_arm % self.generation_chunk_size:
            raise ValueError("candidate budget must be divisible by generation chunk size")
        if self.kinematic_top_k > self.candidate_budget_per_arm:
            raise ValueError("kinematic_top_k exceeds candidate budget")
        if self.use_bfloat16 is not False:
            raise ValueError("latent-search v1 must use float32 inference")
        if self.raw_policy != "top_k_only":
            raise ValueError("latent-search v1 saves only the kinematic Top-K raw")
        if self.latent_contract != "paired_standard_normal_epsilon_v1":
            raise ValueError("unexpected latent-search latent contract")
        if any(
            (
                self.external_smoothing,
                self.interpolation_repair,
                self.filter_threshold_overrides,
            )
        ):
            raise ValueError(
                "latent-search forbids smoothing, interpolation repair, and threshold overrides"
            )
        representatives = tuple(self.representatives)
        actual = tuple(
            (
                item.representative_id,
                item.skill_id,
                item.required_arms,
                item.evidence_contract,
            )
            for item in representatives
        )
        if actual != _EXPECTED_REPRESENTATIVES:
            raise ValueError("latent-search representative contracts differ from frozen v1")
        object.__setattr__(self, "representatives", representatives)


@dataclass(frozen=True)
class FrozenLatentSearchRepresentative:
    representative_id: str
    skill_id: str
    seed_record_id: str
    scenario_id: str
    target_track_id: str
    task_ids_by_arm: Mapping[PilotEvaluationArm, str]
    pilot_evidence_by_arm: Mapping[PilotEvaluationArm, Mapping[str, Any]]
    selection_basis: str

    def __post_init__(self) -> None:
        for name in (
            "representative_id",
            "skill_id",
            "seed_record_id",
            "scenario_id",
            "target_track_id",
            "selection_basis",
        ):
            _required_text(getattr(self, name), name)
        _sha256_text(self.seed_record_id, "seed_record_id")
        task_ids = dict(self.task_ids_by_arm)
        if not task_ids:
            raise ValueError("task_ids_by_arm must not be empty")
        for arm, task_id in task_ids.items():
            _required_text(arm, "task arm")
            _sha256_text(task_id, "task_id")
        evidence = {arm: dict(value) for arm, value in self.pilot_evidence_by_arm.items()}
        if set(evidence) != set(task_ids):
            raise ValueError("pilot evidence must exactly cover representative task arms")
        canonical_json_bytes(evidence)
        object.__setattr__(self, "task_ids_by_arm", MappingProxyType(task_ids))
        object.__setattr__(
            self,
            "pilot_evidence_by_arm",
            MappingProxyType(
                {arm: MappingProxyType(value) for arm, value in evidence.items()}
            ),
        )


@dataclass(frozen=True)
class LatentSearchManifest:
    version: int
    kind: str
    status: str
    pilot: Mapping[str, Any]
    representatives: tuple[FrozenLatentSearchRepresentative, ...]
    formal_train_only: bool
    validation_manifests_opened: bool
    final_validation_accessed: bool
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if self.version != LATENT_SEARCH_MANIFEST_VERSION:
            raise ValueError("unsupported latent-search manifest version")
        if self.kind != "latent_search_representative_manifest":
            raise ValueError("unexpected latent-search manifest kind")
        if self.status != "frozen":
            raise ValueError("latent-search representative manifest is not frozen")
        if self.formal_train_only is not True:
            raise ValueError("latent-search representatives must be Formal Train only")
        if self.validation_manifests_opened or self.final_validation_accessed:
            raise ValueError("latent-search manifest reports Validation access")
        _sha256_text(self.sha256, "manifest sha256")
        object.__setattr__(self, "pilot", MappingProxyType(dict(self.pilot)))
        object.__setattr__(self, "representatives", tuple(self.representatives))


@dataclass(frozen=True)
class LatentSearchTask:
    representative_id: str
    evaluation_arm: PilotEvaluationArm
    task: GenerationTask


@dataclass
class _PilotTaskStats:
    task: GenerationTask
    evaluation_arm: PilotEvaluationArm
    candidate_indices: set[int] = field(default_factory=set)
    accepted_count: int = 0
    first_failed_stages: Counter[str] = field(default_factory=Counter)
    stage_reached: Counter[str] = field(default_factory=Counter)
    stage_passed: Counter[str] = field(default_factory=Counter)
    normalized_kinematic_violation_scores: list[float] = field(default_factory=list)

    def add(self, row: Mapping[str, Any], *, accepted: bool) -> None:
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("Pilot filter row is missing metrics")
        candidate_index = metrics.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index < 0
            or candidate_index >= self.task.candidate_budget
        ):
            raise ValueError("Pilot filter row has an invalid candidate_index")
        if candidate_index in self.candidate_indices:
            raise ValueError("Pilot filter rows contain a duplicate task candidate")
        self.candidate_indices.add(candidate_index)
        if metrics.get("skill_id") != self.task.skill_id:
            raise ValueError("Pilot filter row skill differs from its task")
        if metrics.get("scenario_id") != self.task.scenario_id:
            raise ValueError("Pilot filter row scenario differs from its task")
        first_failed = metrics.get("first_failed_stage")
        if accepted:
            if first_failed is not None:
                raise ValueError("accepted Pilot row reports a failed stage")
            self.accepted_count += 1
        else:
            if first_failed not in _FILTER_STAGE_ORDER:
                raise ValueError("rejected Pilot row lacks a valid first_failed_stage")
            self.first_failed_stages[str(first_failed)] += 1
        stage_evidence = metrics.get("stage_evidence")
        if not isinstance(stage_evidence, list):
            raise ValueError("Pilot filter row lacks stage_evidence")
        for item in stage_evidence:
            if not isinstance(item, Mapping) or item.get("stage") not in _FILTER_STAGE_ORDER:
                raise ValueError("Pilot stage evidence is invalid")
            stage = str(item["stage"])
            self.stage_reached[stage] += 1
            if item.get("passed") is True:
                self.stage_passed[stage] += 1
            if stage == "kinematics":
                stage_metrics = item.get("metrics")
                score = (
                    stage_metrics.get("normalized_violation_score")
                    if isinstance(stage_metrics, Mapping)
                    else None
                )
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    self.normalized_kinematic_violation_scores.append(float(score))

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_indices)

    @property
    def deepest_reached_stage(self) -> str:
        reached = [stage for stage in _FILTER_STAGE_ORDER if self.stage_reached[stage]]
        if not reached:
            raise ValueError("Pilot task has no reached filter stage")
        return reached[-1]

    @property
    def deepest_reached_stage_index(self) -> int:
        return _FILTER_STAGE_ORDER.index(self.deepest_reached_stage)

    @property
    def minimum_normalized_kinematic_violation_score(self) -> float | None:
        if not self.normalized_kinematic_violation_scores:
            return None
        return min(self.normalized_kinematic_violation_scores)

    def evidence(self) -> dict[str, Any]:
        return {
            "pilot_task_id": self.task.task_id,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "deepest_reached_stage": self.deepest_reached_stage,
            "deepest_reached_count": self.stage_reached[
                self.deepest_reached_stage
            ],
            "kinematic_passed_count": self.stage_passed["kinematics"],
            "minimum_normalized_kinematic_violation_score": (
                self.minimum_normalized_kinematic_violation_score
            ),
            "skill_trigger_passed_count": self.stage_passed["skill_trigger"],
            "parameter_realization_passed_count": self.stage_passed[
                "parameter_realization"
            ],
            "first_failed_stages": dict(sorted(self.first_failed_stages.items())),
        }


def load_latent_search_config(path: str | Path) -> LatentSearchConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to load latent-search config: {config_path}") from error
    root = _strict_mapping(
        raw,
        "latent-search config",
        (
            "version",
            "contract_name",
            "base_seed",
            "candidate_budget_per_arm",
            "generation_chunk_size",
            "kinematic_top_k",
            "use_bfloat16",
            "raw_policy",
            "latent_contract",
            "postprocess",
            "representatives",
        ),
    )
    postprocess = _strict_mapping(
        root["postprocess"],
        "postprocess",
        ("external_smoothing", "interpolation_repair", "filter_threshold_overrides"),
    )
    raw_representatives = root["representatives"]
    if not isinstance(raw_representatives, list):
        raise ValueError("representatives must be a list")
    representatives = []
    for index, value in enumerate(raw_representatives):
        item = _strict_mapping(
            value,
            f"representatives[{index}]",
            ("representative_id", "skill_id", "required_arms", "evidence_contract"),
        )
        arms = item["required_arms"]
        if not isinstance(arms, list) or any(not isinstance(arm, str) for arm in arms):
            raise ValueError(f"representatives[{index}].required_arms must be strings")
        representatives.append(
            LatentSearchRepresentativeSpec(
                representative_id=item["representative_id"],
                skill_id=item["skill_id"],
                required_arms=tuple(arms),
                evidence_contract=item["evidence_contract"],
            )
        )
    return LatentSearchConfig(
        version=root["version"],
        contract_name=root["contract_name"],
        base_seed=root["base_seed"],
        candidate_budget_per_arm=root["candidate_budget_per_arm"],
        generation_chunk_size=root["generation_chunk_size"],
        kinematic_top_k=root["kinematic_top_k"],
        use_bfloat16=root["use_bfloat16"],
        raw_policy=root["raw_policy"],
        latent_contract=root["latent_contract"],
        external_smoothing=postprocess["external_smoothing"],
        interpolation_repair=postprocess["interpolation_repair"],
        filter_threshold_overrides=postprocess["filter_threshold_overrides"],
        representatives=tuple(representatives),
    )


def _task_from_row(value: Any) -> GenerationTask:
    fields = (
        "task_id",
        "task_index",
        "seed_record_id",
        "scenario_id",
        "skill_id",
        "target_track_id",
        "proposal_mode",
        "condition_skill_id",
        "candidate_budget",
        "checkpoint_sha256",
        "semantic_config_sha256",
    )
    row = _strict_mapping(value, "Pilot task row", fields)
    return GenerationTask(status="pending", **row)


def _read_task_plan(path: Path) -> tuple[GenerationTask, ...]:
    tasks: list[GenerationTask] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Pilot task plan contains a blank line: {line_number}")
            try:
                tasks.append(_task_from_row(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid Pilot task plan row {line_number}: {path}: {error}"
                ) from error
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Pilot task plan must contain unique tasks")
    return tuple(tasks)


def _load_pilot_task_stats(
    tasks: Sequence[GenerationTask],
    accepted_path: Path,
    rejected_path: Path,
    *,
    none_skill_id: str,
) -> dict[str, _PilotTaskStats]:
    stats = {
        task.task_id: _PilotTaskStats(
            task=task,
            evaluation_arm=pilot_evaluation_arm(task, none_skill_id=none_skill_id),
        )
        for task in tasks
    }
    for path, accepted in ((accepted_path, True), (rejected_path, False)):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Pilot filter index contains a blank line: {path}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid Pilot filter row {line_number}: {path}"
                    ) from error
                if not isinstance(row, Mapping):
                    raise ValueError("Pilot filter row must be a mapping")
                metrics = row.get("metrics")
                task_id = metrics.get("task_id") if isinstance(metrics, Mapping) else None
                task_stats = stats.get(task_id)
                if task_stats is None:
                    raise ValueError("Pilot filter row references an unknown task")
                task_stats.add(row, accepted=accepted)
    incomplete = {
        task_id: (item.candidate_count, item.task.candidate_budget)
        for task_id, item in stats.items()
        if item.candidate_count != item.task.candidate_budget
    }
    if incomplete:
        raise ValueError(f"Pilot filter indexes do not cover every candidate: {incomplete}")
    return stats


def _same_paired_seed(task_stats: Sequence[_PilotTaskStats]) -> bool:
    if len(task_stats) != 2:
        return False
    left, right = (item.task for item in task_stats)
    return (
        left.seed_record_id == right.seed_record_id
        and left.scenario_id == right.scenario_id
        and left.skill_id == right.skill_id
        and left.target_track_id == right.target_track_id
        and left.proposal_mode == right.proposal_mode
        and left.checkpoint_sha256 == right.checkpoint_sha256
        and left.semantic_config_sha256 == right.semantic_config_sha256
    )


def _representative_from_stats(
    spec: LatentSearchRepresentativeSpec,
    selected: Sequence[_PilotTaskStats],
    *,
    selection_basis: str,
) -> FrozenLatentSearchRepresentative:
    if not selected:
        raise ValueError(f"no Pilot task selected for {spec.representative_id}")
    first = selected[0].task
    if any(
        item.task.seed_record_id != first.seed_record_id
        or item.task.scenario_id != first.scenario_id
        or item.task.skill_id != first.skill_id
        or item.task.target_track_id != first.target_track_id
        for item in selected
    ):
        raise ValueError("representative arms do not share one Pilot seed/target")
    by_arm = {item.evaluation_arm: item for item in selected}
    if tuple(by_arm) != spec.required_arms:
        by_arm = {arm: by_arm[arm] for arm in spec.required_arms}
    return FrozenLatentSearchRepresentative(
        representative_id=spec.representative_id,
        skill_id=spec.skill_id,
        seed_record_id=first.seed_record_id,
        scenario_id=first.scenario_id,
        target_track_id=first.target_track_id,
        task_ids_by_arm={arm: item.task.task_id for arm, item in by_arm.items()},
        pilot_evidence_by_arm={arm: item.evidence() for arm, item in by_arm.items()},
        selection_basis=selection_basis,
    )


def _select_representatives(
    config: LatentSearchConfig,
    stats_by_task: Mapping[str, _PilotTaskStats],
) -> tuple[FrozenLatentSearchRepresentative, ...]:
    by_skill_arm: dict[tuple[str, PilotEvaluationArm], list[_PilotTaskStats]] = {}
    for item in stats_by_task.values():
        by_skill_arm.setdefault((item.task.skill_id, item.evaluation_arm), []).append(item)
    for values in by_skill_arm.values():
        values.sort(key=lambda item: item.task.task_id)

    result: list[FrozenLatentSearchRepresentative] = []
    for spec in config.representatives:
        if spec.evidence_contract == "deepest_formal_funnel":
            choices = by_skill_arm.get((spec.skill_id, "rule_guided_none"), [])
            if not choices:
                raise ValueError(
                    f"no rule-guided Pilot task exists for {spec.skill_id}"
                )
            selected = min(
                choices,
                key=lambda item: (
                    -item.deepest_reached_stage_index,
                    -item.stage_reached[item.deepest_reached_stage],
                    -item.stage_passed[item.deepest_reached_stage],
                    -item.stage_passed["kinematics"],
                    item.task.task_id,
                ),
            )
            result.append(
                _representative_from_stats(
                    spec,
                    (selected,),
                    selection_basis=(
                        "deepest reached filter stage, then number reaching that stage, "
                        "then number passing it, kinematic-pass count, and task_id"
                    ),
                )
            )
            continue

        pairs: list[tuple[_PilotTaskStats, _PilotTaskStats]] = []
        conditioned = by_skill_arm.get((spec.skill_id, "learned_conditioned"), [])
        controls = {
            item.task.seed_record_id: item
            for item in by_skill_arm.get((spec.skill_id, "learned_none_control"), [])
        }
        for item in conditioned:
            control = controls.get(item.task.seed_record_id)
            if control is not None and _same_paired_seed((item, control)):
                pairs.append((item, control))

        if spec.evidence_contract == "none_control_outperforms_conditioned":
            eligible = [
                pair
                for pair in pairs
                if pair[1].accepted_count > 0 and pair[0].accepted_count == 0
            ]
            if not eligible:
                raise ValueError(
                    "no jaywalking Pilot pair preserves the required reverse evidence"
                )
            selected_pair = min(
                eligible,
                key=lambda pair: (
                    -(pair[1].accepted_count - pair[0].accepted_count),
                    -(
                        pair[1].stage_passed["skill_trigger"]
                        - pair[0].stage_passed["skill_trigger"]
                    ),
                    pair[0].task.seed_record_id,
                ),
            )
            result.append(
                _representative_from_stats(
                    spec,
                    selected_pair,
                    selection_basis=(
                        "largest none-control accepted advantage, then trigger-pass "
                        "advantage, then seed_record_id"
                    ),
                )
            )
            continue

        if spec.evidence_contract == "both_arms_fail_kinematics":
            eligible = [
                pair
                for pair in pairs
                if all(
                    item.accepted_count == 0
                    and item.first_failed_stages["kinematics"]
                    == item.candidate_count
                    for item in pair
                )
            ]
            if not eligible:
                raise ValueError(
                    "no slow-lead Pilot pair failed entirely at kinematics in both arms"
                )
            scored_pairs = [
                pair
                for pair in eligible
                if all(
                    item.minimum_normalized_kinematic_violation_score is not None
                    for item in pair
                )
            ]
            if scored_pairs and len(scored_pairs) == len(eligible):
                selected_pair = min(
                    scored_pairs,
                    key=lambda pair: (
                        max(
                            float(item.minimum_normalized_kinematic_violation_score)
                            for item in pair
                        ),
                        sum(
                            float(item.minimum_normalized_kinematic_violation_score)
                            for item in pair
                        ),
                        tuple(sorted(item.task.task_id for item in pair)),
                    ),
                )
                selection_basis = (
                    "both arms fail every Pilot candidate at kinematics; minimize "
                    "the worst-arm then summed minimum normalized violation score, "
                    "then paired task IDs"
                )
            else:
                selected_pair = min(
                    eligible,
                    key=lambda pair: (
                        tuple(sorted(item.task.task_id for item in pair)),
                        pair[0].task.seed_record_id,
                    ),
                )
                selection_basis = (
                    "both arms fail every Pilot candidate at kinematics; Pilot rows "
                    "lack a complete normalized violation score, so use the stable "
                    "lexicographic paired task IDs, then seed_record_id"
                )
            result.append(
                _representative_from_stats(
                    spec,
                    selected_pair,
                    selection_basis=selection_basis,
                )
            )
            continue

        raise ValueError(f"unsupported evidence contract: {spec.evidence_contract}")
    return tuple(result)


def build_latent_search_manifest(
    *,
    pilot_summary_path: str | Path,
    output_path: str | Path,
    config: LatentSearchConfig,
    repository_root: str | Path,
    none_skill_id: str = "<none>",
) -> LatentSearchManifest:
    """Freeze exact representative Pilot tasks after the Pilot filter is trusted."""

    root = Path(repository_root).resolve()
    summary_path = Path(pilot_summary_path).resolve()
    summary = _read_json(summary_path, "Pilot summary")
    if summary.get("stage") != "pilot" or summary.get("status") != "completed":
        raise ValueError("representatives require one completed Pilot summary")
    if summary.get("validation_manifests_opened") is not False:
        raise ValueError("Pilot summary does not prove Validation remained unopened")
    if summary.get("final_validation_accessed") is not False:
        raise ValueError("Pilot summary reports Final Validation access")
    outputs = summary.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Pilot summary is missing outputs")
    task_plan_path = _resolve_repo_path(outputs.get("task_plan"), root, "task plan")
    accepted_path = _resolve_repo_path(outputs.get("accepted"), root, "accepted index")
    rejected_path = _resolve_repo_path(outputs.get("rejected"), root, "rejected index")
    task_plan_summary_path = task_plan_path.with_name("task_plan.summary.json")
    if not task_plan_summary_path.is_file():
        raise FileNotFoundError("Pilot task-plan summary is missing")

    task_plan_sha256 = _file_sha256(task_plan_path)
    task_plan_summary_sha256 = _file_sha256(task_plan_summary_path)
    if summary.get("task_plan_sha256") != task_plan_sha256:
        raise ValueError("Pilot task plan SHA-256 differs from its summary")
    if summary.get("task_plan_summary_sha256") != task_plan_summary_sha256:
        raise ValueError("Pilot task-plan summary SHA-256 differs")
    task_plan_summary = _read_json(task_plan_summary_path, "Pilot task-plan summary")
    if task_plan_summary.get("task_plan_id") != summary.get("task_plan_id"):
        raise ValueError("Pilot task-plan ID differs between artifacts")
    if task_plan_summary.get("base_seed") != config.base_seed:
        raise ValueError("latent-search base seed differs from the Pilot seed contract")

    tasks = _read_task_plan(task_plan_path)
    stats = _load_pilot_task_stats(
        tasks,
        accepted_path,
        rejected_path,
        none_skill_id=none_skill_id,
    )
    representatives = _select_representatives(config, stats)
    value = {
        "version": LATENT_SEARCH_MANIFEST_VERSION,
        "kind": "latent_search_representative_manifest",
        "status": "frozen",
        "pilot": {
            "pilot_run_id": _sha256_text(summary.get("pilot_run_id"), "pilot_run_id"),
            "summary_path": _repo_relative(summary_path, root),
            "summary_sha256": _file_sha256(summary_path),
            "task_plan_path": _repo_relative(task_plan_path, root),
            "task_plan_sha256": task_plan_sha256,
            "task_plan_summary_path": _repo_relative(task_plan_summary_path, root),
            "task_plan_summary_sha256": task_plan_summary_sha256,
            "task_plan_id": _sha256_text(summary.get("task_plan_id"), "task_plan_id"),
            "base_seed": config.base_seed,
            "checkpoint_sha256": _sha256_text(
                summary.get("checkpoint_sha256"), "checkpoint_sha256"
            ),
            "generation_semantic_sha256": _sha256_text(
                summary.get("generation_semantic_sha256"),
                "generation_semantic_sha256",
            ),
            "generation_execution_sha256": _sha256_text(
                summary.get("generation_execution_sha256"),
                "generation_execution_sha256",
            ),
            "filter_semantic_sha256": _sha256_text(
                summary.get("filter_semantic_sha256"), "filter_semantic_sha256"
            ),
            "accepted_index_path": _repo_relative(accepted_path, root),
            "accepted_index_sha256": _file_sha256(accepted_path),
            "rejected_index_path": _repo_relative(rejected_path, root),
            "rejected_index_sha256": _file_sha256(rejected_path),
        },
        "representatives": [
            {
                "representative_id": item.representative_id,
                "skill_id": item.skill_id,
                "seed_record_id": item.seed_record_id,
                "scenario_id": item.scenario_id,
                "target_track_id": item.target_track_id,
                "task_ids_by_arm": dict(item.task_ids_by_arm),
                "pilot_evidence_by_arm": {
                    arm: dict(evidence)
                    for arm, evidence in item.pilot_evidence_by_arm.items()
                },
                "selection_basis": item.selection_basis,
            }
            for item in representatives
        ],
        "formal_train_only": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    target = _atomic_write_json(Path(output_path), value)
    return load_latent_search_manifest(target, config=config, repository_root=root)


def load_latent_search_manifest(
    path: str | Path,
    *,
    config: LatentSearchConfig,
    repository_root: str | Path,
) -> LatentSearchManifest:
    manifest_path = Path(path)
    root = Path(repository_root).resolve()
    raw = _read_json(manifest_path, "latent-search representative manifest")
    value = _strict_mapping(
        raw,
        "latent-search representative manifest",
        (
            "version",
            "kind",
            "status",
            "pilot",
            "representatives",
            "formal_train_only",
            "validation_manifests_opened",
            "final_validation_accessed",
        ),
    )
    pilot = _strict_mapping(
        value["pilot"],
        "manifest.pilot",
        (
            "pilot_run_id",
            "summary_path",
            "summary_sha256",
            "task_plan_path",
            "task_plan_sha256",
            "task_plan_summary_path",
            "task_plan_summary_sha256",
            "task_plan_id",
            "base_seed",
            "checkpoint_sha256",
            "generation_semantic_sha256",
            "generation_execution_sha256",
            "filter_semantic_sha256",
            "accepted_index_path",
            "accepted_index_sha256",
            "rejected_index_path",
            "rejected_index_sha256",
        ),
    )
    for name in (
        "pilot_run_id",
        "summary_sha256",
        "task_plan_sha256",
        "task_plan_summary_sha256",
        "task_plan_id",
        "checkpoint_sha256",
        "generation_semantic_sha256",
        "generation_execution_sha256",
        "filter_semantic_sha256",
        "accepted_index_sha256",
        "rejected_index_sha256",
    ):
        _sha256_text(pilot[name], f"manifest.pilot.{name}")
    if pilot["base_seed"] != config.base_seed:
        raise ValueError("manifest Pilot base seed differs from latent-search config")
    for path_name, sha_name in (
        ("summary_path", "summary_sha256"),
        ("task_plan_path", "task_plan_sha256"),
        ("task_plan_summary_path", "task_plan_summary_sha256"),
        ("accepted_index_path", "accepted_index_sha256"),
        ("rejected_index_path", "rejected_index_sha256"),
    ):
        artifact = _resolve_repo_path(pilot[path_name], root, f"manifest.pilot.{path_name}")
        if _file_sha256(artifact) != pilot[sha_name]:
            raise ValueError(f"manifest Pilot artifact changed: {path_name}")

    raw_representatives = value["representatives"]
    if not isinstance(raw_representatives, list):
        raise ValueError("manifest representatives must be a list")
    representatives: list[FrozenLatentSearchRepresentative] = []
    for index, raw_item in enumerate(raw_representatives):
        item = _strict_mapping(
            raw_item,
            f"manifest.representatives[{index}]",
            (
                "representative_id",
                "skill_id",
                "seed_record_id",
                "scenario_id",
                "target_track_id",
                "task_ids_by_arm",
                "pilot_evidence_by_arm",
                "selection_basis",
            ),
        )
        if not isinstance(item["task_ids_by_arm"], Mapping):
            raise ValueError("task_ids_by_arm must be a mapping")
        if not isinstance(item["pilot_evidence_by_arm"], Mapping):
            raise ValueError("pilot_evidence_by_arm must be a mapping")
        representatives.append(
            FrozenLatentSearchRepresentative(
                representative_id=item["representative_id"],
                skill_id=item["skill_id"],
                seed_record_id=item["seed_record_id"],
                scenario_id=item["scenario_id"],
                target_track_id=item["target_track_id"],
                task_ids_by_arm=item["task_ids_by_arm"],
                pilot_evidence_by_arm=item["pilot_evidence_by_arm"],
                selection_basis=item["selection_basis"],
            )
        )
    expected_specs = {item.representative_id: item for item in config.representatives}
    if [item.representative_id for item in representatives] != [
        item.representative_id for item in config.representatives
    ]:
        raise ValueError("manifest representative order differs from frozen config")
    for item in representatives:
        spec = expected_specs[item.representative_id]
        if item.skill_id != spec.skill_id:
            raise ValueError("manifest representative skill differs from frozen config")
        if tuple(item.task_ids_by_arm) != spec.required_arms:
            raise ValueError("manifest representative arms differ from frozen config")

    tasks = {
        task.task_id: task
        for task in _read_task_plan(
            _resolve_repo_path(pilot["task_plan_path"], root, "manifest task plan")
        )
    }
    for representative in representatives:
        for arm, task_id in representative.task_ids_by_arm.items():
            task = tasks.get(task_id)
            if task is None:
                raise ValueError("manifest representative task is absent from Pilot plan")
            if pilot_evaluation_arm(task) != arm:
                raise ValueError("manifest representative task arm is invalid")
            expected_identity = (
                representative.seed_record_id,
                representative.scenario_id,
                representative.skill_id,
                representative.target_track_id,
            )
            actual_identity = (
                task.seed_record_id,
                task.scenario_id,
                task.skill_id,
                task.target_track_id,
            )
            if actual_identity != expected_identity:
                raise ValueError("manifest representative task identity changed")
    return LatentSearchManifest(
        version=value["version"],
        kind=value["kind"],
        status=value["status"],
        pilot=dict(pilot),
        representatives=tuple(representatives),
        formal_train_only=value["formal_train_only"],
        validation_manifests_opened=value["validation_manifests_opened"],
        final_validation_accessed=value["final_validation_accessed"],
        path=manifest_path,
        sha256=_file_sha256(manifest_path),
    )


def build_latent_search_tasks(
    manifest: LatentSearchManifest,
    *,
    config: LatentSearchConfig,
    generation_config: CounterfactualGenerationConfig,
    repository_root: str | Path,
) -> tuple[LatentSearchTask, ...]:
    semantic_sha256 = semantic_generation_config_sha256(generation_config)
    if semantic_sha256 != manifest.pilot["generation_semantic_sha256"]:
        raise ValueError("current generation semantics differ from the frozen Pilot")
    if generation_config.active_checkpoint.sha256 != manifest.pilot["checkpoint_sha256"]:
        raise ValueError("current checkpoint differs from the frozen Pilot")
    if generation_config.sampling.base_seed != config.base_seed:
        raise ValueError("generation and latent-search base seeds differ")
    task_plan_path = _resolve_repo_path(
        manifest.pilot["task_plan_path"],
        Path(repository_root).resolve(),
        "manifest task plan",
    )
    pilot_tasks = {task.task_id: task for task in _read_task_plan(task_plan_path)}
    result: list[LatentSearchTask] = []
    for representative in manifest.representatives:
        for arm, task_id in representative.task_ids_by_arm.items():
            pilot_task = pilot_tasks[task_id]
            expanded = replace(
                pilot_task,
                task_index=len(result),
                candidate_budget=config.candidate_budget_per_arm,
            )
            if expanded.task_id != pilot_task.task_id:
                raise RuntimeError("candidate-budget expansion changed the task identity")
            result.append(
                LatentSearchTask(
                    representative_id=representative.representative_id,
                    evaluation_arm=arm,
                    task=expanded,
                )
            )
    if len(result) != 6:
        raise ValueError("latent-search v1 must contain exactly six task arms")
    return tuple(result)


def latent_search_plan_payload(
    tasks: Sequence[LatentSearchTask],
    *,
    config: LatentSearchConfig,
    manifest: LatentSearchManifest,
) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "latent_search_task_plan",
        "latent_search_config": {
            "base_seed": config.base_seed,
            "candidate_budget_per_arm": config.candidate_budget_per_arm,
            "generation_chunk_size": config.generation_chunk_size,
            "kinematic_top_k": config.kinematic_top_k,
            "latent_contract": config.latent_contract,
            "raw_policy": config.raw_policy,
        },
        "representative_manifest_sha256": manifest.sha256,
        "tasks": [
            {
                "task_index": item.task.task_index,
                "representative_id": item.representative_id,
                "evaluation_arm": item.evaluation_arm,
                "task_id": item.task.task_id,
                "seed_record_id": item.task.seed_record_id,
                "scenario_id": item.task.scenario_id,
                "skill_id": item.task.skill_id,
                "target_track_id": item.task.target_track_id,
                "proposal_mode": item.task.proposal_mode,
                "condition_skill_id": item.task.condition_skill_id,
                "candidate_budget": item.task.candidate_budget,
            }
            for item in tasks
        ],
    }


def latent_search_plan_id(
    tasks: Sequence[LatentSearchTask],
    *,
    config: LatentSearchConfig,
    manifest: LatentSearchManifest,
) -> str:
    return canonical_sha256(
        latent_search_plan_payload(tasks, config=config, manifest=manifest)
    )


__all__ = [
    "LATENT_SEARCH_CANDIDATE_BUDGET",
    "LATENT_SEARCH_CHUNK_SIZE",
    "LATENT_SEARCH_TOP_K",
    "FrozenLatentSearchRepresentative",
    "LatentSearchConfig",
    "LatentSearchManifest",
    "LatentSearchRepresentativeSpec",
    "LatentSearchTask",
    "build_latent_search_manifest",
    "build_latent_search_tasks",
    "latent_search_plan_id",
    "latent_search_plan_payload",
    "load_latent_search_config",
    "load_latent_search_manifest",
]

"""Strict, deterministic quality filtering for generated raw candidates."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from skilldrive.filtering.collision import check_proxy_collisions
from skilldrive.filtering.common import (
    FutureKinematics,
    KinematicLimits,
    check_kinematics,
    check_schema_and_finite,
    derive_future_kinematics,
)
from skilldrive.filtering.context import (
    BoundRawCandidate,
    CandidateEvaluationContext,
    build_candidate_evaluation_context,
    validate_bound_candidate_contract,
)
from skilldrive.filtering.contracts import FilterCheck, FilterStage
from skilldrive.filtering.diversity import (
    DiversityCandidate,
    apply_diversity_filter,
)
from skilldrive.filtering.history import check_history_and_coordinates
from skilldrive.filtering.map import (
    check_map_compliance,
    check_map_compliance_batch,
)
from skilldrive.filtering.prepared_map import (
    PreparedMapGeometry,
    PreparedMapVerificationSession,
)
from skilldrive.filtering.novelty import check_observed_future_novelty
from skilldrive.filtering.parameters import check_parameter_realization
from skilldrive.filtering.risk import (
    RiskEvaluation,
    check_target_risk,
    evaluate_skill_risk,
)
from skilldrive.filtering.skill_validity import (
    prepare_risk_context,
    validate_skill_trigger,
)
from skilldrive.generation.config import CounterfactualFilterConfig, FILTER_STAGES
from skilldrive.generation.contracts import FilterDecision, canonical_json
from skilldrive.schemas import Scenario, SkillSpec
from skilldrive.skills.detection import DetectionConfig


FILTER_CONTRACT_VERSION = "filters_v1"
_STAGE_ORDER = tuple(FilterStage(value) for value in FILTER_STAGES)
_INDIVIDUAL_STAGE_ORDER = _STAGE_ORDER[:-1]
_DEFAULT_DIVERSITY_COHORT = "default"
MAP_BATCH_SIZES = frozenset({8, 16, 32})
DEFAULT_MAP_BATCH_SIZE = 16


def _freeze_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class CandidateFilterInput:
    """One fully bound raw candidate and the authoritative objects it references."""

    bound: BoundRawCandidate
    skill: SkillSpec
    source_scenario: Scenario
    primary_generated_role: str
    prepared_map: PreparedMapGeometry | None = None
    map_verification_session: PreparedMapVerificationSession | None = None

    def __post_init__(self) -> None:
        session = self.map_verification_session
        if session is not None and self.prepared_map is not session.prepared_map:
            raise ValueError(
                "map_verification_session requires its bound prepared_map"
            )


@dataclass(frozen=True)
class TimedFilterCheck:
    """A semantic check plus runtime evidence kept outside FilterDecision."""

    check: FilterCheck
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.check, FilterCheck):
            raise TypeError("check must be a FilterCheck")
        elapsed = float(self.elapsed_seconds)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        object.__setattr__(self, "elapsed_seconds", elapsed)

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "stage": self.check.stage.value,
            "passed": self.check.passed,
            "rejection_reasons": list(self.check.rejection_values),
            "metrics": dict(self.check.metrics),
        }


@dataclass(frozen=True)
class CandidateValidationResult:
    """All deterministic evidence produced before the batch diversity gate."""

    candidate: CandidateFilterInput
    context: CandidateEvaluationContext | None
    checks: tuple[TimedFilterCheck, ...]
    kinematics: FutureKinematics | None
    risk: RiskEvaluation | None
    quality_score: float | None

    @property
    def quality_passed(self) -> bool:
        return (
            tuple(item.check.stage for item in self.checks)
            == _INDIVIDUAL_STAGE_ORDER
            and all(item.check.passed for item in self.checks)
        )

    @property
    def first_failed(self) -> TimedFilterCheck | None:
        return next((item for item in self.checks if not item.check.passed), None)

    @property
    def stage_elapsed_seconds(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                item.check.stage.value: item.elapsed_seconds
                for item in self.checks
            }
        )

    def diversity_candidate(self) -> DiversityCandidate:
        if (
            not self.quality_passed
            or self.context is None
            or self.risk is None
            or self.risk.value is None
            or self.quality_score is None
        ):
            raise ValueError("only quality-passed candidates can enter diversity filtering")
        parameter_check = next(
            item.check
            for item in self.checks
            if item.check.stage is FilterStage.PARAMETER_REALIZATION
        )
        return DiversityCandidate(
            candidate_id=self.candidate.bound.raw.candidate_id,
            scenario_id=self.candidate.bound.raw.scenario_id,
            skill_id=self.candidate.bound.raw.skill_id,
            future_xy_local=self.context.future_xy_local,
            target_risk_value=self.risk.value,
            quality_score=self.quality_score,
            realized_parameter_bins=_realized_parameter_bins(parameter_check),
        )

    def compact(self, *, cohort: str) -> "CompactCandidateValidationResult":
        """Drop scenarios and evaluation context after individual gates finish."""

        raw = self.candidate.bound.raw
        task = self.candidate.bound.task
        compact_checks = tuple(
            TimedFilterCheck(
                check=FilterCheck(
                    stage=item.check.stage,
                    rejection_reasons=item.check.rejection_reasons,
                    metrics=_freeze_json_value(
                        json.loads(canonical_json(dict(item.check.metrics)))
                    ),
                ),
                elapsed_seconds=item.elapsed_seconds,
            )
            for item in self.checks
        )
        return CompactCandidateValidationResult(
            identity=CandidateFilterIdentity(
                candidate_id=raw.candidate_id,
                task_id=raw.task_id,
                candidate_index=raw.candidate_index,
                latent_seed=raw.latent_seed,
                scenario_id=raw.scenario_id,
                skill_id=raw.skill_id,
                target_track_id=raw.target_track_id,
                seed_record_id=task.seed_record_id,
                proposal_mode=raw.proposal_mode,
                checkpoint_sha256=raw.checkpoint_sha256,
                semantic_config_sha256=raw.semantic_config_sha256,
            ),
            cohort=cohort,
            checks=compact_checks,
            quality_score=self.quality_score,
            diversity_candidate=(
                self.diversity_candidate() if self.quality_passed else None
            ),
        )

    def to_filter_decision(
        self,
        *,
        filter_semantic_sha256: str,
        diversity_check: FilterCheck | None,
    ) -> FilterDecision:
        return self.compact(cohort=_DEFAULT_DIVERSITY_COHORT).to_filter_decision(
            filter_semantic_sha256=filter_semantic_sha256,
            diversity_check=diversity_check,
        )


@dataclass(frozen=True)
class CandidateFilterIdentity:
    """Small immutable identity needed to audit and index one decision."""

    candidate_id: str
    task_id: str
    candidate_index: int
    latent_seed: int
    scenario_id: str
    skill_id: str
    target_track_id: str
    seed_record_id: str
    proposal_mode: str
    checkpoint_sha256: str
    semantic_config_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "task_id",
            "scenario_id",
            "skill_id",
            "target_track_id",
            "seed_record_id",
            "proposal_mode",
            "checkpoint_sha256",
            "semantic_config_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("candidate_index", "latent_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class CompactCandidateValidationResult:
    """Scenario-free individual result retained until cohort finalization."""

    identity: CandidateFilterIdentity
    cohort: str
    checks: tuple[TimedFilterCheck, ...]
    quality_score: float | None
    diversity_candidate: DiversityCandidate | None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CandidateFilterIdentity):
            raise TypeError("identity must be a CandidateFilterIdentity")
        if (
            not isinstance(self.cohort, str)
            or not self.cohort
            or self.cohort.strip() != self.cohort
        ):
            raise ValueError("cohort must be a non-empty trimmed string")
        checks = tuple(self.checks)
        if not checks or any(not isinstance(item, TimedFilterCheck) for item in checks):
            raise ValueError("checks must contain at least one TimedFilterCheck")
        stages = tuple(item.check.stage for item in checks)
        expected = _INDIVIDUAL_STAGE_ORDER[: len(checks)]
        if stages != expected:
            raise ValueError(
                "checks must be a prefix of the frozen individual stage order"
            )
        failed = [index for index, item in enumerate(checks) if not item.check.passed]
        if failed and failed != [len(checks) - 1]:
            raise ValueError(
                "individual filtering must stop after the first failed stage"
            )
        if not failed and len(checks) != len(_INDIVIDUAL_STAGE_ORDER):
            raise ValueError("all-passed result must cover every individual stage")
        passed = len(checks) == len(_INDIVIDUAL_STAGE_ORDER) and not failed
        if passed:
            if self.quality_score is None or not math.isfinite(
                float(self.quality_score)
            ):
                raise ValueError(
                    "quality-passed result requires a finite quality_score"
                )
            if not isinstance(self.diversity_candidate, DiversityCandidate):
                raise ValueError(
                    "quality-passed result requires one DiversityCandidate"
                )
            diversity_identity = (
                self.diversity_candidate.candidate_id,
                self.diversity_candidate.scenario_id,
                self.diversity_candidate.skill_id,
            )
            expected_identity = (
                self.identity.candidate_id,
                self.identity.scenario_id,
                self.identity.skill_id,
            )
            if diversity_identity != expected_identity:
                raise ValueError(
                    "diversity candidate identity differs from result identity"
                )
            object.__setattr__(self, "quality_score", float(self.quality_score))
        elif self.quality_score is not None or self.diversity_candidate is not None:
            raise ValueError(
                "quality-rejected result cannot retain quality or diversity payload"
            )
        object.__setattr__(self, "checks", checks)

    @property
    def quality_passed(self) -> bool:
        return (
            len(self.checks) == len(_INDIVIDUAL_STAGE_ORDER)
            and all(item.check.passed for item in self.checks)
        )

    @property
    def first_failed(self) -> TimedFilterCheck | None:
        return next((item for item in self.checks if not item.check.passed), None)

    @property
    def stage_elapsed_seconds(self) -> Mapping[str, float]:
        return MappingProxyType(
            {
                item.check.stage.value: item.elapsed_seconds
                for item in self.checks
            }
        )

    @property
    def runtime_seconds(self) -> float:
        return sum(item.elapsed_seconds for item in self.checks)

    def to_filter_decision(
        self,
        *,
        filter_semantic_sha256: str,
        diversity_check: FilterCheck | None,
    ) -> FilterDecision:
        if self.quality_passed:
            if (
                diversity_check is None
                or diversity_check.stage is not FilterStage.DIVERSITY
            ):
                raise ValueError(
                    "quality-passed candidate requires one diversity decision"
                )
            final_checks = (*self.checks, TimedFilterCheck(diversity_check, 0.0))
        else:
            if diversity_check is not None:
                raise ValueError(
                    "quality-rejected candidate must not run diversity filtering"
                )
            final_checks = self.checks

        first_failed = next(
            (item for item in final_checks if not item.check.passed),
            None,
        )
        evaluated_stages = [item.check.stage.value for item in final_checks]
        skipped_stages = [
            stage.value for stage in _STAGE_ORDER if stage.value not in evaluated_stages
        ]
        identity = self.identity
        metrics = {
            "task_id": identity.task_id,
            "candidate_index": identity.candidate_index,
            "latent_seed": identity.latent_seed,
            "scenario_id": identity.scenario_id,
            "skill_id": identity.skill_id,
            "target_track_id": identity.target_track_id,
            "seed_record_id": identity.seed_record_id,
            "diversity_cohort": self.cohort,
            "first_failed_stage": (
                None if first_failed is None else first_failed.check.stage.value
            ),
            "primary_rejection_reason": (
                None if first_failed is None else first_failed.check.rejection_values[0]
            ),
            "evaluated_stages": evaluated_stages,
            "skipped_stages": skipped_stages,
            "stage_evidence": [item.evidence_dict() for item in final_checks],
            "quality_score": self.quality_score,
        }
        return FilterDecision.create(
            candidate_id=identity.candidate_id,
            filter_config_sha256=filter_semantic_sha256,
            filter_contract_version=FILTER_CONTRACT_VERSION,
            accepted=first_failed is None,
            rejection_reasons=(
                () if first_failed is None else first_failed.check.rejection_reasons
            ),
            metrics=metrics,
        )


@dataclass(frozen=True)
class BatchValidationResult:
    validations: tuple[CompactCandidateValidationResult, ...]
    decisions: tuple[FilterDecision, ...]
    stage_elapsed_seconds: Mapping[str, float]
    stage_execution_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stage_elapsed_seconds",
            MappingProxyType(dict(self.stage_elapsed_seconds)),
        )
        object.__setattr__(
            self,
            "stage_execution_counts",
            MappingProxyType(dict(self.stage_execution_counts)),
        )


def _timed(function, *args, **kwargs) -> TimedFilterCheck:
    started = time.perf_counter()
    check = function(*args, **kwargs)
    elapsed = time.perf_counter() - started
    if not isinstance(check, FilterCheck):
        raise TypeError("filter stage function must return FilterCheck")
    return TimedFilterCheck(check=check, elapsed_seconds=elapsed)


def _validate_stage_order(config: CounterfactualFilterConfig) -> None:
    if tuple(config.filter_stages) != tuple(stage.value for stage in _STAGE_ORDER):
        raise ValueError("filter config stages differ from the frozen pipeline order")


def _kinematic_limits(
    context: CandidateEvaluationContext,
    config: CounterfactualFilterConfig,
) -> KinematicLimits | None:
    target = next(
        agent
        for agent in context.source_scenario.agents
        if agent.track_id == context.task.target_track_id
    )
    policy = config.kinematics_by_type.get(target.object_type.lower())
    if policy is None:
        return None
    return KinematicLimits(
        maximum_seam_speed_mps=policy.maximum_seam_speed_mps,
        maximum_speed_mps=policy.maximum_speed_mps,
        maximum_acceleration_mps2=policy.maximum_acceleration_mps2,
        maximum_deceleration_mps2=policy.maximum_deceleration_mps2,
        maximum_jerk_mps3=policy.maximum_jerk_mps3,
        maximum_curvature_per_m=policy.maximum_curvature_per_m,
        maximum_heading_rate_rad_s=policy.maximum_heading_rate_rad_s,
        minimum_heading_speed_mps=policy.minimum_heading_speed_mps,
    )


def _unsupported_kinematic_class(context: CandidateEvaluationContext) -> FilterCheck:
    from skilldrive.filtering.contracts import FilterRejection

    target = next(
        agent
        for agent in context.source_scenario.agents
        if agent.track_id == context.task.target_track_id
    )
    return FilterCheck(
        stage=FilterStage.KINEMATICS,
        rejection_reasons=(FilterRejection.KINEMATIC_CLASS_UNSUPPORTED,),
        metrics={"target_object_type": target.object_type.lower()},
    )


def _skill_trigger_check(
    context: CandidateEvaluationContext,
    risk_scenario: Scenario,
    detection_config: DetectionConfig,
    filter_config: CounterfactualFilterConfig,
) -> FilterCheck:
    trigger = validate_skill_trigger(
        source_scenario=context.source_scenario,
        generated_scenario=risk_scenario,
        skill=context.skill,
        role_track_ids=context.seed_record.role_track_ids,
        seed_evidence=context.seed_record.evidence,
        detection_config=detection_config,
    )
    reasons = list(trigger.rejection_reasons)
    metrics: dict[str, Any] = {"trigger": dict(trigger.metrics)}
    if trigger.passed and context.skill.detection["mode"] == "observed_trigger":
        novelty = check_observed_future_novelty(
            context.source_scenario,
            context.task.target_track_id,
            context.raw.future_xy_global,
            filter_config.novelty_policy,
        )
        reasons.extend(novelty.rejection_reasons)
        metrics["novelty"] = dict(novelty.metrics)
    else:
        metrics["novelty"] = {
            "status": "not_applicable" if trigger.passed else "skipped_after_trigger_failure"
        }
    return FilterCheck(
        stage=FilterStage.SKILL_TRIGGER,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        metrics=metrics,
    )


def _realized_parameter_bins(check: FilterCheck) -> tuple[tuple[str, int], ...]:
    raw_parameters = check.metrics.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        return ()
    bins: list[tuple[str, int]] = []
    for name, raw_item in sorted(raw_parameters.items()):
        if not isinstance(name, str) or not isinstance(raw_item, Mapping):
            continue
        if raw_item.get("status") != "computed":
            continue
        realized = raw_item.get("realized")
        tolerance = raw_item.get("absolute_tolerance")
        if (
            isinstance(realized, bool)
            or not isinstance(realized, (int, float))
            or isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
        ):
            continue
        value = float(realized)
        width = float(tolerance)
        if math.isfinite(value) and math.isfinite(width) and width > 0.0:
            bins.append((name, int(round(value / width))))
    return tuple(bins)


def _finish(
    candidate: CandidateFilterInput,
    context: CandidateEvaluationContext | None,
    checks: list[TimedFilterCheck],
    kinematics: FutureKinematics | None,
    risk: RiskEvaluation | None,
    quality_score: float | None,
) -> CandidateValidationResult:
    return CandidateValidationResult(
        candidate=candidate,
        context=context,
        checks=tuple(checks),
        kinematics=kinematics,
        risk=risk,
        quality_score=quality_score,
    )


@dataclass(frozen=True)
class _MapPendingValidation:
    candidate: CandidateFilterInput
    context: CandidateEvaluationContext
    checks: tuple[TimedFilterCheck, ...]
    kinematics: FutureKinematics


def _append_check(checks: list[TimedFilterCheck], item: TimedFilterCheck) -> bool:
    expected = _INDIVIDUAL_STAGE_ORDER[len(checks)]
    if item.check.stage is not expected:
        raise ValueError(
            f"filter stage order violation: expected {expected.value}, "
            f"got {item.check.stage.value}"
        )
    checks.append(item)
    return item.check.passed


def _run_until_map(
    candidate: CandidateFilterInput,
    *,
    filter_config: CounterfactualFilterConfig,
) -> CandidateValidationResult | _MapPendingValidation:
    """Run the frozen individual stages before the map gate."""

    validate_bound_candidate_contract(
        candidate.bound,
        primary_generated_role=candidate.primary_generated_role,
    )
    if candidate.skill.skill_id != candidate.bound.task.skill_id:
        raise ValueError("candidate skill differs from the bound generation task")
    if candidate.source_scenario.scenario_id != candidate.bound.task.scenario_id:
        raise ValueError("source scenario differs from the bound generation task")

    checks: list[TimedFilterCheck] = []

    schema = _timed(
        check_schema_and_finite,
        candidate.source_scenario,
        candidate.bound.task.target_track_id,
        candidate.bound.raw.future_xy_global,
    )
    if not _append_check(checks, schema):
        return _finish(candidate, None, checks, None, None, None)

    context = build_candidate_evaluation_context(
        candidate.bound,
        skill=candidate.skill,
        source_scenario=candidate.source_scenario,
    )
    history = _timed(
        check_history_and_coordinates,
        context.source_scenario,
        context.generated_scenario,
        context.task.target_track_id,
        context.future_xy_local,
        context.raw.future_xy_global,
        context.anchor_origin_global,
        context.anchor_heading_global,
    )
    if not _append_check(checks, history):
        return _finish(candidate, context, checks, None, None, None)

    limits = _kinematic_limits(context, filter_config)
    kinematic_check = (
        _timed(_unsupported_kinematic_class, context)
        if limits is None
        else _timed(
            check_kinematics,
            context.source_scenario,
            context.task.target_track_id,
            context.raw.future_xy_global,
            limits,
        )
    )
    if not _append_check(checks, kinematic_check):
        return _finish(candidate, context, checks, None, None, None)
    assert limits is not None
    kinematics = derive_future_kinematics(
        context.source_scenario,
        context.task.target_track_id,
        context.raw.future_xy_global,
        minimum_heading_speed_mps=limits.minimum_heading_speed_mps,
    )
    return _MapPendingValidation(
        candidate=candidate,
        context=context,
        checks=tuple(checks),
        kinematics=kinematics,
    )


def _resume_after_map(
    pending: _MapPendingValidation,
    map_check: TimedFilterCheck,
    *,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
) -> CandidateValidationResult:
    """Resume one candidate after its map check, preserving frozen short-circuiting."""

    candidate = pending.candidate
    context = pending.context
    kinematics = pending.kinematics
    checks = list(pending.checks)
    if not _append_check(checks, map_check):
        return _finish(candidate, context, checks, kinematics, None, None)

    collision = _timed(
        check_proxy_collisions,
        context.generated_scenario,
        context.task.target_track_id,
        filter_config,
    )
    if not _append_check(checks, collision):
        return _finish(candidate, context, checks, kinematics, None, None)

    risk_started = time.perf_counter()
    risk_scenario = prepare_risk_context(
        source_scenario=context.source_scenario,
        generated_scenario=context.generated_scenario,
        skill=context.skill,
        role_track_ids=context.seed_record.role_track_ids,
        seed_evidence=context.seed_record.evidence,
    )
    risk = evaluate_skill_risk(
        risk_scenario,
        context.skill.skill_id,
        context.seed_record.role_track_ids,
    )
    risk_check = TimedFilterCheck(
        check_target_risk(context.skill, risk),
        time.perf_counter() - risk_started,
    )
    if not _append_check(checks, risk_check):
        return _finish(candidate, context, checks, kinematics, risk, None)

    trigger = _timed(
        _skill_trigger_check,
        context,
        risk_scenario,
        detection_config,
        filter_config,
    )
    if not _append_check(checks, trigger):
        return _finish(candidate, context, checks, kinematics, risk, None)

    parameter = _timed(
        check_parameter_realization,
        requested_parameters=context.seed_record.sampled_parameters,
        source_scenario=context.source_scenario,
        generated_scenario=risk_scenario,
        target_track_id=context.task.target_track_id,
        kinematics=kinematics,
        risk_metric=risk.metric,
        risk_value=risk.value,
        policy=filter_config.parameter_policy,
    )
    if not _append_check(checks, parameter):
        return _finish(candidate, context, checks, kinematics, risk, None)

    low, high = (float(value) for value in context.skill.risk_definition["target_range"])
    midpoint = (low + high) / 2.0
    width = max(high - low, 1e-12)
    quality_score = abs(float(risk.value) - midpoint) / width
    return _finish(candidate, context, checks, kinematics, risk, quality_score)


def validate_candidate(
    candidate: CandidateFilterInput,
    *,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
) -> CandidateValidationResult:
    """Run individual gates in frozen order and stop after the first failure."""

    _validate_stage_order(filter_config)
    pending = _run_until_map(candidate, filter_config=filter_config)
    if isinstance(pending, CandidateValidationResult):
        return pending
    map_check = _timed(
        check_map_compliance,
        pending.context.generated_scenario,
        pending.context.task.target_track_id,
        pending.context.skill.skill_id,
        filter_config.map_policy,
        prepared_map=candidate.prepared_map,
        verification_session=candidate.map_verification_session,
    )
    return _resume_after_map(
        pending,
        map_check,
        filter_config=filter_config,
        detection_config=detection_config,
    )


def _validate_map_batch_size(map_batch_size: int) -> None:
    if map_batch_size not in MAP_BATCH_SIZES:
        raise ValueError("map_batch_size must be one of 8, 16, or 32")


def _amortized_elapsed(total_seconds: float, count: int) -> tuple[float, ...]:
    if count <= 0:
        raise ValueError("elapsed time can only be distributed across positive count")
    total = float(total_seconds)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("total_seconds must be finite and nonnegative")
    common = total / count
    prefix = [common] * (count - 1)
    return (*prefix, total - sum(prefix))


def validate_candidate_individual_batch(
    candidates: Sequence[CandidateFilterInput],
    *,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
    map_batch_size: int = DEFAULT_MAP_BATCH_SIZE,
) -> tuple[CandidateValidationResult, ...]:
    """Batch only the map gate for one source scenario; do not run diversity."""

    _validate_stage_order(filter_config)
    _validate_map_batch_size(map_batch_size)
    values = tuple(candidates)
    if not values:
        return ()
    source = values[0].source_scenario
    prepared_map = values[0].prepared_map
    verification_session = values[0].map_verification_session
    if prepared_map is None:
        raise ValueError("individual map batching requires prepared_map")
    if any(
        item.source_scenario is not source
        or item.prepared_map is not prepared_map
        or item.map_verification_session is not verification_session
        for item in values
    ):
        raise ValueError(
            "individual map batching requires one source, prepared map, and session"
        )

    results: list[CandidateValidationResult | None] = [None] * len(values)
    pending_values: list[tuple[int, _MapPendingValidation]] = []

    def flush_pending() -> None:
        if not pending_values:
            return

        started = time.perf_counter()
        map_checks = check_map_compliance_batch(
            [item.context.generated_scenario for _, item in pending_values],
            [item.context.task.target_track_id for _, item in pending_values],
            [item.context.skill.skill_id for _, item in pending_values],
            filter_config.map_policy,
            prepared_map=prepared_map,
            verification_session=verification_session,
        )
        elapsed = time.perf_counter() - started
        if len(map_checks) != len(pending_values):
            raise ValueError("batch map checks did not cover every pre-map survivor")
        shares = _amortized_elapsed(elapsed, len(map_checks))
        for (index, pending), check, share in zip(
            pending_values,
            map_checks,
            shares,
            strict=True,
        ):
            results[index] = _resume_after_map(
                pending,
                TimedFilterCheck(check=check, elapsed_seconds=share),
                filter_config=filter_config,
                detection_config=detection_config,
            )
        pending_values.clear()

    for index, candidate in enumerate(values):
        pending = _run_until_map(candidate, filter_config=filter_config)
        if isinstance(pending, CandidateValidationResult):
            results[index] = pending
        else:
            pending_values.append((index, pending))
            if len(pending_values) == map_batch_size:
                flush_pending()
    flush_pending()

    if any(item is None for item in results):
        raise RuntimeError("individual map batching omitted a candidate result")
    return tuple(item for item in results if item is not None)


def validate_candidates(
    candidates: Sequence[CandidateFilterInput],
    *,
    filter_config: CounterfactualFilterConfig,
    detection_config: DetectionConfig,
    filter_semantic_sha256: str,
    cohort: str = _DEFAULT_DIVERSITY_COHORT,
) -> BatchValidationResult:
    """Validate a complete batch, then apply diversity once across all survivors."""

    _validate_stage_order(filter_config)
    candidate_ids = [item.bound.raw.candidate_id for item in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate batch contains duplicate candidate IDs")
    validations = tuple(
        validate_candidate(
            item,
            filter_config=filter_config,
            detection_config=detection_config,
        ).compact(cohort=cohort)
        for item in candidates
    )
    return finalize_candidate_validations(
        validations,
        filter_config=filter_config,
        filter_semantic_sha256=filter_semantic_sha256,
    )


def finalize_candidate_validations(
    validations: Sequence[CompactCandidateValidationResult],
    *,
    filter_config: CounterfactualFilterConfig,
    filter_semantic_sha256: str,
) -> BatchValidationResult:
    """Run diversity exactly once per cohort and emit one decision per raw ID."""

    _validate_stage_order(filter_config)
    compact = tuple(validations)
    if any(not isinstance(item, CompactCandidateValidationResult) for item in compact):
        raise TypeError(
            "validations must contain CompactCandidateValidationResult values"
        )
    candidate_ids = [item.identity.candidate_id for item in compact]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("compact validations contain duplicate candidate IDs")

    by_cohort: dict[str, list[CompactCandidateValidationResult]] = {}
    for item in compact:
        by_cohort.setdefault(item.cohort, []).append(item)

    diversity_checks: dict[str, FilterCheck] = {}
    diversity_elapsed = 0.0
    for cohort in sorted(by_cohort):
        survivors = [item for item in by_cohort[cohort] if item.quality_passed]
        diversity_candidates = [
            item.diversity_candidate
            for item in survivors
            if item.diversity_candidate is not None
        ]
        if len(diversity_candidates) != len(survivors):
            raise ValueError("quality-passed compact result lacks diversity payload")
        started = time.perf_counter()
        cohort_checks = apply_diversity_filter(
            diversity_candidates,
            filter_config.diversity_policy,
        )
        diversity_elapsed += time.perf_counter() - started
        expected_ids = {item.identity.candidate_id for item in survivors}
        if set(cohort_checks) != expected_ids:
            raise ValueError(
                f"diversity decisions do not exactly cover cohort {cohort!r}"
            )
        if any(
            not isinstance(check, FilterCheck)
            or check.stage is not FilterStage.DIVERSITY
            for check in cohort_checks.values()
        ):
            raise ValueError("diversity must return one diversity-stage FilterCheck")
        if set(diversity_checks).intersection(cohort_checks):
            raise ValueError("diversity returned a candidate ID in multiple cohorts")
        diversity_checks.update(cohort_checks)

    quality_ids = {
        item.identity.candidate_id for item in compact if item.quality_passed
    }
    if set(diversity_checks) != quality_ids:
        raise ValueError(
            "diversity decisions do not cover every quality-passed candidate"
        )

    decisions = tuple(
        item.to_filter_decision(
            filter_semantic_sha256=filter_semantic_sha256,
            diversity_check=(
                diversity_checks[item.identity.candidate_id]
                if item.quality_passed
                else None
            ),
        )
        for item in compact
    )
    decision_ids = [item.candidate_id for item in decisions]
    if len(decision_ids) != len(candidate_ids) or set(decision_ids) != set(
        candidate_ids
    ):
        raise ValueError("finalization must emit exactly one decision per candidate ID")

    elapsed_by_stage = {stage.value: 0.0 for stage in _STAGE_ORDER}
    counts_by_stage = {stage.value: 0 for stage in _STAGE_ORDER}
    for validation in compact:
        for item in validation.checks:
            elapsed_by_stage[item.check.stage.value] += item.elapsed_seconds
            counts_by_stage[item.check.stage.value] += 1
    elapsed_by_stage[FilterStage.DIVERSITY.value] = diversity_elapsed
    counts_by_stage[FilterStage.DIVERSITY.value] = len(quality_ids)
    return BatchValidationResult(
        validations=compact,
        decisions=decisions,
        stage_elapsed_seconds=elapsed_by_stage,
        stage_execution_counts=counts_by_stage,
    )


__all__ = [
    "DEFAULT_MAP_BATCH_SIZE",
    "FILTER_CONTRACT_VERSION",
    "MAP_BATCH_SIZES",
    "BatchValidationResult",
    "CandidateFilterInput",
    "CandidateFilterIdentity",
    "CandidateValidationResult",
    "CompactCandidateValidationResult",
    "TimedFilterCheck",
    "finalize_candidate_validations",
    "validate_candidate",
    "validate_candidate_individual_batch",
    "validate_candidates",
]

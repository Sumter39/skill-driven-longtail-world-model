"""Build a leakage-free Formal Train repair split from the v5 tensor cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from skilldrive.data.manifests import FIELDNAMES, ManifestRow, read_manifest
from skilldrive.generation.contracts import GenerationTask, canonical_json_bytes
from skilldrive.generation.scheduler import TaskPlan, load_task_plan, write_task_plan


REPAIR_SPLIT_SCHEMA_VERSION = 1
FORMAL_CACHE_VERSION = 5
DEFAULT_FORMAL_MANIFEST = Path("manifests/splits/formal_train.csv")
DEFAULT_CACHE_DIR = Path("data/processed/cvae_baseline/formal_train")
DEFAULT_PILOT_DIR = Path(
    "outputs/generation/counterfactual_v1/pilot/skill-pilot-v1/"
    "3a6d305509f75a1acf3029fc6efab208b632e4da4136471d4cbdc5afe227a315"
)
DEFAULT_OUTPUT_MANIFEST = Path("manifests/splits/formal_train_repair_v1.csv")
DEFAULT_OUTPUT_AUDIT = Path("manifests/splits/formal_train_repair_v1.audit.json")
DEFAULT_OUTPUT_INDEX_DIR = Path("data/processed/cvae_baseline/repair_v1")
DEFAULT_HELDOUT_TASK_DIR = Path(
    "manifests/generation/repair_v1/heldout_ability"
)

REPAIR_TRAIN = "repair_train"
REPAIR_DEV = "repair_dev"
NONE_SKILL_ID = "<none>"
LEARNED_MODE = "learned_conditioned_prior"
RULE_MODE = "rule_guided_prior_search"


@dataclass(frozen=True)
class CacheIndex:
    rows: tuple[dict[str, Any], ...]
    positive_counts: Counter[str]
    positive_by_scenario: Mapping[str, Counter[str]]
    scenarios: frozenset[str]
    offsets: frozenset[tuple[str, int]]
    shard_names: frozenset[str]
    manifest: Mapping[str, Any]
    manifest_path: Path
    index_path: Path


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(namespace: str, *parts: str) -> str:
    return hashlib.sha256(
        (namespace + "|".join(parts)).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
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


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, canonical_json_bytes(value, indent=2))


def _manifest_payload(rows: Iterable[ManifestRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "scenario_id": row.scenario_id,
                "split": row.split,
                "source_path": row.source_path,
                "city_name": row.city_name,
                "selected_reason": row.selected_reason,
            }
        )
    return stream.getvalue().encode("utf-8")


def _task_sort_key(task: GenerationTask) -> tuple[str, str, str, str, str]:
    return (
        task.scenario_id,
        task.skill_id,
        task.seed_record_id,
        task.condition_skill_id,
        task.task_id,
    )


def _load_trusted_pilot(
    pilot_dir: Path,
) -> tuple[TaskPlan, dict[str, Any], Path, Path]:
    summary_path = pilot_dir / "task_plan.summary.json"
    eligibility_path = pilot_dir / "eligibility_audit.json"
    summary = _read_json(summary_path, "Pilot task plan summary")
    semantic_sha256 = summary.get("semantic_config_sha256")
    execution_sha256 = summary.get("execution_config_sha256")
    if not isinstance(semantic_sha256, str) or not isinstance(execution_sha256, str):
        raise ValueError("Pilot summary is missing configuration fingerprints")
    loaded = load_task_plan(
        pilot_dir,
        expected_semantic_config_sha256=semantic_sha256,
        current_execution_config_sha256=execution_sha256,
    )
    eligibility = _read_json(eligibility_path, "Pilot eligibility audit")
    if eligibility.get("validation_manifests_opened") is not False:
        raise ValueError(
            "Pilot eligibility must prove validation_manifests_opened=false"
        )
    if eligibility.get("final_validation_accessed") not in (None, False):
        raise ValueError("Pilot eligibility reports Final Validation access")
    if eligibility.get("formal_train_only") not in (None, True):
        raise ValueError("Pilot eligibility is not Formal Train only")
    if eligibility.get("selection_stable") is not True:
        raise ValueError("Pilot eligibility selection is not deterministic")
    return loaded.plan, eligibility, summary_path, eligibility_path


def _validate_task_pairs(plan: TaskPlan) -> tuple[set[str], set[str]]:
    learned_skills = {
        task.skill_id for task in plan.tasks if task.proposal_mode == LEARNED_MODE
    }
    rule_skills = {
        task.skill_id for task in plan.tasks if task.proposal_mode == RULE_MODE
    }
    unknown_modes = {
        task.proposal_mode
        for task in plan.tasks
        if task.proposal_mode not in {LEARNED_MODE, RULE_MODE}
    }
    if unknown_modes:
        raise ValueError(f"Pilot contains unsupported proposal modes: {unknown_modes}")
    groups: dict[tuple[str, str, str, str], list[GenerationTask]] = defaultdict(list)
    for task in plan.tasks:
        groups[
            (
                task.scenario_id,
                task.seed_record_id,
                task.skill_id,
                task.target_track_id,
            )
        ].append(task)
    for key, tasks in groups.items():
        modes = {task.proposal_mode for task in tasks}
        if modes == {LEARNED_MODE}:
            if len(tasks) != 2 or {task.condition_skill_id for task in tasks} != {
                key[2],
                NONE_SKILL_ID,
            }:
                raise ValueError(
                    "learned Pilot task group is not a conditioned/control pair: "
                    f"{key}"
                )
        elif modes == {RULE_MODE}:
            if len(tasks) != 1 or tasks[0].condition_skill_id != NONE_SKILL_ID:
                raise ValueError(f"rule-guided Pilot task group is invalid: {key}")
        else:
            raise ValueError(f"Pilot task group mixes proposal modes: {key}")
    return learned_skills, rule_skills


def _load_cache_index(cache_dir: Path, formal_ids: set[str]) -> CacheIndex:
    manifest_path = cache_dir / "cache_manifest.json"
    manifest = _read_json(manifest_path, "Formal Train v5 cache manifest")
    if manifest.get("version") != FORMAL_CACHE_VERSION:
        raise ValueError("Formal Train cache is not the current v5 contract")
    if manifest.get("status") != "complete":
        raise ValueError("Formal Train cache is not complete")
    if manifest.get("partition") != "formal_train":
        raise ValueError("cache partition must be formal_train")
    descriptor = manifest.get("sample_index")
    if not isinstance(descriptor, dict):
        raise ValueError("cache sample_index descriptor is missing")
    relative_index = descriptor.get("path")
    if not isinstance(relative_index, str) or not relative_index:
        raise ValueError("cache sample_index path is invalid")
    index_path = cache_dir / relative_index
    if _sha256(index_path) != descriptor.get("sha256"):
        raise ValueError("Formal Train cache sample index SHA-256 differs")
    source_manifest_sha256 = manifest.get("inputs", {}).get("manifest_sha256")
    if not isinstance(source_manifest_sha256, str):
        raise ValueError("cache manifest does not bind its source manifest")

    shard_names = {
        value.get("path")
        for value in manifest.get("shards", [])
        if isinstance(value, dict) and isinstance(value.get("path"), str)
    }
    if not shard_names:
        raise ValueError("cache manifest has no shard descriptors")
    rows: list[dict[str, Any]] = []
    positive_counts: Counter[str] = Counter()
    positive_by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    scenarios: set[str] = set()
    offsets: set[tuple[str, int]] = set()
    sample_ids: set[str] = set()
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise ValueError(f"cache sample index has a blank line at {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid cache sample index line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"cache sample index line {line_number} is not an object")
        scenario_id = row.get("scenario_id")
        shard = row.get("shard")
        offset = row.get("offset")
        sample_id = row.get("sample_id")
        spec = row.get("spec")
        if scenario_id not in formal_ids:
            raise ValueError(
                f"cache sample index references a non-Formal scenario: {scenario_id}"
            )
        if shard not in shard_names:
            raise ValueError(f"cache sample index references an unknown shard: {shard}")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("cache sample index offset must be nonnegative")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("cache sample index sample_id is invalid")
        if not isinstance(spec, dict) or spec.get("scenario_id") != scenario_id:
            raise ValueError("cache sample spec scenario differs from its row")
        skill_id = spec.get("skill_id")
        supervised = spec.get("skill_supervision_mask")
        if not isinstance(skill_id, str) or not isinstance(supervised, bool):
            raise ValueError("cache sample skill contract is invalid")
        if supervised != (skill_id != NONE_SKILL_ID):
            raise ValueError("cache sample skill ID and supervision mask disagree")
        offset_key = (shard, offset)
        if offset_key in offsets:
            raise ValueError(f"duplicate cache shard offset: {offset_key}")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate cache sample_id: {sample_id}")
        offsets.add(offset_key)
        sample_ids.add(sample_id)
        scenarios.add(scenario_id)
        if supervised:
            positive_counts[skill_id] += 1
            positive_by_scenario[scenario_id][skill_id] += 1
        rows.append(row)
    if len(rows) != descriptor.get("records"):
        raise ValueError("cache sample index record count differs from cache_manifest")
    if scenarios != formal_ids:
        missing = sorted(formal_ids - scenarios)[:5]
        raise ValueError(f"Formal Train scenarios are missing from cache: {missing}")
    return CacheIndex(
        rows=tuple(rows),
        positive_counts=positive_counts,
        positive_by_scenario=positive_by_scenario,
        scenarios=frozenset(scenarios),
        offsets=frozenset(offsets),
        shard_names=frozenset(shard_names),
        manifest=manifest,
        manifest_path=manifest_path,
        index_path=index_path,
    )


def _positive_counts(
    scenario_ids: Iterable[str],
    positive_by_scenario: Mapping[str, Counter[str]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for scenario_id in scenario_ids:
        counts.update(positive_by_scenario.get(scenario_id, Counter()))
    return counts


def _select_pilot_dev_scenarios(
    *,
    plan: TaskPlan,
    cache: CacheIndex,
    learned_skills: set[str],
    pinned_task_ids: Sequence[str],
) -> tuple[set[str], set[str], dict[str, str]]:
    tasks_by_scenario: dict[str, list[GenerationTask]] = defaultdict(list)
    for task in plan.tasks:
        tasks_by_scenario[task.scenario_id].append(task)
    pilot_scenarios = set(tasks_by_scenario)
    pinned: set[str] = {
        task.scenario_id for task in plan.tasks if task.proposal_mode == RULE_MODE
    }
    tasks_by_id = {task.task_id: task for task in plan.tasks}
    unknown_pins = sorted(set(pinned_task_ids) - set(tasks_by_id))
    if unknown_pins:
        raise ValueError(f"pinned Pilot task IDs are unknown: {unknown_pins[:5]}")
    pinned.update(tasks_by_id[task_id].scenario_id for task_id in pinned_task_ids)

    representatives: dict[str, str] = {}
    for skill_id in sorted(learned_skills):
        candidates = [
            scenario_id
            for scenario_id in pilot_scenarios
            if cache.positive_by_scenario.get(scenario_id, Counter())[skill_id] > 0
            and any(
                task.skill_id == skill_id and task.proposal_mode == LEARNED_MODE
                for task in tasks_by_scenario[scenario_id]
            )
        ]
        if not candidates:
            raise ValueError(
                f"learned skill has no independent positive Pilot scenario: {skill_id}"
            )
        representative = min(
            candidates,
            key=lambda scenario_id: (
                _stable_sha256(
                    "repair-heldout-positive-v1|", skill_id, scenario_id
                ),
                scenario_id,
            ),
        )
        representatives[skill_id] = representative
        pinned.add(representative)

    totals = cache.positive_counts
    dev_maximum = {skill_id: totals[skill_id] // 2 for skill_id in learned_skills}
    for skill_id in learned_skills:
        if totals[skill_id] < 2:
            raise ValueError(
                f"learned skill cannot have independent train/dev positives: {skill_id}"
            )
    pinned_counts = _positive_counts(pinned, cache.positive_by_scenario)
    violations = {
        skill_id: pinned_counts[skill_id] - dev_maximum[skill_id]
        for skill_id in learned_skills
        if pinned_counts[skill_id] > dev_maximum[skill_id]
    }
    if violations:
        raise ValueError(
            "pinned heldout tasks leave fewer than half the positives for training: "
            f"{dict(sorted(violations.items()))}"
        )

    dev_scenarios = set(pilot_scenarios)
    dev_counts = _positive_counts(dev_scenarios, cache.positive_by_scenario)
    moved_to_train: set[str] = set()
    while any(
        dev_counts[skill_id] > dev_maximum[skill_id]
        for skill_id in learned_skills
    ):
        excess = {
            skill_id: max(0, dev_counts[skill_id] - dev_maximum[skill_id])
            for skill_id in learned_skills
        }
        candidates: list[
            tuple[Fraction, int, str, str]
        ] = []
        for scenario_id in dev_scenarios - pinned:
            scenario_positive = cache.positive_by_scenario.get(
                scenario_id, Counter()
            )
            relief = sum(
                min(count, excess.get(skill_id, 0))
                for skill_id, count in scenario_positive.items()
            )
            if relief <= 0:
                continue
            if any(
                dev_counts[skill_id] - count < 1
                for skill_id, count in scenario_positive.items()
                if skill_id in learned_skills
            ):
                continue
            task_loss = len(tasks_by_scenario[scenario_id])
            candidates.append(
                (
                    Fraction(task_loss, relief),
                    -relief,
                    _stable_sha256("repair-train-move-v1|", scenario_id),
                    scenario_id,
                )
            )
        if not candidates:
            remaining = {
                skill_id: dev_counts[skill_id] - dev_maximum[skill_id]
                for skill_id in learned_skills
                if dev_counts[skill_id] > dev_maximum[skill_id]
            }
            raise ValueError(
                "cannot satisfy learned train/dev positive constraints: "
                f"{dict(sorted(remaining.items()))}"
            )
        candidates.sort()
        scenario_id = candidates[0][-1]
        dev_scenarios.remove(scenario_id)
        moved_to_train.add(scenario_id)
        dev_counts.subtract(
            cache.positive_by_scenario.get(scenario_id, Counter())
        )
    for skill_id in learned_skills:
        train_count = totals[skill_id] - dev_counts[skill_id]
        if dev_counts[skill_id] < 1 or train_count < math.ceil(totals[skill_id] / 2):
            raise AssertionError(f"learned split invariant failed for {skill_id}")
    return dev_scenarios, moved_to_train, representatives


def _source_descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, root),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_repair_split(
    *,
    project_root: str | Path = ".",
    formal_manifest: str | Path = DEFAULT_FORMAL_MANIFEST,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    pilot_dir: str | Path = DEFAULT_PILOT_DIR,
    output_manifest: str | Path = DEFAULT_OUTPUT_MANIFEST,
    output_audit: str | Path = DEFAULT_OUTPUT_AUDIT,
    output_index_dir: str | Path = DEFAULT_OUTPUT_INDEX_DIR,
    heldout_task_dir: str | Path = DEFAULT_HELDOUT_TASK_DIR,
    dev_size: int = 2_000,
    expected_train_size: int = 18_000,
    expected_learned_skill_count: int = 13,
    expected_rule_task_count: int = 320,
    base_seed: int = 2026,
    pinned_task_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and publish the immutable repair split and zero-copy cache views."""

    root = Path(project_root)
    formal_path = _resolve(root, formal_manifest)
    cache_path = _resolve(root, cache_dir)
    pilot_path = _resolve(root, pilot_dir)
    output_manifest_path = _resolve(root, output_manifest)
    output_audit_path = _resolve(root, output_audit)
    output_index_path = _resolve(root, output_index_dir)
    heldout_path = _resolve(root, heldout_task_dir)

    formal_rows = read_manifest(formal_path)
    formal_by_id = {row.scenario_id: row for row in formal_rows}
    if len(formal_by_id) != len(formal_rows):
        raise ValueError("Formal Train manifest contains duplicate scenario IDs")
    if len(formal_rows) != dev_size + expected_train_size:
        raise ValueError(
            "Formal Train size differs from the frozen repair split contract"
        )
    formal_ids = set(formal_by_id)
    plan, eligibility, pilot_summary_path, eligibility_path = _load_trusted_pilot(
        pilot_path
    )
    learned_skills, rule_skills = _validate_task_pairs(plan)
    if len(learned_skills) != expected_learned_skill_count:
        raise ValueError(
            "Pilot learned skill count differs from the frozen repair contract"
        )
    rule_tasks = [task for task in plan.tasks if task.proposal_mode == RULE_MODE]
    if len(rule_tasks) != expected_rule_task_count:
        raise ValueError("Pilot rule task count differs from the frozen repair contract")
    pilot_scenarios = {task.scenario_id for task in plan.tasks}
    outside_formal = sorted(pilot_scenarios - formal_ids)
    if outside_formal:
        raise ValueError(f"Pilot scenarios are outside Formal Train: {outside_formal[:5]}")

    cache = _load_cache_index(cache_path, formal_ids)
    if cache.manifest.get("inputs", {}).get("manifest_sha256") != _sha256(
        formal_path
    ):
        raise ValueError("v5 cache was not built from the supplied Formal Train manifest")
    positive_skills = set(cache.positive_counts)
    if positive_skills != learned_skills:
        raise ValueError(
            "v5 supervised skill set differs from the learned Pilot skills: "
            f"cache_only={sorted(positive_skills - learned_skills)}, "
            f"pilot_only={sorted(learned_skills - positive_skills)}"
        )

    pilot_dev, pilot_train, representatives = _select_pilot_dev_scenarios(
        plan=plan,
        cache=cache,
        learned_skills=learned_skills,
        pinned_task_ids=pinned_task_ids,
    )
    if len(pilot_dev) > dev_size:
        raise ValueError("heldout Pilot scenarios exceed the repair dev size")
    positive_scenarios = set(cache.positive_by_scenario)
    background_candidates = formal_ids - pilot_scenarios - positive_scenarios
    background_needed = dev_size - len(pilot_dev)
    if len(background_candidates) < background_needed:
        raise ValueError("not enough no-positive Formal Train scenarios for dev top-up")
    ranked_background = sorted(
        background_candidates,
        key=lambda scenario_id: (
            _stable_sha256(
                f"repair-background-topup-v1|seed={base_seed}|", scenario_id
            ),
            scenario_id,
        ),
    )
    background_topup = set(ranked_background[:background_needed])
    dev_scenarios = pilot_dev | background_topup
    train_scenarios = formal_ids - dev_scenarios
    if len(dev_scenarios) != dev_size or len(train_scenarios) != expected_train_size:
        raise AssertionError("repair scenario counts differ from the frozen contract")
    if dev_scenarios & train_scenarios or dev_scenarios | train_scenarios != formal_ids:
        raise AssertionError("repair scenario split is not a disjoint Formal Train union")
    if any(cache.positive_by_scenario.get(value) for value in background_topup):
        raise AssertionError("background top-up contains a positive skill sample")

    output_rows: list[ManifestRow] = []
    for scenario_id in sorted(formal_ids):
        source = formal_by_id[scenario_id]
        if scenario_id in pilot_dev:
            split = REPAIR_DEV
            reason = "repair_v1:heldout_ability"
        elif scenario_id in background_topup:
            split = REPAIR_DEV
            reason = "repair_v1:background_topup_no_positive"
        elif scenario_id in pilot_train:
            split = REPAIR_TRAIN
            reason = "repair_v1:pilot_training_support"
        else:
            split = REPAIR_TRAIN
            reason = "repair_v1:training_remainder"
        output_rows.append(
            replace(source, split=split, selected_reason=reason)
        )
    _atomic_write(output_manifest_path, _manifest_payload(output_rows))

    train_index_rows = [
        row for row in cache.rows if row["scenario_id"] in train_scenarios
    ]
    dev_index_rows = [row for row in cache.rows if row["scenario_id"] in dev_scenarios]
    train_index_file = output_index_path / "train.sample_index.jsonl"
    dev_index_file = output_index_path / "dev.sample_index.jsonl"
    _atomic_write(
        train_index_file,
        b"".join(canonical_json_bytes(row) + b"\n" for row in train_index_rows),
    )
    _atomic_write(
        dev_index_file,
        b"".join(canonical_json_bytes(row) + b"\n" for row in dev_index_rows),
    )

    heldout_tasks = tuple(
        replace(task, task_index=index)
        for index, task in enumerate(
            task for task in plan.tasks if task.scenario_id in pilot_dev
        )
    )
    if tuple(sorted(heldout_tasks, key=_task_sort_key)) != heldout_tasks:
        raise AssertionError("heldout tasks lost deterministic scenario grouping")
    heldout_plan = TaskPlan(
        semantic_config_sha256=plan.semantic_config_sha256,
        execution_config_sha256=plan.execution_config_sha256,
        base_seed=plan.base_seed,
        per_skill=plan.per_skill,
        candidate_budget=plan.candidate_budget,
        tasks=heldout_tasks,
    )
    heldout_artifacts = write_task_plan(heldout_path, heldout_plan)
    _validate_task_pairs(heldout_plan)
    heldout_rule_tasks = [
        task for task in heldout_plan.tasks if task.proposal_mode == RULE_MODE
    ]
    if {task.task_id for task in heldout_rule_tasks} != {
        task.task_id for task in rule_tasks
    }:
        raise AssertionError("not every source rule-guided task remains held out")

    train_offsets = {
        (row["shard"], row["offset"]) for row in train_index_rows
    }
    dev_offsets = {(row["shard"], row["offset"]) for row in dev_index_rows}
    if train_offsets & dev_offsets or train_offsets | dev_offsets != cache.offsets:
        raise AssertionError("repair sample-index offsets are not a disjoint cache union")
    train_positive = _positive_counts(train_scenarios, cache.positive_by_scenario)
    dev_positive = _positive_counts(dev_scenarios, cache.positive_by_scenario)
    learned_distribution: dict[str, Any] = {}
    for skill_id in sorted(learned_skills):
        total = cache.positive_counts[skill_id]
        minimum_train = math.ceil(total / 2)
        if train_positive[skill_id] < minimum_train or dev_positive[skill_id] < 1:
            raise AssertionError(f"final learned distribution failed for {skill_id}")
        heldout_skill_tasks = [
            task
            for task in heldout_plan.tasks
            if task.skill_id == skill_id and task.proposal_mode == LEARNED_MODE
        ]
        learned_distribution[skill_id] = {
            "total_positive_samples": total,
            "repair_train_positive_samples": train_positive[skill_id],
            "repair_dev_positive_samples": dev_positive[skill_id],
            "minimum_train_positive_samples": minimum_train,
            "heldout_paired_scenarios": len(heldout_skill_tasks) // 2,
            "heldout_tasks": len(heldout_skill_tasks),
            "representative_scenario_id": representatives[skill_id],
        }

    train_shards = {row["shard"] for row in train_index_rows}
    dev_shards = {row["shard"] for row in dev_index_rows}
    source_task_plan_path = pilot_path / "task_plan.jsonl"
    audit = {
        "schema_version": REPAIR_SPLIT_SCHEMA_VERSION,
        "kind": "formal_train_repair_split_audit",
        "status": "complete",
        "validation_manifests_opened": False,
        "contract": {
            "split_unit": "scenario_id",
            "repair_dev_scenarios": dev_size,
            "repair_train_scenarios": expected_train_size,
            "learned_skill_count": expected_learned_skill_count,
            "expected_rule_tasks": expected_rule_task_count,
            "learned_train_minimum": "ceil(total_positive_samples / 2)",
            "learned_dev_minimum": 1,
            "background_topup_requires_no_positive_samples": True,
            "tensor_files_copied_or_resharded": False,
            "heldout_plan_requires_rebind_to_new_checkpoint": True,
        },
        "sources": {
            "formal_train_manifest": _source_descriptor(formal_path, root),
            "formal_train_v5_cache_manifest": _source_descriptor(
                cache.manifest_path, root
            ),
            "formal_train_v5_sample_index": _source_descriptor(
                cache.index_path, root
            ),
            "pilot_task_plan": {
                **_source_descriptor(source_task_plan_path, root),
                "task_plan_id": plan.task_plan_id,
            },
            "pilot_task_plan_summary": _source_descriptor(
                pilot_summary_path, root
            ),
            "pilot_eligibility_audit": _source_descriptor(
                eligibility_path, root
            ),
        },
        "selection": {
            "base_seed": base_seed,
            "pilot_scenarios": len(pilot_scenarios),
            "heldout_pilot_scenarios": len(pilot_dev),
            "pilot_scenarios_moved_to_train": len(pilot_train),
            "pilot_training_scenario_ids": sorted(pilot_train),
            "background_candidates": len(background_candidates),
            "background_topup_scenarios": len(background_topup),
            "explicit_pinned_task_ids": sorted(set(pinned_task_ids)),
            "learned_representative_scenarios": dict(sorted(representatives.items())),
        },
        "counts": {
            "formal_scenarios": len(formal_ids),
            "repair_train_scenarios": len(train_scenarios),
            "repair_dev_scenarios": len(dev_scenarios),
            "source_cache_samples": len(cache.rows),
            "repair_train_samples": len(train_index_rows),
            "repair_dev_samples": len(dev_index_rows),
            "source_positive_samples": sum(cache.positive_counts.values()),
            "repair_train_positive_samples": sum(train_positive.values()),
            "repair_dev_positive_samples": sum(dev_positive.values()),
            "source_pilot_tasks": len(plan.tasks),
            "heldout_ability_tasks": len(heldout_plan.tasks),
            "source_rule_tasks": len(rule_tasks),
            "heldout_rule_tasks": len(heldout_rule_tasks),
            "source_cache_shards": len(cache.shard_names),
            "repair_train_shards_touched": len(train_shards),
            "repair_dev_shards_touched": len(dev_shards),
            "shared_shards": len(train_shards & dev_shards),
        },
        "learned_skill_distribution": learned_distribution,
        "integrity": {
            "scenario_overlap": len(train_scenarios & dev_scenarios),
            "scenario_union_matches_formal_train": (
                train_scenarios | dev_scenarios == formal_ids
            ),
            "sample_offset_overlap": len(train_offsets & dev_offsets),
            "sample_offset_union_matches_v5_cache": (
                train_offsets | dev_offsets == cache.offsets
            ),
            "learned_pairs_complete": True,
            "all_rule_tasks_heldout": len(heldout_rule_tasks) == len(rule_tasks),
            "background_topup_positive_samples": sum(
                sum(cache.positive_by_scenario.get(value, Counter()).values())
                for value in background_topup
            ),
            "pilot_validation_manifests_opened": eligibility.get(
                "validation_manifests_opened"
            ),
        },
        "outputs": {
            "scenario_manifest": _source_descriptor(output_manifest_path, root),
            "repair_train_sample_index": _source_descriptor(
                train_index_file, root
            ),
            "repair_dev_sample_index": _source_descriptor(dev_index_file, root),
            "heldout_task_plan": {
                **_source_descriptor(heldout_artifacts.task_plan_path, root),
                "task_plan_id": heldout_plan.task_plan_id,
                "source_checkpoint_sha256": sorted(
                    {task.checkpoint_sha256 for task in heldout_plan.tasks}
                ),
                "requires_rebind_to_new_checkpoint": True,
            },
            "heldout_task_plan_summary": _source_descriptor(
                heldout_artifacts.summary_path, root
            ),
        },
    }
    _atomic_write_json(output_audit_path, audit)
    return audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen 18k/2k Formal Train repair split without opening "
            "Internal or Final Validation."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--pinned-task-id", action="append", default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    audit = build_repair_split(
        project_root=args.project_root,
        pilot_dir=args.pilot_dir,
        pinned_task_ids=tuple(args.pinned_task_id),
    )
    counts = audit["counts"]
    print(
        "Formal Train repair split complete: "
        f"train={counts['repair_train_scenarios']} scenarios / "
        f"{counts['repair_train_samples']} samples, "
        f"dev={counts['repair_dev_scenarios']} scenarios / "
        f"{counts['repair_dev_samples']} samples, "
        f"heldout_tasks={counts['heldout_ability_tasks']}"
    )


if __name__ == "__main__":
    main()

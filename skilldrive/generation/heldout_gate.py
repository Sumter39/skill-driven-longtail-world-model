"""Hash-bound repair-formal heldout rebind and promotion recommendation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from skilldrive.filtering.fingerprint import build_filter_semantic_fingerprint
from skilldrive.filtering.pipeline import FILTER_CONTRACT_VERSION
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.contracts import (
    GenerationTask,
    canonical_json_bytes,
    canonical_sha256,
    filter_evaluation_id,
)
from skilldrive.generation.planning import (
    latent_group_id,
    pilot_evaluation_arm,
    semantic_generation_config_sha256,
)
from skilldrive.generation.scheduler import (
    TaskPlan,
    load_task_plan,
    recover_paired_pilot_tasks,
    write_task_plan,
)
from skilldrive.generation.storage import load_raw_shard_candidates
from skilldrive.training.checkpoint import read_checkpoint_metadata


HELDOUT_GATE_VERSION = 1
HELDOUT_GATE_CONTRACT = "repair_heldout_gate_v1"
REPAIR_CONTRACT = "cvae_generation_repair_v1"
DEFAULT_REPAIR_AUDIT = Path("manifests/splits/formal_train_repair_v1.audit.json")
DEFAULT_HELDOUT_PLAN = Path("manifests/generation/repair_v1/heldout_ability")
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/generation/counterfactual_v1/pilot/repair-heldout-gate-v1"
)
HELDOUT_EXECUTION_ADAPTER = {
    "status": "available",
    "cli_stage": "repair-heldout-execute",
    "raw_directory": "raw",
    "filter_directory": "filter",
    "resume_contract": "recover_paired_pilot_tasks_source_seed_v1",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EPOCH_CHECKPOINT_PATTERN = re.compile(
    r"^epoch-(?P<epoch>\d{4})-step-(?P<step>\d{8})\.pt$"
)


@dataclass(frozen=True)
class RepairFormalCandidate:
    checkpoint_path: Path
    checkpoint_sha256: str
    run_manifest_path: Path
    run_manifest_sha256: str
    run_manifest_canonical_sha256: str
    schema_sha256: str
    candidate_epoch: int
    global_step: int
    run_manifest: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one mapping: {path}")
    return value


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"{name} contains a blank line at {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{name} contains invalid JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(f"{name} line {line_number} must be a mapping")
        rows.append(value)
    return rows


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


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical_json_bytes(value, indent=2)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable heldout artifact differs: {path}")
        return path
    _atomic_write(path, payload)
    return path


def _resolved(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _path_label(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repair_candidate_config(
    config,
    candidate: RepairFormalCandidate,
    repository_root: Path,
):
    return replace(
        config,
        active_checkpoint=replace(
            config.active_checkpoint,
            path=Path(_path_label(repository_root, candidate.checkpoint_path)),
            sha256=candidate.checkpoint_sha256,
            run_manifest=Path(
                _path_label(repository_root, candidate.run_manifest_path)
            ),
            run_manifest_sha256=candidate.run_manifest_sha256,
            schema_sha256=candidate.schema_sha256,
        ),
    )


def _heldout_filter_fingerprint(
    *,
    repository_root: Path,
    config_path: str | Path,
    filter_config_path: str | Path,
    detection_config_path: str | Path,
):
    return build_filter_semantic_fingerprint(
        repository_root=repository_root,
        generation_config_path=Path(config_path),
        filter_config_path=Path(filter_config_path),
        detection_config_path=Path(detection_config_path),
        additional_paths=(
            Path("skilldrive/generation/heldout_gate.py"),
            Path("skilldrive/training/checkpoint.py"),
        ),
    )


def _formal_train_source_path(
    repository_root: Path,
    data_root: Path,
    source_path: str,
) -> Path:
    pure = PurePosixPath(source_path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or not pure.parts
        or pure.parts[0] != "train"
    ):
        raise ValueError(
            f"heldout seed source_path must stay inside Formal Train: {source_path}"
        )
    path = _resolved(repository_root, data_root).joinpath(*pure.parts)
    if not path.is_file():
        raise FileNotFoundError(f"heldout seed scenario file is missing: {path}")
    return path


def validate_repair_formal_candidate(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    run_manifest_path: str | Path,
    run_manifest_sha256: str,
    expected_schema_sha256: str,
    repository_root: str | Path = ".",
) -> RepairFormalCandidate:
    """Accept only an unpromoted repair-formal epoch candidate."""

    root = Path(repository_root).resolve()
    checkpoint = _resolved(root, checkpoint_path)
    manifest_path = _resolved(root, run_manifest_path)
    expected_checkpoint = _sha256_text(checkpoint_sha256, "checkpoint_sha256")
    expected_manifest = _sha256_text(run_manifest_sha256, "run_manifest_sha256")
    if _file_sha256(checkpoint) != expected_checkpoint:
        raise ValueError("repair-formal checkpoint SHA-256 mismatch")
    if _file_sha256(manifest_path) != expected_manifest:
        raise ValueError("repair-formal run manifest SHA-256 mismatch")

    manifest = _read_json(manifest_path, "repair-formal run manifest")
    if manifest.get("stage") != "repair-formal":
        raise ValueError("heldout gate accepts only stage=repair-formal")
    if manifest.get("repair_contract") != REPAIR_CONTRACT:
        raise ValueError("repair-formal run manifest contract mismatch")
    schema_sha256 = _sha256_text(
        manifest.get("schema_sha256"),
        "run_manifest.schema_sha256",
    )
    if schema_sha256 != _sha256_text(
        expected_schema_sha256,
        "expected_schema_sha256",
    ):
        raise ValueError("repair-formal schema SHA-256 mismatch")
    fingerprints = manifest.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("repair-formal run manifest fingerprints are missing")
    selection = manifest.get("formal_selection")
    if not isinstance(selection, Mapping) or selection.get(
        "active_checkpoint_gate"
    ) != "heldout_generation_capability_gate_required":
        raise ValueError("repair-formal run manifest lacks the heldout gate contract")

    metadata = read_checkpoint_metadata(checkpoint)
    if dict(metadata.fingerprints) != fingerprints:
        raise ValueError("repair-formal checkpoint fingerprints differ from run manifest")
    canonical_manifest_sha256 = canonical_sha256(manifest)
    if metadata.extra.get("run_manifest_sha256") != canonical_manifest_sha256:
        raise ValueError("checkpoint immutable run manifest fingerprint mismatch")
    checkpoint_meta = metadata.extra.get("checkpoint")
    if not isinstance(checkpoint_meta, Mapping):
        raise ValueError("repair-formal checkpoint candidate metadata is missing")
    expected_metadata = {
        "role": "epoch_validation_candidate",
        "active_checkpoint": False,
        "selection_status": "unpromoted_epoch_candidate",
        "active_checkpoint_gate": "heldout_generation_capability_gate_required",
    }
    mismatches = {
        key: (checkpoint_meta.get(key), expected)
        for key, expected in expected_metadata.items()
        if checkpoint_meta.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"repair-formal epoch candidate metadata mismatch: {mismatches}")
    candidate_epoch = checkpoint_meta.get("candidate_epoch")
    if isinstance(candidate_epoch, bool) or not isinstance(candidate_epoch, int):
        raise ValueError("repair-formal candidate_epoch must be a positive integer")
    if candidate_epoch <= 0 or metadata.progress.epoch != candidate_epoch:
        raise ValueError("repair-formal candidate epoch differs from training progress")
    if metadata.progress.next_batch_index != 0:
        raise ValueError("repair-formal epoch candidate must end at an epoch boundary")
    match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ValueError("heldout gate requires a named repair-formal epoch checkpoint")
    if (
        int(match.group("epoch")) != candidate_epoch
        or int(match.group("step")) != metadata.progress.global_step
    ):
        raise ValueError("repair-formal epoch checkpoint filename differs from metadata")
    epoch_directory = selection.get("epoch_candidate_directory")
    if not isinstance(epoch_directory, str) or not epoch_directory:
        raise ValueError("repair-formal epoch candidate directory is missing")
    if checkpoint.parent != _resolved(root, epoch_directory):
        raise ValueError("repair-formal checkpoint is outside the frozen epoch directory")

    return RepairFormalCandidate(
        checkpoint_path=checkpoint,
        checkpoint_sha256=expected_checkpoint,
        run_manifest_path=manifest_path,
        run_manifest_sha256=expected_manifest,
        run_manifest_canonical_sha256=canonical_manifest_sha256,
        schema_sha256=schema_sha256,
        candidate_epoch=candidate_epoch,
        global_step=metadata.progress.global_step,
        run_manifest=manifest,
    )


def _load_frozen_source_plan(
    *,
    source_plan_dir: Path,
    repair_audit_path: Path,
    repository_root: Path,
) -> tuple[TaskPlan, dict[str, Any]]:
    audit = _read_json(repair_audit_path, "repair split audit")
    if audit.get("status") != "complete" or audit.get(
        "validation_manifests_opened"
    ) is not False:
        raise ValueError("repair split audit is incomplete or accessed validation")
    contract = audit.get("contract")
    integrity = audit.get("integrity")
    if not isinstance(contract, Mapping) or contract.get(
        "heldout_plan_requires_rebind_to_new_checkpoint"
    ) is not True:
        raise ValueError("repair split audit lacks the heldout rebind contract")
    if not isinstance(integrity, Mapping) or integrity.get(
        "pilot_validation_manifests_opened"
    ) is not False:
        raise ValueError("repair split audit heldout evidence is not validation-free")
    outputs = audit.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("repair split audit outputs are missing")
    descriptors = {
        "task_plan": outputs.get("heldout_task_plan"),
        "summary": outputs.get("heldout_task_plan_summary"),
    }
    paths = {
        "task_plan": source_plan_dir / "task_plan.jsonl",
        "summary": source_plan_dir / "task_plan.summary.json",
    }
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"repair split audit {name} descriptor is missing")
        path = paths[name]
        declared = _resolved(repository_root, descriptor.get("path", ""))
        if path.resolve() != declared:
            raise ValueError(f"heldout {name} path differs from repair split audit")
        if path.stat().st_size != descriptor.get("size_bytes"):
            raise ValueError(f"heldout {name} size differs from repair split audit")
        if _file_sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"heldout {name} SHA-256 differs from repair split audit")
    summary = _read_json(paths["summary"], "heldout task plan summary")
    loaded = load_task_plan(
        source_plan_dir,
        expected_semantic_config_sha256=summary.get("semantic_config_sha256"),
        current_execution_config_sha256=summary.get("execution_config_sha256"),
    )
    if loaded.execution_config_changed:
        raise ValueError("frozen heldout task plan execution identity changed")
    counts = audit.get("counts")
    if not isinstance(counts, Mapping) or counts.get(
        "heldout_ability_tasks"
    ) != len(loaded.plan.tasks):
        raise ValueError("heldout task count differs from repair split audit")
    return loaded.plan, audit


def _latent_seed_source_tasks(
    source_plan: TaskPlan,
    rebound_plan: TaskPlan,
) -> dict[str, GenerationTask]:
    """Bind rebound task identities to the frozen cross-checkpoint epsilon source."""

    if (
        source_plan.base_seed != rebound_plan.base_seed
        or source_plan.candidate_budget != rebound_plan.candidate_budget
        or len(source_plan.tasks) != len(rebound_plan.tasks)
    ):
        raise ValueError("rebound heldout plan changed the frozen latent budget")
    source_by_index = {task.task_index: task for task in source_plan.tasks}
    if len(source_by_index) != len(source_plan.tasks):
        raise ValueError("source heldout task indices are not unique")
    identity_fields = (
        "seed_record_id",
        "scenario_id",
        "skill_id",
        "target_track_id",
        "proposal_mode",
        "condition_skill_id",
        "candidate_budget",
    )
    mapping: dict[str, GenerationTask] = {}
    for rebound in rebound_plan.tasks:
        source = source_by_index.get(rebound.task_index)
        if source is None or any(
            getattr(source, field) != getattr(rebound, field)
            for field in identity_fields
        ):
            raise ValueError(
                "rebound heldout task differs from its frozen source task: "
                f"index={rebound.task_index}"
            )
        mapping[rebound.task_id] = source
    if len(mapping) != len(rebound_plan.tasks):
        raise ValueError("rebound heldout task IDs are not unique")
    return mapping


def _heldout_generation_source_sha256(repository_root: Path) -> dict[str, str]:
    paths = {
        *repository_root.joinpath("skilldrive/generation").glob("*.py"),
        repository_root / "skilldrive/data/av2_reader.py",
        repository_root / "skilldrive/data/coordinates.py",
        repository_root / "skilldrive/data/cvae_samples.py",
        repository_root / "skilldrive/models/conditional_cvae.py",
        repository_root / "skilldrive/training/checkpoint.py",
    }
    hashes: dict[str, str] = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(
                f"heldout generation dependency is missing: {path}"
            )
        hashes[path.relative_to(repository_root).as_posix()] = _file_sha256(path)
    return hashes


def _task_rebind_contract(
    source_plan: TaskPlan,
    rebound_plan: TaskPlan,
) -> dict[str, Any]:
    source_tasks = _latent_seed_source_tasks(source_plan, rebound_plan)
    return {
        "schema_version": 1,
        "kind": "repair_heldout_task_rebind",
        "source_task_plan_id": source_plan.task_plan_id,
        "rebound_task_plan_id": rebound_plan.task_plan_id,
        "base_seed": source_plan.base_seed,
        "latent_contract": (
            "frozen_source_task_paired_standard_normal_epsilon_v1"
        ),
        "tasks": [
            {
                "task_index": task.task_index,
                "source_task_id": source_tasks[task.task_id].task_id,
                "rebound_task_id": task.task_id,
                "heldout_latent_group_id": latent_group_id(
                    source_tasks[task.task_id]
                ),
            }
            for task in rebound_plan.tasks
        ],
    }


def _heldout_execution_config(
    *,
    repository_root: Path,
    candidate: RepairFormalCandidate,
    source_plan_path: Path,
    source_summary_path: Path,
    repair_audit_path: Path,
    filter_semantic_sha256: str,
    device: str,
    task_batch_size: int,
) -> dict[str, Any]:
    return {
        "version": HELDOUT_GATE_VERSION,
        "contract": "repair_heldout_execution_v1",
        "raw_shard_policy": "one_task_per_shard",
        "latent_contract": (
            "frozen_source_task_paired_standard_normal_epsilon_v1"
        ),
        "device": device,
        "task_batch_size": task_batch_size,
        "use_bfloat16": False,
        "generation_source_sha256": _heldout_generation_source_sha256(
            repository_root
        ),
        "source_task_plan_sha256": _file_sha256(source_plan_path),
        "source_task_plan_summary_sha256": _file_sha256(source_summary_path),
        "repair_split_audit_sha256": _file_sha256(repair_audit_path),
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "run_manifest_sha256": candidate.run_manifest_sha256,
        "candidate_epoch": candidate.candidate_epoch,
        "filter_semantic_sha256": filter_semantic_sha256,
    }


def rebind_repair_heldout_plan(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    run_manifest_path: str | Path,
    run_manifest_sha256: str,
    config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    filter_config_path: str | Path = "configs/generation/filters_v1.yaml",
    detection_config_path: str | Path = "configs/seed_detection.yaml",
    source_plan_dir: str | Path = DEFAULT_HELDOUT_PLAN,
    repair_audit_path: str | Path = DEFAULT_REPAIR_AUDIT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda",
    task_batch_size: int = 8,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Rebind the frozen heldout plan without marking the checkpoint active."""

    root = Path(repository_root).resolve()
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if (
        isinstance(task_batch_size, bool)
        or not isinstance(task_batch_size, int)
        or task_batch_size <= 0
    ):
        raise ValueError("task_batch_size must be a positive integer")
    config_source = _resolved(root, config_path)
    config = load_counterfactual_config(config_source, repository_root=root)
    candidate = validate_repair_formal_candidate(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        expected_schema_sha256=config.active_checkpoint.schema_sha256,
        repository_root=root,
    )
    source_dir = _resolved(root, source_plan_dir)
    audit_path = _resolved(root, repair_audit_path)
    source_plan, audit = _load_frozen_source_plan(
        source_plan_dir=source_dir,
        repair_audit_path=audit_path,
        repository_root=root,
    )
    rebound_config = _repair_candidate_config(config, candidate, root)
    semantic_sha256 = semantic_generation_config_sha256(rebound_config)
    rebound_tasks = tuple(
        GenerationTask.create(
            task_index=task.task_index,
            seed_record_id=task.seed_record_id,
            scenario_id=task.scenario_id,
            skill_id=task.skill_id,
            target_track_id=task.target_track_id,
            proposal_mode=task.proposal_mode,
            condition_skill_id=task.condition_skill_id,
            candidate_budget=task.candidate_budget,
            checkpoint_sha256=candidate.checkpoint_sha256,
            semantic_config_sha256=semantic_sha256,
        )
        for task in source_plan.tasks
    )
    filter_fingerprint = _heldout_filter_fingerprint(
        repository_root=root,
        config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    source_plan_path = source_dir / "task_plan.jsonl"
    source_summary_path = source_dir / "task_plan.summary.json"
    execution_config = _heldout_execution_config(
        repository_root=root,
        candidate=candidate,
        source_plan_path=source_plan_path,
        source_summary_path=source_summary_path,
        repair_audit_path=audit_path,
        filter_semantic_sha256=filter_fingerprint.semantic_sha256,
        device=device,
        task_batch_size=task_batch_size,
    )
    rebound_plan = TaskPlan(
        semantic_config_sha256=semantic_sha256,
        execution_config_sha256=canonical_sha256(execution_config),
        base_seed=source_plan.base_seed,
        per_skill=source_plan.per_skill,
        candidate_budget=source_plan.candidate_budget,
        tasks=rebound_tasks,
    )
    _latent_seed_source_tasks(source_plan, rebound_plan)
    gate_root = _resolved(root, output_root) / candidate.checkpoint_sha256
    plan_path = gate_root / "task_plan.jsonl"
    summary_path = gate_root / "task_plan.summary.json"
    if plan_path.exists() or summary_path.exists():
        if not plan_path.is_file() or not summary_path.is_file():
            raise ValueError("rebound heldout task plan is partially present")
        loaded = load_task_plan(
            gate_root,
            expected_semantic_config_sha256=rebound_plan.semantic_config_sha256,
            current_execution_config_sha256=rebound_plan.execution_config_sha256,
        )
        if loaded.execution_config_changed or loaded.plan != rebound_plan:
            raise ValueError("existing rebound heldout task plan differs")
        artifacts = None
    else:
        artifacts = write_task_plan(gate_root, rebound_plan)
    plan_sha256 = (
        _file_sha256(plan_path)
        if artifacts is None
        else artifacts.task_plan_sha256
    )
    summary_sha256 = (
        _file_sha256(summary_path)
        if artifacts is None
        else artifacts.summary_sha256
    )
    task_rebind_path = _write_immutable_json(
        gate_root / "task_rebind.json",
        _task_rebind_contract(source_plan, rebound_plan),
    )
    contract = {
        "schema_version": HELDOUT_GATE_VERSION,
        "kind": "repair_heldout_rebind_contract",
        "contract": HELDOUT_GATE_CONTRACT,
        "status": "rebound_pending_execution",
        "formal_active": False,
        "active_config_modified": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "checkpoint": {
            "path": _path_label(root, candidate.checkpoint_path),
            "sha256": candidate.checkpoint_sha256,
            "candidate_epoch": candidate.candidate_epoch,
            "global_step": candidate.global_step,
        },
        "run_manifest": {
            "path": _path_label(root, candidate.run_manifest_path),
            "sha256": candidate.run_manifest_sha256,
            "canonical_sha256": candidate.run_manifest_canonical_sha256,
            "stage": "repair-formal",
            "repair_contract": REPAIR_CONTRACT,
        },
        "source": {
            "repair_split_audit": {
                "path": _path_label(root, audit_path),
                "sha256": _file_sha256(audit_path),
            },
            "task_plan": {
                "path": _path_label(root, source_plan_path),
                "sha256": _file_sha256(source_plan_path),
            },
            "task_plan_summary": {
                "path": _path_label(root, source_summary_path),
                "sha256": _file_sha256(source_summary_path),
            },
            "source_task_plan_id": source_plan.task_plan_id,
        },
        "rebound": {
            "task_plan_id": rebound_plan.task_plan_id,
            "task_plan_sha256": plan_sha256,
            "task_plan_summary_sha256": summary_sha256,
            "task_rebind_path": _path_label(root, task_rebind_path),
            "task_rebind_sha256": _file_sha256(task_rebind_path),
            "semantic_config_sha256": rebound_plan.semantic_config_sha256,
            "execution_config_sha256": rebound_plan.execution_config_sha256,
            "task_count": len(rebound_plan.tasks),
            "candidate_count": rebound_plan.total_candidates,
            "filter_semantic_sha256": filter_fingerprint.semantic_sha256,
            "filter_dependency_sha256": dict(filter_fingerprint.file_sha256),
        },
        "execution_adapter": dict(HELDOUT_EXECUTION_ADAPTER),
        "audit_counts": {
            "heldout_tasks": audit["counts"]["heldout_ability_tasks"],
            "heldout_rule_tasks": audit["counts"]["heldout_rule_tasks"],
        },
    }
    contract_path = _write_immutable_json(
        gate_root / "rebind_contract.json",
        contract,
    )
    return {
        **contract,
        "outputs": {
            "root": _path_label(root, gate_root),
            "task_plan": _path_label(root, plan_path),
            "task_plan_summary": _path_label(root, summary_path),
            "task_rebind": _path_label(root, task_rebind_path),
            "rebind_contract": _path_label(root, contract_path),
        },
    }


def execute_repair_heldout_plan(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    run_manifest_path: str | Path,
    run_manifest_sha256: str,
    config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    filter_config_path: str | Path = "configs/generation/filters_v1.yaml",
    detection_config_path: str | Path = "configs/seed_detection.yaml",
    source_plan_dir: str | Path = DEFAULT_HELDOUT_PLAN,
    repair_audit_path: str | Path = DEFAULT_REPAIR_AUDIT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    device: str = "cuda",
    task_batch_size: int = 8,
    progress_interval_seconds: float = 10.0,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Execute one rebound repair-formal heldout plan with durable raw shards."""

    import numpy as np

    from skilldrive.data import build_cvae_schema, tensorize_prior_context
    from skilldrive.data.av2_reader import (
        load_av2_history_scenario,
        load_av2_scenario,
    )
    from skilldrive.filtering.context import bind_raw_candidates
    from skilldrive.filtering.pipeline import (
        CandidateFilterInput,
        finalize_candidate_validations,
        validate_candidate,
    )
    from skilldrive.generation.assembly import local_futures_to_global
    from skilldrive.generation.capability import (
        write_generation_capability_matrix,
    )
    from skilldrive.generation.config import load_filter_config
    from skilldrive.generation.contracts import GeneratedCandidate, GeneratedOverlay
    from skilldrive.generation.inference import generate_prior_batch, load_repair_cvae
    from skilldrive.generation.planning import (
        paired_latent_seeds_for_task,
        prior_context_fingerprint,
        prior_context_spec_for_task,
        seed_record_id,
    )
    from skilldrive.generation.storage import write_filter_indexes, write_raw_shard
    from skilldrive.seeds import read_seed_records
    from skilldrive.skills.detection import load_detection_config
    from skilldrive.skills.loader import load_skill

    started = time.perf_counter()
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a non-empty string")
    if (
        isinstance(task_batch_size, bool)
        or not isinstance(task_batch_size, int)
        or task_batch_size <= 0
    ):
        raise ValueError("task_batch_size must be a positive integer")
    if (
        isinstance(progress_interval_seconds, bool)
        or not isinstance(progress_interval_seconds, (int, float))
        or not math.isfinite(float(progress_interval_seconds))
        or float(progress_interval_seconds) <= 0.0
    ):
        raise ValueError("progress_interval_seconds must be a positive finite number")

    root = Path(repository_root).resolve()
    config_source = _resolved(root, config_path)
    filter_source = _resolved(root, filter_config_path)
    detection_source = _resolved(root, detection_config_path)
    config = load_counterfactual_config(config_source, repository_root=root)
    candidate = validate_repair_formal_candidate(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        expected_schema_sha256=config.active_checkpoint.schema_sha256,
        repository_root=root,
    )
    rebound_config = _repair_candidate_config(config, candidate, root)
    gate_root = _resolved(root, output_root) / candidate.checkpoint_sha256
    rebind_path = gate_root / "rebind_contract.json"
    rebind = _read_json(rebind_path, "repair heldout rebind contract")
    expected_rebind_fields = {
        "schema_version": HELDOUT_GATE_VERSION,
        "kind": "repair_heldout_rebind_contract",
        "contract": HELDOUT_GATE_CONTRACT,
        "status": "rebound_pending_execution",
        "formal_active": False,
        "active_config_modified": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    if (
        any(rebind.get(key) != value for key, value in expected_rebind_fields.items())
        or rebind.get("checkpoint", {}).get("sha256")
        != candidate.checkpoint_sha256
        or rebind.get("run_manifest", {}).get("sha256")
        != candidate.run_manifest_sha256
    ):
        raise ValueError("repair heldout rebind identity differs from candidate")
    adapter = rebind.get("execution_adapter")
    if not isinstance(adapter, Mapping) or dict(adapter) != HELDOUT_EXECUTION_ADAPTER:
        raise ValueError("repair heldout execution adapter is not available")
    rebound = rebind.get("rebound")
    if not isinstance(rebound, Mapping):
        raise ValueError("repair heldout rebound descriptor is missing")
    plan_loaded = load_task_plan(
        gate_root,
        expected_semantic_config_sha256=rebound.get("semantic_config_sha256"),
        current_execution_config_sha256=rebound.get("execution_config_sha256"),
    )
    if plan_loaded.execution_config_changed:
        raise ValueError("repair heldout execution configuration changed")
    plan = plan_loaded.plan
    if (
        _file_sha256(gate_root / "task_plan.jsonl")
        != rebound.get("task_plan_sha256")
        or _file_sha256(gate_root / "task_plan.summary.json")
        != rebound.get("task_plan_summary_sha256")
    ):
        raise ValueError("repair heldout rebound task plan hash changed")

    source_dir = _resolved(root, source_plan_dir)
    audit_path = _resolved(root, repair_audit_path)
    source_plan, _ = _load_frozen_source_plan(
        source_plan_dir=source_dir,
        repair_audit_path=audit_path,
        repository_root=root,
    )
    source_task_by_rebound_id = _latent_seed_source_tasks(source_plan, plan)
    task_rebind_path = _resolved(root, rebound.get("task_rebind_path", ""))
    if (
        _file_sha256(task_rebind_path) != rebound.get("task_rebind_sha256")
        or _read_json(task_rebind_path, "heldout task rebind")
        != _task_rebind_contract(source_plan, plan)
    ):
        raise ValueError("repair heldout task rebind mapping changed")

    filter_fingerprint = _heldout_filter_fingerprint(
        repository_root=root,
        config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    if (
        filter_fingerprint.semantic_sha256
        != rebound.get("filter_semantic_sha256")
        or dict(filter_fingerprint.file_sha256)
        != rebound.get("filter_dependency_sha256")
    ):
        raise ValueError("repair heldout filter semantics changed after rebind")
    execution_config = _heldout_execution_config(
        repository_root=root,
        candidate=candidate,
        source_plan_path=source_dir / "task_plan.jsonl",
        source_summary_path=source_dir / "task_plan.summary.json",
        repair_audit_path=audit_path,
        filter_semantic_sha256=filter_fingerprint.semantic_sha256,
        device=device,
        task_batch_size=task_batch_size,
    )
    if canonical_sha256(execution_config) != plan.execution_config_sha256:
        raise ValueError("repair heldout runtime differs from rebound execution contract")
    if semantic_generation_config_sha256(rebound_config) != plan.semantic_config_sha256:
        raise ValueError("repair heldout generation semantics differ from rebound plan")

    seed_manifest_path = _resolved(root, config.inputs.seed_manifest)
    if _file_sha256(seed_manifest_path) != config.inputs.seed_manifest_sha256:
        raise ValueError("formal seed manifest SHA-256 changed")
    records = read_seed_records(seed_manifest_path)
    records_by_id = {seed_record_id(record): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")
    source_paths: dict[str, Path] = {}
    arms_by_task: dict[str, str] = {}
    for task in plan.tasks:
        record = records_by_id.get(task.seed_record_id)
        if record is None:
            raise ValueError("heldout task references an unknown formal seed")
        skill_config = config.skills_by_id.get(task.skill_id)
        if (
            skill_config is None
            or record.scenario_id != task.scenario_id
            or record.skill_id != task.skill_id
            or record.role_track_ids.get(skill_config.primary_generated_role)
            != task.target_track_id
            or skill_config.proposal_mode != task.proposal_mode
            or task.checkpoint_sha256 != candidate.checkpoint_sha256
            or task.semantic_config_sha256 != plan.semantic_config_sha256
        ):
            raise ValueError("heldout task identity differs from its formal seed")
        arms_by_task[task.task_id] = pilot_evaluation_arm(
            task,
            none_skill_id=config.none_skill_id,
        )
        source_paths[task.task_id] = _formal_train_source_path(
            root,
            config.inputs.data_root,
            record.source_path,
        )

    schema = build_cvae_schema(_resolved(root, config.formal_catalog).parent)
    raw_root = gate_root / str(adapter.get("raw_directory", "raw"))
    filter_root = gate_root / str(adapter.get("filter_directory", "filter"))
    filter_commit_path = filter_root / "filter-index.commit.json"
    recovery = recover_paired_pilot_tasks(
        plan,
        raw_root,
        latent_seed_source_tasks=source_task_by_rebound_id,
    )
    tasks_to_generate = [
        task for task in plan.tasks if task.task_id in recovery.rebuild_task_ids
    ]
    if tasks_to_generate and filter_commit_path.is_file():
        filter_commit_path.unlink()
        print(
            "repair heldout resume: invalidated stale filter commit before raw rebuild",
            flush=True,
        )
    resumed_task_count = len(recovery.durable_task_ids)
    print(
        "repair heldout generation: "
        f"{resumed_task_count}/{len(plan.tasks)} task shards durable, "
        f"{len(tasks_to_generate)} to generate, 0/"
        f"{len(tasks_to_generate) * plan.candidate_budget} new candidates",
        flush=True,
    )

    generation_started = time.perf_counter()
    generated_count = 0
    if tasks_to_generate:
        runtime = load_repair_cvae(
            checkpoint_path=candidate.checkpoint_path,
            run_manifest_path=candidate.run_manifest_path,
            schema=schema,
            expected_checkpoint_sha256=candidate.checkpoint_sha256,
            expected_run_manifest_sha256=candidate.run_manifest_sha256,
            expected_schema_sha256=candidate.schema_sha256,
            device=device,
            checkpoint_mode="formal",
        )
        history_cache: dict[Path, Any] = {}
        context_cache: dict[str, Any] = {}
        last_progress = generation_started
        for offset in range(0, len(tasks_to_generate), task_batch_size):
            batch_tasks = tasks_to_generate[offset : offset + task_batch_size]
            contexts = []
            latent_rows = []
            batch_records = []
            for task in batch_tasks:
                record = records_by_id[task.seed_record_id]
                path = source_paths[task.task_id]
                history = history_cache.get(path)
                if history is None:
                    history = load_av2_history_scenario(path)
                    if (
                        len(history.timestamps) != 50
                        or history.metadata.get("temporal_scope") != "history_only"
                    ):
                        raise ValueError(
                            "heldout Prior generation must receive history-only scenes"
                        )
                    history_cache[path] = history
                fingerprint = prior_context_fingerprint(task, record)
                context = context_cache.get(fingerprint)
                if context is None:
                    context = tensorize_prior_context(
                        history,
                        prior_context_spec_for_task(task, record),
                        schema,
                    )
                    if context.target_track_id != task.target_track_id:
                        raise ValueError("heldout tensor target differs from task target")
                    if hasattr(context, "target_future") or hasattr(
                        context, "target_future_mask"
                    ):
                        raise ValueError("heldout Prior context exposed future tensors")
                    context_cache[fingerprint] = context
                contexts.append(context)
                source_task = source_task_by_rebound_id[task.task_id]
                latent_rows.append(
                    paired_latent_seeds_for_task(
                        source_task,
                        base_seed=source_plan.base_seed,
                    )
                )
                batch_records.append(record)

            latent_seed_matrix = np.stack(latent_rows)
            generated = generate_prior_batch(
                runtime,
                contexts,
                latent_seed_matrix,
                use_bfloat16=False,
            )
            for batch_index, (task, record, context) in enumerate(
                zip(batch_tasks, batch_records, contexts)
            ):
                futures = local_futures_to_global(
                    generated.future_position_local[batch_index],
                    context.anchor_origin_global,
                    float(context.anchor_heading_global),
                )
                skill_config = config.skills_by_id[task.skill_id]
                source_task = source_task_by_rebound_id[task.task_id]
                candidates = [
                    GeneratedCandidate(
                        task_id=task.task_id,
                        candidate_index=candidate_index,
                        latent_seed=int(
                            latent_seed_matrix[batch_index, candidate_index]
                        ),
                        scenario_id=task.scenario_id,
                        skill_id=task.skill_id,
                        proposal_mode=task.proposal_mode,
                        checkpoint_sha256=task.checkpoint_sha256,
                        semantic_config_sha256=task.semantic_config_sha256,
                        overlay=GeneratedOverlay(
                            target_track_id=task.target_track_id,
                            future_xy_global=futures[candidate_index],
                        ),
                        metadata={
                            "condition_skill_id": task.condition_skill_id,
                            "evaluation_arm": arms_by_task[task.task_id],
                            "latent_group_id": latent_group_id(source_task),
                            "source_task_id": source_task.task_id,
                            "primary_generated_role": (
                                skill_config.primary_generated_role
                            ),
                            "requested_parameters": record.sampled_parameters,
                            "detection_mode": record.evidence["detection_mode"],
                        },
                    )
                    for candidate_index in range(task.candidate_budget)
                ]
                write_raw_shard(
                    raw_root,
                    task.task_index,
                    candidates,
                    semantic_config_sha256=plan.semantic_config_sha256,
                    execution_config_sha256=plan.execution_config_sha256,
                )
                generated_count += len(candidates)

            now = time.perf_counter()
            completed = min(offset + len(batch_tasks), len(tasks_to_generate))
            if (
                now - last_progress >= float(progress_interval_seconds)
                or completed == len(tasks_to_generate)
            ):
                elapsed = max(now - generation_started, 1e-9)
                rate = generated_count / elapsed
                remaining = (
                    len(tasks_to_generate) * plan.candidate_budget - generated_count
                )
                eta = None if rate <= 0.0 else remaining / rate
                eta_label = (
                    "--:--"
                    if eta is None
                    else f"{int(eta // 60):02d}:{int(eta % 60):02d}"
                )
                print(
                    "repair heldout generation: "
                    f"{completed}/{len(tasks_to_generate)} new tasks, "
                    f"{generated_count}/"
                    f"{len(tasks_to_generate) * plan.candidate_budget} candidates, "
                    f"{rate:.1f} candidates/s, ETA {eta_label}",
                    flush=True,
                )
                last_progress = now
        history_cache.clear()
        context_cache.clear()
    generation_elapsed = time.perf_counter() - generation_started

    recovery = recover_paired_pilot_tasks(
        plan,
        raw_root,
        latent_seed_source_tasks=source_task_by_rebound_id,
    )
    if recovery.rebuild_task_ids or len(recovery.durable_task_ids) != len(plan.tasks):
        raise RuntimeError("repair heldout generation did not finish every task shard")
    raw_commits = tuple(
        sorted(recovery.raw_scan.valid_shards, key=lambda item: item.shard_index)
    )
    if (
        len(raw_commits) != len(plan.tasks)
        or sum(item.candidate_count for item in raw_commits) != plan.total_candidates
    ):
        raise RuntimeError("repair heldout raw totals differ from frozen plan")
    raw_snapshot = {
        path.resolve(): {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for shard in raw_commits
        for path in (shard.arrays_path, shard.metadata_path, shard.commit_path)
    }
    raw_commit_set_sha256 = canonical_sha256(
        [
            {
                "shard_index": shard.shard_index,
                "commit_sha256": _file_sha256(shard.commit_path),
            }
            for shard in raw_commits
        ]
    )
    summary_path = gate_root / "execution_summary.json"

    def finish(
        *,
        accepted_count: int,
        rejected_count: int,
        filter_commit_path: Path,
        filter_reused: bool,
        filtering_elapsed: float,
        stage_execution_counts: Mapping[str, Any],
        stage_elapsed_seconds: Mapping[str, Any],
    ) -> dict[str, Any]:
        executed_skills = sorted({task.skill_id for task in plan.tasks})
        missing_skills = sorted(set(config.formal_skill_ids) - set(executed_skills))
        value = {
            "schema_version": 1,
            "kind": "repair_heldout_execution_summary",
            "contract": "repair_heldout_execution_v1",
            "status": "completed",
            "formal_active": False,
            "active_config_modified": False,
            "checkpoint_sha256": candidate.checkpoint_sha256,
            "run_manifest_sha256": candidate.run_manifest_sha256,
            "candidate_epoch": candidate.candidate_epoch,
            "source_task_plan_id": source_plan.task_plan_id,
            "task_plan_id": plan.task_plan_id,
            "task_count": len(plan.tasks),
            "candidate_count": plan.total_candidates,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "formal_skill_count": len(config.formal_skill_ids),
            "executed_skill_count": len(executed_skills),
            "executed_skills": executed_skills,
            "frozen_skills_without_heldout_tasks": missing_skills,
            "cross_checkpoint_latent_source_frozen": True,
            "raw_commit_set_sha256": raw_commit_set_sha256,
            "raw_snapshot_sha256": canonical_sha256(
                {
                    _path_label(root, path): identity
                    for path, identity in sorted(
                        raw_snapshot.items(), key=lambda item: item[0].as_posix()
                    )
                }
            ),
            "raw_immutable_verified": True,
            "filter_semantic_sha256": filter_fingerprint.semantic_sha256,
            "filter_dependency_sha256": dict(filter_fingerprint.file_sha256),
            "filter_contract_version": FILTER_CONTRACT_VERSION,
            "filter_commit_sha256": _file_sha256(filter_commit_path),
            "execution_config": execution_config,
            "resume": {
                "durable_task_count_before_run": resumed_task_count,
                "newly_generated_task_count": len(tasks_to_generate),
                "newly_generated_candidate_count": generated_count,
                "filter_reused": filter_reused,
            },
            "stage_execution_counts": dict(stage_execution_counts),
            "stage_elapsed_seconds": dict(stage_elapsed_seconds),
            "timing_seconds": {
                "generation": generation_elapsed,
                "filtering_and_diversity": filtering_elapsed,
                "end_to_end": time.perf_counter() - started,
            },
            "validation_manifests_opened": False,
            "final_validation_accessed": False,
            "outputs": {
                "rebind_contract": _path_label(root, rebind_path),
                "task_rebind": _path_label(root, task_rebind_path),
                "task_plan": _path_label(root, gate_root / "task_plan.jsonl"),
                "raw": _path_label(root, raw_root),
                "accepted": _path_label(root, filter_root / "accepted.jsonl"),
                "rejected": _path_label(root, filter_root / "rejected.jsonl"),
                "filter_commit": _path_label(root, filter_commit_path),
                "execution_summary": _path_label(root, summary_path),
            },
        }
        write_generation_capability_matrix(summary_path, value)
        return value

    if filter_commit_path.is_file() and not tasks_to_generate:
        try:
            accepted, rejected, _, verified_raw_sha = _verify_filter_result(
                gate_root=gate_root,
                plan=plan,
                expected_filter_sha256=filter_fingerprint.semantic_sha256,
                latent_seed_source_tasks=source_task_by_rebound_id,
            )
        except ValueError as error:
            filter_commit_path.unlink(missing_ok=True)
            print(
                "repair heldout resume: invalid filter commit; rebuilding from "
                f"durable raw ({error})",
                flush=True,
            )
        else:
            if verified_raw_sha != raw_commit_set_sha256:
                raise ValueError("repair heldout raw commit identity changed")
            if summary_path.is_file():
                existing = _read_json(summary_path, "repair heldout execution summary")
                expected = {
                    "status": "completed",
                    "checkpoint_sha256": candidate.checkpoint_sha256,
                    "task_plan_id": plan.task_plan_id,
                    "raw_commit_set_sha256": raw_commit_set_sha256,
                    "filter_commit_sha256": _file_sha256(filter_commit_path),
                }
                if any(existing.get(key) != value for key, value in expected.items()):
                    raise ValueError("repair heldout execution summary identity changed")
                print(
                    "repair heldout execute resume: complete filter commit verified; "
                    "no model or filter work required",
                    flush=True,
                )
                return existing
            return finish(
                accepted_count=len(accepted),
                rejected_count=len(rejected),
                filter_commit_path=filter_commit_path,
                filter_reused=True,
                filtering_elapsed=0.0,
                stage_execution_counts={},
                stage_elapsed_seconds={},
            )

    filter_config = load_filter_config(filter_source)
    detection_config = load_detection_config(detection_source)
    skills = {
        skill_id: load_skill(
            _resolved(root, config.formal_catalog).parent / f"{skill_id}.yaml"
        )
        for skill_id in {task.skill_id for task in plan.tasks}
    }
    tasks_by_index = {task.task_index: task for task in plan.tasks}
    validations = []
    source_cache: dict[Path, Any] = {}
    filtering_started = time.perf_counter()
    last_progress = filtering_started
    filtered_count = 0
    print(
        "repair heldout filtering: "
        f"0/{plan.total_candidates} candidates, 0 quality survivors",
        flush=True,
    )
    for shard in raw_commits:
        task = tasks_by_index[shard.shard_index]
        record = records_by_id[task.seed_record_id]
        path = source_paths[task.task_id]
        source = source_cache.get(path)
        if source is None:
            source = load_av2_scenario(path)
            source_cache[path] = source
        raw_candidates = load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
        )
        bound = bind_raw_candidates(raw_candidates, [task], [record])
        arm = arms_by_task[task.task_id]
        cohort = "learned_none_control" if arm == "learned_none_control" else "formal"
        primary_role = config.skills_by_id[task.skill_id].primary_generated_role
        for item in bound:
            validation = validate_candidate(
                CandidateFilterInput(
                    bound=item,
                    skill=skills[task.skill_id],
                    source_scenario=source,
                    primary_generated_role=primary_role,
                ),
                filter_config=filter_config,
                detection_config=detection_config,
            )
            validations.append(validation.compact(cohort=cohort))
            filtered_count += 1
        now = time.perf_counter()
        if (
            now - last_progress >= float(progress_interval_seconds)
            or filtered_count == plan.total_candidates
        ):
            elapsed = max(now - filtering_started, 1e-9)
            rate = filtered_count / elapsed
            remaining = plan.total_candidates - filtered_count
            eta = None if rate <= 0.0 else remaining / rate
            eta_label = (
                "--:--"
                if eta is None
                else f"{int(eta // 60):02d}:{int(eta % 60):02d}"
            )
            quality_count = sum(item.quality_passed for item in validations)
            print(
                "repair heldout filtering: "
                f"{filtered_count}/{plan.total_candidates} candidates, "
                f"{quality_count} quality survivors, {rate:.1f} candidates/s, "
                f"ETA {eta_label}",
                flush=True,
            )
            last_progress = now
    batch = finalize_candidate_validations(
        validations,
        filter_config=filter_config,
        filter_semantic_sha256=filter_fingerprint.semantic_sha256,
    )
    index = write_filter_indexes(
        filter_root,
        raw_commits,
        batch.decisions,
        filter_config_sha256=filter_fingerprint.semantic_sha256,
        filter_contract_version=FILTER_CONTRACT_VERSION,
    )
    filtering_elapsed = time.perf_counter() - filtering_started
    raw_snapshot_after = {
        path: {
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in raw_snapshot
        if path.is_file()
    }
    if raw_snapshot_after != raw_snapshot:
        raise RuntimeError("repair heldout filtering modified committed raw files")
    result = finish(
        accepted_count=index.accepted_count,
        rejected_count=index.rejected_count,
        filter_commit_path=index.commit_path,
        filter_reused=False,
        filtering_elapsed=filtering_elapsed,
        stage_execution_counts=batch.stage_execution_counts,
        stage_elapsed_seconds=batch.stage_elapsed_seconds,
    )
    print(
        "repair heldout execute complete: "
        f"{len(plan.tasks)} tasks, {plan.total_candidates} candidates, "
        f"{index.accepted_count} accepted",
        flush=True,
    )
    print(f"repair heldout execution summary: {summary_path}", flush=True)
    return result


def _verify_file_descriptor(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError(f"{name} file descriptor is invalid")
    path = root / str(value["path"])
    if not path.is_file() or path.stat().st_size != value["size_bytes"]:
        raise ValueError(f"{name} file size differs from filter commit")
    if _file_sha256(path) != value["sha256"]:
        raise ValueError(f"{name} file SHA-256 differs from filter commit")
    return path


def _verify_filter_result(
    *,
    gate_root: Path,
    plan: TaskPlan,
    expected_filter_sha256: str,
    latent_seed_source_tasks: Mapping[str, GenerationTask] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    raw_root = gate_root / "raw"
    recovery = recover_paired_pilot_tasks(
        plan,
        raw_root,
        latent_seed_source_tasks=latent_seed_source_tasks,
    )
    if recovery.rebuild_task_ids or len(recovery.durable_task_ids) != len(plan.tasks):
        raise ValueError(
            "heldout execution is incomplete: "
            f"durable={len(recovery.durable_task_ids)}, "
            f"rebuild={len(recovery.rebuild_task_ids)}"
        )
    raw_candidates = {
        candidate.candidate_id: candidate
        for shard in recovery.raw_scan.valid_shards
        for candidate in load_raw_shard_candidates(
            shard,
            expected_semantic_config_sha256=plan.semantic_config_sha256,
        )
    }
    if len(raw_candidates) != plan.total_candidates:
        raise ValueError("heldout raw candidate count differs from task plan")

    filter_root = gate_root / "filter"
    commit_path = filter_root / "filter-index.commit.json"
    commit = _read_json(commit_path, "heldout filter commit")
    if commit.get("schema_version") != 1 or commit.get("kind") != "filter_index_commit":
        raise ValueError("heldout filter commit schema is invalid")
    if commit.get("filter_config_sha256") != expected_filter_sha256:
        raise ValueError("heldout filter semantic SHA-256 differs from rebind contract")
    if commit.get("filter_contract_version") != FILTER_CONTRACT_VERSION:
        raise ValueError("heldout filter contract version differs")
    files = commit.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("heldout filter commit file descriptors are missing")
    accepted_path = _verify_file_descriptor(
        filter_root,
        files.get("accepted"),
        "accepted",
    )
    rejected_path = _verify_file_descriptor(
        filter_root,
        files.get("rejected"),
        "rejected",
    )
    accepted = _read_jsonl(accepted_path, "heldout accepted index")
    rejected = _read_jsonl(rejected_path, "heldout rejected index")
    counts = commit.get("counts")
    if not isinstance(counts, Mapping) or counts != {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "tasks": len(plan.tasks),
    }:
        raise ValueError("heldout filter commit counts differ from indexes")
    statuses = commit.get("task_statuses")
    if not isinstance(statuses, Mapping) or statuses != {
        task.task_id: "complete" for task in sorted(plan.tasks, key=lambda item: item.task_id)
    }:
        raise ValueError("heldout filter task statuses are incomplete")

    seen: set[str] = set()
    for accepted_flag, rows in ((True, accepted), (False, rejected)):
        for row in rows:
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or candidate_id in seen:
                raise ValueError("heldout filter indexes contain duplicate candidate IDs")
            seen.add(candidate_id)
            raw = raw_candidates.get(candidate_id)
            if raw is None:
                raise ValueError("heldout filter index references an unknown raw candidate")
            if (
                row.get("task_id") != raw.task_id
                or row.get("candidate_index") != raw.candidate_index
                or row.get("latent_seed") != raw.latent_seed
            ):
                raise ValueError("heldout filter index identity differs from raw candidate")
            raw_reference = row.get("raw")
            if not isinstance(raw_reference, Mapping):
                raise ValueError("heldout filter raw reference is missing")
            expected_paths = {
                "commit": raw.reference.commit_path.resolve(),
                "arrays": raw.reference.arrays_path.resolve(),
                "metadata": raw.reference.metadata_path.resolve(),
            }
            if raw_reference.get("offset") != raw.reference.raw_offset or any(
                (filter_root / str(raw_reference.get(name, ""))).resolve()
                != expected_path
                for name, expected_path in expected_paths.items()
            ):
                raise ValueError("heldout filter raw reference differs from committed shard")
            expected_evaluation_id = filter_evaluation_id(
                candidate_id=candidate_id,
                filter_config_sha256=expected_filter_sha256,
                filter_contract_version=FILTER_CONTRACT_VERSION,
            )
            if row.get("filter_evaluation_id") != expected_evaluation_id:
                raise ValueError("heldout filter evaluation ID differs from contract")
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("heldout filter result metrics are missing")
            if accepted_flag:
                if metrics.get("first_failed_stage") is not None:
                    raise ValueError("accepted heldout candidate has a failed stage")
            else:
                reasons = row.get("rejection_reasons")
                if (
                    not isinstance(reasons, list)
                    or not reasons
                    or row.get("primary_rejection_reason") != reasons[0]
                    or metrics.get("first_failed_stage") is None
                ):
                    raise ValueError("rejected heldout candidate evidence is invalid")
    if seen != set(raw_candidates):
        raise ValueError("heldout filter indexes do not cover every raw candidate")
    raw_commit_set_sha256 = canonical_sha256(
        [
            {
                "shard_index": shard.shard_index,
                "commit_sha256": _file_sha256(shard.commit_path),
            }
            for shard in recovery.raw_scan.valid_shards
        ]
    )
    return accepted, rejected, commit, raw_commit_set_sha256


def _summarize_funnel(
    plan: TaskPlan,
    accepted: Iterable[Mapping[str, Any]],
    rejected: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    tasks = {task.task_id: task for task in plan.tasks}
    accepted_rows = list(accepted)
    rejected_rows = list(rejected)
    accepted_keys = {
        (str(row["task_id"]), int(row["candidate_index"]))
        for row in accepted_rows
    }
    outcome_by_task_index = {
        (row["task_id"], int(row["candidate_index"])): "accepted"
        for row in accepted_rows
    }
    outcome_by_task_index.update(
        {
            (row["task_id"], int(row["candidate_index"])): str(
                row["metrics"]["first_failed_stage"]
            )
            for row in rejected_rows
        }
    )
    rows_by_skill_arm: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for (task_id, _), outcome in outcome_by_task_index.items():
        task = tasks[task_id]
        arm = pilot_evaluation_arm(task)
        rows_by_skill_arm[(task.skill_id, arm)][outcome] += 1
    by_skill_and_arm = []
    for (skill_id, arm), counts in sorted(rows_by_skill_arm.items()):
        by_skill_and_arm.append(
            {
                "skill_id": skill_id,
                "evaluation_arm": arm,
                "candidate_count": sum(counts.values()),
                "accepted_count": counts["accepted"],
                "funnel": dict(sorted(counts.items())),
            }
        )

    paired_outcomes = Counter()
    grouped_tasks: dict[str, list[GenerationTask]] = defaultdict(list)
    for task in plan.tasks:
        if task.proposal_mode == "learned_conditioned_prior":
            grouped_tasks[latent_group_id(task)].append(task)
    for group in grouped_tasks.values():
        arms = {pilot_evaluation_arm(task): task for task in group}
        if set(arms) != {"learned_conditioned", "learned_none_control"}:
            raise ValueError("heldout learned pair is incomplete")
        conditioned = arms["learned_conditioned"]
        control = arms["learned_none_control"]
        for candidate_index in range(plan.candidate_budget):
            conditioned_pass = (
                conditioned.task_id,
                candidate_index,
            ) in accepted_keys
            control_pass = (control.task_id, candidate_index) in accepted_keys
            if conditioned_pass and control_pass:
                outcome = "both_accepted"
            elif conditioned_pass:
                outcome = "conditioned_only_accepted"
            elif control_pass:
                outcome = "control_only_accepted"
            else:
                outcome = "neither_accepted"
            paired_outcomes[outcome] += 1

    arm_totals: dict[str, Counter[str]] = defaultdict(Counter)
    for row in by_skill_and_arm:
        arm_totals[row["evaluation_arm"]]["candidates"] += row["candidate_count"]
        arm_totals[row["evaluation_arm"]]["accepted"] += row["accepted_count"]
    formal_accepted = (
        arm_totals["learned_conditioned"]["accepted"]
        + arm_totals["rule_guided_none"]["accepted"]
    )
    return {
        "by_skill_and_arm": by_skill_and_arm,
        "by_arm": [
            {
                "evaluation_arm": arm,
                "candidate_count": counts["candidates"],
                "accepted_count": counts["accepted"],
            }
            for arm, counts in sorted(arm_totals.items())
        ],
        "paired_control": dict(sorted(paired_outcomes.items())),
        "formal_accepted_count": formal_accepted,
        "control_accepted_count": arm_totals["learned_none_control"]["accepted"],
        "skill_count": len({task.skill_id for task in plan.tasks}),
    }


def build_repair_dev_candidate_evidence(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    run_manifest_path: str | Path,
    run_manifest_sha256: str,
    config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    training_summary_path: str | Path | None = None,
    training_metrics_path: str | Path | None = None,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Derive one immutable Repair Dev gate from formal training artifacts."""

    root = Path(repository_root).resolve()
    config = load_counterfactual_config(
        _resolved(root, config_path),
        repository_root=root,
    )
    candidate = validate_repair_formal_candidate(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        expected_schema_sha256=config.active_checkpoint.schema_sha256,
        repository_root=root,
    )
    formal_root = candidate.run_manifest_path.parent
    summary_path = (
        formal_root / "summary.json"
        if training_summary_path is None
        else _resolved(root, training_summary_path)
    )
    metrics_path = (
        formal_root / "metrics.jsonl"
        if training_metrics_path is None
        else _resolved(root, training_metrics_path)
    )
    summary = _read_json(summary_path, "repair-formal training summary")
    selection = summary.get("formal_selection")
    progress = summary.get("progress")
    if not isinstance(selection, Mapping) or not isinstance(progress, Mapping):
        raise ValueError("repair-formal summary lacks selection or progress")
    frozen_budget = selection.get("frozen_epoch_budget")
    completed_epochs = progress.get("epoch")
    formal_training_complete = (
        summary.get("status") == "complete"
        and summary.get("stage") == "repair-formal"
        and summary.get("stop_reason") == "fixed_epoch_budget"
        and isinstance(frozen_budget, int)
        and not isinstance(frozen_budget, bool)
        and frozen_budget > 0
        and completed_epochs == frozen_budget
        and summary.get("epoch_records_written") == frozen_budget
        and summary.get("metrics_records_written") == frozen_budget
        and selection.get("active_checkpoint_selected") is False
        and selection.get("active_checkpoint_gate")
        == "heldout_generation_capability_gate_required"
    )
    if not formal_training_complete:
        raise ValueError("repair-formal training summary is not complete and frozen")

    rows = _read_jsonl(metrics_path, "repair-formal training metrics")
    matches = [
        row
        for row in rows
        if row.get("kind") == "epoch"
        and row.get("stage") == "repair-formal"
        and row.get("completed_epoch") is True
        and row.get("epoch") == candidate.candidate_epoch - 1
        and row.get("global_step") == candidate.global_step
    ]
    if len(matches) != 1:
        raise ValueError("repair-formal metrics do not identify exactly one candidate epoch")
    row = matches[0]
    checkpoint_selection = row.get("checkpoint_selection")
    if (
        not isinstance(checkpoint_selection, Mapping)
        or checkpoint_selection.get("active_checkpoint_gate")
        != "heldout_generation_capability_gate_required"
        or _resolved(root, checkpoint_selection.get("epoch_candidate", ""))
        != candidate.checkpoint_path
    ):
        raise ValueError("repair-formal metric row checkpoint identity differs")
    validation = row.get("validation")
    validation_loss = row.get("validation_loss")
    if not isinstance(validation, Mapping) or not isinstance(
        validation_loss, Mapping
    ):
        raise ValueError("repair-formal candidate lacks Repair Dev metrics")
    prior = validation.get("prior")
    constant = validation.get("constant_velocity")
    if not isinstance(prior, Mapping) or not isinstance(constant, Mapping):
        raise ValueError("repair-formal candidate lacks prior or baseline metrics")

    metric_values = {
        "repair_dev_sample_count": validation.get("sample_count"),
        "prior_min_ade_6": prior.get("min_ade"),
        "prior_min_fde_6": prior.get("min_fde"),
        "constant_velocity_ade": constant.get("ade"),
        "constant_velocity_fde": constant.get("fde"),
        "validation_total_loss": validation_loss.get("total_loss"),
        "validation_seam_velocity_loss": validation_loss.get(
            "seam_velocity_loss"
        ),
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in metric_values.values()
    ):
        raise ValueError("repair-formal candidate metrics must be finite numbers")
    final_evaluation = summary.get("final_evaluation")
    if not isinstance(final_evaluation, Mapping):
        raise ValueError("repair-formal summary final evaluation is missing")
    expected_sample_count = final_evaluation.get("sample_count")
    if (
        isinstance(expected_sample_count, bool)
        or not isinstance(expected_sample_count, int)
        or expected_sample_count <= 0
    ):
        raise ValueError("repair-formal summary sample count is invalid")
    source = (
        f"repair-formal metrics={_file_sha256(metrics_path)};"
        f"summary={_file_sha256(summary_path)};epoch={candidate.candidate_epoch}"
    )
    gates = [
        {
            "name": "formal_epoch_budget_complete",
            "passed": completed_epochs == frozen_budget,
            "comparison": "value == threshold",
            "value": completed_epochs,
            "threshold": frozen_budget,
            "source": source,
        },
        {
            "name": "repair_dev_sample_count_complete",
            "passed": metric_values["repair_dev_sample_count"]
            == expected_sample_count,
            "comparison": "value == threshold",
            "value": metric_values["repair_dev_sample_count"],
            "threshold": expected_sample_count,
            "source": source,
        },
        {
            "name": "prior_min_ade_6_beats_constant_velocity",
            "passed": metric_values["prior_min_ade_6"]
            < metric_values["constant_velocity_ade"],
            "comparison": "value < threshold",
            "value": metric_values["prior_min_ade_6"],
            "threshold": metric_values["constant_velocity_ade"],
            "source": source,
        },
        {
            "name": "prior_min_fde_6_beats_constant_velocity",
            "passed": metric_values["prior_min_fde_6"]
            < metric_values["constant_velocity_fde"],
            "comparison": "value < threshold",
            "value": metric_values["prior_min_fde_6"],
            "threshold": metric_values["constant_velocity_fde"],
            "source": source,
        },
    ]
    evidence = {
        "schema_version": 1,
        "kind": "repair_dev_candidate_gate",
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "run_manifest_sha256": candidate.run_manifest_sha256,
        "candidate_epoch": candidate.candidate_epoch,
        "source_partition": "repair_dev_from_formal_train",
        "formal_training_complete": True,
        "metrics": {key: float(value) for key, value in metric_values.items()},
        "gates": gates,
        "status": "passed" if all(gate["passed"] for gate in gates) else "failed",
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    gate_root = _resolved(root, output_root) / candidate.checkpoint_sha256
    evidence_path = _write_immutable_json(
        gate_root / "repair_dev_candidate_gate.json",
        evidence,
    )
    return {
        "evidence": evidence,
        "outputs": {
            "repair_dev_candidate_gate": _path_label(root, evidence_path),
            "sha256": _file_sha256(evidence_path),
        },
    }


def _load_dev_evidence(
    path: Path,
    expected_sha256: str,
    candidate: RepairFormalCandidate,
) -> dict[str, Any]:
    if _file_sha256(path) != _sha256_text(
        expected_sha256,
        "repair_dev_evidence_sha256",
    ):
        raise ValueError("repair-dev evidence SHA-256 mismatch")
    value = _read_json(path, "repair-dev candidate evidence")
    required = {
        "schema_version",
        "kind",
        "checkpoint_sha256",
        "run_manifest_sha256",
        "candidate_epoch",
        "source_partition",
        "formal_training_complete",
        "metrics",
        "gates",
        "status",
        "validation_manifests_opened",
        "final_validation_accessed",
    }
    if set(value) != required:
        raise ValueError("repair-dev evidence has missing or unknown fields")
    if value["schema_version"] != 1 or value["kind"] != "repair_dev_candidate_gate":
        raise ValueError("repair-dev evidence schema is invalid")
    if (
        value["checkpoint_sha256"] != candidate.checkpoint_sha256
        or value["run_manifest_sha256"] != candidate.run_manifest_sha256
        or value["candidate_epoch"] != candidate.candidate_epoch
    ):
        raise ValueError("repair-dev evidence candidate identity differs")
    if value["source_partition"] != "repair_dev_from_formal_train":
        raise ValueError("repair-dev evidence source partition is invalid")
    if value["validation_manifests_opened"] is not False or value[
        "final_validation_accessed"
    ] is not False:
        raise ValueError("repair-dev evidence accessed a forbidden validation split")
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("repair-dev evidence metrics must be non-empty")
    for name, metric in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("repair-dev metric names must be non-empty strings")
        if (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
        ):
            raise ValueError("repair-dev metrics must be finite numbers")
    gates = value["gates"]
    if not isinstance(gates, list) or not gates:
        raise ValueError("repair-dev evidence gates must be non-empty")
    gate_names: set[str] = set()
    for gate in gates:
        if not isinstance(gate, Mapping) or set(gate) != {
            "name",
            "passed",
            "comparison",
            "value",
            "threshold",
            "source",
        }:
            raise ValueError("repair-dev gate evidence is invalid")
        if not isinstance(gate["name"], str) or gate["name"] in gate_names:
            raise ValueError("repair-dev gate names must be unique")
        if not isinstance(gate["passed"], bool):
            raise ValueError("repair-dev gate passed must be boolean")
        gate_names.add(gate["name"])
    all_passed = all(gate["passed"] for gate in gates)
    expected_status = (
        "passed"
        if value["formal_training_complete"] is True and all_passed
        else "failed"
    )
    if value["status"] != expected_status:
        raise ValueError("repair-dev evidence status differs from its gates")
    return value


def aggregate_repair_heldout_gate(
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    run_manifest_path: str | Path,
    run_manifest_sha256: str,
    repair_dev_evidence_path: str | Path,
    repair_dev_evidence_sha256: str,
    config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    filter_config_path: str | Path = "configs/generation/filters_v1.yaml",
    detection_config_path: str | Path = "configs/seed_detection.yaml",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Verify a complete heldout result and write a non-active recommendation."""

    root = Path(repository_root).resolve()
    config = load_counterfactual_config(
        _resolved(root, config_path),
        repository_root=root,
    )
    candidate = validate_repair_formal_candidate(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        run_manifest_path=run_manifest_path,
        run_manifest_sha256=run_manifest_sha256,
        expected_schema_sha256=config.active_checkpoint.schema_sha256,
        repository_root=root,
    )
    gate_root = _resolved(root, output_root) / candidate.checkpoint_sha256
    rebind_path = gate_root / "rebind_contract.json"
    rebind = _read_json(rebind_path, "repair heldout rebind contract")
    expected_rebind_fields = {
        "schema_version": HELDOUT_GATE_VERSION,
        "kind": "repair_heldout_rebind_contract",
        "contract": HELDOUT_GATE_CONTRACT,
        "status": "rebound_pending_execution",
        "formal_active": False,
        "active_config_modified": False,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    if (
        any(rebind.get(key) != value for key, value in expected_rebind_fields.items())
        or rebind.get("checkpoint", {}).get("sha256")
        != candidate.checkpoint_sha256
        or rebind.get("run_manifest", {}).get("sha256")
        != candidate.run_manifest_sha256
        or rebind.get("execution_adapter") != HELDOUT_EXECUTION_ADAPTER
    ):
        raise ValueError("repair heldout rebind checkpoint identity differs")
    rebound = rebind.get("rebound")
    if not isinstance(rebound, Mapping):
        raise ValueError("repair heldout rebound descriptor is missing")
    filter_fingerprint = _heldout_filter_fingerprint(
        repository_root=root,
        config_path=config_path,
        filter_config_path=filter_config_path,
        detection_config_path=detection_config_path,
    )
    if (
        rebound.get("filter_semantic_sha256")
        != filter_fingerprint.semantic_sha256
        or rebound.get("filter_dependency_sha256")
        != dict(filter_fingerprint.file_sha256)
    ):
        raise ValueError("repair heldout filter semantics changed after execution")
    loaded = load_task_plan(
        gate_root,
        expected_semantic_config_sha256=rebound.get("semantic_config_sha256"),
        current_execution_config_sha256=rebound.get("execution_config_sha256"),
    )
    if loaded.execution_config_changed:
        raise ValueError("repair heldout execution configuration changed")
    if (
        _file_sha256(gate_root / "task_plan.jsonl")
        != rebound.get("task_plan_sha256")
        or _file_sha256(gate_root / "task_plan.summary.json")
        != rebound.get("task_plan_summary_sha256")
    ):
        raise ValueError("repair heldout rebound task plan hash changed")
    source = rebind.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("repair heldout source descriptor is missing")
    source_plan_descriptor = source.get("task_plan")
    audit_descriptor = source.get("repair_split_audit")
    if not isinstance(source_plan_descriptor, Mapping) or not isinstance(
        audit_descriptor, Mapping
    ):
        raise ValueError("repair heldout source plan descriptors are missing")
    source_plan_path = _resolved(root, source_plan_descriptor.get("path", ""))
    audit_path = _resolved(root, audit_descriptor.get("path", ""))
    source_plan, _ = _load_frozen_source_plan(
        source_plan_dir=source_plan_path.parent,
        repair_audit_path=audit_path,
        repository_root=root,
    )
    latent_seed_sources = _latent_seed_source_tasks(source_plan, loaded.plan)
    task_rebind_path = _resolved(root, rebound.get("task_rebind_path", ""))
    if (
        _file_sha256(task_rebind_path) != rebound.get("task_rebind_sha256")
        or _read_json(task_rebind_path, "heldout task rebind")
        != _task_rebind_contract(source_plan, loaded.plan)
    ):
        raise ValueError("repair heldout task rebind mapping changed")
    accepted, rejected, filter_commit, raw_commit_set_sha256 = _verify_filter_result(
        gate_root=gate_root,
        plan=loaded.plan,
        expected_filter_sha256=str(rebound.get("filter_semantic_sha256")),
        latent_seed_source_tasks=latent_seed_sources,
    )
    funnel = _summarize_funnel(loaded.plan, accepted, rejected)
    executed_skills = {task.skill_id for task in loaded.plan.tasks}
    unknown_skills = sorted(executed_skills - set(config.formal_skill_ids))
    if unknown_skills:
        raise ValueError(
            f"repair heldout plan contains non-formal skills: {unknown_skills}"
        )
    frozen_without_tasks = sorted(set(config.formal_skill_ids) - executed_skills)
    formal_skill_status = [
        {
            "skill_id": skill_id,
            "status": (
                "heldout_executed"
                if skill_id in executed_skills
                else "frozen_without_heldout_task"
            ),
        }
        for skill_id in config.formal_skill_ids
    ]
    dev_path = _resolved(root, repair_dev_evidence_path)
    dev_evidence = _load_dev_evidence(
        dev_path,
        repair_dev_evidence_sha256,
        candidate,
    )
    arm = {
        row["evaluation_arm"]: row for row in funnel["by_arm"]
    }
    ability_gates = {
        "complete_task_and_candidate_budget": True,
        "all_formal_skills_have_explicit_status": (
            len(formal_skill_status) == len(config.formal_skill_ids)
        ),
        "formal_candidate_accepted": funnel["formal_accepted_count"] > 0,
        "learned_conditioned_candidate_accepted": arm.get(
            "learned_conditioned", {}
        ).get("accepted_count", 0)
        > 0,
        "rule_guided_candidate_accepted": arm.get("rule_guided_none", {}).get(
            "accepted_count", 0
        )
        > 0,
        "conditioned_acceptance_exceeds_control": arm.get(
            "learned_conditioned", {}
        ).get("accepted_count", 0)
        > arm.get("learned_none_control", {}).get("accepted_count", 0),
    }
    ability_passed = all(ability_gates.values())
    dev_passed = dev_evidence["status"] == "passed"
    failures = [name for name, passed in ability_gates.items() if not passed]
    if not dev_passed:
        failures.append("repair_dev_candidate_gate")
    recommendation = "recommend_promotion" if ability_passed and dev_passed else "reject"
    heldout_summary = {
        "schema_version": HELDOUT_GATE_VERSION,
        "kind": "repair_heldout_gate_summary",
        "status": "passed" if ability_passed else "failed",
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "run_manifest_sha256": candidate.run_manifest_sha256,
        "candidate_epoch": candidate.candidate_epoch,
        "task_plan_id": loaded.plan.task_plan_id,
        "task_count": len(loaded.plan.tasks),
        "candidate_count": loaded.plan.total_candidates,
        "formal_skill_count": len(config.formal_skill_ids),
        "executed_skill_count": len(executed_skills),
        "frozen_skills_without_heldout_tasks": frozen_without_tasks,
        "formal_skill_status": formal_skill_status,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "raw_commit_set_sha256": raw_commit_set_sha256,
        "filter_commit_sha256": _file_sha256(
            gate_root / "filter" / "filter-index.commit.json"
        ),
        "filter_semantic_sha256": filter_commit["filter_config_sha256"],
        "funnel": funnel,
        "ability_gates": ability_gates,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    heldout_summary_path = _write_immutable_json(
        gate_root / "heldout_gate_summary.json",
        heldout_summary,
    )
    promotion = {
        "schema_version": HELDOUT_GATE_VERSION,
        "kind": "repair_checkpoint_promotion_recommendation",
        "contract": HELDOUT_GATE_CONTRACT,
        "status": "completed",
        "recommendation": recommendation,
        "failure_reasons": failures,
        "formal_active": False,
        "active_config_modified": False,
        "requires_separate_active_config_update": True,
        "checkpoint": rebind["checkpoint"],
        "run_manifest": rebind["run_manifest"],
        "evidence": {
            "rebind_contract": {
                "path": _path_label(root, rebind_path),
                "sha256": _file_sha256(rebind_path),
            },
            "heldout_gate_summary": {
                "path": _path_label(root, heldout_summary_path),
                "sha256": _file_sha256(heldout_summary_path),
                "status": heldout_summary["status"],
            },
            "repair_dev_candidate_gate": {
                "path": _path_label(root, dev_path),
                "sha256": _file_sha256(dev_path),
                "status": dev_evidence["status"],
                "metrics": dict(dev_evidence["metrics"]),
                "gates": list(dev_evidence["gates"]),
            },
        },
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    promotion_path = _write_immutable_json(
        gate_root / "promotion_recommendation.json",
        promotion,
    )
    return {
        **promotion,
        "outputs": {
            "heldout_gate_summary": _path_label(root, heldout_summary_path),
            "promotion_recommendation": _path_label(root, promotion_path),
        },
    }


__all__ = [
    "DEFAULT_HELDOUT_PLAN",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_REPAIR_AUDIT",
    "HELDOUT_GATE_CONTRACT",
    "RepairFormalCandidate",
    "aggregate_repair_heldout_gate",
    "build_repair_dev_candidate_evidence",
    "execute_repair_heldout_plan",
    "rebind_repair_heldout_plan",
    "validate_repair_formal_candidate",
]

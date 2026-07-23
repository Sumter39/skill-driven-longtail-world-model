"""Future-free CVAE loading and deterministic Prior inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from skilldrive.data import CVAESchema, TensorizedPriorContext, cvae_schema_fingerprint
from skilldrive.generation.config import ActiveCheckpointConfig
from skilldrive.models import ConditionalCVAE
from skilldrive.training.checkpoint import TrainingProgress, load_checkpoint


_CONTEXT_TENSORS = (
    "actor_history",
    "actor_time_mask",
    "actor_mask",
    "actor_type_id",
    "actor_role_id",
    "map_polylines",
    "map_point_mask",
    "map_polyline_mask",
    "map_type_id",
    "target_actor_index",
    "skill_id",
    "skill_parameters",
    "parameter_mask",
)


@dataclass(frozen=True)
class ActiveCVAERuntime:
    model: ConditionalCVAE
    schema: CVAESchema
    checkpoint_path: Path
    checkpoint_sha256: str
    run_manifest_path: Path
    run_manifest_sha256: str
    schema_sha256: str
    manifest_stage: str
    progress: TrainingProgress
    device: torch.device


@dataclass(frozen=True)
class PriorBatchOutput:
    future_position_local: np.ndarray
    latent: np.ndarray
    prior_mean: np.ndarray
    prior_logvar: np.ndarray


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(
    path: Path,
    *,
    allowed_stages: frozenset[str],
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("CVAE run manifest must contain a mapping")
    for name in ("fingerprints", "model", "stage"):
        if name not in value:
            raise ValueError(f"CVAE run manifest is missing {name}")
    if value["stage"] not in allowed_stages:
        raise ValueError(
            "CVAE run manifest stage must be one of "
            f"{sorted(allowed_stages)}, got {value['stage']!r}"
        )
    if not isinstance(value["fingerprints"], dict) or not isinstance(
        value["model"], dict
    ):
        raise ValueError("CVAE run manifest fingerprints and model must be mappings")
    return value


def _validate_model_schema(model: Mapping[str, Any], schema: CVAESchema) -> None:
    expected = {
        "num_actor_types": len(schema.actor_type_vocabulary.tokens),
        "num_actor_roles": len(schema.role_vocabulary.tokens),
        "num_map_types": len(schema.map_type_vocabulary.tokens),
        "num_skills": len(schema.skill_vocabulary.tokens),
        "parameter_dim": schema.parameter_schema.dimension,
        "future_steps": 60,
    }
    mismatches = {
        name: (model.get(name), value)
        for name, value in expected.items()
        if model.get(name) != value
    }
    if mismatches:
        raise ValueError(f"CVAE run manifest model/schema mismatch: {mismatches}")


def _load_cvae(
    *,
    checkpoint_path: str | Path,
    run_manifest_path: str | Path,
    schema: CVAESchema,
    expected_checkpoint_sha256: str,
    expected_schema_sha256: str,
    device: str | torch.device,
    expected_run_manifest_sha256: str,
    allowed_manifest_stages: frozenset[str],
    required_repair_contract: str | None,
) -> ActiveCVAERuntime:
    """Load one hash-bound inference model without touching validation data or RNG."""

    checkpoint = Path(checkpoint_path)
    manifest_path = Path(run_manifest_path)
    actual_checkpoint_sha256 = file_sha256(checkpoint)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 mismatch")
    actual_manifest_sha256 = file_sha256(manifest_path)
    if actual_manifest_sha256 != expected_run_manifest_sha256:
        raise ValueError("run manifest SHA-256 mismatch")
    actual_schema_sha256 = cvae_schema_fingerprint(schema)
    if actual_schema_sha256 != expected_schema_sha256:
        raise ValueError("CVAE schema SHA-256 mismatch")

    manifest = _read_manifest(
        manifest_path,
        allowed_stages=allowed_manifest_stages,
    )
    if (
        required_repair_contract is not None
        and manifest.get("repair_contract") != required_repair_contract
    ):
        raise ValueError(
            "repair CVAE run manifest contract mismatch: expected "
            f"{required_repair_contract!r}, got {manifest.get('repair_contract')!r}"
        )
    _validate_model_schema(manifest["model"], schema)
    target_device = torch.device(device)
    model = ConditionalCVAE(**manifest["model"]).to(target_device)
    progress, _ = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_fingerprints=manifest["fingerprints"],
        map_location="cpu",
        restore_rng=False,
    )
    model.eval()
    model.requires_grad_(False)
    return ActiveCVAERuntime(
        model=model,
        schema=schema,
        checkpoint_path=checkpoint,
        checkpoint_sha256=actual_checkpoint_sha256,
        run_manifest_path=manifest_path,
        run_manifest_sha256=actual_manifest_sha256,
        schema_sha256=actual_schema_sha256,
        manifest_stage=str(manifest["stage"]),
        progress=progress,
        device=target_device,
    )


def load_active_cvae(
    *,
    checkpoint_path: str | Path,
    run_manifest_path: str | Path,
    schema: CVAESchema,
    expected_checkpoint_sha256: str,
    expected_schema_sha256: str,
    device: str | torch.device,
    expected_run_manifest_sha256: str,
) -> ActiveCVAERuntime:
    """Load the legacy frozen formal model used by the existing generation stages."""

    return _load_cvae(
        checkpoint_path=checkpoint_path,
        run_manifest_path=run_manifest_path,
        schema=schema,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_schema_sha256=expected_schema_sha256,
        device=device,
        expected_run_manifest_sha256=expected_run_manifest_sha256,
        allowed_manifest_stages=frozenset(("formal",)),
        required_repair_contract=None,
    )


RepairCheckpointMode = Literal["diagnostic-overfit", "formal"]


def load_repair_cvae(
    *,
    checkpoint_path: str | Path,
    run_manifest_path: str | Path,
    schema: CVAESchema,
    expected_checkpoint_sha256: str,
    expected_schema_sha256: str,
    device: str | torch.device,
    expected_run_manifest_sha256: str,
    checkpoint_mode: RepairCheckpointMode,
    expected_repair_contract: str = "cvae_generation_repair_v1",
) -> ActiveCVAERuntime:
    """Load a repair checkpoint under an explicit diagnostic or formal policy.

    ``diagnostic-overfit`` can load only ``repair-overfit`` and is never an active
    formal checkpoint. The formal path can load only ``repair-formal``.
    """

    expected_stage_by_mode = {
        "diagnostic-overfit": "repair-overfit",
        "formal": "repair-formal",
    }
    if checkpoint_mode not in expected_stage_by_mode:
        raise ValueError(
            "checkpoint_mode must be diagnostic-overfit or formal"
        )
    expected_stage = expected_stage_by_mode[checkpoint_mode]
    return _load_cvae(
        checkpoint_path=checkpoint_path,
        run_manifest_path=run_manifest_path,
        schema=schema,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_schema_sha256=expected_schema_sha256,
        device=device,
        expected_run_manifest_sha256=expected_run_manifest_sha256,
        allowed_manifest_stages=frozenset((expected_stage,)),
        required_repair_contract=expected_repair_contract,
    )


def _resolved(repository_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _read_json_mapping(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a mapping: {path}")
    return value


def validate_active_checkpoint_promotion(
    active_checkpoint: ActiveCheckpointConfig,
    *,
    repository_root: str | Path = ".",
) -> dict[str, Any] | None:
    """Verify the external promotion evidence before treating a repair model as active."""

    root = Path(repository_root).resolve()
    configured_manifest = _resolved(root, active_checkpoint.run_manifest)
    if file_sha256(configured_manifest) != active_checkpoint.run_manifest_sha256:
        raise ValueError("active checkpoint run manifest SHA-256 mismatch")
    manifest = _read_manifest(
        configured_manifest,
        allowed_stages=frozenset(("formal", "repair-formal")),
    )
    if manifest.get("stage") != active_checkpoint.run_manifest_stage:
        raise ValueError("active checkpoint run manifest stage differs from config")
    if active_checkpoint.run_manifest_stage == "formal":
        if any(
            value is not None
            for value in (
                active_checkpoint.repair_contract,
                active_checkpoint.promotion_recommendation,
                active_checkpoint.promotion_recommendation_sha256,
            )
        ):
            raise ValueError("legacy formal active checkpoint cannot carry repair evidence")
        if manifest.get("repair_contract") is not None:
            raise ValueError("legacy formal active checkpoint has a repair contract")
        return None
    if active_checkpoint.run_manifest_stage != "repair-formal":
        raise ValueError("active checkpoint run manifest stage is unsupported")
    if active_checkpoint.repair_contract != "cvae_generation_repair_v1":
        raise ValueError("active repair checkpoint contract is invalid")
    if manifest.get("repair_contract") != active_checkpoint.repair_contract:
        raise ValueError("active repair run manifest contract differs from config")
    if (
        active_checkpoint.promotion_recommendation is None
        or active_checkpoint.promotion_recommendation_sha256 is None
    ):
        raise ValueError("active repair checkpoint lacks promotion evidence")

    promotion_path = _resolved(root, active_checkpoint.promotion_recommendation)
    if file_sha256(promotion_path) != active_checkpoint.promotion_recommendation_sha256:
        raise ValueError("active checkpoint promotion SHA-256 mismatch")
    promotion = _read_json_mapping(
        promotion_path,
        "active checkpoint promotion recommendation",
    )
    expected = {
        "schema_version": 1,
        "kind": "repair_checkpoint_promotion_recommendation",
        "contract": "repair_heldout_gate_v1",
        "status": "completed",
        "recommendation": "recommend_promotion",
        "formal_active": False,
        "active_config_modified": False,
        "requires_separate_active_config_update": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    if any(promotion.get(key) != value for key, value in expected.items()):
        raise ValueError("active checkpoint promotion recommendation did not pass")
    if promotion.get("failure_reasons") != []:
        raise ValueError("active checkpoint promotion contains failure reasons")

    checkpoint = promotion.get("checkpoint")
    run_manifest = promotion.get("run_manifest")
    if not isinstance(checkpoint, Mapping) or not isinstance(run_manifest, Mapping):
        raise ValueError("active checkpoint promotion identities are missing")
    configured_checkpoint = _resolved(root, active_checkpoint.path)
    if (
        _resolved(root, Path(str(checkpoint.get("path", ""))))
        != configured_checkpoint
        or checkpoint.get("sha256") != active_checkpoint.sha256
        or _resolved(root, Path(str(run_manifest.get("path", ""))))
        != configured_manifest
        or run_manifest.get("sha256") != active_checkpoint.run_manifest_sha256
        or run_manifest.get("stage") != active_checkpoint.run_manifest_stage
        or run_manifest.get("repair_contract") != active_checkpoint.repair_contract
    ):
        raise ValueError("active checkpoint differs from promotion recommendation")

    from skilldrive.generation.heldout_gate import validate_repair_formal_candidate

    candidate = validate_repair_formal_candidate(
        checkpoint_path=configured_checkpoint,
        checkpoint_sha256=active_checkpoint.sha256,
        run_manifest_path=configured_manifest,
        run_manifest_sha256=active_checkpoint.run_manifest_sha256,
        expected_schema_sha256=active_checkpoint.schema_sha256,
        repository_root=root,
    )
    if (
        checkpoint.get("candidate_epoch") != candidate.candidate_epoch
        or checkpoint.get("global_step") != candidate.global_step
    ):
        raise ValueError("active checkpoint candidate metadata differs from promotion")

    evidence = promotion.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("active checkpoint promotion evidence is missing")
    evidence_specs = {
        "rebind_contract": None,
        "heldout_gate_summary": "passed",
        "repair_dev_candidate_gate": "passed",
    }
    evidence_values: dict[str, dict[str, Any]] = {}
    for name, required_status in evidence_specs.items():
        descriptor = evidence.get(name)
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"active checkpoint promotion lacks {name}")
        evidence_path = _resolved(root, Path(str(descriptor.get("path", ""))))
        if file_sha256(evidence_path) != descriptor.get("sha256"):
            raise ValueError(f"active checkpoint {name} SHA-256 mismatch")
        value = _read_json_mapping(evidence_path, name)
        evidence_values[name] = value
        if required_status is not None:
            if descriptor.get("status") != required_status:
                raise ValueError(f"active checkpoint {name} did not pass")
            if (
                value.get("status") != required_status
                or value.get("checkpoint_sha256") != active_checkpoint.sha256
                or value.get("run_manifest_sha256")
                != active_checkpoint.run_manifest_sha256
                or value.get("candidate_epoch") != candidate.candidate_epoch
                or value.get("final_validation_accessed") is not False
                or value.get("validation_manifests_opened") is not False
            ):
                raise ValueError(f"active checkpoint {name} content is invalid")
    rebind = evidence_values["rebind_contract"]
    rebind_checkpoint = rebind.get("checkpoint")
    rebind_manifest = rebind.get("run_manifest")
    if (
        rebind.get("schema_version") != 1
        or rebind.get("kind") != "repair_heldout_rebind_contract"
        or rebind.get("contract") != "repair_heldout_gate_v1"
        or rebind.get("status") != "rebound_pending_execution"
        or not isinstance(rebind_checkpoint, Mapping)
        or not isinstance(rebind_manifest, Mapping)
        or _resolved(root, Path(str(rebind_checkpoint.get("path", ""))))
        != configured_checkpoint
        or rebind_checkpoint.get("sha256") != active_checkpoint.sha256
        or rebind_checkpoint.get("candidate_epoch") != candidate.candidate_epoch
        or rebind_checkpoint.get("global_step") != candidate.global_step
        or _resolved(root, Path(str(rebind_manifest.get("path", ""))))
        != configured_manifest
        or rebind_manifest.get("sha256")
        != active_checkpoint.run_manifest_sha256
        or rebind_manifest.get("stage") != active_checkpoint.run_manifest_stage
        or rebind_manifest.get("repair_contract")
        != active_checkpoint.repair_contract
        or rebind.get("formal_active") is not False
        or rebind.get("active_config_modified") is not False
        or rebind.get("validation_manifests_opened") is not False
        or rebind.get("final_validation_accessed") is not False
    ):
        raise ValueError("active checkpoint rebind contract is invalid")
    heldout = evidence_values["heldout_gate_summary"]
    if (
        heldout.get("schema_version") != 1
        or heldout.get("kind") != "repair_heldout_gate_summary"
    ):
        raise ValueError("active checkpoint heldout gate contract is invalid")
    ability_gates = heldout.get("ability_gates")
    if not isinstance(ability_gates, Mapping) or not ability_gates or not all(
        value is True for value in ability_gates.values()
    ):
        raise ValueError("active checkpoint heldout ability gates did not all pass")
    repair_dev = evidence_values["repair_dev_candidate_gate"]
    if (
        repair_dev.get("schema_version") != 1
        or repair_dev.get("kind") != "repair_dev_candidate_gate"
    ):
        raise ValueError("active checkpoint Repair Dev contract is invalid")
    repair_dev_gates = repair_dev.get("gates")
    if (
        repair_dev.get("source_partition") != "repair_dev_from_formal_train"
        or repair_dev.get("formal_training_complete") is not True
        or not isinstance(repair_dev_gates, list)
        or not repair_dev_gates
        or not all(
            isinstance(gate, Mapping) and gate.get("passed") is True
            for gate in repair_dev_gates
        )
    ):
        raise ValueError("active checkpoint Repair Dev gate is invalid")
    return promotion


def load_configured_cvae(
    *,
    active_checkpoint: ActiveCheckpointConfig,
    schema: CVAESchema,
    device: str | torch.device,
    repository_root: str | Path = ".",
) -> ActiveCVAERuntime:
    """Load exactly the baseline or promoted repair model declared by config."""

    root = Path(repository_root).resolve()
    validate_active_checkpoint_promotion(
        active_checkpoint,
        repository_root=root,
    )
    checkpoint_path = _resolved(root, active_checkpoint.path)
    run_manifest_path = _resolved(root, active_checkpoint.run_manifest)
    if active_checkpoint.run_manifest_stage == "formal":
        return load_active_cvae(
            checkpoint_path=checkpoint_path,
            run_manifest_path=run_manifest_path,
            schema=schema,
            expected_checkpoint_sha256=active_checkpoint.sha256,
            expected_run_manifest_sha256=active_checkpoint.run_manifest_sha256,
            expected_schema_sha256=active_checkpoint.schema_sha256,
            device=device,
        )
    return load_repair_cvae(
        checkpoint_path=checkpoint_path,
        run_manifest_path=run_manifest_path,
        schema=schema,
        expected_checkpoint_sha256=active_checkpoint.sha256,
        expected_run_manifest_sha256=active_checkpoint.run_manifest_sha256,
        expected_schema_sha256=active_checkpoint.schema_sha256,
        device=device,
        checkpoint_mode="formal",
        expected_repair_contract=str(active_checkpoint.repair_contract),
    )


def stack_prior_contexts(
    contexts: Sequence[TensorizedPriorContext],
    *,
    device: str | torch.device,
) -> dict[str, Tensor]:
    """Stack future-free NumPy contexts into the exact model batch contract."""

    if not contexts:
        raise ValueError("at least one Prior context is required")
    target_device = torch.device(device)
    batch: dict[str, Tensor] = {}
    for name in _CONTEXT_TENSORS:
        values = [np.asarray(getattr(context, name)) for context in contexts]
        batch[name] = torch.as_tensor(np.stack(values), device=target_device)
    return batch


def standard_normal_from_seeds(
    latent_seeds: np.ndarray,
    *,
    latent_dim: int,
) -> np.ndarray:
    """Generate task-local standard normal noise independent of batching and resume."""

    seeds = np.asarray(latent_seeds)
    if seeds.ndim != 2:
        raise ValueError("latent_seeds must have shape [B, S]")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if not np.issubdtype(seeds.dtype, np.integer):
        raise ValueError("latent_seeds must contain integers")
    noise = np.empty((*seeds.shape, latent_dim), dtype=np.float32)
    for index in np.ndindex(seeds.shape):
        seed = int(seeds[index])
        if seed < 0:
            raise ValueError("latent seeds must be nonnegative")
        noise[index] = np.random.default_rng(seed).standard_normal(
            latent_dim,
            dtype=np.float32,
        )
    return noise


def generate_prior_batch(
    runtime: ActiveCVAERuntime,
    contexts: Sequence[TensorizedPriorContext],
    latent_seeds: np.ndarray,
    *,
    use_bfloat16: bool = False,
) -> PriorBatchOutput:
    """Run deterministic explicit-noise Prior inference and return CPU float32 arrays."""

    if np.asarray(latent_seeds).shape[:1] != (len(contexts),):
        raise ValueError("latent_seeds batch dimension must match contexts")
    batch = stack_prior_contexts(contexts, device=runtime.device)
    noise = standard_normal_from_seeds(
        latent_seeds,
        latent_dim=runtime.model.latent_dim,
    )
    noise_tensor = torch.as_tensor(noise, device=runtime.device)
    autocast_enabled = use_bfloat16 and runtime.device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=runtime.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        output = runtime.model.sample_prior_from_noise(batch, noise_tensor)
    return PriorBatchOutput(
        future_position_local=output.future_position_local.float().cpu().numpy(),
        latent=output.latent.float().cpu().numpy(),
        prior_mean=output.prior_mean.float().cpu().numpy(),
        prior_logvar=output.prior_logvar.float().cpu().numpy(),
    )


__all__ = [
    "ActiveCVAERuntime",
    "PriorBatchOutput",
    "file_sha256",
    "generate_prior_batch",
    "load_active_cvae",
    "load_configured_cvae",
    "load_repair_cvae",
    "RepairCheckpointMode",
    "stack_prior_contexts",
    "standard_normal_from_seeds",
    "validate_active_checkpoint_promotion",
]

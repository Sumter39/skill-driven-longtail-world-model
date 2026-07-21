"""Atomic, fingerprinted checkpoints for resumable model training."""

from __future__ import annotations

import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainingProgress:
    epoch: int
    next_batch_index: int
    global_step: int
    best_metric: float | None
    best_epoch: int | None

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.next_batch_index < 0 or self.global_step < 0:
            raise ValueError("checkpoint progress counters must be nonnegative")
        if self.best_epoch is not None and self.best_epoch < 0:
            raise ValueError("best_epoch must be nonnegative when present")


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"checkpoint RNG state is missing: {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["torch_cuda"]]
        )


def _state_dict_or_none(component: Any | None) -> dict[str, Any] | None:
    return None if component is None else component.state_dict()


def _validate_fingerprints(
    actual: Mapping[str, str],
    expected: Mapping[str, str],
) -> None:
    if dict(actual) != dict(expected):
        differing = sorted(set(actual) | set(expected))
        details = [
            f"{key}: checkpoint={actual.get(key)!r}, expected={expected.get(key)!r}"
            for key in differing
            if actual.get(key) != expected.get(key)
        ]
        raise ValueError("checkpoint fingerprint mismatch: " + "; ".join(details))


def _load_payload(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as error:
        raise ValueError(f"failed to read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint {path} has unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    required = {
        "fingerprints",
        "progress",
        "model",
        "optimizer",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"checkpoint {path} is missing fields: {sorted(missing)}")
    return payload


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    progress: TrainingProgress,
    fingerprints: Mapping[str, str],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save and verify one complete optimizer-step checkpoint."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "fingerprints": dict(sorted(fingerprints.items())),
        "progress": asdict(progress),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": _state_dict_or_none(scheduler),
        "scaler": _state_dict_or_none(scaler),
        "rng_state": capture_rng_state(),
        "extra": dict(extra or {}),
    }
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        verified = _load_payload(temporary, map_location="cpu")
        _validate_fingerprints(verified["fingerprints"], payload["fingerprints"])
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    expected_fingerprints: Mapping[str, str],
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> tuple[TrainingProgress, dict[str, Any]]:
    """Validate and restore a checkpoint, returning progress and extra metadata."""

    source = Path(path)
    payload = _load_payload(source, map_location=map_location)
    _validate_fingerprints(payload["fingerprints"], expected_fingerprints)

    try:
        progress = TrainingProgress(**payload["progress"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint {source} contains invalid progress: {error}") from error

    model.load_state_dict(payload["model"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint does not contain scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("checkpoint does not contain scaler state")
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return progress, dict(payload.get("extra") or {})


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "TrainingProgress",
    "capture_rng_state",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]

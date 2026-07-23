from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from skilldrive.training.checkpoint import (
    TrainingProgress,
    load_checkpoint,
    read_checkpoint_metadata,
    save_checkpoint,
)


def _components() -> tuple[torch.nn.Linear, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    return model, optimizer


def _optimizer_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
    loss.backward()
    optimizer.step()


def test_checkpoint_round_trip_restores_model_optimizer_progress_and_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    np.random.seed(7)
    random.seed(7)
    model, optimizer = _components()
    _optimizer_step(model, optimizer)
    progress = TrainingProgress(3, 4, 25, 1.25, 2)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=progress,
        fingerprints={"config": "abc", "train": "def"},
        extra={"note": "verified"},
    )

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(2)
    saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_model, restored_optimizer = _components()
    restored_progress, extra = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        expected_fingerprints={"config": "abc", "train": "def"},
    )

    assert restored_progress == progress
    assert extra == {"note": "verified"}
    for actual, expected in zip(restored_model.parameters(), saved_parameters, strict=True):
        torch.testing.assert_close(actual, expected)
    assert random.random() == pytest.approx(expected_python)
    assert float(np.random.random()) == pytest.approx(expected_numpy)
    torch.testing.assert_close(torch.rand(2), expected_torch)
    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_rejects_fingerprint_drift_before_restoring(tmp_path: Path) -> None:
    model, optimizer = _components()
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(0, 0, 0, None, None),
        fingerprints={"config": "old"},
    )
    original = [parameter.detach().clone() for parameter in model.parameters()]

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_fingerprints={"config": "new"},
        )

    for actual, expected in zip(model.parameters(), original, strict=True):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a torch checkpoint")
    model, optimizer = _components()

    with pytest.raises(ValueError, match="failed to read checkpoint"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_fingerprints={},
        )


def test_checkpoint_metadata_is_read_only_and_validated(tmp_path: Path) -> None:
    model, optimizer = _components()
    path = tmp_path / "candidate.pt"
    progress = TrainingProgress(3, 0, 21, 1.0, 2)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=progress,
        fingerprints={"contract": "repair"},
        extra={"checkpoint": {"role": "epoch_validation_candidate"}},
    )

    metadata = read_checkpoint_metadata(path)

    assert metadata.progress == progress
    assert metadata.fingerprints == {"contract": "repair"}
    assert metadata.extra == {
        "checkpoint": {"role": "epoch_validation_candidate"}
    }


def test_training_progress_rejects_negative_counters() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        TrainingProgress(epoch=0, next_batch_index=-1, global_step=0, best_metric=None, best_epoch=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_restore_accepts_rng_tensors_loaded_to_cuda(tmp_path: Path) -> None:
    model, optimizer = _components()
    path = tmp_path / "cuda-mapped.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=TrainingProgress(0, 0, 0, None, None),
        fingerprints={"config": "cuda"},
    )

    load_checkpoint(
        path,
        model=model.cuda(),
        optimizer=optimizer,
        expected_fingerprints={"config": "cuda"},
        map_location="cuda",
    )

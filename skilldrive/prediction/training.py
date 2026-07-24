"""Losses, evaluation, and resumable training for trajectory predictors."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from skilldrive.prediction.data import collate_prediction_samples, prediction_model_inputs
from skilldrive.prediction.metrics import PredictionMetricSums, multimodal_prediction_sums
from skilldrive.prediction.model import PredictionOutput
from skilldrive.training.checkpoint import TrainingProgress, load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class PredictionLoss:
    total: Tensor
    regression: Tensor
    classification: Tensor
    winning_modes: Tensor


@dataclass(frozen=True)
class PredictionTrainConfig:
    batch_size: int = 64
    max_steps: int = 4000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    classification_weight: float = 0.2
    num_workers: int = 2
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    amp: bool = True
    checkpoint_every_steps: int = 250
    validation_every_steps: int = 250
    progress_every_seconds: float = 5.0

    def __post_init__(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "gradient_clip_norm": self.gradient_clip_norm,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "validation_every_steps": self.validation_every_steps,
            "progress_every_seconds": self.progress_every_seconds,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"training configuration values must be positive: {positive}")
        if self.num_workers < 0 or self.prefetch_factor < 1:
            raise ValueError("worker configuration is invalid")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")


def prediction_loss(
    output: PredictionOutput,
    target: Tensor,
    mask: Tensor,
    *,
    classification_weight: float = 0.2,
) -> PredictionLoss:
    """Best-of-K Smooth L1 regression plus winning-mode classification."""

    predictions = output.trajectories
    logits = output.logits
    if predictions.ndim != 4 or target.ndim != 3 or predictions.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes do not align")
    if logits.shape != predictions.shape[:2]:
        raise ValueError("mode logits do not align with predictions")
    if mask.shape != target.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("target mask must be boolean [B, T]")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every training sample needs at least one future point")
    valid = mask[:, None, :, None]
    target_safe = torch.where(mask[:, :, None], target, torch.zeros_like(target))
    expanded_target = target_safe[:, None].expand_as(predictions)
    per_coordinate = F.smooth_l1_loss(
        predictions, expanded_target, reduction="none"
    )
    counts = mask.sum(dim=1).clamp_min(1).to(predictions.dtype) * target.shape[-1]
    per_mode = (per_coordinate * valid).sum(dim=(2, 3)) / counts[:, None]
    winning_modes = per_mode.detach().argmin(dim=1)
    rows = torch.arange(target.shape[0], device=target.device)
    regression = per_mode[rows, winning_modes].mean()
    classification = F.cross_entropy(logits, winning_modes)
    total = regression + classification_weight * classification
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("prediction loss contains NaN or Inf")
    return PredictionLoss(total, regression, classification, winning_modes)


class LocalityInterleaveSampler(Sampler[int]):
    """Shuffle real shards while distributing augmentation indices across an epoch."""

    def __init__(
        self,
        real_dataset: Dataset[Any],
        augmentation_count: int,
        *,
        seed: int,
        epoch: int,
    ) -> None:
        entries = getattr(real_dataset, "entries", None)
        if not isinstance(entries, Sequence):
            raise ValueError("real dataset must expose indexed entries")
        self.real_entries = entries
        self.augmentation_count = int(augmentation_count)
        self.seed = int(seed)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.real_entries) + self.augmentation_count

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        by_shard: dict[str, list[int]] = {}
        for index, entry in enumerate(self.real_entries):
            shard = entry.get("shard") if isinstance(entry, Mapping) else None
            if not isinstance(shard, str):
                raise ValueError("real dataset entry has no shard")
            by_shard.setdefault(shard, []).append(index)
        shard_names = list(by_shard)
        rng.shuffle(shard_names)
        real_indices: list[int] = []
        for shard in shard_names:
            values = by_shard[shard]
            rng.shuffle(values)
            real_indices.extend(values)
        if not self.augmentation_count:
            yield from real_indices
            return
        offset = len(real_indices)
        augmentation = list(range(offset, offset + self.augmentation_count))
        rng.shuffle(augmentation)
        slots: dict[int, list[int]] = {}
        for rank, index in enumerate(augmentation, 1):
            position = round(rank * len(real_indices) / (len(augmentation) + 1))
            slots.setdefault(position, []).append(index)
        for position in range(len(real_indices) + 1):
            yield from slots.get(position, ())
            if position < len(real_indices):
                yield real_indices[position]


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def training_fingerprints(
    *,
    experiment: str,
    seed: int,
    config: PredictionTrainConfig,
    model: nn.Module,
    data_fingerprints: Mapping[str, str],
) -> dict[str, str]:
    return {
        "experiment": _fingerprint(experiment),
        "seed": _fingerprint(seed),
        "training": _fingerprint(asdict(config)),
        "model": _fingerprint({name: list(value.shape) for name, value in model.state_dict().items()}),
        **{f"data.{key}": value for key, value in sorted(data_fingerprints.items())},
    }


def build_training_loader(
    real_dataset: Dataset[Any],
    augmentation_dataset: Dataset[Any] | None,
    *,
    config: PredictionTrainConfig,
    seed: int,
    epoch: int,
) -> DataLoader[Any]:
    dataset: Dataset[Any]
    augmentation_count = 0
    if augmentation_dataset is None:
        dataset = real_dataset
    else:
        dataset = ConcatDataset((real_dataset, augmentation_dataset))
        augmentation_count = len(augmentation_dataset)
    sampler = LocalityInterleaveSampler(
        real_dataset, augmentation_count, seed=seed, epoch=epoch
    )
    worker_args: dict[str, Any] = {}
    if config.num_workers:
        worker_args.update(
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_prediction_samples,
        drop_last=True,
        **worker_args,
    )


@torch.inference_mode()
def evaluate_prediction_model(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    amp: bool,
) -> PredictionMetricSums:
    model.eval()
    totals = PredictionMetricSums(0.0, 0.0, 0, 0)
    for batch in loader:
        tensors = {
            key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for key, value in batch.items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(prediction_model_inputs(tensors))
        totals += multimodal_prediction_sums(
            output.trajectories.float(),
            tensors["target_future"],
            tensors["target_future_mask"],
        )
    return totals


def train_prediction_model(
    *,
    model: nn.Module,
    real_dataset: Dataset[Any],
    augmentation_dataset: Dataset[Any] | None,
    validation_loader: DataLoader[Any],
    output_dir: str | Path,
    experiment: str,
    seed: int,
    config: PredictionTrainConfig,
    data_fingerprints: Mapping[str, str],
    device: str | torch.device = "cuda",
    resume: bool = True,
) -> dict[str, Any]:
    """Train one experiment with atomic latest/best checkpoints and exact resume."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = torch.device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    model.to(resolved)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    fingerprints = training_fingerprints(
        experiment=experiment,
        seed=seed,
        config=config,
        model=model,
        data_fingerprints=data_fingerprints,
    )
    latest = destination / "latest.pt"
    progress = TrainingProgress(0, 0, 0, None, None)
    if resume and latest.is_file():
        progress, _ = load_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            expected_fingerprints=fingerprints,
            map_location=resolved,
        )

    start = time.perf_counter()
    last_report = start
    regression_sum = 0.0
    classification_sum = 0.0
    measured_steps = 0
    epoch = progress.epoch
    next_batch = progress.next_batch_index
    global_step = progress.global_step
    best_metric = progress.best_metric
    best_epoch = progress.best_epoch
    last_validation: dict[str, float] | None = None
    if global_step >= config.max_steps:
        summary_path = destination / "summary.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    loader = build_training_loader(
        real_dataset,
        augmentation_dataset,
        config=config,
        seed=seed,
        epoch=epoch,
    )
    sampler = loader.sampler
    if not isinstance(sampler, LocalityInterleaveSampler):
        raise TypeError("prediction training requires LocalityInterleaveSampler")
    while global_step < config.max_steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch:
                continue
            model.train()
            tensors = {
                key: value.to(resolved, non_blocking=True) if isinstance(value, Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=resolved.type,
                dtype=torch.bfloat16,
                enabled=config.amp and resolved.type == "cuda",
            ):
                output = model(prediction_model_inputs(tensors))
                losses = prediction_loss(
                    output,
                    tensors["target_future"],
                    tensors["target_future_mask"],
                    classification_weight=config.classification_weight,
                )
            losses.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("prediction gradient contains NaN or Inf")
            optimizer.step()
            global_step += 1
            measured_steps += 1
            regression_sum += float(losses.regression.detach().cpu())
            classification_sum += float(losses.classification.detach().cpu())
            next_progress = TrainingProgress(
                epoch, batch_index + 1, global_step, best_metric, best_epoch
            )

            should_validate = (
                global_step % config.validation_every_steps == 0
                or global_step == config.max_steps
            )
            if should_validate:
                metrics = evaluate_prediction_model(
                    model, validation_loader, device=resolved, amp=config.amp
                )
                last_validation = {
                    "min_ade": metrics.min_ade,
                    "min_fde": metrics.min_fde,
                    "miss_rate": metrics.miss_rate,
                }
                if best_metric is None or metrics.min_fde < best_metric:
                    best_metric = metrics.min_fde
                    best_epoch = epoch
                    next_progress = TrainingProgress(
                        epoch, batch_index + 1, global_step, best_metric, best_epoch
                    )
                    save_checkpoint(
                        destination / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        progress=next_progress,
                        fingerprints=fingerprints,
                        extra={"validation_min_fde": metrics.min_fde},
                    )
            if (
                global_step % config.checkpoint_every_steps == 0
                or global_step == config.max_steps
            ):
                next_progress = TrainingProgress(
                    epoch, batch_index + 1, global_step, best_metric, best_epoch
                )
                save_checkpoint(
                    latest,
                    model=model,
                    optimizer=optimizer,
                    progress=next_progress,
                    fingerprints=fingerprints,
                )

            now = time.perf_counter()
            if now - last_report >= config.progress_every_seconds or global_step == config.max_steps:
                elapsed = now - start
                rate = measured_steps / elapsed if elapsed else 0.0
                eta = (config.max_steps - global_step) / rate if rate else math.inf
                memory = (
                    torch.cuda.max_memory_allocated(resolved) / (1024**3)
                    if resolved.type == "cuda"
                    else 0.0
                )
                checkpoint_label = (
                    str(latest) if latest.is_file() else "pending"
                )
                print(
                    f"\r{experiment} seed={seed} {global_step}/{config.max_steps} "
                    f"loss={float(losses.total.detach()):.4f} {rate:.2f} steps/s "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s "
                    f"vram={memory:.2f}GiB checkpoint={checkpoint_label}",
                    end="",
                    flush=True,
                )
                last_report = now
            if global_step >= config.max_steps:
                break
        else:
            epoch += 1
            next_batch = 0
            continue
        break
    print()
    elapsed = time.perf_counter() - start
    result = {
        "experiment": experiment,
        "seed": seed,
        "global_step": global_step,
        "elapsed_seconds": elapsed,
        "mean_regression_loss": regression_sum / max(measured_steps, 1),
        "mean_classification_loss": classification_sum / max(measured_steps, 1),
        "best_internal_min_fde": best_metric,
        "last_internal_metrics": last_validation,
        "fingerprints": fingerprints,
    }
    (destination / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = [
    "LocalityInterleaveSampler",
    "PredictionLoss",
    "PredictionTrainConfig",
    "build_training_loader",
    "evaluate_prediction_model",
    "prediction_loss",
    "train_prediction_model",
    "training_fingerprints",
]

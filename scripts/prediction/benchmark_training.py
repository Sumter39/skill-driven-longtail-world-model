"""Benchmark stable prediction training throughput on a fixed E3 workload."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from skilldrive.data.cvae_cache import CVAECachedDataset
from skilldrive.prediction.data import (
    PredictionAugmentationDataset,
    PredictionRealDataset,
    prediction_model_inputs,
)
from skilldrive.prediction.model import VectorTrajectoryPredictor
from skilldrive.prediction.training import (
    PredictionTrainConfig,
    build_training_loader,
    prediction_loss,
)


def _sync() -> None:
    torch.cuda.synchronize()


def _run_candidate(real, augmentation, *, batch_size: int, workers: int, warmup: int, measured: int, seed: int):
    config = PredictionTrainConfig(
        batch_size=batch_size,
        max_steps=warmup + measured,
        num_workers=workers,
        persistent_workers=bool(workers),
    )
    torch.manual_seed(seed)
    model = VectorTrajectoryPredictor().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = build_training_loader(
        real, augmentation, config=config, seed=seed, epoch=0
    )
    iterator = iter(loader)
    startup_start = time.perf_counter()
    batch = next(iterator)
    startup_seconds = time.perf_counter() - startup_start
    step_seconds = []
    data_seconds = []
    for step in range(warmup + measured):
        if step:
            data_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            data_wait = time.perf_counter() - data_start
        else:
            data_wait = startup_seconds
        batch = {
            key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        _sync()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(prediction_model_inputs(batch))
            loss = prediction_loss(
                output, batch["target_future"], batch["target_future_mask"]
            )
        loss.total.backward()
        optimizer.step()
        _sync()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            step_seconds.append(elapsed + data_wait)
            data_seconds.append(data_wait)
    median = statistics.median(step_seconds)
    return {
        "batch_size": batch_size,
        "num_workers": workers,
        "startup_seconds": startup_seconds,
        "median_step_seconds": median,
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "median_data_wait_seconds": statistics.median(data_seconds),
        "samples_per_second": batch_size / median,
        "measured_steps": measured,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-cache", type=Path, required=True)
    parser.add_argument("--augmentation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(32, 64, 96))
    parser.add_argument("--workers", nargs="+", type=int, default=(0, 2, 4))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measured", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    real = PredictionRealDataset(CVAECachedDataset(args.formal_cache, in_memory_shards=16))
    augmentation = PredictionAugmentationDataset(
        args.augmentation_root, "e3", in_memory_shards=16
    )
    results = []
    for batch_size in args.batch_sizes:
        for workers in args.workers:
            for repeat in range(args.repeats):
                result = _run_candidate(
                    real,
                    augmentation,
                    batch_size=batch_size,
                    workers=workers,
                    warmup=args.warmup,
                    measured=args.measured,
                    seed=2026 + repeat,
                )
                result["repeat"] = repeat
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    payload = {"schema_version": 1, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

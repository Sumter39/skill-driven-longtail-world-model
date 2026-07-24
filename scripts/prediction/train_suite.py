"""Train E0-E3 prediction experiments with fixed, resumable contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from skilldrive.data.cvae_cache import CVAECachedDataset
from skilldrive.prediction.audit import file_sha256
from skilldrive.prediction.data import (
    PredictionAugmentationDataset,
    PredictionRealDataset,
    collate_prediction_samples,
)
from skilldrive.prediction.model import LSTMTrajectoryPredictor, VectorTrajectoryPredictor
from skilldrive.prediction.training import (
    PredictionTrainConfig,
    train_prediction_model,
)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _cache_fingerprint(path: Path) -> str:
    return file_sha256(path / "cache_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-cache", type=Path, required=True)
    parser.add_argument("--internal-cache", type=Path, required=True)
    parser.add_argument("--augmentation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiments", nargs="+", choices=("e0", "e1", "e2", "e3"), default=("e0", "e1", "e2", "e3"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(2026, 2027, 2028))
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-every-steps", type=int, default=250)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--model", choices=("transformer", "lstm"), default="transformer")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    real = PredictionRealDataset(CVAECachedDataset(args.formal_cache, in_memory_shards=16))
    validation = PredictionRealDataset(CVAECachedDataset(args.internal_cache, in_memory_shards=8))
    validation_loader = DataLoader(
        validation,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        collate_fn=collate_prediction_samples,
        drop_last=False,
        prefetch_factor=2 if args.num_workers else None,
        persistent_workers=bool(args.num_workers),
    )
    bundle_manifest = args.augmentation_root / "manifest.json"
    data_fingerprints = {
        "formal_cache": _cache_fingerprint(args.formal_cache),
        "internal_cache": _cache_fingerprint(args.internal_cache),
        "augmentation_bundle": file_sha256(bundle_manifest),
    }
    config = PredictionTrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        persistent_workers=bool(args.num_workers),
        pin_memory=args.device == "cuda",
        validation_every_steps=args.validation_every_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for experiment in args.experiments:
        augmentation = None if experiment == "e0" else PredictionAugmentationDataset(
            args.augmentation_root, experiment, in_memory_shards=16
        )
        for seed in args.seeds:
            _seed(seed)
            model = (
                LSTMTrajectoryPredictor()
                if args.model == "lstm"
                else VectorTrajectoryPredictor()
            )
            output_dir = args.output_root / experiment / f"seed_{seed}"
            result = train_prediction_model(
                model=model,
                real_dataset=real,
                augmentation_dataset=augmentation,
                validation_loader=validation_loader,
                output_dir=output_dir,
                experiment=experiment,
                seed=seed,
                config=config,
                data_fingerprints=data_fingerprints,
                device=args.device,
                resume=not args.no_resume,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

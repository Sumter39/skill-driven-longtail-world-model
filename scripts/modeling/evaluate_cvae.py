"""Independently evaluate a saved conditional CVAE best checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from scripts.modeling.train_cvae import (
    _DatasetView,
    _base_view,
    _effective_training,
    _fingerprints,
    _loader,
    _manifest_path,
    _observed_view,
    _overfit_view,
    _stage_paths,
    _validate_cache_contract,
    _validation_seed,
    build_model_from_config,
    evaluation_to_dict,
)
from skilldrive.data import (
    CVAECachedDataset,
    build_cvae_schema,
    cvae_schema_fingerprint,
)
from skilldrive.training import DEFAULT_CVAE_CONFIG, load_checkpoint, load_cvae_config
from skilldrive.training.trainer import evaluate, move_batch_to_device


EVALUATION_STAGES = ("overfit", "development", "formal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _endpoint_diversity(futures: torch.Tensor, threshold_m: float = 0.1) -> dict[str, Any]:
    if futures.ndim != 4 or futures.shape[1] < 2 or futures.shape[-1] != 2:
        raise ValueError("prior futures must have shape [B, K>=2, T, 2]")
    endpoints = futures[:, :, -1].float()
    maximums = torch.cdist(endpoints, endpoints).amax(dim=(1, 2))
    return {
        "prior_samples": int(futures.shape[1]),
        "threshold_m": threshold_m,
        "maximum_endpoint_separation_m": float(maximums.max().cpu()),
        "conditions_above_threshold": int((maximums > threshold_m).sum().cpu()),
        "condition_count": int(futures.shape[0]),
    }


def run_evaluation(
    *,
    config_path: str | Path = DEFAULT_CVAE_CONFIG,
    stage: str,
    project_root: str | Path = ".",
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the stage best checkpoint and write an independent evaluation JSON."""
    if stage not in EVALUATION_STAGES:
        raise ValueError(f"stage must be one of {EVALUATION_STAGES}")
    root = Path(project_root)
    config = load_cvae_config(config_path)
    schema = build_cvae_schema(root / config.data.skill_dir)
    train_cache, validation_cache, output_dir = _stage_paths(
        config,
        stage,
        root,
        cache_root,
    )
    train_partition = "formal_train" if stage == "formal" else "development_train"
    validation_partition = (
        "internal_validation"
        if stage == "formal"
        else ("development_train" if stage == "overfit" else "development_validation")
    )
    raw_train_dataset = CVAECachedDataset(train_cache, schema=schema)
    raw_validation_dataset = (
        raw_train_dataset
        if validation_cache == train_cache
        else CVAECachedDataset(validation_cache, schema=schema)
    )
    schema_sha256 = cvae_schema_fingerprint(schema)
    candidate_pool_path = root / config.data.formal_candidate_pool
    _validate_cache_contract(
        raw_train_dataset,
        expected_partition=train_partition,
        manifest_path=_manifest_path(config, train_partition, root),
        schema_sha256=schema_sha256,
        candidate_pool_path=candidate_pool_path,
    )
    if raw_validation_dataset is not raw_train_dataset:
        _validate_cache_contract(
            raw_validation_dataset,
            expected_partition=validation_partition,
            manifest_path=_manifest_path(config, validation_partition, root),
            schema_sha256=schema_sha256,
            candidate_pool_path=candidate_pool_path,
        )
    run_manifest_path = output_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    stored_training = run_manifest.get("training")
    if not isinstance(stored_training, dict):
        raise ValueError("run manifest is missing the effective training contract")
    fingerprint_training, learning_rate = _effective_training(
        config,
        stage,
        int(stored_training["batch_size"]),
        int(stored_training["num_workers"]),
        bool(stored_training.get("amp", config.training.amp)),
        int(stored_training.get("prefetch_factor", config.training.prefetch_factor)),
    )
    if float(stored_training["learning_rate"]) != learning_rate:
        raise ValueError("run manifest learning rate differs from current configuration")
    expected_fingerprints = _fingerprints(
        config=config,
        schema=schema,
        stage=stage,
        training=fingerprint_training,
        learning_rate=learning_rate,
        train_cache=train_cache,
        validation_cache=validation_cache,
    )
    if run_manifest.get("fingerprints") != expected_fingerprints:
        raise ValueError("run manifest fingerprint mismatch")
    if run_manifest.get("validation_seed") != _validation_seed(fingerprint_training):
        raise ValueError("run manifest validation seed differs from current contract")

    evaluation_training = replace(
        fingerprint_training,
        batch_size=(
            fingerprint_training.batch_size if batch_size is None else batch_size
        ),
        num_workers=(
            fingerprint_training.num_workers if num_workers is None else num_workers
        ),
    )
    if evaluation_training.batch_size <= 0 or evaluation_training.num_workers < 0:
        raise ValueError("evaluation batch_size must be positive and num_workers nonnegative")
    if evaluation_training.num_workers == 0:
        evaluation_training = replace(evaluation_training, persistent_workers=False)
    device = torch.device(evaluation_training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device=cuda but CUDA is unavailable")

    checkpoint = (
        output_dir / "best.pt"
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    model = build_model_from_config(config, schema).to(device)
    progress, extra = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_fingerprints=expected_fingerprints,
        map_location=device,
        restore_rng=False,
    )
    if extra.get("stage") != stage:
        raise ValueError("checkpoint stage differs from requested evaluation stage")
    def evaluate_dataset(dataset, *, seed: int) -> dict[str, Any]:
        loader: DataLoader = _loader(
            dataset,
            training=evaluation_training,
            sampler=None,
            generator=torch.Generator().manual_seed(seed + 1),
        )
        result = evaluate(
            model,
            loader,
            device=device,
            prior_samples=evaluation_training.prior_samples,
            sample_period_s=config.tensorization.sample_period_s,
            evaluation_seed=seed,
            amp=evaluation_training.amp,
        )
        return evaluation_to_dict(result, evaluation_training.prior_samples)

    validation_seed = _validation_seed(evaluation_training)
    if stage == "overfit":
        overfit_dataset = _overfit_view(raw_train_dataset, config)
        metrics = {
            "overfit_training_subset": evaluate_dataset(
                overfit_dataset,
                seed=validation_seed,
            )
        }
        diversity_loader = _loader(
            overfit_dataset,
            training=evaluation_training,
            sampler=None,
            generator=torch.Generator().manual_seed(validation_seed + 2),
        )
        batch = move_batch_to_device(next(iter(diversity_loader)), device)
        model.eval()
        with torch.no_grad():
            prior = model.sample_prior(
                batch,
                8,
                torch.Generator(device=device).manual_seed(validation_seed + 3),
            )
        metrics["prior_endpoint_diversity"] = _endpoint_diversity(
            prior.future_position_local
        )
    else:
        base = _base_view(raw_validation_dataset)
        observed = _observed_view(raw_validation_dataset)
        metrics = {
            "base_no_skill": evaluate_dataset(base, seed=validation_seed),
            "observed_skill": None,
            "observed_by_skill": {},
        }
        if len(observed):
            metrics["observed_skill"] = evaluate_dataset(
                observed,
                seed=validation_seed + 1,
            )
            counts = Counter(entry["spec"]["skill_id"] for entry in observed.entries)
            by_skill: dict[str, Any] = {}
            for offset, skill_id in enumerate(sorted(counts), start=2):
                indices = [
                    index
                    for index, entry in enumerate(observed.entries)
                    if entry["spec"]["skill_id"] == skill_id
                ]
                view = _DatasetView(observed, indices)
                by_skill[skill_id] = {
                    "sample_count": counts[skill_id],
                    "metrics": evaluate_dataset(
                        view,
                        seed=validation_seed + offset,
                    ),
                }
            metrics["observed_by_skill"] = by_skill
    summary = {
        "stage": stage,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "epoch": progress.epoch,
            "global_step": progress.global_step,
            "best_metric": progress.best_metric,
            "best_epoch": progress.best_epoch,
        },
        "fingerprints": expected_fingerprints,
        "validation_seed": validation_seed,
        "metrics": metrics,
    }
    destination = (
        output_dir / "evaluation.json"
        if output_path is None
        else Path(output_path)
    )
    if not destination.is_absolute():
        destination = root / destination
    _atomic_json(destination, summary)
    summary["output_path"] = str(destination)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an independently loaded conditional CVAE best checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CVAE_CONFIG)
    parser.add_argument("--stage", choices=EVALUATION_STAGES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cache-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_evaluation(
        config_path=args.config,
        stage=args.stage,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_root=args.cache_root,
    )
    group = (
        "overfit_training_subset" if args.stage == "overfit" else "base_no_skill"
    )
    metrics = summary["metrics"][group]["prior"]
    print(
        "CVAE evaluation complete: "
        f"minADE={metrics['min_ade']:.6f}, minFDE={metrics['min_fde']:.6f}"
    )


if __name__ == "__main__":
    main()

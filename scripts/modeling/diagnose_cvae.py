"""Render reproducible trajectory diagnostics from a formal CVAE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import Tensor

from scripts.modeling.train_cvae import (
    _cache_fingerprint,
    _manifest_path,
    _validate_cache_contract,
    build_model_from_config,
    model_kwargs_from_config,
)
from skilldrive.data import (
    CVAECachedDataset,
    build_cvae_schema,
    cvae_schema_fingerprint,
)
from skilldrive.models import CVAEOutput, ConditionalCVAE
from skilldrive.training import DEFAULT_CVAE_CONFIG, load_checkpoint, load_cvae_config
from skilldrive.training.trainer import move_batch_to_device


MAX_DIAGNOSTIC_SAMPLES = 16
DEFAULT_PRIOR_SAMPLES = 6


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


def _stable_rank(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{value}".encode("utf-8")).hexdigest()


def _entry_spec(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = entry.get("spec")
    if not isinstance(spec, Mapping):
        raise ValueError("CVAE sample index entry is missing its sample spec")
    return spec


def select_diagnostic_indices(
    entries: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> list[int]:
    """Select a fixed, balanced set of base and observed validation samples."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise ValueError("sample_count must be an integer")
    if not 1 <= sample_count <= MAX_DIAGNOSTIC_SAMPLES:
        raise ValueError(
            f"sample_count must be between 1 and {MAX_DIAGNOSTIC_SAMPLES}"
        )
    if len(entries) < sample_count:
        raise ValueError(
            f"internal validation contains only {len(entries)} samples; "
            f"cannot select {sample_count}"
        )

    ranked: dict[bool, list[int]] = {False: [], True: []}
    for index, entry in enumerate(entries):
        sample_id = entry.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("CVAE sample index entry has an invalid sample_id")
        spec = _entry_spec(entry)
        observed = spec.get("skill_supervision_mask") is True
        ranked[observed].append(index)
    for observed, indices in ranked.items():
        indices.sort(
            key=lambda index: (
                _stable_rank(
                    seed,
                    "observed" if observed else "base",
                    str(entries[index]["sample_id"]),
                ),
                str(entries[index]["sample_id"]),
            )
        )

    observed_count = sample_count // 2 if ranked[False] and ranked[True] else 0
    observed_count = min(observed_count, len(ranked[True]))
    base_count = min(sample_count - observed_count, len(ranked[False]))
    observed_count = min(sample_count - base_count, len(ranked[True]))
    selected = ranked[False][:base_count] + ranked[True][:observed_count]
    if len(selected) != sample_count:
        selected_set = set(selected)
        fill = sorted(
            (index for index in range(len(entries)) if index not in selected_set),
            key=lambda index: (
                _stable_rank(seed, "fill", str(entries[index]["sample_id"])),
                str(entries[index]["sample_id"]),
            ),
        )
        selected.extend(fill[: sample_count - len(selected)])
    if len(selected) != sample_count:
        raise RuntimeError("failed to select the requested diagnostic samples")
    return selected


def _generator_seed(seed: int, sample_id: str, stream: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _batched_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.unsqueeze(0) if isinstance(value, Tensor) else value
        for name, value in sample.items()
    }


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _trajectory_metrics(
    posterior: Tensor,
    prior: Tensor,
    target: Tensor,
    mask: Tensor,
) -> dict[str, Any]:
    valid = mask.to(dtype=torch.bool)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not len(valid_indices):
        raise ValueError("diagnostic sample has no valid target future")
    posterior_errors = torch.linalg.vector_norm(posterior - target, dim=-1)
    prior_errors = torch.linalg.vector_norm(prior - target.unsqueeze(0), dim=-1)
    posterior_ade = posterior_errors[valid].mean()
    posterior_fde = posterior_errors[valid_indices[-1]]
    prior_ades = prior_errors[:, valid].mean(dim=1)
    prior_fdes = prior_errors[:, valid_indices[-1]]
    endpoints = prior[:, valid_indices[-1]]
    endpoint_separation = (
        torch.cdist(endpoints, endpoints).amax()
        if prior.shape[0] > 1
        else prior.new_zeros(())
    )
    return {
        "posterior_ade_m": float(posterior_ade.cpu()),
        "posterior_fde_m": float(posterior_fde.cpu()),
        "prior_min_ade_m": float(prior_ades.min().cpu()),
        "prior_min_fde_m": float(prior_fdes.min().cpu()),
        "prior_max_endpoint_separation_m": float(endpoint_separation.cpu()),
    }


def infer_diagnostic_record(
    model: ConditionalCVAE,
    sample: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    device: torch.device,
    prior_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run posterior and prior inference for one cached sample."""

    if isinstance(prior_samples, bool) or not isinstance(prior_samples, int):
        raise ValueError("prior_samples must be an integer")
    if prior_samples < 2:
        raise ValueError("prior_samples must be at least 2 for diversity diagnostics")
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str) or sample_id != entry.get("sample_id"):
        raise ValueError("diagnostic sample differs from its index entry")
    batch = move_batch_to_device(_batched_sample(sample), device)
    posterior_seed = _generator_seed(seed, sample_id, "posterior")
    prior_seed = _generator_seed(seed, sample_id, "prior")
    posterior_generator = torch.Generator(device=device).manual_seed(posterior_seed)
    prior_generator = torch.Generator(device=device).manual_seed(prior_seed)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            posterior_output: CVAEOutput = model.forward_train(
                batch,
                posterior_generator,
            )
            prior_output: CVAEOutput = model.sample_prior(
                batch,
                prior_samples,
                prior_generator,
            )
    finally:
        model.train(was_training)

    posterior = posterior_output.future_position_local[0].float().cpu()
    prior = prior_output.future_position_local[0].float().cpu()
    target = sample["target_future"].float().cpu()
    mask = sample["target_future_mask"].bool().cpu()
    if posterior.shape != target.shape:
        raise ValueError("posterior trajectory shape differs from target future")
    if prior.shape != (prior_samples, *target.shape):
        raise ValueError("prior trajectory shape differs from the diagnostic contract")

    spec = _entry_spec(entry)
    record = {
        "sample_id": sample_id,
        "scenario_id": str(sample["scenario_id"]),
        "target_track_id": str(sample["target_track_id"]),
        "condition": {
            "skill_id": str(spec.get("skill_id")),
            "skill_supervision_mask": bool(spec.get("skill_supervision_mask")),
        },
        "anchor": {
            "origin_global_xy": sample["anchor_origin_global"].float().tolist(),
            "heading_global_rad": float(sample["anchor_heading_global"]),
        },
        "inference": {
            "posterior": {
                "method": "ConditionalCVAE.forward_train",
                "generator_seed": posterior_seed,
                "trajectory_sha256": _tensor_sha256(posterior),
                "latent_sha256": _tensor_sha256(posterior_output.latent[0]),
            },
            "prior": {
                "method": "ConditionalCVAE.sample_prior",
                "generator_seed": prior_seed,
                "sample_count": prior_samples,
                "trajectory_sha256": _tensor_sha256(prior),
                "latent_sha256": _tensor_sha256(prior_output.latent[0]),
            },
        },
        "metrics": _trajectory_metrics(posterior, prior, target, mask),
        "trajectories_local_xy": {
            "target_future": target.tolist(),
            "target_future_mask": mask.tolist(),
            "posterior_reconstruction": posterior.tolist(),
            "prior_samples": prior.tolist(),
        },
    }
    return record


def render_diagnostic_plot(
    sample: Mapping[str, Any],
    record: Mapping[str, Any],
    path: Path,
    *,
    checkpoint_sha256: str,
) -> None:
    """Render one local-frame BEV using the exact trajectories stored in the record."""

    trajectories = record["trajectories_local_xy"]
    target = np.asarray(trajectories["target_future"], dtype=np.float32)
    target_mask = np.asarray(trajectories["target_future_mask"], dtype=bool)
    posterior = np.asarray(
        trajectories["posterior_reconstruction"], dtype=np.float32
    )
    priors = np.asarray(trajectories["prior_samples"], dtype=np.float32)

    figure, axis = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
    map_polylines = sample["map_polylines"].numpy()
    map_point_mask = sample["map_point_mask"].numpy()
    map_polyline_mask = sample["map_polyline_mask"].numpy()
    for polyline, point_mask, present in zip(
        map_polylines,
        map_point_mask,
        map_polyline_mask,
    ):
        valid = point_mask & bool(present)
        if valid.any():
            axis.plot(
                polyline[valid, 0],
                polyline[valid, 1],
                color="#C8CDD3",
                linewidth=0.8,
                zorder=1,
            )

    actor_history = sample["actor_history"].numpy()
    actor_time_mask = sample["actor_time_mask"].numpy()
    actor_mask = sample["actor_mask"].numpy()
    target_actor_index = int(sample["target_actor_index"])
    for index, (history, time_mask, present) in enumerate(
        zip(actor_history, actor_time_mask, actor_mask)
    ):
        valid = time_mask & bool(present)
        if not valid.any():
            continue
        is_target = index == target_actor_index
        axis.plot(
            history[valid, 0],
            history[valid, 1],
            color="#20252B" if is_target else "#97A6B5",
            linewidth=2.2 if is_target else 0.9,
            alpha=1.0 if is_target else 0.65,
            label="target history" if is_target else None,
            zorder=3 if is_target else 2,
        )

    for index, prior in enumerate(priors):
        axis.plot(
            prior[:, 0],
            prior[:, 1],
            color="#3977D6",
            linewidth=1.2,
            alpha=0.55,
            label="prior samples" if index == 0 else None,
            zorder=4,
        )
    axis.plot(
        posterior[:, 0],
        posterior[:, 1],
        color="#E67E22",
        linewidth=2.2,
        label="posterior reconstruction",
        zorder=5,
    )
    axis.plot(
        target[target_mask, 0],
        target[target_mask, 1],
        color="#159947",
        linewidth=2.6,
        label="true future",
        zorder=6,
    )
    axis.scatter([0.0], [0.0], marker="x", s=40, color="#20252B", zorder=7)
    condition = record["condition"]
    axis.set_title(
        f"{record['scenario_id']} | {condition['skill_id']}\n"
        f"checkpoint {checkpoint_sha256[:12]} | sample {record['sample_id'][:12]}"
    )
    axis.set_xlabel("local x (m)")
    axis.set_ylabel("local y (m)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_diagnostics(
    *,
    config_path: str | Path = DEFAULT_CVAE_CONFIG,
    project_root: str | Path = ".",
    checkpoint_path: str | Path | None = None,
    cache_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    sample_count: int = MAX_DIAGNOSTIC_SAMPLES,
    prior_samples: int = DEFAULT_PRIOR_SAMPLES,
    seed: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Generate fixed diagnostics from formal best weights and internal validation."""

    root = Path(project_root)
    config = load_cvae_config(config_path)
    schema = build_cvae_schema(root / config.data.skill_dir)
    resolved_cache_root = (
        root / config.cache.root
        if cache_root is None
        else _resolve_path(root, cache_root)
    )
    validation_cache = resolved_cache_root / "internal_validation"
    dataset = CVAECachedDataset(validation_cache, schema=schema)
    _validate_cache_contract(
        dataset,
        expected_partition="internal_validation",
        manifest_path=_manifest_path(config, "internal_validation", root),
        schema_sha256=cvae_schema_fingerprint(schema),
        candidate_pool_path=root / config.data.formal_candidate_pool,
    )
    if dataset.cache_manifest.get("partition") != "internal_validation":
        raise ValueError("trajectory diagnostics may only read internal_validation")

    formal_output = root / config.outputs.formal
    checkpoint = (
        formal_output / "best.pt"
        if checkpoint_path is None
        else _resolve_path(root, checkpoint_path)
    )
    run_manifest_path = formal_output / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"formal run manifest not found: {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("stage") != "formal":
        raise ValueError("trajectory diagnostics require a formal run manifest")
    if run_manifest.get("validation_partition") != "internal_validation":
        raise ValueError("formal run manifest does not use internal_validation")
    expected_model = model_kwargs_from_config(config, schema)
    if run_manifest.get("model") != expected_model:
        raise ValueError("formal run manifest model differs from current configuration")
    fingerprints = run_manifest.get("fingerprints")
    if not isinstance(fingerprints, dict):
        raise ValueError("formal run manifest is missing fingerprints")
    if fingerprints.get("validation_cache") != _cache_fingerprint(validation_cache):
        raise ValueError("internal validation cache differs from the formal run manifest")

    training = run_manifest.get("training")
    if not isinstance(training, dict):
        raise ValueError("formal run manifest is missing the training contract")
    resolved_device = torch.device(str(training.get("device")) if device is None else device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("diagnostic device is CUDA but CUDA is unavailable")
    model = build_model_from_config(config, schema).to(resolved_device)
    progress, extra = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=None,
        expected_fingerprints=fingerprints,
        map_location=resolved_device,
        restore_rng=False,
    )
    if extra.get("stage") != "formal":
        raise ValueError("checkpoint is not a formal training checkpoint")

    selection_seed = int(run_manifest["validation_seed"] if seed is None else seed)
    indices = select_diagnostic_indices(
        dataset.entries,
        sample_count=sample_count,
        seed=selection_seed,
    )
    checkpoint_sha256 = _sha256(checkpoint)
    destination = (
        formal_output / "diagnostics"
        if output_dir is None
        else _resolve_path(root, output_dir)
    )
    figures = destination / "figures"
    records: list[dict[str, Any]] = []
    for order, index in enumerate(indices):
        entry = dataset.entries[index]
        sample = dataset[index]
        record = infer_diagnostic_record(
            model,
            sample,
            entry,
            device=resolved_device,
            prior_samples=prior_samples,
            seed=selection_seed,
        )
        figure_name = f"{order:02d}-{record['sample_id'][:16]}.png"
        figure_path = figures / figure_name
        render_diagnostic_plot(
            sample,
            record,
            figure_path,
            checkpoint_sha256=checkpoint_sha256,
        )
        record["figure"] = {
            "path": str(figure_path),
            "sha256": _sha256(figure_path),
        }
        records.append(record)

    summary = {
        "schema_version": 1,
        "source": {
            "partition": "internal_validation",
            "cache_dir": str(validation_cache),
            "cache_fingerprint": _cache_fingerprint(validation_cache),
            "final_validation_accessed": False,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "epoch": progress.epoch,
            "global_step": progress.global_step,
            "best_metric": progress.best_metric,
            "best_epoch": progress.best_epoch,
        },
        "selection": {
            "strategy": "sha256_seeded_balanced_base_observed",
            "seed": selection_seed,
            "sample_count": len(records),
            "maximum_allowed": MAX_DIAGNOSTIC_SAMPLES,
            "prior_samples_per_condition": prior_samples,
            "sample_ids": [record["sample_id"] for record in records],
        },
        "model_inference": {
            "device": str(resolved_device),
            "posterior_method": "ConditionalCVAE.forward_train",
            "prior_method": "ConditionalCVAE.sample_prior",
        },
        "records": records,
    }
    manifest_path = destination / "diagnostics.json"
    _atomic_json(manifest_path, summary)
    summary["output_path"] = str(manifest_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render fixed internal-validation trajectories from a formal CVAE checkpoint."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CVAE_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-count", type=int, default=MAX_DIAGNOSTIC_SAMPLES)
    parser.add_argument("--prior-samples", type=int, default=DEFAULT_PRIOR_SAMPLES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_diagnostics(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        prior_samples=args.prior_samples,
        seed=args.seed,
        device=args.device,
    )
    print(
        "CVAE trajectory diagnostics complete: "
        f"{summary['selection']['sample_count']} samples -> {summary['output_path']}"
    )


if __name__ == "__main__":
    main()

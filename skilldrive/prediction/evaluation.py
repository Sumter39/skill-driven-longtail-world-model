"""Per-sample trajectory-prediction evaluation and paired uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


METRIC_NAMES = ("min_ade", "min_fde", "miss")


def per_sample_prediction_errors(
    predictions: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    miss_threshold_m: float = 2.0,
) -> dict[str, Tensor]:
    """Return best-of-K errors for every row, respecting each future mask."""

    if predictions.ndim != 4 or target.ndim != 3:
        raise ValueError("predictions and target must have ranks 4 and 3")
    if predictions.shape[0] != target.shape[0] or predictions.shape[2:] != target.shape[1:]:
        raise ValueError("prediction and target shapes do not align")
    if mask.dtype is not torch.bool or mask.shape != target.shape[:2]:
        raise ValueError("mask must be boolean with shape [B, T]")
    if miss_threshold_m <= 0 or not bool(mask.any(dim=1).all()):
        raise ValueError("every sample needs future points and a positive miss threshold")
    distances = torch.linalg.vector_norm(predictions - target[:, None], dim=-1)
    counts = mask.sum(dim=1).clamp_min(1)
    ade = (distances * mask[:, None]).sum(dim=-1) / counts[:, None]
    time = torch.arange(mask.shape[1], device=mask.device)
    last = torch.where(mask, time[None], -1).max(dim=1).values
    rows = torch.arange(target.shape[0], device=target.device)
    modes = torch.arange(predictions.shape[1], device=target.device)
    fde = distances[rows[:, None], modes[None], last[:, None]]
    min_ade = ade.min(dim=1).values
    min_fde = fde.min(dim=1).values
    return {
        "min_ade": min_ade,
        "min_fde": min_fde,
        "miss": (min_fde > miss_threshold_m).to(dtype=torch.float32),
    }


def summarize_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Summarize JSON-compatible per-sample metric rows."""

    if not rows:
        return {"sample_count": 0, "min_ade": float("nan"), "min_fde": float("nan"), "miss_rate": float("nan")}
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for row in rows:
        for name in METRIC_NAMES:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"prediction row has invalid {name}")
            values[name].append(float(value))
    return {
        "sample_count": len(rows),
        "min_ade": float(np.mean(values["min_ade"])),
        "min_fde": float(np.mean(values["min_fde"])),
        "miss_rate": float(np.mean(values["miss"])),
    }


def paired_bootstrap_delta(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    repetitions: int = 2_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Estimate candidate-minus-baseline CIs after grouping repeated labels by scenario."""

    if repetitions < 100:
        raise ValueError("paired bootstrap needs at least 100 repetitions")

    def grouped(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
        accumulators: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {name: [] for name in METRIC_NAMES}
        )
        for row in rows:
            scenario_id = row.get("scenario_id")
            if not isinstance(scenario_id, str) or not scenario_id:
                raise ValueError("prediction row has no scenario_id")
            for name in METRIC_NAMES:
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"prediction row has invalid {name}")
                accumulators[scenario_id][name].append(float(value))
        return {
            scenario_id: {name: float(np.mean(values[name])) for name in METRIC_NAMES}
            for scenario_id, values in accumulators.items()
        }

    baseline = grouped(baseline_rows)
    candidate = grouped(candidate_rows)
    if set(baseline) != set(candidate) or not baseline:
        raise ValueError("paired bootstrap requires identical, non-empty scenario IDs")
    ids = sorted(baseline)
    differences = np.asarray(
        [
            [candidate[scenario_id][name] - baseline[scenario_id][name] for name in METRIC_NAMES]
            for scenario_id in ids
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(ids), size=(repetitions, len(ids)))
    estimates = differences[draws].mean(axis=1)
    means = differences.mean(axis=0)
    lower = np.quantile(estimates, 0.025, axis=0)
    upper = np.quantile(estimates, 0.975, axis=0)
    labels = {"min_ade": "minADE", "min_fde": "minFDE", "miss": "Miss Rate"}
    return {
        "scenario_count": len(ids),
        "repetitions": repetitions,
        "seed": seed,
        "delta_definition": "candidate_minus_baseline; negative is better",
        "metrics": {
            labels[name]: {
                "mean_delta": float(means[index]),
                "ci95": [float(lower[index]), float(upper[index])],
            }
            for index, name in enumerate(METRIC_NAMES)
        },
    }


__all__ = [
    "METRIC_NAMES",
    "paired_bootstrap_delta",
    "per_sample_prediction_errors",
    "summarize_prediction_rows",
]

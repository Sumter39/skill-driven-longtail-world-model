"""Audit all frozen downstream-prediction training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from skilldrive.prediction.audit import file_sha256
from skilldrive.training.checkpoint import read_checkpoint_metadata


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/prediction/formal_v1.json")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("manifests/prediction/formal_contract_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/prediction/formal_run_audit_v1.json"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    experiments = list(config["experiments"]["names"])
    seeds = list(config["experiments"]["seeds"])
    max_steps = int(config["training"]["max_steps"])
    validation_interval = int(config["training"]["validation_every_steps"])
    runs: list[dict[str, Any]] = []
    common_fingerprints: dict[str, set[str]] = {}
    for experiment in experiments:
        for seed in seeds:
            directory = args.run_root / experiment / f"seed_{seed}"
            summary_path = directory / "summary.json"
            latest_path = directory / "latest.pt"
            best_path = directory / "best.pt"
            if not all(path.is_file() for path in (summary_path, latest_path, best_path)):
                raise FileNotFoundError(f"formal run is incomplete: {directory}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("experiment") != experiment
                or summary.get("seed") != seed
                or summary.get("global_step") != max_steps
            ):
                raise ValueError(f"formal summary contract differs: {summary_path}")
            expected_fingerprints = summary.get("fingerprints")
            latest = read_checkpoint_metadata(latest_path)
            best = read_checkpoint_metadata(best_path)
            if dict(latest.fingerprints) != expected_fingerprints or dict(best.fingerprints) != expected_fingerprints:
                raise ValueError(f"checkpoint fingerprints differ from summary: {directory}")
            if latest.progress.global_step != max_steps:
                raise ValueError(f"latest checkpoint has the wrong step: {latest_path}")
            best_step = best.progress.global_step
            if best_step < validation_interval or best_step > max_steps or best_step % validation_interval:
                raise ValueError(f"best checkpoint is not a validation step: {best_path}")
            best_metric = best.extra.get("validation_min_fde")
            if not isinstance(best_metric, (int, float)) or abs(
                float(best_metric) - float(summary["best_internal_min_fde"])
            ) > 1e-9:
                raise ValueError(f"best checkpoint metric differs: {best_path}")
            for name in (
                "data.augmentation_bundle",
                "data.formal_cache",
                "data.internal_cache",
                "model",
                "training",
            ):
                common_fingerprints.setdefault(name, set()).add(expected_fingerprints[name])
            runs.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "global_step": max_steps,
                    "best_step": best_step,
                    "best_internal_min_fde": float(best_metric),
                    "summary_sha256": file_sha256(summary_path),
                    "latest_sha256": file_sha256(latest_path),
                    "best_sha256": file_sha256(best_path),
                    "fingerprints": expected_fingerprints,
                }
            )
    inconsistent = {name: sorted(values) for name, values in common_fingerprints.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"formal groups do not share frozen fingerprints: {inconsistent}")
    payload = {
        "schema_version": 1,
        "kind": "downstream_prediction_formal_run_audit",
        "status": "complete",
        "contract_id": contract["contract_id"],
        "run_count": len(runs),
        "expected_run_count": len(experiments) * len(seeds),
        "max_steps": max_steps,
        "common_fingerprints": {name: next(iter(values)) for name, values in common_fingerprints.items()},
        "runs": runs,
        "final_validation_content_accessed_before_training_completion": False,
    }
    _atomic_json(args.output, payload)
    print(f"formal run audit complete: {len(runs)}/{payload['expected_run_count']} runs")


if __name__ == "__main__":
    main()

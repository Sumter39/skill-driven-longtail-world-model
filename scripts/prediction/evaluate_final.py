"""Run the frozen one-time Final Validation trajectory-prediction evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from skilldrive.data.cvae_cache import CVAECachedDataset
from skilldrive.data.cvae_samples import build_cvae_schema
from skilldrive.prediction.audit import file_sha256
from skilldrive.prediction.data import (
    PredictionRealDataset,
    collate_prediction_samples,
    prediction_model_inputs,
)
from skilldrive.prediction.evaluation import (
    paired_bootstrap_delta,
    per_sample_prediction_errors,
    summarize_prediction_rows,
)
from skilldrive.prediction.metrics import constant_velocity_prediction
from skilldrive.prediction.model import LSTMTrajectoryPredictor, VectorTrajectoryPredictor
from skilldrive.seeds.records import SeedRecord, iter_seed_records


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _checkpoint_model(path: Path, model: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError(f"invalid predictor checkpoint: {path}")
    model.load_state_dict(payload["model"])


def _record_target(record: SeedRecord) -> str:
    if record.skill_id == "short_headway_following":
        target = record.role_track_ids.get("close_follower")
        if not target:
            raise ValueError("short_headway_following record has no close_follower")
        return target
    return record.initiator_track_id


def _label_index(path: Path) -> dict[tuple[str, str, str, str | None], SeedRecord]:
    grouped: dict[tuple[str, str, str, str | None], list[SeedRecord]] = defaultdict(list)
    for record in iter_seed_records(path):
        if record.evidence.get("detection_mode") != "observed_trigger":
            raise ValueError("Final Validation label CSV contains a non-observed record")
        grouped[(record.scenario_id, record.skill_id, _record_target(record), record.responder_track_id)].append(record)
    return {
        key: min(records, key=lambda record: (-record.trigger_score, record.unique_key))
        for key, records in grouped.items()
    }


def _risk_fields(record: SeedRecord) -> dict[str, Any]:
    definition = record.target_risk_definition
    matched = record.seed_risk_metric == definition["metric"]
    severity = None
    stratum = "proxy_metric"
    if matched:
        low, high = map(float, definition["target_range"])
        if high > low:
            fraction = (record.seed_risk_value - low) / (high - low)
            if definition["direction"] == "lower_is_riskier":
                fraction = 1.0 - fraction
            severity = float(np.clip(fraction, 0.0, 1.0))
            stratum = "high" if severity >= 2 / 3 else "medium" if severity >= 1 / 3 else "low"
        else:
            stratum = "degenerate_target_range"
    return {
        "seed_risk_metric": record.seed_risk_metric,
        "seed_risk_value": record.seed_risk_value,
        "target_risk_metric": definition["metric"],
        "risk_metric_matches_target": matched,
        "normalized_risk_severity": severity,
        "risk_stratum": stratum,
    }


def _metadata_by_sample(
    dataset: CVAECachedDataset, labels: Mapping[tuple[str, str, str, str | None], SeedRecord]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in dataset.entries:
        spec = entry["spec"]
        skill_id = spec["skill_id"]
        metadata: dict[str, Any] = {
            "sample_id": entry["sample_id"],
            "scenario_id": entry["scenario_id"],
            "target_track_id": entry["target_track_id"],
            "skill_id": skill_id,
            "is_long_tail": bool(spec["skill_supervision_mask"]),
        }
        if metadata["is_long_tail"]:
            key = (
                entry["scenario_id"],
                skill_id,
                entry["target_track_id"],
                spec.get("responder_track_id"),
            )
            record = labels.get(key)
            if record is None:
                raise ValueError(f"Final Validation cache label has no risk record: {key}")
            metadata.update(_risk_fields(record))
        result[entry["sample_id"]] = metadata
    return result


@torch.inference_mode()
def _evaluate(
    *,
    loader: DataLoader[Any],
    metadata: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    model: torch.nn.Module | None,
    amp: bool,
) -> list[dict[str, Any]]:
    if model is not None:
        model.to(device).eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        tensors = {
            name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for name, value in batch.items()
        }
        if model is None:
            predictions = constant_velocity_prediction(
                tensors["actor_history"],
                tensors["actor_time_mask"],
                tensors["actor_mask"],
                tensors["target_actor_index"],
            )
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                predictions = model(prediction_model_inputs(tensors)).trajectories
        errors = per_sample_prediction_errors(
            predictions.float(), tensors["target_future"], tensors["target_future_mask"]
        )
        for index, item in enumerate(batch["metadata"]):
            sample_id = item["sample_id"]
            row = dict(metadata[sample_id])
            row.update(
                min_ade=float(errors["min_ade"][index].cpu()),
                min_fde=float(errors["min_fde"][index].cpu()),
                miss=float(errors["miss"][index].cpu()),
            )
            rows.append(row)
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    value = summarize_prediction_rows(rows)
    return {name: (None if isinstance(item, float) and not math.isfinite(item) else item) for name, item in value.items()}


def _views(rows: Sequence[Mapping[str, Any]], skill_ids: Sequence[str]) -> dict[str, Any]:
    overall = [row for row in rows if not row["is_long_tail"]]
    long_tail = [row for row in rows if row["is_long_tail"]]
    per_skill = {
        skill_id: _summary([row for row in long_tail if row["skill_id"] == skill_id])
        for skill_id in skill_ids
    }
    strata = {
        name: _summary([row for row in long_tail if row.get("risk_stratum") == name])
        for name in ("low", "medium", "high", "proxy_metric", "degenerate_target_range")
    }
    return {
        "overall": _summary(overall),
        "real_long_tail": _summary(long_tail),
        "per_skill": per_skill,
        "risk_strata": strata,
    }


def _average_seed_rows(rows_by_seed: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    indexed = [{row["sample_id"]: row for row in rows} for rows in rows_by_seed]
    ids = set(indexed[0])
    if any(set(value) != ids for value in indexed[1:]):
        raise ValueError("three-seed evaluation rows do not contain identical samples")
    output: list[dict[str, Any]] = []
    for sample_id in sorted(ids):
        row = dict(indexed[0][sample_id])
        for name in ("min_ade", "min_fde", "miss"):
            row[name] = float(np.mean([value[sample_id][name] for value in indexed]))
        output.append(row)
    return output


def _seed_statistics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"seed_count": len(values)}
    for view in ("overall", "real_long_tail"):
        result[view] = {}
        for metric in ("min_ade", "min_fde", "miss_rate"):
            items = [float(value[view][metric]) for value in values]
            result[view][metric] = {
                "mean": float(np.mean(items)),
                "std": float(np.std(items, ddof=1)),
                "by_seed": items,
            }
    return result


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("min_ade", "min_fde", "miss_rate"):
        candidate_value = float(candidate[name])
        baseline_value = float(baseline[name])
        absolute = candidate_value - baseline_value
        result[name] = {
            "candidate_minus_baseline": absolute,
            "relative_percent": 100.0 * absolute / baseline_value if baseline_value else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-cache", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--lstm-checkpoint", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument(
        "--formal-audit", type=Path, default=Path("manifests/prediction/formal_run_audit_v1.json")
    )
    parser.add_argument(
        "--augmentation-manifest", type=Path, default=Path("manifests/prediction/augmentation_bundle_v1.json")
    )
    parser.add_argument(
        "--summary-output", type=Path, default=Path("manifests/prediction/final_evaluation_v1.json")
    )
    parser.add_argument("--batch-size", type=int, default=288)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2_000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    audit = json.loads(args.formal_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("run_count") != 12:
        raise ValueError("formal run audit is incomplete")
    cache_manifest = args.final_cache / "cache_manifest.json"
    cache_data = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cache_data.get("status") != "complete" or cache_data.get("partition") != "final_validation":
        raise ValueError("Final Validation cache is incomplete")
    dataset = CVAECachedDataset(args.final_cache, in_memory_shards=16)
    real = PredictionRealDataset(dataset)
    labels = _label_index(args.label_csv)
    metadata = _metadata_by_sample(dataset, labels)
    loader = DataLoader(
        real,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prediction_samples,
        persistent_workers=bool(args.num_workers),
        prefetch_factor=2 if args.num_workers else None,
    )
    checkpoints: list[tuple[str, str, int | None, Path | None]] = [("constant_velocity", "constant_velocity", None, None)]
    checkpoints.append(("lstm_e0_seed_2026", "lstm", 2026, args.lstm_checkpoint))
    for experiment in ("e0", "e1", "e2", "e3"):
        for seed in (2026, 2027, 2028):
            checkpoints.append(
                (f"transformer_{experiment}_seed_{seed}", experiment, seed, args.formal_root / experiment / f"seed_{seed}" / "best.pt")
            )
    checkpoint_hashes = {name: (None if path is None else file_sha256(path)) for name, _, _, path in checkpoints}
    identity = _canonical_sha(
        {
            "schema_version": 1,
            "formal_audit": file_sha256(args.formal_audit),
            "final_cache": file_sha256(cache_manifest),
            "label_csv": file_sha256(args.label_csv),
            "checkpoints": checkpoint_hashes,
            "batch_size": args.batch_size,
        }
    )
    args.runtime_output.mkdir(parents=True, exist_ok=True)
    state_path = args.runtime_output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"identity": identity, "completed": {}}
    if state.get("identity") != identity:
        raise ValueError("Final Validation evaluation state identity differs")
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for position, (name, kind, seed, checkpoint) in enumerate(checkpoints, 1):
        output_path = args.runtime_output / f"{name}.jsonl"
        descriptor = state["completed"].get(name)
        if descriptor and output_path.is_file() and file_sha256(output_path) == descriptor["sha256"]:
            rows = _read_jsonl(output_path)
            print(f"evaluation {position}/{len(checkpoints)} {name}: verified resume")
        else:
            model: torch.nn.Module | None
            if kind == "constant_velocity":
                model = None
            elif kind == "lstm":
                model = LSTMTrajectoryPredictor()
                _checkpoint_model(checkpoint, model)  # type: ignore[arg-type]
            else:
                model = VectorTrajectoryPredictor()
                _checkpoint_model(checkpoint, model)  # type: ignore[arg-type]
            run_started = time.perf_counter()
            rows = _evaluate(loader=loader, metadata=metadata, device=device, model=model, amp=True)
            _atomic_jsonl(output_path, rows)
            state["completed"][name] = {
                "checkpoint_sha256": checkpoint_hashes[name],
                "rows": len(rows),
                "sha256": file_sha256(output_path),
                "elapsed_seconds": time.perf_counter() - run_started,
            }
            _atomic_json(state_path, state)
            print(
                f"evaluation {position}/{len(checkpoints)} {name}: {len(rows)} samples "
                f"in {state['completed'][name]['elapsed_seconds']:.1f}s"
            )
        if len(rows) != len(dataset):
            raise ValueError(f"evaluation row count differs for {name}")
        rows_by_name[name] = rows
    schema = build_cvae_schema(Path("configs/skills"))
    skill_ids = list(schema.formal_skill_ids)
    model_views = {name: _views(rows, skill_ids) for name, rows in rows_by_name.items()}
    aggregate_rows: dict[str, list[dict[str, Any]]] = {}
    aggregate_views: dict[str, Any] = {}
    seed_statistics: dict[str, Any] = {}
    for experiment in ("e0", "e1", "e2", "e3"):
        names = [f"transformer_{experiment}_seed_{seed}" for seed in (2026, 2027, 2028)]
        aggregate_rows[experiment] = _average_seed_rows([rows_by_name[name] for name in names])
        aggregate_views[experiment] = _views(aggregate_rows[experiment], skill_ids)
        seed_statistics[experiment] = _seed_statistics([model_views[name] for name in names])
    comparisons: dict[str, Any] = {}
    for candidate in ("e1", "e2", "e3"):
        comparisons[f"{candidate}_vs_e0"] = {}
        for view, predicate in (
            ("overall", lambda row: not row["is_long_tail"]),
            ("real_long_tail", lambda row: row["is_long_tail"]),
        ):
            baseline_rows = [row for row in aggregate_rows["e0"] if predicate(row)]
            candidate_rows = [row for row in aggregate_rows[candidate] if predicate(row)]
            comparisons[f"{candidate}_vs_e0"][view] = {
                "delta": _delta(aggregate_views[candidate][view], aggregate_views["e0"][view]),
                "paired_bootstrap": paired_bootstrap_delta(
                    baseline_rows,
                    candidate_rows,
                    repetitions=args.bootstrap_repetitions,
                    seed=2026,
                ),
            }
    for view, predicate in (
        ("overall", lambda row: not row["is_long_tail"]),
        ("real_long_tail", lambda row: row["is_long_tail"]),
    ):
        comparisons.setdefault("e3_vs_e2", {})[view] = {
            "delta": _delta(aggregate_views["e3"][view], aggregate_views["e2"][view]),
            "paired_bootstrap": paired_bootstrap_delta(
                [row for row in aggregate_rows["e2"] if predicate(row)],
                [row for row in aggregate_rows["e3"] if predicate(row)],
                repetitions=args.bootstrap_repetitions,
                seed=2027,
            ),
        }
    augmentation = json.loads(args.augmentation_manifest.read_text(encoding="utf-8"))
    e3_entries = augmentation["arms"]["e3"]["entries"]
    proposal_counts = Counter(entry["proposal_mode"] for entry in e3_entries)
    proposal_by_skill: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in e3_entries:
        proposal_by_skill[entry["skill_id"]][entry["proposal_mode"]] += 1
    e3_skill_analysis = {
        skill_id: {
            "training_accepted_count": sum(proposal_by_skill[skill_id].values()),
            "training_proposal_mode_counts": dict(sorted(proposal_by_skill[skill_id].items())),
            "final_validation": aggregate_views["e3"]["per_skill"][skill_id],
            "e3_minus_e0": (
                None
                if aggregate_views["e3"]["per_skill"][skill_id]["sample_count"] == 0
                else _delta(
                    aggregate_views["e3"]["per_skill"][skill_id],
                    aggregate_views["e0"]["per_skill"][skill_id],
                )
            ),
        }
        for skill_id in skill_ids
    }
    e3_longtail_fde_change = comparisons["e3_vs_e0"]["real_long_tail"]["delta"]["min_fde"]
    e3_longtail_miss_change = comparisons["e3_vs_e0"]["real_long_tail"]["delta"]["miss_rate"]
    e3_overall = comparisons["e3_vs_e0"]["overall"]["delta"]
    payload = {
        "schema_version": 1,
        "kind": "downstream_prediction_final_evaluation",
        "status": "complete",
        "evaluation_identity": identity,
        "formal_contract_id": audit["contract_id"],
        "final_validation": {
            "cache_manifest_sha256": file_sha256(cache_manifest),
            "base_samples": sum(not value["is_long_tail"] for value in metadata.values()),
            "real_long_tail_samples": sum(value["is_long_tail"] for value in metadata.values()),
            "real_long_tail_scenarios": len({value["scenario_id"] for value in metadata.values() if value["is_long_tail"]}),
            "risk_metric_proxy_samples": sum(value.get("risk_stratum") == "proxy_metric" for value in metadata.values()),
        },
        "models": model_views,
        "three_seed_statistics": seed_statistics,
        "three_seed_averaged_views": aggregate_views,
        "comparisons": comparisons,
        "e3_training_source_analysis": {
            "proposal_mode_counts": dict(sorted(proposal_counts.items())),
            "per_skill": e3_skill_analysis,
        },
        "success_criteria": {
            "long_tail_min_fde_improved_at_least_5_percent": e3_longtail_fde_change["relative_percent"] <= -5.0,
            "long_tail_miss_rate_improved_at_least_5_percent": e3_longtail_miss_change["relative_percent"] <= -5.0,
            "either_long_tail_metric_met": e3_longtail_fde_change["relative_percent"] <= -5.0 or e3_longtail_miss_change["relative_percent"] <= -5.0,
            "overall_min_fde_degradation_at_most_2_percent": e3_overall["min_fde"]["relative_percent"] <= 2.0,
            "overall_miss_rate_degradation_at_most_2_percent": e3_overall["miss_rate"]["relative_percent"] <= 2.0,
        },
        "runtime": {
            "total_elapsed_seconds": time.perf_counter() - started,
            "device": str(device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "per_model_outputs": state["completed"],
        },
        "post_final_validation_tuning_permitted": False,
    }
    _atomic_json(args.summary_output, payload)
    print(
        "Final Validation evaluation complete: "
        f"overall={payload['final_validation']['base_samples']} "
        f"long_tail={payload['final_validation']['real_long_tail_samples']} "
        f"summary={args.summary_output}"
    )


if __name__ == "__main__":
    main()

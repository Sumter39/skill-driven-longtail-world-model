"""Build resumable, leakage-safe E1/E2/E3 augmentation views."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.data.coordinates import global_to_local
from skilldrive.data.cvae_samples import (
    NONE_SKILL_ID,
    PriorContextSpec,
    build_cvae_schema,
    tensorize_prior_context,
)
from skilldrive.data.manifests import read_manifest
from skilldrive.generation.storage import StoredRawCandidate, load_raw_shard_candidates
from skilldrive.prediction.audit import CONTRACT_ID, file_sha256
from skilldrive.prediction.data import PREDICTION_CONTEXT_FIELDS


PREDICTION_BUNDLE_SCHEMA_VERSION = 2
CONTEXT_SHARD_SIZE = 64
FUTURE_SHARD_SIZE = 256


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        ),
    )


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _validate_delivery(rows: list[dict[str, Any]], formal_ids: set[str]) -> None:
    if len(rows) != 1512:
        raise ValueError(f"balanced delivery must contain 1,512 rows, got {len(rows)}")
    ids = [row.get("candidate_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("balanced delivery contains invalid or duplicate candidate IDs")
    required = {
        "candidate_id", "task_id", "scenario_id", "skill_id", "target_track_id",
        "proposal_mode", "raw",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"balanced delivery row is missing fields: {sorted(required - set(row))}")
        if row["scenario_id"] not in formal_ids:
            raise ValueError("augmentation source is outside Formal Train")
        raw = row["raw"]
        if not isinstance(raw, Mapping) or not isinstance(raw.get("commit"), str):
            raise ValueError("balanced delivery row has no raw commit reference")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": 2, "contexts": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("prediction preparation state is incompatible")
    if not isinstance(value.get("contexts"), Mapping):
        raise ValueError("prediction preparation state has invalid contexts")
    return value


def _context_key(scenario_id: str, target_track_id: str) -> str:
    return f"{scenario_id}:{target_track_id}"


def _context_tensor(context: Any, name: str) -> torch.Tensor:
    value = getattr(context, name)
    return torch.as_tensor(value).clone()


def _build_context_shard(
    *,
    keys: list[tuple[str, str]],
    source_by_scenario: Mapping[str, Path],
    schema: Any,
    scenario_cache: dict[str, Any],
    output: Path,
    shard_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values: dict[str, list[torch.Tensor]] = defaultdict(list)
    entries: list[dict[str, Any]] = []
    for offset, (scenario_id, target_track_id) in enumerate(keys):
        scenario = scenario_cache.get(scenario_id)
        if scenario is None:
            scenario = load_av2_scenario(source_by_scenario[scenario_id])
            scenario_cache[scenario_id] = scenario
        context = tensorize_prior_context(
            scenario,
            PriorContextSpec(
                scenario_id=scenario_id,
                target_track_id=target_track_id,
                condition_skill_id=NONE_SKILL_ID,
            ),
            schema,
        )
        for name in PREDICTION_CONTEXT_FIELDS:
            values[name].append(_context_tensor(context, name))
        entries.append(
            {
                "context_key": _context_key(scenario_id, target_track_id),
                "scenario_id": scenario_id,
                "target_track_id": target_track_id,
                "context_shard": shard_name,
                "context_offset": offset,
            }
        )
    payload = {name: torch.stack(items) for name, items in values.items()}
    _atomic_torch(output / shard_name, payload)
    return payload, entries


def _rank_key(seed: int, task_id: str, candidate_id: str) -> str:
    return _canonical_sha({"version": 1, "seed": seed, "task_id": task_id, "candidate_id": candidate_id})


def _load_raw_candidate(
    row: Mapping[str, Any],
    *,
    formal_run_root: Path,
    cache: dict[str, tuple[StoredRawCandidate, ...]],
) -> tuple[StoredRawCandidate, ...]:
    raw = row["raw"]
    commit = formal_run_root / str(raw["commit"])
    key = commit.as_posix()
    values = cache.get(key)
    if values is None:
        values = load_raw_shard_candidates(commit)
        cache[key] = values
    return values


def _filter_outcomes(
    selected: Iterable[str],
    *,
    formal_run_root: Path,
    skill_ids: Iterable[str],
) -> dict[str, str]:
    remaining = set(selected)
    outcomes: dict[str, str] = {}
    for skill_id in sorted(set(skill_ids)):
        for status in ("accepted", "rejected"):
            path = formal_run_root / "filter" / skill_id / f"{status}.jsonl"
            if not path.is_file():
                continue
            for row in _read_jsonl(path):
                candidate_id = row.get("candidate_id")
                if candidate_id in remaining:
                    outcomes[candidate_id] = status
                    remaining.remove(candidate_id)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"selected E2 candidates are missing filter outcomes: {len(remaining)}")
    return outcomes


def _original_future_local(
    scenario: Any, target_track_id: str, origin: np.ndarray, heading: float
) -> tuple[np.ndarray, np.ndarray]:
    agent = next((item for item in scenario.agents if item.track_id == target_track_id), None)
    if agent is None:
        raise ValueError(f"target track is missing: {target_track_id}")
    future = np.asarray(agent.positions[50:110], dtype=np.float64)
    if future.shape != (60, 2):
        raise ValueError(f"E1 source future has invalid shape for {scenario.scenario_id}/{target_track_id}")
    mask = np.isfinite(future).all(axis=1)
    if not bool(mask.any()):
        raise ValueError(f"E1 source future has no valid points for {scenario.scenario_id}/{target_track_id}")
    local = np.zeros((60, 2), dtype=np.float32)
    local[mask] = global_to_local(future[mask], origin, heading).astype(np.float32)
    return local, mask


def _random_future(base: np.ndarray, seed: str) -> tuple[np.ndarray, dict[str, float]]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    second = int.from_bytes(digest[8:16], "big") / float(2**64)
    scale = 0.98 + 0.04 * unit
    lateral = -0.15 + 0.30 * second
    ramp = np.linspace(0.0, 1.0, 60, dtype=np.float32)
    ramp = ramp * ramp * (3.0 - 2.0 * ramp)
    output = base * np.float32(scale)
    output[:, 1] += np.float32(lateral) * ramp
    return output.astype(np.float32), {"position_scale": scale, "lateral_offset_m": lateral}


def _save_future_shards(
    *,
    output: Path,
    arm: str,
    rows: list[dict[str, Any]],
    futures: list[np.ndarray],
    masks: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    future_root = output / "futures" / arm
    entries: list[dict[str, Any]] = []
    descriptors: list[dict[str, Any]] = []
    for start in range(0, len(rows), FUTURE_SHARD_SIZE):
        stop = min(start + FUTURE_SHARD_SIZE, len(rows))
        shard_index = start // FUTURE_SHARD_SIZE
        name = f"futures/{arm}/shard-{shard_index:04d}.pt"
        path = output / name
        tensor = torch.from_numpy(np.stack(futures[start:stop], axis=0)).contiguous()
        mask_tensor = torch.from_numpy(np.stack(masks[start:stop], axis=0)).to(torch.bool)
        _atomic_torch(path, {"target_future": tensor, "target_future_mask": mask_tensor})
        descriptor = _descriptor(path, output)
        descriptors.append(descriptor)
        for offset, row in enumerate(rows[start:stop]):
            entries.append(
                {
                    **row,
                    "future_shard": name,
                    "future_offset": offset,
                }
            )
    return entries, descriptors


def build_prediction_augmentation_bundle(
    *,
    repository_root: str | Path = ".",
    data_root: str | Path,
    formal_run_root: str | Path,
    output_root: str | Path,
    manifest_output: str | Path | None = None,
    seed: int = 2026,
    resume: bool = True,
) -> dict[str, Any]:
    """Build shared contexts and deterministic E1/E2/E3 future arms."""

    root = Path(repository_root).resolve()
    data = Path(data_root).resolve()
    formal_run = Path(formal_run_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    delivery_path = formal_run / "review/formal_delivery_v1/balanced_accepted.jsonl"
    delivery_rows = _read_jsonl(delivery_path)
    formal_rows = read_manifest(root / "manifests/splits/formal_train.csv")
    formal_ids = {row.scenario_id for row in formal_rows}
    _validate_delivery(delivery_rows, formal_ids)
    source_by_scenario = {
        row.scenario_id: data / row.source_path for row in formal_rows
    }
    schema = build_cvae_schema(root / "configs/skills")

    keys = sorted({
        (str(row["scenario_id"]), str(row["target_track_id"]))
        for row in delivery_rows
    })
    state_path = output / "build_state.json"
    state = _load_state(state_path) if resume else {"schema_version": 2, "contexts": {}}
    context_entries_by_key: dict[str, dict[str, Any]] = {}
    context_shards: list[dict[str, Any]] = []
    scenario_cache: dict[str, Any] = {}
    context_root = output / "contexts"
    for start in range(0, len(keys), CONTEXT_SHARD_SIZE):
        stop = min(start + CONTEXT_SHARD_SIZE, len(keys))
        shard_index = start // CONTEXT_SHARD_SIZE
        shard_name = f"contexts/shard-{shard_index:04d}.pt"
        state_descriptor = state.get("contexts", {}).get(shard_name)
        path = output / shard_name
        if resume and state_descriptor and path.is_file() and file_sha256(path) == state_descriptor["sha256"]:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError(f"context shard is invalid: {path}")
            entries = [
                {
                    "context_key": _context_key(*keys[index]),
                    "scenario_id": keys[index][0],
                    "target_track_id": keys[index][1],
                    "context_shard": shard_name,
                    "context_offset": index - start,
                }
                for index in range(start, stop)
            ]
            descriptor = state_descriptor
        else:
            payload, entries = _build_context_shard(
                keys=keys[start:stop],
                source_by_scenario=source_by_scenario,
                schema=schema,
                scenario_cache=scenario_cache,
                output=output,
                shard_name=shard_name,
            )
            descriptor = _descriptor(path, output)
            state.setdefault("contexts", {})[shard_name] = descriptor
            _atomic_json(state_path, state)
        context_shards.append(descriptor)
        for entry in entries:
            context_entries_by_key[entry["context_key"]] = entry

    raw_cache: dict[str, tuple[StoredRawCandidate, ...]] = {}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in delivery_rows:
        by_task[str(row["task_id"])].append(row)
    e2_choice: dict[str, StoredRawCandidate] = {}
    for task_id, task_rows in sorted(by_task.items()):
        candidates = _load_raw_candidate(task_rows[0], formal_run_root=formal_run, cache=raw_cache)
        matching = [
            candidate for candidate in candidates
            if candidate.task_id == task_id
            and candidate.scenario_id == task_rows[0]["scenario_id"]
            and candidate.target_track_id == task_rows[0]["target_track_id"]
        ]
        if len(matching) < len(task_rows):
            raise ValueError(f"raw task {task_id} has too few candidates for E2")
        ordered = sorted(
            matching,
            key=lambda item: _rank_key(seed, task_id, item.candidate_id),
        )
        for row, candidate in zip(task_rows, ordered):
            e2_choice[str(row["candidate_id"])] = candidate

    outcomes = _filter_outcomes(
        (candidate.candidate_id for candidate in e2_choice.values()),
        formal_run_root=formal_run,
        skill_ids=(str(row["skill_id"]) for row in delivery_rows),
    )
    e3_candidates: dict[str, StoredRawCandidate] = {}
    for row in delivery_rows:
        candidates = _load_raw_candidate(row, formal_run_root=formal_run, cache=raw_cache)
        selected = next((item for item in candidates if item.candidate_id == row["candidate_id"]), None)
        if selected is None:
            raise ValueError(f"E3 candidate is missing from its raw shard: {row['candidate_id']}")
        e3_candidates[str(row["candidate_id"])] = selected

    arm_rows: dict[str, list[dict[str, Any]]] = {"e1": [], "e2": [], "e3": []}
    arm_futures: dict[str, list[np.ndarray]] = {"e1": [], "e2": [], "e3": []}
    arm_masks: dict[str, list[np.ndarray]] = {"e1": [], "e2": [], "e3": []}
    incomplete_e1_rows = 0
    incomplete_e1_points = 0
    for row_index, row in enumerate(delivery_rows):
        key = _context_key(str(row["scenario_id"]), str(row["target_track_id"]))
        context_entry = context_entries_by_key.get(key)
        if context_entry is None:
            raise ValueError(f"context entry is missing for {key}")
        context_payload = torch.load(
            output / context_entry["context_shard"], map_location="cpu", weights_only=True
        )
        origin = context_payload["anchor_origin_global"][context_entry["context_offset"]].numpy()
        heading = float(context_payload["anchor_heading_global"][context_entry["context_offset"]].item())
        scenario = scenario_cache.get(str(row["scenario_id"]))
        if scenario is None:
            scenario = load_av2_scenario(source_by_scenario[str(row["scenario_id"])])
            scenario_cache[str(row["scenario_id"])] = scenario
        base_future, base_mask = _original_future_local(
            scenario, str(row["target_track_id"]), origin, heading
        )
        incomplete_e1_rows += int(not bool(base_mask.all()))
        incomplete_e1_points += int((~base_mask).sum())
        e1_future, random_parameters = _random_future(
            base_future, f"prediction-e1-v1:{seed}:{row['candidate_id']}"
        )
        e2 = e2_choice[str(row["candidate_id"])].future_xy_global
        e3 = e3_candidates[str(row["candidate_id"])].future_xy_global
        e2_future = global_to_local(e2, origin, heading).astype(np.float32)
        e3_future = global_to_local(e3, origin, heading).astype(np.float32)
        common = {
            "sample_id": _canonical_sha({"arm": "prediction", "row": row_index, "candidate_id": row["candidate_id"]}),
            "scenario_id": str(row["scenario_id"]),
            "target_track_id": str(row["target_track_id"]),
            "skill_id": str(row["skill_id"]),
            "proposal_mode": str(row["proposal_mode"]),
            "task_id": str(row["task_id"]),
            "candidate_id": str(row["candidate_id"]),
            "context_shard": context_entry["context_shard"],
            "context_offset": context_entry["context_offset"],
        }
        arm_rows["e1"].append({**common, "source_type": "random_augmentation", "augmentation_parameters": random_parameters})
        arm_rows["e2"].append({**common, "source_type": "raw_generated", "raw_candidate_id": e2_choice[str(row["candidate_id"])].candidate_id, "raw_filter_outcome": outcomes[e2_choice[str(row["candidate_id"])].candidate_id]})
        arm_rows["e3"].append({**common, "source_type": "accepted_generated", "raw_candidate_id": str(row["candidate_id"]), "raw_filter_outcome": "accepted"})
        arm_futures["e1"].append(e1_future)
        arm_futures["e2"].append(e2_future)
        arm_futures["e3"].append(e3_future)
        arm_masks["e1"].append(base_mask)
        arm_masks["e2"].append(np.ones((60,), dtype=bool))
        arm_masks["e3"].append(np.ones((60,), dtype=bool))

    arms: dict[str, Any] = {}
    for arm in ("e1", "e2", "e3"):
        entries, descriptors = _save_future_shards(
            output=output,
            arm=arm,
            rows=arm_rows[arm],
            futures=arm_futures[arm],
            masks=arm_masks[arm],
        )
        arms[arm] = {
            "count": len(entries),
            "entries": entries,
            "future_shards": descriptors,
        }
    manifest = {
        "schema_version": PREDICTION_BUNDLE_SCHEMA_VERSION,
        "kind": "downstream_prediction_augmentation_bundle",
        "status": "complete",
        "contract_id": CONTRACT_ID,
        "seed": seed,
        "delivery_index_sha256": file_sha256(delivery_path),
        "context_count": len(keys),
        "context_shards": context_shards,
        "arms": arms,
        "final_validation_accessed": False,
        "e2_filter_outcome_counts": {
            status: sum(1 for value in outcomes.values() if value == status)
            for status in ("accepted", "rejected")
        },
        "e2_e3_candidate_overlap": len({
            value.candidate_id for value in e2_choice.values()
        } & set(e3_candidates)),
        "e1_partial_future_rows": incomplete_e1_rows,
        "e1_masked_future_points": incomplete_e1_points,
    }
    _atomic_json(output / "manifest.json", manifest)
    if manifest_output is not None:
        _atomic_json(Path(manifest_output), manifest)
    return manifest


__all__ = [
    "build_prediction_augmentation_bundle",
    "CONTEXT_SHARD_SIZE",
    "FUTURE_SHARD_SIZE",
]

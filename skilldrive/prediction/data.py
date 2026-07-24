"""Leakage-safe data views for downstream trajectory prediction."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from skilldrive.data.cvae_cache import CVAECachedDataset


PREDICTION_TENSOR_FIELDS = (
    "actor_history",
    "actor_time_mask",
    "actor_mask",
    "actor_type_id",
    "actor_role_id",
    "map_polylines",
    "map_point_mask",
    "map_polyline_mask",
    "map_type_id",
    "target_actor_index",
    "target_future",
    "target_future_mask",
    "anchor_origin_global",
    "anchor_heading_global",
)
# These are the only tensors that may be passed to a predictor. In particular,
# skill labels, parameter masks, filter decisions and provenance never enter it.
MODEL_INPUT_FIELDS = (
    "actor_history",
    "actor_time_mask",
    "actor_mask",
    "actor_type_id",
    "map_polylines",
    "map_point_mask",
    "map_polyline_mask",
    "map_type_id",
    "target_actor_index",
)

PREDICTION_CONTEXT_FIELDS = tuple(name for name in PREDICTION_TENSOR_FIELDS if name not in {
    "target_future", "target_future_mask"
})


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sample_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = set(PREDICTION_TENSOR_FIELDS) - set(value)
    if missing:
        raise ValueError(f"prediction sample is missing fields: {sorted(missing)}")
    result: dict[str, Any] = {
        name: value[name] for name in PREDICTION_TENSOR_FIELDS
    }
    for name in PREDICTION_TENSOR_FIELDS:
        if not isinstance(result[name], Tensor):
            result[name] = torch.as_tensor(result[name])
    if result["target_future"].shape != (60, 2):
        raise ValueError("target_future must have shape (60, 2)")
    if result["target_future_mask"].shape != (60,):
        raise ValueError("target_future_mask must have shape (60,)")
    if not bool(result["target_future_mask"].any()):
        raise ValueError("prediction sample must contain at least one future point")
    valid_future = result["target_future_mask"].to(dtype=torch.bool)
    if not bool(torch.isfinite(result["target_future"][valid_future]).all()):
        raise ValueError("valid target future values must be finite")
    return result


def _metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("sample_id", "scenario_id", "target_track_id", "source_type")
    result = {name: value.get(name) for name in required}
    if any(not isinstance(item, str) or not item for item in result.values()):
        raise ValueError("prediction sample metadata is incomplete")
    # Keep provenance available for stratified reporting, but never pass this
    # mapping to a model. Values are restricted to JSON scalars/lists.
    for name, item in value.items():
        if name in result:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[name] = item  # type: ignore[assignment]
        elif isinstance(item, (list, tuple)) and all(
            isinstance(entry, (str, int, float, bool)) or entry is None
            for entry in item
        ):
            result[name] = list(item)  # type: ignore[assignment]
    return result  # type: ignore[return-value]


def prediction_model_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Extract only predictor-safe tensors from a collated batch."""

    missing = set(MODEL_INPUT_FIELDS) - set(batch)
    if missing:
        raise ValueError(f"prediction batch is missing model inputs: {sorted(missing)}")
    return {name: batch[name] for name in MODEL_INPUT_FIELDS}


class PredictionRealDataset(Dataset[dict[str, Any]]):
    """Real CVAE samples with provenance retained outside the model input."""

    def __init__(self, dataset: CVAECachedDataset, *, source_type: str = "real") -> None:
        if source_type != "real":
            raise ValueError("PredictionRealDataset source_type must be real")
        self.dataset = dataset
        self.source_type = source_type
        self.entries = dataset.entries

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        raw = self.dataset[index]
        sample = _tensor_sample_from_mapping(raw)
        sample_id = raw.get("sample_id")
        scenario_id = raw.get("scenario_id")
        target_track_id = raw.get("target_track_id")
        if not all(isinstance(item, str) and item for item in (sample_id, scenario_id, target_track_id)):
            raise ValueError("CVAE sample identifiers are invalid")
        return {
            **sample,
            "metadata": {
                "sample_id": sample_id,
                "scenario_id": scenario_id,
                "target_track_id": target_track_id,
                "source_type": self.source_type,
            },
        }


class PredictionAugmentationDataset(Dataset[dict[str, Any]]):
    """Load one generated/random arm over a shared context cache."""

    def __init__(self, bundle_dir: str | Path, arm: str, *, in_memory_shards: int = 2) -> None:
        root = Path(bundle_dir)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 2:
            raise ValueError("prediction augmentation manifest version is unsupported")
        arms = manifest.get("arms")
        if not isinstance(arms, Mapping) or arm not in arms:
            raise ValueError(f"prediction augmentation arm is not available: {arm}")
        self.root = root
        self.arm = arm
        self.manifest = manifest
        self.entries = list(arms[arm]["entries"])
        if len(self.entries) != int(arms[arm]["count"]):
            raise ValueError("prediction arm entry count differs from manifest")
        self._context_shards = {
            item["path"]: item for item in manifest.get("context_shards", [])
        }
        self._future_shards = {
            item["path"]: item for item in arms[arm].get("future_shards", [])
        }
        if not self._context_shards or not self._future_shards:
            raise ValueError("prediction augmentation manifest has no shards")
        if isinstance(in_memory_shards, bool) or in_memory_shards < 1:
            raise ValueError("in_memory_shards must be positive")
        self.in_memory_shards = int(in_memory_shards)
        self._contexts: OrderedDict[str, dict[str, Tensor]] = OrderedDict()
        self._futures: OrderedDict[str, tuple[Tensor, Tensor]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def _load_context(self, name: str) -> dict[str, Tensor]:
        value = self._contexts.get(name)
        if value is None:
            descriptor = self._context_shards.get(name)
            if descriptor is None:
                raise ValueError(f"unknown context shard: {name}")
            path = self.root / name
            if path.stat().st_size != descriptor["size_bytes"]:
                raise ValueError(f"context shard size differs: {path}")
            if _file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"context shard hash differs: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError("context shard must contain a mapping")
            value = {name: payload[name] for name in PREDICTION_CONTEXT_FIELDS}
            self._contexts[name] = value
            while len(self._contexts) > self.in_memory_shards:
                self._contexts.popitem(last=False)
        else:
            self._contexts.move_to_end(name)
        return value

    def _load_future(self, name: str) -> tuple[Tensor, Tensor]:
        value = self._futures.get(name)
        if value is None:
            descriptor = self._future_shards.get(name)
            if descriptor is None:
                raise ValueError(f"unknown future shard: {name}")
            path = self.root / name
            if path.stat().st_size != descriptor["size_bytes"]:
                raise ValueError(f"future shard size differs: {path}")
            if _file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"future shard hash differs: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError("future shard must contain a mapping")
            value = payload["target_future"]
            mask = payload["target_future_mask"]
            if not isinstance(value, Tensor) or value.ndim != 3 or tuple(value.shape[1:]) != (60, 2):
                raise ValueError("future shard must contain [N, 60, 2]")
            if not isinstance(mask, Tensor) or mask.shape != value.shape[:2] or mask.dtype is not torch.bool:
                raise ValueError("future shard mask must contain [N, 60] bool")
            value = (value, mask)
            self._futures[name] = value
            while len(self._futures) > self.in_memory_shards:
                self._futures.popitem(last=False)
        else:
            self._futures.move_to_end(name)
        return value

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        context = self._load_context(entry["context_shard"])
        futures, masks = self._load_future(entry["future_shard"])
        offset = entry["future_offset"]
        future = futures[offset]
        mask = masks[offset]
        sample = {
            name: context[name][entry["context_offset"]]
            for name in PREDICTION_CONTEXT_FIELDS
        }
        sample["target_future"] = future
        sample["target_future_mask"] = mask
        sample["metadata"] = _metadata(entry)
        return sample


def collate_prediction_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack fixed-shape tensors while keeping identifiers out of model inputs."""

    if not samples:
        raise ValueError("cannot collate an empty prediction batch")
    result: dict[str, Any] = {
        name: torch.stack([_tensor_sample_from_mapping(item)[name] for item in samples])
        for name in PREDICTION_TENSOR_FIELDS
    }
    result["metadata"] = [dict(item["metadata"]) for item in samples]
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MODEL_INPUT_FIELDS",
    "PREDICTION_CONTEXT_FIELDS",
    "PREDICTION_TENSOR_FIELDS",
    "PredictionAugmentationDataset",
    "PredictionRealDataset",
    "collate_prediction_samples",
    "prediction_model_inputs",
]

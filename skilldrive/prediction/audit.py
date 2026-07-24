"""Stage-A input and split audit for downstream prediction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from skilldrive.data.manifests import read_manifest


CONTRACT_ID = "6b2da617bcf0694b87ea055285f971b58d660ae4591f49d039de1d51de99baf3"
PREDICTION_AUDIT_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _manifest_audit(path: Path, expected_split: str, repository_root: Path) -> dict[str, Any]:
    rows = read_manifest(path)
    ids = [row.scenario_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"manifest contains duplicate scenario IDs: {path}")
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"manifest contains an unexpected split: {path}")
    return {
        "path": path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "sha256": file_sha256(path),
        "scenario_count": len(ids),
        "scenario_ids_sha256": hashlib.sha256(
            "\n".join(ids).encode("utf-8")
        ).hexdigest(),
        "scenario_ids": ids,
    }


def _cache_audit(cache_root: Path, partition: str, expected_samples: int) -> dict[str, Any]:
    cache_dir = cache_root / partition
    manifest_path = cache_dir / "cache_manifest.json"
    data = _read_json(manifest_path)
    counts = data.get("counts")
    if data.get("status") != "complete" or not isinstance(counts, Mapping):
        raise ValueError(f"CVAE cache is not complete: {cache_dir}")
    if counts.get("retained_samples") != expected_samples:
        raise ValueError(
            f"{partition} cache sample count differs: {counts.get('retained_samples')}"
        )
    index = cache_dir / str(data.get("sample_index", {}).get("path", ""))
    if not index.is_file():
        raise ValueError(f"CVAE sample index is missing: {index}")
    if file_sha256(index) != data.get("sample_index", {}).get("sha256"):
        raise ValueError(f"CVAE sample index hash differs: {index}")
    return {
        "partition": partition,
        "cache_manifest_sha256": file_sha256(manifest_path),
        "sample_index_sha256": file_sha256(index),
        "scenario_count": counts.get("manifest_scenarios"),
        "sample_count": counts.get("retained_samples"),
        "shard_count": len(data.get("shards", [])),
    }


def audit_prediction_inputs(
    *,
    repository_root: str | Path = ".",
    cache_root: str | Path,
    formal_run_root: str | Path,
    output_path: str | Path,
    archive_paths: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Audit all stage-A inputs without opening scenario trajectories from Validation."""

    root = Path(repository_root).resolve()
    cache = Path(cache_root).resolve()
    run = Path(formal_run_root).resolve()
    manifest_paths = {
        "formal_train": root / "manifests/splits/formal_train.csv",
        "internal_validation": root / "manifests/splits/internal_validation.csv",
        "final_validation": root / "manifests/splits/final_validation.csv",
    }
    manifests = {
        "formal_train": _manifest_audit(
            manifest_paths["formal_train"], "train", root
        ),
        "internal_validation": _manifest_audit(
            manifest_paths["internal_validation"], "internal_validation", root
        ),
        "final_validation": _manifest_audit(
            manifest_paths["final_validation"], "validation", root
        ),
    }
    id_sets = {name: set(value["scenario_ids"]) for name, value in manifests.items()}
    overlap = {
        f"{left}:{right}": len(id_sets[left] & id_sets[right])
        for left, right in (
            ("formal_train", "internal_validation"),
            ("formal_train", "final_validation"),
            ("internal_validation", "final_validation"),
        )
    }
    if any(overlap.values()):
        raise ValueError(f"prediction split overlap detected: {overlap}")
    for value in manifests.values():
        value.pop("scenario_ids", None)

    caches = {
        "formal_train": _cache_audit(cache, "formal_train", 29382),
        "internal_validation": _cache_audit(cache, "internal_validation", 2954),
    }
    formal_summary = _read_json(run / "summary.json")
    if (
        formal_summary.get("formal_plan_id") != CONTRACT_ID
        or formal_summary.get("status") != "completed"
        or formal_summary.get("validation_manifests_opened") is not False
        or formal_summary.get("final_validation_accessed") is not False
    ):
        raise ValueError("05 formal run does not satisfy the downstream input contract")
    delivery_audit = _read_json(run / "review/formal_delivery_v1/audit.json")
    delivery_path = run / "review/formal_delivery_v1/balanced_accepted.jsonl"
    delivery_rows = [
        json.loads(line)
        for line in delivery_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(delivery_rows) != 1512 or delivery_audit.get("selected_candidate_count") != 1512:
        raise ValueError("05 balanced delivery does not contain exactly 1,512 rows")
    candidate_ids = [row.get("candidate_id") for row in delivery_rows]
    if len(candidate_ids) != len(set(candidate_ids)) or any(
        not isinstance(value, str) for value in candidate_ids
    ):
        raise ValueError("05 balanced delivery contains invalid or duplicate IDs")
    formal_ids = id_sets["formal_train"]
    if not all(row.get("scenario_id") in formal_ids for row in delivery_rows):
        raise ValueError("balanced delivery contains a non-Formal-Train scenario")
    review_summary = _read_json(run / "review/formal_review_v1/automated_review_summary.json")
    if (
        review_summary.get("review_method") != "automated_evidence"
        or review_summary.get("automated_review_count") != 149
        or review_summary.get("validation_manifests_opened") is not False
        or review_summary.get("final_validation_accessed") is not False
    ):
        raise ValueError("05 automated review evidence is incomplete")

    archive_descriptors = []
    for archive in archive_paths:
        path = Path(archive).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        archive_descriptors.append(
            {
                "repository_label": (
                    "outputs/generation/" + path.name
                    if path.parent.name == "generation"
                    else path.name
                ),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )

    output = Path(output_path)
    payload = {
        "schema_version": PREDICTION_AUDIT_SCHEMA_VERSION,
        "kind": "downstream_prediction_input_audit",
        "status": "complete",
        "contract_id": CONTRACT_ID,
        "split_overlap_counts": overlap,
        "manifests": manifests,
        "cvae_caches": caches,
        "formal_generation": {
            "contract_id": CONTRACT_ID,
            "summary_sha256": file_sha256(run / "summary.json"),
            "delivery_audit_sha256": file_sha256(run / "review/formal_delivery_v1/audit.json"),
            "delivery_index_sha256": file_sha256(delivery_path),
            "delivery_count": len(delivery_rows),
            "unique_delivery_scenarios": len({row["scenario_id"] for row in delivery_rows}),
            "proposal_mode_counts": delivery_audit.get("proposal_mode_counts", {}),
        },
        "automated_review": {
            "summary_sha256": file_sha256(run / "review/formal_review_v1/automated_review_summary.json"),
            "review_method": review_summary["review_method"],
            "review_count": review_summary["automated_review_count"],
        },
        "archives": archive_descriptors,
        "final_validation_content_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


__all__ = ["CONTRACT_ID", "audit_prediction_inputs", "file_sha256"]

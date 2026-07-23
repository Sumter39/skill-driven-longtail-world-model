"""Render resumable source/generated BEV pairs for the active Pilot review set."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.generation.assembly import materialize_overlay_scenario
from skilldrive.generation.capability import write_generation_capability_matrix
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.inference import file_sha256
from skilldrive.generation.planning import seed_record_id
from skilldrive.generation.storage import load_raw_shard_candidates
from skilldrive.seeds.records import read_seed_records
from skilldrive.visualization.seed_review import render_seed_review


PILOT_BEV_REVIEW_SCHEMA_VERSION = 1
PILOT_BEV_REVIEW_CONTRACT = "active_pilot_bev_review_v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return value


def _resolved(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_label(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _valid_png(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
            return False
        with path.open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


def _safe_segment(value: str, limit: int = 48) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (result or "unknown")[:limit]


def _source_path(data_root: Path, source_path: str, scenario_id: str) -> Path:
    source = PurePosixPath(source_path.replace("\\", "/"))
    if (
        source.is_absolute()
        or len(source.parts) < 3
        or source.parts[0] != "train"
        or source.suffix.lower() != ".parquet"
        or any(part in {"", ".", ".."} for part in source.parts)
        or ":" in source.parts[0]
    ):
        raise ValueError(
            f"Pilot review source must be one Formal Train parquet: {source_path}"
        )
    root = data_root.resolve()
    resolved = root.joinpath(*source.parts).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError(
            f"Pilot review scenario is absent or outside data_root: {scenario_id}: {resolved}"
        )
    return resolved


def render_active_pilot_review(
    *,
    gate_analysis_path: str | Path,
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    output_root: str | Path | None = None,
    repository_root: str | Path = ".",
) -> dict[str, Any]:
    """Render all frozen Pilot review cases without modifying source/raw artifacts."""

    root = Path(repository_root).resolve()
    analysis_path = _resolved(root, gate_analysis_path)
    config_path = _resolved(root, generation_config_path)
    analysis = _read_json(analysis_path, "Pilot gate analysis")
    if (
        analysis.get("kind") != "active_pilot_gate_analysis"
        or analysis.get("status") != "passed"
        or analysis.get("validation_manifests_opened") is not False
        or analysis.get("final_validation_accessed") is not False
    ):
        raise ValueError("Pilot gate analysis is not a passed Formal Train artifact")
    outputs = analysis.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("Pilot gate analysis lacks review-manifest evidence")
    manifest_path = _resolved(root, str(outputs.get("review_manifest", "")))
    expected_manifest_sha256 = outputs.get("review_manifest_sha256")
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Pilot review manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path, "Pilot review manifest")
    cases = manifest.get("cases")
    if (
        manifest.get("kind") != "active_pilot_review_manifest"
        or manifest.get("analysis_id") != analysis.get("analysis_id")
        or not isinstance(cases, list)
        or len(cases) != manifest.get("case_count")
        or manifest.get("validation_manifests_opened") is not False
        or manifest.get("final_validation_accessed") is not False
    ):
        raise ValueError("Pilot review manifest is invalid")

    config = load_counterfactual_config(config_path)
    records = read_seed_records(config.inputs.seed_manifest)
    records_by_id = {seed_record_id(record): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")
    data_root = _resolved(root, config.inputs.data_root)
    destination = (
        _resolved(root, output_root)
        if output_root is not None
        else analysis_path.parent / "bev-review-v1"
    )

    scenario_cache: dict[str, Any] = {}
    shard_cache: dict[Path, tuple[Any, ...]] = {}
    rendered = 0
    skipped = 0
    index_rows: list[dict[str, Any]] = []
    for rank, case in enumerate(cases, start=1):
        if not isinstance(case, Mapping):
            raise ValueError("Pilot review case must be a JSON object")
        candidate_id = case.get("candidate_id")
        scenario_id = case.get("scenario_id")
        skill_id = case.get("skill_id")
        record_id = case.get("seed_record_id")
        target_track_id = case.get("target_track_id")
        raw = case.get("raw")
        if not all(
            isinstance(value, str) and value
            for value in (candidate_id, scenario_id, skill_id, record_id, target_track_id)
        ) or not isinstance(raw, Mapping):
            raise ValueError("Pilot review case identity is malformed")
        if record_id not in records_by_id:
            raise ValueError("Pilot review case references an unknown seed record")
        record = records_by_id[record_id]
        if record.scenario_id != scenario_id or record.skill_id != skill_id:
            raise ValueError("Pilot review case differs from its seed record")

        commit_path = _resolved(root, str(raw.get("commit", "")))
        candidates = shard_cache.get(commit_path)
        if candidates is None:
            candidates = load_raw_shard_candidates(commit_path)
            shard_cache[commit_path] = candidates
        offset = raw.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset < len(
            candidates
        ):
            raise ValueError("Pilot review raw offset is invalid")
        candidate = candidates[offset]
        if (
            candidate.candidate_id != candidate_id
            or candidate.scenario_id != scenario_id
            or candidate.skill_id != skill_id
            or candidate.target_track_id != target_track_id
        ):
            raise ValueError("Pilot review raw candidate identity mismatch")

        source = scenario_cache.get(scenario_id)
        if source is None:
            source = load_av2_scenario(
                _source_path(data_root, record.source_path, scenario_id)
            )
            scenario_cache[scenario_id] = source
        overlay = materialize_overlay_scenario(
            source,
            target_track_id,
            candidate.future_xy_global,
        )
        case_name = (
            f"{rank:03d}-{_safe_segment(skill_id)}-"
            f"{_safe_segment(str(case.get('disposition', 'case')), 12)}-"
            f"{candidate_id[:12]}"
        )
        case_root = destination / "cases" / case_name
        source_png = case_root / "source" / "placeholder.png"
        generated_png = case_root / "generated" / "placeholder.png"
        expected_source = case_root / "source"
        expected_generated = case_root / "generated"
        if source_png.parent.exists():
            existing = tuple(source_png.parent.glob("*.png"))
            if len(existing) == 1:
                source_png = existing[0]
        if generated_png.parent.exists():
            existing = tuple(generated_png.parent.glob("*.png"))
            if len(existing) == 1:
                generated_png = existing[0]
        if _valid_png(source_png) and _valid_png(generated_png):
            skipped += 1
        else:
            source_png = render_seed_review(source, record, expected_source)
            generated_png = render_seed_review(overlay, record, expected_generated)
            if not _valid_png(source_png) or not _valid_png(generated_png):
                raise ValueError("Pilot BEV renderer did not produce valid PNG files")
            rendered += 1
        index_rows.append(
            {
                "review_rank": rank,
                "candidate_id": candidate_id,
                "task_id": case.get("task_id"),
                "scenario_id": scenario_id,
                "skill_id": skill_id,
                "disposition": case.get("disposition"),
                "evaluation_arm": case.get("evaluation_arm"),
                "first_failed_stage": case.get("first_failed_stage"),
                "primary_rejection_reason": case.get("primary_rejection_reason"),
                "source_png": {
                    "path": _path_label(root, source_png),
                    "sha256": file_sha256(source_png),
                },
                "generated_png": {
                    "path": _path_label(root, generated_png),
                    "sha256": file_sha256(generated_png),
                },
            }
        )

    summary = {
        "schema_version": PILOT_BEV_REVIEW_SCHEMA_VERSION,
        "kind": "active_pilot_bev_review",
        "contract": PILOT_BEV_REVIEW_CONTRACT,
        "status": "rendered",
        "gate_analysis": {
            "path": _path_label(root, analysis_path),
            "sha256": file_sha256(analysis_path),
        },
        "review_manifest": {
            "path": _path_label(root, manifest_path),
            "sha256": expected_manifest_sha256,
        },
        "case_count": len(index_rows),
        "completed_case_count": len(index_rows),
        "image_count": 2 * len(index_rows),
        "cases": index_rows,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
    }
    summary_path = destination / "pilot_bev_review.json"
    if summary_path.exists():
        if _read_json(summary_path, "Pilot BEV review") != summary:
            raise ValueError("existing Pilot BEV review differs from current evidence")
    else:
        write_generation_capability_matrix(summary_path, summary)
    return {
        **summary,
        "rendered_case_count": rendered,
        "resumed_case_count": skipped,
        "output_path": _path_label(root, summary_path),
    }


__all__ = [
    "PILOT_BEV_REVIEW_CONTRACT",
    "PILOT_BEV_REVIEW_SCHEMA_VERSION",
    "render_active_pilot_review",
]

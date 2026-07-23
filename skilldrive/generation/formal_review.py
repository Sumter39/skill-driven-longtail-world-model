"""Deterministic representative selection and BEV review for a completed formal run."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from collections.abc import Iterator
from typing import Any, Mapping

from PIL import Image

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.generation.assembly import materialize_overlay_scenario
from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.inference import file_sha256
from skilldrive.generation.planning import seed_record_id
from skilldrive.generation.storage import load_raw_shard_candidates
from skilldrive.seeds.records import read_seed_records
from skilldrive.visualization.seed_review import render_seed_review


FORMAL_REVIEW_SCHEMA_VERSION = 1
FORMAL_REVIEW_CONTRACT = "formal_generation_review_v1"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
FAILED_STAGE_ORDER = (
    "schema_finite",
    "history_invariants",
    "kinematics",
    "map",
    "collision",
    "target_risk",
    "skill_trigger",
    "parameter_realization",
    "diversity",
)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"failed to read review rows: {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"review row must be an object at {path}:{line_number}")
            yield value


def _resolved(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _path_label(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _safe_segment(value: str, limit: int = 48) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (segment or "unknown")[:limit]


def _valid_png(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= len(PNG_SIGNATURE):
            return False
        with path.open("rb") as handle:
            return handle.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


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
        raise ValueError(f"review source is not a Formal Train parquet: {source_path}")
    root = data_root.resolve()
    resolved = root.joinpath(*source.parts).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError(f"review source is absent: {scenario_id}: {resolved}")
    return resolved


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("formal filter row is missing metrics")
    scenario_id = metrics.get("scenario_id")
    skill_id = metrics.get("skill_id")
    task_id = row.get("task_id")
    candidate_index = row.get("candidate_index")
    candidate_id = row.get("candidate_id")
    if (
        not all(isinstance(value, str) and value for value in (scenario_id, skill_id, task_id, candidate_id))
        or isinstance(candidate_index, bool)
        or not isinstance(candidate_index, int)
    ):
        raise ValueError("formal filter row has invalid identity")
    return scenario_id, skill_id, task_id, candidate_index, candidate_id


def _select_representatives(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Cover distinct first-failure stages before filling by stable identity."""

    if limit <= 0:
        return []
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metrics = row.get("metrics")
        stage = metrics.get("first_failed_stage") if isinstance(metrics, Mapping) else None
        by_stage[str(stage or "unknown")].append(row)
    order = {stage: index for index, stage in enumerate(FAILED_STAGE_ORDER)}
    selected: list[dict[str, Any]] = []
    selected_scenarios: set[str] = set()
    for stage in sorted(by_stage, key=lambda value: (order.get(value, len(order)), value)):
        candidates = sorted(by_stage[stage], key=lambda row: _row_identity(row)[0:1] + _row_identity(row)[3:])
        for row in candidates:
            scenario_id = _row_identity(row)[0]
            if scenario_id in selected_scenarios:
                continue
            selected.append(row)
            selected_scenarios.add(scenario_id)
            break
        if len(selected) >= limit:
            return selected[:limit]
    remaining = sorted(rows, key=lambda row: _row_identity(row))
    for row in remaining:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _select_accepted(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[float, tuple[str, str, str, int, str]]:
        metrics = row.get("metrics")
        score = metrics.get("quality_score") if isinstance(metrics, Mapping) else None
        value = float(score) if isinstance(score, (int, float)) else float("-inf")
        return (-value, _row_identity(row))

    return sorted(rows, key=key)[: max(0, limit)]


def _bounded_identity_rows(
    rows: list[dict[str, Any]],
    row: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    return sorted(rows + [row], key=_row_identity)[:limit]


def _load_filter_rows(
    run_root: Path,
    *,
    per_disposition: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for directory in sorted(path for path in (run_root / "filter").iterdir() if path.is_dir()):
        accepted_path = directory / "accepted.jsonl"
        rejected_path = directory / "rejected.jsonl"
        accepted: list[dict[str, Any]] = []
        if accepted_path.is_file():
            for row in _iter_jsonl(accepted_path):
                accepted = _select_accepted(accepted + [row], per_disposition)
        rejected_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rejected_fallback: list[dict[str, Any]] = []
        if rejected_path.is_file():
            for row in _iter_jsonl(rejected_path):
                metrics = row.get("metrics")
                stage = metrics.get("first_failed_stage") if isinstance(metrics, Mapping) else None
                stage_name = str(stage or "unknown")
                rejected_by_stage[stage_name] = _bounded_identity_rows(
                    rejected_by_stage[stage_name],
                    row,
                    max(3, per_disposition),
                )
                rejected_fallback = _bounded_identity_rows(
                    rejected_fallback,
                    row,
                    max(12, per_disposition * 4),
                )
        rejected_pool = rejected_fallback + [
            row for stage_rows in rejected_by_stage.values() for row in stage_rows
        ]
        rejected = _select_representatives(rejected_pool, per_disposition)
        for row in accepted + rejected:
            _, skill_id, _, _, _ = _row_identity(row)
            if skill_id != directory.name:
                raise ValueError(f"filter row skill differs from directory: {directory}")
        if accepted or rejected:
            result[directory.name] = {"accepted": accepted, "rejected": rejected}
        print(
            f"formal review selection: {directory.name} "
            f"{len(accepted)} accepted / {len(rejected)} rejected representatives",
            flush=True,
        )
    return result


def build_formal_review_manifest(
    *,
    run_root: str | Path,
    repository_root: str | Path = ".",
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    output_root: str | Path | None = None,
    per_disposition: int = 3,
) -> tuple[Path, dict[str, Any]]:
    """Select deterministic review cases without loading trajectory arrays."""

    root = Path(repository_root).resolve()
    run = Path(run_root).resolve()
    summary = _read_json(run / "summary.json", "formal summary")
    if (
        summary.get("kind") != "formal_counterfactual_summary"
        or summary.get("status") != "completed"
        or summary.get("validation_manifests_opened") is not False
        or summary.get("final_validation_accessed") is not False
    ):
        raise ValueError("formal run is not a completed Formal Train artifact")
    config_path = _resolved(root, generation_config_path)
    config = load_counterfactual_config(config_path, repository_root=root)
    records = read_seed_records(_resolved(root, config.inputs.seed_manifest))
    records_by_id = {seed_record_id(record): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("formal seed manifest contains duplicate record identities")
    data_root = _resolved(root, config.inputs.data_root)
    rows_by_skill = _load_filter_rows(run, per_disposition=per_disposition)
    destination = (
        _resolved(root, output_root)
        if output_root is not None
        else run / "review" / "formal_review_v1"
    )
    cases: list[dict[str, Any]] = []
    rank = 0
    for skill_id in sorted(rows_by_skill):
        selected = [("accepted", row) for row in rows_by_skill[skill_id]["accepted"]] + [
            ("rejected", row) for row in rows_by_skill[skill_id]["rejected"]
        ]
        for disposition, row in selected:
            scenario_id, row_skill, task_id, candidate_index, candidate_id = _row_identity(row)
            metrics = row["metrics"]
            record_id = metrics["seed_record_id"]
            record = records_by_id.get(record_id)
            if record is None or record.scenario_id != scenario_id or record.skill_id != row_skill:
                raise ValueError("formal review row does not match seed manifest")
            _source_path(data_root, record.source_path, scenario_id)
            raw = row.get("raw")
            if not isinstance(raw, Mapping) or not isinstance(raw.get("commit"), str):
                raise ValueError("formal review row lacks raw reference")
            rank += 1
            case_name = f"{rank:03d}-{_safe_segment(skill_id)}-{disposition}-{candidate_id[:12]}"
            cases.append(
                {
                    "review_rank": rank,
                    "case_name": case_name,
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "candidate_index": candidate_index,
                    "scenario_id": scenario_id,
                    "skill_id": skill_id,
                    "seed_record_id": record_id,
                    "target_track_id": metrics.get("target_track_id"),
                    "disposition": disposition,
                    "first_failed_stage": metrics.get("first_failed_stage"),
                    "primary_rejection_reason": row.get("primary_rejection_reason"),
                    "quality_score": metrics.get("quality_score"),
                    "source_path": record.source_path,
                    "raw": dict(raw),
                    "review_status": "pending",
                    "reviewer": "",
                    "notes": "",
                }
            )
    if len(cases) < 100:
        raise ValueError(f"formal review manifest has only {len(cases)} cases; expected at least 100")
    if len(cases) > 204:
        raise ValueError(f"formal review manifest has {len(cases)} cases; maximum is 204")
    manifest = {
        "schema_version": FORMAL_REVIEW_SCHEMA_VERSION,
        "kind": "formal_generation_review_manifest",
        "contract": FORMAL_REVIEW_CONTRACT,
        "status": "pending_manual_review",
        "formal_plan_id": summary["formal_plan_id"],
        "formal_summary_sha256": file_sha256(run / "summary.json"),
        "generation_config_sha256": file_sha256(config_path),
        "case_count": len(cases),
        "accepted_case_count": sum(case["disposition"] == "accepted" for case in cases),
        "rejected_case_count": sum(case["disposition"] == "rejected" for case in cases),
        "skills_with_review_cases": sorted(rows_by_skill),
        "formal_train_only": True,
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "cases": cases,
    }
    manifest_path = destination / "review_manifest.json"
    destination.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing formal review manifest differs from current evidence")
    if not manifest_path.exists():
        manifest_path.write_text(payload, encoding="utf-8")
    return manifest_path, manifest


def render_formal_review(
    *,
    run_root: str | Path,
    repository_root: str | Path = ".",
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    output_root: str | Path | None = None,
    per_disposition: int = 3,
) -> dict[str, Any]:
    """Render and index deterministic source/generated BEV review pairs."""

    root = Path(repository_root).resolve()
    run = Path(run_root).resolve()
    manifest_path, manifest = build_formal_review_manifest(
        run_root=run,
        repository_root=root,
        generation_config_path=generation_config_path,
        output_root=output_root,
        per_disposition=per_disposition,
    )
    destination = manifest_path.parent
    config = load_counterfactual_config(_resolved(root, generation_config_path), repository_root=root)
    records = read_seed_records(_resolved(root, config.inputs.seed_manifest))
    records_by_id = {seed_record_id(record): record for record in records}
    data_root = _resolved(root, config.inputs.data_root)
    scenario_cache: dict[str, Any] = {}
    shard_cache: dict[Path, tuple[Any, ...]] = {}
    cases: list[dict[str, Any]] = []
    rendered = 0
    resumed = 0
    for case in manifest["cases"]:
        record = records_by_id[case["seed_record_id"]]
        commit_path = (run / case["raw"]["commit"]).resolve()
        if run not in commit_path.parents:
            raise ValueError("formal review raw reference escapes run root")
        candidates = shard_cache.get(commit_path)
        if candidates is None:
            candidates = load_raw_shard_candidates(commit_path)
            shard_cache[commit_path] = candidates
        offset = case["raw"].get("offset")
        if not isinstance(offset, int) or not 0 <= offset < len(candidates):
            raise ValueError("formal review raw offset is invalid")
        candidate = candidates[offset]
        if candidate.candidate_id != case["candidate_id"]:
            raise ValueError("formal review candidate identity mismatch")
        source = scenario_cache.get(case["scenario_id"])
        if source is None:
            source = load_av2_scenario(_source_path(data_root, record.source_path, case["scenario_id"]))
            scenario_cache[case["scenario_id"]] = source
        overlay = materialize_overlay_scenario(source, candidate.target_track_id, candidate.future_xy_global)
        case_root = destination / "cases" / case["case_name"]
        source_dir = case_root / "source"
        generated_dir = case_root / "generated"
        source_png = next(iter(source_dir.glob("*.png")), None) if source_dir.is_dir() else None
        generated_png = next(iter(generated_dir.glob("*.png")), None) if generated_dir.is_dir() else None
        if source_png is not None and generated_png is not None and _valid_png(source_png) and _valid_png(generated_png):
            resumed += 1
        else:
            source_png = render_seed_review(source, record, source_dir)
            generated_png = render_seed_review(overlay, record, generated_dir)
            if not _valid_png(source_png) or not _valid_png(generated_png):
                raise ValueError("formal review renderer did not produce valid PNG files")
            rendered += 1
        cases.append(
            {
                **case,
                "source_png": {"path": _path_label(root, source_png), "sha256": file_sha256(source_png)},
                "generated_png": {"path": _path_label(root, generated_png), "sha256": file_sha256(generated_png)},
            }
        )
    result = {
        "schema_version": FORMAL_REVIEW_SCHEMA_VERSION,
        "kind": "formal_generation_review",
        "contract": FORMAL_REVIEW_CONTRACT,
        "status": "pending_manual_review",
        "formal_plan_id": manifest["formal_plan_id"],
        "review_manifest": {"path": _path_label(root, manifest_path), "sha256": file_sha256(manifest_path)},
        "case_count": len(cases),
        "image_count": 2 * len(cases),
        "rendered_case_count": rendered,
        "resumed_case_count": resumed,
        "manual_review_status": "pending",
        "validation_manifests_opened": False,
        "final_validation_accessed": False,
        "cases": cases,
    }
    summary_path = destination / "summary.json"
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") != payload:
        raise ValueError("existing formal review summary differs from current evidence")
    if not summary_path.exists():
        summary_path.write_text(payload, encoding="utf-8")
    return {**result, "output_path": _path_label(root, summary_path)}


REVIEW_CRITERIA = (
    "history_invariants",
    "road_relation",
    "motion_continuity",
    "skill_role",
    "target_risk",
    "parameter_realization",
    "background_interaction",
    "visual_artifacts",
)
REVIEW_TEMPLATE_COLUMNS = (
    "review_rank",
    "case_name",
    "disposition",
    "skill_id",
    "scenario_id",
    "candidate_id",
    "first_failed_stage",
    "source_png",
    "generated_png",
    *REVIEW_CRITERIA,
    "review_status",
    "reviewer",
    "issue_categories",
    "notes",
)
REVIEW_STATUSES = frozenset({"passed", "failed", "uncertain"})
REVIEW_CRITERION_STATUSES = frozenset({"pass", "fail", "not_applicable", "uncertain"})


def _image_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not _valid_png(path):
        raise ValueError(f"review image is not a valid PNG: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to verify review image: {path}: {error}") from error
    actual_sha256 = file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"review image hash mismatch: {path}")
    return {"path": path.as_posix(), "sha256": actual_sha256, "width": width, "height": height, "mode": mode}


def audit_formal_review(
    *,
    summary_path: str | Path,
    repository_root: str | Path = ".",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify review manifests, PNGs and hashes without loading AV2 trajectories."""

    root = Path(repository_root).resolve()
    summary_file = Path(summary_path).resolve()
    summary = _read_json(summary_file, "formal review summary")
    if summary.get("kind") != "formal_generation_review":
        raise ValueError("not a formal generation review summary")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("formal review summary has no cases")
    ranks: set[int] = set()
    case_names: set[str] = set()
    candidate_ids: set[str] = set()
    candidate_order: list[str] = []
    image_records: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("formal review case must be an object")
        rank = case.get("review_rank")
        case_name = case.get("case_name")
        candidate_id = case.get("candidate_id")
        if not isinstance(rank, int) or rank <= 0 or rank in ranks:
            raise ValueError("formal review ranks must be unique positive integers")
        if not isinstance(case_name, str) or not case_name or case_name in case_names:
            raise ValueError("formal review case names must be unique")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise ValueError("formal review candidate IDs must be unique")
        ranks.add(rank)
        case_names.add(case_name)
        candidate_ids.add(candidate_id)
        candidate_order.append(candidate_id)
        for field in ("source_png", "generated_png"):
            reference = case.get(field)
            if not isinstance(reference, Mapping):
                raise ValueError(f"formal review case is missing {field}")
            path_value = reference.get("path")
            sha256 = reference.get("sha256")
            if not isinstance(path_value, str) or not isinstance(sha256, str):
                raise ValueError(f"invalid {field} reference in {case_name}")
            path = Path(path_value)
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            image_records.append(_image_reference(resolved, sha256))
    if ranks != set(range(1, len(cases) + 1)):
        raise ValueError("formal review ranks must be contiguous")
    manifest_path_value = summary.get("review_manifest", {}).get("path")
    manifest_sha256 = summary.get("review_manifest", {}).get("sha256")
    if not isinstance(manifest_path_value, str) or not isinstance(manifest_sha256, str):
        raise ValueError("formal review summary is missing its manifest reference")
    manifest_path = Path(manifest_path_value)
    manifest_path = manifest_path.resolve() if manifest_path.is_absolute() else (root / manifest_path).resolve()
    if not manifest_path.is_file() or file_sha256(manifest_path) != manifest_sha256:
        raise ValueError("formal review manifest is absent or has changed")
    manifest = _read_json(manifest_path, "formal review manifest")
    manifest_cases = manifest.get("cases")
    if not isinstance(manifest_cases, list) or [case.get("candidate_id") for case in manifest_cases] != candidate_order:
        raise ValueError("formal review summary and manifest cases differ")
    result = {
        "schema_version": FORMAL_REVIEW_SCHEMA_VERSION,
        "kind": "formal_generation_review_audit",
        "contract": FORMAL_REVIEW_CONTRACT,
        "status": "automated_audit_passed",
        "summary": {"path": summary_file.as_posix(), "sha256": file_sha256(summary_file)},
        "manifest": {"path": manifest_path.as_posix(), "sha256": manifest_sha256},
        "case_count": len(cases),
        "image_count": len(image_records),
        "verified_image_count": len(image_records),
        "manual_review_status": summary.get("manual_review_status", "pending"),
        "validation_manifests_opened": summary.get("validation_manifests_opened"),
        "final_validation_accessed": summary.get("final_validation_accessed"),
        "images": image_records,
    }
    destination = Path(output_path).resolve() if output_path is not None else summary_file.parent / "audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != payload:
        raise ValueError("existing formal review audit differs from current evidence")
    if not destination.exists():
        destination.write_text(payload, encoding="utf-8")
    return {**result, "output_path": destination.as_posix()}


def write_review_template(*, summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Create a stable CSV for human review; never overwrite completed annotations."""

    summary_file = Path(summary_path).resolve()
    summary = _read_json(summary_file, "formal review summary")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("formal review summary has no cases")
    destination = Path(output_path).resolve() if output_path is not None else summary_file.parent / "manual_review.csv"
    rows = []
    for case in cases:
        rows.append(
            {
                "review_rank": case["review_rank"],
                "case_name": case["case_name"],
                "disposition": case["disposition"],
                "skill_id": case["skill_id"],
                "scenario_id": case["scenario_id"],
                "candidate_id": case["candidate_id"],
                "first_failed_stage": case.get("first_failed_stage") or "",
                "source_png": case["source_png"]["path"],
                "generated_png": case["generated_png"]["path"],
                **{criterion: "" for criterion in REVIEW_CRITERIA},
                "review_status": case.get("review_status", "pending"),
                "reviewer": case.get("reviewer", ""),
                "issue_categories": "",
                "notes": case.get("notes", ""),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open(encoding="utf-8", newline="") as handle:
            columns = tuple(csv.DictReader(handle).fieldnames or ())
        if columns != REVIEW_TEMPLATE_COLUMNS:
            raise ValueError("existing manual review CSV uses an incompatible column contract")
        return destination
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_TEMPLATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def finalize_review_annotations(
    *,
    summary_path: str | Path,
    annotations_path: str | Path,
    output_path: str | Path | None = None,
    minimum_reviews: int = 100,
    review_method: str = "manual",
) -> dict[str, Any]:
    """Validate review annotations and write a separately versioned summary."""

    if isinstance(minimum_reviews, bool) or not isinstance(minimum_reviews, int) or minimum_reviews <= 0:
        raise ValueError("minimum_reviews must be a positive integer")
    if review_method not in {"manual", "automated_evidence"}:
        raise ValueError("review_method must be manual or automated_evidence")
    summary_file = Path(summary_path).resolve()
    annotations_file = Path(annotations_path).resolve()
    summary = _read_json(summary_file, "formal review summary")
    cases = summary.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("formal review summary has no cases")
    expected = {case["case_name"]: case for case in cases}
    annotations: dict[str, dict[str, str]] = {}
    seen_cases: set[str] = set()
    try:
        handle = annotations_file.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError(f"failed to read manual review CSV: {annotations_file}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REVIEW_TEMPLATE_COLUMNS:
            raise ValueError("manual review CSV columns do not match the review template")
        for row in reader:
            case_name = row.get("case_name", "")
            if case_name in seen_cases or case_name not in expected:
                raise ValueError(f"manual review CSV has an unknown or duplicate case: {case_name}")
            seen_cases.add(case_name)
            status = row.get("review_status", "").strip().lower()
            reviewer = row.get("reviewer", "").strip()
            if row.get("candidate_id", "") != expected[case_name]["candidate_id"]:
                raise ValueError(f"manual review candidate does not match {case_name}")
            if status in {"", "pending"}:
                if reviewer or any(row.get(criterion, "").strip() for criterion in REVIEW_CRITERIA):
                    raise ValueError(f"pending manual review has completed fields for {case_name}")
                continue
            if status not in REVIEW_STATUSES:
                raise ValueError(f"manual review status is invalid for {case_name}: {status}")
            if not reviewer:
                raise ValueError(f"manual review reviewer is missing for {case_name}")
            criterion_values = {
                criterion: row.get(criterion, "").strip().lower()
                for criterion in REVIEW_CRITERIA
            }
            invalid = {
                criterion: value
                for criterion, value in criterion_values.items()
                if value not in REVIEW_CRITERION_STATUSES
            }
            if invalid:
                raise ValueError(f"manual review criteria are invalid for {case_name}: {invalid}")
            has_failure = "fail" in criterion_values.values()
            has_uncertain = "uncertain" in criterion_values.values()
            expected_status = "failed" if has_failure else "uncertain" if has_uncertain else "passed"
            if status != expected_status:
                raise ValueError(
                    f"manual review status disagrees with criteria for {case_name}: "
                    f"{status} vs {expected_status}"
                )
            annotations[case_name] = {
                **criterion_values,
                "review_status": status,
                "reviewer": reviewer,
                "issue_categories": row.get("issue_categories", "").strip(),
                "notes": row.get("notes", "").strip(),
            }
    if seen_cases != set(expected):
        raise ValueError("manual review CSV does not contain every review case")
    reviewed_count = len(annotations)
    if reviewed_count < minimum_reviews:
        raise ValueError(
            f"manual review has {reviewed_count} completed cases; "
            f"at least {minimum_reviews} are required"
        )
    output_cases = [
        {**case, **annotations[case["case_name"]]}
        if case["case_name"] in annotations
        else case
        for case in cases
    ]
    reviewer_counts = Counter(item["reviewer"] for item in annotations.values())
    criterion_counts = {
        criterion: dict(
            sorted(Counter(item[criterion] for item in annotations.values()).items())
        )
        for criterion in REVIEW_CRITERIA
    }
    status_counts = Counter(item["review_status"] for item in annotations.values())
    review_prefix = "manual" if review_method == "manual" else "automated"
    result: dict[str, Any] = {
        **summary,
        "status": f"{review_method}_review_completed_minimum",
        "review_method": review_method,
        "review_status": "completed_minimum",
        "review_count": reviewed_count,
        "review_minimum": minimum_reviews,
        "review_reviewer_counts": dict(sorted(reviewer_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
        "review_criterion_counts": criterion_counts,
        "review_annotations": {
            "path": annotations_file.as_posix(),
            "sha256": file_sha256(annotations_file),
        },
        "cases": output_cases,
    }
    result.update(
        {
            f"{review_prefix}_review_status": "completed_minimum",
            f"{review_prefix}_review_count": reviewed_count,
        }
    )
    destination = (
        Path(output_path).resolve()
        if output_path is not None
        else summary_file.parent / "manual_review_summary.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != payload:
        raise ValueError("existing finalized review summary differs from current annotations")
    if not destination.exists():
        destination.write_text(payload, encoding="utf-8")
    return {**result, "output_path": destination.as_posix()}


__all__ = [
    "FORMAL_REVIEW_CONTRACT",
    "FORMAL_REVIEW_SCHEMA_VERSION",
    "build_formal_review_manifest",
    "audit_formal_review",
    "finalize_review_annotations",
    "render_formal_review",
    "write_review_template",
]

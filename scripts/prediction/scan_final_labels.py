"""Scan frozen observed-trigger labels on Final Validation after formal freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from scripts.seed_detection.detect_seeds import (
    _entry_records,
    _initialize_scan_worker,
    _load_confirmed_skills,
    _scan_worker_task,
)
from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.data.manifests import read_manifest
from skilldrive.prediction.audit import file_sha256
from skilldrive.seeds.records import SeedRecord, write_seed_records
from skilldrive.skills.detection import detect_scenario, load_detection_config


def _identity(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _read_checkpoint(path: Path, identity: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if line_number == 1:
            if value != {"kind": "header", "identity": identity}:
                raise ValueError("Final Validation label checkpoint identity differs")
            continue
        if value.get("kind") != "scenario" or value.get("manifest_index") != len(entries):
            raise ValueError("Final Validation label checkpoint is not a complete prefix")
        entries.append(value)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("manifests/splits/final_validation.csv")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/av2/motion-forecasting")
    )
    parser.add_argument("--skills-dir", type=Path, default=Path("configs/skills"))
    parser.add_argument("--config", type=Path, default=Path("configs/seed_detection.yaml"))
    parser.add_argument(
        "--formal-audit",
        type=Path,
        default=Path("manifests/prediction/formal_run_audit_v1.json"),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    audit = json.loads(args.formal_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("run_count") != 12:
        raise ValueError("all 12 formal runs must pass audit before Final Validation")
    rows = read_manifest(args.manifest)
    if len(rows) != 5_000 or any(row.split != "validation" for row in rows):
        raise ValueError("canonical Final Validation must contain 5,000 validation rows")
    all_skills = _load_confirmed_skills(args.skills_dir)
    observed = [skill for skill in all_skills if skill.detection["mode"] == "observed_trigger"]
    compatible = [skill.skill_id for skill in all_skills if skill.detection["mode"] == "compatible_seed"]
    identity = _identity(
        [args.manifest, args.config, args.formal_audit, args.skills_dir / "catalog.yaml"]
        + [args.skills_dir / f"{skill.skill_id}.yaml" for skill in all_skills]
    )
    completed = _read_checkpoint(args.checkpoint, identity)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.checkpoint.open(mode, encoding="utf-8", buffering=1) as handle:
        if not completed:
            handle.write(json.dumps({"kind": "header", "identity": identity}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        config = load_detection_config(args.config)
        risk_definitions = {skill.skill_id: skill.risk_definition for skill in observed}
        started = time.perf_counter()
        last_report = started
        tasks = list(enumerate(rows))[len(completed) :]
        if tasks:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_scan_worker,
                initargs=(
                    args.data_root,
                    observed,
                    config,
                    risk_definitions,
                    load_av2_scenario,
                    detect_scenario,
                ),
            ) as executor:
                for entry in executor.map(_scan_worker_task, tasks, chunksize=1):
                    if entry["manifest_index"] != len(completed):
                        raise ValueError("parallel Final Validation scan order changed")
                    handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                    completed.append(entry)
                    if len(completed) % 64 == 0:
                        handle.flush()
                        os.fsync(handle.fileno())
                    now = time.perf_counter()
                    if now - last_report >= 10 or len(completed) == len(rows):
                        elapsed = max(now - started, 1e-9)
                        rate = (len(completed) - (len(rows) - len(tasks))) / elapsed
                        eta = (len(rows) - len(completed)) / rate if rate else float("inf")
                        print(
                            f"\rfinal labels {len(completed)}/{len(rows)} "
                            f"{rate:.1f} scenarios/s ETA {eta:.0f}s checkpoint={args.checkpoint}",
                            end="",
                            flush=True,
                        )
                        last_report = now
        handle.flush()
        os.fsync(handle.fileno())
    print()
    records: list[SeedRecord] = []
    for entry in completed:
        records.extend(_entry_records(entry))
    write_seed_records(args.output_csv, records)
    counts = Counter(record.skill_id for record in records)
    payload = {
        "schema_version": 1,
        "kind": "final_validation_observed_skill_labels",
        "status": "complete",
        "scenario_count": len(rows),
        "record_count": len(records),
        "observed_skill_count": len(observed),
        "compatible_skill_count": len(compatible),
        "records_by_skill": {skill.skill_id: counts[skill.skill_id] for skill in observed},
        "compatible_seed_skills_not_claimed_as_real_events": compatible,
        "manifest_sha256": file_sha256(args.manifest),
        "formal_audit_sha256": file_sha256(args.formal_audit),
        "label_csv_sha256": file_sha256(args.output_csv),
        "checkpoint_identity": identity,
        "workers": args.workers,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Final Validation labels complete: {len(records)} records from {len(rows)} scenarios")


if __name__ == "__main__":
    main()

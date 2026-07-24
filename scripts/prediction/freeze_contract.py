"""Freeze the formal prediction contract with content fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from skilldrive.prediction.audit import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/prediction/formal_v1.json"))
    parser.add_argument("--input-audit", type=Path, default=Path("manifests/prediction/input_audit_v1.json"))
    parser.add_argument("--augmentation-manifest", type=Path, default=Path("manifests/prediction/augmentation_bundle_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("manifests/prediction/formal_contract_v1.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "frozen_before_final_validation":
        raise ValueError("formal configuration is not frozen")
    payload = {
        "schema_version": 1,
        "kind": "downstream_prediction_formal_contract",
        "status": "frozen",
        "config_path": args.config.as_posix(),
        "config_sha256": file_sha256(args.config),
        "input_audit_sha256": file_sha256(args.input_audit),
        "augmentation_manifest_sha256": file_sha256(args.augmentation_manifest),
        "experiments": config["experiments"],
        "final_validation_content_accessed_before_freeze": False,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["contract_id"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(payload["contract_id"])


if __name__ == "__main__":
    main()

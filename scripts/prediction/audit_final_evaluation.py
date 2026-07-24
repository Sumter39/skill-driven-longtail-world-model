"""Audit the frozen Final Validation prediction deliverables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from skilldrive.prediction.audit import file_sha256


def _walk_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite result at {path}")
    if isinstance(value, Mapping):
        for name, item in value.items():
            _walk_finite(item, f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _walk_finite(item, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=Path("manifests/prediction/final_evaluation_v1.json")
    )
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/prediction/final_evaluation_audit_v1.json"),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("post_final_validation_tuning_permitted") is not False:
        raise ValueError("Final Validation summary is not frozen and complete")
    final = summary["final_validation"]
    if final["base_samples"] != 5_000 or final["real_long_tail_samples"] != 2_279:
        raise ValueError("Final Validation sample counts differ from the frozen cache")
    state = json.loads((args.runtime_output / "state.json").read_text(encoding="utf-8"))
    if state.get("identity") != summary.get("evaluation_identity"):
        raise ValueError("runtime state and Final Validation summary identities differ")
    expected_names = {"constant_velocity", "lstm_e0_seed_2026"} | {
        f"transformer_{experiment}_seed_{seed}"
        for experiment in ("e0", "e1", "e2", "e3")
        for seed in (2026, 2027, 2028)
    }
    if set(state.get("completed", {})) != expected_names:
        raise ValueError("Final Validation model result set is incomplete")
    reference_ids: set[str] | None = None
    outputs: dict[str, Any] = {}
    for name in sorted(expected_names):
        descriptor = state["completed"][name]
        path = args.runtime_output / f"{name}.jsonl"
        if not path.is_file() or file_sha256(path) != descriptor["sha256"]:
            raise ValueError(f"Final Validation per-sample output hash differs: {name}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        ids = {row.get("sample_id") for row in rows}
        if len(rows) != 7_279 or len(ids) != 7_279 or None in ids:
            raise ValueError(f"Final Validation per-sample rows are incomplete: {name}")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Final Validation sample IDs differ: {name}")
        outputs[name] = {
            "rows": len(rows),
            "sha256": descriptor["sha256"],
            "checkpoint_sha256": descriptor["checkpoint_sha256"],
        }
    for comparison in ("e1_vs_e0", "e2_vs_e0", "e3_vs_e0", "e3_vs_e2"):
        for view, scenarios in (("overall", 5_000), ("real_long_tail", 1_839)):
            bootstrap = summary["comparisons"][comparison][view]["paired_bootstrap"]
            if bootstrap["scenario_count"] != scenarios or bootstrap["repetitions"] != 2_000:
                raise ValueError(f"paired bootstrap contract differs: {comparison}/{view}")
    _walk_finite(summary)
    payload = {
        "schema_version": 1,
        "kind": "downstream_prediction_final_evaluation_audit",
        "status": "complete",
        "evaluation_identity": summary["evaluation_identity"],
        "summary_sha256": file_sha256(args.summary),
        "model_result_count": len(outputs),
        "sample_ids_identical_across_models": True,
        "overall_samples": 5_000,
        "real_long_tail_samples": 2_279,
        "real_long_tail_scenarios": 1_839,
        "bootstrap_repetitions": 2_000,
        "outputs": outputs,
        "post_evaluation_model_selection_detected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Final Validation audit complete: 14/14 outputs, 7,279 identical samples")


if __name__ == "__main__":
    main()

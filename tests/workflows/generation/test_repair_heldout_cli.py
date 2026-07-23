from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.generation.run_counterfactual_pipeline as pipeline_module
import skilldrive.generation.heldout_gate as heldout_gate


CHECKPOINT_SHA256 = "a" * 64
RUN_MANIFEST_SHA256 = "b" * 64


def _execute_arguments() -> list[str]:
    return [
        "--stage",
        "repair-heldout-execute",
        "--repair-checkpoint-path",
        "candidate.pt",
        "--repair-checkpoint-sha256",
        CHECKPOINT_SHA256,
        "--repair-run-manifest-path",
        "run-manifest.json",
        "--repair-run-manifest-sha256",
        RUN_MANIFEST_SHA256,
        "--repair-checkpoint-mode",
        "formal",
    ]


@pytest.mark.parametrize(
    "removed_flag",
    (
        "--repair-checkpoint-path",
        "--repair-checkpoint-sha256",
        "--repair-run-manifest-path",
        "--repair-run-manifest-sha256",
    ),
)
def test_repair_heldout_execute_requires_all_hash_bound_checkpoint_arguments(
    monkeypatch: pytest.MonkeyPatch,
    removed_flag: str,
) -> None:
    arguments = _execute_arguments()
    index = arguments.index(removed_flag)
    del arguments[index : index + 2]
    monkeypatch.setattr(sys, "argv", ["run_counterfactual_pipeline.py", *arguments])
    monkeypatch.setattr(
        heldout_gate,
        "execute_repair_heldout_plan",
        lambda **kwargs: pytest.fail("invalid CLI arguments must not start execution"),
    )

    with pytest.raises(ValueError, match=removed_flag):
        pipeline_module.main()


@pytest.mark.parametrize("mode", (None, "diagnostic-overfit"))
def test_repair_heldout_execute_requires_formal_checkpoint_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str | None,
) -> None:
    arguments = _execute_arguments()
    mode_index = arguments.index("--repair-checkpoint-mode")
    del arguments[mode_index : mode_index + 2]
    if mode is not None:
        arguments.extend(("--repair-checkpoint-mode", mode))
    monkeypatch.setattr(sys, "argv", ["run_counterfactual_pipeline.py", *arguments])
    monkeypatch.setattr(
        heldout_gate,
        "execute_repair_heldout_plan",
        lambda **kwargs: pytest.fail("non-formal mode must not start execution"),
    )

    with pytest.raises(ValueError, match="repair-formal epoch checkpoint"):
        pipeline_module.main()


def test_repair_heldout_execute_forwards_only_heldout_execution_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, object] = {}

    def execute(**kwargs):
        received.update(kwargs)
        return {
            "status": "completed",
            "outputs": {"execution_summary": "heldout/execution-summary.json"},
        }

    monkeypatch.setattr(heldout_gate, "execute_repair_heldout_plan", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_counterfactual_pipeline.py",
            *_execute_arguments(),
            "--config",
            "cfg/generation.yaml",
            "--filter-config",
            "cfg/filter.yaml",
            "--detection-config",
            "cfg/detection.yaml",
            "--repair-heldout-source-plan",
            "manifests/heldout-plan",
            "--repair-audit",
            "manifests/repair-audit.json",
            "--repair-heldout-output-root",
            "outputs/repair-heldout",
            "--output-root",
            "outputs/must-not-be-used",
            "--device",
            "cpu",
            "--task-batch-size",
            "17",
            "--progress-interval-seconds",
            "2.5",
        ],
    )

    pipeline_module.main()

    assert received == {
        "checkpoint_path": Path("candidate.pt"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "run_manifest_path": Path("run-manifest.json"),
        "run_manifest_sha256": RUN_MANIFEST_SHA256,
        "config_path": Path("cfg/generation.yaml"),
        "filter_config_path": Path("cfg/filter.yaml"),
        "detection_config_path": Path("cfg/detection.yaml"),
        "source_plan_dir": Path("manifests/heldout-plan"),
        "repair_audit_path": Path("manifests/repair-audit.json"),
        "output_root": Path("outputs/repair-heldout"),
        "device": "cpu",
        "task_batch_size": 17,
        "progress_interval_seconds": 2.5,
    }


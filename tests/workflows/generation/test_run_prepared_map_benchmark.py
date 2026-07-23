from __future__ import annotations

import sys
from pathlib import Path

import scripts.generation.run_prepared_map_benchmark as cli_module


def test_cli_passes_optional_expected_decision_digest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    captured = {}
    config = object()
    digest = "d" * 64
    summary_path = tmp_path / "summary.json"
    summary = {
        "aggregate": {
            "decision_sha256": digest,
            "formal_projection": {"hours_p50": 1.25, "hours_p95": 1.5},
        }
    }
    monkeypatch.setattr(cli_module, "load_performance_config", lambda path: config)

    def run(
        received_config,
        *,
        config_path,
        workload_path,
        repository_root,
        expected_decision_sha256,
        map_batch_size,
    ):
        captured.update(
            {
                "config": received_config,
                "config_path": config_path,
                "workload_path": workload_path,
                "repository_root": repository_root,
                "expected": expected_decision_sha256,
                "map_batch_size": map_batch_size,
            }
        )
        return summary_path, summary

    monkeypatch.setattr(cli_module, "run_cpu_filter_prepared_map_benchmark", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_prepared_map_benchmark",
            "--config",
            "config.yaml",
            "--workload",
            "workload.json",
            "--repository-root",
            str(tmp_path),
            "--expected-decision-sha256",
            digest,
            "--map-batch-size",
            "32",
        ],
    )

    cli_module.main()

    assert captured["config"] is config
    assert captured["config_path"] == (tmp_path / "config.yaml").resolve()
    assert captured["workload_path"] == (tmp_path / "workload.json").resolve()
    assert captured["repository_root"] == tmp_path.resolve()
    assert captured["expected"] == digest
    assert captured["map_batch_size"] == 32
    output = capsys.readouterr().out
    assert "decision_sha256=" + digest in output
    assert "p50=1.250h" in output

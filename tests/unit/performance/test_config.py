from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skilldrive.performance.config import load_performance_config


CONFIG_PATH = Path("configs/generation/performance_v1.yaml")


def _raw() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_loads_frozen_performance_contract() -> None:
    config = load_performance_config()

    assert config.workload.max_tasks == 512
    assert config.workload.selection_contract == (
        "deterministic_atomic_condition_pairs_v1"
    )
    assert config.benchmark.runner == "cpu_filter_legacy"
    assert config.benchmark.repeats == 3
    assert config.benchmark.formal_candidate_count == 542_624
    assert "ab25dabcef82a936" in config.inputs.pilot_summary.as_posix()
    assert config.inputs.pilot_summary.name == "summary.json"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["workload"].update({"max_tasks": 513}), "<= 512"),
        (lambda value: value["benchmark"].update({"repeats": 2}), "exactly 3"),
        (
            lambda value: value["benchmark"].update(
                {"formal_candidate_count": 542_623}
            ),
            "542624",
        ),
        (
            lambda value: value["inputs"].update(
                {"generation_config": "../counterfactual.yaml"}
            ),
            "repository-relative",
        ),
        (
            lambda value: value["inputs"].update(
                {"pilot_summary": "../pilot/summary.json"}
            ),
            "repository-relative",
        ),
    ],
)
def test_rejects_performance_contract_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    value = _raw()
    mutation(value)
    path = tmp_path / "performance.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_performance_config(path)

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest
import yaml

from skilldrive.training import CVAEConfig, load_cvae_config


CONFIG_PATH = Path("configs/models/cvae_baseline.yaml")


def _raw_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_config(tmp_path: Path, value: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    parent: dict[str, Any] = value
    for key in path[:-1]:
        child = parent[key]
        assert isinstance(child, dict)
        parent = child
    parent[path[-1]] = replacement


def test_loads_repository_config_as_frozen_dataclasses_and_relative_paths() -> None:
    config = load_cvae_config()

    assert isinstance(config, CVAEConfig)
    assert config.version == 1
    assert config.tensorization.history_steps == 50
    assert config.tensorization.map_types == (
        "lane_centerline",
        "pedestrian_crossing",
        "drivable_area",
    )
    assert config.benchmark.worker_candidates == (0, 4, 8)
    assert config.data.root == Path("data/av2/motion-forecasting")
    assert not config.data.root.is_absolute()
    assert config.outputs.formal == Path("outputs/modeling/cvae_baseline/formal")
    with pytest.raises(FrozenInstanceError):
        config.training.batch_size = 8  # type: ignore[misc]


def test_canonical_dict_and_fingerprint_are_stable_across_yaml_key_order(
    tmp_path: Path,
) -> None:
    raw = _raw_config()
    reversed_raw = {
        key: dict(reversed(list(value.items()))) if isinstance(value, dict) else value
        for key, value in reversed(list(raw.items()))
    }

    first = load_cvae_config(CONFIG_PATH)
    second = load_cvae_config(_write_config(tmp_path, reversed_raw))

    assert first.to_canonical_dict() == second.to_canonical_dict()
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.to_canonical_dict()["data"]["root"] == "data/av2/motion-forecasting"
    assert first.to_canonical_dict()["benchmark"]["worker_candidates"] == [0, 4, 8]


def test_fingerprint_changes_when_a_semantic_value_changes(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["training"]["batch_size"] = 8

    assert load_cvae_config(CONFIG_PATH).fingerprint != load_cvae_config(
        _write_config(tmp_path, raw)
    ).fingerprint


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "unknown"),
        ("training", "unknown"),
        ("data", "unknown"),
    ],
)
def test_rejects_unknown_keys(
    tmp_path: Path,
    section: str | None,
    key: str,
) -> None:
    raw = _raw_config()
    target = raw if section is None else raw[section]
    target[key] = 1

    with pytest.raises(ValueError, match="unknown keys"):
        load_cvae_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "model"),
        ("training", "batch_size"),
        ("data", "manifests"),
    ],
)
def test_rejects_missing_keys(
    tmp_path: Path,
    section: str | None,
    key: str,
) -> None:
    raw = _raw_config()
    target = raw if section is None else raw[section]
    del target[key]

    with pytest.raises(ValueError, match="missing keys"):
        load_cvae_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("training", "batch_size"), True, "must be an integer"),
        (("training", "amp"), 1, "must be a boolean"),
        (("training", "learning_rate"), "fast", "finite number"),
        (("tensorization", "map_types"), "lane", "sequence of strings"),
        (("data", "root"), 3, "non-empty string"),
        (("model",), [], "must be a mapping"),
    ],
)
def test_rejects_wrong_types(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    message: str,
) -> None:
    raw = _raw_config()
    _set_nested(raw, path, replacement)

    with pytest.raises(ValueError, match=message):
        load_cvae_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("tensorization", "history_steps"), 0, "at least 1"),
        (("tensorization", "anchor_frame"), 50, "at most 49"),
        (("tensorization", "sample_period_s"), 0.0, "greater than 0.0"),
        (("model", "dropout"), 1.0, "less than 1.0"),
        (("model", "interaction_heads"), 3, "must be divisible"),
        (("loss", "endpoint_weight"), -1.0, "at least 0.0"),
        (("training", "learning_rate"), 0.0, "greater than 0.0"),
        (("training", "num_workers"), -1, "at least 0"),
        (("overfit", "sample_count"), 0, "at least 1"),
        (("benchmark", "worker_candidates"), [0, 2, 2], "duplicates"),
        (("benchmark", "batch_size_candidates"), [0, 8], "at least 1"),
    ],
)
def test_rejects_illegal_ranges_and_cross_field_values(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    message: str,
) -> None:
    raw = _raw_config()
    _set_nested(raw, path, replacement)

    with pytest.raises(ValueError, match=message):
        load_cvae_config(_write_config(tmp_path, raw))


def test_rejects_final_validation_as_any_training_or_selection_manifest(
    tmp_path: Path,
) -> None:
    raw = _raw_config()
    manifests = raw["data"]["manifests"]
    manifests["formal_train"] = manifests["final_validation"]

    with pytest.raises(ValueError, match="final_validation cannot be used"):
        load_cvae_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/data",
        "../data",
        "C:/data",
        "data\\windows",
    ],
)
def test_rejects_non_repository_relative_paths(tmp_path: Path, path: str) -> None:
    raw = _raw_config()
    raw["data"]["root"] = path

    with pytest.raises(ValueError, match="repository-relative"):
        load_cvae_config(_write_config(tmp_path, raw))


def test_rejects_output_directory_outside_output_root(tmp_path: Path) -> None:
    raw = _raw_config()
    raw["outputs"]["formal"] = "outputs/elsewhere"

    with pytest.raises(ValueError, match="contained by outputs.root"):
        load_cvae_config(_write_config(tmp_path, raw))

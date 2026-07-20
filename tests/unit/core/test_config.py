from pathlib import Path

import pytest

from skilldrive.config import load_paths


def test_local_paths_override_defaults(tmp_path: Path) -> None:
    default = tmp_path / "paths.example.yaml"
    local = tmp_path / "paths.local.yaml"
    default.write_text("data_root: /data\ncache_root: /cache\noutput_root: /output\n", encoding="utf-8")
    local.write_text("cache_root: /local-cache\n", encoding="utf-8")
    values = load_paths(default, local)
    assert values == {
        "data_root": "/data",
        "cache_root": "/local-cache",
        "output_root": "/output",
    }


def test_environment_path_has_highest_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    default = tmp_path / "paths.example.yaml"
    default.write_text("data_root: /data\ncache_root: /cache\noutput_root: /output\n", encoding="utf-8")
    monkeypatch.setenv("SKILLDRIVE_DATA_ROOT", "/environment-data")
    assert load_paths(default, tmp_path / "missing.yaml")["data_root"] == "/environment-data"


def test_unknown_path_key_is_rejected(tmp_path: Path) -> None:
    default = tmp_path / "paths.example.yaml"
    default.write_text(
        "data_root: /data\ncache_root: /cache\noutput_root: /output\nsurprise: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown"):
        load_paths(default, tmp_path / "missing.yaml")

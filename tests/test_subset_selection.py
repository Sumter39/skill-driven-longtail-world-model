import pytest

from scripts.download_av2_subset import (
    _is_retryable_network_error,
    _load_resume_ids,
    _uses_windows_s5cmd,
)
from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.data.subsets import select_ids


def test_subset_selection_is_deterministic_and_unique() -> None:
    ids = [f"scene-{index:03d}" for index in range(100)]
    first = select_ids(ids, count=20, seed=2026)
    second = select_ids(list(reversed(ids)), count=20, seed=2026)
    assert first == second
    assert len(first) == len(set(first)) == 20


def test_subset_selection_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_ids(["a"], count=0, seed=1)
    with pytest.raises(ValueError, match="only 1"):
        select_ids(["a"], count=2, seed=1)


def test_windows_s5cmd_detection() -> None:
    assert _uses_windows_s5cmd("/mnt/c/Users/example/s5cmd.exe")
    assert not _uses_windows_s5cmd("/home/example/.local/bin/s5cmd")


def test_resume_manifest_reuses_exact_ids(tmp_path) -> None:
    manifest = tmp_path / "train.csv"
    rows = [
        ManifestRow(str(index), "train", f"train/{index}", "unknown", "seed")
        for index in range(3)
    ]
    write_manifest(manifest, rows)

    assert _load_resume_ids(manifest, count=3, split="train") == ["0", "1", "2"]


def test_network_error_classification() -> None:
    assert _is_retryable_network_error("proxyconnect tcp: connection actively refused")
    assert _is_retryable_network_error("RequestError: send request failed caused by: EOF")
    assert not _is_retryable_network_error("AccessDenied: permission denied")

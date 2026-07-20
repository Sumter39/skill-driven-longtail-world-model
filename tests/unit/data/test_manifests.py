from pathlib import Path

import pytest

from skilldrive.data.manifests import ManifestRow, assert_disjoint, read_manifest, write_manifest


def test_manifest_round_trip(tmp_path: Path) -> None:
    rows = [ManifestRow("a", "train", "a.parquet", "MIA", "development")]
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    assert read_manifest(path) == rows


def test_manifest_leakage_detection() -> None:
    train = [ManifestRow("same", "train", "a", "MIA", "seed")]
    validation = [ManifestRow("same", "validation", "b", "MIA", "evaluation")]
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint(train, validation)

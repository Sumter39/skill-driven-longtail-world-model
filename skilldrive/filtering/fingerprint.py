"""Aggregate every semantic dependency that can change a filter decision."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from skilldrive.generation.config import load_counterfactual_config
from skilldrive.generation.contracts import canonical_sha256


FILTER_SEMANTIC_FINGERPRINT_VERSION = 1


@dataclass(frozen=True)
class FilterSemanticFingerprint:
    semantic_sha256: str
    file_sha256: Mapping[str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_filter_semantic_fingerprint(
    *,
    repository_root: str | Path = ".",
    generation_config_path: str | Path = "configs/generation/counterfactual_v1.yaml",
    filter_config_path: str | Path = "configs/generation/filters_v1.yaml",
    detection_config_path: str | Path = "configs/seed_detection.yaml",
    additional_paths: Iterable[str | Path] = (),
) -> FilterSemanticFingerprint:
    root = Path(repository_root).resolve()
    generation_path = (root / generation_config_path).resolve()
    config = load_counterfactual_config(generation_path, repository_root=root)
    paths = {
        generation_path,
        (root / filter_config_path).resolve(),
        (root / detection_config_path).resolve(),
        (root / config.formal_catalog).resolve(),
        (root / "skilldrive/generation/assembly.py").resolve(),
        (root / "skilldrive/generation/config.py").resolve(),
        (root / "skilldrive/generation/contracts.py").resolve(),
        (root / "skilldrive/generation/planning.py").resolve(),
        (root / "skilldrive/generation/scheduler.py").resolve(),
        (root / "skilldrive/generation/storage.py").resolve(),
        (root / "skilldrive/data/av2_reader.py").resolve(),
        (root / "skilldrive/data/coordinates.py").resolve(),
        (root / "skilldrive/schemas/core.py").resolve(),
        (root / "skilldrive/seeds/records.py").resolve(),
        (root / "skilldrive/skills/detection.py").resolve(),
        (root / "skilldrive/skills/geometry.py").resolve(),
        (root / "skilldrive/skills/loader.py").resolve(),
        (root / "scripts/generation/run_counterfactual_pipeline.py").resolve(),
    }
    if config.active_checkpoint.promotion_recommendation is not None:
        paths.add(
            (root / config.active_checkpoint.promotion_recommendation).resolve()
        )
    skill_directory = (root / config.formal_catalog).resolve().parent
    paths.update(
        (skill_directory / f"{skill_id}.yaml").resolve()
        for skill_id in config.formal_skill_ids
    )
    paths.update(path.resolve() for path in (root / "skilldrive/filtering").glob("*.py"))
    paths.update((root / path).resolve() for path in additional_paths)

    hashes: dict[str, str] = {}
    for path in sorted(paths):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"filter semantic dependency escapes repository: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"filter semantic dependency is missing: {path}")
        hashes[relative] = _sha256(path)
    semantic = canonical_sha256(
        {
            "version": FILTER_SEMANTIC_FINGERPRINT_VERSION,
            "files": hashes,
        }
    )
    return FilterSemanticFingerprint(
        semantic_sha256=semantic,
        file_sha256=MappingProxyType(hashes),
    )


__all__ = [
    "FILTER_SEMANTIC_FINGERPRINT_VERSION",
    "FilterSemanticFingerprint",
    "build_filter_semantic_fingerprint",
]

"""Prepare deterministic, resumable conditional CVAE scenario caches."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from skilldrive.data.cvae_cache import (
    ScenarioLoader,
    ValidationLabeler,
    prepare_cvae_split,
)
from skilldrive.data.cvae_samples import (
    CVAESchema,
    SampleSpec,
    build_cvae_schema,
    observed_sample_specs,
)
from skilldrive.schemas import Scenario
from skilldrive.skills.detection import detect_scenario, load_detection_config
from skilldrive.training import DEFAULT_CVAE_CONFIG, load_cvae_config


class FrozenObservedValidationLabeler:
    """Apply the frozen 14 observed-trigger rules without creating seed manifests."""

    def __init__(self, *, config_path: Path, schema: CVAESchema) -> None:
        self.detection_config = load_detection_config(config_path)
        self.skills = tuple(
            skill
            for skill in schema.formal_skills
            if skill.detection["mode"] == "observed_trigger"
        )
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.cache_identity = (
            f"frozen_observed_only:{digest}:"
            + ",".join(skill.skill_id for skill in self.skills)
        )

    def __call__(
        self,
        scenario: Scenario,
        schema: CVAESchema,
    ) -> tuple[SampleSpec, ...]:
        run = detect_scenario(scenario, list(self.skills), self.detection_config)
        return observed_sample_specs(run.records, schema)


def run_preparation(
    *,
    config_path: str | Path = DEFAULT_CVAE_CONFIG,
    split: str,
    project_root: str | Path = ".",
    schema: CVAESchema | None = None,
    scenario_loader: ScenarioLoader | None = None,
    validation_labeler: ValidationLabeler | None = None,
    limit: int | None = None,
    force: bool = False,
    cache_root: str | Path | None = None,
    progress_stream: TextIO | None = None,
) -> dict:
    """Load configuration and prepare one development or formal cache pair."""
    config = load_cvae_config(config_path)
    if cache_root is not None:
        config = replace(
            config,
            cache=replace(config.cache, root=Path(cache_root)),
        )
    root = Path(project_root)
    if schema is None:
        schema = build_cvae_schema(root / config.data.skill_dir)
    detection_path = root / config.data.detection_config
    if validation_labeler is None and detection_path.is_file():
        validation_labeler = FrozenObservedValidationLabeler(
            config_path=detection_path,
            schema=schema,
        )
    kwargs = {}
    if scenario_loader is not None:
        kwargs["scenario_loader"] = scenario_loader
    return prepare_cvae_split(
        config,
        split,
        project_root=root,
        schema=schema,
        validation_labeler=validation_labeler,
        limit=limit,
        force=force,
        progress_stream=progress_stream,
        **kwargs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic scenario caches for conditional CVAE training."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CVAE_CONFIG)
    parser.add_argument("--split", choices=("development", "formal"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-root", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    summary = run_preparation(
        config_path=args.config,
        split=args.split,
        limit=args.limit,
        force=args.force,
        cache_root=args.cache_root,
    )
    counts = {
        name: value["counts"]["retained_samples"]
        for name, value in summary["partitions"].items()
    }
    print(f"CVAE cache preparation complete: split={args.split}, samples={counts}")


if __name__ == "__main__":
    main()

"""Freeze exact latent-search representatives from one trusted Pilot summary."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.generation import load_counterfactual_config
from skilldrive.generation.latent_search import (
    build_latent_search_manifest,
    load_latent_search_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/counterfactual_v1.yaml"),
    )
    parser.add_argument(
        "--latent-search-config",
        type=Path,
        default=Path("configs/generation/latent_search_v1.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/generation/latent_search_representatives_v1.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generation_config = load_counterfactual_config(args.config)
    search_config = load_latent_search_config(args.latent_search_config)
    manifest = build_latent_search_manifest(
        pilot_summary_path=args.pilot_summary,
        output_path=args.output,
        config=search_config,
        repository_root=Path.cwd(),
        none_skill_id=generation_config.none_skill_id,
    )
    print(f"latent-search representative manifest: {manifest.path}")
    print(f"sha256: {manifest.sha256}")


if __name__ == "__main__":
    main()

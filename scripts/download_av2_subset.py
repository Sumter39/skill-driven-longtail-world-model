"""Create and optionally execute a deterministic s5cmd AV2 subset download."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from skilldrive.data.manifests import ManifestRow, write_manifest
from skilldrive.data.subsets import select_ids


S3_ROOT = "s3://argoverse/datasets/av2/motion-forecasting"
SCENARIO_PATTERN = re.compile(r"/([0-9a-f-]+)/scenario_\1\.parquet$")


def _list_scenario_ids(s5cmd: str, split: str, cache: Path) -> list[str]:
    if cache.exists():
        return [line.strip() for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = subprocess.run(
        [
            s5cmd,
            "--no-sign-request",
            "ls",
            f"{S3_ROOT}/{split}/*/scenario_*.parquet",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ids: list[str] = []
    for line in result.stdout.splitlines():
        source = line.split()[-1]
        match = SCENARIO_PATTERN.search(source)
        if match:
            ids.append(match.group(1))
    if not ids:
        raise RuntimeError("s5cmd returned no AV2 scenario IDs")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(sorted(set(ids))) + "\n", encoding="utf-8")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("/mnt/d/datasets/av2/motion-forecasting"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--listing-cache",
        type=Path,
        default=None,
        help="Defaults to data/metadata/av2_<split>_scenario_ids.txt.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    s5cmd = shutil.which("s5cmd")
    if not s5cmd:
        raise RuntimeError("s5cmd is not on PATH; install it as described in docs/data/argoverse2.md")
    cache = args.listing_cache or Path(f"data/metadata/av2_{args.split}_scenario_ids.txt")
    selected = select_ids(_list_scenario_ids(s5cmd, args.split, cache), args.count, args.seed)

    if args.manifest.exists() and not args.force_manifest:
        raise FileExistsError(f"manifest already exists: {args.manifest}; pass --force-manifest to replace it")
    manifest_split = "validation" if args.split == "val" else args.split
    rows = [
        ManifestRow(
            scenario_id=scenario_id,
            split=manifest_split,
            source_path=f"{args.split}/{scenario_id}/scenario_{scenario_id}.parquet",
            city_name="unknown_until_loaded",
            selected_reason=f"deterministic_subset_seed_{args.seed}",
        )
        for scenario_id in selected
    ]
    write_manifest(args.manifest, rows)
    print(f"wrote {len(rows)} scenarios to {args.manifest}")

    if not args.execute:
        print("dry run only; pass --execute to download the selected scenario directories")
        return

    commands = [
        f'cp "{S3_ROOT}/{args.split}/{scenario_id}/*" '
        f'"{args.target_root}/{args.split}/{scenario_id}/"'
        for scenario_id in selected
    ]
    with tempfile.TemporaryDirectory(prefix="skilldrive-s5cmd-") as temporary_directory:
        command_file = Path(temporary_directory) / "commands.txt"
        command_file.write_text("\n".join(commands) + "\n", encoding="utf-8")
        subprocess.run([s5cmd, "--no-sign-request", "run", str(command_file)], check=True)
    print(f"downloaded {len(selected)} scenarios to {args.target_root / args.split}")


if __name__ == "__main__":
    main()

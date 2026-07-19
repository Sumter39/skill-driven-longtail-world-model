"""Create and optionally execute a deterministic s5cmd AV2 subset download."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from skilldrive.data.manifests import ManifestRow, read_manifest, write_manifest
from skilldrive.data.subsets import select_ids


S3_ROOT = "s3://argoverse/datasets/av2/motion-forecasting"
SCENARIO_DIRECTORY_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
NETWORK_RETRY_SECONDS = 15
DOWNLOAD_WORKERS = 32
DOWNLOAD_BATCH_SIZE = 200
LOCAL_SCAN_WORKERS = 8
NETWORK_ERROR_MARKERS = (
    "proxyconnect",
    "connection refused",
    "actively refused",
    "connection reset",
    "connection aborted",
    "network is unreachable",
    "no such host",
    "timed out",
    "timeout",
    "tls handshake timeout",
    "unexpected eof",
    "caused by: eof",
    "request canceled",
)


def _resolve_s5cmd(configured: str | None) -> str:
    if configured:
        resolved = shutil.which(configured)
        if resolved:
            return resolved
        if Path(configured).is_file():
            return str(Path(configured))
        raise FileNotFoundError(f"configured s5cmd does not exist: {configured}")
    resolved = shutil.which("s5cmd")
    if not resolved:
        raise RuntimeError("s5cmd is not on PATH; install it as described in docs/data/argoverse2.md")
    return resolved


def _uses_windows_s5cmd(s5cmd: str) -> bool:
    return s5cmd.lower().endswith(".exe")


def _path_for_s5cmd(path: Path, s5cmd: str) -> str:
    resolved = path.resolve()
    if not _uses_windows_s5cmd(s5cmd) or sys.platform == "win32":
        return str(resolved)
    result = subprocess.run(
        ["wslpath", "-w", str(resolved)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _s5cmd_environment(s5cmd: str) -> dict[str, str] | None:
    if not _uses_windows_s5cmd(s5cmd) or sys.platform == "win32":
        return None
    environment = os.environ.copy()
    forwarded = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"]
    existing = [name for name in environment.get("WSLENV", "").split(":") if name]
    environment["WSLENV"] = ":".join(dict.fromkeys([*existing, *forwarded]))
    return environment


def _read_error_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _is_retryable_network_error(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in NETWORK_ERROR_MARKERS)


def _print_fatal_error(message: str) -> None:
    lines = [line for line in message.splitlines() if line.strip()]
    print("fatal s5cmd error; download stopped:", file=sys.stderr)
    print("\n".join(lines[-20:]) or "s5cmd exited without an error message", file=sys.stderr)


def _read_new_directory_ids(path: Path, offset: int) -> tuple[list[str], int]:
    ids: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        handle.seek(offset)
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            candidate = parts[-1].rstrip("/")
            if SCENARIO_DIRECTORY_PATTERN.fullmatch(candidate):
                ids.append(candidate)
        return ids, handle.tell()


def _list_scenario_ids_from_directories(
    s5cmd: str,
    split: str,
    cache: Path,
) -> list[str]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw_listing = cache.with_name(f"{cache.stem}_raw.txt")
    error_log = cache.with_name(f"{cache.stem}_listing_error.log")

    print(
        f"listing official AV2 {split} scenario directories with s5cmd; "
        "progress prints every 10 seconds",
        flush=True,
    )
    started = time.monotonic()
    while True:
        offset = 0
        ids: list[str] = []
        with raw_listing.open("w", encoding="utf-8") as output_handle, error_log.open(
            "w", encoding="utf-8"
        ) as error_handle:
            process = subprocess.Popen(
                [s5cmd, "--no-sign-request", "ls", f"{S3_ROOT}/{split}/"],
                stdout=output_handle,
                stderr=error_handle,
                env=_s5cmd_environment(s5cmd),
            )
            try:
                while process.poll() is None:
                    time.sleep(10)
                    new_ids, offset = _read_new_directory_ids(raw_listing, offset)
                    ids.extend(new_ids)
                    elapsed = time.monotonic() - started
                    print(
                        f"\rlisting {split}: {len(ids)} scenario IDs, {elapsed:.0f}s, "
                        f"{len(ids) / max(elapsed, 1):.1f} IDs/s",
                        end="",
                        flush=True,
                    )
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print()
                raise

        new_ids, offset = _read_new_directory_ids(raw_listing, offset)
        ids.extend(new_ids)
        if process.returncode == 0:
            break
        error_message = _read_error_log(error_log)
        if not _is_retryable_network_error(error_message):
            print()
            _print_fatal_error(error_message)
            raise subprocess.CalledProcessError(process.returncode, process.args)
        print(
            f"\rlisting network unavailable; retrying in {NETWORK_RETRY_SECONDS}s".ljust(100),
            end="",
            flush=True,
        )
        time.sleep(NETWORK_RETRY_SECONDS)

    print()
    if not ids:
        raise RuntimeError("Windows s5cmd returned no AV2 scenario directories")

    unique_ids = sorted(set(ids))
    cache.write_text("\n".join(unique_ids) + "\n", encoding="utf-8")
    elapsed = time.monotonic() - started
    print(f"listed {len(unique_ids)} {split} scenarios in {elapsed:.1f}s total", flush=True)
    return unique_ids


def _list_scenario_ids(s5cmd: str, split: str, cache: Path) -> list[str]:
    if cache.exists():
        cached_ids = [
            line.strip()
            for line in cache.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"loaded {len(cached_ids)} cached {split} scenario IDs from {cache}")
        return cached_ids

    return _list_scenario_ids_from_directories(s5cmd, split, cache)


def _show_progress(label: str, count: int, total: int, status: str, *, final: bool = False) -> None:
    ratio = min(count / max(total, 1), 1.0)
    width = 30
    filled = round(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r{label} [{bar}] {ratio * 100:6.2f}%  {count}/{total}  {status}".ljust(120),
        end="\n" if final else "",
        flush=True,
    )


def _scenario_is_complete(root: Path, split: str, scenario_id: str) -> bool:
    directory = root / split / scenario_id
    scenario_path = directory / f"scenario_{scenario_id}.parquet"
    map_path = directory / f"log_map_archive_{scenario_id}.json"
    try:
        if scenario_path.stat().st_size == 0 or map_path.stat().st_size == 0:
            return False
        pq.ParquetFile(scenario_path).metadata
        with map_path.open(encoding="utf-8") as handle:
            json.load(handle)
    except Exception:
        return False
    return True


def _scan_local_scenarios(
    scenario_ids: list[str],
    *,
    root: Path,
    split: str,
) -> list[str]:
    print("phase 1/2: scanning local files for completeness", flush=True)
    incomplete: list[str] = []
    with ThreadPoolExecutor(max_workers=LOCAL_SCAN_WORKERS) as executor:
        futures = {
            executor.submit(_scenario_is_complete, root, split, scenario_id): scenario_id
            for scenario_id in scenario_ids
        }
        for checked, future in enumerate(as_completed(futures), start=1):
            if not future.result():
                incomplete.append(futures[future])
            if checked % 100 == 0 or checked == len(scenario_ids):
                complete = checked - len(incomplete)
                _show_progress(
                    "scan    ",
                    checked,
                    len(scenario_ids),
                    f"complete={complete}, needs_download={len(incomplete)}",
                    final=checked == len(scenario_ids),
                )
    return sorted(incomplete)


def _run_download_batch(
    command: list[str],
    *,
    environment: dict[str, str] | None,
    error_log: Path,
    completed_before: int,
    batch_size: int,
    expected_total: int,
    batch_number: int,
    total_batches: int,
) -> None:
    status = f"batch {batch_number}/{total_batches}"
    _show_progress("download", completed_before, expected_total, status)
    while True:
        with error_log.open("w", encoding="utf-8") as error_handle:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=error_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                while True:
                    try:
                        return_code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        _show_progress("download", completed_before, expected_total, status)
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                print()
                raise

        if return_code == 0:
            break
        error_message = _read_error_log(error_log)
        if not _is_retryable_network_error(error_message):
            print()
            _print_fatal_error(error_message)
            raise subprocess.CalledProcessError(return_code, command)
        _show_progress(
            "download",
            completed_before,
            expected_total,
            f"network interrupted; retrying batch {batch_number} in {NETWORK_RETRY_SECONDS}s",
        )
        time.sleep(NETWORK_RETRY_SECONDS)

    _show_progress(
        "download",
        completed_before + batch_size,
        expected_total,
        f"batch {batch_number}/{total_batches} complete",
        final=completed_before + batch_size == expected_total,
    )


def _load_resume_ids(manifest: Path, count: int, split: str) -> list[str]:
    if not manifest.exists():
        raise FileNotFoundError(f"resume manifest does not exist: {manifest}")
    rows = read_manifest(manifest)
    ids = [row.scenario_id for row in rows]
    if len(ids) != count:
        raise ValueError(f"resume manifest contains {len(ids)} scenarios, expected {count}")
    if len(set(ids)) != len(ids):
        raise ValueError("resume manifest contains duplicate scenario IDs")
    expected_split = "validation" if split == "val" else split
    if any(row.split != expected_split for row in rows):
        raise ValueError(f"resume manifest contains rows outside split {expected_split}")
    return sorted(ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--s5cmd",
        default=None,
        help="Optional s5cmd executable, including a Windows .exe invoked from WSL.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("data/av2/motion-forecasting"),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude scenario IDs in an existing manifest; may be passed more than once.",
    )
    parser.add_argument(
        "--listing-cache",
        type=Path,
        default=None,
        help="Defaults to data/metadata/av2_<split>_scenario_ids.txt.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    s5cmd = _resolve_s5cmd(args.s5cmd)
    cache = args.listing_cache or Path(f"data/metadata/av2_{args.split}_scenario_ids.txt")
    if args.manifest.exists() and not args.force_manifest:
        selected = _load_resume_ids(args.manifest, args.count, args.split)
        print(
            f"automatically resuming {len(selected)} scenarios from {args.manifest}; "
            "complete files will be skipped",
            flush=True,
        )
    else:
        scenario_ids = _list_scenario_ids(s5cmd, args.split, cache)
        excluded_ids = {
            row.scenario_id
            for manifest in args.exclude_manifest
            for row in read_manifest(manifest)
        }
        available_ids = [scenario_id for scenario_id in scenario_ids if scenario_id not in excluded_ids]
        selected = select_ids(available_ids, args.count, args.seed)

        manifest_split = "validation" if args.split == "val" else args.split
        rows = [
            ManifestRow(
                scenario_id=scenario_id,
                split=manifest_split,
                source_path=f"{args.split}/{scenario_id}/scenario_{scenario_id}.parquet",
                city_name="unknown_until_loaded",
                selected_reason=(
                    f"deterministic_subset_seed_{args.seed}"
                    if not excluded_ids
                    else f"deterministic_subset_seed_{args.seed}_excluding_{len(excluded_ids)}"
                ),
            )
            for scenario_id in selected
        ]
        write_manifest(args.manifest, rows)
        print(f"wrote {len(rows)} scenarios to {args.manifest}")

    if not args.execute:
        print("dry run only; pass --execute to download the selected scenario directories")
        return

    incomplete_ids = _scan_local_scenarios(
        selected,
        root=args.target_root,
        split=args.split,
    )
    completed = len(selected) - len(incomplete_ids)
    if not incomplete_ids:
        print(f"all {len(selected)} scenarios are already complete")
        return

    print(
        f"phase 2/2: downloading {len(incomplete_ids)} incomplete scenarios; "
        f"{completed} complete scenarios will not be sent to s5cmd",
        flush=True,
    )
    target_root = _path_for_s5cmd(args.target_root, s5cmd)
    temporary_parent = cache.parent if _uses_windows_s5cmd(s5cmd) else None
    with tempfile.TemporaryDirectory(
        prefix="skilldrive-s5cmd-",
        dir=temporary_parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        batches = [
            incomplete_ids[index : index + DOWNLOAD_BATCH_SIZE]
            for index in range(0, len(incomplete_ids), DOWNLOAD_BATCH_SIZE)
        ]
        for batch_number, batch_ids in enumerate(batches, start=1):
            commands = [
                f'cp --if-size-differ "{S3_ROOT}/{args.split}/{scenario_id}/*" '
                f'"{target_root}/{args.split}/{scenario_id}/"'
                for scenario_id in batch_ids
            ]
            command_file = temporary_path / f"commands_{batch_number:04d}.txt"
            command_file.write_text("\n".join(commands) + "\n", encoding="utf-8")
            command_file_argument = _path_for_s5cmd(command_file, s5cmd)
            _run_download_batch(
                [
                    s5cmd,
                    "--log",
                    "error",
                    "--numworkers",
                    str(DOWNLOAD_WORKERS),
                    "--no-sign-request",
                    "run",
                    command_file_argument,
                ],
                environment=_s5cmd_environment(s5cmd),
                error_log=temporary_path / f"errors_{batch_number:04d}.log",
                completed_before=completed,
                batch_size=len(batch_ids),
                expected_total=len(selected),
                batch_number=batch_number,
                total_batches=len(batches),
            )
            completed += len(batch_ids)
    print(f"downloaded {len(selected)} scenarios to {args.target_root / args.split}")


if __name__ == "__main__":
    main()

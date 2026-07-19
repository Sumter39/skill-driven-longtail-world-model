"""Download the tiny official AV2 API test scenario used for reader validation."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


SCENARIO_ID = "0a1e6f0a-1817-4a98-b02e-db8c9327d151"
BASE_URL = (
    "https://raw.githubusercontent.com/argoverse/av2-api/main/"
    f"tests/unit/test_data/forecasting_scenarios/{SCENARIO_ID}"
)
FILES = {
    f"scenario_{SCENARIO_ID}.parquet": 123_374,
    f"log_map_archive_{SCENARIO_ID}.json": 99_874,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample/av2") / SCENARIO_ID,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, expected_size in FILES.items():
        target = args.output_dir / filename
        if target.exists() and target.stat().st_size == expected_size and not args.force:
            print(f"reuse {target}")
            continue
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", target)
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"unexpected size for {target}: expected {expected_size}, got {actual_size}"
            )
        print(f"downloaded {target} ({actual_size} bytes)")


if __name__ == "__main__":
    main()

import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data.download_av2_subset import _scenario_is_complete
from skilldrive.data.manifests import ManifestRow
from scripts.data.verify_av2_download import verify_row


def test_verify_row_reports_missing_files(tmp_path) -> None:
    row = ManifestRow(
        "scenario-id",
        "train",
        "train/scenario-id/scenario_scenario-id.parquet",
        "unknown",
        "test",
    )

    errors = verify_row(row, tmp_path)

    assert len(errors) == 2
    assert "missing scenario" in errors[0]
    assert "missing map" in errors[1]


def test_local_scan_accepts_readable_scenario_pair(tmp_path) -> None:
    scenario_id = "example-id"
    directory = tmp_path / "train" / scenario_id
    directory.mkdir(parents=True)
    pq.write_table(
        pa.table({"value": [1]}),
        directory / f"scenario_{scenario_id}.parquet",
    )
    with (directory / f"log_map_archive_{scenario_id}.json").open("w", encoding="utf-8") as handle:
        json.dump({"ok": True}, handle)

    assert _scenario_is_complete(tmp_path, "train", scenario_id)

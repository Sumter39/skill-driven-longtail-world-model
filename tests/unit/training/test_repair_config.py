from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from skilldrive.training import load_cvae_config


BASELINE = Path("configs/models/cvae_baseline.yaml")
REPAIR = Path("configs/models/cvae_generation_repair_v1.yaml")


def test_repair_config_is_versioned_and_baseline_canonical_contract_stays_legacy() -> None:
    baseline = load_cvae_config(BASELINE)
    repair = load_cvae_config(REPAIR)

    assert baseline.version == 1
    assert baseline.repair is None
    assert "repair" not in baseline.to_canonical_dict()
    assert repair.version == 2
    assert repair.repair is not None
    assert repair.repair.contract == "cvae_generation_repair_v1"
    assert repair.repair.source_cache_partition == "formal_train"
    assert repair.repair.model.decoder_initial_delta_mode == "history_velocity"
    assert repair.repair.sampler.max_repeats_per_sample == 8
    assert repair.overfit.sample_count == 64
    assert "internal_validation" not in repair.repair.split.train_sample_index.as_posix()
    assert "final_validation" not in repair.repair.split.development_sample_index.as_posix()


def test_repair_config_rejects_repeat_caps_above_eight(tmp_path: Path) -> None:
    raw = yaml.safe_load(REPAIR.read_text(encoding="utf-8"))
    raw["repair"]["sampler"]["max_repeats_per_sample"] = 9
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="at most 8"):
        load_cvae_config(path)

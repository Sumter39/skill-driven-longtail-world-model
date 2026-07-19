from pathlib import Path

import pytest
import yaml

from skilldrive.skills import load_skill, validate_skill_dict


SKILL_DIR = Path("configs/skills")
CORE_FILES = [
    "vehicle_cut_in.yaml",
    "hard_brake.yaml",
    "stopped_vehicle_blockage.yaml",
    "vulnerable_crossing.yaml",
    "merge_yield.yaml",
]


def test_catalog_has_30_unique_skills_and_five_implemented() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    ids = [entry["skill_id"] for entry in entries]
    assert len(entries) == 30
    assert len(set(ids)) == 30
    assert sum(entry["implemented"] for entry in entries) == 5


@pytest.mark.parametrize("filename", CORE_FILES)
def test_core_skill_is_valid(filename: str) -> None:
    skill = load_skill(SKILL_DIR / filename)
    assert skill.implemented is True
    assert skill.expected_behavior


def test_missing_skill_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_skill_dict({"skill_id": "incomplete"})

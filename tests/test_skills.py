from pathlib import Path

import pytest
import yaml

from skilldrive.skills import load_skill, validate_skill_dict


SKILL_DIR = Path("configs/skills")


def test_catalog_has_30_unique_confirmed_skills() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    ids = [entry["skill_id"] for entry in entries]
    assert len(entries) == 30
    assert len(set(ids)) == 30
    assert {entry["feasibility"] for entry in entries} <= {"A", "B"}
    assert sum(entry["feasibility"] == "A" for entry in entries) == 17
    assert sum(entry["feasibility"] == "B" for entry in entries) == 13
    assert all((SKILL_DIR / f"{skill_id}.yaml").is_file() for skill_id in ids)
    yaml_ids = {path.stem for path in SKILL_DIR.glob("*.yaml")} - {"catalog"}
    assert yaml_ids == set(ids)


def test_all_catalog_skills_are_complete_and_valid() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    entries = [entry for family in catalog["families"].values() for entry in family]
    for entry in entries:
        skill = load_skill(SKILL_DIR / f"{entry['skill_id']}.yaml")
        assert skill.skill_id == entry["skill_id"]
        assert skill.family == entry["family"]
        assert skill.data_support["feasibility"] == entry["feasibility"]
        assert skill.generation_operators
        assert skill.validation_metrics
        assert skill.known_limitations


def test_catalog_families_are_balanced() -> None:
    catalog = yaml.safe_load((SKILL_DIR / "catalog.yaml").read_text(encoding="utf-8"))
    assert len(catalog["families"]) == 6
    assert {family: len(entries) for family, entries in catalog["families"].items()} == {
        family: 5 for family in catalog["families"]
    }


def test_missing_skill_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_skill_dict({"skill_id": "incomplete"})


def test_c_feasibility_is_rejected() -> None:
    data = yaml.safe_load((SKILL_DIR / "lead_hard_brake.yaml").read_text(encoding="utf-8"))
    data["data_support"]["feasibility"] = "C"
    with pytest.raises(ValueError, match="A or B"):
        validate_skill_dict(data)


def test_parameter_source_and_range_are_validated() -> None:
    data = yaml.safe_load((SKILL_DIR / "lead_hard_brake.yaml").read_text(encoding="utf-8"))
    data["parameters"]["brake_onset_s"] = {"range": [3.0, 1.0], "source": "semantic"}
    with pytest.raises(ValueError, match="invalid range"):
        validate_skill_dict(data)

from skilldrive.data.manifests import ManifestRow
from scripts.data.split_av2_train_pool import build_splits


def test_train_pool_split_is_disjoint_and_development_sets_are_nested() -> None:
    rows = [
        ManifestRow(str(index), "train", f"train/{index}", "unknown", "pool")
        for index in range(22)
    ]

    splits = build_splits(
        rows,
        train_count=20,
        internal_validation_count=2,
        development_train_count=5,
        development_validation_count=1,
        seed=2026,
    )

    train_ids = {row.scenario_id for row in splits["formal_train"]}
    validation_ids = {row.scenario_id for row in splits["internal_validation"]}
    development_train_ids = {row.scenario_id for row in splits["development_train"]}
    development_validation_ids = {
        row.scenario_id for row in splits["development_validation"]
    }

    assert len(train_ids) == 20
    assert len(validation_ids) == 2
    assert train_ids.isdisjoint(validation_ids)
    assert development_train_ids <= train_ids
    assert development_validation_ids <= validation_ids

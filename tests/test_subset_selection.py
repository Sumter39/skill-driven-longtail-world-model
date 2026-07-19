import pytest

from skilldrive.data.subsets import select_ids


def test_subset_selection_is_deterministic_and_unique() -> None:
    ids = [f"scene-{index:03d}" for index in range(100)]
    first = select_ids(ids, count=20, seed=2026)
    second = select_ids(list(reversed(ids)), count=20, seed=2026)
    assert first == second
    assert len(first) == len(set(first)) == 20


def test_subset_selection_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_ids(["a"], count=0, seed=1)
    with pytest.raises(ValueError, match="only 1"):
        select_ids(["a"], count=2, seed=1)

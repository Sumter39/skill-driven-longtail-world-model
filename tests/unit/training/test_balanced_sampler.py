from __future__ import annotations

from collections import Counter

import pytest
from torch.utils.data import Dataset

from skilldrive.data import ObservedSkillBalanceSampler


class _IndexedDataset(Dataset[int]):
    def __init__(self) -> None:
        self.entries = []
        for index in range(4):
            self.entries.append(
                {
                    "shard": f"shard-{index % 2}.pt",
                    "spec": {
                        "skill_id": "<none>",
                        "skill_supervision_mask": False,
                    },
                }
            )
        for skill_id, count in (("common", 20), ("medium", 3), ("rare", 1)):
            for index in range(count):
                self.entries.append(
                    {
                        "shard": f"shard-{index % 3}.pt",
                        "spec": {
                            "skill_id": skill_id,
                            "skill_supervision_mask": True,
                        },
                    }
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> int:
        return index


def test_balanced_sampler_covers_every_sample_caps_repeats_and_reports_exposure() -> None:
    dataset = _IndexedDataset()
    sampler = ObservedSkillBalanceSampler(
        dataset,
        seed=2026,
        max_repeats_per_sample=8,
    )

    order = list(sampler)
    repeats = Counter(order)
    assert set(order) == set(range(len(dataset)))
    assert max(repeats.values()) == 8
    assert sampler.contract["observed_epoch_exposure_by_skill"] == {
        "common": 20,
        "medium": 20,
        "rare": 8,
    }
    assert sampler.contract["base_samples"] == 4
    assert len(order) == 52
    exposure = sampler.exposure()
    assert exposure["base"] == 4
    assert exposure["observed_by_skill"] == {
        "common": 20,
        "medium": 20,
        "rare": 8,
    }
    assert exposure["maximum_sample_repeats_in_range"] == 8


def test_balanced_sampler_epoch_and_range_are_exactly_recoverable() -> None:
    sampler = ObservedSkillBalanceSampler(_IndexedDataset(), seed=9)
    epoch_zero = list(sampler)
    assert epoch_zero == list(sampler)

    sampler.set_epoch(1)
    epoch_one = list(sampler)
    assert epoch_one == list(sampler)
    assert epoch_one != epoch_zero
    sampler.set_range(7, 23)
    assert list(sampler) == epoch_one[7:23]
    assert len(sampler) == 16
    assert sampler.exposure()["samples"] == 16
    sampler.set_range()
    assert list(sampler) == epoch_one


def test_balanced_sampler_rejects_compatible_or_invalid_skill_contracts() -> None:
    dataset = _IndexedDataset()
    dataset.entries[0]["spec"] = {
        "skill_id": "compatible_seed_skill",
        "skill_supervision_mask": False,
    }
    with pytest.raises(ValueError, match="only base or observed"):
        ObservedSkillBalanceSampler(dataset, seed=1)

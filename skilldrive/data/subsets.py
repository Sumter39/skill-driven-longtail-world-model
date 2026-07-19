"""Deterministic scenario subset selection."""

from __future__ import annotations

import random


def select_ids(ids: list[str], count: int, seed: int) -> list[str]:
    """Select unique IDs independently of the input ordering."""
    unique = sorted(set(ids))
    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(unique):
        raise ValueError(f"requested {count} scenarios, only {len(unique)} are available")
    generator = random.Random(seed)
    return sorted(generator.sample(unique, count))

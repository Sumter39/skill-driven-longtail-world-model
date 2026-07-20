"""Deterministic stratified selection of formal seed scenarios."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from skilldrive.seeds.records import SeedRecord, sort_seed_records


Stratum = tuple[str, str, int]


def _stable_digest(seed: int, namespace: str, *parts: object) -> bytes:
    payload = "\x1f".join((str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _scenario_strata(records: list[SeedRecord]) -> dict[Stratum, set[str]]:
    risk_groups: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        group_key = (record.skill_id, record.seed_risk_metric)
        scenario_risks = risk_groups[group_key]
        if record.scenario_id in scenario_risks:
            raise ValueError(
                "multiple seed records for the same scenario, skill, and risk "
                f"metric: {(record.scenario_id, *group_key)!r}"
            )
        scenario_risks[record.scenario_id] = record.seed_risk_value

    strata: dict[Stratum, set[str]] = defaultdict(set)
    for (skill_id, risk_metric), group in risk_groups.items():
        ordered = sorted(
            group.items(),
            key=lambda item: (item[1], item[0]),
        )
        size = len(ordered)
        quartile = 0
        previous_risk_value: float | None = None
        for index, (scenario_id, risk_value) in enumerate(ordered):
            if index == 0 or risk_value != previous_risk_value:
                quartile = min(3, index * 4 // size)
            strata[(skill_id, risk_metric, quartile)].add(scenario_id)
            previous_risk_value = risk_value
    return dict(strata)


def select_seed_records(
    records: Iterable[SeedRecord],
    target_scenario_count: int,
    *,
    seed: int = 2026,
) -> list[SeedRecord]:
    """Select unique scenarios while retaining every record from each selection.

    Records are stratified by skill, seed-risk metric, and within-group risk-value
    quartile, without splitting equal risk values across quartiles. Smaller
    occupied strata receive the first turn in every round. A seeded stable hash
    orders equally sized strata and scenarios within a stratum, so iteration order
    never affects the result.
    """

    if (
        isinstance(target_scenario_count, bool)
        or not isinstance(target_scenario_count, int)
        or target_scenario_count < 1
    ):
        raise ValueError("target_scenario_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    try:
        materialized = list(records)
    except TypeError as exc:
        raise ValueError("records must be an iterable of SeedRecord") from exc
    if any(not isinstance(record, SeedRecord) for record in materialized):
        raise ValueError("records must contain only SeedRecord instances")

    canonical = sort_seed_records(materialized)
    strata = _scenario_strata(canonical)
    scenario_ids = {record.scenario_id for record in canonical}
    if target_scenario_count >= len(scenario_ids):
        return canonical

    stratum_order = sorted(
        strata,
        key=lambda stratum: (
            len(strata[stratum]),
            _stable_digest(seed, "stratum", *stratum),
            stratum,
        ),
    )
    queues = {
        stratum: sorted(
            strata[stratum],
            key=lambda scenario_id: (
                _stable_digest(seed, "scenario", *stratum, scenario_id),
                scenario_id,
            ),
        )
        for stratum in stratum_order
    }
    positions = {stratum: 0 for stratum in stratum_order}

    selected: list[str] = []
    selected_set: set[str] = set()
    while len(selected) < target_scenario_count:
        added = False
        for stratum in stratum_order:
            queue = queues[stratum]
            index = positions[stratum]
            while index < len(queue) and queue[index] in selected_set:
                index += 1
            positions[stratum] = index
            if index == len(queue):
                continue

            scenario_id = queue[index]
            positions[stratum] += 1
            selected.append(scenario_id)
            selected_set.add(scenario_id)
            added = True
            if len(selected) == target_scenario_count:
                break
        if not added:
            break

    return sort_seed_records(
        record for record in canonical if record.scenario_id in selected_set
    )

"""Deterministic BEV review images for detected skill seeds."""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skilldrive.schemas import AgentTrack, Scenario
from skilldrive.seeds import SeedRecord, sort_seed_records
from skilldrive.visualization.bev import OBJECT_COLORS


PRIMARY_ROLE_COLORS = {
    "initiator": "#f97316",
    "responder": "#0891b2",
}
EXTRA_ROLE_COLORS = ("#7c3aed", "#16a34a", "#db2777", "#ca8a04")


def _safe_filename_segment(value: str, limit: int = 48) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (segment or "unknown")[:limit]


def seed_review_filename(record: SeedRecord) -> str:
    """Return a short, cross-platform filename derived from a seed's unique key."""

    digest = hashlib.sha256("\x1f".join(record.unique_key).encode("utf-8")).hexdigest()[:12]
    skill = _safe_filename_segment(record.skill_id)
    scenario = _safe_filename_segment(record.scenario_id)
    return f"{skill}__{scenario}__{digest}.png"


def _last_visible_position(agent: AgentTrack) -> np.ndarray | None:
    finite = np.isfinite(agent.positions).all(axis=1)
    observed = finite & agent.observed_mask
    indices = np.flatnonzero(observed if observed.any() else finite)
    return agent.positions[indices[-1]] if len(indices) else None


def _evidence_lines(record: SeedRecord, width: int = 105) -> list[str]:
    evidence = json.dumps(
        record.evidence,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(evidence) > 500:
        evidence = f"{evidence[:497]}..."
    return textwrap.wrap(
        f"evidence={evidence}",
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _risk_detail_lines(record: SeedRecord) -> list[str]:
    target = record.target_risk_definition
    target_range = json.dumps(
        target["target_range"],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    relation = "proxy" if record.seed_risk_is_proxy else "target_metric_observation"
    return [
        f"seed_risk={record.seed_risk_metric}:{record.seed_risk_value:.6g}",
        (
            f"target_risk={target['metric']} range={target_range} "
            f"direction={target['direction']} source={target['source']}"
        ),
        f"risk_relation={relation}",
    ]


def render_seed_review(
    scenario: Scenario,
    record: SeedRecord,
    output_dir: str | Path,
    radius_m: float = 60.0,
) -> Path:
    """Render one candidate with every structured actor role highlighted."""

    if scenario.scenario_id != record.scenario_id:
        raise ValueError("scenario_id does not match the seed record")
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)):
        raise ValueError("radius_m must be a positive finite number")
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius_m must be a positive finite number")

    agents = {agent.track_id: agent for agent in scenario.agents}
    missing = [track_id for track_id in record.role_track_ids.values() if track_id not in agents]
    if missing:
        raise ValueError(f"seed record references missing tracks: {', '.join(missing)}")

    role_by_track = {track_id: role for role, track_id in record.role_track_ids.items()}
    extra_roles = sorted(
        role
        for role, track_id in record.role_track_ids.items()
        if track_id not in {record.initiator_track_id, record.responder_track_id}
    )
    extra_colors = {
        role: EXTRA_ROLE_COLORS[index % len(EXTRA_ROLE_COLORS)]
        for index, role in enumerate(extra_roles)
    }
    role_positions = [
        position
        for track_id in role_by_track
        if (position := _last_visible_position(agents[track_id])) is not None
    ]
    center = np.mean(role_positions, axis=0) if role_positions else np.zeros(2)
    if role_positions:
        required_radius = max(float(np.max(np.abs(position - center))) for position in role_positions)
        radius = max(radius, required_radius + 5.0)

    target = Path(output_dir) / seed_review_filename(record)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 9), dpi=150)
    for polyline in scenario.map_polylines:
        points = polyline.points
        if len(points) < 2:
            continue
        color = "#94a3b8" if "boundary" in polyline.polyline_type else "#cbd5e1"
        width = 1.0 if "boundary" in polyline.polyline_type else 0.7
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=width, zorder=1)

    other_label_added = False
    for agent in scenario.agents:
        finite = np.isfinite(agent.positions).all(axis=1)
        observed = finite & agent.observed_mask
        future = finite & ~agent.observed_mask
        role = role_by_track.get(agent.track_id)
        if role is None:
            color = OBJECT_COLORS.get(agent.object_type.lower(), "#64748b")
            linewidth = 0.9
            alpha = 0.35
            observed_label = "other agents" if not other_label_added else None
            other_label_added = other_label_added or observed.any()
        else:
            if agent.track_id == record.initiator_track_id:
                color = PRIMARY_ROLE_COLORS["initiator"]
            elif agent.track_id == record.responder_track_id:
                color = PRIMARY_ROLE_COLORS["responder"]
            else:
                color = extra_colors[role]
            linewidth = 3.2
            alpha = 1.0
            observed_label = role

        if observed.any():
            axis.plot(
                agent.positions[observed, 0],
                agent.positions[observed, 1],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=observed_label,
                zorder=4 if role else 2,
            )
            last = agent.positions[observed][-1]
            axis.scatter(
                last[0],
                last[1],
                color=color,
                s=78 if role else 18,
                edgecolors="black" if role else "none",
                linewidths=0.8,
                zorder=6 if role else 3,
            )
            if role:
                axis.annotate(
                    f"{role}\n{agent.track_id}",
                    xy=(last[0], last[1]),
                    xytext=(7, 7),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    color=color,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.9},
                    zorder=7,
                )
        if future.any():
            axis.plot(
                agent.positions[future, 0],
                agent.positions[future, 1],
                color=color,
                linewidth=linewidth,
                linestyle="--",
                alpha=0.8 if role else 0.25,
                zorder=3 if role else 2,
            )

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Map x / m")
    axis.set_ylabel("Map y / m")
    axis.set_title(
        f"Skill {record.skill_id}\nScenario {scenario.scenario_id} | {scenario.city_name}"
    )
    axis.grid(True, color="#e2e8f0", linewidth=0.5)
    axis.legend(loc="upper right", fontsize=8)

    details = [
        "roles="
        + json.dumps(
            record.role_track_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        f"score={record.trigger_score:.4f}",
        *_risk_detail_lines(record),
        *_evidence_lines(record),
    ]
    figure.text(
        0.06,
        0.018,
        "\n".join(details),
        ha="left",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.45", "fc": "#f8fafc", "ec": "#94a3b8"},
    )
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target


def select_stratified_review_records(
    records: Iterable[SeedRecord],
    target_count: int = 100,
) -> list[SeedRecord]:
    """Select up to ``target_count`` records in deterministic skill round-robin order.

    The first round covers every skill that has a candidate whenever the target
    permits. Within each skill, stronger trigger scores are reviewed first. If
    fewer candidates exist than requested, all candidates are returned.
    """

    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count < 1:
        raise ValueError("target_count must be a positive integer")

    canonical = sort_seed_records(records)
    groups: dict[str, list[SeedRecord]] = defaultdict(list)
    for record in canonical:
        groups[record.skill_id].append(record)
    for group in groups.values():
        group.sort(key=lambda record: (-record.trigger_score, record.unique_key))

    limit = min(target_count, len(canonical))
    selected: list[SeedRecord] = []
    indices = {skill_id: 0 for skill_id in groups}
    skill_ids = sorted(groups)
    while len(selected) < limit:
        added = False
        for skill_id in skill_ids:
            index = indices[skill_id]
            if index >= len(groups[skill_id]):
                continue
            selected.append(groups[skill_id][index])
            indices[skill_id] += 1
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
    return selected

"""Headless bird's-eye-view rendering for prepared scenarios."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from skilldrive.schemas import Scenario


OBJECT_COLORS = {
    "vehicle": "#2563eb",
    "pedestrian": "#dc2626",
    "cyclist": "#16a34a",
    "motorcyclist": "#9333ea",
}


def render_bev(scenario: Scenario, output_path: str | Path, radius_m: float = 60.0) -> Path:
    """Render map polylines and observed/future tracks to a PNG file."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 8), dpi=150)
    for polyline in scenario.map_polylines:
        points = polyline.points
        if len(points) < 2:
            continue
        color = "#94a3b8" if "boundary" in polyline.polyline_type else "#cbd5e1"
        width = 1.0 if "boundary" in polyline.polyline_type else 0.7
        axis.plot(points[:, 0], points[:, 1], color=color, linewidth=width, zorder=1)

    for agent in scenario.agents:
        valid = np.isfinite(agent.positions).all(axis=1)
        observed = valid & agent.observed_mask
        future = valid & ~agent.observed_mask
        color = OBJECT_COLORS.get(agent.object_type.lower(), "#475569")
        width = 2.8 if agent.is_focal else 1.5
        label = f"{agent.object_type} history"
        if observed.any():
            axis.plot(
                agent.positions[observed, 0],
                agent.positions[observed, 1],
                color=color,
                linewidth=width,
                label=label,
                zorder=3,
            )
            axis.scatter(
                agent.positions[observed, 0][-1],
                agent.positions[observed, 1][-1],
                color=color,
                s=42 if agent.is_focal else 24,
                edgecolors="black" if agent.is_focal else "none",
                zorder=4,
            )
        if future.any():
            axis.plot(
                agent.positions[future, 0],
                agent.positions[future, 1],
                color=color,
                linewidth=width,
                linestyle="--",
                alpha=0.8,
                label=f"{agent.object_type} future",
                zorder=2,
            )

    axis.set_xlim(-radius_m, radius_m)
    axis.set_ylim(-radius_m, radius_m)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Local x / m (forward)")
    axis.set_ylabel("Local y / m (left)")
    axis.set_title(f"Scenario {scenario.scenario_id} | {scenario.city_name}")
    axis.grid(True, color="#e2e8f0", linewidth=0.5)

    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        axis.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)
    figure.tight_layout()
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)
    return target

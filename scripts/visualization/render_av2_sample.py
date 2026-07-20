"""Load and render one AV2 motion-forecasting scenario without training."""

from __future__ import annotations

import argparse
from pathlib import Path

from skilldrive.data.av2_reader import load_av2_scenario
from skilldrive.data.coordinates import to_focal_frame
from skilldrive.visualization import render_bev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=Path, help="Path to scenario_*.parquet")
    parser.add_argument("--map", dest="map_path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/av2_sample_bev.png"))
    parser.add_argument("--radius", type=float, default=60.0)
    args = parser.parse_args()

    scenario = load_av2_scenario(args.scenario, args.map_path)
    local_scenario = to_focal_frame(scenario)
    output = render_bev(local_scenario, args.output, radius_m=args.radius)
    print(
        f"scenario={scenario.scenario_id} agents={len(scenario.agents)} "
        f"map_polylines={len(scenario.map_polylines)} output={output.resolve()}"
    )


if __name__ == "__main__":
    main()

from pathlib import Path

from skilldrive.schemas import Scenario
from skilldrive.visualization import render_bev


def test_bev_renderer_writes_png(tmp_path: Path, synthetic_scenario: Scenario) -> None:
    output = render_bev(synthetic_scenario, tmp_path / "scene.png", radius_m=20)
    assert output.exists()
    assert output.stat().st_size > 1_000

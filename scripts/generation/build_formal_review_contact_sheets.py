"""Build compact contact sheets for the pending formal BEV review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build_contact_sheets(summary_path: Path, *, output_dir: Path, per_sheet: int = 25) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = summary["cases"]
    if per_sheet <= 0:
        raise ValueError("per_sheet must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    outputs: list[Path] = []
    columns = 5
    thumb = 300
    label_height = 34
    rows = math.ceil(per_sheet / columns)
    for offset in range(0, len(cases), per_sheet):
        subset = cases[offset : offset + per_sheet]
        canvas = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, case in enumerate(subset):
            generated = Image.open(summary_path.parent / "cases" / case["case_name"] / "generated" / next(iter((summary_path.parent / "cases" / case["case_name"] / "generated").glob("*.png"))).name).convert("RGB")
            generated.thumbnail((thumb - 4, thumb - 4))
            x = (index % columns) * thumb + (thumb - generated.width) // 2
            y = (index // columns) * (thumb + label_height) + 2
            canvas.paste(generated, (x, y))
            label = f"{case['review_rank']:03d} {case['disposition']} {case['skill_id'][:24]}"
            draw.text(((index % columns) * thumb + 2, y + thumb - 2), label, fill="black", font=font)
            generated.close()
        output = output_dir / f"contact-{offset // per_sheet + 1:03d}.jpg"
        canvas.save(output, quality=90, optimize=True)
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--per-sheet", type=int, default=25)
    args = parser.parse_args()
    output_dir = args.output_dir or args.summary.parent / "contact-sheets"
    outputs = build_contact_sheets(args.summary, output_dir=output_dir, per_sheet=args.per_sheet)
    print(f"contact sheets: {len(outputs)}")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render grouped candidate/training pairs from a leakage near-match report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in report["near_matches"]:
        grouped[row["candidate"]["key"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18
        )
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
    except OSError:
        font = small = ImageFont.load_default()

    index = []
    for key, rows in sorted(grouped.items()):
        candidate = rows[0]["candidate"]
        candidate_image = fit(Image.open(candidate["path"]), (480, 480))
        cell_w, cell_h, label_h = 330, 330, 54
        columns = min(4, max(1, len(rows)))
        grid_rows = (len(rows) + columns - 1) // columns
        width = max(520, columns * cell_w)
        height = 90 + 520 + grid_rows * (cell_h + label_h)
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 10), key, fill="black", font=font)
        draw.text(
            (12, 40),
            f"candidate source={candidate['source_image_id']} near_rows={len(rows)}",
            fill=(60, 60, 60),
            font=small,
        )
        sheet.paste(candidate_image, ((width - candidate_image.width) // 2, 80))
        draw.text((12, 570), "CANDIDATE ABOVE; TRAINING MATCHES BELOW", fill=(150, 0, 0), font=small)
        for index_in_group, row in enumerate(rows):
            column = index_in_group % columns
            grid_row = index_in_group // columns
            x, y = column * cell_w, 610 + grid_row * (cell_h + label_h)
            training = fit(Image.open(row["training_path"]), (cell_w - 16, cell_h - 16))
            sheet.paste(training, (x + (cell_w - training.width) // 2, y))
            relative_training = Path(row["training_path"]).name
            parent = Path(row["training_path"]).parent.parent.name
            draw.text(
                (x + 4, y + cell_h),
                f"{parent}/{relative_training}",
                fill="black",
                font=small,
            )
            draw.text(
                (x + 4, y + cell_h + 22),
                f"pHash={row['phash_distance']} dHash={row['dhash_distance']}",
                fill=(70, 70, 70),
                font=small,
            )
        output = args.output_dir / f"{key.replace(':', '__')}.jpg"
        sheet.save(output, quality=92, subsampling=0)
        index.append(
            {
                "candidate_key": key,
                "case_id": candidate["case_id"],
                "near_match_count": len(rows),
                "sheet": output.name,
            }
        )
    (args.output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"candidate_count": len(index), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

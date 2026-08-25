"""Draw converted YOLO labels back onto the images, so we can eyeball them.

Coordinate conversion bugs are silent — the training loop happily learns from
misplaced boxes. The only reliable check is to look at a few samples before
spending the GPU window on them.

Usage:
    python src/data/visualize_labels.py --data data/processed/cardd_seg --split val -n 6
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
COLOURS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]


def draw_detection(draw, parts, w, h, colour):
    cx, cy, bw, bh = (float(v) for v in parts)
    draw.rectangle(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        outline=colour,
        width=3,
    )
    return (cx - bw / 2) * w, (cy - bh / 2) * h


def draw_polygon(draw, parts, w, h, colour):
    points = [(float(parts[i]) * w, float(parts[i + 1]) * h) for i in range(0, len(parts) - 1, 2)]
    draw.line(points + [points[0]], fill=colour, width=3)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, type=Path, help="cardd_det or cardd_seg directory")
    ap.add_argument("--split", default="val")
    ap.add_argument("-n", type=int, default=6, help="how many samples to draw")
    ap.add_argument("--out", type=Path, default=Path("report/figures/label_check"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_dir = args.data / "labels" / args.split
    image_dir = args.data / "images" / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    labels = sorted(p for p in label_dir.glob("*.txt") if p.stat().st_size > 0)
    random.seed(args.seed)
    picked = random.sample(labels, min(args.n, len(labels)))

    for label_path in picked:
        image_path = next(image_dir.glob(label_path.stem + ".*"))
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        w, h = image.size

        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            cls = int(fields[0])
            colour = COLOURS[cls % len(COLOURS)]
            parts = fields[1:]
            anchor = draw_detection(draw, parts, w, h, colour) if len(parts) == 4 \
                else draw_polygon(draw, parts, w, h, colour)
            draw.text((anchor[0] + 4, max(anchor[1] - 14, 2)), CLASS_NAMES[cls], fill=colour)

        out_path = args.out / f"{args.data.name}_{args.split}_{label_path.stem}.jpg"
        image.save(out_path, quality=90)
        print(f"  {out_path}")


if __name__ == "__main__":
    main()

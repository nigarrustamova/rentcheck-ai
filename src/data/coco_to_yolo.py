"""Convert the CarDD COCO annotations into the layout Ultralytics expects.

CarDD ships bounding boxes and one polygon per instance, so we emit two parallel
datasets from the same source: a detection one (class cx cy w h) and a
segmentation one (class x1 y1 x2 y2 ...). Both are needed for the ablation that
compares box-only output against masks.

The images themselves are hard-linked rather than copied — on one NTFS volume that
costs no extra disk space, and 4,000 high-resolution images would otherwise be
duplicated twice.

CarDD's official train/val/test split is kept as-is. It is the split the CarDD
paper reports on, which makes our numbers directly comparable to theirs.

Usage:
    python src/data/coco_to_yolo.py \
        --src data/raw/cardd_kaggle_mirror/CarDD_COCO \
        --dst data/processed
"""

import argparse
import json
import shutil
from pathlib import Path

# COCO category id -> YOLO class index. Order fixed here and mirrored in the
# dataset YAML; changing it invalidates every label file already written.
CLASS_NAMES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
CATEGORY_TO_INDEX = {i + 1: i for i in range(len(CLASS_NAMES))}

SPLITS = {"train2017": "train", "val2017": "val", "test2017": "test"}

# A polygon needs at least three points to enclose anything.
MIN_POLYGON_COORDS = 6


def clamp01(value):
    return min(max(value, 0.0), 1.0)


def to_detection_row(ann, width, height):
    """COCO [x, y, w, h] in pixels -> YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        return None
    cx = clamp01((x + w / 2) / width)
    cy = clamp01((y + h / 2) / height)
    nw = clamp01(w / width)
    nh = clamp01(h / height)
    if nw <= 0 or nh <= 0:
        return None
    cls = CATEGORY_TO_INDEX[ann["category_id"]]
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def to_segmentation_row(ann, width, height):
    """COCO polygon in pixels -> YOLO polygon normalised.

    CarDD stores exactly one polygon per instance; if that ever changes we keep
    the largest one, since Ultralytics expects a single contour per row.
    """
    polygons = ann["segmentation"]
    if not isinstance(polygons, list) or not polygons:
        return None
    polygon = max(polygons, key=len)
    if len(polygon) < MIN_POLYGON_COORDS:
        return None

    coords = []
    for i in range(0, len(polygon) - 1, 2):
        coords.append(clamp01(polygon[i] / width))
        coords.append(clamp01(polygon[i + 1] / height))

    cls = CATEGORY_TO_INDEX[ann["category_id"]]
    return f"{cls} " + " ".join(f"{c:.6f}" for c in coords)


def link_or_copy(src, dst):
    """Hard-link when possible, fall back to copying across volumes."""
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def convert_split(src_root, dst_root, coco_dir, split_name):
    ann_path = src_root / "annotations" / f"instances_{coco_dir}.json"
    with open(ann_path, encoding="utf-8") as fh:
        coco = json.load(fh)

    images = {img["id"]: img for img in coco["images"]}

    by_image = {}
    skipped_boxes = skipped_polygons = 0
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        det = to_detection_row(ann, img["width"], img["height"])
        seg = to_segmentation_row(ann, img["width"], img["height"])
        if det is None:
            skipped_boxes += 1
        if seg is None:
            skipped_polygons += 1
        rows = by_image.setdefault(ann["image_id"], {"det": [], "seg": []})
        if det:
            rows["det"].append(det)
        if seg:
            rows["seg"].append(seg)

    written = {"det": 0, "seg": 0}
    empty = 0
    for task in ("det", "seg"):
        (dst_root / f"cardd_{task}" / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (dst_root / f"cardd_{task}" / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for img_id, img in images.items():
        src_image = src_root / coco_dir / img["file_name"]
        if not src_image.exists():
            raise FileNotFoundError(f"Image listed in annotations is missing: {src_image}")

        rows = by_image.get(img_id, {"det": [], "seg": []})
        if not rows["det"]:
            # No annotations at all — YOLO reads this as a background image.
            empty += 1

        for task in ("det", "seg"):
            base = dst_root / f"cardd_{task}"
            link_or_copy(src_image, base / "images" / split_name / img["file_name"])
            label_path = (base / "labels" / split_name / img["file_name"]).with_suffix(".txt")
            label_path.write_text("\n".join(rows[task]) + ("\n" if rows[task] else ""), encoding="utf-8")
            written[task] += len(rows[task])

    print(f"  {split_name:<6} images={len(images):>5}  boxes={written['det']:>5}  polygons={written['seg']:>5}", end="")
    if empty:
        print(f"  background-only={empty}", end="")
    if skipped_boxes or skipped_polygons:
        print(f"  skipped: boxes={skipped_boxes} polygons={skipped_polygons}", end="")
    print()
    return len(images)


def write_dataset_yaml(dst_root, task):
    """Ultralytics dataset config. Paths are relative to the repository root."""
    base = dst_root / f"cardd_{task}"
    lines = [
        f"# CarDD, {'detection' if task == 'det' else 'instance segmentation'} labels.",
        "# Generated by src/data/coco_to_yolo.py — do not edit by hand.",
        f"path: {base.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    path = base / "dataset.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="CarDD_COCO directory")
    ap.add_argument("--dst", required=True, type=Path, help="output root, e.g. data/processed")
    args = ap.parse_args()

    if not (args.src / "annotations").is_dir():
        raise SystemExit(f"No annotations/ under {args.src}")

    print(f"\nConverting {args.src} -> {args.dst}")
    total = 0
    for coco_dir, split_name in SPLITS.items():
        total += convert_split(args.src, args.dst, coco_dir, split_name)

    print(f"\n  total images: {total}")
    for task in ("det", "seg"):
        print(f"  dataset config: {write_dataset_yaml(args.dst, task)}")
    print()


if __name__ == "__main__":
    main()

"""Inspect a raw dataset folder and report what is actually inside it.

Answers the four questions we need before any training starts: how many images
there are, which annotation format they come in, how balanced the classes are,
and what resolution we are dealing with. Handles both COCO-style annotations
(CarDD) and YOLO-style label files (the Kaggle sets), so the same command works
on every folder under data/raw/.

Usage:
    python src/data/check_dataset.py --root data/raw/cardd
    python src/data/check_dataset.py --root data/raw/kaggle
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The course brief's floor for a detection/segmentation project.
DETECTION_IMAGE_FLOOR = 2000


def find_images(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def summarise_images(paths):
    """Read image headers only — fast, and still catches truncated files."""
    sizes, corrupt = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception as exc:
            corrupt.append((p, type(exc).__name__))
    return sizes, corrupt


def spread(values):
    """min / median / max of a list of numbers."""
    if not values:
        return None
    v = sorted(values)
    return v[0], v[len(v) // 2], v[-1]


def find_coco_files(root):
    """JSON files that look like COCO annotation dumps."""
    found = []
    for p in root.rglob("*.json"):
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if isinstance(data, dict) and {"images", "annotations"} <= data.keys():
            found.append((p, data))
    return found


def report_coco(path, data, root):
    names = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    per_class = Counter(a["category_id"] for a in data["annotations"])
    with_seg = sum(1 for a in data["annotations"] if a.get("segmentation"))

    print(f"\n  {path.relative_to(root)}")
    print(f"    images listed      : {len(data['images'])}")
    print(f"    instances          : {len(data['annotations'])}")
    print(f"    with segmentation  : {with_seg}")
    print(f"    classes            : {len(names)}")
    for cid, count in per_class.most_common():
        share = 100 * count / len(data["annotations"])
        print(f"      {names.get(cid, cid):<16} {count:>6}  ({share:4.1f}%)")

    missing = [c for c in names if c not in per_class]
    if missing:
        print(f"    !! declared but unused classes: {[names[c] for c in missing]}")


def find_class_names(root):
    """YOLO folders usually ship names in data.yaml, classes.txt or obj.names."""
    for pattern in ("data.yaml", "classes.txt", "obj.names", "*.yaml"):
        for p in root.rglob(pattern):
            return p
    return None


def report_yolo(root, images):
    label_files = [p for p in root.rglob("*.txt") if p.name.lower() != "classes.txt"]
    if not label_files:
        return False

    per_class = Counter()
    empty, malformed = 0, 0
    for p in label_files:
        lines = [ln for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for ln in lines:
            parts = ln.split()
            # A YOLO row is "class cx cy w h", or longer for polygon labels.
            if len(parts) < 5:
                malformed += 1
                continue
            per_class[parts[0]] += 1

    print("\n  YOLO-style labels")
    print(f"    label files        : {len(label_files)}")
    print(f"    empty label files  : {empty}")
    if malformed:
        print(f"    !! malformed rows  : {malformed}")
    print(f"    instances          : {sum(per_class.values())}")
    print(f"    class ids          : {len(per_class)}")

    total = sum(per_class.values()) or 1
    for cid, count in sorted(per_class.items(), key=lambda kv: -kv[1]):
        print(f"      class {cid:<10} {count:>6}  ({100 * count / total:4.1f}%)")

    stems = {p.stem for p in label_files}
    unlabelled = [p for p in images if p.stem not in stems]
    if unlabelled:
        print(f"    !! images with no label file: {len(unlabelled)}")

    names_file = find_class_names(root)
    if names_file:
        print(f"    class names file   : {names_file.relative_to(root)}  <- check ids map to real names")
    else:
        print("    !! no class-names file found — ids alone are not enough for training")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="dataset folder to inspect")
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    print(f"\n=== {root} ===")

    images = find_images(root)
    print(f"\n  images found       : {len(images)}")
    if not images:
        raise SystemExit("  No images here — check the extraction path.")

    sizes, corrupt = summarise_images(images)
    widths = spread([w for w, _ in sizes])
    heights = spread([h for _, h in sizes])
    print(f"    width  min/med/max : {widths[0]} / {widths[1]} / {widths[2]}")
    print(f"    height min/med/max : {heights[0]} / {heights[1]} / {heights[2]}")
    if corrupt:
        print(f"    !! unreadable files: {len(corrupt)}")
        for p, err in corrupt[:5]:
            print(f"       {p.relative_to(root)}  ({err})")

    coco = find_coco_files(root)
    if coco:
        print(f"\n  COCO annotation files: {len(coco)}")
        for path, data in coco:
            report_coco(path, data, root)
    yolo = report_yolo(root, images)

    if not coco and not yolo:
        print("\n  !! No annotations found in either format — this folder is images only.")

    print()
    if len(images) >= DETECTION_IMAGE_FLOOR:
        print(f"  Detection floor (>={DETECTION_IMAGE_FLOOR} images): OK")
    else:
        print(f"  Detection floor (>={DETECTION_IMAGE_FLOOR} images): NOT met by this folder alone")
    print()


if __name__ == "__main__":
    main()

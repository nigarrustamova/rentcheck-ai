"""Turn a folder of inspection photos into a damage inventory the diff can use.

One folder is one session — the six walk-around photos taken at check-out, or the
six taken at check-in. Each photo becomes a list of Damage records keyed by the
view its filename names, which is what src/diff/compare.py consumes.

Areas are recorded as a fraction of the image rather than in pixels, because the
two sessions may be shot on different phones at different resolutions and a raw
pixel count would not be comparable between them. The pixel count is kept as well,
since converting to cm2 later needs it together with a scale reference.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from compare import (
    IMAGE_EXTENSIONS,
    REVIEW_CONFIDENCE,
    Damage,
    compare,
    view_from_filename,
)


def damages_from_result(result, view, names):
    """Convert one Ultralytics result into Damage records.

    Detections without a mask are skipped rather than guessed at: the whole point
    of the segmentation model is that the area is measured, not inferred from a
    bounding box that mostly encloses undamaged paint.
    """
    damages = []
    if result.masks is None or result.boxes is None:
        return damages

    original_h, original_w = result.orig_shape
    mask_h, mask_w = result.masks.data.shape[-2:]
    mask_pixels = mask_h * mask_w

    for mask, box in zip(result.masks.data, result.boxes):
        fraction = float(mask.sum()) / mask_pixels
        damages.append(
            Damage(
                view=view,
                class_name=names[int(box.cls)],
                confidence=float(box.conf),
                area_fraction=round(fraction, 6),
                bbox=tuple(round(float(v), 6) for v in box.xyxyn[0]),
                area_px=int(round(fraction * original_h * original_w)),
            )
        )
    return damages


def detect_session(model, folder, imgsz=640, conf=REVIEW_CONFIDENCE):
    """Run the model over every photo in a session folder.

    The confidence floor is the review threshold, not the confirmed one — anything
    the diff might want to surface for a human has to survive detection first.
    """
    folder = Path(folder)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images in {folder}")

    by_view = {}
    for image_path in images:
        view = view_from_filename(image_path)
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
        by_view.setdefault(view, []).extend(damages_from_result(result, view, model.names))

    return by_view


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--checkout", required=True, type=Path, help="folder of check-out photos")
    ap.add_argument("--checkin", required=True, type=Path, help="folder of check-in photos")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=REVIEW_CONFIDENCE)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    model.to(args.device)

    before = detect_session(model, args.checkout, args.imgsz, args.conf)
    after = detect_session(model, args.checkin, args.imgsz, args.conf)
    report = compare(before, after)

    summary = report.summary()
    print(f"\n  check-out: {sum(len(v) for v in before.values())} damages across {len(before)} views")
    print(f"  check-in : {sum(len(v) for v in after.values())} damages across {len(after)} views\n")

    for view in report.views:
        if not (view.confirmed or view.needs_review):
            continue
        print(f"  {view.view}")
        for damage in view.confirmed:
            print(f"      NEW      {damage.class_name:<14} conf {damage.confidence:.2f}"
                  f"  area {damage.area_fraction * 100:.2f}% of frame")
        for damage in view.needs_review:
            print(f"      REVIEW   {damage.class_name:<14} conf {damage.confidence:.2f}"
                  f"  area {damage.area_fraction * 100:.2f}% of frame")

    if not (report.confirmed or report.needs_review):
        print("  No new damage found.")

    print(f"\n  {summary}\n")
    if report.missing_views:
        print(f"  !! views present in only one session, not compared: {report.missing_views}\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "views": [
                {
                    "view": v.view,
                    "new_damage": [d.to_dict() for d in v.new_damage],
                    "pre_existing": [d.to_dict() for d in v.pre_existing],
                    "repaired": [d.to_dict() for d in v.repaired],
                }
                for v in report.views
            ],
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  written: {args.out}\n")


if __name__ == "__main__":
    main()

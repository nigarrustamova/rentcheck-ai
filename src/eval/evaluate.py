"""Evaluate a trained checkpoint and write the results table the paper needs.

Reports per-class AP alongside the overall mAP. That is not decoration: CarDD is
badly imbalanced — scratch is 41% of the instances and tire flat 3.6% — so a single
averaged number hides which classes the model actually handles. The course brief
asks for per-group results wherever the data is imbalanced, and this is where they
come from.

For a segmentation checkpoint both box and mask metrics are reported, since the
mask numbers are the ones that matter for the damage-area estimate in the report.

    python src/eval/evaluate.py --weights runs/main_seg/weights/best.pt --split val
    python src/eval/evaluate.py --weights runs/main_seg/weights/best.pt --split test --confirm-test

The test split is gated behind --confirm-test on purpose: it is meant to be touched
once, for the final numbers, and an accidental habit of evaluating on it during
development is exactly the leakage the brief penalises.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]


def collect_rows(metrics, model_names):
    """One row per class, plus an 'all' row, from an Ultralytics results object.

    class_result(i) returns (precision, recall, ap50, ap50-95) for the i-th class
    that actually appeared in the split — classes with no instances are absent, so
    we index through ap_class_index rather than assuming all six are present.
    """
    has_masks = getattr(metrics, "seg", None) is not None

    rows = []
    for i, class_index in enumerate(metrics.box.ap_class_index):
        box_p, box_r, box_ap50, box_ap = metrics.box.class_result(i)
        row = {
            "class": model_names[class_index],
            "box_precision": box_p,
            "box_recall": box_r,
            "box_ap50": box_ap50,
            "box_ap50_95": box_ap,
        }
        if has_masks:
            mask_p, mask_r, mask_ap50, mask_ap = metrics.seg.class_result(i)
            row.update({
                "mask_precision": mask_p,
                "mask_recall": mask_r,
                "mask_ap50": mask_ap50,
                "mask_ap50_95": mask_ap,
            })
        rows.append(row)

    overall = {
        "class": "all",
        "box_precision": metrics.box.mp,
        "box_recall": metrics.box.mr,
        "box_ap50": metrics.box.map50,
        "box_ap50_95": metrics.box.map,
    }
    if has_masks:
        overall.update({
            "mask_precision": metrics.seg.mp,
            "mask_recall": metrics.seg.mr,
            "mask_ap50": metrics.seg.map50,
            "mask_ap50_95": metrics.seg.map,
        })
    rows.append(overall)
    return rows


def as_markdown(rows):
    headers = [k for k in rows[0] if k != "class"]
    lines = [
        "| class | " + " | ".join(h.replace("_", " ") for h in headers) + " |",
        "|" + "---|" * (len(headers) + 1),
    ]
    for row in rows:
        cells = [f"{row[h]:.3f}" for h in headers]
        name = f"**{row['class']}**" if row["class"] == "all" else row["class"]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=None,
                    help="dataset yaml; defaults to the one the model was trained on")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--confirm-test", action="store_true",
                    help="required to evaluate on the test split")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "results")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit(
            "Refusing to evaluate on the test split without --confirm-test.\n"
            "The test set is for the final numbers only; use --split val while developing."
        )

    if not args.weights.exists():
        raise SystemExit(f"No such checkpoint: {args.weights}")

    model = YOLO(str(args.weights))
    val_kwargs = {
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "plots": True,
    }
    if args.data is not None:
        val_kwargs["data"] = str(args.data)

    metrics = model.val(**val_kwargs)
    rows = collect_rows(metrics, model.names)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.weights.parent.parent.name}_{args.split}"

    csv_path = args.out / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = as_markdown(rows)
    md_path = args.out / f"{stem}.md"
    md_path.write_text(
        f"# {args.weights.parent.parent.name} — {args.split} split\n\n{table}\n\n"
        f"Speed per image: {metrics.speed['preprocess']:.1f} ms preprocess, "
        f"{metrics.speed['inference']:.1f} ms inference, "
        f"{metrics.speed['postprocess']:.1f} ms postprocess "
        f"({args.device}, imgsz {args.imgsz}).\n",
        encoding="utf-8",
    )

    print("\n" + table)
    print(f"\n  model size: {args.weights.stat().st_size / 1e6:.1f} MB")
    print(f"  speed/image: {metrics.speed['inference']:.1f} ms inference on {args.device}")
    print(f"  written: {csv_path}")
    print(f"  written: {md_path}\n")


if __name__ == "__main__":
    main()

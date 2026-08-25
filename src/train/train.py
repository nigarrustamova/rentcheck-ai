"""Train a YOLO model from a config file.

Every run — the smoke test, the baseline, the main model, the ablations — goes
through this one script with a different config, so the only thing that varies
between them is recorded in version control rather than in someone's shell
history. The resolved settings are written next to the weights.

Usage:
    python src/train/train.py --config configs/smoke.yaml
    python src/train/train.py --config configs/main.yaml --device 0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

# Ultralytics otherwise writes runs wherever its machine-global settings point,
# which on a shared machine is some other project's folder.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything the trainer is allowed to take from a config file. Anything else is
# a typo, and silently ignoring typos in a config is how experiments become
# irreproducible.
ALLOWED_KEYS = {
    "model", "data", "task", "epochs", "imgsz", "batch", "seed", "device",
    "workers", "patience", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "freeze", "pretrained", "amp", "cache", "cos_lr",
    "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "erasing", "auto_augment", "single_cls", "rect", "save_period", "val",
    "plots", "project", "name", "exist_ok", "resume", "deterministic", "fraction",
}


def set_seeds(seed):
    """Ultralytics seeds torch itself, but not the modules we call around it."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path):
    with open(path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    unknown = set(config) - ALLOWED_KEYS
    if unknown:
        raise SystemExit(f"Unknown key(s) in {path}: {sorted(unknown)}")
    for required in ("model", "data"):
        if required not in config:
            raise SystemExit(f"{path} is missing the '{required}' key")
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--device", default=None, help="override the config's device, e.g. 0 or cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override for a quick check")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the config's batch size, e.g. for a smaller GPU")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run this config already started, from its last checkpoint")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch

    config.setdefault("seed", 42)
    config.setdefault("deterministic", True)
    config.setdefault("project", "runs")
    config.setdefault("name", args.config.stem)
    config.setdefault("exist_ok", True)

    project = Path(config["project"])
    config["project"] = str(project if project.is_absolute() else REPO_ROOT / project)

    # Relative paths in a config mean "relative to the repository", not to whatever
    # directory the command happened to be launched from.
    data = Path(config["data"])
    config["data"] = str(data if data.is_absolute() else REPO_ROOT / data)

    if args.resume:
        # The booked GPU window can end mid-training, so picking a run back up has
        # to be one flag, not a hand-edited config.
        last = Path(config["project"]) / config["name"] / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume from: {last}")
        config["model"] = str(last)
        config["resume"] = True
        print(f"  resuming from {last}")

    if not Path(config["data"]).exists():
        raise SystemExit(f"Dataset config not found: {config['data']}\nRun src/data/coco_to_yolo.py first.")

    set_seeds(config["seed"])

    print(f"\n=== {args.config.name} ===")
    for key in ("model", "data", "epochs", "imgsz", "batch", "device", "seed"):
        if key in config:
            print(f"  {key:<8} {config[key]}")
    print()

    model = YOLO(config.pop("model"))
    results = model.train(**config)

    # Keep the exact settings beside the weights — "which config produced this?"
    # is the first question asked of any result.
    run_dir = Path(results.save_dir)
    (run_dir / "resolved_config.json").write_text(
        json.dumps({"config_file": str(args.config), **config}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  run directory: {run_dir}")


if __name__ == "__main__":
    main()

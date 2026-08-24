# RentCheck AI — Automated Car Damage Detection for Rent-a-Car Companies

**Deep Learning Final Project · DLE-AI-202 · Cohort I 2026 · Track 3 (Industry Product)**

A tool that lets a rent-a-car operator photograph a vehicle at check-out and check-in,
automatically detects and localizes body damage in the photos, and produces a
before/after comparison report — so damage disputes are resolved with evidence
instead of guesswork.

---

## Team

Listed alphabetically by surname, matching the author order used in the paper.

| Name | Role / module |
|---|---|
| Aliyev, Orkhan | Baseline model, evaluation, ablation study |
| Mirzayeva, Laman | Web UI, PDF report generation, slide deck |
| Rustamova, Nigar | Data pipeline, splits, augmentation, training runs |
| Samadov, Nijat | FastAPI service, check-out/check-in diff logic, Docker |

Work is divided by module, but every member is expected to be able to explain
any part of the system at the defense.

## Status

Scaffold only — data pipeline, training, and API are not implemented yet.

---

## 1. Setup

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## 2. Getting the data

**Primary dataset — CarDD** (4,000 images, 9,000+ annotated instances, 6 damage
classes, bounding-box + pixel-level annotations):

1. Go to https://cardd-ustc.github.io/
2. Fill in the licence form and email the authors to request the download link.
3. Extract the archive into `data/raw/cardd/`.

**Fallback / supplementary — Kaggle:**

```bash
kaggle datasets download -d nasimetemadi/car-damage-detection -p data/raw/kaggle --unzip
```

Before anything else, inspect whatever you have just downloaded — image count,
annotation format, class balance, resolution, and unreadable files:

```bash
python src/data/check_dataset.py --root data/raw/cardd
```

Then convert the COCO annotations to YOLO format and build the fixed split:

```bash
python src/data/coco_to_yolo.py --src data/raw/cardd --dst data/processed
```

```bash
python src/data/make_splits.py --data data/processed --seed 42
```

Nothing under `data/` is tracked by git.

## 3. Reproducing the headline result

A clean checkout plus the data step above, then a single command:

```bash
bash run_all.sh
```

This runs: data checks → baseline training → main training → evaluation on the
held-out test set → results table and figures into `report/figures/`.

## 4. Running the demo

```bash
docker build -t rentcheck-ai .
```

```bash
docker run -p 8000:8000 rentcheck-ai
```

Then open http://localhost:8000 — upload a check-out photo set and a check-in
photo set, and download the damage report.

---

## Reproducibility

- Fixed seed (`42`) for splits, training, and evaluation.
- Train/validation/test split committed under `data/splits/` (filename lists only).
- The test set is touched once, for the final numbers.
- Pinned dependency versions in `requirements.txt`.

## Licence and data usage

CarDD is used under its own licence terms (see the dataset's licence form).
No personal data is collected. No site is scraped in violation of its terms of service.

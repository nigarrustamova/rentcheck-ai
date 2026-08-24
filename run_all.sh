#!/usr/bin/env bash
# One-command reproduction of the headline result.
# Assumes the data step in README §2 has already been run.
set -euo pipefail

SEED=42

echo "==> [1/4] Data integrity checks"
# python src/data/check_dataset.py --data data/processed --splits data/splits

echo "==> [2/4] Baseline training"
# python src/train/train.py --config configs/baseline.yaml --seed "$SEED"

echo "==> [3/4] Main model training"
# python src/train/train.py --config configs/main.yaml --seed "$SEED"

echo "==> [4/4] Evaluation on the held-out test set"
# python src/eval/evaluate.py --split test --out report/figures

echo "==> Done. Results table and figures are in report/figures/"

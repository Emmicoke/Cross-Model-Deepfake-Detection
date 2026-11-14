#!/usr/bin/env bash
# One-command reproduction of the M2 experiments.
#   bash run_all.sh                       # defaults: CIFAKE in data/cifake, OOD sets found in data/ood/*
#   SIZE=32 SEEDS="42 43 44" EPOCHS=15 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
TRAIN_ROOT="${TRAIN_ROOT:-data/cifake}"
SIZE="${SIZE:-32}"
SEEDS="${SEEDS:-42}"
EPOCHS="${EPOCHS:-15}"
MAXPC="${MAXPC:-}"          # e.g. MAXPC=20000 to cap images per class
OOD=()
if [ -d data/ood ]; then for d in data/ood/*/; do OOD+=("${d%/}"); done; fi
FO=()
if [ -d data/ood_fake_only ]; then for d in data/ood_fake_only/*/; do FO+=("${d%/}"); done; fi
EXTRA=(); [ ${#FO[@]} -gt 0 ] && EXTRA+=(--ood-fake-only "${FO[@]}")
[ -n "$MAXPC" ] && EXTRA+=(--max-per-class "$MAXPC")

echo "== EDA =="
python scripts/eda.py --root "$TRAIN_ROOT/train" --size "$SIZE" --name train
for d in "${OOD[@]:-}"; do [ -n "$d" ] && python scripts/eda.py --root "$d" --size "$SIZE" --name "$(basename "$d")"; done

echo "== Baselines + main approach =="
for seed in $SEEDS; do
  for m in naive_majority naive_stats pixel_logreg lbp_logreg lbp_svm fft_radial_logreg fft_mlp fft_resnet18 pixel_resnet18; do
    python scripts/run_experiment.py --model "$m" --train-root "$TRAIN_ROOT" --size "$SIZE" \
      --seed "$seed" --epochs "$EPOCHS" ${OOD[@]:+--ood "${OOD[@]}"} ${EXTRA[@]:+"${EXTRA[@]}"}
  done
  # robustness-augmented variant of the main approach
  python scripts/run_experiment.py --model fft_resnet18 --augment --train-root "$TRAIN_ROOT" --size "$SIZE" \
    --seed "$seed" --epochs "$EPOCHS" ${OOD[@]:+--ood "${OOD[@]}"} ${EXTRA[@]:+"${EXTRA[@]}"}
done
python scripts/plot_results.py results
echo "Done. See results/results.csv, results/experiment_table.md, results/figures/"

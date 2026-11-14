#!/usr/bin/env bash
# M2 experiment driver.  EXP=e2|e1  SEEDS="42"  EPOCHS=20  bash scripts/run_m2.sh
#   e2: train data/g281,    OOD = data/ood/g281_*            results/g281/results.csv
#   e1: train data/tinygen, OOD = every data/ood/* folder    results/tinygen/results.csv
# Both evaluate every data/ood_fake_only/* folder as fake-class recall. One log per run in logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
EXP="${EXP:-e2}"; SEEDS="${SEEDS:-42}"; EPOCHS="${EPOCHS:-20}"; SIZE="${SIZE:-256}"
MODELS="${MODELS:-naive_majority naive_stats pixel_logreg lbp_logreg lbp_svm fft_radial_logreg fft_mlp fft_resnet18 pixel_resnet18 fft_resnet18+aug}"
if [ "$EXP" = e2 ]; then
  TRAIN=data/g281; OUT=results/g281/results.csv; OOD=(data/ood/g281_*)
else
  TRAIN=data/tinygen; OUT=results/tinygen/results.csv; OOD=(data/ood/*)
fi
FO=(data/ood_fake_only/*)
mkdir -p logs
for seed in $SEEDS; do
  for m in $MODELS; do
    aug=(); model="$m"; tag="$m"
    if [ "$m" = "fft_resnet18+aug" ]; then model=fft_resnet18; aug=(--augment); tag=fft_resnet18_aug; fi
    id="${EXP}_${tag}_s${seed}"
    if [ -f "$OUT" ] && grep -q "^${id}," "$OUT"; then echo "skip $id (done)"; continue; fi
    echo "== $id  $(date +%H:%M:%S)"
    "$PY" scripts/run_experiment.py --model "$model" "${aug[@]}" --train-root "$TRAIN" --size "$SIZE" \
      --seed "$seed" --epochs "$EPOCHS" --exp-id "$id" --out "$OUT" \
      --ood "${OOD[@]}" --ood-fake-only "${FO[@]}" > "logs/${id}.log" 2>&1
    rc=$?; tail -n 1 "logs/${id}.log"; [ $rc -ne 0 ] && echo "!! $id failed (rc=$rc), see logs/${id}.log"
  done
done

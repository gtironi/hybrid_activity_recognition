#!/usr/bin/env bash
# TSFEL-only baseline: SelectKBest + RandomForest.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUT="${REPO_ROOT}/experiments/tsfel_baseline_${DATASET_ID}_s${SEED}"

if [ -f "${OUT}/DONE" ]; then
    echo ">>> TSFEL baseline already complete, skipping"
    exit 0
fi
mkdir -p "${OUT}"

echo ">>> Starting TSFEL baseline at $(date)"
python -m random_forest_baseline.tsfel_baseline \
    --train "$TRAIN_PARQUET" \
    --test "$TEST_PARQUET" \
    --output_dir "$OUT" \
    --seed "$SEED" \
    2>&1 | tee -a "${OUT}/train.log"

touch "${OUT}/DONE"
echo ">>> TSFEL baseline done at $(date)"

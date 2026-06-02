#!/usr/bin/env bash
# RF default-vs-tuned hyperparameter search on the SAME DL val split.
# Mirrors run_tsfel_baseline.sh; uses TRAIN/TEST_PARQUET + VAL_FRACTION from _common.sh.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/rf_hypersearch"
mkdir -p "${EXPERIMENTS_BASE}"
OUT="${EXPERIMENTS_BASE}/rf_hypersearch_${DATASET_ID}_s${SEED}"

if [ -f "${OUT}/DONE" ]; then
    echo ">>> RF hypersearch already complete, skipping"
    exit 0
fi
mkdir -p "${OUT}"

echo ">>> Starting RF hypersearch (val_fraction=${VAL_FRACTION}) at $(date)"
python -m random_forest_baseline.rf_hypersearch \
    --train "$TRAIN_PARQUET" \
    --test "$TEST_PARQUET" \
    --val_fraction "$VAL_FRACTION" \
    --seed "$SEED" \
    --output_dir "$OUT" \
    2>&1 | tee -a "${OUT}/train.log"

touch "${OUT}/DONE"
echo ">>> RF hypersearch done at $(date)"

#!/usr/bin/env bash
# K-fold subject-level cross-validation orchestrator.
#
# For each of N_FOLDS:
#   1. Window train.parquet / test.parquet for the fold (TSFEL manifest fitted on
#      that fold's train set only — no cross-fold leakage)
#   2. RF baseline + TSFEL-MLP (fast)
#   3. CNN+LSTM, Robust, PatchTST (each: deep_only + hybrid, with/without pretrain)
#
# Pretraining (TS2Vec for CNN-based, MAE for PatchTST) is GLOBAL: done once on
# the separate raw unlabeled dataset (Time_Adj_Raw_Data.csv). Same pretrained
# encoder is reused across all folds — no leakage because the raw dataset is
# disjoint from the labeled AcTBeCalf data.
#
# Existing single-split scripts (run_all.sh, run_cnn_lstm.sh, etc.) are NOT
# affected. This script is fully self-contained.
#
# Usage:
#   bash scripts/experiments/run_kfold_cv.sh
# Override via env vars:
#   N_FOLDS=5 KFOLD_DATA_DIR=dataset/processed/kfold bash scripts/experiments/run_kfold_cv.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"

# --- Config ---
N_FOLDS="${N_FOLDS:-5}"
KFOLD_DATA_DIR="${KFOLD_DATA_DIR:-${REPO_ROOT}/dataset/processed/kfold}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
KFOLD_RUN_BASE="${KFOLD_RUN_BASE:-${REPO_ROOT}/experiments/kfold/runs/kfold_${TIMESTAMP}}"
mkdir -p "${KFOLD_RUN_BASE}"

echo ">>> K-fold run: ${KFOLD_RUN_BASE}"
echo ">>> N_FOLDS=${N_FOLDS}  PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}  PATCHTST_PRETRAIN_EPOCHS=${PATCHTST_PRETRAIN_EPOCHS}"

# --- 1. Generate fold parquets (subject-level partition + label filter) ---
if [ ! -f "${KFOLD_DATA_DIR}/fold_assignments.json" ]; then
    echo ">>> Generating ${N_FOLDS} folds → ${KFOLD_DATA_DIR}"
    python "${REPO_ROOT}/scripts/generate_kfold_splits.py" \
        --csv "${REPO_ROOT}/dataset/AcTBeCalf.csv" \
        --out-dir "${KFOLD_DATA_DIR}" \
        --n-folds "${N_FOLDS}"
else
    echo ">>> ${KFOLD_DATA_DIR}/fold_assignments.json exists — skipping fold generation"
fi

# --- 2. Global pretraining (idempotent; reused across all folds) ---
ensure_raw_parquet
run_ts2vec_pretrain_raw cnn_lstm >/dev/null
run_patchtst_mae_raw             >/dev/null

TS2VEC_CNNLSTM_DIR=$(ts2vec_pretrain_raw_dir cnn_lstm)
PATCHTST_PRETRAIN_DIR=$(patchtst_mae_raw_dir)

TS2VEC_CNNLSTM_CKPT="${TS2VEC_CNNLSTM_DIR}/ts2vec_best.pt"
PATCHTST_CKPT="${PATCHTST_PRETRAIN_DIR}/pretrain_ep${PATCHTST_PRETRAIN_EPOCHS}.pt"

for ckpt in "$TS2VEC_CNNLSTM_CKPT" "$PATCHTST_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "!!! Pretrain checkpoint missing: $ckpt"
        exit 1
    fi
done

# --- 3. Per-fold experiments ---
for k in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DATA_DIR="${KFOLD_DATA_DIR}/fold_${k}"
    FOLD_RUN_DIR="${KFOLD_RUN_BASE}/fold_${k}"
    mkdir -p "${FOLD_RUN_DIR}"

    if [ ! -f "${FOLD_DATA_DIR}/train.parquet" ] || [ ! -f "${FOLD_DATA_DIR}/test.parquet" ]; then
        echo "!!! fold_${k}: train.parquet or test.parquet missing in ${FOLD_DATA_DIR}"
        exit 1
    fi

    echo ""
    echo "========================================================"
    echo ">>> FOLD ${k} / $((N_FOLDS - 1))"
    echo "========================================================"

    # --- 3a. Windowing for this fold ---
    FOLD_WIN_TRAIN="${FOLD_DATA_DIR}/windowed_train.parquet"
    FOLD_WIN_TEST="${FOLD_DATA_DIR}/windowed_test.parquet"
    FOLD_MANIFEST="${FOLD_DATA_DIR}/tsfel_feature_manifest.json"

    if [ ! -f "$FOLD_WIN_TRAIN" ] || [ ! -f "$FOLD_MANIFEST" ]; then
        echo ">>> fold_${k}: windowing train + discovering TSFEL features"
        python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
            --input "${FOLD_DATA_DIR}/train.parquet" \
            --output "$FOLD_WIN_TRAIN" \
            --feature-manifest-out "$FOLD_MANIFEST"
    else
        echo ">>> fold_${k}: train windowing already done, skipping"
    fi

    if [ ! -f "$FOLD_WIN_TEST" ]; then
        echo ">>> fold_${k}: windowing test (apply manifest)"
        python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
            --input "${FOLD_DATA_DIR}/test.parquet" \
            --output "$FOLD_WIN_TEST" \
            --feature-manifest-in "$FOLD_MANIFEST"
    else
        echo ">>> fold_${k}: test windowing already done, skipping"
    fi

    # Override paths for this fold's experiments
    export TRAIN_PARQUET="$FOLD_WIN_TRAIN"
    export TEST_PARQUET="$FOLD_WIN_TEST"
    export EXPERIMENTS_BASE="$FOLD_RUN_DIR"

    # --- 3b. RF baseline (FIRST, fastest) ---
    echo ""
    echo ">>> fold_${k}: RF baseline"
    RF_OUT="${EXPERIMENTS_BASE}/tsfel_baseline_${DATASET_ID}_s${SEED}"
    if [ ! -f "${RF_OUT}/DONE" ]; then
        mkdir -p "$RF_OUT"
        python -m random_forest_baseline.tsfel_baseline \
            --train "$TRAIN_PARQUET" \
            --test "$TEST_PARQUET" \
            --output_dir "$RF_OUT" \
            --seed "$SEED" \
            2>&1 | tee -a "${RF_OUT}/train.log"
        touch "${RF_OUT}/DONE"
    else
        echo ">>> fold_${k}: RF baseline already done, skipping"
    fi

    # --- 3c. CNN+LSTM: deep_only/hybrid × fromscratch/frompretrain ---
    for MODE in deep_only hybrid; do
        echo ""
        echo ">>> fold_${k}: cnn_lstm ${MODE} fromscratch"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch run_experiment cnn_lstm "$MODE"

        echo ">>> fold_${k}: cnn_lstm ${MODE} frompretrain_raw_ep${PRETRAIN_EPOCHS}"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_experiment cnn_lstm "$MODE" --init_encoder_from "$TS2VEC_CNNLSTM_CKPT"
    done

    # --- 3d. PatchTST: deep_only/hybrid × fromscratch/frompretrain ---
    for MODE in deep_only hybrid; do
        echo ""
        echo ">>> fold_${k}: patchtst ${MODE} fromscratch"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX=fromscratch run_experiment patchtst "$MODE"

        echo ">>> fold_${k}: patchtst ${MODE} frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" \
            run_experiment patchtst "$MODE" --patchtst_checkpoint "$PATCHTST_CKPT"
    done

    echo ""
    echo ">>> FOLD ${k} done"
done

# --- 4. Aggregate ---
echo ""
echo "========================================================"
echo ">>> Aggregating results"
echo "========================================================"
python "${REPO_ROOT}/scripts/aggregate_kfold_results.py" \
    --run-dir "$KFOLD_RUN_BASE" \
    --n-folds "$N_FOLDS"

echo ""
echo ">>> K-fold CV complete: ${KFOLD_RUN_BASE}"

#!/usr/bin/env bash
# Append TSFEL feature-set experiments to the existing paper6c kfold run.
#
# The hc/catch22/rocket sets already ran under:
#   experiments/kfold_paper6c_w125/runs/kfold_2026-06-02_13-25-15/
# This script adds a parallel "tsfel" branch using TSFEL-windowed parquets
# built from the same 6-class remapped fold splits.
#
# Usage:
#   bash scripts/experiments/run_kfold_paper6c_tsfel.sh
# Overrides:
#   KFOLD_RUN_DIR=experiments/kfold_paper6c_w125/runs/kfold_2026-06-02_13-25-15 \
#   N_FOLDS=5 SEED=2026 \
#   bash scripts/experiments/run_kfold_paper6c_tsfel.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pretrain checkpoints were built at w125 with DATASET_ID=AcTBeCalf_paper_w125.
export WINDOW_LEN=125
export WINDOW_STRIDE=62
export DATASET_ID="AcTBeCalf_paper_w125"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

N_FOLDS="${N_FOLDS:-5}"
KFOLD_DATA_DIR="${KFOLD_DATA_DIR:-${REPO_ROOT}/dataset/processed/kfold_paper6c_w125}"
KFOLD_RUN_DIR="${KFOLD_RUN_DIR:-${REPO_ROOT}/experiments/kfold_paper6c_w125/runs/kfold_2026-06-02_13-25-15}"
LABEL_MAP="${REPO_ROOT}/scripts/label_maps/paper_6class.json"

if [ ! -d "$KFOLD_RUN_DIR" ]; then
    echo "ERROR: run dir not found: ${KFOLD_RUN_DIR}" >&2
    exit 1
fi
if [ ! -f "$LABEL_MAP" ]; then
    echo "ERROR: label map not found: ${LABEL_MAP}" >&2
    exit 1
fi

# Resolve pretrain checkpoints (built with DATASET_ID=AcTBeCalf_paper_w125).
CNN_CKPT="${PRETRAIN_BASE}/ts2vec_pretrain_cnn_lstm_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}/ts2vec_best.pt"
ROB_CKPT="${PRETRAIN_BASE}/ts2vec_pretrain_robust_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}/ts2vec_best.pt"
PATCHTST_CKPT="${PRETRAIN_BASE}/patchtst_pretrain_raw_${DATASET_ID}_ep${PATCHTST_PRETRAIN_EPOCHS}_bs${PATCHTST_PRETRAIN_BATCH_SIZE}_lr${PRETRAIN_LR}_s${SEED}/best.pt"

for ckpt in "$CNN_CKPT" "$ROB_CKPT" "$PATCHTST_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: pretrain checkpoint missing: $ckpt" >&2
        exit 1
    fi
done

echo ">>> kfold paper6c TSFEL | folds=${N_FOLDS} seed=${SEED} window=${WINDOW_LEN}"
echo ">>> run dir: ${KFOLD_RUN_DIR}"
echo ">>> cnn ckpt:      ${CNN_CKPT}"
echo ">>> robust ckpt:   ${ROB_CKPT}"
echo ">>> patchtst ckpt: ${PATCHTST_CKPT}"

for k in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DATA_DIR="${KFOLD_DATA_DIR}/fold_${k}"
    TSFEL_DIR="${FOLD_DATA_DIR}/tsfel"
    mkdir -p "$TSFEL_DIR"

    FOLD_WIN_TRAIN="${TSFEL_DIR}/windowed_tsfel_train.parquet"
    FOLD_WIN_TEST="${TSFEL_DIR}/windowed_tsfel_test.parquet"
    FOLD_MANIFEST="${TSFEL_DIR}/tsfel_feature_manifest.json"

    echo ""
    echo "========================================================"
    echo ">>> FOLD ${k} / $((N_FOLDS - 1)) — windowing"
    echo "========================================================"

    if [ ! -f "$FOLD_WIN_TRAIN" ] || [ ! -f "$FOLD_MANIFEST" ]; then
        echo ">>> fold_${k}: windowing train (6-class remap, w${WINDOW_LEN})"
        python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
            --input "${FOLD_DATA_DIR}/train.parquet" \
            --output "$FOLD_WIN_TRAIN" \
            --feature-manifest-out "$FOLD_MANIFEST" \
            --window-size "$WINDOW_LEN" \
            --overlap 0.5 \
            --remap-labels "$LABEL_MAP"
    else
        echo ">>> fold_${k}: train windowing already done, skipping"
    fi

    if [ ! -f "$FOLD_WIN_TEST" ]; then
        echo ">>> fold_${k}: windowing test (apply manifest)"
        python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
            --input "${FOLD_DATA_DIR}/test.parquet" \
            --output "$FOLD_WIN_TEST" \
            --feature-manifest-in "$FOLD_MANIFEST" \
            --window-size "$WINDOW_LEN" \
            --overlap 0.5 \
            --remap-labels "$LABEL_MAP"
    else
        echo ">>> fold_${k}: test windowing already done, skipping"
    fi

    # Override globals for this fold's supervised runs.
    export TRAIN_PARQUET="$FOLD_WIN_TRAIN"
    export TEST_PARQUET="$FOLD_WIN_TEST"
    export DATASET_ID="AcTBeCalf_paper6c_tsfel"
    export EXPERIMENTS_BASE="${KFOLD_RUN_DIR}/tsfel/fold_${k}"
    mkdir -p "${EXPERIMENTS_BASE}"

    echo ""
    echo ">>> fold_${k}: RF baseline"
    RF_OUT="${EXPERIMENTS_BASE}/rf_baseline_${DATASET_ID}_s${SEED}"
    if [ ! -f "${RF_OUT}/DONE" ]; then
        mkdir -p "$RF_OUT"
        python -m random_forest_baseline.tsfel_baseline \
            --train "$TRAIN_PARQUET" \
            --test  "$TEST_PARQUET" \
            --output_dir "$RF_OUT" \
            --seed "$SEED" \
            2>&1 | tee -a "${RF_OUT}/train.log"
        touch "${RF_OUT}/DONE"
    else
        echo ">>> fold_${k}: RF baseline already done, skipping"
    fi

    echo ""
    echo ">>> fold_${k}: RF hypersearch (val_fraction=${VAL_FRACTION})"
    RFH_OUT="${EXPERIMENTS_BASE}/rf_hypersearch_${DATASET_ID}_s${SEED}"
    if [ ! -f "${RFH_OUT}/DONE" ]; then
        mkdir -p "$RFH_OUT"
        python -m random_forest_baseline.rf_hypersearch \
            --train "$TRAIN_PARQUET" \
            --test  "$TEST_PARQUET" \
            --val_fraction "$VAL_FRACTION" \
            --seed "$SEED" \
            --output_dir "$RFH_OUT" \
            2>&1 | tee -a "${RFH_OUT}/train.log"
        touch "${RFH_OUT}/DONE"
    else
        echo ">>> fold_${k}: RF hypersearch already done, skipping"
    fi

    echo ""
    echo ">>> fold_${k}: TSFEL-MLP"
    BATCH_SIZE="$BATCH_SIZE_LARGE" run_experiment tsfel_mlp tsfel_only
    BATCH_SIZE="$BATCH_SIZE_LARGE" run_finetune   tsfel_mlp tsfel_only

    for MODE in deep_only hybrid; do
        echo ""
        echo ">>> fold_${k}: cnn_lstm ${MODE}"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
            run_experiment cnn_lstm "$MODE"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
            run_finetune   cnn_lstm "$MODE"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_experiment cnn_lstm "$MODE" --init_encoder_from "$CNN_CKPT"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_finetune   cnn_lstm "$MODE"

        echo ">>> fold_${k}: robust ${MODE}"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
            run_experiment robust "$MODE"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
            run_finetune   robust "$MODE"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_experiment robust "$MODE" --init_encoder_from "$ROB_CKPT"
        BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_finetune   robust "$MODE"

        echo ">>> fold_${k}: patchtst ${MODE}"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX=fromscratch \
            run_experiment patchtst "$MODE" --context_length "$WINDOW_LEN"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX=fromscratch \
            run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" \
            run_experiment patchtst "$MODE" --patchtst_checkpoint "$PATCHTST_CKPT" --context_length "$WINDOW_LEN"
        BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" \
            run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
    done

    echo ""
    echo ">>> FOLD ${k} done"
done

echo ""
echo ">>> All paper6c TSFEL kfold experiments done"
echo ">>> Results under: ${KFOLD_RUN_DIR}/tsfel/"

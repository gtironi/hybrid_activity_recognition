#!/usr/bin/env bash
# Append hc/catch22/rocket feature-set experiments to the existing original kfold run.
#
# The TSFEL-windowed experiments already ran under:
#   experiments/kfold/runs/kfold_2026-05-27_21-59-49/fold_*/
# This script adds parallel "hc", "catch22", "rocket" branches using the
# same paper-feature pipeline as run_paper_features.sh but per fold (no
# label remapping — original 10-class labels).
#
# Usage:
#   bash scripts/experiments/run_kfold_hc_catch22_rocket.sh
# Overrides:
#   KFOLD_RUN_DIR=experiments/kfold/runs/kfold_2026-05-27_21-59-49 \
#   N_FOLDS=5 SEED=2026 \
#   bash scripts/experiments/run_kfold_hc_catch22_rocket.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pretrain checkpoints were built at w75 with DATASET_ID=AcTBeCalf (default).
export WINDOW_LEN=75
export WINDOW_STRIDE=37
export DATASET_ID="AcTBeCalf"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

N_FOLDS="${N_FOLDS:-5}"
KFOLD_DATA_DIR="${KFOLD_DATA_DIR:-${REPO_ROOT}/dataset/processed/kfold}"
KFOLD_RUN_DIR="${KFOLD_RUN_DIR:-${REPO_ROOT}/experiments/kfold/runs/kfold_2026-05-27_21-59-49}"

if [ ! -d "$KFOLD_RUN_DIR" ]; then
    echo "ERROR: run dir not found: ${KFOLD_RUN_DIR}" >&2
    exit 1
fi

# Resolve pretrain checkpoints (built with DATASET_ID=AcTBeCalf, w75).
CNN_CKPT="${PRETRAIN_BASE}/ts2vec_pretrain_cnn_lstm_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}/ts2vec_best.pt"
ROB_CKPT="${PRETRAIN_BASE}/ts2vec_pretrain_robust_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}/ts2vec_best.pt"
PATCHTST_CKPT="${PRETRAIN_BASE}/patchtst_pretrain_raw_${DATASET_ID}_ep${PATCHTST_PRETRAIN_EPOCHS}_bs${PATCHTST_PRETRAIN_BATCH_SIZE}_lr${PRETRAIN_LR}_s${SEED}/best.pt"

for ckpt in "$CNN_CKPT" "$ROB_CKPT" "$PATCHTST_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: pretrain checkpoint missing: $ckpt" >&2
        exit 1
    fi
done

echo ">>> kfold hc/catch22/rocket | folds=${N_FOLDS} seed=${SEED} window=${WINDOW_LEN}"
echo ">>> run dir: ${KFOLD_RUN_DIR}"
echo ">>> cnn ckpt:      ${CNN_CKPT}"
echo ">>> robust ckpt:   ${ROB_CKPT}"
echo ">>> patchtst ckpt: ${PATCHTST_CKPT}"

for FEAT in hc catch22 rocket; do
    echo ""
    echo "========================================================"
    echo ">>> FEATURE SET: ${FEAT}"
    echo "========================================================"

    for k in $(seq 0 $((N_FOLDS - 1))); do
        FOLD_DATA_DIR="${KFOLD_DATA_DIR}/fold_${k}"
        FEAT_DIR="${FOLD_DATA_DIR}/${FEAT}"
        mkdir -p "$FEAT_DIR"

        FOLD_WIN_TRAIN="${FEAT_DIR}/windowed_${FEAT}_train.parquet"
        FOLD_WIN_TEST="${FEAT_DIR}/windowed_${FEAT}_test.parquet"
        ROCKET_MODEL="${FEAT_DIR}/rocket_model.joblib"

        echo ""
        echo ">>> fold_${k}: windowing ${FEAT}"

        EXTRA_TRAIN=()
        EXTRA_TEST=()
        if [ "$FEAT" = rocket ]; then
            EXTRA_TRAIN+=(--rocket-manifest-out "$ROCKET_MODEL")
            EXTRA_TEST+=(--rocket-manifest-in  "$ROCKET_MODEL")
        fi

        if [ ! -f "$FOLD_WIN_TRAIN" ]; then
            python "${REPO_ROOT}/scripts/prepare_paper_features_parquet.py" \
                --features "$FEAT" \
                --input "${FOLD_DATA_DIR}/train.parquet" \
                --output "$FOLD_WIN_TRAIN" \
                --window-size "$WINDOW_LEN" \
                --overlap 0.5 \
                --purity-threshold 0.9 \
                "${EXTRA_TRAIN[@]}"
        else
            echo ">>> fold_${k}/${FEAT}: train parquet already exists, skipping"
        fi

        if [ ! -f "$FOLD_WIN_TEST" ]; then
            python "${REPO_ROOT}/scripts/prepare_paper_features_parquet.py" \
                --features "$FEAT" \
                --input "${FOLD_DATA_DIR}/test.parquet" \
                --output "$FOLD_WIN_TEST" \
                --window-size "$WINDOW_LEN" \
                --overlap 0.5 \
                --purity-threshold 0.9 \
                "${EXTRA_TEST[@]}"
        else
            echo ">>> fold_${k}/${FEAT}: test parquet already exists, skipping"
        fi

        # Override globals for this fold/feature supervised runs.
        export TRAIN_PARQUET="$FOLD_WIN_TRAIN"
        export TEST_PARQUET="$FOLD_WIN_TEST"
        export DATASET_ID="AcTBeCalf_${FEAT}"
        export EXPERIMENTS_BASE="${KFOLD_RUN_DIR}/${FEAT}/fold_${k}"
        mkdir -p "${EXPERIMENTS_BASE}"

        echo ">>> fold_${k}/${FEAT}: RF baseline"
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
            echo ">>> fold_${k}/${FEAT}: RF baseline already done, skipping"
        fi

        echo ">>> fold_${k}/${FEAT}: RF hypersearch (val_fraction=${VAL_FRACTION})"
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
            echo ">>> fold_${k}/${FEAT}: RF hypersearch already done, skipping"
        fi

        echo ">>> fold_${k}/${FEAT}: TSFEL-MLP"
        BATCH_SIZE="$BATCH_SIZE_LARGE" run_experiment tsfel_mlp tsfel_only
        BATCH_SIZE="$BATCH_SIZE_LARGE" run_finetune   tsfel_mlp tsfel_only

        for MODE in deep_only hybrid; do
            echo ">>> fold_${k}/${FEAT}: cnn_lstm ${MODE}"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
                run_experiment cnn_lstm "$MODE"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
                run_finetune   cnn_lstm "$MODE"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_experiment cnn_lstm "$MODE" --init_encoder_from "$CNN_CKPT"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_finetune   cnn_lstm "$MODE"

            echo ">>> fold_${k}/${FEAT}: robust ${MODE}"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
                run_experiment robust "$MODE"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX=fromscratch \
                run_finetune   robust "$MODE"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_experiment robust "$MODE" --init_encoder_from "$ROB_CKPT"
            BATCH_SIZE="$BATCH_SIZE_LARGE" RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_finetune   robust "$MODE"

            echo ">>> fold_${k}/${FEAT}: patchtst ${MODE}"
            BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX=fromscratch \
                run_experiment patchtst "$MODE"
            BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX=fromscratch \
                run_finetune   patchtst "$MODE"
            BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" \
                run_experiment patchtst "$MODE" --patchtst_checkpoint "$PATCHTST_CKPT"
            BATCH_SIZE="$PATCHTST_BATCH_SIZE" RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" \
                run_finetune   patchtst "$MODE"
        done

        echo ">>> fold_${k}/${FEAT} done"
    done

    echo ">>> Feature set ${FEAT} complete"
done

echo ""
echo ">>> All hc/catch22/rocket kfold experiments done"
echo ">>> Results under: ${KFOLD_RUN_DIR}/{hc,catch22,rocket}/"

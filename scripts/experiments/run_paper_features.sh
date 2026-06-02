#!/usr/bin/env bash
# Paper feature experiments (Dissanayake et al., 2025): HC, Catch22, ROCKET.
#
# For each feature set:
#   - CNN-LSTM  × {deep_only, hybrid} × {fromscratch, frompretrain_raw_ep40} (+ finetune)
#   - Robust    × {deep_only, hybrid} × {fromscratch, frompretrain_raw_ep40} (+ finetune)
#   - PatchTST  × {deep_only, hybrid} × {fromscratch, frompretrain_raw_ep40} (+ finetune)
#   - tsfel_mlp tsfel_only (+ finetune)   -- "MLP baseline on paper features"
#   - RF baseline (random_forest_baseline.tsfel_baseline)
#
# No frozen encoder, no multi-checkpoint sweep — always last-epoch pretrain.
# PatchTST always reads raw signals (deep_only); hybrid adds the feature cols.
#
# Pretraining (TS2Vec + PatchTST MAE) runs ONCE on a paper-windowed raw parquet
# (3 s / 50 % overlap = 75 samples) and is shared across the 3 feature datasets.
# Datasets are built under dataset/processed/AcTBeCalf/paper_features/.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Paper-specific window config (must be set BEFORE sourcing _common.sh so the
# raw windowed parquet path / DATASET_ID encode 3 s @ 25 Hz windows).
# ---------------------------------------------------------------------------
export WINDOW_LEN="${WINDOW_LEN:-75}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-37}"   # 3 s with ~50% overlap @ 25 Hz
export DATASET_ID="${DATASET_ID:-AcTBeCalf_paper_w75}"
# Paper-windowed raw parquet for pretraining (independent of TSFEL pipeline).
export RAW_PARQUET="${RAW_PARQUET:-$(cd "${DIR}/../.." && pwd)/dataset/processed/pretrain_raw_windowed_w75.parquet}"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

# ---------------------------------------------------------------------------
# Dataset paths.
# ---------------------------------------------------------------------------
PAPER_DATA="${REPO_ROOT}/dataset/processed/AcTBeCalf/paper_w${WINDOW_LEN}"
mkdir -p "${PAPER_DATA}/hc" "${PAPER_DATA}/catch22" "${PAPER_DATA}/rocket"
ROCKET_MODEL="${PAPER_DATA}/rocket/rocket_model.joblib"

# Raw row-level parquets produced by scripts/dataset_processing.py.
SRC_TRAIN="${REPO_ROOT}/dataset/processed/AcTBeCalf/train.parquet"
SRC_TEST="${REPO_ROOT}/dataset/processed/AcTBeCalf/test.parquet"

if [ ! -f "$SRC_TRAIN" ] || [ ! -f "$SRC_TEST" ]; then
    echo "ERROR: ${SRC_TRAIN} or ${SRC_TEST} not found. Run scripts/dataset_processing.py first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the 3 feature parquets (idempotent: skip when output already exists).
# ---------------------------------------------------------------------------
build_feature_parquet() {
    # Usage: build_feature_parquet FEAT MODE
    #   FEAT in {hc, catch22, rocket}
    #   MODE in {train, test}
    local FEAT="$1" MODE="$2"
    local IN_PATH OUT_PATH
    if [ "$MODE" = train ]; then IN_PATH="$SRC_TRAIN"; else IN_PATH="$SRC_TEST"; fi
    OUT_PATH="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_${MODE}.parquet"

    if [ -f "$OUT_PATH" ]; then
        echo ">>> ${OUT_PATH} already exists, skipping"
        return 0
    fi

    local EXTRA=()
    if [ "$FEAT" = rocket ]; then
        if [ "$MODE" = train ]; then
            EXTRA+=(--rocket-manifest-out "$ROCKET_MODEL")
        else
            EXTRA+=(--rocket-manifest-in "$ROCKET_MODEL")
        fi
    fi

    echo ">>> Building ${FEAT}/${MODE} parquet → ${OUT_PATH}"
    python "${REPO_ROOT}/scripts/prepare_paper_features_parquet.py" \
        --features "$FEAT" \
        --input "$IN_PATH" \
        --output "$OUT_PATH" \
        --window-size "$WINDOW_LEN" \
        --overlap 0.5 \
        --purity-threshold 0.9 \
        "${EXTRA[@]}"
}

for FEAT in hc catch22 rocket; do
    # train MUST run before test for rocket (test mode needs the fitted manifest).
    build_feature_parquet "$FEAT" train
    build_feature_parquet "$FEAT" test
done

# ---------------------------------------------------------------------------
# Pretraining (shared across feature sets — uses raw windowed parquet).
# ---------------------------------------------------------------------------
echo ">>> Stage 1: encoder pretraining on paper-windowed raw data (${WINDOW_LEN} samples, best checkpoint)"
TS2VEC_CNN_DIR=$(ts2vec_pretrain_raw_dir "cnn_lstm")
TS2VEC_ROB_DIR=$(ts2vec_pretrain_raw_dir "robust")
PATCHTST_PRETRAIN_DIR=$(patchtst_mae_raw_dir)

ENCODER=cnn_lstm bash "${DIR}/pretrain_encoder.sh" || true
ENCODER=robust   bash "${DIR}/pretrain_encoder.sh" || true
bash "${DIR}/pretrain_patchtst.sh" || true

CNN_CKPT="${TS2VEC_CNN_DIR}/ts2vec_best.pt"
ROB_CKPT="${TS2VEC_ROB_DIR}/ts2vec_best.pt"
PATCHTST_CKPT="${PATCHTST_PRETRAIN_DIR}/best.pt"

[ -f "$CNN_CKPT" ]      || { echo "ERROR: missing $CNN_CKPT"; exit 1; }
[ -f "$ROB_CKPT" ]      || { echo "ERROR: missing $ROB_CKPT"; exit 1; }
[ -f "$PATCHTST_CKPT" ] || { echo "ERROR: missing $PATCHTST_CKPT"; exit 1; }

# ---------------------------------------------------------------------------
# Per-feature-set supervised + baseline runs.
# ---------------------------------------------------------------------------
PAPER_EXP_ROOT="${EXPERIMENTS_BASE}/paper_features"
mkdir -p "${PAPER_EXP_ROOT}"

for FEAT in hc catch22 rocket; do
    echo ""
    echo "============================================================"
    echo ">>> Feature set: ${FEAT}"
    echo "============================================================"

    export TRAIN_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_train.parquet"
    export TEST_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_test.parquet"
    export PRETRAIN_PARQUET="$TRAIN_PARQUET"
    export DATASET_ID="AcTBeCalf_paper_${FEAT}"
    export EXPERIMENTS_BASE="${PAPER_EXP_ROOT}/${FEAT}"
    mkdir -p "${EXPERIMENTS_BASE}"
    unset FREEZE_ENCODER || true

    # ----- CNN-LSTM & Robust (large batch) -------------------------------
    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    for ENC_AND_CKPT in "cnn_lstm:${CNN_CKPT}" "robust:${ROB_CKPT}"; do
        ENC="${ENC_AND_CKPT%%:*}"
        CKPT="${ENC_AND_CKPT##*:}"
        for MODE in deep_only hybrid; do
            RUN_SUFFIX=fromscratch        run_experiment "$ENC" "$MODE"
            RUN_SUFFIX=fromscratch        run_finetune   "$ENC" "$MODE"
            RUN_SUFFIX=frompretrain_raw_ep${PRETRAIN_EPOCHS} run_experiment "$ENC" "$MODE" \
                --init_encoder_from "$CKPT"
            RUN_SUFFIX=frompretrain_raw_ep${PRETRAIN_EPOCHS} run_finetune   "$ENC" "$MODE"
        done
    done

    # ----- PatchTST (smaller batch; deep_only reads raw signals only) ----
    export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
    for MODE in deep_only hybrid; do
        RUN_SUFFIX=fromscratch        run_experiment patchtst "$MODE"
        RUN_SUFFIX=fromscratch        run_finetune   patchtst "$MODE"
        RUN_SUFFIX=frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS} run_experiment patchtst "$MODE" \
            --patchtst_checkpoint "$PATCHTST_CKPT"
        RUN_SUFFIX=frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS} run_finetune   patchtst "$MODE"
    done

    # ----- MLP baseline on the paper features (tsfel_only mode) ---------
    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    unset RUN_SUFFIX || true
    run_experiment "tsfel_mlp" "tsfel_only"
    run_finetune   "tsfel_mlp" "tsfel_only"

    # ----- RandomForest baseline on the paper features ------------------
    RF_OUT="${EXPERIMENTS_BASE}/rf_baseline_${DATASET_ID}_s${SEED}"
    if [ -f "${RF_OUT}/DONE" ]; then
        echo ">>> RF baseline ${FEAT}: already complete, skipping"
    else
        mkdir -p "${RF_OUT}"
        echo ">>> RF baseline ${FEAT} at $(date)"
        python -m random_forest_baseline.tsfel_baseline \
            --train "$TRAIN_PARQUET" \
            --test  "$TEST_PARQUET" \
            --output_dir "$RF_OUT" \
            --seed "$SEED" \
            2>&1 | tee -a "${RF_OUT}/train.log"
        touch "${RF_OUT}/DONE"
    fi

    # ----- RF hyperparameter search on the SAME DL val split -----------
    RFH_OUT="${EXPERIMENTS_BASE}/rf_hypersearch_${DATASET_ID}_s${SEED}"
    if [ -f "${RFH_OUT}/DONE" ]; then
        echo ">>> RF hypersearch ${FEAT}: already complete, skipping"
    else
        mkdir -p "${RFH_OUT}"
        echo ">>> RF hypersearch ${FEAT} (val_fraction=${VAL_FRACTION}) at $(date)"
        python -m random_forest_baseline.rf_hypersearch \
            --train "$TRAIN_PARQUET" \
            --test  "$TEST_PARQUET" \
            --val_fraction "$VAL_FRACTION" \
            --seed "$SEED" \
            --output_dir "$RFH_OUT" \
            2>&1 | tee -a "${RFH_OUT}/train.log"
        touch "${RFH_OUT}/DONE"
    fi
done

echo ""
echo ">>> All paper-feature experiments finished at $(date)"
echo ">>> Results under: ${PAPER_EXP_ROOT}"

#!/usr/bin/env bash
# Paper feature experiments — 10-class ethogram.
#
# 9 named behaviours + Other:
#   Drinking | Eating | Grooming | Lying | Oral manipulation of pen
#   Play | Run | Standing | Walking | Other
#
# Step 0 (one-time): generate the 10-class parquets with dataset_processing.py:
#   python scripts/dataset_processing.py --label-map 10class
#
# Datasets are saved under dataset/processed/AcTBeCalf_10class/
# Feature parquets go under dataset/processed/AcTBeCalf_10class/paper_10class_w<WINDOW_LEN>/
# Pretrain reuses the same raw checkpoints as run_paper_features.sh (labels
# don't affect pretraining), so DATASET_ID is temporarily set to the non-10class
# value when resolving pretrain dirs.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WINDOW_LEN="${WINDOW_LEN:-75}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-37}"
export DATASET_ID="${DATASET_ID:-AcTBeCalf_paper_w${WINDOW_LEN}}"
export RAW_PARQUET="${RAW_PARQUET:-$(cd "${DIR}/../.." && pwd)/dataset/processed/pretrain_raw_windowed_w${WINDOW_LEN}.parquet}"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

LABEL_MAP="${REPO_ROOT}/scripts/label_maps/paper_10class.json"
if [ ! -f "$LABEL_MAP" ]; then
    echo "ERROR: label map not found at ${LABEL_MAP}" >&2
    exit 1
fi

SRC_TRAIN="${REPO_ROOT}/dataset/processed/AcTBeCalf_10class/train.parquet"
SRC_TEST="${REPO_ROOT}/dataset/processed/AcTBeCalf_10class/test.parquet"

if [ ! -f "$SRC_TRAIN" ] || [ ! -f "$SRC_TEST" ]; then
    echo "ERROR: 10-class parquets not found. Run first:" >&2
    echo "  python scripts/dataset_processing.py --label-map 10class" >&2
    exit 1
fi

PAPER_DATA="${REPO_ROOT}/dataset/processed/AcTBeCalf_10class/paper_10class_w${WINDOW_LEN}"
mkdir -p "${PAPER_DATA}/hc" "${PAPER_DATA}/catch22" "${PAPER_DATA}/rocket"
ROCKET_MODEL="${PAPER_DATA}/rocket/rocket_model.joblib"

build_feature_parquet() {
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
        --remap-labels "$LABEL_MAP" \
        "${EXTRA[@]}"
}

for FEAT in hc catch22 rocket; do
    build_feature_parquet "$FEAT" train
    build_feature_parquet "$FEAT" test
done

echo ">>> Stage 1: encoder pretraining (${WINDOW_LEN} samples)"
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

PAPER_EXP_ROOT="${EXPERIMENTS_BASE}/paper_10class"
mkdir -p "${PAPER_EXP_ROOT}"

for FEAT in hc catch22 rocket; do
    echo ""
    echo "============================================================"
    echo ">>> Feature set: ${FEAT}"
    echo "============================================================"

    export TRAIN_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_train.parquet"
    export TEST_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_test.parquet"
    export PRETRAIN_PARQUET="$TRAIN_PARQUET"
    export DATASET_ID="AcTBeCalf_paper10c_${FEAT}"
    export EXPERIMENTS_BASE="${PAPER_EXP_ROOT}/${FEAT}"
    mkdir -p "${EXPERIMENTS_BASE}"
    unset FREEZE_ENCODER || true

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

    export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
    for MODE in deep_only hybrid; do
        RUN_SUFFIX=fromscratch        run_experiment patchtst "$MODE" --context_length "$WINDOW_LEN"
        RUN_SUFFIX=fromscratch        run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
        RUN_SUFFIX=frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS} run_experiment patchtst "$MODE" \
            --patchtst_checkpoint "$PATCHTST_CKPT" --context_length "$WINDOW_LEN"
        RUN_SUFFIX=frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS} run_finetune   patchtst "$MODE" \
            --context_length "$WINDOW_LEN"
    done

    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    unset RUN_SUFFIX || true
    run_experiment "tsfel_mlp" "tsfel_only"
    run_finetune   "tsfel_mlp" "tsfel_only"

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
done

echo ""
echo ">>> All 10-class paper-feature experiments finished at $(date)"
echo ">>> Results under: ${PAPER_EXP_ROOT}"

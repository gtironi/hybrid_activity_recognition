#!/usr/bin/env bash
# Paper feature experiments — 10-class ethogram.
#
# 9 named behaviours + Other (everything else):
#   Drinking | Eating | Grooming | Lying | Oral manipulation of pen
#   Play | Run | Standing | Walking | Other
#
# Source parquets: dataset/processed/AcTBeCalf/train.parquet (already has Play).
# Label remapping to 10 classes is applied at feature-extraction time via
# --remap-labels, exactly like run_paper_6class.sh does for 6 classes.
#
# Usage:
#   bash scripts/experiments/run_paper_10class.sh
# Override window size:
#   WINDOW_LEN=125 bash scripts/experiments/run_paper_10class.sh

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WINDOW_LEN="${WINDOW_LEN:-75}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-37}"
# Pretrain dirs are keyed on DATASET_ID; temporarily use the non-10class value
# so ts2vec_pretrain_raw_dir / patchtst_mae_raw_dir resolve to existing checkpoints.
export DATASET_ID="${DATASET_ID:-AcTBeCalf_paper_w${WINDOW_LEN}}"
export RAW_PARQUET="${RAW_PARQUET:-$(cd "${DIR}/../.." && pwd)/dataset/processed/pretrain_raw_windowed_w${WINDOW_LEN}.parquet}"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

LABEL_MAP="${REPO_ROOT}/scripts/label_maps/paper_10class.json"
[ -f "$LABEL_MAP" ] || { echo "ERROR: label map not found: ${LABEL_MAP}" >&2; exit 1; }

SRC_TRAIN="${REPO_ROOT}/dataset/processed/AcTBeCalf/train.parquet"
SRC_TEST="${REPO_ROOT}/dataset/processed/AcTBeCalf/test.parquet"
[ -f "$SRC_TRAIN" ] || { echo "ERROR: ${SRC_TRAIN} not found" >&2; exit 1; }
[ -f "$SRC_TEST"  ] || { echo "ERROR: ${SRC_TEST} not found"  >&2; exit 1; }

PAPER_DATA="${REPO_ROOT}/dataset/processed/AcTBeCalf/paper_10class_w${WINDOW_LEN}"
mkdir -p "${PAPER_DATA}/hc" "${PAPER_DATA}/catch22" "${PAPER_DATA}/rocket" "${PAPER_DATA}/tsfel"
ROCKET_MODEL="${PAPER_DATA}/rocket/rocket_model.joblib"
TSFEL_MANIFEST="${PAPER_DATA}/tsfel/tsfel_feature_manifest.json"

# --- Build hc / catch22 / rocket feature parquets ---
build_feature_parquet() {
    local FEAT="$1" MODE="$2"
    local IN_PATH OUT_PATH
    if [ "$MODE" = train ]; then IN_PATH="$SRC_TRAIN"; else IN_PATH="$SRC_TEST"; fi
    OUT_PATH="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_${MODE}.parquet"

    if [ -f "$OUT_PATH" ]; then echo ">>> ${OUT_PATH} exists, skipping"; return 0; fi

    local EXTRA=()
    if [ "$FEAT" = rocket ]; then
        if [ "$MODE" = train ]; then EXTRA+=(--rocket-manifest-out "$ROCKET_MODEL")
        else                        EXTRA+=(--rocket-manifest-in  "$ROCKET_MODEL"); fi
    fi

    echo ">>> Building ${FEAT}/${MODE} → ${OUT_PATH}"
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

# --- Build tsfel feature parquets ---
TSFEL_TRAIN="${PAPER_DATA}/tsfel/windowed_tsfel_train.parquet"
TSFEL_TEST="${PAPER_DATA}/tsfel/windowed_tsfel_test.parquet"

if [ ! -f "$TSFEL_TRAIN" ] || [ ! -f "$TSFEL_MANIFEST" ]; then
    echo ">>> Building tsfel/train → ${TSFEL_TRAIN}"
    python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
        --input "$SRC_TRAIN" \
        --output "$TSFEL_TRAIN" \
        --feature-manifest-out "$TSFEL_MANIFEST" \
        --window-size "$WINDOW_LEN" \
        --overlap 0.5 \
        --remap-labels "$LABEL_MAP"
else
    echo ">>> tsfel/train exists, skipping"
fi

if [ ! -f "$TSFEL_TEST" ]; then
    echo ">>> Building tsfel/test → ${TSFEL_TEST}"
    python "${REPO_ROOT}/scripts/prepare_windowed_parquet.py" \
        --input "$SRC_TEST" \
        --output "$TSFEL_TEST" \
        --feature-manifest-in "$TSFEL_MANIFEST" \
        --window-size "$WINDOW_LEN" \
        --overlap 0.5 \
        --remap-labels "$LABEL_MAP"
else
    echo ">>> tsfel/test exists, skipping"
fi

# --- Pretrain (idempotent) ---
echo ">>> Encoder pretraining (w${WINDOW_LEN})"
TS2VEC_CNN_DIR=$(ts2vec_pretrain_raw_dir "cnn_lstm")
TS2VEC_ROB_DIR=$(ts2vec_pretrain_raw_dir "robust")
PATCHTST_PRETRAIN_DIR=$(patchtst_mae_raw_dir)

ENCODER=cnn_lstm bash "${DIR}/pretrain_encoder.sh" || true
ENCODER=robust   bash "${DIR}/pretrain_encoder.sh" || true
bash "${DIR}/pretrain_patchtst.sh" || true

CNN_CKPT="${TS2VEC_CNN_DIR}/ts2vec_best.pt"
ROB_CKPT="${TS2VEC_ROB_DIR}/ts2vec_best.pt"
PATCHTST_CKPT="${PATCHTST_PRETRAIN_DIR}/best.pt"

[ -f "$CNN_CKPT" ]      || { echo "ERROR: missing ${CNN_CKPT}";      exit 1; }
[ -f "$ROB_CKPT" ]      || { echo "ERROR: missing ${ROB_CKPT}";      exit 1; }
[ -f "$PATCHTST_CKPT" ] || { echo "ERROR: missing ${PATCHTST_CKPT}"; exit 1; }

PAPER_EXP_ROOT="${EXPERIMENTS_BASE}/paper_10class"
mkdir -p "${PAPER_EXP_ROOT}"

# --- Per-feature-set supervised experiments ---
for FEAT in hc catch22 rocket tsfel; do
    echo ""
    echo "============================================================"
    echo ">>> Feature set: ${FEAT}  (w${WINDOW_LEN})"
    echo "============================================================"

    if [ "$FEAT" = tsfel ]; then
        export TRAIN_PARQUET="$TSFEL_TRAIN"
        export TEST_PARQUET="$TSFEL_TEST"
    else
        export TRAIN_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_train.parquet"
        export TEST_PARQUET="${PAPER_DATA}/${FEAT}/windowed_${FEAT}_test.parquet"
    fi
    export PRETRAIN_PARQUET="$TRAIN_PARQUET"
    export DATASET_ID="AcTBeCalf_paper10c_${FEAT}"
    export EXPERIMENTS_BASE="${PAPER_EXP_ROOT}/${FEAT}"
    mkdir -p "${EXPERIMENTS_BASE}"
    unset FREEZE_ENCODER || true

    # RF baseline
    RF_OUT="${EXPERIMENTS_BASE}/rf_baseline_${DATASET_ID}_s${SEED}"
    if [ ! -f "${RF_OUT}/DONE" ]; then
        mkdir -p "$RF_OUT"
        echo ">>> RF baseline ${FEAT}"
        python -m random_forest_baseline.tsfel_baseline \
            --train "$TRAIN_PARQUET" --test "$TEST_PARQUET" \
            --output_dir "$RF_OUT" --seed "$SEED" \
            2>&1 | tee -a "${RF_OUT}/train.log"
        touch "${RF_OUT}/DONE"
    else
        echo ">>> RF baseline ${FEAT}: already done, skipping"
    fi

    # RF hypersearch
    RFH_OUT="${EXPERIMENTS_BASE}/rf_hypersearch_${DATASET_ID}_s${SEED}"
    if [ ! -f "${RFH_OUT}/DONE" ]; then
        mkdir -p "$RFH_OUT"
        echo ">>> RF hypersearch ${FEAT} (val_fraction=${VAL_FRACTION})"
        python -m random_forest_baseline.rf_hypersearch \
            --train "$TRAIN_PARQUET" --test "$TEST_PARQUET" \
            --val_fraction "$VAL_FRACTION" --seed "$SEED" \
            --output_dir "$RFH_OUT" \
            2>&1 | tee -a "${RFH_OUT}/train.log"
        touch "${RFH_OUT}/DONE"
    else
        echo ">>> RF hypersearch ${FEAT}: already done, skipping"
    fi

    # TSFEL-MLP
    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    unset RUN_SUFFIX || true
    run_experiment tsfel_mlp tsfel_only
    run_finetune   tsfel_mlp tsfel_only

    # CNN+LSTM and Robust
    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    for ENC_AND_CKPT in "cnn_lstm:${CNN_CKPT}" "robust:${ROB_CKPT}"; do
        ENC="${ENC_AND_CKPT%%:*}"
        CKPT="${ENC_AND_CKPT##*:}"
        for MODE in deep_only hybrid; do
            RUN_SUFFIX=fromscratch \
                run_experiment "$ENC" "$MODE"
            RUN_SUFFIX=fromscratch \
                run_finetune   "$ENC" "$MODE"
            RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_experiment "$ENC" "$MODE" --init_encoder_from "$CKPT"
            RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
                run_finetune   "$ENC" "$MODE"
        done
    done

    # PatchTST
    export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
    for MODE in deep_only hybrid; do
        RUN_SUFFIX=fromscratch \
            run_experiment patchtst "$MODE" --context_length "$WINDOW_LEN"
        RUN_SUFFIX=fromscratch \
            run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
        RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_experiment patchtst "$MODE" \
                --patchtst_checkpoint "$PATCHTST_CKPT" --context_length "$WINDOW_LEN"
        RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" \
            run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
    done

    echo ">>> Feature set ${FEAT} done"
done

echo ""
echo ">>> All 10-class paper-feature experiments finished at $(date)"
echo ">>> Results under: ${PAPER_EXP_ROOT}"

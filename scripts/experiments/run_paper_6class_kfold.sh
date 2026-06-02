#!/usr/bin/env bash
# K-fold cross-validation for the 6-class paper-feature pipeline (hc/catch22/rocket).
#
# Same grid as run_paper_6class.sh but evaluated over N_FOLDS non-overlapping
# subject folds (reusing the same fold assignments as the TSFEL k-fold for
# cross-experiment comparability). Window = 125 samples (5 s @ 25 Hz), 6 classes.
#
# Per fold, per feature family (hc/catch22/rocket):
#   cnn_lstm/robust/patchtst × {deep_only,hybrid} × {scratch,pretrain}  (12)
#   + tsfel_mlp (MLP on the paper features)                              (1)
#   + RF baseline                                                        (1)
# = 14 experiments × 3 families × 5 folds.
#
# Pretraining (TS2Vec / MAE, window 125) is GLOBAL — reused from
# experiments/pretrain/ (already produced for the single-split paper run).
#
# Output:  experiments/kfold_paper6c_w125/runs/kfold_<ts>/<feat>/fold_<k>/<exp>/
# Aggregated per family into <feat>/summary_stage{1,2}.json
#
# NOTE: rare behaviours (Cough, Fall, Rumination, …) are folded into "Other"
# here (no upstream rare-class drop), so "Other" is ~3% larger than in the
# single-split paper run where they were removed. The 6-class label space is
# identical across all folds.

set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WINDOW_LEN="${WINDOW_LEN:-125}"
export WINDOW_STRIDE="${WINDOW_STRIDE:-62}"
# Pretrain checkpoints are keyed by this DATASET_ID; matches the single-split paper
# run so the already-trained w125 encoders are reused. Overridden per-feature below.
export DATASET_ID="${DATASET_ID:-AcTBeCalf_paper_w125}"
export RAW_PARQUET="${RAW_PARQUET:-$(cd "${DIR}/../.." && pwd)/dataset/processed/pretrain_raw_windowed_w${WINDOW_LEN}.parquet}"

# shellcheck source=_common.sh
source "${DIR}/_common.sh"

N_FOLDS="${N_FOLDS:-5}"
LABEL_MAP="${REPO_ROOT}/scripts/label_maps/paper_6class.json"
[ -f "$LABEL_MAP" ] || { echo "ERROR: label map not found at ${LABEL_MAP}" >&2; exit 1; }

# Reuse the SAME subject folds as the TSFEL k-fold for direct comparability.
REUSE_ASSIGNMENTS="${REUSE_ASSIGNMENTS:-${REPO_ROOT}/dataset/processed/kfold/fold_assignments.json}"
KFOLD_DATA_DIR="${KFOLD_DATA_DIR:-${REPO_ROOT}/dataset/processed/kfold_paper6c_w${WINDOW_LEN}}"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_BASE="${RUN_BASE:-${REPO_ROOT}/experiments/kfold_paper6c_w${WINDOW_LEN}/runs/kfold_${TIMESTAMP}}"
mkdir -p "${RUN_BASE}"

echo ">>> Paper 6-class k-fold run: ${RUN_BASE}"
echo ">>> N_FOLDS=${N_FOLDS}  WINDOW_LEN=${WINDOW_LEN}  PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS}"

# --- 1. Per-fold raw splits (canonical labels, no rare filter; remap happens later) ---
if [ ! -f "${KFOLD_DATA_DIR}/fold_assignments.json" ]; then
    echo ">>> Generating ${N_FOLDS} folds → ${KFOLD_DATA_DIR}"
    python "${REPO_ROOT}/scripts/generate_kfold_splits.py" \
        --csv "${REPO_ROOT}/dataset/AcTBeCalf.csv" \
        --out-dir "${KFOLD_DATA_DIR}" \
        --n-folds "${N_FOLDS}" \
        --reuse-assignments "${REUSE_ASSIGNMENTS}" \
        --skip-rare-filter
else
    echo ">>> ${KFOLD_DATA_DIR}/fold_assignments.json exists — skipping fold generation"
fi

# --- 2. Resolve GLOBAL pretrain checkpoints (window 125; reused, idempotent) ---
ensure_raw_parquet
TS2VEC_CNN_DIR=$(ts2vec_pretrain_raw_dir cnn_lstm)
TS2VEC_ROB_DIR=$(ts2vec_pretrain_raw_dir robust)
PATCHTST_PRETRAIN_DIR=$(patchtst_mae_raw_dir)

ENCODER=cnn_lstm bash "${DIR}/pretrain_encoder.sh" || true
ENCODER=robust   bash "${DIR}/pretrain_encoder.sh" || true
bash "${DIR}/pretrain_patchtst.sh" || true

CNN_CKPT="${TS2VEC_CNN_DIR}/ts2vec_best.pt"
ROB_CKPT="${TS2VEC_ROB_DIR}/ts2vec_best.pt"
PATCHTST_CKPT="${PATCHTST_PRETRAIN_DIR}/best.pt"
for ckpt in "$CNN_CKPT" "$ROB_CKPT" "$PATCHTST_CKPT"; do
    [ -f "$ckpt" ] || { echo "ERROR: missing pretrain checkpoint $ckpt" >&2; exit 1; }
done

ROCKET_NK="${ROCKET_NK:-10000}"

# --- 3. Per-fold, per-feature-family experiments ---
for k in $(seq 0 $((N_FOLDS - 1))); do
    FOLD_DATA="${KFOLD_DATA_DIR}/fold_${k}"
    SRC_TRAIN="${FOLD_DATA}/train.parquet"
    SRC_TEST="${FOLD_DATA}/test.parquet"
    [ -f "$SRC_TRAIN" ] && [ -f "$SRC_TEST" ] || { echo "ERROR: missing ${SRC_TRAIN} or ${SRC_TEST}" >&2; exit 1; }

    echo ""
    echo "########################################################"
    echo ">>> FOLD ${k} / $((N_FOLDS - 1))"
    echo "########################################################"

    for FEAT in hc catch22 rocket; do
        echo ""
        echo ">>> fold_${k} · feature set: ${FEAT}"

        FEAT_DATA="${FOLD_DATA}/${FEAT}"
        mkdir -p "${FEAT_DATA}"
        WIN_TRAIN="${FEAT_DATA}/windowed_${FEAT}_train.parquet"
        WIN_TEST="${FEAT_DATA}/windowed_${FEAT}_test.parquet"
        ROCKET_MODEL="${FEAT_DATA}/rocket_model.joblib"

        # 3a. Build feature parquets (remap to 6 classes inside the script)
        if [ ! -f "$WIN_TRAIN" ]; then
            EXTRA=()
            [ "$FEAT" = rocket ] && EXTRA+=(--rocket-manifest-out "$ROCKET_MODEL" --rocket-n-kernels "$ROCKET_NK")
            echo ">>> building ${FEAT}/train → ${WIN_TRAIN}"
            python "${REPO_ROOT}/scripts/prepare_paper_features_parquet.py" \
                --features "$FEAT" --input "$SRC_TRAIN" --output "$WIN_TRAIN" \
                --window-size "$WINDOW_LEN" --overlap 0.5 --purity-threshold 0.9 \
                --remap-labels "$LABEL_MAP" "${EXTRA[@]}"
        fi
        if [ ! -f "$WIN_TEST" ]; then
            EXTRA=()
            [ "$FEAT" = rocket ] && EXTRA+=(--rocket-manifest-in "$ROCKET_MODEL")
            echo ">>> building ${FEAT}/test → ${WIN_TEST}"
            python "${REPO_ROOT}/scripts/prepare_paper_features_parquet.py" \
                --features "$FEAT" --input "$SRC_TEST" --output "$WIN_TEST" \
                --window-size "$WINDOW_LEN" --overlap 0.5 --purity-threshold 0.9 \
                --remap-labels "$LABEL_MAP" "${EXTRA[@]}"
        fi

        # 3b. Experiment context for this fold × feature
        export TRAIN_PARQUET="$WIN_TRAIN"
        export TEST_PARQUET="$WIN_TEST"
        export PRETRAIN_PARQUET="$WIN_TRAIN"
        export DATASET_ID="AcTBeCalf_paper6c_${FEAT}"
        export EXPERIMENTS_BASE="${RUN_BASE}/${FEAT}/fold_${k}"
        mkdir -p "${EXPERIMENTS_BASE}"
        unset FREEZE_ENCODER || true

        # RF baseline (fast, first)
        RF_OUT="${EXPERIMENTS_BASE}/rf_baseline_${DATASET_ID}_s${SEED}"
        if [ ! -f "${RF_OUT}/DONE" ]; then
            mkdir -p "${RF_OUT}"
            echo ">>> fold_${k} ${FEAT}: RF baseline"
            python -m random_forest_baseline.tsfel_baseline \
                --train "$TRAIN_PARQUET" --test "$TEST_PARQUET" \
                --output_dir "$RF_OUT" --seed "$SEED" \
                2>&1 | tee -a "${RF_OUT}/train.log"
            touch "${RF_OUT}/DONE"
        else
            echo ">>> fold_${k} ${FEAT}: RF baseline already done, skipping"
        fi

        # MLP on the paper features (fast, no pretrain)
        export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
        unset RUN_SUFFIX || true
        run_experiment "tsfel_mlp" "tsfel_only"
        run_finetune   "tsfel_mlp" "tsfel_only"

        # CNN+LSTM and Robust: deep_only/hybrid × scratch/pretrain
        for ENC_AND_CKPT in "cnn_lstm:${CNN_CKPT}" "robust:${ROB_CKPT}"; do
            ENC="${ENC_AND_CKPT%%:*}"
            CKPT="${ENC_AND_CKPT##*:}"
            for MODE in deep_only hybrid; do
                RUN_SUFFIX=fromscratch run_experiment "$ENC" "$MODE"
                RUN_SUFFIX=fromscratch run_finetune   "$ENC" "$MODE"
                RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" run_experiment "$ENC" "$MODE" \
                    --init_encoder_from "$CKPT"
                RUN_SUFFIX="frompretrain_raw_ep${PRETRAIN_EPOCHS}" run_finetune   "$ENC" "$MODE"
            done
        done

        # PatchTST: deep_only/hybrid × scratch/pretrain (context_length = window)
        export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
        for MODE in deep_only hybrid; do
            RUN_SUFFIX=fromscratch run_experiment patchtst "$MODE" --context_length "$WINDOW_LEN"
            RUN_SUFFIX=fromscratch run_finetune   patchtst "$MODE" --context_length "$WINDOW_LEN"
            RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" run_experiment patchtst "$MODE" \
                --patchtst_checkpoint "$PATCHTST_CKPT" --context_length "$WINDOW_LEN"
            RUN_SUFFIX="frompretrain_raw_ep${PATCHTST_PRETRAIN_EPOCHS}" run_finetune   patchtst "$MODE" \
                --context_length "$WINDOW_LEN"
        done
    done

    echo ">>> FOLD ${k} done"
done

# --- 4. Aggregate per feature family ---
echo ""
echo "########################################################"
echo ">>> Aggregating per feature family"
echo "########################################################"
for FEAT in hc catch22 rocket; do
    echo ""
    echo ">>> Aggregating ${FEAT}"
    python "${REPO_ROOT}/scripts/aggregate_kfold_results.py" \
        --run-dir "${RUN_BASE}/${FEAT}" \
        --n-folds "${N_FOLDS}" \
        --stages 1 2
done

echo ""
echo ">>> Paper 6-class k-fold complete: ${RUN_BASE}"

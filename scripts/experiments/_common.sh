#!/usr/bin/env bash
# Shared variables and helpers for all experiment scripts.
# Source this file; do not execute it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src"

# --- Default paths (override from environment if needed) ---
TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_train.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_test.parquet}"
PRETRAIN_PARQUET="${PRETRAIN_PARQUET:-${TRAIN_PARQUET}}"
DATASET_ID="${DATASET_ID:-AcTBeCalf}"

# --- Default hyperparameters ---
SEED="${SEED:-2026}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-500}"          # Stage 1: balanced CE, ES patience=25 will stop early
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"  # Stage 2: plain CE, low LR
VAL_FRACTION="${VAL_FRACTION:-0.1}"
LR="${LR:-1e-3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
# Larger batches for CNN/LSTM/robust/TSFEL-MLP; PatchTST scripts set PATCHTST_BATCH_SIZE.
BATCH_SIZE_LARGE="${BATCH_SIZE_LARGE:-512}"
PATCHTST_BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-${BATCH_SIZE_LARGE}}"

# --- Helpers ---

make_run_dir() {
    # Usage: make_run_dir MODEL MODE
    # Optional RUN_SUFFIX (e.g. fromscratch, frompretrain_hf) disambiguates PatchTST runs.
    local MODEL="$1" MODE="$2"
    local suf="${RUN_SUFFIX:-}"
    if [ -n "$suf" ]; then
        suf="_${suf}"
    fi
    echo "${REPO_ROOT}/experiments/${MODEL}_${MODE}${suf}_${DATASET_ID}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_s${SEED}"
}

run_experiment() {
    # Usage: run_experiment MODEL MODE [-- extra CLI args ...]
    # Remaining words after MODEL MODE are passed verbatim to main.py (quoted-safe for paths with spaces).
    local MODEL="$1" MODE="$2"
    shift 2
    local OUT
    OUT=$(make_run_dir "$MODEL" "$MODE")

    if [ -f "${OUT}/DONE" ]; then
        echo ">>> ${MODEL}_${MODE}: already complete, skipping"
        return 0
    fi
    mkdir -p "${OUT}"

    echo ">>> Starting ${MODEL}_${MODE} at $(date)"
    local FREEZE_ARGS=()
    if [ "${FREEZE_ENCODER:-0}" = "1" ]; then
        FREEZE_ARGS+=(--freeze_encoder)
    fi
    if [ -f "${OUT}/checkpoint.pt" ]; then
        echo ">>> Checkpoint found at ${OUT}/checkpoint.pt, resuming..."
        python -m hybrid_activity_recognition.main \
            --mode supervised \
            --model "$MODEL" \
            --input_mode "$MODE" \
            --labeled_parquet_train "$TRAIN_PARQUET" \
            --labeled_parquet_test "$TEST_PARQUET" \
            --output_dir "$OUT" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --val_fraction "$VAL_FRACTION" \
            --checkpoint "${OUT}/checkpoint.pt" \
            "${FREEZE_ARGS[@]}" \
            "$@" \
            2>&1 | tee -a "${OUT}/train.log"
    else
        python -m hybrid_activity_recognition.main \
            --mode supervised \
            --model "$MODEL" \
            --input_mode "$MODE" \
            --labeled_parquet_train "$TRAIN_PARQUET" \
            --labeled_parquet_test "$TEST_PARQUET" \
            --output_dir "$OUT" \
            --epochs "$EPOCHS" \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --val_fraction "$VAL_FRACTION" \
            "${FREEZE_ARGS[@]}" \
            "$@" \
            2>&1 | tee -a "${OUT}/train.log"
    fi

    touch "${OUT}/DONE"
    echo ">>> ${MODEL}_${MODE} done at $(date)"
}

run_finetune() {
    # Stage 2: load best.pt from Stage 1, fine-tune with plain CE (no class weights).
    # Usage: run_finetune MODEL MODE [-- extra CLI args ...]
    local MODEL="$1" MODE="$2"
    shift 2
    local OUT
    OUT=$(make_run_dir "$MODEL" "$MODE")
    local STAGE1_CKPT="${OUT}/best.pt"
    local FINETUNE_DONE="${OUT}/DONE_finetune"

    if [ -f "${FINETUNE_DONE}" ]; then
        echo ">>> ${MODEL}_${MODE} finetune: already complete, skipping"
        return 0
    fi
    if [ ! -f "${STAGE1_CKPT}" ]; then
        echo ">>> ${MODEL}_${MODE} finetune: Stage 1 checkpoint not found at ${STAGE1_CKPT}, skipping"
        return 1
    fi

    echo ">>> Fine-tuning ${MODEL}_${MODE} at $(date)"
    python -m hybrid_activity_recognition.main \
        --mode finetune \
        --model "$MODEL" \
        --input_mode "$MODE" \
        --labeled_parquet_train "$TRAIN_PARQUET" \
        --labeled_parquet_test "$TEST_PARQUET" \
        --output_dir "$OUT" \
        --epochs "$FINETUNE_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr 1e-4 \
        --seed "$SEED" \
        --device "$DEVICE" \
        --val_fraction "$VAL_FRACTION" \
        --checkpoint "${STAGE1_CKPT}" \
        "$@" \
        2>&1 | tee -a "${OUT}/finetune.log"

    touch "${FINETUNE_DONE}"
    echo ">>> ${MODEL}_${MODE} finetune done at $(date)"
}

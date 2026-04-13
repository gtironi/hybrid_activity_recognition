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
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"

# --- Helpers ---

make_run_dir() {
    # Usage: make_run_dir MODEL MODE
    local MODEL="$1" MODE="$2"
    echo "${REPO_ROOT}/experiments/${MODEL}_${MODE}_${DATASET_ID}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_s${SEED}"
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
            --checkpoint "${OUT}/checkpoint.pt" \
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
            "$@" \
            2>&1 | tee -a "${OUT}/train.log"
    fi

    touch "${OUT}/DONE"
    echo ">>> ${MODEL}_${MODE} done at $(date)"
}

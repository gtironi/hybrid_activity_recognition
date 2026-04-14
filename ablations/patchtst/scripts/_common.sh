#!/usr/bin/env bash
# Shared helpers for isolated PatchTST ablations.
set -euo pipefail

ABL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABL_ROOT="$(cd "${ABL_SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ABL_ROOT}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src"

TRAIN_PARQUET="${TRAIN_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_train.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${REPO_ROOT}/dataset/processed/AcTBeCalf/windowed_test.parquet}"
PRETRAIN_PARQUET="${PRETRAIN_PARQUET:-${TRAIN_PARQUET}}"
DATASET_ID="${DATASET_ID:-AcTBeCalf}"

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-516}"
LR="${LR:-1e-3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-3}"
PRETRAIN_MASK_RATIO="${PRETRAIN_MASK_RATIO:-0.4}"

EXP_ROOT="${EXP_ROOT:-${REPO_ROOT}/experiments/ablations_patchtst}"
LOG_ROOT="${LOG_ROOT:-${ABL_ROOT}/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-${ABL_ROOT}/results}"
TMP_ROOT="${TMP_ROOT:-${ABL_ROOT}/tmp}"

mkdir -p "${EXP_ROOT}" "${LOG_ROOT}" "${RESULTS_ROOT}" "${TMP_ROOT}"

ts() {
    date +"%Y-%m-%dT%H:%M:%S%z"
}

run_pretrain_patchtst_variant() {
    # Usage: run_pretrain_patchtst_variant TAG [-- extra patchtst args]
    local TAG="$1"
    shift

    local OUT="${EXP_ROOT}/patchtst_pretrain_${DATASET_ID}_${TAG}_s${SEED}"
    mkdir -p "${OUT}"

    if [ -f "${OUT}/DONE" ]; then
        echo "[$(ts)] pretrain ${TAG}: already complete, skipping"
        echo "${OUT}/best.pt"
        return 0
    fi

    echo "[$(ts)] pretrain ${TAG}: start"

    local cmd=(
        python -m hybrid_activity_recognition.main
        --mode pretrain
        --pretrain_parquet "${PRETRAIN_PARQUET}"
        --output_dir "${OUT}"
        --pretrain_epochs "${PRETRAIN_EPOCHS}"
        --pretrain_lr "${PRETRAIN_LR}"
        --pretrain_mask_ratio "${PRETRAIN_MASK_RATIO}"
        --batch_size "${BATCH_SIZE}"
        --seed "${SEED}"
        --device "${DEVICE}"
    )

    if [ -f "${OUT}/checkpoint.pt" ]; then
        cmd+=(--checkpoint "${OUT}/checkpoint.pt")
    fi
    cmd+=("$@")

    "${cmd[@]}" 2>&1 | tee -a "${OUT}/train.log"
    touch "${OUT}/DONE"
    echo "[$(ts)] pretrain ${TAG}: done"

    echo "${OUT}/best.pt"
}

run_supervised_patchtst_variant() {
    # Usage: run_supervised_patchtst_variant MODE TAG PATCHTST_CHECKPOINT [-- extra patchtst args]
    local MODE="$1"
    local TAG="$2"
    local PATCHTST_CKPT="$3"
    shift 3

    local OUT="${EXP_ROOT}/patchtst_${MODE}_${DATASET_ID}_${TAG}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_s${SEED}"
    mkdir -p "${OUT}"

    if [ -f "${OUT}/DONE" ]; then
        echo "[$(ts)] supervised ${MODE} ${TAG}: already complete, skipping"
        return 0
    fi

    echo "[$(ts)] supervised ${MODE} ${TAG}: start"

    local cmd=(
        python -m hybrid_activity_recognition.main
        --mode supervised
        --model patchtst
        --input_mode "${MODE}"
        --labeled_parquet_train "${TRAIN_PARQUET}"
        --labeled_parquet_test "${TEST_PARQUET}"
        --output_dir "${OUT}"
        --epochs "${EPOCHS}"
        --batch_size "${BATCH_SIZE}"
        --lr "${LR}"
        --seed "${SEED}"
        --device "${DEVICE}"
    )

    if [ "${PATCHTST_CKPT}" != "NONE" ] && [ -n "${PATCHTST_CKPT}" ]; then
        cmd+=(--patchtst_checkpoint "${PATCHTST_CKPT}")
    fi
    if [ -f "${OUT}/checkpoint.pt" ]; then
        cmd+=(--checkpoint "${OUT}/checkpoint.pt")
    fi

    cmd+=("$@")

    "${cmd[@]}" 2>&1 | tee -a "${OUT}/train.log"
    touch "${OUT}/DONE"
    echo "[$(ts)] supervised ${MODE} ${TAG}: done"
}

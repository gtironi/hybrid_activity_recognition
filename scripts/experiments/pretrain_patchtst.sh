#!/usr/bin/env bash
# PatchTST self-supervised MAE pretraining.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

OUT=$(make_run_dir "patchtst" "pretrain")

if [ -f "${OUT}/DONE" ]; then
    echo ">>> PatchTST pretraining already complete, skipping"
    exit 0
fi
mkdir -p "${OUT}"

echo ">>> Starting PatchTST pretraining at $(date)"
    if [ -f "${OUT}/checkpoint.pt" ]; then
    echo ">>> Checkpoint found, resuming pretraining..."
    python -m hybrid_activity_recognition.main \
        --mode pretrain \
        --pretrain_parquet "$PRETRAIN_PARQUET" \
        --output_dir "$OUT" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --pretrain_lr "$PRETRAIN_LR" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --device "$DEVICE" \
            --checkpoint "${OUT}/checkpoint.pt" \
            ${CONTEXT_LENGTH:+--context_length "$CONTEXT_LENGTH"} \
            ${PATCH_LEN:+--patch_len "$PATCH_LEN"} \
            ${STRIDE:+--stride "$STRIDE"} \
        2>&1 | tee -a "${OUT}/train.log"
else
    python -m hybrid_activity_recognition.main \
        --mode pretrain \
        --pretrain_parquet "$PRETRAIN_PARQUET" \
        --output_dir "$OUT" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --pretrain_lr "$PRETRAIN_LR" \
        --batch_size "$BATCH_SIZE" \
        --seed "$SEED" \
        --device "$DEVICE" \
            ${CONTEXT_LENGTH:+--context_length "$CONTEXT_LENGTH"} \
            ${PATCH_LEN:+--patch_len "$PATCH_LEN"} \
            ${STRIDE:+--stride "$STRIDE"} \
            2>&1 | tee -a "${OUT}/train.log"
fi

touch "${OUT}/DONE"
echo ">>> PatchTST pretraining done at $(date)"

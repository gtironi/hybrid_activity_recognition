#!/usr/bin/env bash
# PatchTST experiments: {deep_only, hybrid} × {fromscratch, frompretrain_raw}.
#
# Checkpoint guard: if the MAE raw checkpoint does not exist yet, this
# script triggers pretrain_patchtst.sh to create it. If it already exists,
# pretraining is skipped and the existing checkpoint is reused.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

ENCODER="patchtst"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/${ENCODER}"
mkdir -p "${EXPERIMENTS_BASE}"

# Default: use only the last (best-trained) pretrain snapshot.
# Override with PRETRAIN_EPOCHS_LIST="10 25 40" to sweep multiple snapshots
# (or use run_pretrain_ablation.sh which keeps the ablation in its own folder).
PRETRAIN_EPOCHS_LIST="${PRETRAIN_EPOCHS_LIST:-40}"

# --- Checkpoint guard: pretrain on raw data if no checkpoint yet ---
RAW_PRETRAIN_DIR=$(patchtst_mae_raw_dir)
if [ ! -f "${RAW_PRETRAIN_DIR}/DONE" ]; then
    echo ">>> ${ENCODER}: raw MAE checkpoint missing — running pretrain_patchtst.sh"
    bash "${DIR}/pretrain_patchtst.sh"
else
    echo ">>> ${ENCODER}: raw MAE checkpoint found at ${RAW_PRETRAIN_DIR}, skipping pretrain"
fi

# --- Supervised experiments: 2 input modes × {fromscratch, frompretrain_raw} ---
unset FREEZE_ENCODER || true
for MODE in deep_only hybrid; do
    # From scratch
    RUN_SUFFIX=fromscratch run_experiment "$ENCODER" "$MODE"

    # From raw-data MAE checkpoints
    for EP in $PRETRAIN_EPOCHS_LIST; do
        CKPT="${RAW_PRETRAIN_DIR}/pretrain_ep${EP}.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> SKIP frompretrain_raw_ep${EP}: checkpoint not found at ${CKPT}"
            continue
        fi
        RUN_SUFFIX="frompretrain_raw_ep${EP}" run_experiment "$ENCODER" "$MODE" \
            --patchtst_checkpoint "$CKPT"
    done
done

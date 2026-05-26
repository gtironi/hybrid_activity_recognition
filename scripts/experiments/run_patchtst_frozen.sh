#!/usr/bin/env bash
# PatchTST frozen-encoder experiments (encoder capacity ablation).
# Loads the raw-data MAE checkpoint (ep40) and trains only the head / TSFEL / fusion
# while keeping the PatchTST encoder weights frozen.
#
# Requires: raw MAE checkpoint produced by pretrain_patchtst.sh.
# Errors out (does not pretrain) if the checkpoint is missing.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"
export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"

ENCODER="patchtst"
EP="${EP:-40}"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/patchtst_frozen"
mkdir -p "${EXPERIMENTS_BASE}"

RAW_PRETRAIN_DIR=$(patchtst_mae_raw_dir)
RAW_CKPT="${RAW_PRETRAIN_DIR}/pretrain_ep${EP}.pt"

if [ ! -f "$RAW_CKPT" ]; then
    echo "ERROR: PatchTST raw MAE checkpoint not found at: $RAW_CKPT" >&2
    echo "       Run pretrain_patchtst.sh first." >&2
    exit 1
fi

echo ">>> PatchTST frozen encoder, using checkpoint: $RAW_CKPT"

export FREEZE_ENCODER=1
for MODE in deep_only hybrid; do
    RUN_SUFFIX="frompretrain_raw_ep${EP}_encfrozen" run_experiment "$ENCODER" "$MODE" \
        --patchtst_checkpoint "$RAW_CKPT"
done
unset FREEZE_ENCODER

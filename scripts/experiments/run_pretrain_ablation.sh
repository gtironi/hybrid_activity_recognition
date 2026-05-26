#!/usr/bin/env bash
# Pretrain checkpoint ablation: for a given encoder, runs the frompretrain
# supervised pipeline against EVERY saved pretrain snapshot (not just the last).
# Outputs are isolated in their own sub-folder so they don't mix with the
# default per-encoder runs.
#
# Usage:
#   ENCODER=cnn_lstm bash run_pretrain_ablation.sh
#   ENCODER=robust   bash run_pretrain_ablation.sh
#   ENCODER=patchtst bash run_pretrain_ablation.sh
#   bash run_pretrain_ablation.sh cnn_lstm    # positional arg also accepted
#
# Output sub-folder: ${EXPERIMENTS_BASE}/pretrain_ablation_${ENCODER}/
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"

ENCODER="${1:-${ENCODER:-}}"
case "$ENCODER" in
    cnn_lstm|robust)
        DEFAULT_LIST="20 50 100"
        PRETRAIN_DIR=$(ts2vec_pretrain_raw_dir "$ENCODER")
        CKPT_PREFIX="ts2vec_ep"
        INIT_FLAG="--init_encoder_from"
        export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
        # Require at least the last snapshot to be present.
        REQUIRED_CKPT="${PRETRAIN_DIR}/${CKPT_PREFIX}100.pt"
        ;;
    patchtst)
        DEFAULT_LIST="10 25 40"
        PRETRAIN_DIR=$(patchtst_mae_raw_dir)
        CKPT_PREFIX="pretrain_ep"
        INIT_FLAG="--patchtst_checkpoint"
        export BATCH_SIZE="${PATCHTST_BATCH_SIZE:-128}"
        REQUIRED_CKPT="${PRETRAIN_DIR}/${CKPT_PREFIX}40.pt"
        ;;
    "")
        echo "ERROR: ENCODER not set. Usage: ENCODER=cnn_lstm|robust|patchtst bash $0" >&2
        exit 2
        ;;
    *)
        echo "ERROR: ENCODER='$ENCODER' not supported. Use cnn_lstm | robust | patchtst." >&2
        exit 2
        ;;
esac

if [ ! -f "$REQUIRED_CKPT" ]; then
    echo "ERROR: pretrain checkpoint not found at $REQUIRED_CKPT" >&2
    echo "       Run pretrain_encoder.sh / pretrain_patchtst.sh first." >&2
    exit 1
fi

EPOCHS_LIST="${PRETRAIN_EPOCHS_LIST:-$DEFAULT_LIST}"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/pretrain_ablation_${ENCODER}"
mkdir -p "${EXPERIMENTS_BASE}"

echo ">>> Pretrain ablation: ${ENCODER}"
echo ">>> Snapshots:         ${EPOCHS_LIST}"
echo ">>> Pretrain dir:      ${PRETRAIN_DIR}"
echo ">>> Output dir:        ${EXPERIMENTS_BASE}"

unset FREEZE_ENCODER || true
for MODE in deep_only hybrid; do
    for EP in $EPOCHS_LIST; do
        CKPT="${PRETRAIN_DIR}/${CKPT_PREFIX}${EP}.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> SKIP frompretrain_raw_ep${EP}: checkpoint not found at ${CKPT}"
            continue
        fi
        RUN_SUFFIX="frompretrain_raw_ep${EP}" run_experiment "$ENCODER" "$MODE" \
            "$INIT_FLAG" "$CKPT"
        RUN_SUFFIX="frompretrain_raw_ep${EP}" run_finetune   "$ENCODER" "$MODE"
    done
done

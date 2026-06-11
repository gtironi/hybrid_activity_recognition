#!/usr/bin/env bash
# CNN+LSTM experiments: {deep_only, hybrid} × {fromscratch, frompretrain_raw}.
#
# Checkpoint guard: if the TS2Vec raw checkpoint does not exist yet, this
# script triggers pretrain_encoder.sh to create it. If it already exists,
# pretraining is skipped and the existing checkpoint is reused.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${DIR}/_common.sh"
export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"

ENCODER="cnn_lstm"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE}/${ENCODER}"
mkdir -p "${EXPERIMENTS_BASE}"

# --- Checkpoint guard: pretrain on raw data if no checkpoint yet ---
TS2VEC_DIR=$(ts2vec_pretrain_raw_dir "$ENCODER")
if [ ! -f "${TS2VEC_DIR}/ts2vec_best.pt" ]; then
    echo ">>> ${ENCODER}: TS2Vec raw checkpoint missing — running pretrain_encoder.sh"
    ENCODER="$ENCODER" bash "${DIR}/pretrain_encoder.sh"
else
    echo ">>> ${ENCODER}: TS2Vec raw checkpoint found at ${TS2VEC_DIR}, skipping pretrain"
fi

# Default: use the best checkpoint saved by the new trainer.
PRETRAIN_EPOCHS_LIST="${PRETRAIN_EPOCHS_LIST:-best}"

# --- Supervised experiments: 2 input modes × {fromscratch, frompretrain_raw} ---
for MODE in deep_only hybrid; do
    # From scratch
    RUN_SUFFIX=fromscratch run_experiment "$ENCODER" "$MODE"

    # From raw-data TS2Vec checkpoints
    for EP in $PRETRAIN_EPOCHS_LIST; do
        if [ "$EP" = "best" ]; then
            CKPT="${TS2VEC_DIR}/ts2vec_best.pt"
        else
            CKPT="${TS2VEC_DIR}/ts2vec_ep${EP}.pt"
        fi
        if [ ! -f "$CKPT" ]; then
            echo ">>> SKIP frompretrain_raw_ep${EP}: checkpoint not found at ${CKPT}"
            continue
        fi
        RUN_SUFFIX="frompretrain_raw_ep${EP}" run_experiment "$ENCODER" "$MODE" \
            --init_encoder_from "$CKPT"
    done
done

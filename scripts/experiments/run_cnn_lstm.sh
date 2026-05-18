#!/usr/bin/env bash
# CNN+LSTM encoder: deep_only and hybrid experiments (fromscratch + frompretrain ep20/50/100).
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"

# Pretrain once; returns output_dir containing ep20/ep50/ep100 snapshots.
TS2VEC_DIR=$(run_ts2vec_pretrain "cnn_lstm")

for MODE in deep_only hybrid; do
    # From scratch (default initialisation)
    RUN_SUFFIX=fromscratch run_experiment "cnn_lstm" "$MODE"
    RUN_SUFFIX=fromscratch run_finetune   "cnn_lstm" "$MODE"

    # From TS2Vec pretrained encoder — one run per snapshot epoch.
    for EP in 20 50 100; do
        CKPT="${TS2VEC_DIR}/ts2vec_ep${EP}.pt"
        if [ ! -f "$CKPT" ]; then
            echo ">>> SKIP frompretrain_ep${EP}: checkpoint not found at ${CKPT}"
            continue
        fi
        RUN_SUFFIX="frompretrain_ep${EP}" run_experiment "cnn_lstm" "$MODE" \
            --init_encoder_from "$CKPT"
        RUN_SUFFIX="frompretrain_ep${EP}" run_finetune   "cnn_lstm" "$MODE"
    done
done

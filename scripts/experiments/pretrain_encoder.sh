#!/usr/bin/env bash
# TS2Vec self-supervised pretraining on the unlabeled raw data.
# Produces checkpoints only — no downstream supervised experiments.
#
# Usage:
#   ENCODER=cnn_lstm bash pretrain_encoder.sh
#   ENCODER=robust   bash pretrain_encoder.sh
#   bash pretrain_encoder.sh cnn_lstm    # positional arg also accepted
#
# Output: experiments/ts2vec_pretrain_raw_${ENCODER}_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_s${SEED}/
#         containing ts2vec_ep20.pt, ts2vec_ep50.pt, ts2vec_ep100.pt
set -euo pipefail
source "$(dirname "$0")/_common.sh"

ENCODER="${1:-${ENCODER:-}}"
case "$ENCODER" in
    cnn_lstm|robust) ;;
    "")
        echo "ERROR: ENCODER not set. Usage: ENCODER=cnn_lstm|robust bash $0" >&2
        exit 2
        ;;
    *)
        echo "ERROR: ENCODER='$ENCODER' not supported by TS2Vec. Use cnn_lstm or robust." >&2
        echo "       (For PatchTST use pretrain_patchtst.sh instead.)" >&2
        exit 2
        ;;
esac

OUT=$(run_ts2vec_pretrain_raw "$ENCODER")
echo ">>> Done. Checkpoints in: $OUT"

#!/usr/bin/env bash
# PatchTST MAE self-supervised pretraining on the unlabeled raw data.
# Produces checkpoints only — no downstream supervised experiments.
#
# Usage:
#   bash pretrain_patchtst.sh
#
# Output: experiments/patchtst_pretrain_raw_${DATASET_ID}_ep${PRETRAIN_EPOCHS}_bs${PATCHTST_BATCH_SIZE}_lr${PRETRAIN_LR}_s${SEED}/
#         containing per-epoch snapshots (pretrain_ep{N}.pt) and best.pt
set -euo pipefail
source "$(dirname "$0")/_common.sh"

OUT=$(run_patchtst_mae_raw)
echo ">>> Done. Checkpoints in: $OUT"

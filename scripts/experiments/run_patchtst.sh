#!/usr/bin/env bash
# PatchTST encoder: pretrain (if needed) + deep_only and hybrid experiments.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

PATCHTST_CHECKPOINT="${PATCHTST_CHECKPOINT:-}"

# Run pretraining if no checkpoint provided
if [ -z "$PATCHTST_CHECKPOINT" ]; then
    bash "$(dirname "$0")/pretrain_patchtst.sh"
    PRETRAIN_DIR=$(make_run_dir "patchtst" "pretrain")
    PATCHTST_CHECKPOINT="${PRETRAIN_DIR}/best.pt"
fi

for MODE in deep_only hybrid; do
    run_experiment "patchtst" "$MODE" --patchtst_checkpoint "$PATCHTST_CHECKPOINT"
done

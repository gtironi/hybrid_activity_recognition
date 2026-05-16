#!/usr/bin/env bash
# Standalone: just the warmup_encoder combo, with corrected checkpoint paths.
#
# The original run_warmup_ablation.sh looked for deep_only checkpoints at
#   experiments/${MODEL}_deep_only_..._bs${BATCH_SIZE_LARGE}_...
# but they actually live at
#   experiments/0.2_val/${MODEL}_deep_only_..._bs512_...
# so the combo was silently skipped. This script hard-codes the right paths.
#
# Outputs go to experiments/ alongside the other abl_* runs (bs=2048) for a
# fair comparison against abl_baseline.
#
# Usage:
#   bash scripts/experiments/run_warmup_encoder_only.sh
#   nohup bash scripts/experiments/run_warmup_encoder_only.sh > logs/warmup_encoder.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"

# Match the other ablation runs (bs=2048) so abl_baseline is the comparable.
export BATCH_SIZE_LARGE="${BATCH_SIZE_LARGE:-2048}"
export BATCH_SIZE="${BATCH_SIZE_LARGE}"
export EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-${REPO_ROOT}/experiments/ablations}"

MODELS="${MODELS:-cnn_lstm robust}"

# Deep_only checkpoints live in experiments/0.2_val/ with bs=512.
deep_only_ckpt() {
    local MODEL="$1"
    echo "${REPO_ROOT}/experiments/0.2_val/${MODEL}_deep_only_${DATASET_ID}_ep${EPOCHS}_bs512_lr${LR}_s${SEED}/best.pt"
}

run_warmup() {
    local MODEL="$1"
    local CKPT
    CKPT="$(deep_only_ckpt "$MODEL")"
    if [ ! -f "$CKPT" ]; then
        echo ">>> SKIP ${MODEL}: deep_only checkpoint not found at ${CKPT}"
        return 1
    fi
    export RUN_SUFFIX="abl_warmup_encoder"
    echo "=== ${MODEL}/hybrid (abl_warmup_encoder) ==="
    echo "    using encoder from: ${CKPT}"
    run_experiment "$MODEL" "hybrid" --init_encoder_from "$CKPT"
    run_finetune   "$MODEL" "hybrid"
    unset RUN_SUFFIX
}

echo "=== warmup_encoder-only run start: $(date) ==="

for MODEL in $MODELS; do
    run_warmup "$MODEL"
done

echo "=== warmup_encoder-only run done: $(date) ==="

#!/usr/bin/env bash
# Dry run: 1 epoch per combo, just to verify CLI args and code paths work.
# Outputs go to experiments/dryrun/ — safe to delete after.
#
# Usage:
#   bash scripts/experiments/dry_run_ablations.sh
#   bash scripts/experiments/dry_run_ablations.sh 2>&1 | tee logs/dry_run.log
set -euo pipefail
source "$(dirname "$0")/_common.sh"

export EPOCHS=1
export BATCH_SIZE_LARGE=64
export PATCHTST_BATCH_SIZE=32
export DATASET_ID="dryrun"
export EXPERIMENTS_BASE="${REPO_ROOT}/experiments/dryrun"

# Real deep_only checkpoints (ep500) to test --init_encoder_from.
REAL_CNNLSTM_CKPT="${REPO_ROOT}/experiments/0.2_val/cnn_lstm_deep_only_AcTBeCalf_ep500_bs512_lr1e-3_s2026/best.pt"
REAL_ROBUST_CKPT="${REPO_ROOT}/experiments/0.2_val/robust_deep_only_AcTBeCalf_ep500_bs512_lr1e-3_s2026/best.pt"

run_dry() {
    local MODEL="$1" SUFFIX="$2"
    shift 2
    export RUN_SUFFIX="$SUFFIX"
    export BATCH_SIZE="${BATCH_SIZE_LARGE}"
    [ "$MODEL" = "patchtst" ] && export BATCH_SIZE="${PATCHTST_BATCH_SIZE}"
    run_experiment "$MODEL" "hybrid" "$@" || { echo "FAIL: ${MODEL} ${SUFFIX}"; exit 1; }
    unset RUN_SUFFIX
}

echo "=== DRY RUN start: $(date) ==="

# ── Loss ablation ─────────────────────────────────────────────────────────────
echo "── loss ablation ──"
for MODEL in cnn_lstm robust; do
    export BATCH_SIZE="${BATCH_SIZE_LARGE}"
    run_dry "$MODEL" "lossce"            --loss_type ce
    run_dry "$MODEL" "lossce_balsamp"    --loss_type ce   --balanced_sampler
    run_dry "$MODEL" "lossfocal"         --loss_type focal
    run_dry "$MODEL" "lossfocal_balsamp" --loss_type focal --balanced_sampler
done

# ── Warmup ablation ───────────────────────────────────────────────────────────
echo "── warmup ablation ──"

for MODEL in cnn_lstm robust; do
    REAL_CKPT="${REPO_ROOT}/experiments/0.2_val/${MODEL}_deep_only_AcTBeCalf_ep500_bs512_lr1e-3_s2026/best.pt"
    run_dry "$MODEL" "abl_baseline"
    run_dry "$MODEL" "abl_curriculum10" --tsfel_dropout_warmup_epochs 10
    run_dry "$MODEL" "abl_tsfel_drop25" --tsfel_dropout_p 0.25
    if [ -f "$REAL_CKPT" ]; then
        run_dry "$MODEL" "abl_warmup_encoder" --init_encoder_from "$REAL_CKPT"
    else
        echo ">>> SKIP ${MODEL} abl_warmup_encoder (not found: ${REAL_CKPT})"
    fi
done

echo ""
echo "=== DRY RUN done: $(date) ==="
echo "Dirs created (safe to delete):"
ls -d "${EXPERIMENTS_BASE}/"* 2>/dev/null || echo "(none found)"
echo ""
echo "To clean up: rm -rf ${EXPERIMENTS_BASE}"

#!/usr/bin/env bash
# Ablation: warm-up / TSFEL-dropout strategies on hybrid models.
# Runs 4 configs per model:
#   abl_baseline           : current training (no warm-up, no TSFEL dropout)
#   abl_warmup_encoder     : load encoder weights from deep_only checkpoint
#   abl_curriculum10       : first 10 epochs zero TSFEL entirely (p=1.0), then normal
#   abl_tsfel_dropout25    : per batch, with 25% probability zero the entire TSFEL
#                            vector (NOT feature-level dropout — all-or-nothing per batch)
#
# Assumes the deep_only checkpoints already exist at the standard paths
# (run scripts/experiments/run_cnn_lstm.sh and run_robust.sh first).
# Does not include models that already use SSL pretraining.
#
# Usage:
#   bash scripts/experiments/run_warmup_ablation.sh
#   MODELS="cnn_lstm" bash scripts/experiments/run_warmup_ablation.sh
#   nohup bash scripts/experiments/run_warmup_ablation.sh > logs/warmup_ablation.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/_common.sh"

EXPERIMENTS_BASE="${EXPERIMENTS_BASE:-${REPO_ROOT}/experiments/ablations/warmup}"
MODELS="${MODELS:-cnn_lstm robust}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
DROPOUT_P="${DROPOUT_P:-0.25}"

deep_only_ckpt() {
    # Points to the canonical (non-ablation) experiments dir — deep_only runs
    # live at the top-level experiments/, not inside ablations/.
    local MODEL="$1"
    echo "${REPO_ROOT}/experiments/${MODEL}_deep_only_${DATASET_ID}_ep${EPOCHS}_bs${BATCH_SIZE_LARGE}_lr${LR}_s${SEED}/best.pt"
}

run_combo() {
    # Args: model suffix [-- extra CLI args ...]
    local MODEL="$1" SUFFIX="$2"
    shift 2
    export RUN_SUFFIX="$SUFFIX"
    export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"
    echo "=== ${MODEL}/hybrid (${SUFFIX}) ==="
    run_experiment "$MODEL" "hybrid" "$@"
    run_finetune   "$MODEL" "hybrid"
    unset RUN_SUFFIX
}

echo "=== Warm-up / TSFEL-dropout ablation start: $(date) ==="

for MODEL in $MODELS; do
    CKPT="$(deep_only_ckpt "$MODEL")"

    # 1. Baseline (matches existing hybrid training).
    run_combo "$MODEL" "abl_baseline"

    # 2. Encoder warm-up from deep_only checkpoint.
    if [ -f "$CKPT" ]; then
        run_combo "$MODEL" "abl_warmup_encoder" --init_encoder_from "$CKPT"
    else
        echo ">>> ${MODEL}: deep_only checkpoint not found at ${CKPT}, skipping abl_warmup_encoder"
    fi

    # 3. Curriculum: first N epochs train with TSFEL fully zeroed.
    run_combo "$MODEL" "abl_curriculum${WARMUP_EPOCHS}" \
        --tsfel_dropout_warmup_epochs "$WARMUP_EPOCHS"

    # 4. Stochastic TSFEL dropout throughout training.
    run_combo "$MODEL" "abl_tsfel_dropout$(printf '%g' "$(echo "$DROPOUT_P * 100" | bc -l)" | cut -d. -f1)" \
        --tsfel_dropout_p "$DROPOUT_P"
done

echo "=== Warm-up / TSFEL-dropout ablation done: $(date) ==="

#!/usr/bin/env bash
# Master script: runs loss/sampler ablation + warm-up/TSFEL-dropout ablation.
#
# Loss ablation:  4 combos × {cnn_lstm:hybrid, patchtst:hybrid, robust:hybrid, tsfel_mlp:tsfel_only} = 16 runs
# Warmup ablation: 4 combos × {cnn_lstm, robust} hybrid = 8 runs
# Total: 24 runs.
#
# Each run is skipped if its DONE marker already exists, so this script is safe
# to re-run after a partial failure.
#
# Usage:
#   bash scripts/experiments/run_all_ablations.sh
#   nohup bash scripts/experiments/run_all_ablations.sh > logs/all_ablations.log 2>&1 &
#
# Restrict to a subset:
#   ONLY=loss     bash scripts/experiments/run_all_ablations.sh
#   ONLY=warmup   bash scripts/experiments/run_all_ablations.sh
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONLY="${ONLY:-both}"

echo "=== Full ablation suite start: $(date) ==="

if [ "$ONLY" = "both" ] || [ "$ONLY" = "loss" ]; then
    bash "${DIR}/run_loss_ablation.sh"
fi

if [ "$ONLY" = "both" ] || [ "$ONLY" = "warmup" ]; then
    bash "${DIR}/run_warmup_ablation.sh"
fi

echo ""
echo "=== Generating summary table ==="
python "${DIR}/../summarize_experiments.py" || true

echo "=== Full ablation suite done: $(date) ==="

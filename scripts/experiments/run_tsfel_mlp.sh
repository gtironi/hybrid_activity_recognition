#!/usr/bin/env bash
# TSFEL-only deep baseline: TSFEL branch + MLP head.
set -euo pipefail
source "$(dirname "$0")/_common.sh"
export BATCH_SIZE="${BATCH_SIZE_LARGE:-512}"

# Stage 1: balanced CE loss, ES patience=25
run_experiment "tsfel_mlp" "tsfel_only"
# Stage 2: plain CE loss, LR=1e-4, 50 epochs
run_finetune   "tsfel_mlp" "tsfel_only"
